"""Azimuth sub-aperture decomposition of NISAR RSLC data."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from scipy.fft import fft, ifft, fftshift, ifftshift, fftfreq, next_fast_len
from scipy.ndimage import uniform_filter1d
import math
import warnings
import numpy as np
import h5py

from ioput import read_block, get_available_pols, copy_h5_structure


@dataclass
class SubapertureMetaData:
    """Azimuth parameters needed to form subapertures from an RSLC product."""
    zero_doppler_time_spacing: float   # seconds; azimuth sampling interval
    azimuth_bandwidth: float           # Hz; processed azimuth bandwidth
    doppler_centroid: np.ndarray       # (n_az, n_rg) Hz; on the coarse grid
    doppler_az_time: np.ndarray        # (n_az,) zeroDopplerTime coords [s]
    doppler_slant_range: np.ndarray    # (n_rg,) slant-range coords [m]
    az_time: np.ndarray                # (n_az_full,) full-res zeroDopplerTime [s]
    slant_range: np.ndarray            # (n_rg_full,) full-res slantRange [m]
    freq: str = 'A'                    # frequency band the swath parameters refer to

    @classmethod
    def load_from_rslc(cls, fh: h5py.File, freq: str = 'A') -> 'SubapertureMetaData':
        """Load subaperture metadata from an open RSLC HDF5 file handle."""
        base_proc = f'science/LSAR/RSLC/metadata/processingInformation/parameters/frequency{freq}'
        base_swath = f'science/LSAR/RSLC/swaths/frequency{freq}'
        return cls(
            zero_doppler_time_spacing=float(
                fh['science/LSAR/RSLC/swaths/zeroDopplerTimeSpacing'][()]),
            azimuth_bandwidth=float(
                fh[f'{base_swath}/processedAzimuthBandwidth'][()]),
            doppler_centroid=fh[f'{base_proc}/dopplerCentroid'][()],
            doppler_az_time=fh[f'{base_proc}/zeroDopplerTime'][()],
            doppler_slant_range=fh[f'{base_proc}/slantRange'][()],
            az_time=fh['science/LSAR/RSLC/swaths/zeroDopplerTime'][()],
            slant_range=fh[f'{base_swath}/slantRange'][()],
            freq=freq,
        )

    @classmethod
    def load_from_rslc_path(cls, path: Path, freq: str = 'A') -> 'SubapertureMetaData':
        """Load subaperture metadata from an RSLC HDF5 file path."""
        with h5py.File(path, 'r') as fh:
            return cls.load_from_rslc(fh, freq=freq)


def _linear_bracket(knots, x):
    """Left knot index and unclipped weight for linear interpolation that extrapolates."""
    i = np.clip(np.searchsorted(knots, x, side='right') - 1, 0, len(knots) - 2)
    return i, (x - knots[i]) / (knots[i + 1] - knots[i])


@dataclass(frozen=True)
class DopplerCentroid:
    """Doppler centroid f_dc, bilinear in a LUT; phase 2 pi int f_dc dt from az_time_ref."""
    az_time: np.ndarray       # (K,) LUT zeroDopplerTime knots [s]
    slant_range: np.ndarray   # (M,) LUT slant-range knots [m]; M = 1 is range-independent
    doppler: np.ndarray       # (K, M) Doppler centroid [Hz]
    az_time_ref: float        # [s]; phase is zero here

    @classmethod
    def from_meta(cls, meta: SubapertureMetaData, az_time_ref: float = None) -> 'DopplerCentroid':
        """RSLC Doppler centroid LUT; az_time_ref defaults to the scene-centre line."""
        if az_time_ref is None:
            az_time_ref = float(meta.az_time[len(meta.az_time) // 2])
        return cls(meta.doppler_az_time, meta.doppler_slant_range, meta.doppler_centroid,
                   float(az_time_ref))

    @classmethod
    def constant(cls, doppler_centroid_hz: float, az_time_ref: float) -> 'DopplerCentroid':
        return cls(np.array([az_time_ref - 1.0, az_time_ref + 1.0]), np.array([0.0]),
                   np.full((2, 1), float(doppler_centroid_hz)), float(az_time_ref))

    @classmethod
    def from_h5(cls, group: h5py.Group) -> 'DopplerCentroid':
        return cls(group['az_time'][()], group['slant_range'][()], group['doppler'][()],
                   float(group['az_time_ref'][()]))

    def to_h5(self, group: h5py.Group) -> None:
        for name in ('az_time', 'slant_range', 'doppler', 'az_time_ref'):
            group.create_dataset(name, data=getattr(self, name))

    @property
    def is_constant(self) -> bool:
        return bool(np.all(self.doppler == self.doppler.flat[0]))

    @property
    def range_dependent(self) -> bool:
        return bool(np.any(np.diff(self.doppler, axis=1)))

    def _at_range(self, slant_range):
        """DC (K, m) at the azimuth knots and the given slant ranges."""
        slant_range = np.atleast_1d(slant_range)
        if len(self.slant_range) == 1:
            return np.repeat(self.doppler, len(slant_range), axis=1)
        j, w = _linear_bracket(self.slant_range, slant_range)
        return self.doppler[:, j] * (1 - w) + self.doppler[:, j + 1] * w

    def frequency(self, az_time, slant_range) -> np.ndarray:
        """DC [Hz] on the grid az_time (n,) x slant_range (m,)."""
        f_knots = self._at_range(slant_range)
        i, w = _linear_bracket(self.az_time, np.atleast_1d(az_time))
        return f_knots[i] * (1 - w)[:, None] + f_knots[i + 1] * w[:, None]

    def phase(self, az_time, slant_range) -> np.ndarray:
        """Doppler centroid phase [rad] on the grid az_time (n,) x slant_range (m,)."""
        f_knots = self._at_range(slant_range)
        dt_knots = np.diff(self.az_time)[:, None]
        # exact integral of the piecewise-linear f_dc: trapezoids up to the knot, quadratic beyond
        cycles_knots = np.concatenate([np.zeros((1, f_knots.shape[1])), np.cumsum(
            0.5 * (f_knots[1:] + f_knots[:-1]) * dt_knots, axis=0)])

        def cycles(t):
            i, w = _linear_bracket(self.az_time, np.atleast_1d(t))
            u = w[:, None] * dt_knots[i]
            slope = (f_knots[i + 1] - f_knots[i]) / dt_knots[i]
            return cycles_knots[i] + f_knots[i] * u + 0.5 * slope * u ** 2

        return 2 * np.pi * (cycles(az_time) - cycles(self.az_time_ref))


class AzimuthSubaperture:
    """Overlapping azimuth sub-apertures from raised-cosine windows on the Doppler spectrum."""

    def __init__(
        self,
        meta: SubapertureMetaData,
        n_subapertures: int = 5,
        overlap: float = 0.3,
        shrink_fraction_useful: float = 0.02,
        raised_cosine_beta: float = 0.5,
        normalize_window_gain: bool = True,
        demodulate_subaperture: bool = True,
        remodulate_to_full_dc: bool = False,
        deweight_spectrum: bool = False,
        deweight_smooth_bins: int = None,
    ):
        """Without estimate_deweighting_profile, de-weighting is estimated from the first block."""
        self.meta = meta
        self.n_subapertures = n_subapertures
        self.overlap = overlap
        self.shrink_fraction_useful = shrink_fraction_useful
        self.raised_cosine_beta = raised_cosine_beta
        self.normalize_window_gain = normalize_window_gain
        self.demodulate_subaperture = demodulate_subaperture
        self.remodulate_to_full_dc = remodulate_to_full_dc
        self.deweight_spectrum = deweight_spectrum
        self.deweight_smooth_bins = deweight_smooth_bins
        self._deweight_profile = None

    def _zero_pad(self, block):
        """Zero-pad a 2-D SLC block (n_az, n_rg) along azimuth to the next fast FFT length."""
        n = next_fast_len(block.shape[0])
        block_zp = np.zeros((n, block.shape[1]), dtype=np.complex64)
        block_zp[:block.shape[0], :] = block
        return block_zp

    @cached_property
    def scene_doppler_centroid(self) -> DopplerCentroid:
        """RSLC Doppler centroid LUT, phase referenced to the scene-centre line."""
        return DopplerCentroid.from_meta(self.meta)

    def _demodulate_slc(self, block_zp, doppler_centroid, az_time, slant_range):
        """Demodulate by the Doppler centroid phase; returns (demodulated, modulation)."""
        phase = doppler_centroid.phase(az_time, slant_range)
        modulation = np.empty(phase.shape, dtype=np.complex64)
        modulation.real = np.cos(phase)
        modulation.imag = np.sin(phase)
        return block_zp * modulation.conj(), modulation

    def _compute_spectrum(self, block_zp, bin_useful_lo=None, bin_useful_hi=None, bin_width=None):
        """Compute the azimuth DFT spectrum (centered), applying de-weighting if enabled."""
        spectrum = fftshift(fft(block_zp, axis=0, workers=-1), axes=0)
        if self.deweight_spectrum:
            if self._deweight_profile is None:
                avg_power = np.mean(np.abs(spectrum) ** 2, axis=1)
                self._deweight_profile = self._build_deweight_profile(
                    avg_power, bin_useful_lo, bin_useful_hi, bin_width)
            spectrum = spectrum * self._resampled_deweight_profile(spectrum.shape[0])[:, np.newaxis]
        return spectrum

    def _resampled_deweight_profile(self, n):
        """De-weighting profile interpolated onto an n-bin grid in normalised frequency."""
        n_p = self._deweight_profile.shape[0]
        if n_p == n:
            return self._deweight_profile
        freq_profile = (np.arange(n_p) - n_p // 2) / n_p
        freq_block = (np.arange(n) - n // 2) / n
        return np.interp(freq_block, freq_profile, self._deweight_profile).astype(np.float32)

    def _bandwidth_fractions(self, azimuth_bandwidth, zero_doppler_time_spacing):
        """Useful and per-sub-aperture bandwidth as fractions of the PRF."""
        fraction_useful = (azimuth_bandwidth * zero_doppler_time_spacing
                           * (1 - self.shrink_fraction_useful))
        fraction_sub = fraction_useful / (1 + (self.n_subapertures - 1) * (1 - self.overlap))
        return fraction_useful, fraction_sub

    def _compute_subaperture_params(self, n, azimuth_bandwidth, zero_doppler_time_spacing):
        """DFT bin width, non-integer sub-aperture centroid bins, and useful-band bin range."""
        fraction_useful, fraction_sub = self._bandwidth_fractions(
            azimuth_bandwidth, zero_doppler_time_spacing)
        bin_useful_lo = int(n * (1 - fraction_useful) // 2)
        bin_useful_hi = int(n * (1 + fraction_useful) // 2)
        # float so that sub-aperture bands in normalised frequency do not depend on n
        bin_width = n * fraction_sub
        # for both even and odd N
        half_span = bin_width * (1 - self.overlap) * (self.n_subapertures - 1) / 2
        # symmetric and equally spaced about n // 2
        bin_centroids = np.linspace(n // 2 - half_span, n // 2 + half_span, self.n_subapertures)
        return bin_width, bin_centroids, bin_useful_lo, bin_useful_hi

    def _build_deweight_profile(self, avg_power, bin_useful_lo, bin_useful_hi, bin_width, eps=1e-6):
        """Smoothed correction from avg_power within the useful band; 1 elsewhere."""
        smooth_bins = self.deweight_smooth_bins or max(int(bin_width / 16), 1)
        segment = avg_power[bin_useful_lo:bin_useful_hi]
        smoothed = uniform_filter1d(segment, size=smooth_bins, mode='nearest')
        smoothed = np.maximum(smoothed, eps * float(np.max(smoothed)))
        shape = smoothed / float(np.mean(smoothed))
        profile = np.ones(avg_power.shape[0], dtype=np.float32)
        profile[bin_useful_lo:bin_useful_hi] = 1.0 / np.sqrt(shape)
        return profile

    def estimate_deweighting_profile(self, rslc_path, doppler_centroid: DopplerCentroid = None,
                                     az_start=0, az_stop=None, rg_start=0, rg_stop=None, freq='A',
                                     pol=None, blocksize_range=512):
        """Store de-weighting from the mean power spectrum; default scene_doppler_centroid."""
        rslc_path = Path(rslc_path)
        if doppler_centroid is None:
            doppler_centroid = self.scene_doppler_centroid
        with h5py.File(rslc_path, 'r') as fh:
            if pol is None:
                pol = get_available_pols(fh, freq)[0]
            n_az, n_rg = fh[f'science/LSAR/RSLC/swaths/frequency{freq}/{pol}'].shape
        if az_stop is None:
            az_stop = n_az
        if rg_stop is None:
            rg_stop = n_rg

        n_az_padded = next_fast_len(az_stop - az_start)
        az_time = (self.meta.az_time[az_start]
                   + np.arange(n_az_padded) * self.meta.zero_doppler_time_spacing)
        power_sum = np.zeros(n_az_padded)
        n_rg_total = 0
        for rg_block_start in range(rg_start, rg_stop, blocksize_range):
            rg_block_stop = min(rg_block_start + blocksize_range, rg_stop)
            block = read_block(rslc_path, az_start, az_stop, rg_block_start, rg_block_stop,
                               freq=freq, pol=pol)
            block_zp = self._zero_pad(block)
            block_zp, _ = self._demodulate_slc(
                block_zp, doppler_centroid, az_time,
                self.meta.slant_range[rg_block_start:rg_block_stop])
            spectrum = fftshift(fft(block_zp, axis=0, workers=-1), axes=0)
            power_sum += np.sum(np.abs(spectrum) ** 2, axis=1)
            n_rg_total += rg_block_stop - rg_block_start

        avg_power = power_sum / n_rg_total
        bin_width, _, bin_useful_lo, bin_useful_hi = self._compute_subaperture_params(
            n_az_padded, self.meta.azimuth_bandwidth, self.meta.zero_doppler_time_spacing)
        self._deweight_profile = self._build_deweight_profile(
            avg_power, bin_useful_lo, bin_useful_hi, bin_width)
        return self._deweight_profile

    def _window_gain_scale(self, win, bin_useful_width, eps=1e-12):
        """Scale factor so subaperture power matches full-aperture power for distributed target."""
        win_energy = float(np.sum(win.astype(np.float64) ** 2))
        if win_energy < eps:
            warnings.warn(
                "Sub-aperture window has near-zero passband energy; skipping normalization",
                UserWarning, stacklevel=3)
            return 1.0
        return np.sqrt(bin_useful_width / win_energy)

    def _apply_subaperture_windows(self, spectrum, bin_centroids, bin_width, bin_useful_width):
        """Raised-cosine windowed spectra (n_subapertures, n, n_rg)."""
        spectrum_sub = np.zeros((self.n_subapertures,) + spectrum.shape, dtype=np.complex64)
        for j, centroid in enumerate(bin_centroids):
            win = construct_raised_cosine(
                spectrum.shape[0], centroid, bin_width, beta=self.raised_cosine_beta)
            spectrum_sub[j] = spectrum * win[:, np.newaxis]
            if self.normalize_window_gain:
                scale = self._window_gain_scale(win, bin_useful_width)
                spectrum_sub[j] *= np.float32(scale)
        return spectrum_sub

    def _invert_subapertures(self, spectrum_sub, bin_centroids, modulation, n_unpadded,
                             az_line_offset=0.0):
        """Inverse-FFT to n_unpadded lines; az_line_offset is first line minus reference line."""
        n = spectrum_sub.shape[1]
        block_sub = ifft(ifftshift(spectrum_sub, axes=1), axis=1, workers=-1)
        if self.demodulate_subaperture:
            lines = np.arange(n) + az_line_offset
            for j, centroid in enumerate(bin_centroids):
                sub_mod = np.exp(2j * np.pi * (centroid - n // 2) * lines / n)
                block_sub[j] *= sub_mod.conj()[:, np.newaxis]
        if self.remodulate_to_full_dc:
            block_sub *= modulation[np.newaxis]
        return block_sub[:, :n_unpadded, :]

    def _process_block(self, block, doppler_centroid: DopplerCentroid = None, az_time_start=0.0,
                       slant_range=None, zero_doppler_time_spacing=None, azimuth_bandwidth=None):
        """Sub-apertures (N, n_az, n_rg); default DC: scene mean, phase zero at az_time_start."""
        if doppler_centroid is None:
            doppler_centroid = DopplerCentroid.constant(
                float(np.mean(self.meta.doppler_centroid)), az_time_start)
        if slant_range is None:
            if doppler_centroid.range_dependent:
                raise ValueError('a range-dependent Doppler centroid needs slant_range')
            slant_range = np.zeros(block.shape[1])
        if zero_doppler_time_spacing is None:
            zero_doppler_time_spacing = self.meta.zero_doppler_time_spacing
        if azimuth_bandwidth is None:
            azimuth_bandwidth = self.meta.azimuth_bandwidth

        n_unpadded = block.shape[0]
        block_zp = self._zero_pad(block)
        az_time = az_time_start + np.arange(block_zp.shape[0]) * zero_doppler_time_spacing
        block_zp, modulation = self._demodulate_slc(
            block_zp, doppler_centroid, az_time, slant_range)
        bin_width, bin_centroids, bin_useful_lo, bin_useful_hi = self._compute_subaperture_params(
            block_zp.shape[0], azimuth_bandwidth, zero_doppler_time_spacing)
        spectrum = self._compute_spectrum(block_zp, bin_useful_lo, bin_useful_hi, bin_width)
        bin_useful_width = bin_useful_hi - bin_useful_lo
        spectrum_sub = self._apply_subaperture_windows(
            spectrum, bin_centroids, bin_width, bin_useful_width)
        az_line_offset = (az_time_start - doppler_centroid.az_time_ref) / zero_doppler_time_spacing
        return self._invert_subapertures(spectrum_sub, bin_centroids, modulation, n_unpadded,
                                         az_line_offset=az_line_offset)

    def process_tile(self, rslc_path, az_start, az_stop, rg_start, rg_stop,
                     doppler_centroid: DopplerCentroid = None, freq='A', pol='HH',
                     az_margin=None):
        """Sub-apertures (N, n_az, n_rg) of a tile, read with az_margin extra lines on each side."""
        if freq != self.meta.freq:
            raise ValueError(f'freq {freq} does not match metadata frequency {self.meta.freq}')
        if doppler_centroid is None:
            doppler_centroid = self.scene_doppler_centroid
        if az_margin is None:
            az_margin = azimuth_margin(self)
        read_start = max(az_start - az_margin, 0)
        read_stop = min(az_stop + az_margin, len(self.meta.az_time))
        block = read_block(rslc_path, read_start, read_stop, rg_start, rg_stop, freq=freq, pol=pol)
        block_sub = self._process_block(
            block, doppler_centroid=doppler_centroid,
            az_time_start=float(self.meta.az_time[read_start]),
            slant_range=self.meta.slant_range[rg_start:rg_stop])
        crop_start = az_start - read_start
        return block_sub[:, crop_start:crop_start + az_stop - az_start, :]

    def interpolate_doppler_centroid(self, az_idx, rg_idx) -> float:
        """Doppler centroid [Hz] bilinearly interpolated at full-resolution (az_idx, rg_idx)."""
        f = self.scene_doppler_centroid.frequency(
            self.meta.az_time[az_idx], self.meta.slant_range[rg_idx])
        return float(f[0, 0])

    def _setup_output_h5(self, src, dst, freq, pols, n_az, n_rg, blocksize_range):
        swath_grp = src[f'science/LSAR/RSLC/swaths/frequency{freq}']
        skip_paths = {swath_grp[pol].name for pol in pols}
        copy_h5_structure(src['/'], dst['/'], skip=skip_paths)
        out_swath = dst[f'science/LSAR/RSLC/swaths/frequency{freq}']
        for pol in pols:
            ds = out_swath.create_dataset(
                pol,
                shape=(self.n_subapertures, n_az, n_rg),
                dtype=np.complex64,
                chunks=(1, min(256, n_az), min(blocksize_range, n_rg)),
            )
            ds.attrs['description'] = (
                f'Subaperture SLC for polarization {pol}; '
                f'dim 0 = subaperture index (0..{self.n_subapertures - 1})'
            )
        return out_swath

    @staticmethod
    def iter_tiles(az_start, az_stop, rg_start, rg_stop, blocksize_az=None, blocksize_rg=None):
        """(az_start, az_stop, rg_start, rg_stop) tiles of an extent; None is one tile per axis."""
        az_step = blocksize_az or az_stop - az_start
        rg_step = blocksize_rg or rg_stop - rg_start
        for az0 in range(az_start, az_stop, az_step):
            for rg0 in range(rg_start, rg_stop, rg_step):
                yield az0, min(az0 + az_step, az_stop), rg0, min(rg0 + rg_step, rg_stop)

    def process_rslc(
        self,
        rslc_path: Path,
        output_h5: Path,
        freq: str = 'A',
        pols: list = None,
        blocksize_range: int = 512,
        blocksize_az: int = None,
        doppler_centroid: DopplerCentroid = None,
    ) -> None:
        """Sub-aperture RSLC; default scene_doppler_centroid; pols=None means all."""
        if doppler_centroid is None:
            doppler_centroid = self.scene_doppler_centroid
        warnings.warn(
            "Output HDF5 metadata is copied verbatim from the NISAR RSLC source; "
            "product-level metadata (e.g. product type, doppler centroid) "
            "is not updated and needs to be reviewed and fixed.",
            UserWarning,
            stacklevel=2,
        )
        rslc_path = Path(rslc_path)
        with h5py.File(rslc_path, 'r') as src:
            if pols is None:
                pols = get_available_pols(src, freq)
            n_az, n_rg = src[f'science/LSAR/RSLC/swaths/frequency{freq}/{pols[0]}'].shape
            with h5py.File(output_h5, 'w') as dst:
                out_swath = self._setup_output_h5(src, dst, freq, pols, n_az, n_rg, blocksize_range)
                # read back with DopplerCentroid.from_h5 to rebuild the phase
                doppler_centroid.to_h5(out_swath.create_group('subapertureDopplerCentroid'))
                az_start_prev = None
                for az_start, az_stop, rg_start, rg_stop in \
                        self.iter_tiles(0, n_az, 0, n_rg, blocksize_az, blocksize_range):
                    if self.deweight_spectrum and az_start != az_start_prev:
                        self.estimate_deweighting_profile(
                            rslc_path, doppler_centroid=doppler_centroid, az_start=az_start,
                            az_stop=az_stop,
                            freq=freq, pol=pols[0], blocksize_range=blocksize_range)
                        az_start_prev = az_start
                    for pol in pols:
                        out_swath[pol][:, az_start:az_stop, rg_start:rg_stop] = self.process_tile(
                            rslc_path, az_start, az_stop, rg_start, rg_stop,
                            doppler_centroid=doppler_centroid, freq=freq, pol=pol)

    def process_rslc_uniform_dc(
        self,
        rslc_path: Path,
        output_h5: Path,
        freq: str = 'A',
        pols: list = None,
        blocksize_range: int = 512,
    ) -> None:
        """Sub-aperture RSLC at the scene-mean Doppler centroid; pols=None means all."""
        doppler_centroid = DopplerCentroid.constant(
            float(np.mean(self.meta.doppler_centroid)), self.scene_doppler_centroid.az_time_ref)
        self.process_rslc(rslc_path, output_h5, freq, pols, blocksize_range,
                          doppler_centroid=doppler_centroid)


def azimuth_margin(sub: AzimuthSubaperture, n_widths: float = 6) -> int:
    """Azimuth margin [pixels] of n_widths sub-aperture impulse-response widths."""
    _, fraction_sub = sub._bandwidth_fractions(
        sub.meta.azimuth_bandwidth, sub.meta.zero_doppler_time_spacing)
    return math.ceil(n_widths / fraction_sub)


# TODO: shares Tukey/raised-cosine kernel with ISCE3 range split-spectrum
def construct_raised_cosine(n, bin_centroid, bin_width, beta=0.5):
    """Raised-cosine bandpass (roll-off beta) over n DFT bins; bin_centroid may be non-integer."""
    dist = np.abs(np.arange(n) - bin_centroid)
    half_width = bin_width / 2
    rolloff_hwidth = beta * bin_width / 2
    flat_hwidth = half_width - rolloff_hwidth

    # clips at array edges; may change to circular
    rc = np.zeros(n, dtype=np.float32)
    rc[dist < flat_hwidth] = 1.0
    if rolloff_hwidth > 0:
        in_taper = (dist >= flat_hwidth) & (dist < half_width)
        rc[in_taper] = 0.5 * (1 + np.cos(np.pi * (dist[in_taper] - flat_hwidth) / rolloff_hwidth))
    else:
        # half weight at exact edges keeps the rectangle symmetric
        rc[np.isclose(dist, half_width)] = 0.5
    return rc


def plot_subaperture_diagnostics(
    sub: AzimuthSubaperture,
    block: np.ndarray,
    doppler_centroid_hz: float = None,
    zero_doppler_time_spacing: float = None,
    azimuth_bandwidth: float = None,
    demodulate: bool = True,
    save_path: Path = None,
):
    import matplotlib.pyplot as plt

    if doppler_centroid_hz is None:
        doppler_centroid_hz = float(np.mean(sub.meta.doppler_centroid))
    if zero_doppler_time_spacing is None:
        zero_doppler_time_spacing = sub.meta.zero_doppler_time_spacing
    if azimuth_bandwidth is None:
        azimuth_bandwidth = sub.meta.azimuth_bandwidth

    doppler_centroid = DopplerCentroid.constant(doppler_centroid_hz, 0.0)
    block_subaperture = sub._process_block(
        block, doppler_centroid=doppler_centroid,
        zero_doppler_time_spacing=zero_doppler_time_spacing, azimuth_bandwidth=azimuth_bandwidth)

    # DFT-domain: zero-pad, demodulate, compute spectrum and window params
    block_zp = sub._zero_pad(block)
    n = block_zp.shape[0]
    block_zp_dm, _ = sub._demodulate_slc(
        block_zp, doppler_centroid, np.arange(n) * zero_doppler_time_spacing,
        np.zeros(block.shape[1]))

    bin_width, bin_centroids, bin_useful_lo, bin_useful_hi = sub._compute_subaperture_params(
        n, azimuth_bandwidth, zero_doppler_time_spacing)
    bin_useful_width = bin_useful_hi - bin_useful_lo
    spectrum = sub._compute_spectrum(block_zp_dm, bin_useful_lo, bin_useful_hi, bin_width)

    # Frequency-domain: power spectra of SLC and each subaperture
    n_az_slc = block.shape[0]
    n_az_sub = block_subaperture.shape[1]
    freqs_slc = fftshift(fftfreq(n_az_slc, d=zero_doppler_time_spacing))
    freqs_sub = fftshift(fftfreq(n_az_sub, d=zero_doppler_time_spacing))

    if demodulate:
        mod_slc = np.exp(
            -2j * np.pi * doppler_centroid_hz * np.arange(n_az_slc) * zero_doppler_time_spacing)
        slc_input = block * mod_slc[:, np.newaxis]
    else:
        slc_input = block

    if demodulate and sub.remodulate_to_full_dc:
        # still at native Doppler centroid; demodulate for display
        mod_sub = np.exp(
            -2j * np.pi * doppler_centroid_hz * np.arange(n_az_sub) * zero_doppler_time_spacing)
        sub_input = block_subaperture * mod_sub[np.newaxis, :, np.newaxis]
    else:
        sub_input = block_subaperture

    # Hann taper suppresses leakage from the block-edge discontinuity; power-preserving
    taper = construct_raised_cosine(n_az_slc, (n_az_slc - 1) / 2, n_az_slc, beta=1.0)
    taper = (taper / np.sqrt(np.mean(taper ** 2)))[:, np.newaxis]

    slc_spec = np.mean(
        np.abs(fftshift(fft(slc_input * taper, axis=0, workers=-1), axes=0)) ** 2, axis=1
    )
    n_sub = block_subaperture.shape[0]
    sub_specs = np.array([
        np.mean(
            np.abs(fftshift(fft(sub_input[j] * taper, axis=0, workers=-1), axes=0)) ** 2, axis=1
        )
        for j in range(n_sub)
    ])

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_sub))

    fig, (ax_slc, ax_sub, ax_win) = plt.subplots(3, 1, figsize=(8, 10))

    # Panel 1: SLC azimuth power spectrum (Hz)
    ax_slc.plot(freqs_slc, 10 * np.log10(slc_spec))
    ax_slc.set_ylabel('Power (dB)')
    ax_slc.set_title('SLC azimuth spectrum')
    if demodulate:
        ax_slc.axvline(0, c='k', lw=0.5, ls='--')

    # Panel 2: Subaperture azimuth power spectra (Hz)
    for j, (spec, color) in enumerate(zip(sub_specs, colors)):
        ax_sub.plot(freqs_sub, 10 * np.log10(spec), color=color, label=f'sub {j}', alpha=0.5)
    ax_sub.set_xlabel('Frequency (Hz)')
    ax_sub.set_ylabel('Power (dB)')
    ax_sub.set_title('Subaperture azimuth spectra')
    if demodulate:
        ax_sub.axvline(0, c='k', lw=0.5, ls='--')

    # Panel 3: DFT-bin power spectrum with window overlays
    power = np.mean(np.abs(spectrum) ** 2, axis=1)
    ax_win.plot(np.arange(n), 10 * np.log10(power), c='k', lw=0.8)
    ax_win.axvline(n // 2, c='k', lw=0.5)
    ax_win.axvspan(bin_useful_lo, bin_useful_hi, alpha=0.15, color='#aaaaaa')
    ax_win.set_ylabel('Power (dB)')
    ax_win2 = ax_win.twinx()
    bin_indices = np.arange(n)
    max_weight = 1.0
    for centroid, color in zip(bin_centroids, colors):
        win = construct_raised_cosine(n, centroid, bin_width, beta=sub.raised_cosine_beta)
        if sub.normalize_window_gain:
            scale = sub._window_gain_scale(win, bin_useful_width)
            win = win * scale
        ax_win2.plot(bin_indices, win.real, color=color, alpha=0.8)
        max_weight = max(max_weight, float(win.real.max()))
        ax_win.axvline(centroid, c='#aaaaaa', alpha=0.5, zorder=2)
    if sub.deweight_spectrum and sub._deweight_profile is not None:
        ax_win2.plot(bin_indices, sub._resampled_deweight_profile(n), c='k', ls='--', lw=1,
                     alpha=0.6, label='de-weight profile')
    ax_win2.set_ylim(0, 1.2 * max_weight)
    ax_win2.set_ylabel('Window weight')
    ax_win.set_xlabel('DFT index')
    ax_win.set_title('Spectrum and subaperture windows')

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    p0 = Path('/home/simon/Work/fmnisar/NISAR/')
    path_nisar = p0 / 'NISAR_L1_PR_RSLC_010_073_D_053_4005_DHDH_A_20260114T062318_20260114T062355_X05010_N_F_J_001.h5'

    meta = SubapertureMetaData.load_from_rslc_path(path_nisar)
    doppler_centroid = meta.doppler_centroid
    block = np.load(p0 / 'block.npy')

    sub = AzimuthSubaperture(meta)

    # plot_subaperture_diagnostics(sub, block)

    folder_out = Path('/media/simon/Extreme SSD/fmnisar/')
    sub.process_rslc_uniform_dc(path_nisar, folder_out / 'uniform_doppler_centroid.h5', pols=['HH'])
    sub.process_rslc(path_nisar, folder_out / 'variable_doppler_centroid.h5', pols=['HH'])
