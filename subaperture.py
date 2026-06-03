'''
Created on Jun 2, 2026

@author: simon
'''
from dataclasses import dataclass
from pathlib import Path
from scipy.fft import next_fast_len
from scipy.interpolate import RegularGridInterpolator
import warnings
import numpy as np
import h5py

from inspect_nisar import extract_rslc_params
from slc_io import read_block, get_available_pols, copy_h5_structure


@dataclass
class SubapertureMetaData:
    """Azimuth parameters needed to form subapertures from an RSLC product."""
    zero_doppler_time_spacing: float   # seconds; 1/PRF
    azimuth_bandwidth: float           # Hz; processed azimuth bandwidth
    doppler_centroid: np.ndarray       # (n_az, n_rg) Hz; on the coarse grid
    doppler_az_time: np.ndarray        # (n_az,) zeroDopplerTime coords [s]
    doppler_slant_range: np.ndarray    # (n_rg,) slant-range coords [m]
    az_time: np.ndarray                # (n_az_full,) full-res zeroDopplerTime [s]
    slant_range: np.ndarray            # (n_rg_full,) full-res slantRange [m]

    @classmethod
    def load_from_rslc(cls, fh: h5py.File, freq: str = 'A') -> 'SubapertureMetaData':
        """Load subaperture metadata from an open RSLC HDF5 file handle.

        Parameters
        ----------
        fh : h5py.File
            Open handle to the NISAR RSLC HDF5 file.
        freq : {'A', 'B'}
            Frequency band to extract.
        """
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
        )

    @classmethod
    def load_from_rslc_path(cls, path: Path, freq: str = 'A') -> 'SubapertureMetaData':
        """Load subaperture metadata from an RSLC HDF5 file path.

        Parameters
        ----------
        path : Path
            Path to the NISAR RSLC HDF5 file.
        freq : {'A', 'B'}
            Frequency band to extract.
        """
        with h5py.File(path, 'r') as fh:
            return cls.load_from_rslc(fh, freq=freq)


class AzimuthSubaperture:
    """Subaperture decomposition of an RSLC azimuth spectrum.

    Works on level-1 SLC data. Currently assumes a spatially uniform doppler centroid. 
    Subapertures can optionally be demodulated to baseband. At the end, the doppler centroid modulation
    of the full-azimuth-bandwitdth image is not re-applied by default.
    
    # To do: multicore, metadata, resample
    """

    def __init__(
        self,
        meta: SubapertureMetaData,
        n_subapertures: int = 5,
        overlap: float = 0.3,
        shrink_fraction_useful: float = 0.02,
        raised_cosine_beta: float = 0.5,
        demodulate_subaperture: bool = False,
        remodulate_to_full_dc: bool = True
        
    ):
        self.meta = meta
        self.n_subapertures = n_subapertures
        self.overlap = overlap
        self.shrink_fraction_useful = shrink_fraction_useful
        self.raised_cosine_beta = raised_cosine_beta
        self.demodulate_subaperture = demodulate_subaperture
        self.remodulate_to_full_dc = remodulate_to_full_dc

    def _zero_pad(self, block):
        N = next_fast_len(block.shape[0])
        block_zp = np.zeros((N, block.shape[1]), dtype=np.complex64)
        block_zp[:block.shape[0], :] = block
        return block_zp

    def _demodulate(self, block_zp, dc, zdts, az_time_start=0.0):
        N = block_zp.shape[0]
        az_time = az_time_start + np.arange(N) * zdts
        modulation = np.exp(2j * np.pi * dc * az_time)
        return block_zp * modulation.conj()[:, np.newaxis], modulation

    def _compute_spectrum(self, block_zp):
        return np.fft.fftshift(np.fft.fft(block_zp, axis=0), axes=0)

    def _compute_subaperture_params(self, N, az_bw, zdts):
        fraction_useful = az_bw * zdts * (1 - self.shrink_fraction_useful)
        bin_useful_lo = int(N * (1 - fraction_useful) // 2)
        bin_useful_hi = int(N * (1 + fraction_useful) // 2)
        bin_useful_width = bin_useful_hi - bin_useful_lo
        bin_width = int(bin_useful_width // (1 + (self.n_subapertures - 1) * (1 - self.overlap)))
        bin_centroids = (
            bin_useful_lo + bin_width // 2
            + np.arange(self.n_subapertures) * bin_width * (1 - self.overlap)
        ).astype(np.int64)
        return bin_width, bin_centroids

    def _apply_subaperture_windows(self, spectrum, bin_centroids, bin_width):
        spectrum_sub = np.zeros((self.n_subapertures,) + spectrum.shape, dtype=np.complex64)
        for j, centroid in enumerate(bin_centroids):
            win = construct_raised_cosine(spectrum.shape[0], centroid, bin_width, beta=self.raised_cosine_beta)
            spectrum_sub[j] = spectrum * win[:, np.newaxis]
        return spectrum_sub

    def _invert_subapertures(self, spectrum_sub, bin_centroids, modulation, N_orig):
        N = spectrum_sub.shape[1]
        block_sub = np.fft.ifft(np.fft.ifftshift(spectrum_sub, axes=1), axis=1)
        if self.demodulate_subaperture:
            for j, centroid in enumerate(bin_centroids):
                sub_mod = np.exp(2j * np.pi * (centroid - N // 2) * np.arange(N) / N)
                block_sub[j] *= sub_mod.conj()[:, np.newaxis]
        if self.remodulate_to_full_dc:
            block_sub *= modulation[np.newaxis, :, np.newaxis]
        return block_sub[:, :N_orig, :]

    def _process_block(self, block, dc=None, zdts=None, az_bw=None, az_time_start=0.0):
        if dc is None:
            dc = float(np.mean(self.meta.doppler_centroid))
        if zdts is None:
            zdts = self.meta.zero_doppler_time_spacing
        if az_bw is None:
            az_bw = self.meta.azimuth_bandwidth

        N_orig = block.shape[0]
        block_zp = self._zero_pad(block)
        block_zp, modulation = self._demodulate(block_zp, dc, zdts, az_time_start=az_time_start)
        spectrum = self._compute_spectrum(block_zp)
        bin_width, bin_centroids = self._compute_subaperture_params(block_zp.shape[0], az_bw, zdts)
        spectrum_sub = self._apply_subaperture_windows(spectrum, bin_centroids, bin_width)
        return self._invert_subapertures(spectrum_sub, bin_centroids, modulation, N_orig)

    def _build_dc_interpolator(self) -> RegularGridInterpolator:
        return RegularGridInterpolator(
            (self.meta.doppler_az_time, self.meta.doppler_slant_range),
            self.meta.doppler_centroid,
            method='linear',
            bounds_error=False,
            fill_value=None,
        )

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
    def _iter_blocks(n_az, n_rg, blocksize_az, blocksize_range):
        az_starts = range(0, n_az, blocksize_az) if blocksize_az is not None else [0]
        for az_start in az_starts:
            az_stop = min(az_start + blocksize_az, n_az) if blocksize_az is not None else n_az
            for rg_start in range(0, n_rg, blocksize_range):
                yield az_start, az_stop, rg_start, min(rg_start + blocksize_range, n_rg)

    def _run_blocks(self, rslc_path, output_h5, freq, pols, blocksize_range, blocksize_az,
                    dc_for_block):
        warnings.warn(
            "Output HDF5 metadata is copied verbatim from the NISAR RSLC source; "
            "product-level metadata (e.g. product type, doppler centroid) "
            "is not updated and needs to be reviewed and fixed.",
            UserWarning,
            stacklevel=3,
        )
        rslc_path = Path(rslc_path)
        with h5py.File(rslc_path, 'r') as src:
            if pols is None:
                pols = get_available_pols(src, freq)
            n_az, n_rg = src[f'science/LSAR/RSLC/swaths/frequency{freq}/{pols[0]}'].shape
            with h5py.File(output_h5, 'w') as dst:
                out_swath = self._setup_output_h5(src, dst, freq, pols, n_az, n_rg, blocksize_range)
                for az_start, az_stop, rg_start, rg_stop in \
                        self._iter_blocks(n_az, n_rg, blocksize_az, blocksize_range):
                    dc = dc_for_block(az_start, az_stop, rg_start, rg_stop)
                    az_time_start = float(self.meta.az_time[az_start])
                    for pol in pols:
                        block = read_block(rslc_path, az_start, az_stop, rg_start, rg_stop,
                                           freq=freq, pol=pol)
                        sub_block = self._process_block(block, dc=dc, az_time_start=az_time_start)
                        out_swath[pol][:, az_start:az_stop, rg_start:rg_stop] = sub_block

    def process_rslc_uniform_dc(
        self,
        rslc_path: Path,
        output_h5: Path,
        freq: str = 'A',
        pols: list = None,
        blocksize_range: int = 512,
    ) -> None:
        """Process an RSLC into subapertures using the scene-mean Doppler centroid.

        Parameters
        ----------
        rslc_path : Path
        output_h5 : Path
            Output HDF5 (created or overwritten).
        freq : str
            Frequency band, ``'A'`` or ``'B'``.
        pols : list of str, optional
            Polarizations to process. Defaults to all available.
        blocksize_range : int
            Range bins per block.
        """
        mean_dc = float(np.mean(self.meta.doppler_centroid))
        self._run_blocks(rslc_path, output_h5, freq, pols, blocksize_range,
                         blocksize_az=None, dc_for_block=lambda *_: mean_dc)

    def process_rslc(
        self,
        rslc_path: Path,
        output_h5: Path,
        freq: str = 'A',
        pols: list = None,
        blocksize_range: int = 512,
        blocksize_az: int = None,
    ) -> None:
        """Process an RSLC into subapertures with per-block interpolated Doppler centroid.

        Interpolates the Doppler centroid to each block center using bilinear
        interpolation on the coarse DC grid. ``blocksize_az=None`` processes the full
        azimuth extent as a single block.

        Parameters
        ----------
        rslc_path : Path
        output_h5 : Path
            Output HDF5 (created or overwritten).
        freq : str
            Frequency band, ``'A'`` or ``'B'``.
        pols : list of str, optional
            Polarizations to process. Defaults to all available.
        blocksize_range : int
            Range bins per block.
        blocksize_az : int or None
            Azimuth lines per block. ``None`` uses the full azimuth extent as one block.
        """
        rgi = self._build_dc_interpolator()

        def dc_for_block(az_start, az_stop, rg_start, rg_stop):
            az_center = self.meta.az_time[(az_start + az_stop) // 2]
            rg_center = self.meta.slant_range[(rg_start + rg_stop) // 2]
            return float(rgi([[az_center, rg_center]]))

        self._run_blocks(rslc_path, output_h5, freq, pols, blocksize_range,
                         blocksize_az=blocksize_az, dc_for_block=dc_for_block)


def construct_raised_cosine(n, bin_centroid, bin_width, beta=0.5):
    # bin indices are after fftshift, from 0 to n-1
    rc = np.zeros(n, dtype=np.float32)
    bin_rolloff_hwidth = int((beta * bin_width) // 2)

    bin_left_cutoff  = bin_centroid - bin_width // 2
    bin_left_flat    = bin_left_cutoff + bin_rolloff_hwidth
    bin_right_flat   = bin_centroid + bin_width // 2 - bin_rolloff_hwidth
    bin_right_cutoff = bin_centroid + bin_width // 2

    # clips; may change to circular
    rc[max(0, bin_left_flat):min(n, bin_right_flat)] = 1.0

    if bin_rolloff_hwidth > 0:
        i_left = np.arange(max(0, bin_left_cutoff), min(n, bin_left_flat))
        if len(i_left):
            rc[i_left] = 0.5 * (1 + np.cos(np.pi * (bin_left_flat - i_left) / bin_rolloff_hwidth))

        i_right = np.arange(max(0, bin_right_flat), min(n, bin_right_cutoff))
        if len(i_right):
            rc[i_right] = 0.5 * (1 + np.cos(np.pi * (i_right - bin_right_flat) / bin_rolloff_hwidth))

    return rc


def plot_subaperture_diagnostics(
    sub: AzimuthSubaperture,
    block: np.ndarray,
    dc: float = None,
    zdts: float = None,
    az_bw: float = None,
    demodulate: bool = True,
):
    import matplotlib.pyplot as plt

    if dc is None:
        dc = float(np.mean(sub.meta.doppler_centroid))
    if zdts is None:
        zdts = sub.meta.zero_doppler_time_spacing
    if az_bw is None:
        az_bw = sub.meta.azimuth_bandwidth

    block_subaperture = sub._process_block(block, dc=dc, zdts=zdts, az_bw=az_bw)

    # DFT-domain: zero-pad, demodulate, compute spectrum and window params
    block_zp = sub._zero_pad(block)
    block_zp_dm, _ = sub._demodulate(block_zp, dc, zdts)
    spectrum = sub._compute_spectrum(block_zp_dm)

    N = block_zp.shape[0]
    bin_width, bin_centroids = sub._compute_subaperture_params(N, az_bw, zdts)
    fraction_useful = az_bw * zdts * (1 - sub.shrink_fraction_useful)
    bin_useful_lo = int(N * (1 - fraction_useful) // 2)
    bin_useful_hi = int(N * (1 + fraction_useful) // 2)

    # Frequency-domain: power spectra of SLC and each subaperture
    N_slc = block.shape[0]
    N_sub = block_subaperture.shape[1]
    freqs_slc = np.fft.fftshift(np.fft.fftfreq(N_slc, d=zdts))
    freqs_sub = np.fft.fftshift(np.fft.fftfreq(N_sub, d=zdts))

    if demodulate:
        mod_slc = np.exp(-2j * np.pi * dc * np.arange(N_slc) * zdts)
        mod_sub = np.exp(-2j * np.pi * dc * np.arange(N_sub) * zdts)
        slc_input = block * mod_slc[:, np.newaxis]
        sub_input = block_subaperture * mod_sub[np.newaxis, :, np.newaxis]
    else:
        slc_input = block
        sub_input = block_subaperture

    slc_spec = np.mean(
        np.abs(np.fft.fftshift(np.fft.fft(slc_input, axis=0), axes=0)) ** 2, axis=1
    )
    n_sub = block_subaperture.shape[0]
    sub_specs = np.array([
        np.mean(
            np.abs(np.fft.fftshift(np.fft.fft(sub_input[j], axis=0), axes=0)) ** 2, axis=1
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
    ax_win.plot(np.arange(N), 10 * np.log10(power), c='k', lw=0.8)
    ax_win.axvline(N // 2, c='k', lw=0.5)
    ax_win.axvspan(bin_useful_lo, bin_useful_hi, alpha=0.15, color='#aaaaaa')
    ax_win.set_ylabel('Power (dB)')
    ax_win2 = ax_win.twinx()
    bin_indices = np.arange(N)
    for centroid, color in zip(bin_centroids, colors):
        win = construct_raised_cosine(N, int(centroid), int(bin_width), beta=sub.raised_cosine_beta)
        ax_win2.plot(bin_indices, win.real, color=color, alpha=0.7)
        ax_win.axvline(centroid, c='#aaaaaa', alpha=0.5, zorder=2)
    ax_win2.set_ylim(0, 1.5)
    ax_win2.set_ylabel('Window weight')
    ax_win.set_xlabel('DFT index')
    ax_win.set_title('Spectrum and subaperture windows')

    fig.tight_layout()
    plt.show()



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    p0 = Path('/home/simon/Work/fmnisar/NISAR/')
    path_nisar = p0 / 'NISAR_L1_PR_RSLC_010_073_D_053_4005_DHDH_A_20260114T062318_20260114T062355_X05010_N_F_J_001.h5'

    meta = SubapertureMetaData.load_from_rslc_path(path_nisar)
    dc = meta.doppler_centroid
    block = np.load(p0 / 'block.npy')

    sub = AzimuthSubaperture(meta)

    # plot_subaperture_diagnostics(sub, block)
    
    folder_out = Path('/media/simon/Extreme SSD/fmnisar/')
    sub.process_rslc_uniform_dc(path_nisar, folder_out /  'uniform_doppler_centroid.h5', pols=['HH'])
    sub.process_rslc(path_nisar, folder_out /  'variable_doppler_centroid.h5', pols=['HH'])
    
    
    