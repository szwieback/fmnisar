"""Run every function in the repo against the synthetic RSLC.

This is the scheduled-CI suite. It is deliberately broad rather than deep: the
goal is to confirm that each entry point still runs on a realistic product and
returns something well formed, so a dependency upgrade or a refactor that breaks
the pipeline shows up on the next weekly run.
"""

import h5py
import numpy as np
import pytest

from ioput import (read_block, read_range_block, get_available_pols,
                   copy_h5_structure, latlon_box_to_radar_bbox, _box_corners)
from products import (coherence, covariance_pair, covariance_matrix, multilook_intensity,
                      entropy, diffphase_statistics, diffphase_statistics_slc,
                      normalized_variance, intensity_to_rgb)
from subaperture import (SubapertureMetaData, AzimuthSubaperture,
                         construct_raised_cosine, plot_subaperture_diagnostics)

SWATH = 'science/LSAR/RSLC/swaths/frequencyA'
LOOKS = (4, 4)


# --------------------------------------------------------------------------
# ioput
# --------------------------------------------------------------------------

def test_read_block(rslc):
    """A requested window comes back at that size, complex, and with signal in it."""
    block = read_block(rslc['path'], 10, 138, 4, 68, pol='HH')
    assert block.shape == (128, 64)
    assert block.dtype == np.complex64
    assert np.isfinite(block).all()
    assert np.abs(block).mean() > 0


def test_read_block_matches_raw_dataset(rslc):
    """The reader returns the same samples as a direct h5py slice."""
    with h5py.File(rslc['path'], 'r') as fh:
        expected = fh[f'{SWATH}/HH'][10:42, 4:20]
    assert np.array_equal(read_block(rslc['path'], 10, 42, 4, 20), expected)


def test_read_range_block_spans_full_azimuth(rslc):
    assert read_range_block(rslc['path'], 0, 16, pol='HH').shape == (rslc['n_az'], 16)


def test_read_block_rejects_missing_polarization(rslc):
    with pytest.raises(KeyError):
        read_block(rslc['path'], 0, 8, 0, 8, pol='VV')


def test_get_available_pols(rslc):
    """Both polarizations are found, and the scalar datasets alongside them are not."""
    with h5py.File(rslc['path'], 'r') as fh:
        assert sorted(get_available_pols(fh, 'A')) == ['HH', 'HV']


def test_copy_h5_structure(rslc, tmp_path):
    """Copying reproduces the tree, minus explicitly skipped datasets."""
    destination = tmp_path / 'copy.h5'
    with h5py.File(rslc['path'], 'r') as src, h5py.File(destination, 'w') as dst:
        copy_h5_structure(src['/'], dst['/'], skip={src[f'{SWATH}/HH'].name})
    with h5py.File(destination, 'r') as dst:
        assert 'science/LSAR/identification/lookDirection' in dst
        assert 'science/LSAR/RSLC/metadata/orbit/position' in dst
        assert f'{SWATH}/HH' not in dst


def test_box_corners_are_centred():
    """The helper returns four corners plus the centre of the requested box."""
    lons, lats = _box_corners(-122.0, 37.5, 6.0, 6.0)
    assert len(lons) == len(lats) == 5
    assert lons[-1] == pytest.approx(-122.0)
    assert lats[-1] == pytest.approx(37.5)


def test_latlon_box_to_radar_bbox(rslc):
    """Geolocation resolves a box at the scene centre to indices inside the image."""
    bbox = latlon_box_to_radar_bbox(
        rslc['path'], rslc['lon'], rslc['lat'], width_km=0.3, height_km=0.3)

    assert 0 <= bbox['az_start'] < bbox['az_stop'] <= rslc['n_az']
    assert 0 <= bbox['rg_start'] < bbox['rg_stop'] <= rslc['n_rg']
    # a box on the scene centre should land near the middle of the grid
    assert abs((bbox['az_start'] + bbox['az_stop']) / 2 - rslc['n_az'] / 2) < rslc['n_az'] / 4
    assert abs((bbox['rg_start'] + bbox['rg_stop']) / 2 - rslc['n_rg'] / 2) < rslc['n_rg'] / 4


# --------------------------------------------------------------------------
# subaperture
# --------------------------------------------------------------------------

def test_metadata_round_trip(rslc, meta):
    """Metadata read from the product is complete and self-consistent."""
    assert meta.zero_doppler_time_spacing > 0
    assert meta.azimuth_bandwidth > 0
    # the processed bandwidth must fit inside the PRF
    assert meta.azimuth_bandwidth * meta.zero_doppler_time_spacing < 1.0
    assert meta.az_time.shape == (rslc['n_az'],)
    assert meta.slant_range.shape == (rslc['n_rg'],)
    assert meta.doppler_centroid.shape == (len(meta.doppler_az_time),
                                           len(meta.doppler_slant_range))


def test_metadata_from_open_handle(rslc):
    """Both metadata constructors agree."""
    with h5py.File(rslc['path'], 'r') as fh:
        from_handle = SubapertureMetaData.load_from_rslc(fh)
    from_path = SubapertureMetaData.load_from_rslc_path(rslc['path'])
    assert from_handle.azimuth_bandwidth == from_path.azimuth_bandwidth
    assert np.array_equal(from_handle.doppler_centroid, from_path.doppler_centroid)


@pytest.mark.parametrize('flags', [
    dict(),
    dict(demodulate_subaperture=False),
    dict(demodulate_subaperture=False, remodulate_to_full_dc=True),
    dict(normalize_window_gain=False),
    dict(deweight_spectrum=True),
], ids=['default', 'no-demod', 'remod-full-dc', 'no-gain-norm', 'deweight'])
@pytest.mark.parametrize('n_subapertures', [1, 3, 5])
def test_process_block(meta, block, flags, n_subapertures):
    """Decomposition runs for every option combination and preserves the grid."""
    sub = AzimuthSubaperture(meta, n_subapertures=n_subapertures, overlap=0.2, **flags)
    out = sub._process_block(block, az_time_start=float(meta.az_time[0]))
    assert out.shape == (n_subapertures,) + block.shape
    assert out.dtype == np.complex64
    assert np.isfinite(out).all()
    assert np.abs(out).mean() > 0


def test_subaperture_power_is_normalised(meta, block):
    """With gain normalisation on, each look carries roughly the full-aperture power."""
    sub = AzimuthSubaperture(meta, n_subapertures=3, overlap=0.2, deweight_spectrum=True)
    out = sub._process_block(block, az_time_start=float(meta.az_time[0]))
    full_power = float((np.abs(block) ** 2).mean())
    ratios = [float((np.abs(out[j]) ** 2).mean()) / full_power for j in range(3)]
    assert np.allclose(ratios, 1.0, atol=0.05), f'power ratios {ratios}'


def test_deweighting_flattens_the_antenna_taper(meta, block):
    """Without de-weighting the antenna taper skews power towards the middle look.

    The synthetic product carries a tapered azimuth spectrum, as a real one does,
    so this pins the correction actually doing something.
    """
    full_power = float((np.abs(block) ** 2).mean())

    def look_powers(deweight):
        sub = AzimuthSubaperture(meta, n_subapertures=3, overlap=0.2,
                                 deweight_spectrum=deweight)
        out = sub._process_block(block, az_time_start=float(meta.az_time[0]))
        return np.array([float((np.abs(out[j]) ** 2).mean()) / full_power for j in range(3)])

    tapered = look_powers(False)
    corrected = look_powers(True)
    # the centre look is over-weighted when the taper is left in
    assert tapered[1] > 1.2 * tapered[0]
    assert tapered[1] > 1.2 * tapered[2]
    # and the correction removes the spread
    assert corrected.max() - corrected.min() < 0.05


def test_estimate_deweighting_profile(rslc, meta):
    """The profile is estimated over the file and is positive and finite."""
    sub = AzimuthSubaperture(meta, n_subapertures=3, overlap=0.2, deweight_spectrum=True)
    profile = sub.estimate_deweighting_profile(rslc['path'], pol='HH', blocksize_range=64)
    assert profile.ndim == 1
    assert np.isfinite(profile).all()
    assert (profile > 0).all()


def test_construct_raised_cosine():
    """The window is bounded, reaches unity, and is zero outside its band."""
    win = construct_raised_cosine(1024, 512, 200, beta=0.5)
    assert win.shape == (1024,)
    assert ((win >= 0.0) & (win <= 1.0)).all()
    assert np.isclose(win.max(), 1.0)
    assert win[:512 - 100].max() == 0.0
    assert win[512 + 101:].max() == 0.0


@pytest.mark.parametrize('uniform_dc', [True, False], ids=['uniform-dc', 'interpolated-dc'])
def test_process_rslc_writes_product(rslc, meta, tmp_path, uniform_dc):
    """Both full-product entry points write a valid sub-aperture file."""
    output = tmp_path / f'subapertures_{uniform_dc}.h5'
    sub = AzimuthSubaperture(meta, n_subapertures=3, overlap=0.2)

    with pytest.warns(UserWarning, match='metadata'):
        if uniform_dc:
            sub.process_rslc_uniform_dc(rslc['path'], output, pols=['HH'], blocksize_range=64)
        else:
            sub.process_rslc(rslc['path'], output, pols=['HH'], blocksize_range=64)

    with h5py.File(output, 'r') as dst:
        data = dst[f'{SWATH}/HH']
        assert data.shape == (3, rslc['n_az'], rslc['n_rg'])
        assert data.dtype == np.complex64
        assert 'description' in data.attrs
        assert np.isfinite(data[:]).all()
        # source metadata is carried across
        assert 'science/LSAR/identification/lookDirection' in dst
        assert 'science/LSAR/RSLC/metadata/orbit/position' in dst


def test_process_rslc_matches_in_memory(rslc, meta, tmp_path):
    """Writing through the file path agrees with processing the array directly."""
    output = tmp_path / 'subapertures.h5'
    sub = AzimuthSubaperture(meta, n_subapertures=3, overlap=0.2)
    with pytest.warns(UserWarning):
        sub.process_rslc_uniform_dc(rslc['path'], output, pols=['HH'],
                                    blocksize_range=rslc['n_rg'])

    block = read_block(rslc['path'], None, None, 0, rslc['n_rg'], pol='HH')
    expected = sub._process_block(block, az_time_start=float(meta.az_time[0]))
    with h5py.File(output, 'r') as dst:
        written = dst[f'{SWATH}/HH'][:]
    assert np.allclose(written, expected, atol=1e-4)


# --------------------------------------------------------------------------
# products
# --------------------------------------------------------------------------

def test_multilook_intensity(stack):
    out = multilook_intensity(stack, LOOKS)
    assert out.shape == (stack.shape[0], stack.shape[1] // LOOKS[0], stack.shape[2] // LOOKS[1])
    assert np.isrealobj(out) and (out >= 0).all() and np.isfinite(out).all()


def test_coherence(stack):
    """Coherence is bounded, and a look is perfectly coherent with itself."""
    out = coherence(stack[0], stack[1], LOOKS)
    assert np.isfinite(out).all()
    assert ((out >= 0) & (out <= 1 + 1e-6)).all()
    assert np.allclose(coherence(stack[0], stack[0], LOOKS), 1.0, atol=1e-5)


def test_covariance_pair_matches_matrix(stack):
    """The pairwise and full-matrix paths agree."""
    cov, coh = covariance_matrix(stack, LOOKS)
    cov_01, coh_01 = covariance_pair(stack[0], stack[1], LOOKS)
    assert np.allclose(cov[0, 1], cov_01, atol=1e-4, rtol=1e-4)
    assert np.allclose(coh[0, 1], coh_01, atol=1e-4)


def test_covariance_matrix_is_hermitian(stack):
    cov, coh = covariance_matrix(stack, LOOKS)
    assert np.allclose(cov, np.conj(np.moveaxis(cov, 0, 1)), rtol=1e-4, atol=1e-4)
    assert np.allclose(np.diagonal(coh, axis1=0, axis2=1), 1.0, atol=1e-5)


def test_entropy(stack):
    """Entropy is finite and inside [0, log(n_sub)]."""
    n_sub = stack.shape[0]
    cov, _ = covariance_matrix(stack, LOOKS)
    values = entropy(cov)
    assert values.shape == cov.shape[2:]
    assert np.isfinite(values).all()
    assert ((values >= -1e-9) & (values <= np.log(n_sub) + 1e-6)).all()


def test_diffphase_statistics(stack):
    """Both phase-statistics paths give wrapped phase and bounded circular variance."""
    cov, _ = covariance_matrix(stack, LOOKS)
    for mean_phase, variance in (diffphase_statistics(cov), diffphase_statistics_slc(stack)):
        assert np.isfinite(mean_phase).all() and np.isfinite(variance).all()
        assert ((mean_phase >= -np.pi) & (mean_phase <= np.pi)).all()
        assert ((variance >= -1e-9) & (variance <= 1 + 1e-6)).all()


def test_normalized_variance(stack):
    out = normalized_variance(stack)
    assert out.shape == stack.shape[1:]
    assert np.isfinite(out).all() and (out >= 0).all()


def test_intensity_to_rgb(stack):
    out = intensity_to_rgb(np.abs(stack) ** 2, gamma=2.5)
    assert out.shape == stack.shape[1:] + (3,)
    assert out.dtype == np.uint8


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------

def test_point_target_simulation_compresses(meta):
    """A simulated point target compresses to a peak at the expected sample."""
    from simulation import (azimuth_fm_rate, simulate_phase_history, azimuth_compress,
                            slant_range_history, azimuth_chirp,
                            effective_velocity, wavelength)

    dt = meta.zero_doppler_time_spacing
    bandwidth = meta.azimuth_bandwidth
    slant_range = float(np.mean(meta.slant_range))
    fm_rate = azimuth_fm_rate(effective_velocity, wavelength, slant_range)
    n_az = 4096

    times = (np.arange(n_az) - n_az // 2) * dt
    assert np.isfinite(slant_range_history(times, slant_range, effective_velocity)).all()
    assert np.isfinite(azimuth_chirp(times, fm_rate, bandwidth, 0.0)).all()

    _, raw = simulate_phase_history(n_az, dt, slant_range, effective_velocity,
                                    wavelength, bandwidth, 0.0)
    slc = azimuth_compress(raw, dt, fm_rate, bandwidth, 0.0)
    assert np.isfinite(slc).all()
    # a stationary target focuses at the centre sample
    assert abs(int(np.argmax(np.abs(slc))) - n_az // 2) <= 1


def test_moving_target_shifts_in_azimuth(meta):
    """A radial velocity displaces the compressed peak by the predicted amount."""
    from simulation import (azimuth_fm_rate, simulate_phase_history, azimuth_compress,
                            effective_velocity, wavelength)

    dt = meta.zero_doppler_time_spacing
    bandwidth = meta.azimuth_bandwidth
    slant_range = float(np.mean(meta.slant_range))
    fm_rate = azimuth_fm_rate(effective_velocity, wavelength, slant_range)
    n_az = 8192
    v_range = 2.0

    _, raw = simulate_phase_history(n_az, dt, slant_range, effective_velocity, wavelength,
                                    bandwidth, 0.0, v_range=v_range)
    slc = azimuth_compress(raw, dt, fm_rate, bandwidth, 0.0)
    measured = int(np.argmax(np.abs(slc))) - n_az // 2
    predicted = (2.0 * v_range / wavelength) / (fm_rate * dt)
    assert abs(measured - predicted) <= max(2.0, 0.1 * abs(predicted))


# --------------------------------------------------------------------------
# plotting entry points
# --------------------------------------------------------------------------

def test_plot_subaperture_diagnostics(sub, block, figure_dir):
    path = figure_dir / 'diagnostics.png'
    plot_subaperture_diagnostics(sub, block, save_path=path)
    assert path.stat().st_size > 0


def test_plot_simulation(meta, sub, figure_dir):
    from simulation import (azimuth_fm_rate, simulate_phase_history, azimuth_compress,
                            plot_simulation, effective_velocity, wavelength)

    dt = meta.zero_doppler_time_spacing
    slant_range = float(np.mean(meta.slant_range))
    fm_rate = azimuth_fm_rate(effective_velocity, wavelength, slant_range)
    times, raw = simulate_phase_history(4096, dt, slant_range, effective_velocity,
                                        wavelength, meta.azimuth_bandwidth, 0.0)
    slc = azimuth_compress(raw, dt, fm_rate, meta.azimuth_bandwidth, 0.0)
    stack = sub._process_block(slc[:, None])[:, :, 0]

    path = figure_dir / 'simulation.png'
    plot_simulation(times, raw, slc, stack, save_path=path)
    assert path.stat().st_size > 0


def test_presentation_figures(figure_dir):
    """The slide-deck figure functions still run end to end."""
    from figures_presentation import (make_meta, subaperture_values,
                                      plot_moving_target_phase, plot_phase_vs_subaperture,
                                      plot_clutter_phase)
    from subaperture import AzimuthSubaperture

    assert make_meta(1 / 1520.0, 1264.0).azimuth_bandwidth == 1264.0

    plot_moving_target_phase(figure_dir / 'moving_target.png', slant_range=900e3,
                             bandwidth=1264.0, doppler_centroid=0.0, prf=1520.0,
                             n_subapertures=5, overlap=0.2, v_range=0.03, v_az=10.0)
    plot_phase_vs_subaperture(figure_dir / 'phase_vs_sub.png', bandwidth=1264.0,
                              n_subapertures=5, dx_targets=(0.3, -0.2))
    plot_clutter_phase(figure_dir / 'clutter.png', bandwidth=1264.0, n_subapertures=5)

    for name in ['moving_target.png', 'phase_vs_sub.png', 'clutter.png']:
        assert (figure_dir / name).stat().st_size > 0

    # subaperture_values returns one frequency and one complex sample per look
    sub = AzimuthSubaperture(make_meta(1 / 1520.0, 1264.0), n_subapertures=3, overlap=0.2)
    slc = np.zeros(2048, dtype=np.complex64)
    slc[1024] = 1.0
    freqs, values = subaperture_values(sub, slc, np.arange(2048) / 1520.0, 1024)
    assert freqs.shape == values.shape == (3,)
    assert np.isfinite(values).all()
