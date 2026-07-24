"""1-D azimuth simulation: raw point-target phase history, matched-filter SLC, and sub-aperture looks."""

from pathlib import Path

import numpy as np
from scipy.fft import fft, ifft, fftshift, next_fast_len

from subaperture import SubapertureMetaData, AzimuthSubaperture, plot_subaperture_diagnostics


# physical parameters not carried in the RSLC metadata (guessed NISAR L-band values)
effective_velocity = 7500.0   # m/s
wavelength = 0.24             # m


def azimuth_fm_rate(velocity: float, wavelength: float, slant_range: float) -> float:
    """Azimuth FM rate Ka [Hz/s] for a broadside point target."""
    return -2.0 * velocity ** 2 / (wavelength * slant_range)


def slant_range_history(t: np.ndarray, slant_range: float, velocity: float, t0: float = 0.0,
                        v_range: float = 0.0, v_az: float = 0.0) -> np.ndarray:
    """Slant-range history R(t) [m] of a point target in uniform motion (v=0 -> broadside stationary)."""
    tau = t - t0
    return np.sqrt((slant_range + v_range * tau) ** 2 + ((velocity - v_az) * tau) ** 2)


def azimuth_chirp(t: np.ndarray, fm_rate: float, bandwidth: float, doppler_centroid: float,
                  t0: float = 0.0) -> np.ndarray:
    """Windowed azimuth chirp of a point target: instantaneous Doppler fdc + Ka*(t-t0) over its aperture."""
    aperture_time = bandwidth / abs(fm_rate)
    window = np.abs(t - t0) <= aperture_time / 2
    phase = np.pi * fm_rate * (t - t0) ** 2 + 2 * np.pi * doppler_centroid * (t - t0)
    return (window * np.exp(1j * phase)).astype(np.complex64)


def simulate_phase_history(n_az: int, dt: float, slant_range: float, velocity: float,
                           wavelength: float, bandwidth: float, doppler_centroid: float,
                           t0: float = 0.0, v_range: float = 0.0, v_az: float = 0.0,
                           noise: float = 0.0):
    """Raw azimuth phase history of a (possibly moving) point target from its slant-range history R(t)."""
    t = (np.arange(n_az) - n_az // 2) * dt
    R = slant_range_history(t, slant_range, velocity, t0=t0, v_range=v_range, v_az=v_az)
    tau = t - t0
    fm_rate = azimuth_fm_rate(velocity - v_az, wavelength, slant_range)   # sets illumination window only
    window = np.abs(tau) <= (bandwidth / abs(fm_rate)) / 2
    phase = -4.0 * np.pi / wavelength * (R - slant_range) + 2.0 * np.pi * doppler_centroid * tau
    raw = window * np.exp(1j * phase)
    if noise:
        raw = raw + noise * (np.random.randn(n_az) + 1j * np.random.randn(n_az)) / np.sqrt(2)
    return t, raw.astype(np.complex64)


def azimuth_compress(raw: np.ndarray, dt: float, fm_rate: float, bandwidth: float,
                   doppler_centroid: float) -> np.ndarray:
    """Azimuth-compress a raw phase history to an SLC via a chirp matched filter (peak at the target sample)."""
    n = raw.shape[0]
    t = (np.arange(n) - n // 2) * dt
    replica = azimuth_chirp(t, fm_rate, bandwidth, doppler_centroid, t0=0.0)
    # frequency-domain cross-correlation; fftshift moves the zero-lag peak to the centre sample
    slc = fftshift(ifft(fft(raw) * np.conj(fft(replica))))
    return slc.astype(np.complex64)


def plot_simulation(t, raw, slc, slc_sub, save_path: Path = None):
    """Three-panel view: raw chirp, compressed SLC, and broadened sub-aperture slc_sub."""
    import matplotlib.pyplot as plt

    n = t.shape[0]
    ref = np.abs(slc).max()

    def db(x):
        return 20 * np.log10(np.maximum(np.abs(x) / ref, 1e-6))

    # zoom around the compressed peak (may be shifted off-centre for a moving target)
    peak = int(np.argmax(np.abs(slc)))
    half = min(n // 2, max(64, int(0.02 * n)))
    sl = slice(max(0, peak - half), min(n, peak + half))
    t_center = t[peak]

    n_sub = slc_sub.shape[0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_sub))

    fig, (ax_slc, ax_sub, ax_phase) = plt.subplots(3, 1, figsize=(8, 10))

    ax_slc.plot(t[sl], db(slc[sl]), c='k')
    ax_slc.set_xlabel('azimuth time (s)')
    ax_slc.set_ylabel('magnitude (dB)')
    ax_slc.set_xlim(t_center - 0.03, t_center + 0.03)
    ax_slc.set_ylim(-20, 0)
    ax_slc.set_title('Matched-filter SLC (compressed impulse)')

    for j, color in enumerate(colors):
        ax_sub.plot(t[sl], db(slc_sub[j][sl]), color=color, label=f'look {j + 1}', alpha=0.9)
    ax_sub.set_xlabel('azimuth time (s)')
    ax_sub.set_ylabel('magnitude (dB)')
    ax_sub.set_xlim(t_center - 0.03, t_center + 0.03)
    ax_sub.set_ylim(-20, 0)
    ax_sub.set_title('Sub-aperture slc_sub (broadened mainlobes)')
    ax_sub.legend(fontsize=8)

    for j, color in enumerate(colors):
        ax_phase.plot(t[sl], np.angle(slc_sub[j][sl]), color=color, label=f'look {j + 1}', alpha=0.9)
    ax_phase.set_xlabel('azimuth time (s)')
    ax_phase.set_ylabel('phase (rad)')
    ax_phase.set_xlim(t_center - 0.03, t_center + 0.03)
    ax_phase.set_ylim(-np.pi, np.pi)
    ax_phase.set_title('Sub-aperture phase (wrapped)')
    ax_phase.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    path_nisar = Path(
        '/media/simon/Extreme SSD/fmnisar/Sydney/'
        'NISAR_L1_PR_RSLC_025_016_D_105_4005_DHDH_A_20260709T075915_20260709T075950_P05023_N_F_J_001.h5')
    folder_out = path_nisar.parent / 'processed'
    folder_out.mkdir(exist_ok=True)

    n_subapertures = 3
    overlap = 0.2

    meta = SubapertureMetaData.load_from_rslc_path(path_nisar)
    # meta.doppler_centroid = 0.0 * meta.doppler_centroid
    dt = meta.zero_doppler_time_spacing
    bandwidth = meta.azimuth_bandwidth
    doppler_centroid = float(np.mean(meta.doppler_centroid))
    slant_range = float(np.mean(meta.slant_range))

    fm_rate = azimuth_fm_rate(effective_velocity, wavelength, slant_range)
    aperture_time = bandwidth / abs(fm_rate)
    aperture_samples = aperture_time / dt
    n_az = next_fast_len(int(4 * aperture_samples))

    print(f'PRF: {1 / dt:.1f} Hz, azimuth bandwidth: {bandwidth:.1f} Hz, '
          f'doppler centroid: {doppler_centroid:.1f} Hz')
    print(f'slant range: {slant_range / 1e3:.1f} km, Ka: {fm_rate:.1f} Hz/s')
    print(f'aperture: {aperture_time:.3f} s ({aperture_samples:.0f} samples), '
          f'azimuth resolution: {effective_velocity / bandwidth:.2f} m')
    print(f'image length: {n_az} samples')

    v_range, v_az = 0.0, 5.0 # m/s
    # raw phase history from the moving-target geometry, compressed with the stationary filter
    t, raw = simulate_phase_history(n_az, dt, slant_range, effective_velocity, wavelength,
                                    bandwidth, doppler_centroid, t0=0*1e-3,
                                    v_range=v_range, v_az=v_az)
    slc = azimuth_compress(raw, dt, fm_rate, bandwidth, doppler_centroid)
    peak = int(np.argmax(np.abs(slc)))
    shift_pred = (2.0 * v_range / wavelength) / (fm_rate * dt)   # [samples]
    print(f'moving target: v_range={v_range} m/s, v_az={v_az} m/s')
    print(f'SLC peak sample: {peak} (centre: {n_az // 2}), '
          f'shift: {peak - n_az // 2} (predicted: {shift_pred:.1f})')

    # azimuth sub-aperture decomposition of the 1-D SLC
    sub = AzimuthSubaperture(
        meta, n_subapertures=n_subapertures, overlap=overlap,
        demodulate_subaperture=True, deweight_spectrum=False, remodulate_to_full_dc=False)
    slc_sub = sub._process_block(slc[:, None], az_time_start=float(t[0]))[:, :, 0]
    print(np.angle(slc[peak]), np.angle(slc_sub[:, peak]))
    # adjacent phase difference
    # expect ~uniform for v_az = 0, with value [slope of ramp] depending on subpixel az position and v_r
    print(np.angle(slc_sub[1:, peak] * slc_sub[:-1, peak].conj()))


    plot_simulation(t, raw, slc, slc_sub, save_path=folder_out / 'simulation_pointtarget.png')
    plot_subaperture_diagnostics(
        sub, slc[:, None], save_path=folder_out / 'simulation_subaperture_diagnostics.png')
