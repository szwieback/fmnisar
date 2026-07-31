"""Figures for slide decks explaining sub-aperture processing."""

from pathlib import Path

import numpy as np
from scipy.fft import next_fast_len, fftfreq, fftshift
import matplotlib.pyplot as plt

from simulation import azimuth_fm_rate, simulate_phase_history, azimuth_compress, effective_velocity, wavelength
from subaperture import SubapertureMetaData, AzimuthSubaperture


def make_meta(dt: float, bandwidth: float) -> SubapertureMetaData:
    return SubapertureMetaData(
        zero_doppler_time_spacing=dt,
        azimuth_bandwidth=bandwidth,
        doppler_centroid=np.zeros((2, 2)),
        doppler_az_time=np.zeros(2),
        doppler_slant_range=np.zeros(2),
        az_time=np.zeros(2),
        slant_range=np.zeros(2),
    )


def subaperture_values(sub: AzimuthSubaperture, slc: np.ndarray, t: np.ndarray, peak: int):
    # evaluate subaperture slc at peak
    slc_sub = sub._process_block(slc[:, None], az_time_start=float(t[0]))[:, :, 0]
    n_pad = next_fast_len(slc.shape[0])
    _, bin_centroids, _, _, _ = sub._compute_subaperture_params(
        n_pad, sub.meta.azimuth_bandwidth, sub.meta.zero_doppler_time_spacing)
    freqs = fftshift(fftfreq(n_pad, d=sub.meta.zero_doppler_time_spacing))[bin_centroids]
    return freqs, slc_sub[:, peak]


def plot_moving_target_phase(save_path: Path, slant_range: float, bandwidth: float, doppler_centroid: float,
                              prf: float, n_subapertures: int, overlap: float, v_range: float, v_az: float):
    """Wrapped sub-aperture phase for a moving point target, focused with a stationary matched filter."""
    dt = 1 / prf

    fm_rate = azimuth_fm_rate(effective_velocity, wavelength, slant_range)
    aperture_time = bandwidth / abs(fm_rate)
    n_az = next_fast_len(int(4 * aperture_time / dt))
    peak = n_az // 2

    meta = make_meta(dt, bandwidth)
    sub = AzimuthSubaperture(
        meta, n_subapertures=n_subapertures, overlap=overlap,
        demodulate_subaperture=True, deweight_spectrum=False, remodulate_to_full_dc=False)

    cases = [
        ('stationary target', dict(v_range=0.0, v_az=0.0), 'k', 'o'),
        (f'radial velocity ($v_r$={v_range*100:.0f} cm/s)', dict(v_range=v_range, v_az=0.0), 'C0', 's'),
        (f'azimuth velocity ($v_{{az}}$={v_az:.0f} m/s)', dict(v_range=0.0, v_az=v_az), 'C3', '^'),
    ]

    t0, raw0 = simulate_phase_history(
        n_az, dt, slant_range, effective_velocity, wavelength, bandwidth, doppler_centroid)
    slc0 = azimuth_compress(raw0, dt, fm_rate, bandwidth, doppler_centroid)
    _, val_stationary = subaperture_values(sub, slc0, t0, peak)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for label, v, color, marker in cases:
        t, raw = simulate_phase_history(
            n_az, dt, slant_range, effective_velocity, wavelength, bandwidth, doppler_centroid,
            v_range=v['v_range'], v_az=v['v_az'])
        slc = azimuth_compress(raw, dt, fm_rate, bandwidth, doppler_centroid)
        freqs, val = subaperture_values(sub, slc, t, peak)
        # phase relative to the stationary case
        phase = np.angle(val * val_stationary.conj())
        ax.plot(freqs, phase, marker=marker, color=color, label=label, lw=1.5, ms=6)

    ax.axhline(0, color='0.85', lw=0.8, zorder=0)
    ax.set_xlabel('sub-aperture center frequency (Hz)')
    ax.set_ylabel('wrapped phase (rad)')
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_yticklabels(['$-\\pi$', '$-\\pi/2$', '0', '$\\pi/2$', '$\\pi$'])
    ax.legend(fontsize=8, loc='upper left', framealpha=1)
    ax.set_title('Simulated wrapped phase vs. sub-aperture for a moving target', fontsize=10)
    fig.tight_layout()

    save_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path, dpi=150, facecolor='white')
    plt.close(fig)
    print(f'saved {save_path}')


def plot_phase_vs_subaperture(save_path: Path, bandwidth: float, n_subapertures: int, dx_targets):
    """Shift-theorem phase-vs-Doppler-frequency lines for two targets at different sub-pixel azimuth offsets."""
    resolution = effective_velocity / bandwidth
    f = np.linspace(-bandwidth / 2, bandwidth / 2, 400)
    sub_centers = np.linspace(-bandwidth / 3, bandwidth / 3, n_subapertures)

    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor='white')
    colors = ['#1f77b4', '#d62728']
    for dx, color in zip(dx_targets, colors):
        t0 = dx * resolution / effective_velocity
        phase = -2 * np.pi * f * t0
        ax.plot(f, phase, color=color, lw=2, label=f'target at {dx:+.1f} px')
        ax.plot(sub_centers, -2 * np.pi * sub_centers * t0, 'o', color=color, ms=7, zorder=3)

    ax.set_xlabel('azimuth (Doppler) frequency (Hz)', fontsize=12)
    ax.set_ylabel('phase (rad)', fontsize=12)
    ax.set_title('Phase vs. sub-aperture frequency: stationary point targets', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11, loc='best', frameon=False)
    ax.set_facecolor('white')
    fig.tight_layout()

    save_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path, dpi=150, facecolor='white')
    plt.close(fig)
    print(f'saved {save_path}')


def plot_clutter_phase(save_path: Path, bandwidth: float, n_subapertures: int, n_scenarios: int = 5, seed: int = 0):
    """Wrapped phase vs. sub-aperture for pure clutter (iid circular Gaussian sub-aperture SLC), for demonstration only."""
    rng = np.random.default_rng(seed)
    sub_centers = np.linspace(-bandwidth / 3, bandwidth / 3, n_subapertures)

    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor='white')
    for n in range(n_scenarios):
        slc = rng.standard_normal(n_subapertures) + 1j * rng.standard_normal(n_subapertures)
        phase = np.angle(slc)
        ax.plot(sub_centers, phase, marker='o', ms=6, lw=1, color=f'C{n}', label=f'clutter realization {n + 1}')

    ax.axhline(0, color='0.85', lw=0.8, zorder=0)
    ax.set_xlabel('sub-aperture center frequency (Hz)')
    ax.set_ylabel('wrapped phase (rad)')
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_yticklabels(['$-\\pi$', '$-\\pi/2$', '0', '$\\pi/2$', '$\\pi$'])
    ax.legend(fontsize=8, loc='upper left', framealpha=1, ncol=2)
    ax.set_title('Simulated wrapped phase vs. sub-aperture for pure clutter (iid)', fontsize=10)
    fig.tight_layout()

    save_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path, dpi=150, facecolor='white')
    plt.close(fig)
    print(f'saved {save_path}')


if __name__ == '__main__':
    folder_out = Path('/media/simon/Extreme SSD/fmnisar/slides/figures')
    slant_range = 800e3
    bandwidth = 1200.0        # Hz, processed azimuth bandwidth
    doppler_centroid = 0.0
    prf = 3000.0               # Hz
    n_subapertures = 9
    overlap = 0.2
    v_range, v_az = 0.03, 10.0   # m/s
    dx_targets = (0.3, -0.2)     # sub-pixel azimuth offsets, in pixels

    plot_moving_target_phase(
        folder_out / 'moving_target_phase.png', slant_range=slant_range, bandwidth=bandwidth,
        doppler_centroid=doppler_centroid, prf=prf, n_subapertures=n_subapertures, overlap=overlap,
        v_range=v_range, v_az=v_az)

    plot_phase_vs_subaperture(
        folder_out / 'schematic_phase_vs_subaperture.png', bandwidth=bandwidth,
        n_subapertures=n_subapertures, dx_targets=dx_targets)

    plot_clutter_phase(
        folder_out / 'clutter_phase.png', bandwidth=bandwidth, n_subapertures=n_subapertures)
