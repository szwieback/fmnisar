"""Coherence and RGB composite products derived from sub-aperture SLC stacks."""

import numpy as np


def estimate_coherence(look_a: np.ndarray, look_b: np.ndarray, az_looks: int, rg_looks: int) -> np.ndarray:
    """Estimate coherence magnitude between two complex looks over a boxcar window.

    look_a, look_b are complex arrays of shape (n_az, n_rg). az_looks/rg_looks
    is the size of the non-overlapping estimation window; coherence at 1x1 is
    trivially unity, so a window > 1 is required.

    Returns
    -------
    coherence : numpy.ndarray, shape (n_az // az_looks, n_rg // rg_looks)
    """
    n_az, n_rg = look_a.shape
    n_az_ml = n_az // az_looks
    n_rg_ml = n_rg // rg_looks
    a = look_a[:n_az_ml * az_looks, :n_rg_ml * rg_looks].reshape(n_az_ml, az_looks, n_rg_ml, rg_looks)
    b = look_b[:n_az_ml * az_looks, :n_rg_ml * rg_looks].reshape(n_az_ml, az_looks, n_rg_ml, rg_looks)
    cross = (a * np.conj(b)).mean(axis=(1, 3))
    power_a = (np.abs(a) ** 2).mean(axis=(1, 3))
    power_b = (np.abs(b) ** 2).mean(axis=(1, 3))
    return np.abs(cross) / np.sqrt(power_a * power_b)


def multilook_intensity(block_sub: np.ndarray, az_looks: int, rg_looks: int) -> np.ndarray:
    """Boxcar-average intensity over non-overlapping az/range look windows.

    block_sub is a complex look stack of shape (n_looks, n_az, n_rg).

    Returns
    -------
    intensity : numpy.ndarray, shape (n_looks, n_az // az_looks, n_rg // rg_looks)
    """
    n_looks, n_az, n_rg = block_sub.shape
    n_az_ml = n_az // az_looks
    n_rg_ml = n_rg // rg_looks
    intensity = np.abs(block_sub[:, :n_az_ml * az_looks, :n_rg_ml * rg_looks]) ** 2
    intensity = intensity.reshape(n_looks, n_az_ml, az_looks, n_rg_ml, rg_looks)
    return intensity.mean(axis=(2, 4))


def intensity_to_rgb(
    intensity: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0, gamma: float = 1.0
) -> np.ndarray:
    """Percentile-stretch a 3-channel intensity stack into an 8-bit RGB image.

    intensity has shape (3, n_az, n_rg); channel order maps to R, G, B.
    low_pct/high_pct are the per-channel percentiles used to clip/stretch to
    [0, 1]; gamma > 1 brightens mid/low tones (gamma=1 leaves it unchanged).

    Returns
    -------
    rgb : numpy.ndarray, shape (n_az, n_rg, 3), uint8
    """
    channels = []
    for c in range(intensity.shape[0]):
        chan = intensity[c]
        lo, hi = np.percentile(chan, [low_pct, high_pct])
        chan = np.clip((chan - lo) / (hi - lo), 0.0, 1.0)
        if gamma != 1.0:
            chan = chan ** (1.0 / gamma)
        channels.append(chan)
    rgb = np.stack(channels, axis=-1)
    return (rgb * 255).astype(np.uint8)
