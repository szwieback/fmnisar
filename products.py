"""Coherence and RGB composite products derived from sub-aperture SLC stacks."""

import numpy as np


def _boxcar_multilook(x: np.ndarray, az_looks: int, rg_looks: int) -> np.ndarray:
    """Crop the trailing two axes to a multiple of the look window and boxcar-average over it."""
    *lead, n_az, n_rg = x.shape
    n_az_ml = n_az // az_looks
    n_rg_ml = n_rg // rg_looks
    cropped = x[..., :n_az_ml * az_looks, :n_rg_ml * rg_looks]
    reshaped = cropped.reshape(*lead, n_az_ml, az_looks, n_rg_ml, rg_looks)
    return reshaped.mean(axis=(-3, -1))


def estimate_coherence(look_a: np.ndarray, look_b: np.ndarray, az_looks: int, rg_looks: int) -> np.ndarray:
    """Estimate coherence magnitude between two complex looks over a boxcar window.

    look_a, look_b are complex arrays of shape (n_az, n_rg). az_looks/rg_looks
    is the size of the non-overlapping estimation window; coherence at 1x1 is
    trivially unity, so a window > 1 is required.

    Returns
    -------
    coherence : numpy.ndarray, shape (n_az // az_looks, n_rg // rg_looks)
    """
    _, coherence = estimate_covariance_pair(look_a, look_b, (az_looks, rg_looks))
    return coherence


def estimate_covariance_pair(
    look_a: np.ndarray, look_b: np.ndarray, looks: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Boxcar complex covariance and derived coherence magnitude for one pair of looks.

    look_a, look_b are complex arrays of shape (n_az, n_rg); looks is (az_looks, rg_looks),
    the non-overlapping estimation window (coherence at 1x1 is trivially unity).

    Returns
    -------
    covariance : numpy.ndarray, complex, shape (n_az // az_looks, n_rg // rg_looks)
    coherence : numpy.ndarray, real, same shape
    """
    az_looks, rg_looks = looks
    covariance = _boxcar_multilook(look_a * np.conj(look_b), az_looks, rg_looks)
    power_a = _boxcar_multilook(np.abs(look_a) ** 2, az_looks, rg_looks)
    power_b = _boxcar_multilook(np.abs(look_b) ** 2, az_looks, rg_looks)
    coherence = np.abs(covariance) / np.sqrt(power_a * power_b)
    return covariance, coherence


def estimate_covariance_matrix(
    block_sub: np.ndarray, looks: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Full pairwise complex covariance and coherence-magnitude matrices across a look stack.

    block_sub is a complex look stack of shape (n_looks, n_az, n_rg); looks is
    (az_looks, rg_looks), the non-overlapping boxcar window. Diagonal terms are the real
    power of each look; off-diagonal terms are the complex cross terms mean(look_i * conj(look_j)).

    Returns
    -------
    covariance : numpy.ndarray, complex, shape (n_looks, n_looks, n_az_ml, n_rg_ml)
    coherence : numpy.ndarray, real, shape (n_looks, n_looks, n_az_ml, n_rg_ml)
    """
    az_looks, rg_looks = looks
    n_looks = block_sub.shape[0]
    outer = block_sub[:, None, :, :] * np.conj(block_sub[None, :, :, :])
    covariance = _boxcar_multilook(outer, az_looks, rg_looks)
    diag_idx = np.arange(n_looks)
    power = np.real(covariance[diag_idx, diag_idx])
    covariance[diag_idx, diag_idx] = power
    coherence = np.abs(covariance) / np.sqrt(power[:, None] * power[None, :])
    return covariance, coherence


def multilook_intensity(block_sub: np.ndarray, az_looks: int, rg_looks: int) -> np.ndarray:
    """Boxcar-average intensity over non-overlapping az/range look windows.

    block_sub is a complex look stack of shape (n_looks, n_az, n_rg).

    Returns
    -------
    intensity : numpy.ndarray, shape (n_looks, n_az // az_looks, n_rg // rg_looks)
    """
    return _boxcar_multilook(np.abs(block_sub) ** 2, az_looks, rg_looks)


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
