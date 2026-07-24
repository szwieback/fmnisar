import numpy as np


def _boxcar_multilook(slc: np.ndarray, looks: tuple[int, int]) -> np.ndarray:
    # full stride, based on reshape
    *lead, n_az, n_rg = slc.shape
    az_looks, rg_looks = looks
    n_az_ml = n_az // az_looks
    n_rg_ml = n_rg // rg_looks
    cropped = slc[..., :n_az_ml * az_looks, :n_rg_ml * rg_looks]
    reshaped = cropped.reshape(*lead, n_az_ml, az_looks, n_rg_ml, rg_looks)
    return reshaped.mean(axis=(-3, -1))


def coherence(slc1: np.ndarray, slc2: np.ndarray, looks: tuple[int, int]) -> np.ndarray:
    _, coherence = covariance_pair(slc1, slc2, looks)
    return coherence


def covariance_pair(
    slc1: np.ndarray, slc2: np.ndarray, looks: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    covariance = _boxcar_multilook(slc1 * np.conj(slc2), looks) # complex
    power_a = _boxcar_multilook(np.abs(slc1) ** 2, looks)
    power_b = _boxcar_multilook(np.abs(slc2) ** 2, looks)
    coherence = np.abs(covariance) / np.sqrt(power_a * power_b) # magnitude
    return covariance, coherence


def covariance_matrix(
        slc_sub: np.ndarray, looks: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Full pairwise complex covariance and coherence-magnitude matrices across a look stack.
    slc_sub is a complex look stack of shape (n_sub, n_az, n_rg)
    
    Returns
    -------
    covariance : numpy.ndarray, complex, shape (n_sub, n_sub, n_az_ml, n_rg_ml)
    coherence : numpy.ndarray, real, shape (n_sub, n_sub, n_az_ml, n_rg_ml)
    """
    n_sub = slc_sub.shape[0]
    outer = slc_sub[:, None, :, :] * np.conj(slc_sub[None, :, :, :])
    covariance = _boxcar_multilook(outer, looks)
    diag_idx = np.arange(n_sub)
    power = np.real(covariance[diag_idx, diag_idx])
    covariance[diag_idx, diag_idx] = power
    coherence = np.abs(covariance) / np.sqrt(power[:, None] * power[None, :])
    return covariance, coherence


def multilook_intensity(slc_sub: np.ndarray, looks: tuple[int, int]) -> np.ndarray:
    return _boxcar_multilook(np.abs(slc_sub) ** 2, looks)

def diffphase_statistics(covariance_sub):
    # actually inter-aperture phase mean/variance from normalized phasors
    # check sarpy implementation [nonuniform aperture spacing]
    cv_nn = np.diagonal(covariance_sub, offset=1, axis1=0, axis2=1) # nearest subaperture neighb. covariance
    cross = cv_nn[..., :-1] * cv_nn[..., 1:].conj()
    cross_phasor = cross / np.abs(cross)
    mean_phasor = np.mean(cross_phasor, axis=-1)
    M = np.angle(mean_phasor)
    V = 1 - np.abs(mean_phasor)
    return M, V
    

def intensity_to_rgb(
    intensity: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0, gamma: float = 1.0
) -> np.ndarray:
    #Percentile-stretch a 3-channel intensity of shape (3, H, W) into an 8-bit RGB image.
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
