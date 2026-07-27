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


def entropy(covariance: np.ndarray) -> np.ndarray:
    """Shannon entropy (nats) of the normalized covariance eigenvalue spectrum, per pixel."""
    mat = np.moveaxis(covariance, (0, 1), (-2, -1))   # (n_az_ml, n_rg_ml, n_sub, n_sub)
    eigvals = np.linalg.eigvalsh(mat)
    eigvals = np.clip(eigvals, 0.0, None)
    p = eigvals / eigvals.sum(axis=-1, keepdims=True)
    return -np.sum(p * np.log(np.where(p > 0, p, 1.0)), axis=-1)  # (n_az_ml, n_rg_ml)

def _diffphase_from_nn_covariance(cv_nn, sub_axis=-1):
    """Inter-aperture phase mean/variance from nearest-neighbour covariances along sub_axis."""
    # check sarpy implementation [nonuniform aperture spacing, not working on nearest neighb.]
    phasor = cv_nn / np.abs(cv_nn)
    mean_phasor = np.mean(phasor, axis=sub_axis)
    M = np.angle(mean_phasor)
    V = 1 - np.abs(mean_phasor)
    return M, V


def diffphase_statistics(covariance_sub):
    """Inter-aperture phase mean/variance from a multilooked sub-aperture covariance matrix."""
    cv_nn = np.diagonal(covariance_sub, offset=1, axis1=0, axis2=1) # nearest subaperture neighb. covariance
    return _diffphase_from_nn_covariance(cv_nn, sub_axis=-1)


def diffphase_statistics_slc(slc_sub):
    #full resolution; directly from slc_sub
    cv_nn = slc_sub[:-1] * slc_sub[1:].conj() # nearest neighb. covariance, shape (n_sub-1, n_az, n_rg)
    return _diffphase_from_nn_covariance(cv_nn, sub_axis=0)
    

def normalized_variance(slc_sub: np.ndarray) -> np.ndarray:
    #Variance of single-look log-intensity across sub-apertures
    log_intensity = np.log(np.abs(slc_sub) ** 2)
    # log_intensity is Gumbel with Var = pi^2/6 for Gaussian speckle
    return log_intensity.var(axis=0, ddof=1)


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
