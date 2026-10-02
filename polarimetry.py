"""Polarimetric decompositions of NISAR RSLC data."""

import numpy as np

from products import covariance_matrix


def dualpol_h_alpha(
    copol: np.ndarray, crosspol: np.ndarray, looks: tuple[int, int],
    stride: tuple[int, int] | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dual-pol entropy/alpha decomposition of a co-pol and cross-pol channel pair.

    Follows Cloude (2007), "The dual polarisation entropy/alpha decomposition: a PALSAR
    case study", eqs. 1 and 2. For each pixel the two channels are averaged over a window
    into a 2x2 matrix J of their powers and cross-product. Its two eigenvalues, scaled to
    sum to one, say how much of the return sits in the dominant scattering state versus
    the leftover one, and its principal eigenvector says what that dominant state is.

    entropy measures how evenly the return is split between the two states. 0 means one
    clean scattering state, 1 means the two channels are equal in power and unrelated,
    as for noise. Volume scatter such as forest tops out near 0.81.

    alpha is how far the dominant return has rotated from the transmitted polarisation.
    0 means all co-pol (smooth surface), pi/2 means all cross-pol, pi/4 for noise. It is
    the probability-weighted mean over both states, following the paper.

    cross_co_phase_difference is the phase of the cross-pol channel relative to the
    co-pol channel in the dominant state. Flat surfaces give no cross-pol so it carries
    nothing there; tilted surfaces and corner-like targets give it a definite value.

    Anisotropy is not returned: with two eigenvalues it is a function of entropy alone.

    Parameters
    ----------
    copol, crosspol : complex, shape (n_az, n_rg); single-look SLC channels, e.g. HH, HV
    looks : (az_looks, rg_looks) window size for the averaging
    stride : window step, defaults to looks (non-overlapping)

    Returns
    -------
    entropy : real, shape (n_az_ml, n_rg_ml); in [0, 1]
    alpha : real, shape (n_az_ml, n_rg_ml); radians, in [0, pi/2]
    cross_co_phase_difference : real, shape (n_az_ml, n_rg_ml); radians, in (-pi, pi]
    """
    # stack the two channels so covariance_matrix treats them as a 2-element look stack
    stack = np.stack([copol, crosspol])

    # eq. 1: boxcar-average the outer products to get the 2x2 matrix J per pixel
    covariance, _ = covariance_matrix(stack, looks, stride)      # (2, 2, n_az_ml, n_rg_ml)
    # eigh wants the matrix axes last, covariance_matrix puts them first, so move them
    j = np.moveaxis(covariance, (0, 1), (-2, -1))                # (n_az_ml, n_rg_ml, 2, 2)

    # eq. 2: eigendecompose every pixel's Hermitian 2x2 matrix in one call
    eigvals, eigvecs = np.linalg.eigh(j)
    # eigh sorts ascending, flip so index 0 is the principal eigenvalue and eigenvector
    eigvals = np.clip(eigvals[..., ::-1], 0.0, None)             # tiny negatives are round-off
    eigvecs = eigvecs[..., ::-1]
    # normalise the eigenvalues to sum to one; these are the probabilities P_i
    p = eigvals / eigvals.sum(axis=-1, keepdims=True)
    # base-2 entropy so two eigenvalues span 0 to 1; np.where makes 0 * log(0) give 0
    entropy = -np.sum(p * np.log2(np.where(p > 0, p, 1.0)), axis=-1)

    # column 0 of the eigenvector matrix is the dominant state (cos a, sin a e^{i d})
    u1 = eigvecs[..., :, 0]
    # modulus ignores the eigenvector's arbitrary overall phase; clip guards arccos round-off
    a = np.arccos(np.clip(np.abs(u1[..., 0]), 0.0, 1.0))
    # phase of component 1 relative to component 0, so the overall phase cancels here too
    cross_co_phase_difference = np.angle(u1[..., 1] * np.conj(u1[..., 0]))
    # second eigenvector is orthogonal so its angle is pi/2 - a; weight both by probability
    alpha = p[..., 0] * a + p[..., 1] * (np.pi / 2 - a)
    return entropy, alpha, cross_co_phase_difference

def quadpol_h_alpha(
    hh: np.ndarray, hv: np.ndarray, vh: np.ndarray, vv: np.ndarray, looks: tuple[int, int],
    stride: tuple[int, int] | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quad-pol entropy/alpha/anisotropy decomposition of the 3x3 coherency matrix T3.

    Follows Cloude and Pottier (1997), "An entropy based classification scheme for land
    applications of polarimetric SAR". Each pixel's scattering is written as the Pauli
    vector k = [HH + VV, HH - VV, HV + VH] / sqrt(2), which assumes reciprocity (HV = VH)
    and so uses the mean of the two cross-pol channels. The outer product k k^H is
    averaged over a window into T3, whose three eigenvalues, scaled to sum to one, are the
    probabilities of three orthogonal scattering mechanisms.

    entropy is base 3 so it spans 0 (one mechanism) to 1 (three equal ones, as for noise).
    A random cloud of dipoles gives 0.95.

    alpha is the probability-weighted mean of each eigenvector's angle away from the
    first Pauli component: 0 for a smooth surface or trihedral, pi/4 for a dipole,
    pi/2 for a dihedral. A random dipole cloud gives pi/4.

    anisotropy is (lambda2 - lambda3) / (lambda2 + lambda3), the split between the two
    minor mechanisms. It carries information only where entropy is moderate to high.

    Parameters
    ----------
    hh, hv, vh, vv : complex, shape (n_az, n_rg); single-look SLC channels
    looks : (az_looks, rg_looks) window size for the averaging
    stride : window step, defaults to looks (non-overlapping)

    Returns
    -------
    entropy : real, shape (n_az_ml, n_rg_ml); in [0, 1]
    alpha : real, shape (n_az_ml, n_rg_ml); radians, in [0, pi/2]
    anisotropy : real, shape (n_az_ml, n_rg_ml); in [0, 1]
    """
    # Pauli vector, with the cross-pol term from the HV/VH mean (2 * mean / sqrt(2))
    k = np.stack([hh + vv, hh - vv, hv + vh]) / np.sqrt(2)

    # boxcar-average the outer products to get the 3x3 coherency matrix per pixel
    coherency, _ = covariance_matrix(k, looks, stride)           # (3, 3, n_az_ml, n_rg_ml)
    t3 = np.moveaxis(coherency, (0, 1), (-2, -1))                # (n_az_ml, n_rg_ml, 3, 3)

    # eigh sorts ascending, flip so index 0 is the largest eigenvalue
    eigvals, eigvecs = np.linalg.eigh(t3)
    eigvals = np.clip(eigvals[..., ::-1], 0.0, None)             # tiny negatives are round-off
    eigvecs = eigvecs[..., ::-1]
    # normalise the eigenvalues to sum to one; these are the probabilities P_i
    p = eigvals / eigvals.sum(axis=-1, keepdims=True)
    # base-3 entropy so three eigenvalues span 0 to 1; np.where makes 0 * log(0) give 0
    entropy = -np.sum(p * np.log(np.where(p > 0, p, 1.0)), axis=-1) / np.log(3)

    # each eigenvector's alpha is the angle off the first Pauli component; modulus drops its phase
    alpha_i = np.arccos(np.clip(np.abs(eigvecs[..., 0, :]), 0.0, 1.0))   # (n_az_ml, n_rg_ml, 3)
    alpha = np.sum(p * alpha_i, axis=-1)

    # split between the two minor eigenvalues
    anisotropy = (eigvals[..., 1] - eigvals[..., 2]) / (eigvals[..., 1] + eigvals[..., 2])
    return entropy, alpha, anisotropy
