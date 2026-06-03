"""I/O utilities for reading NISAR RSLC HDF5 data."""

import h5py
import numpy as np
from pathlib import Path


def read_block(path, az_start, az_stop, rg_start, rg_stop, freq='A', pol='HH'):
    """Read a 2D block from an RSLC file.

    Parameters
    ----------
    path : str or Path
    az_start, az_stop : int or None
        Azimuth line indices (start inclusive, stop exclusive). None means full extent.
    rg_start, rg_stop : int
        Range bin indices (start inclusive, stop exclusive).
    freq : str
        Frequency band, e.g. 'A' or 'B'.
    pol : str
        Polarization, e.g. 'HH', 'HV', 'VH', 'VV'.

    Returns
    -------
    np.ndarray, shape (az_stop - az_start, rg_stop - rg_start), complex64
    """
    ds_path = f'science/LSAR/RSLC/swaths/frequency{freq}/{pol}'
    with h5py.File(path, 'r') as fh:
        if ds_path not in fh:
            avail = list(fh[f'science/LSAR/RSLC/swaths/frequency{freq}'].keys())
            raise KeyError(f'Dataset not found: {ds_path}. Available: {avail}')
        block = fh[ds_path][az_start:az_stop, rg_start:rg_stop]
        if block.dtype.names is not None and {'r', 'i'} <= set(block.dtype.names):
            block = block['r'].astype(np.float32) + 1j * block['i'].astype(np.float32)
    return block.astype(np.complex64)


def read_range_block(path, rg_start, rg_stop, freq='A', pol='HH'):
    """Read a contiguous range block across all azimuth lines from an RSLC file.

    Parameters
    ----------
    path : str or Path
    rg_start, rg_stop : int
        Range bin indices (start inclusive, stop exclusive).
    freq : str
        Frequency band, e.g. 'A' or 'B'.
    pol : str
        Polarization, e.g. 'HH', 'HV', 'VH', 'VV'.

    Returns
    -------
    np.ndarray, shape (n_az, rg_stop - rg_start), complex64
    """
    return read_block(path, None, None, rg_start, rg_stop, freq=freq, pol=pol)


def get_available_pols(fh: h5py.File, freq: str) -> list:
    """Return polarization strings present in the RSLC swath group."""
    grp = fh[f'science/LSAR/RSLC/swaths/frequency{freq}']
    pols = []
    for key, ds in grp.items():
        if not isinstance(ds, h5py.Dataset):
            continue
        is_structured = ds.dtype.names is not None and {'r', 'i'} <= set(ds.dtype.names)
        if is_structured or np.issubdtype(ds.dtype, np.complexfloating):
            pols.append(key)
    return pols


def copy_h5_structure(src_grp: h5py.Group, dst_grp: h5py.Group, skip: set = None) -> None:
    """Recursively copy groups/datasets from src to dst, skipping paths in *skip*."""
    skip = skip or set()
    for k, v in src_grp.items():
        if isinstance(v, h5py.Dataset):
            if v.name not in skip:
                src_grp.copy(k, dst_grp)
        else:
            sub = dst_grp.require_group(k)
            for ak, av in v.attrs.items():
                sub.attrs[ak] = av
            copy_h5_structure(v, sub, skip=skip)
    for ak, av in src_grp.attrs.items():
        dst_grp.attrs[ak] = av


if __name__ == '__main__':
    p0 = Path('/home/simon/Work/fmnisar/NISAR/')
    path_nisar = p0 / 'NISAR_L1_PR_RSLC_010_073_D_053_4005_DHDH_A_20260114T062318_20260114T062355_X05010_N_F_J_001.h5'
    r_start = 32000
    block = read_range_block(path_nisar, r_start, r_start + 512)
    np.save(p0 / 'block.npy', block)

    import matplotlib.pyplot as plt
    spectrum = np.fft.fftshift(np.fft.fft(block, axis=0), axes=0)
    power = np.mean(np.abs(spectrum) ** 2, axis=1)
    plt.plot(np.arange(len(power)), 10 * np.log10(power))
    plt.show()
