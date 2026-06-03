"""I/O utilities for reading NISAR RSLC HDF5 data."""

import h5py
import numpy as np
from pathlib import Path

def read_range_block(path, rg_start, rg_stop, freq="A", pol="HH"):
    """
    Read a contiguous range block across all azimuth lines from an RSLC file.

    Parameters
    ----------
    path : str or Path
        Path to the NISAR RSLC HDF5 file.
    rg_start : int
        First range bin index (inclusive).
    rg_stop : int
        Last range bin index (exclusive).
    freq : str
        Frequency band identifier, e.g. "A" (main band) or "B".
    pol : str
        Polarization, e.g. "HH", "HV", "VH", "VV".

    Returns
    -------
    block : np.ndarray, shape (n_az, rg_stop - rg_start), complex64
        All azimuth lines for the requested range columns.
    """
    dataset_path = f"science/LSAR/RSLC/swaths/frequency{freq}/{pol}"
    with h5py.File(path, "r") as fh:
        if dataset_path not in fh:
            raise KeyError(
                f"Dataset not found: {dataset_path}. "
                f"Available polarizations: {list(fh[f'science/LSAR/RSLC/swaths/frequency{freq}'].keys())}"
            )
        ds = fh[dataset_path]
        print(f"dataset dtype: {ds.dtype}, shape: {ds.shape}")
        block = ds[:, rg_start:rg_stop]
        # NISAR RSLC stores data as complex32 (float16 pairs); h5py returns a
        # structured array that must be explicitly cast to complex64.
        if block.dtype.names is not None and set(block.dtype.names) >= {'r', 'i'}:
            block = (block['r'].astype(np.float32) + 1j * block['i'].astype(np.float32))
        block = block.astype(np.complex64)
    return block

if __name__ == '__main__':
    p0 = Path('/home/simon/Work/fmnisar/NISAR/')
    path_nisar = p0 / 'NISAR_L1_PR_RSLC_010_073_D_053_4005_DHDH_A_20260114T062318_20260114T062355_X05010_N_F_J_001.h5'
    r_start = 32000
    block = read_range_block(path_nisar, r_start, r_start+512)
    
    #save as npy
    np.save(p0 / 'block.npy', block)
    
    from scipy.fft import next_fast_len
    # zero padded block
    
    # demodulate
    
    # dewindow?
    
    # 

    
    print(block.shape)

    spectrum = np.fft.fftshift(np.fft.fft(block, axis=0), axes=0)
    print(spectrum.shape, spectrum.dtype)
    import matplotlib.pyplot as plt
    # plt.imshow(np.abs(block))
    power = np.mean(np.abs(spectrum) ** 2, axis=1)
    print(power.shape)
    plt.plot(np.arange(len(power)), 10 * np.log10(power))
    plt.show()
    

