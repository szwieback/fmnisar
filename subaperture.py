'''
Created on Jun 2, 2026

@author: simon
'''
from pathlib import Path
from scipy.fft import next_fast_len
import numpy as np
import h5py

from inspect_nisar import extract_rslc_params

def raised_cosine_window(dft_len, dft_centroid, dft_width, beta=0.5):
    # dft are dft indices after fftshift, from 0 to dft_len-1
    rc = np.zeros(dft_len, dtype=np.float32)
    dft_rolloff_hwidth = int((beta * dft_width) // 2)

    dft_left_cutoff  = dft_centroid - dft_width // 2
    dft_left_flat   = dft_left_cutoff  + dft_rolloff_hwidth
    dft_right_flat  = dft_centroid + dft_width // 2 - dft_rolloff_hwidth
    dft_right_cutoff = dft_centroid + dft_width // 2

    # clips; may change to circular
    rc[max(0, dft_left_flat):min(dft_len, dft_right_flat)] = 1.0

    if dft_rolloff_hwidth > 0:
        i_left = np.arange(max(0, dft_left_cutoff), min(dft_len, dft_left_flat))
        if len(i_left):
            rc[i_left] = 0.5 * (1 + np.cos(np.pi * (dft_left_flat - i_left) / dft_rolloff_hwidth))

        i_right = np.arange(max(0, dft_right_flat), min(dft_len, dft_right_cutoff))
        if len(i_right):
            rc[i_right] = 0.5 * (1 + np.cos(np.pi * (i_right - dft_right_flat) / dft_rolloff_hwidth))

    return rc


def inspect_subaperture_spectra(block, block_subaperture, zdts, dc=0.0, demodulate=True):
    import matplotlib.pyplot as plt

    N_slc = block.shape[0]
    N_sub = block_subaperture.shape[1]

    freqs_slc = np.fft.fftshift(np.fft.fftfreq(N_slc, d=zdts))
    freqs_sub = np.fft.fftshift(np.fft.fftfreq(N_sub, d=zdts))

    if demodulate:
        modulation_slc = np.exp(-2j * np.pi * dc * np.arange(N_slc) * zdts)
        modulation_sub = np.exp(-2j * np.pi * dc * np.arange(N_sub) * zdts)
        slc_input = block * modulation_slc[:, np.newaxis]
        sub_input = block_subaperture * modulation_sub[np.newaxis, :, np.newaxis]
    else:
        slc_input = block
        sub_input = block_subaperture

    # power spectra averaged over range
    slc_spec = np.mean(
        np.abs(np.fft.fftshift(np.fft.fft(slc_input, axis=0), axes=0)) ** 2, axis=1
    )

    n_sub = block_subaperture.shape[0]
    sub_specs = np.array([
        np.mean(
            np.abs(np.fft.fftshift(np.fft.fft(sub_input[j], axis=0), axes=0)) ** 2,
            axis=1,
        )
        for j in range(n_sub)
    ])

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

    ax_slc, ax_sub = axes
    ax_slc.plot(freqs_slc, 10 * np.log10(slc_spec))
    ax_slc.set_ylabel('Power (dB)')
    ax_slc.set_title('SLC azimuth spectrum')
    if demodulate:
        ax_slc.axvline(0, c='k', lw=0.5, ls='--')

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_sub))
    for j, (spec, color) in enumerate(zip(sub_specs, colors)):
        ax_sub.plot(freqs_sub, 10 * np.log10(spec), color=color, label=f'sub {j}')
    ax_sub.set_xlabel('Frequency (Hz)')
    ax_sub.set_ylabel('Power (dB)')
    ax_sub.set_title('Subaperture azimuth spectra')
    ax_sub.legend(fontsize='small', ncol=min(n_sub, 5))
    if demodulate:
        ax_sub.axvline(0, c='k', lw=0.5, ls='--')

    fig.tight_layout()
    return fig, axes


if __name__ == '__main__':
    p0 = Path('/home/simon/Work/fmnisar/NISAR/')
    path_nisar = p0 / 'NISAR_L1_PR_RSLC_010_073_D_053_4005_DHDH_A_20260114T062318_20260114T062355_X05010_N_F_J_001.h5'
    dewindow = False
    with h5py.File(path_nisar, 'r') as fh:
        p = extract_rslc_params(fh, freq='A')

    # fast loading lof block
    block = np.load(p0 / 'block.npy')

    # zero padded block
    N = next_fast_len(block.shape[0])
    oversampling = N / block.shape[0]
    block_zp = np.zeros((N, block.shape[1]), dtype=np.complex64)
    block_zp[:block.shape[0],:] = block

    # demodulate
    dc = np.mean(p['doppler_centroid'])  # make block-specific later
    az_spacing = p['az_spacing_m']
    zdts = p['zero_doppler_time_spacing']
    az_bw = p['azimuth_bandwidth']

    az_time = np.arange(N) * zdts
    modulation = np.exp(2j * np.pi * dc * az_time)
    block_zp *= modulation.conj()[:, np.newaxis]

    # dewindow?
    if dewindow:
        # get from antenna pattern or data-driven
        # no additional window is applied, according to RSLC document
        raise NotImplementedError("Need to figure out antenna pattern")

    # subapertures
    spectrum = np.fft.fftshift(np.fft.fft(block_zp, axis=0), axes=0)

    shrink_fraction_useful = 0.02  # 0.00
    fraction_useful = az_bw * zdts * (1 - shrink_fraction_useful)
    dft_useful_range = (N * (1 - fraction_useful) // 2, N * (1 + fraction_useful) // 2)
    dft_useful_width = dft_useful_range[1] - dft_useful_range[0]

    subapertures = 5
    overlap = 0.3

    dft_subaperture_width = int(dft_useful_width // (1 + (subapertures - 1) * (1 - overlap)))
    dft_subaperture_centroids = ((dft_useful_range[0] + dft_subaperture_width // 2
         + np.arange(subapertures) * dft_subaperture_width * (1 - overlap))).astype(np.uint32)

    # subaperture spectrum
    spectrum_subaperture = np.zeros((subapertures,) + spectrum.shape, dtype=np.complex64)
    for jsubaperture, dft_centroid in enumerate(dft_subaperture_centroids):
        win = raised_cosine_window(spectrum.shape[0], dft_centroid, dft_subaperture_width)
        spectrum_subaperture[jsubaperture, ...] = spectrum * win[:, np.newaxis]
    
    # inverse fourier transform
    block_subaperture_zp = np.fft.ifft(np.fft.ifftshift(spectrum_subaperture, axes=1), axis=1)
    
    # demodulate subaperture
    demodulate_subaperture = True # for coherence etc.
    N = block_subaperture_zp.shape[1]
    if demodulate_subaperture:
        for jsubaperture, dft_centroid in enumerate(dft_subaperture_centroids):
            modulation_subaperture = np.exp(2j * np.pi * (dft_centroid - N // 2) * np.arange(N) / N)
            block_subaperture_zp[jsubaperture, ...] *= modulation_subaperture.conj()[:, np.newaxis]
    # add back doppler centroid modulation
    # may need to tweak for variable doppler centroid
    block_subaperture_zp *= modulation[np.newaxis, :, np.newaxis]    
    
    # de-zeropad
    block_subaperture = block_subaperture_zp[:, :block.shape[0], :]

    # to do: add block based processing

    inspect_subaperture_spectra(block, block_subaperture, zdts, dc=dc, demodulate=True)

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    power = np.mean(np.abs(spectrum) ** 2, axis=1)
    ax1.plot(np.arange(len(power)), 10 * np.log10(power))
    ax1.axvline(N // 2, c='k', lw=0.5)
    ax1.axvline(dft_useful_range[0], c='#aaaaaa')
    ax1.axvline(dft_useful_range[1], c='#aaaaaa')
    for centroid in dft_subaperture_centroids:
        ax1.axvline(centroid, c='#66aa66')
    ax1.set_ylabel('Power (dB)')

    dft_indices = np.arange(N)
    for centroid in dft_subaperture_centroids:
        win = raised_cosine_window(N, int(centroid), int(dft_subaperture_width))
        ax2.plot(dft_indices, win.real)
    ax2.set_xlabel('DFT index')
    ax2.set_ylabel('Window weight')

    plt.show()

