from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from subaperture import AzimuthSubaperture, SubapertureMetaData, plot_subaperture_diagnostics
from ioput import read_block, latlon_box_to_radar_bbox
from products import (
    multilook_intensity, intensity_to_rgb, coherence, covariance_matrix,
    diffphase_statistics, diffphase_statistics_slc, entropy, normalized_variance,
    normalized_variance_ml, mean_coherence, point_target_score)

path_nisar = Path(
    '/media/simon/Extreme SSD/fmnisar/Sydney/'
    'NISAR_L1_PR_RSLC_025_016_D_105_4005_DHDH_A_20260709T075915_20260709T075950_P05023_N_F_J_001.h5')
folder_out = path_nisar.parent / 'processed'
folder_out.mkdir(exist_ok=True)

pol = 'HH'
overlap = 0.2

study_areas = [
    dict(name='sydney', lon_center=151.215256, lat_center=-33.856784,
         aoi_extent_km=(6.0, 6.0)),
    dict(name='mountains', lon_center=150.49945, lat_center=-33.8196737,
         aoi_extent_km=(6.0, 6.0)),
]

az_looks, rg_looks = 9, 9
stride = 3
gamma = 2.5

def process_sydney(meta, n_subapertures):
    sub = AzimuthSubaperture(
        meta, n_subapertures=n_subapertures, overlap=overlap,
        demodulate_subaperture=False, remodulate_to_full_dc=True, deweight_spectrum=True)
    sub_baseband = AzimuthSubaperture(
        meta, n_subapertures=n_subapertures, overlap=overlap,
        demodulate_subaperture=True, remodulate_to_full_dc=False, deweight_spectrum=True)

    folder_out_n = folder_out / f'N{n_subapertures}'
    folder_out_n.mkdir(parents=True, exist_ok=True)
    rgb_idx = [0, n_subapertures // 2, n_subapertures - 1]

    for area in study_areas:
        name = area['name']

        aoi_width_km, aoi_height_km = area['aoi_extent_km']
        bbox = latlon_box_to_radar_bbox(
            path_nisar, area['lon_center'], area['lat_center'],
            aoi_width_km, aoi_height_km)
        print(f'[{name}] radar bbox: {bbox}')

        block = read_block(
            path_nisar, bbox['az_start'], bbox['az_stop'], bbox['rg_start'], bbox['rg_stop'], pol=pol)

        block_sub = sub._process_block(block, az_time_start=float(meta.az_time[bbox['az_start']]))

        power_full = (abs(block) ** 2).mean()
        power_subapertures = [(abs(block_sub[i]) ** 2).mean() for i in range(n_subapertures)]
        print(f'[{name}] mean power full block: {power_full}')
        for i, power_sub in enumerate(power_subapertures):
            print(f'[{name}] mean power subaperture {i + 1}: {power_sub} '
                  f'(ratio to full: {power_sub / power_full:.3f})')

        # single-look RGB composite
        intensity_sl = abs(block_sub[rgb_idx]) ** 2
        rgb_sl = intensity_to_rgb(intensity_sl, gamma=gamma)
        plt.imsave(folder_out_n / f'{name}_subaperture_rgb_singlelook.png', rgb_sl)

        # single-look full-resolution full-aperture image (grayscale, same stretch/gamma)
        intensity_full_sl = (abs(block) ** 2)[None].repeat(3, axis=0)
        rgb_full_sl = intensity_to_rgb(intensity_full_sl, gamma=gamma)
        plt.imsave(folder_out_n / f'{name}_fullaperture_singlelook.png', rgb_full_sl)

        # log-intensity variance across sub-apertures
        nv = normalized_variance(block_sub)
        plt.imsave(folder_out_n / f'{name}_subaperture_normvariance.png',
                   nv, cmap='magma', vmin=0, vmax=5)

        # multilooked RGB composite
        intensity_ml = multilook_intensity(block_sub[rgb_idx], (az_looks, rg_looks), stride=stride)
        rgb_ml = intensity_to_rgb(intensity_ml, gamma=gamma)
        plt.imsave(folder_out_n / f'{name}_subaperture_rgb_{az_looks}az{rg_looks}rg_looks.png', rgb_ml)

        # coherence between looks, with and without recentering to baseband
        coherence_look_idx = (0, n_subapertures - 1)
        block_sub_baseband = sub_baseband._process_block(
            block, az_time_start=float(meta.az_time[bbox['az_start']]))

        for tag, block_sub_variant in (('nobaseband', block_sub), ('baseband', block_sub_baseband)):
            i, j = coherence_look_idx
            coh = coherence(
                block_sub_variant[i], block_sub_variant[j], (az_looks, rg_looks), stride=1)
            plt.imsave(
                folder_out_n / f'{name}_subaperture_coherence_look{i + 1}{j + 1}_{tag}.png',
                coh, cmap='gray', vmin=0, vmax=1)

        # full pairwise covariance/coherence matrix, baseband subapertures only
        cov, coh = covariance_matrix(
            block_sub_baseband, (az_looks, rg_looks), stride=stride)
        # np.save(folder_out_n / f'{name}_subaperture_covariance_matrix.npy', cov)
        # np.save(folder_out_n / f'{name}_subaperture_coherence_matrix.npy', coh)

        # multilooked log-power variance across sub-apertures, from the covariance matrix diagonal
        nv_ml = normalized_variance_ml(cov)
        plt.imsave(folder_out_n / f'{name}_subaperture_normvariance_multilooked.png',
                   nv_ml, cmap='magma', vmin=0, vmax=2)

        # mean coherence across all N-choose-2 sub-aperture pairs
        mean_coh_ml = mean_coherence(coh)
        plt.imsave(folder_out_n / f'{name}_subaperture_meancoherence_{az_looks}az{rg_looks}rg_looks.png',
                   mean_coh_ml, cmap='gray', vmin=0, vmax=1)

        # entropy of the sub-aperture covariance spectrum, in [0, log(n_subapertures)]
        H = entropy(cov)
        plt.imsave(folder_out_n / f'{name}_subaperture_entropy.png',
                   H, cmap='magma', vmin=0, vmax=np.log(n_subapertures))

        M, V = diffphase_statistics(cov)
        plt.imsave(folder_out_n / f'{name}_subaperture_phasevariance.png',
                   V, cmap='gray', vmin=0, vmax=1)
        plt.imsave(folder_out_n / f'{name}_subaperture_phasemean.png',
                   M, cmap='twilight_shifted', vmin=-np.pi, vmax=np.pi)

        # full-resolution phase mean/variance directly from the baseband subaperture SLC stack
        M_full, V_full = diffphase_statistics_slc(block_sub_baseband)
        plt.imsave(folder_out_n / f'{name}_subaperture_phasevariance_singlelook.png',
                   V_full, cmap='gray', vmin=0, vmax=1)
        plt.imsave(folder_out_n / f'{name}_subaperture_phasemean_singlelook.png',
                   M_full, cmap='twilight_shifted', vmin=-np.pi, vmax=np.pi)

        # single-look point target score: correlation with the best-fitting phase ramp
        pts = point_target_score(block_sub_baseband, M_full)
        plt.imsave(folder_out_n / f'{name}_subaperture_pointtargetscore.png',
                   pts, cmap='gray', vmin=0, vmax=1)


        # diagnostics: spectra and sub-aperture windows over the AOI block
        plot_subaperture_diagnostics(
            sub_baseband, block, save_path=folder_out_n / f'{name}_subaperture_diagnostics.png')

        # process the entire image, subapertures centered to baseband
        # output_h5 = folder_out_n / f'{name}_subaperture_baseband.h5'
        # sub_baseband.process_rslc(path_nisar, output_h5, pols=[pol])


if __name__ == '__main__':
    meta = SubapertureMetaData.load_from_rslc_path(path_nisar)
    for n_subapertures in (3, 5, 7, 9):
        process_sydney(meta, n_subapertures)
