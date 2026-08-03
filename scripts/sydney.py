from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from subaperture import AzimuthSubaperture, SubapertureMetaData, plot_subaperture_diagnostics
from ioput import read_block, latlon_box_to_radar_bbox
from products import (
    multilook_intensity, intensity_to_rgb, coherence, covariance_matrix,
    diffphase_statistics, diffphase_statistics_slc, entropy, normalized_variance,
    normalized_variance_ml, mean_coherence, point_target_score, intensity_polynomial,
    intensity_polynomial_ml, qdeviation)

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

quantity_style = {
    'normalized_variance':    dict(cmap='magma', vmin=0, vmax=5),
    'normalized_variance_ml': dict(cmap='magma', vmin=0, vmax=2),
    'intensity_linear':       dict(cmap='RdBu_r', vmin=-0.5, vmax=0.5),
    'intensity_linear_ml':    dict(cmap='RdBu_r', vmin=-0.2, vmax=0.2),
    'intensity_quadratic':    dict(cmap='RdBu_r', vmin=-0.5, vmax=0.5),
    'intensity_quadratic_ml': dict(cmap='RdBu_r', vmin=-0.2, vmax=0.2),
    'qdeviation':             dict(cmap='magma', vmin=0, vmax=1.0),
    'qdeviation_ml':          dict(cmap='magma', vmin=0, vmax=1.0),
    'coherence':              dict(cmap='gray', vmin=0, vmax=1),
    'mean_coherence':         dict(cmap='gray', vmin=0, vmax=1),
    'entropy':                dict(cmap='magma', vmin=0, vmax=1),
    'phase_variance':         dict(cmap='gray', vmin=0, vmax=1),
    'phase_mean':             dict(cmap='twilight_shifted', vmin=-np.pi, vmax=np.pi),
    'point_target_score':     dict(cmap='gray', vmin=0, vmax=1),
}


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
                   nv, **quantity_style['normalized_variance'])

        # single-look linear and quadratic coefficients of log-intensity across sub-apertures
        linear, quadratic = intensity_polynomial(block_sub)
        plt.imsave(folder_out_n / f'{name}_subaperture_linear.png',
                   linear, **quantity_style['intensity_linear'])
        plt.imsave(folder_out_n / f'{name}_subaperture_quadratic.png',
                   quadratic, **quantity_style['intensity_quadratic'])

        # single-look max deviation of the linear+quadratic fit from the intercept-only model
        qdev = qdeviation(linear, quadratic, n_subapertures)
        plt.imsave(folder_out_n / f'{name}_subaperture_qdeviation.png',
                   qdev, **quantity_style['qdeviation'])

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
                coh, **quantity_style['coherence'])

        # full pairwise covariance/coherence matrix, baseband subapertures only
        cov, coh = covariance_matrix(
            block_sub_baseband, (az_looks, rg_looks), stride=stride)
        # np.save(folder_out_n / f'{name}_subaperture_covariance_matrix.npy', cov)
        # np.save(folder_out_n / f'{name}_subaperture_coherence_matrix.npy', coh)

        # multilooked log-power variance across sub-apertures, from the covariance matrix diagonal
        nv_ml = normalized_variance_ml(cov)
        plt.imsave(folder_out_n / f'{name}_subaperture_normvariance_multilooked.png',
                   nv_ml, **quantity_style['normalized_variance_ml'])

        # multilooked linear and quadratic coefficients of log-power across sub-apertures
        linear_ml, quadratic_ml = intensity_polynomial_ml(cov)
        plt.imsave(folder_out_n / f'{name}_subaperture_linear_multilooked.png',
                   linear_ml, **quantity_style['intensity_linear_ml'])
        plt.imsave(folder_out_n / f'{name}_subaperture_quadratic_multilooked.png',
                   quadratic_ml, **quantity_style['intensity_quadratic_ml'])

        # multilooked max deviation of the linear+quadratic fit from the intercept-only model
        qdev_ml = qdeviation(linear_ml, quadratic_ml, n_subapertures)
        plt.imsave(folder_out_n / f'{name}_subaperture_qdeviation_multilooked.png',
                   qdev_ml, **quantity_style['qdeviation_ml'])

        # mean coherence across all N-choose-2 sub-aperture pairs
        mean_coh_ml = mean_coherence(coh)
        plt.imsave(folder_out_n / f'{name}_subaperture_meancoherence_{az_looks}az{rg_looks}rg_looks.png',
                   mean_coh_ml, **quantity_style['mean_coherence'])

        # entropy of the sub-aperture covariance spectrum, in [0, 1]
        H = entropy(cov)
        plt.imsave(folder_out_n / f'{name}_subaperture_entropy.png',
                   H, **quantity_style['entropy'])

        M, V = diffphase_statistics(cov)
        plt.imsave(folder_out_n / f'{name}_subaperture_phasevariance.png',
                   V, **quantity_style['phase_variance'])
        plt.imsave(folder_out_n / f'{name}_subaperture_phasemean.png',
                   M, **quantity_style['phase_mean'])

        # full-resolution phase mean/variance directly from the baseband subaperture SLC stack
        M_full, V_full = diffphase_statistics_slc(block_sub_baseband)
        plt.imsave(folder_out_n / f'{name}_subaperture_phasevariance_singlelook.png',
                   V_full, **quantity_style['phase_variance'])
        plt.imsave(folder_out_n / f'{name}_subaperture_phasemean_singlelook.png',
                   M_full, **quantity_style['phase_mean'])

        # single-look point target score: correlation with the best-fitting phase ramp
        pts = point_target_score(block_sub_baseband, M_full)
        plt.imsave(folder_out_n / f'{name}_subaperture_pointtargetscore.png',
                   pts, **quantity_style['point_target_score'])


        # diagnostics: spectra and sub-aperture windows over the AOI block
        plot_subaperture_diagnostics(
            sub_baseband, block, save_path=folder_out_n / f'{name}_subaperture_diagnostics.png')

        # process the entire image, subapertures centered to baseband
        # output_h5 = folder_out_n / f'{name}_subaperture_baseband.h5'
        # sub_baseband.process_rslc(path_nisar, output_h5, pols=[pol])


def _read_area_block(meta, area):
    aoi_width_km, aoi_height_km = area['aoi_extent_km']
    bbox = latlon_box_to_radar_bbox(
        path_nisar, area['lon_center'], area['lat_center'], aoi_width_km, aoi_height_km)
    block = read_block(
        path_nisar, bbox['az_start'], bbox['az_stop'], bbox['rg_start'], bbox['rg_stop'], pol=pol)
    return bbox, block


def _baseband_subaperture_block(meta, block, bbox, n_subapertures):
    sub_baseband = AzimuthSubaperture(
        meta, n_subapertures=n_subapertures, overlap=overlap,
        demodulate_subaperture=True, remodulate_to_full_dc=False, deweight_spectrum=True)
    return sub_baseband._process_block(block, az_time_start=float(meta.az_time[bbox['az_start']]))


def plot_ml_sweep(meta, area, quantity, n_subapertures_list, looks_list, stride=stride):
    """Sweep n_subapertures x looks for a covariance-matrix quantity ('mean_coherence',
    'entropy', 'intensity_linear_ml', 'intensity_quadratic_ml', or 'qdeviation_ml')."""
    style = quantity_style[quantity]
    bbox, block = _read_area_block(meta, area)

    fig, axes = plt.subplots(
        len(n_subapertures_list), len(looks_list),
        figsize=(3 * len(looks_list), 3 * len(n_subapertures_list)), squeeze=False)
    for i, n_subapertures in enumerate(n_subapertures_list):
        block_sub_baseband = _baseband_subaperture_block(meta, block, bbox, n_subapertures)
        for j, looks in enumerate(looks_list):
            cov, coh = covariance_matrix(block_sub_baseband, (looks, looks), stride=stride)
            if quantity == 'mean_coherence':
                value = mean_coherence(coh)
            elif quantity == 'entropy':
                value = entropy(cov)
            elif quantity == 'intensity_linear_ml':
                value, _ = intensity_polynomial_ml(cov)
            elif quantity == 'intensity_quadratic_ml':
                _, value = intensity_polynomial_ml(cov)
            else:
                linear_ml, quadratic_ml = intensity_polynomial_ml(cov)
                value = qdeviation(linear_ml, quadratic_ml, n_subapertures)
            ax = axes[i, j]
            ax.imshow(value, cmap=style['cmap'], vmin=style['vmin'], vmax=style['vmax'])
            ax.set_title(f'N={n_subapertures}, L={looks}')
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"{area['name']}: {quantity}")
    fig.tight_layout()
    fig.savefig(folder_out / f"{area['name']}_sweep_{quantity}.png", dpi=150)
    plt.close(fig)


def plot_slc_sweep(meta, area, quantity, n_subapertures_list):
    """Sweep n_subapertures for a single-look SLC quantity ('phase_variance', 'point_target_score',
    'intensity_linear', 'intensity_quadratic', or 'qdeviation')."""
    style = quantity_style[quantity]
    bbox, block = _read_area_block(meta, area)

    fig, axes = plt.subplots(1, len(n_subapertures_list), figsize=(3 * len(n_subapertures_list), 3), squeeze=False)
    for j, n_subapertures in enumerate(n_subapertures_list):
        block_sub_baseband = _baseband_subaperture_block(meta, block, bbox, n_subapertures)
        M_full, V_full = diffphase_statistics_slc(block_sub_baseband)
        if quantity == 'phase_variance':
            value = V_full
        elif quantity == 'point_target_score':
            value = point_target_score(block_sub_baseband, M_full)
        elif quantity == 'intensity_linear':
            value, _ = intensity_polynomial(block_sub_baseband)
        elif quantity == 'intensity_quadratic':
            _, value = intensity_polynomial(block_sub_baseband)
        elif quantity == 'qdeviation':
            linear, quadratic = intensity_polynomial(block_sub_baseband)
            value = qdeviation(linear, quadratic, n_subapertures)
        else:
            raise ValueError(f"Unknown quantity {quantity}")
        ax = axes[0, j]
        ax.imshow(value, cmap=style['cmap'], vmin=style['vmin'], vmax=style['vmax'])
        ax.set_title(f'N={n_subapertures}')
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{area['name']}: {quantity}")
    fig.tight_layout()
    fig.savefig(folder_out / f"{area['name']}_sweep_{quantity}.png", dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    meta = SubapertureMetaData.load_from_rslc_path(path_nisar)
    for n_subapertures in (3, 5, 7, 9):
        process_sydney(meta, n_subapertures)

    for area in study_areas:
        for quantity in ('mean_coherence', 'entropy', 'intensity_linear_ml', 'intensity_quadratic_ml',
                          'qdeviation_ml'):
            plot_ml_sweep(meta, area, quantity, (3, 7, 11), (3, 7, 11))
        for quantity in ('phase_variance', 'point_target_score', 'intensity_linear', 'intensity_quadratic',
                          'qdeviation'):
            plot_slc_sweep(meta, area, quantity, (3, 7, 11, 15))
