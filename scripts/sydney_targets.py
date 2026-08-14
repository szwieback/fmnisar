"""Candidate coherent targets in downtown Sydney: bright pixels combining high inter-sub-aperture
coherence with a large quadratic deviation, i.e. a stable but strongly anisotropic response.
Hand-specified lon/lat targets are analyzed alongside the automatically selected ones."""

from pathlib import Path

import h5py
import isce3
import matplotlib.pyplot as plt
import numpy as np

from subaperture import AzimuthSubaperture, SubapertureMetaData
from ioput import read_block, latlon_box_to_radar_bbox
from products import (
    covariance_matrix, mean_coherence, intensity_polynomial_ml, qdeviation,
    multilook_intensity, intensity_to_rgb)

path_nisar = Path(
    '/media/simon/Extreme SSD/fmnisar/Sydney/'
    'NISAR_L1_PR_RSLC_025_016_D_105_4005_DHDH_A_20260709T075915_20260709T075950_P05023_N_F_J_001.h5')
folder_out = path_nisar.parent / 'processed' / 'targets'

area = dict(name='sydney', lon_center=151.215256, lat_center=-33.856784, aoi_extent_km=(6.0, 6.0))

# targets specified by hand, in addition to the automatically selected ones
named_targets = {'fish_harbour': (151.19081251247505, -33.874305642784755),
                 'cockle_bay': (151.2021231923605, -33.87195986136284)}

pol = 'HH'
overlap = 0.2
n_subapertures = 9
n_subapertures_rgb = 3
az_looks, rg_looks = 9, 9
stride = 2
gamma = 3.0

n_targets = 10
coherence_min = 0.7
intensity_percentile = 99.0   # on the center sub-aperture of the RGB stack
min_separation = 50      # multilooked pixels between selected targets
chip_halfwidth = 50      # multilooked pixels

qdeviation_style = dict(cmap='magma', vmin=0, vmax=2.0)
coherence_style = dict(cmap='gray', vmin=0, vmax=1)


def read_area(area):
    """SLC block over the AOI, plus the AOI radar bounding box."""
    width_km, height_km = area['aoi_extent_km']
    bbox = latlon_box_to_radar_bbox(
        path_nisar, area['lon_center'], area['lat_center'], width_km, height_km)
    block = read_block(
        path_nisar, bbox['az_start'], bbox['az_stop'], bbox['rg_start'], bbox['rg_stop'], pol=pol)
    return bbox, block


def subaperture_stack(meta, block, bbox, n_subapertures):
    """Baseband sub-aperture stack of an AOI block."""
    sub = AzimuthSubaperture(
        meta, n_subapertures=n_subapertures, overlap=overlap,
        demodulate_subaperture=True, remodulate_to_full_dc=False, deweight_spectrum=True)
    return sub._process_block(block, az_time_start=float(meta.az_time[bbox['az_start']]))


def find_targets(qdev, coh, intensity, n_targets, coherence_min, intensity_percentile,
                 min_separation, exclude=()):
    """Indices of the n_targets highest-qdeviation pixels that are both coherent
    (>= coherence_min) and bright (intensity above intensity_percentile of the AOI),
    separated by at least min_separation pixels from each other and from exclude."""
    intensity_min = np.percentile(intensity, intensity_percentile)
    eligible = (coh >= coherence_min) & (intensity >= intensity_min)
    score = np.where(eligible, qdev, -np.inf)
    for i, j in exclude:
        sl_az, sl_rg = chip_slices(i, j, score.shape, min_separation)
        score[sl_az, sl_rg] = -np.inf
    print(f'{eligible.sum()} pixels with mean coherence >= {coherence_min} and center-aperture '
          f'intensity >= {10 * np.log10(intensity_min):.1f} dB (p{intensity_percentile:g}), '
          f'{100 * eligible.mean():.2f}% of the AOI')
    pct = (50, 90, 99, 99.9)
    print(f'qdeviation percentiles {pct}: {np.percentile(qdev, pct).round(2)}')
    print(f'mean coherence percentiles {pct}: {np.percentile(coh, pct).round(2)}')

    targets = []
    for _ in range(n_targets):
        i, j = np.unravel_index(np.argmax(score), score.shape)
        if not np.isfinite(score[i, j]):
            break
        targets.append((i, j))
        i0, i1 = max(i - min_separation, 0), i + min_separation + 1
        j0, j1 = max(j - min_separation, 0), j + min_separation + 1
        score[i0:i1, j0:j1] = -np.inf
    return targets


def load_geometry(path, freq='A'):
    """Orbit and radar-grid vectors needed to geolocate a radar sample."""
    with h5py.File(path, 'r') as fh:
        orbit = isce3.core.load_orbit_from_h5_group(fh['science/LSAR/RSLC/metadata/orbit'])
        look = fh['science/LSAR/identification/lookDirection'][()]
        fc = float(fh[f'science/LSAR/RSLC/swaths/frequency{freq}/processedCenterFrequency'][()])
        slant_range = fh[f'science/LSAR/RSLC/swaths/frequency{freq}/slantRange'][()]
        az_time = fh['science/LSAR/RSLC/swaths/zeroDopplerTime'][()]
    return dict(orbit=orbit, side='left' if look == b'Left' else 'right',
                wavelength=isce3.core.speed_of_light / fc,
                az_time=az_time, slant_range=slant_range)


def radar_to_lonlat(geometry, az_idx, rg_idx, height_m=0.0):
    """Lon/lat (degrees) of a full-resolution radar sample on a constant-height ellipsoid."""
    n_az, n_rg = len(geometry['az_time']), len(geometry['slant_range'])
    aztime = float(np.interp(az_idx, np.arange(n_az), geometry['az_time']))
    srange = float(np.interp(rg_idx, np.arange(n_rg), geometry['slant_range']))
    dem = isce3.geometry.DEMInterpolator(height_m)
    xyz = isce3.geometry.rdr2geo_bracket(
        aztime, srange, geometry['orbit'], geometry['side'], 0.0, geometry['wavelength'], dem)
    llh = isce3.core.WGS84_ELLIPSOID.xyz_to_lon_lat(xyz)
    return float(np.degrees(llh[0])), float(np.degrees(llh[1]))


def lonlat_to_radar(geometry, lon, lat, height_m=0.0):
    """Full-resolution azimuth/range index of a lon/lat position on a constant-height ellipsoid."""
    xyz = isce3.core.WGS84_ELLIPSOID.lon_lat_to_xyz(
        np.array([np.radians(lon), np.radians(lat), height_m]))
    aztime, srange = isce3.geometry.geo2rdr_bracket(
        xyz, geometry['orbit'], isce3.core.LUT2d(), geometry['wavelength'], geometry['side'])
    az_idx = np.interp(aztime, geometry['az_time'], np.arange(len(geometry['az_time'])))
    rg_idx = np.interp(srange, geometry['slant_range'], np.arange(len(geometry['slant_range'])))
    return float(az_idx), float(rg_idx)


def to_multilooked(az_full, rg_full, bbox):
    """Multilooked pixel containing a full-resolution radar sample."""
    return (int(round((az_full - bbox['az_start'] - az_looks // 2) / stride)),
            int(round((rg_full - bbox['rg_start'] - rg_looks // 2) / stride)))


def to_full_resolution(i, j, bbox):
    """Full-resolution radar sample at the center of a multilooked pixel."""
    return (bbox['az_start'] + i * stride + az_looks // 2,
            bbox['rg_start'] + j * stride + rg_looks // 2)


def resolve_named_targets(named, geometry, bbox, shape):
    """Multilooked indices of the hand-specified targets, dropping those outside the AOI block."""
    resolved = {}
    for name, (lon, lat) in named.items():
        i, j = to_multilooked(*lonlat_to_radar(geometry, lon, lat), bbox)
        if 0 <= i < shape[0] and 0 <= j < shape[1]:
            resolved[name] = (i, j)
        else:
            print(f'named target {name} ({lat}, {lon}) maps to ({i}, {j}), outside the AOI; skipped')
    return resolved


def chip_slices(i, j, shape, halfwidth):
    """Slices of a square chip around (i, j), clipped to shape."""
    i0, i1 = max(i - halfwidth, 0), min(i + halfwidth + 1, shape[0])
    j0, j1 = max(j - halfwidth, 0), min(j + halfwidth + 1, shape[1])
    return slice(i0, i1), slice(j0, j1)


def plot_target(name, target, coh, qdev, intensity_rgb_stack, geometry, bbox, save_path):
    """Coherence, quadratic deviation and sub-aperture RGB chips around one target."""
    i, j = target
    sl_az, sl_rg = chip_slices(i, j, coh.shape, chip_halfwidth)
    extent = (sl_rg.start - j - 0.5, sl_rg.stop - j - 0.5, sl_az.stop - i - 0.5, sl_az.start - i - 0.5)

    az_full, rg_full = to_full_resolution(i, j, bbox)
    lon, lat = radar_to_lonlat(geometry, az_full, rg_full)

    rgb = intensity_to_rgb(
        intensity_rgb_stack[:, sl_az, sl_rg], low_pct=2.0, high_pct=99.9, gamma=gamma)

    panels = [
        ('mean coherence', coh[sl_az, sl_rg], coherence_style),
        ('quadratic deviation', qdev[sl_az, sl_rg], qdeviation_style),
        (f'sub-aperture RGB ({n_subapertures_rgb} sub-apertures)', rgb, None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for ax, (title, value, style) in zip(axes, panels):
        if style is None:
            ax.imshow(value, extent=extent, interpolation='nearest')
        else:
            im = ax.imshow(value, extent=extent, interpolation='nearest', **style)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('range offset (px)')
    axes[0].set_ylabel('azimuth offset (px)')

    fig.suptitle(
        f'{name}: qdeviation {qdev[i, j]:.2f}, mean coherence {coh[i, j]:.2f}\n'
        f'az {az_full}, rg {rg_full}  |  {lat:.5f}, {lon:.5f}', fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor='white')
    plt.close(fig)
    print(f'saved {save_path}')
    return dict(name=name, az=az_full, rg=rg_full, lon=lon, lat=lat,
                qdeviation=float(qdev[i, j]), coherence=float(coh[i, j]))


def plot_overview(targets, coh, qdev, save_path):
    """AOI-wide coherence and quadratic deviation with the selected targets marked."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, (title, value, style) in zip(
            axes, [('mean coherence', coh, coherence_style),
                   ('quadratic deviation', qdev, qdeviation_style)]):
        im = ax.imshow(value, interpolation='nearest', **style)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        for name, (i, j) in targets:
            ax.plot(j, i, marker='o', mfc='none', mec='#00ff88', ms=12, mew=1.5)
            ax.annotate(name.removeprefix('target'), (j, i), textcoords='offset points',
                        xytext=(10, 6), color='#00ff88', fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{area['name']}: candidate coherent targets", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor='white')
    plt.close(fig)
    print(f'saved {save_path}')


def write_summary(records, save_path):
    lines = ['name,az,rg,lat,lon,qdeviation,coherence']
    lines += [f"{r['name']},{r['az']},{r['rg']},{r['lat']:.6f},{r['lon']:.6f},"
              f"{r['qdeviation']:.4f},{r['coherence']:.4f}" for r in records]
    save_path.write_text('\n'.join(lines) + '\n')
    print(f'saved {save_path}')


def main():
    folder_out.mkdir(parents=True, exist_ok=True)
    meta = SubapertureMetaData.load_from_rslc_path(path_nisar)

    bbox, block = read_area(area)
    print(f"[{area['name']}] radar bbox: {bbox}, block {block.shape}")

    block_sub = subaperture_stack(meta, block, bbox, n_subapertures)
    cov, coh_matrix = covariance_matrix(block_sub, (az_looks, rg_looks), stride=stride)
    coh = mean_coherence(coh_matrix)
    linear, quadratic = intensity_polynomial_ml(cov)
    qdev = qdeviation(linear, quadratic, n_subapertures)
    del cov, coh_matrix, block_sub

    # the RGB composite gets its own coarser decomposition, so each channel spans a
    # third of the aperture instead of a narrow slice of the N-sub-aperture split
    block_sub_rgb = subaperture_stack(meta, block, bbox, n_subapertures_rgb)
    intensity_rgb_stack = multilook_intensity(
        block_sub_rgb, (az_looks, rg_looks), stride=stride)

    geometry = load_geometry(path_nisar)
    named = resolve_named_targets(named_targets, geometry, bbox, coh.shape)

    auto = find_targets(
        qdev, coh, intensity_rgb_stack[n_subapertures_rgb // 2], n_targets, coherence_min,
        intensity_percentile, min_separation, exclude=named.values())
    if len(auto) < n_targets:
        print(f'only {len(auto)} targets found; consider lowering coherence_min')

    targets = list(named.items()) + [
        (f'target{rank}', target) for rank, target in enumerate(auto, start=1)]
    records = [
        plot_target(name, target, coh, qdev, intensity_rgb_stack, geometry, bbox,
                    folder_out / f"{area['name']}_{name}.png")
        for name, target in targets]

    plot_overview(targets, coh, qdev, folder_out / f"{area['name']}_targets_overview.png")
    write_summary(records, folder_out / f"{area['name']}_targets.csv")


if __name__ == '__main__':
    main()
