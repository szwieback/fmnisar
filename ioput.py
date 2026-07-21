"""I/O and geolocation utilities for NISAR RSLC HDF5 data."""

from pathlib import Path

import h5py
import isce3
import numpy as np

_earth_radius = 6371000.0


def read_block(path, az_start, az_stop, rg_start, rg_stop, freq='A', pol='HH'):
    """Read a 2D block from an RSLC file.

    az_start/az_stop, rg_start/rg_stop are azimuth/range indices (stop
    exclusive); None for az_start/az_stop means full azimuth extent.

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
    """Read a contiguous range block across all azimuth lines from an RSLC file."""
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


def _box_corners(lon0, lat0, width_km, height_km):
    """Return lon/lat corners (and center) of a box centered on (lon0, lat0)."""
    dlat = (height_km * 1000.0 / 2.0) / _earth_radius
    dlon = (width_km * 1000.0 / 2.0) / (_earth_radius * np.cos(np.radians(lat0)))
    dlat, dlon = np.degrees(dlat), np.degrees(dlon)
    lats = lat0 + np.array([-dlat, -dlat, dlat, dlat, 0.0])
    lons = lon0 + np.array([-dlon, dlon, -dlon, dlon, 0.0])
    return lons, lats


def latlon_box_to_radar_bbox(
    rslc_path: Path,
    lon0: float,
    lat0: float,
    width_km: float,
    height_km: float,
    height_m: float = 0.0,
    freq: str = 'A',
) -> dict:
    """Find the azimuth-line / range-bin bounding box covering a lon/lat area.

    Uses ISCE3 geo2rdr on the corners (and center) of the box on a
    constant-height ellipsoid; adequate for a several-km AOI. width_km/
    height_km are applied as an east-west/north-south box, a good enough
    proxy for a square AOI since the exact range/azimuth orientation isn't
    known a priori.

    Returns
    -------
    bbox : dict
        Keys ``az_start``, ``az_stop``, ``rg_start``, ``rg_stop`` (indices
        into the full-resolution RSLC grid, stop exclusive).
    """
    rslc_path = Path(rslc_path)
    lons, lats = _box_corners(lon0, lat0, width_km, height_km)

    with h5py.File(rslc_path, 'r') as fh:
        orbit = isce3.core.load_orbit_from_h5_group(fh['science/LSAR/RSLC/metadata/orbit'])
        look = fh['science/LSAR/identification/lookDirection'][()]
        fc = float(fh[f'science/LSAR/RSLC/swaths/frequency{freq}/processedCenterFrequency'][()])
        slant_range = fh[f'science/LSAR/RSLC/swaths/frequency{freq}/slantRange'][()]
        az_time = fh['science/LSAR/RSLC/swaths/zeroDopplerTime'][()]

    side = 'left' if look == b'Left' else 'right'
    ellipsoid = isce3.core.WGS84_ELLIPSOID
    wavelength = isce3.core.speed_of_light / fc
    doppler = isce3.core.LUT2d()

    aztimes, sranges = [], []
    for lon, lat in zip(lons, lats):
        llh = np.array([np.radians(lon), np.radians(lat), height_m])
        xyz = ellipsoid.lon_lat_to_xyz(llh)
        aztime, srange = isce3.geometry.geo2rdr_bracket(xyz, orbit, doppler, wavelength, side)
        aztimes.append(aztime)
        sranges.append(srange)

    az_idx = np.interp(aztimes, az_time, np.arange(len(az_time)))
    rg_idx = np.interp(sranges, slant_range, np.arange(len(slant_range)))

    return dict(
        az_start=int(np.floor(az_idx.min())),
        az_stop=int(np.ceil(az_idx.max())) + 1,
        rg_start=int(np.floor(rg_idx.min())),
        rg_stop=int(np.ceil(rg_idx.max())) + 1,
    )


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
