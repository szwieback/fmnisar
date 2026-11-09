"""Geocode Sentinel-1 IW burst sub-look stacks onto a map grid with isce3, reusing geocode.py.

Input stacks are baseband, as geocode_slc requires: deramped bursts from sentinel_ioput and sub-looks from
AzimuthSubaperture with demodulate_subaperture=True. Orbit comes from the annotation, not a precise orbit
file. Bistatic delay, azimuth FM-rate mismatch, tropospheric delay and ionospheric delay are not applied.
"""
from datetime import timedelta
import xml.etree.ElementTree as ET

import isce3
import numpy as np
from osgeo import gdal, osr

from geocode import geocode_complex
from sentinel.sentinel_ioput import S1Burst, _files, _time


def annotation_orbit(safe, swath: str, pol: str = 'VV') -> isce3.core.Orbit:
    """isce3 Orbit from the annotation state vectors of one swath."""
    root = ET.parse(_files(safe, swath, pol)[0]).getroot()
    vectors = root.findall('generalAnnotation/orbitList/orbit')
    start = _time(vectors[0].find('time').text)
    spacing = (_time(vectors[1].find('time').text) - start).total_seconds()
    states = []
    for n, v in enumerate(vectors):
        # annotation times jitter by about 1 microsecond and isce3 requires exactly uniform spacing
        snapped = start + timedelta(seconds=round(spacing) * n)
        if abs((snapped - _time(v.find('time').text)).total_seconds()) > 1e-3:
            raise ValueError(f'orbit state vector {n} is not on the uniform {round(spacing)} s lattice')
        epoch = isce3.core.DateTime(snapped.isoformat())
        position = [float(v.find(f'position/{k}').text) for k in 'xyz']
        velocity = [float(v.find(f'velocity/{k}').text) for k in 'xyz']
        states.append(isce3.core.StateVector(epoch, position, velocity))
    return isce3.core.Orbit(states, states[0].datetime)


def burst_radar_grid(burst: S1Burst, safe, swath: str, orbit: isce3.core.Orbit, pol: str = 'VV',
                     az_start: int = 0, rg_start: int = 0, n_az: int = None,
                     n_rg: int = None) -> isce3.product.RadarGridParameters:
    """Radar grid of one burst, or of the lines and samples starting at (az_start, rg_start)."""
    root = ET.parse(_files(safe, swath, pol)[0]).getroot()
    wavelength = isce3.core.speed_of_light / float(root.find('generalAnnotation/productInformation/radarFrequency').text)
    # same range axis as the deramp and sub-looks
    range_spacing = float(burst.meta.slant_range[1] - burst.meta.slant_range[0])
    n_az_full, n_rg_full = burst.slc.shape
    epoch = orbit.reference_epoch
    sensing_start = (burst.first_line_utc - _time(epoch.isoformat()[:26])).total_seconds()
    grid = isce3.product.RadarGridParameters(sensing_start, wavelength, 1 / burst.meta.zero_doppler_time_spacing,
                                             float(burst.meta.slant_range[0]), range_spacing, isce3.core.LookSide.Right,
                                             n_az_full, n_rg_full, epoch)
    return grid.offset_and_resize(az_start, rg_start, n_az_full - az_start if n_az is None else n_az,
                                  n_rg_full - rg_start if n_rg is None else n_rg)


def geogrid_from_raster(path, row0: int, col0: int, n_rows: int, n_cols: int) -> isce3.product.GeoGridParameters:
    """Map grid matching a crop of an existing north-up projected GeoTIFF."""
    with gdal.ExceptionMgr():
        ds = gdal.Open(str(path))   # keep the dataset alive while it is read
    x0, dx, rotation_x, y0, rotation_y, dy = ds.GetGeoTransform()
    if rotation_x != 0 or rotation_y != 0:
        raise ValueError(f'{path} has a rotated geotransform')
    srs = osr.SpatialReference(ds.GetProjection())
    srs.AutoIdentifyEPSG()
    code = srs.GetAuthorityCode(None)
    if code is None:
        raise ValueError(f'{path} has no EPSG code')
    return isce3.product.GeoGridParameters(x0 + col0 * dx, y0 + row0 * dy, dx, dy, n_cols, n_rows, int(code))


def _check_dem_covers(dem_path, geogrid) -> None:
    """Raise if the EPSG:4326 DEM does not cover the four geogrid corners with a 0.01 deg margin."""
    with gdal.ExceptionMgr():
        ds = gdal.Open(str(dem_path))
    x0, dx, _, y0, _, dy = ds.GetGeoTransform()
    lon0, lon1, lat1, lat0 = x0, x0 + ds.RasterXSize * dx, y0, y0 + ds.RasterYSize * dy
    src, dst = osr.SpatialReference(), osr.SpatialReference()
    src.ImportFromEPSG(geogrid.epsg)
    dst.ImportFromEPSG(4326)
    for ref in (src, dst):
        ref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    to_lonlat = osr.CoordinateTransformation(src, dst)
    xs = geogrid.start_x + np.array([0, geogrid.width]) * geogrid.spacing_x
    ys = geogrid.start_y + np.array([0, geogrid.length]) * geogrid.spacing_y
    for x in xs:
        for y in ys:
            lon, lat = to_lonlat.TransformPoint(float(x), float(y))[:2]
            if not (lon0 + 0.01 <= lon <= lon1 - 0.01 and lat0 + 0.01 <= lat <= lat1 - 0.01):
                raise ValueError(f'DEM {dem_path} does not cover geogrid corner ({lon:.4f}, {lat:.4f})')


def geocode_burst(stack: np.ndarray, burst: S1Burst, safe, swath: str, dem_path, geogrid, valid: np.ndarray,
                  az_start: int = 0, rg_start: int = 0, pol: str = 'VV') -> tuple:
    """Geocode baseband bands (N, n_az, n_rg) of a burst onto geogrid; returns (geo (N, ny, nx), mask).

    valid (n_az, n_rg) marks the radar samples to trust. Mask is 255 where a map pixel is outside the
    burst or its interpolation reached an invalid sample; those pixels are 0. A burst that misses the
    geogrid returns an all-255 mask.
    """
    if valid.shape != stack.shape[-2:]:
        raise ValueError(f'valid shape {valid.shape} differs from stack {stack.shape[-2:]}')
    _check_dem_covers(dem_path, geogrid)
    orbit = annotation_orbit(safe, swath, pol)
    grid = burst_radar_grid(burst, safe, swath, orbit, pol, az_start, rg_start, *stack.shape[-2:])
    dem_raster = isce3.io.Raster(str(dem_path))
    bands = np.concatenate([stack, valid[None].astype(np.complex64)])   # interpolate the mask with the data
    try:
        geo, mask = geocode_complex(bands, grid, geogrid, orbit, dem_raster, isce3.core.LUT2d(), isce3.core.LUT2d())
    except ValueError as error:
        if 'no radar sample' not in str(error):
            raise
        shape = (stack.shape[0], geogrid.length, geogrid.width)
        return np.zeros(shape, dtype=np.complex64), np.full(shape[1:], 255, dtype=np.uint8)
    # a sinc interpolation that touched an invalid sample leaves the geocoded mask below 1
    bad = (np.abs(geo[-1]) < 0.999) | (mask == 255)
    geo = geo[:-1]
    geo[:, bad] = 0
    mask = mask.copy()
    mask[bad] = 255
    return geo, mask
