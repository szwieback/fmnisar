"""Geocode sub-aperture chips onto a map grid with isce3.

Complex sub-look stacks (N, n_az, n_rg) go through isce3.geocode.geocode_slc. Real
multilooked products (B, n_az_ml, n_rg_ml) go through GeocodeFloat32 in area-projection
mode, as GCOV does. Both paths share the chip's radar grid, one DEM and one geogrid, so
the outputs overlay pixel for pixel.

The complex path assumes the stack is baseband (AzimuthSubaperture with
demodulate_subaperture=True and remodulate_to_full_dc=False). A sub-look that still
carries its Doppler carrier would be aliased by the geocoding interpolator.
"""

from pathlib import Path
import argparse

import isce3
import numpy as np
from isce3.atmosphere.tec_product import tec_lut2d_from_json_srg
from nisar.workflows.dumpconfig import point_to_epsg
from osgeo import gdal, osr

from products import covariance_matrix, entropy, mean_coherence, multilook_intensity
from subaperture import AzimuthSubaperture, SubapertureMetaData
from ioput import _box_corners, load_orbit
from subaperture_chip import ChipExtent, SubapertureChip

_threshold_geo2rdr = 1e-8
_numiter_geo2rdr = 25
_sinc_half = 4   # isce3 SINC_HALF: reach of the 8-tap sinc in geocode_slc
_freq = 'A'      # NISAR frequency band the chips are built from


def chip_radar_grid(rslc_path, extent: ChipExtent) -> isce3.product.RadarGridParameters:
    """Radar grid of the chip."""
    full = isce3.product.RadarGridParameters(str(rslc_path), _freq)
    n_az, n_rg = extent.shape
    return full.offset_and_resize(extent.az_start, extent.rg_start, n_az, n_rg)


def _transform(epsg_from: int, epsg_to: int) -> osr.CoordinateTransformation:
    """Coordinate transform with (x, y) / (lon, lat) axis order."""
    srs = []
    for code in (epsg_from, epsg_to):
        ref = osr.SpatialReference()
        ref.ImportFromEPSG(code)
        ref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        srs.append(ref)
    return osr.CoordinateTransformation(*srs)


def box_geogrid(lon_min: float, lat_min: float, lon_max: float, lat_max: float,
                posting_m: float) -> isce3.product.GeoGridParameters:
    """Map grid of a lon/lat box at posting_m [m]: the envelope of its corners in local UTM.

    The projection follows the NISAR convention: UTM zone of the box centre, polar
    stereographic beyond 60 deg latitude.
    """
    epsg = point_to_epsg(0.5 * (lon_min + lon_max), 0.5 * (lat_min + lat_max))
    to_map = _transform(4326, epsg)
    corners = [(lon_min, lat_min), (lon_max, lat_min), (lon_min, lat_max), (lon_max, lat_max)]
    xy = np.array([to_map.TransformPoint(lon, lat)[:2] for lon, lat in corners])
    x0, y1 = xy[:, 0].min(), xy[:, 1].max()
    width = int(np.ceil((xy[:, 0].max() - x0) / posting_m))
    length = int(np.ceil((y1 - xy[:, 1].min()) / posting_m))
    return isce3.product.GeoGridParameters(x0, y1, posting_m, -posting_m, width, length, epsg)


def chip_extent_for_geogrid(rslc_path, geogrid, orbit, dem, srange_correction: isce3.core.LUT2d,
                            looks: tuple = None) -> ChipExtent:
    """Radar extent whose geocoding fills geogrid without gaps.

    isce3 get_radar_bbox solves the geogrid perimeter at the lowest and highest DEM height
    inside it, which bounds the radar coordinates of every pixel in the box for side-looking
    zero-Doppler geometry. Padded by the sinc half-width, one look window when multilooking,
    and in range by the largest slant range correction, then clamped to the scene. A grid
    wholly outside the scene raises ValueError.
    """
    # lon/lat envelope of the geogrid perimeter, 50 points per edge
    to_lonlat = _transform(geogrid.epsg, 4326)
    t = np.linspace(0.0, 1.0, 50)
    edges = np.concatenate([np.c_[t, 0 * t], np.c_[t, 0 * t + 1], np.c_[0 * t, t], np.c_[0 * t + 1, t]])
    xs = geogrid.start_x + edges[:, 0] * geogrid.width * geogrid.spacing_x
    ys = geogrid.start_y + edges[:, 1] * geogrid.length * geogrid.spacing_y
    lonlat = np.array([to_lonlat.TransformPoint(float(x), float(y))[:2] for x, y in zip(xs, ys)])
    lon0, lat0 = lonlat.min(axis=0)
    lon1, lat1 = lonlat.max(axis=0)

    # DEM pixels under the envelope; x_start, y_start are the first pixel's centre
    cols = (lonlat[:, 0] - dem.x_start) / dem.delta_x
    rows = (lonlat[:, 1] - dem.y_start) / dem.delta_y
    c0, c1 = int(np.floor(cols.min())), int(np.ceil(cols.max()))
    r0, r1 = int(np.floor(rows.min())), int(np.ceil(rows.max()))
    if c0 < 0 or r0 < 0 or c1 >= dem.width or r1 >= dem.length:
        raise ValueError(f'DEM does not cover the output grid (lon {lon0:.4f} to {lon1:.4f}, '
                         f'lat {lat0:.4f} to {lat1:.4f})')
    heights = dem.data[r0:r1 + 1, c0:c1 + 1]

    scene = isce3.product.RadarGridParameters(str(rslc_path), _freq)
    try:
        bb = isce3.geometry.get_radar_bbox(geogrid, scene, orbit, float(np.nanmin(heights)),
                                           float(np.nanmax(heights)), isce3.core.LUT2d(), 0)
    except RuntimeError as e:
        raise ValueError(f'output grid (lon {lon0:.4f} to {lon1:.4f}, lat {lat0:.4f} to {lat1:.4f}) '
                         f'is outside the scene {Path(rslc_path).name}') from e
    pad_az = _sinc_half + (looks[0] if looks else 0)
    pad_rg = _sinc_half + (looks[1] if looks else 0)
    if srange_correction.have_data:
        # the geocoder samples at geo2rdr range plus the correction, so cover its largest value
        pad_rg += int(np.ceil(np.abs(srange_correction.data).max() / scene.range_pixel_spacing))
    return ChipExtent(max(bb.first_azimuth_line - pad_az, 0), min(bb.last_azimuth_line + 1 + pad_az, scene.length),
                      max(bb.first_range_sample - pad_rg, 0), min(bb.last_range_sample + 1 + pad_rg, scene.width))


def sublook_centroids_hz(sub: AzimuthSubaperture) -> tuple:
    """(centroid offsets [Hz] per sub-look relative to the chip DC, sub-look bandwidth [Hz])."""
    dt = sub.meta.zero_doppler_time_spacing
    _, fraction_sub = sub._bandwidth_fractions(sub.meta.azimuth_bandwidth, dt)
    half_span = fraction_sub * (1 - sub.overlap) * (sub.n_subapertures - 1) / 2
    offsets = np.linspace(-half_span, half_span, sub.n_subapertures) / dt
    return offsets, fraction_sub / dt


def ionosphere_range_correction(rslc_path, tec_path, orbit, dem_path) -> isce3.core.LUT2d:
    """Ionospheric slant range delay LUT [m] from the NISAR TEC product, as GSLC/GCOV apply it."""
    # built over the whole scene grid so every map pixel the geocoder tries falls inside it
    scene_grid = isce3.product.RadarGridParameters(str(rslc_path), _freq)
    fc = isce3.core.speed_of_light / scene_grid.wavelength
    lut = tec_lut2d_from_json_srg(str(tec_path), fc, orbit, scene_grid, isce3.core.LUT2d(),
                                  str(dem_path))
    lut.bounds_error = False
    return lut


def geocode_complex(stack: np.ndarray, radar_grid, geogrid, orbit, dem_raster,
                    srange_correction: isce3.core.LUT2d) -> tuple:
    """Geocode a baseband sub-look stack (N, n_az, n_rg); returns (geo (N, ny, nx), mask)."""
    shape = (geogrid.length, geogrid.width)
    geo_blocks = [np.zeros(shape, dtype=np.complex64) for _ in stack]
    rdr_blocks = [np.ascontiguousarray(s, dtype=np.complex64) for s in stack]
    # mask is 255 where no radar sample maps to the map pixel
    mask = np.full(shape, 255, dtype=np.uint8)
    # zero native Doppler: the stack is baseband. flatten=False keeps the same geometric
    # phase in every sub-look so sub-look phase differences survive geocoding.
    isce3.geocode.geocode_slc(
        geo_blocks, rdr_blocks, dem_raster, radar_grid, geogrid, orbit,
        isce3.core.LUT2d(), isce3.core.LUT2d(), isce3.core.WGS84_ELLIPSOID,
        _threshold_geo2rdr, _numiter_geo2rdr, mask_block=mask, flatten=False,
        srange_correction=srange_correction)
    return np.stack(geo_blocks), mask


def geocode_real(bands: np.ndarray, radar_grid, geogrid, orbit, dem_raster,
                 srange_correction: isce3.core.LUT2d) -> np.ndarray:
    """Geocode real bands (B, n_az, n_rg) by area projection; returns (B, ny, nx) float32."""
    bands = np.asarray(bands, dtype=np.float32)
    n_bands, n_az, n_rg = bands.shape
    geocoder = isce3.geocode.GeocodeFloat32()
    geocoder.orbit = orbit
    geocoder.ellipsoid = isce3.core.WGS84_ELLIPSOID
    geocoder.doppler = isce3.core.LUT2d()
    geocoder.threshold_geo2rdr = _threshold_geo2rdr
    geocoder.numiter_geo2rdr = _numiter_geo2rdr
    geocoder.geogrid(geogrid.start_x, geogrid.start_y, geogrid.spacing_x, geogrid.spacing_y,
                     geogrid.width, geogrid.length, geogrid.epsg)
    # GeocodeFloat32 works on GDAL rasters, so the arrays go through in-memory GDAL datasets
    mem = gdal.GetDriverByName('MEM')
    in_ds = mem.Create('', n_rg, n_az, n_bands, gdal.GDT_Float32)
    in_ds.WriteArray(bands)
    out_ds = mem.Create('', geogrid.width, geogrid.length, n_bands, gdal.GDT_Float32)
    geocoder.geocode(radar_grid, isce3.io.Raster(in_ds.this), isce3.io.Raster(out_ds.this), dem_raster,
                     output_mode=isce3.geocode.GeocodeOutputMode.AREA_PROJECTION,
                     slant_range_correction=srange_correction)
    return out_ds.ReadAsArray().reshape(n_bands, geogrid.length, geogrid.width)


def write_geotiff(path, data: np.ndarray, geogrid, band_names: list, metadata: dict) -> None:
    """Write (B, ny, nx) float32 or complex64 bands with the geogrid's georeferencing."""
    n_bands = data.shape[0]
    dtype = gdal.GDT_CFloat32 if np.iscomplexobj(data) else gdal.GDT_Float32
    ds = gdal.GetDriverByName('GTiff').Create(str(path), geogrid.width, geogrid.length,
                                              n_bands, dtype)
    ds.SetGeoTransform((geogrid.start_x, geogrid.spacing_x, 0.0,
                        geogrid.start_y, 0.0, geogrid.spacing_y))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(geogrid.epsg)
    ds.SetProjection(srs.ExportToWkt())
    ds.SetMetadata({k: str(v) for k, v in metadata.items()})
    for b in range(n_bands):
        band = ds.GetRasterBand(b + 1)
        band.WriteArray(data[b])
        band.SetDescription(band_names[b])
        band.SetNoDataValue(float('nan'))  # geocoded gaps are NaN
    ds = None
    print(f'wrote {path}')


def geocode_chip(rslc_path, dem_path, geogrid, name: str, folder_out,
                 n_subapertures: int = 5, overlap: float = 0.2, looks: int = 9,
                 pol: str = 'HH', tec_path=None, ionosphere: bool = True) -> tuple:
    """Sub-aperture process the radar extent that fills geogrid and geocode stack and products.

    Returns the two GeoTIFF paths. The radar extent comes from chip_extent_for_geogrid.

    ionosphere=True applies the NISAR TEC slant range correction (tec_path, the scene's
    NISAR_ANC_TEC JSON) in both geocoders. The RSLC carries the ionospheric delay, about
    5 m slant at L-band, which plain geo2rdr ignores; see scratch/cr_range_delay_budget.py.
    """
    if ionosphere and tec_path is None:
        raise ValueError('ionosphere=True needs tec_path (NISAR_ANC_TEC JSON); '
                         'pass ionosphere=False to geocode without the correction')
    rslc_path, folder_out = Path(rslc_path), Path(folder_out)
    folder_out.mkdir(parents=True, exist_ok=True)
    dem_raster = isce3.io.Raster(str(dem_path))
    dem = isce3.geometry.DEMInterpolator(dem_raster)  # loads the raster on construction
    orbit = load_orbit(rslc_path)
    # an empty LUT means no slant range correction in the isce3 geocoders
    srange_lut = (ionosphere_range_correction(rslc_path, tec_path, orbit, dem_path) if ionosphere
                  else isce3.core.LUT2d())
    extent = chip_extent_for_geogrid(rslc_path, geogrid, orbit, dem, srange_lut, looks=(looks, looks))
    print(f'geogrid EPSG:{geogrid.epsg}, {geogrid.length} x {geogrid.width} at {geogrid.spacing_x} m; '
          f'radar extent {extent}')

    meta = SubapertureMetaData.load_from_rslc_path(rslc_path)
    sub = AzimuthSubaperture(
        meta, n_subapertures=n_subapertures, overlap=overlap,
        demodulate_subaperture=True, remodulate_to_full_dc=False, deweight_spectrum=True)
    chip = SubapertureChip(sub, rslc_path, extent, pol=pol)
    # de-weighting area clamped so cropped RSLCs smaller than the default still work
    deweight_area = tuple(min(d, n) for d, n in zip((8192, 1024), chip.scene_shape))
    stack = chip.process(deweight_area=deweight_area)
    centroids_hz, bandwidth_hz = sublook_centroids_hz(sub)
    print(f'sub-look stack {stack.shape}, chip DC {chip.doppler_centroid:.1f} Hz, '
          f'sub-look bandwidth {bandwidth_hz:.1f} Hz')

    grid_full = chip_radar_grid(rslc_path, extent)
    # isce3 shifts the multilooked origin by (looks - 1) / 2, matching products.boxcar window centres
    grid_ml = grid_full.multilook(looks, looks)

    if ionosphere:
        print(f'ionospheric slant range correction at chip centre: '
              f'{srange_lut.eval(grid_full.sensing_mid, grid_full.mid_range):.2f} m')

    common = dict(rslc=rslc_path.name, dem=Path(dem_path).name, pol=pol,
                  az_start=extent.az_start, rg_start=extent.rg_start,
                  n_subapertures=n_subapertures, overlap=overlap,
                  chip_doppler_centroid_hz=f'{chip.doppler_centroid:.3f}',
                  sublook_centroid_offsets_hz=' '.join(f'{c:.3f}' for c in centroids_hz),
                  sublook_bandwidth_hz=f'{bandwidth_hz:.3f}', baseband=True,
                  ionosphere_correction=ionosphere,
                  tec=Path(tec_path).name if ionosphere else '')
    # product name through the acquisition start time (cycle, track, frame, mode, date)
    stem = '_'.join(rslc_path.stem.split('_')[:12]) + f'_{name}'
    sublooks_path = folder_out / f'{stem}_sublooks.tif'
    products_path = folder_out / f'{stem}_products.tif'
    sub_names = [f'sublook_{j}' for j in range(n_subapertures)]

    geo_stack, mask = geocode_complex(stack, grid_full, geogrid, orbit, dem_raster, srange_lut)
    print(f'complex: {np.mean(mask != 255) * 100:.1f}% of map pixels covered')
    write_geotiff(sublooks_path, geo_stack, geogrid, sub_names, dict(common, flatten=False))

    # multilooked products on the (looks, looks) grid: per-sub-look intensity, mean coherence, entropy
    intensity = multilook_intensity(stack, (looks, looks))
    cov, coh = covariance_matrix(stack, (looks, looks))
    bands = np.concatenate([intensity, mean_coherence(coh)[None], entropy(cov)[None]])
    geo_bands = geocode_real(bands, grid_ml, geogrid, orbit, dem_raster, srange_lut)
    print(f'real: {np.mean(np.isfinite(geo_bands[0])) * 100:.1f}% of map pixels covered')
    write_geotiff(products_path, geo_bands, geogrid,
                  [f'intensity_{n}' for n in sub_names] + ['mean_coherence', 'entropy'],
                  dict(common, looks=looks))
    return sublooks_path, products_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('rslc')
    parser.add_argument('dem', help='WGS84 lon/lat DEM GeoTIFF covering the chip')
    parser.add_argument('--lon', type=float, required=True)
    parser.add_argument('--lat', type=float, required=True)
    parser.add_argument('--km', type=float, default=1.0, help='chip side [km]')
    parser.add_argument('--out', default='data/processed/geocode')
    parser.add_argument('--n-subapertures', type=int, default=5)
    parser.add_argument('--looks', type=int, default=9)
    parser.add_argument('--posting', type=float, default=20.0, help='map posting [m]')
    parser.add_argument('--pol', default='HH')
    parser.add_argument('--tec', help='NISAR_ANC_TEC JSON for the scene (hyp3_isce3.process.get_tec)')
    parser.add_argument('--no-ionosphere', dest='ionosphere', action='store_false',
                        help='skip the TEC slant range correction')
    args = parser.parse_args()

    # output grid: the km x km lon/lat box about (lon, lat); the radar extent follows from it
    lons, lats = _box_corners(args.lon, args.lat, args.km, args.km)
    geogrid = box_geogrid(lons.min(), lats.min(), lons.max(), lats.max(), args.posting)
    geocode_chip(args.rslc, args.dem, geogrid, f'{args.lon:.4f}_{args.lat:.4f}', args.out,
                 n_subapertures=args.n_subapertures, looks=args.looks,
                 pol=args.pol, tec_path=args.tec, ionosphere=args.ionosphere)
