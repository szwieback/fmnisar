"""ESA WorldCover land cover for a lon/lat box, on a geocode.py map grid or the native grid."""

from pathlib import Path
import math
import shutil
import time
import urllib.error
import urllib.request
import uuid

import numpy as np
from osgeo import gdal, osr


class WorldCover:
    """ESA WorldCover 10 m tiles cached locally, clipped to a lon/lat box."""

    labels = {
        10: 'Tree cover',
        20: 'Shrubland',
        30: 'Grassland',
        40: 'Cropland',
        50: 'Built-up',
        60: 'Bare/sparse vegetation',
        70: 'Snow and ice',
        80: 'Permanent water bodies',
        90: 'Herbaceous wetland',
        95: 'Mangroves',
        100: 'Moss and lichen',
    }
    groups = {
        10: 'Vegetation',
        20: 'Sparse / Low Vegetation',
        30: 'Sparse / Low Vegetation',
        40: 'Sparse / Low Vegetation',
        50: 'Urban/Built',
        60: 'Bare surfaces',
        70: 'Cryosphere',
        80: 'Water',
        90: 'Other wet vegetation',
        95: 'Other wet vegetation',
        100: 'Sparse / Low Vegetation',
    }
    nodata = 0
    _years = {'v100': 2020, 'v200': 2021}
    _base = 'https://esa-worldcover.s3.eu-central-1.amazonaws.com'
    _tile_deg = 3
    _res_deg = 1 / 12000  # 10 m native posting; tile origins are whole degrees

    def __init__(self, tile_cache, version: str = 'v200', retries: int = 3):
        if version not in self._years:
            raise ValueError(f'unknown WorldCover version {version}; known: {list(self._years)}')
        self.tile_cache = Path(tile_cache)
        self.version = version
        self.year = self._years[version]
        self.retries = retries

    @property
    def source(self) -> str:
        return f'ESA WorldCover 10m {self.year} {self.version}'

    @property
    def base_url(self) -> str:
        return f'{self._base}/{self.version}/{self.year}/map'

    def tile_name(self, tile: str) -> str:
        return f'ESA_WorldCover_10m_{self.year}_{self.version}_{tile}_Map.tif'

    def tile_ids(self, lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> list:
        """Ids of the 3 x 3 deg tiles intersecting the box, named by their SW corner."""
        d = self._tile_deg
        lon0, lon1 = math.floor(lon_min / d) * d, math.floor((lon_max - 1e-12) / d) * d
        lat0, lat1 = math.floor(lat_min / d) * d, math.floor((lat_max - 1e-12) / d) * d
        return [f'{"N" if lat >= 0 else "S"}{abs(lat):02d}{"E" if lon >= 0 else "W"}{abs(lon):03d}'
                for lat in range(lat0, lat1 + 1, d) for lon in range(lon0, lon1 + 1, d)]

    def tile_path(self, tile: str):
        """Local path of the tile, downloaded if not cached; None if WorldCover has no such tile."""
        path = self.tile_cache / self.tile_name(tile)
        if path.exists() and path.stat().st_size > 0:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(path.suffix + '.part')
        url = f'{self.base_url}/{path.name}'
        for attempt in range(1, self.retries + 1):
            try:
                print(f'downloading {url}')
                with urllib.request.urlopen(url, timeout=120) as response, part.open('wb') as dst:
                    shutil.copyfileobj(response, dst, length=1024 * 1024)
                part.replace(path)
                return path
            except urllib.error.HTTPError as e:
                part.unlink(missing_ok=True)
                # tiles exist only over land
                if e.code == 404:
                    return None
                if attempt == self.retries:
                    raise
            except OSError:
                part.unlink(missing_ok=True)
                if attempt == self.retries:
                    raise
            time.sleep(5 * attempt)

    def clip(self, lon_min: float, lat_min: float, lon_max: float, lat_max: float,
             geogrid=None, path=None, resampling: str = 'near') -> np.ndarray:
        """Class codes (ny, nx) uint8 on geogrid, or on the native EPSG:4326 grid if None."""
        if geogrid is None:
            epsg, geotransform, width, length = self._native_grid(lon_min, lat_min, lon_max, lat_max)
            alignment = 'worldcover_native_grid'
        else:
            epsg, width, length = geogrid.epsg, geogrid.width, geogrid.length
            geotransform = (geogrid.start_x, geogrid.spacing_x, 0.0,
                            geogrid.start_y, 0.0, geogrid.spacing_y)
            alignment = 'geogrid'
            # snapped corners in a projected CRS can reach past the lon/lat box
            env = _lonlat_envelope(geogrid)
            lon_min, lat_min = min(lon_min, env[0]), min(lat_min, env[1])
            lon_max, lat_max = max(lon_max, env[2]), max(lat_max, env[3])

        tiles = self.tile_ids(lon_min, lat_min, lon_max, lat_max)
        tile_paths = [p for p in (self.tile_path(t) for t in tiles) if p is not None]

        with gdal.ExceptionMgr():
            if path is None:
                dst = gdal.GetDriverByName('MEM').Create('', width, length, 1, gdal.GDT_Byte)
            else:
                dst = gdal.GetDriverByName('GTiff').Create(
                    str(path), width, length, 1, gdal.GDT_Byte,
                    options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER'])
            dst.SetGeoTransform(geotransform)
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(epsg)
            dst.SetProjection(srs.ExportToWkt())
            band = dst.GetRasterBand(1)
            band.SetNoDataValue(self.nodata)
            band.Fill(self.nodata)
            dst.SetMetadata({'source': self.source, 'source_url': self.base_url,
                             'tiles': ';'.join(tiles), 'alignment': alignment,
                             'epsg': str(epsg), 'resampling': resampling})
            if tile_paths:
                vrt_path = f'/vsimem/worldcover_{uuid.uuid4().hex}.vrt'
                vrt = gdal.BuildVRT(vrt_path, [str(p) for p in tile_paths],
                                    srcNodata=self.nodata, VRTNodata=self.nodata)
                gdal.Warp(dst, vrt, resampleAlg=resampling,
                          srcNodata=self.nodata, dstNodata=self.nodata)
                vrt = None
                gdal.Unlink(vrt_path)
            else:
                print(f'no WorldCover tile covers lon {lon_min:.4f} to {lon_max:.4f}, '
                      f'lat {lat_min:.4f} to {lat_max:.4f}')
            dst.FlushCache()
            codes = band.ReadAsArray()
            band, dst = None, None
        if path is not None:
            print(f'wrote {path}')
        return codes

    def _native_grid(self, lon_min, lat_min, lon_max, lat_max) -> tuple:
        """(epsg, geotransform, width, length) of the native pixels intersecting the box."""
        n = 1 / self._res_deg
        # 1e-6 pixel tolerance so box edges on the lattice do not gain a pixel to rounding
        c0, c1 = math.floor(lon_min * n + 1e-6), math.ceil(lon_max * n - 1e-6)
        r0, r1 = math.floor(lat_min * n + 1e-6), math.ceil(lat_max * n - 1e-6)
        geotransform = (c0 / n, self._res_deg, 0.0, r1 / n, 0.0, -self._res_deg)
        return 4326, geotransform, c1 - c0, r1 - r0

    @classmethod
    def label(cls, code):
        if code is None or not np.isfinite(code) or int(code) == cls.nodata:
            return None
        return cls.labels.get(int(code), f'class_{int(code)}')

    @classmethod
    def group(cls, code):
        if code is None or not np.isfinite(code) or int(code) == cls.nodata:
            return None
        return cls.groups.get(int(code), 'Other')


def _lonlat_envelope(geogrid, n_edge: int = 50) -> tuple:
    """(lon_min, lat_min, lon_max, lat_max) of the geogrid perimeter."""
    src, dst = osr.SpatialReference(), osr.SpatialReference()
    src.ImportFromEPSG(geogrid.epsg)
    dst.ImportFromEPSG(4326)
    for ref in (src, dst):
        ref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    to_lonlat = osr.CoordinateTransformation(src, dst)
    t = np.linspace(0.0, 1.0, n_edge)
    edges = np.concatenate([np.c_[t, 0 * t], np.c_[t, 0 * t + 1], np.c_[0 * t, t], np.c_[0 * t + 1, t]])
    xs = geogrid.start_x + edges[:, 0] * geogrid.width * geogrid.spacing_x
    ys = geogrid.start_y + edges[:, 1] * geogrid.length * geogrid.spacing_y
    lonlat = np.array([to_lonlat.TransformPoint(float(x), float(y))[:2] for x, y in zip(xs, ys)])
    return (*lonlat.min(axis=0), *lonlat.max(axis=0))
