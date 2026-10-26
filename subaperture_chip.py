"""Subaperture processing of a single chip with a constant chip-centre Doppler centroid."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import math
import warnings
import numpy as np

from ioput import (read_block, read_grid_shape, latlon_to_radar_index, latlon_points_to_radar_bbox,
                   latlon_box_to_radar_bbox)
from subaperture import AzimuthSubaperture


def _interval_within(center: int, length: int, n: int) -> tuple:
    """[start, stop) of the given length about center, shifted to lie within [0, n)."""
    if length > n:
        raise ValueError(f'requested {length} samples, only {n} available')
    start = min(max(center - length // 2, 0), n - length)
    return start, start + length


@dataclass(frozen=True)
class ChipExtent:
    """Azimuth/range index extent of a chip on the full-resolution RSLC grid (stop exclusive)."""
    az_start: int
    az_stop: int
    rg_start: int
    rg_stop: int

    @classmethod
    def from_center(cls, az_center: int, rg_center: int, n_az: int, n_rg: int,
                    n_az_scene: int, n_rg_scene: int) -> 'ChipExtent':
        """n_az x n_rg extent about (az_center, rg_center), shifted to lie within the scene."""
        az_start, az_stop = _interval_within(az_center, n_az, n_az_scene)
        rg_start, rg_stop = _interval_within(rg_center, n_rg, n_rg_scene)
        return cls(az_start, az_stop, rg_start, rg_stop)

    @classmethod
    def from_latlon_center(cls, rslc_path: Path, lon: float, lat: float, n_az: int, n_rg: int,
                           height_m: float = 0.0, freq: str = 'A') -> 'ChipExtent':
        """n_az x n_rg extent about the radar position of (lon, lat)."""
        az_idx, rg_idx = latlon_to_radar_index(rslc_path, lon, lat, height_m=height_m, freq=freq)
        n_az_scene, n_rg_scene = read_grid_shape(rslc_path, freq=freq)
        return cls.from_center(int(round(az_idx[0])), int(round(rg_idx[0])), n_az, n_rg,
                               n_az_scene, n_rg_scene)

    @classmethod
    def from_latlon_bbox(cls, rslc_path: Path, lon_min: float, lat_min: float, lon_max: float,
                         lat_max: float, height_m: float = 0.0, freq: str = 'A') -> 'ChipExtent':
        """Extent enclosing the radar positions of the lon/lat bbox corners."""
        lons = [lon_min, lon_max, lon_min, lon_max]
        lats = [lat_min, lat_min, lat_max, lat_max]
        return cls(**latlon_points_to_radar_bbox(rslc_path, lons, lats, height_m=height_m,
                                                 freq=freq))

    @classmethod
    def from_latlon_area(cls, rslc_path: Path, lon: float, lat: float, width_km: float,
                         height_km: float, height_m: float = 0.0, freq: str = 'A') -> 'ChipExtent':
        """Extent enclosing an east-west x north-south box [km] about (lon, lat)."""
        return cls(**latlon_box_to_radar_bbox(rslc_path, lon, lat, width_km, height_km,
                                              height_m=height_m, freq=freq))

    @property
    def center(self) -> tuple:
        return (self.az_start + self.az_stop) // 2, (self.rg_start + self.rg_stop) // 2

    @property
    def shape(self) -> tuple:
        return self.az_stop - self.az_start, self.rg_stop - self.rg_start

    def resized(self, n_az: int, n_rg: int, n_az_scene: int, n_rg_scene: int) -> 'ChipExtent':
        """n_az x n_rg extent about this extent's centre, shifted to lie within the scene."""
        return ChipExtent.from_center(*self.center, n_az, n_rg, n_az_scene, n_rg_scene)


def azimuth_margin(sub: AzimuthSubaperture, n_widths: float = 6) -> int:
    """Azimuth margin [pixels] of n_widths sub-aperture impulse-response widths."""
    _, fraction_sub = sub._bandwidth_fractions(
        sub.meta.azimuth_bandwidth, sub.meta.zero_doppler_time_spacing)
    return math.ceil(n_widths / fraction_sub)


class SubapertureChip:
    """Chip sub-apertures at the chip-centre Doppler centroid and sub-aperture phase reference."""

    def __init__(self, sub: AzimuthSubaperture, rslc_path: Path, chip_extent: ChipExtent,
                 freq='A', pol='HH'):
        if freq != sub.meta.freq:
            raise ValueError(f'freq {freq} does not match metadata frequency {sub.meta.freq}')
        self.sub = sub
        self.rslc_path = Path(rslc_path)
        self.chip_extent = chip_extent
        self.freq = freq
        self.pol = pol

    @property
    def scene_shape(self) -> tuple:
        return len(self.sub.meta.az_time), len(self.sub.meta.slant_range)

    @cached_property
    def doppler_centroid(self) -> float:
        """Doppler centroid [Hz] at the chip centre."""
        return self.sub.interpolate_doppler_centroid(*self.chip_extent.center)

    def deweighting_extent(self, n_az_area: int, n_rg_area: int) -> ChipExtent:
        return self.chip_extent.resized(n_az_area, n_rg_area, *self.scene_shape)

    def estimate_deweighting(self, n_az_area: int = 8192, n_rg_area: int = 1024) -> np.ndarray:
        """Store sub's de-weighting profile from n_az_area x n_rg_area about the chip centre."""
        area = self.deweighting_extent(n_az_area, n_rg_area)
        return self.sub.estimate_deweighting_profile(
            self.rslc_path, az_start=area.az_start, az_stop=area.az_stop,
            rg_start=area.rg_start, rg_stop=area.rg_stop,
            freq=self.freq, pol=self.pol, doppler_centroid=self.doppler_centroid)

    def read(self) -> np.ndarray:
        e = self.chip_extent
        return read_block(self.rslc_path, e.az_start, e.az_stop, e.rg_start, e.rg_stop,
                          freq=self.freq, pol=self.pol)

    def process(self, az_margin=None, deweight_area=(8192, 1024)) -> np.ndarray:
        """Sub-aperture stack (N, n_az, n_rg); deweight_area=None reuses sub's de-weighting."""
        if self.sub.deweight_spectrum and deweight_area is not None:
            self.estimate_deweighting(*deweight_area)
        elif self.sub.deweight_spectrum and self.sub._deweight_profile is None:
            warnings.warn('no de-weighting profile stored; estimating it from the chip block alone '
                          '(noisy) and caching it on sub', UserWarning, stacklevel=2)
        if az_margin is None:
            az_margin = azimuth_margin(self.sub)
        e = self.chip_extent
        n_az_scene, _ = self.scene_shape
        read_start = max(e.az_start - az_margin, 0)
        read_stop = min(e.az_stop + az_margin, n_az_scene)
        if read_start > e.az_start - az_margin or read_stop < e.az_stop + az_margin:
            warnings.warn(f'azimuth margin of {az_margin} pixels truncated at scene edge',
                          UserWarning, stacklevel=2)
        block = read_block(self.rslc_path, read_start, read_stop, e.rg_start, e.rg_stop,
                           freq=self.freq, pol=self.pol)
        block_sub = self.sub._process_block(
            block, doppler_centroid=self.doppler_centroid,
            az_time_start=float(self.sub.meta.az_time[read_start]),
            az_time_ref=float(self.sub.meta.az_time[self.chip_extent.center[0]]))
        crop_start = e.az_start - read_start
        n_az_chip, _ = e.shape
        return block_sub[:, crop_start:crop_start + n_az_chip, :]


if __name__ == '__main__':
    from subaperture import SubapertureMetaData, plot_subaperture_diagnostics

    path_nisar = Path(
        '/media/simon/Extreme SSD/fmnisar/Sydney/'
        'NISAR_L1_PR_RSLC_025_016_D_105_4005_DHDH_A_20260709T075915_20260709T075950_P05023_N_F_J_001.h5')
    folder_out = path_nisar.parent / 'processed' / 'chip'
    folder_out.mkdir(parents=True, exist_ok=True)

    meta = SubapertureMetaData.load_from_rslc_path(path_nisar)
    sub = AzimuthSubaperture(
        meta, n_subapertures=9, overlap=0.2,
        demodulate_subaperture=True, remodulate_to_full_dc=False, deweight_spectrum=True)

    chip_extent = ChipExtent.from_latlon_area(path_nisar, 151.215256, -33.856784, 1.0, 1.0)
    chip = SubapertureChip(sub, path_nisar, chip_extent, pol='HH')
    print(f'chip extent: {chip_extent}, DC: {chip.doppler_centroid:.2f} Hz, '
          f'area extent: {chip.deweighting_extent(8192, 1024)}')

    block_sub = chip.process()
    print(f'chip stack shape: {block_sub.shape}')

    plot_subaperture_diagnostics(
        sub, chip.read(), doppler_centroid=chip.doppler_centroid,
        save_path=folder_out / 'chip_subaperture_diagnostics.png')
