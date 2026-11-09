"""I/O for Sentinel-1 IW SLC bursts: read one burst from a SAFE folder and deramp it for sub-apertures."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET

import isce3
import numpy as np
from osgeo import gdal
from scipy.ndimage import binary_erosion

from subaperture import AzimuthSubaperture, SubapertureMetaData
from subaperture_chip import azimuth_margin


@dataclass
class S1Burst:
    """One IW burst: raw and deramped SLC (n_az, n_rg), with the metadata for AzimuthSubaperture."""
    slc: np.ndarray            # complex64, as stored; invalid samples are 0
    deramped: np.ndarray       # complex64, azimuth spectrum stationary and centered at 0 Hz
    valid: np.ndarray          # bool (n_az, n_rg), samples inside the burst's valid region
    meta: SubapertureMetaData  # doppler centroid is 0 because the burst is deramped
    first_line_utc: datetime   # zero-Doppler time of the first line


def _time(text: str) -> datetime:
    return datetime.fromisoformat(text)


def _floats(text: str) -> np.ndarray:
    return np.array(text.split(), dtype=float)


def _nearest(items, t: datetime):
    """The annotation record whose azimuthTime is closest to t."""
    return min(items, key=lambda e: abs((_time(e.find('azimuthTime').text) - t).total_seconds()))


def _polyval(element, name: str, t0_name: str, tau: np.ndarray) -> np.ndarray:
    """Evaluate an ascending-order annotation polynomial at two-way range time tau [s]."""
    coefficients = _floats(element.find(name).text)
    return np.polynomial.polynomial.polyval(tau - float(element.find(t0_name).text), coefficients)


def _velocity(root, t: datetime) -> float:
    """Platform speed [m/s] at time t, linearly interpolated from the annotation orbit."""
    orbit = root.findall('generalAnnotation/orbitList/orbit')
    times = np.array([(_time(o.find('time').text) - t).total_seconds() for o in orbit])
    vel = np.array([[float(o.find(f'velocity/{k}').text) for k in 'xyz'] for o in orbit])
    return float(np.linalg.norm([np.interp(0.0, times, vel[:, i]) for i in range(3)]))


def _files(safe, swath: str, pol: str) -> tuple:
    """(annotation XML, measurement TIFF) of one swath and polarization in a SAFE folder."""
    pattern = f's1?-{swath.lower()}-slc-{pol.lower()}-*'
    annotations = sorted((Path(safe) / 'annotation').glob(pattern + '.xml'))
    measurements = sorted((Path(safe) / 'measurement').glob(pattern + '.tiff'))
    if not annotations or not measurements:
        raise FileNotFoundError(f'no {swath} {pol} annotation and measurement in {safe}')
    return annotations[0], measurements[0]


def deramp_phase(root, burst, tau: np.ndarray, n_az: int, dt: float) -> np.ndarray:
    """TOPS signal phase (n_az, n_rg) [rad]; multiplying the SLC by exp(-j phase) removes the ramp.

    Quantities follow ESA's TOPS deramping note. The overall sign was set by measurement: the raw
    local azimuth frequency rises along the burst at +k_t, see scratch/check_sentinel_ioput.py.
    """
    # azimuthTime is the first line; annotation records are chosen at the burst center
    mid_time = _time(burst.find('azimuthTime').text) + timedelta(seconds=(n_az - 1) / 2 * dt)
    product = root.find('generalAnnotation/productInformation')
    wavelength = isce3.core.speed_of_light / float(product.find('radarFrequency').text)
    steering_deg = float(product.find('azimuthSteeringRate').text)
    if not 0.5 < steering_deg < 2.5:
        raise ValueError(f'azimuth steering rate {steering_deg} deg/s outside the S1 IW range')
    steering = np.radians(steering_deg)
    # Doppler rate from the beam steering [Hz/s], from the velocity at the burst
    k_s = 2 * _velocity(root, mid_time) / wavelength * steering
    fm = _nearest(root.findall('generalAnnotation/azimuthFmRateList/azimuthFmRate'), mid_time)
    dc = _nearest(root.findall('dopplerCentroid/dcEstimateList/dcEstimate'), mid_time)
    if fm.find('azimuthFmRatePolynomial') is None:
        raise ValueError('unsupported annotation: azimuth FM rate is not stored as azimuthFmRatePolynomial')
    # an unreliable data Doppler estimate would mis-center the deramped spectrum without warning
    if dc.find('dataDcRmsErrorAboveThreshold').text.strip().lower() == 'true':
        raise ValueError(f'data Doppler estimate at {dc.find("azimuthTime").text} is flagged unreliable')
    k_a = _polyval(fm, 'azimuthFmRatePolynomial', 't0', tau)     # azimuth FM rate [Hz/s]
    f_dc = _polyval(dc, 'dataDcPolynomial', 't0', tau)           # Doppler centroid [Hz]
    k_t = k_a * k_s / (k_a - k_s)                                # Doppler rate in the focused burst [Hz/s]
    # reference time of the ramp in each range sample, relative to the swath center
    eta_c = -f_dc / k_a
    eta_ref = eta_c - eta_c[tau.size // 2]
    # zero-Doppler time from the burst center [s]
    eta = (np.arange(n_az) - (n_az - 1) / 2)[:, None] * dt
    return np.pi * k_t * (eta - eta_ref) ** 2 + 2 * np.pi * f_dc * (eta - eta_ref)


def n_bursts(safe, swath: str, pol: str = 'VV') -> int:
    """Number of bursts in one swath."""
    root = ET.parse(_files(safe, swath, pol)[0]).getroot()
    return len(root.findall('swathTiming/burstList/burst'))


def read_burst(safe, swath: str, burst_index: int, pol: str = 'VV') -> S1Burst:
    """Burst burst_index (0-based) of swath ('IW2') from a SAFE folder, raw and deramped."""
    xml_path, tiff_path = _files(safe, swath, pol)
    root = ET.parse(xml_path).getroot()
    n_az = int(root.find('swathTiming/linesPerBurst').text)
    n_rg = int(root.find('swathTiming/samplesPerBurst').text)
    bursts = root.findall('swathTiming/burstList/burst')
    if not 0 <= burst_index < len(bursts):
        raise IndexError(f'burst_index {burst_index} outside 0 to {len(bursts) - 1}')
    burst = bursts[burst_index]
    info = root.find('imageAnnotation/imageInformation')
    dt = float(info.find('azimuthTimeInterval').text)
    fs = float(root.find('generalAnnotation/productInformation/rangeSamplingRate').text)
    tau = float(info.find('slantRangeTime').text) + np.arange(n_rg) / fs   # two-way range time [s]
    slant_range = tau * isce3.core.speed_of_light / 2
    if not 6e5 < slant_range.mean() < 1.2e6:
        raise ValueError(f'slant range {slant_range.mean():.0f} m outside the S1 IW range')
    az_processing = root.find('imageAnnotation/processingInformation/swathProcParamsList/swathProcParams/azimuthProcessing')
    bandwidth = float(az_processing.find('processingBandwidth').text)

    # bursts are stored one after another in the swath TIFF
    with gdal.ExceptionMgr():
        ds = gdal.Open(str(tiff_path))   # keep the dataset alive while its band is read
        slc = ds.GetRasterBand(1).ReadAsArray(0, burst_index * n_az, n_rg, n_az)
    slc = slc.astype(np.complex64)

    # each line has its own valid sample range; -1 marks a line with no valid samples
    first, last = _floats(burst.find('firstValidSample').text), _floats(burst.find('lastValidSample').text)
    samples = np.arange(n_rg)[None, :]
    valid = (first[:, None] >= 0) & (samples >= first[:, None]) & (samples <= last[:, None])
    slc[~valid] = 0

    phase = deramp_phase(root, burst, tau, n_az, dt)
    deramped = slc * np.exp(-1j * phase).astype(np.complex64)

    start = _time(burst.find('azimuthTime').text)
    az_time = np.arange(n_az) * dt
    meta = SubapertureMetaData(
        zero_doppler_time_spacing=dt, azimuth_bandwidth=bandwidth,
        doppler_centroid=np.zeros((2, 2)),   # 0 Hz everywhere after deramping
        doppler_az_time=az_time[[0, -1]], doppler_slant_range=slant_range[[0, -1]],
        az_time=az_time, slant_range=slant_range)
    return S1Burst(slc, deramped, valid, meta, start)


def valid_for_looks(burst: S1Burst, sub: AzimuthSubaperture) -> np.ndarray:
    """Burst valid mask shrunk in azimuth by the look filter margin, away from FFT edge ringing."""
    margin = azimuth_margin(sub)
    return binary_erosion(burst.valid, structure=np.ones((2 * margin + 1, 1), dtype=bool), border_value=0)
