"""Build a tiny, fully synthetic NISAR RSLC for automated testing.

Nothing is copied from a real product. The file is generated from physical
parameters, so it can be rebuilt in CI on any machine and no binary data needs
to live in the repository.

What makes it realistic enough to be worth testing against:

- the NISAR RSLC group layout and dataset names the repo actually reads
- metadata values taken from a real L-band product (PRF, processed bandwidth,
  centre frequency, range spacing)
- a closed circular orbit written through isce3, so geolocation really solves
- a slant-range vector consistent with that orbit at a realistic look angle
- azimuth spectra band-limited to the processed bandwidth, tapered by an
  antenna pattern, and centred on a Doppler centroid that varies across range
- speckle over a backscatter map with point targets, so products such as
  coherence and entropy see structure rather than pure noise

Run standalone to write a file:
    python tests/make_test_rslc.py [output.h5]
"""

import sys
from pathlib import Path

import h5py
import isce3
import numpy as np
from scipy.fft import fft, ifft, fftshift, ifftshift

# --- scene size -------------------------------------------------------------
N_AZ = 256          # azimuth lines
N_RG = 128          # range bins

# --- radar parameters, matching a real NISAR L-band RSLC --------------------
PRF = 1520.0                     # Hz
AZIMUTH_BANDWIDTH = 1264.14      # Hz, processed
CENTRE_FREQUENCY = 1.239e9       # Hz
RANGE_BANDWIDTH = 20e6           # Hz
SLANT_RANGE_SPACING = 3.1228     # m
SLANT_RANGE_NEAR = 900e3         # m, gives a ~34 degree look angle at 747 km
LOOK_SIDE = 'left'               # NISAR L-SAR looks left

# --- orbit ------------------------------------------------------------------
ALTITUDE = 747e3                 # m
INCLINATION = np.radians(98.4)   # sun-synchronous
EARTH_GM = 3.986004418e14        # m^3/s^2
EARTH_SEMI_MAJOR = 6378137.0     # m
REFERENCE_EPOCH = '2026-06-24T03:04:28'
TARGET_LATITUDE = np.radians(37.8)    # place the pass over a mid-latitude scene
TARGET_LONGITUDE = np.radians(-122.4)

# --- scene content ----------------------------------------------------------
DOPPLER_CENTROID_MEAN = -35.0    # Hz
DOPPLER_CENTROID_SLOPE = 40.0    # Hz across the full range swath
CROSS_POL_RATIO = 0.15           # HV power relative to HH
CROSS_POL_CORRELATION = 0.3      # complex correlation between HH and HV

# deterministic bright targets, as (azimuth index, range index, amplitude);
# tests use these coordinates to probe the impulse response
POINT_TARGETS = [(64, 32, 60.0), (150, 90, 40.0), (200, 48, 25.0)]

SWATH = 'science/LSAR/RSLC/swaths'
FREQ = f'{SWATH}/frequencyA'
PARAMS = 'science/LSAR/RSLC/metadata/processingInformation/parameters/frequencyA'
ORBIT = 'science/LSAR/RSLC/metadata/orbit'
IDENT = 'science/LSAR/identification'


def build_orbit():
    """A closed circular orbit, positioned so the scene lands at a mid-latitude.

    Treated as an Earth-fixed frame. That is not a true sun-synchronous ground
    track, but it is internally consistent, which is all the geometry code needs.
    """
    radius = EARTH_SEMI_MAJOR + ALTITUDE
    mean_motion = np.sqrt(EARTH_GM / radius ** 3)      # rad/s

    # argument of latitude placing the satellite at the target latitude, descending
    arg_latitude = np.pi - np.arcsin(np.sin(TARGET_LATITUDE) / np.sin(INCLINATION))

    # a couple of minutes of state vectors either side of the scene
    duration = N_AZ / PRF
    times = np.arange(-120.0, duration + 120.0, 1.0)
    angle = arg_latitude + mean_motion * times

    # in-plane circular motion, then tilted by the inclination
    x_plane, y_plane = radius * np.cos(angle), radius * np.sin(angle)
    vx_plane = -radius * mean_motion * np.sin(angle)
    vy_plane = radius * mean_motion * np.cos(angle)

    position = np.column_stack([x_plane,
                                y_plane * np.cos(INCLINATION),
                                y_plane * np.sin(INCLINATION)])
    velocity = np.column_stack([vx_plane,
                                vy_plane * np.cos(INCLINATION),
                                vy_plane * np.sin(INCLINATION)])

    # rotate about the pole so the pass crosses the target longitude
    mid = len(times) // 2
    raan = TARGET_LONGITUDE - np.arctan2(position[mid, 1], position[mid, 0])
    rotation = np.array([[np.cos(raan), -np.sin(raan), 0.0],
                         [np.sin(raan), np.cos(raan), 0.0],
                         [0.0, 0.0, 1.0]])
    position = position @ rotation.T
    velocity = velocity @ rotation.T

    epoch = isce3.core.DateTime(REFERENCE_EPOCH)
    state_vectors = [
        isce3.core.StateVector(epoch + isce3.core.TimeDelta(float(t)), p, v)
        for t, p, v in zip(times, position, velocity)
    ]
    return isce3.core.Orbit(state_vectors, epoch)


def doppler_centroid_grid(az_time, slant_range, n_az_nodes=5, n_rg_nodes=4):
    """Coarse Doppler centroid grid with a linear trend across range."""
    az_nodes = np.linspace(az_time[0], az_time[-1], n_az_nodes)
    rg_nodes = np.linspace(slant_range[0], slant_range[-1], n_rg_nodes)
    # normalised range coordinate, -0.5 at near range to +0.5 at far range
    across = (rg_nodes - rg_nodes.mean()) / (rg_nodes[-1] - rg_nodes[0])
    centroid = DOPPLER_CENTROID_MEAN + DOPPLER_CENTROID_SLOPE * across[None, :]
    return az_nodes, rg_nodes, np.broadcast_to(centroid, (n_az_nodes, n_rg_nodes)).copy()


def backscatter_map(rng):
    """A backscatter field with distinct regions and a few point targets."""
    az, rg = np.meshgrid(np.arange(N_AZ), np.arange(N_RG), indexing='ij')

    sigma = np.full((N_AZ, N_RG), 1.0)
    sigma[az > 0.62 * N_AZ] = 3.0                       # bright, urban-like
    sigma[(az < 0.22 * N_AZ) & (rg > 0.55 * N_RG)] = 0.08   # dark, water-like
    # a smooth gradient so nothing is piecewise constant
    sigma *= 1.0 + 0.3 * np.sin(2 * np.pi * rg / N_RG)
    # mild texture
    sigma *= np.exp(0.25 * rng.standard_normal((N_AZ, N_RG)))
    return sigma


def synthesise_image(rng, az_time, slant_range, doppler_centroid_per_column):
    """Speckle over the backscatter map, band-limited and Doppler-centred.

    Point targets are inserted before band-limiting so they emerge with the
    correct band-limited impulse response rather than as single hot pixels.
    """
    sigma = backscatter_map(rng)
    field = np.sqrt(sigma / 2.0) * (rng.standard_normal((N_AZ, N_RG))
                                    + 1j * rng.standard_normal((N_AZ, N_RG)))

    for az_idx, rg_idx, amplitude in POINT_TARGETS:
        field[az_idx, rg_idx] += amplitude

    # restrict to the processed azimuth bandwidth, tapered by an antenna pattern
    frequencies = fftshift(np.fft.fftfreq(N_AZ, d=1.0 / PRF))
    inside = np.abs(frequencies) <= AZIMUTH_BANDWIDTH / 2
    normalised = np.where(inside, 2 * frequencies / AZIMUTH_BANDWIDTH, 0.0)
    taper = np.where(inside, 1.0 - 0.6 * normalised ** 2, 0.0)   # parabolic roll-off

    spectrum = fftshift(fft(field, axis=0), axes=0)
    spectrum *= taper[:, None]
    image = ifft(ifftshift(spectrum, axes=0), axis=0)

    # shift each column's spectrum onto its own Doppler centroid
    relative_time = az_time - az_time[0]
    ramp = np.exp(2j * np.pi * doppler_centroid_per_column[None, :] * relative_time[:, None])
    return (image * ramp).astype(np.complex64)


def write_test_rslc(path, seed=0):
    """Write the synthetic product and return its scene-centre (lon, lat) in degrees."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    orbit = build_orbit()
    wavelength = isce3.core.speed_of_light / CENTRE_FREQUENCY

    # radar grid centred on the middle of the orbit arc
    az_time = orbit.mid_time + (np.arange(N_AZ) - N_AZ // 2) / PRF
    slant_range = SLANT_RANGE_NEAR + np.arange(N_RG) * SLANT_RANGE_SPACING

    # scene centre on the ellipsoid, solved from the orbit itself
    centre_xyz = isce3.geometry.rdr2geo_bracket(
        float(az_time[N_AZ // 2]), float(slant_range[N_RG // 2]),
        orbit, LOOK_SIDE, 0.0, wavelength)
    centre_llh = isce3.core.WGS84_ELLIPSOID.xyz_to_lon_lat(centre_xyz)
    scene_lon, scene_lat = np.degrees(centre_llh[0]), np.degrees(centre_llh[1])

    az_nodes, rg_nodes, centroid = doppler_centroid_grid(az_time, slant_range)
    # per-column Doppler centroid, interpolated from the coarse grid
    centroid_per_column = np.interp(slant_range, rg_nodes, centroid[0])

    hh = synthesise_image(rng, az_time, slant_range, centroid_per_column)
    # cross-pol: partially correlated with co-pol, and weaker
    independent = synthesise_image(rng, az_time, slant_range, centroid_per_column)
    hv = np.sqrt(CROSS_POL_RATIO) * (
        CROSS_POL_CORRELATION * hh
        + np.sqrt(1.0 - CROSS_POL_CORRELATION ** 2) * independent).astype(np.complex64)

    epoch_units = f'seconds since {REFERENCE_EPOCH.replace("T", " ")}'

    with h5py.File(path, 'w') as fh:
        # --- image data and its coordinates ---
        fh.create_dataset(f'{FREQ}/HH', data=hh)
        fh.create_dataset(f'{FREQ}/HV', data=hv)
        time_ds = fh.create_dataset(f'{SWATH}/zeroDopplerTime', data=az_time)
        time_ds.attrs['units'] = np.bytes_(epoch_units.encode())
        fh.create_dataset(f'{SWATH}/zeroDopplerTimeSpacing', data=1.0 / PRF)
        fh.create_dataset(f'{FREQ}/slantRange', data=slant_range)

        # --- per-frequency scalars a real product carries ---
        fh.create_dataset(f'{FREQ}/processedAzimuthBandwidth', data=AZIMUTH_BANDWIDTH)
        fh.create_dataset(f'{FREQ}/processedCenterFrequency', data=CENTRE_FREQUENCY)
        fh.create_dataset(f'{FREQ}/processedRangeBandwidth', data=RANGE_BANDWIDTH)
        fh.create_dataset(f'{FREQ}/slantRangeSpacing', data=SLANT_RANGE_SPACING)
        fh.create_dataset(f'{FREQ}/nominalAcquisitionPRF', data=PRF)
        fh.create_dataset(f'{FREQ}/listOfPolarizations',
                          data=np.array([b'HH', b'HV']))

        # --- processing metadata ---
        fh.create_dataset(f'{PARAMS}/dopplerCentroid', data=centroid)
        fh.create_dataset(f'{PARAMS}/zeroDopplerTime', data=az_nodes)
        fh.create_dataset(f'{PARAMS}/slantRange', data=rg_nodes)

        # --- orbit, written by isce3 so it reads back cleanly ---
        orbit.save_to_h5(fh.create_group(ORBIT))

        # --- identification ---
        fh.create_dataset(f'{IDENT}/lookDirection', data=np.bytes_(b'Left'))
        fh.create_dataset(f'{IDENT}/orbitPassDirection', data=np.bytes_(b'Descending'))
        fh.create_dataset(f'{IDENT}/absoluteOrbitNumber', data=np.int64(12345))

        fh.attrs['scene_center_lon'] = scene_lon
        fh.attrs['scene_center_lat'] = scene_lat
        fh.attrs['synthetic'] = np.bytes_(b'generated by tests/make_test_rslc.py')

    return float(scene_lon), float(scene_lat)


if __name__ == '__main__':
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/processed/test_rslc.h5')
    lon, lat = write_test_rslc(output)
    size_kb = output.stat().st_size / 1024
    print(f'wrote {output} ({size_kb:.0f} KB)')
    print(f'scene centre: lon {lon:.4f}, lat {lat:.4f}')
