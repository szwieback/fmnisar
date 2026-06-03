"""Inspect a NISAR RSLC HDF5 file: print structure, access SLC data, and extract metadata."""

import h5py
import numpy as np



def get_slc_datasets(fh):
    """Return dict of {(freq, pol): dataset} for all SLC rasters."""
    slc = {}
    base = "science/LSAR/RSLC/swaths"
    if base not in fh:
        print(f"Path not found: {base}")
        return slc
    swaths = fh[base]
    for freq in swaths:
        if not freq.startswith("frequency"):
            continue
        freq_grp = swaths[freq]
        for pol in freq_grp:
            ds = freq_grp[pol]
            if isinstance(ds, h5py.Dataset) and np.issubdtype(ds.dtype, np.complexfloating):
                slc[(freq, pol)] = ds
    return slc


def extract_rslc_params(fh, freq="A"):
    """
    Extract processing parameters for one frequency band from an RSLC file.

    Returns a dict with:
      doppler_centroid   : (n_az, n_rg) float64 array [Hz], on the coarse grid
      doppler_az_time    : (n_az,) zeroDopplerTime coordinates of the centroid grid [s]
      doppler_slant_range: (n_rg,) slant-range coordinates of the centroid grid [m]
      orbit_velocity     : (n_sv, 3) ECEF velocity state vectors [m/s]
      orbit_time         : (n_sv,) UTC times of state vectors [s since epoch]
      slant_range        : (n_rg_full,) full-resolution slant range vector [m]
      az_time            : (n_az_full,) full-resolution zero-Doppler time vector [s]
      az_spacing_m       : along-track pixel spacing at scene centre [m]
      rg_spacing_m       : ground-range pixel spacing at scene centre [m]
      az_chirp_weighting : (256,) normalised azimuth taper (the stored weighting function)
      rg_chirp_weighting : (256,) normalised range taper
      az_bandwidth_hz    : azimuth bandwidth derived from azimuth_resolution and v_eff [Hz]
      v_eff              : effective (along-track) speed at scene centre [m/s]
    """
    base_proc = f"science/LSAR/RSLC/metadata/processingInformation/parameters"
    base_swath = f"science/LSAR/RSLC/swaths/frequency{freq}"
    base_orbit = "science/LSAR/RSLC/metadata/orbit"
    out = {}

    # --- Doppler centroid grid ---
    freq_grp = fh[f"{base_proc}/frequency{freq}"]
    out["doppler_centroid"] = freq_grp["dopplerCentroid"][()]
    out["doppler_az_time"] = freq_grp["zeroDopplerTime"][()]
    out["doppler_slant_range"] = freq_grp["slantRange"][()]

    # --- Orbit state vectors ---
    orbit_grp = fh[base_orbit]
    out["orbit_velocity"] = orbit_grp["velocity"][()]
    out["orbit_time"] = orbit_grp["time"][()]

    # --- Swath geometry ---
    swath_grp = fh[base_swath]
    out["slant_range"] = swath_grp["slantRange"][()]
    out["az_time"] = fh["science/LSAR/RSLC/swaths/zeroDopplerTime"][()]
    out["az_spacing_m"] = float(swath_grp["sceneCenterAlongTrackSpacing"][()])
    out["rg_spacing_m"] = float(swath_grp["sceneCenterGroundRangeSpacing"][()])
    out["azimuth_bandwidth"] = float(swath_grp["processedAzimuthBandwidth"][()])

    # --- Window / taper arrays ---
    params_grp = fh[base_proc]
    out["az_chirp_weighting"] = params_grp["azimuthChirpWeighting"][()]
    out["rg_chirp_weighting"] = params_grp["rangeChirpWeighting"][()]

    # --- Azimuth bandwidth and effective velocity ---
    # v_eff: median orbit speed as proxy (ECEF speed ≈ along-track for near-polar orbits)
    speeds = np.linalg.norm(out["orbit_velocity"], axis=1)
    out["v_eff"] = float(np.median(speeds))

    # PRF is 1 / azimuth pixel spacing in time
    dt = float(np.median(np.diff(out["az_time"])))
    prf = 1.0 / dt
    out['prf'] = prf
    out['zero_doppler_time_spacing'] = float(
        fh['/science/LSAR/RSLC/swaths/zeroDopplerTimeSpacing'][()])
    return out


def print_metadata(fh, root="science/LSAR"):
    """Print all datasets under root in a readable format."""
    def _visit(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        val = obj[()]
        if isinstance(val, bytes):
            # skip long blobs like runConfigurationContents
            if len(val) > 200:
                print(f"  {name}: <bytes, {len(val)} chars>")
            else:
                print(f"  {name}: {val.decode()}")
        elif isinstance(val, np.ndarray) and val.size > 6:
            print(f"  {name}: shape={val.shape} dtype={obj.dtype}  "
                  f"min={np.nanmin(val):.4f}  mean={np.nanmean(val):.4f}  max={np.nanmax(val):.4f}")
        else:
            v = val.item() if isinstance(val, np.ndarray) and val.size == 1 else val
            print(f"  {name}: {v}")

    fh[root].visititems(_visit)


if __name__ == "__main__":
    path = "/home/simon/Work/fmnisar/NISAR/NISAR_L1_PR_RSLC_010_073_D_053_4005_DHDH_A_20260114T062318_20260114T062355_X05010_N_F_J_001.h5"

    with h5py.File(path, "r") as fh:
        print("=== SLC datasets ===")
        slc = get_slc_datasets(fh)
        for (freq, pol), ds in slc.items():
            print(f"  {freq}/{pol}: shape={ds.shape} dtype={ds.dtype}")

        print("\n=== Processing parameters (freq A) ===")
        p = extract_rslc_params(fh, freq='A')
        print(f"  slant_range:         {p['slant_range'].shape}  "
              f"{p['slant_range'][0]:.1f} … {p['slant_range'][-1]:.1f} m")
        print(f"  az_time:             {p['az_time'].shape}  "
              f"{p['az_time'][0]:.3f} … {p['az_time'][-1]:.3f} s")
        print(f"  az_spacing_m:        {p['az_spacing_m']:.4f} m")
        print(f"  rg_spacing_m:        {p['rg_spacing_m']:.4f} m")
        print(f"  doppler_centroid:    {p['doppler_centroid'].shape}  "
              f"mean={p['doppler_centroid'].mean():.2f} Hz")
        print(f"  orbit_velocity:      {p['orbit_velocity'].shape}  "
              f"speed range {np.linalg.norm(p['orbit_velocity'], axis=1).min():.1f}"
              f"–{np.linalg.norm(p['orbit_velocity'], axis=1).max():.1f} m/s")
        print(f"  v_eff (median |v|):  {p['v_eff']:.1f} m/s")
        print(f"  az_chirp_weighting:  {p['az_chirp_weighting'].shape}  "
              f"min={p['az_chirp_weighting'].min():.4f} max={p['az_chirp_weighting'].max():.4f}")
        print(f"  rg_chirp_weighting:  {p['rg_chirp_weighting'].shape}  "
              f"min={p['rg_chirp_weighting'].min():.4f} max={p['rg_chirp_weighting'].max():.4f}")

        print("\n=== All metadata (science/LSAR) ===")
        # print_metadata(fh)

        print(p['az_chirp_weighting'])