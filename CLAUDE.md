# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Prototype pipeline for processing NISAR SAR data to generate pre-training datasets for geoscience foundation models. Uses ISCE3 (InSAR Scientific Computing Environment v3) Python API as the core SAR processing backend, supplemented by custom Python code.

## Environment

The project uses a conda environment named `isce3` (defined in `environment.yml`). Activate it before running anything:

```bash
conda activate isce3
```

## Key Domain Concepts

- **NISAR**: NASA-ISRO SAR Mission — L-band and S-band dual-frequency SAR satellite. Data products include RSLC (Range-compressed Single-Look Complex), GSLC, GCOV, GUNW, etc.
- **Foundation model pre-training**: The output of this pipeline is patches/chips of SAR amplitude, coherence, or phase data intended as self-supervised pre-training inputs.

## Next task: window-selection evaluation framework

Reference package (Sentinel-1, external to this repo): `/home/simon/Work/fmnisar/evaluation_framework/`.
It selects the local estimation window `W` for a joint feature stack. The task is to reimplement it
for NISAR RSLC on top of this repo's code. The S1 code is a reference for the method only: do not port
it as-is. It has the flaws listed below and is tied to SNAP/sarpyx.

Contents:

- `documentation/Methodology - val. Framework.pdf`: full method (equations, normalization, scoring).
  Read this first; `pdftotext -layout` works.
- `documentation/Validation Framework SD derived feats.pdf`: workshop slides with the S1 results.
- `issues.md`: known flaws in S1 sub-aperture generation and features.
- `code/subap_config_analysis/`: S1 implementation. `compute_window_selection_metrics.py` (NMI,
  stability, land cover) and `review_window_selection.py` (scoring, ranking, plots) are the parts
  worth consulting. `generate_*_features.py` and `space_hpc/` are SNAP/sarpyx/PBS-specific.
- `reference_data/`: AOI geometry, WorldCover index template, WorldCover clip download script.
- `reference_experiment/`: S1 acquisition/AOI manifest. Inputs (~316 GB) are not included.

### Method summary (S1 reference)

- `L = 3` azimuth sub-apertures; candidate windows `W ∈ {3, 5, 7, 9, 15}`; each co/cross-pol
  channel evaluated separately.
- Full stack per W: `gamma12, gamma13, gamma23` (inter-look coherence, boxcar W×W),
  `pd_mean` (circular mean of locally averaged successive-look phase differences),
  `phase_variance` (local mean of per-pixel circular variance across looks), dual-pol
  `entropy`, `alpha` (full aperture, same W). Sensitivity stack drops `phase_variance`.
- Evaluation on complete 128×128 patches with a common finite-value mask.
- Complementarity: `C = 1 - mean pairwise NMI`, each band discretized into 64 bins between its own
  P1 and P99 within the patch.
- Stability (median over patches): CV for coherences, circular dispersion `1 - |mean exp(jθ)|`
  for `pd_mean`, spatial std for phase variance, entropy, alpha.
- Min-max normalization over windows × sources within stack kind and polarization. Stability is
  averaged per family (coherence, phase difference, phase variance, H/alpha), then
  `q = λ C̃ + (1-λ) S`, `λ = 0.5`. Scores averaged over sources, then combined over polarizations:
  `Q_mean`, `Q_min`, `Q_robust = 0.7 Q_mean + 0.3 Q_min`.
- Land-cover stratification: ESA WorldCover dominant native class ≥ 60 % per patch, mapped to six
  groups. Two normalization scopes, shared and within-class; only within-class is valid for
  class-specific W selection, and the two must not be compared.

### Known flaws in the S1 reference (see `issues.md`)

1. `phase_variance` uses absolute look phases. For a point target these ramp across looks with
   sub-pixel azimuth offset, so the feature mixes position with decorrelation. Fix: circular variance
   of adjacent-look phase differences (from windowed cross-products `s_k s_{k+1}*`), which requires
   equally spaced look centres. `pd_mean` averaging `pd12` and `pd23` has the same requirement.
2. sarpyx does not demodulate looks to baseband, so cross-products carry a ramp
   `exp(j 2π (f_i - f_j) t)` and windowed coherence is biased low in a W-dependent way (the quantity
   being selected). Fix: demodulate each look by its centre frequency with a common reference time.
3. sarpyx conjugates and azimuth-flips looks, reversing the sign of the phase differences. This does
   not affect the metrics.
4. WARNING: sarpyx's dask path corrupts IW looks (`subaperture_full_img.py`, commit `ba2bf57`, code
   present since `baa1fa4`). `_ifftshift_freq_axis` applies `ifftshift` per dask chunk before the
   rechunk, so each look moves by about half the line rate (243 Hz) and its band halves swap. sarpyx
   picks dask automatically when its peak memory estimate exceeds half of available RAM (32.5 GB
   estimated for a full IW swath), so results depend on the machine. Measured on two S1 blocks: the
   look nearest Nyquist loses about 17-26 % power at terrain correction, gamma12 drops 3-20 %, and
   per-pixel features barely correlate with the numpy path. Fix: pass `force_dask=False` to
   `do_subaps`; a global `ifftshift` then matches numpy to rounding error. Whether Juan's reference
   run was affected is not known.
5. Minor: sarpyx can de-weight IW data with the wrong swath's Hamming coefficient
   (`sarpyx/processor/core/meta.py:84,104`, commit `ba2bf57`). `Handler` takes the first annotation
   file from an unsorted `os.walk`, so the value depends on filesystem order. The azimuth coefficient
   is 0.70 for IW1 and 0.75 for IW2 and IW3; 4 of the 7 S1 AOIs used the wrong one in our run. It
   barely changes `local_shift` (up to 9 Hz). Its effect on the look split and features is not
   measured. `AzimBand` comes from the same file; other uses of those values are not traced.
6. sarpyx's `local_shift` can discard about half the IW azimuth band
   (`sarpyx/processor/core/subaperture_full_img.py:729,1183,1225`, commit `ba2bf57`, code present
   since `baa1fa4`). After `iw_roll_shift` has centered the spectrum, it rolls again to put the peak
   of the de-weighted, nearly flat profile at the window center, but `ss` and `ee` stay fixed, so
   signal moved past them is lost. On both the numpy and dask paths, and on by default
   (`iw_doppler_centroid_correction=True`). Measured on VV: shifts of about 145 Hz keep 53-57 % of
   the power in AOI 30, 2, 20 and 23, where the peak is at the band edge; AOI 26 shifts -95 Hz and
   keeps 95 %, and IW1 AOI 33 and 10 shift 9-16 Hz and keep over 99 %. Using the correct Hamming
   coefficient (flaw 5) does not remove these shifts. The shift is not rolled back, but it is common
   to all looks, so it should cancel in inter-look features. Its effect on the selected `W` is not
   measured; that needs a full run with `iw_doppler_centroid_correction=False`. Whether Juan's run
   had it on is not known.

`subaperture.py` already avoids 1 and 2: `AzimuthSubaperture` uses equally spaced centroids
(`_compute_subaperture_params`) and per-look demodulation with a common reference line
(`demodulate_subaperture=True`, `_invert_subapertures`). Unlike sarpyx, its sub-bands overlap
(`overlap`).

### Adapting to NISAR

- Looks: `subaperture.py` / `subaperture_chip.py`. Features: `products.py` (`coherence`,
  `covariance_matrix`, `diffphase_statistics*`). H/alpha: `polarimetry.py` (`dualpol_h_alpha`,
  `quadpol_h_alpha`). I/O and radar↔geo indexing: `ioput.py`.
- Polarizations are HH/HV (and VV/VH for quad-pol) instead of VV/VH; frequency A/B adds a
  configuration dimension.
- The S1 run evaluated on a 10 m terrain-corrected grid. `geocode.py` geocodes sub-aperture chips
  onto a map grid: baseband complex sub-look stacks via `geocode_complex` (`geocode_slc`), and
  sub-look covariance matrices averaged over a `looks × looks` radar window, area-projected, with
  products formed on the map grid (`products_from_covariance`). The complex path requires
  `demodulate_subaperture=True`, `remodulate_to_full_dc=False`. Features computed after geocoding
  can use WorldCover directly; evaluating in radar geometry instead requires mapping WorldCover
  onto the radar grid.
