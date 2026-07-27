# Azimuth Sub-Aperture Processing — TODO




## Subaperture
- post-hoc brightness equalization (probably not needed for NISAR)?
- optionally, subapertures based on distribution (equal power), rather than equally spaced, but equalization works well


## Coherence and covariance computation

- **Windowing choice**: decide whether outputs should stay decimated
  (current non-overlapping reshape approach — cheap, coarser grid) or move to
  a sliding boxcar (`scipy.ndimage.uniform_filter`, as sarpyx uses) for
  full-resolution coherence/covariance maps. This should be driven by what
  resolution the pretraining patches need.
- Optionally add a mean-coherence-across-pairs convenience band
  (`gamma_mean` in sarpyx) if a single scalar coherence summary per pixel is
  useful for patch labeling/filtering.

## Geocoding

Run through ISCE workflow; same grid as native RSLC, so should be easy.

## Metadata

Update metadata

