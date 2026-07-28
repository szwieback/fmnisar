"""Pin the pipeline's numbers, not just its shapes.

The rest of the suite checks that every function returns something well formed.
That passes even if a new isce3 or scipy release shifts every value, which is
exactly the failure a scheduled run exists to catch. This module reduces a full
pass over the synthetic product to a handful of scalars and compares them
against stored references.

The references are platform specific in the last few digits (BLAS, FFT backend,
and LAPACK all differ between builds), so they are compared with a relative
tolerance rather than exactly. To regenerate them on a new platform:

    pytest tests/test_reference_values.py --update-reference

Review the diff before committing: a change here means the pipeline's output
moved, which is either a bug or something you meant to do.
"""

import json
import platform
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from ioput import read_block, latlon_box_to_radar_bbox
from make_test_rslc import POINT_TARGETS
from products import covariance_matrix, entropy, diffphase_statistics
from subaperture import AzimuthSubaperture

REFERENCE_PATH = Path(__file__).parent / 'reference_values.json'

# loose enough to absorb library-build differences, tight enough that a real
# numerical change in any stage shows up
RTOL = 1e-4

LOOKS = (4, 4)
N_SUBAPERTURES = 3
OVERLAP = 0.2


def compute_statistics(rslc, meta, block):
    """Reduce a full pipeline pass to scalars, one or more per processing stage."""
    stats = {}

    # --- geolocation: solved from the orbit, sensitive to any isce3 change ---
    stats['scene_center_lon'] = rslc['lon']
    stats['scene_center_lat'] = rslc['lat']
    bbox = latlon_box_to_radar_bbox(rslc['path'], rslc['lon'], rslc['lat'], 0.3, 0.3)
    # discrete indices, compared exactly rather than by tolerance
    stats['bbox'] = [bbox['az_start'], bbox['az_stop'], bbox['rg_start'], bbox['rg_stop']]

    # --- the generator itself, so a change in synthesis is not mistaken for a
    #     change in processing ---
    hv = read_block(rslc['path'], 0, rslc['n_az'], 0, rslc['n_rg'], pol='HV')
    full_power = float((np.abs(block) ** 2).mean())
    stats['mean_intensity_hh'] = full_power
    stats['mean_intensity_hv'] = float((np.abs(hv) ** 2).mean())

    # --- sub-aperture decomposition: FFT normalisation and window gain ---
    sub = AzimuthSubaperture(meta, n_subapertures=N_SUBAPERTURES, overlap=OVERLAP,
                             deweight_spectrum=True)
    stack = sub._process_block(block, az_time_start=float(meta.az_time[0]))
    stats['look_power_ratios'] = [
        float((np.abs(stack[j]) ** 2).mean()) / full_power for j in range(N_SUBAPERTURES)
    ]
    stats['mean_amplitude'] = float(np.abs(stack).mean())

    # the point targets probe the impulse response and the inter-look phase ramp.
    # phase is recorded as a cosine: it stays continuous across the +/-pi branch
    # cut, where one target happens to sit, and it does not cancel to near zero
    # the way an average over random clutter phase would.
    stats['point_target_intensity'] = [
        float(np.abs(stack[0][az, rg]) ** 2) for az, rg, _ in POINT_TARGETS
    ]
    stats['point_target_diffphase_cosine'] = [
        float(np.cos(np.angle(stack[1][az, rg] * np.conj(stack[0][az, rg]))))
        for az, rg, _ in POINT_TARGETS
    ]

    # --- covariance, coherence, eigenvalues ---
    covariance, coherence_matrix = covariance_matrix(stack, LOOKS)
    stats['mean_adjacent_coherence'] = float(coherence_matrix[0, 1].mean())
    stats['mean_outer_coherence'] = float(coherence_matrix[0, 2].mean())
    stats['mean_entropy'] = float(entropy(covariance).mean())

    _, phase_variance = diffphase_statistics(covariance)
    stats['mean_diffphase_variance'] = float(phase_variance.mean())

    return stats


def environment_description():
    """Recorded alongside the values, to make a future mismatch diagnosable."""
    import isce3
    import scipy
    return {
        'platform': f'{platform.system()}-{platform.machine()}',
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'h5py': h5py.__version__,
        'isce3': isce3.__version__,
    }


def test_pipeline_statistics_match_reference(rslc, meta, block, request):
    """Every stage of the pipeline still produces the numbers it used to."""
    measured = compute_statistics(rslc, meta, block)

    if request.config.getoption('--update-reference'):
        REFERENCE_PATH.write_text(json.dumps(
            {'environment': environment_description(), 'values': measured},
            indent=2, sort_keys=True) + '\n')
        pytest.skip(f'reference values rewritten to {REFERENCE_PATH.name}')

    if not REFERENCE_PATH.exists():
        pytest.fail(f'{REFERENCE_PATH.name} is missing; '
                    f'create it with: pytest {Path(__file__).name} --update-reference')

    reference = json.loads(REFERENCE_PATH.read_text())['values']

    missing = sorted(set(measured) - set(reference))
    assert not missing, (f'no reference stored for {missing}; '
                         f'regenerate with --update-reference')

    mismatches = []
    for name, expected in sorted(reference.items()):
        actual = measured[name]
        if name == 'bbox':
            # integer indices: any change is a real change
            if list(actual) != list(expected):
                mismatches.append(f'  {name}: {expected} -> {actual}')
        elif isinstance(expected, list):
            for i, (want, got) in enumerate(zip(expected, actual)):
                if not np.isclose(got, want, rtol=RTOL, atol=0.0):
                    mismatches.append(f'  {name}[{i}]: {want!r} -> {got!r}')
        elif not np.isclose(actual, expected, rtol=RTOL, atol=0.0):
            drift = abs(actual - expected) / max(abs(expected), 1e-30)
            mismatches.append(f'  {name}: {expected!r} -> {actual!r} ({drift:.2e} relative)')

    assert not mismatches, (
        'pipeline output has moved against the stored reference:\n'
        + '\n'.join(mismatches)
        + f'\n\nIf this is intended, regenerate with:\n'
          f'  pytest tests/{Path(__file__).name} --update-reference')
