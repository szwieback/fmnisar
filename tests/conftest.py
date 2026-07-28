"""Test fixtures: one synthetic RSLC, generated once per session.

The product is built from physical parameters by make_test_rslc.py rather than
committed as a binary, so a scheduled CI run always exercises the generator too.
"""

import matplotlib
import pytest

from ioput import read_block
from make_test_rslc import write_test_rslc, N_AZ, N_RG
from subaperture import SubapertureMetaData, AzimuthSubaperture

# figures must render without a display
matplotlib.use('Agg')


def pytest_addoption(parser):
    """Allow the stored reference statistics to be regenerated in place."""
    parser.addoption(
        '--update-reference', action='store_true', default=False,
        help='rewrite tests/reference_values.json from this run instead of comparing')


@pytest.fixture(scope='session')
def rslc(tmp_path_factory):
    """Path to the synthetic product, plus the scene centre solved from its orbit."""
    path = tmp_path_factory.mktemp('rslc') / 'test_rslc.h5'
    lon, lat = write_test_rslc(path)
    return dict(path=path, lon=lon, lat=lat, n_az=N_AZ, n_rg=N_RG)


@pytest.fixture(scope='session')
def meta(rslc):
    """Metadata read back out of the synthetic product."""
    return SubapertureMetaData.load_from_rslc_path(rslc['path'])


@pytest.fixture
def sub(meta):
    """A default decomposition: 3 sub-apertures, 20 percent overlap."""
    return AzimuthSubaperture(meta, n_subapertures=3, overlap=0.2)


@pytest.fixture(scope='session')
def block(rslc):
    """The full HH image as a single block."""
    return read_block(rslc['path'], 0, rslc['n_az'], 0, rslc['n_rg'], pol='HH')


@pytest.fixture
def stack(sub, block, meta):
    """A sub-aperture stack (n_sub, n_az, n_rg), the input to everything in products."""
    return sub._process_block(block, az_time_start=float(meta.az_time[0]))


@pytest.fixture
def figure_dir(tmp_path):
    """Somewhere for plotting functions to write, so nothing lands in the repo."""
    out = tmp_path / 'figures'
    out.mkdir()
    return out
