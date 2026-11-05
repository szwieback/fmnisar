"""Patch metrics for window selection: medians, stability, cross-family NMI, group separation."""

from dataclasses import dataclass, asdict, fields
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from selection.landcover import WorldCover


@dataclass(frozen=True, order=True)
class BandKey:
    """One feature band; pol and n_sub are None for dual-pol H/alpha, window is 1 for PV."""
    family: str
    feature: str
    pol: str | None
    n_sub: int | None
    window: int

    @property
    def statistic(self) -> str:
        return 'cv' if self.family == 'coherence' else 'std'

    @property
    def configuration(self) -> tuple:
        if self.family == 'coherence':
            return self.n_sub, self.window
        if self.family == 'phase_variance':
            return (self.n_sub,)
        return (self.window,)


def band_fields() -> list:
    return [f.name for f in fields(BandKey)]


def band_keys(n_subs, windows, pols) -> list:
    keys = [BandKey('halpha', feature, None, None, w)
            for w in windows for feature in ('entropy', 'alpha')]
    keys += [BandKey('phase_variance', 'phase_variance', pol, n, 1)
             for pol in pols for n in n_subs]
    keys += [BandKey('coherence', 'coherence', pol, n, w)
             for pol in pols for n in n_subs for w in windows]
    return keys


def nmi_pairs(keys) -> list:
    """Cross-family pairs with compatible pol, plus entropy-alpha at the same W; ordered pairs."""
    order = ['halpha', 'phase_variance', 'coherence', 'entropy', 'alpha']
    pairs = []
    for a, b in combinations(keys, 2):
        if a.family != b.family:
            if a.pol is None or b.pol is None or a.pol == b.pol:
                pairs.append((a, b) if order.index(a.family) < order.index(b.family) else (b, a))
        elif a.family == 'halpha' and a.feature != b.feature and a.window == b.window:
            pairs.append((a, b) if order.index(a.feature) < order.index(b.feature) else (b, a))
    return pairs


def patch_origins(shape, patch: int = 128) -> list:
    ny, nx = shape[-2:]
    return [(i, j) for i in range(0, ny - patch + 1, patch)
            for j in range(0, nx - patch + 1, patch)]


def common_valid(bands) -> np.ndarray:
    arrays = list(bands.values())
    valid = np.ones(arrays[0].shape, dtype=bool)
    all_zero = np.ones(arrays[0].shape, dtype=bool)
    for x in arrays:
        valid &= np.isfinite(x)
        all_zero &= x == 0
    return valid & ~all_zero


def patch_landcover(codes, min_fraction: float = 0.6) -> tuple:
    """(dominant code, its fraction of non-nodata pixels, group or None below min_fraction)."""
    values = codes[codes != WorldCover.nodata]
    if values.size == 0:
        return None, np.nan, None
    classes, counts = np.unique(values, return_counts=True)
    code = int(classes[np.argmax(counts)])
    fraction = counts.max() / values.size
    group = WorldCover.group(code) if fraction >= min_fraction else None
    return code, float(fraction), group


def discretize(x, bins: int = 64, pct=(1, 99)) -> tuple:
    """Bin codes between the band's own percentiles; all zero and False if the band is constant."""
    lo, hi = np.percentile(x, pct)
    if not (np.isfinite(lo) and np.isfinite(hi)) or np.isclose(lo, hi):
        return np.zeros(x.shape, dtype=np.uint8), False
    edges = np.linspace(lo, hi, bins + 1)
    return np.digitize(np.clip(x, lo, hi), edges[1:-1]).astype(np.uint8), True


def _entropy(counts) -> float:
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def _nmi(codes_b, codes_c, h_b, h_c, bins) -> float:
    if h_b == 0 or h_c == 0:
        return np.nan
    joint = np.bincount(codes_b.astype(np.int64) * bins + codes_c, minlength=bins * bins)
    return 2 * (h_b + h_c - _entropy(joint)) / (h_b + h_c)


def nmi(codes_b, codes_c, bins: int = 64) -> float:
    h_b = _entropy(np.bincount(codes_b, minlength=bins))
    h_c = _entropy(np.bincount(codes_c, minlength=bins))
    return _nmi(codes_b, codes_c, h_b, h_c, bins)


def patch_statistic(x, statistic: str, min_abs_mean: float = 1e-8) -> float:
    sigma = float(np.std(x, dtype=np.float64))
    if statistic == 'std':
        return sigma
    if statistic == 'cv':
        mu = abs(float(np.mean(x, dtype=np.float64)))
        return sigma / mu if mu >= min_abs_mean else np.nan
    raise ValueError(f'unknown statistic {statistic}')


def evaluate(bands, landcover=None, patch: int = 128, bins: int = 64, source: str = '',
             min_fraction: float = 0.6) -> tuple:
    """Per-patch band records (m, v) and cross-family NMI records for one source."""
    keys = list(bands)
    shape = bands[keys[0]].shape
    if any(bands[k].shape != shape for k in keys):
        raise ValueError('all bands must share one grid')
    if landcover is not None and landcover.shape != shape:
        raise ValueError(f'landcover shape {landcover.shape} differs from bands {shape}')
    pairs = nmi_pairs(keys)
    valid = common_valid(bands)
    band_rows, nmi_rows = [], []
    for index, (i, j) in enumerate(patch_origins(shape, patch)):
        window = np.s_[i:i + patch, j:j + patch]
        mask = valid[window]
        n_valid = int(mask.sum())
        if n_valid == 0:
            continue
        code, fraction, group = (None, np.nan, None) if landcover is None \
            else patch_landcover(landcover[window], min_fraction)
        meta = dict(source=source, patch=index, row=i, col=j, n_valid=n_valid,
                    landcover=code, landcover_fraction=fraction, group=group)
        codes, h = {}, {}
        for k in keys:
            x = bands[k][window][mask]
            codes[k], _ = discretize(x, bins)
            h[k] = _entropy(np.bincount(codes[k], minlength=bins))
            band_rows.append({**meta, **asdict(k), 'm': float(np.median(x)),
                              'v': patch_statistic(x, k.statistic)})
        for a, b in pairs:
            nmi_rows.append({**meta, **_suffixed(a, '_1'), **_suffixed(b, '_2'),
                             'nmi': _nmi(codes[a], codes[b], h[a], h[b], bins)})
    return pd.DataFrame(band_rows), pd.DataFrame(nmi_rows)


def _suffixed(key, suffix) -> dict:
    return {f'{name}{suffix}': value for name, value in asdict(key).items()}


def key_columns(records) -> list:
    """Band key columns of band records, or of both bands of NMI records."""
    names = band_fields()
    if 'nmi' in records.columns:
        return [f'{n}{s}' for s in ('_1', '_2') for n in names]
    return names


def _by_scope(records):
    """(group label, rows) over all patches, then per land-cover group; group None excluded."""
    yield 'all', records
    grouped = records[records['group'].notna()]
    for group, rows in grouped.groupby('group', sort=True):
        yield group, rows


def summarize(records, value: str, q=(0.1, 0.25, 0.5, 0.75, 0.9)) -> pd.DataFrame:
    """Quantiles over patches and n_patches per source, scope and band (or band pair)."""
    by = ['source'] + key_columns(records)
    tables = []
    for group, rows in _by_scope(records):
        grouped = rows.groupby(by, dropna=False, sort=True)[value]
        table = grouped.quantile(list(q)).unstack()
        table.columns = [f'q{round(100 * p):02d}' for p in q]
        table['n_patches'] = grouped.count()
        table.insert(0, 'group', group)
        tables.append(table.reset_index())
    return pd.concat(tables, ignore_index=True)


def rank_eta2(values, labels) -> tuple:
    """(eta2, its maximum under the group sizes) of the pooled ranks."""
    ranks = rankdata(values)
    n = ranks.size
    mean = ranks.mean()
    ss_total = np.sum((ranks - mean) ** 2)
    ss_between, size_term = 0.0, 0.0
    for label in np.unique(labels):
        r = ranks[labels == label]
        ss_between += r.size * (r.mean() - mean) ** 2
        size_term += r.size ** 3 - r.size
    eta2 = ss_between / ss_total if ss_total > 0 else np.nan
    return eta2, 1 - size_term / (n ** 3 - n)


def group_separation(band_records, value: str, min_patches: int = 10) -> pd.DataFrame:
    """Rank eta2 between land-cover groups per source and band; one row per band and group."""
    rows = band_records[band_records['group'].notna() & band_records[value].notna()]
    by = ['source'] + key_columns(rows)
    out = []
    for key, band in rows.groupby(by, dropna=False, sort=True):
        sizes = band['group'].value_counts()
        band = band[band['group'].isin(sizes.index[sizes >= min_patches])]
        if band['group'].nunique() < 2:
            continue
        eta2, eta2_max = rank_eta2(band[value].to_numpy(), band['group'].to_numpy())
        for group, members in band.groupby('group', sort=True):
            out.append({**dict(zip(by, key)), 'value': value, 'eta2': eta2,
                        'eta2_max': eta2_max, 'n_total': len(band), 'group': group,
                        'n': len(members), 'median': float(members[value].median())})
    return pd.DataFrame(out)
