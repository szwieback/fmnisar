"""Diagnostic plots of the window-selection patch metrics."""

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from selection.landcover import WorldCover


def _group_colors() -> dict:
    """Fixed categorical colour per land-cover group, in WorldCover order; 'all' in neutral ink."""
    palette = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6',
               '#e34948']
    groups = list(dict.fromkeys(list(WorldCover.groups.values()) + ['Other']))
    return {'all': '#52514e', **dict(zip(groups, palette))}


def _n_sub_colors(n_subs) -> dict:
    n_subs = sorted(n_subs)
    shades = plt.get_cmap('Blues')(np.linspace(0.4, 0.95, len(n_subs)))
    return dict(zip(n_subs, shades))


def _scopes(groups) -> list:
    order = list(_group_colors())
    return sorted(set(groups), key=lambda g: order.index(g) if g in order else len(order))


def _select(df, source, feature, pol=None) -> pd.DataFrame:
    rows = df[(df['source'] == source) & (df['feature'] == feature)]
    rows = rows[rows['pol'].isna()] if pol is None else rows[rows['pol'] == pol]
    if rows.empty:
        raise ValueError(f'no rows for {feature} {pol} {source}')
    return rows


def _configuration(family, n_sub, window) -> tuple:
    """(sort key, label) of a band configuration."""
    if family == 'coherence':
        return (int(n_sub), int(window)), f'{int(n_sub)}/{int(window)}'
    if family == 'phase_variance':
        return (int(n_sub),), f'{int(n_sub)}'
    return (int(window),), f'{int(window)}'


def _config_axis_label(family) -> str:
    return {'coherence': 'N / W', 'phase_variance': 'N', 'halpha': 'W'}[family]


def _with_configuration(rows, suffix='') -> pd.DataFrame:
    rows = rows.copy()
    configs = [_configuration(f, n, w) for f, n, w in
               zip(rows[f'family{suffix}'], rows[f'n_sub{suffix}'], rows[f'window{suffix}'])]
    rows[f'config_key{suffix}'] = [c[0] for c in configs]
    rows[f'config{suffix}'] = [c[1] for c in configs]
    return rows


def _name(feature, pol) -> str:
    return feature if pol is None else f'{feature} {pol}'


def _style(ax):
    ax.grid(True, color='0.9', linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def _finish(fig, path):
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def metric_curves(summary, source, feature, pol=None, ylabel='', path=None):
    """Median and interquartile range of a summarized metric against configuration, per scope."""
    rows = _select(summary, source, feature, pol)
    family = rows['family'].iloc[0]
    x_name = 'n_sub' if family == 'phase_variance' else 'window'
    scopes = _scopes(rows['group'])
    fig, axes = plt.subplots(1, len(scopes), figsize=(2.6 * len(scopes) + 0.6, 2.8),
                             sharey=True, squeeze=False)
    if family == 'coherence':
        lines = [(n, c, f'N = {n:.0f}') for n, c in _n_sub_colors(rows['n_sub'].unique()).items()]
    else:
        lines = [(None, '#0b0b0b', None)]
    for ax, scope in zip(axes[0], scopes):
        scope_rows = rows[rows['group'] == scope]
        for n_sub, color, label in lines:
            r = scope_rows if n_sub is None else scope_rows[scope_rows['n_sub'] == n_sub]
            r = r.sort_values(x_name)
            ax.fill_between(r[x_name], r['q25'], r['q75'], color=color, alpha=0.15, linewidth=0)
            ax.plot(r[x_name], r['q50'], color=color, linewidth=2, marker='o', markersize=4,
                    label=label)
        ax.set_title(f'{scope} (n = {int(scope_rows["n_patches"].max())})', fontsize=9)
        ax.set_xlabel('N' if x_name == 'n_sub' else 'W')
        ax.set_xticks(sorted(rows[x_name].unique()))
        _style(ax)
    axes[0, 0].set_ylabel(ylabel)
    if family == 'coherence':
        axes[0, -1].legend(fontsize=8, frameon=False, loc='center left',
                           bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f'{_name(feature, pol)}: median, IQR over patches', fontsize=10)
    fig.tight_layout()
    return _finish(fig, path)


def group_separation_plot(separation, source, feature, pol=None, path=None):
    """Rank eta2 (top) and group medians (bottom) against configuration, one column per value."""
    rows = _with_configuration(_select(separation, source, feature, pol))
    family = rows['family'].iloc[0]
    values = list(dict.fromkeys(rows['value']))
    configs = rows[['config_key', 'config']].drop_duplicates().sort_values('config_key')
    position = {c: i for i, c in enumerate(configs['config'])}
    colors = _group_colors()
    fig, axes = plt.subplots(2, len(values), figsize=(max(4.0, 0.22 * len(configs) + 2.5)
                                                     * len(values), 5.2),
                             sharex=True, squeeze=False)
    for col, value in enumerate(values):
        v_rows = rows[rows['value'] == value]
        eta = v_rows.drop_duplicates('config').sort_values('config_key')
        x = eta['config'].map(position)
        ax = axes[0, col]
        ax.plot(x, eta['eta2'], color='#0b0b0b', linewidth=2, marker='o', markersize=4,
                label='rank η²')
        ax.plot(x, eta['eta2_max'], color='0.6', linewidth=1, linestyle='--', label='maximum')
        ax.set_ylim(0, None)
        ax.set_title(f'{value}', fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        ax = axes[1, col]
        for group in _scopes(v_rows['group']):
            g = v_rows[v_rows['group'] == group].sort_values('config_key')
            ax.plot(g['config'].map(position), g['median'], color=colors.get(group, '0.3'),
                    linewidth=2, marker='o', markersize=4,
                    label=f'{group} (n = {int(g["n"].max())})')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs['config'], rotation=90 if len(configs) > 8 else 0, fontsize=8)
        ax.set_xlabel(_config_axis_label(family))
        if family == 'coherence':
            n_subs = [k[0] for k in configs['config_key']]
            for i in range(1, len(n_subs)):
                if n_subs[i] != n_subs[i - 1]:
                    for a in axes[:, col]:
                        a.axvline(i - 0.5, color='0.8', linewidth=0.8)
        for a in axes[:, col]:
            _style(a)
    axes[0, 0].set_ylabel('rank η²')
    axes[1, 0].set_ylabel('group median')
    axes[1, -1].legend(fontsize=8, frameon=False, loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f'{_name(feature, pol)}: land-cover group separation', fontsize=10)
    fig.tight_layout()
    return _finish(fig, path)


def nmi_table(nmi_summary, source, feature_1, feature_2, pol=None, group='all', stat='q50',
              path=None):
    """Heat map of an NMI quantile, configurations of band 1 (rows) by band 2 (columns)."""
    rows = nmi_summary[(nmi_summary['source'] == source)
                       & (nmi_summary['feature_1'] == feature_1)
                       & (nmi_summary['feature_2'] == feature_2)
                       & (nmi_summary['group'] == group)]
    if pol is not None:
        rows = rows[(rows['pol_1'] == pol) | rows['pol_1'].isna()]
        rows = rows[(rows['pol_2'] == pol) | rows['pol_2'].isna()]
    if rows.empty:
        raise ValueError(f'no NMI rows for {feature_1} x {feature_2}, {pol}, {group}')
    rows = _with_configuration(_with_configuration(rows, '_1'), '_2')
    order_1 = rows[['config_key_1', 'config_1']].drop_duplicates().sort_values('config_key_1')
    order_2 = rows[['config_key_2', 'config_2']].drop_duplicates().sort_values('config_key_2')
    table = rows.pivot_table(index='config_1', columns='config_2', values=stat, aggfunc='first')
    table = table.reindex(index=order_1['config_1'], columns=order_2['config_2'])
    n_r, n_c = table.shape
    fig, ax = plt.subplots(figsize=(0.45 * n_c + 2.2, 0.3 * n_r + 1.6))
    data = table.to_numpy(dtype=float)
    vmin, vmax = (np.nanmin(data), np.nanmax(data)) if np.isfinite(data).any() else (0, 1)
    image = ax.imshow(np.ma.masked_invalid(data), cmap='Blues', vmin=vmin, vmax=vmax,
                      aspect='auto')
    if n_r * n_c <= 400:
        for (i, j), value in np.ndenumerate(data):
            if np.isfinite(value):
                ax.text(j, i, f'{value:.3f}', ha='center', va='center', fontsize=5,
                        color='white' if value > vmin + 0.6 * (vmax - vmin) else '#0b0b0b')
    ax.set_xticks(range(n_c))
    ax.set_xticklabels(table.columns, rotation=90 if n_c > 8 else 0, fontsize=7)
    ax.set_yticks(range(n_r))
    ax.set_yticklabels(table.index, fontsize=7)
    ax.set_xlabel(f'{feature_2} ({_config_axis_label(rows["family_2"].iloc[0])})')
    ax.set_ylabel(f'{feature_1} ({_config_axis_label(rows["family_1"].iloc[0])})')
    fig.colorbar(image, ax=ax, label=f'NMI {stat}', shrink=0.8)
    ax.set_title(f'{feature_1} vs {_name(feature_2, pol)}, {group}', fontsize=9)
    fig.tight_layout()
    return _finish(fig, path)


def chip_gallery(bands, band_records, source, feature, pol=None, keys=None, n_chips: int = 3,
                 patch: int = 128, pct=(2, 98), seed: int = 0, cmap='gray', path=None):
    """Chips per land-cover group (columns) across configurations (rows), one colour scale."""
    if keys is None:
        keys = sorted(k for k in bands if k.feature == feature and k.pol == pol)
    patches = band_records[(band_records['source'] == source) & band_records['group'].notna()]
    patches = patches[['patch', 'row', 'col', 'group']].drop_duplicates()
    rng = np.random.default_rng(seed)
    columns = []
    for group in _scopes(patches['group']):
        members = patches[patches['group'] == group]
        chosen = rng.choice(len(members), size=min(n_chips, len(members)), replace=False)
        columns += [(group, r, c) for r, c in members.iloc[np.sort(chosen)][['row', 'col']]
                    .itertuples(index=False)]
    if not columns:
        raise ValueError('no patches with a land-cover group')
    chips = [[bands[k][r:r + patch, c:c + patch] for _, r, c in columns] for k in keys]
    values = np.concatenate([chip[np.isfinite(chip)] for row in chips for chip in row])
    vmin, vmax = np.percentile(values, pct)
    size = 1.1
    fig, axes = plt.subplots(len(keys), len(columns), squeeze=False, layout='constrained',
                             figsize=(size * len(columns) + 1.4, size * len(keys) + 0.8))
    colors = _group_colors()
    for i, (k, row) in enumerate(zip(keys, chips)):
        for j, chip in enumerate(row):
            ax = axes[i, j]
            image = ax.imshow(chip, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                group = columns[j][0]
                ax.set_title(textwrap.fill(group, 14), fontsize=6,
                             color=colors.get(group, '#0b0b0b'))
        _, label = _configuration(k.family, k.n_sub, k.window)
        axes[i, 0].set_ylabel(label, fontsize=7, rotation=0, ha='right', va='center')
    fig.colorbar(image, ax=axes, shrink=min(0.6, 4 / len(keys)), aspect=15,
                 label=f'{_name(feature, pol)}')
    fig.suptitle(f'{_name(feature, pol)} ({_config_axis_label(keys[0].family)})', fontsize=10)
    return _finish(fig, path)

