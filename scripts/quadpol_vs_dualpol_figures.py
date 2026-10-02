"""Quad-pol vs dual-pol entropy/alpha from the same NISAR quad-pol RSLC crop.

Quad-pol uses HH, HV, VH, VV (polarimetry.quadpol_h_alpha); dual-pol uses only the HH and HV
channels of the same file (polarimetry.dualpol_h_alpha), so any difference is the
polarimetry alone. Samples outside the product's validSamplesSubSwath ranges are masked.

Run from the repo root:
    PYTHONPATH=. conda run --no-capture-output -n fmnisar python scripts/quadpol_vs_dualpol_figures.py
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ioput import read_block
from polarimetry import dualpol_h_alpha, quadpol_h_alpha
from products import boxcar

crops = ['alaska_bona', 'alaska_bonanza', 'alaska_cr_tfield']
dates = ['20260726', '20251116']
folder_out = Path('figures/alaska_quadpol')
folder_out.mkdir(parents=True, exist_ok=True)

# target ground footprint of one multilook window, metres
window_m = 30.0


def swath_metadata(path):
    """Pixel spacing (azimuth, ground range) in metres and the valid-sample mask."""
    with h5py.File(path, 'r') as fh:
        swath = fh['science/LSAR/RSLC/swaths/frequencyA']
        spacing = (float(swath['sceneCenterAlongTrackSpacing'][()]),
                   float(swath['sceneCenterGroundRangeSpacing'][()]))
        n_az, n_rg = swath['HH'].shape
        # union of every subswath's [start, end) sample range on each line
        valid = np.zeros((n_az, n_rg), bool)
        columns = np.arange(n_rg)
        for key in [k for k in swath if k.startswith('validSamplesSubSwath')]:
            bounds = swath[key][()]
            valid |= (columns[None, :] >= bounds[:, :1]) & (columns[None, :] < bounds[:, 1:])
    return spacing, valid


for date in dates:
    for crop in crops:
        path = next(Path(f'data/raw/alaska_quadpol/{crop}').glob(f'*_{date}T*_sub.h5'))
        (az_m, rg_m), valid = swath_metadata(path)
        looks = (round(window_m / az_m), round(window_m / rg_m))
        print(f'[{crop} {date}] pixel {az_m:.1f} x {rg_m:.1f} m, looks {looks}, '
              f'valid {100 * valid.mean():.1f}% of samples')

        # zero the invalid samples, then drop any window that touched one
        channels = {pol: np.where(valid, read_block(path, None, None, None, None, pol=pol), 0)
                    for pol in ['HH', 'HV', 'VH', 'VV']}
        window_valid = boxcar(valid.astype(float), looks) == 1.0

        with np.errstate(invalid='ignore', divide='ignore'):
            h_quad, alpha_quad, _ = quadpol_h_alpha(channels['HH'], channels['HV'], channels['VH'],
                                                    channels['VV'], looks)
            h_dual, alpha_dual, _ = dualpol_h_alpha(channels['HH'], channels['HV'], looks)
        for field in (h_quad, alpha_quad, h_dual, alpha_dual):
            field[~window_valid] = np.nan

        both = np.isfinite(h_quad) & np.isfinite(h_dual)
        print(f'[{crop} {date}] entropy mean quad {np.nanmean(h_quad):.3f} dual {np.nanmean(h_dual):.3f}, '
              f'alpha mean quad {np.degrees(np.nanmean(alpha_quad)):.1f} dual '
              f'{np.degrees(np.nanmean(alpha_dual)):.1f} deg, '
              f'corr H {np.corrcoef(h_quad[both], h_dual[both])[0, 1]:.2f} '
              f'alpha {np.corrcoef(alpha_quad[both], alpha_dual[both])[0, 1]:.2f}')

        # axes in km on the ground so all panels share one frame
        n_az, n_rg = h_quad.shape
        extent = [0, n_rg * looks[1] * rg_m / 1000, n_az * looks[0] * az_m / 1000, 0]
        panels = [
            ('Quad-pol entropy', h_quad, dict(cmap='magma', vmin=0, vmax=1)),
            ('Dual-pol entropy (HH, HV)', h_dual, dict(cmap='magma', vmin=0, vmax=1)),
            ('Quad-pol alpha (deg)', np.degrees(alpha_quad), dict(cmap='viridis', vmin=0, vmax=90)),
            ('Dual-pol alpha (deg)', np.degrees(alpha_dual), dict(cmap='viridis', vmin=0, vmax=90)),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        for ax, (title, image, style) in zip(axes.ravel(), panels):
            im = ax.imshow(image, extent=extent, interpolation='nearest', **style)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, shrink=0.8)
        for ax in axes[-1]:
            ax.set_xlabel('ground range (km)')
        for ax in axes[:, 0]:
            ax.set_ylabel('azimuth (km)')
        fig.tight_layout()
        out = folder_out / f'{crop.removeprefix("alaska_")}_{date}_quad_vs_dual.png'
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f'[{crop} {date}] wrote {out}')
