"""Dual-pol entropy/alpha maps for one RSLC crop per site.

Sites with a NAIP image warped to radar geometry (scratch/naip_to_radar_sites.py) get it as
the first panel; the others show the HV-HH phase there instead.

Run from the repo root:
    PYTHONPATH=. conda run --no-capture-output -n fmnisar python scripts/h_alpha_figures.py
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ioput import read_block
from polarimetry import dualpol_h_alpha
from products import boxcar

sites = ['bair_island', 'rice_colusa', 'sf_bridge', 'vasylivka', 'rosamond']
folder_out = Path('figures/h_alpha')
folder_out.mkdir(parents=True, exist_ok=True)

# target ground footprint of one multilook window, metres
window_m = 30.0


def pixel_spacing(path):
    """Azimuth and ground-range pixel spacing in metres from the RSLC metadata."""
    with h5py.File(path, 'r') as fh:
        swath = fh['science/LSAR/RSLC/swaths/frequencyA']
        return (float(swath['sceneCenterAlongTrackSpacing'][()]),
                float(swath['sceneCenterGroundRangeSpacing'][()]))


for site in sites:
    path = sorted(Path(f'data/raw/{site}').glob('*_sub.h5'))[0]
    date = path.name.split('_')[12][:8]
    az_m, rg_m = pixel_spacing(path)
    looks = (round(window_m / az_m), round(window_m / rg_m))
    print(f'[{site}] {path.name}')
    print(f'[{site}] pixel {az_m:.1f} x {rg_m:.1f} m, looks {looks}')

    hh = read_block(path, None, None, None, None, pol='HH')
    hv = read_block(path, None, None, None, None, pol='HV')
    entropy, alpha, phase = dualpol_h_alpha(hh, hv, looks)
    power_db = 10 * np.log10(boxcar(np.abs(hh) ** 2, looks))
    print(f'[{site}] output {entropy.shape}, entropy mean {np.nanmean(entropy):.3f}, '
          f'alpha mean {np.degrees(np.nanmean(alpha)):.1f} deg')

    # axes in km on the ground so the four panels share one frame
    n_az, n_rg = entropy.shape
    extent = [0, n_rg * looks[1] * rg_m / 1000, n_az * looks[0] * az_m / 1000, 0]
    naip_path = Path(f'data/processed/naip_{site}_radar.npy')
    panels = [
        ('HH power (dB)', power_db, dict(cmap='gray', vmin=np.nanpercentile(power_db, 2),
                                         vmax=np.nanpercentile(power_db, 98))),
        ('Alpha (deg)', np.degrees(alpha), dict(cmap='viridis', vmin=0, vmax=90)),
        ('Entropy', entropy, dict(cmap='magma', vmin=0, vmax=1)),
    ]
    # NAIP first when it exists, otherwise the phase panel; the RGB image needs no colormap
    if naip_path.exists():
        panels.insert(0, ('NAIP', np.load(naip_path), dict()))
    else:
        panels.append(('HV-HH phase (deg)', np.degrees(phase),
                       dict(cmap='twilight', vmin=-180, vmax=180)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, (title, image, style) in zip(axes.ravel(), panels):
        im = ax.imshow(image, extent=extent, interpolation='nearest', **style)
        ax.set_title(title)
        # RGB gets an invisible colorbar so it takes the same width as the other panels
        if image.ndim == 2:
            fig.colorbar(im, ax=ax, shrink=0.8)
        else:
            fig.colorbar(plt.cm.ScalarMappable(), ax=ax, shrink=0.8).ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('ground range (km)')
    for ax in axes[:, 0]:
        ax.set_ylabel('azimuth (km)')
    fig.tight_layout()
    out = folder_out / f'{site}_{date}.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'[{site}] wrote {out}')
