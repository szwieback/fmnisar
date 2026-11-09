"""Process one Sentinel-1 IW SLC and AOI into Juan's L3 sub-aperture layout with sarpyx and SNAP.

Juan's package does not include this step. It follows sarpyx's S1 TOPS recipe
(sarpyx.pipelines.single_product.s1_tops: same operators, sub-aperture options, decomposition and
terrain-correction settings) with three sub-apertures and an AOI subset before terrain correction:

  TOPSAR-Split (one swath, all bursts) -> Apply-Orbit-File -> Calibration (complex)
  -> TOPSAR-Deramp-Demod -> TOPSAR-Deburst -> do_subaps (i/q_<IW>_<POL>_SA1..3 in the deburst product)
  -> H-Alpha dual-pol decomposition -> copy full-aperture and sub-aperture I/Q into it
  -> Subset to the AOI polygon -> Terrain-Correction (10 m, AUTO:42001, complex output)

Outputs, matching Juan's manifest and the inputs his feature scripts expect:
  <experiment-root>/<product>/<aoi>/<swath>/L3/subset/<product>_PDEC_<aoi>_SUB.dim
  <experiment-root>/<product>/<aoi>/<swath>/L3/tc/<product>_PDEC_<aoi>_SUB_TC.dim
Intermediates go to <work-root>/<product>/<aoi>/<swath>, outside the experiment tree so the
metric and WorldCover scripts never see them. A finished TC product is not redone.
"""
import argparse
import re
import shutil
from pathlib import Path

import geopandas as gpd
from sarpyx.pipelines.single_product.s1_tops import (DEFAULT_MAP_PROJECTION, DEFAULT_ORBIT_TYPE,
                                                     DEFAULT_PIXEL_SPACING_M, PDEC_PARAMS,
                                                     TOPS_SUBAP_PARAMS)
from sarpyx.snapflow.engine import GPT
from sarpyx.snapflow.merge import merge_iq_into_pdec


def aoi_wkt(gpkg: Path, aoi_id: str) -> str:
    """WKT polygon of one AOI from Juan's geopackage (EPSG:4326)."""
    aois = gpd.read_file(gpkg)
    rows = aois[aois['aoi_id'] == aoi_id]
    if len(rows) != 1:
        raise ValueError(f'expected one polygon for {aoi_id} in {gpkg.name}, found {len(rows)}')
    return rows.geometry.iloc[0].wkt


def add_missing_band_types(dim: Path) -> None:
    """Add the DATA_TYPE tag that sarpyx's merge leaves off; SNAP silently drops bands without it."""
    text = dim.read_text()
    for block in re.findall(r'<Spectral_Band_Info>.*?</Spectral_Band_Info>', text, flags=re.S):
        # virtual bands have an expression instead of stored data
        if '<DATA_TYPE>' not in block and '<EXPRESSION>' not in block:
            typed = block.replace('</BAND_RASTER_HEIGHT>',
                                  '</BAND_RASTER_HEIGHT>\n            <DATA_TYPE>float32</DATA_TYPE>')
            text = text.replace(block, typed, 1)
    dim.write_text(text)


def iq_band_pairs(dim: Path) -> list:
    """Stored I/Q band names in i, q order; Terrain-Correction fails on SA bands if left to pair them."""
    i_bands = re.findall(r'<BAND_NAME>(i_IW\d_(?:VV|VH)(?:_SA\d+)?)</BAND_NAME>', dim.read_text())
    return [name for i in i_bands for name in (i, f'q{i[1:]}')]


def require(path, step: str, op: GPT) -> str:
    """Stop with SNAP's error when a GPT step returns no product."""
    if path is None:
        raise RuntimeError(f'{step} failed: {op.last_error_summary()}')
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--safe', type=Path, required=True, help='unzipped S1 IW SLC .SAFE directory')
    parser.add_argument('--aoi-id', required=True)
    parser.add_argument('--swath', required=True, choices=['IW1', 'IW2', 'IW3'])
    parser.add_argument('--aoi-gpkg', type=Path, required=True)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--n-subapertures', type=int, default=3)
    parser.add_argument('--gpt-path', required=True)
    parser.add_argument('--snap-userdir', default=None)
    parser.add_argument('--gpt-memory', default='16G')
    parser.add_argument('--gpt-parallelism', type=int, default=4)
    parser.add_argument('--no-local-shift', action='store_true',
                        help="turn off sarpyx's iw_doppler_centroid_correction (CLAUDE.md, known flaw 6)")
    parser.add_argument('--delete-work', action='store_true',
                        help='delete this AOI\'s intermediates once its TC product is verified')
    args = parser.parse_args()

    product = args.safe.name.removesuffix('.SAFE')
    lookset = f'L{args.n_subapertures}'
    l_dir = args.experiment_root / product / args.aoi_id / args.swath / lookset
    subset_name = f'{product}_PDEC_{args.aoi_id}_SUB'
    tc_dim = l_dir / 'tc' / f'{subset_name}_TC.dim'
    if tc_dim.exists():
        print(f'SKIP {product} {args.aoi_id}: {tc_dim} exists')
        return
    work_dir = args.work_root / product / args.aoi_id / args.swath
    gpt_kwargs = dict(gpt_path=args.gpt_path, snap_userdir=args.snap_userdir,
                      memory=args.gpt_memory, parallelism=args.gpt_parallelism, timeout=None)
    wkt = aoi_wkt(args.aoi_gpkg, args.aoi_id)
    print(f'{product} {args.aoi_id} {args.swath}: radar chain in {work_dir}')

    # radar-geometry chain; each GPT call writes a product and becomes the next one's input
    op = GPT(product=args.safe, outdir=work_dir, **gpt_kwargs)
    require(op.topsar_split(subswath=args.swath, output_name=f'{product}_split'), 'TOPSAR-Split', op)
    require(op.apply_orbit_file(orbit_type=DEFAULT_ORBIT_TYPE, output_name=f'{product}_orbit'),
            'Apply-Orbit-File', op)
    require(op.calibration(output_complex=True, output_name=f'{product}_cal'), 'Calibration', op)
    require(op.topsar_deramp_demod(output_name=f'{product}_deramp'), 'TOPSAR-Deramp-Demod', op)
    deb_dim = require(op.deburst(output_name=f'{product}_deb'), 'TOPSAR-Deburst', op)

    # sub-apertures are written into the deburst .data folder; its DIM is left as is (update_dim).
    # force_dask=False: sarpyx's dask path misplaces the look bands (see CLAUDE.md, known flaw 4)
    subap_params = dict(TOPS_SUBAP_PARAMS, n_decompositions=args.n_subapertures, force_dask=False)
    if args.no_local_shift:
        subap_params['iw_doppler_centroid_correction'] = False
    op.do_subaps(dim_path=deb_dim, safe_path=args.safe, **subap_params)

    # decomposition of the deburst product, then its I/Q (incl. SA bands) copied into the result
    pdec_dim = require(op.polarimetric_decomposition(**PDEC_PARAMS, output_name=f'{product}_PDEC'),
                       'Polarimetric-Decomposition', op)
    merge_iq_into_pdec(src_dim=deb_dim, pdec_dim=pdec_dim, is_tops=True,
                       overwrite_copied_files=False, backup=False)
    add_missing_band_types(Path(pdec_dim))

    # AOI subset in radar geometry, then terrain correction of the I/Q bands, into Juan's layout
    op_sub = GPT(product=pdec_dim, outdir=l_dir / 'subset', **gpt_kwargs)
    subset_dim = require(op_sub.subset(geo_region=wkt, copy_metadata=True, output_name=subset_name),
                         'Subset', op_sub)
    iq_bands = iq_band_pairs(Path(subset_dim))
    op_tc = GPT(product=subset_dim, outdir=l_dir / 'tc', **gpt_kwargs)
    require(op_tc.terrain_correction(source_bands=iq_bands, map_projection=DEFAULT_MAP_PROJECTION,
                                     pixel_spacing_in_meter=DEFAULT_PIXEL_SPACING_M,
                                     save_selected_source_band=True,
                                     output_name=f'{subset_name}_TC'),
            'Terrain-Correction', op_tc)

    # every polarization must reach the TC product with all its looks, or the features are empty
    n_sa = len(list(tc_dim.with_suffix('.data').glob('[iq]_*_SA*.img')))
    n_expected = 2 * 2 * args.n_subapertures  # i and q, VV and VH
    if n_sa != n_expected:
        raise RuntimeError(f'{tc_dim.name} has {n_sa} sub-aperture bands, expected {n_expected}')
    print(f'wrote {tc_dim} with {n_sa} sub-aperture bands')
    if args.delete_work:
        shutil.rmtree(work_dir)


if __name__ == '__main__':
    main()
