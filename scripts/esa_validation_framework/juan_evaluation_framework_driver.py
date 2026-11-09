"""Run Juan's Sentinel-1 window-selection experiment end to end into one output directory.

Steps (select with --steps; each script skips work it has already finished where it can):
  1. download the manifest's S1 IW SLCs from ASF                       (download_s1_slc.py)
  2. per product and AOI, sarpyx/SNAP sub-aperture chain to terrain-corrected L3 products
     (process_s1_subaperture_aoi.py), then Juan's generate_subap_features.py once per window W
  3. Juan's generate_halpha_features.py: W-specific Entropy/Alpha from the TC full-aperture I/Q
  4. Juan's download_worldcover_aoi_clips.py, then compute_window_selection_metrics.py for the
     full stack and the no-phase-variance stack. WorldCover runs here, not in step 1, because
     the clips are cut onto the feature grids that step 2 makes.
  5. Juan's review_window_selection.py for each stack
  6. Juan's plot_aoi_feature_maps.py

Products, AOIs, swaths and windows come from the framework's reference manifest. Run with the
Python of an env holding sarpyx (with its geo and io extras), SNAP's gpt, and asf_search; every
step runs under that same interpreter.

Output layout under --output-dir:
  slc/                    unzipped .SAFE products
  work/                   SNAP intermediates of step 2
  experiment/             <product>/<aoi>/<swath>/L3/{subset,tc,features/W*}, <product>/<aoi>/landcover
  worldcover_tiles/       WorldCover tile cache;  worldcover_index.csv
  review/<stack kind>/    metrics workbook, rankings and review plots
  feature_maps/           AOI feature maps

    python scripts/esa_validation_framework/juan_evaluation_framework_driver.py \
        "validation framework subapertures" --output-dir /data/juan_run --gpt-path /path/to/gpt
"""
import argparse
import csv
import shlex
import subprocess  # nosec B404
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STACK_KINDS = ('feature_stack', 'feature_stack_no_phase_variance')


def run(cmd: list) -> None:
    """Echo and run one step command; stop the driver if it fails."""
    cmd = [str(c) for c in cmd]
    print(f'\n+ {shlex.join(cmd)}', flush=True)
    subprocess.run(cmd, check=True)  # nosec B603


def read_manifest(path: Path) -> tuple:
    """(product/AOI/swath rows, candidate windows) from Juan's acquisitions_and_subsets.csv."""
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    windows = {row['windows'] for row in rows}
    if len(windows) != 1:
        raise ValueError(f'manifest rows disagree on the window set: {windows}')
    return rows, [int(w) for w in windows.pop().split(',')]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('framework_dir', type=Path,
                        help="Juan's package root ('validation framework subapertures')")
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpt-path', required=True, help="SNAP gpt executable")
    parser.add_argument('--snap-userdir', default=None)
    parser.add_argument('--gpt-memory', default='16G')
    parser.add_argument('--gpt-parallelism', type=int, default=4)
    parser.add_argument('--steps', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6],
                        choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--no-local-shift', action='store_true',
                        help="turn off sarpyx's iw_doppler_centroid_correction in step 2")
    parser.add_argument('--delete-work', action='store_true',
                        help='delete each AOI\'s step 2 intermediates once its TC product is verified')
    args = parser.parse_args()

    framework = args.framework_dir.resolve()
    code = framework / 'code' / 'subap_config_analysis'
    manifest = framework / 'reference_experiment' / 'acquisitions_and_subsets.csv'
    gpkg = framework / 'reference_data' / 'samples_s1iw_landcover_subapAnalysis.gpkg'
    for path in (code, manifest, gpkg):
        if not path.exists():
            raise SystemExit(f'not found in the framework directory: {path}')
    out = args.output_dir.resolve()
    # sarpyx joins GPT arguments into one string, so product paths must not contain spaces
    if ' ' in str(out):
        raise SystemExit(f'--output-dir must not contain spaces: {out}')
    rows, windows = read_manifest(manifest)
    print(f'{len(rows)} product-AOI observations, windows {windows}, output {out}')

    slc_dir, work_dir, experiment = out / 'slc', out / 'work', out / 'experiment'
    landcover_index = out / 'worldcover_index.csv'
    gpt_args = ['--gpt-path', args.gpt_path, '--gpt-memory', args.gpt_memory]
    if args.snap_userdir:
        gpt_args += ['--snap-userdir', args.snap_userdir]

    process_args = [flag for flag, on in (('--no-local-shift', args.no_local_shift),
                                          ('--delete-work', args.delete_work)) if on]

    if 1 in args.steps:
        run([sys.executable, HERE / 'download_s1_slc.py', '--manifest', manifest,
             '--output-dir', slc_dir])

    if 2 in args.steps:
        for row in rows:
            product, aoi, swath = row['source_product'], row['aoi_id'], row['swath']
            run([sys.executable, HERE / 'process_s1_subaperture_aoi.py',
                 '--safe', slc_dir / f'{product}.SAFE', '--aoi-id', aoi, '--swath', swath,
                 '--aoi-gpkg', gpkg, '--experiment-root', experiment, '--work-root', work_dir,
                 '--gpt-parallelism', args.gpt_parallelism, *gpt_args, *process_args])
            l_dir = experiment / product / aoi / swath / 'L3'
            tc_data = l_dir / 'tc' / f'{product}_PDEC_{aoi}_SUB_TC.data'
            for w in windows:
                run([sys.executable, code / 'generate_subap_features.py', '--input-root', tc_data,
                     '--output-root', l_dir / 'features' / f'W{w}', '--win-size', w, '--quiet'])
                # Juan's script only skips quietly when it finds no sub-aperture bands
                if not list((l_dir / 'features' / f'W{w}').rglob('coherence_*.tif')):
                    raise SystemExit(f'no features written for {product} {aoi} W{w}')

    if 3 in args.steps:
        run([sys.executable, code / 'generate_halpha_features.py', '--input-root', experiment,
             '--only-l', 3, '--only-w', *windows,
             '--gpt-parallelism', args.gpt_parallelism, *gpt_args])

    if 4 in args.steps:
        run([sys.executable, framework / 'reference_data' / 'download_worldcover_aoi_clips.py',
             '--input-root', experiment, '--tile-cache', out / 'worldcover_tiles',
             '--index-csv', landcover_index])
        # same settings as Juan's run_window_selection.py
        for kind in STACK_KINDS:
            review_dir = out / 'review' / kind
            review_dir.mkdir(parents=True, exist_ok=True)
            run([sys.executable, code / 'compute_window_selection_metrics.py',
                 '--input-root', experiment, '--process-kinds', kind, '--only-l', 3,
                 '--patch-size', 128, '--landcover-min-fraction', 0.6,
                 '--landcover-index', landcover_index,
                 '--output-xlsx', review_dir / 'window_selection_metrics.xlsx'])

    if 5 in args.steps:
        for kind in STACK_KINDS:
            review_dir = out / 'review' / kind
            run([sys.executable, code / 'review_window_selection.py',
                 '--metrics-xlsx', review_dir / 'window_selection_metrics.xlsx',
                 '--output-dir', review_dir, '--kinds', kind, '--only-l', 3,
                 '--per-pol-png', 'configuration_selection_per_pol.png',
                 '--agg-png', 'configuration_selection_aggregated.png'])

    if 6 in args.steps:
        run([sys.executable, code / 'plot_aoi_feature_maps.py', '--input-root', experiment,
             '--landcover-index', landcover_index, '--output-dir', out / 'feature_maps',
             '--only-l', 3])

    print(f'\ndone: steps {args.steps} in {out}')


if __name__ == '__main__':
    main()
