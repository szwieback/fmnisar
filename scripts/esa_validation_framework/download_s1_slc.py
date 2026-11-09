"""Download the Sentinel-1 IW SLCs listed in Juan's reference manifest from ASF and unzip them to .SAFE.

Credentials come from the urs.earthdata.nasa.gov entry in ~/.netrc. Products whose .SAFE directory
already exists are skipped. Each zip is deleted once it has been extracted. A dropped connection
or truncated zip is retried, because asf_search only retries burst products.

    python download_s1_slc.py --manifest <framework>/reference_experiment/acquisitions_and_subsets.csv \
        --output-dir <out>/slc
"""
import argparse
import csv
import netrc
import time
import zipfile
from pathlib import Path

import asf_search as asf
import requests

max_attempts = 5
retry_wait_s = 60


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    # one entry per product; the manifest repeats a product that covers two AOIs
    with open(args.manifest, newline='') as f:
        products = sorted({row['source_product'] for row in csv.DictReader(f)})
    args.output_dir.mkdir(parents=True, exist_ok=True)

    login, _, password = netrc.netrc().authenticators('urs.earthdata.nasa.gov')
    session = asf.ASFSession().auth_with_creds(login, password)

    for name in products:
        safe_dir = args.output_dir / f'{name}.SAFE'
        if safe_dir.is_dir():
            print(f'SKIP {safe_dir.name}: already extracted')
            continue
        results = asf.search(granule_list=[name], processingLevel='SLC')
        if len(results) != 1:
            raise RuntimeError(f'expected one SLC for {name} on ASF, found {len(results)}')
        zip_path = args.output_dir / results[0].properties['fileName']
        for attempt in range(1, max_attempts + 1):
            # asf_search skips any existing file, so a partial zip must not survive to the next try
            zip_path.unlink(missing_ok=True)
            print(f'downloading {name} (attempt {attempt}/{max_attempts})', flush=True)
            try:
                results[0].download(path=args.output_dir, session=session)
                with zipfile.ZipFile(zip_path) as z:
                    z.extractall(args.output_dir)
                break
            except (requests.RequestException, zipfile.BadZipFile) as e:
                if attempt == max_attempts:
                    raise
                print(f'attempt {attempt} failed ({type(e).__name__}); retrying in {retry_wait_s} s')
                time.sleep(retry_wait_s)
        if not safe_dir.is_dir():
            raise RuntimeError(f'{zip_path.name} did not contain {safe_dir.name}')
        zip_path.unlink()
        print(f'extracted {safe_dir}')


if __name__ == '__main__':
    main()
