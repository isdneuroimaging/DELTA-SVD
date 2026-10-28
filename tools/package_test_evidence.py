"""Package the small numerical evidence set after release validation succeeds."""
import argparse
import json
import os
from pathlib import Path
import re
import zipfile


SUBJECTS = ('sub-test01', 'sub-test02')
FILES = ('delta-svd_results.csv', 'longitudinal-changes.csv',
         'comparison.txt', 'delta-svd_run_manifest.json')


def package_evidence(results, output, version, metadata):
    if not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', version):
        raise ValueError('Invalid release version')
    contents = {}
    for subject in SUBJECTS:
        for name in FILES:
            contents[f'{subject}/{name}'] = (results / subject / name).read_bytes()
        manifest = json.loads(contents[f'{subject}/delta-svd_run_manifest.json'])
        if (manifest['subject_id'] != subject
                or manifest['pipeline_version'] != version
                or manifest['source_revision'] != metadata['source_revision']
                or manifest['processing_mode'] != 'longitudinal'):
            raise ValueError(f'{subject}: run manifest does not match this release')
        lines = contents[f'{subject}/comparison.txt'].decode().splitlines()
        if (not lines
                or 'Rows keyed by ID, timepoint, skeleton, region, metric.' not in lines
                or 'Metric values and voxel counts compared exactly.' not in lines
                or any('do(es) not identify a row uniquely' in line for line in lines)
                or lines[-1] != 'RESULT: no metric value and no voxel count changed.'):
            raise ValueError(f'{subject}: comparison did not pass with all keys and exact values')
    metadata = {
        **metadata,
        'comparison': {'metric_values': 'exact', 'voxel_counts': 'exact', 'ignored_keys': []},
        'subjects': dict.fromkeys(SUBJECTS, 'passed'),
    }
    contents['evidence.json'] = (json.dumps(metadata, indent=2) + '\n').encode()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f'delta-svd_{version}_test-evidence.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, data in sorted(contents.items()):
            bundle.writestr(name, data)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    metadata = {
        'image': os.environ['IMAGE'],
        'source_revision': os.environ['GITHUB_SHA'],
        'workflow_run': (f'{os.environ["GITHUB_SERVER_URL"]}/{os.environ["GITHUB_REPOSITORY"]}'
                         f'/actions/runs/{os.environ["GITHUB_RUN_ID"]}'),
        'run_attempt': int(os.environ['GITHUB_RUN_ATTEMPT']),
        'dataset': {'url': os.environ['DATASET_URL'], 'sha256': os.environ['DATASET_SHA256']},
    }
    version = (Path(__file__).resolve().parents[1] / 'VERSION').read_text().strip()
    print(package_evidence(args.results, args.output, version, metadata))


if __name__ == '__main__':
    main()
