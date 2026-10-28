"""Exercise runner wiring and failure propagation without image processing."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(tmp_path):
    dataset = tmp_path / 'dataset'
    reference = dataset / 'reference' / 'sub-test01'
    reference.mkdir(parents=True)
    csv = reference / 'delta-svd_results.csv'
    csv.write_text(
        'ID,timepoint,skeleton,region,metric,voxels,value\n'
        'sub-test01,ses-1,default,whole,PSMD,100,1.25\n'
        'sub-test01,ses-2,default,whole,PSMD,100,1.5\n'
    )
    (dataset / 'SHA256SUMS').write_text(
        f'{hashlib.sha256(csv.read_bytes()).hexdigest()}  '
        'reference/sub-test01/delta-svd_results.csv\n'
    )
    docker = tmp_path / 'docker'
    docker.write_text('''#!/usr/bin/env bash
set -euo pipefail
if [ "$1" = image ]; then
    echo '{}'
    exit 0
fi
exit_code=${MOCK_DOCKER_EXIT:-0}
if [ "$exit_code" -ne 0 ]; then exit "$exit_code"; fi
[[ " $* " == *" --env USER=nonroot "* ]]
[[ " $* " == *" --user "* ]]
for arg in "$@"; do
    case "$arg" in
        type=bind,src=*,dst=/output) output=${arg#type=bind,src=}; output=${output%,dst=/output} ;;
    esac
done
cp "$MOCK_REFERENCE" "$output/delta-svd_results.csv"
if [ "${MOCK_CHANGED:-0}" = 1 ]; then
    sed -i 's/1.5$/1.75/' "$output/delta-svd_results.csv"
fi
''')
    docker.chmod(0o755)

    def run(**extra_env):
        return subprocess.run(
            ['bash', str(REPO / 'tests/run-longitudinal.sh'), str(dataset),
             'test-image', str(tmp_path / 'results'), 'sub-test01'],
            env={**os.environ, 'PATH': f'{tmp_path}:{os.environ["PATH"]}',
                 'PYTHON': sys.executable, 'MOCK_REFERENCE': str(csv), **extra_env},
            capture_output=True, text=True,
        )

    return run, csv, tmp_path / 'results' / 'sub-test01'


def test_runner_compares_results_and_writes_changes(runner):
    run, _, output = runner
    result = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert '0.25' in (output / 'longitudinal-changes.csv').read_text()
    assert (output / 'comparison.txt').is_file()
    assert (output / 'image-inspect.json').is_file()
    assert run().returncode == 2  # Existing results must not be overwritten.


def test_runner_propagates_processing_failure_through_tee(runner):
    run, _, output = runner
    assert run(MOCK_DOCKER_EXIT='7').returncode == 7
    assert not (output / 'comparison.txt').exists()


def test_runner_rejects_changed_endpoint(runner):
    run, _, output = runner
    assert run(MOCK_CHANGED='1').returncode == 1
    assert 'difference(s) found' in (output / 'comparison.txt').read_text()
    assert not (output / 'longitudinal-changes.csv').exists()


def test_comparison_report_survives_summary_failure(runner):
    run, csv, output = runner
    csv.write_text(csv.read_text().replace(',1.25\n', ',\n'))
    (csv.parents[2] / 'SHA256SUMS').write_text(
        f'{hashlib.sha256(csv.read_bytes()).hexdigest()}  '
        'reference/sub-test01/delta-svd_results.csv\n'
    )
    assert run().returncode != 0
    assert 'no metric value and no voxel count changed' in (output / 'comparison.txt').read_text()


def test_runner_verifies_dataset_before_processing(runner):
    run, csv, output = runner
    csv.write_text(csv.read_text() + '\n')
    assert run().returncode != 0
    assert not output.exists()
