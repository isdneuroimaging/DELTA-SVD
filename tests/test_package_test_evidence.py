"""Check release evidence completeness and identity before it is attested."""
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


spec = importlib.util.spec_from_file_location(
    'package_test_evidence', Path(__file__).resolve().parents[1] / 'tools/package_test_evidence.py'
)
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)

PASSING_COMPARISON = (
    'Rows keyed by ID, timepoint, skeleton, region, metric.\n'
    'Metric values and voxel counts compared exactly.\n'
    '\nRESULT: no metric value and no voxel count changed.\n'
)


@pytest.fixture
def evidence(tmp_path):
    results = tmp_path / 'results'
    metadata = {
        'image': 'ghcr.io/isdneuroimaging/delta-svd-staging@sha256:' + 'a' * 64,
        'source_revision': 'b' * 40,
        'workflow_run': 'https://github.com/isdneuroimaging/DELTA-SVD/actions/runs/123',
        'run_attempt': 2,
        'dataset': {'url': 'https://example.org/data.zip', 'sha256': 'c' * 64},
    }
    for subject in packager.SUBJECTS:
        folder = results / subject
        folder.mkdir(parents=True)
        (folder / 'delta-svd_results.csv').write_text('candidate results\n')
        (folder / 'longitudinal-changes.csv').write_text('candidate changes\n')
        (folder / 'comparison.txt').write_text(PASSING_COMPARISON)
        (folder / 'delta-svd_run_manifest.json').write_text(json.dumps({
            'subject_id': subject, 'pipeline_version': '1.0.6',
            'source_revision': metadata['source_revision'], 'processing_mode': 'longitudinal',
        }))
        (folder / 'delta-svd_qc.html').write_text('large QC report excluded from release evidence')
        (folder / 'console.log').write_text('processing log excluded from release evidence')
    return results, tmp_path / 'evidence', metadata


def test_packages_only_nine_evidence_files(evidence):
    results, output, metadata = evidence
    archive = packager.package_evidence(results, output, '1.0.6', metadata)
    assert archive.name == 'delta-svd_1.0.6_test-evidence.zip'
    with zipfile.ZipFile(archive) as bundle:
        expected = {f'{subject}/{name}' for subject in packager.SUBJECTS for name in packager.FILES}
        assert set(bundle.namelist()) == expected | {'evidence.json'}
        for name in expected:
            assert bundle.read(name) == (results / name).read_bytes()
        saved = json.loads(bundle.read('evidence.json'))
        assert saved == {
            **metadata,
            'comparison': {'metric_values': 'exact', 'voxel_counts': 'exact', 'ignored_keys': []},
            'subjects': {'sub-test01': 'passed', 'sub-test02': 'passed'},
        }


def test_missing_subject_file_blocks_packaging(evidence):
    results, output, metadata = evidence
    (results / 'sub-test02/delta-svd_results.csv').unlink()
    with pytest.raises(FileNotFoundError):
        packager.package_evidence(results, output, '1.0.6', metadata)
    assert not output.exists()


@pytest.mark.parametrize('field,value', [
    ('subject_id', 'other-subject'), ('pipeline_version', '1.0.5'),
    ('source_revision', 'd' * 40), ('processing_mode', 'cross_sectional'),
])
def test_wrong_manifest_blocks_packaging(evidence, field, value):
    results, output, metadata = evidence
    path = results / 'sub-test01/delta-svd_run_manifest.json'
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='manifest'):
        packager.package_evidence(results, output, '1.0.6', metadata)
    assert not output.exists()


def test_failed_comparison_blocks_packaging(evidence):
    results, output, metadata = evidence
    (results / 'sub-test02/comparison.txt').write_text('RESULT: 1 difference(s) found\n')
    with pytest.raises(ValueError, match='comparison'):
        packager.package_evidence(results, output, '1.0.6', metadata)
    assert not output.exists()


@pytest.mark.parametrize('comparison', [
    'RESULT: no metric value and no voxel count changed.\n',  # -q
    PASSING_COMPARISON.replace(
        'Rows keyed by ID, timepoint, skeleton, region, metric.',
        'Rows keyed by ID, timepoint, region, metric; skeleton ignored.'),
    PASSING_COMPARISON.replace('Metric values and voxel counts compared exactly.\n', ''),
    'WARNING: ID, timepoint, skeleton, region, metric do(es) not identify a row uniquely; '
    'rows sharing a key are matched in file order.\n' + PASSING_COMPARISON,
    PASSING_COMPARISON.replace(
        '\nRESULT:', '\nWARNING: keys do(es) not identify a row uniquely\nRESULT:'),
    '',
])
def test_invalid_comparison_blocks_packaging(evidence, comparison):
    results, output, metadata = evidence
    (results / 'sub-test01/comparison.txt').write_text(comparison)
    with pytest.raises(ValueError, match='comparison'):
        packager.package_evidence(results, output, '1.0.6', metadata)
    assert not output.exists()


def test_cli_resolves_version_outside_repository(evidence, monkeypatch):
    import sys

    results, output, metadata = evidence
    repo = output.parent / 'checkout'
    tools = repo / 'tools'
    tools.mkdir(parents=True)
    (repo / 'VERSION').write_text('1.0.6\n')
    monkeypatch.setattr(packager, '__file__', str(tools / 'package_test_evidence.py'))
    monkeypatch.setattr(sys, 'argv', ['package_test_evidence.py', str(results), str(output)])
    monkeypatch.chdir(output.parent)
    for name, value in {
        'IMAGE': metadata['image'], 'GITHUB_SHA': metadata['source_revision'],
        'GITHUB_SERVER_URL': 'https://github.com', 'GITHUB_REPOSITORY': 'isdneuroimaging/DELTA-SVD',
        'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '2',
        'DATASET_URL': metadata['dataset']['url'], 'DATASET_SHA256': metadata['dataset']['sha256'],
    }.items():
        monkeypatch.setenv(name, value)
    packager.main()
    assert (output / 'delta-svd_1.0.6_test-evidence.zip').is_file()


def verify_promotion(archive, metadata):
    """Run the workflow's actual read-only ZIP check with fixture metadata."""
    import os
    import subprocess
    import sys
    import textwrap

    workflow = (Path(__file__).resolve().parents[1] / '.github/workflows/release-promote.yml').read_text()
    code = textwrap.dedent(workflow.split("python3 - <<'PY'\n", 1)[1].split('\n          PY', 1)[0])
    return subprocess.run(
        [sys.executable, '-c', code], cwd=archive.parent.parent,
        env={**os.environ, 'SRC': metadata['image'], 'REVISION': metadata['source_revision'],
             'BUILD_RUN': '123', 'BUILD_ATTEMPT': '2', 'VERSION': '1.0.6'},
        capture_output=True, text=True,
    )


def test_promotion_accepts_matching_archive(evidence):
    results, output, metadata = evidence
    archive = packager.package_evidence(results, output, '1.0.6', metadata)
    result = verify_promotion(archive, metadata)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('key,value', [
    ('image', 'different-image'), ('source_revision', 'd' * 40),
    ('workflow_run', 'https://github.com/isdneuroimaging/DELTA-SVD/actions/runs/999'),
    ('run_attempt', 1),
])
def test_promotion_rejects_different_identity(evidence, key, value):
    results, output, metadata = evidence
    archive = packager.package_evidence(results, output, '1.0.6', metadata)
    with zipfile.ZipFile(archive) as bundle:
        contents = {name: bundle.read(name) for name in bundle.namelist()}
    saved = json.loads(contents['evidence.json'])
    saved[key] = value
    contents['evidence.json'] = json.dumps(saved).encode()
    with zipfile.ZipFile(archive, 'w') as bundle:
        for name, data in contents.items():
            bundle.writestr(name, data)
    result = verify_promotion(archive, metadata)
    assert result.returncode != 0
    assert key in result.stderr


def test_promotion_rejects_incomplete_zip(evidence):
    results, output, metadata = evidence
    archive = packager.package_evidence(results, output, '1.0.6', metadata)
    with zipfile.ZipFile(archive) as bundle:
        data = bundle.read('evidence.json')
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('evidence.json', data)
    result = verify_promotion(archive, metadata)
    assert result.returncode != 0
    assert 'Incomplete' in result.stderr
