import argparse
import csv
import datetime
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "container" / "scripts" / "delta-svd_aggregate_results.py"

CSV_HEADER = "ID,timepoint,skeleton,region,voxels,metric,value\n"
CSV_HEADER_NO_ID = "timepoint,skeleton,region,voxels,metric,value\n"


def _write_fixture_csv(dirpath, patient_id, value_psmd, value_msmd):
    dirpath.mkdir(parents=True, exist_ok=True)
    content = (
        CSV_HEADER
        + f"{patient_id},TP01,skel.nii.gz,intersection,100,PSMD,{value_psmd}\n"
        + f"{patient_id},TP01,skel.nii.gz,intersection,100,MSMD,{value_msmd}\n"
    )
    (dirpath / "delta-svd_results.csv").write_text(content)


def _write_fixture_csv_without_id(dirpath, value_psmd, value_msmd):
    # what a run started without '--id' produces: no identifier column at all
    dirpath.mkdir(parents=True, exist_ok=True)
    content = (
        CSV_HEADER_NO_ID
        + f"TP01,skel.nii.gz,intersection,100,PSMD,{value_psmd}\n"
        + f"TP01,skel.nii.gz,intersection,100,MSMD,{value_msmd}\n"
    )
    (dirpath / "delta-svd_results.csv").write_text(content)


def _write_fixture_csv_with_debug_row(dirpath, patient_id, value_psmd, value_msmd):
    # a bookkeeping row as written by integrate_masks(): 'NA'/'NaN' sentinels
    # that read_csv coerces to real NaN, which is how the aggregator tells
    # bookkeeping rows apart from metric rows.
    dirpath.mkdir(parents=True, exist_ok=True)
    content = (
        CSV_HEADER
        + f"{patient_id},TP01,skel.nii.gz,intersection,100,PSMD,{value_psmd}\n"
        + f"{patient_id},TP01,skel.nii.gz,intersection,100,MSMD,{value_msmd}\n"
        + f"{patient_id},TP01,skel.nii.gz,LH,40,NA,NaN\n"
    )
    (dirpath / "delta-svd_results.csv").write_text(content)


def _write_header_only_csv(dirpath):
    # what an interrupted or failed run can leave behind: a table without rows
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "delta-svd_results.csv").write_text(CSV_HEADER)


def _read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True,
    )


# ---------------------------------------------------------------------------
# Argparse validators
#
# The aggregation logic itself lives in the `if __name__ == "__main__":`
# block rather than in a function, so it is exercised end-to-end via
# subprocess below. Only the argparse type validators and parser
# construction (iniParser) are plain module-level functions.

def test_isDir_accepts_existing_directory(aggregate_results, tmp_path):
    assert aggregate_results.isDir(str(tmp_path)) == str(tmp_path)


def test_isDir_rejects_non_directory(aggregate_results, tmp_path):
    with pytest.raises(argparse.ArgumentTypeError):
        aggregate_results.isDir(str(tmp_path / "missing"))


def test_extCSV_accepts_csv_extension(aggregate_results):
    assert aggregate_results.extCSV("results.csv") == "results.csv"


def test_extCSV_rejects_other_extensions(aggregate_results):
    with pytest.raises(argparse.ArgumentTypeError):
        aggregate_results.extCSV("results.txt")


def test_plausiblePath_accepts_existing_directory(aggregate_results, tmp_path):
    assert aggregate_results.plausiblePath(str(tmp_path)) == str(tmp_path)


def test_plausiblePath_accepts_bare_csv_filename(aggregate_results):
    assert aggregate_results.plausiblePath("out.csv") == "out.csv"


def test_plausiblePath_accepts_csv_in_existing_directory(aggregate_results, tmp_path):
    path = str(tmp_path / "out.csv")
    assert aggregate_results.plausiblePath(path) == path


def test_plausiblePath_rejects_csv_in_missing_directory(aggregate_results, tmp_path):
    with pytest.raises(argparse.ArgumentTypeError):
        aggregate_results.plausiblePath(str(tmp_path / "missing" / "out.csv"))


def test_plausiblePath_rejects_non_csv_filename(aggregate_results, tmp_path):
    with pytest.raises(argparse.ArgumentTypeError):
        aggregate_results.plausiblePath(str(tmp_path / "out.txt"))


def test_iniparser_defaults(aggregate_results):
    parser = aggregate_results.iniParser()
    assert parser.get_default("filename") == "delta-svd_results.csv"
    assert parser.get_default("depth") == -1
    assert parser.get_default("output") == aggregate_results.fnOutDefault


# ---------------------------------------------------------------------------
# End-to-end aggregation (subprocess, since the logic is script-level code)

def test_aggregate_globs_and_concatenates_all_result_files(tmp_path):
    _write_fixture_csv(tmp_path / "patient1", "P01", "0.0004", "0.0007")
    _write_fixture_csv(tmp_path / "patient2", "P02", "0.0005", "0.0008")

    out = tmp_path / "agg.csv"
    result = _run(str(tmp_path), "-o", str(out))

    assert result.returncode == 0, result.stderr
    content = out.read_text()
    assert "P01" in content
    assert "P02" in content
    assert content.count("PSMD") == 2


def test_aggregate_split_writes_metrics_and_debugging_files(tmp_path):
    _write_fixture_csv(tmp_path / "patient1", "P01", "0.0004", "0.0007")
    _write_fixture_csv(tmp_path / "patient2", "P02", "0.0005", "0.0008")

    out = tmp_path / "agg.csv"
    result = _run(str(tmp_path), "-o", str(out), "-s")

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "agg_metrics.csv").exists()
    assert (tmp_path / "agg_debugging.csv").exists()


def test_aggregate_split_separates_bookkeeping_rows_from_metrics(tmp_path):
    _write_fixture_csv_with_debug_row(tmp_path / "patient1", "P01", "0.0004", "0.0007")

    out = tmp_path / "agg.csv"
    result = _run(str(tmp_path), "-o", str(out), "-s")

    assert result.returncode == 0, result.stderr
    metric_rows = _read_rows(tmp_path / "agg_metrics.csv")
    debug_rows = _read_rows(tmp_path / "agg_debugging.csv")

    assert len(metric_rows) == 2
    assert {row["metric"] for row in metric_rows} == {"PSMD", "MSMD"}

    assert len(debug_rows) == 1
    assert debug_rows[0]["region"] == "LH"
    # 'metric'/'value' are entirely NaN across all bookkeeping rows, so they
    # get dropped from the debugging table rather than rendered empty.
    assert "metric" not in debug_rows[0]
    assert "value" not in debug_rows[0]


def test_aggregate_insert_path_column(tmp_path):
    _write_fixture_csv(tmp_path / "patient1", "P01", "0.0004", "0.0007")

    out = tmp_path / "agg.csv"
    result = _run(str(tmp_path), "-o", str(out), "-p")

    assert result.returncode == 0, result.stderr
    header = out.read_text().splitlines()[0]
    assert header.split(",") == ["ID", "path", "timepoint", "skeleton", "region", "voxels", "metric", "value"]


# ---------------------------------------------------------------------------
# The 'path' column has to identify the file a row actually came from. '-p' is
# the documented remedy for tables carrying no 'ID' column, so it must work on
# exactly those (it used to raise "ValueError: 'ID' is not in list"), and a file
# contributing no rows must not shift the path assignment of the files after it.

def test_aggregate_insert_path_column_without_id_column(tmp_path):
    _write_fixture_csv_without_id(tmp_path / "patient1", "0.0004", "0.0007")
    _write_fixture_csv_without_id(tmp_path / "patient2", "0.0005", "0.0008")

    out = tmp_path / "agg.csv"
    result = _run(str(tmp_path), "-o", str(out), "-p")

    assert result.returncode == 0, result.stderr
    rows = _read_rows(out)
    assert out.read_text().splitlines()[0].split(",")[0] == "path"
    assert [row["path"] for row in rows].count(
        str(tmp_path / "patient1" / "delta-svd_results.csv")) == 2
    assert [row["path"] for row in rows].count(
        str(tmp_path / "patient2" / "delta-svd_results.csv")) == 2


def test_aggregate_path_column_tracks_source_when_a_csv_has_no_rows(tmp_path):
    _write_fixture_csv(tmp_path / "patient1", "P01", "0.0004", "0.0007")
    _write_header_only_csv(tmp_path / "patient2")
    _write_fixture_csv(tmp_path / "patient3", "P03", "0.0005", "0.0008")

    out = tmp_path / "agg.csv"
    result = _run(str(tmp_path), "-o", str(out), "-p")

    assert result.returncode == 0, result.stderr
    rows = _read_rows(out)
    assert len(rows) == 4
    for row in rows:
        expected = {"P01": "patient1", "P03": "patient3"}[row["ID"]]
        assert expected in row["path"], row


def test_aggregate_excludes_csv_without_data_rows(tmp_path):
    _write_fixture_csv(tmp_path / "patient1", "P01", "0.0004", "0.0007")
    _write_header_only_csv(tmp_path / "patient2")

    out = tmp_path / "agg.csv"
    result = _run(str(tmp_path), "-o", str(out))

    assert result.returncode == 0, result.stderr
    assert "without any data rows" in result.stdout
    assert len(_read_rows(out)) == 2


def test_aggregate_errors_when_no_csv_has_data_rows(tmp_path):
    _write_header_only_csv(tmp_path / "patient1")
    _write_header_only_csv(tmp_path / "patient2")

    result = _run(str(tmp_path), "-o", str(tmp_path / "agg.csv"))

    assert result.returncode != 0
    assert not (tmp_path / "agg.csv").exists()


# ---------------------------------------------------------------------------
# Without '-p', the 'ID' column is the only identifier, so it must be constant
# within each file and distinct between files (both grouped per source file).

def test_aggregate_rejects_ids_that_are_not_distinct_between_files(tmp_path):
    _write_fixture_csv(tmp_path / "patient1", "P01", "0.0004", "0.0007")
    _write_fixture_csv(tmp_path / "patient2", "P01", "0.0005", "0.0008")

    result = _run(str(tmp_path), "-o", str(tmp_path / "agg.csv"))

    assert result.returncode != 0
    assert "not all distinct" in result.stderr


def test_aggregate_rejects_id_varying_within_one_file(tmp_path):
    dirpath = tmp_path / "patient1"
    dirpath.mkdir()
    (dirpath / "delta-svd_results.csv").write_text(
        CSV_HEADER
        + "P01,TP01,skel.nii.gz,intersection,100,PSMD,0.0004\n"
        + "P02,TP01,skel.nii.gz,intersection,100,MSMD,0.0007\n"
    )

    result = _run(str(tmp_path), "-o", str(tmp_path / "agg.csv"))

    assert result.returncode != 0
    assert "not constant" in result.stderr


def test_aggregate_overwrite_guard_requires_explicit_flag(tmp_path):
    _write_fixture_csv(tmp_path / "patient1", "P01", "0.0004", "0.0007")
    out = tmp_path / "agg.csv"

    first = _run(str(tmp_path), "-o", str(out))
    assert first.returncode == 0, first.stderr

    second = _run(str(tmp_path), "-o", str(out))
    assert second.returncode != 0

    third = _run(str(tmp_path), "-o", str(out), "-x")
    assert third.returncode == 0, third.stderr


def test_aggregate_appends_date_suffix_to_output_filename(tmp_path):
    _write_fixture_csv(tmp_path / "patient1", "P01", "0.0004", "0.0007")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = _run(str(tmp_path), "-o", str(out_dir / "agg.csv"), "-t", "date")

    assert result.returncode == 0, result.stderr
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    produced = os.listdir(out_dir)
    assert any(today in fn for fn in produced), produced


# ---------------------------------------------------------------------------
# Errors go to stderr with an 'ERROR:' prefix and no traceback, matching
# delta-svd.py. Every check here sits in one linear script-level flow, so this
# uses sys.exit() directly rather than delta-svd.py's DeltaSvdError.

def test_aggregate_errors_go_to_stderr_without_a_traceback(tmp_path):
    _write_fixture_csv(tmp_path / "patient1", "P01", "0.0004", "0.0007")
    out = tmp_path / "agg.csv"
    assert _run(str(tmp_path), "-o", str(out)).returncode == 0

    result = _run(str(tmp_path), "-o", str(out))

    assert result.returncode == 1
    assert result.stderr.startswith("\nERROR: ")
    assert "Traceback" not in result.stderr
    # the hint naming the option that resolves it travels with the message
    assert "'-x'" in result.stderr


def test_aggregate_reports_an_empty_glob_on_stderr(tmp_path):
    result = _run(str(tmp_path), "-o", str(tmp_path / "agg.csv"))

    assert result.returncode == 1
    assert "No CSV files found" in result.stderr
    assert "'-f'" in result.stderr and "'-d'" in result.stderr

