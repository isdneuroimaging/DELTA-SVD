import pandas as pd
import pytest

# tools/compare_results.py diffs two results tables so that a metric that moved
# cannot be missed by eye. Its whole point is to be conservative: anything it
# reports as unchanged has to genuinely be unchanged.

HEADER = "ID,timepoint,skeleton,region,voxels,metric,value\n"


def _rows(msmd="0.00073214", psmd="0.00042100", voxels="118432",
          skel="skel.nii.gz", psmd_name="PSMD"):
    # a metrics row pair plus the kind of debugging row integrate_masks emits
    # (metric 'NA' / value 'NaN', which pandas reads back as missing values).
    # 'skel' and 'psmd_name' model a label rename between two runs.
    return (
        HEADER
        + f"P01,ses-1,{skel},intersection,{voxels},MSMD,{msmd}\n"
        + f"P01,ses-1,{skel},intersection,{voxels},{psmd_name},{psmd}\n"
        + f"P01,all,{skel},total,{voxels},NA,NaN\n"
    )


def _write(tmp_path, name, content):
    fn = tmp_path / name
    fn.write_text(content)
    return fn


def _compare(compare_results, tmp_path, before, after, **kwargs):
    fnB = _write(tmp_path, "before.csv", before)
    fnA = _write(tmp_path, "after.csv", after)
    return compare_results.compare(
        compare_results.read_results(fnB), compare_results.read_results(fnA), **kwargs)


def test_identical_tables_report_no_difference(compare_results, tmp_path):
    assert _compare(compare_results, tmp_path, _rows(), _rows()) == []


def test_debugging_rows_with_missing_values_are_not_flagged(compare_results, tmp_path):
    # 'NA'/'NaN' on both sides must compare equal, not as NaN != NaN
    diffs = _compare(compare_results, tmp_path, _rows(), _rows())
    assert not any("total" in d for d in diffs)


def test_row_order_alone_is_not_a_difference(compare_results, tmp_path):
    shuffled = HEADER + "\n".join(reversed(_rows().strip().split("\n")[1:])) + "\n"
    assert _compare(compare_results, tmp_path, _rows(), shuffled) == []


def test_changed_metric_value_is_reported(compare_results, tmp_path):
    diffs = _compare(compare_results, tmp_path, _rows(), _rows(msmd="0.00073219"))
    assert len(diffs) == 1
    assert "value changed" in diffs[0]
    assert "MSMD" in diffs[0]


def test_tiny_metric_change_is_reported(compare_results, tmp_path):
    # comparison is exact: a last-digit move must not slip through
    diffs = _compare(compare_results, tmp_path, _rows(), _rows(msmd="0.000732140001"))
    assert len(diffs) == 1


def test_single_float32_ulp_metric_change_is_reported(compare_results, tmp_path):
    # the differences this tool exists to catch can be one float32 ULP: exactly
    # such a change in one FA voxel moved the longitudinal skeleton by 146 voxels
    # (see CONTRIBUTING.md). The pipeline writes float32, so these are adjacent.
    diffs = _compare(compare_results, tmp_path,
                     _rows(msmd="0.0007321399752981961"),
                     _rows(msmd="0.000732140033505857"))
    assert len(diffs) == 1, "a one-ULP move must be reported, not absorbed"


def test_changed_voxel_count_is_reported(compare_results, tmp_path):
    diffs = _compare(compare_results, tmp_path, _rows(), _rows(voxels="118429"))
    assert diffs, "a changed voxel count is a changed skeleton"
    assert all("voxels changed" in d for d in diffs)


def test_compare_takes_no_tolerance_argument(compare_results, tmp_path):
    # --rtol was retired: both run modes are bit-reproducible once --threads is
    # pinned, so a tolerance could only ever hide a real change
    with pytest.raises(TypeError):
        _compare(compare_results, tmp_path, _rows(), _rows(), rtol=0.5)


def test_added_region_is_reported_as_a_structural_difference(compare_results, tmp_path):
    extra = _rows() + "P01,ses-1,skel.nii.gz,intersection_LH,5000,MSMD,0.0007\n"
    diffs = _compare(compare_results, tmp_path, _rows(), extra)
    assert len(diffs) == 1
    assert "only in AFTER" in diffs[0]
    assert "intersection_LH" in diffs[0]


def test_table_without_the_expected_columns_is_rejected(compare_results, tmp_path):
    fn = _write(tmp_path, "junk.csv", "a,b\n1,2\n")
    with pytest.raises(ValueError, match="missing expected column"):
        compare_results.read_results(fn)


# ---------------------------------------------------------------------------
# --ignore-key: a label was renamed, the numbers are still the question

def test_renamed_skeleton_is_a_difference_by_default(compare_results, tmp_path):
    # the conservative default must not wave a renamed mask through
    diffs = _compare(compare_results, tmp_path, _rows(), _rows(skel="renamed.nii.gz"))
    assert diffs and all("only in" in d for d in diffs)


def test_ignoring_the_renamed_key_compares_the_values(compare_results, tmp_path):
    diffs = _compare(compare_results, tmp_path, _rows(), _rows(skel="renamed.nii.gz"),
                     ignore_keys=["skeleton"])
    assert diffs == []


def test_ignoring_a_key_still_reports_a_moved_value(compare_results, tmp_path):
    # the whole point: ignoring a label must not also hide a changed number
    diffs = _compare(compare_results, tmp_path, _rows(),
                     _rows(skel="renamed.nii.gz", msmd="0.00073219"),
                     ignore_keys=["skeleton"])
    assert len(diffs) == 1
    assert "value changed" in diffs[0]


def test_ignoring_a_key_still_reports_a_moved_voxel_count(compare_results, tmp_path):
    diffs = _compare(compare_results, tmp_path, _rows(),
                     _rows(skel="renamed.nii.gz", voxels="118429"),
                     ignore_keys=["skeleton"])
    assert diffs and all("voxels changed" in d for d in diffs)


def test_several_keys_can_be_ignored_at_once(compare_results, tmp_path):
    # mask file and metric both renamed, numbers unchanged
    diffs = _compare(compare_results, tmp_path, _rows(),
                     _rows(skel="renamed.nii.gz", psmd_name="RENAMED"),
                     ignore_keys=["skeleton", "metric"])
    assert diffs == []


def test_ignoring_an_unknown_column_is_rejected(compare_results, tmp_path):
    with pytest.raises(ValueError, match="not a key column"):
        _compare(compare_results, tmp_path, _rows(), _rows(), ignore_keys=["voxels"])


def test_ignoring_every_key_column_is_reported(compare_results, tmp_path):
    diffs = _compare(compare_results, tmp_path, _rows(), _rows(),
                     ignore_keys=["ID", "timepoint", "skeleton", "region", "metric"])
    assert len(diffs) == 1
    assert "no key column is left" in diffs[0]


# ---------------------------------------------------------------------------
# Exit status is what a caller scripts against

def test_main_returns_zero_when_tables_agree(compare_results, tmp_path, capsys):
    fnB = _write(tmp_path, "before.csv", _rows())
    fnA = _write(tmp_path, "after.csv", _rows())
    assert compare_results.main([str(fnB), str(fnA)]) == 0
    assert "no metric value and no voxel count changed" in capsys.readouterr().out


def test_main_returns_one_when_a_metric_moved(compare_results, tmp_path, capsys):
    fnB = _write(tmp_path, "before.csv", _rows())
    fnA = _write(tmp_path, "after.csv", _rows(msmd="0.00073219"))
    assert compare_results.main([str(fnB), str(fnA)]) == 1
    assert "metric-affecting" in capsys.readouterr().out


def test_main_accepts_repeated_ignore_key(compare_results, tmp_path, capsys):
    fnB = _write(tmp_path, "before.csv", _rows())
    fnA = _write(tmp_path, "after.csv", _rows(skel="renamed.nii.gz", psmd_name="RENAMED"))
    assert compare_results.main(
        [str(fnB), str(fnA), "--ignore-key", "skeleton", "--ignore-key", "metric"]) == 0
    captured = capsys.readouterr()
    assert "no metric value and no voxel count changed" in captured.out
    assert "skeleton, metric ignored" in captured.out
    # ignoring 'metric' leaves the two intersection rows indistinguishable
    assert "WARNING" in captured.err


def test_main_returns_two_for_an_unknown_ignore_key(compare_results, tmp_path, capsys):
    fnB = _write(tmp_path, "before.csv", _rows())
    fnA = _write(tmp_path, "after.csv", _rows())
    assert compare_results.main([str(fnB), str(fnA), "--ignore-key", "nonsense"]) == 2
    assert "ERROR" in capsys.readouterr().err


def test_main_returns_two_when_a_table_cannot_be_read(compare_results, tmp_path, capsys):
    fnB = _write(tmp_path, "before.csv", _rows())
    assert compare_results.main([str(fnB), str(tmp_path / "missing.csv")]) == 2
    assert "ERROR" in capsys.readouterr().err
