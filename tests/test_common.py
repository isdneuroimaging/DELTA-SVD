import numpy as np
import pytest

import delta_svd_common


def test_validate_roi_labels_accepts_documented_boundaries():
    labels = delta_svd_common.validate_roi_labels(
        np.array([0, 255, 256, 65535], dtype=float), "ROI mask", "roi.nii.gz")
    assert labels.dtype == np.dtype("uint16")
    assert labels.tolist() == [0, 255, 256, 65535]


@pytest.mark.parametrize("invalid", [-1, 0.5, 65536, np.nan, np.inf])
def test_validate_roi_labels_rejects_unsupported_values(invalid):
    with pytest.raises(delta_svd_common.DeltaSvdError, match="finite integer labels from 0 through 65535"):
        delta_svd_common.validate_roi_labels(np.array([0, invalid]), "ROI mask", "roi.nii.gz")


def test_recursive_delete_requires_containment(tmp_path):
    allowed = tmp_path / "output"
    unrelated = tmp_path / "other"
    allowed.mkdir()
    unrelated.mkdir()
    sentinel = unrelated / "keep.txt"
    sentinel.touch()

    with pytest.raises(ValueError, match="Refusing to recursively delete"):
        delta_svd_common.safe_rmtree(str(unrelated), str(allowed))

    assert sentinel.exists()


# ---------------------------------------------------------------------------
# Every user-fixable error is a DeltaSvdError, so the __main__ guard can report
# it without a traceback. A ValueError from anywhere else is a bug and has to
# keep its traceback, so the guard must not swallow it.

def test_delta_svd_error_is_a_value_error():
    assert issubclass(delta_svd_common.DeltaSvdError, ValueError)


# ---------------------------------------------------------------------------
# run_subprocess(): displayed output is streamed; hidden output only on failure.

@pytest.mark.parametrize("display", [True, False])
def test_run_subprocess_success(capsys, display):
    delta_svd_common.run_subprocess("echo hello; echo oops >&2", display, "demo")
    out = capsys.readouterr().out
    assert ("hello\noops\n" in out) == display


@pytest.mark.parametrize("display", [True, False])
def test_run_subprocess_failure_shows_the_output(capsys, display):
    with pytest.raises(delta_svd_common.DeltaSvdError, match="'demo' command failed with exit code 3"):
        delta_svd_common.run_subprocess("echo partial; echo why >&2; exit 3", display, "demo")
    out = capsys.readouterr().out
    assert "partial\nwhy\n" in out
