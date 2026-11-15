import os
import shlex
import sys

import nibabel as nib
import numpy as np
import pytest

import delta_svd_inputs
# grid helpers shared with the input tests; _mni_reference is a fixture
from test_inputs import _DWI_AFFINE, _mni_reference, _save_dwi, _save_grid


# ---------------------------------------------------------------------------
# "steps must be contiguous" / "--qc 0 contradicts --steps qc" error paths

def _minimal_pipeline_inputs(tmp_path):
    dwi = tmp_path / "sub01.nii.gz"
    skel = tmp_path / "skel.nii.gz"
    for fn in [tmp_path / "sub01.bval", tmp_path / "sub01.bvec", skel]:
        fn.touch()
    # real headers: the brain mask is checked against the DWI grid up front
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 2), dtype="float32"), np.eye(4)), str(dwi))
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype="uint8"), np.eye(4)),
             str(tmp_path / "sub01_brainmask.nii.gz"))
    return dwi, skel


def test_pipeline_rejects_noncontiguous_steps(delta_svd, tmp_path, monkeypatch):
    dwi, skel = _minimal_pipeline_inputs(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "delta-svd.py", "--dwi", str(dwi), "--skeletonMask", str(skel),
        "--steps", "fwc", "extract",
    ])
    with pytest.raises(ValueError, match="have to be contiguous"):
        delta_svd.pipeline_delta_svd()


def test_pipeline_records_unambiguous_function_call(
        delta_svd, tmp_path, monkeypatch, capsys):
    dwi, skel = _minimal_pipeline_inputs(tmp_path)
    argv = [
        "delta-svd.py", "--dwi", str(dwi), "--skeletonMask", str(skel),
        "--id", "subject '01'", "--steps", "fwc", "extract",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(ValueError, match="have to be contiguous"):
        delta_svd.pipeline_delta_svd()

    command = capsys.readouterr().out.split("Running: ", 1)[1].splitlines()[0]
    assert shlex.split(command) == argv


def test_pipeline_rejects_qc_zero_with_qc_step_requested(delta_svd, tmp_path, monkeypatch):
    dwi, skel = _minimal_pipeline_inputs(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "delta-svd.py", "--dwi", str(dwi), "--skeletonMask", str(skel),
        "--steps", "qc", "--qc", "0",
    ])
    with pytest.raises(ValueError, match="contradictory"):
        delta_svd.pipeline_delta_svd()


# ---------------------------------------------------------------------------
# A repeated label makes two timepoints share a working folder and their rows:
# the second overwrites the first, and the run reports it twice. 'all' is what
# integrate_masks() calls the aggregated rows, so it is reserved as well.

def _two_timepoint_inputs(tmp_path):
    dwi, skel = _minimal_pipeline_inputs(tmp_path)
    dwi2 = tmp_path / "sub02.nii.gz"
    for fn in [dwi2, tmp_path / "sub02.bval", tmp_path / "sub02.bvec",
               tmp_path / "sub02_brainmask.nii.gz"]:
        fn.touch()
    return dwi, dwi2, skel


def _argv(dwis, skel, *rest):
    return (["delta-svd.py", "--dwi"] + [str(d) for d in dwis]
            + ["--skeletonMask", str(skel)] + list(rest))


def test_pipeline_rejects_duplicate_timepoint_labels(delta_svd, tmp_path, monkeypatch):
    dwi, dwi2, skel = _two_timepoint_inputs(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv([dwi, dwi2], skel, "--tp", "V1", "V1"))

    with pytest.raises(ValueError, match="have to be unique"):
        delta_svd.pipeline_delta_svd()


def test_pipeline_rejects_all_as_a_longitudinal_timepoint_label(delta_svd, tmp_path, monkeypatch):
    dwi, dwi2, skel = _two_timepoint_inputs(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv([dwi, dwi2], skel, "--tp", "all", "V2"))

    with pytest.raises(ValueError, match="'all' is reserved"):
        delta_svd.pipeline_delta_svd()


@pytest.mark.parametrize("kind", ["absolute", "traversal"])
def test_unsafe_reprocessing_leaves_unrelated_files_untouched(
        delta_svd, tmp_path, monkeypatch, kind):
    dwi, skel = _minimal_pipeline_inputs(tmp_path)
    output = tmp_path / "output"
    (output / "delta-svd_temp").mkdir(parents=True)
    unrelated = (tmp_path if kind == "absolute" else output) / "other"
    unrelated.mkdir()
    sentinel = unrelated / "keep.txt"
    sentinel.write_text("untouched")
    label = str(unrelated) if kind == "absolute" else "../other"
    monkeypatch.setattr(sys, "argv", _argv(
        [dwi], skel, "--tp", label, "--dirOutput", str(output),
        "--steps", "fwc", "--reprocess"))

    with pytest.raises(ValueError, match="absolute path|path traversal"):
        delta_svd.pipeline_delta_svd()

    assert sentinel.read_text() == "untouched"


def test_valid_timepoint_reprocessing_only_replaces_its_working_directory(
        delta_svd, tmp_path, monkeypatch):
    dwi, skel = _minimal_pipeline_inputs(tmp_path)
    output = tmp_path / "output"
    timepoint = output / "delta-svd_temp" / "V1"
    timepoint.mkdir(parents=True)
    stale = timepoint / "stale.txt"
    stale.touch()
    sentinel = output / "keep.txt"
    sentinel.write_text("untouched")
    monkeypatch.setattr(sys, "argv", _argv(
        [dwi], skel, "--tp", "V1", "--dirOutput", str(output),
        "--steps", "fwc", "--reprocess"))
    # the touched skeleton mask is no NIfTI; the grid check has its own tests
    monkeypatch.setattr(delta_svd_inputs, "check_mni_grid", lambda *args, **kwargs: None)
    monkeypatch.setattr(delta_svd, "filter_b_values",
                        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("stop")))

    with pytest.raises(RuntimeError, match="stop"):
        delta_svd.pipeline_delta_svd()

    assert timepoint.is_dir() and not stale.exists()
    assert sentinel.read_text() == "untouched"


def test_pipeline_accepts_distinct_timepoint_labels(delta_svd, tmp_path, monkeypatch):
    # a well-formed run must get through, as far as the step check further down
    dwi, dwi2, skel = _two_timepoint_inputs(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv([dwi, dwi2], skel, "--tp", "V1", "V2",
                                           "--steps", "fwc", "extract"))

    with pytest.raises(ValueError, match="have to be contiguous"):
        delta_svd.pipeline_delta_svd()


# ---------------------------------------------------------------------------
# '--qc 0' used to do `stepsImplemented.remove('qc')` on the module-level list
# that also backs the '--steps' argparse choices, so a second run in the same
# process died on "list.remove(x): x not in list" and lost 'qc' as a valid
# choice. The step list is now copied per run.

def test_qc_zero_does_not_erode_the_module_level_step_list(delta_svd, tmp_path, monkeypatch):
    dwi, skel = _minimal_pipeline_inputs(tmp_path)
    argv = ["delta-svd.py", "--dwi", str(dwi), "--skeletonMask", str(skel),
            "--qc", "0", "--steps", "fwc", "extract"]

    # non-contiguous steps, so each run aborts just after the 'qc' removal point
    for _ in range(2):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(ValueError, match="have to be contiguous"):
            delta_svd.pipeline_delta_svd()

    assert "qc" in delta_svd_inputs.stepsImplemented
    # and '--steps qc' is still an accepted choice afterwards
    ns = delta_svd.iniParser().parse_args(
        ["--dwi", str(dwi), "--skeletonMask", str(skel), "--steps", "qc"])
    assert ns.steps == ["qc"]


# ---------------------------------------------------------------------------
# "Hemispheric skeleton analysis" message should only print when --hemispheres
# was actually passed (regression test: previously guarded by
# `if args.hemispheres is not None`, which is always true since the default
# is False, not None).

def test_hemispheres_message_not_printed_by_default(delta_svd, tmp_path, monkeypatch, capsys):
    dwi, skel = _minimal_pipeline_inputs(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "delta-svd.py", "--dwi", str(dwi), "--skeletonMask", str(skel),
        "--steps", "qc", "--qc", "0",
    ])
    with pytest.raises(ValueError):
        delta_svd.pipeline_delta_svd()
    assert "Hemispheric skeleton analysis" not in capsys.readouterr().out


def test_hemispheres_message_printed_when_flag_set(delta_svd, tmp_path, monkeypatch, capsys):
    dwi, skel = _minimal_pipeline_inputs(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "delta-svd.py", "--dwi", str(dwi), "--skeletonMask", str(skel),
        "--hemispheres", "--steps", "qc", "--qc", "0",
    ])
    with pytest.raises(ValueError):
        delta_svd.pipeline_delta_svd()
    assert "Hemispheric skeleton analysis" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Resolution of the bval/bvec/bmask paths. Both NIfTI extensions are supported
# for the brain mask, whether it is inferred from the DWI path or passed
# explicitly. The runs below are aborted (deliberately contradictory
# '--steps qc --qc 0') right after the resolved inputs have been echoed.

def _resolved_inputs(delta_svd, monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, "argv", ["delta-svd.py"] + argv)
    with pytest.raises(ValueError, match="contradictory"):
        delta_svd.pipeline_delta_svd()
    return capsys.readouterr().out


@pytest.mark.parametrize("dwiExt", [".nii.gz", ".nii"])
@pytest.mark.parametrize("maskExt", [".nii.gz", ".nii"])
def test_bmask_inference_accepts_both_nifti_extensions(delta_svd, tmp_path, monkeypatch, capsys, dwiExt, maskExt):
    dwi = tmp_path / ("sub01" + dwiExt)
    bmask = tmp_path / ("sub01_brainmask" + maskExt)
    skel = tmp_path / "skel.nii.gz"
    for fn in [dwi, bmask, skel, tmp_path / "sub01.bval", tmp_path / "sub01.bvec"]:
        fn.touch()

    out = _resolved_inputs(delta_svd, monkeypatch, capsys, [
        "--dwi", str(dwi), "--skeletonMask", str(skel), "--steps", "qc", "--qc", "0",
    ])
    assert f"Bmask :{bmask}" in out


def test_bmask_inference_prefers_gzipped_mask(delta_svd, tmp_path, monkeypatch, capsys):
    dwi = tmp_path / "sub01.nii"
    skel = tmp_path / "skel.nii.gz"
    gz = tmp_path / "sub01_brainmask.nii.gz"
    for fn in [dwi, skel, gz, tmp_path / "sub01_brainmask.nii",
               tmp_path / "sub01.bval", tmp_path / "sub01.bvec"]:
        fn.touch()

    out = _resolved_inputs(delta_svd, monkeypatch, capsys, [
        "--dwi", str(dwi), "--skeletonMask", str(skel), "--steps", "qc", "--qc", "0",
    ])
    assert f"Bmask :{gz}" in out


@pytest.mark.parametrize("given", ["sub01_brainmask.nii", "sub01_brainmask"])
def test_explicit_bmask_resolves_basename_and_extension(delta_svd, tmp_path, monkeypatch, capsys, given):
    # a basename is resolved against the DWI folder, a missing extension by probing
    dwi = tmp_path / "sub01.nii"
    bmask = tmp_path / "sub01_brainmask.nii"
    skel = tmp_path / "skel.nii.gz"
    for fn in [dwi, bmask, skel, tmp_path / "sub01.bval", tmp_path / "sub01.bvec"]:
        fn.touch()

    out = _resolved_inputs(delta_svd, monkeypatch, capsys, [
        "--dwi", str(dwi), "--bmask", given, "--skeletonMask", str(skel),
        "--steps", "qc", "--qc", "0",
    ])
    assert f"Bmask :{bmask}" in out


def test_missing_bmask_raises_naming_the_option(delta_svd, tmp_path, monkeypatch):
    dwi = tmp_path / "sub01.nii"
    skel = tmp_path / "skel.nii.gz"
    for fn in [dwi, skel, tmp_path / "sub01.bval", tmp_path / "sub01.bvec"]:
        fn.touch()
    monkeypatch.setattr(sys, "argv", [
        "delta-svd.py", "--dwi", str(dwi), "--skeletonMask", str(skel),
        "--steps", "qc", "--qc", "0",
    ])
    with pytest.raises(delta_svd.DeltaSvdError) as excinfo:
        delta_svd.pipeline_delta_svd()

    # the message has to name the DWI it belongs to, say the path was inferred
    # rather than given, list every path probed, and name the option that fixes it
    msg = str(excinfo.value)
    assert str(dwi) in msg
    assert "inferred from the DWI path" in msg
    assert str(tmp_path / "sub01_brainmask.nii.gz") in msg
    assert str(tmp_path / "sub01_brainmask.nii") in msg
    assert "'--bmask'" in msg


# ---------------------------------------------------------------------------
# An inferred path already carries the DWI folder, so re-anchoring it to that
# folder would only probe '<dwiDir>/<dwiDir>/<name>'. Absolute paths hide this,
# because os.path.join() discards its first argument for them -- these tests use
# relative ones on purpose.

def test_inferred_path_is_not_re_anchored_to_the_dwi_folder(delta_svd, tmp_path, monkeypatch):
    sub = tmp_path / "Notch004" / "BL" / "diffusion"
    sub.mkdir(parents=True)
    skel = tmp_path / "skel.nii.gz"
    skel.touch()
    for fn in ["data.nii.gz", "data.bval", "data.bvec"]:
        (sub / fn).touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "delta-svd.py", "--dwi", "Notch004/BL/diffusion/data.nii.gz",
        "--skeletonMask", "skel.nii.gz", "--steps", "qc", "--qc", "0",
    ])
    with pytest.raises(delta_svd.DeltaSvdError) as excinfo:
        delta_svd.pipeline_delta_svd()

    lines = str(excinfo.value).splitlines()
    looked = [ln.strip() for ln in lines[lines.index(" Looked for:") + 1:]
              if ln.startswith("   ")]
    assert looked == ["Notch004/BL/diffusion/data_brainmask.nii.gz",
                      "Notch004/BL/diffusion/data_brainmask.nii"]


def test_explicit_basename_still_resolves_against_the_dwi_folder(delta_svd, tmp_path, monkeypatch, capsys):
    # relative DWI path, so the DWI folder differs from the working directory
    sub = tmp_path / "study" / "sub01"
    sub.mkdir(parents=True)
    skel = tmp_path / "skel.nii.gz"
    skel.touch()
    for fn in ["data.nii.gz", "data.bval", "data.bvec", "mask.nii.gz"]:
        (sub / fn).touch()
    monkeypatch.chdir(tmp_path)

    out = _resolved_inputs(delta_svd, monkeypatch, capsys, [
        "--dwi", "study/sub01/data.nii.gz", "--bmask", "mask.nii.gz",
        "--skeletonMask", "skel.nii.gz", "--steps", "qc", "--qc", "0",
    ])
    assert f"Bmask :{os.path.abspath('study/sub01/mask.nii.gz')}" in out


# ---------------------------------------------------------------------------
# A bare name means the file next to each timepoint's DWI image. It used to be
# looked for in the working directory first, so running from inside one
# session's folder silently gave every session that session's files.

def _two_sessions(tmp_path):
    skel = tmp_path / "skel.nii.gz"
    skel.touch()
    dwis = []
    for ses in ["ses-1", "ses-2"]:
        d = tmp_path / ses
        d.mkdir()
        for fn in ["data.nii.gz", "bvals", "bvecs", "mask.nii.gz", "lesion.nii.gz"]:
            (d / fn).touch()
        dwis.append(str(d / "data.nii.gz"))
    return skel, dwis


def test_bare_names_ignore_a_same_named_file_in_the_working_directory(delta_svd, tmp_path, monkeypatch, capsys):
    skel, dwis = _two_sessions(tmp_path)
    monkeypatch.chdir(tmp_path / "ses-1")

    out = _resolved_inputs(delta_svd, monkeypatch, capsys, [
        "--dwi", *dwis, "--bval", "bvals", "--bvec", "bvecs", "--bmask", "mask",
        "--Emask", "lesion.nii.gz", "lesion.nii.gz",
        "--skeletonMask", str(skel), "--steps", "qc", "--qc", "0",
    ])
    tp1, tp2 = out.split("Timepoint TP02:")
    ses1, ses2 = tmp_path / "ses-1", tmp_path / "ses-2"
    for label, fn in [("bval  ", "bvals"), ("bvec  ", "bvecs"),
                      ("Bmask ", "mask.nii.gz"), ("Emask ", "lesion.nii.gz")]:
        assert f"{label}:{ses1 / fn}" in tp1
        assert f"{label}:{ses2 / fn}" in tp2
    assert str(ses1) not in tp2


def test_bare_name_found_only_in_the_working_directory_is_missing(delta_svd, tmp_path, monkeypatch):
    skel, dwis = _two_sessions(tmp_path)
    (tmp_path / "ses-2" / "bvals").unlink()
    monkeypatch.chdir(tmp_path / "ses-1")
    monkeypatch.setattr(sys, "argv", [
        "delta-svd.py", "--dwi", *dwis, "--bval", "bvals", "--bvec", "bvecs",
        "--bmask", "mask", "--skeletonMask", str(skel), "--steps", "qc", "--qc", "0",
    ])
    with pytest.raises(delta_svd.DeltaSvdError) as excinfo:
        delta_svd.pipeline_delta_svd()

    msg = str(excinfo.value)
    assert "'--bval'" in msg
    assert dwis[1] in msg
    assert str(tmp_path / "ses-2" / "bvals") in msg


def test_path_with_a_folder_is_used_as_given(delta_svd, tmp_path, monkeypatch, capsys):
    skel, dwis = _two_sessions(tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "bvals").touch()
    monkeypatch.chdir(tmp_path)

    out = _resolved_inputs(delta_svd, monkeypatch, capsys, [
        "--dwi", *dwis, "--bval", "./shared/bvals", "--bvec", "bvecs", "--bmask", "mask",
        "--skeletonMask", str(skel), "--steps", "qc", "--qc", "0",
    ])
    assert out.count(f"bval  :{os.path.abspath('shared/bvals')}") == 2


def test_path_with_a_folder_is_not_re_anchored_to_the_dwi_folder(delta_svd, tmp_path, monkeypatch):
    # 'grad/bvals' exists only relative to the DWI folder, not to the working
    # directory; it is a path, so it is not looked for there
    skel, dwis = _two_sessions(tmp_path)
    for ses in ["ses-1", "ses-2"]:
        (tmp_path / ses / "grad").mkdir()
        (tmp_path / ses / "grad" / "bvals").touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "delta-svd.py", "--dwi", *dwis, "--bval", "grad/bvals", "--bvec", "bvecs",
        "--bmask", "mask", "--skeletonMask", str(skel), "--steps", "qc", "--qc", "0",
    ])
    with pytest.raises(delta_svd.DeltaSvdError, match="'--bval'"):
        delta_svd.pipeline_delta_svd()


def test_explicit_bmask_with_wrong_extension_names_the_sibling(delta_svd, tmp_path, monkeypatch):
    dwi = tmp_path / "sub01.nii.gz"
    skel = tmp_path / "skel.nii.gz"
    for fn in [dwi, skel, tmp_path / "sub01.bval", tmp_path / "sub01.bvec",
               tmp_path / "mask.nii.gz"]:
        fn.touch()
    monkeypatch.setattr(sys, "argv", [
        "delta-svd.py", "--dwi", str(dwi), "--bmask", "mask.nii",
        "--skeletonMask", str(skel), "--steps", "qc", "--qc", "0",
    ])
    with pytest.raises(delta_svd.DeltaSvdError) as excinfo:
        delta_svd.pipeline_delta_svd()

    msg = str(excinfo.value)
    assert "other NIfTI extension" in msg
    assert str(tmp_path / "mask.nii.gz") in msg


def test_missing_explicit_bmask_reports_it_as_given(delta_svd, tmp_path, monkeypatch):
    dwi = tmp_path / "sub01.nii"
    skel = tmp_path / "skel.nii.gz"
    for fn in [dwi, skel, tmp_path / "sub01.bval", tmp_path / "sub01.bvec"]:
        fn.touch()
    monkeypatch.setattr(sys, "argv", [
        "delta-svd.py", "--dwi", str(dwi), "--bmask", "nowhere.nii.gz",
        "--skeletonMask", str(skel), "--steps", "qc", "--qc", "0",
    ])
    with pytest.raises(delta_svd.DeltaSvdError) as excinfo:
        delta_svd.pipeline_delta_svd()

    msg = str(excinfo.value)
    assert "given as 'nowhere.nii.gz'" in msg
    assert "inferred" not in msg
    # a bare name is probed next to the DWI, and that is reported
    assert str(tmp_path / "nowhere.nii.gz") in msg


def test_missing_emask_names_the_option_and_the_NA_escape(delta_svd, tmp_path, monkeypatch):
    dwi = tmp_path / "sub01.nii"
    skel = tmp_path / "skel.nii.gz"
    for fn in [dwi, skel, tmp_path / "sub01.bval", tmp_path / "sub01.bvec",
               tmp_path / "sub01_brainmask.nii"]:
        fn.touch()
    monkeypatch.setattr(sys, "argv", [
        "delta-svd.py", "--dwi", str(dwi), "--Emask", "lesion.nii.gz",
        "--skeletonMask", str(skel), "--steps", "qc", "--qc", "0",
    ])
    # used to escape as a bare argparse.ArgumentTypeError, from outside argparse
    with pytest.raises(delta_svd.DeltaSvdError) as excinfo:
        delta_svd.pipeline_delta_svd()

    msg = str(excinfo.value)
    assert "'--Emask'" in msg
    assert "'NA'" in msg


@pytest.mark.parametrize("attr", ["Emask", "Rmask"])
def test_mask_given_without_extension_is_resolved_for_later_steps(delta_svd, tmp_path, monkeypatch, capsys, attr):
    # the resolved name used to be discarded, leaving the bare basename on
    # 'args' for nibabel to choke on much later
    dwi = tmp_path / "sub01.nii"
    skel = tmp_path / "skel.nii.gz"
    mask = tmp_path / "lesion.nii.gz"
    for fn in [dwi, skel, mask, tmp_path / "sub01.bval", tmp_path / "sub01.bvec",
               tmp_path / "sub01_brainmask.nii"]:
        fn.touch()
    if attr == "Rmask":
        nib.save(nib.Nifti1Image(np.zeros((1, 1, 1), dtype="uint8"), np.eye(4)), str(mask))

    out = _resolved_inputs(delta_svd, monkeypatch, capsys, [
        "--dwi", str(dwi), "--" + attr, "lesion", "--skeletonMask", str(skel),
        "--steps", "qc", "--qc", "0",
    ])
    assert f"{attr} :{mask}" in out


def test_bval_bvec_inferred_from_dwi_path(delta_svd, tmp_path, monkeypatch, capsys):
    dwi = tmp_path / "sub01.nii"
    skel = tmp_path / "skel.nii.gz"
    for fn in [dwi, skel, tmp_path / "sub01.bval", tmp_path / "sub01.bvec",
               tmp_path / "sub01_brainmask.nii"]:
        fn.touch()

    out = _resolved_inputs(delta_svd, monkeypatch, capsys, [
        "--dwi", str(dwi), "--skeletonMask", str(skel), "--steps", "qc", "--qc", "0",
    ])
    assert f"bval  :{tmp_path / 'sub01.bval'}" in out
    assert f"bvec  :{tmp_path / 'sub01.bvec'}" in out


def test_wrong_number_of_bvec_files_names_that_option(delta_svd, tmp_path, monkeypatch):
    # the count error used to report '--bval' regardless of which option was wrong
    skel = tmp_path / "skel.nii.gz"
    skel.touch()
    dwis = []
    for tp in ["tp1", "tp2"]:
        for suffix in [".nii.gz", ".bval", ".bvec", "_brainmask.nii.gz"]:
            (tmp_path / (tp + suffix)).touch()
        dwis.append(str(tmp_path / (tp + ".nii.gz")))
    monkeypatch.setattr(sys, "argv", [
        "delta-svd.py", "--dwi", *dwis, "--skeletonMask", str(skel),
        "--bvec", "tp1.bvec", "tp2.bvec", "tp1.bvec",
        "--steps", "qc", "--qc", "0",
    ])
    with pytest.raises(ValueError, match=r"option '--bvec'"):
        delta_svd.pipeline_delta_svd()


@pytest.mark.parametrize("option", ["--skeletonMask", "--RmaskMNI"])
def test_pipeline_rejects_an_off_grid_mask_before_processing(
        delta_svd, tmp_path, monkeypatch, _mni_reference, option):
    dwi, skel = _minimal_pipeline_inputs(tmp_path)
    skel = _save_grid(skel)
    offGrid = _save_grid(tmp_path / "off.nii.gz", shape=(91, 109, 91))
    argv = ["delta-svd.py", "--dwi", str(dwi), "--skeletonMask", skel]
    if option == "--skeletonMask":
        argv[-1] = offGrid
    else:
        argv += ["--RmaskMNI", offGrid]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(delta_svd.DeltaSvdError, match="image dimensions"):
        delta_svd.pipeline_delta_svd()
    assert not (tmp_path / "delta-svd_temp").exists()


@pytest.mark.parametrize("option", ["--bmask", "--Emask", "--Rmask"])
def test_pipeline_rejects_an_off_grid_dwi_mask_before_processing(
        delta_svd, tmp_path, monkeypatch, option):
    dwi = _save_dwi(tmp_path / "sub01.nii.gz")
    for fn in ["sub01.bval", "sub01.bvec"]:
        (tmp_path / fn).touch()
    _save_grid(tmp_path / "sub01_brainmask.nii.gz", (8, 9, 7), _DWI_AFFINE)
    offGrid = _save_grid(tmp_path / "off.nii.gz", (16, 18, 14), _DWI_AFFINE)
    skel = tmp_path / "skel.nii.gz"
    skel.touch()
    monkeypatch.setattr(sys, "argv", ["delta-svd.py", "--dwi", dwi, option, offGrid,
                                      "--skeletonMask", str(skel)])

    with pytest.raises(delta_svd.DeltaSvdError, match="image dimensions") as excinfo:
        delta_svd.pipeline_delta_svd()
    assert f"'{option}'" in str(excinfo.value)
    assert not (tmp_path / "delta-svd_temp").exists()


# ---------------------------------------------------------------------------
# ANTs' template construction uses its paths unquoted, so whitespace in the
# output path must never reach its command line.

def test_longitudinal_run_accepts_whitespace_in_the_output_path(delta_svd, tmp_path, monkeypatch):
    folder = tmp_path / "My Study"
    folder.mkdir()
    dwis = []
    for tp in ["tp1", "tp2"]:
        for suffix in [".nii.gz", ".bval", ".bvec", "_brainmask.nii.gz"]:
            (folder / (tp + suffix)).touch()
        dwis.append(str(folder / (tp + ".nii.gz")))
    skel = tmp_path / "skel.nii.gz"
    skel.touch()
    monkeypatch.setattr(sys, "argv", ["delta-svd.py", "--dwi", *dwis,
                                      "--skeletonMask", str(skel), "--steps", "qc", "--qc", "0"])
    # gets past the path handling to the contradictory-options error that follows it
    with pytest.raises(delta_svd.DeltaSvdError, match="contradictory"):
        delta_svd.pipeline_delta_svd()


def test_cross_sectional_run_accepts_whitespace_in_the_output_path(delta_svd, tmp_path, monkeypatch):
    folder = tmp_path / "My Study"
    folder.mkdir()
    for fn in ["tp1.nii.gz", "tp1.bval", "tp1.bvec", "tp1_brainmask.nii.gz"]:
        (folder / fn).touch()
    skel = tmp_path / "skel.nii.gz"
    skel.touch()
    monkeypatch.setattr(sys, "argv", ["delta-svd.py", "--dwi", str(folder / "tp1.nii.gz"),
                                      "--skeletonMask", str(skel), "--steps", "qc", "--qc", "0"])
    with pytest.raises(delta_svd.DeltaSvdError, match="contradictory"):
        delta_svd.pipeline_delta_svd()
