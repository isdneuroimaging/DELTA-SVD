import argparse
import sys

import nibabel as nib
import numpy as np
import pytest

import delta_svd_common
import delta_svd_inputs


# ---------------------------------------------------------------------------
# Argparse validators

def test_isNIfTI_accepts_existing_file_with_extension(tmp_path):
    fn = tmp_path / "sub01.nii.gz"
    fn.touch()
    assert delta_svd_inputs.isNIfTI(str(fn)) == str(fn)


def test_isNIfTI_appends_extension_if_missing(tmp_path):
    fn = tmp_path / "sub01.nii"
    fn.touch()
    base = str(tmp_path / "sub01")
    assert delta_svd_inputs.isNIfTI(base) == base + ".nii"


def test_isNIfTI_missing_file_aborts_by_default(tmp_path):
    with pytest.raises(argparse.ArgumentTypeError):
        delta_svd_inputs.isNIfTI(str(tmp_path / "missing"))


def test_isNIfTI_missing_file_returns_none_without_abort(tmp_path):
    assert delta_svd_inputs.isNIfTI(str(tmp_path / "missing"), abort=False) is None


def test_isCSV_accepts_overwrite_keyword():
    assert delta_svd_inputs.isCSV("overwrite") == "overwrite"


@pytest.mark.parametrize("name", ["results.csv", "results.CSV"])
def test_isCSV_accepts_csv_extension(name):
    assert delta_svd_inputs.isCSV(name) == name


def test_isCSV_rejects_other_extensions():
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
        delta_svd_inputs.isCSV("results.txt")
    # every other type= validator names its own option; this one used to be the
    # odd one out, leaving the reader to guess which flag it meant
    assert "--reprocess" in str(excinfo.value)


@pytest.mark.parametrize("name", ["../results.csv", "./results.csv", "/tmp/results.csv",
                                         r"..\results.csv"])
def test_isCSV_rejects_paths_instead_of_base_names(name):
    with pytest.raises(argparse.ArgumentTypeError, match="base name"):
        delta_svd_inputs.isCSV(name)


def test_assertPositiveJobs_accepts_values_at_or_above_one():
    assert delta_svd_inputs.assertPositiveJobs("1") == 1
    assert delta_svd_inputs.assertPositiveJobs("5") == 5


def test_assertPositiveJobs_below_one_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        delta_svd_inputs.assertPositiveJobs("0")


def test_assertPositiveJobs_non_integer_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        delta_svd_inputs.assertPositiveJobs("two")


def test_threadBudget_accepts_auto():
    assert delta_svd_inputs.threadBudget("auto") == "auto"
    assert delta_svd_inputs.threadBudget("AUTO") == "auto"


def test_threadBudget_accepts_positive_integer():
    assert delta_svd_inputs.threadBudget("8") == 8


def test_threadBudget_rejects_invalid():
    with pytest.raises(argparse.ArgumentTypeError):
        delta_svd_inputs.threadBudget("0")
    with pytest.raises(argparse.ArgumentTypeError):
        delta_svd_inputs.threadBudget("-1")
    with pytest.raises(argparse.ArgumentTypeError):
        delta_svd_inputs.threadBudget("lots")


def test_assertPositiveItkThreads_rejects_non_positive_and_non_integer():
    assert delta_svd_inputs.assertPositiveItkThreads("12") == 12
    for bad in ("0", "-1", "auto", "1.5"):
        with pytest.raises(argparse.ArgumentTypeError):
            delta_svd_inputs.assertPositiveItkThreads(bad)


# ---------------------------------------------------------------------------
# CustomArgumentParser single-dash rule

def _minimal_parser():
    parser = delta_svd_inputs.CustomArgumentParser()
    parser.add_argument("-o", "--dirOutput")
    return parser


def test_custom_parser_allows_single_dash_with_space():
    parser = _minimal_parser()
    ns = parser.parse_args(["-o", "foo"])
    assert ns.dirOutput == "foo"


def test_custom_parser_rejects_attached_single_dash_argument(capsys):
    parser = _minimal_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["-ofoo"])
    assert excinfo.value.code == 2
    assert "-ofoo" in capsys.readouterr().err


def test_custom_parser_allows_double_dash_attached_form():
    parser = _minimal_parser()
    ns = parser.parse_args(["--dirOutput=foo"])
    assert ns.dirOutput == "foo"


def test_custom_parser_falls_back_to_the_command_line(monkeypatch):
    # the argparse default: parse_args() with no argument reads sys.argv[1:],
    # which the single-dash check used to iterate as None
    parser = _minimal_parser()
    monkeypatch.setattr(sys, "argv", ["delta-svd.py", "-o", "foo"])

    assert parser.parse_args().dirOutput == "foo"


def test_custom_parser_suggests_the_double_dash_form_when_known(capsys):
    # the likeliest real-world case: '--shells' mistyped with one dash
    parser = _minimal_parser()
    parser.add_argument("--shells", nargs="+")
    with pytest.raises(SystemExit):
        parser.parse_args(["-shells", "700"])
    assert 'Did you mean "--shells"?' in capsys.readouterr().err


def test_custom_parser_suggests_the_double_dash_form_with_attached_value(capsys):
    # '=' attaches a value to a long option; the suggestion must strip it first
    parser = _minimal_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["-dirOutput=foo"])
    assert 'Did you mean "--dirOutput"?' in capsys.readouterr().err


def test_custom_parser_omits_suggestion_for_unknown_options(capsys):
    # a typo that doesn't match any registered option gets the plain message
    parser = _minimal_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["-bogus"])
    assert "Did you mean" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# iniParser() defaults

def test_iniparser_defaults():
    parser = delta_svd_inputs.iniParser()
    assert parser.get_default("qc") == 1
    assert parser.get_default("para") is None
    assert parser.get_default("threads") == "auto"
    assert parser.get_default("itkThreads") == delta_svd_inputs.ITK_THREADS_DEFAULT
    assert parser.get_default("bRange") == [800, 1200]
    assert parser.get_default("shells") is None
    assert parser.get_default("skeletonMask") == "/opt/scripts/delta-svd_skeletonmask_v1.nii.gz"
    assert parser.get_default("adjustBmaskForFW") is True
    assert parser.get_default("debug") is False
    assert parser.get_default("hemispheres") is False


# Anything that actually parses arguments has to pass an explicit
# '--skeletonMask': argparse runs string defaults through their 'type=', and the
# default here is the in-container path '/opt/scripts/delta-svd_skeletonmask_v1.nii.gz',
# which isNIfTI() rejects outside the image. Defaults are asserted via
# parser.get_default() instead (see test_iniparser_defaults above).
def _touch_dwi_and_skeleton(tmp_path):
    dwi = tmp_path / "sub01.nii.gz"
    skel = tmp_path / "skel.nii.gz"
    dwi.touch()
    skel.touch()
    return dwi, skel


def test_iniparser_para_one_is_accepted_by_cli(tmp_path):
    # '--para 1' is now valid (serial registration); only non-positive is rejected
    dwi, skel = _touch_dwi_and_skeleton(tmp_path)
    parser = delta_svd_inputs.iniParser()
    ns = parser.parse_args(["--dwi", str(dwi), "--skeletonMask", str(skel), "--para", "1"])
    assert ns.para == 1


def test_iniparser_para_zero_is_rejected_by_cli(tmp_path, capsys):
    dwi, skel = _touch_dwi_and_skeleton(tmp_path)
    parser = delta_svd_inputs.iniParser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--dwi", str(dwi), "--skeletonMask", str(skel), "--para", "0"])
    # assert the exit really came from '--para', not from some other argument
    assert "--para" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# assertBValue(): the window the diffusion tensor model is valid in
#
# Outside it the tensor is not interpretable - below the floor the signal is
# contaminated by perfusion, above the ceiling by non-Gaussian diffusion - so
# such a request is refused before anything runs rather than fitted.

@pytest.mark.parametrize("value", ["250", "1000", "1800"])
def test_assertBValue_accepts_the_valid_window(value):
    assert delta_svd_inputs.assertBValue(value) == int(value)


@pytest.mark.parametrize("value", ["249", "1801", "0", "-100"])
def test_assertBValue_rejects_b_values_outside_the_window(value):
    with pytest.raises(argparse.ArgumentTypeError, match="have to lie between 250 and 1800"):
        delta_svd_inputs.assertBValue(value)


def test_assertBValue_points_a_zero_at_the_b0_rule():
    # the mistake this catches: passing 0 as the lower limit to 'keep the b0s',
    # which they never needed - it only drags every shell below the upper limit
    # into the fit
    with pytest.raises(argparse.ArgumentTypeError, match="always included"):
        delta_svd_inputs.assertBValue("0")


def test_assertBValue_rejects_non_numeric():
    with pytest.raises(argparse.ArgumentTypeError, match="whole numbers"):
        delta_svd_inputs.assertBValue("1000.5")


def test_iniparser_shells_accepts_several_shells(tmp_path):
    dwi, skel = _touch_dwi_and_skeleton(tmp_path)
    parser = delta_svd_inputs.iniParser()
    ns = parser.parse_args(["--dwi", str(dwi), "--skeletonMask", str(skel),
                            "--shells", "700", "1000", "1500"])
    assert ns.shells == [700, 1000, 1500]


def test_iniparser_rejects_brange_and_shells_together(tmp_path, capsys):
    # they answer the same question in two ways; taking both would leave it
    # undefined which one selected the data
    dwi, skel = _touch_dwi_and_skeleton(tmp_path)
    parser = delta_svd_inputs.iniParser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--dwi", str(dwi), "--skeletonMask", str(skel),
                           "--bRange", "800", "1200", "--shells", "1000"])
    assert "not allowed with argument" in capsys.readouterr().err


def test_iniparser_brange_outside_the_valid_window_is_rejected_by_cli(tmp_path, capsys):
    dwi, skel = _touch_dwi_and_skeleton(tmp_path)
    parser = delta_svd_inputs.iniParser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--dwi", str(dwi), "--skeletonMask", str(skel),
                           "--bRange", "800", "2500"])
    assert "--bRange" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("label", "message"),
    [("../other", "path traversal"), ("visit/one", "path separator"),
     (r"visit\one", "path separator"), ("template", "internal working directory"),
     ("intermediateTemplates", "internal working directory"),
     ("TBSS", "internal working directory"),
     # FSL's tbss_non_FA and ANTs' template script split names at whitespace
     ("visit one", "whitespace"), ("visit\tone", "whitespace")],
)
def test_rejects_unsafe_timepoint_labels(label, message):
    with pytest.raises(ValueError, match=message):
        delta_svd_inputs.validate_timepoint_labels([label])


@pytest.mark.parametrize("label", ["TP01", "ses-1", "visit.1"])
def test_accepts_plain_timepoint_labels(label):
    delta_svd_inputs.validate_timepoint_labels([label])


# ---------------------------------------------------------------------------
# --numRegistrations reaches an ANTs shell command, so it is validated like the
# other numeric options rather than accepting any int.

def test_numRegistrations_accepts_positive_integer():
    assert delta_svd_inputs.assertPositiveRegistrations("5") == 5


@pytest.mark.parametrize("bad", ["0", "-2", "many"])
def test_numRegistrations_rejects_non_positive_and_non_integer(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        delta_svd_inputs.assertPositiveRegistrations(bad)


def test_candidate_paths_anchors_only_a_bare_name():
    dwi = "study/sub01/data.nii.gz"
    # inferred: already carries the folder, so it is probed as-is only
    assert delta_svd_inputs.candidate_paths("study/sub01/data_brainmask", dwi, True) == [
        "study/sub01/data_brainmask.nii.gz", "study/sub01/data_brainmask.nii"]
    # a bare name is probed next to the DWI only, never in the working directory
    assert delta_svd_inputs.candidate_paths("m.nii.gz", dwi, True) == ["study/sub01/m.nii.gz"]
    # a name carrying a folder is probed as given only
    assert delta_svd_inputs.candidate_paths("./m.nii.gz", dwi, True) == ["./m.nii.gz"]


def test_per_timepoint_path():
    dwi = "/data/ses-2/data.nii.gz"
    assert delta_svd_inputs.per_timepoint_path("bvals", dwi) == "/data/ses-2/bvals"
    assert delta_svd_inputs.per_timepoint_path("./bvals", dwi) == "./bvals"
    assert delta_svd_inputs.per_timepoint_path("ses-1/bvals", dwi) == "ses-1/bvals"
    assert delta_svd_inputs.per_timepoint_path("/abs/bvals", dwi) == "/abs/bvals"
    # a DWI given as a bare name lives in the working directory, so does its file
    assert delta_svd_inputs.per_timepoint_path("bvals", "data.nii.gz") == "bvals"


# ---------------------------------------------------------------------------
# The extension probing appends to the name as given, so for 'data.nii' it looks
# for 'data.nii.nii.gz' -- it never reaches 'data.nii.gz'. The message must not
# claim otherwise, and the sibling that is actually there is worth naming.

def test_isNIfTI_names_the_sibling_with_the_other_extension(tmp_path):
    (tmp_path / "data.nii.gz").touch()
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
        delta_svd_inputs.isNIfTI(str(tmp_path / "data.nii"))

    msg = str(excinfo.value)
    assert "also tried" not in msg
    assert str(tmp_path / "data.nii.gz") in msg


def test_isNIfTI_names_the_sibling_in_the_other_direction(tmp_path):
    (tmp_path / "data.nii").touch()
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
        delta_svd_inputs.isNIfTI(str(tmp_path / "data.nii.gz"))

    assert str(tmp_path / "data.nii") in str(excinfo.value)


def test_isNIfTI_does_not_claim_probes_it_did_not_make(tmp_path):
    # nothing there at all, and the name already carries an extension
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
        delta_svd_inputs.isNIfTI(str(tmp_path / "data.nii"))

    assert "also tried" not in str(excinfo.value)


def test_isNIfTI_still_reports_the_probes_for_an_extensionless_name(tmp_path):
    # here the probes are real, and naming them tells the user what was searched
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
        delta_svd_inputs.isNIfTI(str(tmp_path / "data"))

    msg = str(excinfo.value)
    assert str(tmp_path / "data.nii.gz") in msg
    assert str(tmp_path / "data.nii") in msg


def test_nifti_sibling_returns_none_without_a_counterpart(tmp_path):
    (tmp_path / "data.nii.gz").touch()
    assert delta_svd_inputs.nifti_sibling(str(tmp_path / "other.nii")) is None
    assert delta_svd_inputs.nifti_sibling(str(tmp_path / "notes.txt")) is None
    # the file asked for is never substituted, only reported
    assert delta_svd_inputs.nifti_sibling(str(tmp_path / "data.nii")) == str(tmp_path / "data.nii.gz")


# ---------------------------------------------------------------------------
# check_mni_grid(). The reference is a stand-in with the shipped mask's header,
# since '/opt/scripts' only exists in the image.

_MNI_AFFINE = np.array([[-1., 0, 0, 90], [0, 1, 0, -126], [0, 0, 1, -72], [0, 0, 0, 1]])
_MNI_SHAPE = (182, 218, 182)


def _save_grid(path, shape=_MNI_SHAPE, affine=_MNI_AFFINE):
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype="uint8"), affine), str(path))
    return str(path)


@pytest.fixture
def _mni_reference(tmp_path, monkeypatch):
    ref = _save_grid(tmp_path / "reference.nii.gz")
    monkeypatch.setattr(delta_svd_inputs, "SKELETON_MASK_DEFAULT", ref)
    return ref


def test_mni_reference_matches_the_shipped_skeleton_mask():
    from conftest import CONTAINER_FILES
    img = nib.load(str(CONTAINER_FILES / "delta-svd_skeletonmask_v1.nii.gz"))
    assert img.shape == _MNI_SHAPE
    assert np.allclose(img.affine, _MNI_AFFINE)


def test_check_mni_grid_accepts_a_matching_grid(tmp_path, _mni_reference):
    affine = _MNI_AFFINE.copy()
    affine[0, 3] += 1e-6                          # float32 header round-off
    delta_svd_inputs.check_mni_grid(_save_grid(tmp_path / "mask.nii.gz", affine=affine), "skeleton mask")


def test_check_mni_grid_is_a_no_op_for_the_default_mask(tmp_path, monkeypatch):
    # the default path is never even opened
    missing = str(tmp_path / "not-there.nii.gz")
    monkeypatch.setattr(delta_svd_inputs, "SKELETON_MASK_DEFAULT", missing)
    delta_svd_inputs.check_mni_grid(missing, "skeleton mask")


def test_check_mni_grid_rejects_a_flipped_x_axis(tmp_path, _mni_reference):
    affine = _MNI_AFFINE.copy()
    affine[0, 0], affine[0, 3] = 1, -91           # RAS storage of the same space
    fn = _save_grid(tmp_path / "mask.nii.gz", affine=affine)
    with pytest.raises(delta_svd_common.DeltaSvdError, match="affine") as excinfo:
        delta_svd_inputs.check_mni_grid(fn, "skeleton mask")
    assert fn in str(excinfo.value)


def test_check_mni_grid_rejects_a_wrong_shape(tmp_path, _mni_reference):
    fn = _save_grid(tmp_path / "mask.nii.gz", shape=(91, 109, 91))
    with pytest.raises(delta_svd_common.DeltaSvdError, match="image dimensions") as excinfo:
        delta_svd_inputs.check_mni_grid(fn, "ROI mask in MNI space")
    assert "ROI mask in MNI space" in str(excinfo.value)
    assert fn in str(excinfo.value)


# ---------------------------------------------------------------------------
# check_dwi_grid(): masks given in DWI space have to be on their DWI's grid.

_DWI_AFFINE = np.diag([2.0, 2.0, 2.5, 1.0])
_DWI_AFFINE[:3, 3] = [-100.0, -120.0, -60.0]


def _save_dwi(path, shape=(8, 9, 7, 3), affine=_DWI_AFFINE):
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype="float32"), affine), str(path))
    return str(path)


def test_check_dwi_grid_accepts_a_matching_grid(tmp_path):
    dwi = _save_dwi(tmp_path / "dwi.nii.gz")
    affine = _DWI_AFFINE.copy()
    affine[0, 3] += 1e-6                          # float32 header round-off
    delta_svd_inputs.check_dwi_grid(_save_grid(tmp_path / "mask.nii.gz", (8, 9, 7), affine),
                             "brain mask", "bmask", dwi)


def test_check_dwi_grid_rejects_a_wrong_shape(tmp_path):
    dwi = _save_dwi(tmp_path / "dwi.nii.gz")
    fn = _save_grid(tmp_path / "lesion.nii.gz", (16, 18, 14), _DWI_AFFINE)
    with pytest.raises(delta_svd_common.DeltaSvdError, match="image dimensions") as excinfo:
        delta_svd_inputs.check_dwi_grid(fn, "exclusion mask", "Emask", dwi)
    msg = str(excinfo.value)
    assert fn in msg and dwi in msg and "'--Emask'" in msg


def test_check_dwi_grid_rejects_a_shifted_affine(tmp_path):
    dwi = _save_dwi(tmp_path / "dwi.nii.gz")
    affine = _DWI_AFFINE.copy()
    affine[2, 3] += 2.5                           # one slice off
    fn = _save_grid(tmp_path / "roi.nii.gz", (8, 9, 7), affine)
    with pytest.raises(delta_svd_common.DeltaSvdError, match="affine"):
        delta_svd_inputs.check_dwi_grid(fn, "ROI mask in DWI space", "Rmask", dwi)
