import argparse
import io
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import decompose_tensor, design_matrix, from_lower_triangular
import dipy.reconst.dti as dti


# ---------------------------------------------------------------------------
# Argparse validators

def test_isNIfTI_accepts_existing_file_with_extension(delta_svd, tmp_path):
    fn = tmp_path / "sub01.nii.gz"
    fn.touch()
    assert delta_svd.isNIfTI(str(fn)) == str(fn)


def test_isNIfTI_appends_extension_if_missing(delta_svd, tmp_path):
    fn = tmp_path / "sub01.nii"
    fn.touch()
    base = str(tmp_path / "sub01")
    assert delta_svd.isNIfTI(base) == base + ".nii"


def test_isNIfTI_missing_file_aborts_by_default(delta_svd, tmp_path):
    with pytest.raises(argparse.ArgumentTypeError):
        delta_svd.isNIfTI(str(tmp_path / "missing"))


def test_isNIfTI_missing_file_returns_none_without_abort(delta_svd, tmp_path):
    assert delta_svd.isNIfTI(str(tmp_path / "missing"), abort=False) is None


def test_isCSV_accepts_overwrite_keyword(delta_svd):
    assert delta_svd.isCSV("overwrite") == "overwrite"


@pytest.mark.parametrize("name", ["results.csv", "results.CSV"])
def test_isCSV_accepts_csv_extension(delta_svd, name):
    assert delta_svd.isCSV(name) == name


def test_isCSV_rejects_other_extensions(delta_svd):
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
        delta_svd.isCSV("results.txt")
    # every other type= validator names its own option; this one used to be the
    # odd one out, leaving the reader to guess which flag it meant
    assert "--reprocess" in str(excinfo.value)


def test_assertPositiveJobs_accepts_values_at_or_above_one(delta_svd):
    assert delta_svd.assertPositiveJobs("1") == 1
    assert delta_svd.assertPositiveJobs("5") == 5


def test_assertPositiveJobs_below_one_raises(delta_svd):
    with pytest.raises(argparse.ArgumentTypeError):
        delta_svd.assertPositiveJobs("0")


def test_assertPositiveJobs_non_integer_raises(delta_svd):
    with pytest.raises(argparse.ArgumentTypeError):
        delta_svd.assertPositiveJobs("two")


def test_threadBudget_accepts_auto(delta_svd):
    assert delta_svd.threadBudget("auto") == "auto"
    assert delta_svd.threadBudget("AUTO") == "auto"


def test_threadBudget_accepts_positive_integer(delta_svd):
    assert delta_svd.threadBudget("8") == 8


def test_threadBudget_rejects_invalid(delta_svd):
    with pytest.raises(argparse.ArgumentTypeError):
        delta_svd.threadBudget("0")
    with pytest.raises(argparse.ArgumentTypeError):
        delta_svd.threadBudget("-1")
    with pytest.raises(argparse.ArgumentTypeError):
        delta_svd.threadBudget("lots")


# ---------------------------------------------------------------------------
# ANTs parallelism planner

def test_plan_ants_parallelism_itk_threads_never_vary(delta_svd):
    # The regression guard for the whole planner: the ITK thread count is the one
    # quantity here that moves the metric values, so it must depend on nothing but
    # --itkThreads. Deriving it from the core budget is what previously made
    # results differ between machines and between subjects with different numbers
    # of timepoints.
    varied = [
        (nTP, budget, para, delta_svd.plan_ants_parallelism(nTP, budget, para)[1])
        for nTP in (1, 2, 3, 4, 6, 12)
        for budget in (1, 2, 8, 12, 13, 16, 24, 32, 36, 48, 64, 96)
        for para in (None, 1, 2, 3, 12)
    ]
    assert [v for v in varied if v[3] != delta_svd.ITK_THREADS_DEFAULT] == []


def test_plan_ants_parallelism_fills_the_budget(delta_svd):
    # 'para' spends whatever cores are there (ceil), but never oversubscribes past
    # 1.5 threads per core -- which is why 13 cores stays at one job rather than
    # jumping to two (24 threads on 13 cores, 1.85x).
    expected = {12: 1, 13: 1, 16: 2, 24: 2, 32: 3, 36: 3, 48: 4, 64: 6}
    got = {b: delta_svd.plan_ants_parallelism(99, b)[0] for b in expected}
    assert got == expected


def test_plan_ants_parallelism_single_job_is_serial(delta_svd):
    # one job -> serial path ('-c 0'), never '-c 2 -j 1' (pexec aborts on 1 job).
    # The branch selects the execution mode only: itk is unchanged either way.
    para, itk, control = delta_svd.plan_ants_parallelism(2, 1)
    assert (para, itk, control) == (1, delta_svd.ITK_THREADS_DEFAULT, "-c 0")


def test_plan_ants_parallelism_caps_jobs_at_timepoints(delta_svd):
    # 2 timepoints -> at most 2 jobs however large the budget; there is no third
    # registration to run in parallel
    para, itk, control = delta_svd.plan_ants_parallelism(2, 64)
    assert (para, itk) == (2, delta_svd.ITK_THREADS_DEFAULT)
    assert control == "-c 2 -j 2"


def test_plan_ants_parallelism_override_is_not_clamped_to_the_budget(delta_svd):
    # explicit --para 1 forces the serial path
    assert delta_svd.plan_ants_parallelism(6, 12, paraOverride=1)[2] == "-c 0"
    # an override above the budget is honoured (deliberate oversubscription is a
    # valid choice and costs only runtime), but never exceeds the timepoint count
    para, _, control = delta_svd.plan_ants_parallelism(3, 4, paraOverride=3)
    assert para == 3 and control == "-c 2 -j 3"
    assert delta_svd.plan_ants_parallelism(3, 4, paraOverride=99)[0] == 3


def test_plan_ants_parallelism_honours_an_explicit_itk_thread_count(delta_svd):
    para, itk, _ = delta_svd.plan_ants_parallelism(4, 24, itkThreads=6)
    assert itk == 6
    assert para == 4          # 24 cores / 6 threads = 4 jobs, capped by 4 timepoints


# ---------------------------------------------------------------------------
# --itkThreads resolution and export

def test_resolve_itk_threads_defaults_when_absent(delta_svd):
    assert delta_svd.resolve_itk_threads([]) == delta_svd.ITK_THREADS_DEFAULT
    assert delta_svd.resolve_itk_threads(["--dwi", "a.nii.gz"]) == delta_svd.ITK_THREADS_DEFAULT


def test_resolve_itk_threads_reads_an_explicit_value(delta_svd):
    assert delta_svd.resolve_itk_threads(["--itkThreads", "6"]) == 6
    assert delta_svd.resolve_itk_threads(["--dwi", "a.nii.gz", "--itkThreads", "1"]) == 1


def test_resolve_itk_threads_falls_back_on_garbage(delta_svd):
    # the full parser reports the error properly; this pre-pass must not crash
    # before numpy is even imported
    assert delta_svd.resolve_itk_threads(["--itkThreads", "many"]) == delta_svd.ITK_THREADS_DEFAULT


def test_blas_thread_pools_are_pinned_to_one(delta_svd):
    # no thread count in the numerics may vary with the machine; assigned, not
    # setdefault, so a host-forwarded value cannot reach BLAS either
    assert delta_svd.BLAS_THREADS == 1
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        assert os.environ[var] == "1", var


def test_itk_thread_count_is_exported_to_every_ants_call(delta_svd):
    # assigned, not setdefault: a value forwarded in from the host must not be
    # able to change the metrics (Apptainer passes the host environment through)
    assert os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] == str(delta_svd.ITK_THREADS)


def test_cpu_check_passes_when_the_required_flags_are_present(delta_svd, tmp_path):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("flags : sse2 avx avx2 fma bmi2\n")
    assert delta_svd.check_cpu_supports_coretype("Haswell", str(cpuinfo)) == ()


def test_cpu_check_names_every_missing_flag(delta_svd, tmp_path):
    # OpenBLAS would die with SIGILL and no message; the check exists to say why
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("flags : sse2 sse4_2\n")
    assert set(delta_svd.check_cpu_supports_coretype("Haswell", str(cpuinfo))) == {"avx2", "fma"}


def test_cpu_check_is_silent_when_the_kernel_is_not_pinned(delta_svd, tmp_path):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("flags : sse2\n")
    assert delta_svd.check_cpu_supports_coretype("", str(cpuinfo)) == ()
    assert delta_svd.check_cpu_supports_coretype("SomeFutureKernel", str(cpuinfo)) == ()


def test_cpu_check_does_not_fail_without_proc_cpuinfo(delta_svd, tmp_path):
    # non-Linux hosts have no /proc/cpuinfo; leave the decision to OpenBLAS
    assert delta_svd.check_cpu_supports_coretype("Haswell", str(tmp_path / "absent")) == ()


def test_assertPositiveItkThreads_rejects_non_positive_and_non_integer(delta_svd):
    assert delta_svd.assertPositiveItkThreads("12") == 12
    for bad in ("0", "-1", "auto", "1.5"):
        with pytest.raises(argparse.ArgumentTypeError):
            delta_svd.assertPositiveItkThreads(bad)


def test_detect_physical_cores_returns_positive_int(delta_svd):
    n = delta_svd.detect_physical_cores()
    assert isinstance(n, int) and n >= 1


# ---------------------------------------------------------------------------
# resolve_thread_budget() / detect_physical_cores()
#
# These run at import time, *before* the real parser and before numpy is
# imported, to size the BLAS/OpenMP pools. So they must never raise on input the
# real parser is supposed to reject, and never choke on the rest of the command
# line - a traceback here kills the run before argparse can print a usage error.

def test_resolve_thread_budget_autodetects_when_threads_absent(delta_svd, monkeypatch):
    monkeypatch.setattr(delta_svd, "detect_physical_cores", lambda: 99)
    assert delta_svd.resolve_thread_budget([]) == 99


@pytest.mark.parametrize("value", ["auto", "AUTO", " auto "])
def test_resolve_thread_budget_autodetects_for_auto(delta_svd, monkeypatch, value):
    monkeypatch.setattr(delta_svd, "detect_physical_cores", lambda: 99)
    assert delta_svd.resolve_thread_budget(["--threads", value]) == 99


def test_resolve_thread_budget_uses_explicit_integer(delta_svd, monkeypatch):
    monkeypatch.setattr(delta_svd, "detect_physical_cores", lambda: 99)
    assert delta_svd.resolve_thread_budget(["--threads", "8"]) == 8
    assert delta_svd.resolve_thread_budget(["--threads", " 8 "]) == 8


def test_resolve_thread_budget_clamps_negative_to_one(delta_svd, monkeypatch):
    monkeypatch.setattr(delta_svd, "detect_physical_cores", lambda: 99)
    assert delta_svd.resolve_thread_budget(["--threads", "-4"]) == 1


def test_resolve_thread_budget_falls_back_on_garbage(delta_svd, monkeypatch):
    # 'lots' is rejected later by threadBudget(); here it must only fall back,
    # because raising would abort before argparse can report the real error.
    monkeypatch.setattr(delta_svd, "detect_physical_cores", lambda: 99)
    assert delta_svd.resolve_thread_budget(["--threads", "lots"]) == 99


def test_resolve_thread_budget_ignores_the_rest_of_the_command_line(delta_svd, monkeypatch):
    monkeypatch.setattr(delta_svd, "detect_physical_cores", lambda: 99)
    argv = ["--dwi", "sub01.nii.gz", "--threads", "4", "--steps", "1", "2", "-o", "/out"]
    assert delta_svd.resolve_thread_budget(argv) == 4


def _fake_cpuinfo(monkeypatch, text, allowed):
    """Serve 'text' as /proc/cpuinfo and pin CPU affinity to 'allowed', so the
    Linux-only topology path can be exercised on any platform."""
    realOpen = io.open
    def fakeOpen(fn, *a, **kw):
        if str(fn) == "/proc/cpuinfo":
            if text is None:
                raise OSError("no such file")
            return io.StringIO(text)
        return realOpen(fn, *a, **kw)
    monkeypatch.setattr("builtins.open", fakeOpen)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(allowed), raising=False)


# Four logical CPUs, two physical cores: hyperthread siblings share a
# (physical id, core id) pair. Counting logical CPUs here would double the ANTs
# job count and thrash a machine that gains nothing from SMT.
_CPUINFO_2CORE_SMT = """processor : 0
physical id : 0
core id : 0

processor : 1
physical id : 0
core id : 1

processor : 2
physical id : 0
core id : 0

processor : 3
physical id : 0
core id : 1

"""


def test_detect_physical_cores_ignores_hyperthread_siblings(delta_svd, monkeypatch):
    _fake_cpuinfo(monkeypatch, _CPUINFO_2CORE_SMT, allowed=[0, 1, 2, 3])
    assert delta_svd.detect_physical_cores() == 2


def test_detect_physical_cores_honours_cpu_affinity(delta_svd, monkeypatch):
    # A scheduler cpuset handing us CPUs 0 and 2 - the two siblings of one core -
    # is a budget of one core, not two.
    _fake_cpuinfo(monkeypatch, _CPUINFO_2CORE_SMT, allowed=[0, 2])
    assert delta_svd.detect_physical_cores() == 1


def test_detect_physical_cores_falls_back_to_affinity_size(delta_svd, monkeypatch):
    # No /proc/cpuinfo (non-Linux, or a locked-down container): assume no SMT
    # rather than crashing at import time.
    _fake_cpuinfo(monkeypatch, None, allowed=[0, 1, 2])
    assert delta_svd.detect_physical_cores() == 3


# ---------------------------------------------------------------------------
# CustomArgumentParser single-dash rule

def _minimal_parser(delta_svd):
    parser = delta_svd.CustomArgumentParser()
    parser.add_argument("-o", "--dirOutput")
    return parser


def test_custom_parser_allows_single_dash_with_space(delta_svd):
    parser = _minimal_parser(delta_svd)
    ns = parser.parse_args(["-o", "foo"])
    assert ns.dirOutput == "foo"


def test_custom_parser_rejects_attached_single_dash_argument(delta_svd, capsys):
    parser = _minimal_parser(delta_svd)
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["-ofoo"])
    assert excinfo.value.code == 2
    assert "-ofoo" in capsys.readouterr().err


def test_custom_parser_allows_double_dash_attached_form(delta_svd):
    parser = _minimal_parser(delta_svd)
    ns = parser.parse_args(["--dirOutput=foo"])
    assert ns.dirOutput == "foo"


def test_custom_parser_falls_back_to_the_command_line(delta_svd, monkeypatch):
    # the argparse default: parse_args() with no argument reads sys.argv[1:],
    # which the single-dash check used to iterate as None
    parser = _minimal_parser(delta_svd)
    monkeypatch.setattr(sys, "argv", ["delta-svd.py", "-o", "foo"])

    assert parser.parse_args().dirOutput == "foo"


def test_custom_parser_suggests_the_double_dash_form_when_known(delta_svd, capsys):
    # the likeliest real-world case: '--shells' mistyped with one dash
    parser = _minimal_parser(delta_svd)
    parser.add_argument("--shells", nargs="+")
    with pytest.raises(SystemExit):
        parser.parse_args(["-shells", "700"])
    assert 'Did you mean "--shells"?' in capsys.readouterr().err


def test_custom_parser_suggests_the_double_dash_form_with_attached_value(delta_svd, capsys):
    # '=' attaches a value to a long option; the suggestion must strip it first
    parser = _minimal_parser(delta_svd)
    with pytest.raises(SystemExit):
        parser.parse_args(["-dirOutput=foo"])
    assert 'Did you mean "--dirOutput"?' in capsys.readouterr().err


def test_custom_parser_omits_suggestion_for_unknown_options(delta_svd, capsys):
    # a typo that doesn't match any registered option gets the plain message
    parser = _minimal_parser(delta_svd)
    with pytest.raises(SystemExit):
        parser.parse_args(["-bogus"])
    assert "Did you mean" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# iniParser() defaults

def test_iniparser_defaults(delta_svd):
    parser = delta_svd.iniParser()
    assert parser.get_default("qc") == 1
    assert parser.get_default("para") is None
    assert parser.get_default("threads") == "auto"
    assert parser.get_default("itkThreads") == delta_svd.ITK_THREADS_DEFAULT
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


def test_iniparser_para_one_is_accepted_by_cli(delta_svd, tmp_path):
    # '--para 1' is now valid (serial registration); only non-positive is rejected
    dwi, skel = _touch_dwi_and_skeleton(tmp_path)
    parser = delta_svd.iniParser()
    ns = parser.parse_args(["--dwi", str(dwi), "--skeletonMask", str(skel), "--para", "1"])
    assert ns.para == 1


def test_iniparser_para_zero_is_rejected_by_cli(delta_svd, tmp_path, capsys):
    dwi, skel = _touch_dwi_and_skeleton(tmp_path)
    parser = delta_svd.iniParser()
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
def test_assertBValue_accepts_the_valid_window(delta_svd, value):
    assert delta_svd.assertBValue(value) == int(value)


@pytest.mark.parametrize("value", ["249", "1801", "0", "-100"])
def test_assertBValue_rejects_b_values_outside_the_window(delta_svd, value):
    with pytest.raises(argparse.ArgumentTypeError, match="have to lie between 250 and 1800"):
        delta_svd.assertBValue(value)


def test_assertBValue_points_a_zero_at_the_b0_rule(delta_svd):
    # the mistake this catches: passing 0 as the lower limit to 'keep the b0s',
    # which they never needed - it only drags every shell below the upper limit
    # into the fit
    with pytest.raises(argparse.ArgumentTypeError, match="always included"):
        delta_svd.assertBValue("0")


def test_assertBValue_rejects_non_numeric(delta_svd):
    with pytest.raises(argparse.ArgumentTypeError, match="whole numbers"):
        delta_svd.assertBValue("1000.5")


def test_iniparser_shells_accepts_several_shells(delta_svd, tmp_path):
    dwi, skel = _touch_dwi_and_skeleton(tmp_path)
    parser = delta_svd.iniParser()
    ns = parser.parse_args(["--dwi", str(dwi), "--skeletonMask", str(skel),
                            "--shells", "700", "1000", "1500"])
    assert ns.shells == [700, 1000, 1500]


def test_iniparser_rejects_brange_and_shells_together(delta_svd, tmp_path, capsys):
    # they answer the same question in two ways; taking both would leave it
    # undefined which one selected the data
    dwi, skel = _touch_dwi_and_skeleton(tmp_path)
    parser = delta_svd.iniParser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--dwi", str(dwi), "--skeletonMask", str(skel),
                           "--bRange", "800", "1200", "--shells", "1000"])
    assert "not allowed with argument" in capsys.readouterr().err


def test_iniparser_brange_outside_the_valid_window_is_rejected_by_cli(delta_svd, tmp_path, capsys):
    dwi, skel = _touch_dwi_and_skeleton(tmp_path)
    parser = delta_svd.iniParser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--dwi", str(dwi), "--skeletonMask", str(skel),
                           "--bRange", "800", "2500"])
    assert "--bRange" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# bval/bvec round trip

def test_bval_roundtrip(delta_svd, tmp_path):
    content = "0 0 1000 1000\n"
    fn = tmp_path / "test.bval"
    fn.write_text(content)

    arrFloat, arrStr = delta_svd.read_bval_or_bvec(str(fn))
    assert arrFloat.tolist() == [0.0, 0.0, 1000.0, 1000.0]

    out = tmp_path / "out.bval"
    delta_svd.write_bval_or_bvec(arrStr, str(out))
    assert out.read_text() == content


def test_bvec_roundtrip(delta_svd, tmp_path):
    content = "1 0 0.7071\n0 1 0.7071\n0 0 0\n"
    fn = tmp_path / "test.bvec"
    fn.write_text(content)

    arrFloat, arrStr = delta_svd.read_bval_or_bvec(str(fn))
    assert arrFloat.shape == (3, 3)

    out = tmp_path / "out.bvec"
    delta_svd.write_bval_or_bvec(arrStr, str(out))
    assert out.read_text() == content


def test_read_bval_or_bvec_rejects_malformed_shape(delta_svd, tmp_path):
    fn = tmp_path / "bad.bval"
    fn.write_text("1 2\n3 4\n")
    with pytest.raises(ValueError):
        delta_svd.read_bval_or_bvec(str(fn))


# A ragged or non-numeric file used to escape as numpy's own ValueError, whose
# message ("inhomogeneous shape", "could not convert string to float") names
# neither the file nor the line. Both are the user's to fix, so both are ours
# to report.

def test_read_bval_or_bvec_rejects_lines_of_unequal_length(delta_svd, tmp_path):
    fn = tmp_path / "bad.bvec"
    fn.write_text("0 1 0\n0 0 1\n0 0\n")          # third direction truncated
    with pytest.raises(delta_svd.DeltaSvdError) as excinfo:
        delta_svd.read_bval_or_bvec(str(fn))

    msg = str(excinfo.value)
    assert str(fn) in msg
    assert "line 1: 3 value(s)" in msg
    assert "line 3: 2 value(s)" in msg
    assert "blank line" not in msg               # none here, so no misleading hint


def test_read_bval_or_bvec_points_at_a_blank_line(delta_svd, tmp_path):
    fn = tmp_path / "bad.bval"
    fn.write_text("0 1000 1000\n\n")             # trailing blank line
    with pytest.raises(delta_svd.DeltaSvdError, match="blank line"):
        delta_svd.read_bval_or_bvec(str(fn))


def test_read_bval_or_bvec_locates_a_non_numeric_value(delta_svd, tmp_path):
    fn = tmp_path / "bad.bval"
    fn.write_text("0 1000 abc 1000\n")
    with pytest.raises(delta_svd.DeltaSvdError) as excinfo:
        delta_svd.read_bval_or_bvec(str(fn))

    msg = str(excinfo.value)
    assert "'abc'" in msg
    assert "line 1 at position 3" in msg
    assert str(fn) in msg


def test_read_bval_or_bvec_locates_a_non_numeric_value_in_a_bvec(delta_svd, tmp_path):
    # the reported position is the one in the file, not in the transposed array
    fn = tmp_path / "bad.bvec"
    fn.write_text("1 0 0\n0 1 0\n0 0 x\n")
    with pytest.raises(delta_svd.DeltaSvdError) as excinfo:
        delta_svd.read_bval_or_bvec(str(fn))

    assert "line 3 at position 3" in str(excinfo.value)


def test_read_bval_or_bvec_truncates_a_long_list_of_lines(delta_svd, tmp_path):
    fn = tmp_path / "bad.bval"
    fn.write_text("\n".join(["1 2 3"] * 12 + ["1 2"]) + "\n")
    with pytest.raises(delta_svd.DeltaSvdError) as excinfo:
        delta_svd.read_bval_or_bvec(str(fn))

    msg = str(excinfo.value)
    assert f"line {delta_svd.MAX_LINES_REPORTED}:" in msg
    assert f"line {delta_svd.MAX_LINES_REPORTED + 1}:" not in msg
    assert "and 3 more line(s)" in msg


@pytest.mark.parametrize("content", ["0 0 1000 1000\n", "1 0 0.7071\n0 1 0.7071\n0 0 0\n"])
def test_read_bval_or_bvec_still_accepts_well_formed_files(delta_svd, tmp_path, content):
    # the added checks must not narrow what parses: 'nan'/'inf'/exponents included
    fn = tmp_path / "ok.bval"
    fn.write_text(content)
    arrFloat, arrStr = delta_svd.read_bval_or_bvec(str(fn))
    assert arrFloat.size == arrStr.size


def test_read_bval_or_bvec_accepts_exponent_notation(delta_svd, tmp_path):
    fn = tmp_path / "ok.bval"
    fn.write_text("0 1e3 1.2E+03\n")
    arrFloat, _ = delta_svd.read_bval_or_bvec(str(fn))
    assert arrFloat.tolist() == [0.0, 1000.0, 1200.0]


# ---------------------------------------------------------------------------
# resolve_b_intervals(): the '--bRange' / '--shells' request as b-value windows

def test_resolve_b_intervals_widens_a_range_by_its_tolerance(delta_svd):
    assert delta_svd.resolve_b_intervals(bRange=[800, 1200]) == [(795.0, 1205.0)]


def test_resolve_b_intervals_orders_the_range_limits(delta_svd):
    # argparse takes the two limits positionally, so nothing stops a user from
    # passing them the wrong way round
    assert delta_svd.resolve_b_intervals(bRange=[1200, 800]) == [(795.0, 1205.0)]


def test_resolve_b_intervals_gives_each_shell_its_own_window(delta_svd):
    # one window per shell, so the b-values *between* the shells stay excluded -
    # the whole point of '--shells' over a range spanning them
    assert delta_svd.resolve_b_intervals(shells=[700, 1000]) == [(675.0, 725.0), (975.0, 1025.0)]


def test_resolve_b_intervals_sorts_and_deduplicates_shells(delta_svd):
    assert delta_svd.resolve_b_intervals(shells=[1000, 700, 1000]) == [(675.0, 725.0), (975.0, 1025.0)]


def test_resolve_b_intervals_prefers_shells_over_the_range_default(delta_svd):
    # '--bRange' keeps its default even when '--shells' is given, so the shells
    # have to win - argparse guarantees the user set only one of the two
    assert delta_svd.resolve_b_intervals(bRange=[800, 1200], shells=[1000]) == [(975.0, 1025.0)]


# ---------------------------------------------------------------------------
# describe_directions(): angular sampling, and whether the tensor is estimable
#
# The count drives the hard floor in filter_b_values(), and it counts *unique*
# directions: the free-water fraction is grid-searched against the residual, so
# a repeated direction adds no constraint on it, however many times it was
# acquired.

def _directions(n):
    """n distinct unit vectors, spread over a hemisphere by the golden-section
    spiral. A hemisphere because directions are antipodally equivalent here, so
    a full sphere would hand back near-duplicate pairs. Well-conditioned for a
    tensor fit at any n >= 6."""
    i = np.arange(n) + 0.5
    z = i / n
    r = np.sqrt(np.maximum(0.0, 1 - z**2))
    phi = np.pi * (1 + 5**0.5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def _plane_directions(n):
    """n distinct unit vectors, all in the xy-plane. However many there are,
    their outer products span only three of the six tensor components, so the
    design matrix stays rank-deficient - a damaged gradient table looks like
    this."""
    a = np.linspace(0, np.pi, n, endpoint=False)
    return np.stack([np.cos(a), np.sin(a), np.zeros(n)], axis=1)


def test_describe_directions_counts_directions_and_reports_full_rank(delta_svd):
    g = _directions(30)
    bvals = np.r_[0.0, np.full(30, 1000.0)]
    bvecs = np.vstack([np.zeros(3), g])

    assert delta_svd.describe_directions(bvals, bvecs) == (30, 7)


def test_describe_directions_counts_a_repeated_direction_once(delta_svd):
    # 24 volumes, 12 directions: the extra acquisitions average noise, they do
    # not constrain the free-water fraction any further
    g = _directions(12)
    bvals = np.r_[0.0, np.full(24, 1000.0)]
    bvecs = np.vstack([np.zeros(3), g, g])

    assert delta_svd.describe_directions(bvals, bvecs)[0] == 12


def test_describe_directions_counts_antipodal_directions_once(delta_svd):
    # g and -g probe the same tensor element, so a full-sphere scheme that
    # includes both must not be credited with twice the angular sampling
    g = _directions(12)
    bvals = np.r_[0.0, np.full(24, 1000.0)]
    bvecs = np.vstack([np.zeros(3), g, -g])

    assert delta_svd.describe_directions(bvals, bvecs)[0] == 12


def test_describe_directions_ignores_b0_volumes_and_zero_bvecs(delta_svd):
    g = _directions(12)
    # three b0s, and one diffusion-weighted volume whose bvec is all zeros: it
    # probes no direction at all
    bvals = np.r_[np.zeros(3), np.full(13, 1000.0)]
    bvecs = np.vstack([np.zeros((3, 3)), g, np.zeros(3)])

    assert delta_svd.describe_directions(bvals, bvecs)[0] == 12


def test_describe_directions_reports_deficient_rank_for_coplanar_directions(delta_svd):
    # plenty of directions, but they span only part of the tensor
    bvals = np.r_[0.0, np.full(20, 1000.0)]
    bvecs = np.vstack([np.zeros(3), _plane_directions(20)])

    nDirections, rank = delta_svd.describe_directions(bvals, bvecs)
    assert nDirections == 20
    assert rank < delta_svd.DESIGN_MATRIX_RANK


def test_describe_directions_reaches_full_rank_at_six_directions(delta_svd):
    # six non-collinear directions plus a b0 are what the seven-column design
    # matrix needs; the floor in filter_b_values() sits well above this, but the
    # rank check must not fire here
    bvals = np.r_[0.0, np.full(6, 1000.0)]
    bvecs = np.vstack([np.zeros(3), _directions(6)])

    assert delta_svd.describe_directions(bvals, bvecs)[1] == 7


# ---------------------------------------------------------------------------
# format_b_values(): the b-values present, for error messages

def test_format_b_values_groups_within_shell_deviation(delta_svd):
    # scanners report b-values that scatter around the nominal shell; an error
    # message listing each of them verbatim would be unreadable
    assert delta_svd.format_b_values([0, 0, 998, 1000, 1002, 2000]) == '0 (n=2), 1000 (n=3), 2000 (n=1)'


# ---------------------------------------------------------------------------
# filter_b_values(): shell selection of the b-value range
#
# This is the gate every later step depends on - the DWI volumes, the bvals and
# the bvecs have to stay index-aligned after filtering, or the tensor fit is
# silently fitted against the wrong gradient directions. It is also where data
# the fits cannot be trusted on is refused: the bi-tensor fit solves with a
# pseudo-inverse, so nothing downstream raises on a degenerate gradient table.

DEFAULT_INTERVALS = [(795.0, 1205.0)]


def _write_dwi_set(tmp_path, bvals, bvecs):
    """Write a matched data/bval/bvec triplet from a b-value sequence and an
    (n, 3) array of directions. Volume i is filled with the constant i, so a
    test can tell from the voxel values which volumes survived."""
    bvals = list(bvals)
    bvecs = np.asarray(bvecs, dtype=float)
    (tmp_path / "in").mkdir(exist_ok=True)
    fnBval = tmp_path / "in" / "sub01.bval"
    fnBvec = tmp_path / "in" / "sub01.bvec"
    fnData = tmp_path / "in" / "sub01.nii.gz"
    fnBval.write_text(" ".join(str(b) for b in bvals) + "\n")
    fnBvec.write_text("\n".join(" ".join(f"{v:.6f}" for v in row) for row in bvecs.T) + "\n")
    img = np.arange(len(bvals), dtype=float) * np.ones((2, 2, 2, len(bvals)))
    nib.save(nib.Nifti1Image(img, np.eye(4)), str(fnData))
    outDir = tmp_path / "out"
    outDir.mkdir(exist_ok=True)
    return str(fnData), str(fnBval), str(fnBvec), str(outDir)


def _shelled_set(tmp_path, shells, nPerShell=20, nB0=1):
    """A b0 plus 'nPerShell' whole-sphere directions on each of 'shells'."""
    g = _directions(nPerShell)
    bvals = [0] * nB0 + [b for b in shells for _ in range(nPerShell)]
    bvecs = np.vstack([np.zeros((nB0, 3))] + [g for _ in shells])
    return _write_dwi_set(tmp_path, bvals, bvecs)


def test_filter_b_values_drops_volumes_outside_brange(delta_svd, tmp_path):
    # b=2000 falls outside the default [800, 1200] shell and must go, together
    # with its bvec columns and its image volumes.
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [1000, 2000], nPerShell=20)

    fnDataO, fnBvalO, fnBvecO, _ = delta_svd.filter_b_values(
        fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
        out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
    )

    assert [Path(f).parent.name for f in (fnDataO, fnBvalO, fnBvecO)] == ["out"] * 3
    # b-value strings are carried over verbatim, not reformatted
    assert Path(fnBvalO).read_text().split() == ["0"] + ["1000"] * 20
    # bvecs stay index-aligned with what survived: three rows, 21 columns
    assert [len(line.split()) for line in Path(fnBvecO).read_text().splitlines()] == [21] * 3
    # volumes 0..20 survive - and in that order
    kept = nib.load(fnDataO).get_fdata()
    assert kept.shape == (2, 2, 2, 21)
    assert kept[0, 0, 0].tolist() == list(range(21))


def test_filter_b_values_rounds_near_zero_bvalues_to_zero(delta_svd, tmp_path):
    # b=3 is a b0 in all but name; dipy's gradient_table would otherwise treat it
    # as a (useless) diffusion-weighted direction. Nothing is dropped here - the
    # volume is kept, only its b-value is rewritten.
    fnData, fnBval, fnBvec, outDir = _write_dwi_set(
        tmp_path,
        [3] + [1000] * 20,
        np.vstack([np.zeros(3), _directions(20)]),
    )

    fnDataO, fnBvalO, fnBvecO, _ = delta_svd.filter_b_values(
        fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
        out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
    )

    assert Path(fnBvalO).read_text().split() == ["0"] + ["1000"] * 20
    assert nib.load(fnDataO).get_fdata().shape[-1] == 21     # no volume dropped


def test_filter_b_values_is_a_no_op_when_nothing_needs_filtering(delta_svd, tmp_path):
    # The no-op path must hand back the *input* paths untouched: it writes no
    # files, so returning out_dir paths would point the pipeline at nothing.
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [1000], nPerShell=20)

    fnDataO, fnBvalO, fnBvecO, _ = delta_svd.filter_b_values(
        fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
        out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
    )

    assert (fnDataO, fnBvalO, fnBvecO) == (fnData, fnBval, fnBvec)
    assert list(Path(outDir).iterdir()) == []


def test_filter_b_values_checks_the_unfiltered_data_too(delta_svd, tmp_path):
    # the no-op path returns early, so the guards have to sit ahead of it - a
    # dataset that needs no filtering can still be unfittable
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [1000], nPerShell=8)

    with pytest.raises(ValueError, match="unique diffusion direction"):
        delta_svd.filter_b_values(
            fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
            out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
        )


def test_filter_b_values_keeps_b0s_regardless_of_brange(delta_svd, tmp_path):
    # b0 volumes are selected by the '<= 5' rule, not by the requested range - a
    # narrow range must never strip the b0s the fit needs for S0.
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [700, 1000], nPerShell=20)

    _, fnBvalO, _, _ = delta_svd.filter_b_values(
        fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
        out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
    )

    assert Path(fnBvalO).read_text().split() == ["0"] + ["1000"] * 20


def test_filter_b_values_returns_the_direction_count(delta_svd, tmp_path):
    # the count qualifies every metric the run produces, so it is handed back
    # for the QC report rather than only printed
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [1000], nPerShell=32)

    assert delta_svd.filter_b_values(
        fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
        out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
    )[3] == 32


# ---------------------------------------------------------------------------
# filter_b_values(): the b-value tolerance
#
# Scanners report b-values that deviate from the nominal shell, so a limit met
# exactly would discard volumes of the very shell that was asked for.

@pytest.mark.parametrize("bval,kept", [(1205, True), (1206, False),
                                       (795, True), (794, False)])
def test_filter_b_values_meets_the_range_limits_with_a_tolerance(delta_svd, tmp_path, bval, kept):
    fnData, fnBval, fnBvec, outDir = _write_dwi_set(
        tmp_path,
        [0] + [1000] * 20 + [bval],
        np.vstack([np.zeros(3), _directions(21)]),
    )

    _, fnBvalO, _, _ = delta_svd.filter_b_values(
        fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
        out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
    )

    assert (str(bval) in Path(fnBvalO).read_text().split()) is kept


@pytest.mark.parametrize("bval,kept", [(1025, True), (1026, False),
                                       (975, True), (974, False)])
def test_filter_b_values_matches_a_shell_with_its_tolerance(delta_svd, tmp_path, bval, kept):
    fnData, fnBval, fnBvec, outDir = _write_dwi_set(
        tmp_path,
        [0] + [1000] * 20 + [bval],
        np.vstack([np.zeros(3), _directions(21)]),
    )

    _, fnBvalO, _, _ = delta_svd.filter_b_values(
        fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
        out_dir=outDir, bIntervals=delta_svd.resolve_b_intervals(shells=[1000]),
    )

    assert (str(bval) in Path(fnBvalO).read_text().split()) is kept


def test_filter_b_values_excludes_shells_between_requested_ones(delta_svd, tmp_path):
    # what '--shells' is for: a range spanning 700 and 1500 would drag the 1000
    # shell into the fit with it
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [700, 1000, 1500], nPerShell=20)

    _, fnBvalO, _, _ = delta_svd.filter_b_values(
        fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
        out_dir=outDir, bIntervals=delta_svd.resolve_b_intervals(shells=[700, 1500]),
    )

    kept = set(Path(fnBvalO).read_text().split())
    assert kept == {"0", "700", "1500"}


# ---------------------------------------------------------------------------
# filter_b_values(): the guards
#
# Each of these would otherwise reach the fits, which solve with a pseudo-inverse
# and so return a minimum-norm solution instead of raising - the run would finish
# and report plausible-looking numbers.

def test_filter_b_values_rejects_a_shell_that_matches_nothing(delta_svd, tmp_path):
    # the user named the shell explicitly, so its absence is a misconfiguration,
    # not something to quietly carry on without
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [1000], nPerShell=20)

    with pytest.raises(ValueError, match=r"No volume has a b-value in \[1475, 1525\]"):
        delta_svd.filter_b_values(
            fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
            out_dir=outDir, bIntervals=delta_svd.resolve_b_intervals(shells=[1000, 1500]),
        )


def test_filter_b_values_reports_the_b_values_present_when_nothing_matches(delta_svd, tmp_path):
    # the likeliest cause is a selection that does not fit the data, so the
    # error has to show what the data actually holds
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [2000], nPerShell=20)

    with pytest.raises(ValueError, match=r"0 \(n=1\), 2000 \(n=20\)"):
        delta_svd.filter_b_values(
            fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
            out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
        )


def test_filter_b_values_rejects_data_without_a_b0(delta_svd, tmp_path):
    # S0 is averaged from the b0 volumes; without one the fits have no scale
    fnData, fnBval, fnBvec, outDir = _write_dwi_set(
        tmp_path, [1000] * 20, _directions(20))

    with pytest.raises(ValueError, match="close to zero"):
        delta_svd.filter_b_values(
            fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
            out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
        )


def test_filter_b_values_rejects_too_few_directions(delta_svd, tmp_path):
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [1000], nPerShell=11)

    with pytest.raises(ValueError, match="Only 11 unique diffusion direction"):
        delta_svd.filter_b_values(
            fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
            out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
        )


def test_filter_b_values_accepts_exactly_the_minimum_directions(delta_svd, tmp_path):
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [1000], nPerShell=12)

    assert delta_svd.filter_b_values(
        fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
        out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
    )[3] == 12


def test_filter_b_values_does_not_count_repeats_towards_the_minimum(delta_svd, tmp_path):
    # 24 volumes, but only 6 directions: enough to solve for the tensor, and
    # nowhere near enough to pin the free-water fraction down
    g = _directions(6)
    fnData, fnBval, fnBvec, outDir = _write_dwi_set(
        tmp_path,
        [0] + [1000] * 24,
        np.vstack([np.zeros(3), g, g, g, g]),
    )

    with pytest.raises(ValueError, match="Only 6 unique diffusion direction"):
        delta_svd.filter_b_values(
            fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
            out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
        )


def test_filter_b_values_counts_directions_after_the_selection(delta_svd, tmp_path):
    # 20 directions per shell, but a range that keeps only eight of them: the
    # floor applies to what reaches the fit, not to what was acquired
    g = _directions(20)
    fnData, fnBval, fnBvec, outDir = _write_dwi_set(
        tmp_path,
        [0] + [1000] * 8 + [2000] * 12,
        np.vstack([np.zeros(3), g[:8], g[8:]]),
    )

    with pytest.raises(ValueError, match="Only 8 unique diffusion direction"):
        delta_svd.filter_b_values(
            fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
            out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
        )


def test_filter_b_values_rejects_a_degenerate_gradient_table(delta_svd, tmp_path):
    # enough directions to pass the count, but they lie in one plane and so
    # cannot span the tensor - what a damaged bvec file looks like
    fnData, fnBval, fnBvec, outDir = _write_dwi_set(
        tmp_path,
        [0] + [1000] * 20,
        np.vstack([np.zeros(3), _plane_directions(20)]),
    )

    with pytest.raises(ValueError, match="do not span the diffusion tensor"):
        delta_svd.filter_b_values(
            fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
            out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
        )


def test_filter_b_values_rejects_a_bvec_file_of_the_wrong_length(delta_svd, tmp_path):
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [1000], nPerShell=20)
    Path(fnBvec).write_text("\n".join(" ".join("0" for _ in range(19)) for _ in range(3)) + "\n")

    with pytest.raises(ValueError, match="19 directions but the bval file holds 21"):
        delta_svd.filter_b_values(
            fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
            out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
        )


def test_filter_b_values_warns_below_the_recommended_direction_count(delta_svd, tmp_path, capsys):
    # between the floor and the recommendation the fit runs, but the free-water
    # fraction is noisy - a warning, not an error
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [1000], nPerShell=16)

    delta_svd.filter_b_values(
        fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
        out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
    )

    out = capsys.readouterr().out
    assert "WARNING" in out and "recommended minimum of 20" in out


def test_filter_b_values_does_not_warn_at_the_recommended_direction_count(delta_svd, tmp_path, capsys):
    fnData, fnBval, fnBvec, outDir = _shelled_set(tmp_path, [1000], nPerShell=20)

    delta_svd.filter_b_values(
        fn_data=fnData, fn_bval=fnBval, fn_bvec=fnBvec,
        out_dir=outDir, bIntervals=DEFAULT_INTERVALS,
    )

    assert "WARNING" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# integrate_masks()
#
# Produces every voxel count that reaches the results CSV, and every skeleton
# mask extract_stats() later globs for. The counts are what a reader uses to
# judge whether a timepoint is usable, so a silent off-by-one here is invisible
# downstream but changes the reported numbers. Each test below pins one of the
# structurally distinct branches.

def _vol(values):
    """A (n, 1, 1) volume - a one-dimensional 'brain' is enough for every branch
    here, and keeps the expected voxel counts readable."""
    return np.asarray(values, dtype=float).reshape(-1, 1, 1)


def _tbss_tree(tmp_path, skeleton, bmasks, emask=None, rois=None, mni=None):
    """Lay out the minimal TBSS tree integrate_masks() reads: one skeletonised
    brain mask per timepoint, plus the optional exclusion / ROI inputs."""
    stats = tmp_path / "TBSS" / "stats"
    stats.mkdir(parents=True)
    affine = np.eye(4)

    def save(path, arr):
        nib.save(nib.Nifti1Image(_vol(arr), affine), str(path))

    skelMask = tmp_path / "skel.nii.gz"
    save(skelMask, skeleton)

    dirTP = []
    for tp, arr in bmasks.items():
        (tmp_path / tp).mkdir()
        dirTP.append(str(tmp_path / tp))
        save(stats / f"all_{tp}_bmask_skeletonised.nii.gz", arr)

    if emask is not None:
        save(stats / "all_E-MASK_skeletonised.nii.gz", emask)
    for label, arr in (rois or {}).items():
        save(stats / f"all_ROI-{label}_skeletonised.nii.gz", arr)

    fnROI_MNI = None
    if mni is not None:
        fnROI_MNI = str(tmp_path / "roi_mni.nii.gz")
        save(fnROI_MNI, mni)

    return dict(dirTP=dirTP, dirTBSS=str(tmp_path / "TBSS"),
                skelMask=str(skelMask), fnROI_MNI=fnROI_MNI), stats


def _counts(df):
    return dict(zip(df["region"], df["voxels"]))


def test_integrate_masks_single_timepoint_drops_interpolated_voxels(delta_svd, tmp_path):
    # A skeletonised brain mask is resampled, so its edge voxels come back with
    # 0 < value < 1 - foreground blended with background. Those must be dropped
    # ('bmask < 1'), not rounded in, or the skeleton spills past the brain.
    kwargs, stats = _tbss_tree(
        tmp_path,
        skeleton=[1, 1, 1, 1, 1, 0],
        bmasks={"TP01": [1, 1, 1, 1, 0.5, 1]},
    )

    df = delta_svd.integrate_masks(**kwargs)

    assert _counts(df) == {"total": 5, "intersection": 4}
    # a single timepoint is named, not aggregated as 'all'
    assert list(df["timepoint"]) == ["TP01", "TP01"]
    assert list(df["skeleton"]) == ["skel.nii.gz"] * 2
    # voxel-count rows carry no metric; extract_stats() appends those separately
    assert list(df["metric"]) == ["NA"] * 2 and list(df["value"]) == ["NaN"] * 2

    saved = nib.load(str(stats / "skel_intersection.nii.gz")).get_fdata()
    assert saved.ravel().tolist() == [1, 1, 1, 1, 0, 0]


def test_integrate_masks_multi_timepoint_intersects_and_reports_set_difference(delta_svd, tmp_path):
    # With several timepoints the analysis runs on the voxels common to all of
    # them, so a longitudinal change cannot come from a changing denominator.
    # The per-timepoint 'set_difference' rows say how much each one gave up.
    kwargs, stats = _tbss_tree(
        tmp_path,
        skeleton=[1, 1, 1, 1, 1, 0],
        bmasks={"TP01": [1, 1, 1, 1, 1, 1],       # adjusted -> 5 voxels
                "TP02": [1, 1, 1, 0.4, 1, 1]},    # adjusted -> 4 voxels
    )

    df = delta_svd.integrate_masks(**kwargs)

    assert list(df["region"]) == ["total", "intersection",
                                 "set_difference", "set_difference"]
    assert list(df["voxels"]) == [5, 4, 1, 0]
    assert list(df["timepoint"]) == ["all", "all", "TP01", "TP02"]

    saved = nib.load(str(stats / "skel_intersection.nii.gz")).get_fdata()
    assert saved.ravel().tolist() == [1, 1, 1, 0, 1, 0]


def test_integrate_masks_excludes_emask_and_labels_it_for_qc(delta_svd, tmp_path):
    # The exclusion mask (e.g. an infarct) is cut out of the analysed skeleton.
    # Two things are easy to get wrong and both are checked here: the 0.05
    # threshold is deliberately conservative, so interpolated fringe voxels are
    # excluded too; and the QC image must label 2 only where exclusion actually
    # removed something, i.e. inside the intersection.
    kwargs, stats = _tbss_tree(
        tmp_path,
        skeleton=[1, 1, 1, 1, 1, 0],
        bmasks={"TP01": [1, 1, 1, 1, 0.5, 1]},    # intersection -> [1,1,1,1,0,0]
        emask=[0, 0.04, 0.5, 0, 0.9, 0],
    )

    df = delta_svd.integrate_masks(**kwargs)

    assert _counts(df) == {"total": 5, "intersection": 4, "intersection_Emask": 3}

    final = nib.load(str(stats / "skel_intersection_Emask.nii.gz")).get_fdata()
    assert final.ravel().tolist() == [1, 1, 0, 1, 0, 0]

    qc = nib.load(str(stats / "skel_intersection_Emask-as-label2.nii.gz")).get_fdata()
    # index 1 (0.04) is under threshold and stays skeleton; index 2 is excluded
    # and labelled 2; index 4 is excluded but was never in the intersection, so
    # it must stay 0 rather than show up as a phantom lesion in the QC image.
    assert qc.ravel().tolist() == [1, 1, 2, 1, 0, 0]


def test_integrate_masks_inserts_the_background_roi_before_the_named_rois(delta_svd, tmp_path):
    # Subject ROIs partition the skeleton, and Rmask-00 is the complement that
    # makes the partition complete. It is *inserted* ahead of the named ROI rows
    # rather than appended - the insert offset is index arithmetic over a list
    # that has already grown, which is exactly the kind of thing that silently
    # slips a row into the wrong place in the CSV.
    kwargs, stats = _tbss_tree(
        tmp_path,
        skeleton=[1, 1, 1, 1, 1, 0],
        bmasks={"TP01": [1, 1, 1, 1, 0.5, 1]},    # intersection -> [1,1,1,1,0,0]
        rois={"01": [1, 0, 0, 0, 0, 0],
              "02": [0, 0.5, 0.04, 0, 0.9, 0]},
    )

    df = delta_svd.integrate_masks(**kwargs)

    assert list(df["region"]) == [
        "total", "intersection",
        "intersection_Rmask-00", "intersection_Rmask-01", "intersection_Rmask-02",
    ]
    assert list(df["voxels"]) == [5, 4, 2, 1, 1]
    # the ROIs and the background partition the intersection exactly
    assert sum(list(df["voxels"])[2:]) == df["voxels"].iloc[1]

    # index 2 (0.04) is below the 0.05 threshold and index 4 is outside the
    # intersection, so ROI-02 keeps only index 1
    roi02 = nib.load(str(stats / "skel_intersection_Rmask-02.nii.gz")).get_fdata()
    assert roi02.ravel().tolist() == [0, 1, 0, 0, 0, 0]

    background = nib.load(str(stats / "skel_intersection_Rmask-00.nii.gz")).get_fdata()
    assert background.ravel().tolist() == [0, 0, 1, 1, 0, 0]

    # the merged file re-labels each ROI with its own number, not a binary union
    merged = nib.load(str(stats / "skel_intersection_Rmask.nii.gz")).get_fdata()
    assert merged.ravel().tolist() == [1, 2, 0, 0, 0, 0]


def test_integrate_masks_roi_suffixes_follow_the_emask_exclusion(delta_svd, tmp_path):
    # Once an exclusion mask is present the ROI outputs have to hang off the
    # *excluded* skeleton, both in name and in content - extract_stats() globs
    # '*_Rmask-*' and would otherwise mix pre- and post-exclusion regions.
    kwargs, stats = _tbss_tree(
        tmp_path,
        skeleton=[1, 1, 1, 1, 1, 0],
        bmasks={"TP01": [1, 1, 1, 1, 0.5, 1]},
        emask=[0, 1, 0, 0, 0, 0],                 # removes index 1
        rois={"01": [1, 1, 0, 0, 0, 0]},
    )

    df = delta_svd.integrate_masks(**kwargs)

    assert list(df["region"]) == ["total", "intersection", "intersection_Emask",
                                 "intersection_Emask_Rmask-00",
                                 "intersection_Emask_Rmask-01"]
    roi01 = nib.load(str(stats / "skel_intersection_Emask_Rmask-01.nii.gz")).get_fdata()
    assert roi01.ravel().tolist() == [1, 0, 0, 0, 0, 0]      # index 1 was excluded


def test_integrate_masks_relabels_mni_rois_including_the_background(delta_svd, tmp_path):
    # An atlas file carries several labels in one volume; each becomes its own
    # region. Label 0 is deliberately kept as a region ('the rest of the
    # skeleton'), so a future 'uROI = uROI[uROI>0]' tidy-up would silently drop a
    # column from the results CSV.
    kwargs, stats = _tbss_tree(
        tmp_path,
        skeleton=[1, 1, 1, 1, 1, 0],
        bmasks={"TP01": [1, 1, 1, 1, 0.5, 1]},    # intersection -> [1,1,1,1,0,0]
        mni=[0, 0, 1, 1, 2, 2],
    )

    df = delta_svd.integrate_masks(**kwargs)

    assert list(df["region"]) == ["total", "intersection",
                                 "intersection_RmaskMNI-00",
                                 "intersection_RmaskMNI-01",
                                 "intersection_RmaskMNI-02"]
    # label 2 lies entirely outside the intersection -> an honest zero, not a
    # missing row
    assert list(df["voxels"]) == [5, 4, 2, 2, 0]

    merged = nib.load(str(stats / "skel_intersection_RmaskMNI.nii.gz")).get_fdata()
    # merged after intersecting with the final skeleton, so it differs from the
    # atlas that went in: label 2 is gone and label 0 is not written
    assert merged.ravel().tolist() == [0, 0, 1, 1, 0, 0]


def test_integrate_masks_splits_hemispheres_along_the_first_axis(delta_svd, tmp_path):
    # Optional per-hemisphere analysis. The split is a plain index cut at
    # shape[0] // 2 with no image-orientation check, and 'LH' is the half that
    # survives after the *low* indices are zeroed - pinned here because nothing
    # else in the pipeline would notice the two labels swapping.
    kwargs, stats = _tbss_tree(
        tmp_path,
        skeleton=[1, 1, 1, 1, 1, 0],
        bmasks={"TP01": [1, 1, 1, 1, 1, 1]},      # intersection -> [1,1,1,1,1,0]
    )

    df = delta_svd.integrate_masks(analyseHemispheres=True, **kwargs)

    counts = _counts(df)
    assert counts["intersection_LH"] == 2
    assert counts["intersection_RH"] == 3
    assert counts["intersection_LH"] + counts["intersection_RH"] == counts["intersection"]

    lh = nib.load(str(stats / "skel_intersection_LH.nii.gz")).get_fdata()
    assert lh.ravel().tolist() == [0, 0, 0, 1, 1, 0]


@pytest.mark.parametrize("bmasks", [
    {"TP01": [1, 1, 1, 1, 1]},                              # cross-sectional
    {"TP01": [1, 1, 1, 1, 1], "TP02": [1, 1, 1, 1, 1]},     # longitudinal
])
def test_integrate_masks_binarises_a_non_binary_skeleton_mask(delta_svd, tmp_path, bmasks):
    # Fractional values used to be truncated by the uint8 write yet counted in
    # full, and only cross-sectionally (np.all() binarises the longitudinal
    # path). Both modes must agree, and the count must match what was written.
    kwargs, stats = _tbss_tree(tmp_path, skeleton=[1, 1, 0.5, 0.5, 0], bmasks=bmasks)

    df = delta_svd.integrate_masks(**kwargs)

    saved = nib.load(str(stats / "skel_intersection.nii.gz")).get_fdata()
    assert saved.ravel().tolist() == [1, 1, 1, 1, 0]
    assert _counts(df)["intersection"] == int(np.count_nonzero(saved)) == 4


# ---------------------------------------------------------------------------
# extract_stats() metric naming
#
# The emitted names are part of the published output contract, so they are
# pinned exactly: a peak-width statistic is only defined for MD, and the
# free-water and mean-diffusivity means carry the 'MS' prefix.

def test_extract_stats_emits_correct_metric_names(delta_svd, tmp_path):
    dirTBSS = tmp_path / "TBSS"
    stats_dir = dirTBSS / "stats"
    stats_dir.mkdir(parents=True)
    affine = np.eye(4)

    mask = np.array([[[1, 1], [0, 1]]], dtype=float)
    nib.save(nib.Nifti1Image(mask, affine), str(stats_dir / "skel_intersection.nii.gz"))

    fw = np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=float)
    md = np.array([[[0.0005, 0.0007], [0.0006, 0.0009]]], dtype=float)
    nib.save(nib.Nifti1Image(fw, affine), str(stats_dir / "all_TP01_FW_skeletonised.nii.gz"))
    nib.save(nib.Nifti1Image(md, affine), str(stats_dir / "all_TP01_MD_skeletonised.nii.gz"))

    df = delta_svd.extract_stats(
        dirTP="/some/path/TP01",
        dirTBSS=str(dirTBSS),
        fnNonFA={"FW": "unused", "MD": "unused"},
        skelMask="skel.nii.gz",
    )

    assert set(df["metric"]) == {"PSMD", "MSMD", "MSFW"}


# PSMD is defined as the 95th minus the 5th percentile of MD on the skeleton.
# The name test above would still pass if that arithmetic were wrong (an IQR, a
# standard deviation, the wrong percentiles, or the difference taken the wrong
# way round), so pin the number itself on an array whose percentiles are exact.
def test_extract_stats_psmd_is_p95_minus_p5(delta_svd, tmp_path):
    dirTBSS = tmp_path / "TBSS"
    stats_dir = dirTBSS / "stats"
    stats_dir.mkdir(parents=True)
    affine = np.eye(4)

    # 21 in-mask samples so numpy's linear interpolation lands exactly on one:
    # p5 -> index (21-1)*0.05 = 1, p95 -> index 19. PSMD is therefore 18e-4.
    mdIn = np.arange(21) * 1e-4
    # The two out-of-mask voxels carry a CSF-like value that would blow up p95
    # if the ROI masking were dropped.
    mask = np.array([0.0] + [1.0] * 21 + [0.0]).reshape(-1, 1, 1)
    md = np.concatenate(([9e-3], mdIn, [9e-3])).reshape(-1, 1, 1)

    nib.save(nib.Nifti1Image(mask, affine), str(stats_dir / "skel_intersection.nii.gz"))
    nib.save(nib.Nifti1Image(md, affine), str(stats_dir / "all_TP01_MD_skeletonised.nii.gz"))

    df = delta_svd.extract_stats(
        dirTP="/some/path/TP01",
        dirTBSS=str(dirTBSS),
        fnNonFA={"MD": "unused"},
        skelMask="skel.nii.gz",
    )

    row = df.set_index("metric")
    assert row.loc["PSMD", "voxels"] == 21
    assert row.loc["PSMD", "value"] == pytest.approx(18e-4, rel=1e-12)
    assert row.loc["MSMD", "value"] == pytest.approx(mdIn.mean(), rel=1e-12)


# ---------------------------------------------------------------------------
# Tensor fitting (wls_fit_dti / wls_iter_dti)

def test_wls_fit_dti_recovers_known_tensor(delta_svd):
    bvecs = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.7071, 0.7071, 0],
            [0.7071, 0, 0.7071],
            [0, 0.7071, 0.7071],
        ],
        dtype=float,
    )
    bvals = np.array([0, 0, 1000, 1000, 1000, 1000, 1000, 1000], dtype=float)
    gtab = gradient_table(bvals, bvecs=bvecs)
    W = design_matrix(gtab)

    evals_true = np.array([1.7e-3, 0.3e-3, 0.3e-3])
    D = np.diag(evals_true)
    S0 = 1000.0
    signal = np.array([S0 * np.exp(-b * g.dot(D).dot(g)) for b, g in zip(bvals, bvecs)])

    data = signal.reshape(1, 1, 1, -1)
    mask = np.ones((1, 1, 1), dtype=bool)

    fw_params = delta_svd.wls_fit_dti(W, data, mask=mask)
    evals, _ = decompose_tensor(from_lower_triangular(fw_params))

    FA = dti.fractional_anisotropy(evals).ravel()[0]
    MD = dti.mean_diffusivity(evals).ravel()[0]

    assert FA == pytest.approx(dti.fractional_anisotropy(evals_true), abs=1e-4)
    assert MD == pytest.approx(np.mean(evals_true), abs=1e-8)


# ---------------------------------------------------------------------------
# Free-water two-tensor fitting (wls_fit_tensor_fw / wls_iter_fw)
#
# The output row is [Dxx, Dxy, Dyy, Dxz, Dyz, Dzz, log(S0), f, dF2] - index 7 is
# the free-water fraction written out as wls_dti_FW.nii.gz, index 8 the residual
# difference. These tests fix the *structure* and the qualitative behaviour of
# the fit, not an accuracy claim: the estimator is deliberately regularised
# towards MDm and is biased, so asserting f == f_true would be wrong.

@pytest.fixture(scope="module")
def _fw_gtab():
    bvecs = np.array(
        [
            [0, 0, 0], [0, 0, 0],
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [0.7071, 0.7071, 0], [0.7071, 0, 0.7071], [0, 0.7071, 0.7071],
            [-0.7071, 0.7071, 0], [-0.7071, 0, 0.7071], [0, -0.7071, 0.7071],
            [0.5774, 0.5774, 0.5774], [-0.5774, 0.5774, 0.5774],
            [0.5774, -0.5774, 0.5774], [0.5774, 0.5774, -0.5774],
        ],
        dtype=float,
    )
    bvals = np.array([0, 0] + [1000] * 13, dtype=float)
    return gradient_table(bvals, bvecs=bvecs), bvals, bvecs


def _two_compartment_signal(bvals, bvecs, fFree, evals=(1.6e-3, 0.4e-3, 0.4e-3),
                            S0=1000.0, Diso=3e-3):
    """Signal of a voxel that is a fFree mixture of isotropic free water with a
    tissue tensor - the model wls_fit_tensor_fw() is supposed to invert."""
    D = np.diag(evals)
    tissue = np.array([np.exp(-b * g.dot(D).dot(g)) for b, g in zip(bvals, bvecs)])
    water = np.exp(-bvals * Diso)
    return S0 * ((1 - fFree) * tissue + fFree * water)


def _fit_fw(delta_svd, gtab, bvals, bvecs, fFree, MDm=0.0006, mdreg=2.0e-3):
    W = design_matrix(gtab)
    data = _two_compartment_signal(bvals, bvecs, fFree).reshape(1, 1, 1, -1)
    mask = np.ones((1, 1, 1), dtype=bool)
    # MD0 comes from the uncorrected single-tensor fit, exactly as
    # free_water_correction() feeds it in.
    dtiParams = delta_svd.wls_fit_dti(W, data, mask=mask)
    MD0 = dti.mean_diffusivity(decompose_tensor(from_lower_triangular(dtiParams))[0])
    S0 = np.mean(data[..., gtab.b0s_mask], axis=-1)
    params = delta_svd.wls_fit_tensor_fw(W, data, MD0, S0, Diso=3e-3, mask=mask,
                                         min_signal=1.0e-6, piterations=2,
                                         mdreg=mdreg, MDm=MDm)
    evals, _ = decompose_tensor(from_lower_triangular(params))
    return params[0, 0, 0], MD0.ravel()[0], dti.mean_diffusivity(evals).ravel()[0]


def test_wls_fit_tensor_fw_free_water_fraction_tracks_contamination(delta_svd, _fw_gtab):
    # The one thing the free-water map has to get right: more free water in the
    # voxel must mean a larger f. Same tissue tensor throughout, so nothing but
    # the contamination differs.
    gtab, bvals, bvecs = _fw_gtab
    fEst = [_fit_fw(delta_svd, gtab, bvals, bvecs, f)[0][7] for f in (0.0, 0.2, 0.4, 0.6)]

    assert all(0.0 <= f <= 0.98 for f in fEst)          # 0.98 is the sampler's cap
    assert fEst == sorted(fEst) and len(set(fEst)) == 4


def test_wls_fit_tensor_fw_removes_free_water_from_the_tissue_md(delta_svd, _fw_gtab):
    # This is what separates the two-tensor fit from wls_fit_dti: free water
    # inflates the single-tensor MD, and the corrected tissue MD must come back
    # down to the MDm the fit is regularised towards.
    gtab, bvals, bvecs = _fw_gtab
    for fFree in (0.2, 0.4):
        params, MD0, MD1 = _fit_fw(delta_svd, gtab, bvals, bvecs, fFree, MDm=0.0006)
        assert MD0 > 0.0006                              # contaminated, uncorrected
        assert MD1 == pytest.approx(0.0006, rel=0.01)
        assert MD1 < MD0


def test_wls_fit_tensor_fw_follows_the_mdm_target(delta_svd, _fw_gtab):
    # MDm is a free parameter of the fit (free_water_correction() passes 0.0006);
    # if the MD-matching sample selection broke, the tissue MD would stop
    # tracking it.
    gtab, bvals, bvecs = _fw_gtab
    _, _, mdLow = _fit_fw(delta_svd, gtab, bvals, bvecs, 0.3, MDm=0.0004)
    _, _, mdHigh = _fit_fw(delta_svd, gtab, bvals, bvecs, 0.3, MDm=0.0006)
    assert mdLow == pytest.approx(0.0004, rel=0.05)
    assert mdHigh == pytest.approx(0.0006, rel=0.05)
    assert mdLow < mdHigh


def test_wls_fit_tensor_fw_declares_csf_voxels_fully_free_water(delta_svd, _fw_gtab):
    # A voxel whose single-tensor MD is at or above mdreg is CSF; it short-cuts
    # the fit and is written out as f = 1. Everything else stays zero, so the
    # tensor is null and FA comes out 0 - which is what the '--- FA==0 inside the
    # brain mask -> 0.05' fixup downstream keys off.
    gtab, bvals, bvecs = _fw_gtab
    W = design_matrix(gtab)
    data = np.ones((2, 1, 1, len(bvals))) * 500.0
    md = np.array([[[1.0e-3]], [[2.5e-3]]])              # below / above mdreg
    S0 = np.ones((2, 1, 1)) * 500.0

    params = delta_svd.wls_fit_tensor_fw(W, data, md, S0, mask=np.ones((2, 1, 1), bool),
                                         mdreg=2.0e-3)

    assert params.shape == (2, 1, 1, 9)
    assert params[1, 0, 0].tolist() == [0, 0, 0, 0, 0, 0, 0, 1.0, 0]
    assert params[0, 0, 0, 7] != 1.0                     # sub-threshold voxel was fitted


def test_wls_fit_tensor_fw_leaves_signalless_voxels_at_zero(delta_svd, _fw_gtab):
    # Below min_signal the fit is skipped entirely. It has to be: handed noise,
    # the WLS solve takes log() of a clipped floor and the fitted log(S0) runs
    # away (~20 for a 1e-9 signal), so a mask that leaks background would seed
    # nonsense into the FA and FW maps. The contrast with the CSF branch above
    # also matters - background must not come out as 100 % free water.
    gtab, bvals, bvecs = _fw_gtab
    W = design_matrix(gtab)
    # voxel 0 is exactly zero, voxel 1 sits just under min_signal; both have an
    # MD below mdreg, so only the low-signal guard can zero them
    data = np.stack([np.zeros(len(bvals)),
                     np.full(len(bvals), 1.0e-9)]).reshape(2, 1, 1, -1)
    md = np.array([[[1.0e-3]], [[1.0e-3]]])
    S0 = np.array([[[0.0]], [[1.0e-9]]])

    params = delta_svd.wls_fit_tensor_fw(W, data, md, S0, min_signal=1.0e-6,
                                         mask=np.ones((2, 1, 1), bool))

    assert not params.any()


def test_wls_fit_tensor_fw_respects_the_mask(delta_svd, _fw_gtab):
    gtab, bvals, bvecs = _fw_gtab
    W = design_matrix(gtab)
    data = np.ones((2, 1, 1, len(bvals))) * 500.0
    md = np.array([[[1.0e-3]], [[1.0e-3]]])
    S0 = np.ones((2, 1, 1)) * 500.0

    params = delta_svd.wls_fit_tensor_fw(W, data, md, S0,
                                         mask=np.array([[[True]], [[False]]]))
    assert params[0, 0, 0].any()
    assert not params[1, 0, 0].any()

    with pytest.raises(ValueError):
        delta_svd.wls_fit_tensor_fw(W, data, md, S0, mask=np.ones((3, 1, 1), bool))


# ---------------------------------------------------------------------------
# Parallel driver for the vendored fits (fit_voxelwise)
#
# What matters is bit-identity with the serial loop, not approximate agreement:
# a 1-ULP difference in the FA maps has been observed to move delta-PSMD by
# percent. Hence the raw-byte comparisons against the vendored fit itself.

@pytest.fixture(scope="module")
def _fw_volume(_fw_gtab):
    """A small multi-slab volume: 11 along axis 0 so the slab edges do not divide
    it evenly, with masked-out slices and a CSF slice to cover the fit's branches."""
    gtab, bvals, bvecs = _fw_gtab
    rng = np.random.default_rng(0)
    nx, ny, nz = 11, 3, 2
    data = np.empty((nx, ny, nz, len(bvals)))
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                data[ix, iy, iz] = _two_compartment_signal(
                    bvals, bvecs, fFree=rng.uniform(0.0, 0.6))
    mask = np.ones((nx, ny, nz), dtype=bool)
    mask[3] = False                                      # a wholly empty slab
    mask[7, 1] = False                                   # a partly empty one
    md = np.full((nx, ny, nz), 1.0e-3)
    md[5] = 2.5e-3                                       # above mdreg: CSF short-cut
    S0 = np.mean(data[..., gtab.b0s_mask], axis=-1)
    return design_matrix(gtab), data, md, S0, mask


@pytest.mark.parametrize("nproc", [2, 3, 4, 32])          # 32 > 11 slabs available
def test_fit_voxelwise_is_bit_identical_to_the_serial_fw_fit(delta_svd, _fw_volume, nproc):
    W, data, md, S0, mask = _fw_volume
    kwargs = dict(Diso=3e-3, min_signal=1.0e-6, piterations=2,
                  mdreg=2.0e-3, MDm=0.0006)

    serial = delta_svd.wls_fit_tensor_fw(W, data, md, S0, mask=mask, **kwargs)
    parallel = delta_svd.fit_voxelwise(delta_svd.wls_fit_tensor_fw, data,
                                       {'md_data': md, 'S0': S0, 'mask': mask},
                                       {'W': W, **kwargs}, nproc)

    assert serial.any()                                   # guard against comparing zeros
    assert parallel.tobytes() == serial.tobytes()


@pytest.mark.parametrize("nproc", [1, 4])
def test_fit_voxelwise_is_bit_identical_to_the_serial_dti_fit(delta_svd, _fw_volume, nproc):
    W, data, _, _, mask = _fw_volume

    serial = delta_svd.wls_fit_dti(W, data, mask=mask, min_signal=1.0e-6)
    parallel = delta_svd.fit_voxelwise(delta_svd.wls_fit_dti, data, {'mask': mask},
                                       {'W': W, 'min_signal': 1.0e-6}, nproc)

    assert serial.any()
    assert parallel.tobytes() == serial.tobytes()


def test_fit_voxelwise_handles_an_empty_mask(delta_svd, _fw_volume):
    # Every slab is a no-op, but the driver still owes the full zero volume.
    W, data, md, S0, _ = _fw_volume
    empty = np.zeros(data.shape[:-1], dtype=bool)

    parallel = delta_svd.fit_voxelwise(delta_svd.wls_fit_tensor_fw, data,
                                       {'md_data': md, 'S0': S0, 'mask': empty},
                                       {'W': W, 'mdreg': 2.0e-3}, 4)

    assert parallel.shape == data.shape[:-1] + (9,)
    assert not parallel.any()


def test_fit_voxelwise_propagates_worker_exceptions(delta_svd, _fw_volume):
    # A wrong-shaped mask raises inside the vendored fit; that has to surface in
    # the parent rather than hang the pool or go unnoticed.
    W, data, md, S0, _ = _fw_volume
    bad = np.ones((data.shape[0], data.shape[1] + 1, data.shape[2]), dtype=bool)

    with pytest.raises(ValueError):
        delta_svd.fit_voxelwise(delta_svd.wls_fit_tensor_fw, data,
                                {'md_data': md, 'S0': S0, 'mask': bad},
                                {'W': W, 'mdreg': 2.0e-3}, 4)


# ---------------------------------------------------------------------------
# Mask merging

def test_merge_masks_takes_voxelwise_union(delta_svd, tmp_path):
    affine = np.eye(4)
    m1 = np.array([[[1, 0], [0, 0]]], dtype="uint8")
    m2 = np.array([[[0, 0], [0, 1]]], dtype="uint8")
    fn1 = tmp_path / "m1.nii.gz"
    fn2 = tmp_path / "m2.nii.gz"
    nib.save(nib.Nifti1Image(m1, affine), str(fn1))
    nib.save(nib.Nifti1Image(m2, affine), str(fn2))

    out = tmp_path / "merged.nii.gz"
    fnOut = delta_svd.merge_masks([str(fn1), str(fn2)], str(out))

    result = nib.load(fnOut).get_fdata()
    assert np.array_equal(result, np.array([[[1, 0], [0, 1]]]))


def test_coreg_merge_masks_single_timepoint_binarises(delta_svd, tmp_path):
    tp_dir = tmp_path / "TP01"
    tp_dir.mkdir()
    affine = np.eye(4)
    src = tmp_path / "roi.nii.gz"
    img = np.array([[[0, 2], [0, -3]]], dtype=float)
    nib.save(nib.Nifti1Image(img, affine), str(src))

    fnOut = delta_svd.coreg_merge_masks(
        timepoints=[str(tp_dir)], masks=[str(src)], label="mask_test",
        dirTemplate=str(tmp_path), binarise=True,
    )

    result = nib.load(fnOut).get_fdata()
    assert np.array_equal(result, (img > 0).astype(float))


def test_coreg_merge_masks_returns_none_when_no_masks_provided(delta_svd, tmp_path):
    tp_dir = tmp_path / "TP01"
    tp_dir.mkdir()
    result = delta_svd.coreg_merge_masks(
        timepoints=[str(tp_dir)], masks=[None], label="x", dirTemplate=str(tmp_path),
    )
    assert result is None


# An uncompressed '.nii' mask must not be carried into the temp tree under its
# own extension: with a single timepoint there is no antsApplyTransforms step to
# normalise it, and every downstream consumer (tbss_non_FA in particular) assumes
# '.nii.gz'. Regression test for a mask silently ending up as 'mask_test.nii'.
@pytest.mark.parametrize("binarise", [True, False])
def test_coreg_merge_masks_normalises_uncompressed_input_to_nii_gz(delta_svd, tmp_path, binarise):
    tp_dir = tmp_path / "TP01"
    tp_dir.mkdir()
    src = tmp_path / "roi.nii"
    img = np.array([[[0, 2], [0, 1]]], dtype="uint8")
    nib.save(nib.Nifti1Image(img, np.eye(4)), str(src))

    fnOut = delta_svd.coreg_merge_masks(
        timepoints=[str(tp_dir)], masks=[str(src)], label="mask_test",
        dirTemplate=str(tmp_path), binarise=binarise,
    )

    assert fnOut.endswith(".nii.gz")
    expected = (img > 0) if binarise else img
    assert np.array_equal(nib.load(fnOut).get_fdata(), expected.astype(float))


# ---------------------------------------------------------------------------
# copy_as_nii_gz(): the destination name always claims gzip, so an uncompressed
# input has to be re-encoded rather than copied - nibabel, FSL and ANTs all pick
# the codec from the file name and fail on a plain NIfTI named '.nii.gz'.

def test_copy_as_nii_gz_copies_gzipped_input_verbatim(delta_svd, tmp_path):
    src = tmp_path / "in.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), "uint8"), np.eye(4)), str(src))
    dst = tmp_path / "out.nii.gz"

    delta_svd.copy_as_nii_gz(str(src), str(dst))

    assert dst.read_bytes() == src.read_bytes()


def test_copy_as_nii_gz_reencodes_uncompressed_input(delta_svd, tmp_path):
    img = np.array([[[1, 0], [0, 3]]], dtype="uint8")
    src = tmp_path / "in.nii"
    nib.save(nib.Nifti1Image(img, np.eye(4)), str(src))
    dst = tmp_path / "out.nii.gz"

    delta_svd.copy_as_nii_gz(str(src), str(dst))

    assert dst.read_bytes()[:2] == b"\x1f\x8b"          # really gzipped
    assert np.array_equal(nib.load(str(dst)).get_fdata(), img.astype(float))
    assert nib.load(str(dst)).get_data_dtype() == img.dtype  # dtype preserved by default


# ---------------------------------------------------------------------------
# binarise_mask()
#
# The threshold at 1 and the uint8 writes only behave on a strictly {0,1} mask.
# Binarising on the way in makes that true, and must stay a no-op for a mask
# that already is - the case every validated run was made with.

@pytest.mark.parametrize("values", [
    [0.0, 1.0, 1.0, 0.0],                                   # float 0/1
    np.array([0, 1, 1, 0], dtype="uint8"),                  # integer 0/1
    [0.0, 0.0, 0.0, 0.0],                                   # empty mask
])
def test_binarise_mask_leaves_a_binary_mask_untouched(delta_svd, capsys, values):
    img = np.asarray(values, dtype=float)

    out = delta_svd.binarise_mask(img, "brain mask", "mask.nii.gz")

    assert np.array_equal(out.astype(float), img)           # values unchanged
    assert capsys.readouterr().out == ""                    # and reported as nothing to do


@pytest.mark.parametrize("values, expected", [
    ([0.0, 255.0, 255.0, 0.0], [0, 1, 1, 0]),               # ITK-style 0/255 mask
    ([0.0, 0.3, 0.9, 1.0], [0, 1, 1, 1]),                   # probabilistic mask
    ([0.0, 2.0, 0.0, 7.0], [0, 1, 0, 1]),                   # arbitrary labels
    ([-2.0, -0.1, 0.0, 1.0], [0, 0, 0, 1]),                # negative values are background
])
def test_binarise_mask_binarises_and_reports_anything_else(delta_svd, capsys, values, expected):
    out = delta_svd.binarise_mask(np.asarray(values), "brain mask", "mask.nii.gz")

    assert out.astype(int).tolist() == expected
    assert "other than 0 and 1" in capsys.readouterr().out


def test_binarise_mask_write_path_is_byte_identical_for_a_binary_mask(delta_svd, tmp_path):
    # For a {0,1} mask this has to produce the very same file as before, or the
    # change would move the validated numbers.
    img = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    src = tmp_path / "bmask.nii.gz"
    nib.save(nib.Nifti1Image(img, np.eye(4)), str(src))
    nii = nib.load(str(src))

    plain = tmp_path / "plain.nii.gz"
    binarised = tmp_path / "binarised.nii.gz"
    delta_svd.save_nifti(str(plain), nii.get_fdata(), nii.affine, nii.header, dtype="uint8")
    delta_svd.save_nifti(str(binarised), delta_svd.binarise_mask(nii.get_fdata(), "brain mask", str(src)),
                         nii.affine, nii.header, dtype="uint8")

    assert binarised.read_bytes() == plain.read_bytes()


# ---------------------------------------------------------------------------
# "steps must be contiguous" / "--qc 0 contradicts --steps qc" error paths

def _minimal_pipeline_inputs(tmp_path):
    dwi = tmp_path / "sub01.nii.gz"
    skel = tmp_path / "skel.nii.gz"
    for fn in [
        dwi, tmp_path / "sub01.bval", tmp_path / "sub01.bvec",
        tmp_path / "sub01_brainmask.nii.gz", skel,
    ]:
        fn.touch()
    return dwi, skel


def test_pipeline_rejects_noncontiguous_steps(delta_svd, tmp_path, monkeypatch):
    dwi, skel = _minimal_pipeline_inputs(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "delta-svd.py", "--dwi", str(dwi), "--skeletonMask", str(skel),
        "--steps", "fwc", "extract",
    ])
    with pytest.raises(ValueError, match="have to be contiguous"):
        delta_svd.pipeline_delta_svd()


def test_pipeline_rejects_qc_zero_with_qc_step_requested(delta_svd, tmp_path, monkeypatch):
    dwi, skel = _minimal_pipeline_inputs(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "delta-svd.py", "--dwi", str(dwi), "--skeletonMask", str(skel),
        "--steps", "qc", "--qc", "0",
    ])
    with pytest.raises(ValueError, match="contradictory"):
        delta_svd.pipeline_delta_svd()


# ---------------------------------------------------------------------------
# A repeated label makes two time-points share a working folder and their rows:
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

    assert "qc" in delta_svd.stepsImplemented
    # and '--steps qc' is still an accepted choice afterwards
    ns = delta_svd.iniParser().parse_args(
        ["--dwi", str(dwi), "--skeletonMask", str(skel), "--steps", "qc"])
    assert ns.steps == ["qc"]


# ---------------------------------------------------------------------------
# --numRegistrations reaches an ANTs shell command, so it is validated like the
# other numeric options rather than accepting any int.

def test_numRegistrations_accepts_positive_integer(delta_svd):
    assert delta_svd.assertPositiveRegistrations("5") == 5


@pytest.mark.parametrize("bad", ["0", "-2", "many"])
def test_numRegistrations_rejects_non_positive_and_non_integer(delta_svd, bad):
    with pytest.raises(argparse.ArgumentTypeError):
        delta_svd.assertPositiveRegistrations(bad)


class _StopAfterCommand(Exception):
    """Abort create_template once the ANTs command has been captured."""


def _capture_ants_command(delta_svd, tmp_path, monkeypatch, **kwargs):
    calls = []

    def record(cmd, *a, **k):
        calls.append(cmd)
        raise _StopAfterCommand

    monkeypatch.setattr(delta_svd, "run_subprocess", record)
    monkeypatch.setattr(delta_svd, "copy2", lambda *a, **k: None)
    tp = tmp_path / "TP01"
    tp.mkdir(exist_ok=True)
    with pytest.raises(_StopAfterCommand):
        delta_svd.create_template(timepoints=[str(tp)], fnCoreg=[], dirOut=str(tmp_path),
                                  coreBudget=1, **kwargs)
    return calls[0]


def test_default_iterations_reach_ants_unquoted_and_unchanged(delta_svd, tmp_path, monkeypatch):
    # shlex.quote is a no-op for the default, so the command stays byte-identical
    cmd = _capture_ants_command(delta_svd, tmp_path, monkeypatch,
                                iterations="30x30x8", numRegistrations=3)
    assert " -i 3 " in cmd
    assert " -q 30x30x8 " in cmd


def test_iterations_with_shell_metacharacters_cannot_break_out(delta_svd, tmp_path, monkeypatch):
    cmd = _capture_ants_command(delta_svd, tmp_path, monkeypatch,
                                iterations="30x30x8; rm -rf /", numRegistrations=3)
    assert " -q '30x30x8; rm -rf /' " in cmd


# ---------------------------------------------------------------------------
# "Hemispheric ROI analysis" message should only print when --hemispheres
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
    assert "Hemispheric ROI analysis" not in capsys.readouterr().out


def test_hemispheres_message_printed_when_flag_set(delta_svd, tmp_path, monkeypatch, capsys):
    dwi, skel = _minimal_pipeline_inputs(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "delta-svd.py", "--dwi", str(dwi), "--skeletonMask", str(skel),
        "--hemispheres", "--steps", "qc", "--qc", "0",
    ])
    with pytest.raises(ValueError):
        delta_svd.pipeline_delta_svd()
    assert "Hemispheric ROI analysis" in capsys.readouterr().out


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


def test_candidate_paths_re_anchors_only_what_the_user_typed(delta_svd):
    dwi = "study/sub01/data.nii.gz"
    # inferred: already carries the folder, so it is probed as-is only
    assert delta_svd.candidate_paths("study/sub01/data_brainmask", dwi, True, inferred=True) == [
        "study/sub01/data_brainmask.nii.gz", "study/sub01/data_brainmask.nii"]
    # typed: a bare basename has to be resolvable against the DWI folder
    assert delta_svd.candidate_paths("m.nii.gz", dwi, True, inferred=False) == [
        "m.nii.gz", "study/sub01/m.nii.gz"]


def test_explicit_basename_still_resolves_against_the_dwi_folder(delta_svd, tmp_path, monkeypatch, capsys):
    # the feature the second probe exists for must survive the fix, relative too
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
    assert "Bmask :study/sub01/mask.nii.gz" in out


# ---------------------------------------------------------------------------
# The extension probing appends to the name as given, so for 'data.nii' it looks
# for 'data.nii.nii.gz' -- it never reaches 'data.nii.gz'. The message must not
# claim otherwise, and the sibling that is actually there is worth naming.

def test_isNIfTI_names_the_sibling_with_the_other_extension(delta_svd, tmp_path):
    (tmp_path / "data.nii.gz").touch()
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
        delta_svd.isNIfTI(str(tmp_path / "data.nii"))

    msg = str(excinfo.value)
    assert "also tried" not in msg
    assert str(tmp_path / "data.nii.gz") in msg


def test_isNIfTI_names_the_sibling_in_the_other_direction(delta_svd, tmp_path):
    (tmp_path / "data.nii").touch()
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
        delta_svd.isNIfTI(str(tmp_path / "data.nii.gz"))

    assert str(tmp_path / "data.nii") in str(excinfo.value)


def test_isNIfTI_does_not_claim_probes_it_did_not_make(delta_svd, tmp_path):
    # nothing there at all, and the name already carries an extension
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
        delta_svd.isNIfTI(str(tmp_path / "data.nii"))

    assert "also tried" not in str(excinfo.value)


def test_isNIfTI_still_reports_the_probes_for_an_extensionless_name(delta_svd, tmp_path):
    # here the probes are real, and naming them tells the user what was searched
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
        delta_svd.isNIfTI(str(tmp_path / "data"))

    msg = str(excinfo.value)
    assert str(tmp_path / "data.nii.gz") in msg
    assert str(tmp_path / "data.nii") in msg


def test_nifti_sibling_returns_none_without_a_counterpart(delta_svd, tmp_path):
    (tmp_path / "data.nii.gz").touch()
    assert delta_svd.nifti_sibling(str(tmp_path / "other.nii")) is None
    assert delta_svd.nifti_sibling(str(tmp_path / "notes.txt")) is None
    # the file asked for is never substituted, only reported
    assert delta_svd.nifti_sibling(str(tmp_path / "data.nii")) == str(tmp_path / "data.nii.gz")


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
    # probed relative to the DWI folder as well, and that is reported
    assert str(tmp_path / "nowhere.nii.gz") in msg


# ---------------------------------------------------------------------------
# Every user-fixable error is a DeltaSvdError, so the __main__ guard can report
# it without a traceback. A ValueError from anywhere else is a bug and has to
# keep its traceback, so the guard must not swallow it.

def test_delta_svd_error_is_a_value_error(delta_svd):
    assert issubclass(delta_svd.DeltaSvdError, ValueError)


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
