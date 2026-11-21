import io
import os

import pytest

import delta_svd_runtime


# ---------------------------------------------------------------------------
# --itkThreads resolution

def test_resolve_itk_threads_defaults_when_absent():
    assert delta_svd_runtime.resolve_itk_threads([]) == delta_svd_runtime.ITK_THREADS_DEFAULT
    assert delta_svd_runtime.resolve_itk_threads(["--dwi", "a.nii.gz"]) == delta_svd_runtime.ITK_THREADS_DEFAULT


def test_resolve_itk_threads_reads_an_explicit_value():
    assert delta_svd_runtime.resolve_itk_threads(["--itkThreads", "6"]) == 6
    assert delta_svd_runtime.resolve_itk_threads(["--dwi", "a.nii.gz", "--itkThreads", "1"]) == 1


def test_resolve_itk_threads_falls_back_on_garbage():
    # the full parser reports the error properly; this pre-pass must not crash
    # before numpy is even imported
    assert delta_svd_runtime.resolve_itk_threads(["--itkThreads", "many"]) == delta_svd_runtime.ITK_THREADS_DEFAULT


def test_cpu_check_passes_when_the_required_flags_are_present(tmp_path):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("flags : sse2 avx avx2 fma bmi2\n")
    assert delta_svd_runtime.check_cpu_supports_coretype("Haswell", str(cpuinfo)) == ()


def test_cpu_check_names_every_missing_flag(tmp_path):
    # OpenBLAS would die with SIGILL and no message; the check exists to say why
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("flags : sse2 sse4_2\n")
    assert set(delta_svd_runtime.check_cpu_supports_coretype("Haswell", str(cpuinfo))) == {"avx2", "fma"}


def test_cpu_check_is_silent_when_the_kernel_is_not_pinned(tmp_path):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("flags : sse2\n")
    assert delta_svd_runtime.check_cpu_supports_coretype("", str(cpuinfo)) == ()
    assert delta_svd_runtime.check_cpu_supports_coretype("SomeFutureKernel", str(cpuinfo)) == ()


def test_cpu_check_does_not_fail_without_proc_cpuinfo(tmp_path):
    # non-Linux hosts have no /proc/cpuinfo; leave the decision to OpenBLAS
    assert delta_svd_runtime.check_cpu_supports_coretype("Haswell", str(tmp_path / "absent")) == ()


def test_detect_physical_cores_returns_positive_int():
    n = delta_svd_runtime.detect_physical_cores()
    assert isinstance(n, int) and n >= 1


# ---------------------------------------------------------------------------
# cgroup_cpu_limit(): a CPU quota (Docker '--cpus', a Kubernetes CPU limit)
# does not show in the affinity mask, so it is read from the cgroup files.

def _cgroup_file(root, rel, name, text):
    d = root.joinpath(*[p for p in rel.split("/") if p])
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(text)


def test_cgroup_v2_quota_takes_the_tightest_ancestor(tmp_path):
    proc = tmp_path / "cgroup"
    proc.write_text("0::/user.slice/job\n")
    root = tmp_path / "fs"
    _cgroup_file(root, "/user.slice/job", "cpu.max", "150000 100000\n")
    _cgroup_file(root, "/user.slice", "cpu.max", "400000 100000\n")
    assert delta_svd_runtime.cgroup_cpu_limit(str(proc), str(root)) == 1.5


def test_cgroup_v2_without_quota(tmp_path):
    proc = tmp_path / "cgroup"
    proc.write_text("0::/\n")
    root = tmp_path / "fs"
    _cgroup_file(root, "/", "cpu.max", "max 100000\n")
    assert delta_svd_runtime.cgroup_cpu_limit(str(proc), str(root)) is None


@pytest.mark.parametrize("quota, expected", [("-1", None), ("200000", 2.0)])
def test_cgroup_v1_quota(tmp_path, quota, expected):
    proc = tmp_path / "cgroup"
    proc.write_text("4:memory:/job\n3:cpu,cpuacct:/job\n0::/\n")
    root = tmp_path / "fs"
    _cgroup_file(root / "cpu,cpuacct", "/job", "cpu.cfs_quota_us", quota + "\n")
    _cgroup_file(root / "cpu,cpuacct", "/job", "cpu.cfs_period_us", "100000\n")
    assert delta_svd_runtime.cgroup_cpu_limit(str(proc), str(root)) == expected


def test_cgroup_limit_none_without_cgroup_information(tmp_path):
    assert delta_svd_runtime.cgroup_cpu_limit(str(tmp_path / "missing"), str(tmp_path)) is None


def test_detect_physical_cores_is_capped_by_the_quota(monkeypatch):
    monkeypatch.setattr(delta_svd_runtime, "cgroup_cpu_limit", lambda: None)
    uncapped = delta_svd_runtime.detect_physical_cores()
    monkeypatch.setattr(delta_svd_runtime, "cgroup_cpu_limit", lambda: 0.5)
    assert delta_svd_runtime.detect_physical_cores() == 1          # never below one core
    monkeypatch.setattr(delta_svd_runtime, "cgroup_cpu_limit", lambda: 1000.0)
    assert delta_svd_runtime.detect_physical_cores() == uncapped   # a loose quota changes nothing


# ---------------------------------------------------------------------------
# resolve_thread_budget() / detect_physical_cores() run at import, before the
# real parser, so they must never raise on bad input: argparse reports it later.

def test_resolve_thread_budget_autodetects_when_threads_absent(monkeypatch):
    monkeypatch.setattr(delta_svd_runtime, "detect_physical_cores", lambda: 99)
    assert delta_svd_runtime.resolve_thread_budget([]) == 99


@pytest.mark.parametrize("value", ["auto", "AUTO", " auto "])
def test_resolve_thread_budget_autodetects_for_auto(monkeypatch, value):
    monkeypatch.setattr(delta_svd_runtime, "detect_physical_cores", lambda: 99)
    assert delta_svd_runtime.resolve_thread_budget(["--threads", value]) == 99


def test_resolve_thread_budget_uses_explicit_integer(monkeypatch):
    monkeypatch.setattr(delta_svd_runtime, "detect_physical_cores", lambda: 99)
    assert delta_svd_runtime.resolve_thread_budget(["--threads", "8"]) == 8
    assert delta_svd_runtime.resolve_thread_budget(["--threads", " 8 "]) == 8


def test_resolve_thread_budget_clamps_negative_to_one(monkeypatch):
    monkeypatch.setattr(delta_svd_runtime, "detect_physical_cores", lambda: 99)
    assert delta_svd_runtime.resolve_thread_budget(["--threads", "-4"]) == 1


def test_resolve_thread_budget_falls_back_on_garbage(monkeypatch):
    # 'lots' is rejected later by threadBudget(); here it must only fall back,
    # because raising would abort before argparse can report the real error.
    monkeypatch.setattr(delta_svd_runtime, "detect_physical_cores", lambda: 99)
    assert delta_svd_runtime.resolve_thread_budget(["--threads", "lots"]) == 99


def test_resolve_thread_budget_ignores_the_rest_of_the_command_line(monkeypatch):
    monkeypatch.setattr(delta_svd_runtime, "detect_physical_cores", lambda: 99)
    argv = ["--dwi", "sub01.nii.gz", "--threads", "4", "--steps", "1", "2", "-o", "/out"]
    assert delta_svd_runtime.resolve_thread_budget(argv) == 4


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


def test_detect_physical_cores_ignores_hyperthread_siblings(monkeypatch):
    _fake_cpuinfo(monkeypatch, _CPUINFO_2CORE_SMT, allowed=[0, 1, 2, 3])
    assert delta_svd_runtime.detect_physical_cores() == 2


def test_detect_physical_cores_honours_cpu_affinity(monkeypatch):
    # A scheduler cpuset handing us CPUs 0 and 2 - the two siblings of one core -
    # is a budget of one core, not two.
    _fake_cpuinfo(monkeypatch, _CPUINFO_2CORE_SMT, allowed=[0, 2])
    assert delta_svd_runtime.detect_physical_cores() == 1


def test_detect_physical_cores_falls_back_to_affinity_size(monkeypatch):
    # No /proc/cpuinfo (non-Linux, or a locked-down container): assume no SMT
    # rather than crashing at import time.
    _fake_cpuinfo(monkeypatch, None, allowed=[0, 1, 2])
    assert delta_svd_runtime.detect_physical_cores() == 3
