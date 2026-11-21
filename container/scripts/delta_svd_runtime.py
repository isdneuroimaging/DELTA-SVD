#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image environment, core detection and the CPU check for delta-svd.py.

Definitions only: delta-svd.py calls them in its start-up statements, before
the thread pinning and the first third-party import. So this module imports
the standard library and delta_svd_constants only.
"""

import os, sys, argparse
import importlib.util
from os.path import join, dirname

from delta_svd_constants import (ITK_THREADS_DEFAULT,
                                 IMAGE_SCRIPT_DIR, IMAGE_ENVIRONMENT)


def running_in_image(scriptPath=None):
    """True when this is the copy of the script installed in the image."""
    if scriptPath is None:
        scriptPath = __file__
    return dirname(os.path.realpath(scriptPath)) == IMAGE_SCRIPT_DIR


def fsl_sub_config(fsldir):
    """The fsl_sub configuration the image resolves to by itself, i.e. with no
    FSLSUB_CONF and no ~/.fsl_sub.yml (as under Docker, where the image user has
    no home): the first non-empty file of fsl_sub's own search path after those
    two. Located without importing fsl_sub. None if neither exists."""
    candidates = [join(fsldir, 'etc', 'fslconf', 'fsl_sub.yml')]
    spec = importlib.util.find_spec('fsl_sub')
    if spec is not None and spec.submodule_search_locations:
        candidates.append(join(list(spec.submodule_search_locations)[0],
                               'plugins', 'fsl_sub_shell.yml'))
    for p in candidates:
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            return p
    return None


def pin_image_environment(environ, fslSubConfig=fsl_sub_config):
    """Give 'environ' the image's own settings, whatever came in from the host.

    Apptainer passes in host variables the image does not set (PYTHONPATH,
    FSLSUB_*) and mounts the host's home, and the image's ENV values can still
    be replaced on purpose (see IMAGE_ENVIRONMENT). Left alone, that can run
    the host's FSL (FSLDIR: the TBSS scripts call
    $FSLDIR/bin/...) or ANTs (ANTSPATH: the template script calls
    ${ANTSPATH}/antsRegistration), another BLAS kernel (OPENBLAS_CORETYPE), the
    host's Python packages (PYTHONPATH, ~/.local), or the host's fsl_sub
    settings (FSLSUB_*). Every one of these changes the metrics or breaks the
    run, so they are assigned or removed here rather than defaulted. Also
    inherited by every FSL and ANTs subprocess.

    Not covered: a ~/.fsl_sub.yml in the mounted home. fsl_sub merges it over
    whatever configuration it loads, FSLSUB_CONF included. It can only change
    how jobs are launched, not what they compute; asking for a cluster
    scheduler, for which the image has no plugin, makes fsl_sub fail."""
    for name in [n for n in environ if n.startswith(('PYTHON', 'FSLSUB_'))]:
        del environ[name]
    environ.update(IMAGE_ENVIRONMENT)
    # the file fsl_sub would find by itself, named explicitly so a host value
    # cannot select another one
    conf = fslSubConfig(environ['FSLDIR'])
    if conf is not None:
        environ['FSLSUB_CONF'] = conf


def isolated_interpreter_argv(flags, argv, script):
    """The command re-running 'script' as 'python -E -s -u', or None when the
    interpreter already ignores PYTHON* variables and the user site-packages.
    Otherwise numpy and the rest could be imported from the host's PYTHONPATH or
    ~/.local instead of the image. -u stands in for PYTHONUNBUFFERED, which -E
    ignores."""
    if flags.ignore_environment and flags.no_user_site:
        return None
    return [sys.executable, '-E', '-s', '-u', script] + list(argv)


def detect_physical_cores():
    """Number of *physical* CPU cores this process may use, honouring CPU affinity
    (an HPC scheduler's cpuset). Hyperthreads are deliberately not counted: ANTs
    registration gains little from SMT. Falls back to the affinity size. Capped
    by a cgroup CPU quota, if the process runs under one."""
    try:
        allowed = os.sched_getaffinity(0)          # logical CPUs this process may use
    except AttributeError:                         # non-Linux platforms
        allowed = set(range(os.cpu_count() or 1))
    try:
        cores = set()
        cur = {}
        with open('/proc/cpuinfo') as fh:
            for line in fh:
                line = line.strip()
                if not line:                       # blank line ends one processor block
                    if cur.get('processor') in allowed and 'physical id' in cur and 'core id' in cur:
                        cores.add((cur['physical id'], cur['core id']))
                    cur = {}
                    continue
                key, _, val = line.partition(':')
                key, val = key.strip(), val.strip()
                if key == 'processor':
                    cur['processor'] = int(val)
                elif key in ('physical id', 'core id'):
                    cur[key] = int(val)
        if not cores:
            raise ValueError('no core topology in /proc/cpuinfo')
        count = len(cores)
    except (OSError, ValueError):
        count = max(1, len(allowed))                # fallback: assume no SMT
    # A CPU quota (Docker '--cpus', a Kubernetes CPU limit) leaves the affinity
    # untouched, so it has to be read separately; without it the budget would
    # be every core of the host.
    quota = cgroup_cpu_limit()
    if quota is not None:
        count = min(count, max(1, int(-(-quota // 1))))      # rounded up
    return count


def cgroup_cpu_limit(procCgroup='/proc/self/cgroup', root='/sys/fs/cgroup'):
    """CPUs' worth of time the cgroup CPU quota allows this process, or None
    without a quota. The tightest of its own cgroup and every ancestor counts.
    Covers cgroup v2 ('cpu.max') and v1 ('cpu.cfs_quota_us')."""
    try:
        with open(procCgroup) as fh:
            entries = [line.rstrip('\n').split(':', 2) for line in fh if line.count(':') >= 2]
    except OSError:
        return None

    def read(path):
        try:
            with open(path) as fh:
                return fh.read().split()
        except OSError:
            return None

    def ratio(quota, period):
        try:
            return int(quota) / int(period)
        except (ValueError, ZeroDivisionError):
            return -1

    def ancestors(base, rel):
        parts = [p for p in rel.split('/') if p]
        return [os.path.join(base, *parts[:n]) for n in range(len(parts), -1, -1)]

    limits = []
    for _, controllers, rel in entries:
        if controllers == '':                                    # cgroup v2
            for d in ancestors(root, rel):
                val = read(os.path.join(d, 'cpu.max'))
                if val and len(val) == 2 and val[0] != 'max':
                    limits.append(ratio(val[0], val[1]))
        elif 'cpu' in controllers.split(','):                   # cgroup v1
            for mount in ('cpu', 'cpu,cpuacct', 'cpuacct,cpu'):
                for d in ancestors(os.path.join(root, mount), rel):
                    quota = read(os.path.join(d, 'cpu.cfs_quota_us'))
                    period = read(os.path.join(d, 'cpu.cfs_period_us'))
                    if quota and period:
                        limits.append(ratio(quota[0], period[0]))    # -1: no quota
    positive = [x for x in limits if x > 0]
    return min(positive) if positive else None


def resolve_thread_budget(argv):
    """Core budget from --threads, auto-detected when absent or 'auto'.
    Resolved by delta-svd.py at import, ahead of the full parser, alongside the ITK count."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--threads', default=None)
    ns, _ = pre.parse_known_args(argv)
    val = ns.threads
    if val is None or str(val).strip().lower() == 'auto':
        return detect_physical_cores()
    try:
        return max(1, int(val))
    except ValueError:
        return detect_physical_cores()              # the full parser reports the error later


def resolve_itk_threads(argv):
    """ITK threads per registration job from --itkThreads. Resolved alongside the
    core budget because delta-svd.py exports it into the environment at import."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--itkThreads', default=None)
    ns, _ = pre.parse_known_args(argv)
    if ns.itkThreads is None:
        return ITK_THREADS_DEFAULT
    try:
        return max(1, int(ns.itkThreads))
    except ValueError:
        return ITK_THREADS_DEFAULT                  # the full parser reports the error later


#--- CPU features each pinned OpenBLAS kernel family needs to execute at all
CORETYPE_REQUIRED_FLAGS = {'haswell': ('avx2', 'fma'), 'zen': ('avx2', 'fma'),
                           'skylakex': ('avx512f',), 'sandybridge': ('avx',)}


def check_cpu_supports_coretype(coretype=None, cpuinfo='/proc/cpuinfo'):
    """Fail early, and legibly, if the CPU cannot run the pinned BLAS kernel.

    The image pins OPENBLAS_CORETYPE so the numerics cannot vary with the CPU
    (see the Dockerfile). Asking OpenBLAS for a kernel the hardware lacks does
    not fall back -- it dies with SIGILL and no message -- so the requirement is
    checked here instead. Returns the missing flags, empty when all is well."""
    coretype = os.environ.get('OPENBLAS_CORETYPE') if coretype is None else coretype
    required = CORETYPE_REQUIRED_FLAGS.get(str(coretype).strip().lower(), ())
    if not required:
        return ()                                  # unpinned, or a family we make no claim about
    try:
        with open(cpuinfo) as fh:
            flags = set(fh.read().split())
    except OSError:
        return ()                                  # not Linux: leave it to OpenBLAS
    return tuple(f for f in required if f not in flags)
