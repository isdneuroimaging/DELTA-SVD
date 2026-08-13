#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys, argparse, re, subprocess, time, glob, shlex, multiprocessing
from os.path import join, exists, dirname, basename
from shutil import copy2, rmtree
from pathlib import Path


def detect_physical_cores():
    """Number of *physical* CPU cores this process may use, honouring CPU affinity
    (an HPC scheduler's cpuset). Hyperthreads are deliberately not counted: ANTs
    registration gains little from SMT. Falls back to the affinity size."""
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
        if cores:
            return len(cores)
    except (OSError, ValueError):
        pass
    return max(1, len(allowed))                     # fallback: assume no SMT


def resolve_thread_budget(argv):
    """Core budget from --threads, auto-detected when absent or 'auto'. Kept
    dependency-free so it can run *before* numpy/OpenBLAS are imported."""
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


# ITK threads per registration job. This is the one threading quantity that
# reaches the metric values: ITK sums the registration metric and its gradient
# per thread, so a different count sums them in a different order and the last
# bits move. The skeleton amplifies that from there -- 1 thread instead of 12
# shifts delta-PSMD by ~23% on the reference subject. 12 is the value the method
# was validated at; see CONTRIBUTING.md before changing it.
ITK_THREADS_DEFAULT = 12

# How far the template step may oversubscribe the core budget, as a fraction:
# 3/2 = 1.5 threads per core. Measured faster than an exactly-fitting plan, and
# without numerical consequence, so this is purely a throughput choice.
OVERSUBSCRIBE_NUM, OVERSUBSCRIBE_DEN = 3, 2


def resolve_itk_threads(argv):
    """ITK threads per registration job from --itkThreads. Resolved alongside the
    core budget because it is exported into the environment below."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--itkThreads', default=None)
    ns, _ = pre.parse_known_args(argv)
    if ns.itkThreads is None:
        return ITK_THREADS_DEFAULT
    try:
        return max(1, int(ns.itkThreads))
    except ValueError:
        return ITK_THREADS_DEFAULT                  # the full parser reports the error later


BLAS_THREADS = 1

# Pin the BLAS/OpenMP thread pools *before* importing numpy/dipy: OpenBLAS reads
# these when first loaded, and setting them later is ignored. Fixed at 1 rather
# than at the core budget, so no thread count anywhere in the numerics can vary
# with the machine: the per-voxel fits are parallelised by process instead (see
# fit_voxelwise) and their matrices are far too small to thread anyway, so
# nothing is given up. Assigned rather than setdefault(), for the reason given
# for the ITK count below.
CORE_BUDGET = resolve_thread_budget(sys.argv[1:])
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_var] = str(BLAS_THREADS)

# Every ANTs call inherits this, not only the template step: antsApplyTransforms
# would otherwise fall back to the hardware concurrency -- or, on a Grid Engine
# cluster, to NSLOTS, which ITK also consults -- putting a metric-affecting
# thread count outside this pipeline's control. Assigned rather than
# setdefault(): a value forwarded in from the host (Apptainer passes the whole
# environment through by default) must not be able to change the results.
ITK_THREADS = resolve_itk_threads(sys.argv[1:])
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(ITK_THREADS)


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


_missing = check_cpu_supports_coretype()
if _missing:
    sys.exit(f"ERROR: this CPU does not support the instructions DELTA-SVD needs.\n"
             f"  Missing: {', '.join(_missing)}\n"
             f"  DELTA-SVD requires an x86-64-v3 CPU (AVX2 and FMA): Intel Haswell (2013) or\n"
             f"  newer, AMD Zen (2017) or newer. Note that Atom, Celeron and Pentium parts\n"
             f"  often lack these regardless of age.\n"
             f"  The BLAS kernel is pinned so results cannot vary between machines; running\n"
             f"  without it would silently produce different metrics.")

import nibabel as nib
import numpy as np
import pandas as pd

from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from dipy.reconst.dti import (design_matrix, decompose_tensor,
                           from_lower_triangular)

from scipy.ndimage import gaussian_filter

from markvcid_fw_mrn import wls_fit_tensor_fw, wls_fit_dti
from delta_svd_version import __version__

###########################################################################
# Functions for reading/writing bval/bvec files

def read_bval_or_bvec(fname):

    with open(fname, "r") as file:
        ll = file.readlines()

    for i,l in enumerate(ll):
        ll[i] = l.split()

    ll = np.array(ll)

    if ll.shape[0]==1:
        ll = ll[0]
    elif ll.shape[0]==3:
        ll = ll.T
    else:
        raise ValueError('Input file has to contain either one row (for bval files) or three rows (for bvec files).')

    arrStr   = ll
    arrFloat = ll.astype('float')
    
    return arrFloat, arrStr


def write_bval_or_bvec(arrStr, fname):
    with open(fname, "w") as file:
        if arrStr.ndim==2:
            for i in range(arrStr.shape[1]):
                line = " ".join([x for x in arrStr[:,i]])+'\n'
                file.write(line)
        elif arrStr.ndim==1:
            line = " ".join([x for x in arrStr])+'\n'
            file.write(line)



###########################################################################
# Functions for major processing steps


#--- b-values at or below this count as b = 0: they carry no usable diffusion
#    weighting, and are what S0 is averaged from. Always kept, whatever b-value
#    selection the user asks for.
B0_MAX = 5

#--- Scanners report b-values that deviate from the nominal shell (rounding, and
#    cross-terms with the imaging gradients), so a requested limit is met with a
#    tolerance. A range carries slack in its endpoints already, so it only needs
#    enough to absorb rounding; a shell is a point, where the tolerance is the
#    whole acceptance window, so it gets more. Neither can merge adjacent shells,
#    which sit at least 100 s/mm2 apart in practice.
BRANGE_TOL = 5
SHELL_TOL = 25

#--- The window the diffusion-tensor model is valid in. Below the floor the
#    signal is contaminated by perfusion (IVIM), above the ceiling by
#    non-Gaussian diffusion; a tensor fitted outside it is not interpretable, so
#    b-value selections beyond these are refused rather than fitted.
BVAL_MIN = 250
BVAL_MAX = 1800

#--- Identifiability of the fits.
#
#    The design matrix both fits solve has seven columns -- the six tensor
#    components plus the log-S0 intercept (see markvcid_fw_mrn.wls_iter_fw) --
#    so rank 7 is what "the tensor is estimable at all" means, and six
#    non-collinear directions reach it. The free-water fraction is not part of
#    that linear solve: it is grid-searched and scored by the residual, so it is
#    constrained only by *distinct* design rows. Repeated acquisitions of one
#    direction average noise but add no constraint on it -- hence the floor
#    counts unique directions, not volumes.
#
#    Twelve is the lowest direction count of a real clinical DTI protocol: below
#    it a dataset is far more likely truncated, corrupted, or over-filtered than
#    deliberately acquired, so it is refused. Between twelve and the twenty
#    directions docs/requirements.md recommends the fit works but the free-water
#    fraction is noisy, which is a warning rather than an error. None of this
#    fails on its own: the bi-tensor fit solves with a pseudo-inverse, which
#    returns a minimum-norm solution for a degenerate gradient table instead of
#    raising, so an unchecked run produces plausible-looking numbers.
DESIGN_MATRIX_RANK = 7
MIN_DIRECTIONS = 12
RECOMMENDED_DIRECTIONS = 20

#--- Two gradient directions count as one below an angle of ~2.6 degrees. Even a
#    128-direction scheme separates its directions by more than 10 degrees, so
#    this can only ever merge genuine repeats.
DIRECTION_COS_TOL = 0.999


def resolve_b_intervals(bRange=None, shells=None):
    """The accepted non-zero b-value windows, tolerance included.

    One window for '--bRange', one per shell for '--shells'; the two options are
    mutually exclusive on the command line."""
    if shells:
        return [(float(s) - SHELL_TOL, float(s) + SHELL_TOL) for s in sorted(set(shells))]
    return [(float(min(bRange)) - BRANGE_TOL, float(max(bRange)) + BRANGE_TOL)]


def describe_directions(bvals, bvecs):
    """Unique diffusion directions, and design-matrix rank, of a gradient table.

    Directions are identified antipodally -- g and -g probe the same tensor
    element -- and up to DIRECTION_COS_TOL, so repeats of a direction count once.
    The rank returned is that of the seven-column design matrix the fits solve:
    scaling a row by its b-value and repeating rows change neither, so it is
    computed from the unique unit directions plus a single b = 0 row."""
    bvals = np.asarray(bvals, dtype=float)
    dwi = bvals > B0_MAX
    g = np.asarray(bvecs, dtype=float).reshape(len(bvals), 3)[dwi]
    norms = np.linalg.norm(g, axis=1)
    g = g[norms > 0] / norms[norms > 0, None]        # a zero bvec probes no direction

    unique = []
    for v in g:
        if not any(abs(float(np.dot(v, u))) >= DIRECTION_COS_TOL for u in unique):
            unique.append(v)

    #--- lower-triangular ordering, matching dipy's design_matrix()
    rows = [[v[0]**2, 2*v[0]*v[1], v[1]**2, 2*v[0]*v[2], 2*v[1]*v[2], v[2]**2]
            for v in unique]
    if np.any(~dwi):
        rows.append([0.0]*6)                         # every b = 0 volume gives this row
    W = np.column_stack([np.array(rows).reshape(len(rows), 6), np.ones(len(rows))])

    return len(unique), int(np.linalg.matrix_rank(W))


def format_b_values(bvals):
    """The b-values present with their counts, rounded to 10 s/mm2 so that
    per-direction deviation within a shell reads as one shell. For messages."""
    shells = np.round(np.asarray(bvals, dtype=float) / 10.0) * 10.0
    values, counts = np.unique(shells, return_counts=True)
    return ', '.join(f'{int(v)} (n={n})' for v, n in zip(values, counts))


def filter_b_values(fn_data = 'data.nii.gz',
                fn_bval = 'file.bval',
                fn_bvec = 'file.bvec',
                out_dir = None,
                bIntervals = ((800-BRANGE_TOL, 1200+BRANGE_TOL),)):

    accepted = ', '.join(f'[{lo:g}, {hi:g}]' for lo, hi in bIntervals)
    print("Filtering DWI data according to b-values:")
    print(f"Accepted are b-values close to Zero (b-value <= {B0_MAX}) and in: {accepted}")

    bvals, bvalsStr = read_bval_or_bvec(fn_bval)
    bvecs, bvecsStr = read_bval_or_bvec(fn_bvec)
    if len(bvecs) != len(bvals):
        raise ValueError(f"The bvec file holds {len(bvecs)} directions but the bval file holds "
                         f"{len(bvals)} b-values. They have to describe the same volumes.\n"
                         f" bval: {fn_bval}\n bvec: {fn_bvec}")

    selB0 =  (bvals <= B0_MAX)
    perInterval = [((bvals >= lo) & (bvals <= hi) & ~selB0) for lo, hi in bIntervals]
    selBRange = np.logical_or.reduce(perInterval)
    sel = selB0 | selBRange
    print(f' total number of images        : n={len(bvals)}')
    print(f' images with b-value <= {B0_MAX}      : n={sum(selB0==True)}')
    print(f' images with b-value in range  : n={sum(selBRange==True)}')
    print(f' images with excluded b-values : n={sum(sel==False)}')

    #--- Everything below refuses data the fits cannot be trusted on, before any
    #    image is loaded: a too-narrow selection then costs seconds rather than a
    #    tensor fit. The checks run on the unfiltered path too, so they are
    #    placed ahead of the 'applyFilter' branch.
    for (lo, hi), selInterval in zip(bIntervals, perInterval):
        if not np.any(selInterval):
            raise ValueError(f"No volume has a b-value in [{lo:g}, {hi:g}]. Check the requested "
                             f"b-values ('--bRange' / '--shells', tolerance included above) "
                             f"against the b-values in the data: {format_b_values(bvals)}.\n"
                             f" bval: {fn_bval}")
    if not np.any(selB0):
        raise ValueError(f"No volume with a b-value close to zero (b <= {B0_MAX}) was found. The "
                         f"fits need at least one to estimate S0. b-values in the data: "
                         f"{format_b_values(bvals)}.\n bval: {fn_bval}")

    nDirections, rank = describe_directions(bvals[sel], bvecs[sel])
    print(f' unique diffusion directions   : n={nDirections}')
    if nDirections < MIN_DIRECTIONS:
        raise ValueError(f"Only {nDirections} unique diffusion direction(s) remain after the "
                         f"b-value selection, but the free-water bi-tensor fit needs at least "
                         f"{MIN_DIRECTIONS} (repeated directions constrain the free-water "
                         f"fraction no further than a single one does, so they are counted "
                         f"once). Check the requested b-values ('--bRange' / '--shells') "
                         f"against the b-values in the data: {format_b_values(bvals)}.\n"
                         f" bval: {fn_bval}\n bvec: {fn_bvec}")
    if rank < DESIGN_MATRIX_RANK:
        raise ValueError(f"The {nDirections} diffusion directions do not span the diffusion "
                         f"tensor: the design matrix has rank {rank} instead of "
                         f"{DESIGN_MATRIX_RANK}, so the tensor cannot be estimated from them. "
                         f"They are collinear or lie in a single plane, which usually means a "
                         f"damaged gradient table.\n bvec: {fn_bvec}")
    if nDirections < RECOMMENDED_DIRECTIONS:
        print(f'WARNING: {nDirections} unique diffusion directions is below the recommended '
              f'minimum of {RECOMMENDED_DIRECTIONS}.\n The fit runs, but the free-water fraction '
              f'and hence the reported metrics are noisy.\n Please interpret the results with '
              f'care, and do not pool them with results from data with more directions.')

    applyFilter = False

    if sum(sel==False)>0:
        applyFilter = True
        print('Removing excluded b-values!')

    b5 = (bvals>0) & (bvals<=B0_MAX)
    if sum(b5==True) > 0:
        print(f'Some b-values (n={sum(b5)}) are close but not exactly Zero:\n {bvals[b5]}')
        print('These values are set to Zero!')
        bvalsStr[b5] = '0'
        applyFilter = True

    if not applyFilter:
        print('Nothing to do.')
    else:

        bvals = bvalsStr[sel]
        print('New set of b-values:')
        print(bvals)

        bvecs = bvecsStr[sel]

        nii = nib.load(fn_data)
        img = nii.get_fdata()
        img = img[:,:,:,sel]

        fn_bval = join(out_dir, basename(fn_bval))
        fn_bvec = join(out_dir, basename(fn_bvec))
        fn_data = join(out_dir, basename(fn_data))
        write_bval_or_bvec(bvals, fn_bval)
        write_bval_or_bvec(bvecs, fn_bvec)
        # dtype='float32' explicit: a lossless round-trip for the float32 input the
        # pipeline expects (do not change -- float64 would alter results for
        # int16-with-scaling input)
        save_nifti(fn_data, img, nii.affine, nii.header, dtype='float32')

        print('New data saved to:')
        print(fn_bval)
        print(fn_bvec)
        print(fn_data)


    return fn_data, fn_bval, fn_bvec, nDirections


###########################################################################
# Parallel driver for the vendored per-voxel fits

# markvcid_fw_mrn.py is vendored verbatim and stays that way. Both fits there
# loop over independent voxels, so fitting slabs of the volume in worker
# processes leaves every voxel's arithmetic untouched and the result is
# bit-identical to the serial loop - required, because a 1-ULP difference in the
# fitted FA can move delta-PSMD by percent (see conda-explicit-linux-64.txt).

# Capped independently of the core budget: 16 workers already take the fit from
# ~2 min per timepoint to ~10-15 s, and slabbing along axis 0 cannot beat the
# single most expensive slice anyway (~52x).
FW_MAX_WORKERS = 16

_FW_SHARED = {}


def _fw_pool_init(payload):
    _FW_SHARED.update(payload)


def _fw_fit_slab(bounds):
    lo, hi = bounds
    kwargs = dict(_FW_SHARED['kwargs'])
    for name in _FW_SHARED['slabbed']:
        kwargs[name] = _FW_SHARED[name][lo:hi]
    return lo, hi, _FW_SHARED['fn'](**kwargs)


def fit_voxelwise(fn, data, volumes, kwargs, nproc):
    """Run one of the vendored per-voxel fits, spreading slabs of the volume
    along axis 0 over 'nproc' worker processes. 'volumes' maps a keyword name to
    a volume that has to be sliced alongside 'data'; 'kwargs' is passed through
    unsliced. Returns what the serial call returns, bit for bit."""

    volumes = {'data': data, **volumes}
    if nproc <= 1:
        return fn(**volumes, **kwargs)

    # Four slabs per worker, so dynamic scheduling can absorb the uneven masked-
    # voxel count per slice. Slabs are contiguous, hence no more than axis 0 is long.
    n0 = data.shape[0]
    edges = np.linspace(0, n0, min(n0, nproc * 4) + 1).round().astype(int)
    jobs = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]

    out = np.zeros(data.shape[:-1] + (9,))
    payload = {'fn': fn, 'kwargs': kwargs, 'slabbed': tuple(volumes), **volumes}
    # 'fork' explicitly: workers inherit the volumes copy-on-write instead of
    # pickling them, and it stops being the Linux default in Python 3.14.
    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(nproc, initializer=_fw_pool_init, initargs=(payload,)) as pool:
        for lo, hi, params in pool.imap_unordered(_fw_fit_slab, jobs, chunksize=1):
            out[lo:hi] = params        # indexed, so completion order is irrelevant
    return out


def free_water_correction(fn_data = 'data.nii.gz',
                fn_mask = 'brain_mask.nii.gz',
                fn_bval = 'file.bval',
                fn_bvec = 'file.bvec',
                out_dir = None,
                smooth=True,
                nproc=1):

    mdreg=2.0e-3
 
    print('Reading data from:')
    print(fn_bval)
    print(fn_bvec)
    print(fn_data)
    nii = nib.load(fn_data)
    niim = nib.load(fn_mask)
    data = nii.get_fdata()
    # Same conversion the vendored fits do internally, just done before the call:
    # from numpy 2.0 their np.array(mask, dtype=bool, copy=False) raises on a
    # float mask, because copy=False came to mean "never copy".
    mask = niim.get_fdata().astype(bool)
    bvals, bvecs = read_bvals_bvecs(fn_bval, fn_bvec)
    print(f'bvals = \n{bvals}\n')

    gtab = gradient_table(bvals, bvecs)
    
    W = design_matrix(gtab)
    
    if smooth:
        print('Smoothing DWI data')
        fwhm = 1.25
        gauss_std = fwhm / np.sqrt(8 * np.log(2))  # converting fwhm to Gaussian std
        for v in range(data.shape[-1]):
            data[..., v] = gaussian_filter(data[..., v], sigma=gauss_std)
    
    
    print(f'Fitting single tensor model, not accounting for free water ({nproc} worker(s))')
    dti_params = fit_voxelwise(wls_fit_dti, data, {'mask': mask},
                               {'W': W, 'min_signal': 1.0e-6}, nproc)
    evals, _ = decompose_tensor(from_lower_triangular(dti_params))
    FA0 = dti.fractional_anisotropy(evals)
    MD0 = dti.mean_diffusivity(evals)
    save_nifti(join(out_dir, 'wls_dti_FA.nii.gz'), FA0, nii.affine, nii.header)
    save_nifti(join(out_dir, 'wls_dti_MD.nii.gz'), MD0, nii.affine, nii.header)
    
    print('Fitting two-tensor model, for tissue and free water')
    S0 = np.mean(data[..., gtab.b0s_mask], axis=-1)
    pCSF = (MD0 > 0.002)
    mCSF = np.mean(MD0[pCSF])    
    mdreg1 = 0.002*mCSF/0.0025
    mdreg = np.min([mdreg,mdreg1])
    MDm = 0.0006
    
    dti_params1 = fit_voxelwise(wls_fit_tensor_fw, data,
                                {'md_data': MD0, 'S0': S0, 'mask': mask},
                                {'W': W, 'Diso': 3e-3, 'min_signal': 1.0e-6,
                                 'piterations': 2, 'mdreg': mdreg, 'MDm': MDm}, nproc)
    evals, _ = decompose_tensor(from_lower_triangular(dti_params1))
    FA1 = dti.fractional_anisotropy(evals)
    FW1 = dti_params1[..., 7]
    save_nifti(join(out_dir, 'fwc_wls_dti_FA.nii.gz'), FA1, nii.affine, nii.header)
    save_nifti(join(out_dir, 'wls_dti_FW.nii.gz'), FW1, nii.affine, nii.header)
    
    print('Setting voxels in fwc-FA image to 0.05, if located inside brain mask and fwc-FA equals 0')
    FA1[(FA1==0) & (mask>0)] = 0.05
    save_nifti(join(out_dir, 'fwc_wls_dti_FA_05.nii.gz'), FA1, nii.affine, nii.header)


def plan_ants_parallelism(nTP, coreBudget, paraOverride=None, itkThreads=ITK_THREADS_DEFAULT):
    """Returns (para, itkThreads, control) for the ANTs template step, 'control'
    being its '-c' argument.

    'itkThreads' is handed straight back. It must not depend on the machine or on
    the timepoint count, because it is the one quantity here that moves the metric
    values; deriving it from the core budget is what made results differ between
    machines and between subjects with different numbers of visits.

    Only 'para' is derived, and it is numerically inert -- serial and pexec runs
    are byte-identical across all 82 intermediates -- so it is free to fill
    whatever cores happen to be available. A para of 1 must take the serial path:
    ANTs' pexec aborts on '-j 1' without running anything at all."""
    coreBudget = max(1, int(coreBudget))
    nTP = max(1, int(nTP))
    itkThreads = max(1, int(itkThreads))

    if paraOverride is None:
        # 'fill' spends the whole budget rather than idling the remainder; 'cap'
        # holds it back where doing so would oversubscribe beyond the limit above.
        fill = -(-coreBudget // itkThreads)
        cap = (OVERSUBSCRIBE_NUM * coreBudget) // (OVERSUBSCRIBE_DEN * itkThreads)
        para = max(1, min(fill, cap))
    else:
        # An explicit override is a deliberate choice and is not clamped to the
        # budget -- only to the timepoint count, beyond which there is no work
        # left to run in parallel.
        para = max(1, int(paraOverride))
    para = min(para, nTP)

    return para, itkThreads, ("-c 0" if para == 1 else f"-c 2 -j {para}")


def create_template(timepoints = [], fnCoreg = [], dirOut = None, coreBudget = 1, paraOverride = None, iterations="30x30x8", numRegistrations=3, itkThreads=ITK_THREADS_DEFAULT):

    fnFA = []
    for i,tp in enumerate(timepoints):
        fnFA.append(join(dirOut, basename(tp)+'_fwc_wls_dti_FA_05.nii.gz'))
        copy2(join(tp, 'fwc_wls_dti_FA_05.nii.gz'), fnFA[i])

    para, itkThreads, control = plan_ants_parallelism(len(timepoints), coreBudget, paraOverride, itkThreads)
    mode = 'serial' if para == 1 else 'pexec'
    perCore = para * itkThreads / coreBudget
    print(f"Template construction: {len(timepoints)} timepoint(s), core budget {coreBudget} "
          f"-> {para} parallel registration job(s) x {itkThreads} ITK thread(s) per job ({mode} mode)")
    print(f"  {para * itkThreads} thread(s) over {coreBudget} core(s) = {perCore:.2f} per core")
    if perCore > 2:
        print("  NOTE: the threads outnumber the cores several times over. This affects only the "
              "runtime, never the results; allocate more cores to bring it down.")

    cmd = (f"export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={itkThreads}; "
           "export ANTS_RANDOM_SEED=1; "
           f"antsMultivariateTemplateConstruction2.sh -d 3 -i {int(numRegistrations)} -f 4x2x1 -s 2x1x0vox -q {shlex.quote(iterations)} -t SyN -m CC "
           f" -r 1 -z /opt/scripts/FMRIB58_FA_2mm_crop.nii.gz -y 0 {control} -o {shlex.quote(dirOut + '/')} {' '.join(shlex.quote(f) for f in fnFA)}")
    run_subprocess(cmd, False, 'antsMultivariateTemplateConstruction2.sh')

    fnAverage = []
    for iTP, tp in enumerate(timepoints):
        tpB = basename(tp)
        for iFn, fn in enumerate(fnCoreg):
            fnIn = join(tp, fn)
            fnOut = re.sub(r'\.nii(\.gz)?$','_to_template.nii.gz', fnIn)
            ref = join(dirOut, 'template0.nii.gz')
            warp = join(dirOut, f'{tpB}_fwc_wls_dti_FA_05{iTP}1Warp.nii.gz')
            affine = join(dirOut, f'{tpB}_fwc_wls_dti_FA_05{iTP}0GenericAffine.mat')
            cmd = (f"antsApplyTransforms -d 3 --float 1 -i {shlex.quote(fnIn)}  -o {shlex.quote(fnOut)} "
                   f"-r {shlex.quote(ref)} -t {shlex.quote(warp)} -t {shlex.quote(affine)}")
            run_subprocess(cmd, True, 'antsApplyTransforms')

            if iFn==0: # iFn==0 corresponds to the FA (by default the fwc-FA), which shall be averaged across timepoints right after this loop
                fnAverage.append(fnOut)
    
    # average the co-registered fwc-FA, because intensity histogram in template drifts
    print('Averaging the registered fwc-FA images:\n',fnAverage)
    img = []
    for fn in fnAverage:
        nii = nib.load(fn)
        img.append(nii.get_fdata())
    img = np.mean(np.stack(img,-1), -1)
    # 'nii' here is deliberately the loop variable leaked from above: all timepoints
    # share one affine/header, so any of them will do.
    save_nifti(join(dirOut, 'FA-for-tbss-long.nii.gz'), img, nii.affine, nii.header, 'float32')
    print('Saved mean image to:', join(dirOut, 'FA-for-tbss-long.nii.gz'))

def coreg_merge_masks(timepoints = [], masks = [], label=None, dirTemplate = None, binarise = False):

    if any(masks):
        fnMerge = []
        for iTP, tp in enumerate(timepoints):
            tpB = basename(tp)
            if masks[iTP] is not None:
                fnIn = masks[iTP]
                # always '.nii.gz' here: an uncompressed input mask must not be
                # carried into the temp tree, where every consumer assumes gzip
                fnOut = join(tp,label+'.nii.gz')
                if binarise:
                    nii = nib.load(fnIn)
                    img = nii.get_fdata()
                    save_nifti(fnOut, img>0, nii.affine, nii.header, 'uint8')
                else:
                    copy_as_nii_gz(fnIn, fnOut)
                if len(timepoints)>1:
                    fnIn = fnOut
                    fnOut = join(tp, label+'_to_template.nii.gz')
                    ref = join(dirTemplate, 'template0.nii.gz')
                    warp = join(dirTemplate, f'{tpB}_fwc_wls_dti_FA_05{iTP}1Warp.nii.gz')
                    affine = join(dirTemplate, f'{tpB}_fwc_wls_dti_FA_05{iTP}0GenericAffine.mat')
                    cmd = (f"antsApplyTransforms -d 3 --float 1 -i {shlex.quote(fnIn)}  -o {shlex.quote(fnOut)} "
                           f"-r {shlex.quote(ref)} -t {shlex.quote(warp)} -t {shlex.quote(affine)}  -n NearestNeighbor")
                    run_subprocess(cmd, True, 'antsApplyTransforms')
                fnMerge.append(fnOut)
        if len(fnMerge)>1:
            fnNewMask = merge_masks(fnMerge, join(dirTemplate, label+'_merged.nii.gz'))
        else:
            fnNewMask = fnMerge[0]
    else:
            fnNewMask = None
        
    return fnNewMask

def merge_masks(fnMerge, fnOut):
    print('Merging masks across time-points:\n',fnMerge)
    img = []
    for fn in fnMerge:
        nii = nib.load(fn)
        img.append(nii.get_fdata())
    img = np.amax(np.stack(img,-1), -1)
    save_nifti(fnOut, img, nii.affine, nii.header, 'uint8')
    
    return fnOut


def run_tbss(fnameFAt = None, dirTBSS = None):
    
    copy2(fnameFAt, join(dirTBSS, basename(fnameFAt)))

    dirBase = os.getcwd()
    os.chdir(dirTBSS)
    cmd = f'tbss_1_preproc {shlex.quote(basename(fnameFAt))}'
    run_subprocess(cmd, True, 'tbss_1_preproc')
    cmd = 'tbss_2_reg -T'
    run_subprocess(cmd, True, 'tbss_2_reg')
    cmd = 'tbss_3_postreg -T'
    run_subprocess(cmd, True, 'tbss_3_postreg')
    cmd = 'tbss_4_prestats 0.2'
    run_subprocess(cmd, True, 'tbss_4_prestats')
    os.chdir(dirBase)

    
def batch_tbss_non_fa(dirTP = None, dirTBSS = None, fnNonFA = []):
    
    fnameFAt = glob.glob(join(dirTBSS, 'FA', '*_FA.nii.gz'))
    fnameFAt = re.sub(r'_FA\.nii\.gz','.nii.gz',basename(fnameFAt[0]))

    tpB = basename(dirTP)

    for mapName, fn in fnNonFA.items():
        run_tbss_non_fa(join(dirTP,fn), tpB+'_'+mapName, dirTBSS, fnameFAt)


def run_tbss_non_fa(fn = None, label=None, dirTBSS = None, fnameFAt = None):
    
    # get name of FA used for projection onto skeleton
    if fnameFAt is None:
        fnameFAt = glob.glob(join(dirTBSS, 'FA', '*_FA.nii.gz'))
        fnameFAt = re.sub(r'_FA\.nii\.gz','.nii.gz',basename(fnameFAt[0]))
    
    dirBase = os.getcwd()

    dirTBSS_nonFA = join(dirTBSS, label)
    Path(dirTBSS_nonFA).mkdir(exist_ok=True)
    copy_as_nii_gz(fn, join(dirTBSS_nonFA, fnameFAt))

    os.chdir(dirTBSS)
    cmd = f"tbss_non_FA {shlex.quote(label)}"
    run_subprocess(cmd, True, 'tbss_non_FA')
    os.chdir(dirBase)


def integrate_masks(dirTP = [], dirTBSS = None, skelMask = None, fnROI_MNI = None, analyseHemispheres = False):
    
    tpAll = [basename(tp) for tp in dirTP]
    
    skelBase = re.sub(r'\.nii(\.gz)?$','',basename(skelMask))
    niiMask = nib.load(skelMask)
    mask = binarise_mask(niiMask.get_fdata(), 'skeleton mask', skelMask)

    voxels = [np.count_nonzero(mask)]
    timept = ['all'] if len(dirTP)>1 else [tpAll[0]]
    region = ['total']


    # Intersect the skeleton mask with each timepoint's skeletonised brain mask.
    # Values 0<v<1 there are foreground blended with background by the transforms
    # and have to go too, hence the threshold at 1 (see binarise_mask).
    allMasksAdjusted = []
    for tp in tpAll:
        nii = nib.load(join(dirTBSS, 'stats', 'all_'+tp+'_bmask_skeletonised.nii.gz'))
        bmask = nii.get_fdata()
        tpMaskAdjusted = mask.copy()
        tpMaskAdjusted[bmask<1] = 0
        allMasksAdjusted.append(tpMaskAdjusted)
        

    if len(dirTP)>1:
        maskIntersection = np.all(np.stack(allMasksAdjusted, -1), -1)
        timeptT = 'all'
    else:
        # 'tpMaskAdjusted' is the loop variable leaked from above; safe because
        # this branch implies the loop ran exactly once.
        maskIntersection = tpMaskAdjusted
        timeptT = tpAll[0]

    skelSuffix = 'intersection'
    save_nifti(join(dirTBSS, 'stats', skelBase+'_'+skelSuffix+'.nii.gz'), maskIntersection, niiMask.affine, niiMask.header, dtype='uint8', scale=False)

    voxelsIntersection = np.count_nonzero(maskIntersection)
    voxels.append(voxelsIntersection)
    timept.append(timeptT)
    region.append(skelSuffix)
    
        
    if len(dirTP)>1:
        for iTP, tp in enumerate(tpAll):            
            voxels.append(np.count_nonzero(allMasksAdjusted[iTP]) - voxelsIntersection)
            timept.append(tp)
            region.append('set_difference')
    
    fnEmask = join(dirTBSS, 'stats', 'all_E-MASK_skeletonised.nii.gz')
    if exists(fnEmask):
        niiEmask = nib.load(fnEmask)
        imgEmask = niiEmask.get_fdata()
        imgEmask = (imgEmask>0.05) *2 #--- be conservative, excluding also (most) interpolated voxels; only needed for QC: set voxels of exclusion mask equal 2, for later combination with intersection mask
        imgEmask[maskIntersection==0] = 0 #--- only needed for QC: remove voxels from exclusion mask, which are anyways outside the intersection mask
        maskIntersection[imgEmask>0] = 0
        imgEmask[maskIntersection>0] = 1 #--- only needed for QC: combine intersection mask (label=1) and exclusion mask (label=2)

        # label-2 version is only used for the QC image
        skelSuffixL2 = skelSuffix + '_Emask-as-label2'
        pnameSkelIntersExcLabeled = join(dirTBSS, 'stats', skelBase+'_'+skelSuffixL2+'.nii.gz')
        save_nifti(pnameSkelIntersExcLabeled, imgEmask, niiMask.affine, niiMask.header, dtype='uint8', scale=False)

        skelSuffix = skelSuffix + '_Emask'
        pnameSkelIntersExc = join(dirTBSS, 'stats', skelBase+'_'+skelSuffix+'.nii.gz')
        save_nifti(pnameSkelIntersExc, maskIntersection, niiMask.affine, niiMask.header, dtype='uint8', scale=False)

        voxels.append(np.count_nonzero(maskIntersection))
        timept.append(timeptT)
        region.append(skelSuffix)

    fnROI = sorted(glob.glob(join(dirTBSS, 'stats', 'all_ROI-*_skeletonised.nii.gz')))
    for iFn,fn in enumerate(fnROI):
        roi = re.sub(r'.*all_ROI-([0-9]*)_skeletonised.nii.gz','\\1',fn)
        niiROI = nib.load(fn)
        imgROI = niiROI.get_fdata()

        imgROI[maskIntersection==0] = 0
        imgROI[imgROI>0.05] = 1
        imgROI[imgROI<1] = 0

        skelSuffixT = skelSuffix + f'_Rmask-{roi}'
        pnameROI = join(dirTBSS, 'stats', skelBase+'_'+skelSuffixT+'.nii.gz')
        save_nifti(pnameROI, imgROI, niiMask.affine, niiMask.header, dtype='uint8')

        voxels.append(np.count_nonzero(imgROI))
        timept.append(timeptT)
        region.append(skelSuffixT)

        if iFn==0: imgROImerged = np.zeros(imgROI.shape, 'uint8')
        imgROImerged[imgROI>0] = int(roi)
        if iFn==len(fnROI)-1 and np.count_nonzero(imgROImerged)>0: 
            pnameROImerged = join(dirTBSS, 'stats', skelBase+'_'+skelSuffix+'_Rmask.nii.gz')
            save_nifti(pnameROImerged, imgROImerged, niiROI.affine, niiROI.header, dtype='uint8')
    # complementary ROI for the background
    if len(fnROI)>0:
        imgROI = maskIntersection.copy()
        imgROI[imgROImerged>0] = 0
        skelSuffixT = skelSuffix + '_Rmask-00'
        pnameROI = join(dirTBSS, 'stats', skelBase+'_'+skelSuffixT+'.nii.gz')
        save_nifti(pnameROI, imgROI, niiMask.affine, niiMask.header, dtype='uint8')
        # Inserts the background row just before the per-ROI rows appended above,
        # keeping these three parallel lists in sync by index arithmetic.
        voxels.insert(len(voxels)-len(fnROI), np.count_nonzero(imgROI))
        timept.insert(len(timept)-len(fnROI), timeptT)
        region.insert(len(region)-len(fnROI), skelSuffixT)

    


    if fnROI_MNI is not None:
        niiROI_MNI = nib.load(fnROI_MNI)
        imgROI_MNI = niiROI_MNI.get_fdata()
        #- one file may carry several ROI labels
        uROI = np.unique(imgROI_MNI.astype('uint8'))
        # label 0 is deliberately kept, so the background is analysed as a ROI too
        for iRoi,roi in enumerate(uROI):
            imgROI_MNI_roi = maskIntersection.copy()
            imgROI_MNI_roi[imgROI_MNI!=roi] = 0
            
            skelSuffixT = skelSuffix + f'_RmaskMNI-{roi:02d}'
            pnameROI = join(dirTBSS, 'stats', skelBase+'_'+skelSuffixT+'.nii.gz')
            save_nifti(pnameROI, imgROI_MNI_roi, niiMask.affine, niiMask.header, dtype='uint8')

            voxels.append(np.count_nonzero(imgROI_MNI_roi))
            timept.append(timeptT)
            region.append(skelSuffixT)            

            #- merged after intersecting with the skeleton, so this differs from the input
            if iRoi==0: imgROImerged = np.zeros(imgROI_MNI.shape, 'uint8')
            if roi>0:
                imgROImerged[imgROI_MNI_roi>0] = roi
            if iRoi==len(uROI)-1 and np.count_nonzero(imgROImerged)>0: 
                pnameROImerged = join(dirTBSS, 'stats', skelBase+'_'+skelSuffix+'_RmaskMNI.nii.gz')
                save_nifti(pnameROImerged, imgROImerged, niiROI_MNI.affine, niiROI_MNI.header, dtype='uint8')

    
    if analyseHemispheres:
        sh = maskIntersection.shape
        for hemi,bounds in zip(['LH', 'RH'],[[0,sh[0]//2],[sh[0]//2,sh[0]+1]]):
            maskHemi = maskIntersection.copy()
            maskHemi[bounds[0]:bounds[1],:,:] = 0

            skelSuffixT = skelSuffix + f'_{hemi}'
            pnameROI = join(dirTBSS, 'stats', skelBase+'_'+skelSuffixT+'.nii.gz')
            save_nifti(pnameROI, maskHemi, niiMask.affine, niiMask.header, dtype='uint8')

            voxels.append(np.count_nonzero(maskHemi))
            timept.append(timeptT)
            region.append(skelSuffixT)
        

    skeleton = [basename(skelMask)] * len(voxels)
    # 'NA'/'NaN' are load-bearing sentinels, not placeholders: delta-svd_aggregate_results.py
    # relies on them (via pandas' read_csv coercing 'NA' to NaN) to split these bookkeeping
    # rows from real metric rows. Keep them literal strings; do not switch to np.nan.
    df = pd.DataFrame(
        {'timepoint': timept,
         'skeleton': skeleton,
         'region': region,
         'voxels': voxels,
         'metric': ['NA'] * len(voxels),
         'value': ['NaN'] * len(voxels)}
    )
    return df

def extract_stats(dirTP = None, dirTBSS = None, fnNonFA = [], skelMask = None):
    
    skelBase = re.sub(r'\.nii(\.gz)?$','', basename(skelMask))
    skelMaskInters = join(dirTBSS, 'stats', skelBase+'_intersection_Emask.nii.gz')
    if not exists(skelMaskInters):
        skelMaskInters = join(dirTBSS, 'stats', skelBase+'_intersection.nii.gz')

    fnROI = [skelMaskInters]
    fnROI = fnROI + sorted(glob.glob(join(dirTBSS, 'stats', '*_Rmask-*.nii.gz')))
    fnROI = fnROI + sorted(glob.glob(join(dirTBSS, 'stats', '*_RmaskMNI-*.nii.gz')))
    fnROI = fnROI + sorted(glob.glob(join(dirTBSS, 'stats', '*_LH.nii.gz')))
    fnROI = fnROI + sorted(glob.glob(join(dirTBSS, 'stats', '*_RH.nii.gz')))



    dd = []
    tpB = basename(dirTP)
    for fnR in fnROI:
        
        niiROI = nib.load(fnR)
        roi = niiROI.get_fdata()
        roiBase = re.sub(r'\.nii(\.gz)?$','', basename(fnR))
        roiSuffix = re.sub(skelBase+'_','', roiBase)

        for mapName, _ in fnNonFA.items():

            print('\nExtracting histogram parameters for:')
            print(f' region    : {roiSuffix}')
            print(f' map       : {mapName}')

            nii = nib.load(join(dirTBSS, 'stats', 'all_'+tpB+'_'+mapName+'_skeletonised.nii.gz'))
            img = nii.get_fdata()
            skel = img[roi>0]
            print( ' voxels  :',len(skel))
            mean = np.mean(skel) if len(skel)>0 else np.nan

            mT = re.sub(r'^nc','',mapName)
            if mT == 'MD':
                prcts = np.percentile(skel,[5,95]) if len(skel)>0 else [np.nan]*2
                pw = prcts[1] - prcts[0]
                metrics = ['PSMD', 'MS'+mT]
                values = [pw, mean]
            else:
                metrics = ['MS'+mT]
                values = [mean]
            dd.append(pd.DataFrame(
                {'timepoint': [tpB]*len(metrics),
                'skeleton': [basename(skelMask)]*len(metrics),
                'region': [roiSuffix]*len(metrics),
                'voxels': [len(skel)]*len(metrics),
                'metric': metrics,
                'value': values}
            ))

    df = pd.concat(dd)

    return df

def prepare_qc(dirQC, fnHTML, skelMask, dirTBSS, dirTemplate, dirTP, fnCSV, args):
        
    from create_qc_image import create_qc_image
    from create_html_with_png import create_html_with_png
    
    skelBase = re.sub(r'\.nii(\.gz)?$','', basename(skelMask))
    skelMask = join(dirTBSS, 'stats', skelBase+'_intersection_Emask-as-label2.nii.gz')
    emaskExists = 1
    if not exists(skelMask):
        skelMask = join(dirTBSS, 'stats', skelBase+'_intersection.nii.gz')
        emaskExists = 0
    
    fnROI = [skelMask]
    fnROI = fnROI + sorted(glob.glob(join(dirTBSS, 'stats', '*_Rmask.nii.gz')))
    fnROI = fnROI + sorted(glob.glob(join(dirTBSS, 'stats', '*_RmaskMNI.nii.gz')))

    if len(dirTP)>1:
        fnameFAt = "FA-for-tbss-long"
    else:
        fnameFAt = "fwc_wls_dti_FA_05"
    dirBase = os.getcwd()

    for fn in fnROI:
        os.chdir(join(dirTBSS,'stats'))
        cmd = f'tbss_deproject {shlex.quote(basename(fn))} 2 -n'
        run_subprocess(cmd, True, 'tbss_deproject')
        os.chdir(dirBase)
        fnBase = basename(fn)
        fnDeprojectedTemplateSpace = join(dirTBSS, 'FA', fnameFAt+'_FA_'+fnBase)
            
        for iTP in range(len(dirTP)):
            tpB = basename(dirTP[iTP])
            if len(dirTP)>1:
                fnOut = join(dirQC, tpB+'_'+fnBase)
                ref = join(dirTP[iTP], 'fwc_wls_dti_FA.nii.gz')
                affine = join(dirTemplate, f'{tpB}_fwc_wls_dti_FA_05{iTP}0GenericAffine.mat')
                invwarp = join(dirTemplate, f'{tpB}_fwc_wls_dti_FA_05{iTP}1InverseWarp.nii.gz')
                cmd = (f"antsApplyTransforms -d 3 --float 1 -i {shlex.quote(fnDeprojectedTemplateSpace)}  -o {shlex.quote(fnOut)} "
                       f"-r {shlex.quote(ref)} -t [{shlex.quote(affine)},1] -t {shlex.quote(invwarp)} -n NearestNeighbor")
                run_subprocess(cmd, True, 'antsApplyTransforms')
            else:
                copy2(fnDeprojectedTemplateSpace, join(dirQC, tpB+'_'+fnBase))    
    
    #--- in space of the input, per timepoint
    vlim = [
        [0.05, 0.7],
        [0.00035, 0.0026]
    ]
    labels = [
        'free water-corrected FA',
        'MD'
    ]

    fnPNG = []
    captions = []
    for iTP in range(len(dirTP)):
        tpB = basename(dirTP[iTP])
        fnFA = join(dirTP[iTP], 'fwc_wls_dti_FA_05.nii.gz')
        copy2(fnFA, join(dirQC, tpB+'_fwcFA.nii.gz'))
        copy2(join(dirTP[iTP], "wls_dti_FA.nii.gz"), join(dirQC, tpB+'_FA.nii.gz'))
        fnMD = join(dirTP[iTP], 'wls_dti_MD.nii.gz')
        copy2(fnMD, join(dirQC, tpB+'_MD.nii.gz'))
        fnSkeleton = join(dirQC, tpB+'_'+basename(fnROI[0]))
        fnBmask = join(dirTP[iTP], 'brain_mask.nii.gz')
        addLegends=(1,emaskExists) if iTP==0 else (0,0)

        fnPNG = fnPNG + create_qc_image([fnFA, fnMD], vlim, labels, fnSkeleton, fnBmask, animate = False, addLegends=addLegends)
        captions = captions + [f'Timepoint "{tpB}"','']

    #--- in space of patient template
    vlim = []
    labels = []
    fnames = []
    fnBmask = []
    if len(dirTP)>1:
        for iTP in range(len(dirTP)):
            tpB = basename(dirTP[iTP])
            vlim.append([0.00035, 0.0026])
            labels.append(f'MD at timepoint "{tpB}"')
            fnames.append(join(dirTP[iTP], 'wls_dti_MD_to_template.nii.gz'))
            fnBmask.append(join(dirTP[iTP], 'brain_mask_to_template.nii.gz'))

        fnSkeleton = join(dirTBSS, 'FA', fnameFAt+'_FA_'+basename(fnROI[0])) 
        fnPNG = fnPNG + [create_qc_image(fnames, vlim, labels, fnSkeleton, fnBmask)]
        captions = captions + ['Within-subject template space']

    #--- in space of MNI template
    vlim = []
    labels = []
    fnames = []
    fnamesBmask = []
    for iTP in range(len(dirTP)):
        tpB = basename(dirTP[iTP])
        vlim.append([0.00035, 0.0026])
        labels.append(f'MD at timepoint "{tpB}"')
        fnames.append(join(dirTBSS, 'stats', f'all_{tpB}_MD.nii.gz'))
        fnamesBmask.append(join(dirTBSS, 'stats', f'all_{tpB}_bmask.nii.gz'))

    fnROI_base = re.sub(r'\.nii(\.gz)?$','', basename(fnROI[0]))
    fnSkeleton = join(dirTBSS, 'stats', fnROI_base+'_to_all_FA.nii.gz')
    fnameCmask = join(dirTBSS, 'stats', 'mean_FA_mask.nii.gz')
    fnPNG = fnPNG + [create_qc_image(fnames, vlim, labels, fnSkeleton, fnamesBmask, fnameCmask)]
    captions = captions + ['MNI space']

    print('\nSaving QC to:\n ', fnHTML)

    create_html_with_png(fnHTML, fnPNG, captions, None, fnCSV, args)

    if args.qc < 2:
        rmtree(dirQC)


###########################################################################
# Helper functions

def save_nifti(fname, arr, affine, header, dtype='float32', scale=False):
    niiNew = nib.Nifti1Image(arr, affine, header)    
    niiNew.set_data_dtype(dtype)
    if not scale:
        niiNew.header.set_slope_inter(1, 0)
    nib.save(niiNew, fname)


def binarise_mask(img, label, fname):
    """Return 'img' as a strict {0,1} mask, reporting anything it had to change.
    Masks are thresholded strictly above 0 after resampling and written as uint8, neither of
    which behaves on a 0/255 or probabilistic mask. No-op on a binary one."""
    binary = img > 0
    if not np.array_equal(img, binary):
        print(f"NOTE: the provided {label} holds values other than 0 and 1:\n"
              f"  {fname}\n"
              f"  Binarising it (values greater than zero become 1; zero and negative values become 0) for processing.")
    return binary


def copy_as_nii_gz(fnIn, fnOut, dtype=None):
    """Place 'fnIn' at 'fnOut', whose name always ends in '.nii.gz'. A gzipped
    input is copied verbatim; an uncompressed one is re-encoded, because nibabel,
    FSL and ANTs pick the codec from the file *name*. 'dtype' defaults to the
    input's own, keeping the re-encode faithful."""
    if fnIn.endswith('.nii.gz'):
        copy2(fnIn, fnOut)
    else:
        nii = nib.load(fnIn)
        save_nifti(fnOut, nii.get_fdata(), nii.affine, nii.header,
                   nii.get_data_dtype() if dtype is None else dtype)


def run_subprocess(cmd, displayStdout, label):
    print(f"Calling {label} command with:")
    print(cmd, '\n')
    output = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    if output.returncode != 0:
        print("STDOUT/STDERR:")
        print(output.stdout.decode("utf-8"))
        raise ValueError(f"ERROR during call of {label} command! For stdout/stderr of the command see above!")
    else:
        stdout = output.stdout.decode("utf-8")
        if displayStdout and len(stdout)>0 and not stdout.isspace(): print(stdout)


def section_header(text, startPrevious = None):
    lengthFrame = len(text) if len(text)>60 else 60
    print('\n\n'+'#'*lengthFrame)
    if startPrevious is not None:
        endPrevious = time.time()
        elapsed = endPrevious - startPrevious
        print('(previous step lasted: {:02.0f}:{:02.0f})'.format(elapsed//60, elapsed%60))
    print(text)
    print('#'*lengthFrame+'\n')

    return time.time()


def isNIfTI(s, abort=True):
    if os.path.isfile(s) and (s.endswith('.nii.gz') or s.endswith('.nii')):
        return s
    elif os.path.isfile(s+'.nii.gz'):
        return s+'.nii.gz'
    elif os.path.isfile(s+'.nii'):
        return s+'.nii'
    else:
        if abort:
            raise argparse.ArgumentTypeError("File path does not exist or is not NIfTI. Please check: %s"%(s))
        else:
            return None

def isCSV(s):
    if s == 'overwrite' or s.endswith('.csv') or s.endswith('.CSV'):
        return s
    else:
        raise argparse.ArgumentTypeError("The provided filename does not have the required '.csv' extension. Please check: %s"%(s))
    
def assertPositiveJobs(s):
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--para (number of parallel ANTs registration jobs) must be a positive integer; you provided '{s}'")
    if v < 1:
        raise argparse.ArgumentTypeError(f'--para (number of parallel ANTs registration jobs) must be at least 1; you provided {s}')
    return v


def assertPositiveRegistrations(s):
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--numRegistrations (iterations of the template construction) must be a positive integer; you provided '{s}'")
    if v < 1:
        raise argparse.ArgumentTypeError(f'--numRegistrations (iterations of the template construction) must be at least 1; you provided {s}')
    return v


def assertPositiveItkThreads(s):
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--itkThreads (ITK threads per ANTs registration job) must be a positive integer; you provided '{s}'")
    if v < 1:
        raise argparse.ArgumentTypeError(f'--itkThreads (ITK threads per ANTs registration job) must be at least 1; you provided {s}')
    return v


def assertBValue(s):
    """A b-value for '--bRange' / '--shells', inside the window the tensor fit
    is valid in."""
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"b-values given with --bRange / --shells must be whole numbers; you provided '{s}'")
    if not BVAL_MIN <= v <= BVAL_MAX:
        raise argparse.ArgumentTypeError(
            f"b-values given with --bRange / --shells have to lie between {BVAL_MIN} and "
            f"{BVAL_MAX} s/mm2; you provided {v}. Below {BVAL_MIN} the diffusion signal is "
            f"contaminated by perfusion and above {BVAL_MAX} by non-Gaussian diffusion, so a "
            f"diffusion tensor fitted there is not interpretable. Note that b-values close to "
            f"zero (b <= {B0_MAX}) are always included and must not be given here.")
    return v


def threadBudget(s):
    """A positive integer, or 'auto' to auto-detect the physical cores."""
    if str(s).strip().lower() == 'auto':
        return 'auto'
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--threads must be a positive integer or 'auto'; you provided '{s}'")
    if v < 1:
        raise argparse.ArgumentTypeError(f"--threads must be at least 1 (or 'auto'); you provided {s}")
    return v

class CustomArgumentParser(argparse.ArgumentParser):
    # Single-dash options must be one character and separated from their argument
    def parse_known_args(self, args=None, namespace=None):
        if args is None:                       # the argparse default: read the command line
            args = sys.argv[1:]
        for arg_string in args:
            if arg_string.startswith('-') and not arg_string.startswith('--'):
                if len(arg_string) > 2 and not arg_string[2].isspace():
                    self.error(f'single-dash options must be one character and separated from their argument by a space: "{arg_string}"')
        return super().parse_known_args(args, namespace)
    
stepsImplemented = ['fwc','template','tbss','tbss_non_fa','extract','qc']
argparseDescription = f"DELTA-SVD {__version__} ('Diffusion Endpoints for Longitudinal Tracking of white matter Alterations in cerebral Small Vessel Disease') processes multi-directional diffusion MRI data to fully automatically extract clinically and technically validated white matter diffusion metrics. Key steps include diffusion tensor fitting (with and without free water imaging), skeletonization based on the free water-corrected FA (fwc-FA) via FSL's TBSS, and enhanced CSF partial volume masking. The final metrics MSMD, PSMD, and MSFW are computed over the skeleton. For longitudinal data, a within-subject template is created using ANTs."

def iniParser():
    parser = CustomArgumentParser(description=argparseDescription, epilog='Notice: By using DELTA-SVD, you agree to the license terms (CC BY-NC-ND 4.0) described in the LICENSE file at "https://github.com/isdneuroimaging/DELTA-SVD"')
    parser.add_argument("--version", action='version', version=f'DELTA-SVD {__version__}', help="show the DELTA-SVD version and exit")
    group0 = parser.add_argument_group('input/output data specification')
    group0.add_argument("--dwi", required=True, metavar='NIfTI', type=isNIfTI, nargs="+", action='extend', help="input path(s) to 4D DWI image(s) in NIfTI format. Number of arguments should correspond to number of time-points.")
    group0.add_argument("--bval", metavar='text-file', type=str, nargs="+", action='extend', help="input path(s) to text file(s) with b-values in FSL format, corresponding to DWI image(s). If parent folders are identical to those of corresponding DWI images, providing basename(s) is sufficient. If all basenames are identical, repetition is not needed. If argument not provided, path(s) will be constructed from DWI image path(s), substituting extension with '.bval'")
    group0.add_argument("--bvec", metavar='text-file', type=str, nargs="+", action='extend', help="input path(s) to text file(s) with b-vectors in FSL format, corresponding to DWI image(s). If parent folders are identical to those of corresponding DWI images, providing basename(s) is sufficient.  If all basenames are identical, repetition is not needed. If argument not provided, path(s) will be constructed from DWI image path(s), substituting extension with '.bvec'")
    group0.add_argument("--bmask", metavar='NIfTI', type=str, nargs="+", action='extend', help="input path(s) to DWI brain mask(s) in NIfTI format, corresponding to DWI image(s). If parent folders are identical to those of corresponding DWI images, providing basename(s) is sufficient. If all basenames are identical, repetition is not needed. If argument not provided, path(s) will be constructed from DWI image path(s), substituting the extension with '_brainmask.nii.gz' or, if that file does not exist, with '_brainmask.nii'. Masks are binarised: values greater than zero are set to 1; zero and negative values are set to 0.")
    group0.add_argument("--tp", metavar='label', type=str, nargs="+", action='extend', help="label(s) for all time-points. Number of arguments should correspond to number of DWI image(s). Labels have to be unique, and 'all' is reserved for the rows summarising all time-points. If argument not provided, time-points are labeled consecutively as TP01, TP02, and so on.")
    group0.add_argument("--id", metavar='label', type=str, help="optional patient/subject identifier. If provided, an additional column with this identifier will be added to the results table 'delta-svd_results.csv', meant to facilitate aggregation of results tables for multiple patients/subjects.")
    group0.add_argument("-o", "--dirOutput", type=str, help="path to output folder. If not provided, the parent folder of the first DWI image will be used. The results table ('delta-svd_results.csv') and a subfolder and HTML for quality checking ('delta-svd_qc' and 'delta-svd_qc.html') will be saved here. Furthermore, intermediate/temporary files will be created here inside a subfolder called 'delta-svd_temp'.")
    group1 = parser.add_argument_group('additional masking')
    group1.add_argument("--Emask", metavar='NIfTI', type=str, default = [], nargs="+", action='extend', help="input path(s) to custom exclusion mask(s) in DWI image space, used for 'exclusive' masking. One per timepoint can be provided, which will be merged in template space. Time-points will be matched by position of provided paths. Skip time-points by entering NA instead of a path. The masked area (e.g. lesion) will be excluded from analysis. Provided masks are binarised: values greater than zero are set to 1; zero and negative values are set to 0.")
    group1.add_argument("--Rmask", metavar='NIfTI', type=str, default = [], nargs="+", action='extend', help="input path(s) to custom ROI mask(s) in DWI image space. One per timepoint can be provided, which will be merged in template space. Timepoint matching and/or skipping works as explained for option 'Emask'. Each mask can contain more than one integer label corresponding to different ROI, which will be analysed separately. However, masks will be merged in template space and if labels in masks from different time-points disagree, the respectively highest integer label will overwrite the other labels.")
    group1.add_argument("--RmaskMNI", metavar='NIfTI', type=isNIfTI, help="input path to a single custom ROI mask in MNI space. Can contain integer labels for multiple ROI, which will be analysed separately.")
    group1.add_argument("--hemispheres", action='store_true', help="calculate skeleton metrics also separately for left and right hemispheres. Please note, however, that this does not affect ROI masks, which will not be split between hemispheres.")
    group2 = parser.add_argument_group('advanced options')
    group2.add_argument("--skeletonMask", metavar='NIfTI', type=isNIfTI, default="/opt/scripts/delta-svd_skeletonmask_v1.nii.gz", help="input path to an alternative skeleton mask. It will be binarised: values greater than zero are set to 1; zero and negative values are set to 0. Defaults to the mask validated with DELTA-SVD ('delta-svd_skeletonmask_v1') and designed to exclude regions with frequent CSF partial volume effects.")
    group2b = group2.add_mutually_exclusive_group()
    group2b.add_argument("--bRange", metavar='Integer', type=assertBValue, default = [800, 1200], nargs=2, help=f"range of b-values to consider for diffusion tensor fitting, given as the lower and upper limit of the non-zero shell(s) to include. Defaults to range [800,1200]. The limits are met with a tolerance of {BRANGE_TOL} s/mm2, so that shells the scanner reports slightly off their nominal value are not discarded. Volumes with a b-value close to zero (b <= {B0_MAX}) are always included and are not affected by this option. Both limits have to lie between {BVAL_MIN} and {BVAL_MAX} s/mm2, outside of which the diffusion tensor model is not valid. Mutually exclusive with '--shells'.")
    group2b.add_argument("--shells", metavar='Integer', type=assertBValue, nargs="+", action='extend', help=f"b-value shell(s) to consider for diffusion tensor fitting, e.g. '--shells 700 1000'. An alternative to '--bRange' for selecting shells individually rather than as one range, which avoids pulling in the shells in between. Each shell is matched with a tolerance of {SHELL_TOL} s/mm2, and a shell that matches no volume in the data is an error. As for '--bRange', volumes with a b-value close to zero (b <= {B0_MAX}) are always included, and each shell has to lie between {BVAL_MIN} and {BVAL_MAX} s/mm2. Mutually exclusive with '--bRange'.")
    group2.add_argument("--smooth", action='store_true', help=argparse.SUPPRESS) #--- "apply Gaussian filter (fwhm = 1.25) to DWI data"
    group2.add_argument("--dontAdjustBmaskForFW", dest='adjustBmaskForFW', action='store_false', help=argparse.SUPPRESS) #--- "don't correct the brain mask for free-water. By default, the brain mask is set to zero, where free water equals 1 (and hence fwc-FA equals 0)."
    group2.add_argument("--para", metavar='ANTs-jobs', type=assertPositiveJobs, default=None, help="number of ANTs registration jobs run at once during longitudinal template construction. Derived from the '--threads' budget by default, and capped at the number of time-points either way. Peak memory scales with it, so '--para 1' is the lowest-memory setting. It has no effect on the results, only on runtime and memory.")
    group2.add_argument("--threads", metavar='cores', type=threadBudget, default='auto', help="number of physical CPU cores DELTA-SVD may use. Two steps are multi-core: the diffusion tensor / free-water fit, and (for longitudinal input only) the within-subject template construction; TBSS and the remaining steps are single-threaded. Defaults to 'auto', which detects the cores available to the process, honouring an HPC scheduler's allocation. It has no effect on the results, only on runtime, so it can be tuned freely.")
    group2.add_argument("--itkThreads", metavar='threads', type=assertPositiveItkThreads, default=ITK_THREADS_DEFAULT, help=argparse.SUPPRESS) #--- "Expert override for the ITK threads used per ANTs registration job. WARNING: this changes the computed metrics -- ITK sums the registration metric per thread, so a different count sums in a different order. Defaults to 12, the value DELTA-SVD was validated at. Results produced with different values must not be compared or pooled."
    group2.add_argument("--iterations", type=str, default='30x30x8', help=argparse.SUPPRESS) #--- "Iterations at each resolution level of the pairwise ANTs registrations during template creation. Must be three levels and specified in the format: 'L1xL2xL3'. Defaults to '30x30x8'."
    group2.add_argument("--numRegistrations", type=assertPositiveRegistrations, default=3, help=argparse.SUPPRESS) #--- "Iterations of the template construction. Each iteration comprises averaging of images and pairwise registrations of each timepoint to the template. Defaults to 3 iterations."
    group2.add_argument("--qc", type=int, choices=[0,1,2], default=1, help="create (with argument 1; the default) a HTML file (delta-svd_qc.html) for quality checking and create (with argument 2) additionally a subfolder 'delta-svd_qc' with a series of NIfTI images showing skeleton and masks in native space, or (with argument 0) skip creation of both.")
    group2.add_argument("--debug", action='store_true', help="don't delete temporary folder 'delta-svd_temp', containing intermediate files created during processing.")
    group2.add_argument("--steps", choices = stepsImplemented, nargs="+", action='extend', help=argparse.SUPPRESS) #--- "choose step(s) to be conducted. By default all steps will be conducted. If the output for preceding steps is missing, an error will be raised. If different masks are provided, step 'tbss_non_fa' and following have to be repeated."
    group2.add_argument("--reprocess", metavar='csv-file',  type=isCSV, nargs="?", const='overwrite', help='allow reprocessing and overwriting of previously created output. You can, however, keep a previously created results file (default name: delta-svd_results.csv) by specifying here an alternative name for the new one (provide only the base name; the output folder cannot be changed here).') #--- "allow reprocessing of previously conducted steps. Warning: this will delete the previous results for all processing steps or, if '--steps' is used, for the selected (and all following/depending) steps. Deleting the final output table 'delta-svd_results.csv' can however be avoided, by providing here an alternative CSV filename (provide basename only; will be saved into the output folder, see '--dirOutput')"
    return parser


###########################################################################
# Pipeline

def pipeline_delta_svd():

    start_script = time.time()
    parser = iniParser()
    if len(sys.argv)<2:
        parser.print_usage()
        print(f'\nDELTA-SVD {__version__}\n'
              'Run "delta-svd.py -h" for detailed help\n'
              'Notice: By using DELTA-SVD, you agree to the license terms (CC BY-NC-ND 4.0) described in the LICENSE file at "https://github.com/isdneuroimaging/DELTA-SVD"\n')
        parser.exit()
    else:
        args = parser.parse_args(sys.argv[1::])

    # On its own line, so it survives a log truncated on the long call below.
    # Carried on 'args' too, for the QC report to record.
    args.version = __version__
    print(f"DELTA-SVD {__version__}")
    args.function_call = " ".join([basename(sys.argv[0])]+sys.argv[1::])
    print("Running: " + args.function_call,'\n')

    # Check bval, bvec and bmask files. The brain mask may be '.nii.gz' or '.nii',
    # so it is resolved through isNIfTI(), which probes both (gzip first) and also
    # completes a path with no extension; bval/bvec are matched verbatim.
    for attr, ext in zip(['bval','bvec','bmask'], ['.bval','.bvec','_brainmask']):
        anyExtension = (attr == 'bmask')
        resolve = (lambda fn: isNIfTI(fn, abort=False)) if anyExtension else (lambda fn: fn if exists(fn) else None)
        flist = getattr(args,attr)
        if flist is None:
            flist = [re.sub(r'\.nii(\.gz)?$', ext, fn) for fn in args.dwi]
        elif len(flist) != len(args.dwi):
            if len(flist) == 1:
                flist = flist * len(args.dwi)
            else:
                raise ValueError(f"Number of files provided with option '--{attr}' has to be zero or correspond to number of DWI files. Please refer to '--help'")
        for i, fn in enumerate(flist):
            fnResolved = resolve(fn)
            if fnResolved is None:
                fnResolved = resolve(join(dirname(args.dwi[i]), fn))
            if fnResolved is None:
                raise ValueError(f"The {i+1}. of the expected '{attr}' files does not exist")
            flist[i] = fnResolved
        setattr(args,attr,flist)
    
    # Check timepoint labels
    if args.tp is None:
        args.tp = ['TP{:02d}'.format(i+1) for i in range(len(args.dwi))] #-- folders for all time-points
    if len(args.tp) != len(args.dwi):
        raise ValueError(f'If timepoint labels are provided, their number has to correspond to the number of provided DWI files! You passed {len(args.tp)} labels for {len(args.dwi)} DWI files.')
    duplicates = sorted({tp for tp in args.tp if args.tp.count(tp) > 1})
    if duplicates:
        raise ValueError(f'Timepoint labels have to be unique! You passed {", ".join(duplicates)} more than once.')
    if len(args.dwi) > 1 and 'all' in args.tp:
        raise ValueError("'all' is reserved as the label for the rows summarising all time-points and cannot be used as a timepoint label!")

    # Check exclusion and ROI masks
    if len(args.Emask) > len(args.dwi):
        raise ValueError(f'Number of provided exclusion masks (n={len(args.Emask)}) exceeds number of time-points (n={len(args.dwi)})! Allowed is max. one mask per timepoint!')
    if len(args.Rmask) > len(args.dwi):
        raise ValueError(f'Number of provided ROI masks in DWI space (n={len(args.Rmask)}) exceeds number of time-points (n={len(args.dwi)})! Allowed is max. one mask per timepoint!')
    for i,_ in enumerate(args.dwi):
            if len(args.Emask)>i :
                if args.Emask[i]!='NA':
                    if isNIfTI(args.Emask[i], abort=False) is None:
                        args.Emask[i] = isNIfTI(join(dirname(args.dwi[i]), args.Emask[i]))
                else:
                    args.Emask[i] = None
            else:
                args.Emask.append(None)
            if len(args.Rmask)>i:
                if args.Rmask[i]!='NA':
                    if isNIfTI(args.Rmask[i], abort=False) is None:
                        args.Rmask[i] = isNIfTI(join(dirname(args.dwi[i]), args.Rmask[i]))
                else:
                    args.Rmask[i] = None
            else:
                args.Rmask.append(None)

    print(f"\nInput contains N={len(args.dwi)} time-points")
    for i in range(len(args.dwi)):
        print(f'Timepoint {args.tp[i]}:')
        print(f' DWI   :{args.dwi[i]}')
        print(f' bval  :{args.bval[i]}')
        print(f' bvec  :{args.bvec[i]}')
        print(f' Bmask :{args.bmask[i]}')
        if args.Emask[i]:
            print(f' Emask :{args.Emask[i]}')
        if args.Rmask[i]:
            print(f' Rmask :{args.Rmask[i]}')

    if args.RmaskMNI is not None:
        print(f'\nAn additional ROI mask in MNI space (RmaskMNI) was provided:\n {args.RmaskMNI}')

    if args.hemispheres:
        print('\nHemispheric ROI analysis will be done as well')

    if args.skeletonMask == "/opt/scripts/delta-svd_skeletonmask_v1.nii.gz":
        print(f'\nUsing the default skeleton mask:\n {args.skeletonMask}')
    else:
        print(f'\nUsing a non-default skeleton mask provided as input:\n {args.skeletonMask}')

    if args.itkThreads != ITK_THREADS_DEFAULT and len(args.dwi) > 1:
        print(f'\nWARNING: --itkThreads is set to {args.itkThreads} instead of the validated '
              f'{ITK_THREADS_DEFAULT}.\n This changes the computed metrics. The results of this run '
              f'must not be compared\n or pooled with results produced at the default.')

    if args.dirOutput is None:
        args.dirOutput = os.path.dirname(args.dwi[0])
    
    # Copy: 'stepsImplemented' is module-level and backs the '--steps' choices, so
    # dropping 'qc' from it would erode that list for a later run in the process.
    stepsAvailable = list(stepsImplemented)
    if args.qc==0:
        if args.steps is not None and 'qc' in args.steps:
            raise ValueError("You asked to do '--step qc' and to skip it '--qc 0' at the same time! Your choice is contradictory!")
        stepsAvailable.remove('qc')
    if args.steps is None:
        args.steps = stepsAvailable
    else:
        stepsIdx = [i for i,x in enumerate(stepsAvailable) if x in args.steps]
        args.steps = [stepsAvailable[i] for i in stepsIdx]
        print('\nOn request, only the following processing steps will be conducted:')
        for i,step in enumerate(args.steps): print(f' {i+1}. {step}')
        if len(stepsIdx)>1 and any(np.diff(stepsIdx)>1):
            print(' '); raise ValueError(f"Requested processing steps have to be contiguous. This is not the case.\nThe available steps in order are: {stepsAvailable}")
        if 'extract' not in args.steps and not args.debug:
            print("NOTE: Given that the final 'extract' step is not selected, we assume that you want to keep intermediate/temporary output and switch on the option '--debug' for you!")
            args.debug = True
    
    dirTemp = join(args.dirOutput, 'delta-svd_temp')
    dirTP = [join(dirTemp, tp) for tp in args.tp] #-- folders for all time-points
    dirTemplate = join(dirTemp,'template')
    dirTemplateInter = join(dirTemp, 'intermediateTemplates')
    dirTBSS = join(dirTemp,'TBSS')
    fnNonTBSS = glob.glob(join(dirTBSS,'*')) + glob.glob(join(dirTBSS, 'stats','*'))
    fnNonTBSS = [x for x in fnNonTBSS if not re.match('FA$|origdata$|stats$|all_FA|mean_FA|thresh',basename(x))]
    if args.reprocess is None or args.reprocess == 'overwrite':
        fnCSV = join(args.dirOutput, 'delta-svd_results.csv')
    else:
        fnCSV = join(args.dirOutput, args.reprocess)
    fnSkelRegions = glob.glob(join(dirTBSS, 'stats','*_intersection*'))
    dirQC = join(args.dirOutput, 'delta-svd_qc')
    fnHTML = join(args.dirOutput, 'delta-svd_qc.html')
    
    # Check if output exists already
    if os.path.exists(dirTemp) or os.path.exists(fnCSV) or os.path.exists(dirQC) or os.path.exists(fnHTML):

        # all steps requested: simply delete everything
        if set(args.steps+['qc']) == set(stepsAvailable+['qc']):
            print('\nChecking existence of output')
            if args.reprocess is None:
                raise ValueError("Output exists already.\n Tip: Use option '--reprocess' if you want to reprocess and overwrite")
            if os.path.exists(dirTemp): print(f'Deleting: {dirTemp}'); rmtree(dirTemp)
            if os.path.exists(fnCSV): print(f'Deleting: {fnCSV}'); os.remove(fnCSV)
            if os.path.exists(dirQC): print(f'Deleting: {dirQC}'); rmtree(dirQC)
            if os.path.exists(fnHTML): print(f'Deleting: {fnHTML}'); os.remove(fnHTML)

        # otherwise check the output of each step separately
        else:
            print('\nChecking existence of output per processing step:')
            stepsOutput = {
                'fwc':dirTP,
                'template':[dirTemplate, dirTemplateInter],
                'tbss':dirTBSS,
                'tbss_non_fa':fnNonTBSS,
                'extract':[fnCSV] + fnSkelRegions,
                'qc':[dirQC, fnHTML]
            }
            if len(args.dwi)==1:
                stepsOutput.pop('template')
            shouldExist = []
            shouldNotExist = []
            afterFirst = False
            warnFlag=False
            for k,v in stepsOutput.items():
                if k not in args.steps and not afterFirst:
                    shouldExist.append(k)
                else:
                    shouldNotExist.append(k)
                    afterFirst = True
            for k in shouldExist:
                fn = stepsOutput[k]
                if not isinstance(fn, list): fn = [fn]
                for fnT in fn:
                    if not exists(fnT):
                        raise ValueError(f"Output from skipped processing steps '{k}' is missing: {fnT}")
            if args.reprocess is None:
                for k in shouldNotExist:
                    fn = stepsOutput[k]
                    if not isinstance(fn, list): fn = [fn]
                    for fnT in fn:
                        if exists(fnT):
                            raise ValueError(f"Output for requested step '{k}' exists already: {fnT}\n Tip: Use option '--reprocess' if you want to reprocess and overwrite")
            else:
                for k in shouldNotExist:
                    fn = stepsOutput[k]
                    if not isinstance(fn, list): fn = [fn]
                    for fnT in fn: 
                        if exists(fnT): 
                            warnFlag=True
                            print(f"Warning: Deleting for step '{k}' the already existing output: {fnT}")
                            rmtree(fnT) if os.path.isdir(fnT) else os.remove(fnT)
            if not warnFlag: print('No problems detected!')


    if not os.path.exists(dirTemp):
        print(f"\nCreating temporary folder for processing:\n"
              f"  {dirTemp}")
        Path(dirTemp).mkdir(parents=True, exist_ok=True)


    #----------------------------------
    # Conduct the main processing steps

    # Free water correction
    startTime=None
    if 'fwc' in args.steps:

        print("\nTime point(s) will be copied and processed in following folder(s):")
        for i,tp in enumerate(dirTP):
            print(f' {tp}')

        bIntervals = resolve_b_intervals(bRange=args.bRange, shells=args.shells)
        #--- carried on 'args' for the QC report: the direction count qualifies
        #    every metric the run produces, so it belongs with them
        args.nDirections = []

        for i in range(len(dirTP)):
            startTime = section_header(f'DTI-fit and free-water correction for {i+1}. timepoint in: {dirTP[i]}', startTime)

            Path(dirTP[i]).mkdir(parents=True, exist_ok=True)

            dwi, bval, bvec, nDirections = filter_b_values(fn_data = args.dwi[i],
                                              fn_bval = args.bval[i],
                                              fn_bvec = args.bvec[i],
                                              out_dir = dirTP[i],
                                              bIntervals = bIntervals)
            args.nDirections.append(nDirections)


            print('')
            free_water_correction(fn_data = dwi, 
                        fn_mask = args.bmask[i], 
                        fn_bval = bval, 
                        fn_bvec = bvec,
                        out_dir = dirTP[i],
                        smooth = args.smooth,
                        nproc = min(CORE_BUDGET, FW_MAX_WORKERS))
            
            # Copy and, by default, set brain mask equal 0, where Free-Water equals 1.
            niiBmask = nib.load(args.bmask[i])
            imgBmask = binarise_mask(niiBmask.get_fdata(), 'brain mask', args.bmask[i])
            if args.adjustBmaskForFW:
                print('Setting brain mask to zero, where free-water equals one.')
                niiFAt = nib.load(join(dirTP[i], 'fwc_wls_dti_FA.nii.gz'))
                imgFAt = niiFAt.get_fdata()
                imgBmask[imgFAt==0] = 0
            save_nifti(join(dirTP[i],'brain_mask.nii.gz'), imgBmask, niiBmask.affine, niiBmask.header, dtype='uint8')

    
    
    
    fnCoreg = [
            'fwc_wls_dti_FA_05.nii.gz',
            'wls_dti_FW.nii.gz',
            'wls_dti_MD.nii.gz',
            'brain_mask.nii.gz'
        ]

    # Run template construction
    if 'template' in args.steps  and  len(dirTP)>1:
        Path(dirTemplate).mkdir(parents=True, exist_ok=True)
        startTime = section_header('Template construction and co-registration of all images for all time-points', startTime)
        create_template(timepoints = dirTP, fnCoreg = fnCoreg, dirOut = dirTemplate, coreBudget = CORE_BUDGET, paraOverride = args.para, iterations=args.iterations, numRegistrations=args.numRegistrations, itkThreads=args.itkThreads)

    # Run TBSS
    if len(dirTP)>1:
        fnameFAt = join(dirTemplate, "FA-for-tbss-long.nii.gz")
        tempText = ["template created from ","s"]
    else:
        fnameFAt = join(dirTP[0], "fwc_wls_dti_FA_05.nii.gz")
        tempText = ["",""]
    if 'tbss' in args.steps:
        startTime = section_header('TBSS on {}FA image{}'.format(*tempText), startTime)
        Path(dirTBSS).mkdir(parents=True, exist_ok=True)
        run_tbss(fnameFAt, dirTBSS)

    # Define non-FA maps
    suffix = '_to_template.nii.gz' if len(dirTP)>1 else '.nii.gz'
    nonFA = {
        'FW'    :'wls_dti_FW'+suffix,
        'MD'    :'wls_dti_MD'+suffix,
        'bmask' :'brain_mask'+suffix
    }


    # Run non-FA TBSS
    if 'tbss_non_fa' in args.steps:
        for i in range(len(dirTP)):
            startTime = section_header(f'Non-FA TBSS for {i+1}. timepoint in: {dirTP[i]}', startTime)
            batch_tbss_non_fa(dirTP[i], dirTBSS, nonFA)
        
        startTime = section_header('Non-FA TBSS for additional mask images', startTime)

        # Transform and merge the optional exclusion/ROI masks (None if none given)
        Emask = coreg_merge_masks(timepoints = dirTP, masks = args.Emask, label='mask_exclusive', dirTemplate = dirTemplate, binarise = True)
        Rmask = coreg_merge_masks(timepoints = dirTP, masks = args.Rmask, label='mask_roi', dirTemplate = dirTemplate, binarise=False)
        flagAdditionalMasks = False
        if Emask is not None:
            run_tbss_non_fa(Emask, 'E-MASK', dirTBSS)
            flagAdditionalMasks = True
        if Rmask is not None: 
            niiROI = nib.load(Rmask)
            imgROI = niiROI.get_fdata()
            uROI = np.unique(imgROI)
            uROI = uROI[uROI>0].astype('uint8')
            # one label at a time: the MNI registration does not use nearest-neighbor
            for roi in uROI:
                imgT = imgROI.copy()
                imgT[imgROI!=roi] = 0
                fnOut = re.sub(r'\.nii(\.gz)?$','-{:02d}.nii.gz'.format(roi), Rmask)
                save_nifti(fnOut, imgT>0, niiROI.affine, niiROI.header, 'uint8')
                run_tbss_non_fa(fnOut, 'ROI-{:02d}'.format(roi), dirTBSS)
            flagAdditionalMasks = True
        if not flagAdditionalMasks:
            print('No additional masks were provided!')
        

    # Extract statistics
    if 'extract' in args.steps:

        dfL = []

        if len(dirTP)>1:
            startTime = section_header('Finding skeleton voxels present in brain masks of all timepoints and in the skeleton mask and ROIs', startTime)
        else:
            startTime = section_header('Finding skeleton voxels present in the brain mask and in the skeleton mask and ROIs', startTime)
        dfT = integrate_masks(dirTP, dirTBSS, skelMask = args.skeletonMask, fnROI_MNI = args.RmaskMNI, analyseHemispheres=args.hemispheres)
        dfL.append(dfT)
        print('Done analysing skeletons and masks:')
        print(dfL[0].iloc[:,0:4])
        
        # Extract values
        nonFA.pop('bmask')
        for i in range(len(dirTP)):
            startTime = section_header(f'Extract histogram statistics for {i+1}. timepoint in: {dirTP[i]}', startTime)
            
            dfT = extract_stats(dirTP[i], dirTBSS, nonFA, skelMask = args.skeletonMask)
            dfL.append(dfT)
            
        df = pd.concat(dfL)
        if args.id is not None:
            df.insert(0,'ID',args.id)

        print('\n\nSummary statistics:\n')
        pd.set_option('display.max_rows', 1000)
        pd.set_option('display.max_columns', 10)
        pd.set_option('display.width', 1000)
        print(df)

        df.to_csv(fnCSV, index = False)
        print(f'\nSummary statistics were saved to:\n{fnCSV}')


    # Prepare images for QC
    if 'qc' in args.steps:

        startTime = section_header('Creating QC images (deprojecting skeleton mask and transforming into native space)', startTime)
        Path(dirQC).mkdir(parents=True, exist_ok=True)
        prepare_qc(dirQC, fnHTML, args.skeletonMask, dirTBSS, dirTemplate, dirTP, fnCSV, args)


    # Clean up
    section_header('Finalising:', startTime)
    if args.debug:
        print(f'Keeping temporary folder: {dirTemp}')
    else:
        print(f'Deleting temporary folder: {dirTemp}')
        rmtree(dirTemp)
    
    elapsed = time.time() - start_script
    print('\nTotal duration: {:02.0f}:{:02.0f}\n'.format(elapsed//60, elapsed%60))



if __name__ == "__main__":
    pipeline_delta_svd()
