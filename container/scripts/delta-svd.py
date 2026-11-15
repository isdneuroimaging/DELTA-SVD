#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, re, time, glob, shlex, multiprocessing, json, datetime, contextlib
from os.path import join, exists, basename
from shutil import copy2
from pathlib import Path

from delta_svd_constants import ITK_THREADS_DEFAULT, ANTS_TEMPLATE_INIT

from delta_svd_runtime import (running_in_image, pin_image_environment,
                               isolated_interpreter_argv, resolve_thread_budget,
                               resolve_itk_threads, check_cpu_supports_coretype)

# Before anything else reads the environment or imports a third-party package.
# The Dockerfile ENTRYPOINT already starts Python with -E -s -u; the re-exec
# covers the script being started any other way inside the image.
if running_in_image():
    pin_image_environment(os.environ)
    if __name__ == '__main__':
        _argv = isolated_interpreter_argv(sys.flags, sys.argv[1:], os.path.realpath(__file__))
        if _argv is not None:
            os.execv(sys.executable, _argv)


# How far the template step may oversubscribe the core budget, as a fraction:
# 3/2 = 1.5 threads per core. Measured faster than an exactly-fitting plan, and
# without numerical consequence, so this is purely a throughput choice.
OVERSUBSCRIBE_NUM, OVERSUBSCRIBE_DEN = 3, 2


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
from nibabel.orientations import (io_orientation, axcodes2ornt, ornt_transform,
                                  apply_orientation, inv_ornt_aff)
import numpy as np
import pandas as pd

from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from dipy.reconst.dti import (design_matrix, decompose_tensor,
                           from_lower_triangular)

from scipy.ndimage import gaussian_filter

from markvcid_fw_mrn import wls_fit_tensor_fw, wls_fit_dti
from delta_svd_version import __version__, __source_revision__
from delta_svd_common import (DeltaSvdError, ROI_LABEL_DTYPE, save_nifti,
                              strip_nifti_ext, tbss_fa_name, validate_roi_labels,
                              run_subprocess, safe_rmtree)
from delta_svd_gradients import resolve_b_intervals, filter_b_values
from delta_svd_inputs import iniParser, resolve_inputs


###########################################################################
# Functions for major processing steps


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
    # Thresholded like binarise_mask() (negative/NaN is not brain), but without its
    # NOTE, which is printed once later. Done here because from numpy 2.0 the
    # vendored fits' np.array(mask, dtype=bool, copy=False) raises on a float mask.
    mask = niim.get_fdata() > 0
    bvals, bvecs = read_bvals_bvecs(fn_bval, fn_bvec)
    print(f'bvals = \n{bvals}\n')

    gtab = gradient_table(bvals, bvecs=bvecs)
    
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
    if not pCSF.any():
        # np.mean() of nothing is NaN, which would silently NaN every FW estimate
        raise DeltaSvdError(f"No voxel inside the brain mask has a mean diffusivity above 0.002 mm2/s "
                            f"in the single-tensor fit, so the CSF reference value needed to "
                            f"regularise the free-water fit cannot be estimated.\n"
                            f"  DWI       : {fn_data}\n  brain mask: {fn_mask}\n"
                            f" The brain mask has to include some CSF (e.g. the ventricles), so "
                            f"check that it is not eroded or empty and that it matches the DWI "
                            f"image. Also check that the b-values are given in s/mm2.")
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
    print(f"Template construction: {len(timepoints)} timepoints, core budget {coreBudget} "
          f"-> {para} parallel registration job(s) x {itkThreads} ITK thread(s) per job ({mode} mode)")
    print(f"  {para * itkThreads} thread(s) over {coreBudget} core(s) = {perCore:.2f} per core")
    if perCore > 2:
        print("  NOTE: the threads outnumber the cores several times over. This affects only the "
              "runtime, never the results; allocate more cores to bring it down.")

    # ANTs' template construction script uses its paths unquoted, so whitespace
    # would split them. Running inside 'dirOut' keeps its path out of the command.
    cmd = (f"export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={itkThreads}; "
           "export ANTS_RANDOM_SEED=1; "
           f"antsMultivariateTemplateConstruction2.sh -d 3 -i {int(numRegistrations)} -f 4x2x1 -s 2x1x0vox -q {shlex.quote(iterations)} -t SyN -m CC "
           f" -r 1 -z {ANTS_TEMPLATE_INIT} -y 0 {control} -o ./ {' '.join(shlex.quote(basename(f)) for f in fnFA)}")
    with contextlib.chdir(dirOut):
        run_subprocess(cmd, False, 'antsMultivariateTemplateConstruction2.sh')

    fnAverage = []
    for iTP, tp in enumerate(timepoints):
        tpB = basename(tp)
        for iFn, fn in enumerate(fnCoreg):
            fnIn = join(tp, fn)
            fnOut = strip_nifti_ext(fnIn) + '_to_template.nii.gz'
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
                    nii = nib.load(fnIn)
                    validate_roi_labels(nii.get_fdata(), 'ROI mask in DWI space', fnIn)
                    copy_as_nii_gz(fnIn, fnOut)
                if len(timepoints)==1:
                    # on the DWI grid, so reoriented like the maps
                    reorient_to_fsl_standard(fnOut)
                else:
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
    print('Merging masks across timepoints:\n',fnMerge)
    img = []
    for fn in fnMerge:
        nii = nib.load(fn)
        img.append(nii.get_fdata())
    img = np.amax(np.stack(img,-1), -1)
    validate_roi_labels(img, 'ROI mask in DWI space', fnOut)
    save_nifti(fnOut, img, nii.affine, nii.header, ROI_LABEL_DTYPE)
    
    return fnOut


def run_tbss(fnameFAt = None, dirTBSS = None):
    
    copy2(fnameFAt, join(dirTBSS, basename(fnameFAt)))

    with contextlib.chdir(dirTBSS):
        cmd = f'tbss_1_preproc {shlex.quote(basename(fnameFAt))}'
        run_subprocess(cmd, True, 'tbss_1_preproc')
        cmd = 'tbss_2_reg -T'
        run_subprocess(cmd, True, 'tbss_2_reg')
        cmd = 'tbss_3_postreg -T'
        run_subprocess(cmd, True, 'tbss_3_postreg')
        cmd = 'tbss_4_prestats 0.2'
        run_subprocess(cmd, True, 'tbss_4_prestats')

    
def batch_tbss_non_fa(dirTP = None, dirTBSS = None, fnNonFA = None):
    
    fnameFAt = glob.glob(join(glob.escape(dirTBSS), 'FA', '*_FA.nii.gz'))
    fnameFAt = re.sub(r'_FA\.nii\.gz','.nii.gz',basename(fnameFAt[0]))

    tpB = basename(dirTP)

    for mapName, fn in fnNonFA.items():
        run_tbss_non_fa(join(dirTP,fn), tpB+'_'+mapName, dirTBSS, fnameFAt)


def run_tbss_non_fa(fn = None, label=None, dirTBSS = None, fnameFAt = None):
    
    # get name of FA used for projection onto skeleton
    if fnameFAt is None:
        fnameFAt = glob.glob(join(glob.escape(dirTBSS), 'FA', '*_FA.nii.gz'))
        fnameFAt = re.sub(r'_FA\.nii\.gz','.nii.gz',basename(fnameFAt[0]))

    dirTBSS_nonFA = join(dirTBSS, label)
    Path(dirTBSS_nonFA).mkdir(exist_ok=True)
    copy_as_nii_gz(fn, join(dirTBSS_nonFA, fnameFAt))

    with contextlib.chdir(dirTBSS):
        cmd = f"tbss_non_FA {shlex.quote(label)}"
        run_subprocess(cmd, True, 'tbss_non_FA')


def integrate_masks(dirTP = [], dirTBSS = None, skelMask = None, fnROI_MNI = None, analyseHemispheres = False):
    
    tpAll = [basename(tp) for tp in dirTP]
    
    skelBase = strip_nifti_ext(basename(skelMask))
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
    save_nifti(join(dirTBSS, 'stats', skelBase+'_'+skelSuffix+'.nii.gz'), maskIntersection, niiMask.affine, niiMask.header, dtype='uint8')

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
        pnameSkelIntersExcLabelled = join(dirTBSS, 'stats', skelBase+'_'+skelSuffixL2+'.nii.gz')
        save_nifti(pnameSkelIntersExcLabelled, imgEmask, niiMask.affine, niiMask.header, dtype='uint8')

        skelSuffix = skelSuffix + '_Emask'
        pnameSkelIntersExc = join(dirTBSS, 'stats', skelBase+'_'+skelSuffix+'.nii.gz')
        save_nifti(pnameSkelIntersExc, maskIntersection, niiMask.affine, niiMask.header, dtype='uint8')

        voxels.append(np.count_nonzero(maskIntersection))
        timept.append(timeptT)
        region.append(skelSuffix)

    # numeric labels only: a timepoint labelled 'ROI-1' writes 'all_ROI-1_FW_skeletonised'
    fnROI = sorted(fn for fn in glob.glob(join(glob.escape(dirTBSS), 'stats', 'all_ROI-*_skeletonised.nii.gz'))
                   if re.fullmatch(r'all_ROI-[0-9]+_skeletonised\.nii\.gz', basename(fn)))
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

        #- no merged label map: ROIs may overlap, which merging would hide
        if iFn==0: roiUnion = np.zeros(imgROI.shape, bool)
        roiUnion |= imgROI>0
    # complementary ROI for the background
    if len(fnROI)>0:
        imgROI = maskIntersection.copy()
        imgROI[roiUnion] = 0
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
        uROI = validate_roi_labels(imgROI_MNI, 'ROI mask in MNI space', fnROI_MNI)
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
            if iRoi==0: imgROImerged = np.zeros(imgROI_MNI.shape, ROI_LABEL_DTYPE)
            if roi>0:
                imgROImerged[imgROI_MNI_roi>0] = roi
            if iRoi==len(uROI)-1 and np.count_nonzero(imgROImerged)>0: 
                pnameROImerged = join(dirTBSS, 'stats', skelBase+'_'+skelSuffix+'_RmaskMNI.nii.gz')
                save_nifti(pnameROImerged, imgROImerged, niiROI_MNI.affine, niiROI_MNI.header, dtype=ROI_LABEL_DTYPE)

    
    if analyseHemispheres:
        # Splitting the first axis in half assumes the FMRIB58 1 mm grid (LAS, midline
        # centred); check_mni_grid() enforces it for custom masks too.
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

def extract_stats(dirTP = None, dirTBSS = None, fnNonFA = None, skelMask = None):
    
    skelBase = strip_nifti_ext(basename(skelMask))
    skelMaskInters = join(dirTBSS, 'stats', skelBase+'_intersection_Emask.nii.gz')
    if not exists(skelMaskInters):
        skelMaskInters = join(dirTBSS, 'stats', skelBase+'_intersection.nii.gz')

    fnROI = [skelMaskInters]
    fnROI = fnROI + sorted(glob.glob(join(glob.escape(dirTBSS), 'stats', '*_Rmask-*.nii.gz')))
    fnROI = fnROI + sorted(glob.glob(join(glob.escape(dirTBSS), 'stats', '*_RmaskMNI-*.nii.gz')))
    fnROI = fnROI + sorted(glob.glob(join(glob.escape(dirTBSS), 'stats', '*_LH.nii.gz')))
    fnROI = fnROI + sorted(glob.glob(join(glob.escape(dirTBSS), 'stats', '*_RH.nii.gz')))



    dd = []
    tpB = basename(dirTP)
    # Loaded once here rather than once per region: with many ROI labels the
    # repeated decompression added up to tens of seconds per timepoint.
    maps = {mapName: nib.load(join(dirTBSS, 'stats', 'all_'+tpB+'_'+mapName+'_skeletonised.nii.gz')).get_fdata()
            for mapName in fnNonFA}
    for fnR in fnROI:
        
        niiROI = nib.load(fnR)
        roi = niiROI.get_fdata()
        roiBase = strip_nifti_ext(basename(fnR))
        roiSuffix = roiBase.removeprefix(skelBase+'_')

        for mapName, _ in fnNonFA.items():

            print('\nCalculating endpoint metrics for:')
            print(f' region    : {roiSuffix}')
            print(f' map       : {mapName}')

            img = maps[mapName]
            skel = img[roi>0]
            print( ' voxels  :',len(skel))
            mean = np.mean(skel) if len(skel)>0 else np.nan

            if mapName == 'MD':
                prcts = np.percentile(skel,[5,95]) if len(skel)>0 else [np.nan]*2
                pw = prcts[1] - prcts[0]
                metrics = ['PSMD', 'MS'+mapName]
                values = [pw, mean]
            else:
                metrics = ['MS'+mapName]
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

def qc_label_maps(dirTBSS):
    """The ROI maps in skeleton space: one binary map per DWI-space ROI label,
    kept separate so overlapping labels remain visible, and the merged
    MNI-space label map."""
    # Match only numeric label suffixes, excluding prior QC deprojection outputs.
    return (sorted(fn for fn in glob.glob(join(glob.escape(dirTBSS), 'stats', '*_Rmask-*.nii.gz'))
                   if re.search(r'_Rmask-[0-9]+\.nii\.gz$', basename(fn)))
            + sorted(glob.glob(join(glob.escape(dirTBSS), 'stats', '*_RmaskMNI.nii.gz'))))


def consecutive_labels(img):
    """'img' with its non-zero labels renumbered 1..K in ascending order, and
    the original label of each: (renumbered, labels), labels[k-1] for k."""
    labels = np.unique(img[img > 0])
    out = np.zeros(img.shape, ROI_LABEL_DTYPE)
    out[img > 0] = np.searchsorted(labels, img[img > 0]) + 1
    return out, labels


def restore_labels(img, labels):
    """Undo consecutive_labels() on an image carrying its numbering."""
    idx = np.rint(img).astype(np.int64)
    out = np.zeros(img.shape, ROI_LABEL_DTYPE)
    out[idx > 0] = labels[idx[idx > 0] - 1]
    return out


def deproject_to_native(fn, dirQC, dirTBSS, dirTemplate, dirTP, relabel=False):
    """Deproject the skeleton-space image 'fn' (in TBSS/stats) into the native
    space of every timepoint, as '<dirQC>/<timepoint>_<name of fn>'.

    'relabel' is for ROI label maps. With '-n', tbss_deproject runs one
    applywarp for every integer from 1 up to the largest label, present or
    not, so a label such as 1000 costs a thousand warps. The labels are
    renumbered 1..K for the deprojection and restored afterwards. The
    renumbering keeps their order, so where labels meet, the one that wins is
    the same as without it."""
    fnameFAt = tbss_fa_name(len(dirTP))
    fnBase = basename(fn)
    labels = None
    src = fn
    if relabel:
        nii = nib.load(fn)
        consecutive, labels = consecutive_labels(nii.get_fdata())
        src = strip_nifti_ext(fn) + '_consecutive.nii.gz'
        save_nifti(src, consecutive, nii.affine, nii.header, ROI_LABEL_DTYPE)

    with contextlib.chdir(join(dirTBSS,'stats')):
        cmd = f'tbss_deproject {shlex.quote(basename(src))} 2 -n'
        run_subprocess(cmd, True, 'tbss_deproject')
    fnDeprojectedTemplateSpace = join(dirTBSS, 'FA', fnameFAt+'_FA_'+basename(src))

    for iTP in range(len(dirTP)):
        tpB = basename(dirTP[iTP])
        fnOut = join(dirQC, tpB+'_'+fnBase)
        if len(dirTP)>1:
            ref = join(dirTP[iTP], 'fwc_wls_dti_FA.nii.gz')
            affine = join(dirTemplate, f'{tpB}_fwc_wls_dti_FA_05{iTP}0GenericAffine.mat')
            invwarp = join(dirTemplate, f'{tpB}_fwc_wls_dti_FA_05{iTP}1InverseWarp.nii.gz')
            cmd = (f"antsApplyTransforms -d 3 --float 1 -i {shlex.quote(fnDeprojectedTemplateSpace)}  -o {shlex.quote(fnOut)} "
                   f"-r {shlex.quote(ref)} -t [{shlex.quote(affine)},1] -t {shlex.quote(invwarp)} -n NearestNeighbor")
            run_subprocess(cmd, True, 'antsApplyTransforms')
        else:
            copy2(fnDeprojectedTemplateSpace, fnOut)
        if labels is not None:
            nii = nib.load(fnOut)
            save_nifti(fnOut, restore_labels(nii.get_fdata(), labels), nii.affine, nii.header,
                       ROI_LABEL_DTYPE)


def prepare_qc(dirQC, fnHTML, skelMask, dirTBSS, dirTemplate, dirTP, fnCSV, args):
        
    from create_qc_image import create_qc_image
    from create_html_with_png import create_html_with_png
    
    skelBase = strip_nifti_ext(basename(skelMask))
    skelMask = join(dirTBSS, 'stats', skelBase+'_intersection_Emask-as-label2.nii.gz')
    emaskExists = 1
    if not exists(skelMask):
        skelMask = join(dirTBSS, 'stats', skelBase+'_intersection.nii.gz')
        emaskExists = 0
    
    # ROI maps are kept for inspection with '--qc 2', not shown in the report.
    fnROI = [skelMask] + (qc_label_maps(dirTBSS) if args.qc >= 2 else [])

    fnameFAt = tbss_fa_name(len(dirTP))

    for fn in fnROI:
        deproject_to_native(fn, dirQC, dirTBSS, dirTemplate, dirTP,
                            relabel=fn.endswith('_RmaskMNI.nii.gz'))
    
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

    #--- in space of within-subject template
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

    fnROI_base = strip_nifti_ext(basename(fnROI[0]))
    fnSkeleton = join(dirTBSS, 'stats', fnROI_base+'_to_all_FA.nii.gz')
    fnameCmask = join(dirTBSS, 'stats', 'mean_FA_mask.nii.gz')
    fnPNG = fnPNG + [create_qc_image(fnames, vlim, labels, fnSkeleton, fnamesBmask, fnameCmask)]
    captions = captions + ['MNI space']

    print('\nSaving QC to:\n ', fnHTML)

    create_html_with_png(fnHTML, fnPNG, captions, fnCSV, args)

    if args.qc < 2:
        safe_rmtree(dirQC, args.dirOutput)


###########################################################################
# Helper functions

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


def copy_as_nii_gz(fnIn, fnOut):
    """Place 'fnIn' at 'fnOut', whose name always ends in '.nii.gz'. A gzipped
    input is copied verbatim; an uncompressed one is re-encoded, because nibabel,
    FSL and ANTs pick the codec from the file *name*. Raw stored values and
    slope/intercept are kept, so the decoded values stay exactly equal."""
    if fnIn.endswith('.nii.gz'):
        copy2(fnIn, fnOut)
    else:
        nii = nib.load(fnIn)
        raw = np.asanyarray(nii.dataobj.get_unscaled())
        niiNew = nib.Nifti1Image(raw, nii.affine, nii.header)
        niiNew.set_data_dtype(nii.get_data_dtype())
        niiNew.header.set_slope_inter(nii.dataobj.slope, nii.dataobj.inter)
        nib.save(niiNew, fnOut)


#--- FSL's standard voxel order. LAS and RAS (validated; FSL handles the x flip)
#    are left alone; any other order starts FLIRT 90-180 degrees off, and FNIRT
#    then ran >15 min instead of ~1 on the reference subject stored as LPS.
FSL_STANDARD_AXCODES = ('L', 'A', 'S')

#--- Per-timepoint maps used by TBSS and QC in a cross-sectional run.
FSL_REORIENTED_MAPS = ('wls_dti_FA.nii.gz', 'wls_dti_MD.nii.gz', 'wls_dti_FW.nii.gz',
                       'fwc_wls_dti_FA.nii.gz', 'fwc_wls_dti_FA_05.nii.gz',
                       'brain_mask.nii.gz')


def needs_fsl_reorientation(affine):
    """True unless 'affine' stores voxels as LAS or RAS."""
    return nib.aff2axcodes(affine)[1:] != FSL_STANDARD_AXCODES[1:]


def fsl_reorientation_source(affines):
    """The voxel order (e.g. 'LPS') the maps are reordered from, or None. Only a
    cross-sectional run is reordered: the longitudinal template works in world space."""
    if len(affines) != 1 or not needs_fsl_reorientation(affines[0]):
        return None
    return ''.join(nib.aff2axcodes(affines[0]))


def mixed_voxel_order_warning(voxelOrder, tp):
    """A warning when the timepoints' voxel orders differ, else None."""
    if len(set(voxelOrder)) < 2:
        return None
    shown = ', '.join(f'{o} ({t})' for o, t in zip(voxelOrder, tp))
    return (f'\nWARNING: the timepoints are stored in different voxel orders: {shown}.\n'
            f' This affects the longitudinal change. Convert and preprocess all timepoints identically.')


def reorient_to_fsl_standard(fn):
    """Rewrite 'fn' in place in LAS order if needs_fsl_reorientation(); returns
    whether it did. Pure voxel permutation: values, dtype, scaling and world
    positions are kept, nothing is resampled."""
    nii = nib.load(fn)
    if not needs_fsl_reorientation(nii.affine):
        return False
    ornt = ornt_transform(io_orientation(nii.affine), axcodes2ornt(FSL_STANDARD_AXCODES))
    raw = apply_orientation(np.asanyarray(nii.dataobj.get_unscaled()), ornt)
    affine = nii.affine @ inv_ornt_aff(ornt, nii.shape)
    niiNew = nib.Nifti1Image(raw, affine, nii.header)
    niiNew.set_data_dtype(nii.get_data_dtype())
    niiNew.header.set_slope_inter(nii.dataobj.slope, nii.dataobj.inter)
    # keep the codes, and the voxel sizes exactly: recomputed from the affine they
    # can move by a float32 ULP, which FSL and ANTs amplify into the metrics
    niiNew.header.set_sform(affine, int(nii.header['sform_code']))
    niiNew.header.set_qform(affine, int(nii.header['qform_code']))
    zooms = list(nii.header.get_zooms())
    for i, (j, _) in enumerate(ornt.astype(int)):
        zooms[j] = nii.header.get_zooms()[i]
    niiNew.header.set_zooms(zooms)
    # freq/phase/slice axes move too, as in nibabel's as_reoriented()
    niiNew.header.set_dim_info(*[None if d is None else int(ornt[d, 0])
                                 for d in nii.header.get_dim_info()])
    nib.save(niiNew, fn)
    return True


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


def utc_timestamp(value=None):
    """Return an RFC 3339 timestamp in UTC, using the manifest's Z form."""
    if value is None:
        value = datetime.datetime.now(datetime.timezone.utc)
    return value.astimezone(datetime.timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def write_run_manifest(fname, args, startedAt, outputs, completedAt=None):
    """Write the successful-run manifest atomically beside the final outputs.

    The temporary file is only for protecting the small manifest write itself;
    pipeline outputs continue to be written in their existing locations.
    """
    manifest = {
        'manifest_schema_version': 1,
        'pipeline': 'DELTA-SVD',
        'pipeline_version': __version__,
        'source_revision': __source_revision__,
        'subject_id': args.id,
        'processing_mode': 'longitudinal' if len(args.dwi) > 1 else 'cross_sectional',
        'command': args.function_call,
        'started_at': utc_timestamp(startedAt),
        'completed_at': utc_timestamp(completedAt),
        'steps_completed': list(args.steps),
        'qc_mode': args.qc,
        'outputs': list(outputs),
        # stored voxel order of each timepoint's DWI, before any reordering
        'voxel_order': dict(zip(args.tp, args.voxelOrder)),
    }
    temporary = fname + '.tmp'
    try:
        with open(temporary, 'w') as fp:
            json.dump(manifest, fp, indent=2)
            fp.write('\n')
        os.replace(temporary, fname)
    except Exception:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


###########################################################################
# Pipeline

def pipeline_delta_svd():

    start_script = time.time()
    started_at = datetime.datetime.now(datetime.timezone.utc)
    parser = iniParser()
    if len(sys.argv)<2:
        parser.print_usage()
        print(f'\nDELTA-SVD {__version__}\n'
              'Run "delta-svd.py -h" for detailed help\n'
              'Notice: By using DELTA-SVD, you agree to the licence terms (CC BY-NC-ND 4.0) described in the LICENSE file at "https://github.com/isdneuroimaging/DELTA-SVD"\n')
        parser.exit()
    else:
        args = parser.parse_args(sys.argv[1::])

    # On its own line, so it survives a log truncated on the long call below.
    # Carried on 'args' too, for the QC report to record.
    args.version = __version__
    print(f"DELTA-SVD {__version__}")
    args.function_call = shlex.join([basename(sys.argv[0])]+sys.argv[1::])
    print("Running: " + args.function_call,'\n')

    stepsAvailable = resolve_inputs(args)

    # Carried on 'args' for the QC report and manifest: storage order affects
    # the results. The masks share the DWI grid.
    affines = [nib.load(dwi).affine for dwi in args.dwi]
    args.voxelOrder = [''.join(nib.aff2axcodes(a)) for a in affines]
    warning = mixed_voxel_order_warning(args.voxelOrder, args.tp)
    if warning:
        print(warning)
    args.reorientedFrom = fsl_reorientation_source(affines)
    if args.reorientedFrom is not None:
        print(f'\nNOTE: the DWI image stores its voxels as {args.reorientedFrom}. The maps '
              f'will be reordered to {"".join(FSL_STANDARD_AXCODES)} (no resampling) '
              f'before TBSS, whose registration needs LAS or RAS input.')

    dirTemp = join(args.dirOutput, 'delta-svd_temp')
    dirTP = [join(dirTemp, tp) for tp in args.tp] #-- folders for all timepoints
    dirTemplate = join(dirTemp,'template')
    dirTemplateInter = join(dirTemplate, 'intermediateTemplates')
    dirTBSS = join(dirTemp,'TBSS')
    fnNonTBSS = glob.glob(join(glob.escape(dirTBSS),'*')) + glob.glob(join(glob.escape(dirTBSS), 'stats','*'))
    fnNonTBSS = [x for x in fnNonTBSS if not re.match('FA$|origdata$|stats$|all_FA|mean_FA|thresh',basename(x))]
    if args.reprocess is None or args.reprocess == 'overwrite':
        fnCSV = join(args.dirOutput, 'delta-svd_results.csv')
    else:
        fnCSV = join(args.dirOutput, args.reprocess)
    fnSkelRegions = glob.glob(join(glob.escape(dirTBSS), 'stats','*_intersection*'))
    dirQC = join(args.dirOutput, 'delta-svd_qc')
    fnHTML = join(args.dirOutput, 'delta-svd_qc.html')
    fnManifest = join(args.dirOutput, 'delta-svd_run_manifest.json')
    fnManifestTemp = fnManifest + '.tmp'
    
    # Check if output exists already
    outputExisting = [p for p in (dirTemp, fnCSV, dirQC, fnHTML, fnManifest, fnManifestTemp)
                      if os.path.exists(p)]
    if outputExisting:

        # all steps requested: simply delete everything
        if set(args.steps+['qc']) == set(stepsAvailable+['qc']):
            print('\nChecking existence of output')
            if args.reprocess is None:
                # the paths themselves, not '--dirOutput': that is an empty string
                # when the DWI was given as a bare name in the working directory
                raise DeltaSvdError("Output from a previous run exists already:\n"
                                    + '\n'.join(f'   {p}' for p in outputExisting)
                                    + "\n Use option '--reprocess' to reprocess and overwrite it.")
            if os.path.exists(dirTemp): print(f'Deleting: {dirTemp}'); safe_rmtree(dirTemp, args.dirOutput)
            if os.path.exists(fnCSV): print(f'Deleting: {fnCSV}'); os.remove(fnCSV)
            if os.path.exists(dirQC): print(f'Deleting: {dirQC}'); safe_rmtree(dirQC, args.dirOutput)
            if os.path.exists(fnHTML): print(f'Deleting: {fnHTML}'); os.remove(fnHTML)
            if os.path.exists(fnManifest): print(f'Deleting: {fnManifest}'); os.remove(fnManifest)
            if os.path.exists(fnManifestTemp): print(f'Deleting: {fnManifestTemp}'); os.remove(fnManifestTemp)

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
                        raise DeltaSvdError(f"Step '{k}' is not among the requested '--steps', but its output "
                                            f"is missing and the requested steps need it:\n {fnT}")
            if args.reprocess is None:
                for k in shouldNotExist:
                    fn = stepsOutput[k]
                    if not isinstance(fn, list): fn = [fn]
                    for fnT in fn:
                        if exists(fnT):
                            raise DeltaSvdError(f"Output for the requested step '{k}' exists already:\n {fnT}\n "
                                                f"Use option '--reprocess' to reprocess and overwrite it.")
            else:
                for k in shouldNotExist:
                    fn = stepsOutput[k]
                    if not isinstance(fn, list): fn = [fn]
                    for fnT in fn: 
                        if exists(fnT): 
                            warnFlag=True
                            print(f"Warning: Deleting for step '{k}' the already existing output: {fnT}")
                            if os.path.isdir(fnT):
                                safe_rmtree(fnT, args.dirOutput if k == 'qc' else dirTemp)
                            else:
                                os.remove(fnT)
            if not warnFlag: print('No problems detected!')

            # The manifest describes the whole run, so any rerun of selected
            # steps invalidates it even though it is not one processing step.
            for fnT in (fnManifest, fnManifestTemp):
                if os.path.exists(fnT):
                    if args.reprocess is None:
                        raise DeltaSvdError(f"Run manifest exists already:\n {fnManifest}\n "
                                            "Use option '--reprocess' to reprocess and overwrite it.")
                    print(f'Warning: Deleting the already existing run manifest: {fnT}')
                    os.remove(fnT)


    if not os.path.exists(dirTemp):
        print(f"\nCreating temporary folder for processing:\n"
              f"  {dirTemp}")
        Path(dirTemp).mkdir(parents=True, exist_ok=True)


    #----------------------------------
    # Conduct the main processing steps

    # Free water correction
    startTime=None
    if 'fwc' in args.steps:

        print("\nTimepoints will be copied and processed in following folder(s):")
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

            # cross-sectional only: the ANTs template works in world space
            if len(dirTP) == 1 and args.reorientedFrom is not None:
                print(f'Reordering the voxels of the maps from {args.reorientedFrom} to '
                      f'{"".join(FSL_STANDARD_AXCODES)} for TBSS.')
                for fn in FSL_REORIENTED_MAPS:
                    reorient_to_fsl_standard(join(dirTP[i], fn))

    
    
    
    fnCoreg = [
            'fwc_wls_dti_FA_05.nii.gz',
            'wls_dti_FW.nii.gz',
            'wls_dti_MD.nii.gz',
            'brain_mask.nii.gz'
        ]

    # Run template construction
    if 'template' in args.steps  and  len(dirTP)>1:
        Path(dirTemplate).mkdir(parents=True, exist_ok=True)
        startTime = section_header('Template construction and co-registration of all images for all timepoints', startTime)
        create_template(timepoints = dirTP, fnCoreg = fnCoreg, dirOut = dirTemplate, coreBudget = CORE_BUDGET, paraOverride = args.para, iterations=args.iterations, numRegistrations=args.numRegistrations, itkThreads=args.itkThreads)

    # Run TBSS
    if len(dirTP)>1:
        fnameFAt = join(dirTemplate, tbss_fa_name(len(dirTP)) + ".nii.gz")
        tempText = ["template created from ","s"]
    else:
        fnameFAt = join(dirTP[0], tbss_fa_name(len(dirTP)) + ".nii.gz")
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
            uROI = validate_roi_labels(imgROI, 'ROI mask in DWI space', Rmask)
            uROI = uROI[uROI>0]
            # one label at a time: the MNI registration does not use nearest-neighbor
            for roi in uROI:
                imgT = imgROI.copy()
                imgT[imgROI!=roi] = 0
                fnOut = strip_nifti_ext(Rmask) + '-{:02d}.nii.gz'.format(roi)
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
            startTime = section_header(f'Calculate endpoint metrics for {i+1}. timepoint in: {dirTP[i]}', startTime)
            
            dfT = extract_stats(dirTP[i], dirTBSS, nonFA, skelMask = args.skeletonMask)
            dfL.append(dfT)
            
        df = pd.concat(dfL)
        if args.id is not None:
            df.insert(0,'ID',args.id)

        print('\n\nResults table:\n')
        pd.set_option('display.max_rows', 1000)
        pd.set_option('display.max_columns', 10)
        pd.set_option('display.width', 1000)
        print(df)

        df.to_csv(fnCSV, index = False)
        print(f'\nResults table was saved to:\n{fnCSV}')


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
        safe_rmtree(dirTemp, args.dirOutput)

    outputs = []
    if 'extract' in args.steps:
        outputs.append(os.path.basename(fnCSV))
    if 'qc' in args.steps:
        outputs.append(os.path.basename(fnHTML))
    write_run_manifest(fnManifest, args, started_at, outputs)
    print(f'Run manifest was saved to:\n{fnManifest}')
    
    elapsed = time.time() - start_script
    print('\nTotal duration: {:02.0f}:{:02.0f}\n'.format(elapsed//60, elapsed%60))



if __name__ == "__main__":
    # Only DeltaSvdError is caught, and only here rather than inside
    # pipeline_delta_svd(): a problem with the input is reported as one message,
    # the way argparse reports its own, while anything else keeps its traceback
    # -- an unexpected exception is a fault in the pipeline, and the traceback is
    # what makes it reportable. Catching it here also leaves the pipeline
    # importable and testable as a plain function.
    try:
        pipeline_delta_svd()
    except DeltaSvdError as err:
        sys.exit(f"\nERROR: {err}\n")
    except KeyboardInterrupt:
        sys.exit("\nABORTED: interrupted by the user\n")
