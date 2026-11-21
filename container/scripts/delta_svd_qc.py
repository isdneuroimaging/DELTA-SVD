#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quality control: deprojection of the skeleton and ROI maps into native
space and assembly of the QC report.

Imports numpy and nibabel, so it must be imported after delta-svd.py has
pinned the threads and the BLAS kernel.
"""

import re, glob, shlex, contextlib
from os.path import join, exists, basename
from shutil import copy2

import nibabel as nib
import numpy as np

from delta_svd_common import (ROI_LABEL_DTYPE, save_nifti,
                              strip_nifti_ext, tbss_fa_name,
                              run_subprocess, safe_rmtree)


###########################################################################
# Functions for quality control

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
