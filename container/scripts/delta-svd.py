#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys, argparse, re, subprocess, time, glob
from os.path import join, exists, dirname, basename
import nibabel as nib
import numpy as np
import pandas as pd
# import string
from shutil import copy2, rmtree
from pathlib import Path

from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from dipy.reconst.dti import (design_matrix, decompose_tensor,
                           from_lower_triangular)
from dipy.core.ndindex import ndindex

from scipy.ndimage import gaussian_filter

###########################################################################
# Functions for tensor fitting and free water correction

# Constant declarations
def wls_fit_tensor_fw(W, data, md_data,S0, Diso=3e-3, mask=None, 
                min_signal=1.0e-6, piterations=2, mdreg=2.0e-3, MDm = 0.0006):

    fw_params = np.zeros(data.shape[:-1] + (9,))


    # Prepare mask
    if mask is None:
        mask = np.ones(data.shape[:-1], dtype=bool)
    else:
        if mask.shape != data.shape[:-1]:
            raise ValueError("Mask is not the same shape as data.")
        mask = np.array(mask, dtype=bool, copy=False)


    index = ndindex(mask.shape)
    for v in index:
        if mask[v]:
            params = wls_iter_fw(W, data, md_data, S0, v, min_signal=min_signal,
                        Diso=3e-3, piterations=piterations, mdreg=mdreg, MDm = MDm)
            fw_params[v] = params
               
    return fw_params
    
    
def wls_iter_fw(W, data, md_data, S0 , v, Diso=3e-3, mdreg=2.0e-3,
             min_signal=1.0e-6, piterations=2, MDm = 0.0006):

    MDm1 = MDm

    sig = data[v]
    MD = md_data[v]
    dmatrix = W

    if (MD < mdreg):
        
        SS = S0[v]
        
        W = dmatrix
    
        # Define weights
        S2 = np.diag(sig**2)
    
        # Defining matrix to solve fwDTI wls solution
        WTS2 = np.dot(W.T, S2)
        inv_WT_S2_W = np.linalg.pinv(np.dot(WTS2, W))
        invWTS2W_WTS2 = np.dot(inv_WT_S2_W, WTS2)
    
        # Process voxel if it has significant signal from tissue
        if np.mean(sig) > min_signal and SS > min_signal:
 
            fwsig = np.exp(np.dot(dmatrix,
                                  np.array([Diso, 0, Diso, 0, 0, Diso, 0])))
    
            df = 1  # initialize precision
            flow = 0  # lower f evaluated
            fhig = 1  # higher f evaluated
            ns = 21  # initial number of samples per iteration
            for p in range(piterations):
                df = df * 0.1
                fs = np.linspace(flow, fhig, num=ns)  # sampling f
                fs[ns-1] = 0.98
    
                SFW = np.array([fwsig, ]*ns)  # repeat contributions for all values
                FS, SI = np.meshgrid(fs, sig)
                SA = SI - FS*SS*SFW.T

                SA[SA <= 0] = min_signal
                y = np.log(SA / (1-FS))
                all_new_params = np.dot(invWTS2W_WTS2, y)
                # Select params for lower F2
                SIpred = (1-FS)*np.exp(np.dot(W, all_new_params)) + FS*SS*SFW.T
                F2 = np.sum(np.square(SI - SIpred), axis=0)
                evals, evecs =decompose_tensor(from_lower_triangular(all_new_params.T))
 
                MD2 = dti.mean_diffusivity(evals)
                FA2 = dti.fractional_anisotropy(evals)
                if ( p == 0):
                    MDa = MD2[0]

                if (MD > MDm1):
                    Mind1 = np.argmin(np.abs(MD2 - MDm1))
                    Mind2 = np.argmin(np.abs(FA2 - 3.0*FA2[0]))              
                    Mind1 = np.min([Mind1,Mind2])
                else:
                    MDm2 = 0.00042*MDa/MDm1
                    Mind1 = np.argmin(np.abs(MD2 - MDm2))
                    Mind2 = np.argmin(np.abs(FA2 - 3.0*FA2[0]))              
                    Mind1 = np.min([Mind1,Mind2])                    
                    
                    
                F2S1 =  F2[0] - F2[Mind1] 

                params1 = all_new_params[:, Mind1]                    
                f = fs[Mind1]  # Updated f
                flow = max([f - df,0])  # refining precision
                fhig = min([f + df, 0.98])

            fw_params = np.concatenate((params1,np.array([f]), np.array([F2S1])), axis=0)

        else:
            fw_params = np.zeros(9)
    else:
        fw_params = np.zeros(9)
        fw_params[7] = 1.0
        
    return fw_params


def wls_fit_dti(W, data, mask=None, min_signal=1.0e-6):

    fw_params = np.zeros(data.shape[:-1] + (9,))


    # Prepare mask
    if mask is None:
        mask = np.ones(data.shape[:-1], dtype=bool)
    else:
        if mask.shape != data.shape[:-1]:
            raise ValueError("Mask is not the same shape as data.")
        mask = np.array(mask, dtype=bool, copy=False)


    index = ndindex(mask.shape)
    for v in index:
        if mask[v]:
            params = wls_iter_dti(W, data, v, min_signal=min_signal)
            fw_params[v] = params
                
    return fw_params


def wls_iter_dti(W, data , v, min_signal=1.0e-6):
    
    sig = data[v]
    
    # Define weights
    S2 = np.diag(sig**2)
    SI = sig.copy()
 
    # solve fwDTI wls solution
    WTS2 = np.dot(W.T, S2)
    inv_WT_S2_W = np.linalg.pinv(np.dot(WTS2, W))
    invWTS2W_WTS2 = np.dot(inv_WT_S2_W, WTS2)
    
    SI[SI <= 0] = min_signal
    y = np.log(SI)
    params = np.dot(invWTS2W_WTS2, y)
    SIpred = np.exp(np.dot(W, params))
    
    F2 = np.sum(np.square(SI - SIpred), axis=0)  
    dti_params = np.concatenate((params,np.array([0]), np.array([F2])), axis=0)
    
    return dti_params
    

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


def filter_b_values(fn_data = 'data.nii.gz', 
                fn_bval = 'file.bval', 
                fn_bvec = 'file.bvec',
                out_dir = None,
                bRange = [800,1200]):
    
    # Load the dti data
    print("Filtering DWI data according to b-values:")
    print(f"Accepted are b-values close to Zero (b-value <= 5) and in the range: {bRange}")
    
    bvals, bvalsStr = read_bval_or_bvec(fn_bval)

    selB0 =  (bvals <= 5)
    selBRange =  ((bvals >= bRange[0]) & (bvals <= bRange[1]))
    sel = selB0 | selBRange
    print(f' total number of images        : n={len(bvals)}')
    print(f' images with b-value <= 5      : n={sum(selB0==True)}')
    print(f' images with b-value in range  : n={sum(selBRange==True)}')
    print(f' images with excluded b-values : n={sum(sel==False)}')

    applyFilter = False
    
    if sum(sel==False)>0:
        applyFilter = True
        print('Removing excluded b-values!')

    b5 = (bvals>0) & (bvals<=5)
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
    
        _, bvecs = read_bval_or_bvec(fn_bvec)
        bvecs = bvecs[sel]

        nii = nib.load(fn_data)
        img = nii.get_fdata()
        img = img[:,:,:,sel]

        fn_bval = join(out_dir, basename(fn_bval))
        fn_bvec = join(out_dir, basename(fn_bvec))
        fn_data = join(out_dir, basename(fn_data))
        write_bval_or_bvec(bvals, fn_bval)
        write_bval_or_bvec(bvecs, fn_bvec)
        save_nifti(fn_data, img, nii.affine, nii.header)

        print('New data saved to:')
        print(fn_bval)
        print(fn_bvec)
        print(fn_data)
        
        
    return fn_data, fn_bval, fn_bvec


def free_water_correction(fn_data = 'data.nii.gz', 
                fn_mask = 'brain_mask.nii.gz', 
                fn_bval = 'file.bval', 
                fn_bvec = 'file.bvec',
                out_dir = None,
                smooth=True):
    
    # Define some parameters
    mdreg=2.0e-3
 
    # Load the dti data
    print('Reading data from:')
    print(fn_bval)
    print(fn_bvec)
    print(fn_data)
    nii = nib.load(fn_data)
    niim = nib.load(fn_mask)
    data = nii.get_fdata()
    mask = niim.get_fdata()
    bvals, bvecs = read_bvals_bvecs(fn_bval, fn_bvec)
    print(f'bvals = \n{bvals}\n')

    # Construct the gradient table
    gtab = gradient_table(bvals, bvecs)
    
    W = design_matrix(gtab)
    
    # smooth the data
    if smooth:
        print('Smoothing DWI data')
        fwhm = 1.25
        gauss_std = fwhm / np.sqrt(8 * np.log(2))  # converting fwhm to Gaussian std
        for v in range(data.shape[-1]):
            data[..., v] = gaussian_filter(data[..., v], sigma=gauss_std)
    
    
    # weighted least squares fit, not accounting for free water
    print('Fitting single tensor model, not accounting for free water')
    dti_params = wls_fit_dti(W, data, mask=mask, min_signal=1.0e-6)
    evals, evecs = decompose_tensor(from_lower_triangular(dti_params))
    FA0 = dti.fractional_anisotropy(evals)   
    MD0 = dti.mean_diffusivity(evals)
    save_nifti(join(out_dir, 'wls_dti_FA.nii.gz'), FA0, nii.affine, nii.header)
    save_nifti(join(out_dir, 'wls_dti_MD.nii.gz'), MD0, nii.affine, nii.header)
    
    # two-tensor model fitting
    print('Fitting two-tensor model, for tissue and free water')
    S0 = np.mean(data[..., gtab.b0s_mask], axis=-1)
    pCSF = (MD0 > 0.002)
    mCSF = np.mean(MD0[pCSF])    
    mdreg1 = 0.002*mCSF/0.0025
    mdreg = np.min([mdreg,mdreg1])
    MDm = 0.0006
    
    dti_params1 = wls_fit_tensor_fw(W, data, MD0, S0, Diso=3e-3, mask=mask, 
                            min_signal=1.0e-6, piterations=2, mdreg=mdreg, MDm = MDm)
    evals, evecs = decompose_tensor(from_lower_triangular(dti_params1))
    FA1 = dti.fractional_anisotropy(evals)   
    FW1 = dti_params1[..., 7]
    save_nifti(join(out_dir, 'fwc_wls_dti_FA.nii.gz'), FA1, nii.affine, nii.header)
    save_nifti(join(out_dir, 'wls_dti_FW.nii.gz'), FW1, nii.affine, nii.header)
    
    print('Setting voxels in fwc-FA image to 0.05, if located inside brain mask and fwc-FA equals 0')
    FA1[(FA1==0) & (mask>0)] = 0.05
    save_nifti(join(out_dir, 'fwc_wls_dti_FA_05.nii.gz'), FA1, nii.affine, nii.header)


def create_template(timepoints = [], fnCoreg = [], dirOut = None, nCPU = 2, nThreads = 12, iterations="30x30x8", numRegistrations=3):

    # copy fwc-FA images for all time-points to the folder for template creation
    fnFA = []
    for i,tp in enumerate(timepoints):
        fnFA.append(join(dirOut, basename(tp)+'_fwc_wls_dti_FA_05.nii.gz'))
        copy2(join(tp, 'fwc_wls_dti_FA_05.nii.gz'), fnFA[i])
    
    # construct command for template creation
    cmd = (f"export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={nThreads}; "
           "export ANTS_RANDOM_SEED=1; "
           f"antsMultivariateTemplateConstruction2.sh -d 3 -i {numRegistrations} -f 4x2x1 -s 2x1x0vox -q {iterations} -t SyN -m CC "
           f" -r 1 -z /opt/scripts/FMRIB58_FA_2mm_crop.nii.gz -y 0 -c 2 -j {nCPU} -o {dirOut}/ {' '.join(fnFA)}")
    run_subprocess(cmd, False, 'antsMultivariateTemplateConstruction2.sh')

    # co-register several images for each timepoint to the template
    fnAverage = []
    for iTP, tp in enumerate(timepoints):
        tpB = basename(tp)
        for iFn, fn in enumerate(fnCoreg):
            fnIn = join(tp, fn) 
            fnOut = re.sub(r'\.nii(\.gz)?$','_to_template.nii.gz', fnIn)
            cmd = f"antsApplyTransforms -d 3 --float 1 -i {fnIn}  -o {fnOut} -r {dirOut}/template0.nii.gz -t {dirOut}/{tpB}_fwc_wls_dti_FA_05{iTP}1Warp.nii.gz -t {dirOut}/{tpB}_fwc_wls_dti_FA_05{iTP}0GenericAffine.mat"
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
    save_nifti(join(dirOut, 'FA-for-tbss-long.nii.gz'), img, nii.affine, nii.header, 'float32')
    print('Saved mean image to:', join(dirOut, 'FA-for-tbss-long.nii.gz'))

def coreg_merge_masks(timepoints = [], masks = [], label=None, dirTemplate = None, binarise = False):

    # co-register masks
    if any(masks):
        fnMerge = []
        for iTP, tp in enumerate(timepoints):
            tpB = basename(tp)
            if masks[iTP] is not None:
                fnIn = masks[iTP]
                ext = '.nii.gz' if fnIn.endswith('.nii.gz') else '.nii'
                fnOut = join(tp,label+ext)
                if binarise:
                    nii = nib.load(fnIn)
                    img = nii.get_fdata()
                    save_nifti(fnOut, img>0, nii.affine, nii.header, 'uint8')
                else:
                    copy2(fnIn, fnOut)
                if len(timepoints)>1:
                    fnIn = fnOut
                    fnOut = join(tp, label+'_to_template.nii.gz')
                    cmd = f"antsApplyTransforms -d 3 --float 1 -i {fnIn}  -o {fnOut} -r {dirTemplate}/template0.nii.gz -t {dirTemplate}/{tpB}_fwc_wls_dti_FA_05{iTP}1Warp.nii.gz -t {dirTemplate}/{tpB}_fwc_wls_dti_FA_05{iTP}0GenericAffine.mat  -n NearestNeighbor"
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
    # print('Saved mean image to:', fnOut)
    
    return fnOut


def run_tbss(fnameFAt = None, dirTBSS = None):
    
    #--- Copy the FAt file to TBSS folder
    copy2(fnameFAt, join(dirTBSS, basename(fnameFAt)))

    #--- TBSS on FA
    dirBase = os.getcwd()
    os.chdir(dirTBSS)
    cmd = f'tbss_1_preproc {basename(fnameFAt)}'
    run_subprocess(cmd, True, 'tbss_1_preproc')
    cmd = 'tbss_2_reg -T'
    run_subprocess(cmd, True, 'tbss_2_reg')
    cmd = 'tbss_3_postreg -T'
    run_subprocess(cmd, True, 'tbss_3_postreg')
    cmd = 'tbss_4_prestats 0.2'
    run_subprocess(cmd, True, 'tbss_4_prestats')
    os.chdir(dirBase)

    
def batch_tbss_non_fa(dirTP = None, dirTBSS = None, fnNonFA = []):
    
    # get name of FA used for projection on skeletion
    fnameFAt = glob.glob(join(dirTBSS, 'FA', '*_FA.nii.gz'))
    fnameFAt = re.sub(r'_FA\.nii\.gz','.nii.gz',basename(fnameFAt[0]))
    # print(fnameFAt)

    tpB = basename(dirTP)

    for modality, fn in fnNonFA.items():        
        run_tbss_non_fa(join(dirTP,fn), tpB+'_'+modality, dirTBSS, fnameFAt)


def run_tbss_non_fa(fn = None, modality=None, dirTBSS = None, fnameFAt = None):
    
    # get name of FA used for projection onto skeleton
    if fnameFAt is None:
        fnameFAt = glob.glob(join(dirTBSS, 'FA', '*_FA.nii.gz'))
        fnameFAt = re.sub(r'_FA\.nii\.gz','.nii.gz',basename(fnameFAt[0]))
        # print(fnameFAt)
    
    dirBase = os.getcwd()

    # copy the modality in template space to a dedicated non-FA folder
    dirTBSS_nonFA = join(dirTBSS, modality)
    Path(dirTBSS_nonFA).mkdir(exist_ok=True)
    copy2(fn, join(dirTBSS_nonFA, fnameFAt))

    # run non-FA
    os.chdir(dirTBSS)
    cmd = f"tbss_non_FA {modality}"
    run_subprocess(cmd, True, 'tbss_non_FA')
    os.chdir(dirBase)


def integrate_masks(dirTP = [], dirTBSS = None, skelMask = None, fnROI_MNI = None, analyseHemispheres = False):
    
    tpAll = [basename(tp) for tp in dirTP]
    
    # load the generic skeleton mask (standardised PSMD-mask)
    skelBase = re.sub(r'\.nii(\.gz)?$','',basename(skelMask))
    niiMask = nib.load(skelMask)
    mask = niiMask.get_fdata()

    # count voxels
    voxels = [np.count_nonzero(mask)]
    timept = ['all'] if len(dirTP)>1 else [tpAll[0]]
    region = ['total']


    # find for each timepoint the skeleton voxels being in the generic skeleton-mask and in the skeletonized brain mask.
    # voxels in the skeletonized brain mask with 0<value<1 have to be removed as well, because these values result from interpolation of foreground with background during transformations.
    allMasksAdjusted = []
    for tp in tpAll:
        # load skeleton for brainmask at that timepoint
        nii = nib.load(join(dirTBSS, 'stats', 'all_'+tp+'_bmask_skeletonised.nii.gz'))
        bmask = nii.get_fdata()
        # find intersection of skeleton voxels present in this timepoint's skeletonized brain mask and present in the generic skeleton-mask
        tpMaskAdjusted = mask.copy()
        tpMaskAdjusted[bmask<1] = 0
        allMasksAdjusted.append(tpMaskAdjusted)
        

    # find intersection of adjusted masks across time-points
    if len(dirTP)>1:
        maskIntersection = np.all(np.stack(allMasksAdjusted, -1), -1)
        timeptT = 'all'
    else:
        maskIntersection = tpMaskAdjusted
        timeptT = tpAll[0]

    # save intersection mask (i.e. intersection (across timepoints) of the interesection masks resulting for each timepoint's brain mask with the generic skeleton mask)
    skelSuffix = 'intersection'
    save_nifti(join(dirTBSS, 'stats', skelBase+'_'+skelSuffix+'.nii.gz'), maskIntersection, niiMask.affine, niiMask.header, dtype='uint8', scale=False)

    # count voxels
    voxelsIntersection = np.count_nonzero(maskIntersection)
    voxels.append(voxelsIntersection)
    timept.append(timeptT)
    region.append(skelSuffix)
    
        
    if len(dirTP)>1:
        for iTP, tp in enumerate(tpAll):            
            voxels.append(np.count_nonzero(allMasksAdjusted[iTP]) - voxelsIntersection)
            timept.append(tp)
            region.append('set_difference')
    
    # load exclusion mask
    fnEmask = join(dirTBSS, 'stats', 'all_E-MASK_skeletonised.nii.gz')
    if exists(fnEmask):
        niiEmask = nib.load(fnEmask)
        imgEmask = niiEmask.get_fdata()
        imgEmask = (imgEmask>0.05) *2 #--- be conservative, excluding also (most) interpolated voxels; only needed for QC: set voxels of exclusion mask equal 2, for later combination with intersection mask
        imgEmask[maskIntersection==0] = 0 #--- only needed for QC: remove voxels from exclusion mask, which are anyways outside the intersection mask
        maskIntersection[imgEmask>0] = 0
        imgEmask[maskIntersection>0] = 1 #--- only needed for QC: combine intersection mask (label=1) and exclusion mask (label=2)

        # save skeleton with excluded area inserted as label-2 (only needed as QC image later)
        skelSuffixL2 = skelSuffix + '_Emask-as-label2'
        pnameSkelIntersExcLabeled = join(dirTBSS, 'stats', skelBase+'_'+skelSuffixL2+'.nii.gz')
        save_nifti(pnameSkelIntersExcLabeled, imgEmask, niiMask.affine, niiMask.header, dtype='uint8', scale=False)

        # save skeleton without the excluded area
        skelSuffix = skelSuffix + '_Emask'
        pnameSkelIntersExc = join(dirTBSS, 'stats', skelBase+'_'+skelSuffix+'.nii.gz')
        save_nifti(pnameSkelIntersExc, maskIntersection, niiMask.affine, niiMask.header, dtype='uint8', scale=False)

        voxels.append(np.count_nonzero(maskIntersection))
        timept.append(timeptT)
        region.append(skelSuffix)

    # load ROI masks provided for patient time-points and merged
    fnROI = sorted(glob.glob(join(dirTBSS, 'stats', 'all_ROI-*_skeletonised.nii.gz')))
    for iFn,fn in enumerate(fnROI):
        roi = re.sub(r'.*all_ROI-([0-9]*)_skeletonised.nii.gz','\\1',fn)
        niiROI = nib.load(fn)
        imgROI = niiROI.get_fdata()

        imgROI[maskIntersection==0] = 0
        imgROI[imgROI>0.05] = 1
        imgROI[imgROI<1] = 0

        #- save
        skelSuffixT = skelSuffix + f'_Rmask-{roi}'
        pnameROI = join(dirTBSS, 'stats', skelBase+'_'+skelSuffixT+'.nii.gz')
        save_nifti(pnameROI, imgROI, niiMask.affine, niiMask.header, dtype='uint8')

        #- count voxels
        voxels.append(np.count_nonzero(imgROI))
        timept.append(timeptT)
        region.append(skelSuffixT)

        if iFn==0: imgROImerged = np.zeros(imgROI.shape, 'uint8')
        imgROImerged[imgROI>0] = int(roi)
        if iFn==len(fnROI)-1 and np.count_nonzero(imgROImerged)>0: 
            pnameROImerged = join(dirTBSS, 'stats', skelBase+'_'+skelSuffix+'_Rmask.nii.gz')
            save_nifti(pnameROImerged, imgROImerged, niiROI.affine, niiROI.header, dtype='uint8')
    # create an additional, complementary ROI for the background
    if len(fnROI)>0:
        imgROI = maskIntersection.copy()
        imgROI[imgROImerged>0] = 0
        #- save
        skelSuffixT = skelSuffix + f'_Rmask-00'
        pnameROI = join(dirTBSS, 'stats', skelBase+'_'+skelSuffixT+'.nii.gz')
        save_nifti(pnameROI, imgROI, niiMask.affine, niiMask.header, dtype='uint8')
        #- count voxels
        voxels.insert(len(voxels)-len(fnROI), np.count_nonzero(imgROI))
        timept.insert(len(timept)-len(fnROI), timeptT)
        region.insert(len(region)-len(fnROI), skelSuffixT)

    


    # load ROI mask in FMRIB FA (MNI) space
    if fnROI_MNI is not None:
        niiROI_MNI = nib.load(fnROI_MNI)
        imgROI_MNI = niiROI_MNI.get_fdata()
        #- extract unique ROI labels (multiple ROI can be provided in the same file)
        uROI = np.unique(imgROI_MNI.astype('uint8'))
        # uROI = uROI[uROI>0] #--- commented out, to include also the background (on the skeleton) as a ROI
        for iRoi,roi in enumerate(uROI):
            imgROI_MNI_roi = maskIntersection.copy()
            imgROI_MNI_roi[imgROI_MNI!=roi] = 0
            
            #- save
            # skelSuffixT = skelSuffix + f'_mni-roi-{roi:02d}'
            skelSuffixT = skelSuffix + f'_RmaskMNI-{roi:02d}'
            pnameROI = join(dirTBSS, 'stats', skelBase+'_'+skelSuffixT+'.nii.gz')
            save_nifti(pnameROI, imgROI_MNI_roi, niiMask.affine, niiMask.header, dtype='uint8')

            #- count voxels
            voxels.append(np.count_nonzero(imgROI_MNI_roi))
            timept.append(timeptT)
            region.append(skelSuffixT)            

            #- merge ROIs after intersecting them with the final skeleton; hence, this is differnt from the input "fnROI_MNI" file
            if iRoi==0: imgROImerged = np.zeros(imgROI_MNI.shape, 'uint8')
            if roi>0:
                imgROImerged[imgROI_MNI_roi>0] = roi
            #- save merged ROI only if at least one ROI is present, after intersecting with the final skeleton
            if iRoi==len(uROI)-1 and np.count_nonzero(imgROImerged)>0: 
                pnameROImerged = join(dirTBSS, 'stats', skelBase+'_'+skelSuffix+'_RmaskMNI.nii.gz')
                save_nifti(pnameROImerged, imgROImerged, niiROI_MNI.affine, niiROI_MNI.header, dtype='uint8')

    
    # analyse hemispheres separately
    if analyseHemispheres:
        sh = maskIntersection.shape
        for hemi,range in zip(['LH', 'RH'],[[0,sh[0]//2],[sh[0]//2,sh[0]+1]]):
            maskHemi = maskIntersection.copy()
            maskHemi[range[0]:range[1],:,:] = 0

            #- save
            skelSuffixT = skelSuffix + f'_{hemi}'
            pnameROI = join(dirTBSS, 'stats', skelBase+'_'+skelSuffixT+'.nii.gz')
            save_nifti(pnameROI, maskHemi, niiMask.affine, niiMask.header, dtype='uint8')

            #- count voxels
            voxels.append(np.count_nonzero(maskHemi))
            timept.append(timeptT)
            region.append(skelSuffixT)
        

    # construct and return dataframe with voxel counts per timpoint, skeleton and region
    skeleton = [basename(skelMask)] * len(voxels)
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
    
    # find the adjusted skeleton mask
    skelBase = re.sub(r'\.nii(\.gz)?$','', basename(skelMask))
    skelMaskInters = join(dirTBSS, 'stats', skelBase+'_intersection_Emask.nii.gz')
    if not exists(skelMaskInters):
        skelMaskInters = join(dirTBSS, 'stats', skelBase+'_intersection.nii.gz')

    # find ROI masks
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

        for modality, _ in fnNonFA.items():

            print('\nExtracting histogram parameters for:')
            print(f' region    : {roiSuffix}')
            print(f' modality: {modality}')            
            
            # extract histogram statistics
            nii = nib.load(join(dirTBSS, 'stats', 'all_'+tpB+'_'+modality+'_skeletonised.nii.gz'))
            img = nii.get_fdata()
            skel = img[roi>0]
            print( ' voxels  :',len(skel))
            if len(skel)>0:
                prcts = np.percentile(skel,[5,95])
                mean = np.mean(skel)
            else:
                prcts = [np.nan]*2
                mean = np.nan
            pw = prcts[1] - prcts[0]

            # create dataframe
            mT = re.sub(r'^nc','',modality)
            dd.append(pd.DataFrame(
                {'timepoint': [tpB]*2,
                'skeleton': [basename(skelMask)]*2,
                'region': [roiSuffix]*2,
                'voxels': [len(skel)]*2,
                'metric': ['PWS'+mT, 'MS'+mT],
                'value': [pw, mean]}
            ))

    df = pd.concat(dd)
    # remove unwanted metric 'PWSFW'
    df = df[df['metric']!='PWSFW']

    return df

def prepare_qc(dirQC, skelMask, dirTBSS, dirTemplate, dirTP, fnCSV, args):
        
    from create_qc_image import create_qc_image
    from create_html_with_png import create_html_with_png
    
    # find the adjusted skeleton mask
    skelBase = re.sub(r'\.nii(\.gz)?$','', basename(skelMask))
    skelMask = join(dirTBSS, 'stats', skelBase+'_intersection_Emask-as-label2.nii.gz')
    emaskExists = 1
    if not exists(skelMask):
        skelMask = join(dirTBSS, 'stats', skelBase+'_intersection.nii.gz')
        emaskExists = 0
    
    # find additional ROI
    fnROI = [skelMask]
    fnROI = fnROI + glob.glob(join(dirTBSS, 'stats', '*_Rmask.nii.gz'))
    fnROI = fnROI + glob.glob(join(dirTBSS, 'stats', '*_RmaskMNI.nii.gz'))

    if len(dirTP)>1:
        fnameFAt = "FA-for-tbss-long"
    else:
        fnameFAt = "fwc_wls_dti_FA_05"
    dirBase = os.getcwd()

    for fn in fnROI:
        os.chdir(join(dirTBSS,'stats'))
        cmd = f'tbss_deproject {basename(fn)} 2 -n'
        run_subprocess(cmd, True, 'tbss_deproject')
        os.chdir(dirBase)
        fnBase = basename(fn)
        fnDeprojectedTemplateSpace = join(dirTBSS, 'FA', fnameFAt+'_FA_'+fnBase)
            
        for iTP in range(len(dirTP)):
            tpB = basename(dirTP[iTP])
            if len(dirTP)>1:
                fnOut = join(dirQC, tpB+'_'+fnBase)
                cmd = f"antsApplyTransforms -d 3 --float 1 -i {fnDeprojectedTemplateSpace}  -o {fnOut} -r {dirTP[iTP]}/fwc_wls_dti_FA.nii.gz -t [{dirTemplate}/{tpB}_fwc_wls_dti_FA_05{iTP}0GenericAffine.mat,1] -t {dirTemplate}/{tpB}_fwc_wls_dti_FA_05{iTP}1InverseWarp.nii.gz -n NearestNeighbor"
                run_subprocess(cmd, True, 'antsApplyTransforms')
            else:
                copy2(fnDeprojectedTemplateSpace, join(dirQC, tpB+'_'+fnBase))    
    
    # Create PNGs for HTML, in space of input and per timepoint
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

    fnHTML = dirQC + '.html'
    print(f'\nSaving QC to:\n ', fnHTML)
    
    create_html_with_png(fnHTML, fnPNG, captions, None, fnCSV, args)

    # delete psmd2_QC folder, if only HTML was requested
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


def run_subprocess(cmd, displayStdout, label):
    verbose = True
    if verbose: print(f"Calling {label} command with:")
    if verbose: print(cmd, '\n')
    output = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    # display output
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
    
def assertMinimumCPU(s):
    if int(s) < 2:
        raise argparse.ArgumentError(f'Number of cpu cores to use for ANTs registration has to be at least 2; You selected only {s}')
    else:
        return int(s)

class CustomArgumentParser(argparse.ArgumentParser):
    # This subclass ensures that single dash options have to be one character long (after the dash) and separated from their arguments by a space
    def parse_args(self, args=None, namespace=None):
        for arg_string in args:
            if arg_string.startswith('-') and not arg_string.startswith('--'):
                if len(arg_string) > 2 and not arg_string[2].isspace():
                    raise ValueError(f'Single dash options have to be one character long (after the dash) and separated from their arguments by a space. Argument "{arg_string}" violates this requirement.')
        args = super().parse_args(args, namespace)
        return args
    
stepsImplemented = ['fwc','template','tbss','tbss_non_fa','extract','qc']
argparseDescription = "The 'Pipeline for Skeletonized Metrics from Diffusion MRI' (PSMD) processes multi-directional diffusion MRI data to fully automatically extract clinically and technically validated white matter diffusion metrics. Key steps include diffusion tensor fitting (with and without free water imaging), skeletonization based on the free water-corrected FA (fwc-FA) via FSL's TBSS, and enhanced CSF partial volume masking. The final metrics MSMD, PWSMD, and MSFW are computed over the skeleton. For longitudinal data, a within-subject template is created using ANTs."

def iniParser():
    parser = CustomArgumentParser(description=argparseDescription, epilog='Notice: By using PSMD, you agree to the software license terms described at "http://psmd-marker.com"')
    group0 = parser.add_argument_group('input/output data specification')
    group0.add_argument("--dwi", required=True, metavar='NIfTI', type=isNIfTI, nargs="+", action='extend', help="input path(s) to 4D DWI image(s) in NIfTI format. Number of arguments should correspond to number of time-points.")
    group0.add_argument("--bval", metavar='text-file', type=str, nargs="+", action='extend', help="input path(s) to text file(s) with b-values in FSL format, corresponding to DWI image(s). If parent folders are identical to those of corresponding DWI images, providing basename(s) is sufficient. If all basenames are identical, repetition is not needed. If argument not provided, path(s) will be constructed from DWI image path(s), substituting extension with '.bval'")
    group0.add_argument("--bvec", metavar='text-file', type=str, nargs="+", action='extend', help="input path(s) to text file(s) with b-vectors in FSL format, corresponding to DWI image(s). If parent folders are identical to those of corresponding DWI images, providing basename(s) is sufficient.  If all basenames are identical, repetition is not needed. If argument not provided, path(s) will be constructed from DWI image path(s), substituting extension with '.bvec'")
    group0.add_argument("--bmask", metavar='NIfTI', type=str, nargs="+", action='extend', help="input path(s) to DWI brain mask(s) in NIfTI format, corresponding to DWI image(s). If parent folders are identical to those of corresponding DWI images, providing basename(s) is sufficient. If all basenames are identical, repetition is not needed. If argument not provided, path(s) will be constructed from DWI image path(s), substituting extension with '_brainmask.nii.gz'")
    group0.add_argument("--tp", metavar='label', type=str, nargs="+", action='extend', help="label(s) for all time-points. Number of arguments should correspond to number of DWI image(s). If argument not provided, time-points are labeled consecutively as TP01, TP02, and so on.")
    group0.add_argument("--id", metavar='label', type=str, help="optional patient/subject identifier. If provided, an additional column with this identifier will be added to the results table 'psmd2_results.csv', meant to facilitate aggregation of results tables for multiple patients/subjects.")
    group0.add_argument("-o", "--dirOutput", type=str, help="path to output folder. If not provided, the parent folder of the first DWI image will be used. The results table ('psmd2_results.csv') and a subfolder and HTML for quality checking ('psmd2_QC' and 'psmd2_QC.html') will be saved here. Furthermore, intermediate/temporary files will be created here inside a subfolder called 'psmd2_temp'.")
    group1 = parser.add_argument_group('additional masking')
    group1.add_argument("--Emask", metavar='NIfTI', type=str, default = [], nargs="+", action='extend', help="input path(s) to custom exclusion mask(s) in DWI image space, used for 'exclusive' masking. One per timepoint can be provided, which will be merged in template space. Time-points will be matched by position of provided paths. Skip time-points by entering NA instead of a path. The masked area (e.g. lesion) will be excluded from analysis. Provided masks will be binarised, i.e. all non-zero values will be set to 1.")
    group1.add_argument("--Rmask", metavar='NIfTI', type=str, default = [], nargs="+", action='extend', help="input path(s) to custom ROI mask(s) in DWI image space. One per timepoint can be provided, which will be merged in template space. Timepoint matching and/or skipping works as explained for option 'Emask'. Each mask can contain more than one integer label corresponding to differnt ROI, which will be analysed separately. However, masks will be merged in template space and if labels in masks from different time-points disagree, the respectively highest interger label will overwrite the other labels.")
    group1.add_argument("--RmaskMNI", metavar='NIfTI', type=isNIfTI, help="input path to a single custom ROI mask in MNI space. Can contain integer labels for multiple ROI, which will be analysed separately.")
    group1.add_argument("--hemispheres", action='store_true', help="calculate skeleton metrics also separately for left and right hemispheres. Please note, however, that this does not affect ROI masks, which will not be split between hemispheres.")
    group2 = parser.add_argument_group('advanced options')
    group2.add_argument("--skeletonMask", metavar='NIfTI', type=isNIfTI, default="/opt/scripts/psmd2-skeletonmask-v1.nii.gz", help="input path to an alternative skeleton mask. Defaults to the mask validated with PSMD2 ('psmd2-skeletonmask-v1') and designed to exclude regions with frequent CSF partial volume effects.")
    group2.add_argument("--bRange", metavar='Integer', type=int, default = [800, 1200], nargs=2, help="range of b-values to consider for diffusion tensor fitting. Defaults to range [800,1200].")
    group2.add_argument("--smooth", action='store_true', help=argparse.SUPPRESS) #--- "apply Gaussian filter (fwhm = 1.25) to DWI data"
    group2.add_argument("--dontAdjustBmaskForFW", dest='adjustBmaskForFW', action='store_false', help=argparse.SUPPRESS) #--- "don't correct the brain mask for free-water. By default, the brain mask is set to zero, where free water equals 1 (and hence fwc-FA equals 0)."
    group2.add_argument("--para", metavar='max-ANTs-jobs', type=assertMinimumCPU, default=2, help=argparse.SUPPRESS) #--- "Limit number of parallel jobs used during ANTs template creation. Requires minimum and defaults to 2. Maximally, ANTs can use as many jobs as time-points."
    group2.add_argument("--threads", metavar='ITK-threads', type=int, default=12, help="number of threads used by ITK during ANTs registrations (i.e. environmental variable ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS). Defaults to 12 threads.")
    group2.add_argument("--iterations", type=str, default='30x30x8', help=argparse.SUPPRESS) #--- "Iterations at each resolution level of the pairwise ANTs registrations during template creation. Must be three levels and specified in the format: 'L1xL2xL3'. Defaults to '30x30x8'."
    group2.add_argument("--numRegistrations", type=int, default=3, help=argparse.SUPPRESS) #--- "Iterations of the template construction. Each iteration comprises averaging of images and pairwise registrations of each timepoint to the template. Defaults to 3 iterations."
    group2.add_argument("--qc", type=int, choices=[0,1,2], default=1, help="create (with argument 1; the default) a HTML file (psmd2_QC.html) for quality checking and create (with argument 2) additionally a subfolder 'psmd2_QC' with a series of NIfTI images showing skeleton and masks in native space, or (with argument 0) skip creation of both.")
    group2.add_argument("--debug", action='store_true', help="don't delete temporary folder 'psmd2_temp', containing intermediate files created during processing.")
    group2.add_argument("--steps", choices = stepsImplemented, nargs="+", action='extend', help=argparse.SUPPRESS) #--- "choose step(s) to be conducted. By default all steps will be conducted. If the output for preceeding steps is missing, an error will be raised. If different masks are provided, step 'tbss_non_fa' and following have to be repeated."
    group2.add_argument("--reprocess", metavar='csv-file',  type=isCSV, nargs="?", const='overwrite', help='allow reprocessing and overwriting of previously created output. You can, however, keep a previously created results files (default name: psmd2_results.csv) by specifying here an alternative name (provide only the base name, output folder can not be changed here.') #--- "allow reprocessing of previously conducted steps. Warning: this will delete the previous results for all processing steps or, if '--steps' is used, for the selected (and all following/depending) steps. Deleting the final output table 'psmd2_results.csv' can however be avoided, by providing here an alternative CSV filename (provide basename only; will be saved into the output folder, see '--dirOutput')"
    return parser


###########################################################################
# Pipeline

def pipeline_psmd2():

    start_script = time.time()
    parser = iniParser()
    if len(sys.argv)<2:
        parser.print_usage()
        print('\nRun "psmd2.py -h" for detailed help\n'
              'Notice: By using PSMD, you agree to the software license terms described at "http://psmd-marker.com"\n')
        parser.exit()
    else:
        args = parser.parse_args(sys.argv[1::])
    
    args.function_call = " ".join([basename(sys.argv[0])]+sys.argv[1::])
    print("Running: " + args.function_call,'\n')

    # Check DWI files (already done by argparse)
    for i, fn in enumerate(args.dwi):
        if not exists(fn):
            raise ValueError(f"The {i+1}. of the files provided with option '--dwi' does not exist")
        
    # Check bval, bvec and bmask files
    for attr, ext in zip(['bval','bvec','bmask'], ['.bval','.bvec','_brainmask.nii.gz']):
        flist = getattr(args,attr)
        if flist is None:
            flist = [re.sub(r'\.nii(\.gz)?$', ext, fn) for fn in args.dwi]
        elif len(flist) != len(args.dwi):
            if len(flist) == 1:
                flist = flist * len(args.dwi)
            else:
                raise ValueError("Number of files provided with option '--bval' has to be zero or correspond to number of DWI files. Please refer to '--help'")
        for i, fn in enumerate(flist):
            if not exists(fn):
                fn = join(dirname(args.dwi[i]), fn)
                if exists(fn):
                    flist[i] = fn
                else:
                    raise ValueError(f"The {i+1}. of the expected '{attr}' files does not exist")
        setattr(args,attr,flist)
    
    # Check timepoint labels
    if args.tp is None:
        args.tp = ['TP{:02d}'.format(i+1) for i in range(len(args.dwi))] #-- folders for all time-points
    assert len(args.tp) == len(args.dwi), f'If timepoint labels are provided, their number has to correspond to the number of provided DWI files! You passed {len(args.tp)} labels for {len(args.dwi)} DWI files.'    

    # Check exclusion and ROI masks 
    assert len(args.Emask) <= len(args.dwi), f'Number of provided exclusion masks (n={len(args.Emask)}) exceeds number of time-points (n={len(args.dwi)})! Allowed is max. one mask per timepoint!'       
    assert len(args.Rmask) <= len(args.dwi), f'Number of provided ROI masks in DWI space (n={len(args.Rmask)}) exceeds number of time-points (n={len(args.dwi)})! Allowed is max. one mask per timepoint!'       
    for i,_ in enumerate(args.dwi):
            # Check exclusion masks
            if len(args.Emask)>i :
                if args.Emask[i]!='NA':
                    if isNIfTI(args.Emask[i], abort=False) is None:
                        args.Emask[i] = isNIfTI(join(dirname(args.dwi[i]), args.Emask[i]))
                else:
                    args.Emask[i] = None
            else:
                args.Emask.append(None)
            # Check ROI mask
            if len(args.Rmask)>i:
                if args.Rmask[i]!='NA':
                    if isNIfTI(args.Rmask[i], abort=False) is None:
                        args.Rmask[i] = isNIfTI(join(dirname(args.dwi[i]), args.Rmask[i]))
                else:
                    args.Rmask[i] = None
            else:
                args.Rmask.append(None)

    # Display files per timepoint
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

    # Display MNI ROI mask
    if args.RmaskMNI is not None:
        print(f'\nAn additional ROI mask in MNI space (RmaskMNI) was provided:\n {args.RmaskMNI}')

    # Display whether hemisphere masking was selected
    if args.hemispheres is not None:
        print(f'\nHemispheric ROI analysis will be done as well')

    # Display skeleton mask
    if args.skeletonMask == "/opt/scripts/psmd2-skeletonmask-v1.nii.gz":
        print(f'\nUsing the default skeleton mask:\n {args.skeletonMask}')
    else:
        print(f'\nUsing a non-default skeleton mask provided as input:\n {args.skeletonMask}')

    # Output folder
    if args.dirOutput is None:
        args.dirOutput = os.path.dirname(args.dwi[0])
    
    # Check requested processing steps
    if args.qc==0:
        assert args.steps is None or 'qc' not in args.steps, "You asked to do '--step qc' and to skip it '--qc 0' at the same time! Your choice is contradictory!"
        stepsImplemented.remove('qc')
    if args.steps is None:
        args.steps = stepsImplemented
    else:
        stepsIdx = [i for i,x in enumerate(stepsImplemented) if x in args.steps]
        args.steps = [stepsImplemented[i] for i in stepsIdx]
        print(f'\nOn request, only the following processing steps will be conducted:')
        for i,step in enumerate(args.steps): print(f' {i+1}. {step}')
        if len(stepsIdx)>1 and any(np.diff(stepsIdx)>1):
            print(' '); raise ValueError(f"Requested processing steps have to be contiguous. This is not the case.\nThe available steps in order are: {stepsImplemented}")       
        if 'extract' not in args.steps and not args.debug:
            print("NOTE: Given that the final 'extract' step is not selected, we assume that you want to keep intermediate/temporary output and switch on the option '--debug' for you!")
            args.debug = True
    
    # Create file paths used for various processing steps and output files
    dirTemp = join(args.dirOutput, 'psmd2_temp')
    dirTP = [join(dirTemp, tp) for tp in args.tp] #-- folders for all time-points
    dirTemplate = join(dirTemp,'template')
    dirTemplateInter = join(dirTemp, 'intermediateTemplates')
    dirTBSS = join(dirTemp,'TBSS')
    fnNonTBSS = glob.glob(join(dirTBSS,'*')) + glob.glob(join(dirTBSS, 'stats','*'))
    fnNonTBSS = [x for x in fnNonTBSS if not re.match('FA$|origdata$|stats$|all_FA|mean_FA|thresh',basename(x))]
    if args.reprocess is None or args.reprocess == 'overwrite':
        fnCSV = join(args.dirOutput, 'psmd2_results.csv')
    else:
        fnCSV = join(args.dirOutput, args.reprocess)
    fnSkelRegions = glob.glob(join(dirTBSS, 'stats','*_intersection*'))
    dirQC = join(args.dirOutput, 'psmd2_QC')
    fnHTML = join(args.dirOutput, 'psmd2_QC.html')
    
    # Check if output exists already
    if os.path.exists(dirTemp) or os.path.exists(fnCSV) or os.path.exists(dirQC) or os.path.exists(fnHTML):

        # if all steps (with or without QC step) are requested, then simply delete everything
        if set(args.steps+['qc']) == set(stepsImplemented+['qc']):
            print(f'\nChecking existence of output')
            assert args.reprocess is not None, "Output exists already.\n Tip: Use option '--reprocess' if you want to reprocess and overwrite"
            if os.path.exists(dirTemp): print(f'Deleting: {dirTemp}'); rmtree(dirTemp)
            if os.path.exists(fnCSV): print(f'Deleting: {fnCSV}'); os.remove(fnCSV)
            if os.path.exists(dirQC): print(f'Deleting: {dirQC}'); rmtree(dirQC)
            if os.path.exists(fnHTML): print(f'Deleting: {fnHTML}'); os.remove(fnHTML)

        # else, check output for each step separately
        else:
            print(f'\nChecking existence of output per processing step:')
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
                for fnT in fn: assert exists(fnT), f"Output from skipped processing steps '{k}' is missing: {fnT}"
            if args.reprocess is None:
                for k in shouldNotExist:
                    fn = stepsOutput[k]
                    if not isinstance(fn, list): fn = [fn]
                    for fnT in fn: assert not exists(fnT), f"Output for requested step '{k}' exists already: {fnT}\n Tip: Use option '--reprocess' if you want to reprocess and overwrite"
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


    # Create temp folder
    if not os.path.exists(dirTemp):
        print(f"\nCreating temporary folder for processing:\n"
              f"  {dirTemp}")
        Path(dirTemp).mkdir(parents=True, exist_ok=True)


    #----------------------------------
    # Conduct the main processing steps

    # Free water corrrection
    startTime=None
    if 'fwc' in args.steps:

        print(f"\nTime point(s) will be copied and processed in following folder(s):")
        for i,tp in enumerate(dirTP):
            print(f' {tp}')


        for i in range(len(dirTP)):
            startTime = section_header(f'DTI-fit and free-water correction for {i+1}. timepoint in: {dirTP[i]}', startTime)

            # Create output directory
            Path(dirTP[i]).mkdir(parents=True, exist_ok=True)

            # Filter DWI data according to b-values
            dwi, bval, bvec = filter_b_values(fn_data = args.dwi[i], 
                                              fn_bval = args.bval[i], 
                                              fn_bvec = args.bvec[i],
                                              out_dir = dirTP[i],
                                              bRange = [min(args.bRange), max(args.bRange)])
            
            # Run FW correction
            print('')
            free_water_correction(fn_data = dwi, 
                        fn_mask = args.bmask[i], 
                        fn_bval = bval, 
                        fn_bvec = bvec,
                        out_dir = dirTP[i],
                        smooth = args.smooth)
            
            # Copy and, by default, set brain mask equal 0, where Free-Water equals 1
            if args.adjustBmaskForFW:
                # if fwc-FA should be used for skeleton projection, set voxels in brain-mask to zero, where FW==1 (corresponding to "fwc-FA"==0)
                print('Setting brain mask to zero, where free-water equals one.')
                niiBmask = nib.load(args.bmask[i])
                imgBmask = niiBmask.get_fdata()
                niiFAt = nib.load(join(dirTP[i], 'fwc_wls_dti_FA.nii.gz'))
                imgFAt = niiFAt.get_fdata()
                imgBmask[imgFAt==0] = 0
                save_nifti(join(dirTP[i],'brain_mask.nii.gz'), imgBmask, niiBmask.affine, niiBmask.header, dtype='uint8')
            else:
                copy2(args.bmask[i], join(dirTP[i],'brain_mask.nii.gz'))

    
    # Decide whether to use the free-water-corrected or the uncorrected FA image for skeleton projection
    # >>> Opion deleted <<< Use always the free water-corrected FA: 'fwc_wls_dti_FA_05.nii.gz'
    
    
    # Define modalities to be coregistered
    fnCoreg = [
            'fwc_wls_dti_FA_05.nii.gz',
            'wls_dti_FW.nii.gz',
            'wls_dti_MD.nii.gz',
            'brain_mask.nii.gz'
        ]

    # Run template construction
    if 'template' in args.steps  and  len(dirTP)>1:
        Path(dirTemplate).mkdir(parents=True, exist_ok=True)
        startTime = section_header(f'Template construction and co-registration of modalities for all time-points', startTime)
        create_template(timepoints = dirTP, fnCoreg = fnCoreg, dirOut = dirTemplate, nCPU = args.para, nThreads = args.threads, iterations=args.iterations, numRegistrations=args.numRegistrations)

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

    # Define non-FA modalities
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
        
        startTime = section_header(f'Non-FA TBSS for additional mask images', startTime)

        # Transform and merge additional masks with ROI or exclusions/lesions (if None, then None is returned)
        Emask = coreg_merge_masks(timepoints = dirTP, masks = args.Emask, label='mask_exclusive', dirTemplate = dirTemplate, binarise = True)
        Rmask = coreg_merge_masks(timepoints = dirTP, masks = args.Rmask, label='mask_roi', dirTemplate = dirTemplate, binarise=False)
        # non-FA TBSS for masks
        flagAdditionalMasks = False
        if Emask is not None:
            run_tbss_non_fa(Emask, 'E-MASK', dirTBSS)
            flagAdditionalMasks = True
        if Rmask is not None: 
            # check if there are multiple labels
            niiROI = nib.load(Rmask)
            imgROI = niiROI.get_fdata()
            uROI = np.unique(imgROI)
            uROI = uROI[uROI>0].astype('uint8')
            # run non-FA TBSS for each ROI label separately, because it doesn't use nearset-neighbor for registration into MNI
            for roi in uROI:
                imgT = imgROI.copy()
                imgT[imgROI!=roi] = 0
                if roi==0:
                    imgT[imgROI!=roi] = 1
                fnOut = re.sub(r'\.nii(\.gz)?$','-{:02d}.nii.gz'.format(roi), Rmask)
                save_nifti(fnOut, imgT>0, niiROI.affine, niiROI.header, 'uint8')
                run_tbss_non_fa(fnOut, 'ROI-{:02d}'.format(roi), dirTBSS)
            flagAdditionalMasks = True
        if not flagAdditionalMasks:
            print('No additional masks were provided!')
        

    # Extract statistics
    if 'extract' in args.steps:

        dfL = []

        # Find common skeleton voxels being in skeleton mask and present at all time-points
        if len(dirTP)>1:
            startTime = section_header(f'Finding skeleton voxels present in brain masks of all timepoints and in the skeleton mask and ROIs', startTime)
        else:
            startTime = section_header(f'Finding skeleton voxels present in the brain mask and in the skeleton mask and ROIs', startTime)
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

        startTime = section_header(f'Creating QC images (deprojecting skeleton mask and transforming into native space)', startTime)
        dirQC = join(args.dirOutput, 'psmd2_QC')
        Path(dirQC).mkdir(parents=True, exist_ok=True)
        prepare_qc(dirQC, args.skeletonMask, dirTBSS, dirTemplate, dirTP, fnCSV, args)


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
    pipeline_psmd2()