#!/usr/bin/python
# -*- coding: utf-8 -*-

# Free-water/single-tensor weighted least-squares fitting, vendored from the
# MarkVCID Free Water Kit (davisidealab/MarkVCID_FW, fw_mrn.py):
#   https://github.com/davisidealab/MarkVCID_FW/blob/main/fw_mrn.py
# Copied here (only cosmetic cleanup: removed a commented-out line and one
# unused variable assignment) with permission for redistribution as part of 
# DELTA-SVD.

import numpy as np
import dipy.reconst.dti as dti
from dipy.reconst.dti import decompose_tensor, from_lower_triangular
from dipy.core.ndindex import ndindex


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
