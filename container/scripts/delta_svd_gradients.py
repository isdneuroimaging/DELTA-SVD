#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reading and checking of bval/bvec files and the b-value selection.

Imports numpy and nibabel, so it must be imported after delta-svd.py has
pinned the threads and the BLAS kernel.
"""

from os.path import join, basename

from delta_svd_constants import (BRANGE_DEFAULT, BRANGE_TOL, SHELL_TOL, MIN_DIRECTIONS,
                                 RECOMMENDED_DIRECTIONS)

import nibabel as nib
from nibabel.volumeutils import apply_read_scaling
import numpy as np

from delta_svd_common import DeltaSvdError, save_nifti


###########################################################################
# Functions for reading/writing bval/bvec files

#--- How many lines of a malformed file to list before summarising the rest.
MAX_LINES_REPORTED = 10


def first_non_numeric(rows):
    """Where the first entry that is not a number sits, as (line, position,
    value) with both indices 1-based. None when every entry is a number.

    For error messages only: numpy names the offending value but not where in
    the file it is, which is the part needed to go and fix it."""
    for iRow, row in enumerate(rows):
        for iVal, val in enumerate(row):
            try:
                float(val)
            except ValueError:
                return iRow + 1, iVal + 1, val
    return None


def read_bval_or_bvec(fname):

    with open(fname, "r") as file:
        ll = file.readlines()

    for i,l in enumerate(ll):
        ll[i] = l.split()

    #--- Lines of unequal length make np.array() raise on its own, with a message
    #    about inhomogeneous shapes that names neither the file nor the line.
    lengths = {len(row) for row in ll}
    if len(lengths) > 1:
        shown = '\n'.join(f'   line {i+1}: {len(row)} value(s)'
                          for i, row in enumerate(ll[:MAX_LINES_REPORTED]))
        more = (f'\n   ... and {len(ll)-MAX_LINES_REPORTED} more line(s)'
                if len(ll) > MAX_LINES_REPORTED else '')
        blanks = ('\n A blank line holds no values and counts here, so remove any.'
                  if 0 in lengths else '')
        raise DeltaSvdError(f"Every line of a bval/bvec file has to hold the same number of values "
                            f"-- one per volume -- but these do not:\n{shown}{more}{blanks}\n"
                            f" {fname}")

    rows = ll                      # kept for reporting: 'll' is reshaped below
    ll = np.array(ll)

    if ll.shape[0]==1:
        ll = ll[0]
    elif ll.shape[0]==3:
        ll = ll.T
    else:
        raise DeltaSvdError(f"Input file has to contain either one row (for bval files) or three "
                            f"rows (for bvec files), but this one has {ll.shape[0]}:\n {fname}")

    arrStr   = ll
    #--- Attempted rather than pre-checked, so that what counts as a number stays
    #    exactly what numpy accepts; the scan below runs only once it has failed.
    try:
        arrFloat = ll.astype('float')
    except ValueError:
        found = first_non_numeric(rows)
        where = (f" The first is '{found[2]}', on line {found[0]} at position {found[1]}."
                 if found else '')
        raise DeltaSvdError(f"A bval/bvec file may only hold numbers, but this one holds a value "
                            f"that is not a number.{where}\n {fname}") from None

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


#--- b-values at or below this count as b = 0: they carry no usable diffusion
#    weighting, and are what S0 is averaged from. Always kept, whatever b-value
#    selection the user asks for.
B0_MAX = 5

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

#--- Two gradient directions count as one below an angle of ~2.6 degrees. Even a
#    128-direction scheme separates its directions by more than 10 degrees, so
#    this can only ever merge genuine repeats.
DIRECTION_COS_TOL = 0.999

#--- How far a b-vector's length may deviate from 1; dipy's gradient_table() default.
BVEC_UNIT_TOL = 1e-2


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


def selected_volumes_float32(nii, idx):
    """nii.get_fdata()[..., idx] cast to float32, bit for bit, without holding
    every volume of the image in float64 at once.

    Only the stored (unscaled) data of the whole file is read, in its own type;
    each selected volume is then scaled exactly as get_fdata() scales it and cast
    to float32 on its own. The scaling below mirrors nibabel's
    ArrayProxy._get_scaled() for dtype float64 (unchanged from nibabel 5.3.3,
    the version the image pins, to 5.4); apply_read_scaling() works element by
    element with a precision chosen from the stored type only, so scaling one
    volume at a time gives the same values as scaling all of them. A test
    compares the result with get_fdata() for scaled and unscaled input.

    On multi-shell data this was the peak memory of the whole fit step: a
    288-volume HCP-sized image is ~8.4 GB in float64, before the selection
    copied part of it again."""
    proxy = nii.dataobj
    if not nib.is_proxy(proxy):                       # in-memory image
        return np.asarray(nii.get_fdata()[..., idx], dtype=np.float32)
    raw = np.asanyarray(proxy.get_unscaled())
    slope, inter = np.asanyarray(proxy.slope), np.asanyarray(proxy.inter)
    if np.can_cast(slope, np.float64):
        slope = slope.astype(np.float64)
    if np.can_cast(inter, np.float64):
        inter = inter.astype(np.float64)
    out = np.empty(raw.shape[:-1] + (len(idx),), dtype=np.float32)
    for k, i in enumerate(idx):
        vol = apply_read_scaling(raw[..., i], slope, inter)
        out[..., k] = vol.astype(np.promote_types(vol.dtype, np.float64), copy=False)
    return out


def filter_b_values(fn_data = 'data.nii.gz',
                fn_bval = 'file.bval',
                fn_bvec = 'file.bvec',
                out_dir = None,
                bIntervals = ((BRANGE_DEFAULT[0]-BRANGE_TOL, BRANGE_DEFAULT[1]+BRANGE_TOL),)):

    accepted = ', '.join(f'[{lo:g}, {hi:g}]' for lo, hi in bIntervals)
    print("Filtering DWI data according to b-values:")
    print(f"Accepted are b-values close to Zero (b-value <= {B0_MAX}) and in: {accepted}")

    bvals, bvalsStr = read_bval_or_bvec(fn_bval)
    bvecs, bvecsStr = read_bval_or_bvec(fn_bvec)
    if len(bvecs) != len(bvals):
        raise DeltaSvdError(f"The bvec file holds {len(bvecs)} directions but the bval file holds "
                            f"{len(bvals)} b-values. They have to describe the same volumes.\n"
                            f" bval: {fn_bval}\n bvec: {fn_bvec}")
    #--- volumes are selected by position below, so a mismatch would go unnoticed
    shape = nib.load(fn_data).shape
    if len(shape) != 4 or shape[3] != len(bvals):
        raise DeltaSvdError(f"The DWI image (dimensions {shape}) does not hold one volume per "
                            f"b-value ({len(bvals)}).\n DWI : {fn_data}\n bval: {fn_bval}")

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
            raise DeltaSvdError(f"No volume has a b-value in [{lo:g}, {hi:g}]. Check the requested "
                                f"b-values ('--bRange' / '--shells', tolerance included above) "
                                f"against the b-values in the data: {format_b_values(bvals)}.\n"
                                f" bval: {fn_bval}")
    if not np.any(selB0):
        raise DeltaSvdError(f"No volume with a b-value close to zero (b <= {B0_MAX}) was found. The "
                            f"fits need at least one to estimate S0. b-values in the data: "
                            f"{format_b_values(bvals)}.\n bval: {fn_bval}")

    #--- dipy's own rule (gradient_table, atol 1e-2), checked here to fail legibly
    norms = np.linalg.norm(bvecs, axis=1)
    notUnit = np.flatnonzero(selBRange & ~(np.abs(norms - 1) <= BVEC_UNIT_TOL))   # NaN fails too
    if notUnit.size:
        shown = ', '.join(f'{i+1} (length {norms[i]:.3g})' for i in notUnit[:MAX_LINES_REPORTED])
        raise DeltaSvdError(f"Every b-vector of a diffusion-weighted volume has to be a unit vector, "
                            f"but {notUnit.size} are not. Volume(s): {shown}"
                            f"{', ...' if notUnit.size > MAX_LINES_REPORTED else ''}\n"
                            f" bvec: {fn_bvec}")

    nDirections, rank = describe_directions(bvals[sel], bvecs[sel])
    print(f' unique diffusion directions   : n={nDirections}')
    if nDirections < MIN_DIRECTIONS:
        raise DeltaSvdError(f"Only {nDirections} unique diffusion direction(s) remain after the "
                            f"b-value selection, but the free-water bi-tensor fit needs at least "
                            f"{MIN_DIRECTIONS} (repeated directions constrain the free-water "
                            f"fraction no further than a single one does, so they are counted "
                            f"once). Check the requested b-values ('--bRange' / '--shells') "
                            f"against the b-values in the data: {format_b_values(bvals)}.\n"
                            f" bval: {fn_bval}\n bvec: {fn_bvec}")
    if rank < DESIGN_MATRIX_RANK:
        raise DeltaSvdError(f"The {nDirections} diffusion directions do not span the diffusion "
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
        img = selected_volumes_float32(nii, np.flatnonzero(sel))

        fn_bval = join(out_dir, basename(fn_bval))
        fn_bvec = join(out_dir, basename(fn_bvec))
        fn_data = join(out_dir, basename(fn_data))
        write_bval_or_bvec(bvals, fn_bval)
        write_bval_or_bvec(bvecs, fn_bvec)
        # dtype='float32' explicit: a lossless round-trip for the float32 input the
        # pipeline expects (do not change -- float64 would alter results for
        # int16-with-scaling input). 'img' is float32 already, see
        # selected_volumes_float32().
        save_nifti(fn_data, img, nii.affine, nii.header, dtype='float32')

        print('New data saved to:')
        print(fn_bval)
        print(fn_bvec)
        print(fn_data)


    return fn_data, fn_bval, fn_bvec, nDirections
