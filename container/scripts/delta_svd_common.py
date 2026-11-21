#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers shared by delta-svd.py and the modules next to it.

Imports numpy and nibabel, so it must be imported after delta-svd.py has
pinned the threads and the BLAS kernel.
"""

import os, re, subprocess
from shutil import rmtree

import nibabel as nib
import numpy as np

ROI_LABEL_DTYPE = 'uint16'
ROI_LABEL_MAX = np.iinfo(ROI_LABEL_DTYPE).max


class DeltaSvdError(ValueError):
    """A problem with the input or with what was asked for -- not a bug.

    Raised for everything a user can fix by changing the command line or the
    data, and caught in the __main__ guard at the bottom of delta-svd.py, which
    reports it the way argparse reports its own errors: one message on stderr,
    no traceback. Anything else that escapes is a fault in the pipeline and
    keeps its traceback, which is what makes it debuggable.

    An exception rather than a direct sys.exit() because these checks live in
    functions that are unit-tested directly and called as a library; exiting
    from inside them would make that impossible. It derives from ValueError
    because that is what they raised before, and callers may still catch it
    as one.

    delta-svd_aggregate_results.py has no such structure -- every check there
    sits in one linear script-level flow -- so it calls sys.exit() directly
    instead of importing this. Both produce the same 'ERROR: ...' on stderr.
    """


def save_nifti(fname, arr, affine, header, dtype='float32'):
    niiNew = nib.Nifti1Image(arr, affine, header)
    niiNew.set_data_dtype(dtype)
    niiNew.header.set_slope_inter(1, 0)
    nib.save(niiNew, fname)


def strip_nifti_ext(fn):
    """'fn' without a trailing '.nii' or '.nii.gz'."""
    return re.sub(r'\.nii(\.gz)?$', '', fn)


def tbss_fa_name(nTP):
    """Base name of the FA image TBSS is run on."""
    return "FA-for-tbss-long" if nTP > 1 else "fwc_wls_dti_FA_05"


def validate_roi_labels(img, label, fname):
    """Return sorted ROI labels after enforcing the public label contract."""
    labels = np.unique(np.asarray(img))
    invalid = labels[(~np.isfinite(labels)) |
                     (labels < 0) |
                     (labels > ROI_LABEL_MAX) |
                     (labels != np.floor(labels))]
    if invalid.size:
        shown = ', '.join(str(value) for value in invalid[:5])
        if invalid.size > 5:
            shown += ', ...'
        raise DeltaSvdError(
            f"The provided {label} must contain only finite integer labels from 0 through "
            f"{ROI_LABEL_MAX}:\n  {fname}\n  Unsupported value(s): {shown}")
    return labels.astype(ROI_LABEL_DTYPE)


def run_subprocess(cmd, displayStdout, label):
    print(f"Calling {label} command with:")
    print(cmd, '\n')
    # streamed when displayed; otherwise kept for the failure report only
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True,
                            text=True, encoding='utf-8', errors='replace')
    held = []
    for line in proc.stdout:
        if displayStdout:
            print(line, end='')
        else:
            held.append(line)
    proc.wait()
    if proc.returncode != 0:
        if held:
            print("STDOUT/STDERR:")
            print(''.join(held))
        raise DeltaSvdError(f"The '{label}' command failed with exit code {proc.returncode}. "
                            f"Its stdout/stderr is printed above.")


def safe_rmtree(path, parent):
    pathReal = os.path.realpath(os.path.abspath(path))
    parentReal = os.path.realpath(os.path.abspath(parent))
    try:
        contained = os.path.commonpath([pathReal, parentReal]) == parentReal
    except ValueError:
        contained = False
    if not contained or pathReal == parentReal:
        raise DeltaSvdError(f"Refusing to recursively delete path outside its intended output "
                            f"directory: {path}")
    rmtree(path)
