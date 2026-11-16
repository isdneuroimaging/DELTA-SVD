#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line parser, argparse types, input path resolution and input
validation.

Imports numpy and nibabel, so it must be imported after delta-svd.py has
pinned the threads and the BLAS kernel.
"""

import os, sys, argparse, re
from os.path import join, exists, dirname

from delta_svd_constants import (SKELETON_MASK_DEFAULT, ITK_THREADS_DEFAULT, BRANGE_DEFAULT,
                                 BRANGE_TOL, SHELL_TOL, ITERATIONS_DEFAULT,
                                 NUM_REGISTRATIONS_DEFAULT, B0_MAX)

import nibabel as nib
import numpy as np

from delta_svd_version import __version__
from delta_svd_common import DeltaSvdError, strip_nifti_ext, validate_roi_labels
from delta_svd_gradients import BVAL_MIN, BVAL_MAX


###########################################################################
# Parser, argparse types and path resolution

NIFTI_EXTENSIONS = ('.nii.gz', '.nii')


def nifti_sibling(s):
    """The same path carrying the other NIfTI extension, when that file exists.

    Asking for 'data.nii' where only 'data.nii.gz' is present is an easy mistake,
    and the extension probing in isNIfTI() cannot catch it: that appends to the
    name as given, so it looks for 'data.nii.nii.gz'. Reported rather than
    accepted -- the file asked for is not the file found, and substituting it
    silently would be the wrong kind of helpful."""
    if s.endswith('.nii.gz'):
        alt = s[:-len('.gz')]
    elif s.endswith('.nii'):
        alt = s + '.gz'
    else:
        return None
    return alt if os.path.isfile(alt) else None


def isNIfTI(s, abort=True):
    if os.path.isfile(s) and s.endswith(NIFTI_EXTENSIONS):
        return s
    elif os.path.isfile(s+'.nii.gz'):
        return s+'.nii.gz'
    elif os.path.isfile(s+'.nii'):
        return s+'.nii'
    else:
        if abort:
            # An existing file that is merely named wrong needs a different fix
            # from one that is not there at all, so the two are told apart. This
            # stays an ArgumentTypeError: argparse calls it as a 'type' and turns
            # that into its own clean 'error:' line.
            if os.path.isfile(s):
                raise argparse.ArgumentTypeError(
                    f"file is not a NIfTI image -- its name ends in neither '.nii' nor "
                    f"'.nii.gz'. Please check: {s}")
            sibling = nifti_sibling(s)
            if sibling is not None:
                raise argparse.ArgumentTypeError(
                    f"file does not exist, but one with the other NIfTI extension does:\n"
                    f"    {sibling}\n  Please check: {s}")
            # The probes above append to the name as given, so they are only worth
            # reporting when it carried no NIfTI extension of its own -- otherwise
            # they looked for 'data.nii.nii.gz' and claiming they tried '.nii.gz'
            # would be untrue.
            if s.endswith(NIFTI_EXTENSIONS):
                raise argparse.ArgumentTypeError(f"file does not exist. Please check: {s}")
            raise argparse.ArgumentTypeError(
                f"file does not exist (also tried '{s}.nii.gz' and '{s}.nii'). "
                f"Please check: {s}")
        else:
            return None


def per_timepoint_path(fn, dwi):
    """Where an input given per timepoint is looked for: a bare file name next
    to that timepoint's DWI image, anything carrying a folder exactly as given.

    A bare name is never looked for in the working directory. It used to be,
    first, so a file of that name there was silently taken for every timepoint
    -- e.g. 'bvals' run from inside one session's folder gave the other
    sessions that session's b-values. A name carrying a folder ('./bvals',
    'ses-1/bvals', an absolute path) is taken literally and never re-anchored.

    Inferred names need no special case: they are built from the DWI path, so
    they carry its folder already, or none when the DWI has none either."""
    if dirname(fn):
        return fn
    return join(dirname(dwi), fn)


def candidate_paths(fn, dwi, anyExtension):
    """The paths an input given per timepoint is looked for at, in order: the
    one per_timepoint_path() names, optionally completed with a NIfTI extension.

    This mirrors the resolution for the sake of the error message only. What is
    accepted stays entirely with isNIfTI()/exists(), so this cannot widen it."""
    p = per_timepoint_path(fn, dwi)
    # isNIfTI() completes a path that carries no NIfTI extension of its own.
    # It does probe 'p' itself either way, but a name already ending in one
    # can only ever match as itself, so listing the completions would be noise.
    if anyExtension and not p.endswith(NIFTI_EXTENSIONS):
        return [p + e for e in NIFTI_EXTENSIONS]
    return [p]


def missing_input_message(attr, fn, dwi, anyExtension, inferred, note=''):
    """Why a per-timepoint input could not be found, and what to do about it.

    Names the timepoint by its DWI image rather than by an index: the '--tp'
    labels are not resolved yet at this point, and the path is what the user has
    to go and look at anyway."""
    paths = candidate_paths(fn, dwi, anyExtension)
    tried = '\n'.join(f'   {p}' for p in paths)
    if inferred:
        origin = f" No '--{attr}' was given, so it was inferred from the DWI path."
        fix = f" Provide the correct path with '--{attr}'."
    else:
        origin = f" It was given as '{fn}'."
        fix = f" Please check the path given with '--{attr}'."
    #--- the likeliest near-miss: the file is there under the other extension
    siblings = []
    for p in paths:
        alt = nifti_sibling(p)
        if alt is not None and alt not in siblings:
            siblings.append(alt)
    hint = ''
    if siblings:
        hint = (' A file with the other NIfTI extension is there:\n'
                + '\n'.join(f'   {a}' for a in siblings) + '\n')
    return (f"No '--{attr}' file was found for the DWI image:\n   {dwi}\n"
            f"{origin}\n Looked for:\n{tried}\n{hint}{fix}{note}")

def isCSV(s):
    if '/' in s or '\\' in s:
        raise argparse.ArgumentTypeError(f"--reprocess requires a CSV base name in the output folder; you provided '{s}'")
    if s == 'overwrite' or s.endswith('.csv') or s.endswith('.CSV'):
        return s
    else:
        raise argparse.ArgumentTypeError(f"--reprocess filename must end in '.csv'; you provided '{s}'")

def validate_timepoint_labels(labels):
    reserved = {'template', 'intermediateTemplates', 'TBSS'}
    for label in labels:
        if os.path.isabs(label):
            reason = 'is an absolute path'
        elif '..' in re.split(r'[\\/]', label):
            reason = 'contains path traversal'
        elif '/' in label or '\\' in label:
            reason = 'contains a path separator'
        elif label in reserved:
            reason = 'collides with an internal working directory'
        elif label in ('', '.'):
            reason = 'is not a usable directory name'
        elif any(ch.isspace() for ch in label):
            # the label becomes part of file names that the FSL and ANTs
            # scripts pass around unquoted, which splits them at whitespace
            reason = 'contains whitespace'
        else:
            continue
        raise DeltaSvdError(f"Invalid timepoint label given with '--tp': '{label}' {reason}. "
                            "Use a plain label such as 'TP01' or 'ses-1', not a path, an "
                            "internal directory name or a name containing spaces.")

def positive_int(option):
    """An argparse 'type' accepting integers >= 1, naming 'option' in its errors."""
    def check(s):
        try:
            v = int(s)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{option} must be a positive integer; you provided '{s}'")
        if v < 1:
            raise argparse.ArgumentTypeError(f'{option} must be at least 1; you provided {s}')
        return v
    return check


assertPositiveJobs = positive_int('--para (number of parallel ANTs registration jobs)')
assertPositiveRegistrations = positive_int('--numRegistrations (iterations of the template construction)')
assertPositiveItkThreads = positive_int('--itkThreads (ITK threads per ANTs registration job)')


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
                    # The likeliest cause: a long option typed with one dash
                    # instead of two, e.g. '-shells' for '--shells'. Only worth
                    # naming when it really is one of ours; self.error() exits,
                    # so this is built once and never on the success path.
                    known = {opt for a in self._actions for opt in a.option_strings}
                    long_form = '--' + arg_string[1:].split('=', 1)[0]
                    hint = f' Did you mean "{long_form}"?' if long_form in known else ''
                    self.error(f'single-dash options must be one character and separated from their argument by a space: "{arg_string}".{hint}')
        return super().parse_known_args(args, namespace)
    
stepsImplemented = ['fwc','template','tbss','tbss_non_fa','extract','qc']
argparseDescription = f"DELTA-SVD {__version__} ('Diffusion Endpoints for Longitudinal Tracking of white matter Alterations in cerebral Small Vessel Disease') processes multi-directional diffusion MRI data to fully automatically extract clinically and technically validated white matter diffusion metrics. Key steps include diffusion tensor fitting (with and without free water imaging), skeletonisation based on the free water-corrected FA (fwc-FA) via FSL's TBSS, and enhanced CSF partial volume masking. The final metrics MSMD, PSMD, and MSFW are computed over the skeleton. For longitudinal data, a within-subject template is created using ANTs."

def iniParser():
    parser = CustomArgumentParser(description=argparseDescription, epilog='Notice: By using DELTA-SVD, you agree to the licence terms (CC BY-NC-ND 4.0) described in the LICENSE file at "https://github.com/isdneuroimaging/DELTA-SVD"')
    parser.add_argument("--version", action='version', version=f'DELTA-SVD {__version__}', help="show the DELTA-SVD version and exit")
    group0 = parser.add_argument_group('input/output data specification')
    group0.add_argument("--dwi", required=True, metavar='NIfTI', type=isNIfTI, nargs="+", action='extend', help="input path(s) to 4D DWI image(s) in NIfTI format. Number of arguments should correspond to number of timepoints.")
    group0.add_argument("--bval", metavar='text-file', type=str, nargs="+", action='extend', help="input path(s) to text file(s) with b-values in FSL format, corresponding to DWI image(s). A bare file name (without a folder) is looked for next to the corresponding DWI image, never in the working directory; a path containing a folder (e.g. './file' or an absolute path) is used as given. A single value is used for all timepoints, so a single bare file name is looked for next to each DWI image. If argument not provided, path(s) will be constructed from DWI image path(s), substituting extension with '.bval'")
    group0.add_argument("--bvec", metavar='text-file', type=str, nargs="+", action='extend', help="input path(s) to text file(s) with b-vectors in FSL format, corresponding to DWI image(s). A bare file name (without a folder) is looked for next to the corresponding DWI image, never in the working directory; a path containing a folder (e.g. './file' or an absolute path) is used as given. A single value is used for all timepoints, so a single bare file name is looked for next to each DWI image. If argument not provided, path(s) will be constructed from DWI image path(s), substituting extension with '.bvec'")
    group0.add_argument("--bmask", metavar='NIfTI', type=str, nargs="+", action='extend', help="input path(s) to DWI brain mask(s) in NIfTI format, corresponding to DWI image(s) and on their grid (same dimensions and affine). A bare file name (without a folder) is looked for next to the corresponding DWI image, never in the working directory; a path containing a folder (e.g. './file' or an absolute path) is used as given. A single value is used for all timepoints, so a single bare file name is looked for next to each DWI image. If argument not provided, path(s) will be constructed from DWI image path(s), substituting the extension with '_brainmask.nii.gz' or, if that file does not exist, with '_brainmask.nii'. Masks are binarised: values greater than zero are set to 1; zero and negative values are set to 0.")
    group0.add_argument("--tp", metavar='label', type=str, nargs="+", action='extend', help="label(s) for all timepoints. Number of arguments should correspond to number of DWI image(s). Labels must be plain names without path separators, traversal or whitespace and must not collide with internal working directories. Labels have to be unique, and 'all' is reserved for the rows summarising all timepoints. If argument not provided, timepoints are labelled consecutively as TP01, TP02, and so on.")
    group0.add_argument("--id", metavar='label', type=str, help="optional subject ID. If provided, an additional column with this identifier will be added to the results table 'delta-svd_results.csv', meant to facilitate aggregation of results tables for multiple subjects.")
    group0.add_argument("-o", "--dirOutput", type=str, help="path to output folder. If not provided, the parent folder of the first DWI image will be used. The results table ('delta-svd_results.csv') and a subfolder and HTML for quality checking ('delta-svd_qc' and 'delta-svd_qc.html') will be saved here. Furthermore, intermediate/temporary files will be created here inside a subfolder called 'delta-svd_temp'.")
    group1 = parser.add_argument_group('additional masking')
    group1.add_argument("--Emask", metavar='NIfTI', type=str, default = [], nargs="+", action='extend', help="input path(s) to custom exclusion mask(s) in DWI image space, i.e. on the grid of the DWI image (same dimensions and affine), used to exclude the masked region from analysis. One per timepoint can be provided, which will be merged in template space. Timepoints will be matched by position of provided paths. A bare file name (without a folder) is looked for next to the DWI image of that timepoint; a path containing a folder is used as given. Skip timepoints by entering NA instead of a path. The masked area (e.g. lesion) will be excluded from analysis. Provided masks are binarised: values greater than zero are set to 1; zero and negative values are set to 0.")
    group1.add_argument("--Rmask", metavar='NIfTI', type=str, default = [], nargs="+", action='extend', help="input path(s) to custom ROI mask(s) in DWI image space, i.e. on the grid of the DWI image (same dimensions and affine). One per timepoint can be provided, which will be merged in template space. Timepoint matching and/or skipping works as explained for option 'Emask'. Labels must be finite integers from 0 through 65535. Every label defines a separate ROI, including background label 0. However, masks will be merged in template space and if labels in masks from different timepoints disagree, the respectively highest integer label will overwrite the other labels.")
    group1.add_argument("--RmaskMNI", metavar='NIfTI', type=isNIfTI, help="input path to a single custom ROI mask in MNI space. It must be on the FMRIB58 1 mm MNI grid (same image dimensions and affine as the default skeleton mask). Labels must be finite integers from 0 through 65535. Every label defines a separate ROI, including background label 0.")
    group1.add_argument("--hemispheres", action='store_true', help="calculate skeleton metrics also separately for left and right hemispheres. Please note, however, that this does not affect ROI masks, which will not be split between hemispheres.")
    group2 = parser.add_argument_group('advanced options')
    group2.add_argument("--skeletonMask", metavar='NIfTI', type=isNIfTI, default=SKELETON_MASK_DEFAULT, help="input path to an alternative skeleton mask. It must be on the FMRIB58 1 mm MNI grid (same image dimensions and affine as the default mask). It will be binarised: values greater than zero are set to 1; zero and negative values are set to 0. Defaults to the mask validated with DELTA-SVD ('delta-svd_skeletonmask_v1') and designed to exclude regions with frequent CSF partial volume effects.")
    group2b = group2.add_mutually_exclusive_group()
    group2b.add_argument("--bRange", metavar='Integer', type=assertBValue, default = list(BRANGE_DEFAULT), nargs=2, help=f"range of b-values to consider for diffusion tensor fitting, given as the lower and upper limit of the non-zero shell(s) to include. Defaults to range [{BRANGE_DEFAULT[0]},{BRANGE_DEFAULT[1]}]. The limits are met with a tolerance of {BRANGE_TOL} s/mm2, so that shells the scanner reports slightly off their nominal value are not discarded. Volumes with a b-value close to zero (b <= {B0_MAX}) are always included and are not affected by this option. Both limits have to lie between {BVAL_MIN} and {BVAL_MAX} s/mm2, outside of which the diffusion tensor model is not valid. Mutually exclusive with '--shells'.")
    group2b.add_argument("--shells", metavar='Integer', type=assertBValue, nargs="+", action='extend', help=f"b-value shell(s) to consider for diffusion tensor fitting, e.g. '--shells 700 1000'. An alternative to '--bRange' for selecting shells individually rather than as one range, which avoids pulling in the shells in between. Each shell is matched with a tolerance of {SHELL_TOL} s/mm2, and a shell that matches no volume in the data is an error. As for '--bRange', volumes with a b-value close to zero (b <= {B0_MAX}) are always included, and each shell has to lie between {BVAL_MIN} and {BVAL_MAX} s/mm2. Mutually exclusive with '--bRange'.")
    group2.add_argument("--smooth", action='store_true', help=argparse.SUPPRESS) #--- "apply Gaussian filter (fwhm = 1.25) to DWI data"
    group2.add_argument("--dontAdjustBmaskForFW", dest='adjustBmaskForFW', action='store_false', help=argparse.SUPPRESS) #--- "don't correct the brain mask for free-water. By default, the brain mask is set to zero, where free water equals 1 (and hence fwc-FA equals 0)."
    group2.add_argument("--para", metavar='ANTs-jobs', type=assertPositiveJobs, default=None, help="number of ANTs registration jobs run at once during longitudinal template construction. Derived from the '--threads' budget by default, and capped at the number of timepoints either way. Peak memory scales with it, so '--para 1' is the lowest-memory setting. It has no effect on the results, only on runtime and memory.")
    group2.add_argument("--threads", metavar='cores', type=threadBudget, default='auto', help="number of physical CPU cores DELTA-SVD may use. Two steps are multi-core: the diffusion tensor / free-water fit, and (for longitudinal input only) the within-subject template construction; TBSS and the remaining steps are single-threaded. Defaults to 'auto', which detects the cores available to the process, honouring an HPC scheduler's allocation and a container CPU limit (e.g. Docker's '--cpus'). It has no effect on the results, only on runtime, so it can be tuned freely.")
    group2.add_argument("--itkThreads", metavar='threads', type=assertPositiveItkThreads, default=ITK_THREADS_DEFAULT, help=argparse.SUPPRESS) #--- "Expert override for the ITK threads used per ANTs registration job. WARNING: this changes the computed metrics -- ITK sums the registration metric per thread, so a different count sums in a different order. Defaults to 12, the value DELTA-SVD was validated at. Results produced with different values must not be compared or pooled."
    group2.add_argument("--iterations", type=str, default=ITERATIONS_DEFAULT, help=argparse.SUPPRESS) #--- "Iterations at each resolution level of the pairwise ANTs registrations during template creation. Must be three levels and specified in the format: 'L1xL2xL3'. Defaults to '30x30x8'."
    group2.add_argument("--numRegistrations", type=assertPositiveRegistrations, default=NUM_REGISTRATIONS_DEFAULT, help=argparse.SUPPRESS) #--- "Iterations of the template construction. Each iteration comprises averaging of images and pairwise registrations of each timepoint to the template. Defaults to 3 iterations."
    group2.add_argument("--qc", type=int, choices=[0,1,2], default=1, help="create (with argument 1; the default) a HTML file (delta-svd_qc.html) for quality checking and create (with argument 2) additionally a subfolder 'delta-svd_qc' with a series of NIfTI images showing skeleton and masks in native space, or (with argument 0) skip creation of both.")
    group2.add_argument("--debug", action='store_true', help="don't delete temporary folder 'delta-svd_temp', containing intermediate files created during processing.")
    group2.add_argument("--steps", choices = stepsImplemented, nargs="+", action='extend', help=argparse.SUPPRESS) #--- "choose step(s) to be conducted. By default all steps will be conducted. If the output for preceding steps is missing, an error will be raised. If different masks are provided, step 'tbss_non_fa' and following have to be repeated."
    group2.add_argument("--reprocess", metavar='csv-file',  type=isCSV, nargs="?", const='overwrite', help='allow reprocessing and overwriting of previously created output. You can, however, keep a previously created results file (default name: delta-svd_results.csv) by specifying here an alternative name for the new one (provide only the base name; the output folder cannot be changed here).')
    return parser


###########################################################################
# Grid checks

# absorbs float32 header round-off in check_mni_grid()
GRID_AFFINE_ATOL = 1e-4


def check_mni_grid(fname, label, reference=None):
    """Raise unless 'fname' has the shape and affine of 'reference' (default: the
    default skeleton mask, i.e. the FMRIB58 1 mm grid). Reads headers only."""
    if reference is None:
        reference = SKELETON_MASK_DEFAULT
    if fname == reference:
        return
    img, ref = nib.load(fname), nib.load(reference)
    if img.shape != ref.shape:
        raise DeltaSvdError(f"The provided {label} does not have the image dimensions of the FMRIB58 "
                            f"1 mm MNI grid the analysis is done on:\n  {fname}\n"
                            f"  Dimensions: {img.shape}, expected: {ref.shape}\n"
                            f" Please resample it to this grid, e.g. using '{reference}' "
                            f"or FSL's FMRIB58_FA_1mm as the reference image.")
    if not np.allclose(img.affine, ref.affine, atol=GRID_AFFINE_ATOL):
        raise DeltaSvdError(f"The provided {label} has the image dimensions of the FMRIB58 1 mm MNI "
                            f"grid the analysis is done on, but not its voxel-to-world mapping "
                            f"(affine), so its voxels would not line up with the skeleton:\n"
                            f"  {fname}\n  Affine:\n{np.array2string(img.affine, precision=4)}\n"
                            f"  Expected:\n{np.array2string(ref.affine, precision=4)}\n"
                            f" Please resample it to this grid, e.g. using '{reference}' "
                            f"or FSL's FMRIB58_FA_1mm as the reference image.")


def check_dwi_grid(fname, label, attr, dwi):
    """Raise unless 'fname' has the spatial dimensions and affine of the DWI
    image 'dwi'. Reads headers only.

    Nothing else would notice: the brain mask only fails later, with a
    traceback from the fit, and an exclusion or ROI mask on another grid (a
    lesion mask left in FLAIR space, say) is carried through TBSS and applied
    misaligned without any error."""
    img, ref = nib.load(fname), nib.load(dwi)
    fix = (f" Masks given with '--{attr}' have to be in the space of their DWI image, i.e. "
           f"resampled to its grid (same dimensions and voxel-to-world mapping).")
    if img.shape[:3] != ref.shape[:3]:
        raise DeltaSvdError(f"The provided {label} does not have the image dimensions of its DWI "
                            f"image:\n  {fname}\n  Dimensions: {img.shape[:3]}, DWI: "
                            f"{ref.shape[:3]}\n  DWI: {dwi}\n{fix}")
    if not np.allclose(img.affine, ref.affine, atol=GRID_AFFINE_ATOL):
        raise DeltaSvdError(f"The provided {label} has the image dimensions of its DWI image, but "
                            f"not its voxel-to-world mapping (affine), so its voxels would not "
                            f"line up with the DWI data:\n  {fname}\n  Affine:\n"
                            f"{np.array2string(img.affine, precision=4)}\n  DWI affine:\n"
                            f"{np.array2string(ref.affine, precision=4)}\n  DWI: {dwi}\n{fix}")


def resolve_inputs(args):
    """Resolve and validate the inputs on 'args', in place. Returns the steps
    available to this run, in order."""
    # Check bval, bvec and bmask files. The brain mask may be '.nii.gz' or '.nii',
    # so it is resolved through isNIfTI(), which probes both (gzip first) and also
    # completes a path with no extension; bval/bvec are matched verbatim.
    for attr, ext in zip(['bval','bvec','bmask'], ['.bval','.bvec','_brainmask']):
        anyExtension = (attr == 'bmask')
        resolve = (lambda fn: isNIfTI(fn, abort=False)) if anyExtension else (lambda fn: fn if exists(fn) else None)
        flist = getattr(args,attr)
        inferred = flist is None
        if inferred:
            flist = [strip_nifti_ext(fn) + ext for fn in args.dwi]
        elif len(flist) != len(args.dwi):
            if len(flist) == 1:
                flist = flist * len(args.dwi)
            else:
                raise DeltaSvdError(f"The number of files given with option '--{attr}' (n={len(flist)}) "
                                    f"has to be either one or match the number of DWI images "
                                    f"(n={len(args.dwi)}). Please refer to '--help'.")
        for i, fn in enumerate(flist):
            fnResolved = resolve(per_timepoint_path(fn, args.dwi[i]))
            if fnResolved is None:
                raise DeltaSvdError(missing_input_message(attr, fn, args.dwi[i], anyExtension, inferred))
            flist[i] = fnResolved
        setattr(args,attr,flist)
    
    # Check timepoint labels
    if args.tp is None:
        args.tp = ['TP{:02d}'.format(i+1) for i in range(len(args.dwi))] #-- folders for all timepoints
    if len(args.tp) != len(args.dwi):
        raise DeltaSvdError(f"The number of timepoint labels given with '--tp' (n={len(args.tp)}) has "
                            f"to match the number of DWI images (n={len(args.dwi)}).")
    validate_timepoint_labels(args.tp)
    duplicates = sorted({tp for tp in args.tp if args.tp.count(tp) > 1})
    if duplicates:
        raise DeltaSvdError(f"Timepoint labels given with '--tp' have to be unique. You passed "
                            f"{', '.join(duplicates)} more than once.")
    if len(args.dwi) > 1 and 'all' in args.tp:
        raise DeltaSvdError("'all' is reserved as the label for the rows summarising all timepoints, "
                            "so it cannot be used as a '--tp' label.")

    # Check exclusion and ROI masks. Both are optional and per timepoint, with
    # 'NA' skipping one, so a short list is padded rather than rejected.
    maskLabels = {'Emask': 'exclusion mask', 'Rmask': 'ROI mask in DWI space'}
    for attr, label in maskLabels.items():
        masks = getattr(args, attr)
        if len(masks) > len(args.dwi):
            raise DeltaSvdError(f"More {label}s were given with '--{attr}' (n={len(masks)}) than there "
                                f"are timepoints (n={len(args.dwi)}). At most one per timepoint is "
                                f"allowed; enter 'NA' to skip a timepoint.")
    for i,_ in enumerate(args.dwi):
        for attr in maskLabels:
            masks = getattr(args, attr)
            if len(masks) <= i:
                masks.append(None)
                continue
            if masks[i] == 'NA':
                masks[i] = None
                continue
            fnResolved = isNIfTI(per_timepoint_path(masks[i], args.dwi[i]), abort=False)
            if fnResolved is None:
                raise DeltaSvdError(missing_input_message(
                    attr, masks[i], args.dwi[i], True, False,
                    note=" Enter 'NA' instead of a path to skip this timepoint."))
            # assigned back, so that a name resolved by extension or against the
            # DWI folder is the one every later step loads
            masks[i] = fnResolved

    for fn in args.Rmask:
        if fn is not None:
            validate_roi_labels(nib.load(fn).get_fdata(), 'ROI mask in DWI space', fn)
    if args.RmaskMNI is not None:
        validate_roi_labels(nib.load(args.RmaskMNI).get_fdata(),
                            'ROI mask in MNI space', args.RmaskMNI)

    # Absolute, so the log shows which file each timepoint really uses: the
    # same relative name can stand for a different file per timepoint.
    print(f"\nInput contains N={len(args.dwi)} timepoints")
    for i in range(len(args.dwi)):
        print(f'Timepoint {args.tp[i]}:')
        print(f' DWI   :{os.path.abspath(args.dwi[i])}')
        print(f' bval  :{os.path.abspath(args.bval[i])}')
        print(f' bvec  :{os.path.abspath(args.bvec[i])}')
        print(f' Bmask :{os.path.abspath(args.bmask[i])}')
        if args.Emask[i]:
            print(f' Emask :{os.path.abspath(args.Emask[i])}')
        if args.Rmask[i]:
            print(f' Rmask :{os.path.abspath(args.Rmask[i])}')

    if args.RmaskMNI is not None:
        print(f'\nAn additional ROI mask in MNI space (RmaskMNI) was provided:\n {args.RmaskMNI}')

    if args.hemispheres:
        print('\nHemispheric skeleton analysis will be done as well')

    if args.skeletonMask == SKELETON_MASK_DEFAULT:
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
            raise DeltaSvdError("You asked for the 'qc' step with '--steps qc' and to skip it with "
                                "'--qc 0' at the same time. That is contradictory; please drop one "
                                "of the two.")
        stepsAvailable.remove('qc')
    if args.steps is None:
        args.steps = stepsAvailable
    else:
        stepsIdx = [i for i,x in enumerate(stepsAvailable) if x in args.steps]
        args.steps = [stepsAvailable[i] for i in stepsIdx]
        print('\nOn request, only the following processing steps will be conducted:')
        for i,step in enumerate(args.steps): print(f' {i+1}. {step}')
        if len(stepsIdx)>1 and any(np.diff(stepsIdx)>1):
            print(' ')
            raise DeltaSvdError(f"The requested processing steps have to be contiguous, and these "
                                f"are not. Each step consumes the output of the one before it.\n "
                                f"The available steps, in order, are: {', '.join(stepsAvailable)}")
        if 'extract' not in args.steps and not args.debug:
            print("NOTE: Given that the final 'extract' step is not selected, we assume that you want to keep intermediate/temporary output and switch on the option '--debug' for you!")
            args.debug = True

    # After the cheaper checks, before any output is touched: a wrong shape would
    # crash hours in, a wrong affine silently misalign the skeleton.
    for i, dwi in enumerate(args.dwi):
        check_dwi_grid(args.bmask[i], 'brain mask', 'bmask', dwi)
        for attr, label in maskLabels.items():
            fn = getattr(args, attr)[i]
            if fn is not None:
                check_dwi_grid(fn, label, attr, dwi)
    check_mni_grid(args.skeletonMask, 'skeleton mask')
    if args.RmaskMNI is not None:
        check_mni_grid(args.RmaskMNI, 'ROI mask in MNI space')

    return stepsAvailable
