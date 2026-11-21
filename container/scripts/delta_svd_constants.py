#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Defaults shared by delta-svd.py, its delta_svd_*.py modules and the QC report
(create_html_with_png.py).

No third-party imports: delta-svd.py imports this before pinning the BLAS
kernel, which must precede the first numpy import.
"""

#--- The validated skeleton mask, as installed in the image.
SKELETON_MASK_DEFAULT = "/opt/scripts/delta-svd_skeletonmask_v1.nii.gz"

#--- ITK threads per registration job, the one threading setting that changes
#    the metrics: ITK sums the metric per thread, so the count sets the summation
#    order, and the skeleton amplifies the last-bit differences (1 vs 12 threads
#    moves delta-PSMD by ~23%). 12 is the validated value; see CONTRIBUTING.md.
ITK_THREADS_DEFAULT = 12

#--- Template construction (ANTs): iterations per resolution level of each
#    pairwise registration ('-q') and number of template iterations ('-i').
#    Both are validated values; see CONTRIBUTING.md.
ITERATIONS_DEFAULT = "30x30x8"
NUM_REGISTRATIONS_DEFAULT = 3

#--- b-values at or below this count as b = 0: they carry no usable diffusion
#    weighting, and are what S0 is averaged from. Always kept, whatever b-value
#    selection the user asks for.
B0_MAX = 5

#--- The default '--bRange'.
BRANGE_DEFAULT = (800, 1200)

#--- Tolerances for scanner b-values off the nominal value: a range only absorbs
#    rounding, while a shell's tolerance is its whole acceptance window. Neither
#    merges adjacent shells (>= 100 s/mm2 apart) or applies to b = 0 (B0_MAX).
BRANGE_TOL = 5
SHELL_TOL = 25

#--- Where the image installs the pipeline scripts; running from here is what
#    identifies a run inside the image.
IMAGE_SCRIPT_DIR = "/opt/scripts"

#--- Initial target of the longitudinal template construction (ANTs '-z').
ANTS_TEMPLATE_INIT = IMAGE_SCRIPT_DIR + "/FMRIB58_FA_2mm_crop.nii.gz"

#--- Environment that selects the image's FSL, ANTs and BLAS, mirrored by the
#    Dockerfile's ENV (a test keeps them in step). A user can still override ENV
#    ('docker run -e', APPTAINERENV_*), and another FSLDIR, ANTSPATH or
#    OPENBLAS_CORETYPE changes the metrics, so delta-svd.py assigns these outright
#    in the image. PYTHONNOUSERSITE keeps Python subprocesses (e.g. fsl_sub) off a
#    mounted ~/.local; PATH selects the FSL and ANTs commands called by name.
IMAGE_ENVIRONMENT = {
    "PATH": "/opt/scripts:/opt/conda/bin:/opt/ants-2.4.3/bin:"
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "FSLDIR": "/opt/conda",
    "FSLOUTPUTTYPE": "NIFTI_GZ",
    "ANTSPATH": "/opt/ants-2.4.3/bin",
    "OPENBLAS_CORETYPE": "Haswell",
    "PYTHONUNBUFFERED": "1",
    "PYTHONNOUSERSITE": "1",
}

#--- Fewer unique directions than the minimum are refused, fewer than the
#    recommended count warned about; see DESIGN_MATRIX_RANK in delta_svd_gradients.py.
MIN_DIRECTIONS = 12
RECOMMENDED_DIRECTIONS = 20
