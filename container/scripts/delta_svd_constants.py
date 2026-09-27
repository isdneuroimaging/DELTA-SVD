#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Defaults shared by delta-svd.py and the QC report (create_html_with_png.py).

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

#--- The default '--bRange'.
BRANGE_DEFAULT = (800, 1200)

#--- Tolerances for scanner b-values off the nominal shell. A range only needs
#    to absorb rounding; for a shell the tolerance is the whole acceptance
#    window, so it is wider. Neither merges adjacent shells (>= 100 s/mm2 apart).
BRANGE_TOL = 5
SHELL_TOL = 25

#--- Where the image installs the pipeline scripts; running from here is what
#    identifies a run inside the image.
IMAGE_SCRIPT_DIR = "/opt/scripts"

#--- Environment the image's FSL, ANTs and BLAS are selected by. The Dockerfile
#    sets the same values as ENV (a test keeps the two in step), but Apptainer
#    turns each ENV into 'export VAR="${VAR:-value}"', so a value set on the host
#    wins there. delta-svd.py therefore assigns these outright when it runs in
#    the image: a host FSLDIR or ANTSPATH would otherwise run the host's FSL or
#    ANTs, and a host OPENBLAS_CORETYPE another BLAS kernel, all of which change
#    the metrics. PYTHONNOUSERSITE keeps the Python subprocesses (e.g. fsl_sub)
#    off the host's ~/.local, which Apptainer mounts.
IMAGE_ENVIRONMENT = {
    "FSLDIR": "/opt/conda",
    "FSLOUTPUTTYPE": "NIFTI_GZ",
    "ANTSPATH": "/opt/ants-2.4.3/bin",
    "OPENBLAS_CORETYPE": "Haswell",
    "PYTHONUNBUFFERED": "1",
    "PYTHONNOUSERSITE": "1",
}

#--- Fewer unique directions than the minimum are refused, fewer than the
#    recommended count warned about; see DESIGN_MATRIX_RANK in delta-svd.py.
MIN_DIRECTIONS = 12
RECOMMENDED_DIRECTIONS = 20
