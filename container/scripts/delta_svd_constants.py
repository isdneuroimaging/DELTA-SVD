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

#--- Fewer unique directions than the minimum are refused, fewer than the
#    recommended count warned about; see DESIGN_MATRIX_RANK in delta-svd.py.
MIN_DIRECTIONS = 12
RECOMMENDED_DIRECTIONS = 20
