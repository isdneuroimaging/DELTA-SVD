#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Defaults shared by delta-svd.py and the QC report (create_html_with_png.py).

The report restates several of these to flag runs that deviate from them, so
they live here, in one place, rather than as literals in both scripts.

Kept free of third-party imports: delta-svd.py imports this module before it
pins the BLAS kernel, which has to happen before numpy is first imported.
"""

#--- The validated skeleton mask, as installed in the image.
SKELETON_MASK_DEFAULT = "/opt/scripts/delta-svd_skeletonmask_v1.nii.gz"

# ITK threads per registration job. This is the one threading quantity that
# reaches the metric values: ITK sums the registration metric and its gradient
# per thread, so a different count sums them in a different order and the last
# bits move. The skeleton amplifies that from there -- 1 thread instead of 12
# shifts delta-PSMD by ~23% on the reference subject. 12 is the value the method
# was validated at; see CONTRIBUTING.md before changing it.
ITK_THREADS_DEFAULT = 12

#--- The default '--bRange': lower and upper limit of the non-zero shell(s).
BRANGE_DEFAULT = (800, 1200)

#--- Scanners report b-values that deviate from the nominal shell (rounding, and
#    cross-terms with the imaging gradients), so a requested limit is met with a
#    tolerance. A range carries slack in its endpoints already, so it only needs
#    enough to absorb rounding; a shell is a point, where the tolerance is the
#    whole acceptance window, so it gets more. Neither can merge adjacent shells,
#    which sit at least 100 s/mm2 apart in practice.
BRANGE_TOL = 5
SHELL_TOL = 25

#--- Unique diffusion directions: fewer than the minimum is refused, fewer than
#    the recommended count is warned about. The rationale is with
#    DESIGN_MATRIX_RANK in delta-svd.py.
MIN_DIRECTIONS = 12
RECOMMENDED_DIRECTIONS = 20
