import create_html_with_png as chp
import delta_svd_constants as c
import delta_svd_gradients
import delta_svd_inputs


# Changing any of these changes the output or the report: update the pin
# deliberately, in the same commit.
def test_constants_pinned_to_their_validated_values():
    assert c.SKELETON_MASK_DEFAULT == "/opt/scripts/delta-svd_skeletonmask_v1.nii.gz"
    assert c.ITK_THREADS_DEFAULT == 12
    assert c.ITERATIONS_DEFAULT == "30x30x8"
    assert c.NUM_REGISTRATIONS_DEFAULT == 3
    assert c.B0_MAX == 5
    assert c.BRANGE_DEFAULT == (800, 1200)
    assert c.BRANGE_TOL == 5
    assert c.SHELL_TOL == 25
    assert c.MIN_DIRECTIONS == 12
    assert c.RECOMMENDED_DIRECTIONS == 20
    assert c.ANTS_TEMPLATE_INIT == "/opt/scripts/FMRIB58_FA_2mm_crop.nii.gz"


def test_pipeline_and_report_share_the_constants(delta_svd):
    for name in ("ITK_THREADS_DEFAULT", "ITERATIONS_DEFAULT", "NUM_REGISTRATIONS_DEFAULT"):
        assert getattr(delta_svd, name) == getattr(c, name)
    for name in ("SKELETON_MASK_DEFAULT", "ITK_THREADS_DEFAULT", "BRANGE_DEFAULT",
                 "BRANGE_TOL", "SHELL_TOL", "ITERATIONS_DEFAULT",
                 "NUM_REGISTRATIONS_DEFAULT", "B0_MAX"):
        assert getattr(delta_svd_inputs, name) == getattr(c, name)
    for name in ("B0_MAX", "BRANGE_DEFAULT", "BRANGE_TOL", "SHELL_TOL", "MIN_DIRECTIONS",
                 "RECOMMENDED_DIRECTIONS"):
        assert getattr(delta_svd_gradients, name) == getattr(c, name)
    for name in ("SKELETON_MASK_DEFAULT", "ITK_THREADS_DEFAULT", "BRANGE_DEFAULT",
                 "BRANGE_TOL", "SHELL_TOL", "RECOMMENDED_DIRECTIONS"):
        assert getattr(chp, name) == getattr(c, name)

