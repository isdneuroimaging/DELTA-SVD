import create_html_with_png as chp
import delta_svd_constants as c


# The literal values these constants replaced. Changing any of them changes the
# pipeline's output or the report, so a change has to be deliberate: update the
# pin here in the same commit.
def test_constants_pinned_to_their_validated_values():
    assert c.SKELETON_MASK_DEFAULT == "/opt/scripts/delta-svd_skeletonmask_v1.nii.gz"
    assert c.ITK_THREADS_DEFAULT == 12
    assert c.BRANGE_DEFAULT == (800, 1200)
    assert c.BRANGE_TOL == 5
    assert c.SHELL_TOL == 25
    assert c.MIN_DIRECTIONS == 12
    assert c.RECOMMENDED_DIRECTIONS == 20


def test_pipeline_and_report_share_the_constants(delta_svd):
    for name in ("SKELETON_MASK_DEFAULT", "ITK_THREADS_DEFAULT", "BRANGE_DEFAULT",
                 "BRANGE_TOL", "SHELL_TOL", "MIN_DIRECTIONS", "RECOMMENDED_DIRECTIONS"):
        assert getattr(delta_svd, name) == getattr(c, name)
    for name in ("SKELETON_MASK_DEFAULT", "ITK_THREADS_DEFAULT", "BRANGE_DEFAULT",
                 "BRANGE_TOL", "SHELL_TOL", "RECOMMENDED_DIRECTIONS"):
        assert getattr(chp, name) == getattr(c, name)

