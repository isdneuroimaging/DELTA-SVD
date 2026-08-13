from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import create_html_with_png as chp

DEFAULT_SKEL_MASK = "/opt/scripts/delta-svd_skeletonmask_v1.nii.gz"


def _args(**overrides):
    base = dict(
        id="P01",
        tp=["TP01"],
        skeletonMask=DEFAULT_SKEL_MASK,
        bRange=[800, 1200],
        shells=None,
        RmaskMNI=None,
        hemispheres=False,
        adjustBmaskForFW=True,
        itkThreads=12,
        dwi=["/data/sub01_dwi.nii.gz"],
        bmask=["/data/sub01_brainmask.nii.gz"],
        Emask=[None],
        Rmask=[None],
        function_call="delta-svd.py --dwi sub01_dwi.nii.gz",
        version="9.9.9",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# define_notes() branch matrix

def test_define_notes_none_args_returns_empty_string():
    assert chp.define_notes(None) == ""


def test_define_notes_single_timepoint_default_mask_no_deviation():
    notes = chp.define_notes(_args(tp=["TP01"]))
    assert "cross-sectionally" in notes
    assert "deviates" not in notes


def test_define_notes_multi_timepoint_default_mask_no_deviation():
    notes = chp.define_notes(_args(tp=["TP01", "TP02"]))
    assert "longitudinally" in notes
    assert "2 timepoints" in notes
    assert "deviates" not in notes


def test_define_notes_flags_deviation_when_bmask_not_adjusted():
    notes = chp.define_notes(_args(adjustBmaskForFW=False))
    assert "deviates from the default behaviour" in notes


def test_define_notes_flags_deviation_for_custom_skeleton_mask():
    notes = chp.define_notes(_args(skeletonMask="/custom/mask.nii.gz"))
    assert "deviates from the default behaviour" in notes
    assert "custom" in notes


# ---------------------------------------------------------------------------
# create_html_with_png() smoke test

@pytest.fixture
def tiny_png(tmp_path):
    fn = tmp_path / "a.png"
    fig = plt.figure(figsize=(1, 1))
    plt.plot([0, 1], [0, 1])
    plt.savefig(fn)
    plt.close(fig)
    return str(fn)


@pytest.fixture
def metrics_df():
    return pd.DataFrame({
        "timepoint": ["TP01", "TP01", "TP01"],
        "skeleton": ["skel.nii.gz"] * 3,
        "region": ["intersection"] * 3,
        "voxels": [100, 100, 100],
        "metric": ["PSMD", "MSMD", "MSFW"],
        "value": [0.00045, 0.0007, 0.15],
    })


def test_create_html_with_png_smoke(tmp_path, tiny_png, metrics_df):
    fnHTML = tmp_path / "report.html"
    chp.create_html_with_png(str(fnHTML), [tiny_png], ["Timepoint TP01"], None, metrics_df, _args())

    html = fnHTML.read_text()
    for token in ["PSMD", "MSMD", "MSFW", "DELTA-SVD", "P01"]:
        assert token in html


def test_create_html_with_png_without_args_or_df(tmp_path, tiny_png):
    fnHTML = tmp_path / "report.html"
    chp.create_html_with_png(str(fnHTML), [tiny_png])

    html = fnHTML.read_text()
    assert "DELTA-SVD QC Report" in html


def test_create_html_with_png_omits_the_id_row_when_no_id_was_given(tmp_path, tiny_png):
    # '--id' is optional; str(None) used to reach the tab and table as "None"
    fnHTML = tmp_path / "report.html"
    chp.create_html_with_png(str(fnHTML), [tiny_png], ["Timepoint TP01"], None, None, _args(id=None))

    html = fnHTML.read_text()
    body = html.split("</style>")[1]                    # the CSS mentions "Patient ID"
    assert "<title>DELTA-SVD QC Report</title>" in html
    assert "Patient ID" not in body
    assert "None" not in body
    assert "Skeleton mask" in body                      # the rest of the table survives


def test_create_html_with_png_omits_the_default_itk_thread_count(tmp_path, tiny_png):
    # the validated 12 is the norm and stays out of the table, like every other
    # default -- only a deviation is worth the reader's attention
    fnHTML = tmp_path / "report.html"
    chp.create_html_with_png(str(fnHTML), [tiny_png], None, None, None,
                             _args(tp=["TP01", "TP02"], dwi=["a.nii.gz"] * 2,
                                   bmask=["m.nii.gz"] * 2, Emask=[None] * 2, Rmask=[None] * 2))
    body = fnHTML.read_text().split("</style>")[1]
    assert "ITK threads" not in body
    assert "tag--tip" not in body                       # the default needs no warning


def test_create_html_with_png_flags_a_non_default_itk_thread_count(tmp_path, tiny_png):
    fnHTML = tmp_path / "report.html"
    chp.create_html_with_png(str(fnHTML), [tiny_png], None, None, None,
                             _args(tp=["TP01", "TP02"], itkThreads=6, dwi=["a.nii.gz"] * 2,
                                   bmask=["m.nii.gz"] * 2, Emask=[None] * 2, Rmask=[None] * 2))
    body = fnHTML.read_text().split("</style>")[1]
    assert "6 per registration job" in body
    assert "tag--tip" in body                           # flagged as a deviation


def test_create_html_with_png_omits_itk_threads_cross_sectionally(tmp_path, tiny_png):
    # a cross-sectional run invokes no ANTs at all, so the setting is meaningless
    fnHTML = tmp_path / "report.html"
    chp.create_html_with_png(str(fnHTML), [tiny_png], None, None, None, _args())
    assert "ITK threads" not in fnHTML.read_text().split("</style>")[1]


def test_create_html_with_png_records_the_version(tmp_path, tiny_png):
    # results from different versions must not be pooled, so the report has to
    # say which release produced it
    fnHTML = tmp_path / "report.html"
    chp.create_html_with_png(str(fnHTML), [tiny_png], None, None, None, _args())

    body = fnHTML.read_text().split("</style>")[1]
    assert "<td class=\"key\">DELTA-SVD version</td>" in body
    assert "9.9.9" in body


def test_create_html_with_png_omits_the_version_row_when_unknown(tmp_path, tiny_png):
    # 'version' is read off args defensively: an older caller that does not set
    # it must still produce a report, just without the row
    fnHTML = tmp_path / "report.html"
    args = _args()
    del args.version
    chp.create_html_with_png(str(fnHTML), [tiny_png], None, None, None, args)

    body = fnHTML.read_text().split("</style>")[1]
    assert "DELTA-SVD version" not in body
    assert "Skeleton mask" in body                       # the rest of the table survives


def test_create_html_with_png_omits_the_default_b_value_range(tmp_path, tiny_png):
    fnHTML = tmp_path / "report.html"
    chp.create_html_with_png(str(fnHTML), [tiny_png], None, None, None, _args())

    body = fnHTML.read_text().split("</style>")[1]
    assert "b-value range" not in body


def test_create_html_with_png_flags_a_custom_b_value_range(tmp_path, tiny_png):
    fnHTML = tmp_path / "report.html"
    chp.create_html_with_png(str(fnHTML), [tiny_png], None, None, None,
                             _args(bRange=[900, 1100]))

    body = fnHTML.read_text().split("</style>")[1]
    assert "b-value range" in body
    assert "900" in body and "1100" in body


def test_create_html_with_png_records_requested_shells(tmp_path, tiny_png):
    # '--shells' selects the data as much as '--bRange' does, so a report that
    # showed only the (untouched, default) range would misdescribe the run
    fnHTML = tmp_path / "report.html"
    chp.create_html_with_png(str(fnHTML), [tiny_png], None, None, None,
                             _args(shells=[700, 1000]))

    body = fnHTML.read_text().split("</style>")[1]
    assert "b-value shells" in body
    assert "700, 1000" in body
    assert "b-value range" not in body                   # the default range did not select anything


def test_create_html_with_png_records_the_direction_count(tmp_path, tiny_png):
    # the angular sampling qualifies every metric in the report
    fnHTML = tmp_path / "report.html"
    chp.create_html_with_png(str(fnHTML), [tiny_png], None, None, None,
                             _args(nDirections=[30]))

    body = fnHTML.read_text().split("</style>")[1]
    assert "<td class=\"key\">Diffusion directions</td>" in body
    assert ">30</td>" in body
    assert "tag--tip" not in body                        # nothing to caution about


def test_create_html_with_png_flags_a_low_direction_count(tmp_path, tiny_png):
    # above the hard floor the run goes through, so the caution has to reach
    # whoever reads the results rather than only the log
    fnHTML = tmp_path / "report.html"
    chp.create_html_with_png(str(fnHTML), [tiny_png], None, None, None,
                             _args(nDirections=[15]))

    body = fnHTML.read_text().split("</style>")[1]
    assert "Diffusion directions" in body
    assert "tag--tip" in body
    assert "recommended minimum of 20" in body


def test_create_html_with_png_lists_direction_counts_per_timepoint(tmp_path, tiny_png):
    # timepoints can differ - a single number would hide the weaker one
    fnHTML = tmp_path / "report.html"
    chp.create_html_with_png(str(fnHTML), [tiny_png], None, None, None,
                             _args(tp=["TP01", "TP02"], nDirections=[30, 24],
                                   dwi=["a.nii.gz"] * 2, bmask=["m.nii.gz"] * 2,
                                   Emask=[None] * 2, Rmask=[None] * 2))

    body = fnHTML.read_text().split("</style>")[1]
    assert "30 (TP01), 24 (TP02)" in body


def test_create_html_with_png_omits_the_direction_row_when_unknown(tmp_path, tiny_png):
    # the count is only known when the fit step ran; '--steps' can skip it
    fnHTML = tmp_path / "report.html"
    chp.create_html_with_png(str(fnHTML), [tiny_png], None, None, None, _args())

    body = fnHTML.read_text().split("</style>")[1]
    assert "Diffusion directions" not in body
    assert "Skeleton mask" in body                       # the rest of the table survives


def test_create_html_with_png_keeps_the_id_row_when_an_id_was_given(tmp_path, tiny_png):
    fnHTML = tmp_path / "report.html"
    chp.create_html_with_png(str(fnHTML), [tiny_png], ["Timepoint TP01"], None, None, _args(id="P01"))

    body = fnHTML.read_text().split("</style>")[1]
    assert "<td class=\"key\">Patient ID</td>" in body
    assert "P01" in body
