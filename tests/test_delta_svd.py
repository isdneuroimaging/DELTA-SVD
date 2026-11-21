import datetime
import json
import os
from types import SimpleNamespace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import decompose_tensor, design_matrix, from_lower_triangular
import dipy.reconst.dti as dti


def test_utc_timestamp_uses_rfc3339_z_suffix(delta_svd):
    value = datetime.datetime(2026, 9, 16, 10, 15, 22,
                              tzinfo=datetime.timezone.utc)
    assert delta_svd.utc_timestamp(value) == "2026-09-16T10:15:22Z"


def test_run_manifest_records_run_details_and_outputs(delta_svd, tmp_path, monkeypatch):
    args = SimpleNamespace(
        id="sub-01",
        dwi=["ses-1.nii.gz", "ses-2.nii.gz"],
        function_call="delta-svd.py --dwi ses-1.nii.gz ses-2.nii.gz",
        steps=["fwc", "template", "extract"],
        qc=0,
        tp=["ses-1", "ses-2"],
        voxelOrder=["LAS", "RAS"],
    )
    monkeypatch.setattr(delta_svd, "__source_revision__", "abc1234")
    started = datetime.datetime(2026, 9, 16, 10, 15, 22,
                                tzinfo=datetime.timezone.utc)
    completed = datetime.datetime(2026, 9, 16, 10, 47, 3,
                                  tzinfo=datetime.timezone.utc)
    manifest = tmp_path / "delta-svd_run_manifest.json"

    delta_svd.write_run_manifest(
        str(manifest), args, started, ["delta-svd_results.csv"], completed)

    data = json.loads(manifest.read_text())
    assert data == {
        "manifest_schema_version": 1,
        "pipeline": "DELTA-SVD",
        "pipeline_version": delta_svd.__version__,
        "source_revision": "abc1234",
        "subject_id": "sub-01",
        "processing_mode": "longitudinal",
        "command": args.function_call,
        "started_at": "2026-09-16T10:15:22Z",
        "completed_at": "2026-09-16T10:47:03Z",
        "steps_completed": ["fwc", "template", "extract"],
        "qc_mode": 0,
        "outputs": ["delta-svd_results.csv"],
        "voxel_order": {"ses-1": "LAS", "ses-2": "RAS"},
    }
    assert not (tmp_path / "delta-svd_run_manifest.json.tmp").exists()


# ---------------------------------------------------------------------------
# ANTs parallelism planner

def test_plan_ants_parallelism_itk_threads_never_vary(delta_svd):
    # The ITK thread count is the one planner output that moves the metrics, so it
    # must depend on --itkThreads alone, not on the cores or timepoints.
    varied = [
        (nTP, budget, para, delta_svd.plan_ants_parallelism(nTP, budget, para)[1])
        for nTP in (1, 2, 3, 4, 6, 12)
        for budget in (1, 2, 8, 12, 13, 16, 24, 32, 36, 48, 64, 96)
        for para in (None, 1, 2, 3, 12)
    ]
    assert [v for v in varied if v[3] != delta_svd.ITK_THREADS_DEFAULT] == []


def test_plan_ants_parallelism_fills_the_budget(delta_svd):
    # 'para' spends whatever cores are there (ceil), but never oversubscribes past
    # 1.5 threads per core -- which is why 13 cores stays at one job rather than
    # jumping to two (24 threads on 13 cores, 1.85x).
    expected = {12: 1, 13: 1, 16: 2, 24: 2, 32: 3, 36: 3, 48: 4, 64: 6}
    got = {b: delta_svd.plan_ants_parallelism(99, b)[0] for b in expected}
    assert got == expected


def test_plan_ants_parallelism_single_job_is_serial(delta_svd):
    # one job -> serial path ('-c 0'), never '-c 2 -j 1' (pexec aborts on 1 job).
    # The branch selects the execution mode only: itk is unchanged either way.
    para, itk, control = delta_svd.plan_ants_parallelism(2, 1)
    assert (para, itk, control) == (1, delta_svd.ITK_THREADS_DEFAULT, "-c 0")


def test_plan_ants_parallelism_caps_jobs_at_timepoints(delta_svd):
    # 2 timepoints -> at most 2 jobs however large the budget; there is no third
    # registration to run in parallel
    para, itk, control = delta_svd.plan_ants_parallelism(2, 64)
    assert (para, itk) == (2, delta_svd.ITK_THREADS_DEFAULT)
    assert control == "-c 2 -j 2"


def test_plan_ants_parallelism_override_is_not_clamped_to_the_budget(delta_svd):
    # explicit --para 1 forces the serial path
    assert delta_svd.plan_ants_parallelism(6, 12, paraOverride=1)[2] == "-c 0"
    # an override above the budget is honoured (deliberate oversubscription is a
    # valid choice and costs only runtime), but never exceeds the timepoint count
    para, _, control = delta_svd.plan_ants_parallelism(3, 4, paraOverride=3)
    assert para == 3 and control == "-c 2 -j 3"
    assert delta_svd.plan_ants_parallelism(3, 4, paraOverride=99)[0] == 3


def test_plan_ants_parallelism_honours_an_explicit_itk_thread_count(delta_svd):
    para, itk, _ = delta_svd.plan_ants_parallelism(4, 24, itkThreads=6)
    assert itk == 6
    assert para == 4          # 24 cores / 6 threads = 4 jobs, capped by 4 timepoints


# ---------------------------------------------------------------------------
# --itkThreads resolution and export

def test_blas_thread_pools_are_pinned_to_one(delta_svd):
    # no thread count in the numerics may vary with the machine; assigned, not
    # setdefault, so a host-forwarded value cannot reach BLAS either
    assert delta_svd.BLAS_THREADS == 1
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        assert os.environ[var] == "1", var


def test_itk_thread_count_is_exported_to_every_ants_call(delta_svd):
    # assigned, not setdefault: a value forwarded in from the host must not be
    # able to change the metrics (Apptainer passes the host environment through)
    assert os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] == str(delta_svd.ITK_THREADS)


# ---------------------------------------------------------------------------
# integrate_masks(): every voxel count in the results CSV and every skeleton mask
# extract_stats() globs for. Each test pins one distinct branch.

def _vol(values):
    """A (n, 1, 1) volume - a one-dimensional 'brain' is enough for every branch
    here, and keeps the expected voxel counts readable."""
    return np.asarray(values, dtype=float).reshape(-1, 1, 1)


def _tbss_tree(tmp_path, skeleton, bmasks, emask=None, rois=None, mni=None):
    """Lay out the minimal TBSS tree integrate_masks() reads: one skeletonised
    brain mask per timepoint, plus the optional exclusion / ROI inputs."""
    stats = tmp_path / "TBSS" / "stats"
    stats.mkdir(parents=True)
    affine = np.eye(4)

    def save(path, arr):
        nib.save(nib.Nifti1Image(_vol(arr), affine), str(path))

    skelMask = tmp_path / "skel.nii.gz"
    save(skelMask, skeleton)

    dirTP = []
    for tp, arr in bmasks.items():
        (tmp_path / tp).mkdir()
        dirTP.append(str(tmp_path / tp))
        save(stats / f"all_{tp}_bmask_skeletonised.nii.gz", arr)

    if emask is not None:
        save(stats / "all_E-MASK_skeletonised.nii.gz", emask)
    for label, arr in (rois or {}).items():
        save(stats / f"all_ROI-{label}_skeletonised.nii.gz", arr)

    fnROI_MNI = None
    if mni is not None:
        fnROI_MNI = str(tmp_path / "roi_mni.nii.gz")
        save(fnROI_MNI, mni)

    return dict(dirTP=dirTP, dirTBSS=str(tmp_path / "TBSS"),
                skelMask=str(skelMask), fnROI_MNI=fnROI_MNI), stats


def _counts(df):
    return dict(zip(df["region"], df["voxels"]))


def test_integrate_masks_single_timepoint_drops_interpolated_voxels(delta_svd, tmp_path):
    # A skeletonised brain mask is resampled, so its edge voxels come back with
    # 0 < value < 1 - foreground blended with background. Those must be dropped
    # ('bmask < 1'), not rounded in, or the skeleton spills past the brain.
    kwargs, stats = _tbss_tree(
        tmp_path,
        skeleton=[1, 1, 1, 1, 1, 0],
        bmasks={"TP01": [1, 1, 1, 1, 0.5, 1]},
    )

    df = delta_svd.integrate_masks(**kwargs)

    assert _counts(df) == {"total": 5, "intersection": 4}
    # a single timepoint is named, not aggregated as 'all'
    assert list(df["timepoint"]) == ["TP01", "TP01"]
    assert list(df["skeleton"]) == ["skel.nii.gz"] * 2
    # voxel-count rows carry no metric; extract_stats() appends those separately
    assert list(df["metric"]) == ["NA"] * 2 and list(df["value"]) == ["NaN"] * 2

    saved = nib.load(str(stats / "skel_intersection.nii.gz")).get_fdata()
    assert saved.ravel().tolist() == [1, 1, 1, 1, 0, 0]


def test_integrate_masks_multi_timepoint_intersects_and_reports_set_difference(delta_svd, tmp_path):
    # With several timepoints the analysis runs on the voxels common to all of
    # them, so a longitudinal change cannot come from a changing denominator.
    # The per-timepoint 'set_difference' rows say how much each one gave up.
    kwargs, stats = _tbss_tree(
        tmp_path,
        skeleton=[1, 1, 1, 1, 1, 0],
        bmasks={"TP01": [1, 1, 1, 1, 1, 1],       # adjusted -> 5 voxels
                "TP02": [1, 1, 1, 0.4, 1, 1]},    # adjusted -> 4 voxels
    )

    df = delta_svd.integrate_masks(**kwargs)

    assert list(df["region"]) == ["total", "intersection",
                                 "set_difference", "set_difference"]
    assert list(df["voxels"]) == [5, 4, 1, 0]
    assert list(df["timepoint"]) == ["all", "all", "TP01", "TP02"]

    saved = nib.load(str(stats / "skel_intersection.nii.gz")).get_fdata()
    assert saved.ravel().tolist() == [1, 1, 1, 0, 1, 0]


def test_integrate_masks_excludes_emask_and_labels_it_for_qc(delta_svd, tmp_path):
    # The exclusion mask (e.g. an infarct) is cut out of the skeleton. The 0.05
    # threshold also excludes interpolated fringe voxels, and the QC image labels 2
    # only where exclusion removed something, i.e. inside the intersection.
    kwargs, stats = _tbss_tree(
        tmp_path,
        skeleton=[1, 1, 1, 1, 1, 0],
        bmasks={"TP01": [1, 1, 1, 1, 0.5, 1]},    # intersection -> [1,1,1,1,0,0]
        emask=[0, 0.04, 0.5, 0, 0.9, 0],
    )

    df = delta_svd.integrate_masks(**kwargs)

    assert _counts(df) == {"total": 5, "intersection": 4, "intersection_Emask": 3}

    final = nib.load(str(stats / "skel_intersection_Emask.nii.gz")).get_fdata()
    assert final.ravel().tolist() == [1, 1, 0, 1, 0, 0]

    qc = nib.load(str(stats / "skel_intersection_Emask-as-label2.nii.gz")).get_fdata()
    # index 1 (0.04) is under threshold and stays skeleton; index 2 is excluded
    # and labelled 2; index 4 is excluded but was never in the intersection, so
    # it must stay 0 rather than show up as a phantom lesion in the QC image.
    assert qc.ravel().tolist() == [1, 1, 2, 1, 0, 0]


def test_integrate_masks_inserts_the_background_roi_before_the_named_rois(delta_svd, tmp_path):
    # Rmask-00 is the complement of the subject ROIs. It is inserted ahead of the
    # named ROI rows by index arithmetic, which can easily slip a row out of place.
    kwargs, stats = _tbss_tree(
        tmp_path,
        skeleton=[1, 1, 1, 1, 1, 0],
        bmasks={"TP01": [1, 1, 1, 1, 0.5, 1]},    # intersection -> [1,1,1,1,0,0]
        rois={"01": [1, 0, 0, 0, 0, 0],
              "02": [0, 0.5, 0.04, 0, 0.9, 0]},
    )

    df = delta_svd.integrate_masks(**kwargs)

    assert list(df["region"]) == [
        "total", "intersection",
        "intersection_Rmask-00", "intersection_Rmask-01", "intersection_Rmask-02",
    ]
    assert list(df["voxels"]) == [5, 4, 2, 1, 1]
    # the ROIs and the background partition the intersection exactly
    assert sum(list(df["voxels"])[2:]) == df["voxels"].iloc[1]

    # index 2 (0.04) is below the 0.05 threshold and index 4 is outside the
    # intersection, so ROI-02 keeps only index 1
    roi02 = nib.load(str(stats / "skel_intersection_Rmask-02.nii.gz")).get_fdata()
    assert roi02.ravel().tolist() == [0, 1, 0, 0, 0, 0]

    background = nib.load(str(stats / "skel_intersection_Rmask-00.nii.gz")).get_fdata()
    assert background.ravel().tolist() == [0, 0, 1, 1, 0, 0]

    # no merged label map: overlapping ROIs would be hidden in it
    assert not (stats / "skel_intersection_Rmask.nii.gz").exists()


def test_integrate_masks_ignores_a_timepoint_named_like_an_roi(delta_svd, tmp_path):
    # timepoint 'ROI-1' writes 'all_ROI-1_FW_skeletonised', which the ROI
    # pattern also matched, so the extract step crashed reading its label
    kwargs, stats = _tbss_tree(
        tmp_path, skeleton=[1, 1, 1, 1, 1, 0], bmasks={"ROI-1": [1, 1, 1, 1, 1, 1]})
    nib.save(nib.Nifti1Image(_vol([0] * 6), np.eye(4)), str(stats / "all_ROI-1_FW_skeletonised.nii.gz"))

    df = delta_svd.integrate_masks(**kwargs)

    assert list(df["region"]) == ["total", "intersection"]


def test_integrate_masks_roi_suffixes_follow_the_emask_exclusion(delta_svd, tmp_path):
    # Once an exclusion mask is present the ROI outputs have to hang off the
    # *excluded* skeleton, both in name and in content - extract_stats() globs
    # '*_Rmask-*' and would otherwise mix pre- and post-exclusion regions.
    kwargs, stats = _tbss_tree(
        tmp_path,
        skeleton=[1, 1, 1, 1, 1, 0],
        bmasks={"TP01": [1, 1, 1, 1, 0.5, 1]},
        emask=[0, 1, 0, 0, 0, 0],                 # removes index 1
        rois={"01": [1, 1, 0, 0, 0, 0]},
    )

    df = delta_svd.integrate_masks(**kwargs)

    assert list(df["region"]) == ["total", "intersection", "intersection_Emask",
                                 "intersection_Emask_Rmask-00",
                                 "intersection_Emask_Rmask-01"]
    roi01 = nib.load(str(stats / "skel_intersection_Emask_Rmask-01.nii.gz")).get_fdata()
    assert roi01.ravel().tolist() == [1, 0, 0, 0, 0, 0]      # index 1 was excluded


def test_prepare_qc_deprojects_the_roi_maps_integrate_masks_writes(delta_svd, tmp_path, monkeypatch):
    # prepare_qc() finds the ROI maps by file name, so a renamed output in
    # integrate_masks() would silently drop them from '--qc 2'. Runs the two
    # together rather than against hand-named placeholder files.
    import create_html_with_png
    import create_qc_image
    import delta_svd_qc

    kwargs, stats = _tbss_tree(
        tmp_path,
        skeleton=[1, 1, 1, 1, 1, 0],
        bmasks={"TP01": [1, 1, 1, 1, 1, 1]},
        emask=[0, 0, 0, 0, 1, 0],
        rois={"01": [1, 1, 0, 0, 0, 0], "02": [0, 1, 1, 0, 0, 0]},   # overlap at index 1
        mni=[0, 0, 1, 1, 2, 2],
    )
    delta_svd.integrate_masks(**kwargs)

    for name in ["fwc_wls_dti_FA_05.nii.gz", "wls_dti_FA.nii.gz", "wls_dti_MD.nii.gz"]:
        (tmp_path / "TP01" / name).touch()
    dirQC = tmp_path / "delta-svd_qc"
    dirQC.mkdir()
    deprojected = []
    monkeypatch.setattr(delta_svd_qc, "deproject_to_native",
                        lambda fn, *a, relabel=False: deprojected.append((os.path.basename(fn), relabel)))
    monkeypatch.setattr(create_qc_image, "create_qc_image",
                        lambda *a, animate=True, **k: "x.png" if animate else [])
    monkeypatch.setattr(create_html_with_png, "create_html_with_png", lambda *a, **k: None)

    delta_svd_qc.prepare_qc(str(dirQC), str(tmp_path / "qc.html"), kwargs["skelMask"],
                            kwargs["dirTBSS"], str(tmp_path / "template"), kwargs["dirTP"],
                            "results.csv", SimpleNamespace(qc=2, dirOutput=str(tmp_path)))

    # every binary ROI map as written, without relabelling; only the merged MNI map is relabelled
    assert deprojected == [
        ("skel_intersection_Emask-as-label2.nii.gz", False),
        ("skel_intersection_Emask_Rmask-00.nii.gz", False),
        ("skel_intersection_Emask_Rmask-01.nii.gz", False),
        ("skel_intersection_Emask_Rmask-02.nii.gz", False),
        ("skel_intersection_Emask_RmaskMNI.nii.gz", True),
    ]
    # the overlap survives in the separate maps
    roi01 = nib.load(str(stats / "skel_intersection_Emask_Rmask-01.nii.gz")).get_fdata()
    roi02 = nib.load(str(stats / "skel_intersection_Emask_Rmask-02.nii.gz")).get_fdata()
    assert roi01.ravel()[1] == roi02.ravel()[1] == 1


def test_integrate_masks_relabels_mni_rois_including_the_background(delta_svd, tmp_path):
    # Each label of an atlas file becomes its own region, label 0 ('the rest of
    # the skeleton') included.
    kwargs, stats = _tbss_tree(
        tmp_path,
        skeleton=[1, 1, 1, 1, 1, 0],
        bmasks={"TP01": [1, 1, 1, 1, 0.5, 1]},    # intersection -> [1,1,1,1,0,0]
        mni=[0, 0, 1, 1, 2, 2],
    )

    df = delta_svd.integrate_masks(**kwargs)

    assert list(df["region"]) == ["total", "intersection",
                                 "intersection_RmaskMNI-00",
                                 "intersection_RmaskMNI-01",
                                 "intersection_RmaskMNI-02"]
    # label 2 lies entirely outside the intersection -> an honest zero, not a
    # missing row
    assert list(df["voxels"]) == [5, 4, 2, 2, 0]

    merged = nib.load(str(stats / "skel_intersection_RmaskMNI.nii.gz")).get_fdata()
    # merged after intersecting with the final skeleton, so it differs from the
    # atlas that went in: label 2 is gone and label 0 is not written
    assert merged.ravel().tolist() == [0, 0, 1, 1, 0, 0]


def test_integrate_masks_preserves_wide_mni_labels(delta_svd, tmp_path):
    kwargs, stats = _tbss_tree(
        tmp_path,
        skeleton=[1, 1, 1],
        bmasks={"TP01": [1, 1, 1]},
        mni=[0, 255, 256],
    )

    df = delta_svd.integrate_masks(**kwargs)

    assert list(df["region"][-3:]) == [
        "intersection_RmaskMNI-00",
        "intersection_RmaskMNI-255",
        "intersection_RmaskMNI-256",
    ]
    merged = nib.load(str(stats / "skel_intersection_RmaskMNI.nii.gz"))
    assert merged.get_data_dtype() == np.dtype("uint16")
    assert merged.get_fdata().ravel().tolist() == [0, 255, 256]


def test_integrate_and_extract_preserve_wide_dwi_labels(delta_svd, tmp_path):
    kwargs, stats = _tbss_tree(
        tmp_path,
        skeleton=[1, 1, 1],
        bmasks={"TP01": [1, 1, 1]},
        rois={"255": [0, 1, 0], "256": [0, 0, 1]},
    )

    df = delta_svd.integrate_masks(**kwargs)
    assert list(df["region"][-3:]) == [
        "intersection_Rmask-00", "intersection_Rmask-255", "intersection_Rmask-256",
    ]

    nib.save(nib.Nifti1Image(_vol([1.0, 2.0, 3.0]), np.eye(4)),
             str(stats / "all_TP01_MD_skeletonised.nii.gz"))
    extracted = delta_svd.extract_stats(
        dirTP=str(tmp_path / "TP01"), dirTBSS=str(tmp_path / "TBSS"),
        fnNonFA={"MD": "unused"}, skelMask=str(tmp_path / "skel.nii.gz"),
    )
    assert {"intersection_Rmask-255", "intersection_Rmask-256"} <= set(extracted["region"])


def test_integrate_masks_splits_hemispheres_along_the_first_axis(delta_svd, tmp_path):
    # The hemisphere split is an index cut at shape[0] // 2; 'LH' is the half left
    # after the low indices are zeroed. Nothing else would notice a swap.
    kwargs, stats = _tbss_tree(
        tmp_path,
        skeleton=[1, 1, 1, 1, 1, 0],
        bmasks={"TP01": [1, 1, 1, 1, 1, 1]},      # intersection -> [1,1,1,1,1,0]
    )

    df = delta_svd.integrate_masks(analyseHemispheres=True, **kwargs)

    counts = _counts(df)
    assert counts["intersection_LH"] == 2
    assert counts["intersection_RH"] == 3
    assert counts["intersection_LH"] + counts["intersection_RH"] == counts["intersection"]

    lh = nib.load(str(stats / "skel_intersection_LH.nii.gz")).get_fdata()
    assert lh.ravel().tolist() == [0, 0, 0, 1, 1, 0]


@pytest.mark.parametrize("bmasks", [
    {"TP01": [1, 1, 1, 1, 1]},                              # cross-sectional
    {"TP01": [1, 1, 1, 1, 1], "TP02": [1, 1, 1, 1, 1]},     # longitudinal
])
def test_integrate_masks_binarises_a_non_binary_skeleton_mask(delta_svd, tmp_path, bmasks):
    # Fractional values used to be truncated by the uint8 write yet counted in
    # full, and only cross-sectionally (np.all() binarises the longitudinal
    # path). Both modes must agree, and the count must match what was written.
    kwargs, stats = _tbss_tree(tmp_path, skeleton=[1, 1, 0.5, 0.5, 0], bmasks=bmasks)

    df = delta_svd.integrate_masks(**kwargs)

    saved = nib.load(str(stats / "skel_intersection.nii.gz")).get_fdata()
    assert saved.ravel().tolist() == [1, 1, 1, 1, 0]
    assert _counts(df)["intersection"] == int(np.count_nonzero(saved)) == 4


# ---------------------------------------------------------------------------
# extract_stats() metric naming
#
# The emitted names are part of the published output contract, so they are
# pinned exactly: a peak-width statistic is only defined for MD, and the
# free-water and mean-diffusivity means carry the 'MS' prefix.

def test_extract_stats_emits_correct_metric_names(delta_svd, tmp_path):
    dirTBSS = tmp_path / "TBSS"
    stats_dir = dirTBSS / "stats"
    stats_dir.mkdir(parents=True)
    affine = np.eye(4)

    mask = np.array([[[1, 1], [0, 1]]], dtype=float)
    nib.save(nib.Nifti1Image(mask, affine), str(stats_dir / "skel_intersection.nii.gz"))

    fw = np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=float)
    md = np.array([[[0.0005, 0.0007], [0.0006, 0.0009]]], dtype=float)
    nib.save(nib.Nifti1Image(fw, affine), str(stats_dir / "all_TP01_FW_skeletonised.nii.gz"))
    nib.save(nib.Nifti1Image(md, affine), str(stats_dir / "all_TP01_MD_skeletonised.nii.gz"))

    df = delta_svd.extract_stats(
        dirTP="/some/path/TP01",
        dirTBSS=str(dirTBSS),
        fnNonFA={"FW": "unused", "MD": "unused"},
        skelMask="skel.nii.gz",
    )

    assert set(df["metric"]) == {"PSMD", "MSMD", "MSFW"}


def test_extract_stats_loads_each_map_once_per_timepoint(delta_svd, tmp_path, monkeypatch):
    # many regions, but the skeletonised maps are the same for all of them
    stats_dir = tmp_path / "TBSS" / "stats"
    stats_dir.mkdir(parents=True)
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(np.ones((1, 2, 2)), affine), str(stats_dir / "skel_intersection.nii.gz"))
    for roi in range(5):
        nib.save(nib.Nifti1Image(np.ones((1, 2, 2)), affine),
                 str(stats_dir / f"skel_intersection_RmaskMNI-{roi:02d}.nii.gz"))
    for mapName in ("FW", "MD"):
        nib.save(nib.Nifti1Image(np.full((1, 2, 2), 0.5), affine),
                 str(stats_dir / f"all_TP01_{mapName}_skeletonised.nii.gz"))
    loads = []
    realLoad = delta_svd.nib.load
    monkeypatch.setattr(delta_svd.nib, "load", lambda fn, *a, **k: loads.append(os.path.basename(fn)) or realLoad(fn, *a, **k))

    df = delta_svd.extract_stats(dirTP="/some/path/TP01", dirTBSS=str(tmp_path / "TBSS"),
                                 fnNonFA={"FW": "unused", "MD": "unused"}, skelMask="skel.nii.gz")

    assert loads.count("all_TP01_FW_skeletonised.nii.gz") == 1
    assert loads.count("all_TP01_MD_skeletonised.nii.gz") == 1
    assert len(df) == 6 * 3                    # 6 regions x (MSFW, PSMD, MSMD)
    assert list(df["metric"][:3]) == ["MSFW", "PSMD", "MSMD"]   # row order unchanged


# PSMD is defined as the 95th minus the 5th percentile of MD on the skeleton.
# The name test above would still pass if that arithmetic were wrong (an IQR, a
# standard deviation, the wrong percentiles, or the difference taken the wrong
# way round), so pin the number itself on an array whose percentiles are exact.
def test_extract_stats_psmd_is_p95_minus_p5(delta_svd, tmp_path):
    dirTBSS = tmp_path / "TBSS"
    stats_dir = dirTBSS / "stats"
    stats_dir.mkdir(parents=True)
    affine = np.eye(4)

    # 21 in-mask samples so numpy's linear interpolation lands exactly on one:
    # p5 -> index (21-1)*0.05 = 1, p95 -> index 19. PSMD is therefore 18e-4.
    mdIn = np.arange(21) * 1e-4
    # The two out-of-mask voxels carry a CSF-like value that would blow up p95
    # if the ROI masking were dropped.
    mask = np.array([0.0] + [1.0] * 21 + [0.0]).reshape(-1, 1, 1)
    md = np.concatenate(([9e-3], mdIn, [9e-3])).reshape(-1, 1, 1)

    nib.save(nib.Nifti1Image(mask, affine), str(stats_dir / "skel_intersection.nii.gz"))
    nib.save(nib.Nifti1Image(md, affine), str(stats_dir / "all_TP01_MD_skeletonised.nii.gz"))

    df = delta_svd.extract_stats(
        dirTP="/some/path/TP01",
        dirTBSS=str(dirTBSS),
        fnNonFA={"MD": "unused"},
        skelMask="skel.nii.gz",
    )

    row = df.set_index("metric")
    assert row.loc["PSMD", "voxels"] == 21
    assert row.loc["PSMD", "value"] == pytest.approx(18e-4, rel=1e-12)
    assert row.loc["MSMD", "value"] == pytest.approx(mdIn.mean(), rel=1e-12)


# ---------------------------------------------------------------------------
# Tensor fitting (wls_fit_dti / wls_iter_dti)

def test_wls_fit_dti_recovers_known_tensor(delta_svd):
    bvecs = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.7071, 0.7071, 0],
            [0.7071, 0, 0.7071],
            [0, 0.7071, 0.7071],
        ],
        dtype=float,
    )
    bvals = np.array([0, 0, 1000, 1000, 1000, 1000, 1000, 1000], dtype=float)
    gtab = gradient_table(bvals, bvecs=bvecs)
    W = design_matrix(gtab)

    evals_true = np.array([1.7e-3, 0.3e-3, 0.3e-3])
    D = np.diag(evals_true)
    S0 = 1000.0
    signal = np.array([S0 * np.exp(-b * g.dot(D).dot(g)) for b, g in zip(bvals, bvecs)])

    data = signal.reshape(1, 1, 1, -1)
    mask = np.ones((1, 1, 1), dtype=bool)

    fw_params = delta_svd.wls_fit_dti(W, data, mask=mask)
    evals, _ = decompose_tensor(from_lower_triangular(fw_params))

    FA = dti.fractional_anisotropy(evals).ravel()[0]
    MD = dti.mean_diffusivity(evals).ravel()[0]

    assert FA == pytest.approx(dti.fractional_anisotropy(evals_true), abs=1e-4)
    assert MD == pytest.approx(np.mean(evals_true), abs=1e-8)


# ---------------------------------------------------------------------------
# Free-water two-tensor fitting (wls_fit_tensor_fw / wls_iter_fw)
#
# The output row is [Dxx, Dxy, Dyy, Dxz, Dyz, Dzz, log(S0), f, dF2]: f is written
# as wls_dti_FW.nii.gz, dF2 is the residual difference. The estimator is
# regularised towards MDm and biased, so these tests pin structure and
# qualitative behaviour, not f == f_true.

@pytest.fixture(scope="module")
def _fw_gtab():
    bvecs = np.array(
        [
            [0, 0, 0], [0, 0, 0],
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [0.7071, 0.7071, 0], [0.7071, 0, 0.7071], [0, 0.7071, 0.7071],
            [-0.7071, 0.7071, 0], [-0.7071, 0, 0.7071], [0, -0.7071, 0.7071],
            [0.5774, 0.5774, 0.5774], [-0.5774, 0.5774, 0.5774],
            [0.5774, -0.5774, 0.5774], [0.5774, 0.5774, -0.5774],
        ],
        dtype=float,
    )
    bvals = np.array([0, 0] + [1000] * 13, dtype=float)
    return gradient_table(bvals, bvecs=bvecs), bvals, bvecs


def _two_compartment_signal(bvals, bvecs, fFree, evals=(1.6e-3, 0.4e-3, 0.4e-3),
                            S0=1000.0, Diso=3e-3):
    """Signal of a voxel that is a fFree mixture of isotropic free water with a
    tissue tensor - the model wls_fit_tensor_fw() is supposed to invert."""
    D = np.diag(evals)
    tissue = np.array([np.exp(-b * g.dot(D).dot(g)) for b, g in zip(bvals, bvecs)])
    water = np.exp(-bvals * Diso)
    return S0 * ((1 - fFree) * tissue + fFree * water)


def _fit_fw(delta_svd, gtab, bvals, bvecs, fFree, MDm=0.0006, mdreg=2.0e-3):
    W = design_matrix(gtab)
    data = _two_compartment_signal(bvals, bvecs, fFree).reshape(1, 1, 1, -1)
    mask = np.ones((1, 1, 1), dtype=bool)
    # MD0 comes from the uncorrected single-tensor fit, exactly as
    # free_water_correction() feeds it in.
    dtiParams = delta_svd.wls_fit_dti(W, data, mask=mask)
    MD0 = dti.mean_diffusivity(decompose_tensor(from_lower_triangular(dtiParams))[0])
    S0 = np.mean(data[..., gtab.b0s_mask], axis=-1)
    params = delta_svd.wls_fit_tensor_fw(W, data, MD0, S0, Diso=3e-3, mask=mask,
                                         min_signal=1.0e-6, piterations=2,
                                         mdreg=mdreg, MDm=MDm)
    evals, _ = decompose_tensor(from_lower_triangular(params))
    return params[0, 0, 0], MD0.ravel()[0], dti.mean_diffusivity(evals).ravel()[0]


def test_wls_fit_tensor_fw_free_water_fraction_tracks_contamination(delta_svd, _fw_gtab):
    # The one thing the free-water map has to get right: more free water in the
    # voxel must mean a larger f. Same tissue tensor throughout, so nothing but
    # the contamination differs.
    gtab, bvals, bvecs = _fw_gtab
    fEst = [_fit_fw(delta_svd, gtab, bvals, bvecs, f)[0][7] for f in (0.0, 0.2, 0.4, 0.6)]

    assert all(0.0 <= f <= 0.98 for f in fEst)          # 0.98 is the sampler's cap
    assert fEst == sorted(fEst) and len(set(fEst)) == 4


def test_wls_fit_tensor_fw_removes_free_water_from_the_tissue_md(delta_svd, _fw_gtab):
    # This is what separates the two-tensor fit from wls_fit_dti: free water
    # inflates the single-tensor MD, and the corrected tissue MD must come back
    # down to the MDm the fit is regularised towards.
    gtab, bvals, bvecs = _fw_gtab
    for fFree in (0.2, 0.4):
        params, MD0, MD1 = _fit_fw(delta_svd, gtab, bvals, bvecs, fFree, MDm=0.0006)
        assert MD0 > 0.0006                              # contaminated, uncorrected
        assert MD1 == pytest.approx(0.0006, rel=0.01)
        assert MD1 < MD0


def test_wls_fit_tensor_fw_follows_the_mdm_target(delta_svd, _fw_gtab):
    # MDm is a free parameter of the fit (free_water_correction() passes 0.0006);
    # if the MD-matching sample selection broke, the tissue MD would stop
    # tracking it.
    gtab, bvals, bvecs = _fw_gtab
    _, _, mdLow = _fit_fw(delta_svd, gtab, bvals, bvecs, 0.3, MDm=0.0004)
    _, _, mdHigh = _fit_fw(delta_svd, gtab, bvals, bvecs, 0.3, MDm=0.0006)
    assert mdLow == pytest.approx(0.0004, rel=0.05)
    assert mdHigh == pytest.approx(0.0006, rel=0.05)
    assert mdLow < mdHigh


def test_wls_fit_tensor_fw_declares_csf_voxels_fully_free_water(delta_svd, _fw_gtab):
    # A voxel with single-tensor MD >= mdreg is CSF: f = 1 and a null tensor, so
    # FA is 0, which the downstream 'FA==0 -> 0.05' fixup keys off.
    gtab, bvals, bvecs = _fw_gtab
    W = design_matrix(gtab)
    data = np.ones((2, 1, 1, len(bvals))) * 500.0
    md = np.array([[[1.0e-3]], [[2.5e-3]]])              # below / above mdreg
    S0 = np.ones((2, 1, 1)) * 500.0

    params = delta_svd.wls_fit_tensor_fw(W, data, md, S0, mask=np.ones((2, 1, 1), bool),
                                         mdreg=2.0e-3)

    assert params.shape == (2, 1, 1, 9)
    assert params[1, 0, 0].tolist() == [0, 0, 0, 0, 0, 0, 0, 1.0, 0]
    assert params[0, 0, 0, 7] != 1.0                     # sub-threshold voxel was fitted


def test_wls_fit_tensor_fw_leaves_signalless_voxels_at_zero(delta_svd, _fw_gtab):
    # Below min_signal the fit is skipped: on noise the fitted log(S0) runs away
    # (~20 for a 1e-9 signal). Unlike CSF, background must not come out as f = 1.
    gtab, bvals, bvecs = _fw_gtab
    W = design_matrix(gtab)
    # voxel 0 is exactly zero, voxel 1 sits just under min_signal; both have an
    # MD below mdreg, so only the low-signal guard can zero them
    data = np.stack([np.zeros(len(bvals)),
                     np.full(len(bvals), 1.0e-9)]).reshape(2, 1, 1, -1)
    md = np.array([[[1.0e-3]], [[1.0e-3]]])
    S0 = np.array([[[0.0]], [[1.0e-9]]])

    params = delta_svd.wls_fit_tensor_fw(W, data, md, S0, min_signal=1.0e-6,
                                         mask=np.ones((2, 1, 1), bool))

    assert not params.any()


def test_wls_fit_tensor_fw_respects_the_mask(delta_svd, _fw_gtab):
    gtab, bvals, bvecs = _fw_gtab
    W = design_matrix(gtab)
    data = np.ones((2, 1, 1, len(bvals))) * 500.0
    md = np.array([[[1.0e-3]], [[1.0e-3]]])
    S0 = np.ones((2, 1, 1)) * 500.0

    params = delta_svd.wls_fit_tensor_fw(W, data, md, S0,
                                         mask=np.array([[[True]], [[False]]]))
    assert params[0, 0, 0].any()
    assert not params[1, 0, 0].any()

    with pytest.raises(ValueError):
        delta_svd.wls_fit_tensor_fw(W, data, md, S0, mask=np.ones((3, 1, 1), bool))


# ---------------------------------------------------------------------------
# Parallel driver for the vendored fits (fit_voxelwise)
#
# What matters is bit-identity with the serial loop, not approximate agreement:
# a 1-ULP difference in the FA maps has been observed to move delta-PSMD by
# percent. Hence the raw-byte comparisons against the vendored fit itself.

@pytest.fixture(scope="module")
def _fw_volume(_fw_gtab):
    """A small multi-slab volume: 11 along axis 0 so the slab edges do not divide
    it evenly, with masked-out slices and a CSF slice to cover the fit's branches."""
    gtab, bvals, bvecs = _fw_gtab
    rng = np.random.default_rng(0)
    nx, ny, nz = 11, 3, 2
    data = np.empty((nx, ny, nz, len(bvals)))
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                data[ix, iy, iz] = _two_compartment_signal(
                    bvals, bvecs, fFree=rng.uniform(0.0, 0.6))
    mask = np.ones((nx, ny, nz), dtype=bool)
    mask[3] = False                                      # a wholly empty slab
    mask[7, 1] = False                                   # a partly empty one
    md = np.full((nx, ny, nz), 1.0e-3)
    md[5] = 2.5e-3                                       # above mdreg: CSF short-cut
    S0 = np.mean(data[..., gtab.b0s_mask], axis=-1)
    return design_matrix(gtab), data, md, S0, mask


@pytest.mark.parametrize("nproc", [2, 3, 4, 32])          # 32 > 11 slabs available
def test_fit_voxelwise_is_bit_identical_to_the_serial_fw_fit(delta_svd, _fw_volume, nproc):
    W, data, md, S0, mask = _fw_volume
    kwargs = dict(Diso=3e-3, min_signal=1.0e-6, piterations=2,
                  mdreg=2.0e-3, MDm=0.0006)

    serial = delta_svd.wls_fit_tensor_fw(W, data, md, S0, mask=mask, **kwargs)
    parallel = delta_svd.fit_voxelwise(delta_svd.wls_fit_tensor_fw, data,
                                       {'md_data': md, 'S0': S0, 'mask': mask},
                                       {'W': W, **kwargs}, nproc)

    assert serial.any()                                   # guard against comparing zeros
    assert parallel.tobytes() == serial.tobytes()


@pytest.mark.parametrize("nproc", [1, 4])
def test_fit_voxelwise_is_bit_identical_to_the_serial_dti_fit(delta_svd, _fw_volume, nproc):
    W, data, _, _, mask = _fw_volume

    serial = delta_svd.wls_fit_dti(W, data, mask=mask, min_signal=1.0e-6)
    parallel = delta_svd.fit_voxelwise(delta_svd.wls_fit_dti, data, {'mask': mask},
                                       {'W': W, 'min_signal': 1.0e-6}, nproc)

    assert serial.any()
    assert parallel.tobytes() == serial.tobytes()


def test_fit_voxelwise_handles_an_empty_mask(delta_svd, _fw_volume):
    # Every slab is a no-op, but the driver still owes the full zero volume.
    W, data, md, S0, _ = _fw_volume
    empty = np.zeros(data.shape[:-1], dtype=bool)

    parallel = delta_svd.fit_voxelwise(delta_svd.wls_fit_tensor_fw, data,
                                       {'md_data': md, 'S0': S0, 'mask': empty},
                                       {'W': W, 'mdreg': 2.0e-3}, 4)

    assert parallel.shape == data.shape[:-1] + (9,)
    assert not parallel.any()


def test_fit_voxelwise_propagates_worker_exceptions(delta_svd, _fw_volume):
    # A wrong-shaped mask raises inside the vendored fit; that has to surface in
    # the parent rather than hang the pool or go unnoticed.
    W, data, md, S0, _ = _fw_volume
    bad = np.ones((data.shape[0], data.shape[1] + 1, data.shape[2]), dtype=bool)

    with pytest.raises(ValueError):
        delta_svd.fit_voxelwise(delta_svd.wls_fit_tensor_fw, data,
                                {'md_data': md, 'S0': S0, 'mask': bad},
                                {'W': W, 'mdreg': 2.0e-3}, 4)


# ---------------------------------------------------------------------------
# Mask merging

def test_merge_masks_takes_voxelwise_union(delta_svd, tmp_path):
    affine = np.eye(4)
    m1 = np.array([[[1, 0], [0, 0]]], dtype="uint8")
    m2 = np.array([[[0, 0], [0, 1]]], dtype="uint8")
    fn1 = tmp_path / "m1.nii.gz"
    fn2 = tmp_path / "m2.nii.gz"
    nib.save(nib.Nifti1Image(m1, affine), str(fn1))
    nib.save(nib.Nifti1Image(m2, affine), str(fn2))

    out = tmp_path / "merged.nii.gz"
    fnOut = delta_svd.merge_masks([str(fn1), str(fn2)], str(out))

    result = nib.load(fnOut).get_fdata()
    assert np.array_equal(result, np.array([[[1, 0], [0, 1]]]))


def test_coreg_merge_masks_single_timepoint_binarises(delta_svd, tmp_path):
    tp_dir = tmp_path / "TP01"
    tp_dir.mkdir()
    affine = np.eye(4)
    src = tmp_path / "roi.nii.gz"
    img = np.array([[[0, 2], [0, -3]]], dtype=float)
    nib.save(nib.Nifti1Image(img, affine), str(src))

    fnOut = delta_svd.coreg_merge_masks(
        timepoints=[str(tp_dir)], masks=[str(src)], label="mask_test",
        dirTemplate=str(tmp_path), binarise=True,
    )

    result = nib.load(fnOut).get_fdata()
    assert np.array_equal(result, (img > 0).astype(float))


def test_coreg_merge_masks_returns_none_when_no_masks_provided(delta_svd, tmp_path):
    tp_dir = tmp_path / "TP01"
    tp_dir.mkdir()
    result = delta_svd.coreg_merge_masks(
        timepoints=[str(tp_dir)], masks=[None], label="x", dirTemplate=str(tmp_path),
    )
    assert result is None


# An uncompressed '.nii' mask must not be carried into the temp tree under its
# own extension: with a single timepoint there is no antsApplyTransforms step to
# normalise it, and every downstream consumer (tbss_non_FA in particular) assumes
# '.nii.gz'. Regression test for a mask silently ending up as 'mask_test.nii'.
@pytest.mark.parametrize("binarise", [True, False])
def test_coreg_merge_masks_normalises_uncompressed_input_to_nii_gz(delta_svd, tmp_path, binarise):
    tp_dir = tmp_path / "TP01"
    tp_dir.mkdir()
    src = tmp_path / "roi.nii"
    img = np.array([[[0, 256], [0, 255]]], dtype="uint16")
    nib.save(nib.Nifti1Image(img, np.eye(4)), str(src))

    fnOut = delta_svd.coreg_merge_masks(
        timepoints=[str(tp_dir)], masks=[str(src)], label="mask_test",
        dirTemplate=str(tmp_path), binarise=binarise,
    )

    assert fnOut.endswith(".nii.gz")
    expected = (img > 0) if binarise else img
    assert np.array_equal(nib.load(fnOut).get_fdata(), expected.astype(float))
    if not binarise:
        assert nib.load(fnOut).get_data_dtype() == img.dtype


# ---------------------------------------------------------------------------
# reorient_to_fsl_standard(): non-LAS/RAS input is reordered to LAS without
# changing any voxel's value or world position; LAS and RAS are untouched.

def _oriented(axcodes, shape=(4, 5, 6), dtype="int16", slope=None):
    """A random image stored in the voxel order 'axcodes', 2 mm voxels."""
    rng = np.random.default_rng(0)
    ornt = nib.orientations.axcodes2ornt(axcodes)
    affine = np.zeros((4, 4))
    affine[3, 3] = 1
    for i, (axis, flip) in enumerate(ornt.astype(int)):
        affine[axis, i] = 2.0 * flip
    affine[:3, 3] = [-10, 20, 5]
    data = rng.integers(0, 1000, shape).astype(dtype)
    nii = nib.Nifti1Image(data, affine)
    nii.header.set_dim_info(freq=0, phase=1, slice=2)
    if slope is not None:
        nii.header.set_slope_inter(slope, 3)
    return nii


def _world_values(nii):
    """Each voxel's value, keyed by its world coordinate."""
    data = np.asanyarray(nii.dataobj)
    idx = np.indices(data.shape).reshape(3, -1)
    world = (nii.affine @ np.vstack([idx, np.ones(idx.shape[1])]))[:3].T.round(6)
    return dict(zip(map(tuple, world), data.ravel()))


@pytest.mark.parametrize("axcodes", [("L", "P", "S"), ("R", "P", "I"), ("A", "L", "S"),
                                     ("S", "R", "A"), ("L", "A", "I")])
@pytest.mark.parametrize("dtype,slope", [("int16", None), ("int16", 0.5), ("float32", None)])
def test_reorient_reorders_voxels_without_changing_them(delta_svd, tmp_path, axcodes, dtype, slope):
    nii = _oriented(axcodes, dtype=dtype, slope=slope)
    fn = tmp_path / "map.nii.gz"
    nib.save(nii, str(fn))
    # read now: nib.load() is lazy and the file is rewritten in place
    before = nib.load(str(fn))
    values, dtype_, scaling = _world_values(before), before.get_data_dtype(), (before.dataobj.slope, before.dataobj.inter)
    codes = (int(before.header["sform_code"]), int(before.header["qform_code"]))
    zooms = before.header.get_zooms()

    assert delta_svd.reorient_to_fsl_standard(str(fn)) is True

    after = nib.load(str(fn))
    assert nib.aff2axcodes(after.affine) == ("L", "A", "S")
    assert np.allclose(after.header.get_sform(), after.affine)
    assert np.allclose(after.header.get_qform(), after.affine)
    assert after.get_data_dtype() == dtype_
    assert (after.dataobj.slope, after.dataobj.inter) == scaling
    assert np.allclose(after.header.get_zooms(), 2.0)
    assert _world_values(after) == values
    # header fields kept exactly, not recomputed from the affine
    assert (int(after.header["sform_code"]), int(after.header["qform_code"])) == codes
    assert sorted(after.header.get_zooms()) == sorted(zooms)
    # the frequency/phase/slice axes follow their data
    ornt = nib.orientations.ornt_transform(nib.orientations.io_orientation(nii.affine),
                                           nib.orientations.axcodes2ornt(("L", "A", "S")))
    assert after.header.get_dim_info() == tuple(int(ornt[d, 0]) for d in (0, 1, 2))
    # idempotent
    assert delta_svd.reorient_to_fsl_standard(str(fn)) is False


@pytest.mark.parametrize("axcodes", [("L", "A", "S"), ("R", "A", "S")])
def test_reorient_leaves_las_and_ras_untouched(delta_svd, tmp_path, axcodes):
    fn = tmp_path / "map.nii.gz"
    nib.save(_oriented(axcodes), str(fn))
    before = fn.read_bytes()
    assert delta_svd.reorient_to_fsl_standard(str(fn)) is False
    assert fn.read_bytes() == before


def test_needs_fsl_reorientation(delta_svd):
    for axcodes, expected in [(("L", "A", "S"), False), (("R", "A", "S"), False),
                              (("L", "P", "S"), True), (("R", "A", "I"), True),
                              (("A", "R", "S"), True)]:
        assert delta_svd.needs_fsl_reorientation(_oriented(axcodes).affine) is expected


@pytest.mark.parametrize("axcodes, expected", [
    ([("L", "P", "S")], "LPS"),                       # cross-sectional, reordered
    ([("R", "A", "S")], None),                        # cross-sectional, standard
    ([("L", "A", "S")], None),
    ([("L", "P", "S"), ("L", "P", "S")], None),       # longitudinal: never reordered
    ([("L", "A", "S"), ("L", "P", "S")], None),
])
def test_fsl_reorientation_source(delta_svd, axcodes, expected):
    affines = [_oriented(a).affine for a in axcodes]
    assert delta_svd.fsl_reorientation_source(affines) == expected


@pytest.mark.parametrize("binarise", [True, False])
def test_coreg_merge_masks_reorients_a_single_timepoint_mask(delta_svd, tmp_path, binarise):
    tp_dir = tmp_path / "TP01"
    tp_dir.mkdir()
    src = tmp_path / "roi.nii.gz"
    nii = _oriented(("L", "P", "S"), dtype="uint16")
    nib.save(nii, str(src))

    fnOut = delta_svd.coreg_merge_masks(
        timepoints=[str(tp_dir)], masks=[str(src)], label="mask_test",
        dirTemplate=str(tmp_path), binarise=binarise,
    )

    out = nib.load(fnOut)
    assert nib.aff2axcodes(out.affine) == ("L", "A", "S")
    expected = {k: int(v > 0) if binarise else v for k, v in _world_values(nii).items()}
    assert _world_values(out) == expected


def test_dwi_and_mni_paths_reject_the_same_invalid_roi_labels(delta_svd, tmp_path):
    invalid = np.array([0, 1.5], dtype=float).reshape(2, 1, 1)
    dwi_roi = tmp_path / "dwi-roi.nii.gz"
    nib.save(nib.Nifti1Image(invalid, np.eye(4)), str(dwi_roi))
    tp_dir = tmp_path / "TP01"
    tp_dir.mkdir()

    with pytest.raises(delta_svd.DeltaSvdError, match="finite integer labels from 0 through 65535"):
        delta_svd.coreg_merge_masks(
            timepoints=[str(tp_dir)], masks=[str(dwi_roi)], label="mask_test",
            dirTemplate=str(tmp_path), binarise=False,
        )

    kwargs, _ = _tbss_tree(
        tmp_path / "mni", skeleton=[1, 1], bmasks={"TP01": [1, 1]}, mni=[0, 1.5])
    with pytest.raises(delta_svd.DeltaSvdError, match="finite integer labels from 0 through 65535"):
        delta_svd.integrate_masks(**kwargs)


# ---------------------------------------------------------------------------
# copy_as_nii_gz(): the destination name always claims gzip, so an uncompressed
# input has to be re-encoded rather than copied - nibabel, FSL and ANTs all pick
# the codec from the file name and fail on a plain NIfTI named '.nii.gz'.

def test_copy_as_nii_gz_copies_gzipped_input_verbatim(delta_svd, tmp_path):
    src = tmp_path / "in.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), "uint8"), np.eye(4)), str(src))
    dst = tmp_path / "out.nii.gz"

    delta_svd.copy_as_nii_gz(str(src), str(dst))

    assert dst.read_bytes() == src.read_bytes()


def test_copy_as_nii_gz_reencodes_uncompressed_input(delta_svd, tmp_path):
    img = np.array([[[1, 0], [0, 3]]], dtype="uint8")
    src = tmp_path / "in.nii"
    nib.save(nib.Nifti1Image(img, np.eye(4)), str(src))
    dst = tmp_path / "out.nii.gz"

    delta_svd.copy_as_nii_gz(str(src), str(dst))

    assert dst.read_bytes()[:2] == b"\x1f\x8b"          # really gzipped
    assert np.array_equal(nib.load(str(dst)).get_fdata(), img.astype(float))
    assert nib.load(str(dst)).get_data_dtype() == img.dtype  # dtype preserved by default


def test_copy_as_nii_gz_preserves_scaled_integer_values(delta_svd, tmp_path):
    raw = np.array([0, 100], dtype="int8").reshape(2, 1, 1)
    src = tmp_path / "scaled.nii"
    image = nib.Nifti1Image(raw, np.eye(4))
    image.header.set_slope_inter(2, 0)
    nib.save(image, str(src))
    dst = tmp_path / "scaled.nii.gz"

    delta_svd.copy_as_nii_gz(str(src), str(dst))

    copied = nib.load(str(dst))
    assert copied.get_data_dtype() == raw.dtype
    assert copied.dataobj.slope == 2
    assert copied.get_fdata().ravel().tolist() == [0, 200]


def test_scaled_compressed_and_uncompressed_inputs_remain_equivalent(delta_svd, tmp_path):
    raw = np.array([0, 100], dtype="int8").reshape(2, 1, 1)
    plain = tmp_path / "plain.nii"
    compressed = tmp_path / "compressed.nii.gz"
    for path in (plain, compressed):
        image = nib.Nifti1Image(raw, np.eye(4))
        image.header.set_slope_inter(2, 0)
        nib.save(image, str(path))

    normalised_plain = tmp_path / "normalised-plain.nii.gz"
    normalised_compressed = tmp_path / "normalised-compressed.nii.gz"
    delta_svd.copy_as_nii_gz(str(plain), str(normalised_plain))
    delta_svd.copy_as_nii_gz(str(compressed), str(normalised_compressed))

    plain_values = nib.load(str(normalised_plain)).get_fdata()
    compressed_values = nib.load(str(normalised_compressed)).get_fdata()
    assert np.array_equal(plain_values, compressed_values)
    assert plain_values.ravel().tolist() == [0, 200]


# ---------------------------------------------------------------------------
# binarise_mask(): the threshold at 1 and the uint8 writes need a {0,1} mask, and
# binarising must be a no-op for one (every validated run).

@pytest.mark.parametrize("values", [
    [0.0, 1.0, 1.0, 0.0],                                   # float 0/1
    np.array([0, 1, 1, 0], dtype="uint8"),                  # integer 0/1
    [0.0, 0.0, 0.0, 0.0],                                   # empty mask
])
def test_binarise_mask_leaves_a_binary_mask_untouched(delta_svd, capsys, values):
    img = np.asarray(values, dtype=float)

    out = delta_svd.binarise_mask(img, "brain mask", "mask.nii.gz")

    assert np.array_equal(out.astype(float), img)           # values unchanged
    assert capsys.readouterr().out == ""                    # and reported as nothing to do


@pytest.mark.parametrize("values, expected", [
    ([0.0, 255.0, 255.0, 0.0], [0, 1, 1, 0]),               # ITK-style 0/255 mask
    ([0.0, 0.3, 0.9, 1.0], [0, 1, 1, 1]),                   # probabilistic mask
    ([0.0, 2.0, 0.0, 7.0], [0, 1, 0, 1]),                   # arbitrary labels
    ([-2.0, -0.1, 0.0, 1.0], [0, 0, 0, 1]),                # negative values are background
])
def test_binarise_mask_binarises_and_reports_anything_else(delta_svd, capsys, values, expected):
    out = delta_svd.binarise_mask(np.asarray(values), "brain mask", "mask.nii.gz")

    assert out.astype(int).tolist() == expected
    assert "other than 0 and 1" in capsys.readouterr().out


def test_binarise_mask_write_path_is_byte_identical_for_a_binary_mask(delta_svd, tmp_path):
    # For a {0,1} mask this has to produce the very same file as before, or the
    # change would move the validated numbers.
    img = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    src = tmp_path / "bmask.nii.gz"
    nib.save(nib.Nifti1Image(img, np.eye(4)), str(src))
    nii = nib.load(str(src))

    plain = tmp_path / "plain.nii.gz"
    binarised = tmp_path / "binarised.nii.gz"
    delta_svd.save_nifti(str(plain), nii.get_fdata(), nii.affine, nii.header, dtype="uint8")
    delta_svd.save_nifti(str(binarised), delta_svd.binarise_mask(nii.get_fdata(), "brain mask", str(src)),
                         nii.affine, nii.header, dtype="uint8")

    assert binarised.read_bytes() == plain.read_bytes()


# ---------------------------------------------------------------------------
# create_template(): the ANTs command line

class _StopAfterCommand(Exception):
    """Abort create_template once the ANTs command has been captured."""


def _capture_ants_command(delta_svd, tmp_path, monkeypatch, **kwargs):
    calls = []

    def record(cmd, *a, **k):
        calls.append(cmd)
        raise _StopAfterCommand

    monkeypatch.setattr(delta_svd, "run_subprocess", record)
    monkeypatch.setattr(delta_svd, "copy2", lambda *a, **k: None)
    tp = tmp_path / "TP01"
    tp.mkdir(exist_ok=True)
    with pytest.raises(_StopAfterCommand):
        delta_svd.create_template(timepoints=[str(tp)], fnCoreg=[], dirOut=str(tmp_path),
                                  coreBudget=1, **kwargs)
    return calls[0]


def test_default_iterations_reach_ants_unquoted_and_unchanged(delta_svd, tmp_path, monkeypatch):
    # shlex.quote is a no-op for the default, so the command stays byte-identical
    cmd = _capture_ants_command(delta_svd, tmp_path, monkeypatch,
                                iterations="30x30x8", numRegistrations=3)
    assert " -i 3 " in cmd
    assert " -q 30x30x8 " in cmd


def test_iterations_with_shell_metacharacters_cannot_break_out(delta_svd, tmp_path, monkeypatch):
    cmd = _capture_ants_command(delta_svd, tmp_path, monkeypatch,
                                iterations="30x30x8; rm -rf /", numRegistrations=3)
    assert " -q '30x30x8; rm -rf /' " in cmd


# ---------------------------------------------------------------------------
# free_water_correction(): fit mask thresholding and the no-CSF error.

def _fw_inputs(tmp_path, gtab_bits, fFree, mask):
    _, bvals, bvecs = gtab_bits
    data = np.stack([_two_compartment_signal(bvals, bvecs, f) for f in fFree]).reshape(len(fFree), 1, 1, -1)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(tmp_path / "dwi.nii.gz"))
    nib.save(nib.Nifti1Image(_vol(mask), np.eye(4)), str(tmp_path / "mask.nii.gz"))
    np.savetxt(tmp_path / "dwi.bval", bvals[None], fmt="%g")
    np.savetxt(tmp_path / "dwi.bvec", bvecs.T, fmt="%g")
    return dict(fn_data=str(tmp_path / "dwi.nii.gz"), fn_mask=str(tmp_path / "mask.nii.gz"),
                fn_bval=str(tmp_path / "dwi.bval"), fn_bvec=str(tmp_path / "dwi.bvec"),
                out_dir=str(tmp_path), smooth=False)


class _StopFit(Exception):
    pass


def _fit_mask(delta_svd, tmp_path, _fw_gtab, monkeypatch, mask):
    """The mask free_water_correction() hands to the first (single-tensor) fit."""
    seen = {}

    def fake_fit(fit, data, voxelArgs, kwargs, nproc):
        seen["mask"] = voxelArgs["mask"]
        raise _StopFit

    monkeypatch.setattr(delta_svd, "fit_voxelwise", fake_fit)
    with pytest.raises(_StopFit):
        delta_svd.free_water_correction(**_fw_inputs(tmp_path, _fw_gtab, [0.0] * len(mask), mask))
    return seen["mask"].ravel()


def test_free_water_fit_mask_excludes_negative_and_nan_voxels(delta_svd, tmp_path, _fw_gtab, monkeypatch):
    mask = _fit_mask(delta_svd, tmp_path, _fw_gtab, monkeypatch, [-1, 0, 0.5, np.nan])
    assert mask.dtype == bool
    assert mask.tolist() == [False, False, True, False]


@pytest.mark.parametrize("values", [[0, 1, 1, 0], [0, 0.2, 0.5, 1], [0, 255, 0, 3]])
def test_free_water_fit_mask_unchanged_for_binary_and_non_negative_masks(
        delta_svd, tmp_path, _fw_gtab, monkeypatch, values):
    mask = _fit_mask(delta_svd, tmp_path, _fw_gtab, monkeypatch, values)
    assert np.array_equal(mask, np.asarray(values, dtype=float).astype(bool))


def test_free_water_correction_rejects_a_mask_without_csf(delta_svd, tmp_path, _fw_gtab):
    # tissue only (MD 0.8e-3), so no voxel exceeds the 0.002 CSF threshold
    kwargs = _fw_inputs(tmp_path, _fw_gtab, [0.0, 0.1, 0.2], [1, 1, 1])
    with pytest.raises(delta_svd.DeltaSvdError, match="mean diffusivity above 0.002") as excinfo:
        delta_svd.free_water_correction(**kwargs)
    assert kwargs["fn_mask"] in str(excinfo.value)
    # raised before the second fit, so none of its output exists
    assert (tmp_path / "wls_dti_MD.nii.gz").exists()
    assert not (tmp_path / "fwc_wls_dti_FA.nii.gz").exists()


# ---------------------------------------------------------------------------
# extract_stats(): the skeleton name is stripped as a literal prefix, not a regex.

@pytest.mark.parametrize("skelBase", ["my+skel", "sk[el", "skel.v1"])
def test_extract_stats_strips_the_skeleton_name_literally(delta_svd, tmp_path, skelBase):
    stats_dir = tmp_path / "TBSS" / "stats"
    stats_dir.mkdir(parents=True)
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(_vol([1, 1, 1, 1]), affine),
             str(stats_dir / f"{skelBase}_intersection.nii.gz"))
    nib.save(nib.Nifti1Image(_vol([1, 1, 0, 0]), affine),
             str(stats_dir / f"{skelBase}_intersection_LH.nii.gz"))
    for m in ["FW", "MD"]:
        nib.save(nib.Nifti1Image(_vol([1e-3, 2e-3, 3e-3, 4e-3]), affine),
                 str(stats_dir / f"all_TP01_{m}_skeletonised.nii.gz"))

    df = delta_svd.extract_stats(dirTP="/some/path/TP01", dirTBSS=str(tmp_path / "TBSS"),
                                 fnNonFA={"FW": "unused", "MD": "unused"},
                                 skelMask=f"/somewhere/{skelBase}.nii.gz")

    assert set(df["region"]) == {"intersection", "intersection_LH"}


# ---------------------------------------------------------------------------
# ANTs' template construction uses its paths unquoted, so whitespace in the
# output path must never reach its command line.

def test_template_construction_keeps_the_output_path_out_of_the_ants_command(delta_svd, tmp_path, monkeypatch):
    dirOut = tmp_path / "My Study"
    dirOut.mkdir()
    tp = tmp_path / "TP01"
    tp.mkdir()
    seen = {}

    def record(cmd, *a, **k):
        seen.update(cmd=cmd, cwd=Path.cwd())
        raise _StopAfterCommand

    monkeypatch.setattr(delta_svd, "run_subprocess", record)
    monkeypatch.setattr(delta_svd, "copy2", lambda *a, **k: None)
    before = Path.cwd()
    with pytest.raises(_StopAfterCommand):
        delta_svd.create_template(timepoints=[str(tp)], fnCoreg=[], dirOut=str(dirOut), coreBudget=1)
    assert seen["cwd"] == dirOut.resolve()
    assert Path.cwd() == before
    assert "My Study" not in seen["cmd"]
    assert seen["cmd"].endswith(" -o ./ TP01_fwc_wls_dti_FA_05.nii.gz")


def test_mixed_voxel_order_warning(delta_svd):
    assert delta_svd.mixed_voxel_order_warning(["LAS"], ["TP01"]) is None
    assert delta_svd.mixed_voxel_order_warning(["RAS", "RAS"], ["BL", "FU"]) is None
    msg = delta_svd.mixed_voxel_order_warning(["LAS", "RAS"], ["BL", "FU"])
    assert "WARNING" in msg and "LAS (BL), RAS (FU)" in msg


def test_reorient_round_trip_restores_the_header_bytes(delta_svd, tmp_path):
    # Oblique, 2.1 mm slices, with a qform (rotation + qfac) and a derived sform as
    # a scanner writes them. Pixdim recomputed from the affine would move by a
    # float32 ULP, which FSL and ANTs amplify, so it has to be copied exactly.
    hdr = nib.Nifti1Header()
    hdr.set_data_shape((4, 5, 6))
    hdr.set_data_dtype("float32")
    hdr["pixdim"][:4] = [-1, 2.0, 2.0, 2.1]
    hdr["quatern_b"], hdr["quatern_c"], hdr["quatern_d"] = -0.054617215, 0.9684012, 0.24286464
    hdr["qoffset_x"], hdr["qoffset_y"], hdr["qoffset_z"] = 122.79986, -79.5609, -97.24512
    hdr["qform_code"] = 1
    hdr.set_sform(hdr.get_qform(), 1)
    las = nib.Nifti1Image(np.arange(4 * 5 * 6, dtype="float32").reshape(4, 5, 6), hdr.get_sform(), hdr)
    fnLas, fnLps = tmp_path / "las.nii.gz", tmp_path / "lps.nii.gz"
    nib.save(las, str(fnLas))
    las = nib.load(str(fnLas))
    assert nib.aff2axcodes(las.affine) == ("L", "A", "S")
    # the same image stored as LPS, header fields kept exactly
    flip = np.eye(4)
    flip[1, 1], flip[1, 3] = -1, las.shape[1] - 1
    lpsAffine = las.affine @ flip
    lps = nib.Nifti1Image(np.asanyarray(las.dataobj)[:, ::-1], lpsAffine, las.header)
    lps.header.set_sform(lpsAffine, int(las.header["sform_code"]))
    lps.header.set_qform(lpsAffine, int(las.header["qform_code"]))
    lps.header.set_zooms(las.header.get_zooms())
    nib.save(lps, str(fnLps))

    assert delta_svd.reorient_to_fsl_standard(str(fnLps)) is True

    back = nib.load(str(fnLps))
    assert np.array_equal(np.asanyarray(back.dataobj), np.asanyarray(las.dataobj))
    for field in ("pixdim", "sform_code", "qform_code", "dim", "datatype", "scl_slope", "scl_inter"):
        assert back.header[field].tobytes() == las.header[field].tobytes(), field
    # the quaternion is recomputed from the reordered matrix and may move by a
    # float32 ULP; FSL reads the sform here, which matches to rounding
    assert np.allclose(back.header.get_sform(), las.header.get_sform(), atol=1e-4)
    assert np.allclose(back.header.get_qform(), las.header.get_qform(), atol=1e-4)
