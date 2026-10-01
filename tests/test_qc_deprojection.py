import os
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

import create_html_with_png
import create_qc_image


def test_consecutive_labels_round_trip(delta_svd):
    img = np.array([[[0, 3, 1000], [65535, 3, 0]]], dtype=float)
    consecutive, labels = delta_svd.consecutive_labels(img)
    assert list(labels) == [3, 1000, 65535]
    assert consecutive.max() == 3
    assert np.array_equal(delta_svd.restore_labels(consecutive.astype(float), labels), img)


def test_consecutive_labels_keep_their_order(delta_svd):
    # where deprojected labels meet, tbss_deproject keeps the highest; a
    # monotone renumbering keeps that choice unchanged
    img = np.array([[[5, 700, 20]]], dtype=float)
    consecutive, _ = delta_svd.consecutive_labels(img)
    assert list(consecutive.ravel()) == [1, 3, 2]


def _tbss(tmp_path, labels):
    stats = tmp_path / "TBSS" / "stats"
    stats.mkdir(parents=True)
    (tmp_path / "TBSS" / "FA").mkdir()
    fn = stats / "skel_intersection_RmaskMNI.nii.gz"
    nib.save(nib.Nifti1Image(labels.astype("uint16"), np.eye(4)), str(fn))
    return str(fn)


@pytest.mark.parametrize("relabel", [True, False])
def test_deproject_to_native_renumbers_label_maps(delta_svd, tmp_path, monkeypatch, relabel):
    labels = np.array([[[0, 7, 1000], [0, 7, 3]]])
    fn = _tbss(tmp_path, labels)
    tp = tmp_path / "TP01"
    tp.mkdir()
    dirQC = tmp_path / "qc"
    dirQC.mkdir()
    seen = {}

    def fake_run(cmd, displayStdout, label):
        # stands in for 'tbss_deproject <name> 2 -n', run in TBSS/stats: an
        # identity deprojection, recording the largest label it had to handle
        assert label == "tbss_deproject"
        name = cmd.split()[1]
        img = nib.load(name)
        seen["max"] = int(img.get_fdata().max())
        nib.save(img, os.path.join("..", "FA", "fwc_wls_dti_FA_05_FA_" + name))

    monkeypatch.setattr(delta_svd, "run_subprocess", fake_run)
    delta_svd.deproject_to_native(fn, str(dirQC), str(tmp_path / "TBSS"), str(tmp_path / "template"),
                                  [str(tp)], relabel=relabel)

    assert seen["max"] == (3 if relabel else 1000)
    out = nib.load(str(dirQC / "TP01_skel_intersection_RmaskMNI.nii.gz")).get_fdata()
    assert np.array_equal(out, labels)


@pytest.mark.parametrize("qc, expected", [
    (1, [("skel_intersection.nii.gz", False)]),
    (2, [("skel_intersection.nii.gz", False), ("skel_intersection_Rmask.nii.gz", True),
         ("skel_intersection_RmaskMNI.nii.gz", True)]),
])
def test_prepare_qc_deprojects_label_maps_only_when_kept(delta_svd, tmp_path, monkeypatch, qc, expected):
    # the label maps are never shown in the report, only kept with '--qc 2'
    stats = tmp_path / "TBSS" / "stats"
    stats.mkdir(parents=True)
    for name in ["skel_intersection.nii.gz", "skel_intersection_Rmask.nii.gz",
                 "skel_intersection_RmaskMNI.nii.gz"]:
        (stats / name).touch()
    tp = tmp_path / "TP01"
    tp.mkdir()
    for name in ["fwc_wls_dti_FA_05.nii.gz", "wls_dti_FA.nii.gz", "wls_dti_MD.nii.gz"]:
        (tp / name).touch()
    dirQC = tmp_path / "delta-svd_qc"
    dirQC.mkdir()
    deprojected = []
    monkeypatch.setattr(delta_svd, "deproject_to_native",
                        lambda fn, *a, relabel=False: deprojected.append((os.path.basename(fn), relabel)))
    monkeypatch.setattr(create_qc_image, "create_qc_image",
                        lambda *a, animate=True, **k: "x.png" if animate else [])
    monkeypatch.setattr(create_html_with_png, "create_html_with_png", lambda *a, **k: None)

    delta_svd.prepare_qc(str(dirQC), str(tmp_path / "qc.html"), "/opt/scripts/skel.nii.gz",
                         str(tmp_path / "TBSS"), str(tmp_path / "template"), [str(tp)],
                         "results.csv", SimpleNamespace(qc=qc, dirOutput=str(tmp_path)))
    assert deprojected == expected
