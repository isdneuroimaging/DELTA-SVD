import matplotlib.image as mpimg
import nibabel as nib
import numpy as np
import pytest

import create_qc_image as cqi


def _save(tmp_path, name, arr, affine):
    fn = tmp_path / name
    nib.save(nib.Nifti1Image(arr, affine), str(fn))
    return str(fn)


def test_create_qc_image_requires_mask():
    with pytest.raises(ValueError):
        cqi.create_qc_image(["a.nii.gz"], [[0, 1]], fnameMask=None)


def test_create_qc_image_produces_expected_png_dimensions(tmp_path):
    affine = np.eye(4)
    shape = (10, 10, 12)
    rng = np.random.default_rng(0)
    fa = rng.random(shape).astype("float32")
    md = (rng.random(shape) * 0.001).astype("float32")
    mask = np.zeros(shape, dtype="uint8")
    mask[3:7, 3:7, 3:9] = 1
    bmask = np.ones(shape, dtype="uint8")

    fnFA = _save(tmp_path, "fa.nii.gz", fa, affine)
    fnMD = _save(tmp_path, "md.nii.gz", md, affine)
    fnMask = _save(tmp_path, "mask.nii.gz", mask, affine)
    fnBmask = _save(tmp_path, "bmask.nii.gz", bmask, affine)

    out = cqi.create_qc_image(
        [fnFA, fnMD], [[0, 1], [0, 0.001]], ["FA", "MD"], fnMask, fnBmask, animate=False,
    )

    assert len(out) == 2
    nSlices = 8
    zoom = 5
    for fn in out:
        img = mpimg.imread(fn)
        height, width = img.shape[:2]
        assert height == shape[0] * zoom
        assert width == shape[1] * nSlices * zoom
