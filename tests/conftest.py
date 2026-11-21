import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTAINER_FILES = REPO_ROOT / "container" / "scripts"

# The delta_svd_*.py modules, create_html_with_png.py and create_qc_image.py
# import each other by plain module name (not via importlib), and the tests
# import them the same way, so they need to be importable normally.
sys.path.insert(0, str(CONTAINER_FILES))


def load_module(name, path):
    # delta-svd.py and delta-svd_aggregate_results.py contain a hyphen, so
    # they can't be imported with a normal `import` statement.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def delta_svd():
    return load_module("delta_svd", CONTAINER_FILES / "delta-svd.py")


@pytest.fixture(scope="session")
def aggregate_results():
    return load_module("delta_svd_aggregate_results",
                       CONTAINER_FILES / "delta-svd_aggregate_results.py")


@pytest.fixture(scope="session")
def compare_results():
    # maintainer tool, lives outside the container image
    return load_module("compare_results", REPO_ROOT / "tools" / "compare_results.py")


# ---------------------------------------------------------------------------
# Grid helpers shared by test_inputs.py and test_pipeline.py. The MNI grid is a
# stand-in with the shipped skeleton mask's header, since '/opt/scripts' only
# exists in the image.

MNI_AFFINE = np.array([[-1., 0, 0, 90], [0, 1, 0, -126], [0, 0, 1, -72], [0, 0, 0, 1]])
MNI_SHAPE = (182, 218, 182)

DWI_AFFINE = np.diag([2.0, 2.0, 2.5, 1.0])
DWI_AFFINE[:3, 3] = [-100.0, -120.0, -60.0]


def save_grid(path, shape=MNI_SHAPE, affine=MNI_AFFINE):
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype="uint8"), affine), str(path))
    return str(path)


def save_dwi(path, shape=(8, 9, 7, 3), affine=DWI_AFFINE):
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype="float32"), affine), str(path))
    return str(path)


@pytest.fixture
def mni_reference(tmp_path, monkeypatch):
    """Point the default skeleton mask at an MNI-grid stand-in."""
    import delta_svd_inputs
    ref = save_grid(tmp_path / "reference.nii.gz")
    monkeypatch.setattr(delta_svd_inputs, "SKELETON_MASK_DEFAULT", ref)
    return ref
