import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTAINER_FILES = REPO_ROOT / "container" / "scripts"

# create_html_with_png.py and create_qc_image.py import each other by plain
# module name (not via importlib), so they need to be importable normally.
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
