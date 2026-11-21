"""Guards on which modules may be imported where.

delta-svd.py pins the BLAS and ITK thread counts and checks the CPU before the
first third-party import, because numpy and friends read that environment when
they are loaded. These tests read the syntax tree, so they see every import
statement, including those inside functions.
"""
import ast
import sys

import pytest

from conftest import CONTAINER_FILES

#--- importable before the thread pinning: the standard library and these
PRE_PINNING_MODULES = {'delta_svd_constants', 'delta_svd_runtime'}


def imported_modules(tree, before_line=None):
    """Top-level package names of all imports in 'tree', optionally only above a line."""
    names = set()
    for node in ast.walk(tree):
        if before_line is not None and getattr(node, 'lineno', before_line) >= before_line:
            continue
        if isinstance(node, ast.Import):
            names.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or '.').split('.')[0])
    return names


def test_nothing_third_party_is_imported_before_the_pinning():
    tree = ast.parse((CONTAINER_FILES / 'delta-svd.py').read_text())
    pinned = [node.lineno for node in tree.body
              if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
              and node.test.id == '_missing']
    assert len(pinned) == 1, "expected exactly one top-level 'if _missing:' statement"
    early = imported_modules(tree, before_line=pinned[0])
    assert early, "no imports found before 'if _missing:'"
    assert early - set(sys.stdlib_module_names) - PRE_PINNING_MODULES == set()


@pytest.mark.parametrize('fname', ['delta_svd_constants.py', 'delta_svd_runtime.py'])
def test_pre_pinning_modules_import_the_standard_library_only(fname):
    imported = imported_modules(ast.parse((CONTAINER_FILES / fname).read_text()))
    assert imported - set(sys.stdlib_module_names) - {'delta_svd_constants'} == set()
