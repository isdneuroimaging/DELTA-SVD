#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve the DELTA-SVD release version from the repo-root VERSION file, the
single source of truth build.sh also stamps into the image's OCI labels.
"""

import os

VERSION_UNKNOWN = 'unknown'


def read_version(scriptDir=None):
    """The release version, or 'unknown' when VERSION is missing or empty.
    Never raises: not knowing the version must not stop a run."""
    if scriptDir is None:
        scriptDir = os.path.dirname(os.path.abspath(__file__))
    candidates = (
        os.path.join(scriptDir, 'VERSION'),                          # inside the image
        os.path.join(scriptDir, os.pardir, os.pardir, 'VERSION'),    # source checkout
    )
    for fn in candidates:
        try:
            with open(fn) as fh:
                version = fh.read().strip()
        except OSError:
            continue
        if version:
            return version
    return VERSION_UNKNOWN


__version__ = read_version()
