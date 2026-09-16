#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve the release metadata embedded in a DELTA-SVD image.

The build writes VERSION and SOURCE_REVISION next to the scripts.  In a source
checkout the version falls back to the repository-root VERSION file; a source
revision is only known when the checkout has been built, so it remains
``unknown`` unless the embedded file is present.
"""

import os

VERSION_UNKNOWN = 'unknown'
SOURCE_REVISION_UNKNOWN = 'unknown'


def _read_metadata(filename, unknown, scriptDir=None):
    """Read an embedded metadata file, with the source-checkout fallback."""
    if scriptDir is None:
        scriptDir = os.path.dirname(os.path.abspath(__file__))
    candidates = (
        os.path.join(scriptDir, filename),
        os.path.join(scriptDir, os.pardir, os.pardir, filename),
    )
    for fn in candidates:
        try:
            with open(fn) as fh:
                value = fh.read().strip()
        except OSError:
            continue
        if value:
            return value
    return unknown


def read_version(scriptDir=None):
    """The release version, or 'unknown' when VERSION is missing or empty."""
    return _read_metadata('VERSION', VERSION_UNKNOWN, scriptDir)


def read_source_revision(scriptDir=None):
    """The built source revision, or 'unknown' when it is not embedded."""
    return _read_metadata('SOURCE_REVISION', SOURCE_REVISION_UNKNOWN, scriptDir)


__version__ = read_version()
__source_revision__ = read_source_revision()
