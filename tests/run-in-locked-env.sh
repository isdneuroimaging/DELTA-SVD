#!/usr/bin/env bash
#
# Run the unit tests against the pinned conda environment
# (container/conda-explicit-linux-64.txt) instead of the PyPI stack they
# normally use.
#
# Not part of the normal workflow. It is worth doing after changing the lock,
# where a dependency bump could remove an API the scripts rely on - that would
# surface here and nowhere else. The image deliberately does not ship pytest, so
# this builds a throwaway environment from the lock rather than adding a test
# framework to the released image.
#
# The repository is mounted read-only, so what gets tested is the working tree.
# On Apple Silicon the amd64 emulation makes this slow (roughly ten minutes);
# on a Linux x86-64 host it is quick.
#
# Usage:
#   tests/run-in-locked-env.sh [extra pytest args...]
#
# Examples:
#   tests/run-in-locked-env.sh
#   tests/run-in-locked-env.sh -k integrate_masks -vv
set -euo pipefail

# Resolve the repo root from this script's location so it works from any CWD.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

echo "Testing the working tree at ${REPO_ROOT} against container/conda-explicit-linux-64.txt"

# The inner script is single-quoted on purpose: $PATH and "$@" are expanded by
# the shell *inside* the container, not by this one. Keep the Miniconda build in
# step with the one the Dockerfile installs.
exec docker run --rm --platform linux/amd64 -v "${REPO_ROOT}:/repo:ro" ubuntu:noble bash -euc '
    apt-get update -qq
    apt-get install -y -q --no-install-recommends ca-certificates wget
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-py311_25.5.1-0-Linux-x86_64.sh -O /tmp/mc.sh
    bash /tmp/mc.sh -b -p /opt/conda
    export PATH=/opt/conda/bin:$PATH
    conda tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/main --channel https://repo.anaconda.com/pkgs/r
    conda install -y -p /opt/conda --file /repo/container/conda-explicit-linux-64.txt
    conda install -y -p /opt/conda -c conda-forge pytest
    cd /repo && python -m pytest tests -q -p no:cacheprovider "$@"' _ "$@"
