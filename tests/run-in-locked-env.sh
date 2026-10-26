#!/usr/bin/env bash
#
# Run the unit tests against the pinned conda environment
# (container/conda-explicit-linux-64.txt) plus the Dockerfile's pinned Pillow
# wheel, instead of the PyPI stack they normally use.
#
# Not part of the normal workflow. It is worth doing after changing the lock,
# where a dependency bump could remove an API the scripts rely on - that would
# surface here and nowhere else. The image deliberately does not ship pytest, so
# this builds a throwaway environment from the lock and wheel rather than adding
# a test framework to the released image.
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
# step with the one the Dockerfile installs. Keep the Pillow URL and SHA-256
# identical to the Dockerfile, too.
exec docker run --rm --platform linux/amd64 -v "${REPO_ROOT}:/repo:ro" ubuntu:noble bash -euc '
    apt-get update -qq
    apt-get install -y -q --no-install-recommends ca-certificates wget
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-py311_25.5.1-0-Linux-x86_64.sh -O /tmp/mc.sh
    bash /tmp/mc.sh -b -p /opt/conda
    export PATH=/opt/conda/bin:$PATH
    conda tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/main --channel https://repo.anaconda.com/pkgs/r
    conda install -y -p /opt/conda --file /repo/container/conda-explicit-linux-64.txt
    # The conda lock deliberately omits Pillow to preserve FSL-compatible libzlib.
    wget -q https://files.pythonhosted.org/packages/36/46/f509b74787f40ce0627501089d630b2c741b63b244c1f40b640b3b01e181/pillow-13.0.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl -P /tmp/
    echo "a625df9a2a5597fa5e5266e01aefc82604bb85d291e3547981e863560d75da6c  /tmp/pillow-13.0.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl" | sha256sum -c -
    python -m pip install --no-deps --no-index --no-cache-dir /tmp/pillow-13.0.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
    rm /tmp/pillow-13.0.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
    # pytest via pip: a conda install re-solves and can move locked packages. Both
    # conda's and pip's view are checked, as conda misses pip replacing a package.
    conda list -p /opt/conda --explicit > /tmp/lock-before.txt
    python -m pip freeze | sort > /tmp/pip-before.txt
    python -m pip install -q --no-cache-dir pytest
    conda list -p /opt/conda --explicit > /tmp/lock-after.txt
    python -m pip freeze | sort > /tmp/pip-after.txt
    diff /tmp/lock-before.txt /tmp/lock-after.txt \
        || { echo "installing pytest changed the locked conda environment" >&2; exit 1; }
    CHANGED="$(comm -23 /tmp/pip-before.txt /tmp/pip-after.txt)"
    [ -z "$CHANGED" ] || { printf "installing pytest replaced locked packages:\n%s\n" "$CHANGED" >&2; exit 1; }
    echo "Locked environment unchanged by the pytest install."
    cd /repo && python -m pytest tests -q -p no:cacheprovider "$@"' _ "$@"
