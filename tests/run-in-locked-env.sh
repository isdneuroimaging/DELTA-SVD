#!/usr/bin/env bash
#
# Run the unit tests against the pinned conda environment
# (container/conda-explicit-linux-64.txt) plus the Dockerfile's pinned Pillow
# wheel. PR CI uses this environment; release CI supplies the staged image.
#
# The image deliberately does not ship pytest. It is installed only inside the
# disposable test container, with checks that no runtime packages changed.
#
# The repository is mounted read-only, so what gets tested is the working tree.
# On Apple Silicon the amd64 emulation makes this slow (roughly ten minutes);
# on a Linux x86-64 host it is quick.
#
# Usage:
#   tests/run-in-locked-env.sh [extra pytest args...]
#   DELTA_SVD_TEST_IMAGE=<image-or-digest> tests/run-in-locked-env.sh
#
# Examples:
#   tests/run-in-locked-env.sh
#   tests/run-in-locked-env.sh -k integrate_masks -vv
set -euo pipefail

# Resolve the repo root from this script's location so it works from any CWD.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

TEST_IMAGE="${DELTA_SVD_TEST_IMAGE:-ubuntu:noble-20260509.1@sha256:786a8b558f7be160c6c8c4a54f9a57274f3b4fb1491cf65146521ae77ff1dc54}"
echo "Testing the working tree at ${REPO_ROOT} using ${TEST_IMAGE}"

# Keep the base image, glibc, Miniconda and Pillow pins in step with Dockerfile.
# An existing image skips environment creation entirely. Root is used only in
# the disposable container to install pytest; the image itself is unchanged.
exec docker run --rm -i --platform linux/amd64 --user 0:0 --entrypoint bash \
    -v "${REPO_ROOT}:/repo:ro" \
    -e DELTA_SVD_TEST_IMAGE="${DELTA_SVD_TEST_IMAGE:-}" \
    -e OPENBLAS_CORETYPE=Haswell \
    -e MPLCONFIGDIR=/tmp/matplotlib \
    -e PYTHONDONTWRITEBYTECODE=1 \
    "${TEST_IMAGE}" -seu -- "$@" <<'BASH'
set -o pipefail
if [ -z "$DELTA_SVD_TEST_IMAGE" ]; then
    apt-get update -qq
    apt-get install -y -q --no-install-recommends \
        ca-certificates=20260816* \
        libc6=2.39-0ubuntu8.9 \
        libc-bin=2.39-0ubuntu8.9 \
        wget=1.21.4*
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-py311_25.5.1-0-Linux-x86_64.sh -O /tmp/mc.sh
    echo "a921abd74e16f5dee8a4d79b124635fac9b939c465ba2e942ea61b3fcd1451d8  /tmp/mc.sh" | sha256sum -c -
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
fi

# pytest via pip: a conda install re-solves and can move locked packages. Both
# conda's and pip's view are checked, as conda misses pip replacing a package.
conda list -p /opt/conda --explicit > /tmp/lock-before.txt
python -m pip freeze | sort > /tmp/pip-before.txt
python -m pip install -q --no-cache-dir pytest==9.1.1
conda list -p /opt/conda --explicit > /tmp/lock-after.txt
python -m pip freeze | sort > /tmp/pip-after.txt
diff /tmp/lock-before.txt /tmp/lock-after.txt \
    || { echo "installing pytest changed the locked conda environment" >&2; exit 1; }
CHANGED="$(comm -23 /tmp/pip-before.txt /tmp/pip-after.txt)"
[ -z "$CHANGED" ] || { printf "installing pytest replaced locked packages:\n%s\n" "$CHANGED" >&2; exit 1; }
echo "Locked environment unchanged by the pytest install."
cd /repo
python -m pytest tests -q -p no:cacheprovider "$@"
BASH
