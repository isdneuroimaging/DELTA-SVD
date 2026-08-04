#!/usr/bin/env bash
#
# Build the DELTA-SVD container image with OCI provenance labels populated.
#
# Injects the current git commit (org.opencontainers.image.revision) and build
# timestamp (org.opencontainers.image.created) as build args so every image can
# be traced back to the exact source it was built from.
#
# The version is read from the repo-root VERSION file (single source of truth,
# also stamped into org.opencontainers.image.version).
#
# Usage:
#   container/build.sh [IMAGE[:TAG]] [extra docker build args...]
#
# Examples:
#   container/build.sh                          # -> delta-svd:<VERSION>
#   container/build.sh delta-svd:dev
#   container/build.sh ghcr.io/isdneuroimaging/delta-svd:1.0.0 --no-cache
#
set -euo pipefail

# Resolve repo root from this script's location so it works from any CWD.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

VERSION="$(tr -d '[:space:]' < "${REPO_ROOT}/VERSION")"

IMAGE="${1:-delta-svd:${VERSION}}"
[ "$#" -gt 0 ] && shift  # remaining args pass through to docker build

VCS_REF="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
# Mark the revision dirty if the working tree differs from HEAD in any way.
# 'status --porcelain' rather than 'diff HEAD': the latter is blind to untracked
# files, yet 'COPY container/scripts/' would carry a new, uncommitted script into
# the image while the label still claimed a clean revision. Ignored paths (the
# virtualenvs, __pycache__) are excluded, so they cannot mark a release dirty.
if [ -n "$(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null)" ]; then
    VCS_REF="${VCS_REF}-dirty"
fi
BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "Building ${IMAGE}"
echo "  version:  ${VERSION}"
echo "  revision: ${VCS_REF}"
echo "  created:  ${BUILD_DATE}"

# --platform is pinned, not left to the host: the conda lock resolves linux-64
# URLs and the bundled ANTs is an x86-64 build, so an arm64 host (an Apple
# Silicon Mac) would otherwise start a build that cannot work. On an x86-64
# host this is a no-op; elsewhere it selects emulation, which is slow but sound.
exec docker build \
    --platform linux/amd64 \
    -f "${REPO_ROOT}/container/Dockerfile" \
    --build-arg VERSION="${VERSION}" \
    --build-arg VCS_REF="${VCS_REF}" \
    --build-arg BUILD_DATE="${BUILD_DATE}" \
    -t "${IMAGE}" \
    "$@" \
    "${REPO_ROOT}"
