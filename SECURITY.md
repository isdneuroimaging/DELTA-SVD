# Security Policy

> [!WARNING]
> **Research use only, not a medical device.** DELTA-SVD is intended solely for research. It is not a medical device, has not been reviewed or approved by any regulatory authority, and must not be used for clinical diagnosis, treatment, or other medical decisions. The software is provided "as is", without warranty of any kind; to the fullest extent permitted by law, the authors accept no liability for any damages arising from its use.

## Supported versions

Only the most recently published release is supported: security fixes land on `main` and go out in the next release, and older release lines are not maintained separately, so upgrading is the only way to get a fix.

DELTA-SVD's endpoints are validated per-version (see [CONTRIBUTING.md](CONTRIBUTING.md#validation-status-read-this-first)), and results from different `MAJOR.MINOR` versions must not be pooled. A patch-only bump is defined as provably metric-identical, so results across patch versions of the same `MAJOR.MINOR` can be pooled.

## Reporting a vulnerability

Please report security issues privately using [GitHub's private vulnerability reporting](https://github.com/isdneuroimaging/DELTA-SVD/security/advisories/new) rather than a public issue. We'll acknowledge the report and work with you on a fix and disclosure timeline.

This pipeline processes local imaging data inside a container and has no network-facing component, so most reports will concern the container image or its dependency chain (e.g. an exploitable vulnerability in a pinned package) rather than the pipeline logic itself.
