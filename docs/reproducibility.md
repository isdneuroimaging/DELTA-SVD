---
icon: lucide/repeat
---

# Reproducibility

## Within a version

Longitudinal ANTs registration divides its similarity metric across threads and sums the parts, so the thread count changes the order of that summation and with it the last bits of the result. That would normally be negligible, but the white matter skeleton is derived by thresholding an interpolated mask, which turns those last bits into whole voxels moving in or out of the skeleton. The metrics at each timepoint change only slightly, but the longitudinal change can shift substantially, especially ΔPSMD.

DELTA-SVD therefore fixes the registration thread count at the value the method was validated with, instead of deriving it from the cores available. This is why `--threads` and `--para` affect only runtime and memory: they decide how many registrations run at once, never how each one is computed.

`--itkThreads` overrides that count. It exists for method development and is deliberately not listed in `--help`. **Do not change it in production use**: results produced with a different value cannot be compared with, or pooled with, results produced at the default.

DELTA-SVD also protects the results from settings in your own environment, which could otherwise select a different FSL, ANTs, BLAS kernel or Python packages. There are two ways this could happen: Apptainer passes host environment variables that the image does not set into the container, such as a `PYTHONPATH` from an HPC module, and the image's own `FSLDIR`, `ANTSPATH` and `OPENBLAS_CORETYPE` can be replaced with `--env`. To prevent this, DELTA-SVD restores the image's own values when it starts, overriding any such changes, and ignores `PYTHONPATH` and Python packages in your home directory (`~/.local`).

## Between releases

DELTA-SVD version numbers have three parts, `MAJOR.MINOR.PATCH`, for example `1.2.3`. The format follows [semantic versioning](https://semver.org/), but which part changes depends on whether results can change:

- **Patch release** (`1.2.3` to `1.2.4`): preserves the validated whole-skeleton endpoints on standard inputs. It can contain bug fixes and new options.
- **`MAJOR` or `MINOR` release** (`1.2.3` to `1.3.0` or `2.0.0`): can shift the validated endpoints, so results must not be pooled with those from earlier versions.

For this reason, [choose one version per project](install.md#getting-the-image).

To test that the endpoints are preserved, each release is compared with a fixed reference, not just with the release before it. The [Zenodo regression dataset](https://zenodo.org/records/23723917) contains two preprocessed subjects with two sessions each, along with reference results generated with DELTA-SVD 1.0.0, the version used in the method publication. Before a release is published, its image processes these subjects, and all metric values and skeleton voxel counts must exactly match the reference.

Starting with version 1.0.6, each GitHub release includes the outcome as `delta-svd_<version>_test-evidence.zip`. The ZIP records the tested image digest, source revision, the reference dataset's URL and checksum, and per-subject verdicts, along with each subject's result tables, comparison output and run manifest.

This is the evidence that a patch release leaves those endpoints unchanged from the published version on these reference data. It doesn't test every possible input or optional feature, and it doesn't replace formal validation. If a patch release corrects optional derived outputs, its compatibility note names the affected inputs and the rows that must not be pooled across the patch release. A release that intentionally changes the endpoints requires formal re-validation and a new approved reference.
