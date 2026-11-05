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

## Testing your installation

We recommend running the release check yourself before starting a project, and whenever your system changes. It confirms that your image, container runtime and hardware reproduce the reference results for all 1.0.x releases. It doesn't check your own data or preprocessing.

**1. Download the [test data](https://zenodo.org/records/23723917) and the comparison script:**

```shell
curl -LO https://zenodo.org/records/23723917/files/delta-svd-longitudinal-test-v2.zip
echo "88d7aab257c5f20d3a6acdfcd837a5d6d4994b825ff4f47b3225341b6ebf4e8f  delta-svd-longitudinal-test-v2.zip" | sha256sum --check
unzip -q delta-svd-longitudinal-test-v2.zip
cd delta-svd-longitudinal-test-v2
curl -LO https://raw.githubusercontent.com/isdneuroimaging/DELTA-SVD/main/tools/compare_results.py
```

**2. Run the test with the [image](install.md#getting-the-image) you plan to use, using either Apptainer or Docker:**

```shell
# Apptainer: replace /path/to/delta-svd.sif with your image
for sub in sub-test01 sub-test02; do
  apptainer run --no-home /path/to/delta-svd.sif \
    --dwi inputs/$sub/ses-1/${sub}_ses-1_dwi.nii.gz inputs/$sub/ses-2/${sub}_ses-2_dwi.nii.gz \
    --tp ses-1 ses-2 --id $sub -o results/$sub
done
for sub in sub-test01 sub-test02; do
  apptainer exec --no-home /path/to/delta-svd.sif python compare_results.py \
    reference/$sub/delta-svd_results.csv results/$sub/delta-svd_results.csv
done
```

```shell
# Docker: replace <version> with your version
for sub in sub-test01 sub-test02; do
  docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/data" -w /data \
    ghcr.io/isdneuroimaging/delta-svd:<version> \
    --dwi inputs/$sub/ses-1/${sub}_ses-1_dwi.nii.gz inputs/$sub/ses-2/${sub}_ses-2_dwi.nii.gz \
    --tp ses-1 ses-2 --id $sub -o results/$sub
done
for sub in sub-test01 sub-test02; do
  docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/data" -w /data --entrypoint python \
    ghcr.io/isdneuroimaging/delta-svd:<version> compare_results.py \
    reference/$sub/delta-svd_results.csv results/$sub/delta-svd_results.csv
done
```

**3. Check the result.** The last lines of the output must report this for both subjects:

```text
RESULT: no metric value and no voxel count changed.
```

If not, don't process study data on that system yet, and [open an issue](https://github.com/isdneuroimaging/DELTA-SVD/issues) with the comparison output, your DELTA-SVD version, container runtime and CPU model.
