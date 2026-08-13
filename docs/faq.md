---
icon: lucide/circle-help
---

# Frequently asked questions

## Where can I get support, report bugs, or ask questions?

DELTA-SVD is maintained on a community, best-effort basis, with no formal or guaranteed support.

- **Questions, usage help, and general discussion**: use [GitHub Discussions](https://github.com/isdneuroimaging/DELTA-SVD/discussions).
- **Bugs and feature requests**: open a [GitHub issue](https://github.com/isdneuroimaging/DELTA-SVD/issues). Please include the version you are running (see [Which version am I running?](#which-version-am-i-running) below) and enough detail to reproduce the problem (command line, input description, and any error output).

We try to respond where we can, but cannot commit to any particular response time.

## Which version am I running?

Ask the image directly:

```
apptainer run delta-svd.sif --version
```

Every run also names its version on the **first line of its console output** and in the table at the top of the **QC report** (`delta-svd_qc.html`), so a finished analysis can be traced back to the version that produced it. This matters because results from different `MAJOR.MINOR` versions must not be compared or pooled; see [Installation](install.md#checking-which-version-you-have).

## What are MSMD, PSMD and MSFW, and which should I report?

Every run reports all three validated endpoints, per timepoint and per region:

- **MSMD**: mean skeletonised mean diffusivity.
- **PSMD**: peak width of skeletonised mean diffusivity.
- **MSFW**: mean skeletonised free water.

For most datasets, **MSMD is the recommended primary endpoint**; PSMD and MSFW are reported alongside it and can add information in specific settings. For the rationale and guidance on when each is most useful, see the method publication ([Citation](index.md#citation)).

## How does DELTA-SVD relate to the original PSMD pipeline?

PSMD (peak width of skeletonised mean diffusivity, Baykara et al., Ann Neurol 2016) is an established diffusion marker of white matter damage in cerebral small vessel disease. DELTA-SVD builds on the same skeleton-based approach and reports PSMD as one of its endpoints, alongside MSMD and the free-water metric MSFW. DELTA-SVD is optimised for longitudinal processing and adds several other improvements, including better elimination of CSF partial-volume effects and a built-in quality-control report. See the method publication ([Citation](index.md#citation)) for details.

Because of these methodological differences, the PSMD values produced by DELTA-SVD are **not** equivalent to those from the original PSMD tool, and the two cannot be compared or combined. Do not mix DELTA-SVD results with values from the original PSMD pipeline in the same analysis; process all data for a given study with a single pipeline.

## Do I need to preprocess my data first?

**Yes.** DELTA-SVD analyses already-preprocessed diffusion MRI. It takes a 4D DWI image, its `bval`/`bvec` files, and a DWI-space brain mask, and does **not** perform denoising, susceptibility-distortion, eddy-current or motion correction itself. Apply a state-of-the-art preprocessing pipeline first. See [Data requirements](requirements.md), which also points to turnkey options such as QSIPrep and MRtrix3.

## Can I use multi-shell data?

**Yes.** DELTA-SVD fits the tensor on b-values around b = 1000 s/mm² (default range 800–1200, together with the b ≈ 0 volumes) and selects the appropriate shell(s) from your data automatically, so multi-shell acquisitions are fine. If you need a different selection, `--bRange` sets the limits and `--shells` names the shells individually — the latter is the safer choice on multi-shell data, since it does not pull in the shells lying between the ones you want. The b ≈ 0 volumes are always kept either way. See [Data requirements](requirements.md) and [Selecting the b-values](advanced-usage.md#selecting-the-b-values).

## How do I exclude a region, or restrict the analysis to specific ROIs?

Use an exclusion mask (`--Emask`) to remove a region, for example an acute infarct or a haemorrhage, from the analysis. To report metrics for specific regions instead, supply ROI masks in DWI space (`--Rmask`; integer labels define separate ROIs) or a single ROI mask in MNI space (`--RmaskMNI`). Add `--hemispheres` to also report the skeleton split by left and right hemisphere. See [Restricting the analysis with masks](usage.md#restricting-the-analysis-with-masks).

## Do I need a GPU, or to install FSL, ANTs or Python?

No. DELTA-SVD is CPU-only, with no GPU requirement. Everything the pipeline needs (FSL, ANTs, and the Python scientific stack) is bundled in the container image, so nothing else has to be installed; you only need a container runtime. See [Installation](install.md).

## Does DELTA-SVD run on Windows or macOS?

The image is built for Linux x86-64 (`linux/amd64`). It runs natively on a Linux x86-64 host (the recommended platform) and on Windows via WSL2, either Apptainer inside a WSL2 distribution or Docker Desktop with its WSL2 backend. On Apple Silicon Macs (M-series) it runs only under x86-64 emulation, which is slow and not recommended; an Intel Mac runs it natively.

## My output files are owned by uid 999 (or root) under Docker. How do I fix that?

That happens because the container runs as its own non-root user. Run it as yourself with `--user "$(id -u):$(id -g)"` (see [Running under Docker](advanced-usage.md#running-under-docker)). Apptainer and rootless Podman avoid this entirely: they map your host identity into the container, so output is owned by you automatically.

## How many CPU cores should I request on a cluster?

For cross-sectional runs, **one core** is enough: there is no registration step at all.

For longitudinal runs, **one core per subject** is usually the most efficient choice: it processes many subjects side by side for the fewest core-hours. If a single subject's turnaround matters instead, **12 to 24 cores** is the sweet spot. Larger allocations help far less than they look like they should, because a substantial part of every run is single-threaded and takes the same time however many cores it has, so those cores sit idle for that whole stretch.

See [Running on an HPC cluster](advanced-usage.md#running-on-an-hpc-cluster).

## The pipeline refuses to overwrite existing output. What do I do?

By design, a run will not overwrite existing results. Pass `--reprocess` to allow reprocessing over existing output; you can optionally give it an alternative results-CSV base name to preserve the previous `delta-svd_results.csv`. See [Advanced usage](advanced-usage.md#other-options).

## Can I use DELTA-SVD commercially?

**No.** DELTA-SVD is for non-commercial research use only. The original source code and documentation are licensed under CC BY-NC-ND 4.0 (non-commercial, no derivatives). In addition, the container image bundles FSL, which is licensed for non-commercial use only, and that restriction applies to the image as a whole. See [LICENSE](https://github.com/isdneuroimaging/DELTA-SVD/blob/main/LICENSE) and [NOTICE](https://github.com/isdneuroimaging/DELTA-SVD/blob/main/NOTICE).

## Is my data sent anywhere?

**No.** DELTA-SVD runs entirely on your own machine and does not transmit your data. Network access is needed only once, to pull the container image; after that it runs fully offline and can be used on an air-gapped system.
