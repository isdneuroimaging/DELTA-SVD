---
icon: lucide/download
---

# Installation

DELTA-SVD is distributed as a single container image: everything the pipeline needs (FSL, ANTs, and the Python scientific stack) is bundled, so there is nothing else to install. You only need a container runtime to pull and run it.

## Requirements

### Hardware

- **CPU** — an x86-64 (amd64) processor supporting **AVX2 and FMA** (the `x86-64-v3` level): Intel Haswell (2013) or newer, AMD Zen (2017) or newer. Note that Atom, Celeron and Pentium parts often lack these regardless of their age; check with `grep -o -m1 -E 'avx2|fma' /proc/cpuinfo`. DELTA-SVD reports the problem and stops if they are missing. Two steps use multiple cores: the diffusion tensor and free-water fit, and (for longitudinal input only) the within-subject template construction; the rest are single-threaded (see [Advanced usage](advanced-usage.md#cpu-usage-and-threading)).
- **Memory** — a single timepoint fits in a few GB of RAM. For longitudinal input, peak memory grows with the number of registrations run in parallel during template construction, so budget additional memory for each. On memory-constrained nodes, lower that count with `--para` (`--para 1` is the lowest-memory setting; see [Advanced usage](advanced-usage.md)).
- **Disk** — allow several GB for the image itself, plus working space for the intermediate files written to `delta-svd_temp/` during a run (removed on success unless you pass `--debug`).

> [!NOTE]
> Exact memory and disk needs depend on your image matrix size, the number of diffusion directions, and the number of timepoints. Start with a generous allocation and tighten it once you have measured a representative run.

### Software

- A **Linux x86-64 host** (the image is built for `linux/amd64`). This includes Windows via WSL2: Apptainer inside a WSL2 distribution, or Docker Desktop with its WSL2 backend.
- A **container runtime**:
    - **Apptainer** (or legacy **Singularity**) — recommended, especially on HPC clusters. It runs rootless and maps your host identity into the container, so output files come out owned by you.
    - **Docker**, or rootless **Podman** — an optional alternative. Under Docker there is an extra step to get output owned by your host user; see [Usage](usage.md) and [Advanced usage](advanced-usage.md).

No separate Python, FSL, or ANTs installation is required; those are all provided inside the image. The bundled third-party components are redistributed under their respective licenses; see [NOTICE](https://github.com/isdneuroimaging/DELTA-SVD/blob/main/NOTICE) for details and license texts.

## Getting the image

The image is published to the GitHub Container Registry at `ghcr.io/isdneuroimaging/delta-svd`, tagged with its release version. There is deliberately **no `latest` tag**: only results produced with the same version can be compared or pooled, so every run has to name the version it uses and none can silently pick up a newer one.

> [!IMPORTANT]
> **Use one version per project.** Only results produced with the same DELTA-SVD version can be compared or pooled. Choose a version at the start of a project and process all data with it; do not upgrade partway through. Version numbers follow `MAJOR.MINOR.PATCH` (e.g. `1.2.0`). The exception is bug-fix releases, which differ only in the last (`PATCH`) digit: these are safe to mix within a project, as they do not change results. Any change in the first two numbers can shift the metrics, so results from different `MAJOR.MINOR` versions must not be combined.

### Apptainer (recommended)

Pull the image and convert it to a local `.sif` file in one step:

```
apptainer pull delta-svd.sif docker://ghcr.io/isdneuroimaging/delta-svd:1.0.0
```

This writes `delta-svd.sif` into the current directory, the file used throughout the [Usage](usage.md) examples. Keep it somewhere stable (or on shared storage on a cluster) and point your runs at it.

### Docker or Podman (optional)

Pull the image into the local daemon's store:

```
docker pull ghcr.io/isdneuroimaging/delta-svd:1.0.0
```

Replace `docker` with `podman` to use rootless Podman instead.

## Verify the installation

Run the pipeline's help to confirm the image works:

```
apptainer run delta-svd.sif --help
```

or, with Docker:

```
docker run --rm ghcr.io/isdneuroimaging/delta-svd:1.0.0 --help
```

If you see the DELTA-SVD option help, you are ready to go; continue with [Usage](usage.md).

## Verify the image's provenance

Every release image is built and pushed by a GitHub Actions workflow that attaches a [Sigstore](https://www.sigstore.dev/)-signed build attestation, verifiable with the [GitHub CLI](https://cli.github.com/) (`gh`, version 2.49 or later):

```
gh attestation verify oci://ghcr.io/isdneuroimaging/delta-svd:1.0.0 --owner isdneuroimaging
```

A successful verification confirms the image was built by that workflow from the corresponding tagged commit in the [DELTA-SVD repository](https://github.com/isdneuroimaging/DELTA-SVD), not assembled or pushed by hand.

## Checking which version you have

Because results from different versions must not be pooled, it is worth being able to confirm which one a `.sif` file or an image tag actually is. Pass `--version`:

```
apptainer run delta-svd.sif --version
```

It prints `DELTA-SVD <version>` and exits. The same works for the aggregator (`apptainer exec delta-svd.sif delta-svd_aggregate_results.py --version`) and under Docker (`docker run --rm ghcr.io/isdneuroimaging/delta-svd:1.0.0 --version`).

Every run also reports its version in two other places, so results can be traced back after the fact:

- the **first line of the run's console output**, ahead of the command line;
- the **QC report** (`delta-svd_qc.html`), in the table at the top.
