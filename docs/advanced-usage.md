---
icon: lucide/sliders-horizontal
---

# Advanced usage

## CPU usage and threading

Two steps use more than one core:

- the **diffusion tensor and free-water fit**, spread across worker processes;
- the **within-subject template construction**, which runs only for longitudinal input (more than one timepoint) and registers the timepoints in parallel.

TBSS and the remaining steps are single-threaded, and a cross-sectional run performs no registration at all.

`--threads` sets how many physical CPU cores DELTA-SVD may use:

| Value | Behaviour |
| --- | --- |
| `auto` (default) | Detect the physical cores available to the run. On a cluster this honours the cores your scheduler assigned, so a correctly sized job needs no setting. |
| `N` | Cap usage at `N` cores. |
| `1` | Use a single core. |

Work is spread across the available cores automatically; you do not need to tune this by hand.

> [!NOTE]
> Only physical cores are counted; hyperthreads are ignored, because registration gains little from them. To use them anyway, pass an explicit `--threads` value.

### Choosing a value

- **Single subject on a workstation** — leave the default, or set `--threads` to the number of cores you want to devote to the run.
- **Many subjects on a cluster** — see [Running on an HPC cluster](#running-on-an-hpc-cluster) below.
- **Limited memory** — peak memory grows with the number of registrations running at once. Lower it with `--para` (`--para 1` runs one registration at a time, the lowest-memory setting).

## Reproducibility

Registration divides its similarity metric across threads and sums the parts, so the thread count changes the order of that summation and with it the last bits of the result. That would normally be negligible, but the white matter skeleton is derived by thresholding an interpolated mask, which turns those last bits into whole voxels moving in or out of the skeleton, enough to shift the longitudinal metrics measurably.

DELTA-SVD therefore fixes the registration thread count at the value the method was validated with, instead of deriving it from the cores available. This is why `--threads` and `--para` affect only runtime and memory: they decide how many registrations run at once, never how each one is computed.

`--itkThreads` overrides that count. It exists for method development and is deliberately not listed in `--help`. **Do not change it**: results produced with a different value cannot be compared with, or pooled with, results produced at the default — unlike a DELTA-SVD `PATCH` version, there is no safe value to deviate to; only the default is validated.

## Running on an HPC cluster

Because only the two steps above are multi-core (and one of them only for longitudinal input), a large allocation sits idle for much of a run. Sizing the request is a trade of wall time against core-hours, and matters more than tuning any option.

### Cross-sectional runs

Allocate **one core**. There is no registration step, and nothing to gain from more.

### Longitudinal runs

**For throughput (the usual case), allocate one core per subject** and run many side by side. The registration still starts its 12 threads; the operating system time-slices them onto that one core at a cost of a few percent. This is much the cheapest in core-hours, because no cores sit idle through the single-threaded phases.

**For latency, use 12 to 24 cores**: 12 runs one registration at a time, 16 and above run two. Beyond that, returns fade fast: registration is capped at one job per timepoint, and the single-threaded part of the run takes the same time however many cores it has. Go higher only with many timepoints, and only after measuring that template construction dominates your runs.

```bash
#SBATCH --cpus-per-task=1          # throughput: one core per subject
#SBATCH --mem=8G                   # size against your image matrix; see below
apptainer run delta-svd.sif --dwi ses-1.nii.gz ses-2.nii.gz --tp ses-1 ses-2 --id sub-01
```

Leave `--threads` at `auto`; it reads the cores your scheduler assigned.

> [!NOTE]
> A one-core job will still start 12 registration threads. This is expected, not a misconfiguration, and the scheduler handles it correctly.

> [!TIP]
> **Memory, not CPU, usually limits how many subjects you can pack onto a node.** Each concurrent subject holds its own working set. Benchmark one representative subject at your intended packing density before committing to a cluster-wide setting; runtimes depend strongly on CPU model, memory bandwidth and storage.

## Running under Docker

The image runs as a non-root user (`nonroot`, uid 999). Under Docker the process therefore writes files owned by `999:999` on the host. To get output owned by your host user, run the container as yourself:

```
docker run --rm --user "$(id -u):$(id -g)" \
  -v /path/to/data:/data \
  ghcr.io/isdneuroimaging/delta-svd:<version> --dwi /data/sub-01_dwi.nii.gz --id sub-01
```

If a step complains about an unset `HOME`, add `-e HOME=/tmp`. To run the aggregator under Docker, override the entry point:

```
docker run --rm -v /path/to/data:/data \
  --entrypoint delta-svd_aggregate_results.py \
  ghcr.io/isdneuroimaging/delta-svd:<version> /data -o /data/study_aggregated.csv
```

Apptainer and rootless Podman map your host identity into the container, so they need none of this: output is owned by you automatically.

## Other options

| Option | Description |
| --- | --- |
| `--reprocess [name]` | Allow reprocessing over existing output (otherwise a run refuses to overwrite). Optionally give an alternative results-CSV base name to preserve the previous `delta-svd_results.csv`. |
| `--debug` | Keep the `delta-svd_temp/` folder of intermediate files instead of deleting it. |
| `--bRange LO HI` | b-value range used for tensor fitting (default `800 1200`). |
| `--skeletonMask <NIfTI>` | Use an alternative skeleton mask instead of the validated default. Binarised on input: values greater than zero become 1; zero and negative values become 0. |
| `--para <n>` | Number of ANTs registration jobs run at once during longitudinal template construction. Derived from the `--threads` budget by default and capped at the number of timepoints. Peak memory scales with it, so `--para 1` is the lowest-memory setting. |
