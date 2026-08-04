---
icon: lucide/compass
---

# Overview

DELTA-SVD is a containerised diffusion-MRI pipeline that turns preprocessed diffusion-weighted images into diffusion metrics designed for **longitudinal** tracking of change in cerebral small vessel disease (cSVD). It fits the diffusion (bi)-tensor, projects the resulting maps onto a white matter skeleton (TBSS-style), applies a custom mask and reports summary endpoints per timepoint and per region.

For a longitudinal run it builds a **within-subject template** so that the skeleton and the skeleton-derived metrics are directly comparable across timepoints, rather than re-deriving them independently at each visit.

## Pipeline at a glance

**Preprocessed DWI (per timepoint)** → **tensor fit → white-matter skeleton projection** → **endpoints table + QC report**

- **Input** — one preprocessed 4D DWI per timepoint, with its gradient table and a brain mask.
- **Processing** — diffusion-tensor fitting, then skeletonisation and masking; longitudinal runs first build a within-subject template so timepoints share one skeleton.
- **Output** — a metrics table (`delta-svd_results.csv`) and an HTML quality-control report.

## Typical input

Each timepoint is a preprocessed 4D DWI plus its b-values, b-vectors, and a DWI-space brain mask. DELTA-SVD does **not** preprocess the data itself. See **[Data requirements](requirements.md)** for acquisition guidance and the expected preprocessing.

## Usage in a nutshell

DELTA-SVD runs as a container; you pass the pipeline's arguments after the image name:

```
apptainer run delta-svd.sif --dwi <image> --id <subject>
```

Pass a single DWI for a **[cross-sectional](usage.md#cross-sectional-single-timepoint)** run, or several (one per timepoint) to trigger **[longitudinal](usage.md#longitudinal-multiple-timepoints)** processing. Full options, mask handling, and cross-subject aggregation are in **[Usage](usage.md)**.

## Outputs

- **`delta-svd_results.csv`** — the metrics table, reporting three validated endpoints per timepoint and per region:
    - **MSMD** — mean skeletonised mean diffusivity; the recommended primary endpoint for most datasets.
    - **PSMD** — peak width of skeletonised mean diffusivity; an established marker of white-matter damage in cSVD.
    - **MSFW** — mean skeletonised free water.
- **`delta-svd_qc.html`** — a quality-control report with the skeleton and masks overlaid on the data.

For guidance on which endpoint to report, see the **[FAQ](faq.md#what-are-msmd-psmd-and-msfw-and-which-should-i-report)**; for the full output detail and QC levels, see **[Usage](usage.md#output)**.
