---
icon: lucide/play
---

# Usage

DELTA-SVD runs as a container. The container's entry point is the main pipeline script `delta-svd.py`, so the arguments you pass after the image name are the pipeline's arguments.

```
apptainer run delta-svd.sif --dwi <image> --id <subject> [options]
```

Under **Apptainer** (recommended) or rootless **Podman**, output files come out owned by you. Under **Docker** you must bind-mount your data (`-v`) and take an extra step to get output owned by your host user; see [Advanced usage](advanced-usage.md).

> [!NOTE]
> Apptainer automatically mounts your home directory, the current working directory, and `/tmp`, so data under any of those is visible with no extra flags. The examples below are run from the folder that holds your data (mounted automatically as the working directory), so the file names need no path. For data on another filesystem (e.g. a `/data` or scratch mount not under your home), bind it in explicitly with `--bind`/`-B`, for example `apptainer run --bind /data delta-svd.sif …`.

## Inputs

A run processes one subject, given one diffusion-weighted image per **timepoint**.

| Option | Required | Description |
| --- | --- | --- |
| `--dwi` | **yes** | Path(s) to 4D, preprocessed DWI NIfTI image(s), one per timepoint. Passing more than one triggers longitudinal processing. |
| `--bval` / `--bvec` | no | FSL-format gradient files. If omitted, they are inferred from each DWI path by swapping the extension for `.bval` / `.bvec`. |
| `--bmask` | no | DWI-space brain mask(s). Binarised on input: values greater than zero become 1; zero and negative values become 0. If omitted, inferred by swapping the DWI extension for `_brainmask.nii.gz`, falling back to `_brainmask.nii` if that file does not exist. |
| `--tp` | no | Timepoint label(s), which must be unique (`all` is reserved for the rows summarising all timepoints). Default: `TP01`, `TP02`, … in the order given. |
| `--id` | no | Subject identifier; added as an `ID` column to the results table. Recommended; it makes [aggregation](#aggregating-across-subjects) across subjects clean. |
| `-o`, `--dirOutput` | no | Output folder. Default: the parent folder of the first `--dwi` image. |

For `--bval`, `--bvec` and `--bmask`, you may give one value (applied to all timepoints) or one per timepoint. When your files follow the naming convention above, you can omit them entirely.

## Cross-sectional (single timepoint)

Run from the folder holding `sub-01_dwi.nii.gz`, `sub-01_dwi.bval`, `sub-01_dwi.bvec` and `sub-01_dwi_brainmask.nii.gz`; everything but the DWI is then inferred:

```
apptainer run delta-svd.sif \
  --dwi sub-01_dwi.nii.gz \
  --id sub-01
```

## Longitudinal (multiple timepoints)

List all the timepoints' images after a single `--dwi` (one image per timepoint). DELTA-SVD builds a within-subject template with ANTs and derives the skeleton on it, so metrics are directly comparable across timepoints:

```
apptainer run delta-svd.sif \
  --dwi ses-1_dwi.nii.gz ses-2_dwi.nii.gz \
  --tp  ses-1 ses-2 \
  --id  sub-01
```

> [!NOTE]
> The within-subject template step is the most CPU-intensive part of the pipeline, and runs only for longitudinal input. For sizing a run, and for cluster use, see [Advanced usage](advanced-usage.md#cpu-usage-and-threading).

## Restricting the analysis with masks

All masks are optional. Per-timepoint masks are given in DWI space (one per timepoint, in the same order as `--dwi`; write `NA` to skip a timepoint) and are merged in template space.

| Option | Description |
| --- | --- |
| `--Emask` | Exclusion mask(s): the masked region (e.g. a lesion) is removed from the analysis. Binarised on input: values greater than zero become 1; zero and negative values become 0. |
| `--Rmask` | ROI mask(s) in DWI space. Integer labels define separate ROIs, each analysed on its own. |
| `--RmaskMNI` | A single ROI mask in MNI space (may hold several integer labels). |
| `--hemispheres` | Additionally report skeleton metrics separately for the left and right hemispheres. |

## Output

Written to the output folder:

- **`delta-svd_results.csv`** — the metrics table. The validated endpoints are **MSMD** (mean skeletonised MD), **PSMD** (peak width of skeletonised MD) and **MSFW** (mean skeletonised free water), reported per timepoint and per region.
- **`delta-svd_qc.html`** — a quality-control report (skeleton and masks overlaid on the data). It also records the DELTA-SVD version and the exact command line that produced the run, so results stay traceable; see [Checking which version you have](install.md#checking-which-version-you-have). Control it with `--qc`: `1` (default) writes the HTML, `2` also keeps the underlying NIfTI images in a `delta-svd_qc/` folder, `0` skips both.

Intermediate files are written under `delta-svd_temp/` and deleted on success; pass `--debug` to keep them.

## Aggregating across subjects

`delta-svd_aggregate_results.py` collects the per-subject `delta-svd_results.csv` tables into one. It is a second script in the image, so run it with `apptainer exec`. Run from the directory that holds your per-subject output folders and point it at the current directory (`.`):

```
apptainer exec delta-svd.sif \
  delta-svd_aggregate_results.py . -o study_aggregated.csv
```

It searches the given directory recursively for `delta-svd_results.csv` and concatenates the matches. Useful options:

| Option | Description |
| --- | --- |
| `-f <pattern>` | CSV file name or pattern to search for (default `delta-svd_results.csv`); wildcards are allowed, but the pattern must end in `.csv`. |
| `-d <n>` | Search depth (default: any depth). |
| `-o <path>` | Output CSV path (must end in `.csv`), or an existing output directory; a bare filename lands in the search directory. |
| `-s` | Split output into separate `_metrics` and `_debugging` tables. |
| `-p` | Add a `path` column identifying each source file (added automatically when the `ID` column is absent). |
| `-x` | Overwrite an existing output file. |
| `-t <when>` | Append the current `date`, `time` or `datetime` to the output file name; an alternative to `-x` when repeating an aggregation. |
| `-q` | Quiet mode. |

> [!TIP]
> Run each subject with `--id`, so the aggregated table carries a clean `ID` column instead of falling back to file paths.
