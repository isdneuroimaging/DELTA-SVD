---
icon: lucide/database
---

# Data requirements

DELTA-SVD analyses multi-directional diffusion MRI that has already been preprocessed. It does not perform image preprocessing itself. The guidance below describes acquisitions and preprocessing that make data well suited to the pipeline. It reflects current best practice rather than a single mandated protocol: sensible deviations are fine, but these recommendations are a safe default.

## Required input

Each timepoint is described by four files. The DWI must already be **preprocessed** (see [Preprocessing](#preprocessing) below); DELTA-SVD does not preprocess it for you.

- **DWI image** — a 4D, preprocessed diffusion-weighted NIfTI (`.nii.gz` or `.nii`).
- **b-values** — an FSL-format `bval` file matching the DWI.
- **b-vectors** — an FSL-format `bvec` file matching the DWI.
- **Brain mask** — preferably a binary brain-mask NIfTI (`.nii.gz` or `.nii`) in DWI space (aligned with the DWI image). DELTA-SVD binarises masks on input: values greater than zero become 1; zero and negative values become 0.

A longitudinal run takes one such set per timepoint. For how to pass these on the command line (and the naming convention that lets the `bval`/`bvec`/mask be inferred from the DWI path), see [Usage](usage.md#inputs).

## Acquisition

- **Field strength** — 3T is recommended.
- **Resolution and directions** — a recommended minimum of 2 mm isotropic voxels and at least 20 diffusion-weighting directions. Higher resolution and more directions improve the tensor estimates, at the cost of scan time. DELTA-SVD refuses data that leaves it fewer than 12 distinct directions after the b-value selection, and warns below the recommended 20: the free-water fraction is estimated against the residual of the tensor fit, so it needs angular sampling well beyond the six directions the tensor itself requires. Repeated acquisitions of the same direction average noise but add no further constraint, and are counted once.
- **b-value** — include at least one shell suitable for diffusion-tensor fitting, around b = 1000 s/mm². By default DELTA-SVD fits the tensor on b-values in the range 800–1200 s/mm² (together with the b ≈ 0 volumes) and automatically selects those shell(s) from the data, so multi-shell acquisitions are fine, and the appropriate shell is picked for you. A different selection can be made with `--bRange` or `--shells`, within the 250–1800 s/mm² window the diffusion tensor model is valid in (see [Advanced usage](advanced-usage.md#selecting-the-b-values)).
- **Direction scheme** — sample the diffusion directions over the whole sphere rather than a half-sphere (a custom whole-sphere direction set is recommended). Full-sphere sampling gives eddy-current and motion correction (FSL `eddy`) better-conditioned data to work with.
- **Reversed phase encoding** — acquire at least one b ≈ 0 volume (or a short series) with reversed phase-encoding direction, so that susceptibility-induced distortions can be corrected (FSL `topup`).

## Preprocessing

DELTA-SVD expects preprocessed input, so apply a state-of-the-art pipeline before running it:

- **Visual QC of the raw data** — inspect for artefacts and gross problems before preprocessing.
- **Conversion** — convert from DICOM with a tool that preserves the metadata required for preprocessing (phase-encoding direction, total readout time, gradient table), for example [`dcm2niix`](https://github.com/rordenlab/dcm2niix) with BIDS sidecars. If you intend to continue with [MRtrix3](https://www.mrtrix.org/), converting with `mrconvert` to the `.mif` format is a convenient alternative, as it embeds this metadata directly in the image file.
- **Core steps** — denoising (e.g. MP-PCA, via [MRtrix3](https://www.mrtrix.org/)'s `dwidenoise`) and Gibbs-ringing removal (MRtrix3's `mrdegibbs`), followed by susceptibility-distortion correction (FSL `topup`) and eddy-current/motion correction (FSL `eddy`).
- **Brain mask** — produce a DWI-space brain mask; DELTA-SVD takes it as input.

> [!TIP]
> Two established options can run the whole preprocessing chain for you: [QSIPrep](https://github.com/PennLINC/qsiprep) bundles these steps into a robust, reproducible, and well-documented workflow; alternatively they can be done end-to-end with [MRtrix3](https://www.mrtrix.org/) (`dwidenoise`, `mrdegibbs`, and `dwifslpreproc`, which wraps FSL `topup`/`eddy`). Either is a convenient way to obtain DELTA-SVD-ready data.
