# Contributing to DELTA-SVD

Thanks for your interest in DELTA-SVD. This guide covers working on the code and on the documentation.

> [!IMPORTANT]
> DELTA-SVD's own code is released under CC BY-NC-ND 4.0 (see [`LICENSE`](LICENSE)), whose NoDerivatives term makes outside contributions legally nuanced. If you'd like to contribute code, please open an issue to discuss it with the maintainers first.

## Code

### Validation status: read this first

DELTA-SVD is a **clinically and technically validated** tool: its endpoints (MSMD, PSMD, MSFW) were validated as produced by a specific version of this pipeline, and that validation holds only as long as the numbers stay the same.

**Any change that can alter the computed metric values invalidates the validation and requires a new formal validation before release.** This includes:

- the processing algorithms and their parameters (tensor/free-water fitting, registration, skeletonisation, statistic extraction);
- the default skeleton mask and other bundled reference data;
- version changes to the scientific stack that does the maths (FSL, ANTs, and the pinned conda packages such as numpy, scipy, dipy, nibabel).

Changes that provably leave every metric untouched (documentation, packaging, tests, or refactors verified to produce byte-identical output) do not need re-validation. When in doubt, assume a change is metric-affecting and raise it with the maintainers first. This is also why `container/scripts/markvcid_fw_mrn.py` is kept verbatim (see [Conventions](#conventions)).

#### External hazards: settings that move the metrics from outside the code

Three quantities shift the endpoints without any source change, because they alter how the *same* arithmetic is executed. All three are pinned in this repository; unpinning any of them, or changing what it's pinned to, is a metric-affecting change requiring re-validation.

| Hazard | Why it moves the numbers | Pinned as |
| --- | --- | --- |
| **ITK threads per registration job** | ITK sums the registration metric per thread, so the count sets the summation order | `ITK_THREADS_DEFAULT = 12` in `delta-svd.py`, overridable only via the hidden `--itkThreads` |
| **BLAS/LAPACK library version** | `np.linalg.pinv` in the free-water fit changes by a few bits between releases | the four hand-maintained BLAS lines in `conda-explicit-linux-64.txt` |
| **BLAS kernel selected for the CPU** | `libopenblas` is a `DYNAMIC_ARCH` build and picks kernels from the CPU's features, so `pinv` differs between kernel families | `ENV OPENBLAS_CORETYPE=Haswell` in the `Dockerfile` |

A last-bit difference in the fitted tensors nudges the deformation field, and the skeleton (thresholding an *interpolated binary* mask at exactly 1) converts that into whole voxels entering or leaving — nothing absorbs it, so it reaches the metrics. If the reference subject ever moves without an obvious cause, check these three before suspecting the code.

#### Checking whether a change moved the numbers

"Provably untouched" means measured, not assumed: nothing in the test suite checks the metric values. For anything that plausibly reaches the numbers (a regenerated conda lock, an FSL or ANTs version bump, an edit to the fitting, masking or skeletonisation code, a change to the `sed` patches in the `Dockerfile`), build the image before and after the change, process the same representative subject with each, and diff the two `delta-svd_results.csv` tables:

```bash
.venv-test/bin/python tools/compare_results.py \
    before/delta-svd_results.csv after/delta-svd_results.csv
```

It runs from the [test virtual environment](#tests), which supplies the numpy and pandas it needs. It compares every metric value and every skeleton voxel count and exits non-zero if anything moved; `--help` covers the rest, including `--ignore-key` for a column that was renamed without the numbers changing.

Everything is compared **exactly**, with no tolerance option: a changed skeleton is a changed result even when the metrics happen to round the same way. A clean run is the evidence that a change is not metric-affecting; any reported difference means re-validation applies.

Rules for the runs being compared:

- **Leave `--threads` and `--para` alone** — they don't affect the numbers. The hidden `--itkThreads` does, and must stay at its default (`ITK_THREADS_DEFAULT = 12`, the value the method was validated at) for any comparison to mean anything.
- **Use a longitudinal subject.** A cross-sectional run alone is not sufficient.
- **Judge by the longitudinal change (ΔPSMD), not the per-timepoint values.** ΔPSMD is by far the most sensitive readout: a diff that looks negligible per timepoint can still be a large change in the endpoint the pipeline exists to produce.

> [!WARNING]
> A cross-sectional-only check is not sufficient. The longitudinal path turns arbitrarily small numerical differences into discrete, reportable ones: the skeleton comes from thresholding an interpolated *binary* brain mask at exactly 1, so every boundary voxel sits on a knife edge, and a sub-voxel shift in the deformation re-decides those ties. The cross-sectional path has no such step and can absorb the same change completely.

### Repository layout

- [`container/`](container/) — the container image: `Dockerfile`, `build.sh`, the pinned conda environment, and the pipeline scripts under `container/scripts/`.
- [`tests/`](tests/) — pytest unit tests for those scripts.
- [`tools/`](tools/) — maintainer utilities that are not part of the image, such as `compare_results.py` above.
- [`docs/`](docs/) — documentation sources (see below).
- [`overrides/`](overrides/) — theme template overrides for the documentation site (see below).

### Workflow

- Development happens on a feature branch; open pull requests against `main`.
- [`VERSION`](VERSION) is the single source of truth for the release version; bump it in the same change that cuts a release.
- Keep the working tree clean before building a release image: `build.sh` marks the image revision `-dirty` when there is any uncommitted change, including an untracked file.

### Building the image

```bash
container/build.sh                 # -> delta-svd:<VERSION>
container/build.sh delta-svd:dev   # custom tag; extra args pass through to docker build
```

`build.sh` reads the version from `VERSION`, builds for `linux/amd64` (the only platform the pinned stack resolves for), and stamps OCI provenance labels (git revision and build date) into the image.

### Releasing

Releases are built and staged by CI ([`release-build.yml`](.github/workflows/release-build.yml)), validated by hand against the staged digest, then published by a second CI workflow ([`release-promote.yml`](.github/workflows/release-promote.yml)) that copies the validated manifest into the production package. Nothing reaches `ghcr.io/isdneuroimaging/delta-svd` without a human having checked the exact digest first.

Expect rebuilding an *older* commit to fail outright rather than merely differ: the `apt` pins resolve against the live Ubuntu archive, and `ca-certificates`' version is itself a date, so the pin stops matching as soon as the archive moves on. **The pushed image digest, not the source tree, is the artefact of record for a release.** Recover an old release by pulling its digest, not by rebuilding its tag.

Only exact version tags are published. There is **no `latest` tag**: results from different versions must not be pooled, so no run should be able to pick up a new version by accident.

1. **Bump [`VERSION`](VERSION)** and commit it. Last digit only for a change that provably leaves every metric untouched (see [Validation status](#validation-status-read-this-first)); otherwise bump `MAJOR.MINOR`, which is what tells users their results cannot be pooled with earlier ones.

    `VERSION` is the single source of truth, but two files restate it by hand because their formats cannot interpolate, and both have to move in the same commit:

    - [`CITATION.cff`](CITATION.cff) — its `version:` field, which GitHub renders in "Cite this repository";
    - [`docs/install.md`](docs/install.md) — the image tag in the `apptainer pull` / `docker pull` / verification commands, which otherwise keeps handing users the *previous* image.

    `tests/test_version.py` fails if either disagrees with `VERSION`. Everything else derives from it automatically: `release-build.yml` stamps the OCI label the same way `build.sh` does for a local build, and the `Dockerfile` copies the file next to the scripts, which is what `--version` reports.

2. **Tag the release commit and push the tag.** This triggers `release-build.yml`, which checks the tag against `VERSION`, builds the image (same semantics as a local `build.sh` run), and pushes it — attested via `actions/attest-build-provenance` — to a public **staging** package, `ghcr.io/isdneuroimaging/delta-svd-staging`, tagged by commit SHA only so it can never be mistaken for a release.

    ```bash
    git tag -a "v$(tr -d '[:space:]' < VERSION)" -m "DELTA-SVD $(tr -d '[:space:]' < VERSION)"
    git push origin "v$(tr -d '[:space:]' < VERSION)"
    ```

    Watch the run's job summary for the staged digest — everything from here on refers to that digest, not the tag.

    If validation below fails and needs a fix, delete the tag, commit the fix, and re-tag: nothing has reached the production package yet, so moving the tag is safe.

3. **Validate**, unless the change is provably not metric-affecting. Run the longitudinal comparison from [Validation status](#validation-status-read-this-first) using *the staged digest*, not a local rebuild:

    ```bash
    DIGEST="sha256:..."   # from the release-build.yml job summary
    docker pull "ghcr.io/isdneuroimaging/delta-svd-staging@$DIGEST"
    ```

4. **Promote.** Once validated, run `release-promote.yml` with that digest — Actions tab → "Run workflow", or `gh workflow run release-promote.yml -f digest="$DIGEST"`. It re-reads the image's own `version`/`revision` labels (refusing anything malformed, `unknown`, or `-dirty`), copies the manifest — never a rebuild — to `ghcr.io/isdneuroimaging/delta-svd:<version>` at the *same* digest, and opens a draft GitHub release recording it.

    **On the first push ever to the production package**, GHCR creates it private, so `docker pull` / `apptainer pull` fail for everyone outside the organisation until it's made public by hand — package page → Package settings → Danger Zone → Change visibility. Check from an unauthenticated shell:

    ```bash
    curl -s -o /dev/null -w '%{http_code}\n' \
        'https://ghcr.io/token?scope=repository:isdneuroimaging/delta-svd:pull&service=ghcr.io'
    ```

    `200` means public; `401` means still private.

5. **Publish the draft release** from the web UI, or `gh release edit "v$VER" --draft=false`.

### Tests

The unit tests exercise the Python scripts in `container/scripts/` with pytest. They cover script logic (argument handling and validation, input path resolution, mask merging, metric naming, results-table assembly, and the QC report) and are deliberately insensitive to the exact dependency versions: the suite passes unchanged both on the stack pinned for the image (Python 3.11, numpy 1.24, dipy 1.5) and on a current one (Python 3.14, numpy 2.5, dipy 1.12).

Two separate things need pinning, and they cover different risks:

- **Python version** (3.11) guards syntax and stdlib compatibility with the shipped interpreter. `ast.parse(source, feature_version=(3, 11))` does not reliably catch version-specific syntax, since its tokenizer runs before the feature-version check — only compiling under a real 3.11 interpreter does.
- **The conda lock** guards the numerical behaviour of the scientific stack. `tests/requirements.txt` installs numpy/dipy/nibabel unpinned from PyPI, so pinning Python alone doesn't cover this; use [`tests/run-in-locked-env.sh`](#tests) below for that.

> [!IMPORTANT]
> The suite does **not** check the metric values the pipeline produces, so a green run says nothing about whether a change is metric-affecting. That judgement, and any re-validation it triggers, still rests on review; see [Validation status](#validation-status-read-this-first).

Run them in a virtual environment; the dependencies are listed in [`tests/requirements.txt`](tests/requirements.txt) and are pulled from PyPI, independently of the container's conda environment. `python3 -m venv` picks up whatever `python3` is on your machine, so pin it to 3.11 if that differs, e.g. with [uv](https://docs.astral.sh/uv/): `uv venv --python 3.11 .venv-test`:

```bash
python3 -m venv .venv-test
.venv-test/bin/pip install -r tests/requirements.txt
.venv-test/bin/python -m pytest
```

The `.venv-test/` directory is git-ignored. Run the suite before opening a pull request and add tests for new behaviour; CI runs it too, on Python 3.11 and 3.14, on every pull request and push to `main`.

CI also compiles every module under Python 3.11 as a fast tripwire for version-specific syntax. Check it locally with just Docker:

```bash
docker run --rm -v "$PWD:/repo:ro" -w /repo -e PYTHONPYCACHEPREFIX=/tmp/pyc \
    python:3.11-slim python -m compileall -q container/scripts tools tests
```

To run the same suite against the *pinned* stack instead (worth doing after changing the conda lock, where a dependency bump could remove an API the scripts rely on), use [`tests/run-in-locked-env.sh`](tests/run-in-locked-env.sh), which builds a throwaway environment from the lock in a container. See the header of that script for details.

### Container dependencies

Runtime dependencies are pinned as an explicit conda environment in `container/conda-explicit-linux-64.txt`. Regenerate that lock rather than hand-editing it when dependencies change, so it stays reproducible. Regeneration is a metric-affecting change in its own right, even when no direct dependency was touched: it can move any transitive package, and the numerics ones reach the numbers.

There is one deliberate exception, documented in the lock's own header: the four BLAS/LAPACK entries (`libopenblas`, `libblas`, `libcblas`, `liblapack`) are **hand-maintained**, pinned to the versions in the container the method was validated against. After any regeneration, re-apply those four URLs and re-run the longitudinal comparison above.

### Conventions

- The `Dockerfile` is kept free of hadolint warnings or errors (info-level findings, such as `DL3066` on the non-numeric `USER nonroot`, are tolerated): run `hadolint container/Dockerfile` before committing changes to it. CI enforces this with `hadolint/hadolint-action@v3` at `failure-threshold: warning`.
- `container/scripts/markvcid_fw_mrn.py` is vendored third-party code (see [`NOTICE`](NOTICE) and `container/third-party-licenses/`). Keep it as-is (don't reformat or refactor it) so it stays traceable to upstream and doesn't need re-validation.

## Documentation

The documentation site is built with [Zensical](https://zensical.org/), a static site generator by the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) team. It reads the standard [`mkdocs.yml`](mkdocs.yml); all content lives in [`docs/`](docs/) as Markdown, which stays readable directly on GitHub. The site is published to GitHub Pages by the [`docs` workflow](.github/workflows/docs.yml) on every push to `main` that touches `docs/`, `overrides/`, `mkdocs.yml`, or the workflow itself. Pull requests touching those same paths run the build without deploying, so a broken link or a stale `nav:` entry fails the strict build on the pull request rather than on `main`.

### One-time setup

Create a virtual environment and install the docs build dependencies (kept separate from the project's runtime dependencies in [`docs/requirements.txt`](docs/requirements.txt)):

```bash
python3 -m venv .venv
.venv/bin/pip install -r docs/requirements.txt
```

The `.venv/` directory is git-ignored.

### Preview locally

Run a live-reloading server; it rebuilds automatically as you edit:

```bash
.venv/bin/zensical serve
```

Then open <http://127.0.0.1:8000/>. Or activate the environment once (`source .venv/bin/activate`) and drop the `.venv/bin/` prefix.

### Check before pushing

CI builds with `--strict`, which fails on broken links or nav entries. Run the same check locally first:

```bash
.venv/bin/zensical build --strict
```

### Editing content

- Add or edit Markdown files under `docs/`, then register new pages in the `nav:` section of [`mkdocs.yml`](mkdocs.yml) so they appear in the site navigation (and to keep the strict build happy).
- GitHub-style alerts work as-is: write `> [!NOTE]`, `> [!TIP]`, `> [!IMPORTANT]`, `> [!WARNING]`, or `> [!CAUTION]` and they render as native alerts on GitHub and as admonitions on the site.

### Link previews

Every page carries Open Graph and Twitter card tags, so a link to the site unfurls with a preview image, its page title, and a description instead of a bare URL. The theme emits none of these itself; they come from [`overrides/main.html`](overrides/main.html), which the `theme.custom_dir` setting in [`mkdocs.yml`](mkdocs.yml) layers over the stock templates.

All pages share one preview image, [`docs/assets/og-image.png`](docs/assets/og-image.png). It is committed to the repository, not generated during the build; [`tools/make_og_image.py`](tools/make_og_image.py) renders it (1200×630, the size every major scraper expects) from the pipeline figure on the landing page. Re-run it after changing that figure or the card design, on any Python that has [Pillow](https://pypi.org/project/pillow/) installed, and commit the result:

```bash
.venv/bin/pip install pillow
.venv/bin/python tools/make_og_image.py
```

The tags are absolute URLs built from `site_url`, which scrapers require, so they can only be checked against the deployed site — a local build serves them pointing at <https://delta-svd.com>. After deploying a change to them, confirm the result with a preview debugger such as [opengraph.xyz](https://www.opengraph.xyz/); note that Slack, LinkedIn, and the rest cache aggressively, so an unchanged preview is usually a stale cache rather than a broken tag.
