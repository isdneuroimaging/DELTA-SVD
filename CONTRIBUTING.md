# Contributing to DELTA-SVD

Thanks for your interest in DELTA-SVD. This guide covers working on the code and on the documentation.

> [!IMPORTANT]
> DELTA-SVD's own code is released under CC BY-NC-ND 4.0 (see [`LICENSE`](LICENSE)), whose NoDerivatives term makes outside contributions legally nuanced. If you'd like to contribute code, please open an issue to discuss it with the maintainers first.

## Code

### Validation status: read this first

DELTA-SVD is a **clinically and technically validated** tool: its whole-skeleton endpoints (MSMD, PSMD, MSFW) were validated as produced by a specific version of this pipeline, and that validation holds only as long as those numbers stay the same.

**Any change that can alter the validated whole-skeleton endpoint values invalidates the validation and requires a new formal validation before release.** This includes:

- the processing algorithms and their parameters (tensor/free-water fitting, registration, skeletonisation, statistic extraction);
- the default skeleton mask and other bundled reference data;
- version changes to the scientific stack that does the maths (FSL, ANTs, and the pinned conda packages such as numpy, scipy, dipy, nibabel).

Changes that provably leave those endpoints untouched (documentation, packaging, tests, or refactors verified to produce identical standard-run output) do not need re-validation. A correction confined to optional derived rows, such as custom-ROI results, may use a patch version only when the affected inputs and pooling restriction are documented explicitly; it does not make those affected rows interchangeable across the patch. When in doubt, assume a change is endpoint-affecting and raise it with the maintainers first. This is also why `container/scripts/markvcid_fw_mrn.py` is kept verbatim (see [Conventions](#conventions)).

#### External hazards: settings that move the metrics from outside the code

Three quantities shift the endpoints without any source change, because they alter how the *same* arithmetic is executed. All three are pinned in this repository; unpinning any of them, or changing what it's pinned to, is a metric-affecting change requiring re-validation.

| Hazard | Why it moves the numbers | Pinned as |
| --- | --- | --- |
| **ITK threads per registration job** | see [Reproducibility](docs/advanced-usage.md#reproducibility) | `ITK_THREADS_DEFAULT = 12` in `delta_svd_constants.py`, overridable only via the hidden `--itkThreads` |
| **BLAS/LAPACK library version** | `np.linalg.pinv` in the free-water fit changes by a few bits between releases | the four hand-maintained BLAS lines in `conda-explicit-linux-64.txt` |
| **BLAS kernel selected for the CPU** | `libopenblas` is a `DYNAMIC_ARCH` build and picks kernels from the CPU's features, so `pinv` differs between kernel families | `OPENBLAS_CORETYPE=Haswell` in `IMAGE_ENVIRONMENT` (`delta_svd_constants.py`), mirrored as `ENV` in the `Dockerfile` |

A last-bit difference in the fitted tensors nudges the deformation field, and the skeleton (thresholding an *interpolated binary* mask at exactly 1) converts that into whole voxels entering or leaving — nothing absorbs it, so it reaches the metrics. If the reference subject ever moves without an obvious cause, check these three before suspecting the code.

The image's tool selection is pinned the same way. The `Dockerfile` `ENV` values can still be replaced at run time (`docker run -e`, Apptainer's `--env` or `APPTAINERENV_*`), and Apptainer passes in host variables the image does not set and mounts the host's home, so `delta-svd.py` assigns everything in `IMAGE_ENVIRONMENT` (`FSLDIR`, `ANTSPATH`, `OPENBLAS_CORETYPE`, ...) itself when it runs in the image, drops host `PYTHON*` and `FSLSUB_*` variables, and runs Python with `-E -s` so neither `PYTHONPATH` nor `~/.local` can supply packages. Otherwise a host FSL, ANTs, BLAS kernel or numpy could do the processing. A test keeps `IMAGE_ENVIRONMENT` and the `Dockerfile` `ENV` in step.

#### Checking whether a change moved the numbers

"Provably untouched" means measured, not assumed: nothing in the test suite checks the endpoint values. For anything that plausibly reaches the validated processing path (a regenerated conda lock, an FSL or ANTs version bump, an edit to the fitting, brain/exclusion masking or skeletonisation code, a change to the `sed` patches in the `Dockerfile`), build the image before and after the change, process the same representative subject with each, and diff the two `delta-svd_results.csv` tables:

```bash
.venv-test/bin/python tools/compare_results.py \
    before/delta-svd_results.csv after/delta-svd_results.csv
```

It runs from the [test virtual environment](#tests), which supplies the numpy and pandas it needs. It compares every metric value and every skeleton voxel count and exits non-zero if anything moved; `--help` covers the rest, including `--ignore-key` for a column that was renamed without the numbers changing.

Everything is compared **exactly**, with no tolerance option: a changed skeleton is a changed result even when the metrics happen to round the same way. A clean standard run is the evidence that the validated endpoints are not affected. If a change deliberately corrects an optional derived output, also compare a representative affected input and document the expected differences and pooling restriction.

Rules for the runs being compared:

- **Leave `--threads` and `--para` alone** — they don't affect the numbers. The hidden `--itkThreads` does, and must stay at its default (`ITK_THREADS_DEFAULT = 12`, the value the method was validated at) for any comparison to mean anything.
- **Use a longitudinal subject.** A cross-sectional run alone is not sufficient.
- **Judge by the longitudinal change (ΔPSMD), not the per-timepoint values.** ΔPSMD is by far the most sensitive readout: a diff that looks negligible per timepoint can still be a large change in the endpoint the pipeline exists to produce.

> [!WARNING]
> A cross-sectional-only check is not sufficient. Both modes threshold the interpolated, skeletonised brain mask at exactly 1, so every boundary voxel sits on a knife edge. A longitudinal check additionally exercises within-subject template construction and measures the longitudinal change (the pipeline's most sensitive readout), so it remains the required validation path.

### Repository layout

- [`container/`](container/) — the container image: `Dockerfile`, `build.sh`, the pinned conda environment, and the pipeline scripts under `container/scripts/`.
- [`tests/`](tests/) — pytest unit tests for those scripts.
- [`tools/`](tools/) — maintainer utilities that are not part of the image, such as `compare_results.py` above.
- [`docs/`](docs/) — documentation sources (see below).
- [`overrides/`](overrides/) — theme template overrides for the documentation site (see below).

### Workflow

- Development happens on a feature branch; open pull requests against `main`.
- `main` is protected: it takes a pull request with at least one approving review and passing status checks (`pytest` on 3.11 and 3.14, `hadolint`, `compileall`). There is no direct-push or force-push exception, including for maintainers.
- [`VERSION`](VERSION) is the single source of truth for the release version; bump it in the feature branch that makes up the release (see [Releasing](#releasing)).
- Keep the working tree clean before building a release image: `build.sh` marks the image revision `-dirty` when there is any uncommitted change, including an untracked file.

### Building the image

```bash
container/build.sh                 # -> delta-svd:<VERSION>
container/build.sh delta-svd:dev   # custom tag; extra args pass through to docker build
```

`build.sh` reads the version from `VERSION`, builds for `linux/amd64` (the only platform the pinned stack resolves for), and stamps OCI provenance labels (git revision and build date) into the image.

### Releasing

A release goes through three stages. CI builds it and stages it ([`release-build.yml`](.github/workflows/release-build.yml)). A maintainer then checks the staged image by hand. Finally, a second workflow ([`release-promote.yml`](.github/workflows/release-promote.yml)) copies that same manifest into the production package. Nothing reaches `ghcr.io/isdneuroimaging/delta-svd` unless a human has checked the exact digest first.

Keep in mind before you start:

- **The pushed image digest is the release, not the source tree.** An *older* commit cannot be rebuilt: the build fails. The `apt` pins resolve against the live Ubuntu archive, and the `ca-certificates` version is itself a date, so the pin stops matching once the archive moves on. To recover an old release, pull its digest. Do not rebuild its tag.
- **Only exact version tags are published.** There is no `latest` tag, so no run can pick up a new version by accident (see [Validation status](#validation-status-read-this-first)).
- **Choosing the version number.** Bump only the last digit when a change provably leaves the validated whole-skeleton endpoints untouched. Otherwise bump `MAJOR.MINOR`, which tells users their results cannot be pooled with earlier ones. A patch that corrects optional derived rows must carry a compatibility note. The note names the affected inputs and tells users not to pool those rows across the patch.

#### 1. Bump the version in the feature branch

Bump the version as the last commit of the feature branch that makes up the release. The reviewer then sees the version choice (patch or `MAJOR.MINOR`) next to the change that justifies it. If a release bundles several pull requests that are already merged, put the bump in a short pull request of its own instead.

[`VERSION`](VERSION) is the single source of truth. Two files restate it by hand because their formats cannot interpolate, and both must change in the same commit:

- [`CITATION.cff`](CITATION.cff): the `version:` field, which GitHub shows under "Cite this repository".
- [`docs/install.md`](docs/install.md): the image tag in the pull and verification commands. If this is missed, the docs keep handing users the *previous* image.

On the feature branch, set `VER` to the new version, then run the block below. This is the only place the version is typed in; every later step reads it back from `VERSION`.

```bash
VER=X.Y.Z
echo "$VER" > VERSION
sed -i "s/^version: .*/version: $VER/" CITATION.cff
sed -i -E "s/delta-svd:[0-9]+\.[0-9]+\.[0-9]+/delta-svd:$VER/g" docs/install.md
git diff
.venv-test/bin/python -m pytest tests/test_version.py
git commit -am "Bump version to $VER"
git push
```

`tests/test_version.py` fails if either file disagrees with `VERSION`. Everything else takes the version from `VERSION` automatically. `build.sh` (which `release-build.yml` also calls) passes it to the `Dockerfile` as the `VERSION` build argument. That argument stamps the OCI label and writes the file that `--version` reports. The `VERSION` file itself is never copied into the image, so a plain `docker build` without that argument reports `unknown`.

`main` is protected, so the **pull request must be merged before the next step**.

#### 2. Tag the merged commit

After merging the PR:

```bash
git switch main
git pull
VER=$(tr -d '[:space:]' < VERSION)
git tag -a "v$VER" -m "DELTA-SVD $VER"
git push origin "v$VER"
```

Pushing the tag triggers `release-build.yml`. The workflow checks the tag against `VERSION`, builds the image the same way a local `build.sh` run would, and attests it with `actions/attest-build-provenance`. It pushes the image to the public **staging** package, `ghcr.io/isdneuroimaging/delta-svd-staging`. The staged image is tagged by commit SHA only, so it cannot be mistaken for a release.

The run's job summary shows the staged image reference, `ghcr.io/isdneuroimaging/delta-svd-staging@sha256:...`. Copy it: every later step uses this reference, not the tag.

#### 3. Validate the staged image

You can skip this step only if the change is provably not metric-affecting. Pull *the staged image*; a local rebuild does not count:

```bash
STAGED="ghcr.io/isdneuroimaging/delta-svd-staging@sha256:..."   # from the job summary
apptainer pull delta-svd_staging.sif "docker://$STAGED"
```

Process the longitudinal reference subject with it, following the rules in [Checking whether a change moved the numbers](#checking-whether-a-change-moved-the-numbers). Then compare the results with those of the previous release:

```bash
.venv-test/bin/python tools/compare_results.py \
    before/delta-svd_results.csv after/delta-svd_results.csv
```

If validation fails, fix it and re-tag. Production has not been touched yet, so moving the tag is safe:

```bash
VER=$(tr -d '[:space:]' < VERSION)
git tag -d "v$VER"
git push origin --delete "v$VER"
# merge the fix through a pull request, then repeat step 2
```

#### 4. Promote

This reuses `STAGED` from step 3. If it is not set in the current shell, the command stops without triggering anything:

```bash
echo "Promoting ${STAGED:?not set, see step 3}" && gh workflow run release-promote.yml -f image="$STAGED"
```

You can also start it from the Actions tab: choose "Run workflow". The workflow first re-reads the image's own `version` and `revision` labels, and refuses the image if either is malformed, `unknown`, or `-dirty`. It then copies the manifest to `ghcr.io/isdneuroimaging/delta-svd:<version>` at the *same* digest, without rebuilding. Finally, it opens a draft GitHub release that records the digest.

#### 5. Publish the draft release

```bash
VER=$(tr -d '[:space:]' < VERSION)
gh release edit "v$VER" --draft=false
```

You can also publish it from the Releases page on GitHub.

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

The documentation site is built with [Zensical](https://zensical.org/), a static site generator by the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) team. It reads [`zensical.toml`](zensical.toml); all content lives in [`docs/`](docs/) as Markdown, which stays readable directly on GitHub. The site is published to GitHub Pages by the [`docs` workflow](.github/workflows/docs.yml) on every push to `main` that touches `docs/`, `overrides/`, `zensical.toml`, or the workflow itself. Pull requests touching those same paths run the build without deploying, so a broken link or a stale navigation entry fails the strict build on the pull request rather than on `main`.

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

- Add or edit Markdown files under `docs/`, then register new pages in the `nav` array of [`zensical.toml`](zensical.toml) so they appear in the site navigation (and to keep the strict build happy).
- GitHub-style alerts work as-is: write `> [!NOTE]`, `> [!TIP]`, `> [!IMPORTANT]`, `> [!WARNING]`, or `> [!CAUTION]` and they render as native alerts on GitHub and as admonitions on the site.

### Link previews

Every page carries Open Graph and Twitter card tags, so a link to the site unfurls with a preview image, its page title, and a description instead of a bare URL. The theme emits none of these itself; they come from [`overrides/main.html`](overrides/main.html), which the `project.theme.custom_dir` setting in [`zensical.toml`](zensical.toml) layers over the stock templates.

All pages share one preview image, [`docs/assets/og-image.png`](docs/assets/og-image.png). It is committed to the repository, not generated during the build; [`tools/make_og_image.py`](tools/make_og_image.py) renders it (1200×630, the size every major scraper expects) from the pipeline figure on the landing page. Re-run it after changing that figure or the card design, on any Python that has [Pillow](https://pypi.org/project/pillow/) installed, and commit the result:

```bash
.venv/bin/pip install pillow
.venv/bin/python tools/make_og_image.py
```

The tags are absolute URLs built from `site_url`, which scrapers require, so they can only be checked against the deployed site — a local build serves them pointing at <https://delta-svd.com>. After deploying a change to them, confirm the result with a preview debugger such as [opengraph.xyz](https://www.opengraph.xyz/); note that Slack, LinkedIn, and the rest cache aggressively, so an unchanged preview is usually a stale cache rather than a broken tag.
