# Refactoring plan: `delta-svd.py`

Working document for the refactoring session(s). Not part of the product:
removed in the last commit (step 11).

All work happens on one branch, one commit per step, and goes to `main` as a
single pull request. That PR must be merged with **rebase** or a **merge
commit**, not squash (the repository's recent PRs were squashed): squashing
would collapse the steps into one commit, so `git bisect` could no longer find
the step that caused a regression and the per-step verbatim proof would be
lost.

Baseline: `main` at `b090b67` (v1.0.7). Line numbers below refer to that commit
and shift as soon as step 1 lands, so locate code by function name.

## 1. Goal and end state

Make the pipeline easier to maintain without changing what it computes.
Maintainability is the main goal, smaller files (context size) a secondary one,
and no regression is a hard constraint.

The main script keeps **the method** (tensor and free-water fit, template,
TBSS, mask integration, statistics) and the step orchestration, readable top to
bottom. Everything *around* the method moves into flat modules next to the
existing `delta_svd_constants.py` and `delta_svd_version.py`.

| File | Content | Size |
| --- | --- | --- |
| `delta-svd.py` | Pinning statements (unchanged order), the method, orchestration, manifest | about 40 to 45 KB, from 120 KB |
| `delta_svd_runtime.py` | Image environment, core detection, CPU check. Standard library only | about 12 KB |
| `delta_svd_inputs.py` | Parser, argparse types, path resolution, input validation | about 31 KB |
| `delta_svd_gradients.py` | bval/bvec reading, direction checks, b-value selection | about 16 KB |
| `delta_svd_qc.py` | Deprojection and `prepare_qc` | about 7 KB |
| `delta_svd_common.py` | The few helpers two or more of the above need | about 4 KB |

`tests/test_delta_svd.py` (142 KB, 221 test functions, 314 cases) is split the
same way.

### Deliberately not done

These were considered and rejected as forced, risky, or low value. Do not add
them without asking.

- Splitting the method itself by pipeline step, or any stage interface or step
  framework. Revisit only if the remaining script proves awkward.
- Removing the import-time pre-parsers (`resolve_thread_budget`,
  `resolve_itk_threads`) or touching `plan_ants_parallelism`: they sit on the
  thread pinning, are tested, and work.
- Converting `run_subprocess` to `shell=False` / argument lists.
- Constants for the intermediate file names, an `isLongitudinal` flag, a
  run-context object replacing `args`.
- Merging `check_mni_grid` / `check_dwi_grid`, or the two input-resolution
  loops: their messages and semantics differ too much.
- Removing the repeated `validate_roi_labels` calls.
- Renaming for style (`isNIfTI`, `iniParser`, ...), restructuring the
  aggregator's `__main__` block, changing `create_qc_image`'s interface.
- Any edit to `markvcid_fw_mrn.py` (vendored, see `CONTRIBUTING.md`).

## 2. Rules

1. **One step per commit, one concern per commit.** A commit either edits code
   in place or moves code, never both. Every commit must pass the checks in
   section 3 on its own, so that each one is a working state.
2. **Move commits are verbatim.** Function bodies, class bodies and constants are
   not edited. Only import lines, the new file's header, and test patch targets
   change. `tools/check_verbatim_move.py` must report `only verbatim moves`
   (see section 3).
3. **The pinning statements stay in `delta-svd.py`, in today's order, ahead of
   the first third-party import.** Only the function *definitions* they call
   move to `delta_svd_runtime.py`. The statements are: the
   `if running_in_image():` block, `BLAS_THREADS`, `CORE_BUDGET`, the
   `for _var in (...)` loop, `ITK_THREADS`, the
   `os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']` assignment, `_missing`
   and `if _missing:`.
4. **`delta_svd_runtime.py` and `delta_svd_constants.py` import the standard
   library only** (plus `delta_svd_constants`). Every other new module may
   import numpy and friends, because `delta-svd.py` imports it after pinning.
5. **`delta-svd.py` imports from the new modules only the names it uses**
   (pyflakes stays clean). Tests for moved code import the new module directly;
   they do not reach it through the `delta_svd` fixture.
6. **Monkeypatch where the name is looked up.** After a move, a patch must
   target the module whose function reads the name, not `delta_svd`. Known
   sites are listed per step.
7. **Tests move verbatim too**, apart from the fixture name and patch targets.
   The number of collected cases must not drop.
8. `PLAN.md`: tick the step in section 5 in the commit that completes it.
9. **Commit messages** name the step (`Step 6: move gradient handling to
   delta_svd_gradients.py`) and, for move commits, include the checker's
   output.

## 3. Checks for every commit

PR CI only tests the branch head, so the first three checks are run locally on
each commit before the next step starts.

| Check | Command | Expectation |
| --- | --- | --- |
| Lint | `python -m pyflakes container/scripts/*.py tools/*.py` | No output for the DELTA-SVD scripts. The vendored `markvcid_fw_mrn.py` is not held to this |
| Unit tests, local | `python -m pytest -q` in a venv from `tests/requirements.txt` | Baseline 484 cases (see note) plus any added |
| Verbatim proof | `tools/check_verbatim_move.py` (compares with the parent commit, `HEAD~1`) | `RESULT: only verbatim moves.` Paste the output into the commit message of move commits |
| Pinned stack | PR CI (`tests.yml`: `run-in-locked-env.sh`, `compileall`, `hadolint`) on the branch head. Locally `tests/run-in-locked-env.sh` where Docker is available | Green |
| Longitudinal, exact | See below | `RESULT: no metric value and no voxel count changed.` for both subjects |

Notes:

- **Local baseline.** With unpinned PyPI packages (Python 3.13, pandas 3.1.1,
  numpy 2.5.4) the suite gives 482 passed, 2 failed. Both failures are
  `tests/test_aggregate_results.py::test_aggregate_preserves_literal_identifiers`
  and exist on `main`; the image pins pandas 2.3.3, where this was not verified
  in the planning session. Treat "the same 2 failures, nothing else" as green
  locally, and PR CI as the authority.
- **The checker's limits.** It compares syntax trees of top-level functions,
  classes and constants across `container/scripts/*.py`, and the remaining
  top-level statements of `delta-svd.py`. It does not prove that a moved
  function still resolves the same names: a missing import shows up in
  pyflakes or the tests, and a *different* object under the same name would
  not. So in a move commit, review the import block of the new module by hand
  against the names the moved functions use.
- **`--allow NAME`** exempts definitions an in-place commit is meant to change.
  Never use it in a move commit, with one exception noted in step 4.
- **Longitudinal comparison.** The unit tests do not check metric values. The
  planning session ran in a cloud container without a Docker daemon, and the
  `longitudinal-tests` workflow only accepts a `ghcr.io` image digest, which
  only a version tag produces. So this check has to be run by a maintainer on a
  machine with Docker (`container/build.sh delta-svd:dev`, then
  `tests/run-longitudinal.sh DATASET delta-svd:dev OUT`, as in
  `CONTRIBUTING.md`), or the release pipeline is the gate. See open decision
  A. Steps marked **L** in section 5 need it.
- **QC outputs** are not in the results table. For step 9 compare a `--qc 2`
  run before and after: the NIfTI files in `delta-svd_qc/` byte for byte, and
  `delta-svd_qc.html` ignoring the creation timestamp.

## 4. Open decisions for the maintainer

Defaults apply unless the maintainer says otherwise.

- **A. Longitudinal runs.** Default: run locally on the commits of step 3,
  step 6 and step 10, and treat the tagged patch release (`release-build.yml`)
  as the final gate. If a checkpoint fails, the commits since the previous
  checkpoint are the suspects; bisect over them. Stricter alternative: on every
  **L** commit.
- **B. Where `PLAN.md` lives.** Only on the branch: it is added in the first
  commit and deleted in the last, so it never stands on `main` on its own. It
  is outside the image (`.dockerignore` admits only `container/`, `LICENSE`,
  `NOTICE`).
- **C. Pull request.** Decided: one PR for all steps, one commit per step,
  reviewed commit by commit, merged by rebase or merge commit (see top of
  file). Open it as a draft early so CI runs on every push.
- **D. Step 10** (`integrate_masks`) rewrites code that writes the masks and
  voxel counts. Default: do it, last and alone. Skipping it is acceptable.
- **E. Release.** The result ships as a patch release, because the endpoints
  are provably unchanged.

## 5. Steps

Legend: **L** = touches code the metrics depend on, needs the longitudinal
comparison (decision A). "Allow" lists the names to pass to
`check_verbatim_move.py --allow` for in-place steps.

### Phase 0: tooling

- [ ] **Step 0. Checker, guard test, plan.** No pipeline code changes.
  - `tools/check_verbatim_move.py`: save the script from the appendix
    unchanged. It was tested in the planning session on an identity run, a
    simulated move and simulated edits.
  - New `tests/test_module_layout.py` with syntax-tree based tests:
    1. In `delta-svd.py`, no module outside the standard library
       (`sys.stdlib_module_names`), `delta_svd_constants` and
       `delta_svd_runtime` is imported before the `if _missing:` statement.
    2. `delta_svd_constants.py`, and `delta_svd_runtime.py` once it exists,
       import the standard library and `delta_svd_constants` only.
  - `PLAN.md`.

### Phase 1: in place

- [ ] **Step 1. Stale code.**
  - Remove the old help text kept as a trailing comment on `--reprocess`
    (line 1849). The trailing comments on the *suppressed* options are their
    only documentation: keep them.
  - Remove the placeholder defaults, making the parameters required:
    `filter_b_values`, `free_water_correction` (`'data.nii.gz'`, `'file.bval'`,
    `out_dir=None`), and the mutable `[]` defaults of `create_template`,
    `coreg_merge_masks`, `integrate_masks`. Check every call site in the tests
    first.
  - Replace the three leaked loop variables with explicit ones, keeping the
    *same object*: in `create_template` the header and affine come from the
    **last** file loaded (keep the last, not the first); in `integrate_masks`
    the single-timepoint branch uses `allMasksAdjusted[0]`; in
    `create_qc_image` keep the last file of `fnames`.
  - Delete commented-out code: `pass #print(...)` in the aggregator,
    `#--- plt.cm.bwr` in `create_qc_image.py`.
  - Allow: `filter_b_values free_water_correction create_template
    coreg_merge_masks integrate_masks create_qc_image`. **L**

- [ ] **Step 2. Cut `pipeline_delta_svd()`.** Contiguous blocks become plain
  functions with explicit arguments. No logic changes, no reordering.
  - `resolve_inputs(args)`: from "Check bval, bvec and bmask files" through the
    two `check_mni_grid` calls (lines 1877 to 2018). It mutates `args` exactly
    as today and returns `stepsAvailable`, which the existing-output check
    needs. The voxel-order block that follows (lines 2020 to 2031) stays in
    `pipeline_delta_svd()`, so the reorientation helpers stay with the method.
  - The additional-masks block of the `tbss_non_fa` step (lines 2240 to 2261)
    becomes one function taking `dirTP, dirTBSS, dirTemplate, Emask, Rmask`.
  - The brain-mask block of the `fwc` step (lines 2178 to 2186) becomes one
    function.
  - Allow: `pipeline_delta_svd` plus the new names. **L**

- [ ] **Step 3. Duplication, clear wins only.**
  - One helper for the ANTs transform file names and the `antsApplyTransforms`
    command, replacing the three copies in `create_template`,
    `coreg_merge_masks`, `deproject_to_native`. The command strings must come
    out character for character as today (note the existing double space
    after `-i ...`); add a test that pins the three strings before changing
    the code.
  - One helper for the TBSS FA base name, replacing the two identical globs in
    `batch_tbss_non_fa` and `run_tbss_non_fa`. Keep deriving it from the files
    on disk.
  - Existing-output check: normalise the `stepsOutput` values to lists once,
    so the three `if not isinstance(fn, list)` loops and the six one-line
    deletes each become one loop. Messages and order of deletion unchanged.
  - Licence notice text: one constant in `delta_svd_constants.py`, used by
    `delta-svd.py`, the aggregator and the HTML report (5 copies today).
  - QC only: one constant for the MD display range (3 copies in
    `prepare_qc`); reuse `custom_tag()`'s markup for the two hand-built
    tooltip badges in `create_html_with_png.py` only if the generated HTML
    stays byte-identical, otherwise leave them.
  - Aggregator: the three identical "output exists" blocks become a loop; the
    manifest condition written twice becomes one variable.
  - Allow: the functions named above. **L**

### Phase 2: verbatim moves

Each step creates one module, moves the listed names, moves their tests into a
new test file, and updates patch targets. Order matters: later modules import
earlier ones.

- [ ] **Step 4. `delta_svd_common.py`.** `DeltaSvdError`, `ROI_LABEL_DTYPE`,
  `ROI_LABEL_MAX`, `save_nifti`, `strip_nifti_ext`, `validate_roi_labels`,
  `run_subprocess`, `safe_rmtree`, `tbss_fa_name`.
  - Everything else in the "Helper functions" section stays in `delta-svd.py`
    for now or moves in a later step.
  - Exception to rule 2: the `DeltaSvdError` docstring says it is caught "at
    the bottom of this file". Correct the wording here and pass
    `--allow DeltaSvdError`.
  - Tests: `tests/test_common.py`. **L**

- [ ] **Step 5. `delta_svd_runtime.py`.** `running_in_image`,
  `fsl_sub_config`, `pin_image_environment`, `isolated_interpreter_argv`,
  `detect_physical_cores`, `cgroup_cpu_limit`, `resolve_thread_budget`,
  `resolve_itk_threads`, `CORETYPE_REQUIRED_FLAGS`,
  `check_cpu_supports_coretype`.
  - Stays in `delta-svd.py`: every statement of rule 3, and
    `OVERSUBSCRIBE_NUM/DEN` (used by `plan_ants_parallelism`).
  - `running_in_image()` defaults to its own module's `__file__`; the new
    module sits in the same directory, so the result is the same. Keep it
    there.
  - Patch targets to change: `detect_physical_cores` (6 sites),
    `cgroup_cpu_limit` (3), and `delta_svd.importlib.util.find_spec` in
    `tests/test_image_environment.py` (3).
  - Tests: the thread and CPU tests go to `tests/test_runtime.py`;
    `test_image_environment.py` switches to the new module. The tests that
    assert the *effect* of the pinning statements (`BLAS_THREADS`,
    `ITK_THREADS`, the environment variables) stay with `delta-svd.py`.
  - Update the step 0 guard test to cover the new module. **L**

- [ ] **Step 6. `delta_svd_gradients.py`.** `MAX_LINES_REPORTED`,
  `first_non_numeric`, `read_bval_or_bvec`, `write_bval_or_bvec`, `B0_MAX`,
  `BVAL_MIN`, `BVAL_MAX`, `DESIGN_MATRIX_RANK`, `DIRECTION_COS_TOL`,
  `BVEC_UNIT_TOL`, `resolve_b_intervals`, `describe_directions`,
  `format_b_values`, `selected_volumes_float32`, `filter_b_values`.
  - The long comment blocks above the constants move with them.
  - The test that patches `delta_svd.filter_b_values` runs the pipeline, which
    still reads the name from `delta-svd.py`: that patch stays as it is.
  - `tests/test_constants.py::test_pipeline_and_report_share_the_constants`
    reads seven constants off `delta_svd`; point each at the module that now
    uses it.
  - Tests: `tests/test_gradients.py` (about 48 test functions). **L**

- [ ] **Step 7. `delta_svd_inputs.py`.** `NIFTI_EXTENSIONS`, `nifti_sibling`,
  `isNIfTI`, `per_timepoint_path`, `candidate_paths`, `missing_input_message`,
  `isCSV`, `validate_timepoint_labels`, `positive_int`, `assertPositiveJobs`,
  `assertPositiveRegistrations`, `assertPositiveItkThreads`, `assertBValue`,
  `threadBudget`, `CustomArgumentParser`, `stepsImplemented`,
  `argparseDescription`, `iniParser`, `GRID_AFFINE_ATOL`, `check_mni_grid`,
  `check_dwi_grid`, and `resolve_inputs` from step 2.
  - Imports `B0_MAX`, `BVAL_MIN`, `BVAL_MAX` from `delta_svd_gradients`.
  - Patch targets to change: `check_mni_grid` (1 site) and
    `SKELETON_MASK_DEFAULT` (the `_mni_reference` fixture, used by 4 tests;
    `check_mni_grid` reads it in its new module). A test that also runs the
    pipeline needs the constant patched in both modules.
  - Tests: parser, argparse-type, path-resolution and grid tests go to
    `tests/test_inputs.py`. Tests that call `pipeline_delta_svd()` stay with
    the main script even when they exercise input errors.
  - Not metric-relevant, but runs before every step.

- [ ] **Step 8. Split what is left of `tests/test_delta_svd.py`** if it is
  still above about 40 KB: pipeline-level tests to `tests/test_pipeline.py`,
  method tests (fit, template, masks, statistics) stay. Tests only.

- [ ] **Step 9. `delta_svd_qc.py`.** `qc_label_maps`, `consecutive_labels`,
  `restore_labels`, `deproject_to_native`, `prepare_qc`.
  - `prepare_qc` keeps importing `create_qc_image` and `create_html_with_png`
    inside the function, so the existing patches on those two modules keep
    working.
  - Patch targets to change: `deproject_to_native` (2 sites) and
    `run_subprocess` in `tests/test_qc_deprojection.py` (1).
  - Tests: `tests/test_qc_deprojection.py` switches to the new module.
  - Check the QC outputs as described in section 3.

### Phase 3: finish

- [ ] **Step 10. `integrate_masks()` bookkeeping** (decision D). Replace the
  three parallel lists and the `insert` index arithmetic with one list of row
  records and one small "save mask, record row" helper. Keep the row order,
  the file names, the dtypes and the `'NA'` / `'NaN'` sentinel strings exactly
  (the aggregator depends on them). Move the two "initialise inside the loop"
  statements (`roiUnion`, `imgROImerged`) in front of their loops. Before
  editing, add a test that pins the complete returned table and the set of
  files written for a case with exclusion mask, both ROI kinds and
  hemispheres. Allow: `integrate_masks`. **L**

- [ ] **Step 11. Documentation and cleanup.**
  - `CONTRIBUTING.md`: "Repository layout" and "Tests" describe the modules and
    the rule "code that can move the numbers lives in `delta-svd.py`,
    `delta_svd_runtime.py`, `delta_svd_gradients.py` and
    `delta_svd_common.py`".
  - Fix cross-references that name `delta-svd.py` for moved code: the
    `delta_svd_constants.py` docstring and comments, `tests/conftest.py`,
    comments in `tests/test_aggregate_results.py`. Search with
    `grep -rn "delta-svd.py" container tests tools docs CONTRIBUTING.md`.
  - Decide whether `tools/check_verbatim_move.py` stays as a maintainer tool
    (then mention it in `CONTRIBUTING.md`) or is removed.
  - Delete `PLAN.md`.

## 6. Facts the executing session should know

- The Dockerfile copies the whole `container/scripts/` folder and CI compiles
  it, so new modules need no Dockerfile or workflow change.
- `tests/conftest.py` puts `container/scripts` on `sys.path`, so a test can
  `import delta_svd_gradients` directly. `delta-svd.py` itself is loaded
  through the `delta_svd` fixture because of the hyphen.
- The fit's worker pool uses `fork` with the module-level `_FW_SHARED`,
  `_fw_pool_init` and `_fw_fit_slab`. They stay in `delta-svd.py`; do not move
  or wrap them.
- `write_run_manifest`, `utc_timestamp`, `section_header`, `binarise_mask`,
  `copy_as_nii_gz` and the reorientation helpers have a single user, the main
  script, and stay there.
- Known limitation, not to be fixed here: `args.nDirections` is only set when
  the `fwc` step runs, so a `--steps qc` report omits the directions row.
- `args.threads` is parsed but never read; the core budget comes from the
  import-time `CORE_BUDGET`. Leave it (see "Deliberately not done").
- Do not put a model name or session link into code comments or this file.

## Appendix: `tools/check_verbatim_move.py`

```python
#!/usr/bin/env python3
"""Prove that a refactoring commit only moved code between the pipeline scripts.

Compares every top-level function, class and constant of container/scripts/*.py
at a base git revision with the working tree, by syntax tree (so comments and
blank lines may differ, code may not). Also requires that the remaining
top-level statements of delta-svd.py (environment and thread pinning, the CPU
check, the __main__ guard) are unchanged and in the same order.

Usage: tools/check_verbatim_move.py [--base REF] [--allow NAME ...]
       REF defaults to HEAD~1, the parent of the commit being checked; run it
       after committing, or with --base HEAD before.
Exit status: 0 = only moves (plus allowed names), 1 = something changed.
"""
import argparse, ast, pathlib, subprocess, sys

SCRIPTS = 'container/scripts'
VENDORED = {'markvcid_fw_mrn.py'}
MAIN = 'delta-svd.py'


def definitions(sources):
    """{name: [(file, dump), ...]} and the main script's other statements."""
    defs, residual = {}, []
    for fname, text in sorted(sources.items()):
        for node in ast.parse(text).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.Assign) and all(
                    isinstance(n, ast.Name) for t in node.targets for n in
                    (t.elts if isinstance(t, ast.Tuple) else [t])):
                names = [n.id for t in node.targets for n in
                         (t.elts if isinstance(t, ast.Tuple) else [t])]
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue                                   # module docstring
            else:
                if fname == MAIN:
                    residual.append(ast.dump(node))
                continue
            for name in names:
                defs.setdefault(name, []).append((fname, ast.dump(node)))
    return defs, residual


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--base', default='HEAD~1')
    parser.add_argument('--allow', nargs='*', default=[], metavar='NAME',
                        help='definitions that are expected to change in this commit')
    args = parser.parse_args()
    root = pathlib.Path(subprocess.check_output(
        ['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    listed = subprocess.check_output(
        ['git', 'ls-tree', '--name-only', args.base, SCRIPTS + '/'], text=True, cwd=root).split()
    old = {pathlib.PurePath(p).name: subprocess.check_output(
               ['git', 'show', f'{args.base}:{p}'], text=True, cwd=root)
           for p in listed if p.endswith('.py') and pathlib.PurePath(p).name not in VENDORED}
    new = {p.name: p.read_text() for p in (root / SCRIPTS).glob('*.py')
           if p.name not in VENDORED}
    oldDefs, oldResidual = definitions(old)
    newDefs, newResidual = definitions(new)

    problems, moved = [], []
    for name, entries in sorted(oldDefs.items()):
        if name in args.allow:
            continue
        now = newDefs.get(name, [])
        if not now:
            problems.append(f'missing: {name} (was in {entries[0][0]})')
        elif sorted(d for _, d in now) != sorted(d for _, d in entries):
            problems.append(f'changed: {name} ({entries[0][0]} -> {now[0][0]})')
        elif sorted(f for f, _ in now) != sorted(f for f, _ in entries):
            moved.append(f'{name}: {entries[0][0]} -> {now[0][0]}')
    added = sorted(set(newDefs) - set(oldDefs) - set(args.allow))
    if oldResidual != newResidual:
        problems.append(f'{MAIN}: the top-level statements outside definitions changed')

    print(f'{len(moved)} definition(s) moved unchanged:')
    for m in moved:
        print(f'  {m}')
    if added:
        print(f'{len(added)} new definition(s), review by hand: {", ".join(added)}')
    for p in problems:
        print(f'PROBLEM {p}')
    print('RESULT: ' + ('only verbatim moves.' if not problems else f'{len(problems)} problem(s).'))
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
```
