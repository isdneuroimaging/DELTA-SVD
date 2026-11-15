# Refactoring plan: `delta-svd.py`

Working document for the refactoring session. Not part of the product: the
last commit removes it.

Baseline: `main` at `b090b67` (v1.0.8). Line numbers refer to that commit;
locate code by function name once the first step has landed.

## 1. Principle

**This refactoring changes where code lives, not what it says.** Every change
to pipeline code is either a verbatim move of whole definitions or one
mechanical extraction of a contiguous block, and both are proven by a script
(section 4), not by review alone. Nothing is rewritten, renamed or tidied.

Why this scope:

- The unit tests do not check metric values, and a last-bit difference can
  reach the endpoints (`CONTRIBUTING.md`, "Validation status"). A move can be
  proven harmless mechanically; a rewrite, however small, cannot.
- The real costs today are structural: one 120 KB script, one 142 KB test
  file, and a 470-line pipeline function that mixes input validation with the
  processing steps. Moves fix all three.
- The functions themselves are small, named and tested. Tidying them would be
  change for its own sake (section 3).

## 2. End state

`delta-svd.py` keeps **the method** (tensor and free-water fit, template,
TBSS, mask integration, statistics) and the step orchestration, readable top
to bottom. What surrounds the method moves into flat modules next to the
existing `delta_svd_constants.py` and `delta_svd_version.py`.

| File | Content | Why it is separate | Size |
| --- | --- | --- | --- |
| `delta-svd.py` | Start-up statements (unchanged), the method, orchestration, manifest | The validated pipeline in one place | about 51 KB, from 120 KB |
| `delta_svd_runtime.py` | Image environment, core detection, CPU check (definitions only) | Changes with the container, not with the method; leaves the start-up sequence in `delta-svd.py` short enough to read at a glance | about 9 KB |
| `delta_svd_inputs.py` | Parser, argparse types, path resolution, input validation | Where most recent changes happened; mostly help and error text | about 34 KB |
| `delta_svd_gradients.py` | bval/bvec reading, direction checks, b-value selection | Input checking for the fit, with long error messages | about 16 KB |
| `delta_svd_qc.py` | Deprojection and `prepare_qc` | Cannot affect the metrics; joins the two existing QC scripts | about 7 KB |
| `delta_svd_common.py` | Nine helpers that two or more of the above need | The modules cannot import from a hyphenated script with import side effects | about 4 KB |

Sizes are measured from the current definitions. The split was checked for
import cycles: `common` is imported by `gradients`, `inputs`, `qc` and the main
script; `inputs` imports three constants from `gradients`; nothing imports the
main script.

`tests/test_delta_svd.py` (142 KB, 221 test functions, 314 cases) is split the
same way.

## 3. Left alone on purpose

### Not in this refactoring, do when that code is next changed anyway

Each of these is working, tested code. Cleaning it up now would be an
unprovable edit with no pending need. When one of them has to change for a
functional reason, do the clean-up first as its own commit.

- **`integrate_masks()`**: three parallel lists kept in step by `insert` index
  arithmetic, and two variables initialised inside their loops. On touch:
  one list of row records and a "save mask, record row" helper; keep row
  order, file names, dtypes and the `'NA'` / `'NaN'` sentinel strings (the
  aggregator depends on them); pin the full returned table in a test first.
- **ANTs transform names**: the warp/affine file-name pattern appears in
  `create_template`, `coreg_merge_masks` and `deproject_to_native`. It is
  dictated by ANTs' output naming and only changes with an ANTs upgrade, which
  needs re-validation anyway.
- **Existing-output check**: three near-identical loops and six one-line
  deletes. It deletes user files, so it is not worth touching for looks.
- **Leaked loop variables** in `create_template`, `integrate_masks`,
  `create_qc_image` (each documented by a comment). If fixed, keep the same
  object: in `create_template` the header comes from the **last** file loaded.
- **Placeholder and mutable default arguments** (`fn_data='data.nii.gz'`,
  `out_dir=None`, `timepoints=[]`): misleading but harmless; no caller relies
  on them.
- **Licence notice** repeated five times across three scripts; the aggregator's
  three identical "output exists" blocks; the hand-built tooltip badges in the
  HTML report; `pass #print(...)` in the aggregator.
- **Further cuts of `pipeline_delta_svd()`**. After step 1 it is about 330
  lines of linear orchestration; cutting 10 to 20 line blocks out of it is
  cosmetic.

### Rejected

- Splitting the method by pipeline step, or any stage interface or step
  framework. Revisit only if the remaining script proves awkward in practice.
- Removing the import-time pre-parsers or touching `plan_ants_parallelism`:
  they sit on the thread pinning.
- `run_subprocess` with `shell=False`; constants for intermediate file names;
  a run-context object replacing `args`; merging the two grid checks;
  style renames (`isNIfTI`, `iniParser`); restructuring the aggregator.
- Any edit to `markvcid_fw_mrn.py` (vendored, see `CONTRIBUTING.md`).

## 4. Rules and checks

### Rules

1. **One step per commit.** Every commit passes the per-commit checks on its
   own.
2. **Moves are verbatim.** Whole functions, classes and constants move with
   their comments; nothing inside them changes, including docstrings (those
   are corrected once, in step 8).
3. **The start-up statements stay in `delta-svd.py`, in today's order, ahead
   of the first third-party import**: the `if running_in_image():` block,
   `BLAS_THREADS`, `CORE_BUDGET`, the `for _var in (...)` loop, `ITK_THREADS`,
   the `os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']` assignment,
   `_missing` and `if _missing:`. Only the function definitions they call move.
4. **`delta_svd_runtime.py` imports the standard library and
   `delta_svd_constants` only.** It is imported before the pinning. Every
   other new module is imported after it and may import numpy and friends.
5. **Import blocks are copied, not written.** Build a new module's imports by
   copying the import lines of `delta-svd.py` verbatim and deleting the names
   pyflakes reports as unused, so every name resolves to the same object as
   before. Names from sibling modules come in as
   `from delta_svd_common import ...`.
6. **`delta-svd.py` imports only the names it still uses** (pyflakes clean).
   No re-exports for the tests' convenience.
7. **Tests move with their code**, unchanged apart from the module they
   address and patch targets. A monkeypatch must target the module whose
   function reads the name, which after a move is no longer `delta_svd`. New
   test files import their module directly (`tests/conftest.py` already puts
   `container/scripts` on `sys.path`).
8. Tick the step in section 6 in the commit that completes it.

### Per commit (local)

| Check | Command | Expectation |
| --- | --- | --- |
| Lint | `python -m pyflakes container/scripts/delta*.py container/scripts/create*.py tools/*.py` | No output |
| Unit tests | `python -m pytest -q` in a venv from `tests/requirements.txt` | Baseline, see note |
| No test lost | `python -m pytest --collect-only -q \| sed 's/.*:://' \| sort` before and after | Identical lists, apart from tests the step adds |
| Proof | `tools/check_verbatim_move.py` (compares with the parent commit) | `RESULT: only verbatim moves.` Put the output in the commit message |

Note on the local baseline: with unpinned PyPI packages (Python 3.13, pandas
3.1.1, numpy 2.5.4) `main` gives 482 passed, 2 failed. Both failures are
`tests/test_aggregate_results.py::test_aggregate_preserves_literal_identifiers`;
the image pins pandas 2.3.3, where this was not verified during planning.
Locally, "the same 2 failures and nothing else" is green; PR CI is the
authority.

### Once, at the head of the pull request

| Check | How | Expectation |
| --- | --- | --- |
| Whole-change proof | `tools/check_verbatim_move.py --base origin/main --extract pipeline_delta_svd resolve_inputs --allow DeltaSvdError` (run before step 8 removes the tool) | `RESULT: only verbatim moves.` Put the output in the PR description |
| Pinned stack | PR CI: `run-in-locked-env.sh`, `compileall`, `hadolint` | Green |
| Longitudinal, exact | By a maintainer on a machine with Docker: `container/build.sh delta-svd:dev`, then `tests/run-longitudinal.sh DATASET delta-svd:dev OUT` | `RESULT: no metric value and no voxel count changed.` for both subjects; open one QC report |

Why one longitudinal run is enough: every pipeline change is proven verbatim,
so the run confirms the one thing the proof cannot see, that imports and their
order behave the same in the real image. `CONTRIBUTING.md` asks for exactly
this for refactors ("verified to produce identical standard-run output"). It
cannot run in a cloud session (no Docker daemon), and the `longitudinal-tests`
workflow only accepts a `ghcr.io` digest, which only a version tag produces.

What the proof does and does not cover: the checker compares syntax trees of
all top-level functions, classes and constants across `container/scripts/*.py`
(the vendored file aside), requires the start-up statements of rule 3 to stay
in `delta-svd.py` unchanged and in order, and proves the extraction of step 1
statement by statement. It does not look at import lines; rule 5, pyflakes and
the guard test of step 0 cover those.

## 5. Decisions

- **Pull request.** One PR, one commit per step, reviewed commit by commit.
  Any merge method is fine, squash included: the whole-change proof holds
  against `main` regardless of how the commits are combined.
- **Longitudinal run.** Once, on the final commit, before merging.
- **`PLAN.md` and the checker.** They live on the branch only; step 8 deletes
  both. Keep `tools/check_verbatim_move.py` instead if the maintainers want it
  for a later split of the method; then mention it in `CONTRIBUTING.md`.
- **Release.** No release of its own is needed. The change rides with the next
  one, as a patch-level change, where release CI repeats the longitudinal
  comparison.

## 6. Steps

- [x] **Step 0. Tooling.** No pipeline code changes.
  - `tools/check_verbatim_move.py`: save the script from the appendix
    unchanged. It was tested during planning on an identity run, a simulated
    move, a simulated extraction, and on tampering (an edited function, an
    edited constant, an edit inside the extracted block, a pinned assignment
    moved to another file).
  - New `tests/test_module_layout.py`, two syntax-tree based tests that stay
    in the repository:
    1. In `delta-svd.py`, nothing outside the standard library
       (`sys.stdlib_module_names`), `delta_svd_constants` and
       `delta_svd_runtime` is imported before the `if _missing:` statement.
    2. `delta_svd_constants.py`, and `delta_svd_runtime.py` once it exists,
       import the standard library and `delta_svd_constants` only.

- [x] **Step 1. Extract `resolve_inputs(args)` from `pipeline_delta_svd()`.**
  The only non-move change to pipeline code.
  - The block runs from the comment "Check bval, bvec and bmask files"
    through the second `check_mni_grid` call (lines 1877 to 2018), 25
    statements. It becomes the body of `resolve_inputs(args)`, which ends with
    `return stepsAvailable`; the pipeline calls it as
    `stepsAvailable = resolve_inputs(args)`. `stepsAvailable` is the only
    local the rest of the pipeline reads from the block; `args` is the only
    one the block reads from before it. The voxel-order block that follows
    stays in the pipeline, so the reorientation helpers stay with the method.
  - Place the new function directly above the "Pipeline" banner.
  - Proof: `tools/check_verbatim_move.py --extract pipeline_delta_svd
    resolve_inputs`. This exact extraction was simulated during planning: the
    proof passed and the unit tests were unchanged (482 passed, the same 2
    failures).

- [ ] **Step 2. `delta_svd_common.py`.** `DeltaSvdError`, `ROI_LABEL_DTYPE`,
  `ROI_LABEL_MAX`, `save_nifti`, `strip_nifti_ext`, `validate_roi_labels`,
  `run_subprocess`, `safe_rmtree`, `tbss_fa_name`.
  - Nothing else from the "Helper functions" section: `binarise_mask`,
    `copy_as_nii_gz`, the reorientation helpers, `section_header`, the manifest
    functions are used by the main script only and stay there.
  - Tests: `tests/test_common.py`.

- [ ] **Step 3. `delta_svd_runtime.py`.** `running_in_image`,
  `fsl_sub_config`, `pin_image_environment`, `isolated_interpreter_argv`,
  `detect_physical_cores`, `cgroup_cpu_limit`, `resolve_thread_budget`,
  `resolve_itk_threads`, `CORETYPE_REQUIRED_FLAGS`,
  `check_cpu_supports_coretype`.
  - Stays in `delta-svd.py`: every statement of rule 3 with its comment, and
    `OVERSUBSCRIBE_NUM/DEN` (used by `plan_ants_parallelism`).
  - `running_in_image()` defaults to its own module's `__file__`. The new
    module sits in the same directory, so the result is the same.
  - Patch targets to change: `detect_physical_cores` (6 sites),
    `cgroup_cpu_limit` (3), and `delta_svd.importlib.util.find_spec` in
    `tests/test_image_environment.py` (3).
  - Tests: thread-detection and CPU-check tests go to `tests/test_runtime.py`;
    `test_image_environment.py` addresses the new module. Tests that assert the
    *effect* of the start-up statements (`BLAS_THREADS`, `ITK_THREADS`, the
    environment variables) stay with the main script.

- [ ] **Step 4. `delta_svd_gradients.py`.** `MAX_LINES_REPORTED`,
  `first_non_numeric`, `read_bval_or_bvec`, `write_bval_or_bvec`, `B0_MAX`,
  `BVAL_MIN`, `BVAL_MAX`, `DESIGN_MATRIX_RANK`, `DIRECTION_COS_TOL`,
  `BVEC_UNIT_TOL`, `resolve_b_intervals`, `describe_directions`,
  `format_b_values`, `selected_volumes_float32`, `filter_b_values`.
  - The comment blocks above the constants move with them.
  - The test that patches `delta_svd.filter_b_values` runs the pipeline, which
    still reads that name in `delta-svd.py`: leave that patch as it is.
  - Tests: `tests/test_gradients.py`.

- [ ] **Step 5. `delta_svd_inputs.py`.** `NIFTI_EXTENSIONS`, `nifti_sibling`,
  `isNIfTI`, `per_timepoint_path`, `candidate_paths`, `missing_input_message`,
  `isCSV`, `validate_timepoint_labels`, `positive_int`, `assertPositiveJobs`,
  `assertPositiveRegistrations`, `assertPositiveItkThreads`, `assertBValue`,
  `threadBudget`, `CustomArgumentParser`, `stepsImplemented`,
  `argparseDescription`, `iniParser`, `GRID_AFFINE_ATOL`, `check_mni_grid`,
  `check_dwi_grid`, `resolve_inputs`.
  - Imports `B0_MAX`, `BVAL_MIN`, `BVAL_MAX` from `delta_svd_gradients`.
  - Patch targets to change: `check_mni_grid` (1 site; that test runs the
    pipeline, but the function is now called from `resolve_inputs`) and
    `SKELETON_MASK_DEFAULT` (2 sites, one of them the `_mni_reference`
    fixture; `check_mni_grid` reads the constant in its new module).
  - `tests/test_constants.py::test_pipeline_and_report_share_the_constants`
    reads seven constants off `delta_svd`. After steps 4 and 5 most are no
    longer used there: point each at the module that uses it.
  - Tests: parser, argparse-type, path-resolution and grid tests go to
    `tests/test_inputs.py`. Tests that call `pipeline_delta_svd()` stay with
    the main script even when they exercise input errors.

- [ ] **Step 6. `delta_svd_qc.py`.** `qc_label_maps`, `consecutive_labels`,
  `restore_labels`, `deproject_to_native`, `prepare_qc`.
  - `prepare_qc` keeps importing `create_qc_image` and `create_html_with_png`
    inside the function, so the existing patches on those two modules keep
    working.
  - Patch targets to change: `deproject_to_native` (2 sites) and
    `run_subprocess` in `tests/test_qc_deprojection.py` (1). The patches of
    `run_subprocess` and `copy2` around `create_template` stay as they are.
  - Tests: `tests/test_qc_deprojection.py` addresses the new module; the QC
    test in `tests/test_delta_svd.py` joins it.

- [ ] **Step 7. Split the rest of `tests/test_delta_svd.py`.** Tests only.
  Pipeline-level tests (those calling `pipeline_delta_svd()`, with their
  helpers) go to `tests/test_pipeline.py`; method tests (fit, template, masks,
  statistics) stay.

- [ ] **Step 8. Wrap up.**
  - Run the whole-change proof (section 4) and keep its output for the PR
    description.
  - Docstrings and comments that the moves made inaccurate: the
    `DeltaSvdError` docstring ("at the bottom of this file"), the "Helper
    functions" and "Pipeline" banners, `tests/conftest.py`, comments in
    `tests/test_aggregate_results.py`. Search with
    `grep -rn "delta-svd.py" container tests tools CONTRIBUTING.md`.
  - Two stale comments, comment-only: the old help text trailing the
    `--reprocess` option (the trailing comments on the *suppressed* options
    are their only documentation, keep those), and `#--- plt.cm.bwr` in
    `create_qc_image.py`.
  - `CONTRIBUTING.md`: "Repository layout" and "Tests" name the modules and
    state two rules: code that can move the numbers lives in `delta-svd.py`,
    `delta_svd_runtime.py`, `delta_svd_gradients.py` and
    `delta_svd_common.py`; and `delta_svd_runtime.py` stays standard-library
    only.
  - Delete `PLAN.md` and, unless the maintainers keep it,
    `tools/check_verbatim_move.py`.

## 7. Facts the executing session should know

- The Dockerfile copies the whole `container/scripts/` folder and CI compiles
  and tests everything under it, so new modules and test files need no
  Dockerfile or workflow change.
- `delta-svd.py` is loaded in tests through the `delta_svd` fixture because of
  the hyphen. Loading it has side effects (reads `sys.argv`, sets environment
  variables); the new modules have none.
- The fit's worker pool uses `fork` with the module-level `_FW_SHARED`,
  `_fw_pool_init` and `_fw_fit_slab`. They stay in `delta-svd.py`; do not move
  or wrap them.
- `test_iniparser_defaults` exists in two test files (pipeline and
  aggregator); keep them in different files.
- The cloud session has no Docker daemon: `tests/run-in-locked-env.sh` and the
  longitudinal run are not available there. Open the PR as a draft early so
  PR CI runs the pinned-stack tests on every push.
- Known limitations that are not to be fixed here: `args.nDirections` is only
  set when the `fwc` step runs, so a `--steps qc` report omits the directions
  row; `args.threads` is parsed but never read, the core budget comes from the
  import-time `CORE_BUDGET`.
- Do not put a model name or session link into code comments or this file.

## Appendix: `tools/check_verbatim_move.py`

```python
#!/usr/bin/env python3
"""Prove that a refactoring commit only moved code between the pipeline scripts.

Compares every top-level function, class and constant of container/scripts/*.py
at a base git revision with the working tree, by syntax tree (so comments and
blank lines may differ, code may not). Also requires that the start-up
statements of delta-svd.py (environment and thread pinning, the CPU check, the
__main__ guard) are unchanged, still in delta-svd.py and in the same order.

Usage: tools/check_verbatim_move.py [--base REF] [--allow NAME ...]
                                    [--extract CALLER NEW]
       REF defaults to HEAD~1, the parent of the commit being checked; run it
       after committing, or with --base HEAD before.
       --extract proves that NEW is a contiguous block cut out of CALLER:
       CALLER's old statements must equal its new ones with the single
       statement calling NEW replaced by NEW's body (docstring and final
       'return' aside).
Exit status: 0 = only moves (plus allowed names), 1 = something changed.
"""
import argparse, ast, pathlib, subprocess, sys

SCRIPTS = 'container/scripts'
VENDORED = {'markvcid_fw_mrn.py'}
MAIN = 'delta-svd.py'
#--- assignments that are part of the start-up sequence and must not move
PINNED = {'BLAS_THREADS', 'CORE_BUDGET', 'ITK_THREADS', '_missing'}


def definitions(sources):
    """({name: [(file, dump, node), ...]}, start-up statements of MAIN)."""
    defs, startup = {}, []
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
                names = []
            if fname == MAIN and (not names or PINNED & set(names)):
                startup.append(ast.dump(node))
            for name in names:
                defs.setdefault(name, []).append((fname, ast.dump(node), node))
    return defs, startup


def body(node):
    """Statements of a function, without a leading docstring."""
    stmts = list(node.body)
    if (stmts and isinstance(stmts[0], ast.Expr) and isinstance(stmts[0].value, ast.Constant)
            and isinstance(stmts[0].value.value, str)):
        stmts = stmts[1:]
    return stmts


def check_extraction(oldDefs, newDefs, caller, new):
    """None if 'new' is a contiguous block cut out of 'caller', else the reason."""
    if caller not in oldDefs or caller not in newDefs or new not in newDefs:
        return f'{caller} or {new} not found'
    if new in oldDefs:
        return f'{new} existed already'
    old = [ast.dump(s) for s in body(oldDefs[caller][0][2])]
    cut = body(newDefs[new][0][2])
    if cut and isinstance(cut[-1], ast.Return):
        cut = cut[:-1]
    calls = [i for i, s in enumerate(body(newDefs[caller][0][2]))
             if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == new
                    for n in ast.walk(s))]
    if len(calls) != 1:
        return f'{caller} has {len(calls)} statements calling {new}, expected 1'
    now = [ast.dump(s) for s in body(newDefs[caller][0][2])]
    spliced = now[:calls[0]] + [ast.dump(s) for s in cut] + now[calls[0] + 1:]
    return None if spliced == old else f'{caller} is not its old self with {new} cut out'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--base', default='HEAD~1')
    parser.add_argument('--allow', nargs='*', default=[], metavar='NAME',
                        help='definitions that are expected to change in this commit')
    parser.add_argument('--extract', nargs=2, metavar=('CALLER', 'NEW'),
                        help='NEW is a contiguous block cut out of CALLER')
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
    oldDefs, oldStartup = definitions(old)
    newDefs, newStartup = definitions(new)
    allowed = set(args.allow) | set(args.extract or [])

    problems, moved = [], []
    for name, entries in sorted(oldDefs.items()):
        if name in allowed:
            continue
        now = newDefs.get(name, [])
        if not now:
            problems.append(f'missing: {name} (was in {entries[0][0]})')
        elif sorted(e[1] for e in now) != sorted(e[1] for e in entries):
            problems.append(f'changed: {name} ({entries[0][0]} -> {now[0][0]})')
        elif sorted(e[0] for e in now) != sorted(e[0] for e in entries):
            moved.append(f'{name}: {entries[0][0]} -> {now[0][0]}')
    added = sorted(set(newDefs) - set(oldDefs) - allowed)
    if oldStartup != newStartup:
        problems.append(f'{MAIN}: the start-up statements changed, moved or were reordered')
    if args.extract:
        reason = check_extraction(oldDefs, newDefs, *args.extract)
        if reason:
            problems.append(f'extraction: {reason}')
        else:
            print(f'{args.extract[1]} is a contiguous block cut out of {args.extract[0]}, unchanged.')

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
