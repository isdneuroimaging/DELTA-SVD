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
