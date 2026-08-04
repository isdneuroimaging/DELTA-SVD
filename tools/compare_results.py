#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two DELTA-SVD results tables and report anything that moved.

For the check described in CONTRIBUTING.md under "Validation status": run the
pipeline on a representative subject before and after a change that could affect
the numbers, then diff the two 'delta-svd_results.csv' tables with this script.
Any difference in a metric value or a skeleton voxel count means the change is
metric-affecting and requires re-validation.

The script ships no data of its own and compares whatever two runs it is pointed
at, so the input never has to leave the machine it was processed on.

It is a maintainer tool and lives outside the container image, so it runs on the
Python it is invoked with and needs numpy and pandas there - in practice the test
virtual environment: '.venv-test/bin/python tools/compare_results.py ...'.

Exit status: 0 if the tables agree, 1 if anything differs, 2 if they cannot be
compared at all.
"""

import argparse
import sys

try:
    import numpy as np
    import pandas as pd
except ImportError as err:
    print(f"ERROR: {err.name} is not installed. compare_results.py needs numpy and pandas.\n"
          f"Run it from the test virtual environment (see CONTRIBUTING.md):\n"
          f"    python3 -m venv .venv-test\n"
          f"    .venv-test/bin/pip install -r tests/requirements.txt\n"
          f"    .venv-test/bin/python tools/compare_results.py BEFORE.csv AFTER.csv",
          file=sys.stderr)
    #--- 2 = 'not comparable'; exiting 1 here would read as 'differences found'
    sys.exit(2)

#--- Columns identifying a row. 'ID' appears only when --id was used; the
#    aggregator's 'path' is deliberately not a key, as two runs of the same
#    subject legitimately live in different folders.
KEY_COLUMNS = ['ID', 'timepoint', 'skeleton', 'region', 'metric']

#--- Compared exactly: a changed voxel count is a changed skeleton, and a
#    metric that moved in its last bits still moved.
REQUIRED_COLUMNS = ['timepoint', 'region', 'metric', 'voxels', 'value']


def iniParser():
    parser = argparse.ArgumentParser(
        description="Compare two DELTA-SVD results tables (delta-svd_results.csv) and report "
                    "every metric value and skeleton voxel count that changed between them.",
        epilog="Exit status: 0 = tables agree, 1 = something differs, 2 = tables not comparable.")
    parser.add_argument("before", metavar="BEFORE.csv", help="results table produced before the change")
    parser.add_argument("after", metavar="AFTER.csv", help="results table produced after the change")
    parser.add_argument("--ignore-key", action="append", default=[], metavar="COLUMN",
                        dest="ignore_keys",
                        help="drop COLUMN from the columns that identify a row (one of: "
                             + ", ".join(KEY_COLUMNS) + "); may be given more than once. Use it "
                             "when a label was renamed without the numbers changing, e.g. "
                             "'--ignore-key skeleton' after the skeleton mask file was renamed. "
                             "Rows the remaining keys no longer identify uniquely are matched in "
                             "file order, so only ignore a key for two runs of the same pipeline.")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="print only the verdict, not the individual differences")
    return parser


def read_results(path):
    """Read a results table, failing loudly if it is not one."""
    try:
        df = pd.read_csv(path)
    except Exception as err:
        raise ValueError(f"cannot read '{path}': {err}") from err
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"'{path}' is missing expected column(s): {', '.join(missing)}")
    return df


def key_columns(dfBefore, dfAfter, ignore_keys=()):
    """Those present in both tables, minus any the caller declared cosmetic. An
    unknown column is rejected: a typo would quietly weaken the comparison."""
    unknown = [c for c in ignore_keys if c not in KEY_COLUMNS]
    if unknown:
        raise ValueError(f"--ignore-key: {', '.join(unknown)} is not a key column "
                         f"(expected one of: {', '.join(KEY_COLUMNS)})")
    return [c for c in KEY_COLUMNS
            if c not in ignore_keys and c in dfBefore.columns and c in dfAfter.columns]


def _format_key(row, keys):
    return '  '.join(f'{k}={row[k]}' for k in keys)


def _format_number(x):
    """Plain rendering of a numpy/pandas scalar (repr() would show 'np.int64(1)')."""
    if pd.isna(x):
        return 'NaN'
    x = float(x)
    return f'{int(x)}' if x.is_integer() else f'{x:.10g}'


def _describe_row_changes(before, after, keys):
    """Rows that exist on only one side, i.e. the tables are not even comparable
    row by row (a region appeared or vanished, or a timepoint was renamed)."""
    merged = before[keys].merge(after[keys], on=keys, how='outer', indicator=True)
    differences = []
    for label, side in (('only in BEFORE', 'left_only'), ('only in AFTER', 'right_only')):
        for _, row in merged[merged['_merge'] == side].iterrows():
            differences.append(f'row {label}: {_format_key(row, keys)}')
    if not differences:
        #--- same key set, but some key occurs a different number of times
        differences.append(f'the tables hold the same set of rows in different multiplicity '
                           f'({len(before)} rows before, {len(after)} rows after)')
    return differences


def compare(dfBefore, dfAfter, ignore_keys=()):
    """Return a list of human-readable differences; an empty list means the two
    tables agree. Rows are matched by sorting on the key columns rather than by
    position, so a reordered table is not reported as a difference. Ignoring a key
    column weakens that: rows it no longer separates keep their file order.

    Everything is compared exactly. Both run modes are bit-for-bit reproducible
    for a fixed image, and longitudinal runs are too once '--threads'/'--para' are
    pinned, so any difference at all is a real one - and the differences that
    matter here can be as small as one float32 ULP (see CONTRIBUTING.md)."""
    keys = key_columns(dfBefore, dfAfter, ignore_keys)
    if not keys:
        if ignore_keys:
            return [f'no key column is left to identify a row '
                    f'(--ignore-key dropped {", ".join(ignore_keys)})']
        return ['the two tables share none of the expected key columns']

    before = dfBefore.sort_values(keys, kind='stable').reset_index(drop=True)
    after = dfAfter.sort_values(keys, kind='stable').reset_index(drop=True)

    #--- once the rows differ, comparing values by position is only noise
    if len(before) != len(after) or not before[keys].equals(after[keys]):
        return _describe_row_changes(before, after, keys)

    differences = []
    for column in ('voxels', 'value'):
        b = pd.to_numeric(before[column], errors='coerce')
        a = pd.to_numeric(after[column], errors='coerce')
        #--- NaN on both sides counts as unchanged, NaN on one side does not
        differs = (b != a) & ~(b.isna() & a.isna())
        for i in before.index[differs]:
            delta = a[i] - b[i]
            rel = abs(delta / b[i]) if b[i] not in (0, np.nan) and pd.notna(b[i]) else float('nan')
            differences.append(
                f'{column} changed: {_format_key(before.loc[i], keys)}\n'
                f'    before {_format_number(b[i])}  after {_format_number(a[i])}  delta {delta:+g}'
                + ('' if pd.isna(rel) else f'  (relative {rel:.3g})'))
    return differences


def main(argv=None):
    args = iniParser().parse_args(argv)

    try:
        dfBefore = read_results(args.before)
        dfAfter = read_results(args.after)
        keys = key_columns(dfBefore, dfAfter, args.ignore_keys)
    except ValueError as err:
        print(f'ERROR: {err}', file=sys.stderr)
        return 2

    #--- indistinguishable rows get paired off in file order, which is only
    #    sound for two runs of the same pipeline
    if keys and (dfBefore.duplicated(subset=keys).any() or dfAfter.duplicated(subset=keys).any()):
        print(f'WARNING: {", ".join(keys)} do(es) not identify a row uniquely; rows sharing a key '
              f'are matched in file order.', file=sys.stderr)

    differences = compare(dfBefore, dfAfter, ignore_keys=args.ignore_keys)

    if not args.quiet:
        print(f'Comparing {len(dfBefore)} rows in "{args.before}"')
        print(f'     with {len(dfAfter)} rows in "{args.after}"')
        print(f'Rows keyed by {", ".join(keys) if keys else "(nothing)"}'
              + (f'; {", ".join(args.ignore_keys)} ignored' if args.ignore_keys else '') + '.')
        print('Metric values and voxel counts compared exactly.\n')
        for d in differences:
            print(f'  {d}')
        if differences:
            print()

    if differences:
        print(f'RESULT: {len(differences)} difference(s) found; this change is metric-affecting '
              f'and requires re-validation (see CONTRIBUTING.md).')
        return 1
    print('RESULT: no metric value and no voxel count changed.')
    return 0


if __name__ == "__main__":
    sys.exit(main())
