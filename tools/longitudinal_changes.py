"""Write follow-up minus baseline metric changes without annualizing them."""
import argparse
import csv
from decimal import Decimal

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('results')
parser.add_argument('output')
args = parser.parse_args()
groups = {}
with open(args.results, newline='') as stream:
    for row in csv.DictReader(stream):
        if row['timepoint'] not in ('ses-1', 'ses-2') or row['metric'] not in ('MSMD', 'PSMD', 'MSFW'):
            continue
        key = (row.get('ID', ''), row.get('skeleton', ''), row['region'], row['metric'])
        sessions = groups.setdefault(key, {})
        if row['timepoint'] in sessions:
            raise ValueError(f'Duplicate row for {key}: {row["timepoint"]}')
        value = Decimal(row['value'])
        if not value.is_finite():
            raise ValueError(f'Nonfinite endpoint for {key}')
        sessions[row['timepoint']] = value
if not groups:
    raise ValueError('No endpoint rows found')
with open(args.output, 'w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['ID', 'skeleton', 'region', 'metric', 'ses-1', 'ses-2', 'change_ses-2_minus_ses-1'])
    for key, sessions in sorted(groups.items()):
        if set(sessions) != {'ses-1', 'ses-2'}:
            raise ValueError(f'Missing session for {key}')
        writer.writerow([*key, sessions['ses-1'], sessions['ses-2'], sessions['ses-2'] - sessions['ses-1']])
