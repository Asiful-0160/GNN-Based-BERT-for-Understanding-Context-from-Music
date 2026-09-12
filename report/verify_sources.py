from pathlib import Path
import json, hashlib, re, csv
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parent
results = root.parent / 'RESULTS'
files = list(results.rglob('*'))
files = [p for p in files if p.is_file()]
records = []
for p in files:
    data = p.read_bytes()
    if p.suffix == '.json':
        json.loads(data)
    if p.suffix == '.png':
        with Image.open(p) as im:
            im.verify()
    if p.suffix == '.csv':
        rows = list(csv.reader(data.decode('utf-8-sig').splitlines()))
        assert rows and all(len(row) == len(rows[0]) for row in rows), p
    if p.suffix == '.npz':
        with np.load(p, allow_pickle=False) as archive:
            for key in archive.files:
                _ = archive[key].shape
    records.append({'path': str(p.relative_to(results)), 'bytes': len(data),
                    'sha256': hashlib.sha256(data).hexdigest()})
checks = []
for name, prefix in [('report_assets/report_assets_manifest.json', 'report_assets'),
                     ('final_test/final_test_evaluation.json', 'final_test'),
                     ('final_test/posthoc_analysis/output_hashes.json', 'final_test/posthoc_analysis')]:
    obj = json.loads((results / name).read_text(encoding='utf-8'))
    mapping = obj.get('output_hashes', obj if 'output_hashes' in name else {})
    for filename, expected in mapping.items():
        if not isinstance(expected, str):
            continue
        p = results / prefix / filename
        actual = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
        checks.append({'path': str(p.relative_to(results)), 'matches_recorded_hash': actual == expected})
summary = json.loads((results / 'final_t4/task4_final_summary.json').read_text())
for group in ['figures', 'tables']:
    for record in summary[group].values():
        p = results / 'final_t4' / group / Path(record['path']).name
        checks.append({'path': str(p.relative_to(results)),
                       'matches_recorded_hash': hashlib.sha256(p.read_bytes()).hexdigest() == record['sha256']})
tex = (root / 'main.tex').read_text()
lock = json.loads((results / 'final_t4/task4_final_artifact_lock.json').read_text())
for record in lock['final_outputs'].values():
    original = record['path'].split('/final/', 1)[1]
    p = results / 'final_t4' / original
    checks.append({'path': str(p.relative_to(results)),
                   'matches_recorded_hash': p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == record['sha256']})
for p in results.rglob('*.png'):
    copy = root / 'figures' / p.name
    checks.append({'path': 'report/figures/' + p.name,
                   'matches_recorded_hash': copy.is_file() and copy.read_bytes() == p.read_bytes()})
labels = set(re.findall(r'\\label\{([^}]+)\}', tex))
refs = set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', tex))
figures = re.findall(r'\\includegraphics\[[^]]*\]\{([^}]+)\}', tex)
figures += re.findall(r'\\pendingfigure\{([^}]+)\}', tex)
figures = [f for f in figures if '#' not in f]
bibkeys = set(re.findall(r'@\w+\{([^,]+),', (root / 'references.bib').read_text()))
cites = {k for group in re.findall(r'\\cite\{([^}]+)\}', tex) for k in group.split(',')}
report = {'files_read': len(files), 'files': records, 'hash_checks': checks,
          'hash_mismatches': [x for x in checks if not x['matches_recorded_hash']],
          'undefined_source_references': sorted(refs-labels),
          'missing_figure_assets': [f for f in figures if not (root / 'figures' / f).is_file()],
          'undefined_citations': sorted(cites-bibkeys),
          'unused_float_labels': sorted(k for k in labels-refs if k.startswith(('fig:', 'tab:'))),
          'scope': 'Local export only. No training, inference, metric recalculation, or upstream checkpoint verification.'}
(root / 'source_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['files','hash_checks']}, indent=2))

assert not report['hash_mismatches'], 'Source hash mismatch'
assert not report['undefined_source_references'], 'Undefined LaTeX reference'
assert not report['undefined_citations'], 'Undefined citation'
