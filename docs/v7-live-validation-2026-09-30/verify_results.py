"""Reconcile this development regression against immutable database records."""
from collections import Counter
import csv
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import tarfile

from brd_srs_testgen.benchmark import digest, trial_order, _trial_id
from brd_srs_testgen.storage import RunRepository

root = Path(__file__).resolve().parent
pilot = root.parent / 'benchmark-pilot-2026-09-29'
manifest = json.loads((root / 'manifest.json').read_text())
assert digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'}) == manifest['manifest_sha256']
with tarfile.open(root / 'frozen-implementation.tar.gz') as archive:
    for name, expected in manifest['implementation']['source_sha256'].items():
        assert hashlib.sha256(archive.extractfile('src/brd_srs_testgen/' + name).read()).hexdigest() == expected
repository = RunRepository(os.environ['DATABASE_URL'])
old_rows = json.loads((pilot / 'results/results.json').read_text())
rows = json.loads((root / 'results/results.json').read_text())
planned = {f"{doc['sha256'][:12]}-{_trial_id(manifest, doc, condition, repeat)}"
           for doc, condition, repeat in trial_order(manifest)}
assert len(rows) == len(planned) == 2 and {row['run_id'] for row in rows} == planned
for row in old_rows + rows:
    assert repository.load_run(row['run_id']).model_dump(mode='json') == row['result']
for document in manifest['documents']:
    assert hashlib.sha256(Path(document['path']).read_bytes()).hexdigest() == document['sha256']
    expected = manifest['source_catalog_drafts'][document['document_id']]
    data = (pilot / expected['file']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected['sha256']
    catalog = repository.load_coverage_catalog(document['sha256'], manifest['evaluator_version'])
    assert catalog.status.value == 'approved'
    assert [u.model_dump(mode='json') for u in catalog.units] == json.loads(data)['units']
preparation = sum(json.loads(line)['budget_tokens'] for line in (pilot / 'D001-calls.jsonl').read_text().splitlines())
prior = preparation + sum(c['budget_tokens'] for row in old_rows for c in row['result']['call_attempts'])
assert prior == manifest['previously_accounted_tokens'] == 522328
trials = []
for row in rows:
    result = row['result']
    assert result['manifest']['status'] != 'running'
    phases, stages = Counter(), Counter()
    for call in result['call_attempts']:
        phases[call['phase']] += call['budget_tokens']
        stages[call['phase'] + '/' + call['stage']] += call['budget_tokens']
    spent = sum(phases.values())
    if result['metrics'] is not None:
        assert spent == result['metrics']['charged_tokens']
    else:
        assert row['document_id'] == 'D001' and spent == 18406
    trials.append({
        'document_id': row['document_id'], 'run_id': row['run_id'],
        'status': result['manifest']['status'],
        'failure_category': result['manifest']['failure_category'],
        'failure_message': result['manifest']['failure_message'],
        'budget_tokens': spent, 'phases': dict(phases), 'stages': dict(stages),
        'calls': len(result['call_attempts']),
        'estimated_tokens': sum(c['estimated_tokens'] for c in result['call_attempts']),
        'artifact_counts': {k: len(v) for k, v in (result['bundle'] or {}).items()},
        'f1': result['coverage']['f1'] if result['coverage'] else None,
    })
spent = sum(t['budget_tokens'] for t in trials)
summary = json.loads((root / 'results/summary.json').read_text())
assert spent == summary['budget_accounted_tokens'] <= manifest['max_total_budget_tokens']
assert prior + spent <= manifest['total_user_authorized_budget_tokens']
ratings = []
for path in (pilot / 'results/blind').glob('ratings-R*.csv'):
    with path.open(newline='') as handle:
        ratings.extend(row for row in csv.DictReader(handle)
                       if any(row[k] for k in ('source_coverage', 'groundedness', 'executability', 'redundancy_control', 'critical_errors')))
report = {
    'verified_at': datetime.now(UTC).isoformat(),
    'manifest_sha256': manifest['manifest_sha256'],
    'checks': ['Both expected trials terminal and identical to database exports.',
               'All eight original pilot results unchanged.',
               'Frozen source archive, source PDFs and approved catalogs match recorded hashes.',
               'Costs reconcile from immutable call attempts and remain inside both budgets.'],
    'trials': trials, 'prior_spent_tokens': prior, 'validation_spent_tokens': spent,
    'total_spent_tokens': prior + spent,
    'remaining_authorized_tokens': 1600000 - prior - spent,
    'original_pilot_rows_with_human_scores': len(ratings),
    'quality_preservation': 'Not established. Development documents; changed generation ceiling/evaluator; human reviews incomplete.',
}
(root / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
