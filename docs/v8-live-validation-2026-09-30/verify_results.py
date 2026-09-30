"""Verify the gated development results without invoking a model."""
from collections import Counter
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
v7 = root.parent / 'v7-live-validation-2026-09-30'
manifest = json.loads((root / 'manifest.json').read_text())
assert digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'}) == manifest['manifest_sha256']
with tarfile.open(root / 'frozen-implementation.tar.gz') as archive:
    for name, expected in manifest['implementation']['source_sha256'].items():
        assert hashlib.sha256(archive.extractfile('src/brd_srs_testgen/' + name).read()).hexdigest() == expected
repository = RunRepository(os.environ['DATABASE_URL'])
old_rows = json.loads((pilot / 'results/results.json').read_text()) + json.loads((v7 / 'results/results.json').read_text())
rows = json.loads((root / 'results/results.json').read_text())
planned = [f"{doc['sha256'][:12]}-{_trial_id(manifest, doc, condition, repeat)}" for doc, condition, repeat in trial_order(manifest)]
assert 1 <= len(rows) <= 2 and [row['run_id'] for row in rows] == planned[:len(rows)]
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
assert prior == manifest['previously_accounted_tokens'] == 732219
trials = []
for row in rows:
    result = row['result']
    assert result['manifest']['status'] != 'running'
    stages, phases, statuses = Counter(), Counter(), Counter()
    for call in result['call_attempts']:
        stages[call['phase'] + '/' + call['stage']] += call['budget_tokens']
        phases[call['phase']] += call['budget_tokens']
        statuses[call['status']] += 1
    spent = sum(phases.values())
    assert spent == result['metrics']['charged_tokens']
    trials.append({'document_id': row['document_id'], 'run_id': row['run_id'],
        'status': result['manifest']['status'], 'failure_message': result['manifest']['failure_message'],
        'budget_tokens': spent, 'phases': dict(phases), 'stages': dict(stages), 'call_statuses': dict(statuses),
        'estimated_tokens': sum(c['estimated_tokens'] for c in result['call_attempts']),
        'artifact_counts': {k: len(v) for k, v in (result['bundle'] or {}).items()},
        'f1': result['coverage']['f1'] if result['coverage'] else None,
        'evaluation_error': result['diagnostics']['evaluation_error'] if result['diagnostics'] else None})
summary = json.loads((root / 'results/summary.json').read_text())
spent = sum(t['budget_tokens'] for t in trials)
assert spent == summary['budget_accounted_tokens'] <= manifest['max_total_budget_tokens']
assert prior + spent <= 1600000
if len(rows) == 1:
    assert summary['stopped_by_gate']
    with repository._connect() as connection:
        assert connection.execute('SELECT count(*) AS n FROM runs WHERE run_id=%s', (planned[1],)).fetchone()['n'] == 0
report = {'verified_at': datetime.now(UTC).isoformat(), 'manifest_sha256': manifest['manifest_sha256'],
    'checks': ['Recorded trial results equal the database and all ten earlier trials are unchanged.',
        'Frozen source archive, PDFs and approved catalogs match their recorded hashes.',
        'Per-call accounting reconciles with run metrics and both budgets.',
        'Unpurchased second trial is absent when the gate stops execution.'],
    'trials': trials, 'prior_spent_tokens': prior, 'validation_spent_tokens': spent,
    'total_spent_tokens': prior + spent, 'remaining_authorized_tokens': 1600000 - prior - spent,
    'recorded_trials': len(rows), 'planned_trials': 2, 'stopped_by_gate': summary.get('stopped_by_gate'),
    'quality_preservation': 'Not established by development trials; held-out comparisons and independent human reviews remain required.'}
(root / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
