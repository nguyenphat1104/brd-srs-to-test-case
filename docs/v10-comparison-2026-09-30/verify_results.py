"""Verify the gated development results without invoking a model."""
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import tarfile

from brd_srs_testgen.benchmark import digest, trial_order, _trial_id, export_blind
from brd_srs_testgen.storage import RunRepository
from brd_srs_testgen.validation import validate_bundle

root = Path(__file__).resolve().parent
pilot = root.parent / 'benchmark-pilot-2026-09-29'
v7 = root.parent / 'v7-live-validation-2026-09-30'
v8 = root.parent / 'v8-live-validation-2026-09-30'
v9 = root.parent / 'v9-live-validation-2026-09-30'
manifest = json.loads((root / 'manifest.json').read_text())
assert digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'}) == manifest['manifest_sha256']
with tarfile.open(root / 'frozen-implementation.tar.gz') as archive:
    for name, expected in manifest['implementation']['source_sha256'].items():
        assert hashlib.sha256(archive.extractfile('src/brd_srs_testgen/' + name).read()).hexdigest() == expected
repository = RunRepository(os.environ['DATABASE_URL'])
old_rows = json.loads((pilot / 'results/results.json').read_text()) + json.loads((v7 / 'results/results.json').read_text()) + json.loads((v8 / 'results/results.json').read_text()) + json.loads((v9 / 'results/results.json').read_text())
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
assert prior == manifest['previously_accounted_tokens'] == 1092824
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
    trials.append({'document_id': row['document_id'], 'condition': row['condition'], 'run_id': row['run_id'],
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
    'checks': ['Recorded trial results equal the database and all twelve earlier trials are unchanged.',
        'Frozen source archive, PDFs and approved catalogs match their recorded hashes.',
        'Per-call accounting reconciles with run metrics and both budgets.',
        'Unpurchased efficient trial is absent when the gate stops execution.'],
    'trials': trials, 'prior_spent_tokens': prior, 'validation_spent_tokens': spent,
    'total_spent_tokens': prior + spent, 'remaining_authorized_tokens': 1600000 - prior - spent,
    'recorded_trials': len(rows), 'planned_trials': 2, 'stopped_by_gate': summary.get('stopped_by_gate'),
    'quality_preservation': 'Not established by development trials; held-out comparisons and independent human reviews remain required.'}
(root / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
extension = root / 'billing-resumption'
if (extension / 'result.json').exists():
    continuation_row = json.loads((extension / 'result.json').read_text())
    result = repository.load_run(continuation_row['run_id'])
    assert result.model_dump(mode='json') == continuation_row['result']
    blocked_row = json.loads((root / 'recovery/result.json').read_text())
    assert repository.load_run(blocked_row['run_id']).model_dump(mode='json') == blocked_row['result']
    protocol = json.loads((extension / 'protocol.json').read_text())
    assert digest({k: v for k, v in protocol.items() if k != 'sha256'}) == protocol['sha256']
    assert hashlib.sha256((root / 'resume_after_billing.py').read_bytes()).hexdigest() == protocol['harness_sha256']
    assert protocol['previous_conservative_total'] + protocol['maximum_additional_tokens'] <= 1600000
    ancestry, seen = [], {result.manifest.run_id}
    parent_id = result.manifest.configuration.get('recovery_parent_id')
    while parent_id:
        assert parent_id not in seen, 'Recovery ancestry must not cycle.'
        seen.add(parent_id)
        parent = repository.load_run(parent_id)
        ancestry.append({'run_id': parent_id, 'own_tokens': sum(c.budget_tokens for c in parent.call_attempts)})
        parent_id = parent.manifest.configuration.get('recovery_parent_id')
    inherited = sum(item['own_tokens'] for item in ancestry)
    assert inherited == 201599
    reused = [s for s in result.stage_outputs if s.reused_from]
    assert len(reused) == result.diagnostics.reused_tasks == 31
    source_checkpoints = {(s['stage'], s['task_index'], s['fingerprint']): s for s in blocked_row['result']['stage_outputs']}
    for task in reused:
        assert task.reused_from == blocked_row['run_id']
        original = source_checkpoints[(task.stage, task.task_index, task.fingerprint)]
        assert task.output == original['output'] and task.input_ids == original['input_ids']
    phases = Counter()
    for call in result.call_attempts:
        phases[call.phase] += call.budget_tokens
        assert call.status == 'completed' and call.estimated_tokens == 0
        assert call.budget_tokens == call.reported_total_tokens
    additional = sum(phases.values())
    assert phases == {'generation': 14767} and additional == result.metrics.charged_tokens
    assert phases['generation'] <= protocol['additional_generation_limit']
    assert phases['evaluation'] <= protocol['additional_judge_limit']
    assert result.manifest.status.value == 'failed' and result.manifest.failure_category.value == 'semantic_validation'
    assert result.coverage is None and result.coverage_evaluation is None
    assert validate_bundle(result.bundle, repository.load_chunks(result.manifest.run_id)).valid
    unresolved = [f.finding_id for f in result.diagnostics.unresolved_findings]
    assert unresolved == ['FIND-002']
    baseline = next(r for r in rows if r['condition'] == 'staged')
    export_blind([baseline, continuation_row], extension)
    recovery_report = {
        'verified_at': datetime.now(UTC).isoformat(), 'status': result.manifest.status.value,
        'run_id': result.manifest.run_id, 'failure': result.manifest.failure_message,
        'provider_billing_block_cleared': True, 'new_calls': len(result.call_attempts),
        'additional_tokens': additional, 'additional_estimated_tokens': 0, 'phases': dict(phases),
        'ancestry': ancestry, 'inherited_accounted_tokens': inherited,
        'raw_diagnostics_inherited_tokens': result.diagnostics.inherited_budget_tokens,
        'accounting_note': 'The frozen runner inherited only the interrupted direct parent cost because that parent has no aggregate diagnostics. Its reporting assertion stopped after the terminal result was saved. This read-only verification traverses both ancestor records; the immutable historical run is not rewritten.',
        'unobserved_retry_hold_tokens': protocol['unobserved_retry_hold_tokens'],
        'cumulative_workflow_accounted_tokens': inherited + additional,
        'cumulative_workflow_conservative_tokens': inherited + additional + protocol['unobserved_retry_hold_tokens'],
        'total_accounted_tokens': protocol['previously_accounted_tokens'] + additional,
        'conservative_total_tokens': protocol['previous_conservative_total'] + additional,
        'remaining_authorized_tokens': 1600000 - protocol['previous_conservative_total'] - additional,
        'reused_tasks': len(reused), 'new_saved_tasks': len(result.stage_outputs) - len(reused),
        'artifact_counts': {k: len(v) for k, v in result.bundle.model_dump().items()},
        'deterministic_validation': True, 'semantic_status': result.diagnostics.semantic_status,
        'unresolved_finding_ids': unresolved, 'f1': None,
        'historical_records_unchanged': len(old_rows) + len(rows) + 1,
        'note': 'Separate post-hoc continuation after user reported increasing the provider cap. No new recovery or Judge call purchased after semantic verification failed. No human scores supplied.'}
    assert recovery_report['conservative_total_tokens'] <= 1600000
    (extension / 'status.json').write_text(json.dumps(recovery_report, indent=2) + '\n')
    report['billing_resumption'] = recovery_report
print(json.dumps(report, indent=2))
