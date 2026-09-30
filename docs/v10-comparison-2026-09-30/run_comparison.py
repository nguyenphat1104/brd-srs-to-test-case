"""Gated development validation on the same approved content; no held-out claim."""
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile

from brd_srs_testgen.benchmark import digest, execute, export_blind, import_catalog, summarize, trial_order, validate_manifest
from brd_srs_testgen.runner import EVALUATOR_VERSION, implementation_snapshot, run_generation
from brd_srs_testgen.storage import RunRepository

root = Path(__file__).resolve().parent
pilot = root.parent / 'benchmark-pilot-2026-09-29'
v7 = root.parent / 'v7-live-validation-2026-09-30'
v8 = root.parent / 'v8-live-validation-2026-09-30'
v9 = root.parent / 'v9-live-validation-2026-09-30'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def log(message):
    line = f'{datetime.now(UTC).isoformat()} {message}'
    with (root / 'progress.log').open('a') as handle:
        handle.write(line + '\n')
    print(line, flush=True)


class StopValidation(Exception):
    pass


with (root / 'execution.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    original = json.loads((pilot / 'pilot-manifest.json').read_text())
    prior_rows = json.loads((pilot / 'results/results.json').read_text()) + json.loads((v7 / 'results/results.json').read_text()) + json.loads((v8 / 'results/results.json').read_text()) + json.loads((v9 / 'results/results.json').read_text())
    preparation = sum(json.loads(line)['budget_tokens'] for line in (pilot / 'D001-calls.jsonl').read_text().splitlines())
    prior_spent = preparation + sum(c['budget_tokens'] for row in prior_rows for c in row['result']['call_attempts'])
    assert prior_spent == 1092824 and prior_spent + 507176 <= 1600000
    deployed = json.loads((root / 'validation-build.json').read_text())
    assert implementation_snapshot() == deployed['implementation'], 'Code differs from tested v10 image.'
    manifest = {
        'protocol': 'v10-matched-development-comparison-v1', 'held_out_approved': False,
        'documents': original['documents'][:1], 'source_catalog_drafts': {'D001': original['source_catalog_drafts']['D001']},
        'conditions': ['staged', 'multi_efficient'], 'repeats': 1, 'seed': 20260930,
        'model': 'gemini-3.6-flash', 'thinking_level': 'minimal',
        'run_token_ceiling': 200000, 'max_total_budget_tokens': 507176,
        'total_user_authorized_budget_tokens': 1600000, 'previously_accounted_tokens': prior_spent,
        'f1_tolerance': None, 'implementation': implementation_snapshot(), 'evaluator_version': EVALUATOR_VERSION,
        'gate': 'Run the staged baseline first; stop before the efficient trial unless staged completes generation and evaluation; no automatic recovery purchases.',
        'authorization': 'User continued implementation and validation; same PDFs and derived content already approved for Gemini, within the original total budget.',
        'comparability_limits': ['Previously inspected development documents, not held out.',
            'Critic payload sharing and evaluator serialization/batching changed; exact data reconstruction does not establish model quality preservation.',
            'Both conditions use 200000 generation tokens, the same coverage-v7 evaluator, model and approved catalog; historical scores are not directly comparable.'],
    }
    manifest['manifest_sha256'] = digest(manifest)
    path = root / 'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest, 'Frozen protocol changed.'
    else:
        save(path, manifest)
    validate_manifest(manifest)
    assert [condition for _, condition, _ in trial_order(manifest)] == ['staged', 'multi_efficient']
    archive_path = root / 'frozen-implementation.tar.gz'
    if not archive_path.exists():
        with tarfile.open(archive_path, 'w:gz') as archive:
            for name in manifest['implementation']['source_sha256']:
                archive.add(root.parent.parent / 'src/brd_srs_testgen' / name, arcname='src/brd_srs_testgen/' + name)
    with tarfile.open(archive_path) as archive:
        for name, expected in manifest['implementation']['source_sha256'].items():
            assert hashlib.sha256(archive.extractfile('src/brd_srs_testgen/' + name).read()).hexdigest() == expected
    repository = RunRepository(os.environ['DATABASE_URL'])
    for row in prior_rows:
        assert repository.load_run(row['run_id']).model_dump(mode='json') == row['result']
    assert json.loads((pilot / 'catalog-approval.json').read_text())['catalogs']['D001'] == manifest['source_catalog_drafts']['D001']
    for document_id, expected in manifest['source_catalog_drafts'].items():
        data = (pilot / expected['file']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected['sha256']
        import_catalog(manifest, document_id, json.loads(data), repository, approve=True)
    log('Preflight passed: twelve earlier trials unchanged, budgets reconciled, v10 source archived, approved content unchanged.')
    if '--prepare-only' in sys.argv:
        raise SystemExit(0)
    status = {'state': 'running', 'manifest_sha256': manifest['manifest_sha256'],
        'started_at': datetime.now(UTC).isoformat(), 'previously_accounted_tokens': prior_spent}
    save(root / 'status.json', status)
    output = root / 'results'

    def observed_generation(*args, **kwargs):
        result_path = output / 'results.json'
        if result_path.exists():
            recorded = json.loads(result_path.read_text())
            if any(row['result']['manifest']['status'] != 'completed' or row['result']['coverage'] is None for row in recorded):
                raise StopValidation('Staged baseline did not complete generation and evaluation; efficient comparison not purchased.')
        log(f'TRIAL START {args[1]}')
        result = run_generation(*args, **kwargs, progress=log)
        log(f'TRIAL END {result.manifest.run_id}: {result.manifest.status.value}; {sum(c.budget_tokens for c in result.call_attempts)} tokens.')
        return result

    try:
        try:
            summary = execute(manifest, repository, output, generation=observed_generation)
        except StopValidation as error:
            log(str(error))
            rows = json.loads((output / 'results.json').read_text())
            export_blind(rows, output)
            summary = summarize(rows, None, manifest['seed'])
            summary.update(planned_trials=2, recorded_trials=len(rows),
                budget_accounted_tokens=sum(c['budget_tokens'] for row in rows for c in row['result']['call_attempts']),
                stopped_by_gate=str(error))
            save(output / 'summary.json', summary)
        spent = summary['budget_accounted_tokens']
        assert spent <= 507176 and prior_spent + spent <= 1600000
        status.update(state='gated_validation_recorded', recorded_trials=summary['recorded_trials'],
            stopped_by_gate=summary.get('stopped_by_gate'), validation_spent_tokens=spent,
            total_spent_tokens=prior_spent + spent, remaining_authorized_tokens=1600000 - prior_spent - spent)
        log(json.dumps(status))
    except Exception as error:
        status.update(state='interrupted_inspect_durable_records', error_type=type(error).__name__)
        raise
    finally:
        status['updated_at'] = datetime.now(UTC).isoformat()
        save(root / 'status.json', status)
