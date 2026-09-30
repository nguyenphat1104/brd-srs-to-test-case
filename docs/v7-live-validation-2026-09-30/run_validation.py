"""Two bounded v7 development regressions on previously approved source content."""
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

from brd_srs_testgen.benchmark import digest, execute, import_catalog, validate_manifest
from brd_srs_testgen.runner import EVALUATOR_VERSION, implementation_snapshot, run_generation
from brd_srs_testgen.storage import RunRepository


root = Path(__file__).resolve().parent
pilot = root.parent / 'benchmark-pilot-2026-09-29'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def log(message):
    line = f'{datetime.now(UTC).isoformat()} {message}'
    with (root / 'progress.log').open('a') as handle:
        handle.write(line + '\n')
    print(line, flush=True)


with (root / 'execution.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    original = json.loads((pilot / 'pilot-manifest.json').read_text())
    approval = json.loads((pilot / 'catalog-approval.json').read_text())
    assert approval['catalogs'] == original['source_catalog_drafts']
    prior_rows = json.loads((pilot / 'results/results.json').read_text())
    preparation = sum(json.loads(line)['budget_tokens'] for line in (pilot / 'D001-calls.jsonl').read_text().splitlines())
    prior_spent = preparation + sum(c['budget_tokens'] for row in prior_rows for c in row['result']['call_attempts'])
    assert prior_spent == 522328
    assert prior_spent + 600000 <= 1600000
    deployed = json.loads((root.parent / 'token-efficiency-v7-deployment-2026-09-30.json').read_text())
    assert implementation_snapshot() == deployed['implementation']
    manifest = {
        'protocol': 'v7-development-regression-v1', 'held_out_approved': False,
        'study_scope': 'Development regression on previously inspected pilot documents; not a held-out comparison or quality-preservation claim.',
        'documents': original['documents'], 'source_catalog_drafts': original['source_catalog_drafts'],
        'conditions': ['multi_efficient'], 'repeats': 1, 'seed': 20260930,
        'model': 'gemini-3.6-flash', 'thinking_level': 'minimal',
        'run_token_ceiling': 200000, 'max_total_budget_tokens': 600000,
        'total_user_authorized_budget_tokens': 1600000, 'previously_accounted_tokens': prior_spent,
        'unallocated_authorization_tokens': 477672, 'f1_tolerance': None,
        'implementation': implementation_snapshot(), 'evaluator_version': EVALUATOR_VERSION,
        'authorization': 'User requested continuation of live v7 validation; same PDFs and derived content already explicitly authorized for Gemini; existing total budget and exact catalog content approvals retained.',
        'comparability_limits': ['Generation ceiling increased from pilot 100000 to application default 200000.',
                                 'Evaluator prompt/version changed; old and new F1 are not like-for-like.',
                                 'The two source documents have already informed development.'],
    }
    manifest['manifest_sha256'] = digest(manifest)
    path = root / 'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest, 'Frozen development protocol changed.'
    else:
        save(path, manifest)
    validate_manifest(manifest)
    repository = RunRepository(os.environ['DATABASE_URL'])
    for row in prior_rows:
        assert repository.load_run(row['run_id']).model_dump(mode='json') == row['result']
    for document_id, expected in manifest['source_catalog_drafts'].items():
        data = (pilot / expected['file']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected['sha256']
        catalog = import_catalog(manifest, document_id, json.loads(data), repository, approve=True)
        log(f'{document_id}: unchanged approved catalog imported for {EVALUATOR_VERSION}; {len(catalog.units)} units.')
    log('Preflight passed: previous costs reconciled, v7 code frozen, source and catalog hashes checked.')
    if '--prepare-only' in sys.argv:
        log('Prepared only; no model calls.')
        raise SystemExit(0)

    status = {'state': 'running', 'manifest_sha256': manifest['manifest_sha256'],
              'started_at': datetime.now(UTC).isoformat(), 'previously_accounted_tokens': prior_spent,
              'validation_budget_tokens': 600000, 'human_ratings_collected': 0}
    save(root / 'status.json', status)

    def observed_generation(*args, **kwargs):
        log(f'TRIAL START {args[1]} / {args[3].pipeline_profile}')
        result = run_generation(*args, **kwargs, progress=log)
        log(f'TRIAL END {result.manifest.run_id}: {result.manifest.status.value}; '
            f'{sum(c.budget_tokens for c in result.call_attempts)} accounted tokens.')
        return result

    try:
        summary = execute(manifest, repository, root / 'results', generation=observed_generation)
        spent = summary['budget_accounted_tokens']
        assert spent <= 600000 and prior_spent + spent <= 1600000
        status.update(state='live_trials_recorded_human_review_pending',
            recorded_trials=summary['recorded_trials'], validation_accounted_tokens=spent,
            total_accounted_tokens=prior_spent + spent, remaining_authorized_tokens=1600000-prior_spent-spent)
        log(json.dumps({k: status[k] for k in ('recorded_trials','validation_accounted_tokens','total_accounted_tokens','remaining_authorized_tokens')}))
    except Exception as error:
        status.update(state='interrupted_inspect_durable_records', error_type=type(error).__name__)
        raise
    finally:
        status['updated_at'] = datetime.now(UTC).isoformat()
        save(root / 'status.json', status)
