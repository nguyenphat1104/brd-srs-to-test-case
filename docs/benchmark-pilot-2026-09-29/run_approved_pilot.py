"""Execute the frozen pilot; this wrapper only records approval and progress."""
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

from brd_srs_testgen.benchmark import execute, import_catalog, trial_order, validate_manifest
from brd_srs_testgen.runner import run_generation
from brd_srs_testgen.storage import RunRepository


root = Path(__file__).resolve().parent
manifest = json.loads((root / 'pilot-manifest.json').read_text())
validate_manifest(manifest)
catalogs = {}
for document_id, expected in manifest['source_catalog_drafts'].items():
    data = (root / expected['file']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected['sha256']
    catalogs[document_id] = json.loads(data)
    assert len(catalogs[document_id]['units']) == expected['units']
preparation = sum(json.loads(line)['budget_tokens'] for line in (root / 'D001-calls.jsonl').read_text().splitlines())
assert preparation == manifest['catalog_preparation_accounted_tokens'] == 23613
assert preparation + manifest['max_total_budget_tokens'] == 1600000
assert len(trial_order(manifest)) == 8
print('Preflight passed: frozen implementation, source/catalog hashes, eight trials, total budget reconciled.', flush=True)
if '--dry-run' in sys.argv:
    raise SystemExit(0)


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


with (root / 'execution.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    approval_path = root / 'catalog-approval.json'
    if not approval_path.exists():
        save(approval_path, {
            'recorded_at': datetime.now(UTC).isoformat(),
            'user_statement': 'Approve both source catalogs',
            'manifest_sha256': manifest['manifest_sha256'],
            'catalogs': manifest['source_catalog_drafts'],
            'scope': 'Source catalog contents only; generated-suite human ratings remain pending.',
        })
    approval = json.loads(approval_path.read_text())
    assert approval['manifest_sha256'] == manifest['manifest_sha256']
    assert approval['catalogs'] == manifest['source_catalog_drafts']
    repository = RunRepository(os.environ['DATABASE_URL'])
    repository.initialize()
    for document_id, units in catalogs.items():
        catalog = import_catalog(manifest, document_id, units, repository, approve=True)
        print(f'{document_id}: {len(catalog.units)} source units {catalog.status.value}.', flush=True)
    status_path = root / 'pilot-status.json'
    status = json.loads(status_path.read_text())
    status.update(state='catalogs_approved_external_transfer_permission_pending' if '--approve-only' in sys.argv else 'running_comparisons', human_catalogs_approved=2,
                  updated_at=datetime.now(UTC).isoformat())
    save(status_path, status)
    if '--approve-only' in sys.argv:
        print('Catalog approval recorded. No model calls made.', flush=True)
        raise SystemExit(0)
    output = root / 'results'

    def observed_generation(*args, **kwargs):
        print(f'TRIAL START {args[1]} / {args[2].value} / {args[3].pipeline_profile}', flush=True)
        result = run_generation(*args, **kwargs,
            progress=lambda message: print(f'{datetime.now(UTC).isoformat()} {message}', flush=True))
        tokens = sum(call.budget_tokens for call in result.call_attempts)
        print(f'TRIAL END {result.manifest.run_id}: {result.manifest.status.value}; {tokens} tokens', flush=True)
        return result

    try:
        summary = execute(manifest, repository, output, generation=observed_generation)
        status.update(state='comparisons_recorded_human_ratings_pending',
                      benchmark_trials_executed=summary['recorded_trials'],
                      comparison_accounted_tokens=summary['budget_accounted_tokens'],
                      total_accounted_tokens=preparation + summary['budget_accounted_tokens'],
                      remaining_authorized_tokens=1600000 - preparation - summary['budget_accounted_tokens'])
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        status.update(state='execution_interrupted_inspect_durable_run_records', error_type=type(error).__name__)
        raise
    finally:
        status['updated_at'] = datetime.now(UTC).isoformat()
        save(status_path, status)
