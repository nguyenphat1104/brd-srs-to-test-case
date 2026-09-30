"""Record an observed process failure without retrying any model calls."""
from datetime import datetime
import json
import os
from pathlib import Path

from brd_srs_testgen.models import FailureCategory, RunStatus
from brd_srs_testgen.storage import RunRepository

root = Path(__file__).resolve().parent
repository = RunRepository(os.environ['DATABASE_URL'])
run_id = '228f8d258152-9b91a42b8a9c5070aadffefd721b8bf1'
result = repository.load_run(run_id)
if result.manifest.status is RunStatus.RUNNING:
    assert len(result.call_attempts) == 4
    assert sum(c.budget_tokens for c in result.call_attempts) == 18406
    (root / 'D001-before-finalization.json').write_text(result.model_dump_json(indent=2) + '\n')
    result.manifest = result.manifest.model_copy(update={
        'status': RunStatus.FAILED,
        'completed_at': datetime.fromisoformat('2026-09-30T06:11:35.518164+00:00'),
        'failure_category': FailureCategory.SEMANTIC_VALIDATION,
        'failure_message': 'Unhandled ValueError after one local correction: Unknown or out-of-scope evidence ID. Process exited; recorded from durable calls and checkpoints without retry.',
    })
    repository.finalize(result)
    assert repository.load_run(run_id).model_dump(mode='json') == result.model_dump(mode='json')
    (root / 'D001-finalization-provenance.json').write_text(json.dumps({
        'run_id': run_id, 'reason': result.manifest.failure_message,
        'charged_tokens': 18406, 'calls_preserved': 4,
        'metrics': 'Unavailable: exception escaped runner before aggregate metrics were persisted. Use immutable per-call accounting.',
        'application_source_changed': False, 'model_calls_repeated': 0,
    }, indent=2) + '\n')
else:
    assert result.manifest.status is RunStatus.FAILED
print(f'{run_id}: {result.manifest.status.value}; {sum(c.budget_tokens for c in result.call_attempts)} tokens preserved.')
