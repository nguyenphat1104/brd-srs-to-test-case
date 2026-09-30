"""Seal the stopped billing-blocked recovery without sending any provider request."""
from datetime import UTC, datetime
import json
import os
from pathlib import Path

from brd_srs_testgen.models import FailureCategory, RunStatus
from brd_srs_testgen.storage import RunRepository

root = Path(__file__).resolve().parent
output = root / 'recovery'
repo = RunRepository(os.environ['DATABASE_URL'])
run_id = '228f8d258152-93f5e43fcdcc593997ed320b56ffd569'
parent_id = '228f8d258152-ab53f20cf7e65c37b3abbc83c947bb90'
result = repo.load_run(run_id)
assert len(result.call_attempts) == 1
call = result.call_attempts[0]
assert call.budget_tokens == call.estimated_tokens == 12497
assert 'monthly spending cap' in call.error
assert len(result.stage_outputs) == 31
assert all(row.reused_from == parent_id for row in result.stage_outputs)
if result.manifest.status is RunStatus.RUNNING:
    (output / 'before-finalization.json').write_text(result.model_dump_json(indent=2) + '\n')
    result.manifest = result.manifest.model_copy(update={
        'status': RunStatus.FAILED, 'completed_at': datetime.now(UTC),
        'failure_category': FailureCategory.PROVIDER_REJECTION,
        'failure_message': 'Gemini rejected the repair: project monthly spending cap exceeded. Recovery container stopped to prevent repeated attempts. One failed call has estimated usage; an in-flight retry may be unrecorded. No repair or evaluation completed.',
    })
    repo.finalize(result)
assert result.manifest.status is RunStatus.FAILED
assert repo.load_run(run_id).model_dump(mode='json') == result.model_dump(mode='json')
for row in json.loads((root / 'results/results.json').read_text()):
    assert repo.load_run(row['run_id']).model_dump(mode='json') == row['result']
row = {'document_id':'D001','condition':'multi_efficient_recovery','repeat':0,
       'run_id':run_id,'result':result.model_dump(mode='json')}
(output / 'result.json').write_text(json.dumps(row, indent=2) + '\n')
status = {
    'status':'failed_provider_billing_cap', 'run_id':run_id, 'parent_run_id':parent_id,
    'prior_accounted_tokens':1346446, 'additional_provider_reported_tokens':None,
    'persisted_estimated_tokens':12497, 'unobserved_retry_reserve_tokens':12497,
    'recorded_accounted_total':1358943, 'conservative_total_including_retry_reserve':1371440,
    'conservative_remaining_authorized_tokens':228560,
    'reused_tasks':31, 'new_completed_tasks':0, 'f1':None,
    'note':'A second sequential attempt may have been in flight at termination; its same-request reservation is held separately, not fabricated as an observed call. Reconcile with provider usage before releasing this hold.',
    'metrics':'Unavailable because the process was stopped; no aggregate metrics or human scores fabricated.',
    'completed_at_semantics':'Manual failure-recording timestamp, not an inferred provider completion time.',
}
(output / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
print(json.dumps(status, indent=2))
