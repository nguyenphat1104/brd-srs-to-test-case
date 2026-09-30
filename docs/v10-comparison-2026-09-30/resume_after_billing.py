"""User-authorized continuation after a reported billing-cap increase."""
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from uuid import NAMESPACE_URL, uuid5

from brd_srs_testgen.benchmark import digest, export_blind, validate_manifest
from brd_srs_testgen.efficient import run_efficient_pipeline
from brd_srs_testgen.models import ArtifactPatch, RunStatus
from brd_srs_testgen.pipelines import PipelineContext, default_agent_setups
from brd_srs_testgen.providers import BudgetLedger, ProviderError
from brd_srs_testgen.runner import ProviderSettings, _make_provider, _make_judge_provider, implementation_snapshot, run_generation
from brd_srs_testgen.storage import RunRepository, StorageError

root = Path(__file__).resolve().parent
output = root / 'billing-resumption'
output.mkdir(exist_ok=True)
manifest = json.loads((root / 'manifest.json').read_text())
validate_manifest(manifest)
comparison = json.loads((root / 'results/results.json').read_text())
verified = json.loads((root / 'verification.json').read_text())
assert verified['total_spent_tokens'] == 1346446
assert verified['remaining_authorized_tokens'] == 253554
parent_row = json.loads((root / 'recovery/result.json').read_text())
blocked = json.loads((root / 'recovery/status.json').read_text())
assert blocked['conservative_total_including_retry_reserve'] == 1371440
assert blocked['conservative_remaining_authorized_tokens'] == 228560
repo = RunRepository(os.environ['DATABASE_URL'])
parent = repo.load_run(parent_row['run_id'])
assert parent.model_dump(mode='json') == parent_row['result']
assert parent.manifest.status is RunStatus.FAILED and parent.manifest.failure_category.value == 'provider_rejection'
assert parent.manifest.configuration['implementation'] == implementation_snapshot()
protocol = {
    'type': 'posthoc-billing-resumption-v1', 'parent_run_id': parent.manifest.run_id,
    'original_trial_run_id': blocked['parent_run_id'],
    'authorization': 'User: increased budget try again. This refers to the provider spending cap; the original 1600000-token experiment ceiling remains unchanged.',
    'comparison_manifest_sha256': manifest['manifest_sha256'],
    'previously_accounted_tokens': 1358943, 'unobserved_retry_hold_tokens': 12497,
    'previous_conservative_total': 1371440, 'additional_generation_limit': 100000,
    'additional_judge_limit': 100000, 'maximum_additional_tokens': 200000,
    'total_authorized_tokens': 1600000,
    'configured_generation_limit': 200000,
    'budget_note': 'Original settings remain identical for checkpoint fingerprints. An external study limit tightens the new generation ledger to 100000. Inherited cost is reported separately and included in cumulative workflow cost.',
    'interpretation': 'Supplementary recovery after a failed matched trial; not a fresh repetition or a pass at the original cumulative 200000 ceiling.',
    'gate': 'Verify all 31 saved tasks are reused before the first new repair call. No new recovery run if this continuation fails; ordinary bounded in-run correction and transient retries remain unchanged.',
    'frozen_image': 'sha256:641da14a8f074ed98d550410ec0dca1e9f498ed59f91d6485c7f2494654a9ca1',
    'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'external_provider_guard': 'Frozen v10 generation, prompts, schemas and checkpoints. This harness additionally marks explicit monthly/project spending-cap 429 errors nonretryable, matching v11, for generation and Judge. This external failure-only guard is not part of the frozen backend fingerprint.',
}
protocol['sha256'] = digest(protocol)
path = output / 'protocol.json'
if path.exists():
    assert json.loads(path.read_text()) == protocol
else:
    path.write_text(json.dumps(protocol, indent=2) + '\n')
assert protocol['previous_conservative_total'] + protocol['maximum_additional_tokens'] <= 1600000
settings = ProviderSettings(provider='gemini', model=manifest['model'],
    thinking_level=manifest['thinking_level'], token_ceiling=200000, pipeline_profile='efficient',
    api_key=os.environ.get('GEMINI_API_KEY', ''), agent_setups=default_agent_setups())
configuration = {**settings.snapshot(parent.manifest.run_type), 'implementation': implementation_snapshot()}
assert configuration == {k: v for k, v in parent.manifest.configuration.items() if k != 'recovery_parent_id'}
signature = hashlib.sha256(json.dumps({'document_hash': parent.manifest.document_hash,
    'configuration': configuration}, sort_keys=True).encode()).hexdigest()


class ReachedRepair(Exception):
    pass


class DryProvider:
    model = manifest['model']
    ledger = BudgetLedger(100000)
    def generate(self, messages, schema, **kwargs):
        assert schema is ArtifactPatch, 'A saved upstream task would be purchased again.'
        raise ReachedRepair


preview = PipelineContext(provider=DryProvider(), efficient=True, recovery_signature=signature,
    recovery_parent=parent.manifest.run_id, checkpoints=parent.stage_outputs)
try:
    run_efficient_pipeline(preview, repo.load_chunks(parent.manifest.run_id))
except ReachedRepair:
    pass
else:
    raise AssertionError('Expected the unfinished repair boundary.')
assert preview.diagnostics.reused_tasks == len(parent.stage_outputs) == 31
assert len(preview.call_attempts) == 1 and preview.call_attempts[0].stage == 'repair'
assert preview.call_attempts[0].budget_tokens == 0


def guard_billing(provider):
    original_generate = provider.generate
    def generate(*args, **kwargs):
        try:
            return original_generate(*args, **kwargs)
        except ProviderError as error:
            if error.code == 429 and any(marker in str(error).lower() for marker in ('monthly spending cap', 'project spend cap')):
                error.retryable = False
            raise
    provider.generate = generate
    return provider


class RejectedProvider:
    def __init__(self, message):
        self.message = message
    def generate(self):
        raise ProviderError(self.message, code=429, retryable=True)


for message, retryable in [('monthly spending cap', False), ('PROJECT SPEND CAP', False), ('rate limit', True)]:
    try:
        guard_billing(RejectedProvider(message)).generate()
    except ProviderError as error:
        assert error.retryable is retryable

print('Dry recovery verified: all 31 validated checkpoints reused; first new request is the scoped repair.', flush=True)
if '--prepare-only' in sys.argv:
    raise SystemExit(0)


def log(message):
    line = f'{datetime.now(UTC).isoformat()} {message}'
    with (output / 'progress.log').open('a') as handle:
        handle.write(line + '\n')
    print(line, flush=True)


def capped_provider(run_type, ledger):
    assert ledger.used == ledger.reserved == 0
    ledger.limit = protocol['additional_generation_limit']
    return guard_billing(_make_provider(settings.for_agent('scout'), ledger))


def guarded_judge(ledger):
    assert ledger.limit == protocol['additional_judge_limit']
    return guard_billing(_make_judge_provider(settings, ledger))


with (output / 'execution.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    request_id = uuid5(NAMESPACE_URL, protocol['sha256']).hex
    run_id = f'{parent.manifest.document_hash[:12]}-{request_id}'
    try:
        result = repo.load_run(run_id)
    except StorageError as error:
        if str(error) != 'Run does not exist.':
            raise
        result = None
    if result and result.manifest.status is RunStatus.RUNNING:
        raise RuntimeError('Interrupted recovery exists; inspect durable records rather than purchase a duplicate.')
    if result is None:
        log(f'Start separate recovery {run_id}')
        result = run_generation(b'', parent.manifest.source_filename, parent.manifest.run_type,
            settings, repository=repo, resume_from=parent.manifest.run_id, request_id=request_id,
            provider_factory=capped_provider, judge_provider_factory=guarded_judge, progress=log)
    row = {'document_id':'D001', 'condition':'multi_efficient_recovery', 'repeat':0,
        'run_id':result.manifest.run_id, 'result':result.model_dump(mode='json')}
    (output / 'result.json').write_text(json.dumps(row, indent=2) + '\n')
    spent = sum(c.budget_tokens for c in result.call_attempts)
    assert result.diagnostics.inherited_budget_tokens == 201599
    assert result.diagnostics.reused_tasks >= 31
    assert not any(c.stage in {'scout','source_audit','curator'} or (c.stage == 'test_writer' and c.task_index < 1000000)
                   for c in result.call_attempts)
    assert spent <= 200000 and 1371440 + spent <= 1600000
    assert sum(c.budget_tokens for c in result.call_attempts if c.phase == 'generation') <= 100000
    assert sum(c.budget_tokens for c in result.call_attempts if c.phase == 'evaluation') <= 100000
    assert spent == result.metrics.charged_tokens
    assert repo.load_run(run_id).model_dump(mode='json') == row['result']
    for old in [*comparison, parent_row]:
        assert repo.load_run(old['run_id']).model_dump(mode='json') == old['result']
    baseline = next(r for r in comparison if r['condition'] == 'staged')
    export_blind([baseline, row], output)
    status = {'status':result.manifest.status.value, 'run_id':run_id,
        'failure':result.manifest.failure_message, 'additional_tokens':spent,
        'inherited_accounted_tokens':201599, 'unobserved_retry_hold_tokens':12497,
        'cumulative_workflow_accounted_tokens':201599 + spent,
        'cumulative_workflow_conservative_tokens':214096 + spent,
        'total_accounted_tokens':1358943 + spent, 'conservative_total_tokens':1371440 + spent,
        'remaining_authorized_tokens':1600000 - 1371440 - spent,
        'additional_estimated_tokens':sum(c.estimated_tokens for c in result.call_attempts),
        'evaluation_error':result.diagnostics.evaluation_error,
        'reused_tasks':result.diagnostics.reused_tasks, 'f1':result.coverage.f1 if result.coverage else None,
        'note':'Separate post-hoc recovery; original matched failure remains unchanged.'}
    (output / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    log(json.dumps(status))
