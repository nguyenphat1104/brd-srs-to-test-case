"""Frozen post-hoc reassessment of a saved draft; never a fresh comparison."""
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
from uuid import NAMESPACE_URL, uuid5

from brd_srs_testgen import runner
from brd_srs_testgen.benchmark import digest, export_blind
from brd_srs_testgen.efficient import verify_repairs, bounded_task
from brd_srs_testgen.models import ArtifactPatch, CriticFinding, RunStatus
from brd_srs_testgen.pipelines import (PipelineOutputError, _canonicalize_grounded,
    _merge_patch, _validate_repair_scope, _validate_repair_citations, default_agent_setups)
from brd_srs_testgen.prompts import repair_prompt
from brd_srs_testgen.storage import RunRepository, StorageError
from brd_srs_testgen.validation import validate_bundle

root = Path(__file__).resolve().parent
previous = root.parent / 'v10-comparison-2026-09-30'
repo = RunRepository(os.environ['DATABASE_URL'])
parent_row = json.loads((previous / 'billing-resumption/result.json').read_text())
parent = repo.load_run(parent_row['run_id'])
assert parent.model_dump(mode='json') == parent_row['result']
prior_status = json.loads((previous / 'billing-resumption/status.json').read_text())
assert prior_status['conservative_total_tokens'] == 1386207
assert prior_status['remaining_authorized_tokens'] == 213793
comparison = json.loads((previous / 'results/results.json').read_text())
original = next(r for r in comparison if r['condition'] == 'multi_efficient')['result']
findings = []
for stage in original['stage_outputs']:
    if stage['stage'] == 'critic':
        for value in stage['output'].get('findings', []):
            findings.append(CriticFinding.model_validate({**value, 'finding_id': f'FIND-{len(findings) + 1:03d}'}))
assert [f.artifact_ids for f in findings] == [['TC-074'], ['REQ-092']]
chunks = repo.load_chunks(parent.manifest.run_id)
assert validate_bundle(parent.bundle, chunks).valid
assert runner.inherited_budget_tokens(repo, parent) == 216366
catalog = repo.load_coverage_catalog(parent.manifest.document_hash, runner.EVALUATOR_VERSION)
assert catalog.status.value == 'approved' and len(catalog.units) == 91
study = json.loads((previous / 'manifest.json').read_text())
doc = study['documents'][0]
assert hashlib.sha256(Path(doc['path']).read_bytes()).hexdigest() == doc['sha256']
catalog_source = previous.parent / 'benchmark-pilot-2026-09-29' / study['source_catalog_drafts']['D001']['file']
assert [u.model_dump(mode='json') for u in catalog.units] == json.loads(catalog_source.read_text())['units']
settings = runner.ProviderSettings(provider='gemini', model=study['model'],
    thinking_level='minimal', token_ceiling=40000, pipeline_profile='efficient',
    api_key=os.environ.get('GEMINI_API_KEY', ''), agent_setups=default_agent_setups())
protocol = {
    'type': 'saved-draft-reassessment-v12', 'authorization': 'User explicitly requested items 1–4, including fixing verification/accounting and running evaluation.',
    'parent_run_id': parent.manifest.run_id, 'parent_bundle_sha256': digest(parent.bundle.model_dump(mode='json')),
    'implementation': runner.implementation_snapshot(),
    'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'configuration': settings.snapshot(parent.manifest.run_type),
    'previous_accounted_tokens': 1373710, 'uncertain_retry_hold_tokens': 12497,
    'previous_conservative_total': 1386207, 'total_authorized_tokens': 1600000,
    'generation_cap': 40000, 'judge_cap': 100000, 'maximum_additional_tokens': 140000,
    'scope': 'Import the exact saved 96-test draft as experimental input. This is explicit artifact reuse across versions, not fingerprint-matched task reuse or a fresh generation trial. No old checkpoint is accepted as a current-version validation.',
    'procedure': 'Reassess both original findings with source-backed explanations and downstream artifacts. If unresolved, permit one semantic patch of TC-074 for FIND-001 and TC-096 for FIND-002; preserve all other artifacts and IDs. Verify the original findings again. Judge runs only if both pass and deterministic validation passes. No further recovery in this protocol.',
    'limits': 'Same approved 91-unit catalog and coverage-v7 Judge as the staged baseline. Supplementary and adaptive on a reused document, with unequal cumulative budgets. Historical failures remain. Human validation remains pending; model scores do not replace it.',
}
protocol['sha256'] = digest(protocol)
protocol_file = root / 'protocol.json'
if protocol_file.exists():
    assert json.loads(protocol_file.read_text()) == protocol
else:
    protocol_file.write_text(json.dumps(protocol, indent=2) + '\n')
    with tarfile.open(root / 'frozen-implementation.tar.gz', 'w:gz') as archive:
        for name in protocol['implementation']['source_sha256']:
            source = Path(runner.__file__).parent / name
            archive.add(source, arcname='src/brd_srs_testgen/' + name)
assert protocol['previous_conservative_total'] + protocol['maximum_additional_tokens'] <= 1600000


def log(message):
    line = f'{datetime.now(UTC).isoformat()} {message}'
    with (root / 'progress.log').open('a') as handle:
        handle.write(line + '\n')
    print(line, flush=True)


def reassess(context, source_chunks):
    bundle = parent.bundle.model_copy(deep=True)
    context.partial_bundle = bundle
    context.diagnostics.unresolved_findings = findings
    context.diagnostics.semantic_status = 'unresolved'
    log('Imported saved draft; no extraction, curation, writing or previous Critic calls purchased.')
    verification = verify_repairs(context, bundle, findings, source_chunks, 0)
    log('Initial verification: ' + verification.model_dump_json())
    for index, finding in enumerate(findings):
        if finding.finding_id not in verification.unresolved_finding_ids:
            continue
        target = 'TC-074' if finding.finding_id == 'FIND-001' else 'TC-096'
        scoped = finding.model_copy(update={'artifact_ids': [target], 'responsible_role': 'test_writer',
            'repair_kind': 'semantic', 'required_action': 'Correct only this existing test using the source and linked scenario. Do not invent a UI message, permission rule or default. Address this verifier explanation: ' + verification.reasons[finding.finding_id]})
        def validate(patch, _):
            _canonicalize_grounded(patch, source_chunks)
            merged = _merge_patch(bundle, patch, {target})
            _validate_repair_scope(bundle, merged, {target}, [scoped])
            _validate_repair_citations(bundle, merged, [scoped])
            if not validate_bundle(merged, source_chunks).valid:
                raise PipelineOutputError('Scoped patch failed deterministic validation.')
        patch = bounded_task(context, 'repair', index * 3, [[scoped]], ArtifactPatch,
            lambda rows: repair_prompt(bundle, rows[0], source_chunks), validate,
            agent='test_writer', combine=lambda _: None)
        bundle = _merge_patch(bundle, patch, {target})
        context.partial_bundle = bundle
        context.semantic_revisions += 1
    if verification.unresolved_finding_ids:
        verification = verify_repairs(context, bundle, findings, source_chunks, 1)
        log('Final verification: ' + verification.model_dump_json())
    context.diagnostics.unresolved_findings = [f for f in findings if f.finding_id in verification.unresolved_finding_ids]
    if context.diagnostics.unresolved_findings:
        raise PipelineOutputError('Detailed verification left unresolved findings: ' + json.dumps(verification.reasons))
    context.diagnostics.semantic_status = 'accepted'
    return bundle


if '--prepare-only' in sys.argv:
    print('Verified saved draft, all ancestor costs, source/catalog hashes, new source snapshot and 140000-token limit. No model calls.')
    raise SystemExit(0)

with (root / 'execution.lock').open('a') as lock:
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
        raise RuntimeError('Existing in-progress attempt; inspect it, never duplicate.')
    if result is None:
        # Explicit protocol-level intervention, recorded by this harness hash above.
        runner.PIPELINES[parent.manifest.run_type] = reassess
        result = runner.run_generation(b'', parent.manifest.source_filename, parent.manifest.run_type,
            settings, repository=repo, resume_from=parent.manifest.run_id, request_id=request_id, progress=log)
    row = {'document_id': 'D001', 'condition': 'multi_efficient_draft_reassessment', 'repeat': 0,
        'run_id': result.manifest.run_id, 'result': result.model_dump(mode='json')}
    (root / 'result.json').write_text(json.dumps(row, indent=2) + '\n')
    assert repo.load_run(run_id).model_dump(mode='json') == row['result']
    assert repo.load_run(parent.manifest.run_id).model_dump(mode='json') == parent_row['result']
    assert result.diagnostics.inherited_budget_tokens == 216366
    assert result.diagnostics.reused_tasks == 0
    spent = sum(c.budget_tokens for c in result.call_attempts)
    assert spent == result.metrics.charged_tokens <= 140000
    phases = {p: sum(c.budget_tokens for c in result.call_attempts if c.phase == p) for p in ['generation', 'evaluation', 'catalog']}
    assert phases['generation'] <= 40000 and phases['evaluation'] <= 100000 and phases['catalog'] == 0
    assert not any(c.stage in ['scout', 'source_audit', 'curator', 'test_writer'] for c in result.call_attempts)
    status = {'run_id': run_id, 'status': result.manifest.status.value, 'failure': result.manifest.failure_message,
        'additional_tokens': spent, 'phases': phases, 'estimated_tokens': sum(c.estimated_tokens for c in result.call_attempts),
        'inherited_accounted_tokens': 216366, 'workflow_accounted_tokens': 216366 + spent,
        'total_accounted_tokens': 1373710 + spent, 'conservative_total_tokens': 1386207 + spent,
        'remaining_authorized_tokens': 213793 - spent, 'f1': result.coverage.f1 if result.coverage else None,
        'semantic_status': result.diagnostics.semantic_status, 'evaluation_error': result.diagnostics.evaluation_error,
        'human_reviews': 'Pending; no reviewers identified or ratings supplied.',
        'interpretation': protocol['limits']}
    (root / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    export_blind([next(r for r in comparison if r['condition'] == 'staged'), row], root)
    log(json.dumps(status))
