"""Offline regressions for token accounting, bounded repairs and failure isolation."""
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from brd_srs_testgen import runner
from brd_srs_testgen.documents import canonicalize_source_references, chunk_pages, verify_source_reference
from brd_srs_testgen.models import (
    ArtifactPatch, CallAttempt, CriticReport, RepairVerification,
    RequirementBatch, RunDiagnostics, RunResult, RunStatus, RunType, SourceReference,
)
from brd_srs_testgen.pipelines import PipelineContext, PipelineOutputError, _critique_bundle, _merge_patch
from brd_srs_testgen.prompts import repair_prompt
from brd_srs_testgen.providers import BudgetExceeded, BudgetLedger, GeminiProvider, ProviderError, StructuredOutputError
from brd_srs_testgen.storage import ImmutableRunError
from tests.factories import bundle, chunk
from tests.test_pipelines import CritiqueProvider, critic_finding
from tests.test_providers import FakeInteractions, FakeModels
from tests.test_runner import RecordingRepository, ScriptedProvider, NamedProvider, frozen_catalog, settings
from tests.test_storage import manifest


@pytest.mark.parametrize('text,status,error', [
    ('{"requirements": []}', 'completed', None),
    ('{"requirements": [', 'incomplete', StructuredOutputError),
    ('broken JSON', 'completed', StructuredOutputError),
])
def test_usage_survives_failed_response_and_keeps_reasoning_separate(text, status, error):
    usage = SimpleNamespace(total_input_tokens=10, total_output_tokens=5, total_tokens=18,
                            total_thought_tokens=3, total_cached_tokens=4)
    provider = GeminiProvider(SimpleNamespace(models=FakeModels(), interactions=FakeInteractions(text, usage, status)), 'gemini-test', BudgetLedger(10000))
    saved = []
    context = PipelineContext(provider=provider, call_recorder=saved.append)
    if error:
        with pytest.raises(error):
            context.generate([{'role': 'user', 'content': 'extract'}], RequirementBatch, 100, allow_schema_repair=False)
    else:
        context.generate([{'role': 'user', 'content': 'extract'}], RequirementBatch, 100)
    row, = context.call_attempts
    assert saved == [row]
    assert row.input_tokens == 10 and row.output_tokens == 5
    assert row.reported_total_tokens == row.budget_tokens == provider.ledger.used == 18
    assert row.usage['total_thought_tokens'] == 3
    assert row.usage['total_cached_tokens'] == 4
    assert row.estimated_tokens == 0
    assert row.finish_reason == status
    assert row.status == ('failed' if error else 'completed')
    assert len(row.request_hash) == len(row.schema_hash) == 64


def test_unknown_usage_is_null_and_estimate_is_explicit():
    provider = GeminiProvider(SimpleNamespace(models=FakeModels(), interactions=FakeInteractions(usage=SimpleNamespace())), 'gemini-test', BudgetLedger(10000))
    context = PipelineContext(provider=provider)
    with pytest.raises(ProviderError):
        context.generate([{'role': 'user', 'content': 'extract'}], RequirementBatch, 100)
    row, = context.call_attempts
    assert row.input_tokens is row.output_tokens is row.reported_total_tokens is None
    assert row.estimated_tokens == row.budget_tokens == provider.ledger.used == 110


def test_blocked_call_has_no_usage_and_overspend_retains_observed_usage():
    usage = SimpleNamespace(total_input_tokens=10, total_output_tokens=5, total_tokens=150)
    for limit, expected_status, expected_usage in [(5, 'blocked', None), (120, 'failed', 150)]:
        provider = GeminiProvider(SimpleNamespace(models=FakeModels(), interactions=FakeInteractions(usage=usage)), 'gemini-test', BudgetLedger(limit))
        context = PipelineContext(provider=provider)
        with pytest.raises(BudgetExceeded):
            context.generate([{'role': 'user', 'content': 'extract'}], RequirementBatch, 100)
        row, = context.call_attempts
        assert row.status == expected_status
        assert row.reported_total_tokens == expected_usage
        assert row.budget_tokens == (expected_usage or 0)


def test_failed_response_excerpt_is_bounded_and_sanitized():
    provider = GeminiProvider(SimpleNamespace(models=FakeModels(), interactions=FakeInteractions('secret-' * 1000, status='incomplete')), 'gemini-test', BudgetLedger(10000))
    context = PipelineContext(provider=provider, sanitize=lambda value: value.replace('secret', '[redacted]'))
    with pytest.raises(StructuredOutputError):
        context.generate([], RequirementBatch, 100)
    row, = context.call_attempts
    assert 'secret' not in row.response_excerpt
    assert len(row.response_excerpt) == 4000


@pytest.mark.parametrize('source,quote', [
    ('The voltage shall remain < 5 V.', 'The voltage shall remain > 5 V.'),
    ('The voltage shall remain <= 5 V.', 'The voltage shall remain < 5 V.'),
    ('The temperature shall be -5 C.', 'The temperature shall be 5 C.'),
    ('The current shall be 1.5 A.', 'The current shall be 15 A.'),
    ('The system shall not enable cooling.', 'The system shall enable cooling.'),
    ('The limit shall be 5 mV.', 'The limit shall be 5 V.'),
])
def test_citation_cannot_erase_meaningful_symbols_or_negation(source, quote):
    chunks = chunk_pages([(1, source)])
    ref = SourceReference(chunk_id=chunks[0].chunk_id, page_number=1, excerpt=quote)
    assert not verify_source_reference(ref, chunks)
    value = RequirementBatch(requirements=[bundle().requirements[0].model_copy(update={'source_references': [ref]})])
    fixed = canonicalize_source_references(value, chunks)
    assert fixed == value
    assert not verify_source_reference(fixed.requirements[0].source_references[0], chunks)


def test_citation_only_repair_is_accepted_without_rewriting_authored_content():
    artifacts = bundle()
    old = artifacts.requirements[0].model_copy(update={'source_references': [artifacts.requirements[0].source_references[0].model_copy(update={'excerpt': 'The system shall authenticate registered'})]})
    original = artifacts.model_copy(update={'requirements': [old]})
    finding = critic_finding(artifact_ids=['REQ-001'], responsible_role='curator').model_copy(update={'repair_kind': 'citation'})
    context = PipelineContext(provider=CritiqueProvider([
        CriticReport(accepted=False, findings=[finding]),
        ArtifactPatch(requirements=artifacts.requirements),
        RepairVerification(resolved_finding_ids=['FIND-001'], unresolved_finding_ids=[], reasons={'FIND-001': 'Source-backed correction.'}),
    ]))
    assert _critique_bundle(context, original, [chunk()]) == artifacts
    assert context.diagnostics.semantic_status == 'accepted'


@pytest.mark.parametrize('ids', [[], ['FIND-999'], ['FIND-001', 'FIND-001']])
def test_verification_cannot_omit_invent_or_repeat_findings(ids):
    artifacts = bundle()
    changed = artifacts.test_cases[0].model_copy(update={'title': 'Corrected'})
    context = PipelineContext(provider=CritiqueProvider([
        CriticReport(accepted=False, findings=[critic_finding()]),
        ArtifactPatch(test_cases=[changed]),
        RepairVerification(resolved_finding_ids=ids, unresolved_finding_ids=[], reasons={key: 'Explanation.' for key in ids}),
    ]))
    with pytest.raises(PipelineOutputError, match='partition'):
        _critique_bundle(context, artifacts, [chunk()])
    assert context.diagnostics.semantic_status == 'unresolved'


def test_repair_prompt_excludes_unrelated_artifacts_and_evidence():
    artifacts = bundle()
    other = artifacts.test_cases[0].model_copy(update={'test_case_id': 'TC-999', 'title': 'UNRELATED ARTIFACT'})
    artifacts = artifacts.model_copy(update={'test_cases': [*artifacts.test_cases, other]})
    other_chunk = chunk().model_copy(update={'chunk_id': 'unrelated', 'text': 'UNRELATED EVIDENCE'})
    prompt = repair_prompt(artifacts, [critic_finding()], [chunk(), other_chunk])
    assert 'UNRELATED ARTIFACT' not in prompt and 'UNRELATED EVIDENCE' not in prompt
    assert 'TC-001' in prompt and 'REQ-001' in prompt and 'SCN-001' in prompt
    assert 'ArtifactPatch' in prompt


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'extra'])
def test_patch_has_exact_requested_identity_set(kind):
    case = bundle().test_cases[0]
    cases = [] if kind == 'missing' else [case, case] if kind == 'duplicate' else [case, case.model_copy(update={'test_case_id': 'TC-999'})]
    with pytest.raises(PipelineOutputError, match='exactly once'):
        _merge_patch(bundle(), ArtifactPatch(test_cases=cases), {'TC-001'})


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('failure', ['budget', 'schema'])
def test_judge_failures_preserve_generation_with_or_without_catalog(monkeypatch, cached, failure):
    repository = RecordingRepository()
    if cached:
        repository.save_coverage_catalog(frozen_catalog(runner.hashlib.sha256(b'pdf').hexdigest()))
    monkeypatch.setattr(runner, 'parse_pdf', lambda _: [chunk()])
    monkeypatch.setitem(runner.PIPELINES, RunType.SINGLE_PROMPT, lambda *_: bundle())
    class FailingJudge(NamedProvider):
        def generate(self, *args, **kwargs):
            if failure == 'budget':
                raise BudgetExceeded('judge exhausted')
            raise StructuredOutputError('truncated', input_tokens=10, output_tokens=5, incomplete=True)
    result = runner.run_generation(b'pdf', 'sample.pdf', RunType.SINGLE_PROMPT, settings(), repository=repository,
        provider_factory=lambda _, ledger: ScriptedProvider(ledger, []),
        judge_provider_factory=lambda ledger: FailingJudge(ledger, runner.JUDGE_MODEL))
    assert result.manifest.status is RunStatus.COMPLETED
    assert result.bundle == bundle() and result.validation.valid
    assert result.coverage is None and result.diagnostics.evaluation_error
    assert result.call_attempts[0].phase == ('evaluation' if cached else 'catalog')
    assert bool(result.coverage_evaluation) == cached
    assert repository.call_attempts[0][1] == result.call_attempts[0]


def test_truncated_repair_preserves_draft_and_unresolved_findings(monkeypatch):
    repository = RecordingRepository()
    monkeypatch.setattr(runner, 'parse_pdf', lambda _: [chunk()])
    monkeypatch.setitem(runner.PIPELINES, RunType.CENTRALIZED_MULTI_AGENT, lambda c, chunks: _critique_bundle(c, bundle(), chunks))
    class TruncatedRepair(ScriptedProvider):
        def generate(self, messages, schema, **kwargs):
            if schema is ArtifactPatch:
                raise StructuredOutputError('truncated', input_tokens=10, output_tokens=5, incomplete=True)
            return super().generate(messages, schema, **kwargs)
    result = runner.run_generation(b'pdf', 'sample.pdf', RunType.CENTRALIZED_MULTI_AGENT, settings(), repository=repository,
        provider_factory=lambda _, ledger: TruncatedRepair(ledger, [CriticReport(accepted=False, findings=[critic_finding()])]))
    assert result.manifest.status is RunStatus.FAILED
    assert result.bundle == bundle() and result.validation.valid and result.rtm
    assert result.metrics.requirement_count == result.metrics.test_case_count == 1
    assert not result.metrics.completion
    assert result.diagnostics.semantic_status == 'unresolved'
    assert len(result.diagnostics.unresolved_findings) == 1
    assert len(result.call_attempts) == 2
    assert result.download_bundle()['test_cases']


def test_call_attempts_persist_before_finalize_and_are_immutable(repository):
    running = manifest()
    repository.create_run(running)
    attempt = CallAttempt(call_id='call-1', phase='generation', stage='repair', task_index=0, attempt=1,
        model='test', provider='test', schema_name='ArtifactPatch', request_hash='a'*64, schema_hash='b'*64,
        max_output_tokens=100, status='failed', budget_tokens=110, estimated_tokens=110,
        started_at=datetime.now(UTC), latency_seconds=0.1)
    repository.append_call_attempt(running.run_id, attempt)
    assert repository.load_run(running.run_id).call_attempts == [attempt]
    repository.append_call_attempt(running.run_id, attempt)
    with pytest.raises(ImmutableRunError):
        repository.append_call_attempt(running.run_id, attempt.model_copy(update={'budget_tokens': 99}))
    terminal = running.model_copy(update={'status': RunStatus.FAILED, 'completed_at': datetime.now(UTC), 'failure_category': runner.FailureCategory.SCHEMA_FAILURE, 'failure_message': 'truncated'})
    result = RunResult(manifest=terminal, call_attempts=[attempt], diagnostics=RunDiagnostics(evaluation_error='catalog failed'))
    repository.finalize(result)
    assert repository.load_run(running.run_id) == result
    with pytest.raises(ImmutableRunError):
        repository.append_call_attempt(running.run_id, attempt.model_copy(update={'call_id': 'new'}))


def test_absent_provider_total_is_not_fabricated_from_components():
    provider = GeminiProvider(SimpleNamespace(models=FakeModels(), interactions=FakeInteractions(usage=SimpleNamespace(total_input_tokens=10, total_output_tokens=5))), 'gemini-test', BudgetLedger(10000))
    context = PipelineContext(provider=provider)
    context.generate([], RequirementBatch, 100)
    row, = context.call_attempts
    assert row.reported_total_tokens is None
    assert row.input_tokens == 10 and row.output_tokens == 5
    assert row.budget_tokens == 15 and row.estimated_tokens == 0


def test_ambiguous_quote_does_not_silently_select_a_chunk():
    chunks = chunk_pages([(1, 'The system shall authenticate registered users.'), (2, 'The system shall authenticate registered users.')])
    reference = SourceReference(chunk_id='unknown', page_number=99, excerpt=chunks[0].text)
    original = RequirementBatch(requirements=[bundle().requirements[0].model_copy(update={'source_references': [reference]})])
    assert canonicalize_source_references(original, chunks) == original


def test_concurrent_calls_keep_usage_and_task_records_separate():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from brd_srs_testgen.providers import CALL_USAGE, GenerationResult
    barrier = Barrier(2)
    class ConcurrentProvider:
        model = 'test'
        ledger = BudgetLedger(1000)
        def generate(self, messages, schema, **kwargs):
            amount = int(messages[0]['content'])
            reservation = self.ledger.reserve(amount)
            CALL_USAGE.get().update(input_tokens=amount, output_tokens=0, reported_total_tokens=amount)
            barrier.wait(timeout=5)
            self.ledger.settle(reservation, amount)
            return GenerationResult(value=RequirementBatch(requirements=[]), input_tokens=amount, output_tokens=0, latency_seconds=0.01, billed_tokens=amount)
    context = PipelineContext(provider=ConcurrentProvider())
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda n: context.generate([{'role': 'user', 'content': str(n)}], RequirementBatch, 100), [11, 17]))
    assert {r.task_index for r in context.call_attempts} == {0, 1}
    assert sorted((r.input_tokens, r.budget_tokens) for r in context.call_attempts) == [(11, 11), (17, 17)]
    assert sum(r.budget_tokens for r in context.call_attempts) == context.charged_tokens == 28


def test_inherited_cost_traverses_interrupted_parent_without_diagnostics():
    def ancestor(key, previous, cost, complete):
        return SimpleNamespace(manifest=SimpleNamespace(run_id=key, document_hash='same',
            configuration={'recovery_parent_id': previous} if previous else {}),
            metrics=SimpleNamespace(charged_tokens=cost) if complete else None,
            call_attempts=[SimpleNamespace(budget_tokens=cost)], diagnostics=None)
    original = ancestor('original', None, 189102, True)
    blocked = ancestor('blocked', 'original', 12497, False)
    retry = ancestor('retry', 'blocked', 14767, True)
    rows = {r.manifest.run_id: r for r in [original, blocked, retry]}
    repo = SimpleNamespace(load_run=rows.__getitem__)
    assert runner.inherited_budget_tokens(repo, retry) == 216366
    original.manifest.configuration['recovery_parent_id'] = 'retry'
    with pytest.raises(runner.ConfigurationError, match='cyclic'):
        runner.inherited_budget_tokens(repo, retry)
    original.manifest.configuration.clear()
    original.manifest.document_hash = 'different'
    with pytest.raises(runner.ConfigurationError, match='another document'):
        runner.inherited_budget_tokens(repo, retry)


def test_requirement_verification_sees_implementing_scenarios_and_tests():
    from brd_srs_testgen.prompts import repair_verification_prompt
    artifacts = bundle()
    finding = critic_finding(artifact_ids=['REQ-001'], responsible_role='curator')
    unrelated = artifacts.test_cases[0].model_copy(update={'test_case_id': 'TC-999',
        'scenario_id': 'SCN-999', 'requirement_ids': ['REQ-999'], 'title': 'UNRELATED'})
    artifacts.test_cases.append(unrelated)
    prompt = repair_verification_prompt(artifacts, [finding], iter([chunk()]))
    assert 'SCN-001' in prompt and 'TC-001' in prompt and 'UNRELATED' not in prompt
    assert chunk().text in prompt and 'reasons' in prompt
    assert 'label alone neither proves nor disproves' in prompt


@pytest.mark.parametrize('reasons', [{}, {'FIND-001': ' '}, {'FIND-999': 'Wrong finding.'}])
def test_repair_verification_rejects_missing_blank_or_unrelated_explanation(reasons):
    from brd_srs_testgen.pipelines import _validate_repair_verification
    verification = RepairVerification(resolved_finding_ids=[], unresolved_finding_ids=['FIND-001'], reasons=reasons)
    with pytest.raises(PipelineOutputError, match='explain every finding'):
        _validate_repair_verification(verification, [critic_finding()])
