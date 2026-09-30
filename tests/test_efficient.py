import json
from dataclasses import replace
import re
from collections import Counter

import pytest

from brd_srs_testgen.documents import chunk_pages, verify_source_reference
from brd_srs_testgen.evidence import (
    EvidenceRegistry, ExtractionBatch, GapAudit, CuratorChoices, SpanScenarios,
    SpanTests, SpanCritique, TestDesignBatch as DesignBatch, alias_evidence, restore_evidence,
)
from brd_srs_testgen.models import CandidateRequirementBatch, CoverageUnitBatch, CoverageMappingBatch
from brd_srs_testgen.pipelines import PipelineContext, PipelineOutputError, run_centralized_multi_agent
from brd_srs_testgen.providers import BudgetLedger, GenerationResult, StructuredOutputError
from brd_srs_testgen.validation import validate_bundle
from tests.factories import bundle, chunk


class EfficientProvider:
    model = 'test-model'
    def __init__(self, *, fail_writer=False, truncate_designs=False):
        self.ledger = BudgetLedger(200000)
        self.calls = []
        self.fail_writer = fail_writer
        self.truncate_designs = truncate_designs

    def generate(self, messages, schema, *, max_output_tokens):
        text = '\n'.join(message['content'] for message in messages)
        self.calls.append((schema, text))
        match = re.search(r'<<<BEGIN TASK JSON DATA>>>\n(.*?)\n<<<END TASK JSON DATA>>>', text, re.S)
        data = json.loads(match[1]) if match else {}
        evidence = re.findall(r'(E\d+):', text)
        source = [{'evidence_id': evidence[0]}] if evidence else []
        base = bundle()
        if schema is ExtractionBatch:
            chunks = data['owned_chunk_ids']
            # Match each owned chunk's first span; contexts cannot become owners.
            candidate_rows = []
            for i,key in enumerate(chunks,1):
                span = re.search(r'\[' + re.escape(key) + r'[^\n]*\]\n(E\d+):', text)[1]
                candidate_rows.append({**base.requirements[0].model_dump(exclude={'requirement_id','dependency_ids'}),
                    'candidate_id': f'CAND-001-{i:03d}', 'source_references':[{'evidence_id':span}]})
            value = {'candidates':candidate_rows, 'coverage':[{'chunk_id':key,'status':'extracted','reason':'Supported requirement.'} for key in chunks]}
        elif schema is GapAudit:
            value = {'missing_candidates':[], 'unresolved_chunk_ids':[]}
        elif schema is CuratorChoices:
            value = {'choices':[{'candidate_id':r['candidate_id'], 'action':'retain','reason':'Distinct supported rule.'} for r in data['candidates']]}
        elif schema is SpanScenarios:
            assigned = data['assigned_requirements']
            value = {'scenarios':[{**base.scenarios[0].model_dump(), 'scenario_id':f'SCN-{i:03d}',
                'requirement_ids':[r['requirement_id']], 'source_references':r['source_references']} for i,r in enumerate(assigned,1)]}
        elif schema is DesignBatch:
            assigned = data['assigned_requirements']
            if self.fail_writer or (self.truncate_designs and len(assigned) > 1):
                raise StructuredOutputError('truncated', incomplete=True)
            value = {'designs': [{**base.test_cases[0].model_dump(exclude={'test_case_id','scenario_id'}),
                'objective': base.scenarios[0].objective, 'scenario_type': base.scenarios[0].scenario_type,
                'requirement_ids': [r['requirement_id']], 'source_references': r['source_references']} for r in assigned]}
        elif schema is SpanTests:
            if self.fail_writer:
                raise StructuredOutputError('truncated', incomplete=True)
            value = {'test_cases':[{**base.test_cases[0].model_dump(), 'test_case_id':f'TC-{i:03d}',
                'scenario_id':r['scenario_id'], 'requirement_ids':r['requirement_ids'], 'source_references':r['source_references']} for i,r in enumerate(data['scenarios'],1)]}
        elif schema is SpanCritique:
            value = {'accepted':True, 'findings':[]}
        else:
            raise AssertionError(f'Unexpected schema {schema}')
        return GenerationResult(value=schema.model_validate(value), input_tokens=1, output_tokens=1, latency_seconds=0.01)


def test_efficient_pipeline_owns_source_once_and_exports_full_citations():
    provider = EfficientProvider()
    saved = []
    context = PipelineContext(provider=provider, efficient=True, stage_recorder=saved.append, recovery_signature='v1')
    result = run_centralized_multi_agent(context,[chunk()])
    assert validate_bundle(result,[chunk()]).valid
    assert Counter(schema for schema,_ in provider.calls)[ExtractionBatch] == 1
    assert len(context.diagnostics.source_dispositions) == 1
    assert context.diagnostics.semantic_status == 'accepted'
    assert context.diagnostics.evidence_spans
    assert saved == context.stage_outputs
    assert all(row.fingerprint for row in saved)
    assert verify_source_reference(result.test_cases[0].source_references[0],[chunk()])
    assert not any('excerpt' in json.loads(re.search(r'<<<BEGIN TASK JSON DATA>>>\n(.*?)\n<<<END TASK JSON DATA>>>',text,re.S)[1]) for _,text in provider.calls)


def test_resume_reuses_validated_tasks_and_does_not_reissue_completed_work():
    first = PipelineContext(provider=EfficientProvider(fail_writer=True), efficient=True, recovery_signature='v1')
    with pytest.raises(PipelineOutputError,match='saved work'):
        run_centralized_multi_agent(first,[chunk()])
    assert first.partial_bundle.requirements and not first.partial_bundle.scenarios
    assert {r.stage for r in first.stage_outputs} == {'scout','source_audit','curator'}
    provider = EfficientProvider()
    resumed = PipelineContext(provider=provider, efficient=True, recovery_signature='v1', checkpoints=first.stage_outputs, recovery_parent='failed-run')
    result = run_centralized_multi_agent(resumed,[chunk()])
    assert validate_bundle(result,[chunk()]).valid
    assert [schema for schema,_ in provider.calls] == [DesignBatch,SpanCritique]
    assert resumed.diagnostics.reused_tasks == 3
    assert sum(row.reused_from == 'failed-run' for row in resumed.stage_outputs) == 3
    changed = PipelineContext(provider=EfficientProvider(),efficient=True,recovery_signature='v2',checkpoints=first.stage_outputs)
    run_centralized_multi_agent(changed,[chunk()])
    assert changed.diagnostics.reused_tasks == 0


def test_truncated_design_splits_once_with_stable_scenario_and_test_links():
    chunks = chunk_pages([(1,'The system shall authenticate registered users.'), (2,'The system shall terminate authenticated sessions.')])
    provider = EfficientProvider(truncate_designs=True)
    context = PipelineContext(provider=provider,efficient=True)
    result = run_centralized_multi_agent(context,chunks)
    assert validate_bundle(result,chunks).valid
    assert [r.scenario_id for r in result.scenarios] == ['SCN-001','SCN-002']
    assert [(t.test_case_id, t.scenario_id) for t in result.test_cases] == [('TC-001','SCN-001'), ('TC-002','SCN-002')]
    assert Counter(schema for schema,_ in provider.calls)[DesignBatch] == 3
    assert {r.task_index for r in context.stage_outputs if r.stage == 'test_writer'} == {0,1,2}


def test_evidence_aliases_preserve_source_text_and_reject_unsupplied_references():
    from brd_srs_testgen.evidence import json_text
    stable = 'E-0123456789abcdef'
    source_text = f'The literal product code {stable} must not change.'
    messages = [{'role': 'user', 'content': json_text({'source_references': [{'evidence_id': stable}], 'text': source_text})
        + '\n' + stable + ': ' + source_text}]
    wire, aliases = alias_evidence(messages)
    assert source_text in wire[0]['content']
    assert wire[0]['content'].endswith('\nE1: ' + source_text)
    assert '"evidence_id":"E1"' in wire[0]['content']
    assert restore_evidence({'evidence_id': 'E1'}, aliases) == {'evidence_id': stable}
    for invalid in ('E2', stable, 'E-ffffffffffffffff'):
        with pytest.raises(ValueError, match='Unknown or out-of-scope'):
            restore_evidence({'evidence_id': invalid}, aliases)


def test_invalid_alias_is_identified_and_corrected_without_repeating_finished_tasks():
    class Typo(EfficientProvider):
        extraction_calls = 0
        def generate(self, messages, schema, **kwargs):
            result = super().generate(messages, schema, **kwargs)
            if schema is ExtractionBatch:
                self.extraction_calls += 1
                if self.extraction_calls == 1:
                    result.value.candidates[0].source_references[0].evidence_id = 'E9999'
                else:
                    assert 'E9999' in messages[-1]['content']
                    assert 'Copy an exact label E1 through E' in messages[-1]['content']
            return result
    provider = Typo()
    context = PipelineContext(provider=provider, efficient=True)
    result = run_centralized_multi_agent(context, [chunk()])
    assert validate_bundle(result, [chunk()]).valid
    assert provider.extraction_calls == 2 and context.semantic_revisions == 1
    assert Counter(schema for schema, _ in provider.calls)[DesignBatch] == 1
    assert not any(schema in (SpanScenarios, SpanTests) for schema, _ in provider.calls)
    assert context.stage_outputs[0].output['candidates'][0]['source_references'][0]['evidence_id'].startswith('E-')


def test_joint_design_preserves_preconditions_boundaries_and_partial_checkpoint_on_failure():
    from brd_srs_testgen.efficient import _design_tests
    from brd_srs_testgen.models import Requirement
    base = bundle().requirements[0].model_dump()
    requirements = [Requirement.model_validate({**base, 'requirement_id': f'REQ-{i:03d}'}) for i in range(1, 10)]
    class FailsLater(EfficientProvider):
        batches = 0
        def generate(self, messages, schema, **kwargs):
            if schema is DesignBatch:
                self.batches += 1
                if self.batches > 1:
                    raise StructuredOutputError('truncated', incomplete=True)
            result = super().generate(messages, schema, **kwargs)
            if schema is DesignBatch:
                for design in result.value.designs:
                    design.preconditions = ['Maintenance mode is disabled.']
                    design.test_data = {'voltage': -5.5, 'unit': 'V'}
                    design.steps[0].expected_result = 'Voltage must remain <= -5.5 V; it must not restart.'
            return result
    context = PipelineContext(provider=FailsLater(), efficient=True)
    with pytest.raises(PipelineOutputError, match='saved work'):
        _design_tests(context, requirements, EvidenceRegistry([chunk()]), [chunk()])
    saved = context.partial_bundle
    assert len(saved.requirements) == 9 and len(saved.scenarios) == len(saved.test_cases) == 8
    assert len(context.stage_outputs) == 1
    for scenario, case in zip(saved.scenarios, saved.test_cases):
        assert case.scenario_id == scenario.scenario_id
        assert scenario.preconditions == case.preconditions == ['Maintenance mode is disabled.']
        assert case.test_data == {'voltage': -5.5, 'unit': 'V'}
        assert '<= -5.5 V' in case.steps[0].expected_result and 'must not restart' in case.steps[0].expected_result


def test_evidence_ids_are_stable_and_reject_unknown_or_out_of_scope_references():
    chunks = chunk_pages([(1,'The voltage shall remain <= -5.5 V under load.'), (2,'The system shall terminate authenticated sessions.')])
    registry = EvidenceRegistry(chunks)
    assert registry.dump() == EvidenceRegistry(chunks).dump()
    key = registry.by_chunk[chunks[0].chunk_id][0]
    ref = registry.hydrate({'evidence_id':key},{chunks[0].chunk_id})
    assert '<=' in ref['excerpt'] and '-5.5' in ref['excerpt']
    assert registry.compact(ref) == {'evidence_id':key}
    for key in ('E-invented',registry.by_chunk[chunks[1].chunk_id][0]):
        with pytest.raises(ValueError,match='out-of-scope'):
            registry.hydrate({'evidence_id':key},{chunks[0].chunk_id})


@pytest.mark.parametrize('artifact_id,role', [('REQ-001','curator'), ('SCN-001','scenario_architect'), ('TC-001','test_writer')])
def test_critic_owner_is_derived_from_target_and_unknown_or_mixed_targets_still_fail(artifact_id, role):
    from brd_srs_testgen.efficient import _hydrate
    from brd_srs_testgen.models import CriticReport
    from brd_srs_testgen.pipelines import _validate_critic_scope
    registry = EvidenceRegistry([chunk()])
    finding = {'finding_id':'FIND-001', 'severity':'high', 'finding_type':'missing_scenario',
        'artifact_ids':[artifact_id], 'required_action':'Add the supported missing scenario.',
        'source_references':[{'evidence_id':registry.by_chunk[chunk().chunk_id][0]}]}
    wire = SpanCritique(accepted=False, findings=[finding])
    assert 'responsible_role' not in type(wire.findings[0]).model_fields
    hydrated = _hydrate(registry, wire, [chunk()], CriticReport)
    assert hydrated.findings[0].responsible_role == role
    assert _validate_critic_scope(hydrated, bundle()) == {artifact_id}
    wire.findings[0].artifact_ids = ['REQ-999']
    with pytest.raises(PipelineOutputError, match='unknown artifact ID'):
        _validate_critic_scope(_hydrate(registry, wire, [chunk()], CriticReport), bundle())
    wire.findings[0].artifact_ids = ['REQ-001','TC-001']
    with pytest.raises(PipelineOutputError, match='one artifact kind'):
        _hydrate(registry, wire, [chunk()], CriticReport)


def expand_review_tables(tables):
    rows = {kind: [dict(zip(table['columns'], row)) for row in table['rows']]
            for kind, table in tables.items()}
    scenarios = {s['scenario_id']: s for s in rows['scenarios']}
    for case in rows['test_cases']:
        for key in case.pop('scenario_fields'):
            case[key] = scenarios[case['scenario_id']][key]
    return rows


def test_review_encoding_roundtrips_equal_and_different_fields_without_mutation():
    from brd_srs_testgen.evidence import review_artifacts, json_text
    artifacts = bundle()
    scenario = artifacts.scenarios[0]
    case = artifacts.test_cases[0]
    for key in ('title', 'preconditions', 'requirement_ids', 'source_references'):
        setattr(case, key, getattr(scenario, key))
    variant = case.model_copy(deep=True, update={'test_case_id': 'TC-002'})
    variant.title = 'A separate boundary case'
    variant.preconditions = ['Voltage <= -5.5 V; do not restart.']
    variant.test_data = {'empty': None, 'scenario_fields': ['literal data'], 'voltage': -5.5}
    artifacts.test_cases.append(variant)
    original = artifacts.model_dump(mode='json')
    tables = review_artifacts(artifacts)
    assert expand_review_tables(tables) == original
    assert artifacts.model_dump(mode='json') == original
    registry = EvidenceRegistry([chunk()])
    wire, aliases = alias_evidence([{'role': 'user', 'content': json_text(registry.compact(tables))}])
    restored = restore_evidence(json.loads(wire[0]['content']), aliases)
    hydrated = registry.hydrate(restored, {chunk().chunk_id})
    assert expand_review_tables(hydrated) == original


def test_compact_review_retains_every_artifact_full_index_and_source_with_bounded_groups():
    from brd_srs_testgen.efficient import _review
    from brd_srs_testgen.models import ArtifactBundle
    artifacts = ArtifactBundle(requirements=[], scenarios=[], test_cases=[])
    for i in range(1, 26):
        base = bundle()
        req, scenario, case = base.requirements[0], base.scenarios[0], base.test_cases[0]
        req.requirement_id = f'REQ-{i:03d}'
        scenario.scenario_id = case.scenario_id = f'SCN-{i:03d}'
        case.test_case_id = f'TC-{i:03d}'
        scenario.requirement_ids = case.requirement_ids = [req.requirement_id]
        artifacts.requirements.append(req)
        artifacts.scenarios.append(scenario)
        artifacts.test_cases.append(case)
    provider = EfficientProvider()
    context = PipelineContext(provider=provider, efficient=True)
    _review(context, artifacts, EvidenceRegistry([chunk()]), [chunk()])
    seen = []
    for schema, text in provider.calls:
        assert schema is SpanCritique
        payload = json.loads(re.search(r'<<<BEGIN TASK JSON DATA>>>\n(.*?)\n<<<END TASK JSON DATA>>>', text, re.S)[1])
        local = expand_review_tables(payload['artifacts'])
        seen.extend(r['requirement_id'] for r in local['requirements'])
        assert len(payload['global_index']['rows']) == 25
        assert 'scenario_fields' in payload['encoding']
        assert chunk().text in text
    assert seen == [r.requirement_id for r in artifacts.requirements]
    assert len(provider.calls) < 3  # Old 12-item bound required at least three calls.


def test_scoped_evidence_keeps_exact_anchor_and_context_without_resending_whole_chunk():
    text = 'The voltage shall remain <= -5.5 V under load. It must not restart during maintenance. '
    text += 'Unrelated background and explanatory material. ' * 50 + 'DISTANT_SENTINEL remains unrelated.'
    chunks = chunk_pages([(1, text)])
    registry = EvidenceRegistry(chunks)
    key = registry.by_chunk[chunks[0].chunk_id][0]
    focused = registry.render(chunks, focus={'requirement': {'source_references': [{'evidence_id': key}]}})
    assert key in focused and '<= -5.5 V' in focused and 'must not restart' in focused
    assert 'DISTANT_SENTINEL' not in focused
    assert len(focused) < len(registry.render(chunks)) / 2
    from brd_srs_testgen.models import SourceReference
    reference = SourceReference.model_validate(registry.hydrate({'evidence_id': key}, {chunks[0].chunk_id}))
    assert verify_source_reference(reference, chunks)


@pytest.mark.parametrize('persistently_invalid', [False, True])
def test_source_audit_corrects_neighbor_only_evidence_once_and_keeps_ownership(persistently_invalid):
    text = 'The system shall authenticate registered users. ' + 'Source background detail. ' * 150
    chunks = chunk_pages([(1, text), (2, text.replace('authenticate', 'authorize'))])
    class NeighborAudit(EfficientProvider):
        audit_attempts = 0
        def generate(self, messages, schema, **kwargs):
            result = super().generate(messages, schema, **kwargs)
            if schema is GapAudit:
                self.audit_attempts += 1
                content = '\n'.join(message['content'] for message in messages)
                data = json.loads(re.search(r'<<<BEGIN TASK JSON DATA>>>\n(.*?)\n<<<END TASK JSON DATA>>>', content, re.S)[1])
                assert data['owned_chunk_ids']
                if self.audit_attempts == 1 or persistently_invalid:
                    candidate = {**bundle().requirements[0].model_dump(exclude={'requirement_id','dependency_ids'}),
                        'candidate_id': 'CAND-999-001', 'source_references': [{'evidence_id': data['boundary_context'][0]['evidence_id']}]}
                    return replace(result, value=GapAudit(missing_candidates=[candidate], unresolved_chunk_ids=[]))
            return result
    provider = NeighborAudit()
    context = PipelineContext(provider=provider, efficient=True)
    if persistently_invalid:
        with pytest.raises(PipelineOutputError, match='no owned evidence'):
            run_centralized_multi_agent(context, chunks)
        assert provider.audit_attempts == 2
        assert [row.stage for row in context.stage_outputs] == ['scout']
    else:
        result = run_centralized_multi_agent(context, chunks)
        assert validate_bundle(result, chunks).valid
        assert provider.audit_attempts == 3  # correction plus the second owned group
        assert len(context.diagnostics.source_dispositions) == len(chunks)
    assert context.semantic_revisions == 1


def test_compact_curator_handles_more_candidates_in_one_bounded_call():
    from brd_srs_testgen.efficient import _curate
    from brd_srs_testgen.models import CandidateRequirement
    base = bundle().requirements[0].model_dump(exclude={'requirement_id','dependency_ids'})
    candidates = [CandidateRequirement(candidate_id=f'CAND-001-{i:03d}', **base) for i in range(1, 38)]
    provider = EfficientProvider()
    result = _curate(PipelineContext(provider=provider), candidates, EvidenceRegistry([chunk()]), [chunk()])
    assert len(result) == 37
    assert [schema for schema, _ in provider.calls] == [CuratorChoices]


def test_batch_sizing_includes_escaped_json_and_trusted_prompt_overhead():
    from brd_srs_testgen.efficient import _groups, bounded_task
    from brd_srs_testgen.models import TestCaseBatch
    seen = []
    class EmptyWriter:
        model = 'test'
        ledger = BudgetLedger(100000)
        def generate(self, messages, schema, **kwargs):
            seen.append(messages)
            return GenerationResult(value=TestCaseBatch(test_cases=[]), input_tokens=1, output_tokens=1, latency_seconds=0)
    context = PipelineContext(provider=EmptyWriter(), agent_prompts={'test_writer': 'Preserve constraints. ' * 500})
    prompt = lambda group: '\n'.join(group)
    groups = _groups(['"\\' * 1800] * 6, prompt, TestCaseBatch, context=context, agent='test_writer')
    assert len(groups) > 1
    for index, group in enumerate(groups):
        bounded_task(context, 'test_writer', index * 3, group, TestCaseBatch, prompt,
            lambda *_: None, agent='test_writer', combine=lambda _: (_ for _ in ()).throw(AssertionError('Unexpected split')))
    assert len(seen) == len(groups)
    for messages in seen:
        payload = json.dumps(messages, ensure_ascii=False, separators=(',', ':'), sort_keys=True)
        schema = json.dumps(TestCaseBatch.model_json_schema(), ensure_ascii=False, separators=(',', ':'), sort_keys=True)
        assert len(payload) + len(schema) < 48000


def test_unresolved_source_audit_stops_without_silently_dropping_evidence():
    class Unresolved(EfficientProvider):
        def generate(self,messages,schema,**kwargs):
            result = super().generate(messages,schema,**kwargs)
            if schema is GapAudit:
                return GenerationResult(value=GapAudit(missing_candidates=[],unresolved_chunk_ids=[chunk().chunk_id]), input_tokens=1,output_tokens=1,latency_seconds=0.01)
            return result
    context = PipelineContext(provider=Unresolved(),efficient=True)
    with pytest.raises(PipelineOutputError,match='Source obligations remain unresolved'):
        run_centralized_multi_agent(context,[chunk()])
    assert [r.stage for r in context.stage_outputs] == ['scout','source_audit']


def test_runner_recovery_persists_checkpoints_and_keeps_original_immutable(repository, monkeypatch):
    from brd_srs_testgen import runner
    from brd_srs_testgen.models import RunStatus, RunType
    from tests.test_runner import settings
    monkeypatch.setattr(runner,'parse_pdf',lambda _: [chunk()])
    config = settings(pipeline_profile='efficient')
    failed = runner.run_generation(b'pdf','sample.pdf',RunType.CENTRALIZED_MULTI_AGENT,config,repository=repository,
        provider_factory=lambda _,ledger: _with_ledger(EfficientProvider(fail_writer=True),ledger))
    stored = repository.load_run(failed.manifest.run_id)
    assert stored == failed
    assert stored.stage_outputs and stored.bundle.requirements
    provider = EfficientProvider()
    resumed = runner.run_generation(b'','sample.pdf',RunType.CENTRALIZED_MULTI_AGENT,config,repository=repository,
        resume_from=failed.manifest.run_id,request_id='b'*32,
        provider_factory=lambda _,ledger: _with_ledger(provider,ledger))
    assert resumed.manifest.status is RunStatus.COMPLETED
    assert resumed.diagnostics.recovery_parent_id == failed.manifest.run_id
    assert resumed.diagnostics.reused_tasks == 3
    assert [schema for schema,_ in provider.calls] == [DesignBatch,SpanCritique]
    assert repository.load_run(failed.manifest.run_id) == failed
    assert repository.load_run(resumed.manifest.run_id) == resumed
    repeated = runner.run_generation(b'','sample.pdf',RunType.CENTRALIZED_MULTI_AGENT,config,repository=repository,
        resume_from=failed.manifest.run_id,request_id='b'*32,
        provider_factory=lambda *_: (_ for _ in ()).throw(AssertionError('Duplicate purchase')))
    assert repeated == resumed
    from brd_srs_testgen.storage import ImmutableRunError
    with repository.run_lease(failed.manifest.run_id):
        with pytest.raises(ImmutableRunError,match='already executing'):
            runner.run_generation(b'','sample.pdf',RunType.CENTRALIZED_MULTI_AGENT,config,repository=repository,
                resume_from=failed.manifest.run_id,request_id='c'*32)


def _with_ledger(provider,ledger):
    provider.ledger = ledger
    return provider


def test_valid_generation_can_be_reused_to_retry_evaluation(repository, monkeypatch):
    from brd_srs_testgen import runner
    from brd_srs_testgen.models import RunStatus, RunType
    from tests.test_runner import settings
    monkeypatch.setattr(runner,'parse_pdf',lambda _: [chunk()])
    config = settings(pipeline_profile='efficient')
    first = runner.run_generation(b'pdf','sample.pdf',RunType.CENTRALIZED_MULTI_AGENT,config,repository=repository,
        provider_factory=lambda _,ledger: _with_ledger(EfficientProvider(),ledger))
    assert first.manifest.status is RunStatus.COMPLETED and first.diagnostics.evaluation_error
    provider = EfficientProvider()
    resumed = runner.run_generation(b'','sample.pdf',RunType.CENTRALIZED_MULTI_AGENT,config,repository=repository,
        resume_from=first.manifest.run_id,
        provider_factory=lambda _,ledger: _with_ledger(provider,ledger))
    assert resumed.bundle == first.bundle
    assert provider.calls == []
    assert resumed.diagnostics.reused_tasks == 1


def test_bounded_evaluator_checks_every_test_against_all_catalog_batches(monkeypatch):
    from datetime import UTC, datetime
    from brd_srs_testgen import efficient
    from brd_srs_testgen.coverage import evaluate_against_catalog
    from brd_srs_testgen.models import CoverageCatalog, CoverageCatalogStatus, CoverageUnit, CoverageMappingBatch
    base = bundle()
    units = [CoverageUnit(unit_id=f'CU-{i:03d}',title=f'Rule {i}',description='The system authenticates registered users.',unit_type='functional',source_references=base.requirements[0].source_references) for i in range(1,102)]
    base.test_cases = [base.test_cases[0].model_copy(update={'test_case_id': f'TC-{i:03d}'}) for i in range(1,50)]
    catalog = CoverageCatalog(catalog_id='catalog',document_hash='a'*64,evaluator_version='test',status=CoverageCatalogStatus.MACHINE_FROZEN,units=units,created_at=datetime.now(UTC))
    calls = []
    class Judge:
        model='test'
        ledger=BudgetLedger(100000)
        def generate(self,messages,schema,**kwargs):
            text=messages[-1]['content']
            assert efficient._request_size(context, 'coverage_analyzer', messages, schema) <= efficient.EVALUATION_REQUEST_CHAR_LIMIT
            payload=json.loads(re.search(r'<<<BEGIN COVERAGE UNITS JSON DATA>>>\n(.*?)\n<<<END COVERAGE UNITS JSON DATA>>>',text,re.S)[1])
            assigned = [dict(zip(payload['units']['columns'], row)) for row in payload['units']['rows']]
            cases_table=json.loads(re.search(r'<<<BEGIN TEST CASES JSON DATA>>>\n(.*?)\n<<<END TEST CASES JSON DATA>>>',text,re.S)[1])
            cases=[dict(zip(cases_table['columns'], row)) for row in cases_table['rows']]
            calls.extend((case['test_case_id'],u['unit_id']) for case in cases for u in assigned)
            return GenerationResult(value=CoverageMappingBatch(mappings=[{'test_case_id':case['test_case_id'],'covered_unit_ids':[u['unit_id'] for u in assigned]} for case in cases]),input_tokens=1,output_tokens=1,latency_seconds=0.01)
    context=PipelineContext(provider=Judge(),efficient=True)
    result=evaluate_against_catalog(context,run_id='run',bundle=base,catalog=catalog,evaluated_at=datetime.now(UTC))
    assert result.score.recall == 1
    assert len(context.call_attempts) >= 4
    assert Counter(calls) == Counter((case.test_case_id, unit.unit_id) for case in base.test_cases for unit in units)
    assert all(c.phase=='evaluation' for c in context.call_attempts)


@pytest.mark.parametrize('kind', ['weak_expected_result','missing_scenario','missing_test_case'])
@pytest.mark.parametrize('resolved', [True,False])
def test_scoped_repairs_and_gap_additions_require_verification(kind,resolved):
    from brd_srs_testgen.models import ArtifactPatch, RepairVerification
    class RepairProvider(EfficientProvider):
        def generate(self,messages,schema,**kwargs):
            if schema is ArtifactPatch:
                self.calls.append((schema,messages[-1]['content']))
                case=bundle().test_cases[0]
                case.steps[0].expected_result='The registered user is authenticated.'
                return GenerationResult(value=ArtifactPatch(test_cases=[case]),input_tokens=1,output_tokens=1,latency_seconds=.01)
            if schema is RepairVerification:
                self.calls.append((schema,messages[-1]['content']))
                return GenerationResult(value=RepairVerification(
                    resolved_finding_ids=['FIND-001'] if resolved else [],
                    unresolved_finding_ids=[] if resolved else ['FIND-001'], reasons={'FIND-001': 'Source-backed correction.' if resolved else 'The defect remains.'}),input_tokens=1,output_tokens=1,latency_seconds=.01)
            result=super().generate(messages,schema,**kwargs)
            if schema is SpanCritique:
                key=re.search(r'(E\d+):',messages[-1]['content'])[1]
                result=replace(result,value=SpanCritique(accepted=False,findings=[{
                    'finding_id':'FIND-001','severity':'high','finding_type':kind,
                    'artifact_ids':['TC-001'],
                    'repair_kind':'semantic','required_action':'Make the supported authentication outcome explicit.',
                    'source_references':[{'evidence_id':key}],
                }]))
            return result
    context=PipelineContext(provider=RepairProvider(),efficient=True)
    if not resolved:
        with pytest.raises(PipelineOutputError,match='unresolved findings'):
            run_centralized_multi_agent(context,[chunk()])
        assert context.diagnostics.semantic_status=='unresolved'
        assert context.partial_bundle.test_cases
        return
    result=run_centralized_multi_agent(context,[chunk()])
    assert validate_bundle(result,[chunk()]).valid
    assert context.diagnostics.semantic_status=='accepted'
    assert context.semantic_revisions==1
    if kind=='weak_expected_result':
        assert len(result.test_cases)==1
        assert result.test_cases[0].steps[0].expected_result=='The registered user is authenticated.'
    else:
        assert [case.test_case_id for case in result.test_cases]==['TC-001','TC-002']
        assert len(result.scenarios)==(2 if kind=='missing_scenario' else 1)


def test_failed_gap_expansion_keeps_the_entire_original_draft(monkeypatch):
    from brd_srs_testgen import efficient
    from brd_srs_testgen.models import ArtifactBundle
    original=bundle()
    context=PipelineContext(provider=EfficientProvider(),efficient=True,partial_bundle=original)
    def interrupted(context,*args):
        context.partial_bundle=ArtifactBundle(requirements=original.requirements,scenarios=[],test_cases=[])
        raise PipelineOutputError('Interrupted gap writer')
    def finding_task(stage,index,inputs,schema,produce,validate,**kwargs):
        from brd_srs_testgen.models import CriticReport
        return CriticReport(accepted=False,findings=[{
            'finding_id':'FIND-001','severity':'high','finding_type':'missing_scenario',
            'artifact_ids':['TC-001'],'responsible_role':'test_writer','required_action':'Add the missing case.',
            'source_references':original.test_cases[0].source_references}])
    monkeypatch.setattr(efficient,'_expand_gap',interrupted)
    monkeypatch.setattr(context,'task',finding_task)
    with pytest.raises(PipelineOutputError,match='Interrupted gap writer'):
        efficient._review(context,original,EvidenceRegistry([chunk()]),[chunk()])
    assert context.partial_bundle==original
