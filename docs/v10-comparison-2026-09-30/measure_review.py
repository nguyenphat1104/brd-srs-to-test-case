"""Compare frozen v9 and v10 review planning on one fixed draft, without models."""
from collections import Counter
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tarfile
import tempfile

from brd_srs_testgen import efficient, coverage
from brd_srs_testgen.documents import parse_pdf
from brd_srs_testgen.evidence import EvidenceRegistry, alias_evidence, compact_index, json_text
from brd_srs_testgen.models import ArtifactBundle, CriticReport, CoverageCatalog, CoverageMappingBatch
from brd_srs_testgen.pipelines import PipelineContext
from brd_srs_testgen.runner import implementation_snapshot

root = Path(__file__).resolve().parent
old = root.parent / 'v9-live-validation-2026-09-30'
manifest = json.loads((old / 'manifest.json').read_text())
result_path = old / 'results/results.json'
result = json.loads(result_path.read_text())[0]['result']
document = manifest['documents'][0]
pdf = Path(document['path']).read_bytes()
assert hashlib.sha256(pdf).hexdigest() == document['sha256']
chunks = parse_pdf(pdf)
bundle = ArtifactBundle.model_validate(result['bundle'])


def records(table):
    return [dict(zip(table['columns'], row)) for row in table['rows']]


def measure(module):
    requests = []
    registry = EvidenceRegistry(chunks)
    def capture(stage, index, inputs, schema, produce, validate, **kwargs):
        assert stage == 'critic'
        content = inputs['messages'][0]['content']
        payload = json.loads(re.search(r'<<<BEGIN TASK JSON DATA>>>\n(.*?)\n<<<END TASK JSON DATA>>>', content, re.S)[1])
        data = payload['artifacts']
        if isinstance(data['requirements'], dict):
            data = {kind: records(table) for kind, table in data.items()}
            scenarios = {s['scenario_id']: s for s in data['scenarios']}
            for case in data['test_cases']:
                for key in case.pop('scenario_fields'):
                    case[key] = scenarios[case['scenario_id']][key]
            assert records(payload['global_index']) == compact_index(bundle.requirements)
        else:
            assert payload['global_index'] == compact_index(bundle.requirements)
        for kind, rows in data.items():
            key = {'requirements':'requirement_id', 'scenarios':'scenario_id', 'test_cases':'test_case_id'}[kind]
            original = {row[key]: row for row in registry.compact(getattr(bundle, kind))}
            assert all(row == original[row[key]] for row in rows), 'Artifact meaning changed in serialization.'
        wire, _ = alias_evidence(inputs['messages'])
        requests.append({'task_index': index, 'requirement_ids': [r['requirement_id'] for r in data['requirements']],
            'request_characters': len(json_text(wire)) + len(json_text(inputs['wire_schema'])),
            'bounded_characters': len(json_text(inputs['messages'])) + len(json_text(inputs['wire_schema'])) + len(json_text(context.prompt_for('critic'))) + 3072,
            'global_index_characters': len(json_text(payload['global_index']))})
        return CriticReport(accepted=True, findings=[])  # Planning stub only, never a quality judgment.
    context = PipelineContext(provider=None)
    context.task = capture
    module._review(context, bundle, registry, chunks)
    assert [key for r in requests for key in r['requirement_ids']] == [r.requirement_id for r in bundle.requirements]
    assert all(r['bounded_characters'] <= efficient.REQUEST_CHAR_LIMIT for r in requests)
    return {'initial_tasks': len(requests), 'request_characters': sum(r['request_characters'] for r in requests),
        'largest_bounded_request': max(r['bounded_characters'] for r in requests),
        'global_index_characters': sum(r['global_index_characters'] for r in requests), 'tasks': requests}


def measure_evaluation(module):
    catalog = CoverageCatalog(catalog_id='planning-only', document_hash=document['sha256'],
        evaluator_version='planning-only', status='machine_frozen', created_at=datetime.now(UTC),
        units=json.loads((root.parent / 'benchmark-pilot-2026-09-29/D001-catalog.draft.json').read_text())['units'])
    requests, pairs = [], []
    def capture(stage, index, inputs, schema, produce, validate, **kwargs):
        text = inputs['messages'][0]['content']
        def block(label):
            return json.loads(re.search(r'<<<BEGIN ' + label + r' DATA>>>\n(.*?)\n<<<END ' + label + r' DATA>>>', text, re.S)[1])
        data, cases = block('COVERAGE UNITS JSON'), block('TEST CASES JSON')
        units = records(data['units']) if isinstance(data['units'], dict) else data['units']
        cases = records(cases) if isinstance(cases, dict) else cases
        for unit in units:
            restored = {**unit, 'source_references': [data['source_evidence'][key] for key in unit['source_references']]}
            assert restored == next(u.model_dump(mode='json') for u in catalog.units if u.unit_id == unit['unit_id'])
        for case in cases:
            original = next(t for t in bundle.test_cases if t.test_case_id == case['test_case_id'])
            expected = original.model_dump(include={'test_case_id','title','requirement_ids','preconditions','test_data'}, mode='json')
            expected['steps'] = [s.model_dump(include={'action','expected_result'}, mode='json') for s in original.steps]
            assert case == expected
        pairs.extend((case['test_case_id'], unit['unit_id']) for case in cases for unit in units)
        requests.append(len(json_text(inputs['messages'])) + len(json_text(inputs['wire_schema'])))
        return CoverageMappingBatch(mappings=[{'test_case_id':case['test_case_id'], 'covered_unit_ids':[]} for case in cases])
    context = PipelineContext(provider=None)
    context.task = capture
    module._bounded_evaluation(context, 'planning-only', bundle, catalog, datetime.now(UTC))
    assert Counter(pairs) == Counter((case.test_case_id, unit.unit_id) for case in bundle.test_cases for unit in catalog.units)
    return {'initial_tasks': len(requests), 'request_characters': sum(requests),
        'all_test_unit_pairs_verified': len(pairs), 'largest_request_characters': max(requests)}


with tarfile.open(old / 'frozen-implementation.tar.gz') as archive, tempfile.TemporaryDirectory() as temporary:
    for name, expected in manifest['implementation']['source_sha256'].items():
        assert hashlib.sha256(archive.extractfile('src/brd_srs_testgen/' + name).read()).hexdigest() == expected
    path = Path(temporary) / 'efficient.py'
    path.write_bytes(archive.extractfile('src/brd_srs_testgen/efficient.py').read())
    spec = importlib.util.spec_from_file_location('brd_srs_testgen.frozen_v9_efficient', path)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    measured = {'v9': measure(frozen), 'v10': measure(efficient)}
    path = Path(temporary) / 'coverage.py'
    path.write_bytes(archive.extractfile('src/brd_srs_testgen/coverage.py').read())
    spec = importlib.util.spec_from_file_location('brd_srs_testgen.frozen_v9_coverage', path)
    old_coverage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_coverage)
    evaluator = {'v6': measure_evaluation(old_coverage), 'v7': measure_evaluation(coverage)}
report = {'source_run': result['manifest']['run_id'], 'results_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
    'implementation': implementation_snapshot(), 'paid_calls': 0, 'measured': measured, 'evaluator': evaluator,
    'request_character_reduction': 1 - measured['v10']['request_characters'] / measured['v9']['request_characters'],
    'checks': ['All scoped artifacts reconstruct exactly, including differences between a test and its scenario.',
        'Every requirement remains assigned and each batch retains the complete requirement index.',
        'All requests fit the unchanged 48,000-character bound. Full relevant source rendering is unchanged.'],
    'limitations': ['Characters and planned initial calls, not measured model-token savings or semantic quality.',
        'This fixed-draft replay excludes generated findings, repairs and independent evaluation.']}
(root / 'offline-measurement.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({version: {k:v for k,v in values.items() if k != 'tasks'} for version,values in measured.items()}, indent=2))
print('Request character reduction:', report['request_character_reduction'])
print('Evaluator:', json.dumps(evaluator))
