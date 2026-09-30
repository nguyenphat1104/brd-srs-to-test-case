"""Offline request planning on saved Telescope artifacts; never calls a model."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile

from brd_srs_testgen import coverage, efficient
from brd_srs_testgen.evidence import CuratorChoices, EvidenceRegistry
from brd_srs_testgen.models import ArtifactBundle, CandidateRequirement, CoverageUnitBatch, DocumentChunk, TestCaseBatch
from brd_srs_testgen.pipelines import PipelineContext

root = Path(__file__).resolve().parents[1]
trace_path = Path(sys.argv[1])
data = json.loads(trace_path.read_text())
chunks = [DocumentChunk.model_validate(c) for c in data['chunks']]
outputs = data['outputs']
candidates = [CandidateRequirement.model_validate(c) for row in outputs if row['stage'] == 'scout' for c in row['output']['candidates']]
requirements = next(row['output']['requirements'] for row in outputs if row['stage'] == 'curator')
scenarios = next(row['output']['scenarios'] for row in outputs if row['stage'] == 'scenario_architect')
tests = [c for row in outputs if row['stage'] == 'test_writer' for c in row['output']['test_cases']]
bundle = ArtifactBundle(requirements=requirements, scenarios=scenarios, test_cases=tests)
manifest = json.loads((root / 'docs/benchmark-pilot-2026-09-29/pilot-manifest.json').read_text())


def load_frozen(name, directory, archive):
    source = archive.extractfile(f'src/brd_srs_testgen/{name}.py').read()
    assert hashlib.sha256(source).hexdigest() == manifest['implementation']['source_sha256'][f'{name}.py']
    path = directory / f'{name}.py'
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location(f'brd_srs_testgen.frozen_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def planned(module, registry_type, phase):
    requests = []
    context = PipelineContext(provider=None)
    def capture(stage, index, inputs, schema, produce, validate, **kwargs):
        # Intentionally observe planning only; never invoke model production or fabricate quality scores.
        size = len(json.dumps(inputs['messages'], ensure_ascii=False)) + len(json.dumps(inputs['wire_schema'], ensure_ascii=False))
        requests.append(size)
        if stage == 'curator':
            import re
            text = inputs['messages'][0]['content']
            task = json.loads(re.search(r'<<<BEGIN TASK JSON DATA>>>\n(.*?)\n<<<END TASK JSON DATA>>>', text, re.S)[1])
            return CuratorChoices(choices=[{'candidate_id': c['candidate_id'], 'action': 'retain', 'reason': 'Serialization replay only.'} for c in task['candidates']])
        return TestCaseBatch(test_cases=[])
    context.task = capture
    registry = registry_type(chunks)
    if phase == 'curator':
        module._curate(context, candidates, registry, chunks)
    else:
        module._tests(context, bundle.scenarios, bundle.requirements, registry, chunks)
    return {'initial_tasks': len(requests), 'total_request_characters': sum(requests),
            'largest_request_characters': max(requests),
            'tasks_over_48000_before_runtime_splitting': sum(n > 48000 for n in requests)}


with tempfile.TemporaryDirectory() as temporary, tarfile.open(root / 'docs/benchmark-pilot-2026-09-29/frozen-implementation.tar.gz') as archive:
    directory = Path(temporary)
    old_efficient = load_frozen('efficient', directory, archive)
    old_evidence = load_frozen('evidence', directory, archive)
    old_coverage = load_frozen('coverage', directory, archive)
    results = {phase: {'v6': planned(old_efficient, old_evidence.EvidenceRegistry, phase),
                       'v7': planned(efficient, EvidenceRegistry, phase)} for phase in ('curator', 'test_writer')}
    # Synthetic catalog from development requirements: payload measurement only, not a reference annotation.
    units = CoverageUnitBatch(units=[{'unit_id': f'CU-{i:03d}', 'title': r.title, 'description': r.description,
        'unit_type': 'business_rule' if r.requirement_type.value == 'business' else r.requirement_type.value,
        'source_references': r.source_references} for i, r in enumerate(bundle.requirements, 1)])
    judge = {'v5': 0, 'v6': 0}
    for start in range(0, len(units.units), 50):
        assigned = CoverageUnitBatch(units=units.units[start:start+50])
        for offset in range(0, len(bundle.test_cases), 12):
            cases = ArtifactBundle(requirements=[], scenarios=[], test_cases=bundle.test_cases[offset:offset+12])
            judge['v5'] += len(old_coverage.map_test_cases_prompt(cases, assigned))
            judge['v6'] += len(coverage.map_test_cases_prompt(cases, assigned))
    results['judge_fixed_batches_request_characters'] = judge
results.update(source_run=data['run_id'], trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
    paid_calls=0, counts={'candidates': len(candidates), 'requirements': len(bundle.requirements),
                         'scenarios': len(bundle.scenarios), 'test_cases': len(bundle.test_cases)},
    limitations=['Request characters and initial task counts, not model tokens, latency or end-to-end savings.',
                 'Curator replay retains all candidates identically in both versions; real decisions may differ.',
                 'Runtime splits, corrections and generated outputs are not measured.',
                 'Evaluator comparison adds previously omitted preconditions/test data; semantic inputs differ.',
                 'No quality claim or automatic/human evaluation was produced.'])
target = root / 'docs/token-efficiency-v7-offline-2026-09-30.json'
target.write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
