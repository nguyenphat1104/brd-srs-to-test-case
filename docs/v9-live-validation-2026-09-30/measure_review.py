"""Replay review request planning only. No model, database or quality scoring."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from brd_srs_testgen import efficient
from brd_srs_testgen.documents import parse_pdf
from brd_srs_testgen.evidence import EvidenceRegistry, alias_evidence, json_text
from brd_srs_testgen.models import ArtifactBundle, CriticReport
from brd_srs_testgen.pipelines import PipelineContext
from brd_srs_testgen.runner import implementation_snapshot

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())
assert implementation_snapshot() == manifest['implementation']
result_path = root / 'results/results.json'
result = json.loads(result_path.read_text())[0]['result']
document = manifest['documents'][0]
pdf = Path(document['path']).read_bytes()
assert hashlib.sha256(pdf).hexdigest() == document['sha256']
chunks = parse_pdf(pdf)
bundle = ArtifactBundle.model_validate(result['bundle'])
requests = []


class PlanningContext(PipelineContext):
    def generate(self, *args, **kwargs):
        raise AssertionError('Offline planning must never invoke a provider.')

    def task(self, stage, index, inputs, schema, produce, validate, **kwargs):
        assert stage == 'critic'
        messages, _ = alias_evidence(inputs['messages'])
        content = messages[0]['content']
        payload = json.loads(re.search(
            r'<<<BEGIN TASK JSON DATA>>>\n(.*?)\n<<<END TASK JSON DATA>>>',
            content, re.S)[1])
        evidence = re.search(r'<<<BEGIN EVIDENCE DATA>>>\n(.*?)\n<<<END EVIDENCE DATA>>>', content, re.S)[1]
        requests.append({
            'task_index': index,
            'requirement_ids': [r['requirement_id'] for r in payload['artifacts']['requirements']],
            'scenario_ids': [s['scenario_id'] for s in payload['artifacts']['scenarios']],
            'test_case_ids': [t['test_case_id'] for t in payload['artifacts']['test_cases']],
            'message_characters': len(json_text(messages)),
            'schema_characters': len(json_text(inputs['wire_schema'])),
            'artifact_characters': len(json_text(payload['artifacts'])),
            'global_index_characters': len(json_text(payload['global_index'])),
            'evidence_characters': len(evidence),
        })
        # In-memory placeholder lets planning enumerate all initial groups only.
        # It is never saved or used as a quality judgment about the actual draft.
        return CriticReport(accepted=True, findings=[])


efficient._review(PlanningContext(provider=None), bundle, EvidenceRegistry(chunks), chunks)
observed = [c for c in result['call_attempts'] if c['stage'] == 'critic' and c['status'] == 'completed']
assert [c['task_index'] for c in observed] == [r['task_index'] for r in requests[:len(observed)]]
assert set(r.requirement_id for r in bundle.requirements) == {i for r in requests for i in r['requirement_ids']}
appearances = Counter(i for r in requests for i in r['test_case_ids'])
report = {
    'run_id': result['manifest']['run_id'],
    'results_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
    'paid_calls': 0,
    'source_characters': sum(len(c.text) for c in chunks),
    'planned_initial_review_tasks': len(requests),
    'completed_initial_review_tasks': len(observed),
    'remaining_initial_review_tasks': len(requests) - len(observed),
    'actual_review_input_tokens': sum(c['input_tokens'] for c in observed),
    'actual_review_output_tokens': sum(c['output_tokens'] for c in observed),
    'planned_totals': {key: sum(r[key] for r in requests) for key in requests[0] if key.endswith('_characters')},
    'test_case_appearances': sum(appearances.values()),
    'unique_test_cases': len(appearances),
    'maximum_reviews_per_test_case': max(appearances.values()),
    'requests': requests,
    'limitations': [
        'Request characters are not model tokens or billed savings.',
        'Planning enumerates initial reviews, without generated findings, repairs or evaluation.',
        'The accepted placeholders are control-flow stubs; no semantic approval is inferred.',
        'The final saved draft and frozen implementation are fixed; this is not a new experiment.',
    ],
}
(root / 'review-payload.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'requests'}, indent=2))
