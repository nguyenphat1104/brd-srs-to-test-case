"""Read-only database, provenance, cost and quality-result verification."""
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import tarfile

from brd_srs_testgen.benchmark import digest
from brd_srs_testgen.models import RepairVerification
from brd_srs_testgen.runner import implementation_snapshot, inherited_budget_tokens
from brd_srs_testgen.storage import RunRepository
from brd_srs_testgen.validation import validate_bundle

root = Path(__file__).resolve().parent
repo = RunRepository(os.environ['DATABASE_URL'])
protocol = json.loads((root / 'protocol.json').read_text())
assert digest({k: v for k, v in protocol.items() if k != 'sha256'}) == protocol['sha256']
assert hashlib.sha256((root / 'run_evaluation.py').read_bytes()).hexdigest() == protocol['harness_sha256']
assert implementation_snapshot() == protocol['implementation']
with tarfile.open(root / 'frozen-implementation.tar.gz') as archive:
    for name, sha in protocol['implementation']['source_sha256'].items():
        assert hashlib.sha256(archive.extractfile('src/brd_srs_testgen/' + name).read()).hexdigest() == sha
old_rows = []
for directory in ['benchmark-pilot-2026-09-29', 'v7-live-validation-2026-09-30',
                  'v8-live-validation-2026-09-30', 'v9-live-validation-2026-09-30', 'v10-comparison-2026-09-30']:
    old_rows.extend(json.loads((root.parent / directory / 'results/results.json').read_text()))
for directory in ['recovery', 'billing-resumption']:
    old_rows.append(json.loads((root.parent / 'v10-comparison-2026-09-30' / directory / 'result.json').read_text()))
for row in old_rows:
    assert repo.load_run(row['run_id']).model_dump(mode='json') == row['result']
row = json.loads((root / 'result.json').read_text())
result = repo.load_run(row['run_id'])
assert result.model_dump(mode='json') == row['result']
parent = repo.load_run(protocol['parent_run_id'])
assert digest(parent.bundle.model_dump(mode='json')) == protocol['parent_bundle_sha256']
assert result.bundle == parent.bundle, 'This execution verified the existing draft without new edits.'
assert result.manifest.status.value == 'completed' and result.validation.valid
assert validate_bundle(result.bundle, repo.load_chunks(result.manifest.run_id)).valid
assert result.diagnostics.semantic_status == 'accepted' and not result.diagnostics.unresolved_findings
assert result.diagnostics.reused_tasks == 0
assert inherited_budget_tokens(repo, parent) == result.diagnostics.inherited_budget_tokens == 216366
verification, = [RepairVerification.model_validate(s.output) for s in result.stage_outputs if s.stage == 'critic']
assert set(verification.resolved_finding_ids) == set(verification.reasons) == {'FIND-001', 'FIND-002'}
assert not verification.unresolved_finding_ids and all(verification.reasons.values())
phases, stages = Counter(), Counter()
for call in result.call_attempts:
    phases[call.phase] += call.budget_tokens
    stages[call.stage] += call.budget_tokens
    assert call.estimated_tokens == 0 and call.reported_total_tokens == call.budget_tokens
spent = sum(phases.values())
assert spent == result.metrics.charged_tokens
assert phases['generation'] <= 40000 and phases['evaluation'] <= 100000 and phases['catalog'] == 0
assert set(stages) <= {'critic', 'evaluation'}
assert result.coverage_evaluation.status.value == 'completed' and result.coverage is not None
status = json.loads((root / 'status.json').read_text())
assert status['additional_tokens'] == spent
assert status['conservative_total_tokens'] + status['remaining_authorized_tokens'] == 1600000
assert status['conservative_total_tokens'] == 1386207 + spent
report = {'verified_at': datetime.now(UTC).isoformat(), 'run_id': result.manifest.run_id,
    'checks': ['Frozen implementation and harness match their pre-call hashes.',
        'All sixteen earlier run records equal their saved exports and are unchanged.',
        'The retained draft is unchanged and passes deterministic validation.',
        'Both findings have saved explanations and zero unresolved findings.',
        'All ancestor costs are included; reported usage matches call totals and budgets.'],
    'historical_runs_checked': len(old_rows), 'charged_tokens': spent, 'phases': dict(phases),
    'stages': dict(stages), 'calls': len(result.call_attempts),
    'score': result.coverage.model_dump(mode='json'), 'metrics': result.metrics.model_dump(mode='json'),
    'remaining_authorized_tokens': status['remaining_authorized_tokens'],
    'human_validation': 'Pending. No people identified and no ratings supplied.',
    'limitations': protocol['limits']}
(root / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
