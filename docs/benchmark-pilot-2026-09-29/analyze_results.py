"""Reconcile frozen trial records and write a descriptive pilot report offline."""
from collections import Counter, defaultdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'pilot-manifest.json').read_text())
rows = json.loads((root / 'results/results.json').read_text())
summary = json.loads((root / 'results/summary.json').read_text())
expected = {(d['document_id'], c, 0) for d in manifest['documents'] for c in manifest['conditions']}
assert len(rows) == 8 and {(r['document_id'], r['condition'], r['repeat']) for r in rows} == expected
assert len({row['run_id'] for row in rows}) == 8
calls = [call for row in rows for call in row['result']['call_attempts']]
comparison = sum(call['budget_tokens'] for call in calls)
preparation = sum(json.loads(line)['budget_tokens'] for line in (root / 'D001-calls.jsonl').read_text().splitlines())
assert comparison == summary['budget_accounted_tokens']
assert preparation == manifest['catalog_preparation_accounted_tokens'] == 23613
assert comparison <= manifest['max_total_budget_tokens']
assert comparison + preparation <= 1600000
stages = defaultdict(lambda: defaultdict(int))
table = []
for row in rows:
    result = row['result']
    assert result['manifest']['status'] in ('completed', 'failed')
    assert result['manifest']['configuration']['implementation'] == manifest['implementation']
    phase = Counter()
    for call in result['call_attempts']:
        phase[call['phase']] += call['budget_tokens']
        stage = stages[f"{row['condition']}/{call['stage']}"]
        stage['paid_calls'] += call['budget_tokens'] > 0
        stage['budget_tokens'] += call['budget_tokens']
        stage['input_tokens'] += call['input_tokens'] or 0
        stage['visible_output_tokens'] += call['output_tokens'] or 0
        stage['thinking_tokens'] += call['usage'].get('total_thought_tokens', 0)
        stage['estimated_tokens'] += call['estimated_tokens']
    coverage = result.get('coverage')
    table.append({'document_id': row['document_id'], 'condition': row['condition'],
        'run_id': row['run_id'], 'status': result['manifest']['status'],
        'generation_tokens': phase['generation'], 'evaluation_tokens': phase['evaluation'],
        'catalog_tokens': phase['catalog'], 'total_tokens': sum(phase.values()),
        'test_cases': len(result['bundle']['test_cases']) if result.get('bundle') else 0,
        'saved_bundle': result.get('bundle') is not None,
        'f1': coverage['f1'] if coverage else None,
        'precision': coverage['precision'] if coverage else None,
        'recall': coverage['recall'] if coverage else None,
        'failure_category': result['manifest'].get('failure_category'),
        'failure_message': result['manifest'].get('failure_message'),
        'evaluation_error': (result.get('diagnostics') or {}).get('evaluation_error'),
    })
for document in manifest['documents']:
    assert hashlib.sha256(Path(document['path']).read_bytes()).hexdigest() == document['sha256']
for expected_catalog in manifest['source_catalog_drafts'].values():
    assert hashlib.sha256((root / expected_catalog['file']).read_bytes()).hexdigest() == expected_catalog['sha256']
completed = sum(row['status'] == 'completed' for row in table)
scored = sum(row['f1'] is not None for row in table)
analysis = {'generated_at': datetime.now(UTC).isoformat(), 'manifest_sha256': manifest['manifest_sha256'],
    'recorded_trials': len(rows), 'completed_trials': completed, 'automatically_scored_trials': scored,
    'catalog_preparation_tokens': preparation, 'comparison_tokens': comparison,
    'total_accounted_tokens': comparison + preparation, 'remaining_authorized_tokens': 1600000 - comparison - preparation,
    'paid_calls': sum(c['budget_tokens'] > 0 for c in calls),
    'blocked_calls': sum(c['status'] == 'blocked' for c in calls),
    'incomplete_calls': sum(c['finish_reason'] == 'incomplete' for c in calls),
    'estimated_tokens': sum(c['estimated_tokens'] for c in calls),
    'input_tokens': sum(c['input_tokens'] or 0 for c in calls),
    'visible_output_tokens': sum(c['output_tokens'] or 0 for c in calls),
    'thinking_tokens': sum(c['usage'].get('total_thought_tokens', 0) for c in calls),
    'trials': table, 'stages': dict(stages), 'human_ratings_collected': 0,
    'quality_preservation': 'Not established.'}
(root / 'results/analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')

report = f'''# Token-efficiency feasibility pilot — 30 September 2026

The eight planned trials are recorded: **{completed} completed generation runs**, **{scored} available automated coverage scores**. Quality preservation is **not established**. Two independent human ratings remain pending.

## Cost and outcomes

Total accounted Gemini tokens: **{comparison + preparation:,} / 1,600,000**. This includes **{preparation:,}** catalog-preparation tokens and **{comparison:,}** comparison tokens, including unsuccessful calls. Unspent authorization: **{1600000 - comparison - preparation:,}** tokens. The budget is a ceiling, not a target to exhaust.

| Document | Condition | Generation result | Generation tokens | Evaluation tokens | Saved test cases | Available F1 |
|---|---|---|---:|---:|---:|---:|
'''
for row in sorted(table, key=lambda r: (r['document_id'], r['condition'])):
    score = f"{row['f1']:.3f}" if row['f1'] is not None else 'Unavailable'
    report += f"| {row['document_id']} | {row['condition']} | {row['status']} | {row['generation_tokens']:,} | {row['evaluation_tokens']:,} | {row['test_cases']} | {score} |\n"
report += f'''
D001 = Model Manager (91 approved coverage units); D002 = DigitalHome 1.3 (84 units). All four conditions reuse the same approved catalog for each document. The catalogs are withheld from generation. Missing evaluations are **not F1 = 0**. Counts of saved tests do not certify correctness. Per-condition totals and exploratory paired statistics are in `results/summary.json`; per-stage costs and trial-level precision/recall are in `results/analysis.json`.

Comparison accounting: paid calls = {analysis['paid_calls']}; locally blocked calls = {analysis['blocked_calls']}; incomplete responses = {analysis['incomplete_calls']}. Reported components: {analysis['input_tokens']:,} input, {analysis['visible_output_tokens']:,} visible output and {analysis['thinking_tokens']:,} thinking tokens. Estimated charges: {analysis['estimated_tokens']:,}. These are application/provider token-accounting records, not a currency invoice. Cumulative run tokens are not the size of one context window.

## Recorded failures

'''
for row in table:
    if row['failure_category']:
        report += f"- **{row['document_id']} / {row['condition']} — {row['failure_category']}:** {row['failure_message']}\n"
    if row['evaluation_error']:
        report += f"- **{row['document_id']} / {row['condition']} — evaluation unavailable:** {row['evaluation_error']}\n"
report += '''
The efficient DigitalHome run used 28,730 tokens for extraction, 23,430 for source audits and 33,857 for curation before the next reservation was blocked. It retained 62 requirements but no scenarios or tests. This is an incomplete pipeline, not a low-cost success. The efficient Model Manager run stopped when a gap candidate lacked evidence owned by its assigned extraction task; successful structured responses did not guarantee semantic validity.

On Model Manager, the staged condition used 37,860 generation tokens and 79,698 evaluator tokens. Evaluator spending therefore exceeded generation spending. Evaluator reasoning and retries are part of the measured cost, not free infrastructure. The earlier catalog attempt also exhausted an 8,000-token output allowance while reasoning consumed most of that allowance. Google's [thinking documentation](https://ai.google.dev/gemini-api/docs/thinking#token-limits-and-max_output_tokens) explains that the response cap includes thinking tokens and can truncate an answer without reducing reasoning effort.

## Interpretation and next engineering experiment

These are observations from a feasibility pilot. They do not validate the label “efficient,” prove that every simpler architecture is better, or establish savings against the earlier 973,822-token Telescope incident: the documents and outcomes differ. A method that fails early cannot win on token count alone. Report completion rate and cost per usable, human-reviewed suite alongside coverage.

The next version should simplify the mandatory pipeline before adding more agents:

1. **Reserve enough budget to finish.** Estimate extraction, writing and verification costs before generation. Allocate stage allowances and report whether the document can fit. A global ceiling currently prevents continued spending but can stop after expensive upstream work with no tests.
2. **Reduce repeated payloads and mandatory passes.** Pass cited spans with necessary surrounding context; retain full documents locally. Investigate a staged generator with targeted audits for ambiguous, uncovered or conflicting obligations. Compare it with the current mandatory per-group audit/curation path. Do not assume that smaller batches mean fewer total tokens.
3. **Repair the failing unit.** For evidence-ownership violations, use one bounded local correction referencing the actual validation error and allowed evidence IDs. Keep the semantic check; do not relax it merely to make runs pass. Preserve valid checkpoints and make incomplete output visible.
4. **Tune the evaluator separately.** Test lower thinking effort and adequate response headroom on development data; compare its mappings with human judgments before freezing a new evaluator. Reduce repeated catalog serialization. Any candidate-retrieval shortcut must measure missed matches; skipping difficult coverage pairs would inflate apparent efficiency.
5. **Run a new versioned study.** Keep these eight outcomes unchanged. Use development documents for fixes, then freeze code/settings and use new held-out documents, repeated runs and two actual human raters. These two documents have now been inspected and must not be described as untouched held-out data for later tuning.

These are proposed engineering changes, not changes applied during this pilot. Quality and cost effects need measurement.

## Research basis

- Zhang et al., [*Cut the Crap / AgentPrune*](https://arxiv.org/abs/2410.02506), studies redundant communication in multi-agent systems. It motivates testing fewer repeated messages and unnecessary interactions. This application does not implement AgentPrune and cannot inherit the paper's savings.
- Liu et al., [*Lost in the Middle*](https://aclanthology.org/2024.tacl-1.9/), finds that relevant-information position affects performance on long-context tasks. This motivates scoped evidence and explicit coverage checks; a large advertised context window does not guarantee effective use of all evidence.
- Smit et al., [*Should we be going MAD?*](https://proceedings.mlr.press/v235/smit24a.html), reports that multi-agent debate does not reliably beat simpler alternatives under its tested protocols. That supports retaining competitive simple baselines; its debate findings are not direct evidence about this requirements pipeline.
- Chen et al., [*Humans or LLMs as the Judge?*](https://aclanthology.org/2024.emnlp-main.474/), documents biases in human and LLM judgment. Automatic F1 should therefore remain one measure alongside independent human review and critical-error checks.
- Dror et al., [*The Hitchhiker’s Guide to Testing Statistical Significance in NLP*](https://aclanthology.org/P18-1128/), motivates choosing analysis suited to the study design. Two documents with one repetition and no predeclared noninferiority margin cannot establish general quality equivalence; paired intervals here are exploratory.

## Controls, provenance and remaining work

Generation uses Gemini 3.6 Flash with minimal thinking; the shared Judge uses the frozen medium-thinking evaluator. Each trial has a 100,000-token generation allowance and a separate 100,000-token Judge allowance. Execution order was randomized with seed 20260929 before results. The source catalogs were assistant-drafted, exact-citation checked and explicitly approved by the user; approval is not evidence that they are a perfect independent gold standard.

Before any trial, runtime preflight found two dependency differences from the local manifest: google-genai 2.17.0 → 2.19.0 and pypdf 6.15.0 → 6.16.2. The deployed runtime was frozen after confirming no old/new trial records existed and revalidating all 175 source references. Source code, catalogs, model settings and trial order were unchanged. Both manifests and the preflight audit are retained. An initial launch was blocked by automatic approval review; the user then explicitly authorized the Gemini data transfer before execution.

The blinded package is `results/blind/START-HERE.md`. Keep condition mapping, costs, logs and this report away from raters until their original ratings are frozen. Anyone exposed to trial outcomes in this task should disclose that exposure; preferably select two raters who have not viewed it. Collect separate coverage, groundedness, executability and redundancy scores, plus critical errors. Calculate agreement and record adjudication separately without replacing initial ratings. No human scores have been fabricated. Item 5 remains incomplete until these ratings and the final analysis are finished.
'''
(root / 'cost-and-quality-report.md').write_text(report)
print(json.dumps({k: analysis[k] for k in ('recorded_trials', 'completed_trials', 'automatically_scored_trials', 'total_accounted_tokens', 'remaining_authorized_tokens')}, indent=2))
