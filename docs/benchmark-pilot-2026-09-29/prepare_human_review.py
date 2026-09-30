"""Package the existing blinded export; no model calls or generated ratings."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
blind = root / 'results' / 'blind'
manifest = json.loads((root / 'pilot-manifest.json').read_text())
summary = json.loads((root / 'results' / 'summary.json').read_text())
assert summary['recorded_trials'] == summary['planned_trials'] == 8


def plain(value):
    return str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('|', '\\|').replace('\n', '<br>')


items = []
with (blind / 'ratings.csv').open(newline='') as handle:
    suite_ids = sorted(row['suite_id'] for row in csv.DictReader(handle))
for suite_id in suite_ids:
    path = blind / f'{suite_id}.json'
    item = json.loads(path.read_text())
    assert set(item) == {'suite_id', 'document_id', 'artifacts'}
    lines = [f"# Suite {item['suite_id']}", '', f"Source: {item['document_id']}", '',
             'Review the supplied artifacts as they stand. Missing sections remain missing; do not infer tests from requirements alone.', '']
    for collection, artifacts in item['artifacts'].items():
        lines += [f"## {collection.replace('_', ' ').title()} ({len(artifacts)})", '']
        for artifact in artifacts:
            identifier = next(artifact[key] for key in ('test_case_id', 'scenario_id', 'requirement_id') if key in artifact)
            lines += [f"### {identifier}: {plain(artifact['title'])}", '']
            for key, value in artifact.items():
                if key in ('title', 'test_case_id', 'requirement_id') or not value:
                    continue
                if key == 'steps':
                    lines += ['| Step | Action | Expected result |', '|---|---|---|']
                    lines += [f"| {step['step_number']} | {plain(step['action'])} | {plain(step['expected_result'])} |" for step in value]
                elif key == 'source_references':
                    lines += ['**Source evidence**', '']
                    lines += [f"- PDF p. {ref['page_number']}, {plain(ref['section'])}, `{ref['chunk_id']}`: {plain(ref['excerpt'])}" for ref in value]
                else:
                    display = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    lines += [f"**{key.replace('_', ' ').title()}:** {plain(display)}"]
                lines.append('')
    path.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    items.append((item['suite_id'], item['document_id']))

for document in manifest['documents']:
    data = Path(document['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == document['sha256']
    (blind / f"{document['document_id']}-source.pdf").write_bytes(data)
    expected = manifest['source_catalog_drafts'][document['document_id']]
    catalog = (root / expected['file']).read_bytes()
    assert hashlib.sha256(catalog).hexdigest() == expected['sha256']
    (blind / f"{document['document_id']}-catalog.json").write_bytes(catalog)
    shutil.copyfile(root / f"{document['document_id']}-catalog-review.md", blind / f"{document['document_id']}-catalog.md")

for rater_id in ('R1', 'R2'):
    target = blind / f'ratings-{rater_id}.csv'
    if not target.exists():
        with (blind / 'ratings.csv').open(newline='') as source, target.open('w', newline='') as destination:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                row['rater_id'] = rater_id
                writer.writerow(row)

instructions = '''# Independent human review

Choose two people who have not viewed the trial outcomes or logs; disclose any prior exposure. Use one actual person for R1 and a different person for R2. Each person completes only their own ratings file independently. Do not open the parent results folder, private mapping, cost report or trial logs until both original ratings are frozen. Do not use an LLM to supply human scores.

Read each source PDF and its approved catalog, then each suite below. The JSON is the authoritative artifact; Markdown is a reading copy. Catalog files retain their original draft wording, but their contents were approved before these suites were generated. An empty or partial artifact is not a completed suite. Record missing material in notes; do not infer test coverage from requirements or scenarios alone. Runs without any saved bundle have no reviewable artifact and are separately counted as generation failures, not assigned fabricated ratings.

Use the existing quality-rubric-v1 scale: **1 = Major deficiencies; 2 = Material gaps; 3 = Minor gaps; 4 = Strong.**

| CSV column | Assess |
|---|---|
| source_coverage | Completeness of source-backed testable behavior. |
| groundedness | Claims and steps remain supported by the source. |
| executability | A tester can perform the steps and observe the outcome. |
| redundancy_control | Cases add distinct value without avoidable duplication. |
| critical_errors | Integer count of distinct critical errors; 0 only after checking. Explain each in notes with test IDs and source pages. |
| notes | Missing behavior, unsupported outcomes, incorrect boundaries/negation, actor or state mistakes, and any unscorable dimension. |

Check expected outcomes against the source, including exact numbers, inclusive/exclusive limits, actors, timing, negation and state transitions. Citations alone do not prove the test is correct. If a dimension cannot be judged (for example, no test cases to assess for executability), leave that score blank and explain why; do not invent evidence. Missing scores are reported as missing, not converted to zero or forced to pass.

DigitalHome contains unresolved source wording: the humidistat section says manual “temperature”; the relationship between 24 hourly settings and four planner periods is unspecified; alarm/contact wording differs between sections. Model Manager has explicit TBDs. Do not silently invent defaults or external-standard details. Log ambiguities separately from clear contradictions.

Keep the suite_id values intact. Save ratings-R1.csv and ratings-R2.csv and return both. Scores remain separate; any adjudication is recorded later without overwriting them. Eight planned trials on two documents are a feasibility pilot, not evidence of general quality equivalence.

## Review items

'''
instructions += '\n'.join(f'- [{suite_id}]({suite_id}.md) — {document_id}' for suite_id, document_id in items) + '\n'
(blind / 'START-HERE.md').write_text(instructions)
print(f'Prepared {len(items)} artifact reading copies, two blank rater forms and two source PDFs. No human ratings supplied.')
