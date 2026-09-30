# Independent human review

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

Model Manager has explicit TBDs. Do not silently invent defaults or external-standard details. Log ambiguities separately from clear contradictions.

Keep the suite_id values intact. Save ratings-R1.csv and ratings-R2.csv and return both. Scores remain separate; any adjudication is recorded later without overwriting them. These suites come from one previously used development document and cannot establish general quality equivalence. Disclose prior exposure to this document or any earlier suites. Generation status and the costs of failed attempts and recovery are reported separately from your artifact ratings.

## Review items

- [95868c4dd524d34e](95868c4dd524d34e.md) — D001
- [d4a90fffd3189ab4](d4a90fffd3189ab4.md) — D001
