Token-efficiency implementation — first release, 29 September 2026

Phases 0–1 of the [research-based plan](superpowers/plans/2026-09-29-token-efficient-quality-preserving-pipeline.md) are implemented in the working tree. The running application has not been rebuilt or deployed. No paid model request was made.

Implemented behavior:

- Each provider attempt records model, phase, task, attempt, effective output cap, request/schema hashes, latency, finish reason, available usage and bounded, sanitized failure details. Attempts are persisted immediately in an additive PostgreSQL table. Identical re-insertion is idempotent; conflicting records and writes to terminal runs are rejected.
- Provider totals remain authoritative. Missing counts remain null. Conservative charges for unknown usage are labeled estimates. The result view separates generation, catalog and evaluation usage, failed-call accounting and maximum observed request input. Historical runs retain their aggregate totals with an explicit legacy label.
- Repairs return only named replacement artifacts, grouped by responsible role. Python merges exact IDs, rejects missing/extra/duplicate targets and unrelated changes, and distinguishes citation, relationship and semantic corrections. Deterministic validation and a small verification response follow the patches; an unresolved finding cannot become an accepted suite through a cosmetic edit alone.
- Exact citation checks tolerate whitespace while preserving symbols, signs, values, units and negation. Fuzzy quote rewriting no longer proves support. Ambiguous matches do not silently select a chunk.
- Repair failure retains the draft, counts, traceability and unresolved findings. Expected Judge failures preserve completed generation, including failures before a catalog exists. Unexpected programming errors still surface.
- PDF preflight reports pages, extracted characters, chunks and initial non-empty Scout tasks with a clearly labeled source-text estimate. Output caps and run budgets are distinguished from model context limits. New runs snapshot source digests and dependency versions; prompt/schema/evaluator versions changed.

Measured on the original Telescope trace:

| Measure | Original | New offline reconstruction |
|---|---:|---:|
| Repair and verification prompt characters | 436,699 | 39,353 |
| Artifact JSON requested as repair output, based on current target sizes | 275,927 | 3,502 across two patches |
| Artifacts changed in scripted replay | — | SCN-013, TC-013, TC-062 |

Repair plus verification prompts are approximately 91% smaller. Character counts exclude response schemas and configured role instructions. They are not billed token counts or end-to-end savings. The old traceability findings were explicitly classified as citation repairs for replay. Scripted responses corrected the two citations and changed the command expected result to “The command is ignored.” Only the three requested artifacts changed, and deterministic validation passed. Verification acceptance was scripted; it is not an independent semantic quality score. Exact measurements are in [the replay JSON](token-efficiency-phase01-replay-2026-09-29.json).

Verification: **541 tests passed**, including Streamlit result-state checks, provider failures/concurrency, patch validation, Judge failure isolation, legacy storage compatibility and additive persistence behavior. Tests ran in a temporary application-image container with the working tree mounted read-only and PostgreSQL access restricted by the test fixture to the dedicated local `brd_srs_test` database. Three existing pytest collection warnings concern imported model classes named `TestCase`, `TestPriority` and `TestStep`. `git diff --check` passed. Application runs were read only for incident replay.

Remaining work is Phases 2–5: compact evidence handoffs, size-bounded generation/review, durable stage recovery, then a frozen held-out cost/quality experiment. Stage outputs still finalize at run completion; only call-attempt accounting is durable immediately in this release. No claim of equal quality or measured billed-token savings should enter the assignment until that experiment is performed. The existing literature motivates these changes; the offline replay verifies the failure correction, not the research hypothesis.
