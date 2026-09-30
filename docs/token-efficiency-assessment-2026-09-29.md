The current pipeline has avoidable token amplification and a repair-contract defect. The strongest first change is to repair only affected artifacts while preserving completed work. Increasing the run ceiling or changing models alone does not address either problem. Quality preservation must be demonstrated on held-out documents; the literature supports the proposed mechanisms, not a guaranteed saving for this application.

Assessment date: 29 September 2026. Scope: the running `brd-srs-to-test-case` application, its current workspace source, the saved Telescope run, and primary research sources. This assessment made no paid model calls and changed no application code. Machine-readable measurements are in [the audit snapshot](/Users/jun/projects/citd-final/brd-srs-to-test-case/docs/token-efficiency-audit-2026-09-29.json).

**The saved run identifies the failure much more precisely than the screenshot.**

Run: `20260929T083231067296Z-fe7216c0e95e-3bd1bb11`.

| Observation | Verified value |
|---|---:|
| PDF size | 701,945 bytes; 107 pages |
| Saved source | 99 chunks; 152,543 text characters |
| Generation model and thinking | Gemini 3.6 Flash; minimal for all generation roles |
| Configured run ceiling | 2,000,000 tokens |
| Recorded input tokens | 753,182 |
| Recorded output tokens | 220,640 |
| Displayed charged tokens | 973,822 |
| Transport retries / schema repairs | 0 / 0 |
| Recorded semantic revisions | 11 |
| Elapsed generation time | 836.6 seconds, about 14 minutes |
| Completed intermediate work | 283 candidates; 230 requirements; 62 scenarios; 63 tests |
| Last completed stage | Critic, with three findings |
| Subsequent operation | Full-bundle repair, capped at 30,000 output tokens |

The file is small in bytes but contains substantial text. The local PDF's SHA-256 matches the saved document hash. The deployed and workspace hashes also match for `pipelines.py`, `prompts.py`, `providers.py`, `runner.py`, and `documents.py`. Re-extracting locally produced 67 additional text characters; the measurements here use the saved run's chunks, not a replacement extraction.

The displayed total is cumulative across calls. Google documents separate limits of 1,048,576 input tokens and 65,536 output tokens for this model. Comparing a whole-run total with a per-request context limit is invalid. The Interactions API describes `incomplete` as incomplete results, including hitting token limits; this application maps every such status to an output-limit error without preserving a more detailed cause. The saved stage sequence and the large repair contract strongly implicate final repair, but no per-request usage log or raw failed response remains to prove its exact input size or finish reason. [Google model limits](https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash), [Interactions API](https://ai.google.dev/api/interactions-api).

For this run, input plus output exactly equals 973,822, with zero recorded transport retries. There is no evidence that an estimated transport charge inflated this particular total, or that high thinking settings caused it. In general, the provider wrapper charges a full reservation after some API errors, so the UI's “charged” label is not always an invoice-quality measurement. It also does not retain separate thought/cache usage. See [provider accounting](/Users/jun/projects/citd-final/brd-srs-to-test-case/src/brd_srs_testgen/providers.py:530).

**The architecture has useful safeguards, but its handoffs carry too much repeated content.**

The main flow is PDF extraction → ordered Scouts → Curator → global Scenario Architect → parallel Test Writers → Critic → optional repair → deterministic validation → independent coverage evaluation. Three workers means concurrency, not three total model calls.

| Component | Current behavior | Assessment |
|---|---|---|
| Parsing | Page chunks, TOC filtering, stable hashes | Retain; add stable evidence-span IDs and preserve tables/cross-references when segmenting |
| Scouts | Approximately 6,000-character groups, previous whole chunk overlap | Exhaustive scanning is useful; overlap duplicates evidence and candidates |
| Curator | Global assignment; for over 100 candidates, materialize requirements in groups of 24 | Already partly batched; still regenerates every retained requirement and repeats evidence |
| Scenario Architect | All canonical requirements plus all source, one output | Global coordination helps dependencies; full prose generation is an output bottleneck |
| Writers | Up to three scenarios, relevant requirements, dependencies and source chunks | Good existing scoping; batch by actual request and output size as well as item count |
| Critic | Entire bundle plus entire source | Useful semantic check, expensive global payload |
| Repair | All source, all findings, entire bundle; returns entire bundle | Main observed failure and highest-priority change |
| Validation | Schema, IDs, citations, traceability and coverage links | Retain; these checks do not prove semantic correctness |
| Persistence/UI | Typed stage outputs survive handled failures; final bundle becomes empty | Valuable audit trail, but no task resume and misleading zero final counts |
| Coverage judge | Frozen reusable catalog, then test mappings; separate 100k ceiling | Strong basis for fair evaluation; extraction and mapping remain single large requests |

Single-prompt generation requests one entire bundle. Staged generation makes three calls but resends all source with increasing intermediate artifacts. Both also have output-size risks. “Single agent” versus “multi-agent” is not itself an efficiency explanation; the amount of repeated evidence and regenerated output matters.

**The immediate defect is a whole-document rewrite disguised as targeted repair.**

The Critic named only `SCN-013`, `TC-013`, and `TC-062`. Two findings request corrected citations; the third distinguishes a command being *ignored* from being *rejected*. That last distinction demonstrates why simply disabling the Critic could reduce quality.

The complete bundle occupies 275,927 characters of compact JSON. The three affected artifacts occupy 3,401 characters: 98.8% less artifact payload. This is a measured character reduction for that component, not a measured token or whole-run saving. The repair prompt itself is 436,699 characters before schema and configurable instructions. It asks the model to reproduce all 355 artifacts under a 30,000-token output cap. [Repair prompt](/Users/jun/projects/citd-final/brd-srs-to-test-case/src/brd_srs_testgen/prompts.py:340), [repair execution](/Users/jun/projects/citd-final/brd-srs-to-test-case/src/brd_srs_testgen/pipelines.py:1107).

There is a second defect: `_validate_repair_scope` rejects changed artifacts whose authored text is unchanged, with `Repair changed links only.` An offline check using the saved `SCN-013` confirms that the Critic's requested citation-only correction is rejected. Raising the output cap would therefore leave another failure waiting behind the truncation. Citation correction must be allowed when explicitly requested and verified; unrelated link changes should remain forbidden. [Repair guard](/Users/jun/projects/citd-final/brd-srs-to-test-case/src/brd_srs_testgen/pipelines.py:1053).

The reconstructed pre-repair bundle passes deterministic validation. This does not make it semantically correct: the Critic still identified meaningful defects. Preserve that bundle as an incomplete draft with outstanding findings, rather than presenting zero work or silently treating it as approved. Resume should create a linked attempt with provenance while preserving the original immutable run.

**Repeated input is the larger cost category.**

Input accounts for 77.3% of recorded usage. Reconstruction from saved artifacts gives the following prompt sizes. These are characters, not Gemini token counts, and exclude schemas, role setup text, run-specific instructions and original citation-correction requests.

| Stage | Base generation calls | Sum of prompt characters |
|---|---:|---:|
| Scouts | 32 | 259,043 |
| Curator assignment | 1 | 321,922 |
| Curator materialization | 10 | 344,339 |
| Scenario Architect | 1 | 300,325 |
| Writers | 21 | 366,585 |
| Critic | 1 | 435,324 |
| Repair | 1 | 436,699 |
| Total | 67 | 2,464,237 |

The 11 recorded semantic revisions correspond to Scout citation corrections on this path; final repair increments that counter only after success. Thus control flow implies approximately 78 generation requests, excluding token-count requests and any SDK-internal behavior. This is an inference from code and stored outputs, not a per-call provider audit.

Scouts receive 207,823 source characters after overlap versus 152,543 unique saved characters: 36.2% extra source text at that stage. The solution is not to discard boundary information indiscriminately. Use section-aware ownership and only the boundary sentences, table headers, definitions or explicitly referenced passages needed for continuity. Record unresolved cross-references.

Stage “budget shares” allocate maximum output per call, not enforce total stage spending. Input, correction calls and multi-pass Curator work sit outside that intuitive interpretation. A larger run ceiling can allow larger per-call outputs without removing redundant work. [Allocation](/Users/jun/projects/citd-final/brd-srs-to-test-case/src/brd_srs_testgen/pipelines.py:328), [Scout grouping](/Users/jun/projects/citd-final/brd-srs-to-test-case/src/brd_srs_testgen/pipelines.py:427).

**Recommended strategy: keep the role separation initially and shrink what crosses each boundary.**

1. **Return repair deltas.** Send findings, affected artifacts, necessary dependencies and their evidence. Return only allowed changed fields or replacement artifacts. Merge in Python, retain unaffected objects exactly, then validate citations, IDs and semantics. Route mixed findings by responsible role. Permit explicitly verified citation changes; require substantive changes only for substantive findings. Support a separate controlled operation for genuinely missing artifacts, since the current existing-ID-only contract cannot add omitted tests.

2. **Store evidence once.** Split source into stable spans with IDs and offsets; let models select IDs, then resolve page, section and exact excerpts in code. Pass the selected verbatim span and necessary surrounding context when a model must reason about it. An ID alone is not semantic evidence. This avoids repeated model copying of long chunk hashes and quotes, and targets the 11 observed citation correction calls. Deterministic span validity still does not prove that a citation supports a claim; retain semantic checks.

3. **Make Curator output decisions first, edits only where needed.** Reuse unchanged candidate fields and citation unions in code. Give global reconciliation a compact catalog of source anchors, actors, triggers, constraints and outcomes. Fetch full supporting spans for ambiguous merges. Do not deduplicate merely by similar wording, which could merge different limits or state-dependent behavior.

4. **Keep a compact global coverage map; generate detailed artifacts locally.** Have the Scenario Architect allocate obligations and cross-requirement dependencies, then generate scenario detail in bounded groups. Reuse the existing writer evidence selection. Cover each source-supported obligation; do not mechanically generate five scenario types for every requirement. The observed 62 scenarios linking 230 requirements need semantic review, not a forced count increase or decrease.

5. **Use deterministic checks before semantic review.** Check schema, links, missing coverage and duplicates in Python. Review semantic obligations with focused evidence; retain a source-to-requirement gap pass so omissions are not invisible merely because no generated artifact cites them. A compact global index can detect cross-batch inconsistency. Keep the Critic in the first optimized version; test selective review as a later ablation with random audits of apparently easy cases.

6. **Bound both input and expected output.** Size batches using measured serialized artifacts and model limits, with room for schema and reasoning. Split a batch after truncation rather than replaying the same oversized contract. Maintain a real run-cost ceiling separately from a per-call output cap. Example pilot sizes such as 5–10 requirements or 3–6 scenarios are tuning candidates, not research-derived constants.

7. **Checkpoint completed tasks and make accounting explicit.** Persist completion after each task, not only finalization. Key reuse by document/chunk hash, code/prompt/schema versions, model, thinking settings and upstream artifact hashes. Log stage/task/attempt, input, visible output, thought/cache fields where available, reported total, estimated reservation charges separately, latency, status and provider request ID. Preserve incomplete responses for diagnosis under the application's data-retention policy. Show generation, evaluation and recovery costs separately.

The independent judge did not run in the failed 973,822-token attempt. On completed runs, its usage is folded into displayed totals. Its catalog extraction and mapping should also be bounded and versioned. Reuse the approved catalog across conditions, but never feed that evaluation catalog into generation. Separate one-time catalog construction cost from per-run mapping cost. [Judge pipeline](/Users/jun/projects/citd-final/brd-srs-to-test-case/src/brd_srs_testgen/coverage.py:113), [metric aggregation](/Users/jun/projects/citd-final/brd-srs-to-test-case/src/brd_srs_testgen/runner.py:811).

A later ablation can combine scenario detail and test writing in one scoped generation call. That could remove an intermediate prose handoff, but should not be the first change because it alters the study's architecture more substantially. Smaller-model routing may reduce money without reducing token count; caching similarly must be measured separately from logical prompt volume. Neither fixes an oversized output contract.

**The research supports specific mechanisms, with clear limits on transfer.**

| Primary source | Relevant evidence and application | Limitation |
|---|---|---|
| Zhang et al., *Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems* (AgentPrune, ICLR 2025; [paper](https://arxiv.org/abs/2410.02506)) | Studies redundant agent communication and reports 28.1–72.8% token reductions across its experiments. Motivates scoped evidence and smaller handoffs. | Its graph-pruning method and benchmarks differ from this pipeline. Borrow the principle; do not claim to implement AgentPrune or inherit its savings. |
| Smit et al., *Should we be going MAD?* (ICML 2024; [paper](https://proceedings.mlr.press/v235/smit24a.html)) | Debate protocols did not consistently outperform simpler alternatives; tuning affected outcomes. Supports strong single/staged baselines and equal-budget comparison. | Debate is not the same protocol as this role-based pipeline; it does not prove this system is worse. |
| Liu et al., *Lost in the Middle* (TACL 2024; [paper](https://aclanthology.org/2024.tacl-1.9/)) | Long-context QA/retrieval performance depended on where evidence appeared. Motivates relevant evidence close to each task. | The study does not measure Gemini 3.6 or BRD/SRS generation. Shorter scoped prompts are a hypothesis to test here. |
| Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet* (ICLR 2024; [paper](https://arxiv.org/abs/2310.01798)) | Finds limitations of intrinsic self-correction without external feedback. Motivates validator-backed, evidence-specific repair. | It is not a blanket rejection of all review; this system's critic has source evidence and catches a concrete semantic issue. |
| Jiang et al., *LLMLingua* (EMNLP 2023; [paper](https://aclanthology.org/2023.emnlp-main.825/)) | Demonstrates prompt compression with limited degradation on evaluated tasks. Establishes compression as an empirical research direction. | Exact requirement constraints, negation and numeric boundaries are fragile. Start with removing duplicated fields; do not apply lossy compression to authoritative evidence without an ablation. |
| Chen, Zaharia and Zou, *FrugalGPT* (2023 preprint; [paper](https://arxiv.org/abs/2305.05176)) | Studies cost-aware model cascades. Supports optional routing of easy tasks to cheaper models with escalation. | Monetary savings differ from token savings; a router must be calibrated. Keep models fixed first to isolate architecture effects. |
| Alagarsamy et al., *Enhancing Large Language Models for Text-to-Testcase Generation* ([authors' preprint](https://arxiv.org/abs/2402.11910), [journal version](https://www.sciencedirect.com/science/article/pii/S0164121225001992)) | Studies description-to-test generation and evaluates correctness, requirement alignment and coverage. Supports measuring several quality dimensions. | Uses Java method descriptions and executable tests; its code-coverage results cannot be transferred to manual SRS test cases. |

The proposed contribution is therefore an evidence-scoped, bounded generation pipeline with minimal repair and explicit cost/quality evaluation. This is a research-informed engineering design, not an established theorem that fewer calls preserve quality.

**For the final assignment, test quality preservation rather than assume it.**

A suitable research question is: “Can scoped evidence handoffs and minimal artifact repair reduce total generation tokens while preserving requirement coverage, groundedness and executability?”

Use separate development and held-out documents, stratified by length and requirements density. Freeze prompts, parser dependencies, schemas, model versions, thinking, code hashes and budgets before the final experiment. Retain single-prompt, staged and optimized multi-agent conditions. Use the original multi-agent implementation and incremental variants on an ablation subset to attribute improvements to repair deltas, evidence reuse and batching. Do not change models at the same time as architecture in the main comparison.

A practical pilot is six held-out documents × three conditions × three independent repeats = 54 runs, following tuning on other documents. This is a feasibility sample, not a power calculation; wide confidence intervals may leave quality preservation inconclusive. Do not reuse generated artifacts across independent repetitions. Resume/reuse should be reported as operational recovery, or evaluated as a separate condition. Account for each failed attempt and its consumed tokens.

Use the existing approved, frozen coverage catalog with blinded human checks, and measure coverage F1, groundedness, executability, redundancy, completion rate, latency and tokens. Report supported negative, boundary and state-transition obligations separately. The current project-specific F1 is not classifier micro-F1 and does not by itself penalize all redundant or unusable tests. Deterministic trace links must not substitute for checking actions and expected outcomes.

Report both equal-budget comparisons and observed cost/quality tradeoffs. Predeclare a meaningful quality tolerance with the supervisor; for example, a 0.02 absolute F1 margin is a proposed study choice, not a literature standard. Require the confidence interval for optimized-minus-baseline quality to stay above the negative tolerance before claiming non-inferiority, alongside acceptable human ratings and completion rate. Analyze paired document-level outcomes; do not treat hundreds of tests from one document as independent samples. Statistical choices must follow the design and distribution. [Dror et al., ACL 2018](https://aclanthology.org/P18-1128/).

Report savings as `1 - optimized_generation_tokens / baseline_generation_tokens` under clearly specified comparable runs. Include failure costs and separately report tokens per completed, quality-approved suite. Keep evaluator failures separate from quality zero, and always report their rate. Freeze approval before official comparisons; the runner's ability to score a machine-frozen catalog is not itself evidence of human approval.

The defensible immediate result is the demonstrated waste in final repair and the confirmed citation-guard conflict. No end-to-end savings percentage or quality improvement has yet been measured. Implementing only smaller repair responses, valid citation repair and recoverable completed work is the smallest first experiment that directly addresses this incident.
