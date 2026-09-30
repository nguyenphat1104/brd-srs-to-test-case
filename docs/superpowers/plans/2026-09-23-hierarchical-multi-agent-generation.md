# Hierarchical Multi-Agent Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current independent extract-and-merge pipeline with a coverage-first hierarchy that preserves document order, synthesizes one global requirement catalog, plans scenarios globally, writes test cases in parallel, and always performs one evidence-grounded critique with at most one repair.

**Architecture:** Ordered overlapping evidence groups feed parallel Requirement Scouts. One Curator semantically reconciles every candidate without truncation. One Scenario Architect sees the complete canonical catalog. Parallel Test Writers expand assigned canonical scenarios. One Critic checks the complete bundle, and the orchestrator routes at most one repair to the responsible role. Typed stage snapshots form a persisted blackboard and make every handoff auditable.

**Tech Stack:** Python 3.11, Pydantic, existing provider abstraction, `ThreadPoolExecutor`, PostgreSQL/psycopg, Streamlit, pytest.

---

## Scope and invariants

- This plan implements Sections 5–7 and the generation parts of Sections 11–12 in `docs/superpowers/specs/2026-09-23-coverage-first-multi-agent-enhancement-design.md`.
- Complete Plan 1 first. This plan assumes `RunResult` can persist typed evaluation records but does not change the evaluator.
- Keep the public run type `centralized_multi_agent`; existing saved runs remain loadable.
- No final-artifact count cap is allowed. Per-task caps are acceptable only when every assigned input is covered and the orchestrator creates enough tasks.
- Remote concurrency is at most three. Local providers remain sequential through the existing `worker_limit` behavior.
- Every candidate receives exactly one Curator decision: retain, merge, or reject with a reason.
- Every canonical requirement appears in at least one Scenario Architect output.
- Every canonical scenario receives at least one Test Writer test case.
- The Critic runs even when deterministic validation already passes. At most one semantic repair call occurs.
- All five roles are configurable per run: Scout, Curator, Scenario Architect, Test Writer, and Critic.

## Task 1: Define role, blackboard, and handoff contracts

**Files:**
- Modify: `src/brd_srs_testgen/models.py`
- Modify: `src/brd_srs_testgen/prompts.py`
- Modify: `tests/test_pipelines.py`
- Modify: `tests/factories.py`

- [ ] **Step 1: Write failing model tests for each handoff**

Add tests that construct one candidate, one Curator decision, one global scenario batch, one critic finding, and one stage snapshot:

```python
def test_curator_decision_names_exactly_one_candidate():
    decision = RequirementDecision(
        candidate_id="CAND-001-001",
        action=RequirementDecisionAction.MERGE,
        canonical_requirement_id="REQ-001",
        reason="Same rule repeated in an adjacent section.",
    )
    assert decision.canonical_requirement_id == "REQ-001"


def test_critic_finding_routes_one_required_action():
    finding = CriticFinding(
        finding_id="FIND-001",
        severity=CriticSeverity.HIGH,
        finding_type="missing_boundary",
        artifact_ids=["REQ-001", "SCN-001"],
        responsible_role="test_writer",
        required_action="Add an executable boundary test at the stated maximum.",
        source_references=[source_reference()],
    )
    assert finding.responsible_role == "test_writer"
```

- [ ] **Step 2: Run and confirm the contracts are absent**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_pipelines.py`

Expected: FAIL with imports for the new handoff types.

- [ ] **Step 3: Add the minimum contracts**

Add these models in `models.py`:

```python
class RequirementDecisionAction(StrEnum):
    RETAIN = "retain"
    MERGE = "merge"
    REJECT = "reject"


class CriticSeverity(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CandidateRequirement(StrictModel):
    candidate_id: str = Field(pattern=r"^CAND-\d{3}-\d{3,}$")
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    requirement_type: RequirementType
    module: str = Field(min_length=1)
    priority: RequirementPriority
    ambiguities: list[str] = Field(default_factory=list)
    source_references: list[SourceReference] = Field(min_length=1)


class CandidateRequirementBatch(StrictModel):
    candidates: list[CandidateRequirement]


class RequirementDecision(StrictModel):
    candidate_id: str = Field(pattern=r"^CAND-\d{3}-\d{3,}$")
    action: RequirementDecisionAction
    canonical_requirement_id: str | None = Field(
        default=None, pattern=r"^REQ-\d{3,}$"
    )
    reason: str = Field(min_length=1)


class RequirementSynthesis(StrictModel):
    decisions: list[RequirementDecision]
    requirements: list[Requirement]


class CriticFinding(StrictModel):
    finding_id: str = Field(pattern=r"^FIND-\d{3,}$")
    severity: CriticSeverity
    finding_type: str = Field(min_length=1)
    artifact_ids: list[str] = Field(min_length=1)
    responsible_role: Literal["curator", "scenario_architect", "test_writer"]
    required_action: str = Field(min_length=1)
    source_references: list[SourceReference] = Field(min_length=1)


class CriticReport(StrictModel):
    accepted: bool
    findings: list[CriticFinding] = Field(default_factory=list)


class AgentStageOutput(StrictModel):
    stage: Literal["scout", "curator", "scenario_architect", "test_writer", "critic", "repair"]
    task_index: int = Field(ge=0)
    role: str = Field(min_length=1)
    input_ids: list[str] = Field(default_factory=list)
    output: dict[str, JsonValue]
    created_at: AwareDatetime
```

Use a model validator on `RequirementDecision`: reject must have no canonical ID; retain/merge must have one. Use a model validator on `CriticReport`: `accepted=True` requires no findings, and `accepted=False` requires at least one.

- [ ] **Step 4: Add the five configurable roles**

Extend `AgentSetup.agent` and `default_agent_setups()` with `scout`, `curator`, `scenario_architect`, `test_writer`, and `critic`. Retain the old `analyst`, `test_generator`, and `reviewer` literals/defaults for loading old configuration rows, but stop using them for new multi-agent runs.

Add the same five keys to `RUN_PROMPT_DEFAULTS`. Keep the instructions descriptive and evidence-bound; do not include experimental condition names.

- [ ] **Step 5: Run tests and commit**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_pipelines.py tests/test_runner.py`

Expected: PASS after fixture updates.

```bash
rtk proxy git add src/brd_srs_testgen/models.py src/brd_srs_testgen/prompts.py tests/test_pipelines.py tests/test_runner.py tests/factories.py
rtk proxy git commit -m "feat: define hierarchical agent handoffs"
```

## Task 2: Preserve evidence order and remove global truncation

**Files:**
- Modify: `src/brd_srs_testgen/pipelines.py`
- Modify: `src/brd_srs_testgen/prompts.py`
- Modify: `tests/test_pipelines.py`

- [ ] **Step 1: Add failing partition tests**

```python
def test_ordered_evidence_groups_keep_source_order_with_boundary_overlap():
    chunks = [chunk(i, text="x" * 40) for i in range(1, 7)]

    groups = _ordered_evidence_groups(chunks, char_limit=100)

    assert [[item.chunk_id for item in group] for group in groups] == [
        ["chunk-001", "chunk-002"],
        ["chunk-002", "chunk-003", "chunk-004"],
        ["chunk-004", "chunk-005", "chunk-006"],
    ]


def test_ordered_evidence_groups_do_not_duplicate_a_single_group():
    chunks = [chunk(1), chunk(2)]
    assert _ordered_evidence_groups(chunks, char_limit=100_000) == [chunks]
```

Also add a regression test with 25 distinct candidates and assert no helper slices the result to 20.

- [ ] **Step 2: Run the focused tests**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_pipelines.py -k 'ordered_evidence or truncate'`

Expected: FAIL because `_ordered_evidence_groups` is missing and `_merge_worker_requirements` truncates.

- [ ] **Step 3: Implement an ordered wrapper around the existing grouping helper**

```python
def _ordered_evidence_groups(
    chunks: list[DocumentChunk], *, char_limit: int
) -> list[list[DocumentChunk]]:
    groups = _bounded_groups(chunks, lambda chunk: len(chunk.text), char_limit)
    if len(groups) < 2:
        return groups
    return [
        group if index == 0 else [groups[index - 1][-1], *group]
        for index, group in enumerate(groups)
    ]
```

Do not use `_balance` for evidence. It sorts by size and destroys neighboring context.

- [ ] **Step 4: Remove global caps**

Delete this line from `RULES`:

```text
- Consolidate overlapping evidence into at most 20 requirements and 24 scenarios.
```

Delete `_merge_worker_requirements`, `BoundedRequirementBatch`, and the `[:20]` selection. The Curator in Task 4 replaces them. Keep small per-task output schemas for scouts and writers only.

- [ ] **Step 5: Run tests and commit**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_pipelines.py`

Expected: PASS.

```bash
rtk proxy git add src/brd_srs_testgen/pipelines.py src/brd_srs_testgen/prompts.py tests/test_pipelines.py
rtk proxy git commit -m "fix: preserve evidence order without truncation"
```

## Task 3: Implement Requirement Scouts

**Files:**
- Modify: `src/brd_srs_testgen/prompts.py`
- Modify: `src/brd_srs_testgen/pipelines.py`
- Modify: `tests/test_pipelines.py`

- [ ] **Step 1: Add a failing Scout orchestration test**

Script three `CandidateRequirementBatch` responses and assert:

- each call receives a contiguous evidence group;
- overlap appears only at boundaries;
- candidate IDs use worker namespaces such as `CAND-001-001` and `CAND-002-001`;
- calls use agent key `scout`;
- `worker_limit` never exceeds three.

- [ ] **Step 2: Run the test and observe current analyst calls**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_pipelines.py -k scout`

Expected: FAIL because the current pipeline requests `RequirementBatch` from `analyst`.

- [ ] **Step 3: Replace `worker_requirements_prompt` with `scout_prompt`**

The prompt must say:

```text
Extract atomic candidate requirements only from the assigned ordered evidence.
Keep separate rules separate. Preserve ambiguity. Cite a verbatim 5-to-25-word excerpt.
Use candidate IDs CAND-{worker:03d}-001 upward. Do not deduplicate across workers;
the Curator owns cross-worker reconciliation.
```

Return `CandidateRequirementBatch`. Retaining overlap duplicates is intentional and gives the Curator evidence for reconciliation.

- [ ] **Step 4: Add strict Scout validation**

Validate unique IDs within a batch, correct worker prefix, real chunk IDs, exact quoted excerpts, and at least one candidate for every non-empty group unless the model explicitly returns an empty batch. Reuse `canonicalize_source_references`; do not write a second citation checker.

- [ ] **Step 5: Run tests and commit**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_pipelines.py -k 'scout or ordered_evidence'`

Expected: PASS.

```bash
rtk proxy git add src/brd_srs_testgen/prompts.py src/brd_srs_testgen/pipelines.py tests/test_pipelines.py
rtk proxy git commit -m "feat: add ordered requirement scouts"
```

## Task 4: Add one semantic Curator

**Files:**
- Modify: `src/brd_srs_testgen/prompts.py`
- Modify: `src/brd_srs_testgen/pipelines.py`
- Modify: `tests/test_pipelines.py`

- [ ] **Step 1: Add failing Curator completeness tests**

Test that `_validate_synthesis` rejects:

- a missing candidate decision;
- a duplicate candidate decision;
- retain/merge pointing to a nonexistent canonical requirement;
- a rejected candidate with a canonical requirement ID;
- duplicate canonical requirement IDs;
- a canonical requirement whose citations do not come from its retained/merged candidates.

Test a valid semantic merge where differently worded adjacent candidates resolve to one requirement.

- [ ] **Step 2: Run the tests**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_pipelines.py -k curator`

Expected: FAIL on missing prompt and validator.

- [ ] **Step 3: Add `curator_prompt`**

Pass all candidate JSON plus the full ordered evidence. Require one decision per candidate, globally unique IDs beginning at `REQ-001` and increasing by one, preserved ambiguities/dependencies, and explicit rejection reasons. State that wording similarity alone is not enough to merge requirements with different triggers, actors, limits, or outcomes.

- [ ] **Step 4: Implement `_validate_synthesis` and the Curator call**

Use set equality for decision coverage:

```python
candidate_ids = {item.candidate_id for item in candidates}
decision_ids = [item.candidate_id for item in synthesis.decisions]
if len(decision_ids) != len(set(decision_ids)) or set(decision_ids) != candidate_ids:
    raise PipelineOutputError("Curator must decide every candidate exactly once.")
```

Then validate target IDs and citations. The Curator output replaces `_merge_worker_requirements` entirely.

- [ ] **Step 5: Run tests and commit**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_pipelines.py -k 'curator or centralized'`

Expected: PASS.

```bash
rtk proxy git add src/brd_srs_testgen/prompts.py src/brd_srs_testgen/pipelines.py tests/test_pipelines.py
rtk proxy git commit -m "feat: reconcile candidates with semantic curator"
```

## Task 5: Plan scenarios globally, then write tests in parallel

**Files:**
- Modify: `src/brd_srs_testgen/prompts.py`
- Modify: `src/brd_srs_testgen/pipelines.py`
- Modify: `tests/test_pipelines.py`

- [ ] **Step 1: Add failing Scenario Architect tests**

Script a `ScenarioBatch` and assert the one Architect call sees every canonical requirement and all supporting chunks. Validate that every requirement ID appears in at least one scenario and every scenario citation resolves to evidence.

- [ ] **Step 2: Add failing Test Writer tests**

Partition canonical scenarios with `_bounded_groups(scenarios, lambda item: len(item.objective), LOCAL_EVIDENCE_CHARS_PER_TASK, max_items=3)`. Script one `TestCaseBatch` per group and assert:

- writers receive existing scenario IDs and never create scenarios;
- every assigned scenario receives at least one test case;
- test-case IDs are worker-namespaced then deterministically renumbered from `TC-001` upward after merge;
- all test-case requirement IDs are a subset of the assigned scenarios' requirement IDs;
- three workers may run concurrently, never more.

- [ ] **Step 3: Run and observe failures**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_pipelines.py -k 'architect or writer'`

Expected: FAIL because the current workers generate scenarios and tests together from requirement fragments.

- [ ] **Step 4: Add prompts**

`scenario_architect_prompt` returns one `ScenarioBatch` from the complete `RequirementBatch`. It must plan supported positive, negative, boundary, edge, and state-transition coverage without an arbitrary total count.

`test_writer_prompt` receives assigned canonical scenarios, the canonical requirements they reference, and only relevant evidence. It returns `TestCaseBatch`; scenario creation is forbidden.

- [ ] **Step 5: Implement global planning and parallel expansion**

Reuse `_run_parallel_workers`, `_bounded_groups`, `_dependency_context`, `_relevant_chunks`, and `canonicalize_source_references`. Delete `GeneratedCases` use from the multi-agent path. Merge only `TestCaseBatch.test_cases`, renumber deterministically by worker index/output position, and construct:

```python
ArtifactBundle(
    requirements=synthesis.requirements,
    scenarios=scenario_batch.scenarios,
    test_cases=renumbered_test_cases,
)
```

- [ ] **Step 6: Run tests and commit**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_pipelines.py`

Expected: PASS.

```bash
rtk proxy git add src/brd_srs_testgen/prompts.py src/brd_srs_testgen/pipelines.py tests/test_pipelines.py
rtk proxy git commit -m "feat: plan scenarios before parallel test writing"
```

## Task 6: Add always-on critique and one targeted repair

**Files:**
- Modify: `src/brd_srs_testgen/prompts.py`
- Modify: `src/brd_srs_testgen/pipelines.py`
- Modify: `tests/test_pipelines.py`

- [ ] **Step 1: Add failing Critic behavior tests**

Cover these cases:

1. A valid bundle still triggers a Critic call.
2. `accepted=True` returns the bundle unchanged and records zero semantic revisions.
3. Findings trigger exactly one repair call to the highest-severity finding's `responsible_role`.
4. Multiple findings are included in that one repair prompt; there is no second repair.
5. A link-only revision is rejected with `PipelineOutputError`.
6. An invalid repaired bundle is rejected by existing deterministic validation.
7. `context.critic_enabled=False` skips Critic and repair for the required ablation only.

- [ ] **Step 2: Run and observe the missing behavior**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_pipelines.py -k critic`

Expected: FAIL because normal multi-agent output never invokes a reviewer.

- [ ] **Step 3: Add critic and repair prompts**

The Critic sees the complete bundle and evidence. It must cite evidence for every finding and inspect missing source behaviors, unsupported content, weak expected results, non-executable steps, duplicates, invalid trace links, and missing boundary/negative paths.

The repair prompt returns one complete `ArtifactBundle`, changes only artifacts named by findings, preserves unaffected IDs, and applies every finding. It is routed through the responsible role's configured provider/model/prompt.

- [ ] **Step 4: Implement one repair limit**

Add `critic_enabled: bool = True` to `PipelineContext`. Select the repair role by severity order `high`, `medium`, `low`, then finding order. Call `context.generate` once with the repair prompt, `ArtifactBundle` schema, repair output-token allowance, and `agent=repair_role`; increment `semantic_revisions` once.

Implement `_semantic_payload(bundle)` using only authored behavior fields:

```python
def _semantic_payload(bundle: ArtifactBundle) -> tuple[object, ...]:
    return (
        tuple((item.title, item.description, item.ambiguities) for item in bundle.requirements),
        tuple((item.title, item.objective, item.preconditions) for item in bundle.scenarios),
        tuple(
            (
                item.title,
                item.preconditions,
                item.test_data,
                tuple((step.action, step.expected_result) for step in item.steps),
            )
            for item in bundle.test_cases
        ),
    )
```

If the repaired semantic payload equals the original, raise `PipelineOutputError("Repair changed links only.")`.

- [ ] **Step 5: Validate after repair and run tests**

Use the existing `validate_bundle` in the runner after pipeline return; do not duplicate full deterministic validation inside the pipeline. Pipeline-specific ownership checks remain local.

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_pipelines.py tests/test_runner.py`

Expected: PASS and the critic tests show at most one semantic revision.

- [ ] **Step 6: Commit**

```bash
rtk proxy git add src/brd_srs_testgen/prompts.py src/brd_srs_testgen/pipelines.py tests/test_pipelines.py tests/test_runner.py
rtk proxy git commit -m "feat: add one-pass multi-agent critique"
```

## Task 7: Allocate budget and persist the blackboard

**Files:**
- Modify: `src/brd_srs_testgen/pipelines.py`
- Modify: `src/brd_srs_testgen/models.py`
- Modify: `src/brd_srs_testgen/runner.py`
- Modify: `src/brd_srs_testgen/schema.sql`
- Modify: `src/brd_srs_testgen/storage.py`
- Modify: `tests/test_pipelines.py`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_runner.py`

- [ ] **Step 1: Add failing budget tests**

Assert the default multi-agent ceiling is apportioned as:

```python
MULTI_AGENT_BUDGET_SHARES = {
    "scout": 0.25,
    "curator": 0.15,
    "scenario_architect": 0.15,
    "test_writer": 0.30,
    "critic": 0.10,
    "repair": 0.05,
}
```

Scout and Writer shares are divided across their task counts. An explicit per-run `agent_max_output_tokens` value caps that role's call, while the shared `BudgetLedger` remains the final total-ceiling authority.

- [ ] **Step 2: Implement one stage limit helper**

```python
def stage_output_tokens(
    context: PipelineContext,
    stage: str,
    *,
    token_ceiling: int,
    task_count: int = 1,
) -> int:
    allocation = int(token_ceiling * MULTI_AGENT_BUDGET_SHARES[stage] / task_count)
    configured = context.agent_max_output_tokens.get(stage, allocation)
    return max(MIN_OUTPUT_TOKENS, min(configured, allocation))
```

Pass the run ceiling into `PipelineContext`; do not infer it from provider internals.

- [ ] **Step 3: Add failing blackboard round-trip tests**

Assert a completed run saves ordered `AgentStageOutput` rows for every Scout task, Curator, Scenario Architect, every Writer task, Critic, and optional repair.

- [ ] **Step 4: Add one append-only stage table**

```sql
CREATE TABLE IF NOT EXISTS agent_stage_outputs (
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    stage text NOT NULL CHECK (stage IN ('scout', 'curator', 'scenario_architect', 'test_writer', 'critic', 'repair')),
    task_index integer NOT NULL CHECK (task_index >= 0),
    role text NOT NULL CHECK (role <> ''),
    input_ids jsonb NOT NULL CHECK (jsonb_typeof(input_ids) = 'array'),
    output jsonb NOT NULL CHECK (jsonb_typeof(output) = 'object'),
    created_at timestamptz NOT NULL,
    PRIMARY KEY (run_id, stage, task_index)
);
```

Add `stage_outputs: list[AgentStageOutput]` to `PipelineContext` and `RunResult`. Record the validated model dump immediately after every successful handoff. Persist all rows in `RunRepository.finalize` and load them with `load_run`.

- [ ] **Step 5: Snapshot the ablation switch**

Add `critic_enabled` to the run configuration JSON. It defaults to true in normal UI runs. Plan 3 alone exposes the false value as the named ablation arm.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_pipelines.py tests/test_runner.py
rtk env PYTHONPATH=src TEST_DATABASE_URL=postgresql://brd_srs:brd_srs_local@127.0.0.1:5432/brd_srs_test .venv/bin/python -m pytest -q tests/test_storage.py
```

Expected: PASS.

```bash
rtk proxy git add src/brd_srs_testgen/pipelines.py src/brd_srs_testgen/models.py src/brd_srs_testgen/runner.py src/brd_srs_testgen/schema.sql src/brd_srs_testgen/storage.py tests/test_pipelines.py tests/test_runner.py tests/test_storage.py
rtk proxy git commit -m "feat: persist multi-agent blackboard and budgets"
```

## Task 8: Make every role configurable per run

**Files:**
- Modify: `src/brd_srs_testgen/runner.py`
- Modify: `src/brd_srs_testgen/schema.sql`
- Modify: `src/brd_srs_testgen/storage.py`
- Modify: `app.py`
- Modify: `tests/test_runner.py`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Add failing settings tests**

Assert `ProviderSettings.snapshot(RunType.CENTRALIZED_MULTI_AGENT)` contains provider, model, prompt, thinking level, and max output tokens for exactly:

```python
("scout", "curator", "scenario_architect", "test_writer", "critic")
```

Assert changing any field affects only the requested run snapshot and does not mutate shared defaults.

- [ ] **Step 2: Update runner configuration**

Change `RUN_AGENTS[RunType.CENTRALIZED_MULTI_AGENT]` to the five roles. Remove the dedicated `analyst_model`, `test_generator_model`, and `reviewer_model` dataclass fields after migrating all callers to the existing `agent_models` dictionary. Keep deserialization of old manifest JSON tolerant because configuration is stored as untyped historical JSON.

- [ ] **Step 3: Relax and seed the database role constraint**

Because the original inline check has a generated PostgreSQL name, use a guarded migration block that discovers and drops the check constraint attached to `agent_setups.agent`, then adds a named `agent_setups_agent_check` covering legacy and new roles. Insert defaults for all five new roles with `ON CONFLICT DO NOTHING`.

- [ ] **Step 4: Update Streamlit settings**

Set `RUN_CONFIG_AGENTS` and `AGENT_LABELS` to the five active roles. For each role render provider, model, prompt, supported thinking level, and output-token cap. Reuse `_render_provider_model`, `_render_thinking_level`, and `_render_step_output_tokens`; do not add a second settings component.

Normal run creation must not show the critic-disable switch. It remains true. The comparison workspace in Plan 3 owns the named no-critic ablation.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_runner.py tests/test_app.py
rtk env PYTHONPATH=src TEST_DATABASE_URL=postgresql://brd_srs:brd_srs_local@127.0.0.1:5432/brd_srs_test .venv/bin/python -m pytest -q tests/test_storage.py
```

Expected: PASS.

```bash
rtk proxy git add src/brd_srs_testgen/runner.py src/brd_srs_testgen/schema.sql src/brd_srs_testgen/storage.py app.py tests/test_runner.py tests/test_storage.py tests/test_app.py
rtk proxy git commit -m "feat: configure hierarchical roles per run"
```

## Task 9: Show the blackboard and verify the complete pipeline

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app.py`
- Modify: `README.md`

- [ ] **Step 1: Add a failing blackboard UI test**

Assert the run detail page renders stages in deterministic order, includes role/model/task scope, and places raw typed output in an expander. A run without stage records must still render as a legacy run.

- [ ] **Step 2: Implement the stage timeline**

Reuse the existing activity/timeline components and `RUN_AGENT_LABELS`. Show Scout and Writer task indices, Curator decisions, Scenario Architect output, Critic findings, and whether repair occurred.

- [ ] **Step 3: Document the new architecture**

Update README architecture text and configuration examples. Explicitly state that role configuration is saved per run and changing defaults does not modify historical manifests.

- [ ] **Step 4: Run fresh verification**

Run:

```bash
rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_pipelines.py tests/test_runner.py tests/test_app.py
rtk env PYTHONPATH=src TEST_DATABASE_URL=postgresql://brd_srs:brd_srs_local@127.0.0.1:5432/brd_srs_test .venv/bin/python -m pytest -q tests/test_storage.py
rtk env PYTHONPATH=src .venv/bin/python -m pytest -q
rtk proxy git diff --check
```

Expected: all tests PASS and no whitespace errors.

- [ ] **Step 5: Commit**

```bash
rtk proxy git add app.py tests/test_app.py README.md
rtk proxy git commit -m "docs: expose hierarchical agent trace"
```

## Completion gate

- Evidence groups remain ordered and overlap only at boundaries.
- No code path truncates the final requirement or scenario catalog to a fixed global count.
- Curator decisions cover every candidate exactly once.
- The Scenario Architect sees the complete canonical catalog and covers every requirement.
- Test Writers only expand assigned canonical scenarios and collectively cover all of them.
- Critic runs by default; at most one substantive repair occurs.
- Each specialist's provider, model, prompt, thinking level, and output cap are configurable and snapshotted per run.
- Every handoff is persisted and inspectable.
