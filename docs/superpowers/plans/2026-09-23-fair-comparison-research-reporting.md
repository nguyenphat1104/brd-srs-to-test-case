# Fair Comparison and Research Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run and report a reproducible comparison in which generation strategy is the intended independent variable, evaluation is frozen before final runs, ratings are blinded, and uncertainty is computed at the document level.

**Architecture:** A study manifest freezes the model controls, document split, approved catalog IDs, arms, repetitions, and random seed. The runner parses each document once, randomizes arm/repetition order, and creates ordinary immutable runs. A private blind-ID mapping separates review from condition identity. A standard-library statistics module aggregates repetitions within document, then performs paired document-level bootstrap, randomization, and Wilcoxon analyses.

**Tech Stack:** Python 3.11 standard library (`csv`, `hashlib`, `itertools`, `json`, `math`, `random`, `statistics`), Pydantic, PostgreSQL/psycopg, Streamlit, pytest.

---

## Scope and invariants

- Complete Plans 1 and 2 first.
- This plan implements Sections 8–13 of `docs/superpowers/specs/2026-09-23-coverage-first-multi-agent-enhancement-design.md`.
- Primary confirmatory contrast: full hierarchical multi-agent minus staged single-agent.
- Required ablation: hierarchical multi-agent without Critic/repair.
- Single prompt remains a secondary descriptive baseline because it is already part of the assignment.
- All arms use the exact same provider, model, temperature `0.0`, thinking level, total generation token ceiling, output schemas, deterministic validators, and frozen evaluator.
- Each document-arm pair has exactly three repetitions.
- The unit of inference is the document, never an individual test case or repetition.
- Final holdout documents are not used to tune prompts, budgets, grouping, or repair policy.
- Confirmatory language is allowed only with at least 20 holdout documents and passed evaluator/rater agreement gates. Otherwise the report says exploratory.
- No SciPy, pandas, NumPy, or new statistics dependency is added.

## Task 1: Add study, split, arm, and run-link contracts

**Files:**
- Modify: `src/brd_srs_testgen/models.py`
- Modify: `tests/test_models.py`
- Modify: `tests/factories.py`

- [ ] **Step 1: Write failing manifest tests**

```python
def test_study_manifest_freezes_four_arms_and_three_repetitions():
    manifest = study_manifest()

    assert manifest.arms == [
        ComparisonArm.SINGLE_PROMPT,
        ComparisonArm.STAGED,
        ComparisonArm.MULTI_NO_CRITIC,
        ComparisonArm.MULTI_FULL,
    ]
    assert manifest.repetitions == 3
    assert manifest.temperature == 0.0


def test_confirmatory_study_requires_twenty_holdout_documents():
    with pytest.raises(ValueError, match="20 holdout"):
        study_manifest(analysis_mode="confirmatory", holdout_count=19)
```

Also test terminal timestamps/status, unique document hashes, approved catalog IDs, and unique `(document_hash, arm, repetition)` run links.

- [ ] **Step 2: Run and confirm the types are missing**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_models.py`

Expected: FAIL on missing study models.

- [ ] **Step 3: Add the contracts**

```python
class ComparisonArm(StrEnum):
    SINGLE_PROMPT = "single_prompt"
    STAGED = "staged"
    MULTI_NO_CRITIC = "multi_no_critic"
    MULTI_FULL = "multi_full"


class StudyPhase(StrEnum):
    DEVELOPMENT = "development"
    HOLDOUT = "holdout"


class AnalysisMode(StrEnum):
    EXPLORATORY = "exploratory"
    CONFIRMATORY = "confirmatory"


class StudyStatus(StrEnum):
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_FAILURES = "completed_with_failures"
    FAILED = "failed"


class BenchmarkDocument(StrictModel):
    document_id: str = Field(min_length=1)
    source_filename: str = Field(min_length=1)
    document_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_length: int = Field(gt=0)
    length_quartile: int = Field(ge=1, le=4)
    phase: StudyPhase
    catalog_id: str = Field(min_length=1)


class StudyManifest(StrictModel):
    study_id: str = Field(min_length=1)
    status: StudyStatus
    analysis_mode: AnalysisMode
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    temperature: float = Field(ge=0)
    thinking_level: str | None = None
    token_ceiling: int = Field(gt=0)
    arms: list[ComparisonArm] = Field(min_length=3)
    repetitions: int = Field(ge=3, le=3)
    seed: int
    documents: list[BenchmarkDocument] = Field(min_length=1)
    created_at: AwareDatetime
    completed_at: AwareDatetime | None = None


class StudyRunLink(StrictModel):
    study_id: str = Field(min_length=1)
    document_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    arm: ComparisonArm
    repetition: int = Field(ge=1, le=3)
    execution_order: int = Field(gt=0)
    run_id: str | None = None
```

Add model validators for status/timestamps, unique documents/arms, exactly three repetitions, and the 20-holdout confirmatory gate.

- [ ] **Step 4: Run tests and commit**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_models.py`

Expected: PASS.

```bash
rtk proxy git add src/brd_srs_testgen/models.py tests/test_models.py tests/factories.py
rtk proxy git commit -m "feat: model controlled comparison studies"
```

## Task 2: Freeze and validate fair generation settings

**Files:**
- Modify: `src/brd_srs_testgen/runner.py`
- Create: `src/brd_srs_testgen/studies.py`
- Create: `tests/test_studies.py`

- [ ] **Step 1: Add failing equality tests**

```python
def test_fair_settings_apply_one_model_to_every_arm_and_role():
    settings = FairGenerationSettings(
        provider="gemini",
        model="gemini-3.6-flash",
        api_key="secret",
        thinking_level="medium",
        token_ceiling=100_000,
    )

    generated = settings.provider_settings_for(ComparisonArm.MULTI_FULL)

    assert set(generated.agent_models.values()) == {"gemini-3.6-flash"}
    assert generated.token_ceiling == 100_000
    assert generated.thinking_level_for("critic") == "medium"


def test_fair_settings_reject_role_specific_model_override():
    with pytest.raises(ConfigurationError, match="same model"):
        FairGenerationSettings.from_provider_settings(mixed_role_settings())
```

Test that `MULTI_NO_CRITIC` differs from `MULTI_FULL` only by `critic_enabled=False`.

- [ ] **Step 2: Run and observe failure**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_studies.py`

Expected: FAIL because `studies.py` does not exist.

- [ ] **Step 3: Implement one frozen settings dataclass**

```python
@dataclass(frozen=True)
class FairGenerationSettings:
    provider: str
    model: str
    token_ceiling: int
    api_key: str = field(default="", repr=False)
    base_url: str = field(default="", repr=False)
    thinking_level: str | None = None

    def provider_settings_for(self, arm: ComparisonArm) -> ProviderSettings:
        run_type = {
            ComparisonArm.SINGLE_PROMPT: RunType.SINGLE_PROMPT,
            ComparisonArm.STAGED: RunType.STAGED_SINGLE_AGENT,
            ComparisonArm.MULTI_NO_CRITIC: RunType.CENTRALIZED_MULTI_AGENT,
            ComparisonArm.MULTI_FULL: RunType.CENTRALIZED_MULTI_AGENT,
        }[arm]
        agents = RUN_AGENTS[run_type]
        return ProviderSettings(
            provider=self.provider,
            model=self.model,
            token_ceiling=self.token_ceiling,
            api_key=self.api_key,
            base_url=self.base_url,
            agent_providers={agent: self.provider for agent in agents},
            agent_models={agent: self.model for agent in agents},
            agent_thinking_levels={agent: self.thinking_level for agent in agents if self.thinking_level},
        )
```

Temperature remains `0.0` in `run_generation`; assert it when building the study manifest. Prompts necessarily differ by strategy, but all output schemas and deterministic validators stay shared.

- [ ] **Step 4: Add `critic_enabled` to the run call and snapshot**

Pass it to `PipelineContext`; default true. Set false only for `MULTI_NO_CRITIC`. Add the value to `ProviderSettings.snapshot` for multi-agent runs.

- [ ] **Step 5: Run tests and commit**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_studies.py tests/test_runner.py`

Expected: PASS.

```bash
rtk proxy git add src/brd_srs_testgen/studies.py src/brd_srs_testgen/runner.py tests/test_studies.py tests/test_runner.py
rtk proxy git commit -m "feat: enforce fair study settings"
```

## Task 3: Create and persist a locked benchmark split

**Files:**
- Create: `scripts/create_benchmark_split.py`
- Create: `tests/test_benchmark_split.py`
- Modify: `src/brd_srs_testgen/studies.py`

- [ ] **Step 1: Add failing deterministic split tests**

Provide 40 documents with varied source lengths and assert:

- the same seed produces byte-for-byte identical JSON;
- every document appears once;
- each length quartile contributes approximately 30% to development and 70% to holdout;
- the result has 12 development and 28 holdout documents;
- 19 holdout documents force `analysis_mode="exploratory"`;
- split creation rejects duplicate hashes.

- [ ] **Step 2: Run and observe missing split logic**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_benchmark_split.py`

Expected: FAIL because the script/module functions do not exist.

- [ ] **Step 3: Implement quartiles with rank order**

Sort by `(source_length, document_hash)` and assign:

```python
quartile = min(4, (rank * 4 // document_count) + 1)
```

Group by quartile, shuffle each group with `random.Random(seed)`, assign `round(len(group) * 0.30)` to development, and place the rest in holdout. Sort final output by `document_id` so JSON order is stable.

- [ ] **Step 4: Add a CLI using only stdlib**

Input CSV columns:

```text
document_id,source_filename,document_hash,source_length,catalog_id
```

Command:

```bash
rtk env PYTHONPATH=src .venv/bin/python scripts/create_benchmark_split.py dataset.csv benchmark-split.json --seed 20260923
```

The JSON records the algorithm version, seed, generated timestamp, counts, analysis mode, and documents. The script refuses to overwrite an existing output path; a revised split gets a new filename.

- [ ] **Step 5: Run tests and commit**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_benchmark_split.py`

Expected: PASS.

```bash
rtk proxy git add scripts/create_benchmark_split.py src/brd_srs_testgen/studies.py tests/test_benchmark_split.py
rtk proxy git commit -m "feat: create locked stratified benchmark split"
```

## Task 4: Persist studies, documents, schedules, and blind IDs

**Files:**
- Modify: `src/brd_srs_testgen/schema.sql`
- Modify: `src/brd_srs_testgen/storage.py`
- Modify: `tests/test_storage.py`

- [ ] **Step 1: Add failing repository tests**

Test immutable round trips for a manifest, its documents, randomized schedule, and run links. Test that a study cannot start when a catalog is not approved. Test that blind IDs are unique and list methods for reviewers omit run type, run ID, arm, model configuration, and stage trace.

- [ ] **Step 2: Run storage tests**

Run: `rtk env PYTHONPATH=src TEST_DATABASE_URL=postgresql://brd_srs:brd_srs_local@127.0.0.1:5432/brd_srs_test .venv/bin/python -m pytest -q tests/test_storage.py`

Expected: FAIL on missing study repository methods.

- [ ] **Step 3: Add schema tables**

```sql
CREATE TABLE IF NOT EXISTS studies (
    study_id text PRIMARY KEY CHECK (study_id <> ''),
    status text NOT NULL CHECK (status IN ('draft', 'running', 'completed', 'completed_with_failures', 'failed')),
    analysis_mode text NOT NULL CHECK (analysis_mode IN ('exploratory', 'confirmatory')),
    provider text NOT NULL,
    model text NOT NULL,
    temperature double precision NOT NULL CHECK (temperature = 0),
    thinking_level text,
    token_ceiling integer NOT NULL CHECK (token_ceiling > 0),
    arms jsonb NOT NULL CHECK (jsonb_typeof(arms) = 'array'),
    repetitions integer NOT NULL CHECK (repetitions = 3),
    seed bigint NOT NULL,
    created_at timestamptz NOT NULL,
    completed_at timestamptz
);

CREATE TABLE IF NOT EXISTS study_documents (
    study_id text NOT NULL REFERENCES studies(study_id) ON DELETE CASCADE,
    document_id text NOT NULL,
    source_filename text NOT NULL,
    document_hash text NOT NULL CHECK (document_hash ~ '^[0-9a-f]{64}$'),
    source_length integer NOT NULL CHECK (source_length > 0),
    length_quartile smallint NOT NULL CHECK (length_quartile BETWEEN 1 AND 4),
    phase text NOT NULL CHECK (phase IN ('development', 'holdout')),
    catalog_id text NOT NULL REFERENCES coverage_catalogs(catalog_id),
    PRIMARY KEY (study_id, document_hash)
);

CREATE TABLE IF NOT EXISTS study_runs (
    study_id text NOT NULL REFERENCES studies(study_id) ON DELETE CASCADE,
    document_hash text NOT NULL,
    arm text NOT NULL CHECK (arm IN ('single_prompt', 'staged', 'multi_no_critic', 'multi_full')),
    repetition smallint NOT NULL CHECK (repetition BETWEEN 1 AND 3),
    execution_order integer NOT NULL CHECK (execution_order > 0),
    run_id text REFERENCES runs(run_id),
    PRIMARY KEY (study_id, document_hash, arm, repetition),
    UNIQUE (study_id, document_hash, execution_order)
);

CREATE TABLE IF NOT EXISTS blind_assignments (
    study_id text NOT NULL REFERENCES studies(study_id) ON DELETE CASCADE,
    blind_id text NOT NULL CHECK (blind_id <> ''),
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    display_order integer NOT NULL CHECK (display_order > 0),
    revealed_at timestamptz,
    PRIMARY KEY (study_id, blind_id),
    UNIQUE (study_id, run_id),
    UNIQUE (study_id, display_order)
);
```

- [ ] **Step 4: Implement repository operations**

Add `create_study`, `load_study`, `list_studies`, `link_study_run`, `finalize_study`, `create_blind_assignments`, `list_blind_review_items`, and `reveal_blind_assignments`. Use one transaction for manifest/documents/schedule creation. Before moving draft → running, query every linked catalog and require `status='approved'`.

- [ ] **Step 5: Run tests and commit**

Run: `rtk env PYTHONPATH=src TEST_DATABASE_URL=postgresql://brd_srs:brd_srs_local@127.0.0.1:5432/brd_srs_test .venv/bin/python -m pytest -q tests/test_storage.py`

Expected: PASS.

```bash
rtk proxy git add src/brd_srs_testgen/schema.sql src/brd_srs_testgen/storage.py tests/test_storage.py
rtk proxy git commit -m "feat: persist controlled studies and blinding"
```

## Task 5: Run randomized repetitions without reparsing documents

**Files:**
- Modify: `src/brd_srs_testgen/runner.py`
- Modify: `src/brd_srs_testgen/studies.py`
- Modify: `tests/test_runner.py`
- Modify: `tests/test_studies.py`

- [ ] **Step 1: Add failing schedule tests**

For one document, assert 12 cells (four arms × three repetitions), unique execution positions, deterministic order from the seed, and a different order from a different seed. Assert the stored order is used exactly.

- [ ] **Step 2: Add a failing parse-once integration test**

Monkeypatch PDF parsing, run all 12 cells, and assert parsing occurs once while every generated run stores the same chunks/document hash and uses its own run ID.

- [ ] **Step 3: Run and observe failures**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_studies.py tests/test_runner.py`

Expected: FAIL because `run_generation` owns PDF parsing and there is no study scheduler.

- [ ] **Step 4: Extract document preparation**

Add:

```python
@dataclass(frozen=True)
class PreparedDocument:
    source_filename: str
    document_hash: str
    chunks: tuple[DocumentChunk, ...]


def prepare_document(source_filename: str, pdf_bytes: bytes) -> PreparedDocument:
    document_hash = hashlib.sha256(pdf_bytes).hexdigest()
    return PreparedDocument(
        source_filename=source_filename,
        document_hash=document_hash,
        chunks=tuple(parse_pdf(pdf_bytes)),
    )
```

Refactor the existing public `run_generation` into a thin prepare-and-delegate wrapper plus `run_prepared_generation(prepared, run_type, settings, repository=repository, progress=progress)`. Preserve all existing single-run behavior and tests.

- [ ] **Step 5: Build and execute the schedule**

Create all `(arm, repetition)` cells, shuffle with `random.Random(document_seed)`, assign one-based execution order, persist before execution, then run sequentially. Map arms to run types and the critic flag through `FairGenerationSettings`.

Continue after an individual run's expected provider/validation failure, because `run_generation` already returns a failed immutable run. Finalize the study as `completed_with_failures`; statistics later use complete paired documents only.

- [ ] **Step 6: Run tests and commit**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_studies.py tests/test_runner.py`

Expected: PASS.

```bash
rtk proxy git add src/brd_srs_testgen/runner.py src/brd_srs_testgen/studies.py tests/test_runner.py tests/test_studies.py
rtk proxy git commit -m "feat: run randomized repeated study arms"
```

## Task 6: Implement document-level paired statistics

**Files:**
- Create: `src/brd_srs_testgen/statistics.py`
- Create: `tests/test_statistics.py`

- [ ] **Step 1: Add failing aggregation tests**

Assert three repetition scores are averaged within `(document, arm)` before differences are calculated. A document missing any repetition in either compared arm must be reported as incomplete and excluded from that pair, not averaged from fewer trials.

- [ ] **Step 2: Add failing bootstrap tests**

```python
def test_paired_bootstrap_resamples_documents_and_is_seeded():
    result = paired_bootstrap_ci([0.10, 0.20, -0.05], draws=10_000, seed=7)
    assert result.estimate == pytest.approx(0.0833333333)
    assert result.draws == 10_000
    assert result == paired_bootstrap_ci([0.10, 0.20, -0.05], draws=10_000, seed=7)
```

Test all-zero differences and one-document input. The report must label a one-document interval descriptive, not inferential.

- [ ] **Step 3: Add failing paired-randomization tests**

For `n <= 16`, enumerate every sign assignment with `itertools.product((-1, 1), repeat=n)` and verify a known exact two-sided p-value. For larger `n`, use 10,000 seeded sign flips and the Monte Carlo correction `(extreme + 1) / (draws + 1)`.

- [ ] **Step 4: Add failing Wilcoxon sensitivity tests**

Cover zeros, tied absolute differences, all-zero input, and a known small exact distribution. Drop zero differences, assign average ranks to ties, enumerate signs for `n <= 16`, and use a tie-corrected normal approximation for larger samples.

- [ ] **Step 5: Run and observe the missing module**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_statistics.py`

Expected: FAIL because `statistics.py` does not exist.

- [ ] **Step 6: Implement the stdlib functions**

Expose five typed public functions: `aggregate_repetitions(records: list[ScoreRecord]) -> AggregatedScores`, `paired_differences(scores: AggregatedScores, left: ComparisonArm, right: ComparisonArm) -> PairedDifferences`, `paired_bootstrap_ci(differences: list[float], *, draws: int = 10_000, seed: int) -> BootstrapResult`, `paired_randomization_test(differences: list[float], *, draws: int = 10_000, seed: int) -> TestResult`, and `wilcoxon_signed_rank(differences: list[float]) -> TestResult`.

Use `statistics.fmean`, `random.Random`, `itertools.product`, and `math.erfc`. Compute the percentile interval from sorted bootstrap means at indices `int(0.025 * draws)` and `int(0.975 * draws) - 1`. Document this percentile convention in the module docstring and report.

- [ ] **Step 7: Run tests and commit**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_statistics.py`

Expected: PASS.

```bash
rtk proxy git add src/brd_srs_testgen/statistics.py tests/test_statistics.py
rtk proxy git commit -m "feat: add paired document-level inference"
```

## Task 7: Enforce blinded review and agreement gates

**Files:**
- Modify: `src/brd_srs_testgen/studies.py`
- Modify: `src/brd_srs_testgen/storage.py`
- Modify: `app.py`
- Modify: `tests/test_studies.py`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Add failing blinding tests**

Assert review records contain only blind ID, source evidence, final artifacts, and rubric fields. They must omit run ID, run type, arm, execution order, provider/model, token counts, stage outputs, and agent labels.

- [ ] **Step 2: Add failing lock/reveal tests**

Assert mappings cannot be revealed until two distinct raters have saved all four dimensions for every sampled blind ID. Adjudication is allowed only after both initial ratings exist and differ. A rater cannot overwrite a prior rating.

- [ ] **Step 3: Add failing holdout gate tests**

Starting holdout execution must require:

- approved catalogs for all holdout documents;
- development/pilot human-human QWK ≥ 0.70 in every dimension;
- development/pilot Judge-human QWK ≥ 0.70 for coverage;
- no unresolved adjudications;
- at least 20 holdout documents for confirmatory mode.

Insufficient agreement blocks confirmatory execution with a precise message. Exploratory studies may continue only after an explicit `analysis_mode=exploratory` manifest is saved; never silently downgrade.

- [ ] **Step 4: Implement gate functions**

Add pure functions `review_complete`, `agreement_gate_status`, and `holdout_preflight`, then call them from repository state transitions and the UI. Keep the `0.70` threshold in one named constant imported from `evaluation.py`.

- [ ] **Step 5: Implement the blind review page**

Display one randomized blind item at a time. Render source excerpts and the final suite, not the agent trace. Require ratings for coverage, groundedness, executability, and redundancy control. Do not show aggregate scores or condition labels until the review lock passes.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_studies.py tests/test_app.py
rtk env PYTHONPATH=src TEST_DATABASE_URL=postgresql://brd_srs:brd_srs_local@127.0.0.1:5432/brd_srs_test .venv/bin/python -m pytest -q tests/test_storage.py
```

Expected: PASS.

```bash
rtk proxy git add src/brd_srs_testgen/studies.py src/brd_srs_testgen/storage.py app.py tests/test_studies.py tests/test_storage.py tests/test_app.py
rtk proxy git commit -m "feat: enforce blinded study review gates"
```

## Task 8: Build the analysis report and export

**Files:**
- Create: `src/brd_srs_testgen/reporting.py`
- Create: `tests/test_reporting.py`
- Modify: `app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Add a failing report snapshot test**

The report must contain:

- study ID, analysis mode, frozen settings, split algorithm/seed, arms, repetitions, and evaluator version;
- completed/failed/incomplete counts by arm;
- per-arm document-level mean F1 and four human dimensions;
- primary full-multi minus staged paired estimate, 95% bootstrap CI, paired-randomization p-value, and Wilcoxon sensitivity result;
- no-critic ablation contrast and single-prompt descriptive baseline;
- human-human QWK by dimension and Judge-human QWK for coverage;
- explicit limitations and the rule used to exclude incomplete pairs;
- `EXPLORATORY` in the title when the holdout has fewer than 20 documents or a gate was not met.

- [ ] **Step 2: Run and observe the missing report module**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_reporting.py`

Expected: FAIL because `reporting.py` does not exist.

- [ ] **Step 3: Implement one report data builder and Markdown renderer**

Expose `build_study_report(repository: RunRepository, study_id: str) -> StudyReport` and `render_study_markdown(report: StudyReport) -> str`. Keep calculations in `statistics.py`; `reporting.py` only assembles validated records and renders them. Never recompute from values rounded for display.

- [ ] **Step 4: Add machine-readable exports**

Export:

1. `study-report.md` for the assignment;
2. `study-summary.json` containing estimates, intervals, p-values, seeds, exclusions, and versions;
3. `document-level-scores.csv` with one row per document/arm after repetition aggregation;
4. `blind-key.csv` only after reveal, as a separate private export.

Use `csv` and `json`; do not add a dataframe dependency.

- [ ] **Step 5: Render the report in Streamlit**

Show the primary contrast first, then uncertainty, ablation, agreement, completion/exclusions, and configuration. A table is sufficient; do not add chart dependencies. Provide separate download buttons for public report files and the private blind key.

- [ ] **Step 6: Run tests and commit**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_reporting.py tests/test_app.py tests/test_statistics.py`

Expected: PASS.

```bash
rtk proxy git add src/brd_srs_testgen/reporting.py tests/test_reporting.py app.py tests/test_app.py
rtk proxy git commit -m "feat: report controlled multi-agent study"
```

## Task 9: Document the protocol and run the final verification gate

**Files:**
- Modify: `README.md`
- Modify: `docs/research-methodology.md`
- Modify: `docs/research-core-operations.md`

- [ ] **Step 1: Document the pre-registered decision rules**

Before holdout execution, record in `docs/research-methodology.md`:

- primary contrast and F1 definition;
- four arms and why the no-critic arm is an ablation;
- three repetitions and document-level aggregation;
- 30/70 length-quartile split and seed;
- 10,000 bootstrap draws and 10,000 Monte Carlo sign flips when exact enumeration is not used;
- paired-randomization primary test and Wilcoxon sensitivity test;
- two raters, four dimensions, quadratic-weighted Cohen kappa, and 0.70 operational gate;
- missing/failure handling and confirmatory sample-size rule;
- no tuning on holdout outputs.

- [ ] **Step 2: Preserve public research citations**

Carry the primary-source citations from the approved design into the methodology document for Cohen's kappa/weighted agreement, bootstrap resampling, randomization tests, and Wilcoxon signed-rank testing. Distinguish established methods from project-specific choices such as the 0.70 threshold and 20-document minimum.

- [ ] **Step 3: Run fresh verification**

Run:

```bash
rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_models.py tests/test_benchmark_split.py tests/test_studies.py tests/test_statistics.py tests/test_reporting.py tests/test_runner.py tests/test_app.py
rtk env PYTHONPATH=src TEST_DATABASE_URL=postgresql://brd_srs:brd_srs_local@127.0.0.1:5432/brd_srs_test .venv/bin/python -m pytest -q tests/test_storage.py
rtk env PYTHONPATH=src .venv/bin/python -m pytest -q
rtk proxy git diff --check
```

Expected: all tests PASS and no whitespace errors.

- [ ] **Step 4: Perform one deterministic dry run**

Using test providers, create a two-document exploratory study and verify:

- schedules are identical when rerun from the same seed;
- each document has 12 linked runs;
- all linked runs use one catalog per document;
- arm labels stay hidden before review completion;
- the report excludes a deliberately incomplete document from paired inference and names it in exclusions.

- [ ] **Step 5: Commit documentation**

```bash
rtk proxy git add README.md docs/research-methodology.md docs/research-core-operations.md
rtk proxy git commit -m "docs: publish controlled comparison protocol"
```

## Completion gate

- Study setup freezes the same provider/model/temperature/thinking/ceiling across arms.
- All catalog IDs are approved before final execution.
- The 30/70 split and randomized three-repeat schedule are reproducible from stored seeds.
- Reviewers cannot see condition identity before both independent ratings are locked.
- Human-human and Judge-human agreement are reported with quadratic-weighted Cohen kappa.
- Repetitions are averaged within document before paired inference.
- The primary report contains a document-resampled 95% bootstrap interval, paired-randomization p-value, and Wilcoxon sensitivity result.
- The report labels an undersized or ungated study exploratory and never claims the multi-agent architecture is best unless the observed evidence supports that conclusion.
