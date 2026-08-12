# Single-Run PostgreSQL History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run exactly one selected generation strategy, persist its complete normalized result in local PostgreSQL, and reopen detailed test cases from run history.

**Architecture:** Replace the comparison aggregate with `RunManifest`/`RunResult`, then route one selected `RunType` to the existing pipeline map. A `RunRepository` backed by psycopg owns an idempotent normalized schema, transactional finalization, history summaries, and result reconstruction. Streamlit reuses one result renderer for the current run and historical runs.

**Tech Stack:** Python 3.11, Pydantic 2, psycopg 3, PostgreSQL 17 via Docker Compose, Streamlit, pytest

---

## File map

| File | Responsibility |
|---|---|
| `compose.yaml` | Local PostgreSQL service, health check, loopback port, durable volume. |
| `docker/postgres/init-test-db.sql` | Create a separate integration-test database on first container initialization. |
| `.env.example` | Document local application and test database URLs without secrets. |
| `requirements.txt` | Add the installed PostgreSQL driver. |
| `src/brd_srs_testgen/models.py` | Define `RunType`, `RunManifest`, `RunResult`, and `RunHistoryItem`; remove comparison models. |
| `src/brd_srs_testgen/schema.sql` | Fully normalized PostgreSQL schema. |
| `src/brd_srs_testgen/storage.py` | Initialize schema; create, progress, finalize, list, and load runs. |
| `src/brd_srs_testgen/runner.py` | Execute only the selected pipeline and persist its lifecycle. |
| `app.py` | Configure a run type, render one detailed result, initialize the repository, and browse history. |
| `tests/conftest.py` | Isolate PostgreSQL integration tests by truncating only the test database. |
| `tests/factories.py` | Build reusable run manifests/results for repository and UI tests. |
| `tests/test_models.py` | Validate single-run lifecycle invariants. |
| `tests/test_storage.py` | Verify normalized round-trip, terminal immutability, and history summaries. |
| `tests/test_runner.py` | Verify one selected pipeline, lifecycle persistence, and existing failure behavior. |
| `tests/test_app.py` | Verify run selection, detailed artifacts, history loading, and database failure UI. |
| `README.md` | Add the short local startup path. |
| `docs/research-core-operations.md` | Replace filesystem/comparison operations with PostgreSQL/single-run operations. |

Do not modify or delete the ignored legacy `runs/` directory. The new application neither reads nor writes it.

### Task 1: Add local PostgreSQL runtime

**Files:**
- Create: `compose.yaml`
- Create: `docker/postgres/init-test-db.sql`
- Create: `.env.example`
- Modify: `requirements.txt`

- [ ] **Step 1: Add the PostgreSQL driver**

Append this runtime dependency to `requirements.txt`:

```text
psycopg[binary]>=3.2,<4
```

- [ ] **Step 2: Define the local database service**

Create `compose.yaml`:

```yaml
services:
  db:
    image: postgres:17.6-alpine
    environment:
      POSTGRES_DB: brd_srs
      POSTGRES_USER: brd_srs
      POSTGRES_PASSWORD: brd_srs_local
    ports:
      - "127.0.0.1:5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/postgres/init-test-db.sql:/docker-entrypoint-initdb.d/10-test-db.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U brd_srs -d brd_srs"]
      interval: 2s
      timeout: 3s
      retries: 15

volumes:
  postgres_data:
```

Create `docker/postgres/init-test-db.sql`:

```sql
CREATE DATABASE brd_srs_test OWNER brd_srs;
```

Create `.env.example`:

```dotenv
DATABASE_URL=postgresql://brd_srs:brd_srs_local@127.0.0.1:5432/brd_srs
TEST_DATABASE_URL=postgresql://brd_srs:brd_srs_local@127.0.0.1:5432/brd_srs_test
GEMINI_API_KEY=
LM_STUDIO_API_TOKEN=
```

- [ ] **Step 3: Validate Compose and install the driver**

Run:

```bash
rtk docker compose config -q
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -c "import psycopg; print(psycopg.__version__)"
```

Expected: Compose exits 0, dependency installation succeeds, and psycopg prints a `3.x` version.

- [ ] **Step 4: Start and verify both databases**

Run:

```bash
rtk docker compose up -d db
rtk docker compose exec -T db pg_isready -U brd_srs -d brd_srs
rtk docker compose exec -T db pg_isready -U brd_srs -d brd_srs_test
```

Expected: both `pg_isready` commands report `accepting connections`.

- [ ] **Step 5: Commit the runtime setup**

```bash
rtk git add compose.yaml docker/postgres/init-test-db.sql .env.example requirements.txt
rtk git commit -m "build: add local PostgreSQL runtime"
```

### Task 2: Replace comparison domain models with one run

**Files:**
- Modify: `src/brd_srs_testgen/models.py`
- Modify: `src/brd_srs_testgen/__init__.py`
- Modify: `tests/test_models.py`
- Modify: `tests/factories.py`

- [ ] **Step 1: Replace comparison-model tests with run lifecycle tests**

In `tests/test_models.py`, replace `ConditionManifest` and `ComparisonManifest` tests with:

```python
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from brd_srs_testgen.models import (
    FailureCategory,
    RunManifest,
    RunStatus,
    RunType,
)


def run_manifest(**overrides: object) -> RunManifest:
    now = datetime.now(UTC)
    values = {
        "run_id": "20260812T120000000000Z-ecac9f035813-12345678",
        "source_filename": "sample.pdf",
        "document_hash": "a" * 64,
        "run_type": RunType.SINGLE_PROMPT,
        "status": RunStatus.RUNNING,
        "provider": "ollama",
        "model": "gemma4",
        "temperature": 0.0,
        "token_ceiling": 100_000,
        "prompt_version": "research-core-v1",
        "schema_version": "research-core-v1",
        "started_at": now,
    }
    values.update(overrides)
    return RunManifest(**values)


def test_running_run_has_no_terminal_fields() -> None:
    manifest = run_manifest()
    assert manifest.run_type is RunType.SINGLE_PROMPT
    assert manifest.completed_at is None


def test_completed_run_requires_completion_time() -> None:
    with pytest.raises(ValidationError, match="completed runs require completed_at"):
        run_manifest(status=RunStatus.COMPLETED)


def test_failed_run_requires_category_and_completion_time() -> None:
    with pytest.raises(ValidationError, match="failed runs require"):
        run_manifest(status=RunStatus.FAILED)


def test_failed_run_accepts_safe_failure_details() -> None:
    now = datetime.now(UTC)
    manifest = run_manifest(
        status=RunStatus.FAILED,
        completed_at=now,
        failure_category=FailureCategory.PARSING,
        failure_message="PDF contains insufficient extractable text.",
    )
    assert manifest.failure_category is FailureCategory.PARSING
```

- [ ] **Step 2: Run the model tests to verify they fail**

Run:

```bash
env PYTHONPATH=src .venv/bin/python -m pytest tests/test_models.py -q
```

Expected: FAIL because `RunType` and `RunManifest` do not exist.

- [ ] **Step 3: Define the single-run models**

In `src/brd_srs_testgen/models.py`:

1. Rename `Condition` to `RunType` without changing its three values.
2. Delete `ConditionManifest` and `ComparisonManifest`.
3. Add these models after `RunMetrics`:

```python
class RunManifest(StrictModel):
    run_id: str = Field(min_length=1)
    source_filename: str = Field(min_length=1)
    document_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_type: RunType
    status: RunStatus
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    temperature: float = Field(ge=0)
    token_ceiling: int = Field(ge=1)
    prompt_version: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    started_at: AwareDatetime
    completed_at: AwareDatetime | None = None
    failure_category: FailureCategory | None = None
    failure_message: str | None = None

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at cannot be earlier than started_at")
        if self.status is RunStatus.COMPLETED:
            if self.completed_at is None:
                raise ValueError("completed runs require completed_at")
            if self.failure_category is not None or self.failure_message is not None:
                raise ValueError("completed runs cannot have failure details")
        elif self.status is RunStatus.FAILED:
            if self.completed_at is None or self.failure_category is None:
                raise ValueError("failed runs require completed_at and failure_category")
        elif any(
            value is not None
            for value in (self.completed_at, self.failure_category, self.failure_message)
        ):
            raise ValueError("running runs cannot have terminal fields")
        return self


class RunResult(StrictModel):
    manifest: RunManifest
    bundle: ArtifactBundle | None = None
    validation: ValidationReport | None = None
    rtm: list[RTMRow] = Field(default_factory=list)
    metrics: RunMetrics | None = None

    def download_bundle(self) -> dict[str, JsonValue]:
        bundle = self.bundle
        return {
            "manifest": self.manifest.model_dump(mode="json"),
            "requirements": (
                [item.model_dump(mode="json") for item in bundle.requirements]
                if bundle else []
            ),
            "scenarios": (
                [item.model_dump(mode="json") for item in bundle.scenarios]
                if bundle else []
            ),
            "test_cases": (
                [item.model_dump(mode="json") for item in bundle.test_cases]
                if bundle else []
            ),
            "validation": (
                self.validation.model_dump(mode="json") if self.validation else None
            ),
            "rtm": [item.model_dump(mode="json") for item in self.rtm],
            "metrics": self.metrics.model_dump(mode="json") if self.metrics else None,
        }


class RunHistoryItem(StrictModel):
    run_id: str
    source_filename: str
    run_type: RunType
    status: RunStatus
    provider: str
    model: str
    started_at: AwareDatetime
    completed_at: AwareDatetime | None
    requirement_count: int | None = None
    scenario_count: int | None = None
    test_case_count: int | None = None

    @property
    def display_status(self) -> str:
        return "Interrupted" if self.status is RunStatus.RUNNING else self.status.value.title()
```

Keep the existing artifact and metric models unchanged. Update `src/brd_srs_testgen/__init__.py` to export `RunType` instead of `Condition`:

```python
from .models import ArtifactBundle, RunType

__version__ = "0.1.0"
__all__ = ["ArtifactBundle", "RunType"]
```

- [ ] **Step 4: Add reusable run-result factories**

Extend `tests/factories.py`:

```python
from datetime import UTC, datetime

from brd_srs_testgen.models import (
    RunManifest,
    RunResult,
    RunStatus,
    RunType,
)
from brd_srs_testgen.validation import build_rtm, compute_metrics, validate_bundle


def completed_run(
    run_id: str = "20260812T120000000000Z-ecac9f035813-12345678",
    run_type: RunType = RunType.SINGLE_PROMPT,
) -> RunResult:
    now = datetime.now(UTC)
    artifacts = bundle()
    evidence = [chunk()]
    validation = validate_bundle(artifacts, evidence)
    metrics = compute_metrics(
        artifacts,
        validation,
        input_tokens=10,
        output_tokens=20,
        charged_tokens=30,
        latency_seconds=0.1,
        retries=0,
        schema_repairs=0,
        semantic_revisions=0,
        budget_exhausted=False,
    )
    return RunResult(
        manifest=RunManifest(
            run_id=run_id,
            source_filename="sample.pdf",
            document_hash="a" * 64,
            run_type=run_type,
            status=RunStatus.COMPLETED,
            provider="ollama",
            model="gemma4",
            temperature=0,
            token_ceiling=100_000,
            prompt_version="research-core-v1",
            schema_version="research-core-v1",
            started_at=now,
            completed_at=now,
        ),
        bundle=artifacts,
        validation=validation,
        rtm=build_rtm(artifacts),
        metrics=metrics,
    )
```

- [ ] **Step 5: Run the model tests**

Run:

```bash
env PYTHONPATH=src .venv/bin/python -m pytest tests/test_models.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the domain change**

```bash
rtk git add src/brd_srs_testgen/models.py src/brd_srs_testgen/__init__.py tests/test_models.py tests/factories.py
rtk git commit -m "refactor: model independent generation runs"
```

### Task 3: Create the normalized PostgreSQL schema

**Files:**
- Create: `src/brd_srs_testgen/schema.sql`
- Create: `tests/conftest.py`
- Replace: `tests/test_storage.py`
- Replace: `src/brd_srs_testgen/storage.py`

- [ ] **Step 1: Add an isolated integration-test repository fixture**

Create `tests/conftest.py`:

```python
import os

import psycopg
import pytest

from brd_srs_testgen.storage import RunRepository


@pytest.fixture
def repository() -> RunRepository:
    database_url = os.environ.get("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL integration tests")
    if database_url.rsplit("/", 1)[-1] != "brd_srs_test":
        pytest.fail("TEST_DATABASE_URL must target brd_srs_test")
    repo = RunRepository(database_url)
    repo.initialize()
    with psycopg.connect(database_url) as connection:
        connection.execute("TRUNCATE TABLE runs CASCADE")
    yield repo
    with psycopg.connect(database_url) as connection:
        connection.execute("TRUNCATE TABLE runs CASCADE")
```

- [ ] **Step 2: Write failing lifecycle and history tests**

Replace `tests/test_storage.py` with focused PostgreSQL tests. Start with:

```python
from datetime import UTC, datetime, timedelta

import psycopg
import pytest

from brd_srs_testgen.models import RunManifest, RunStatus, RunType
from brd_srs_testgen.storage import ImmutableRunError, RunRepository
from tests.factories import chunk


def running_manifest(run_id: str, *, started_at: datetime | None = None) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        source_filename="sample.pdf",
        document_hash="a" * 64,
        run_type=RunType.SINGLE_PROMPT,
        status=RunStatus.RUNNING,
        provider="ollama",
        model="gemma4",
        temperature=0,
        token_ceiling=100_000,
        prompt_version="research-core-v1",
        schema_version="research-core-v1",
        started_at=started_at or datetime.now(UTC),
    )


def test_create_run_retains_an_interrupted_history_item(repository: RunRepository) -> None:
    manifest = running_manifest("run-interrupted")
    repository.create_run(manifest)
    repository.append_event(manifest.run_id, "started")

    history = repository.list_runs()

    assert [item.run_id for item in history] == [manifest.run_id]
    assert history[0].display_status == "Interrupted"


def test_history_is_newest_first(repository: RunRepository) -> None:
    now = datetime.now(UTC)
    repository.create_run(running_manifest("old", started_at=now - timedelta(minutes=1)))
    repository.create_run(running_manifest("new", started_at=now))

    assert [item.run_id for item in repository.list_runs()] == ["new", "old"]


def test_chunks_and_events_require_a_running_run(repository: RunRepository) -> None:
    manifest = running_manifest("run-active")
    repository.create_run(manifest)
    repository.save_chunks(manifest.run_id, [chunk()])
    repository.append_event(manifest.run_id, "parsed")

    assert repository.load_chunks(manifest.run_id) == [chunk()]
    assert [event["stage"] for event in repository.load_events(manifest.run_id)] == [
        "parsed"
    ]


def test_duplicate_run_id_is_rejected(repository: RunRepository) -> None:
    manifest = running_manifest("same-id")
    repository.create_run(manifest)
    with pytest.raises(ImmutableRunError, match="already exists"):
        repository.create_run(manifest)


def test_schema_has_no_pdf_or_credential_columns(repository: RunRepository) -> None:
    with psycopg.connect(repository.database_url) as connection:
        columns = {
            row[0]
            for row in connection.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                """
            )
        }
    assert columns.isdisjoint({"pdf", "pdf_bytes", "api_key", "credential", "token"})
```

- [ ] **Step 3: Run the storage tests to verify they fail**

Run:

```bash
export TEST_DATABASE_URL=postgresql://brd_srs:brd_srs_local@127.0.0.1:5432/brd_srs_test
env PYTHONPATH=src .venv/bin/python -m pytest tests/test_storage.py -q
```

Expected: FAIL because `RunRepository` does not exist.

- [ ] **Step 4: Add the complete schema**

Create `src/brd_srs_testgen/schema.sql`. Use `text` plus check constraints for application enums, `timestamptz` for timestamps, composite run-scoped keys, and `ON DELETE CASCADE` only to the owning run/artifact. Do not add target foreign keys for generated dependency, requirement, scenario, or chunk IDs because invalid generated output must still be stored.

The schema must create these tables in this order:

```sql
CREATE TABLE IF NOT EXISTS runs (
    run_id text PRIMARY KEY,
    source_filename text NOT NULL CHECK (source_filename <> ''),
    document_hash text NOT NULL CHECK (document_hash ~ '^[0-9a-f]{64}$'),
    run_type text NOT NULL CHECK (run_type IN (
        'single_prompt', 'staged_single_agent', 'centralized_multi_agent'
    )),
    status text NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    provider text NOT NULL CHECK (provider <> ''),
    model text NOT NULL CHECK (model <> ''),
    temperature double precision NOT NULL CHECK (temperature >= 0),
    token_ceiling integer NOT NULL CHECK (token_ceiling > 0),
    prompt_version text NOT NULL,
    schema_version text NOT NULL,
    started_at timestamptz NOT NULL,
    completed_at timestamptz,
    failure_category text CHECK (failure_category IN (
        'parsing', 'configuration', 'provider_rejection',
        'transport_exhaustion', 'timeout', 'budget_exhaustion',
        'schema_failure', 'semantic_validation'
    )),
    failure_message text,
    CHECK (
        (status = 'running' AND completed_at IS NULL
            AND failure_category IS NULL AND failure_message IS NULL)
        OR (status = 'completed' AND completed_at IS NOT NULL
            AND failure_category IS NULL AND failure_message IS NULL)
        OR (status = 'failed' AND completed_at IS NOT NULL
            AND failure_category IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS runs_started_at_idx ON runs (started_at DESC);

CREATE TABLE IF NOT EXISTS document_chunks (
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    chunk_id text NOT NULL,
    page_number integer NOT NULL CHECK (page_number > 0),
    section text NOT NULL,
    text text NOT NULL CHECK (text <> ''),
    content_hash text NOT NULL CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    PRIMARY KEY (run_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS run_events (
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    sequence integer NOT NULL CHECK (sequence > 0),
    occurred_at timestamptz NOT NULL,
    stage text NOT NULL CHECK (stage <> ''),
    PRIMARY KEY (run_id, sequence)
);

CREATE TABLE IF NOT EXISTS run_metrics (
    run_id text PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    completion boolean NOT NULL,
    schema_valid boolean NOT NULL,
    citation_coverage double precision NOT NULL CHECK (citation_coverage BETWEEN 0 AND 1),
    requirement_scenario_coverage double precision NOT NULL CHECK (requirement_scenario_coverage BETWEEN 0 AND 1),
    requirement_test_case_coverage double precision NOT NULL CHECK (requirement_test_case_coverage BETWEEN 0 AND 1),
    positive_scenario_coverage double precision NOT NULL CHECK (positive_scenario_coverage BETWEEN 0 AND 1),
    non_positive_scenario_coverage double precision NOT NULL CHECK (non_positive_scenario_coverage BETWEEN 0 AND 1),
    rtm_completeness double precision NOT NULL CHECK (rtm_completeness BETWEEN 0 AND 1),
    orphan_rate double precision NOT NULL CHECK (orphan_rate BETWEEN 0 AND 1),
    invalid_reference_rate double precision NOT NULL CHECK (invalid_reference_rate BETWEEN 0 AND 1),
    duplicate_test_case_rate double precision NOT NULL CHECK (duplicate_test_case_rate BETWEEN 0 AND 1),
    requirement_count integer NOT NULL CHECK (requirement_count >= 0),
    scenario_count integer NOT NULL CHECK (scenario_count >= 0),
    test_case_count integer NOT NULL CHECK (test_case_count >= 0),
    input_tokens integer NOT NULL CHECK (input_tokens >= 0),
    output_tokens integer NOT NULL CHECK (output_tokens >= 0),
    charged_tokens integer NOT NULL CHECK (charged_tokens >= 0),
    latency_seconds double precision NOT NULL CHECK (latency_seconds >= 0),
    retries integer NOT NULL CHECK (retries >= 0),
    schema_repairs integer NOT NULL CHECK (schema_repairs >= 0),
    semantic_revisions integer NOT NULL CHECK (semantic_revisions >= 0),
    budget_exhausted boolean NOT NULL
);

CREATE TABLE IF NOT EXISTS requirements (
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    requirement_id text NOT NULL,
    title text NOT NULL,
    description text NOT NULL,
    requirement_type text NOT NULL CHECK (requirement_type IN ('functional', 'non_functional', 'business')),
    module text NOT NULL,
    priority text NOT NULL CHECK (priority IN ('high', 'medium', 'low')),
    PRIMARY KEY (run_id, requirement_id)
);

CREATE TABLE IF NOT EXISTS requirement_ambiguities (
    run_id text NOT NULL,
    requirement_id text NOT NULL,
    position integer NOT NULL CHECK (position > 0),
    value text NOT NULL,
    PRIMARY KEY (run_id, requirement_id, position),
    FOREIGN KEY (run_id, requirement_id)
        REFERENCES requirements(run_id, requirement_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS requirement_dependencies (
    run_id text NOT NULL,
    requirement_id text NOT NULL,
    position integer NOT NULL CHECK (position > 0),
    dependency_id text NOT NULL,
    PRIMARY KEY (run_id, requirement_id, position),
    FOREIGN KEY (run_id, requirement_id)
        REFERENCES requirements(run_id, requirement_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS requirement_sources (
    run_id text NOT NULL,
    requirement_id text NOT NULL,
    position integer NOT NULL CHECK (position > 0),
    chunk_id text NOT NULL,
    page_number integer NOT NULL CHECK (page_number > 0),
    section text NOT NULL,
    excerpt text NOT NULL,
    PRIMARY KEY (run_id, requirement_id, position),
    FOREIGN KEY (run_id, requirement_id)
        REFERENCES requirements(run_id, requirement_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scenarios (
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    scenario_id text NOT NULL,
    title text NOT NULL,
    objective text NOT NULL,
    scenario_type text NOT NULL CHECK (scenario_type IN (
        'positive', 'negative', 'boundary', 'edge', 'state_transition'
    )),
    PRIMARY KEY (run_id, scenario_id)
);

CREATE TABLE IF NOT EXISTS scenario_preconditions (
    run_id text NOT NULL,
    scenario_id text NOT NULL,
    position integer NOT NULL CHECK (position > 0),
    value text NOT NULL,
    PRIMARY KEY (run_id, scenario_id, position),
    FOREIGN KEY (run_id, scenario_id)
        REFERENCES scenarios(run_id, scenario_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scenario_requirements (
    run_id text NOT NULL,
    scenario_id text NOT NULL,
    position integer NOT NULL CHECK (position > 0),
    requirement_id text NOT NULL,
    PRIMARY KEY (run_id, scenario_id, position),
    FOREIGN KEY (run_id, scenario_id)
        REFERENCES scenarios(run_id, scenario_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scenario_sources (
    run_id text NOT NULL,
    scenario_id text NOT NULL,
    position integer NOT NULL CHECK (position > 0),
    chunk_id text NOT NULL,
    page_number integer NOT NULL CHECK (page_number > 0),
    section text NOT NULL,
    excerpt text NOT NULL,
    PRIMARY KEY (run_id, scenario_id, position),
    FOREIGN KEY (run_id, scenario_id)
        REFERENCES scenarios(run_id, scenario_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS test_cases (
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    test_case_id text NOT NULL,
    scenario_id text NOT NULL,
    title text NOT NULL,
    priority text NOT NULL CHECK (priority IN ('P1', 'P2', 'P3')),
    PRIMARY KEY (run_id, test_case_id)
);

CREATE TABLE IF NOT EXISTS test_case_preconditions (
    run_id text NOT NULL,
    test_case_id text NOT NULL,
    position integer NOT NULL CHECK (position > 0),
    value text NOT NULL,
    PRIMARY KEY (run_id, test_case_id, position),
    FOREIGN KEY (run_id, test_case_id)
        REFERENCES test_cases(run_id, test_case_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS test_case_requirements (
    run_id text NOT NULL,
    test_case_id text NOT NULL,
    position integer NOT NULL CHECK (position > 0),
    requirement_id text NOT NULL,
    PRIMARY KEY (run_id, test_case_id, position),
    FOREIGN KEY (run_id, test_case_id)
        REFERENCES test_cases(run_id, test_case_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS test_case_data (
    run_id text NOT NULL,
    test_case_id text NOT NULL,
    key text NOT NULL,
    value jsonb NOT NULL,
    PRIMARY KEY (run_id, test_case_id, key),
    FOREIGN KEY (run_id, test_case_id)
        REFERENCES test_cases(run_id, test_case_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS test_steps (
    run_id text NOT NULL,
    test_case_id text NOT NULL,
    step_number integer NOT NULL CHECK (step_number > 0),
    action text NOT NULL,
    expected_result text NOT NULL,
    PRIMARY KEY (run_id, test_case_id, step_number),
    FOREIGN KEY (run_id, test_case_id)
        REFERENCES test_cases(run_id, test_case_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS test_case_sources (
    run_id text NOT NULL,
    test_case_id text NOT NULL,
    position integer NOT NULL CHECK (position > 0),
    chunk_id text NOT NULL,
    page_number integer NOT NULL CHECK (page_number > 0),
    section text NOT NULL,
    excerpt text NOT NULL,
    PRIMARY KEY (run_id, test_case_id, position),
    FOREIGN KEY (run_id, test_case_id)
        REFERENCES test_cases(run_id, test_case_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS validation_reports (
    run_id text PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    valid boolean NOT NULL
);

CREATE TABLE IF NOT EXISTS validation_issues (
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    position integer NOT NULL CHECK (position > 0),
    code text NOT NULL,
    artifact_id text NOT NULL,
    message text NOT NULL,
    PRIMARY KEY (run_id, position)
);

CREATE TABLE IF NOT EXISTS validation_uncovered_requirements (
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    position integer NOT NULL CHECK (position > 0),
    requirement_id text NOT NULL,
    PRIMARY KEY (run_id, position)
);

CREATE TABLE IF NOT EXISTS validation_orphan_scenarios (
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    position integer NOT NULL CHECK (position > 0),
    scenario_id text NOT NULL,
    PRIMARY KEY (run_id, position)
);

CREATE TABLE IF NOT EXISTS validation_orphan_test_cases (
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    position integer NOT NULL CHECK (position > 0),
    test_case_id text NOT NULL,
    PRIMARY KEY (run_id, position)
);
```

- [ ] **Step 5: Implement repository initialization and lifecycle operations**

Replace filesystem persistence in `src/brd_srs_testgen/storage.py` with `RunRepository`. Use `psycopg.rows.dict_row`, read `schema.sql` relative to the module, and define `StorageError`, `ImmutableRunError`, and a `RunRepository` whose constructor rejects a blank URL with `StorageError("DATABASE_URL is required.")`.

The public lifecycle methods and return types are:

| Method | Return | Contract |
|---|---|---|
| `initialize()` | `None` | Execute `schema.sql`. |
| `create_run(manifest)` | `None` | Insert one running manifest. |
| `save_chunks(run_id, chunks)` | `None` | Insert evidence once while running. |
| `load_chunks(run_id)` | `list[DocumentChunk]` | Return page/chunk order. |
| `append_event(run_id, stage, occurred_at=None)` | `None` | Append the next run-local sequence. |
| `load_events(run_id)` | `list[dict[str, object]]` | Return sequence order. |
| `list_runs()` | `list[RunHistoryItem]` | Return newest-first summaries. |

Implementation rules:

- `initialize()` executes the complete schema file in one connection.
- `create_run()` accepts only `RUNNING`, inserts every manifest field, and converts unique violations to `ImmutableRunError("Run already exists.")`.
- `save_chunks()` locks the run row with `SELECT status FROM runs WHERE run_id = %s FOR UPDATE`, requires `running`, rejects a second chunk write, and inserts every chunk with `executemany`.
- `append_event()` locks the run row, requires `running`, and inserts `COALESCE(MAX(sequence), 0) + 1` for that run.
- `list_runs()` left joins `run_metrics`, orders by `started_at DESC, run_id DESC`, and builds `RunHistoryItem` values.
- Convert connection/schema errors to `StorageError` while preserving the original exception as `__cause__`.

Use this shared guard inside write transactions:

```python
def _require_running(connection, run_id: str) -> None:
    row = connection.execute(
        "SELECT status FROM runs WHERE run_id = %s FOR UPDATE", (run_id,)
    ).fetchone()
    if row is None:
        raise StorageError("Run does not exist.")
    if row["status"] != RunStatus.RUNNING.value:
        raise ImmutableRunError("Terminal runs are immutable.")
```

- [ ] **Step 6: Run lifecycle tests**

Run:

```bash
export TEST_DATABASE_URL=postgresql://brd_srs:brd_srs_local@127.0.0.1:5432/brd_srs_test
env PYTHONPATH=src .venv/bin/python -m pytest tests/test_storage.py -q
```

Expected: lifecycle/history tests PASS.

- [ ] **Step 7: Commit the schema and lifecycle repository**

```bash
rtk git add src/brd_srs_testgen/schema.sql src/brd_srs_testgen/storage.py tests/conftest.py tests/test_storage.py
rtk git commit -m "feat: persist run lifecycle in PostgreSQL"
```

### Task 4: Persist and reconstruct the normalized result graph

**Files:**
- Modify: `src/brd_srs_testgen/storage.py`
- Modify: `tests/test_storage.py`

- [ ] **Step 1: Add failing full-round-trip and immutability tests**

Append to `tests/test_storage.py`:

```python
from brd_srs_testgen.models import FailureCategory, RunResult
from tests.factories import completed_run


def test_completed_run_round_trips_all_normalized_artifacts(repository: RunRepository) -> None:
    result = completed_run()
    running = result.manifest.model_copy(
        update={"status": RunStatus.RUNNING, "completed_at": None}
    )
    repository.create_run(running)
    repository.save_chunks(running.run_id, [chunk()])

    repository.finalize(result)

    assert repository.load_run(result.manifest.run_id) == result


def test_normalized_child_tables_receive_the_complete_graph(
    repository: RunRepository,
) -> None:
    result = completed_run("normalized")
    running = result.manifest.model_copy(
        update={"status": RunStatus.RUNNING, "completed_at": None}
    )
    repository.create_run(running)
    repository.save_chunks(running.run_id, [chunk()])
    repository.finalize(result)

    with psycopg.connect(repository.database_url) as connection:
        counts = {
            table: connection.execute(
                f"SELECT count(*) FROM {table} WHERE run_id = %s", (running.run_id,)
            ).fetchone()[0]
            for table in (
                "requirements", "scenarios", "test_cases", "test_steps",
                "requirement_sources", "scenario_sources", "test_case_sources",
                "run_metrics", "validation_reports",
            )
        }
    assert counts == {
        "requirements": 1,
        "scenarios": 1,
        "test_cases": 1,
        "test_steps": 1,
        "requirement_sources": 1,
        "scenario_sources": 1,
        "test_case_sources": 1,
        "run_metrics": 1,
        "validation_reports": 1,
    }


def test_failed_run_without_artifacts_round_trips(repository: RunRepository) -> None:
    running = running_manifest("parse-failure")
    repository.create_run(running)
    failed = RunResult(
        manifest=running.model_copy(
            update={
                "status": RunStatus.FAILED,
                "completed_at": datetime.now(UTC),
                "failure_category": FailureCategory.PARSING,
                "failure_message": "PDF contains insufficient extractable text.",
            }
        )
    )

    repository.finalize(failed)

    assert repository.load_run(failed.manifest.run_id) == failed


def test_terminal_run_cannot_be_finalized_twice(repository: RunRepository) -> None:
    result = completed_run("immutable")
    running = result.manifest.model_copy(
        update={"status": RunStatus.RUNNING, "completed_at": None}
    )
    repository.create_run(running)
    repository.save_chunks(running.run_id, [chunk()])
    repository.finalize(result)

    with pytest.raises(ImmutableRunError, match="immutable"):
        repository.finalize(result)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```bash
export TEST_DATABASE_URL=postgresql://brd_srs:brd_srs_local@127.0.0.1:5432/brd_srs_test
env PYTHONPATH=src .venv/bin/python -m pytest tests/test_storage.py -q
```

Expected: FAIL because `finalize()` and `load_run()` do not exist.

- [ ] **Step 3: Implement transactional finalization**

Add to `RunRepository`:

```python
def finalize(self, result: RunResult) -> None:
    if result.manifest.status is RunStatus.RUNNING:
        raise ImmutableRunError("Finalization requires a terminal run.")
    with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
        _require_running(connection, result.manifest.run_id)
        if result.bundle is not None:
            self._insert_bundle(connection, result.manifest.run_id, result.bundle)
        if result.validation is not None:
            self._insert_validation(connection, result.manifest.run_id, result.validation)
        if result.metrics is not None:
            self._insert_metrics(connection, result.manifest.run_id, result.metrics)
        self._append_event(connection, result.manifest.run_id, "finished")
        updated = connection.execute(
            """
            UPDATE runs
            SET status = %s, completed_at = %s,
                failure_category = %s, failure_message = %s
            WHERE run_id = %s AND status = 'running'
            """,
            (
                result.manifest.status.value,
                result.manifest.completed_at,
                result.manifest.failure_category.value
                if result.manifest.failure_category else None,
                result.manifest.failure_message,
                result.manifest.run_id,
            ),
        )
        if updated.rowcount != 1:
            raise ImmutableRunError("Terminal runs are immutable.")
```

Implement `_insert_metrics`, `_insert_validation`, and `_insert_bundle` with parameterized `INSERT`/`executemany` calls. Preserve every list order with a one-based `position`. Wrap each `test_data` value with `psycopg.types.json.Jsonb`. Insert semantic-validation bundles even when the run status is failed. Do not insert the derived RTM.

`_insert_bundle` must populate all 15 artifact tables:

```text
requirements
requirement_ambiguities
requirement_dependencies
requirement_sources
scenarios
scenario_preconditions
scenario_requirements
scenario_sources
test_cases
test_case_preconditions
test_case_requirements
test_case_data
test_steps
test_case_sources
validation_* (through _insert_validation)
```

- [ ] **Step 4: Implement strict result reconstruction**

Add:

```python
def load_run(self, run_id: str) -> RunResult:
    with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
        manifest_row = connection.execute(
            "SELECT * FROM runs WHERE run_id = %s", (run_id,)
        ).fetchone()
        if manifest_row is None:
            raise StorageError("Run does not exist.")
        manifest = _manifest(manifest_row)
        metrics = self._load_metrics(connection, run_id)
        validation = self._load_validation(connection, run_id)
        bundle = self._load_bundle(connection, run_id)
        return RunResult(
            manifest=manifest,
            bundle=bundle,
            validation=validation,
            rtm=build_rtm(bundle) if bundle else [],
            metrics=metrics,
        )
```

Implement `_load_bundle` with one ordered query per root table and one ordered query for each child-table family. Group rows by their owning artifact ID, then construct the existing strict `Requirement`, `Scenario`, `TestCase`, `TestStep`, and `SourceReference` models. Return `None` when no requirement, scenario, or test-case roots exist. `_load_validation` must preserve issue/list order. `_load_metrics` maps every `RunMetrics` field explicitly; do not pass database-only columns into Pydantic.

Use this manifest conversion so enum/timestamp validation stays centralized:

```python
def _manifest(row: dict[str, object]) -> RunManifest:
    return RunManifest(
        run_id=row["run_id"],
        source_filename=row["source_filename"],
        document_hash=row["document_hash"],
        run_type=row["run_type"],
        status=row["status"],
        provider=row["provider"],
        model=row["model"],
        temperature=row["temperature"],
        token_ceiling=row["token_ceiling"],
        prompt_version=row["prompt_version"],
        schema_version=row["schema_version"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        failure_category=row["failure_category"],
        failure_message=row["failure_message"],
    )
```

- [ ] **Step 5: Run PostgreSQL repository tests**

Run:

```bash
export TEST_DATABASE_URL=postgresql://brd_srs:brd_srs_local@127.0.0.1:5432/brd_srs_test
env PYTHONPATH=src .venv/bin/python -m pytest tests/test_storage.py -q
```

Expected: PASS, including equality of the complete reconstructed `RunResult`.

- [ ] **Step 6: Commit normalized result persistence**

```bash
rtk git add src/brd_srs_testgen/storage.py tests/test_storage.py
rtk git commit -m "feat: store normalized generation results"
```

### Task 5: Execute only the selected pipeline

**Files:**
- Modify: `src/brd_srs_testgen/runner.py`
- Replace comparison-oriented cases in: `tests/test_runner.py`

- [ ] **Step 1: Write a failing selected-pipeline test**

Replace the comparison-order test in `tests/test_runner.py` with:

```python
import pytest

from brd_srs_testgen.models import RunStatus, RunType
from brd_srs_testgen.runner import run_generation


class RecordingRepository:
    def __init__(self) -> None:
        self.created = None
        self.chunks = []
        self.events = []
        self.finalized = None

    def create_run(self, manifest) -> None:
        self.created = manifest

    def save_chunks(self, run_id, chunks) -> None:
        self.chunks = chunks

    def append_event(self, run_id, stage, occurred_at=None) -> None:
        self.events.append(stage)

    def finalize(self, result) -> None:
        self.finalized = result


@pytest.mark.parametrize("selected", list(RunType))
def test_runs_only_the_selected_pipeline(selected, monkeypatch) -> None:
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])
    calls = []
    for run_type in RunType:
        monkeypatch.setitem(
            runner.PIPELINES,
            run_type,
            lambda _context, _chunks, run_type=run_type: (
                calls.append(run_type) or bundle()
            ),
        )
    repository = RecordingRepository()

    result = run_generation(
        b"pdf",
        "sample.pdf",
        selected,
        settings(),
        repository=repository,
        provider_factory=lambda _run_type, ledger: ScriptedProvider(ledger, []),
    )

    assert calls == [selected]
    assert result.manifest.run_type is selected
    assert result.manifest.status is RunStatus.COMPLETED
    assert repository.finalized == result
```

Add two lifecycle cases:

```python
def test_parsing_failure_is_persisted_after_run_creation(monkeypatch) -> None:
    monkeypatch.setattr(
        runner, "parse_pdf", lambda _data: (_ for _ in ()).throw(DocumentError("bad pdf"))
    )
    repository = RecordingRepository()

    result = run_generation(
        b"bad",
        "bad.pdf",
        RunType.SINGLE_PROMPT,
        settings(),
        repository=repository,
    )

    assert repository.created.status is RunStatus.RUNNING
    assert result.manifest.status is RunStatus.FAILED
    assert result.manifest.failure_category is FailureCategory.PARSING
    assert repository.finalized == result


def test_unexpected_error_leaves_run_interrupted(monkeypatch) -> None:
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])
    monkeypatch.setitem(
        runner.PIPELINES,
        RunType.SINGLE_PROMPT,
        lambda *_: (_ for _ in ()).throw(AssertionError("defect")),
    )
    repository = RecordingRepository()

    with pytest.raises(AssertionError, match="defect"):
        run_generation(
            b"pdf",
            "sample.pdf",
            RunType.SINGLE_PROMPT,
            settings(),
            repository=repository,
            provider_factory=lambda _run_type, ledger: ScriptedProvider(ledger, []),
        )

    assert repository.created.status is RunStatus.RUNNING
    assert repository.finalized is None
```

- [ ] **Step 2: Run the selected runner tests to verify they fail**

Run:

```bash
env PYTHONPATH=src .venv/bin/python -m pytest tests/test_runner.py -q
```

Expected: FAIL because `run_generation` and the new result model are not wired.

- [ ] **Step 3: Refactor the runner to one run**

In `src/brd_srs_testgen/runner.py`:

- Replace `Condition` imports/types with `RunType`.
- Move `ConditionResult.download_bundle()` behavior to the already-defined `RunResult` model.
- Delete `ConditionResult`, `ComparisonResult`, and `run_comparison`.
- Keep `PIPELINES`, provider settings, budget accounting, validation, repair, metrics, redaction, and failure categorization.
- Rename `_comparison_id` to `_run_id` without changing its collision-resistant timestamp/hash/UUID format.
- Change progress to `Callable[[str], None]` because one run no longer needs a condition argument.

Add this entry point:

```python
def run_generation(
    pdf_bytes: bytes,
    source_filename: str,
    run_type: RunType,
    settings: ProviderSettings,
    *,
    repository: RunRepository,
    progress: Progress | None = None,
    provider_factory: ProviderFactory | None = None,
) -> RunResult:
```

The body follows this exact lifecycle:

1. `settings.validate()` before creating a run; invalid form/config values are not started runs.
2. Hash `pdf_bytes`, reduce the display name with `Path(source_filename).name or "document.pdf"`, construct a `RUNNING` manifest, call `create_run`, append `started`, and notify `Preparing document`.
3. Parse the PDF. On `DocumentError`, create a failed manifest/result with parsing category, call `finalize`, notify `Failed`, and return.
4. Save chunks and append `parsed`.
5. Create one ledger and provider, call only `PIPELINES[run_type]`, then reuse canonicalization, validation, coverage repair, semantic revision, RTM, and metrics.
6. Convert recognized failures with `_failure_category`, `_empty_metrics`, and `_safe_message` exactly as today.
7. Build one `RunResult`, call `repository.finalize(result)`, notify the terminal status, and return.
8. Re-raise unrecognized internal defects without finalizing, leaving the running row visible as interrupted.

The default provider factory remains:

```python
provider_factory = provider_factory or (
    lambda _run_type, ledger: _make_provider(settings, ledger)
)
```

Keep the selected `RunType` argument in custom provider factories so existing tests can vary behavior by pipeline.

- [ ] **Step 4: Convert existing behavioral tests**

For every remaining `tests/test_runner.py` case:

- call `run_generation(pdf_bytes, source_filename, RunType.SINGLE_PROMPT, settings, repository=repository)` or substitute the run type relevant to that test;
- assert directly on `result.manifest`, `result.metrics`, `result.validation`, and `result.bundle` instead of indexing `result.conditions`;
- inject `RecordingRepository` for pure runner tests;
- keep the charged-token, timeout, retry, budget, deterministic-repair, semantic-validation, provider mismatch, parsing, and safe-message assertions;
- delete assertions that all three conditions execute or that one condition continues after another fails.

- [ ] **Step 5: Run runner and pipeline tests**

Run:

```bash
env PYTHONPATH=src .venv/bin/python -m pytest tests/test_runner.py tests/test_pipelines.py -q
```

Expected: PASS; no live provider calls.

- [ ] **Step 6: Commit single-run orchestration**

```bash
rtk git add src/brd_srs_testgen/runner.py tests/test_runner.py
rtk git commit -m "refactor: run one generation strategy"
```

### Task 6: Render one detailed generation result

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Add failing form and detailed-result tests**

In `tests/test_app.py`, replace comparison-specific fixtures with `completed_run()` and add a no-database fake:

```python
class FakeRepository:
    def __init__(self, results=(), history=()) -> None:
        self.results = {result.manifest.run_id: result for result in results}
        self.history = list(history)

    def initialize(self) -> None:
        pass

    def list_runs(self):
        return self.history

    def load_run(self, run_id):
        return self.results[run_id]


def app_test(repository=None) -> AppTest:
    at = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=10)
    at.session_state["_repository"] = repository or FakeRepository()
    return at
```

Add:

```python
def test_configures_exactly_one_run_type() -> None:
    at = app_test()
    at.run()

    selector = next(item for item in at.selectbox if item.label == "Run type")
    assert selector.options == [
        "Single prompt",
        "Staged single agent",
        "Centralized multi-agent",
    ]
    assert any(button.label == "Generate test cases" for button in at.button)


def test_runner_receives_selected_type_and_filename() -> None:
    received = {}

    def fake_runner(pdf_bytes, source_filename, run_type, settings, *, repository, progress):
        received.update(filename=source_filename, run_type=run_type)
        return completed_run(run_type=run_type)

    at = app_test()
    at.session_state["_runner"] = fake_runner
    at.run()
    next(item for item in at.selectbox if item.label == "Provider").set_value("ollama")
    at.run()
    next(item for item in at.selectbox if item.label == "Run type").set_value(
        "centralized_multi_agent"
    )
    next(item for item in at.file_uploader if item.label == "BRD/SRS PDF").set_value(
        [("source.pdf", b"%PDF-1.4\n", "application/pdf")]
    )
    next(item for item in at.button if item.label == "Generate test cases").click()
    at.run()

    assert received == {
        "filename": "source.pdf",
        "run_type": RunType.CENTRALIZED_MULTI_AGENT,
    }


def test_result_renders_complete_test_case_detail() -> None:
    at = app_test()
    at.session_state["run_result"] = completed_run()
    at.run()

    rendered = "\n".join(
        [item.value for item in at.markdown]
        + [item.value for item in at.caption]
        + [str(item.value) for item in at.table]
    )
    assert "TC-001" in rendered
    assert "Sign in with valid credentials" in rendered
    assert "Submit valid credentials" in rendered
    assert "The dashboard is displayed" in rendered
    assert "user@example.com" in rendered
    assert "REQ-001" in rendered
    assert "SCN-001" in rendered
    assert "Authenticate users" in rendered
    assert "Valid sign in" in rendered
    assert "p0001-c001" in rendered
```

- [ ] **Step 2: Run UI tests to verify they fail**

Run:

```bash
env PYTHONPATH=src .venv/bin/python -m pytest tests/test_app.py -q
```

Expected: FAIL because the UI still exposes the three-condition comparison.

- [ ] **Step 3: Convert the form and current result**

In `app.py`:

- Import `RunResult`, `RunType`, `RunRepository`, and `run_generation`.
- Keep the provider/model/credential/token controls.
- Add a required `Run type` selectbox using `list(RunType)` and `_run_type_label`.
- Rename copy from comparison/condition language to run/generation language.
- Rename the action to `Generate test cases`.
- Pass `uploaded.name`, the selected `RunType`, and the repository to the runner.
- Store the returned value as `st.session_state["run_result"]`.
- Remove `_failed_count`, `_summary_rows`, condition tabs, and three-condition empty cards.
- Preserve safe credential redaction and clear the previous result before starting a later run.
- Treat `metrics` as optional: an interrupted or parsing-failed run renders status, failure guidance when present, and diagnostics without reading metric fields.

Use a single status helper:

```python
def _result_status(result: RunResult) -> tuple[str, str]:
    if result.manifest.status is RunStatus.FAILED:
        return "Generation failed", "error"
    if result.manifest.status is RunStatus.RUNNING:
        return "Generation interrupted", "error"
    return "Generation complete", "complete"
```

- [ ] **Step 4: Add detailed artifact renderers**

Keep the current metrics/failure/download behavior, then add these focused renderers and call them for successful or semantic-validation results that have a bundle:

```python
def _render_sources(references) -> None:
    for reference in references:
        st.caption(
            f"Page {reference.page_number} · {reference.section or 'Unsectioned'} · "
            f"{reference.chunk_id}: {reference.excerpt}"
        )


def _render_requirements(bundle: ArtifactBundle) -> None:
    st.markdown("#### Requirements")
    for requirement in bundle.requirements:
        with st.expander(f"{requirement.requirement_id} · {requirement.title}"):
            st.write(requirement.description)
            st.write(
                f"Type: {requirement.requirement_type.value} · "
                f"Priority: {requirement.priority.value} · Module: {requirement.module}"
            )
            if requirement.dependency_ids:
                st.write("Dependencies: " + ", ".join(requirement.dependency_ids))
            if requirement.ambiguities:
                st.write("Ambiguities: " + "; ".join(requirement.ambiguities))
            _render_sources(requirement.source_references)


def _render_scenarios(bundle: ArtifactBundle) -> None:
    st.markdown("#### Scenarios")
    for scenario in bundle.scenarios:
        with st.expander(f"{scenario.scenario_id} · {scenario.title}"):
            st.write(scenario.objective)
            st.write("Type: " + scenario.scenario_type.value)
            st.write("Requirements: " + ", ".join(scenario.requirement_ids))
            if scenario.preconditions:
                st.write("Preconditions: " + "; ".join(scenario.preconditions))
            _render_sources(scenario.source_references)


def _render_test_cases(bundle: ArtifactBundle) -> None:
    st.markdown("#### Test cases")
    for case in bundle.test_cases:
        with st.expander(f"{case.test_case_id} · {case.title}"):
            st.write(
                f"Priority: {case.priority.value} · Scenario: {case.scenario_id}"
            )
            st.write("Requirements: " + ", ".join(case.requirement_ids))
            if case.preconditions:
                st.write("Preconditions: " + "; ".join(case.preconditions))
            if case.test_data:
                st.markdown("**Test data**")
                st.json(case.test_data)
            st.markdown("**Steps**")
            st.table(
                [
                    {
                        "Step": step.step_number,
                        "Action": step.action,
                        "Expected result": step.expected_result,
                    }
                    for step in case.steps
                ]
            )
            _render_sources(case.source_references)
```

- [ ] **Step 5: Run Streamlit tests**

Run:

```bash
env PYTHONPATH=src .venv/bin/python -m pytest tests/test_app.py -q
```

Expected: PASS, including detailed step/action/expected-result text.

- [ ] **Step 6: Commit the single-run result UI**

```bash
rtk git add app.py tests/test_app.py
rtk git commit -m "feat: show detailed single-run results"
```

### Task 7: Initialize PostgreSQL and add run history

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Add failing history and database-error tests**

Extend `FakeRepository.list_runs()` to return configured `RunHistoryItem` rows, then add:

```python
def test_history_lists_interrupted_and_loads_selected_result() -> None:
    result = completed_run("saved-run")
    interrupted = RunHistoryItem(
        run_id="interrupted-run",
        source_filename="unfinished.pdf",
        run_type=RunType.STAGED_SINGLE_AGENT,
        status=RunStatus.RUNNING,
        provider="ollama",
        model="gemma4",
        started_at=datetime.now(UTC),
        completed_at=None,
    )
    completed = RunHistoryItem(
        run_id=result.manifest.run_id,
        source_filename=result.manifest.source_filename,
        run_type=result.manifest.run_type,
        status=result.manifest.status,
        provider=result.manifest.provider,
        model=result.manifest.model,
        started_at=result.manifest.started_at,
        completed_at=result.manifest.completed_at,
        requirement_count=1,
        scenario_count=1,
        test_case_count=1,
    )
    repository = FakeRepository([result], [interrupted, completed])
    at = app_test(repository)
    at.run()

    rendered = "\n".join(str(item.value) for item in at.dataframe)
    assert "Interrupted" in rendered
    selector = next(item for item in at.selectbox if item.label == "Open saved run")
    selector.set_value("saved-run")
    at.run()

    assert any("TC-001" in item.value for item in at.markdown)


def test_database_initialization_failure_blocks_generation() -> None:
    class BrokenRepository(FakeRepository):
        def initialize(self):
            raise StorageError("database unavailable")

    at = app_test(BrokenRepository())
    at.run()

    assert any("database unavailable" in error.value for error in at.error)
    assert not any(button.label == "Generate test cases" for button in at.button)
```

- [ ] **Step 2: Run history tests to verify they fail**

Run:

```bash
env PYTHONPATH=src .venv/bin/python -m pytest tests/test_app.py -q
```

Expected: FAIL because repository initialization and history rendering are absent.

- [ ] **Step 3: Initialize the repository once and fail visibly**

Add:

```python
@st.cache_resource
def _postgres_repository(database_url: str) -> RunRepository:
    repository = RunRepository(database_url)
    repository.initialize()
    return repository


def _repository() -> RunRepository:
    injected = st.session_state.get("_repository")
    if injected is not None:
        injected.initialize()
        return injected
    return _postgres_repository(_env("DATABASE_URL"))
```

At the start of `main()`, after page/theme setup, resolve the repository inside `try/except StorageError`. On failure, show:

```python
st.error(f"Run history database is unavailable: {error}")
st.info("Start PostgreSQL with `docker compose up -d db` and verify DATABASE_URL.")
st.stop()
```

Do not fall back to `runs/`.

- [ ] **Step 4: Add the fourth workflow tab and history loader**

Use four labels:

```python
step_labels = ["1 · Configure", "2 · Run", "3 · Results", "4 · Run history"]
configure_tab, run_tab, results_tab, history_tab = st.tabs(step_labels)
```

In the history tab:

```python
history = repository.list_runs()
if not history:
    st.info("No saved runs yet.")
else:
    st.dataframe(
        [
            {
                "Started": item.started_at,
                "Source": item.source_filename,
                "Run type": _run_type_label(item.run_type),
                "Provider": _provider_label(item.provider),
                "Model": item.model,
                "Status": item.display_status,
                "Requirements": item.requirement_count,
                "Scenarios": item.scenario_count,
                "Test cases": item.test_case_count,
            }
            for item in history
        ],
        hide_index=True,
        width="stretch",
    )
    selected_run_id = st.selectbox(
        "Open saved run",
        [item.run_id for item in history],
        index=None,
        format_func=lambda run_id: next(
            f"{item.started_at:%Y-%m-%d %H:%M} · {item.source_filename} · "
            f"{_run_type_label(item.run_type)}"
            for item in history if item.run_id == run_id
        ),
        placeholder="Select a saved run",
    )
    if selected_run_id:
        _render_result(repository.load_run(selected_run_id))
```

Catch `StorageError` around history listing/loading and show one actionable error without breaking the configure/run tabs.

- [ ] **Step 5: Run application tests**

Run:

```bash
env PYTHONPATH=src .venv/bin/python -m pytest tests/test_app.py -q
```

Expected: PASS; selecting history renders the same detailed result component.

- [ ] **Step 6: Commit run history**

```bash
rtk git add app.py tests/test_app.py
rtk git commit -m "feat: browse persisted run history"
```

### Task 8: Update operations and remove filesystem assumptions

**Files:**
- Modify: `README.md`
- Modify: `docs/research-core-operations.md`
- Verify: `src/brd_srs_testgen/storage.py`
- Verify: `app.py`
- Verify: `tests/`

- [ ] **Step 1: Update the quick start**

Change `README.md` research setup to the following. The conditional copy preserves an existing local `.env`; if it already exists, add the documented `DATABASE_URL` line to it instead of replacing credentials.

```sh
test -f .env || cp .env.example .env
docker compose up -d db
uv pip install --python .venv/bin/python -r requirements.txt
env PYTHONPATH=src .venv/bin/python -m streamlit run app.py
```

State that each click executes one selected run type and that results remain in PostgreSQL history.

- [ ] **Step 2: Replace the operations guide's comparison protocol**

Update `docs/research-core-operations.md` to document:

- `docker compose up -d db` and the `DATABASE_URL` requirement;
- selecting one of the three run types;
- `Generate test cases` executing only that type;
- reopening completed, failed, and interrupted runs in **Run history**;
- PostgreSQL's named volume as the persistent store;
- raw PDFs and credentials not being stored;
- `runs/` as untouched legacy data that the application ignores;
- test setup with `TEST_DATABASE_URL` targeting `brd_srs_test`; and
- `docker compose down` stopping services without deleting history (`down -v` is deliberately not presented as a normal operation).

Remove statements that each action runs all three conditions or that new artifacts are persisted under `runs/<comparison-id>/`.

- [ ] **Step 3: Scan for stale comparison and filesystem APIs**

Run:

```bash
rtk rg -n "run_comparison|ComparisonResult|ComparisonManifest|ConditionManifest|RunStore|comparison_result|conditions/" app.py src tests README.md docs/research-core-operations.md
```

Expected: no matches. References inside archived design/plan documents are allowed because they describe the earlier system.

- [ ] **Step 4: Run focused verification**

Run:

```bash
export TEST_DATABASE_URL=postgresql://brd_srs:brd_srs_local@127.0.0.1:5432/brd_srs_test
env PYTHONPATH=src .venv/bin/python -m pytest tests/test_models.py tests/test_storage.py tests/test_runner.py tests/test_app.py -q
```

Expected: PASS with zero failures. PostgreSQL integration tests must run, not skip.

- [ ] **Step 5: Run the full offline gate**

Run:

```bash
export TEST_DATABASE_URL=postgresql://brd_srs:brd_srs_local@127.0.0.1:5432/brd_srs_test
env PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q app.py src tests
env PYTHONPATH=src .venv/bin/python -c "from brd_srs_testgen.runner import run_generation; print('imports ok')"
rtk git diff --check
rtk git status --short
```

Expected: all tests pass with no skips from `tests/test_storage.py`; compilation exits 0; import prints `imports ok`; diff check is empty; status contains only the intended task changes before the final commit.

- [ ] **Step 6: Commit documentation and cleanup**

```bash
rtk git add README.md docs/research-core-operations.md
rtk git commit -m "docs: document single-run history"
```

### Task 9: Manual local smoke check

**Files:**
- No source changes expected

- [ ] **Step 1: Start the application**

Run:

```bash
rtk docker compose up -d db
env PYTHONPATH=src .venv/bin/python -m streamlit run app.py
```

Expected: Streamlit starts without a database warning.

- [ ] **Step 2: Exercise one offline/local-model run**

With Ollama or LM Studio already running, upload a small non-sensitive text PDF, select one run type, and click **Generate test cases**.

Expected:

- only the selected strategy appears in progress;
- the result shows requirements, scenarios, full test-case steps, expected results, test data, and citations;
- a completed or actionable failed row appears in **Run history**; and
- selecting that row renders the same details.

- [ ] **Step 3: Verify restart persistence**

Stop Streamlit, start it again, and open **Run history**.

Expected: the saved row and detailed result remain available. No live Gemini call is required for this smoke check.

- [ ] **Step 4: Record final repository evidence**

Run:

```bash
rtk git log --oneline -8
rtk git status --short
```

Expected: the task commits are present and the worktree is clean.
