# Centralized Agent Test-Case Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Streamlit application and experiment runner that compare single-prompt, staged single-agent, and centralized multi-agent generation of traceable manual test cases from BRD/SRS PDFs.

**Architecture:** A page-aware evidence service feeds three generation pipelines through one budgeted Gemini gateway. Pydantic schemas and deterministic validation build traceability and metrics; local immutable run directories preserve experiment outputs, and Streamlit provides generation, artifact, experiment, and blinded-evaluation views.

**Tech Stack:** Python 3.11+, Streamlit, Google Gen AI SDK, Pydantic 2, pypdf, pandas, openpyxl, NumPy, krippendorff, pytest, and the Python standard library.

---

**Design spec:** `docs/superpowers/specs/2026-08-10-centralized-agent-test-case-generation-design.md`

**Execution rule:** Run all shell commands through `rtk`. Do not use the live Gemini API in automated tests.

## File structure

Create the following focused modules:

```text
app.py                                      Streamlit entry point and four tabs
requirements.txt                            Runtime and test dependencies
.gitignore                                  Secrets, virtualenv, caches, and run artifacts
src/brd_srs_testgen/__init__.py             Public package exports
src/brd_srs_testgen/models.py               Canonical Pydantic schemas and enums
src/brd_srs_testgen/documents.py            PDF parsing, chunking, and evidence checks
src/brd_srs_testgen/llm.py                  Gemini gateway and thread-safe token ledger
src/brd_srs_testgen/prompts.py              Versioned prompt builders
src/brd_srs_testgen/validation.py           Referential checks and RTM construction
src/brd_srs_testgen/pipelines.py            Three controlled generation conditions
src/brd_srs_testgen/storage.py              Atomic immutable run persistence
src/brd_srs_testgen/metrics.py              Automated structural and cost metrics
src/brd_srs_testgen/experiments.py          54-run manifest, preflight, execution, resume
src/brd_srs_testgen/human_eval.py           Blinding, sampling, agreement, paired summaries
src/brd_srs_testgen/exports.py               JSON and Excel exports
tests/                                      Focused tests with mocked model responses
```

Keep the existing prototype scripts untouched until the new application is verified. They can be removed in a later explicit cleanup.

### Task 1: Bootstrap the package and canonical schemas

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `src/brd_srs_testgen/__init__.py`
- Create: `src/brd_srs_testgen/models.py`
- Create: `tests/__init__.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Add dependencies and ignore local artifacts**

Create `requirements.txt`:

```text
google-genai>=2,<3
krippendorff>=0.8,<1
numpy>=2,<3
openpyxl>=3.1,<4
pandas>=2.2,<3
pydantic>=2.10,<3
pypdf>=6,<7
pytest>=8,<9
streamlit>=1.50,<2
```

Create `.gitignore`:

```text
.env
.venv/
.pytest_cache/
__pycache__/
*.pyc
.streamlit/secrets.toml
runs/
```

- [ ] **Step 2: Install the declared environment**

Run:

```bash
rtk uv --cache-dir /tmp/citd-final-uv-cache venv --python 3.11 .venv
rtk .venv/bin/python -m pip install -r requirements.txt
rtk .venv/bin/python -c "import streamlit, pydantic, pypdf; print('ok')"
```

Expected: all commands exit `0` and the final command prints `ok`.

- [ ] **Step 3: Write failing schema tests**

Create `tests/test_models.py`:

```python
import pytest
from pydantic import ValidationError

from brd_srs_testgen.models import (
    ArtifactBundle,
    Requirement,
    RequirementType,
    Scenario,
    ScenarioType,
    SourceReference,
    TestCase,
    TestPriority,
    TestStep,
)


def source() -> SourceReference:
    return SourceReference(
        chunk_id="p0001-c001-a1b2c3d4e5f6",
        page_number=1,
        section="Authentication",
        excerpt="The system shall authenticate registered users.",
    )


def test_artifact_bundle_accepts_many_to_many_traceability() -> None:
    requirement = Requirement(
        requirement_id="REQ-001",
        title="Authenticate users",
        description="Registered users can sign in.",
        requirement_type=RequirementType.FUNCTIONAL,
        module="Authentication",
        priority="high",
        source_references=[source()],
    )
    scenario = Scenario(
        scenario_id="SCN-001",
        title="Valid sign in",
        objective="Verify successful authentication.",
        scenario_type=ScenarioType.POSITIVE,
        requirement_ids=["REQ-001"],
        source_references=[source()],
    )
    test_case = TestCase(
        test_case_id="TC-001",
        scenario_id="SCN-001",
        requirement_ids=["REQ-001"],
        title="Sign in with valid credentials",
        priority=TestPriority.P1,
        preconditions=["A registered account exists."],
        test_data={"email": "user@example.com"},
        steps=[
            TestStep(
                step_number=1,
                action="Submit valid credentials.",
                expected_result="The dashboard is displayed.",
            )
        ],
        source_references=[source()],
    )

    bundle = ArtifactBundle(
        requirements=[requirement], scenarios=[scenario], test_cases=[test_case]
    )

    assert bundle.test_cases[0].requirement_ids == ["REQ-001"]


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SourceReference.model_validate(
            {
                "chunk_id": "chunk",
                "page_number": 1,
                "section": "Section",
                "excerpt": "Evidence",
                "invented": True,
            }
        )
```

Create an empty `tests/__init__.py` so test fixtures can be imported explicitly by later tasks:

```python
"""Test package."""
```

- [ ] **Step 4: Run the tests to verify they fail**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_models.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'brd_srs_testgen'`.

- [ ] **Step 5: Implement the canonical schemas**

Create `src/brd_srs_testgen/models.py`:

```python
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RequirementType(StrEnum):
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    BUSINESS = "business"


class RequirementPriority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ScenarioType(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    BOUNDARY = "boundary"
    EDGE = "edge"
    STATE_TRANSITION = "state_transition"


class TestPriority(StrEnum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class Condition(StrEnum):
    SINGLE_PROMPT = "single_prompt"
    STAGED_SINGLE_AGENT = "staged_single_agent"
    CENTRALIZED_MULTI_AGENT = "centralized_multi_agent"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SourceReference(StrictModel):
    chunk_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    section: str = ""
    excerpt: str = Field(min_length=1)


class DocumentChunk(StrictModel):
    chunk_id: str
    page_number: int = Field(ge=1)
    section: str = ""
    text: str = Field(min_length=1)
    content_hash: str


class Requirement(StrictModel):
    requirement_id: str
    title: str
    description: str
    requirement_type: RequirementType
    module: str
    priority: RequirementPriority
    ambiguities: list[str] = Field(default_factory=list)
    dependency_ids: list[str] = Field(default_factory=list)
    source_references: list[SourceReference] = Field(min_length=1)


class Scenario(StrictModel):
    scenario_id: str
    title: str
    objective: str
    scenario_type: ScenarioType
    preconditions: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(min_length=1)
    source_references: list[SourceReference] = Field(min_length=1)


class TestStep(StrictModel):
    step_number: int = Field(ge=1)
    action: str
    expected_result: str


class TestCase(StrictModel):
    test_case_id: str
    scenario_id: str
    requirement_ids: list[str] = Field(min_length=1)
    title: str
    priority: TestPriority
    preconditions: list[str] = Field(default_factory=list)
    test_data: dict[str, Any] = Field(default_factory=dict)
    steps: list[TestStep] = Field(min_length=1)
    source_references: list[SourceReference] = Field(min_length=1)


class ArtifactBundle(StrictModel):
    requirements: list[Requirement]
    scenarios: list[Scenario]
    test_cases: list[TestCase]


class RequirementBatch(StrictModel):
    requirements: list[Requirement]


class ScenarioBatch(StrictModel):
    scenarios: list[Scenario]


class TestCaseBatch(StrictModel):
    test_cases: list[TestCase]


class DocumentOverview(StrictModel):
    purpose: str
    actors: list[str]
    modules: list[str]
    business_rules: list[str]


class ReviewIssue(StrictModel):
    artifact_id: str
    reason: str


class ReviewResult(StrictModel):
    accepted: bool
    issues: list[ReviewIssue] = Field(default_factory=list)
```

Create `src/brd_srs_testgen/__init__.py`:

```python
from .models import ArtifactBundle, Condition

__version__ = "0.1.0"
__all__ = ["ArtifactBundle", "Condition"]
```

- [ ] **Step 6: Run the schema tests**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_models.py -q
```

Expected: `2 passed`.

- [ ] **Step 7: Commit the schema foundation**

```bash
rtk git add requirements.txt .gitignore src/brd_srs_testgen tests
rtk git commit -m "feat: add canonical artifact schemas"
```

### Task 2: Parse PDFs into deterministic evidence chunks

**Files:**
- Create: `src/brd_srs_testgen/documents.py`
- Create: `tests/test_documents.py`

- [ ] **Step 1: Write failing document tests**

Create `tests/test_documents.py`:

```python
import pytest

from brd_srs_testgen.documents import (
    DocumentError,
    chunk_pages,
    normalize_text,
    verify_source_reference,
)
from brd_srs_testgen.models import SourceReference


def test_chunk_ids_are_stable_and_page_aware() -> None:
    pages = [(1, "1 Introduction\n" + "A" * 30), (2, "2 Login\n" + "B" * 30)]

    first = chunk_pages(pages, max_chars=20)
    second = chunk_pages(pages, max_chars=20)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert {chunk.page_number for chunk in first} == {1, 2}


def test_source_excerpt_must_exist_in_referenced_chunk() -> None:
    chunks = chunk_pages([(1, "The system shall authenticate users.")])
    valid = SourceReference(
        chunk_id=chunks[0].chunk_id,
        page_number=1,
        section=chunks[0].section,
        excerpt="system shall authenticate users",
    )
    invalid = valid.model_copy(update={"excerpt": "reset passwords"})

    assert verify_source_reference(valid, chunks)
    assert not verify_source_reference(invalid, chunks)


def test_empty_document_is_rejected() -> None:
    with pytest.raises(DocumentError, match="extractable text"):
        chunk_pages([(1, "  "), (2, "")])


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("A\n  B\tC") == "A B C"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_documents.py -q
```

Expected: collection fails because `brd_srs_testgen.documents` does not exist.

- [ ] **Step 3: Implement extraction, chunking, and evidence verification**

Create `src/brd_srs_testgen/documents.py`:

```python
from __future__ import annotations

import hashlib
import io
import re
from collections.abc import Iterable

from pypdf import PdfReader

from .models import DocumentChunk, SourceReference


class DocumentError(ValueError):
    pass


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def extract_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    if reader.is_encrypted:
        raise DocumentError("Encrypted PDFs are not supported.")
    return [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, 1)]


def _section_heading(text: str) -> str:
    for line in text.splitlines():
        candidate = normalize_text(line)
        if 3 <= len(candidate) <= 100 and (
            re.match(r"^\d+(?:\.\d+)*\s+\S", candidate) or candidate.isupper()
        ):
            return candidate
    # ponytail: heading heuristic; replace with a layout parser only if section accuracy becomes a metric.
    return ""


def chunk_pages(
    pages: Iterable[tuple[int, str]], max_chars: int = 6_000
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for page_number, raw_text in pages:
        text = normalize_text(raw_text)
        if not text:
            continue
        section = _section_heading(raw_text)
        for sequence, start in enumerate(range(0, len(text), max_chars), 1):
            piece = text[start : start + max_chars]
            digest = hashlib.sha256(piece.encode()).hexdigest()
            chunks.append(
                DocumentChunk(
                    chunk_id=f"p{page_number:04d}-c{sequence:03d}-{digest[:12]}",
                    page_number=page_number,
                    section=section,
                    text=piece,
                    content_hash=digest,
                )
            )
    if not chunks:
        raise DocumentError("PDF contains insufficient extractable text.")
    return chunks


def parse_pdf(pdf_bytes: bytes, max_chars: int = 6_000) -> list[DocumentChunk]:
    return chunk_pages(extract_pages(pdf_bytes), max_chars=max_chars)


def verify_source_reference(
    reference: SourceReference, chunks: list[DocumentChunk]
) -> bool:
    chunk = next((item for item in chunks if item.chunk_id == reference.chunk_id), None)
    return bool(
        chunk
        and chunk.page_number == reference.page_number
        and normalize_text(reference.excerpt).lower() in normalize_text(chunk.text).lower()
    )


def render_chunks(chunks: Iterable[DocumentChunk]) -> str:
    return "\n\n".join(
        f"[{chunk.chunk_id} | page {chunk.page_number} | {chunk.section}]\n{chunk.text}"
        for chunk in chunks
    )
```

- [ ] **Step 4: Run document tests**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_documents.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit deterministic document evidence**

```bash
rtk git add src/brd_srs_testgen/documents.py tests/test_documents.py
rtk git commit -m "feat: add page-aware PDF evidence"
```

### Task 3: Persist immutable runs atomically

**Files:**
- Create: `src/brd_srs_testgen/storage.py`
- Create: `tests/test_storage.py`

- [ ] **Step 1: Write failing storage tests**

Create `tests/test_storage.py`:

```python
import json

import pytest

from brd_srs_testgen.storage import CompletedRunError, RunStore


def test_completed_runs_are_immutable(tmp_path) -> None:
    store = RunStore(tmp_path)
    store.create("run-1", {"status": "running"})
    store.write_artifact("run-1", "requirements", [{"requirement_id": "REQ-001"}])
    store.complete("run-1")

    with pytest.raises(CompletedRunError):
        store.write_artifact("run-1", "requirements", [])


def test_events_are_append_only_json_lines(tmp_path) -> None:
    store = RunStore(tmp_path)
    store.create("run-1", {"status": "running"})
    store.append_event("run-1", {"type": "started"})
    store.append_event("run-1", {"type": "finished"})

    lines = (tmp_path / "run-1" / "events.jsonl").read_text().splitlines()
    assert [json.loads(line)["type"] for line in lines] == ["started", "finished"]


def test_partial_artifacts_can_be_read_for_resume(tmp_path) -> None:
    store = RunStore(tmp_path)
    store.create("run-1", {"status": "running"})
    store.write_artifact("run-1", "overview", {"purpose": "Checkout"})

    assert store.exists("run-1")
    assert store.has_artifact("run-1", "overview")
    assert store.read_artifact("run-1", "overview") == {"purpose": "Checkout"}
    assert store.read_manifest("run-1")["status"] == "running"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_storage.py -q
```

Expected: collection fails because `brd_srs_testgen.storage` does not exist.

- [ ] **Step 3: Implement atomic local storage**

Create `src/brd_srs_testgen/storage.py`:

```python
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class CompletedRunError(RuntimeError):
    pass


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _directory(self, run_id: str) -> Path:
        return self.root / run_id

    def _manifest(self, run_id: str) -> dict[str, Any]:
        return json.loads(
            (self._directory(run_id) / "manifest.json").read_text(encoding="utf-8")
        )

    def _ensure_mutable(self, run_id: str) -> None:
        if self._manifest(run_id).get("status") == "completed":
            raise CompletedRunError(f"Run {run_id} is complete and immutable.")

    def exists(self, run_id: str) -> bool:
        return (self._directory(run_id) / "manifest.json").exists()

    def read_manifest(self, run_id: str) -> dict[str, Any]:
        return self._manifest(run_id)

    def has_artifact(self, run_id: str, name: str) -> bool:
        return (self._directory(run_id) / f"{name}.json").exists()

    def read_artifact(self, run_id: str, name: str) -> Any:
        return json.loads(
            (self._directory(run_id) / f"{name}.json").read_text(encoding="utf-8")
        )

    def create(self, run_id: str, manifest: dict[str, Any]) -> None:
        directory = self._directory(run_id)
        if directory.exists():
            raise FileExistsError(run_id)
        directory.mkdir(parents=True)
        _atomic_json(directory / "manifest.json", manifest)

    def write_artifact(self, run_id: str, name: str, value: Any) -> None:
        self._ensure_mutable(run_id)
        _atomic_json(self._directory(run_id) / f"{name}.json", value)

    def append_event(self, run_id: str, event: dict[str, Any]) -> None:
        self._ensure_mutable(run_id)
        with (self._directory(run_id) / "events.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def complete(self, run_id: str) -> None:
        manifest = self._manifest(run_id)
        manifest["status"] = "completed"
        _atomic_json(self._directory(run_id) / "manifest.json", manifest)

    def existing_artifacts(self, run_id: str) -> set[str]:
        return {
            path.stem
            for path in self._directory(run_id).glob("*.json")
            if path.name != "manifest.json"
        }
```

- [ ] **Step 4: Run storage tests**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_storage.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit run persistence**

```bash
rtk git add src/brd_srs_testgen/storage.py tests/test_storage.py
rtk git commit -m "feat: persist immutable experiment runs"
```

### Task 4: Add a budgeted Gemini structured-output gateway

**Files:**
- Create: `src/brd_srs_testgen/llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write failing ledger and gateway tests**

Create `tests/test_llm.py`:

```python
from types import SimpleNamespace

import pytest

from brd_srs_testgen.llm import BudgetExceeded, BudgetLedger, GeminiGateway
from brd_srs_testgen.models import DocumentOverview


class FakeModels:
    def count_tokens(self, **_kwargs):
        return SimpleNamespace(total_tokens=10)

    def generate_content(self, **_kwargs):
        return SimpleNamespace(
            text='{"purpose":"Test","actors":[],"modules":[],"business_rules":[]}',
            usage_metadata=SimpleNamespace(total_token_count=25),
        )


def test_ledger_prevents_concurrent_over_reservation() -> None:
    ledger = BudgetLedger(limit=100)
    reservation = ledger.reserve(80)

    with pytest.raises(BudgetExceeded):
        ledger.reserve(30)

    ledger.settle(reservation, actual_tokens=50)
    assert ledger.used == 50


def test_gateway_validates_json_and_records_actual_usage() -> None:
    client = SimpleNamespace(models=FakeModels())
    ledger = BudgetLedger(limit=100)
    gateway = GeminiGateway(client=client, model="gemini-test", ledger=ledger)

    result = gateway.generate("Summarize", DocumentOverview, max_output_tokens=40)

    assert result.value.purpose == "Test"
    assert result.total_tokens == 25
    assert ledger.used == 25
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_llm.py -q
```

Expected: collection fails because `brd_srs_testgen.llm` does not exist.

- [ ] **Step 3: Implement the thread-safe budget ledger and gateway**

Create `src/brd_srs_testgen/llm.py`:

```python
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Reservation:
    tokens: int


@dataclass
class BudgetLedger:
    limit: int
    used: int = 0
    reserved: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def remaining(self) -> int:
        with self._lock:
            return self.limit - self.used - self.reserved

    def reserve(self, tokens: int) -> Reservation:
        with self._lock:
            if self.used + self.reserved + tokens > self.limit:
                raise BudgetExceeded(f"Need {tokens} tokens; {self.limit - self.used - self.reserved} remain.")
            self.reserved += tokens
        return Reservation(tokens)

    def settle(self, reservation: Reservation, actual_tokens: int) -> None:
        with self._lock:
            self.reserved -= reservation.tokens
            self.used += actual_tokens

    def cancel(self, reservation: Reservation) -> None:
        with self._lock:
            self.reserved -= reservation.tokens


@dataclass(frozen=True)
class GenerationResult(Generic[T]):
    value: T
    total_tokens: int


class GeminiGateway:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.6-flash",
        ledger: BudgetLedger | None = None,
        client=None,
    ) -> None:
        self.client = client or genai.Client(api_key=api_key)
        self.model = model
        self.ledger = ledger or BudgetLedger(limit=200_000)

    def generate(
        self,
        prompt: str,
        schema: type[T],
        max_output_tokens: int,
        temperature: float = 0.0,
    ) -> GenerationResult[T]:
        input_tokens = int(
            self.client.models.count_tokens(model=self.model, contents=prompt).total_tokens
        )
        reservation = self.ledger.reserve(input_tokens + max_output_tokens)
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )
            value = schema.model_validate(json.loads(response.text))
            total = int(response.usage_metadata.total_token_count)
            self.ledger.settle(reservation, total)
            return GenerationResult(value=value, total_tokens=total)
        except Exception:
            self.ledger.cancel(reservation)
            raise
```

Do not add provider retries here. Keep the gateway one-call deterministic; the pipeline records and applies the bounded transport retry policy around it.

- [ ] **Step 4: Run gateway tests**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_llm.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit provider and budget control**

```bash
rtk git add src/brd_srs_testgen/llm.py tests/test_llm.py
rtk git commit -m "feat: add budgeted Gemini gateway"
```

### Task 5: Add versioned prompts, referential validation, and RTM construction

**Files:**
- Create: `src/brd_srs_testgen/prompts.py`
- Create: `src/brd_srs_testgen/validation.py`
- Modify: `src/brd_srs_testgen/models.py`
- Create: `tests/test_validation.py`

- [ ] **Step 1: Write failing RTM and source-validation tests**

Create `tests/test_validation.py`:

```python
from brd_srs_testgen.models import (
    ArtifactBundle,
    DocumentChunk,
    Requirement,
    RequirementType,
    Scenario,
    ScenarioType,
    SourceReference,
    TestCase,
    TestPriority,
    TestStep,
)
from brd_srs_testgen.validation import build_rtm, validate_bundle


def fixture_bundle() -> tuple[ArtifactBundle, list[DocumentChunk]]:
    chunk = DocumentChunk(
        chunk_id="p0001-c001-abc",
        page_number=1,
        section="Login",
        text="The system shall authenticate users.",
        content_hash="abc",
    )
    source = SourceReference(
        chunk_id=chunk.chunk_id,
        page_number=1,
        section="Login",
        excerpt="authenticate users",
    )
    requirement = Requirement(
        requirement_id="REQ-001",
        title="Authenticate",
        description="Authenticate registered users.",
        requirement_type=RequirementType.FUNCTIONAL,
        module="Login",
        priority="high",
        source_references=[source],
    )
    scenario = Scenario(
        scenario_id="SCN-001",
        title="Valid login",
        objective="Verify login.",
        scenario_type=ScenarioType.POSITIVE,
        requirement_ids=[requirement.requirement_id],
        source_references=[source],
    )
    test_case = TestCase(
        test_case_id="TC-001",
        scenario_id=scenario.scenario_id,
        requirement_ids=[requirement.requirement_id],
        title="Valid login",
        priority=TestPriority.P1,
        steps=[TestStep(step_number=1, action="Log in.", expected_result="Login succeeds.")],
        source_references=[source],
    )
    return ArtifactBundle(
        requirements=[requirement], scenarios=[scenario], test_cases=[test_case]
    ), [chunk]


def test_valid_bundle_builds_covered_rtm_row() -> None:
    bundle, chunks = fixture_bundle()

    assert validate_bundle(bundle, chunks) == []
    assert build_rtm(bundle)[0].model_dump() == {
        "requirement_id": "REQ-001",
        "scenario_ids": ["SCN-001"],
        "test_case_ids": ["TC-001"],
        "coverage_status": "covered",
    }


def test_invalid_parent_and_evidence_are_reported_without_deleting_artifacts() -> None:
    bundle, chunks = fixture_bundle()
    broken = bundle.model_copy(
        update={
            "test_cases": [
                bundle.test_cases[0].model_copy(
                    update={
                        "scenario_id": "SCN-MISSING",
                        "source_references": [
                            bundle.test_cases[0].source_references[0].model_copy(
                                update={"excerpt": "invented behavior"}
                            )
                        ],
                    }
                )
            ]
        }
    )

    issues = validate_bundle(broken, chunks)

    assert {issue.code for issue in issues} == {"missing_scenario", "invalid_source"}
    assert broken.test_cases[0].test_case_id == "TC-001"
```

- [ ] **Step 2: Run validation tests to verify they fail**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_validation.py -q
```

Expected: collection fails because `brd_srs_testgen.validation` does not exist.

- [ ] **Step 3: Add validation and RTM models**

Append to `src/brd_srs_testgen/models.py`:

```python
class ValidationIssue(StrictModel):
    code: str
    artifact_id: str
    detail: str


class RTMRow(StrictModel):
    requirement_id: str
    scenario_ids: list[str]
    test_case_ids: list[str]
    coverage_status: str
```

- [ ] **Step 4: Implement deterministic validation and RTM construction**

Create `src/brd_srs_testgen/validation.py`:

```python
from __future__ import annotations

from collections import Counter

from .documents import verify_source_reference
from .models import ArtifactBundle, DocumentChunk, RTMRow, ValidationIssue


def _duplicates(values: list[str]) -> set[str]:
    return {value for value, count in Counter(values).items() if count > 1}


def validate_bundle(
    bundle: ArtifactBundle, chunks: list[DocumentChunk]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    requirement_ids = {item.requirement_id for item in bundle.requirements}
    scenario_ids = {item.scenario_id for item in bundle.scenarios}

    id_groups = [
        [item.requirement_id for item in bundle.requirements],
        [item.scenario_id for item in bundle.scenarios],
        [item.test_case_id for item in bundle.test_cases],
    ]
    for duplicate in set().union(*(_duplicates(group) for group in id_groups)):
        issues.append(ValidationIssue(code="duplicate_id", artifact_id=duplicate, detail="ID is not unique."))

    for scenario in bundle.scenarios:
        for requirement_id in scenario.requirement_ids:
            if requirement_id not in requirement_ids:
                issues.append(
                    ValidationIssue(
                        code="missing_requirement",
                        artifact_id=scenario.scenario_id,
                        detail=f"Unknown requirement {requirement_id}.",
                    )
                )

    for test_case in bundle.test_cases:
        if test_case.scenario_id not in scenario_ids:
            issues.append(
                ValidationIssue(
                    code="missing_scenario",
                    artifact_id=test_case.test_case_id,
                    detail=f"Unknown scenario {test_case.scenario_id}.",
                )
            )
        for requirement_id in test_case.requirement_ids:
            if requirement_id not in requirement_ids:
                issues.append(
                    ValidationIssue(
                        code="missing_requirement",
                        artifact_id=test_case.test_case_id,
                        detail=f"Unknown requirement {requirement_id}.",
                    )
                )
        expected_steps = list(range(1, len(test_case.steps) + 1))
        if [step.step_number for step in test_case.steps] != expected_steps:
            issues.append(
                ValidationIssue(
                    code="step_sequence",
                    artifact_id=test_case.test_case_id,
                    detail="Step numbers must be consecutive from 1.",
                )
            )

    for artifact in [*bundle.requirements, *bundle.scenarios, *bundle.test_cases]:
        artifact_id = getattr(
            artifact,
            "requirement_id",
            getattr(artifact, "scenario_id", getattr(artifact, "test_case_id", "")),
        )
        for reference in artifact.source_references:
            if not verify_source_reference(reference, chunks):
                issues.append(
                    ValidationIssue(
                        code="invalid_source",
                        artifact_id=artifact_id,
                        detail=f"Invalid source reference {reference.chunk_id}.",
                    )
                )
    return issues


def build_rtm(bundle: ArtifactBundle) -> list[RTMRow]:
    rows: list[RTMRow] = []
    for requirement in bundle.requirements:
        scenario_ids = sorted(
            scenario.scenario_id
            for scenario in bundle.scenarios
            if requirement.requirement_id in scenario.requirement_ids
        )
        test_case_ids = sorted(
            test_case.test_case_id
            for test_case in bundle.test_cases
            if requirement.requirement_id in test_case.requirement_ids
        )
        status = "covered" if scenario_ids and test_case_ids else "uncovered"
        rows.append(
            RTMRow(
                requirement_id=requirement.requirement_id,
                scenario_ids=scenario_ids,
                test_case_ids=test_case_ids,
                coverage_status=status,
            )
        )
    return rows
```

- [ ] **Step 5: Add versioned prompt builders**

Create `src/brd_srs_testgen/prompts.py`:

```python
from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel

from .documents import render_chunks
from .models import DocumentChunk


PROMPT_VERSION = "2026-08-10.v1"
GROUNDING_RULE = """
Use only the supplied document evidence. Every artifact must cite chunk_id,
page_number, section, and a short verbatim excerpt. Do not invent behavior.
Return only data matching the response schema.
""".strip()


def _json(value: BaseModel | list[BaseModel]) -> str:
    if isinstance(value, list):
        return "[" + ",".join(item.model_dump_json() for item in value) + "]"
    return value.model_dump_json()


def overview_prompt(chunks: Iterable[DocumentChunk]) -> str:
    return f"""You are a business analyst. Summarize the system purpose, actors,
modules, and explicit business rules. {GROUNDING_RULE}\n\n{render_chunks(chunks)}"""


def requirements_prompt(chunks: Iterable[DocumentChunk], overview: BaseModel) -> str:
    return f"""You are a requirements analyst. Exhaustively extract functional,
non-functional, and business requirements. Assign stable REQ-### IDs and retain
ambiguities instead of guessing.\n{GROUNDING_RULE}\nOVERVIEW:\n{_json(overview)}
\nEVIDENCE:\n{render_chunks(chunks)}"""


def scenarios_prompt(
    requirements: list[BaseModel], chunks: Iterable[DocumentChunk]
) -> str:
    return f"""You are a QA lead. Generate positive, negative, boundary, edge,
and state-transition scenarios where supported. Link every scenario to one or
more requirement IDs.\n{GROUNDING_RULE}\nREQUIREMENTS:\n{_json(requirements)}
\nEVIDENCE:\n{render_chunks(chunks)}"""


def test_cases_prompt(
    requirements: list[BaseModel], scenarios: list[BaseModel], chunks: Iterable[DocumentChunk]
) -> str:
    return f"""You are a manual test designer. Turn every scenario into executable
manual test cases with preconditions, concrete data, ordered actions, and a
measurable expected result for each step.\n{GROUNDING_RULE}
\nREQUIREMENTS:\n{_json(requirements)}\nSCENARIOS:\n{_json(scenarios)}
\nEVIDENCE:\n{render_chunks(chunks)}"""


def review_prompt(bundle: BaseModel, chunks: Iterable[DocumentChunk]) -> str:
    return f"""You are the centralized verifier. Identify unsupported claims,
missing coverage, invalid logic, or duplicate intent. Accept only when no
semantic correction is needed.\n{GROUNDING_RULE}\nBUNDLE:\n{_json(bundle)}
\nEVIDENCE:\n{render_chunks(chunks)}"""


def revision_prompt(
    bundle: BaseModel, review: BaseModel, chunks: Iterable[DocumentChunk]
) -> str:
    return f"""Revise only the listed issues and return the full corrected bundle.
Do not change accepted artifacts.\n{GROUNDING_RULE}\nBUNDLE:\n{_json(bundle)}
\nISSUES:\n{_json(review)}\nEVIDENCE:\n{render_chunks(chunks)}"""


def single_prompt(chunks: Iterable[DocumentChunk]) -> str:
    return f"""In one response, extract requirements, design test scenarios,
write detailed manual test cases, and link every artifact to source evidence.
{GROUNDING_RULE}\n\n{render_chunks(chunks)}"""


def reconcile_requirements_prompt(candidates: list[BaseModel]) -> str:
    return f"""You are the centralized supervisor. Merge duplicate candidate
requirements, preserve distinct requirements, normalize IDs to REQ-###, and
retain all valid source references. Return the reconciled requirement list.
{GROUNDING_RULE}\nCANDIDATES:\n{_json(candidates)}"""
```

- [ ] **Step 6: Run validation tests**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_validation.py -q
```

Expected: `2 passed`.

- [ ] **Step 7: Commit prompts and deterministic validation**

```bash
rtk git add src/brd_srs_testgen/models.py src/brd_srs_testgen/prompts.py src/brd_srs_testgen/validation.py tests/test_validation.py
rtk git commit -m "feat: add grounded prompts and RTM checks"
```

### Task 6: Implement the single-prompt and staged single-agent conditions

**Files:**
- Create: `src/brd_srs_testgen/pipelines.py`
- Create: `tests/test_pipelines.py`

- [ ] **Step 1: Write failing controlled-pipeline tests**

Create `tests/test_pipelines.py` with a queue-backed fake gateway:

```python
from collections import deque
from types import SimpleNamespace

from brd_srs_testgen.models import (
    ArtifactBundle,
    DocumentChunk,
    DocumentOverview,
    RequirementBatch,
    ReviewResult,
    ScenarioBatch,
    TestCaseBatch,
)
from brd_srs_testgen.pipelines import PipelineContext, run_single_prompt, run_staged_single
from tests.test_validation import fixture_bundle


class FakeGateway:
    def __init__(self, values):
        self.values = deque(values)
        self.calls: list[type] = []

    def generate(self, _prompt, schema, max_output_tokens, temperature=0.0):
        self.calls.append(schema)
        return SimpleNamespace(value=self.values.popleft(), total_tokens=1)


def context(gateway) -> PipelineContext:
    _, chunks = fixture_bundle()
    return PipelineContext(chunks=chunks, gateway=gateway)


def test_single_prompt_uses_exactly_one_generation_call() -> None:
    bundle, _ = fixture_bundle()
    gateway = FakeGateway([bundle])

    assert run_single_prompt(context(gateway)) == bundle
    assert gateway.calls == [ArtifactBundle]


def test_staged_single_agent_uses_shared_state_and_one_review() -> None:
    bundle, _ = fixture_bundle()
    gateway = FakeGateway(
        [
            DocumentOverview(purpose="Test", actors=[], modules=["Login"], business_rules=[]),
            RequirementBatch(requirements=bundle.requirements),
            ScenarioBatch(scenarios=bundle.scenarios),
            TestCaseBatch(test_cases=bundle.test_cases),
            ReviewResult(accepted=True),
        ]
    )

    result = run_staged_single(context(gateway))

    assert result == bundle
    assert gateway.calls[-1] is ReviewResult


def test_staged_pipeline_reuses_completed_checkpoints() -> None:
    bundle, chunks = fixture_bundle()
    stored: dict[str, dict] = {}
    first_gateway = FakeGateway(
        [
            DocumentOverview(purpose="Test", actors=[], modules=["Login"], business_rules=[]),
            RequirementBatch(requirements=bundle.requirements),
            ScenarioBatch(scenarios=bundle.scenarios),
            TestCaseBatch(test_cases=bundle.test_cases),
            ReviewResult(accepted=True),
        ]
    )
    first_context = PipelineContext(
        chunks=chunks,
        gateway=first_gateway,
        checkpoint_reader=stored.get,
        checkpoint_writer=stored.__setitem__,
    )
    assert run_staged_single(first_context) == bundle

    resumed_gateway = FakeGateway([])
    resumed_context = PipelineContext(
        chunks=chunks,
        gateway=resumed_gateway,
        checkpoint_reader=stored.get,
        checkpoint_writer=stored.__setitem__,
    )
    assert run_staged_single(resumed_context) == bundle
    assert resumed_gateway.calls == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_pipelines.py -q
```

Expected: collection fails because `PipelineContext` and the pipeline functions do not exist.

- [ ] **Step 3: Implement pipeline context and the first two conditions**

Create `src/brd_srs_testgen/pipelines.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from .llm import GeminiGateway
from .models import (
    ArtifactBundle,
    DocumentChunk,
    DocumentOverview,
    RequirementBatch,
    ReviewResult,
    ScenarioBatch,
    TestCaseBatch,
)


T = TypeVar("T", bound=BaseModel)
from .prompts import (
    overview_prompt,
    requirements_prompt,
    review_prompt,
    revision_prompt,
    scenarios_prompt,
    single_prompt,
    test_cases_prompt,
)


@dataclass
class PipelineContext:
    chunks: list[DocumentChunk]
    gateway: GeminiGateway
    event_sink: Callable[[dict[str, Any]], None] = field(default=lambda _event: None)
    checkpoint_reader: Callable[[str], Any | None] = field(default=lambda _name: None)
    checkpoint_writer: Callable[[str, Any], None] = field(
        default=lambda _name, _value: None
    )

    def event(self, stage: str, **values: Any) -> None:
        self.event_sink({"stage": stage, **values})

    def checkpoint(
        self, name: str, schema: type[T], producer: Callable[[], T]
    ) -> T:
        stored = self.checkpoint_reader(name)
        if stored is not None:
            self.event("checkpoint_loaded", name=name)
            return schema.model_validate(stored)
        value = producer()
        self.checkpoint_writer(name, value.model_dump(mode="json"))
        self.event("checkpoint_written", name=name)
        return value


def run_single_prompt(context: PipelineContext) -> ArtifactBundle:
    context.event("single_prompt_started")
    result = context.checkpoint(
        "single_bundle",
        ArtifactBundle,
        lambda: context.gateway.generate(
            single_prompt(context.chunks), ArtifactBundle, max_output_tokens=24_000
        ).value,
    )
    context.event("single_prompt_finished")
    return result


def run_staged_single(context: PipelineContext) -> ArtifactBundle:
    overview = context.checkpoint(
        "staged_overview",
        DocumentOverview,
        lambda: context.gateway.generate(
            overview_prompt(context.chunks), DocumentOverview, max_output_tokens=2_000
        ).value,
    )
    requirements = context.checkpoint(
        "staged_requirements",
        RequirementBatch,
        lambda: context.gateway.generate(
            requirements_prompt(context.chunks, overview),
            RequirementBatch,
            max_output_tokens=8_000,
        ).value,
    ).requirements
    scenarios = context.checkpoint(
        "staged_scenarios",
        ScenarioBatch,
        lambda: context.gateway.generate(
            scenarios_prompt(requirements, context.chunks),
            ScenarioBatch,
            max_output_tokens=8_000,
        ).value,
    ).scenarios
    test_cases = context.checkpoint(
        "staged_test_cases",
        TestCaseBatch,
        lambda: context.gateway.generate(
            test_cases_prompt(requirements, scenarios, context.chunks),
            TestCaseBatch,
            max_output_tokens=12_000,
        ).value,
    ).test_cases
    bundle = ArtifactBundle(
        requirements=requirements, scenarios=scenarios, test_cases=test_cases
    )
    review = context.checkpoint(
        "staged_review",
        ReviewResult,
        lambda: context.gateway.generate(
            review_prompt(bundle, context.chunks), ReviewResult, max_output_tokens=3_000
        ).value,
    )
    if not review.accepted:
        bundle = context.checkpoint(
            "staged_revision",
            ArtifactBundle,
            lambda: context.gateway.generate(
                revision_prompt(bundle, review, context.chunks),
                ArtifactBundle,
                max_output_tokens=12_000,
            ).value,
        )
    return bundle
```

The single semantic revision is explicit and cannot loop.

- [ ] **Step 4: Run controlled-pipeline tests**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_pipelines.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit the reference and staged conditions**

```bash
rtk git add src/brd_srs_testgen/pipelines.py tests/test_pipelines.py
rtk git commit -m "feat: add reference and staged pipelines"
```

### Task 7: Implement centralized task-aware parallelism

**Files:**
- Modify: `src/brd_srs_testgen/pipelines.py`
- Modify: `tests/test_pipelines.py`

- [ ] **Step 1: Add failing partition and centralized-pipeline tests**

Append to `tests/test_pipelines.py`:

```python
from brd_srs_testgen.pipelines import partition_balanced, run_centralized_multi


def test_partition_balanced_preserves_items_across_three_workers() -> None:
    groups = partition_balanced(list(range(8)), workers=3)

    assert sorted(item for group in groups for item in group) == list(range(8))
    assert max(map(len, groups)) - min(map(len, groups)) <= 1


def test_centralized_pipeline_merges_worker_outputs_and_reviews_once() -> None:
    bundle, _ = fixture_bundle()
    overview = DocumentOverview(purpose="Test", actors=[], modules=["Login"], business_rules=[])
    worker_case = bundle.test_cases[0].model_copy(
        update={"scenario_id": "SCN-W1-SCN-001"}
    )
    gateway = FakeGateway(
        [
            overview,
            RequirementBatch(requirements=bundle.requirements),
            RequirementBatch(requirements=[]),
            RequirementBatch(requirements=[]),
            RequirementBatch(requirements=bundle.requirements),
            ScenarioBatch(scenarios=bundle.scenarios),
            ScenarioBatch(scenarios=[]),
            ScenarioBatch(scenarios=[]),
            TestCaseBatch(test_cases=[worker_case]),
            TestCaseBatch(test_cases=[]),
            TestCaseBatch(test_cases=[]),
            ReviewResult(accepted=True),
        ]
    )

    result = run_centralized_multi(context(gateway), executor_workers=1)

    assert result.requirements == bundle.requirements
    assert result.scenarios[0].scenario_id == "SCN-W1-SCN-001"
    assert result.test_cases[0].scenario_id == "SCN-W1-SCN-001"
    assert result.test_cases[0].test_case_id == "TC-W1-TC-001"
    assert gateway.calls.count(RequirementBatch) == 4
```

Use `executor_workers=1` in this deterministic unit test; production passes `3`.

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_pipelines.py -q
```

Expected: import fails for `partition_balanced` and `run_centralized_multi`.

- [ ] **Step 3: Add balanced partitioning and concurrent worker execution**

Append to `src/brd_srs_testgen/pipelines.py`:

```python
from concurrent.futures import ThreadPoolExecutor
from itertools import chain

from .prompts import reconcile_requirements_prompt


def partition_balanced(items: list[Any], workers: int = 3) -> list[list[Any]]:
    groups = [[] for _ in range(workers)]
    for index, item in enumerate(items):
        groups[index % workers].append(item)
    return groups


def _parallel(executor_workers: int, function, groups):
    with ThreadPoolExecutor(max_workers=executor_workers) as executor:
        return list(executor.map(function, groups))


def run_centralized_multi(
    context: PipelineContext, executor_workers: int = 3
) -> ArtifactBundle:
    overview = context.checkpoint(
        "multi_overview",
        DocumentOverview,
        lambda: context.gateway.generate(
            overview_prompt(context.chunks), DocumentOverview, max_output_tokens=2_000
        ).value,
    )
    chunk_groups = partition_balanced(context.chunks)

    def extract(index_and_group):
        worker_index, group = index_and_group
        if not group:
            return RequirementBatch(requirements=[])
        return context.checkpoint(
            f"multi_requirement_candidates_{worker_index + 1}",
            RequirementBatch,
            lambda: context.gateway.generate(
                requirements_prompt(group, overview),
                RequirementBatch,
                max_output_tokens=4_000,
            ).value,
        )

    candidate_batches = _parallel(
        executor_workers, extract, list(enumerate(chunk_groups))
    )
    candidates = list(chain.from_iterable(batch.requirements for batch in candidate_batches))
    requirements = context.checkpoint(
        "multi_requirements",
        RequirementBatch,
        lambda: context.gateway.generate(
            reconcile_requirements_prompt(candidates),
            RequirementBatch,
            max_output_tokens=8_000,
        ).value,
    ).requirements
    requirement_groups = partition_balanced(requirements)

    def generate_scenarios(index_and_group):
        worker_index, group = index_and_group
        if not group:
            return ScenarioBatch(scenarios=[])
        chunk_ids = {
            reference.chunk_id for requirement in group for reference in requirement.source_references
        }
        evidence = [chunk for chunk in context.chunks if chunk.chunk_id in chunk_ids]
        def produce() -> ScenarioBatch:
            batch = context.gateway.generate(
                scenarios_prompt(group, evidence),
                ScenarioBatch,
                max_output_tokens=5_000,
            ).value
            return ScenarioBatch(
                scenarios=[
                    scenario.model_copy(
                        update={
                            "scenario_id": f"SCN-W{worker_index + 1}-{scenario.scenario_id}"
                        }
                    )
                    for scenario in batch.scenarios
                ]
            )

        return context.checkpoint(
            f"multi_scenarios_{worker_index + 1}", ScenarioBatch, produce
        )

    scenario_batches = _parallel(
        executor_workers, generate_scenarios, list(enumerate(requirement_groups))
    )
    scenarios = list(chain.from_iterable(batch.scenarios for batch in scenario_batches))
    scenario_groups = [
        [scenario for scenario in scenarios if set(scenario.requirement_ids) & {item.requirement_id for item in group}]
        for group in requirement_groups
    ]

    def generate_test_cases(group_index):
        group = requirement_groups[group_index]
        group_scenarios = scenario_groups[group_index]
        if not group_scenarios:
            return TestCaseBatch(test_cases=[])
        chunk_ids = {
            reference.chunk_id for requirement in group for reference in requirement.source_references
        }
        evidence = [chunk for chunk in context.chunks if chunk.chunk_id in chunk_ids]
        def produce() -> TestCaseBatch:
            batch = context.gateway.generate(
                test_cases_prompt(group, group_scenarios, evidence),
                TestCaseBatch,
                max_output_tokens=7_000,
            ).value
            return TestCaseBatch(
                test_cases=[
                    test_case.model_copy(
                        update={
                            "test_case_id": f"TC-W{group_index + 1}-{test_case.test_case_id}"
                        }
                    )
                    for test_case in batch.test_cases
                ]
            )

        return context.checkpoint(
            f"multi_test_cases_{group_index + 1}", TestCaseBatch, produce
        )

    test_case_batches = _parallel(
        executor_workers, generate_test_cases, list(range(len(requirement_groups)))
    )
    test_cases = list(chain.from_iterable(batch.test_cases for batch in test_case_batches))
    bundle = ArtifactBundle(
        requirements=requirements, scenarios=scenarios, test_cases=test_cases
    )
    review = context.checkpoint(
        "multi_review",
        ReviewResult,
        lambda: context.gateway.generate(
            review_prompt(bundle, context.chunks), ReviewResult, max_output_tokens=3_000
        ).value,
    )
    if not review.accepted:
        bundle = context.checkpoint(
            "multi_revision",
            ArtifactBundle,
            lambda: context.gateway.generate(
                revision_prompt(bundle, review, context.chunks),
                ArtifactBundle,
                max_output_tokens=12_000,
            ).value,
        )
    return bundle
```

The three worker groups are isolated. Only the supervisor receives merged candidate outputs. The shared `BudgetLedger` reserves tokens under a lock, so concurrent calls cannot overspend.

- [ ] **Step 4: Run all pipeline tests**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_pipelines.py -q
```

Expected: `5 passed`.

- [ ] **Step 5: Commit centralized multi-agent execution**

```bash
rtk git add src/brd_srs_testgen/pipelines.py tests/test_pipelines.py
rtk git commit -m "feat: add centralized worker pipeline"
```

### Task 8: Compute deterministic RTM, coverage, duplicate, and cost metrics

**Files:**
- Create: `src/brd_srs_testgen/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing metric tests**

Create `tests/test_metrics.py`:

```python
from brd_srs_testgen.metrics import compute_metrics, duplicate_rate
from tests.test_validation import fixture_bundle


def test_metrics_measure_structure_without_claiming_semantic_correctness() -> None:
    bundle, chunks = fixture_bundle()

    metrics = compute_metrics(
        bundle,
        chunks,
        status="completed",
        total_tokens=120,
        latency_seconds=2.5,
        retries=0,
        revisions=0,
    )

    assert metrics["requirement_coverage"] == 1.0
    assert metrics["positive_coverage"] == 1.0
    assert metrics["rtm_completeness"] == 1.0
    assert metrics["invalid_source_rate"] == 0.0
    assert metrics["total_tokens"] == 120


def test_duplicate_rate_uses_preregistered_trigram_threshold() -> None:
    cases = [
        "Login with valid email and password then open dashboard",
        "Login with valid email and password then open dashboard",
        "Reset an expired password token",
    ]

    assert duplicate_rate(cases) == 1 / 3
```

- [ ] **Step 2: Run metric tests to verify they fail**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_metrics.py -q
```

Expected: collection fails because `brd_srs_testgen.metrics` does not exist.

- [ ] **Step 3: Implement preregistered automated metrics**

Create `src/brd_srs_testgen/metrics.py`:

```python
from __future__ import annotations

import re
from itertools import combinations

from .models import ArtifactBundle, DocumentChunk, ScenarioType
from .validation import build_rtm, validate_bundle


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _trigrams(text: str) -> set[tuple[str, str, str]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return set(zip(words, words[1:], words[2:]))


def duplicate_rate(case_texts: list[str], threshold: float = 0.85) -> float:
    if not case_texts:
        return 0.0
    duplicate_indexes: set[int] = set()
    grams = [_trigrams(text) for text in case_texts]
    for (left_index, left), (right_index, right) in combinations(enumerate(grams), 2):
        union = left | right
        similarity = len(left & right) / len(union) if union else 1.0
        if similarity >= threshold:
            duplicate_indexes.add(right_index)
    # ponytail: lexical duplicate heuristic; use human semantic labels if this becomes a primary outcome.
    return len(duplicate_indexes) / len(case_texts)


def compute_metrics(
    bundle: ArtifactBundle,
    chunks: list[DocumentChunk],
    *,
    status: str,
    total_tokens: int,
    latency_seconds: float,
    retries: int,
    revisions: int,
) -> dict[str, float | int | str]:
    requirement_count = len(bundle.requirements)
    rows = build_rtm(bundle)
    covered = sum(row.coverage_status == "covered" for row in rows)
    scenario_requirement_ids = {
        requirement_id for scenario in bundle.scenarios for requirement_id in scenario.requirement_ids
    }
    coverage_by_type = {
        scenario_type: {
            requirement_id
            for scenario in bundle.scenarios
            if scenario.scenario_type == scenario_type
            for requirement_id in scenario.requirement_ids
        }
        for scenario_type in ScenarioType
    }
    references = [
        reference
        for artifact in [*bundle.requirements, *bundle.scenarios, *bundle.test_cases]
        for reference in artifact.source_references
    ]
    invalid_sources = sum(
        issue.code == "invalid_source" for issue in validate_bundle(bundle, chunks)
    )
    case_texts = [
        " ".join(
            [test_case.title]
            + [f"{step.action} {step.expected_result}" for step in test_case.steps]
        )
        for test_case in bundle.test_cases
    ]
    return {
        "status": status,
        "requirement_coverage": _ratio(covered, requirement_count),
        "scenario_coverage": _ratio(len(scenario_requirement_ids), requirement_count),
        "positive_coverage": _ratio(
            len(coverage_by_type[ScenarioType.POSITIVE]), requirement_count
        ),
        "negative_coverage": _ratio(
            len(coverage_by_type[ScenarioType.NEGATIVE]), requirement_count
        ),
        "boundary_coverage": _ratio(
            len(coverage_by_type[ScenarioType.BOUNDARY]), requirement_count
        ),
        "edge_coverage": _ratio(len(coverage_by_type[ScenarioType.EDGE]), requirement_count),
        "state_transition_coverage": _ratio(
            len(coverage_by_type[ScenarioType.STATE_TRANSITION]), requirement_count
        ),
        "rtm_completeness": _ratio(covered, requirement_count),
        "invalid_source_rate": _ratio(invalid_sources, len(references)),
        "duplicate_rate": duplicate_rate(case_texts),
        "requirements": requirement_count,
        "scenarios": len(bundle.scenarios),
        "test_cases": len(bundle.test_cases),
        "total_tokens": total_tokens,
        "latency_seconds": latency_seconds,
        "retries": retries,
        "revisions": revisions,
    }
```

- [ ] **Step 4: Run metric tests**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_metrics.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit automated metrics**

```bash
rtk git add src/brd_srs_testgen/metrics.py tests/test_metrics.py
rtk git commit -m "feat: add preregistered run metrics"
```

### Task 9: Enforce bounded transport retries and schema repair

**Files:**
- Modify: `src/brd_srs_testgen/llm.py`
- Modify: `src/brd_srs_testgen/pipelines.py`
- Modify: `tests/test_llm.py`
- Modify: `tests/test_pipelines.py`

- [ ] **Step 1: Add failing policy tests**

Append to `tests/test_llm.py`:

```python
from brd_srs_testgen.llm import StructuredOutputError


class InvalidModels(FakeModels):
    def generate_content(self, **_kwargs):
        return SimpleNamespace(
            text="not-json",
            usage_metadata=SimpleNamespace(total_token_count=17),
        )


def test_invalid_structured_output_still_consumes_budget() -> None:
    ledger = BudgetLedger(limit=100)
    gateway = GeminiGateway(
        client=SimpleNamespace(models=InvalidModels()), model="gemini-test", ledger=ledger
    )

    with pytest.raises(StructuredOutputError):
        gateway.generate("Summarize", DocumentOverview, max_output_tokens=40)

    assert ledger.used == 17
```

Append to `tests/test_pipelines.py`:

```python
from brd_srs_testgen.llm import StructuredOutputError


class RepairingGateway(FakeGateway):
    def __init__(self, repaired):
        super().__init__([repaired])
        self.failed = False

    def generate(self, prompt, schema, max_output_tokens, temperature=0.0):
        if not self.failed:
            self.failed = True
            raise StructuredOutputError("invalid")
        return super().generate(prompt, schema, max_output_tokens, temperature)


def test_staged_condition_repairs_schema_once() -> None:
    overview = DocumentOverview(purpose="Test", actors=[], modules=[], business_rules=[])
    gateway = RepairingGateway(overview)
    pipeline_context = context(gateway)

    try:
        run_staged_single(pipeline_context)
    except IndexError:
        pass

    assert pipeline_context.schema_repairs == 1
```

- [ ] **Step 2: Run policy tests to verify they fail**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_llm.py tests/test_pipelines.py -q
```

Expected: import fails for `StructuredOutputError` or the repair counter assertion fails.

- [ ] **Step 3: Count invalid responses before raising**

In `src/brd_srs_testgen/llm.py`, add the exception:

```python
class StructuredOutputError(ValueError):
    def __init__(self, raw_text: str) -> None:
        super().__init__("Model returned invalid structured output.")
        self.raw_text = raw_text
```

Replace `GeminiGateway.generate` with the complete method below so malformed responses consume their actual tokens while transport failures release reservations:

```python
def generate(
    self,
    prompt: str,
    schema: type[T],
    max_output_tokens: int,
    temperature: float = 0.0,
) -> GenerationResult[T]:
    input_tokens = int(
        self.client.models.count_tokens(model=self.model, contents=prompt).total_tokens
    )
    reservation = self.ledger.reserve(input_tokens + max_output_tokens)
    settled = False
    try:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        total = int(response.usage_metadata.total_token_count)
        self.ledger.settle(reservation, total)
        settled = True
        try:
            value = schema.model_validate(json.loads(response.text))
        except Exception as error:
            raise StructuredOutputError(response.text) from error
        return GenerationResult(value=value, total_tokens=total)
    except Exception:
        if not settled:
            self.ledger.cancel(reservation)
        raise
```

- [ ] **Step 4: Centralize retry and repair policy in PipelineContext**

Add these imports at module scope in `src/brd_srs_testgen/pipelines.py`:

```python
import time

from google.genai import errors

from .llm import StructuredOutputError
```

Then add these fields and methods inside `PipelineContext`:

```python
sleep: Callable[[float], None] = time.sleep
retries: int = 0
schema_repairs: int = 0
semantic_revisions: int = 0
usage_sink: Callable[[int], None] = field(default=lambda _used: None)

def _record_usage(self) -> None:
    ledger = getattr(self.gateway, "ledger", None)
    if ledger is not None:
        self.usage_sink(ledger.used)

def generate(
    self,
    prompt: str,
    schema,
    max_output_tokens: int,
    *,
    allow_schema_repair: bool = True,
):
    transport_attempt = 0
    repaired = False
    while True:
        try:
            result = self.gateway.generate(
                prompt, schema, max_output_tokens=max_output_tokens, temperature=0.0
            )
            self._record_usage()
            return result.value
        except errors.APIError as error:
            retryable = getattr(error, "code", None) in {429, 500, 502, 503, 504}
            if not retryable or transport_attempt >= 2:
                raise
            transport_attempt += 1
            self.retries += 1
            self.sleep(2 ** (transport_attempt - 1))
        except StructuredOutputError as error:
            self._record_usage()
            if not allow_schema_repair or repaired:
                raise
            repaired = True
            self.schema_repairs += 1
            prompt = (
                "Return valid JSON matching the requested schema. Preserve supported content "
                f"from this invalid response and add nothing else:\n{error.raw_text}"
            )
```

In `run_single_prompt`, `run_staged_single`, and `run_centralized_multi`, replace each direct `context.gateway.generate` invocation with `context.generate`, preserving its prompt, schema, and token limit, and remove the trailing `.value`. Pass `allow_schema_repair=False` only in `run_single_prompt`. Immediately before each verifier-directed `revision_prompt` call, increment:

```python
context.semantic_revisions += 1
```

- [ ] **Step 5: Run the provider and pipeline policy tests**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_llm.py tests/test_pipelines.py -q
```

Expected: all tests pass, including one counted invalid response and exactly one schema repair.

- [ ] **Step 6: Commit bounded recovery policy**

```bash
rtk git add src/brd_srs_testgen/llm.py src/brd_srs_testgen/pipelines.py tests/test_llm.py tests/test_pipelines.py
rtk git commit -m "feat: bound model retries and repairs"
```

### Task 10: Build the frozen 54-run experiment manifest and runner

**Files:**
- Modify: `src/brd_srs_testgen/models.py`
- Modify: `src/brd_srs_testgen/storage.py`
- Create: `src/brd_srs_testgen/experiments.py`
- Create: `tests/test_experiments.py`

- [ ] **Step 1: Write failing manifest and resume tests**

Create `tests/test_experiments.py`:

```python
from types import SimpleNamespace

import pytest

from brd_srs_testgen import experiments
from brd_srs_testgen.experiments import build_manifest, execute_interactive, execute_run
from brd_srs_testgen.models import Condition, DocumentOverview, DocumentRecord
from brd_srs_testgen.storage import CompletedRunError, RunStore
from tests.test_validation import fixture_bundle


def documents() -> list[DocumentRecord]:
    pages = [10, 15, 20, 60, 80, 120]
    return [
        DocumentRecord(
            name=f"doc-{index}.pdf",
            content_hash=f"hash-{index}",
            page_count=page_count,
            length_class=(
                "short" if page_count <= 15 else "medium" if page_count <= 60 else "long"
            ),
        )
        for index, page_count in enumerate(pages)
    ]


def test_manifest_contains_balanced_randomized_54_runs() -> None:
    manifest = build_manifest(
        documents(), model="gemini-test", token_budget=100_000, seed=20260810
    )

    assert len(manifest.runs) == 54
    for document in documents():
        selected = [run for run in manifest.runs if run.document_hash == document.content_hash]
        assert len(selected) == 9
        assert {run.condition for run in selected} == set(Condition)
    assert manifest.seed == 20260810
    assert len(manifest.prompt_hash) == 64
    assert len(manifest.schema_hash) == 64
    assert manifest.software_version == "0.1.0"


def test_manifest_rejects_wrong_length_strata() -> None:
    invalid = documents()[:-1]

    try:
        build_manifest(invalid, model="gemini-test", token_budget=100_000, seed=20260810)
    except ValueError as error:
        assert "two short, two medium, and two long" in str(error)
    else:
        raise AssertionError("Invalid strata were accepted")


def test_execute_run_resumes_only_missing_stages_and_prior_usage(tmp_path, monkeypatch) -> None:
    manifest = build_manifest(
        documents(), model="gemini-test", token_budget=100_000, seed=20260810
    )
    definition = manifest.runs[0]
    bundle, chunks = fixture_bundle()
    store = RunStore(tmp_path)
    store.create(
        definition.run_id,
        {
            **definition.model_dump(mode="json"),
            "status": "failed",
            "total_tokens": 30,
            "latency_seconds": 2.0,
        },
    )
    stored_overview = DocumentOverview(
        purpose="Stored", actors=[], modules=[], business_rules=[]
    )
    store.write_artifact(
        definition.run_id, "staged_overview", stored_overview.model_dump(mode="json")
    )

    def resumed_pipeline(context):
        loaded = context.checkpoint(
            "staged_overview",
            DocumentOverview,
            lambda: (_ for _ in ()).throw(AssertionError("checkpoint regenerated")),
        )
        assert loaded == stored_overview
        return bundle

    monkeypatch.setattr(experiments, "parse_pdf", lambda _data: chunks)
    monkeypatch.setitem(experiments.PIPELINES, definition.condition, resumed_pipeline)

    def gateway_factory(ledger):
        assert ledger.used == 30
        return SimpleNamespace(ledger=ledger)

    assert execute_run(definition, b"pdf", manifest, store, gateway_factory) == "completed"
    assert store.read_artifact(definition.run_id, "staged_overview")["purpose"] == "Stored"
    assert store.status(definition.run_id) == "completed"


def test_frozen_experiment_manifest_cannot_change_in_place(tmp_path) -> None:
    manifest = build_manifest(
        documents(), model="gemini-test", token_budget=100_000, seed=20260810
    )
    store = RunStore(tmp_path)
    store.save_experiment(manifest.experiment_id, manifest.model_dump(mode="json"))
    store.save_experiment(manifest.experiment_id, manifest.model_dump(mode="json"))

    changed = manifest.model_dump(mode="json")
    changed["token_budget"] += 1
    with pytest.raises(CompletedRunError):
        store.save_experiment(manifest.experiment_id, changed)


def test_interactive_generation_uses_the_same_persisted_runner(tmp_path, monkeypatch) -> None:
    _, chunks = fixture_bundle()
    captured = {}
    monkeypatch.setattr(experiments, "parse_pdf", lambda _data: chunks)

    def fake_execute(definition, pdf_bytes, manifest, store, gateway_factory):
        captured.update(definition=definition, manifest=manifest, pdf_bytes=pdf_bytes)
        return "completed"

    monkeypatch.setattr(experiments, "execute_run", fake_execute)
    run_id, status = execute_interactive(
        b"pdf",
        "pilot.pdf",
        Condition.STAGED_SINGLE_AGENT,
        model="gemini-test",
        token_budget=100_000,
        store=RunStore(tmp_path),
        gateway_factory=lambda ledger: SimpleNamespace(ledger=ledger),
    )

    assert status == "completed"
    assert captured["definition"].run_id == run_id
    assert captured["manifest"].model == "gemini-test"
    assert captured["pdf_bytes"] == b"pdf"
```

- [ ] **Step 2: Run experiment tests to verify they fail**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_experiments.py -q
```

Expected: import fails because the experiment models and module do not exist.

- [ ] **Step 3: Add experiment manifest models**

Append to `src/brd_srs_testgen/models.py`:

```python
class DocumentRecord(StrictModel):
    name: str
    content_hash: str
    page_count: int = Field(gt=0)
    length_class: str


class RunDefinition(StrictModel):
    run_id: str
    document_hash: str
    condition: Condition
    repetition: int = Field(ge=1, le=3)
    order: int = Field(ge=1, le=3)


class ExperimentManifest(StrictModel):
    experiment_id: str
    model: str
    temperature: float
    token_budget: int
    seed: int
    prompt_version: str
    prompt_hash: str
    schema_hash: str
    software_version: str
    documents: list[DocumentRecord]
    runs: list[RunDefinition]
```

- [ ] **Step 4: Add resumable run-state and token persistence**

Add `import threading`, create `self._lock = threading.Lock()` in `RunStore.__init__`, and add:

```python
def status(self, run_id: str) -> str:
    return str(self._manifest(run_id).get("status"))

def mark_running(self, run_id: str, started_at: str) -> None:
    manifest = self._manifest(run_id)
    manifest.update(status="running", resumed_at=started_at)
    manifest.pop("failure_category", None)
    manifest.pop("failure_detail", None)
    _atomic_json(self._directory(run_id) / "manifest.json", manifest)

def record_usage(self, run_id: str, total_tokens: int) -> None:
    with self._lock:
        self._ensure_mutable(run_id)
        manifest = self._manifest(run_id)
        manifest["total_tokens"] = max(total_tokens, int(manifest.get("total_tokens", 0)))
        _atomic_json(self._directory(run_id) / "manifest.json", manifest)

def save_experiment(self, experiment_id: str, manifest: dict[str, Any]) -> None:
    if not experiment_id.isalnum():
        raise ValueError("Invalid experiment ID.")
    path = self.root / "experiments" / experiment_id / "manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise CompletedRunError("A frozen experiment manifest cannot be changed.")
        return
    _atomic_json(path, manifest)

def fail(
    self,
    run_id: str,
    category: str,
    detail: str,
    *,
    total_tokens: int,
    latency_seconds: float,
    retries: int,
    revisions: int,
    ended_at: str,
) -> None:
    manifest = self._manifest(run_id)
    manifest.update(
        status="failed",
        failure_category=category,
        failure_detail=detail,
        total_tokens=total_tokens,
        latency_seconds=latency_seconds,
        retries=retries,
        revisions=revisions,
        ended_at=ended_at,
    )
    _atomic_json(self._directory(run_id) / "manifest.json", manifest)
```

Change `complete` to accept the final reproducibility fields while preserving the completed-run immutability guard:

```python
def complete(
    self,
    run_id: str,
    *,
    total_tokens: int = 0,
    latency_seconds: float = 0.0,
    retries: int = 0,
    revisions: int = 0,
    ended_at: str = "",
) -> None:
    manifest = self._manifest(run_id)
    manifest.update(
        status="completed",
        total_tokens=total_tokens,
        latency_seconds=latency_seconds,
        retries=retries,
        revisions=revisions,
        ended_at=ended_at,
    )
    _atomic_json(self._directory(run_id) / "manifest.json", manifest)
```

Also wrap the existing `append_event` body in `with self._lock:` so the three worker threads cannot interleave JSONL writes.

- [ ] **Step 5: Implement deterministic manifest construction**

Create `src/brd_srs_testgen/experiments.py`:

```python
from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, prompts
from .documents import parse_pdf
from .llm import BudgetLedger, GeminiGateway
from .metrics import compute_metrics
from .models import (
    ArtifactBundle,
    Condition,
    DocumentChunk,
    DocumentRecord,
    ExperimentManifest,
    RunDefinition,
)
from .pipelines import (
    PipelineContext,
    run_centralized_multi,
    run_single_prompt,
    run_staged_single,
)
from .prompts import PROMPT_VERSION
from .storage import RunStore
from .validation import build_rtm, validate_bundle


PIPELINES = {
    Condition.SINGLE_PROMPT: run_single_prompt,
    Condition.STAGED_SINGLE_AGENT: run_staged_single,
    Condition.CENTRALIZED_MULTI_AGENT: run_centralized_multi,
}


def configuration_fingerprints() -> tuple[str, str]:
    prompt_hash = hashlib.sha256(Path(prompts.__file__).read_bytes()).hexdigest()
    schema_hash = hashlib.sha256(
        json.dumps(ArtifactBundle.model_json_schema(), sort_keys=True).encode()
    ).hexdigest()
    return prompt_hash, schema_hash


def build_manifest(
    documents: list[DocumentRecord], *, model: str, token_budget: int, seed: int
) -> ExperimentManifest:
    strata = Counter(document.length_class for document in documents)
    if strata != {"short": 2, "medium": 2, "long": 2}:
        raise ValueError("Select exactly two short, two medium, and two long documents.")
    rng = random.Random(seed)
    runs: list[RunDefinition] = []
    for document in sorted(documents, key=lambda item: item.content_hash):
        for repetition in range(1, 4):
            conditions = list(Condition)
            rng.shuffle(conditions)
            for order, condition in enumerate(conditions, 1):
                raw_id = f"{document.content_hash}:{condition}:{repetition}:{seed}"
                run_id = hashlib.sha256(raw_id.encode()).hexdigest()[:16]
                runs.append(
                    RunDefinition(
                        run_id=run_id,
                        document_hash=document.content_hash,
                        condition=condition,
                        repetition=repetition,
                        order=order,
                    )
                )
    prompt_hash, schema_hash = configuration_fingerprints()
    identity = json.dumps(
        {
            "model": model,
            "token_budget": token_budget,
            "seed": seed,
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": prompt_hash,
            "schema_hash": schema_hash,
            "software_version": __version__,
            "documents": [item.model_dump() for item in documents],
        },
        sort_keys=True,
    )
    return ExperimentManifest(
        experiment_id=hashlib.sha256(identity.encode()).hexdigest()[:16],
        model=model,
        temperature=0.0,
        token_budget=token_budget,
        seed=seed,
        prompt_version=PROMPT_VERSION,
        prompt_hash=prompt_hash,
        schema_hash=schema_hash,
        software_version=__version__,
        documents=documents,
        runs=runs,
    )


def execute_run(
    definition: RunDefinition,
    pdf_bytes: bytes,
    manifest: ExperimentManifest,
    store: RunStore,
    gateway_factory: Callable[[BudgetLedger], GeminiGateway],
) -> str:
    run_id = definition.run_id
    if store.exists(run_id) and store.status(run_id) == "completed":
        return "skipped"
    now = datetime.now(timezone.utc).isoformat()
    if store.exists(run_id):
        previous = store.read_manifest(run_id)
        previous_tokens = int(previous.get("total_tokens", 0))
        previous_latency = float(previous.get("latency_seconds", 0.0))
        previous_retries = int(previous.get("retries", 0))
        previous_revisions = int(previous.get("revisions", 0))
        store.mark_running(run_id, now)
    else:
        previous_tokens = 0
        previous_latency = 0.0
        previous_retries = 0
        previous_revisions = 0
        store.create(
            run_id,
            {
                **definition.model_dump(mode="json"),
                "experiment_id": manifest.experiment_id,
                "model": manifest.model,
                "temperature": manifest.temperature,
                "token_budget": manifest.token_budget,
                "prompt_version": manifest.prompt_version,
                "prompt_hash": manifest.prompt_hash,
                "schema_hash": manifest.schema_hash,
                "software_version": manifest.software_version,
                "status": "running",
                "started_at": now,
                "total_tokens": 0,
                "latency_seconds": 0.0,
                "retries": 0,
                "revisions": 0,
            },
        )
    started = time.monotonic()
    ledger = BudgetLedger(manifest.token_budget, used=previous_tokens)
    context: PipelineContext | None = None
    try:
        if store.has_artifact(run_id, "chunks"):
            chunks = [
                DocumentChunk.model_validate(item)
                for item in store.read_artifact(run_id, "chunks")
            ]
        else:
            chunks = parse_pdf(pdf_bytes)
            store.write_artifact(
                run_id, "chunks", [item.model_dump(mode="json") for item in chunks]
            )

        def read_checkpoint(name: str):
            return store.read_artifact(run_id, name) if store.has_artifact(run_id, name) else None

        context = PipelineContext(
            chunks=chunks,
            gateway=gateway_factory(ledger),
            event_sink=lambda event: store.append_event(run_id, event),
            checkpoint_reader=read_checkpoint,
            checkpoint_writer=lambda name, value: store.write_artifact(run_id, name, value),
            usage_sink=lambda used: store.record_usage(run_id, used),
        )
        bundle: ArtifactBundle = PIPELINES[definition.condition](context)
        store.write_artifact(run_id, "requirements", [item.model_dump(mode="json") for item in bundle.requirements])
        store.write_artifact(run_id, "scenarios", [item.model_dump(mode="json") for item in bundle.scenarios])
        store.write_artifact(run_id, "test_cases", [item.model_dump(mode="json") for item in bundle.test_cases])
        store.write_artifact(run_id, "rtm", [item.model_dump(mode="json") for item in build_rtm(bundle)])
        issues = validate_bundle(bundle, chunks)
        latency = previous_latency + time.monotonic() - started
        retries = previous_retries + context.retries
        revisions = previous_revisions + context.semantic_revisions + context.schema_repairs
        metrics = compute_metrics(
            bundle,
            chunks,
            status="completed" if not issues else "failed",
            total_tokens=ledger.used,
            latency_seconds=latency,
            retries=retries,
            revisions=revisions,
        )
        store.write_artifact(run_id, "metrics", metrics)
        if issues:
            store.write_artifact(run_id, "validation_issues", [item.model_dump() for item in issues])
            store.fail(
                run_id,
                "validation",
                f"{len(issues)} deterministic validation issues",
                total_tokens=ledger.used,
                latency_seconds=latency,
                retries=retries,
                revisions=revisions,
                ended_at=datetime.now(timezone.utc).isoformat(),
            )
            return "failed"
        store.complete(
            run_id,
            total_tokens=ledger.used,
            latency_seconds=latency,
            retries=retries,
            revisions=revisions,
            ended_at=datetime.now(timezone.utc).isoformat(),
        )
        return "completed"
    except Exception as error:
        retries = previous_retries + (context.retries if context else 0)
        revisions = previous_revisions + (
            context.semantic_revisions + context.schema_repairs if context else 0
        )
        store.fail(
            run_id,
            type(error).__name__,
            str(error),
            total_tokens=ledger.used,
            latency_seconds=previous_latency + time.monotonic() - started,
            retries=retries,
            revisions=revisions,
            ended_at=datetime.now(timezone.utc).isoformat(),
        )
        return "failed"


def execute_interactive(
    pdf_bytes: bytes,
    pdf_name: str,
    condition: Condition,
    *,
    model: str,
    token_budget: int,
    store: RunStore,
    gateway_factory: Callable[[BudgetLedger], GeminiGateway],
) -> tuple[str, str]:
    content_hash = hashlib.sha256(pdf_bytes).hexdigest()
    chunks = parse_pdf(pdf_bytes)
    page_count = max(chunk.page_number for chunk in chunks)
    length_class = "short" if page_count <= 15 else "medium" if page_count <= 60 else "long"
    nonce = time.time_ns()
    run_id = hashlib.sha256(
        f"interactive:{content_hash}:{condition.value}:{nonce}".encode()
    ).hexdigest()[:16]
    definition = RunDefinition(
        run_id=run_id,
        document_hash=content_hash,
        condition=condition,
        repetition=1,
        order=1,
    )
    prompt_hash, schema_hash = configuration_fingerprints()
    experiment_id = hashlib.sha256(f"interactive:{run_id}".encode()).hexdigest()[:16]
    manifest = ExperimentManifest(
        experiment_id=experiment_id,
        model=model,
        temperature=0.0,
        token_budget=token_budget,
        seed=nonce,
        prompt_version=PROMPT_VERSION,
        prompt_hash=prompt_hash,
        schema_hash=schema_hash,
        software_version=__version__,
        documents=[
            DocumentRecord(
                name=pdf_name,
                content_hash=content_hash,
                page_count=page_count,
                length_class=length_class,
            )
        ],
        runs=[definition],
    )
    store.save_experiment(experiment_id, manifest.model_dump(mode="json"))
    return run_id, execute_run(
        definition, pdf_bytes, manifest, store, gateway_factory
    )
```

- [ ] **Step 6: Run manifest tests**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_experiments.py tests/test_storage.py -q
```

Expected: all tests pass and the manifest contains exactly `54` runs.

- [ ] **Step 7: Commit reproducible experiment execution**

```bash
rtk git add src/brd_srs_testgen/models.py src/brd_srs_testgen/storage.py src/brd_srs_testgen/experiments.py tests/test_experiments.py
rtk git commit -m "feat: add frozen experiment runner"
```

### Task 11: Add blinded sampling, ordinal agreement, and paired summaries

**Files:**
- Modify: `src/brd_srs_testgen/models.py`
- Modify: `src/brd_srs_testgen/storage.py`
- Create: `src/brd_srs_testgen/human_eval.py`
- Create: `tests/test_human_eval.py`
- Modify: `tests/test_storage.py`

- [ ] **Step 1: Write failing blinding and agreement tests**

Create `tests/test_human_eval.py`:

```python
from brd_srs_testgen.human_eval import agreement, blind_sample, paired_bootstrap
from brd_srs_testgen.models import CaseRecord, Condition, EvaluationScore, ScenarioType
from tests.test_validation import fixture_bundle


def records() -> list[CaseRecord]:
    bundle, _ = fixture_bundle()
    case = bundle.test_cases[0]
    return [
        CaseRecord(
            document_hash="doc-1",
            condition=condition,
            repetition=repetition,
            scenario_type=ScenarioType.POSITIVE,
            test_case=case.model_copy(update={"test_case_id": f"TC-{condition}-{repetition}-{index}"}),
        )
        for condition in Condition
        for repetition in range(1, 4)
        for index in range(3)
    ]


def test_blind_sample_is_stable_and_hides_condition() -> None:
    first_cases, first_mapping = blind_sample(records(), per_document_condition=6, seed=20260810)
    second_cases, second_mapping = blind_sample(records(), per_document_condition=6, seed=20260810)

    assert first_cases == second_cases
    assert first_mapping == second_mapping
    assert len(first_cases) == 18
    assert all(not hasattr(item, "condition") for item in first_cases)
    for condition in Condition:
        repetitions = [
            value["repetition"]
            for value in first_mapping.values()
            if value["condition"] == condition.value
        ]
        assert sorted(repetitions) == [1, 1, 2, 2, 3, 3]


def test_ordinal_agreement_is_one_for_identical_ratings() -> None:
    scores = [
        EvaluationScore(
            blind_id=blind_id,
            evaluator_id=evaluator,
            correctness=correctness,
            completeness=4,
            executability_readability=5,
            groundedness_traceability=5,
        )
        for evaluator in ["A", "B", "C"]
        for blind_id, correctness in [("blind-1", 5), ("blind-2", 3)]
    ]

    assert agreement(scores, "correctness") == 1.0


def test_paired_bootstrap_resamples_documents_not_repetitions() -> None:
    result = paired_bootstrap(
        {f"doc-{index}": (float(index), float(index + 1)) for index in range(6)},
        seed=20260810,
        samples=100,
    )

    assert result == {"median_difference": 1.0, "ci_low": 1.0, "ci_high": 1.0}
```

Append to `tests/test_storage.py`:

```python
def test_evaluation_mapping_and_scores_are_separate(tmp_path) -> None:
    store = RunStore(tmp_path)
    store.write_evaluation("experiment1", "blind_mapping", {"blind-1": {"condition": "single"}})
    store.write_evaluation("experiment1", "scores", [{"blind_id": "blind-1"}])

    assert store.read_evaluation("experiment1", "blind_mapping")["blind-1"]["condition"] == "single"
    assert store.read_evaluation("experiment1", "scores") == [{"blind_id": "blind-1"}]
    assert (tmp_path / "evaluations" / "experiment1" / "blind_mapping.json").exists()
    assert (tmp_path / "evaluations" / "experiment1" / "scores.json").exists()
```

- [ ] **Step 2: Run evaluation tests to verify they fail**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_human_eval.py tests/test_storage.py -q
```

Expected: import fails because the human-evaluation module and evaluation-store methods do not exist.

- [ ] **Step 3: Add blinded-evaluation models**

Append to `src/brd_srs_testgen/models.py`:

```python
class CaseRecord(StrictModel):
    document_hash: str
    condition: Condition
    repetition: int
    scenario_type: ScenarioType
    test_case: TestCase


class BlindedCase(StrictModel):
    blind_id: str
    document_hash: str
    scenario_type: ScenarioType
    test_case: TestCase


class EvaluationScore(StrictModel):
    blind_id: str
    evaluator_id: str
    correctness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    executability_readability: int = Field(ge=1, le=5)
    groundedness_traceability: int = Field(ge=1, le=5)
    unsupported_or_hallucinated: bool = False
    comment: str = ""
```

At module scope in `storage.py`, add the fixed allowlist:

```python
EVALUATION_ARTIFACTS = {"blind_cases", "blind_mapping", "scores"}
```

Then add these methods inside `RunStore`; the allowlist keeps user input out of paths while storing the private mapping separately from evaluator scores:

```python
def _evaluation_path(self, experiment_id: str, name: str) -> Path:
    if not experiment_id.isalnum() or name not in EVALUATION_ARTIFACTS:
        raise ValueError("Invalid evaluation artifact path.")
    return self.root / "evaluations" / experiment_id / f"{name}.json"

def write_evaluation(self, experiment_id: str, name: str, value: Any) -> None:
    _atomic_json(self._evaluation_path(experiment_id, name), value)

def read_evaluation(self, experiment_id: str, name: str) -> Any:
    return json.loads(
        self._evaluation_path(experiment_id, name).read_text(encoding="utf-8")
    )

def has_evaluation(self, experiment_id: str, name: str) -> bool:
    return self._evaluation_path(experiment_id, name).exists()
```

- [ ] **Step 4: Implement deterministic blinding and ordinal agreement**

Create `src/brd_srs_testgen/human_eval.py`:

```python
from __future__ import annotations

import hashlib
import random
from collections import defaultdict

import krippendorff
import numpy as np

from .models import BlindedCase, CaseRecord, EvaluationScore


def blind_sample(
    records: list[CaseRecord], *, per_document_condition: int = 6, seed: int
) -> tuple[list[BlindedCase], dict[str, dict[str, str | int]]]:
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[CaseRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.document_hash, record.condition.value)].append(record)

    blinded: list[BlindedCase] = []
    mapping: dict[str, dict[str, str | int]] = {}
    for (document_hash, condition), group in sorted(grouped.items()):
        selected: list[CaseRecord] = []
        remaining = group.copy()
        repetition_counts: dict[int, int] = defaultdict(int)

        def select_one(candidates: list[CaseRecord]) -> None:
            rng.shuffle(candidates)
            choice = min(candidates, key=lambda item: repetition_counts[item.repetition])
            selected.append(choice)
            remaining.remove(choice)
            repetition_counts[choice.repetition] += 1

        for scenario_type in sorted({item.scenario_type.value for item in remaining}):
            if len(selected) == per_document_condition:
                break
            select_one(
                [item for item in remaining if item.scenario_type.value == scenario_type]
            )
        while remaining and len(selected) < per_document_condition:
            select_one(remaining.copy())
        for record in selected[:per_document_condition]:
            key = f"{seed}:{record.document_hash}:{condition}:{record.repetition}:{record.test_case.test_case_id}"
            blind_id = hashlib.sha256(key.encode()).hexdigest()[:12]
            blinded.append(
                BlindedCase(
                    blind_id=blind_id,
                    document_hash=record.document_hash,
                    scenario_type=record.scenario_type,
                    test_case=record.test_case,
                )
            )
            mapping[blind_id] = {
                "condition": condition,
                "repetition": record.repetition,
                "test_case_id": record.test_case.test_case_id,
            }
    rng.shuffle(blinded)
    return blinded, mapping


def agreement(scores: list[EvaluationScore], field: str) -> float:
    evaluators = sorted({score.evaluator_id for score in scores})
    cases = sorted({score.blind_id for score in scores})
    lookup = {(score.evaluator_id, score.blind_id): getattr(score, field) for score in scores}
    matrix = np.array(
        [[lookup.get((evaluator, case), np.nan) for case in cases] for evaluator in evaluators],
        dtype=float,
    )
    return float(
        krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal")
    )
```

- [ ] **Step 5: Add paired document-level bootstrap summaries**

Append to `src/brd_srs_testgen/human_eval.py`:

```python
def paired_bootstrap(
    values: dict[str, tuple[float, float]], *, seed: int = 20260810, samples: int = 10_000
) -> dict[str, float]:
    documents = sorted(values)
    if not documents:
        raise ValueError("At least one paired document is required.")
    rng = random.Random(seed)
    observed = [values[document][1] - values[document][0] for document in documents]
    bootstrapped = [
        float(np.median([observed[rng.randrange(len(observed))] for _ in documents]))
        for _ in range(samples)
    ]
    return {
        "median_difference": float(np.median(observed)),
        "ci_low": float(np.percentile(bootstrapped, 2.5)),
        "ci_high": float(np.percentile(bootstrapped, 97.5)),
    }
```

- [ ] **Step 6: Run blinded-evaluation tests**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_human_eval.py tests/test_storage.py -q
```

Expected: all human-evaluation and storage tests pass.

- [ ] **Step 7: Commit human-evaluation tooling**

```bash
rtk git add src/brd_srs_testgen/models.py src/brd_srs_testgen/storage.py src/brd_srs_testgen/human_eval.py tests/test_human_eval.py tests/test_storage.py
rtk git commit -m "feat: add blinded expert evaluation"
```

### Task 12: Export canonical artifacts to JSON and Excel

**Files:**
- Create: `src/brd_srs_testgen/exports.py`
- Create: `tests/test_exports.py`

- [ ] **Step 1: Write a failing workbook-contract test**

Create `tests/test_exports.py`:

```python
from io import BytesIO

from openpyxl import load_workbook

from brd_srs_testgen.exports import bundle_json, bundle_xlsx
from brd_srs_testgen.validation import build_rtm
from tests.test_validation import fixture_bundle


def test_export_contains_canonical_sheets_and_json() -> None:
    bundle, _ = fixture_bundle()
    workbook = load_workbook(BytesIO(bundle_xlsx(bundle, build_rtm(bundle), {"status": "completed"})))

    assert workbook.sheetnames == ["Requirements", "Scenarios", "Test Cases", "RTM", "Metrics"]
    assert '"requirement_id": "REQ-001"' in bundle_json(bundle)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_exports.py -q
```

Expected: collection fails because `brd_srs_testgen.exports` does not exist.

- [ ] **Step 3: Implement deterministic JSON and workbook exports**

Create `src/brd_srs_testgen/exports.py`:

```python
from __future__ import annotations

import io
import json

import pandas as pd

from .models import ArtifactBundle, RTMRow


def _sources(artifact) -> str:
    return "; ".join(
        f"p.{reference.page_number} {reference.chunk_id}: {reference.excerpt}"
        for reference in artifact.source_references
    )


def bundle_json(bundle: ArtifactBundle) -> str:
    return json.dumps(bundle.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"


def bundle_xlsx(
    bundle: ArtifactBundle, rtm: list[RTMRow], metrics: dict
) -> bytes:
    requirements = [
        {
            **item.model_dump(exclude={"source_references", "ambiguities", "dependency_ids"}, mode="json"),
            "ambiguities": "\n".join(item.ambiguities),
            "dependency_ids": ", ".join(item.dependency_ids),
            "sources": _sources(item),
        }
        for item in bundle.requirements
    ]
    scenarios = [
        {
            **item.model_dump(exclude={"source_references", "preconditions", "requirement_ids"}, mode="json"),
            "requirement_ids": ", ".join(item.requirement_ids),
            "preconditions": "\n".join(item.preconditions),
            "sources": _sources(item),
        }
        for item in bundle.scenarios
    ]
    test_cases = [
        {
            "test_case_id": item.test_case_id,
            "scenario_id": item.scenario_id,
            "requirement_ids": ", ".join(item.requirement_ids),
            "title": item.title,
            "priority": item.priority.value,
            "preconditions": "\n".join(item.preconditions),
            "test_data": json.dumps(item.test_data, ensure_ascii=False),
            "steps": "\n".join(
                f"{step.step_number}. {step.action} -> {step.expected_result}" for step in item.steps
            ),
            "sources": _sources(item),
        }
        for item in bundle.test_cases
    ]
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(requirements).to_excel(writer, index=False, sheet_name="Requirements")
        pd.DataFrame(scenarios).to_excel(writer, index=False, sheet_name="Scenarios")
        pd.DataFrame(test_cases).to_excel(writer, index=False, sheet_name="Test Cases")
        pd.DataFrame([item.model_dump(mode="json") for item in rtm]).to_excel(
            writer, index=False, sheet_name="RTM"
        )
        pd.DataFrame([metrics]).to_excel(writer, index=False, sheet_name="Metrics")
    return output.getvalue()
```

- [ ] **Step 4: Run export tests**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_exports.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit artifact exports**

```bash
rtk git add src/brd_srs_testgen/exports.py tests/test_exports.py
rtk git commit -m "feat: export traceable test artifacts"
```

### Task 13: Build the four-tab Streamlit application

**Files:**
- Create: `app.py`
- Create: `tests/test_app_smoke.py`

- [ ] **Step 1: Write a failing Streamlit smoke test**

Create `tests/test_app_smoke.py`:

```python
from streamlit.testing.v1 import AppTest


def test_app_exposes_four_research_workflow_tabs() -> None:
    app = AppTest.from_file("app.py").run(timeout=10)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Generate",
        "Artifacts",
        "Experiment",
        "Human Evaluation",
    ]
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_app_smoke.py -q
```

Expected: failure because `app.py` does not exist.

- [ ] **Step 3: Create the application shell and shared state**

Create `app.py` with the imports, configuration, and tab shell:

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from brd_srs_testgen.documents import DocumentError, extract_pages
from brd_srs_testgen.experiments import build_manifest, execute_interactive, execute_run
from brd_srs_testgen.exports import bundle_json, bundle_xlsx
from brd_srs_testgen.human_eval import blind_sample, paired_bootstrap
from brd_srs_testgen.llm import GeminiGateway
from brd_srs_testgen.models import (
    ArtifactBundle,
    CaseRecord,
    Condition,
    DocumentChunk,
    DocumentRecord,
    EvaluationScore,
    ExperimentManifest,
    ScenarioType,
    TestCase,
)
from brd_srs_testgen.storage import RunStore
from brd_srs_testgen.validation import build_rtm


st.set_page_config(page_title="BRD/SRS Test Case Research", layout="wide")
st.title("BRD/SRS to Manual Test Cases")
st.caption("Controlled single-agent and centralized multi-agent comparison")

api_key = st.sidebar.text_input("Gemini API key", type="password")
model = st.sidebar.text_input("Frozen Gemini model", value="gemini-3.6-flash")
token_budget = int(
    st.sidebar.number_input("Total token budget per run", min_value=10_000, value=200_000, step=10_000)
)
store = RunStore(Path("runs"))

if "latest_bundle" not in st.session_state:
    st.session_state.latest_bundle = None
    st.session_state.latest_chunks = None
    st.session_state.latest_metrics = None
if "experiment_manifest" not in st.session_state:
    st.session_state.experiment_manifest = None
    st.session_state.experiment_pdfs = {}
if "blind_cases" not in st.session_state:
    st.session_state.blind_cases = []
    st.session_state.blind_mapping = {}
    st.session_state.evaluation_scores = []

generate_tab, artifacts_tab, experiment_tab, evaluation_tab = st.tabs(
    ["Generate", "Artifacts", "Experiment", "Human Evaluation"]
)
```

- [ ] **Step 4: Implement interactive generation**

Append to `app.py`:

```python
with generate_tab:
    uploaded = st.file_uploader("Upload a text-extractable BRD/SRS PDF", type=["pdf"])
    condition = Condition(
        st.selectbox(
            "Generation condition",
            options=[item.value for item in Condition],
            format_func=lambda value: value.replace("_", " ").title(),
        )
    )
    if st.button("Generate artifacts", type="primary"):
        if not api_key or uploaded is None:
            st.error("Provide a Gemini API key and PDF.")
        else:
            try:
                with st.spinner("Generating and validating artifacts"):
                    run_id, status = execute_interactive(
                        uploaded.getvalue(),
                        uploaded.name,
                        condition,
                        model=model,
                        token_budget=token_budget,
                        store=store,
                        gateway_factory=lambda ledger: GeminiGateway(
                            api_key=api_key, model=model, ledger=ledger
                        ),
                    )
                if all(
                    store.has_artifact(run_id, name)
                    for name in ["requirements", "scenarios", "test_cases"]
                ):
                    bundle = ArtifactBundle.model_validate(
                        {
                            "requirements": store.read_artifact(run_id, "requirements"),
                            "scenarios": store.read_artifact(run_id, "scenarios"),
                            "test_cases": store.read_artifact(run_id, "test_cases"),
                        }
                    )
                    chunks = [
                        DocumentChunk.model_validate(item)
                        for item in store.read_artifact(run_id, "chunks")
                    ]
                    metrics = store.read_artifact(run_id, "metrics")
                    st.session_state.latest_bundle = bundle
                    st.session_state.latest_chunks = chunks
                    st.session_state.latest_metrics = metrics
                    st.json(metrics)
                manifest_data = store.read_manifest(run_id)
                if status == "completed":
                    st.success(f"Run {run_id} completed and was persisted.")
                else:
                    st.error(
                        f"Run {run_id} failed: "
                        f"{manifest_data.get('failure_category', 'unknown')} — "
                        f"{manifest_data.get('failure_detail', 'No detail')}."
                    )
                    if store.has_artifact(run_id, "validation_issues"):
                        st.json(store.read_artifact(run_id, "validation_issues"))
            except DocumentError as error:
                st.error(str(error))
            except Exception as error:
                st.exception(error)
```

- [ ] **Step 5: Implement artifact inspection and downloads**

Append to `app.py`:

```python
with artifacts_tab:
    bundle: ArtifactBundle | None = st.session_state.latest_bundle
    if bundle is None:
        st.info("Generate artifacts first.")
    else:
        st.subheader("Requirements")
        st.dataframe(pd.DataFrame([item.model_dump(mode="json") for item in bundle.requirements]))
        st.subheader("Scenarios")
        st.dataframe(pd.DataFrame([item.model_dump(mode="json") for item in bundle.scenarios]))
        st.subheader("Test Cases")
        st.dataframe(pd.DataFrame([item.model_dump(mode="json") for item in bundle.test_cases]))
        rtm = build_rtm(bundle)
        st.subheader("Requirement Traceability Matrix")
        st.dataframe(pd.DataFrame([item.model_dump(mode="json") for item in rtm]))
        st.download_button(
            "Download JSON",
            bundle_json(bundle),
            file_name="test_artifacts.json",
            mime="application/json",
        )
        st.download_button(
            "Download Excel",
            bundle_xlsx(bundle, rtm, st.session_state.latest_metrics),
            file_name="test_artifacts.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
```

- [ ] **Step 6: Implement experiment preflight and execution**

Append to `app.py`:

```python
def length_class(page_count: int) -> str:
    return "short" if page_count <= 15 else "medium" if page_count <= 60 else "long"


with experiment_tab:
    pilot_confirmed = st.checkbox(
        "I used a separate pilot PDF and selected the smallest token ceiling under which all three conditions completed."
    )
    files = st.file_uploader(
        "Select exactly six evaluation PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="experiment_files",
    )
    build_clicked = st.button("Build frozen manifest")
    if build_clicked and not pilot_confirmed:
        st.error("Complete and confirm the excluded pilot calibration first.")
    elif build_clicked:
        try:
            records = []
            pdfs = {}
            for file in files:
                data = file.getvalue()
                content_hash = hashlib.sha256(data).hexdigest()
                page_count = len(extract_pages(data))
                records.append(
                    DocumentRecord(
                        name=file.name,
                        content_hash=content_hash,
                        page_count=page_count,
                        length_class=length_class(page_count),
                    )
                )
                pdfs[content_hash] = data
            manifest = build_manifest(
                records, model=model, token_budget=token_budget, seed=20260810
            )
            store.save_experiment(
                manifest.experiment_id, manifest.model_dump(mode="json")
            )
            st.session_state.experiment_manifest = manifest
            st.session_state.experiment_pdfs = pdfs
            st.success("Manifest frozen for 54 runs.")
        except Exception as error:
            st.error(str(error))

    manifest: ExperimentManifest | None = st.session_state.experiment_manifest
    if manifest is not None:
        st.json(manifest.model_dump(mode="json"))
        if st.button("Run or resume experiment", type="primary"):
            if not api_key:
                st.error("Provide a Gemini API key.")
            else:
                progress = st.progress(0.0)
                statuses = []
                ordered = sorted(manifest.runs, key=lambda item: (item.document_hash, item.repetition, item.order))
                for index, definition in enumerate(ordered, 1):
                    status = execute_run(
                        definition,
                        st.session_state.experiment_pdfs[definition.document_hash],
                        manifest,
                        store,
                        lambda ledger: GeminiGateway(
                            api_key=api_key, model=manifest.model, ledger=ledger
                        ),
                    )
                    statuses.append({"run_id": definition.run_id, "status": status})
                    progress.progress(index / len(ordered))
                st.dataframe(pd.DataFrame(statuses))

        metric_rows = []
        for definition in manifest.runs:
            if store.exists(definition.run_id) and store.has_artifact(definition.run_id, "metrics"):
                metric_rows.append(
                    {
                        "document_hash": definition.document_hash,
                        "condition": definition.condition.value,
                        "repetition": definition.repetition,
                        **store.read_artifact(definition.run_id, "metrics"),
                    }
                )
        if metric_rows:
            metrics_frame = pd.DataFrame(metric_rows)
            st.subheader("Automated metrics")
            st.dataframe(metrics_frame)
            numeric_metrics = [
                column
                for column in metrics_frame.select_dtypes(include="number").columns
                if column != "repetition"
            ]
            selected_metric = st.selectbox("Comparison metric", numeric_metrics)
            condition_summary = (
                metrics_frame.groupby("condition")[selected_metric]
                .agg(
                    median="median",
                    q1=lambda values: values.quantile(0.25),
                    q3=lambda values: values.quantile(0.75),
                    runs="count",
                )
                .reset_index()
            )
            st.dataframe(condition_summary)
            document_means = metrics_frame.pivot_table(
                index="document_hash",
                columns="condition",
                values=selected_metric,
                aggfunc="mean",
            )
            central = Condition.CENTRALIZED_MULTI_AGENT.value
            staged = Condition.STAGED_SINGLE_AGENT.value
            pairs = {
                document_hash: (float(row[staged]), float(row[central]))
                for document_hash, row in document_means.iterrows()
                if pd.notna(row.get(staged)) and pd.notna(row.get(central))
            }
            if pairs:
                st.json(
                    {
                        "comparison": f"{central} minus {staged}",
                        **paired_bootstrap(pairs, seed=manifest.seed),
                    }
                )
```

- [ ] **Step 7: Implement blinded evaluator forms**

Append to `app.py`:

```python
with evaluation_tab:
    st.write("Generate blinded samples from completed experiment runs, then score each case independently.")
    evaluator_id = st.text_input("Evaluator ID")
    if st.button("Prepare blinded sample"):
        manifest = st.session_state.experiment_manifest
        if manifest is None:
            st.error("Build and run an experiment first.")
        else:
            records: list[CaseRecord] = []
            for definition in manifest.runs:
                if not store.exists(definition.run_id) or store.status(definition.run_id) != "completed":
                    continue
                scenarios = store.read_artifact(definition.run_id, "scenarios")
                scenario_types = {
                    item["scenario_id"]: ScenarioType(item["scenario_type"]) for item in scenarios
                }
                for raw_case in store.read_artifact(definition.run_id, "test_cases"):
                    case = TestCase.model_validate(raw_case)
                    records.append(
                        CaseRecord(
                            document_hash=definition.document_hash,
                            condition=definition.condition,
                            repetition=definition.repetition,
                            scenario_type=scenario_types[case.scenario_id],
                            test_case=case,
                        )
                    )
            cases, mapping = blind_sample(records, per_document_condition=6, seed=manifest.seed)
            st.session_state.blind_cases = cases
            st.session_state.blind_mapping = mapping
            store.write_evaluation(
                manifest.experiment_id,
                "blind_cases",
                [item.model_dump(mode="json") for item in cases],
            )
            store.write_evaluation(manifest.experiment_id, "blind_mapping", mapping)
            if store.has_evaluation(manifest.experiment_id, "scores"):
                st.session_state.evaluation_scores = [
                    EvaluationScore.model_validate(item)
                    for item in store.read_evaluation(manifest.experiment_id, "scores")
                ]

    for case in st.session_state.blind_cases:
        with st.expander(f"Case {case.blind_id}"):
            st.json(case.test_case.model_dump(mode="json"))
            with st.form(f"score-{case.blind_id}-{evaluator_id}"):
                values = {
                    "correctness": st.slider("Correctness", 1, 5, 3),
                    "completeness": st.slider("Completeness", 1, 5, 3),
                    "executability_readability": st.slider("Executability / readability", 1, 5, 3),
                    "groundedness_traceability": st.slider("Groundedness / traceability", 1, 5, 3),
                    "unsupported_or_hallucinated": st.checkbox("Unsupported or hallucinated"),
                    "comment": st.text_area("Comment"),
                }
                if st.form_submit_button("Save score"):
                    if not evaluator_id.strip():
                        st.error("Enter an evaluator ID before saving.")
                    else:
                        score = EvaluationScore(
                            blind_id=case.blind_id,
                            evaluator_id=evaluator_id.strip(),
                            **values,
                        )
                        st.session_state.evaluation_scores = [
                            item
                            for item in st.session_state.evaluation_scores
                            if not (
                                item.blind_id == score.blind_id
                                and item.evaluator_id == score.evaluator_id
                            )
                        ] + [score]
                        manifest = st.session_state.experiment_manifest
                        store.write_evaluation(
                            manifest.experiment_id,
                            "scores",
                            [
                                item.model_dump(mode="json")
                                for item in st.session_state.evaluation_scores
                            ],
                        )
                        st.success("Score saved locally.")

    if st.session_state.evaluation_scores:
        st.download_button(
            "Download evaluator scores",
            json.dumps(
                [item.model_dump(mode="json") for item in st.session_state.evaluation_scores],
                ensure_ascii=False,
                indent=2,
            ),
            file_name=f"evaluation-{evaluator_id or 'anonymous'}.json",
            mime="application/json",
        )
```

The evaluator download contains only anonymized scores. The private mapping is persisted separately at `runs/evaluations/<experiment_id>/blind_mapping.json` and is never rendered in the evaluator UI.

- [ ] **Step 8: Run the Streamlit smoke test**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_app_smoke.py -q
```

Expected: `1 passed` and no Streamlit exceptions.

- [ ] **Step 9: Commit the Streamlit application**

```bash
rtk git add app.py src/brd_srs_testgen/storage.py tests/test_app_smoke.py
rtk git commit -m "feat: add Streamlit research workflow"
```

### Task 14: Document operation and run the complete verification gate

**Files:**
- Modify: `README.md`
- Test: `tests/`

- [ ] **Step 1: Replace the stale README with exact setup and usage**

Replace `README.md` with:

````markdown
# BRD/SRS to Manual Test Cases

A Streamlit research application comparing single-prompt, staged single-agent,
and centralized multi-agent generation of traceable manual test cases.
PDF text is extracted exhaustively into page-aware chunks and selected by deterministic
source links; the controlled experiment does not use a vector database.

## Setup

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run

```sh
PYTHONPATH=src .venv/bin/python -m streamlit run app.py
```

Enter a Gemini API key in the sidebar. The key remains in Streamlit session
state and is not written to disk. Use a text-extractable PDF; OCR is out of
scope.

## Verify

```sh
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Automated tests use mocked model responses and do not call Gemini.

## Experiment

Select exactly two PDFs up to 15 pages, two PDFs from 16-60 pages, and two PDFs
over 60 pages. Freeze the model and pilot token budget before creating the
54-run manifest. Completed runs under `runs/` are immutable and git-ignored.

See the approved design and implementation plan under `docs/superpowers/`.
````

- [ ] **Step 2: Run the complete automated suite**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest -q
```

Expected: all tests pass with zero live Gemini calls.

- [ ] **Step 3: Run static syntax compilation**

Run:

```bash
rtk .venv/bin/python -m compileall -q app.py src tests
```

Expected: exit `0` with no syntax errors.

- [ ] **Step 4: Verify the Streamlit process starts**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m streamlit run app.py --server.headless true
```

Expected: Streamlit reports a local URL and no import error. Stop the process with `Ctrl-C` after startup.

- [ ] **Step 5: Perform the optional live pilot smoke test**

With a non-evaluation text PDF and a valid API key, use the Generate tab once for each condition. Confirm each run produces requirements, scenarios, test cases, source excerpts, RTM, token usage, and downloads. Do not use any of these outputs in the six-document evaluation.

- [ ] **Step 6: Inspect the final diff and repository status**

Run:

```bash
rtk git diff --check
rtk git status --short
```

Expected: no whitespace errors; only the intended README change remains unstaged before the final commit.

- [ ] **Step 7: Commit operational documentation**

```bash
rtk git add README.md
rtk git commit -m "docs: document research app operation"
```

- [ ] **Step 8: Run a fresh post-commit verification**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest -q
rtk git status --short
```

Expected: all tests pass and the worktree is clean.
