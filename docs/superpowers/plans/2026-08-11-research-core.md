# BRD/SRS Test-Case Research Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an English Streamlit application that runs one text-extractable PDF through single-prompt, staged single-agent, and centralized multi-agent test-case generation using either Gemini or Ollama, then persists and downloads validated JSON artifacts, RTM rows, and metrics.

**Architecture:** A thin Streamlit entry point calls a provider-neutral research kernel. Page-aware evidence, strict Pydantic models, deterministic validation/metrics, and immutable local storage are shared by all three pipelines; each condition receives its own equal token ledger. Gemini and Ollama differ only inside their transport adapters.

**Tech Stack:** Python 3.11+, Streamlit, Google Gen AI SDK, Pydantic 2, pypdf, pytest, and the Python standard library.

---

**Design spec:** `docs/superpowers/specs/2026-08-11-research-core-design.md`

This plan supersedes `2026-08-10-centralized-agent-test-case-generation.md` for the first delivery. The older full-platform plan remains future-scope reference only.

**Execution rules:** Run shell commands through `rtk`. Do not call live Gemini or Ollama services in automated tests. Leave `app-ba.py`, `app-ba-sys-architect.py`, `llm_provider.py`, and their existing user changes untouched.

## File structure

```text
app.py                              Thin English Streamlit UI
requirements.txt                   Runtime and test dependencies
.gitignore                         Local environment and run artifacts
src/brd_srs_testgen/__init__.py     Package exports/version
src/brd_srs_testgen/models.py       Canonical strict data models
src/brd_srs_testgen/documents.py    PDF parsing, chunks, evidence checks
src/brd_srs_testgen/providers.py    Gemini/Ollama adapters and token ledger
src/brd_srs_testgen/validation.py   Referential checks, RTM, and metrics
src/brd_srs_testgen/storage.py      Atomic immutable run persistence
src/brd_srs_testgen/pipelines.py    Prompts and three generation conditions
src/brd_srs_testgen/runner.py       Three-condition comparison orchestration
tests/factories.py                  Shared deterministic artifact fixtures
tests/test_models.py                Schema tests
tests/test_documents.py             Evidence tests
tests/test_providers.py             Provider, usage, retry-boundary tests
tests/test_validation.py            Traceability and metric tests
tests/test_storage.py               Atomic persistence tests
tests/test_pipelines.py             Three condition tests
tests/test_runner.py                Comparison/failure-isolation tests
tests/test_app.py                   Streamlit smoke test
docs/research-core-operations.md     Setup and manual smoke instructions
```

The existing prototypes remain available for reference until this application passes its verification gate.

### Task 1: Bootstrap the package and canonical models

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `src/brd_srs_testgen/__init__.py`
- Create: `src/brd_srs_testgen/models.py`
- Create: `tests/__init__.py`
- Create: `tests/factories.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Declare only required dependencies and ignored artifacts**

Create `requirements.txt`:

```text
google-genai>=2,<3
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

- [ ] **Step 2: Create the environment and install dependencies**

Run:

```bash
rtk uv --cache-dir /tmp/citd-final-uv-cache venv --python 3.11 .venv
rtk .venv/bin/python -m pip install -r requirements.txt
rtk .venv/bin/python -c "import google.genai, pydantic, pypdf, streamlit; print('ok')"
```

Expected: each command exits `0`; the import command prints `ok`.

- [ ] **Step 3: Write failing model tests**

Create `tests/__init__.py`:

```python
"""Test package."""
```

Create `tests/test_models.py`:

```python
import pytest
from pydantic import ValidationError

from brd_srs_testgen.models import (
    ArtifactBundle,
    Requirement,
    RequirementPriority,
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


def test_bundle_accepts_many_to_many_traceability() -> None:
    requirement = Requirement(
        requirement_id="REQ-001",
        title="Authenticate users",
        description="Registered users can sign in.",
        requirement_type=RequirementType.FUNCTIONAL,
        module="Authentication",
        priority=RequirementPriority.HIGH,
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


def test_artifact_ids_follow_canonical_patterns() -> None:
    with pytest.raises(ValidationError):
        Requirement(
            requirement_id="wrong",
            title="Title",
            description="Description",
            requirement_type=RequirementType.BUSINESS,
            module="Core",
            priority=RequirementPriority.MEDIUM,
            source_references=[source()],
        )
```

- [ ] **Step 4: Run the model tests to verify they fail**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_models.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'brd_srs_testgen'`.

- [ ] **Step 5: Implement the canonical models**

Create `src/brd_srs_testgen/models.py`:

```python
from __future__ import annotations

from enum import StrEnum

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
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class FailureCategory(StrEnum):
    PARSING = "parsing"
    CONFIGURATION = "configuration"
    PROVIDER_REJECTION = "provider_rejection"
    TRANSPORT_EXHAUSTION = "transport_exhaustion"
    TIMEOUT = "timeout"
    BUDGET_EXHAUSTION = "budget_exhaustion"
    SCHEMA_FAILURE = "schema_failure"
    SEMANTIC_VALIDATION = "semantic_validation"


class SourceReference(StrictModel):
    chunk_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    section: str = ""
    excerpt: str = Field(min_length=1)


class DocumentChunk(StrictModel):
    chunk_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    section: str = ""
    text: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class Requirement(StrictModel):
    requirement_id: str = Field(pattern=r"^REQ-\d{3,}$")
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    requirement_type: RequirementType
    module: str = Field(min_length=1)
    priority: RequirementPriority
    ambiguities: list[str] = Field(default_factory=list)
    dependency_ids: list[str] = Field(default_factory=list)
    source_references: list[SourceReference] = Field(min_length=1)


class Scenario(StrictModel):
    scenario_id: str = Field(pattern=r"^SCN-\d{3,}$")
    title: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    scenario_type: ScenarioType
    preconditions: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(min_length=1)
    source_references: list[SourceReference] = Field(min_length=1)


class TestStep(StrictModel):
    step_number: int = Field(ge=1)
    action: str = Field(min_length=1)
    expected_result: str = Field(min_length=1)


class TestCase(StrictModel):
    test_case_id: str = Field(pattern=r"^TC-\d{3,}$")
    scenario_id: str = Field(pattern=r"^SCN-\d{3,}$")
    requirement_ids: list[str] = Field(min_length=1)
    title: str = Field(min_length=1)
    priority: TestPriority
    preconditions: list[str] = Field(default_factory=list)
    test_data: dict[str, str] = Field(default_factory=dict)
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


class GeneratedCases(StrictModel):
    scenarios: list[Scenario]
    test_cases: list[TestCase]


class ReviewIssue(StrictModel):
    artifact_id: str
    reason: str


class ReviewResult(StrictModel):
    accepted: bool
    issues: list[ReviewIssue] = Field(default_factory=list)


class ValidationIssue(StrictModel):
    code: str
    artifact_id: str
    message: str


class ValidationReport(StrictModel):
    valid: bool
    issues: list[ValidationIssue]
    uncovered_requirement_ids: list[str]
    orphan_scenario_ids: list[str]
    orphan_test_case_ids: list[str]


class RTMRow(StrictModel):
    requirement_id: str
    scenario_ids: list[str]
    test_case_ids: list[str]
    source_chunk_ids: list[str]
    covered: bool


class RunMetrics(StrictModel):
    completion: bool
    schema_valid: bool
    citation_coverage: float
    requirement_scenario_coverage: float
    requirement_test_case_coverage: float
    positive_scenario_coverage: float
    non_positive_scenario_coverage: float
    rtm_completeness: float
    orphan_rate: float
    invalid_reference_rate: float
    duplicate_test_case_rate: float
    requirement_count: int
    scenario_count: int
    test_case_count: int
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    retries: int
    schema_repairs: int
    semantic_revisions: int
    budget_exhausted: bool


class ConditionManifest(StrictModel):
    condition: Condition
    status: RunStatus
    provider: str
    model: str
    temperature: float
    token_ceiling: int
    started_at: str
    completed_at: str | None = None
    failure_category: FailureCategory | None = None
    failure_message: str | None = None


class ComparisonManifest(StrictModel):
    comparison_id: str
    document_hash: str
    provider: str
    model: str
    temperature: float
    token_ceiling: int
    condition_order: list[Condition]
    prompt_version: str
    schema_version: str
    started_at: str
    completed_at: str | None = None
```

Create `src/brd_srs_testgen/__init__.py`:

```python
from .models import ArtifactBundle, Condition

__version__ = "0.1.0"
__all__ = ["ArtifactBundle", "Condition"]
```

Create `tests/factories.py`:

```python
from brd_srs_testgen.models import (
    ArtifactBundle,
    DocumentChunk,
    Requirement,
    RequirementPriority,
    RequirementType,
    Scenario,
    ScenarioType,
    SourceReference,
    TestCase,
    TestPriority,
    TestStep,
)


def chunk() -> DocumentChunk:
    text = "The system shall authenticate registered users."
    return DocumentChunk(
        chunk_id="p0001-c001-ecac9f035813",
        page_number=1,
        section="AUTHENTICATION",
        text=text,
        content_hash="ecac9f0358134f174bcbf0d60ddbc7c25bcb4f812ea8e4c57bfbd8c02edaa274",
    )


def source() -> SourceReference:
    item = chunk()
    return SourceReference(
        chunk_id=item.chunk_id,
        page_number=item.page_number,
        section=item.section,
        excerpt=item.text,
    )


def bundle() -> ArtifactBundle:
    requirement = Requirement(
        requirement_id="REQ-001",
        title="Authenticate users",
        description="Registered users can sign in.",
        requirement_type=RequirementType.FUNCTIONAL,
        module="Authentication",
        priority=RequirementPriority.HIGH,
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
    return ArtifactBundle(
        requirements=[requirement], scenarios=[scenario], test_cases=[test_case]
    )
```

- [ ] **Step 6: Run the model tests**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_models.py -q
```

Expected: `3 passed`.

- [ ] **Step 7: Commit the schema foundation**

```bash
rtk git add requirements.txt .gitignore src/brd_srs_testgen/__init__.py src/brd_srs_testgen/models.py tests/__init__.py tests/factories.py tests/test_models.py
rtk git commit -m "feat: add research core models"
```

### Task 2: Parse PDFs into deterministic evidence chunks

**Files:**
- Create: `src/brd_srs_testgen/documents.py`
- Create: `tests/test_documents.py`

- [ ] **Step 1: Write failing evidence tests**

Create `tests/test_documents.py`:

```python
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from brd_srs_testgen.documents import (
    DocumentError,
    chunk_pages,
    extract_pages,
    verify_source_reference,
)
from brd_srs_testgen.models import SourceReference


def test_chunk_ids_are_stable_page_aware_and_bounded() -> None:
    pages = [(1, "AUTHENTICATION\n" + "word " * 20), (2, "2.1 Audit\nAudit text")]

    first = chunk_pages(pages, max_chars=30)
    second = chunk_pages(pages, max_chars=30)

    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert {item.page_number for item in first} == {1, 2}
    assert all(len(item.text) <= 30 for item in first)
    assert first[0].section == "AUTHENTICATION"


def test_source_excerpt_must_exist_in_referenced_chunk() -> None:
    chunks = chunk_pages([(1, "The system shall authenticate users.")])
    valid = SourceReference(
        chunk_id=chunks[0].chunk_id,
        page_number=1,
        excerpt="system shall authenticate users",
    )
    invalid = valid.model_copy(update={"excerpt": "invented evidence"})

    assert verify_source_reference(valid, chunks)
    assert not verify_source_reference(invalid, chunks)


def test_empty_pdf_text_is_rejected() -> None:
    with pytest.raises(DocumentError, match="extractable text"):
        chunk_pages([(1, "  "), (2, "")])


def test_encrypted_pdf_is_rejected() -> None:
    reader = SimpleNamespace(is_encrypted=True, pages=[])
    with patch("brd_srs_testgen.documents.PdfReader", return_value=reader):
        with pytest.raises(DocumentError, match="Encrypted"):
            extract_pages(b"pdf")
```

- [ ] **Step 2: Run the evidence tests to verify they fail**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_documents.py -q
```

Expected: collection fails because `brd_srs_testgen.documents` does not exist.

- [ ] **Step 3: Implement PDF extraction and evidence checks**

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
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as error:
        raise DocumentError("The uploaded file is not a readable PDF.") from error
    if reader.is_encrypted:
        raise DocumentError("Encrypted PDFs are not supported.")
    try:
        return [
            (number, page.extract_text() or "")
            for number, page in enumerate(reader.pages, 1)
        ]
    except Exception as error:
        raise DocumentError("PDF text extraction failed.") from error


def _section_heading(raw_text: str) -> str:
    for line in raw_text.splitlines():
        candidate = normalize_text(line)
        if 3 <= len(candidate) <= 100 and (
            candidate.isupper() or re.match(r"^\d+(?:\.\d+)*\s+\S", candidate)
        ):
            return candidate
    return ""


def _pieces(text: str, max_chars: int) -> Iterable[str]:
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            yield remaining
            return
        cut = remaining.rfind(" ", 0, max_chars + 1)
        if cut <= 0:
            cut = max_chars
        yield remaining[:cut].strip()
        remaining = remaining[cut:].strip()


def chunk_pages(
    pages: Iterable[tuple[int, str]], max_chars: int = 6_000
) -> list[DocumentChunk]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    chunks: list[DocumentChunk] = []
    for page_number, raw_text in pages:
        text = normalize_text(raw_text)
        if not text:
            continue
        section = _section_heading(raw_text)
        for sequence, piece in enumerate(_pieces(text, max_chars), 1):
            digest = hashlib.sha256(piece.encode("utf-8")).hexdigest()
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
        and normalize_text(reference.excerpt).casefold()
        in normalize_text(chunk.text).casefold()
    )


def render_chunks(chunks: Iterable[DocumentChunk]) -> str:
    return "\n\n".join(
        f"[{item.chunk_id} | page {item.page_number} | {item.section}]\n{item.text}"
        for item in chunks
    )
```

- [ ] **Step 4: Run the evidence tests**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_documents.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit deterministic document evidence**

```bash
rtk git add src/brd_srs_testgen/documents.py tests/test_documents.py
rtk git commit -m "feat: add PDF evidence chunks"
```

### Task 3: Add budgeted Gemini and Ollama adapters

**Files:**
- Create: `src/brd_srs_testgen/providers.py`
- Create: `tests/test_providers.py`

- [ ] **Step 1: Write failing ledger and adapter tests**

Create `tests/test_providers.py`:

```python
import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from brd_srs_testgen.models import RequirementBatch
from brd_srs_testgen.providers import (
    BudgetExceeded,
    BudgetLedger,
    GeminiProvider,
    OllamaProvider,
    StructuredOutputError,
)


class FakeModels:
    def count_tokens(self, **_kwargs):
        return SimpleNamespace(total_tokens=10)


class FakeInteractions:
    def __init__(self, text: str = '{"requirements": []}') -> None:
        self.text = text
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            output_text=self.text,
            usage=SimpleNamespace(
                total_input_tokens=10,
                total_output_tokens=5,
                total_tokens=15,
            ),
        )


def test_ledger_prevents_over_reservation() -> None:
    ledger = BudgetLedger(limit=100)
    reservation = ledger.reserve(80)

    with pytest.raises(BudgetExceeded):
        ledger.reserve(21)

    ledger.settle(reservation, actual_tokens=50)
    assert ledger.used == 50
    assert ledger.remaining == 50


def test_gemini_uses_structured_output_and_records_usage() -> None:
    interactions = FakeInteractions()
    client = SimpleNamespace(models=FakeModels(), interactions=interactions)
    ledger = BudgetLedger(limit=100)
    provider = GeminiProvider(client, "gemini-test", ledger)

    result = provider.generate(
        [{"role": "user", "content": "Extract requirements"}],
        RequirementBatch,
        max_output_tokens=40,
    )

    assert result.value.requirements == []
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert ledger.used == 15
    assert interactions.kwargs["response_format"]["mime_type"] == "application/json"
    assert interactions.kwargs["generation_config"]["temperature"] == 0.0


def test_ollama_posts_schema_and_reads_token_counts() -> None:
    response = io.BytesIO(
        json.dumps(
            {
                "message": {"content": '{"requirements": []}'},
                "prompt_eval_count": 8,
                "eval_count": 4,
            }
        ).encode()
    )
    ledger = BudgetLedger(limit=10_000)
    provider = OllamaProvider("http://localhost:11434/", "gemma4", ledger)

    with patch("brd_srs_testgen.providers.urlopen", return_value=response) as opened:
        result = provider.generate(
            [{"role": "user", "content": "Extract requirements"}],
            RequirementBatch,
            max_output_tokens=40,
        )

    request = opened.call_args.args[0]
    payload = json.loads(request.data)
    assert request.full_url == "http://localhost:11434/api/chat"
    assert payload["stream"] is False
    assert payload["format"]["type"] == "object"
    assert result.total_tokens == 12
    assert ledger.used == 12


def test_invalid_json_is_charged_before_schema_error() -> None:
    interactions = FakeInteractions("not-json")
    client = SimpleNamespace(models=FakeModels(), interactions=interactions)
    ledger = BudgetLedger(limit=100)
    provider = GeminiProvider(client, "gemini-test", ledger)

    with pytest.raises(StructuredOutputError, match="structured output"):
        provider.generate(
            [{"role": "user", "content": "Extract"}],
            RequirementBatch,
            max_output_tokens=40,
        )

    assert ledger.used == 15
```

- [ ] **Step 2: Run the provider tests to verify they fail**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_providers.py -q
```

Expected: collection fails because `brd_srs_testgen.providers` does not exist.

- [ ] **Step 3: Implement the token ledger and provider adapters**

Create `src/brd_srs_testgen/providers.py`:

```python
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)
Messages = list[dict[str, str]]
RETRYABLE_CODES = {429, 500, 502, 503, 504}


class BudgetExceeded(RuntimeError):
    pass


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, code: int | None, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class StructuredOutputError(RuntimeError):
    def __init__(
        self,
        raw_text: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_seconds: float = 0.0,
    ) -> None:
        super().__init__("Provider returned invalid structured output.")
        self.raw_text = raw_text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_seconds = latency_seconds


@dataclass(frozen=True)
class Reservation:
    tokens: int


@dataclass
class BudgetLedger:
    limit: int
    used: int = 0
    reserved: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("Token limit must be positive.")

    @property
    def remaining(self) -> int:
        with self._lock:
            return self.limit - self.used - self.reserved

    def reserve(self, tokens: int) -> Reservation:
        if tokens < 1:
            raise ValueError("Reservation must be positive.")
        with self._lock:
            remaining = self.limit - self.used - self.reserved
            if tokens > remaining:
                raise BudgetExceeded(f"Need {tokens} tokens; {remaining} remain.")
            self.reserved += tokens
        return Reservation(tokens)

    def cancel(self, reservation: Reservation) -> None:
        with self._lock:
            self.reserved -= reservation.tokens

    def settle(self, reservation: Reservation, actual_tokens: int) -> None:
        with self._lock:
            self.reserved -= reservation.tokens
            self.used += actual_tokens
            over = self.used > self.limit
        if over:
            raise BudgetExceeded(
                f"Actual usage {self.used} exceeded token limit {self.limit}."
            )


@dataclass(frozen=True)
class GenerationResult(Generic[T]):
    value: T
    input_tokens: int
    output_tokens: int
    latency_seconds: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class StructuredProvider(Protocol):
    model: str
    ledger: BudgetLedger

    def generate(
        self,
        messages: Messages,
        schema: type[T],
        *,
        max_output_tokens: int,
    ) -> GenerationResult[T]:
        pass


def _prompt(messages: Messages) -> str:
    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}" for message in messages
    )


def _error_code(error: Exception) -> int | None:
    code = getattr(error, "code", None)
    if isinstance(code, int):
        return code
    status = getattr(error, "status_code", None)
    return status if isinstance(status, int) else None


class GeminiProvider:
    def __init__(self, client, model: str, ledger: BudgetLedger) -> None:
        self.client = client
        self.model = model
        self.ledger = ledger

    def generate(
        self,
        messages: Messages,
        schema: type[T],
        *,
        max_output_tokens: int,
    ) -> GenerationResult[T]:
        prompt = _prompt(messages)
        try:
            input_estimate = int(
                self.client.models.count_tokens(
                    model=self.model, contents=prompt
                ).total_tokens
            )
        except Exception as error:
            code = _error_code(error)
            raise ProviderError(
                str(error), code=code, retryable=code in RETRYABLE_CODES or code is None
            ) from error

        reservation = self.ledger.reserve(input_estimate + max_output_tokens)
        started = time.perf_counter()
        try:
            interaction = self.client.interactions.create(
                model=self.model,
                input=prompt,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema.model_json_schema(),
                },
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": max_output_tokens,
                },
            )
        except Exception as error:
            self.ledger.cancel(reservation)
            code = _error_code(error)
            raise ProviderError(
                str(error), code=code, retryable=code in RETRYABLE_CODES or code is None
            ) from error

        input_tokens = int(interaction.usage.total_input_tokens)
        output_tokens = int(interaction.usage.total_output_tokens)
        self.ledger.settle(reservation, input_tokens + output_tokens)
        raw_text = interaction.output_text
        try:
            value = schema.model_validate_json(raw_text)
        except Exception as error:
            raise StructuredOutputError(
                raw_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_seconds=time.perf_counter() - started,
            ) from error
        return GenerationResult(
            value=value,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=time.perf_counter() - started,
        )


class OllamaProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        ledger: BudgetLedger,
        *,
        timeout: int = 600,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.ledger = ledger
        self.timeout = timeout

    def generate(
        self,
        messages: Messages,
        schema: type[T],
        *,
        max_output_tokens: int,
    ) -> GenerationResult[T]:
        payload = {
            "model": self.model,
            "messages": messages,
            "format": schema.model_json_schema(),
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0, "num_predict": max_output_tokens},
        }
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        reservation = self.ledger.reserve(len(encoded) + max_output_tokens)
        request = Request(
            f"{self.base_url}/api/chat",
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            self.ledger.cancel(reservation)
            raise ProviderError(
                str(error), code=error.code, retryable=error.code in RETRYABLE_CODES
            ) from error
        except (URLError, TimeoutError) as error:
            self.ledger.cancel(reservation)
            raise ProviderError(str(error), code=None, retryable=True) from error
        except Exception as error:
            self.ledger.cancel(reservation)
            raise ProviderError(str(error), code=None, retryable=False) from error

        try:
            input_tokens = int(result["prompt_eval_count"])
            output_tokens = int(result["eval_count"])
            raw_text = result["message"]["content"]
        except (KeyError, TypeError, ValueError) as error:
            self.ledger.cancel(reservation)
            raise ProviderError(
                "Ollama returned an incomplete response.", code=None, retryable=False
            ) from error

        self.ledger.settle(reservation, input_tokens + output_tokens)
        try:
            value = schema.model_validate_json(raw_text)
        except Exception as error:
            raise StructuredOutputError(
                raw_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_seconds=time.perf_counter() - started,
            ) from error
        return GenerationResult(
            value=value,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=time.perf_counter() - started,
        )
```

- [ ] **Step 4: Run the provider tests**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_providers.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit the provider boundary**

```bash
rtk git add src/brd_srs_testgen/providers.py tests/test_providers.py
rtk git commit -m "feat: add Gemini and Ollama providers"
```

### Task 4: Validate traceability and compute RTM metrics

**Files:**
- Create: `src/brd_srs_testgen/validation.py`
- Create: `tests/test_validation.py`

- [ ] **Step 1: Write failing validation and metric tests**

Create `tests/test_validation.py`:

```python
from brd_srs_testgen.models import Requirement, RequirementPriority, RequirementType
from brd_srs_testgen.validation import build_rtm, compute_metrics, validate_bundle
from tests.factories import bundle, chunk, source


def test_valid_bundle_builds_complete_rtm() -> None:
    artifacts = bundle()
    report = validate_bundle(artifacts, [chunk()])
    rows = build_rtm(artifacts)

    assert report.valid
    assert rows[0].requirement_id == "REQ-001"
    assert rows[0].scenario_ids == ["SCN-001"]
    assert rows[0].test_case_ids == ["TC-001"]
    assert rows[0].covered


def test_invalid_parent_and_excerpt_are_reported() -> None:
    artifacts = bundle()
    bad_case = artifacts.test_cases[0].model_copy(
        update={
            "scenario_id": "SCN-999",
            "source_references": [
                source().model_copy(update={"excerpt": "invented evidence"})
            ],
        }
    )
    artifacts = artifacts.model_copy(update={"test_cases": [bad_case]})

    report = validate_bundle(artifacts, [chunk()])

    assert not report.valid
    assert {issue.code for issue in report.issues} >= {
        "missing_scenario",
        "invalid_source_reference",
    }


def test_uncovered_requirement_remains_in_rtm() -> None:
    artifacts = bundle()
    second = Requirement(
        requirement_id="REQ-002",
        title="Audit sign in",
        description="Sign-in attempts are audited.",
        requirement_type=RequirementType.BUSINESS,
        module="Audit",
        priority=RequirementPriority.MEDIUM,
        source_references=[source()],
    )
    artifacts = artifacts.model_copy(
        update={"requirements": [*artifacts.requirements, second]}
    )

    report = validate_bundle(artifacts, [chunk()])
    rows = build_rtm(artifacts)

    assert report.uncovered_requirement_ids == ["REQ-002"]
    assert rows[1].requirement_id == "REQ-002"
    assert not rows[1].covered


def test_metrics_include_usage_and_duplicate_rate() -> None:
    artifacts = bundle()
    duplicate = artifacts.test_cases[0].model_copy(
        update={"test_case_id": "TC-002"}
    )
    artifacts = artifacts.model_copy(
        update={"test_cases": [*artifacts.test_cases, duplicate]}
    )
    report = validate_bundle(artifacts, [chunk()])

    metrics = compute_metrics(
        artifacts,
        report,
        input_tokens=100,
        output_tokens=50,
        latency_seconds=1.25,
        retries=1,
        schema_repairs=0,
        semantic_revisions=1,
        budget_exhausted=False,
    )

    assert metrics.input_tokens == 100
    assert metrics.output_tokens == 50
    assert metrics.duplicate_test_case_rate == 1.0
```

- [ ] **Step 2: Run the validation tests to verify they fail**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_validation.py -q
```

Expected: collection fails because `brd_srs_testgen.validation` does not exist.

- [ ] **Step 3: Implement deterministic validation, RTM, and metrics**

Create `src/brd_srs_testgen/validation.py`:

```python
from __future__ import annotations

import re
from collections import Counter

from .documents import verify_source_reference
from .models import (
    ArtifactBundle,
    DocumentChunk,
    RTMRow,
    RunMetrics,
    ScenarioType,
    ValidationIssue,
    ValidationReport,
)


def _duplicates(values: list[str]) -> set[str]:
    counts = Counter(values)
    return {value for value, count in counts.items() if count > 1}


def validate_bundle(
    bundle: ArtifactBundle, chunks: list[DocumentChunk]
) -> ValidationReport:
    issues: list[ValidationIssue] = []

    def add(code: str, artifact_id: str, message: str) -> None:
        issues.append(
            ValidationIssue(code=code, artifact_id=artifact_id, message=message)
        )

    requirements = {item.requirement_id: item for item in bundle.requirements}
    scenarios = {item.scenario_id: item for item in bundle.scenarios}
    test_cases = {item.test_case_id: item for item in bundle.test_cases}

    for value in sorted(
        _duplicates([item.requirement_id for item in bundle.requirements])
    ):
        add("duplicate_id", value, "Requirement ID is duplicated.")
    for value in sorted(_duplicates([item.scenario_id for item in bundle.scenarios])):
        add("duplicate_id", value, "Scenario ID is duplicated.")
    for value in sorted(_duplicates([item.test_case_id for item in bundle.test_cases])):
        add("duplicate_id", value, "Test-case ID is duplicated.")

    for requirement in bundle.requirements:
        for dependency_id in requirement.dependency_ids:
            if dependency_id not in requirements:
                add(
                    "missing_dependency",
                    requirement.requirement_id,
                    f"Dependency {dependency_id} does not exist.",
                )
        for reference in requirement.source_references:
            if not verify_source_reference(reference, chunks):
                add(
                    "invalid_source_reference",
                    requirement.requirement_id,
                    f"Invalid source reference {reference.chunk_id}.",
                )

    for scenario in bundle.scenarios:
        for requirement_id in scenario.requirement_ids:
            if requirement_id not in requirements:
                add(
                    "missing_requirement",
                    scenario.scenario_id,
                    f"Requirement {requirement_id} does not exist.",
                )
        for reference in scenario.source_references:
            if not verify_source_reference(reference, chunks):
                add(
                    "invalid_source_reference",
                    scenario.scenario_id,
                    f"Invalid source reference {reference.chunk_id}.",
                )

    for test_case in bundle.test_cases:
        scenario = scenarios.get(test_case.scenario_id)
        if scenario is None:
            add(
                "missing_scenario",
                test_case.test_case_id,
                f"Scenario {test_case.scenario_id} does not exist.",
            )
        for requirement_id in test_case.requirement_ids:
            if requirement_id not in requirements:
                add(
                    "missing_requirement",
                    test_case.test_case_id,
                    f"Requirement {requirement_id} does not exist.",
                )
            elif scenario and requirement_id not in scenario.requirement_ids:
                add(
                    "scenario_requirement_mismatch",
                    test_case.test_case_id,
                    f"Scenario {scenario.scenario_id} does not cover {requirement_id}.",
                )
        for reference in test_case.source_references:
            if not verify_source_reference(reference, chunks):
                add(
                    "invalid_source_reference",
                    test_case.test_case_id,
                    f"Invalid source reference {reference.chunk_id}.",
                )

    scenario_ids_with_cases = {item.scenario_id for item in bundle.test_cases}
    orphan_scenarios = sorted(set(scenarios) - scenario_ids_with_cases)
    orphan_cases = sorted(
        item.test_case_id
        for item in bundle.test_cases
        if item.scenario_id not in scenarios
    )
    covered_by_scenario = {
        requirement_id
        for scenario in bundle.scenarios
        for requirement_id in scenario.requirement_ids
        if requirement_id in requirements
    }
    covered_by_case = {
        requirement_id
        for test_case in bundle.test_cases
        for requirement_id in test_case.requirement_ids
        if requirement_id in requirements
    }
    uncovered = sorted(set(requirements) - (covered_by_scenario & covered_by_case))

    for scenario_id in orphan_scenarios:
        add("orphan_scenario", scenario_id, "Scenario has no test case.")
    for test_case_id in orphan_cases:
        add("orphan_test_case", test_case_id, "Test case has no valid scenario.")
    for requirement_id in uncovered:
        add(
            "uncovered_requirement",
            requirement_id,
            "Requirement lacks both scenario and test-case coverage.",
        )

    return ValidationReport(
        valid=not issues,
        issues=issues,
        uncovered_requirement_ids=uncovered,
        orphan_scenario_ids=orphan_scenarios,
        orphan_test_case_ids=orphan_cases,
    )


def build_rtm(bundle: ArtifactBundle) -> list[RTMRow]:
    rows: list[RTMRow] = []
    for requirement in bundle.requirements:
        scenarios = [
            item
            for item in bundle.scenarios
            if requirement.requirement_id in item.requirement_ids
        ]
        test_cases = [
            item
            for item in bundle.test_cases
            if requirement.requirement_id in item.requirement_ids
        ]
        source_ids = {
            reference.chunk_id for reference in requirement.source_references
        }
        source_ids.update(
            reference.chunk_id
            for scenario in scenarios
            for reference in scenario.source_references
        )
        source_ids.update(
            reference.chunk_id
            for test_case in test_cases
            for reference in test_case.source_references
        )
        rows.append(
            RTMRow(
                requirement_id=requirement.requirement_id,
                scenario_ids=sorted(item.scenario_id for item in scenarios),
                test_case_ids=sorted(item.test_case_id for item in test_cases),
                source_chunk_ids=sorted(source_ids),
                covered=bool(scenarios and test_cases),
            )
        )
    return rows


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _tokens(test_case) -> set[str]:
    text = " ".join(
        [
            test_case.title,
            *(
                f"{step.action} {step.expected_result}"
                for step in test_case.steps
            ),
        ]
    ).casefold()
    words = re.findall(r"\w+", text)
    if len(words) < 3:
        return set(words)
    return {" ".join(words[index : index + 3]) for index in range(len(words) - 2)}


def _duplicate_rate(bundle: ArtifactBundle) -> float:
    cases = bundle.test_cases
    pairs = 0
    duplicates = 0
    for left_index, left in enumerate(cases):
        left_tokens = _tokens(left)
        for right in cases[left_index + 1 :]:
            pairs += 1
            right_tokens = _tokens(right)
            union = left_tokens | right_tokens
            similarity = len(left_tokens & right_tokens) / len(union) if union else 1.0
            duplicates += similarity >= 0.85
    return _ratio(duplicates, pairs)


def compute_metrics(
    bundle: ArtifactBundle,
    report: ValidationReport,
    *,
    input_tokens: int,
    output_tokens: int,
    latency_seconds: float,
    retries: int,
    schema_repairs: int,
    semantic_revisions: int,
    budget_exhausted: bool,
) -> RunMetrics:
    requirement_ids = {item.requirement_id for item in bundle.requirements}
    scenario_covered = {
        requirement_id
        for scenario in bundle.scenarios
        for requirement_id in scenario.requirement_ids
        if requirement_id in requirement_ids
    }
    case_covered = {
        requirement_id
        for test_case in bundle.test_cases
        for requirement_id in test_case.requirement_ids
        if requirement_id in requirement_ids
    }
    positive = {
        requirement_id
        for scenario in bundle.scenarios
        if scenario.scenario_type == ScenarioType.POSITIVE
        for requirement_id in scenario.requirement_ids
        if requirement_id in requirement_ids
    }
    non_positive = {
        requirement_id
        for scenario in bundle.scenarios
        if scenario.scenario_type != ScenarioType.POSITIVE
        for requirement_id in scenario.requirement_ids
        if requirement_id in requirement_ids
    }
    artifacts = [*bundle.requirements, *bundle.scenarios, *bundle.test_cases]
    invalid_ids = {
        issue.artifact_id
        for issue in report.issues
        if issue.code == "invalid_source_reference"
    }
    reference_count = sum(len(item.source_references) for item in artifacts)
    invalid_reference_count = sum(
        issue.code == "invalid_source_reference" for issue in report.issues
    )
    covered_rows = sum(row.covered for row in build_rtm(bundle))
    orphan_count = len(report.orphan_scenario_ids) + len(report.orphan_test_case_ids)

    return RunMetrics(
        completion=report.valid,
        schema_valid=True,
        citation_coverage=_ratio(
            sum(item.requirement_id not in invalid_ids for item in bundle.requirements)
            + sum(item.scenario_id not in invalid_ids for item in bundle.scenarios)
            + sum(item.test_case_id not in invalid_ids for item in bundle.test_cases),
            len(artifacts),
        ),
        requirement_scenario_coverage=_ratio(
            len(scenario_covered), len(requirement_ids)
        ),
        requirement_test_case_coverage=_ratio(
            len(case_covered), len(requirement_ids)
        ),
        positive_scenario_coverage=_ratio(len(positive), len(requirement_ids)),
        non_positive_scenario_coverage=_ratio(
            len(non_positive), len(requirement_ids)
        ),
        rtm_completeness=_ratio(covered_rows, len(requirement_ids)),
        orphan_rate=_ratio(orphan_count, len(bundle.scenarios) + len(bundle.test_cases)),
        invalid_reference_rate=_ratio(invalid_reference_count, reference_count),
        duplicate_test_case_rate=_duplicate_rate(bundle),
        requirement_count=len(bundle.requirements),
        scenario_count=len(bundle.scenarios),
        test_case_count=len(bundle.test_cases),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=latency_seconds,
        retries=retries,
        schema_repairs=schema_repairs,
        semantic_revisions=semantic_revisions,
        budget_exhausted=budget_exhausted,
    )
```

- [ ] **Step 4: Run the validation tests**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_validation.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit deterministic validation and metrics**

```bash
rtk git add src/brd_srs_testgen/validation.py tests/test_validation.py
rtk git commit -m "feat: validate traceability and metrics"
```

### Task 5: Persist comparison runs atomically and immutably

**Files:**
- Create: `src/brd_srs_testgen/storage.py`
- Create: `tests/test_storage.py`

- [ ] **Step 1: Write failing storage tests**

Create `tests/test_storage.py`:

```python
from datetime import UTC, datetime

import pytest

from brd_srs_testgen.models import (
    ComparisonManifest,
    Condition,
    ConditionManifest,
    RunStatus,
)
from brd_srs_testgen.storage import ImmutableArtifactError, RunStore
from tests.factories import bundle, chunk


def comparison_manifest() -> ComparisonManifest:
    return ComparisonManifest(
        comparison_id="20260811T000000Z-ecac9f035813",
        document_hash="ecac9f0358134f174bcbf0d60ddbc7c25bcb4f812ea8e4c57bfbd8c02edaa274",
        provider="ollama",
        model="gemma4",
        temperature=0.0,
        token_ceiling=1000,
        condition_order=list(Condition),
        prompt_version="1",
        schema_version="1",
        started_at=datetime.now(UTC).isoformat(),
    )


def condition_manifest() -> ConditionManifest:
    return ConditionManifest(
        condition=Condition.SINGLE_PROMPT,
        status=RunStatus.RUNNING,
        provider="ollama",
        model="gemma4",
        temperature=0.0,
        token_ceiling=1000,
        started_at=datetime.now(UTC).isoformat(),
    )


def test_create_comparison_writes_manifest_and_chunks(tmp_path) -> None:
    store = RunStore(tmp_path)
    manifest = comparison_manifest()

    directory = store.create_comparison(manifest, [chunk()])

    assert (directory / "manifest.json").exists()
    assert (directory / "chunks.json").exists()


def test_artifacts_cannot_be_overwritten(tmp_path) -> None:
    store = RunStore(tmp_path)
    manifest = comparison_manifest()
    store.create_comparison(manifest, [chunk()])
    store.start_condition(manifest.comparison_id, condition_manifest())
    store.write_artifact(
        manifest.comparison_id,
        Condition.SINGLE_PROMPT,
        "requirements.json",
        [item.model_dump(mode="json") for item in bundle().requirements],
    )

    with pytest.raises(ImmutableArtifactError):
        store.write_artifact(
            manifest.comparison_id,
            Condition.SINGLE_PROMPT,
            "requirements.json",
            [],
        )


def test_events_are_appended_with_atomic_replacement(tmp_path) -> None:
    store = RunStore(tmp_path)
    manifest = comparison_manifest()
    store.create_comparison(manifest, [chunk()])
    store.start_condition(manifest.comparison_id, condition_manifest())

    store.append_event(
        manifest.comparison_id, Condition.SINGLE_PROMPT, {"stage": "start"}
    )
    store.append_event(
        manifest.comparison_id, Condition.SINGLE_PROMPT, {"stage": "finish"}
    )

    events = (
        tmp_path
        / manifest.comparison_id
        / "conditions"
        / Condition.SINGLE_PROMPT
        / "events.jsonl"
    ).read_text().splitlines()
    assert len(events) == 2
```

- [ ] **Step 2: Run the storage tests to verify they fail**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_storage.py -q
```

Expected: collection fails because `brd_srs_testgen.storage` does not exist.

- [ ] **Step 3: Implement atomic immutable storage**

Create `src/brd_srs_testgen/storage.py`:

```python
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .models import (
    ComparisonManifest,
    Condition,
    ConditionManifest,
    DocumentChunk,
)


class ImmutableArtifactError(RuntimeError):
    pass


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


class RunStore:
    def __init__(self, root: str | Path = "runs") -> None:
        self.root = Path(root)

    def comparison_dir(self, comparison_id: str) -> Path:
        return self.root / comparison_id

    def condition_dir(self, comparison_id: str, condition: Condition) -> Path:
        return self.comparison_dir(comparison_id) / "conditions" / condition.value

    def create_comparison(
        self, manifest: ComparisonManifest, chunks: list[DocumentChunk]
    ) -> Path:
        directory = self.comparison_dir(manifest.comparison_id)
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise ImmutableArtifactError(
                f"Comparison {manifest.comparison_id} already exists."
            ) from error
        _atomic_json(directory / "manifest.json", manifest.model_dump(mode="json"))
        _atomic_json(
            directory / "chunks.json",
            [item.model_dump(mode="json") for item in chunks],
        )
        return directory

    def update_comparison(self, manifest: ComparisonManifest) -> None:
        _atomic_json(
            self.comparison_dir(manifest.comparison_id) / "manifest.json",
            manifest.model_dump(mode="json"),
        )

    def start_condition(
        self, comparison_id: str, manifest: ConditionManifest
    ) -> Path:
        directory = self.condition_dir(comparison_id, manifest.condition)
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise ImmutableArtifactError(
                f"Condition {manifest.condition.value} already exists."
            ) from error
        _atomic_json(directory / "manifest.json", manifest.model_dump(mode="json"))
        return directory

    def update_condition(
        self, comparison_id: str, manifest: ConditionManifest
    ) -> None:
        _atomic_json(
            self.condition_dir(comparison_id, manifest.condition) / "manifest.json",
            manifest.model_dump(mode="json"),
        )

    def write_artifact(
        self,
        comparison_id: str,
        condition: Condition,
        filename: str,
        value: Any,
    ) -> Path:
        path = self.condition_dir(comparison_id, condition) / filename
        if path.exists():
            raise ImmutableArtifactError(f"Artifact already exists: {path}")
        _atomic_json(path, value)
        return path

    def append_event(
        self, comparison_id: str, condition: Condition, event: dict[str, Any]
    ) -> None:
        path = self.condition_dir(comparison_id, condition) / "events.jsonl"
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        line = json.dumps(event, ensure_ascii=False, sort_keys=True)
        _atomic_text(path, f"{existing}{line}\n")
```

- [ ] **Step 4: Run the storage tests**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_storage.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit immutable run storage**

```bash
rtk git add src/brd_srs_testgen/storage.py tests/test_storage.py
rtk git commit -m "feat: persist comparison runs"
```

### Task 6: Implement prompts, retry policy, single-prompt, and staged conditions

**Files:**
- Create: `src/brd_srs_testgen/pipelines.py`
- Create: `tests/test_pipelines.py`

- [ ] **Step 1: Write failing single-prompt, staged, retry, and repair tests**

Create `tests/test_pipelines.py`:

```python
from collections import deque

from brd_srs_testgen.models import (
    ArtifactBundle,
    RequirementBatch,
    ReviewResult,
    ScenarioBatch,
    TestCaseBatch,
)
from brd_srs_testgen.pipelines import (
    PipelineContext,
    run_single_prompt,
    run_staged_single_agent,
)
from brd_srs_testgen.providers import (
    BudgetLedger,
    GenerationResult,
    ProviderError,
    StructuredOutputError,
)
from tests.factories import bundle, chunk


class ScriptedProvider:
    model = "test-model"

    def __init__(self, responses) -> None:
        self.responses = deque(responses)
        self.ledger = BudgetLedger(100_000)
        self.calls = []

    def generate(self, messages, schema, *, max_output_tokens):
        self.calls.append((messages, schema, max_output_tokens))
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        value = schema.model_validate(response.model_dump(mode="json"))
        return GenerationResult(
            value=value,
            input_tokens=1,
            output_tokens=1,
            latency_seconds=0.01,
        )


def accepted() -> ReviewResult:
    return ReviewResult(accepted=True)


def test_single_prompt_returns_one_bundle() -> None:
    provider = ScriptedProvider([bundle()])
    context = PipelineContext(provider=provider, sleep=lambda _seconds: None)

    result = run_single_prompt(context, [chunk()])

    assert result.test_cases[0].test_case_id == "TC-001"
    assert len(provider.calls) == 1


def test_staged_condition_preserves_sequential_history() -> None:
    artifacts = bundle()
    provider = ScriptedProvider(
        [
            RequirementBatch(requirements=artifacts.requirements),
            accepted(),
            ScenarioBatch(scenarios=artifacts.scenarios),
            accepted(),
            TestCaseBatch(test_cases=artifacts.test_cases),
            accepted(),
        ]
    )
    context = PipelineContext(provider=provider, sleep=lambda _seconds: None)

    result = run_staged_single_agent(context, [chunk()])

    assert result == artifacts
    assert len(provider.calls[2][0]) > len(provider.calls[0][0])


def test_transport_failure_retries_twice_at_most() -> None:
    transient = ProviderError("busy", code=503, retryable=True)
    provider = ScriptedProvider([transient, transient, bundle()])
    delays = []
    context = PipelineContext(provider=provider, sleep=delays.append)

    result = run_single_prompt(context, [chunk()])

    assert isinstance(result, ArtifactBundle)
    assert context.retries == 2
    assert delays == [1, 2]


def test_single_prompt_gets_one_schema_repair() -> None:
    provider = ScriptedProvider(
        [
            StructuredOutputError("bad", input_tokens=2, output_tokens=3),
            bundle(),
        ]
    )
    context = PipelineContext(provider=provider, sleep=lambda _seconds: None)

    run_single_prompt(context, [chunk()])

    assert context.schema_repairs == 1
    assert (context.input_tokens, context.output_tokens) == (3, 4)
    assert "invalid response" in provider.calls[1][0][-1]["content"]
```

- [ ] **Step 2: Run focused pipeline tests to verify they fail**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_pipelines.py -q
```

Expected: collection fails because `brd_srs_testgen.pipelines` does not exist.

- [ ] **Step 3: Implement prompts and bounded generation context**

Create `src/brd_srs_testgen/pipelines.py` with these imports, constants, prompt builders, and context:

```python
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel

from .documents import render_chunks
from .models import (
    ArtifactBundle,
    DocumentChunk,
    GeneratedCases,
    Requirement,
    RequirementBatch,
    ReviewResult,
    ScenarioBatch,
    TestCaseBatch,
)
from .providers import (
    Messages,
    ProviderError,
    StructuredOutputError,
    StructuredProvider,
)


T = TypeVar("T", bound=BaseModel)
PROMPT_VERSION = "research-core-v1"
WORKER_COUNT = 3

RULES = """Use English. Return only the requested schema. Use canonical IDs:
REQ-001, SCN-001, and TC-001 with unique increasing numbers. Every artifact must
cite real chunk IDs and short verbatim excerpts. Do not invent requirements,
behavior, pages, citations, or expected results unsupported by the evidence.
"""


def _user(content: str) -> dict[str, str]:
    return {"role": "user", "content": content}


def _assistant(value: BaseModel) -> dict[str, str]:
    return {"role": "assistant", "content": value.model_dump_json()}


def single_prompt(chunks: list[DocumentChunk]) -> str:
    return f"""{RULES}
Extract exhaustive functional, non-functional, and business requirements. Create
positive, negative, boundary, edge, and state-transition scenarios where the
evidence supports them. Create executable manual test cases with ordered actions
and expected results. Return one ArtifactBundle.

EVIDENCE
{render_chunks(chunks)}
"""


def requirements_prompt(chunks: list[DocumentChunk]) -> str:
    return f"""{RULES}
Extract and consolidate all supported requirements from every evidence chunk.
Return RequirementBatch.

EVIDENCE
{render_chunks(chunks)}
"""


def scenarios_prompt(
    chunks: list[DocumentChunk], requirements: RequirementBatch
) -> str:
    return f"""{RULES}
Generate traceable scenarios for these validated requirements. Include positive
and relevant non-positive types. Return ScenarioBatch.

REQUIREMENTS
{requirements.model_dump_json()}

EVIDENCE
{render_chunks(chunks)}
"""


def test_cases_prompt(
    chunks: list[DocumentChunk],
    requirements: RequirementBatch,
    scenarios: ScenarioBatch,
) -> str:
    return f"""{RULES}
Generate detailed manual test cases for these scenarios. Each step needs one
action and its observable expected result. Return TestCaseBatch.

REQUIREMENTS
{requirements.model_dump_json()}

SCENARIOS
{scenarios.model_dump_json()}

EVIDENCE
{render_chunks(chunks)}
"""


def review_prompt(label: str, value: BaseModel, chunks: list[DocumentChunk]) -> str:
    return f"""{RULES}
Review the {label} for groundedness, completeness, duplicate IDs, relationship
errors, and invalid citations. Return ReviewResult. accepted must be false when
any semantic correction is required.

ARTIFACT
{value.model_dump_json()}

EVIDENCE
{render_chunks(chunks)}
"""


def revision_prompt(
    label: str,
    value: BaseModel,
    review: ReviewResult,
    chunks: list[DocumentChunk],
) -> str:
    return f"""{RULES}
Revise the {label} once to address every review issue. Preserve supported content
and return the same artifact schema.

ARTIFACT
{value.model_dump_json()}

ISSUES
{review.model_dump_json()}

EVIDENCE
{render_chunks(chunks)}
"""


@dataclass
class PipelineContext:
    provider: StructuredProvider
    sleep: Callable[[float], None] = time.sleep
    retries: int = 0
    schema_repairs: int = 0
    semantic_revisions: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _record(self, result) -> None:
        with self._lock:
            self.input_tokens += result.input_tokens
            self.output_tokens += result.output_tokens
            self.latency_seconds += result.latency_seconds

    def generate(
        self,
        messages: Messages,
        schema: type[T],
        *,
        max_output_tokens: int,
        allow_schema_repair: bool = True,
    ) -> T:
        transport_attempts = 0
        repaired = False
        current_messages = list(messages)
        while True:
            try:
                result = self.provider.generate(
                    current_messages,
                    schema,
                    max_output_tokens=max_output_tokens,
                )
                self._record(result)
                return result.value
            except ProviderError as error:
                if not error.retryable or transport_attempts >= 2:
                    raise
                delay = 2**transport_attempts
                transport_attempts += 1
                with self._lock:
                    self.retries += 1
                self.sleep(delay)
            except StructuredOutputError as error:
                self._record(error)
                if not allow_schema_repair or repaired:
                    raise
                repaired = True
                with self._lock:
                    self.schema_repairs += 1
                current_messages = [
                    _user(
                        f"Return valid JSON matching the supplied schema. Preserve only "
                        f"supported content from this invalid response:\n{error.raw_text}"
                    )
                ]

    def revise(
        self,
        label: str,
        value: T,
        review: ReviewResult,
        chunks: list[DocumentChunk],
        schema: type[T],
        *,
        max_output_tokens: int,
    ) -> T:
        with self._lock:
            self.semantic_revisions += 1
        return self.generate(
            [_user(revision_prompt(label, value, review, chunks))],
            schema,
            max_output_tokens=max_output_tokens,
        )
```

- [ ] **Step 4: Add the complete single-prompt and staged functions**

Append to `src/brd_srs_testgen/pipelines.py`:

```python
def run_single_prompt(
    context: PipelineContext, chunks: list[DocumentChunk]
) -> ArtifactBundle:
    return context.generate(
        [_user(single_prompt(chunks))],
        ArtifactBundle,
        max_output_tokens=30_000,
    )


def _review_once(
    context: PipelineContext,
    history: Messages,
    label: str,
    value: T,
    chunks: list[DocumentChunk],
    schema: type[T],
    *,
    max_output_tokens: int,
) -> T:
    review = context.generate(
        [*history, _user(review_prompt(label, value, chunks))],
        ReviewResult,
        max_output_tokens=4_000,
    )
    if review.accepted:
        return value
    return context.revise(
        label,
        value,
        review,
        chunks,
        schema,
        max_output_tokens=max_output_tokens,
    )


def run_staged_single_agent(
    context: PipelineContext, chunks: list[DocumentChunk]
) -> ArtifactBundle:
    history: Messages = []

    requirement_request = _user(requirements_prompt(chunks))
    requirements = context.generate(
        [*history, requirement_request],
        RequirementBatch,
        max_output_tokens=12_000,
    )
    history.extend([requirement_request, _assistant(requirements)])
    requirements = _review_once(
        context,
        history,
        "requirements",
        requirements,
        chunks,
        RequirementBatch,
        max_output_tokens=12_000,
    )
    history[-1] = _assistant(requirements)

    scenario_request = _user(scenarios_prompt(chunks, requirements))
    scenarios = context.generate(
        [*history, scenario_request],
        ScenarioBatch,
        max_output_tokens=12_000,
    )
    history.extend([scenario_request, _assistant(scenarios)])
    scenarios = _review_once(
        context,
        history,
        "scenarios",
        scenarios,
        chunks,
        ScenarioBatch,
        max_output_tokens=12_000,
    )
    history[-1] = _assistant(scenarios)

    test_case_request = _user(test_cases_prompt(chunks, requirements, scenarios))
    test_cases = context.generate(
        [*history, test_case_request],
        TestCaseBatch,
        max_output_tokens=20_000,
    )
    history.extend([test_case_request, _assistant(test_cases)])
    test_cases = _review_once(
        context,
        history,
        "test cases",
        test_cases,
        chunks,
        TestCaseBatch,
        max_output_tokens=20_000,
    )

    return ArtifactBundle(
        requirements=requirements.requirements,
        scenarios=scenarios.scenarios,
        test_cases=test_cases.test_cases,
    )
```

- [ ] **Step 5: Run the focused pipeline tests**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_pipelines.py -q
```

Expected: `4 passed`.

- [ ] **Step 6: Commit single and staged pipelines**

```bash
rtk git add src/brd_srs_testgen/pipelines.py tests/test_pipelines.py
rtk git commit -m "feat: add single and staged pipelines"
```

### Task 7: Add centralized three-worker coordination

**Files:**
- Modify: `src/brd_srs_testgen/pipelines.py`
- Modify: `tests/test_pipelines.py`

- [ ] **Step 1: Add a failing centralized-isolation test**

Add these imports to `tests/test_pipelines.py`:

```python
import threading

from brd_srs_testgen.models import GeneratedCases
from brd_srs_testgen.pipelines import run_centralized_multi_agent
```

Append this provider and test:

```python
class CentralProvider:
    model = "test-model"

    def __init__(self) -> None:
        self.ledger = BudgetLedger(100_000)
        self.calls = []
        self.lock = threading.Lock()
        self.artifacts = bundle()

    def generate(self, messages, schema, *, max_output_tokens):
        content = messages[-1]["content"]
        with self.lock:
            self.calls.append((messages, schema, max_output_tokens))
        if schema is RequirementBatch:
            value = RequirementBatch(
                requirements=(
                    self.artifacts.requirements
                    if "p0001-c001" in content or "CANDIDATES" in content
                    else []
                )
            )
        elif schema is GeneratedCases:
            assigned = '"requirements":[]' not in content.replace(" ", "")
            value = GeneratedCases(
                scenarios=self.artifacts.scenarios if assigned else [],
                test_cases=self.artifacts.test_cases if assigned else [],
            )
        elif schema is ReviewResult:
            value = ReviewResult(accepted=True)
        else:
            value = self.artifacts
        return GenerationResult(
            value=schema.model_validate(value.model_dump(mode="json")),
            input_tokens=1,
            output_tokens=1,
            latency_seconds=0.01,
        )


def test_centralized_workers_receive_isolated_assignments() -> None:
    provider = CentralProvider()
    context = PipelineContext(provider=provider, sleep=lambda _seconds: None)

    result = run_centralized_multi_agent(context, [chunk()])

    assert result == bundle()
    worker_calls = [
        call for call in provider.calls if "WORKER REQUIREMENT EXTRACTION" in call[0][0]["content"]
    ]
    assert len(worker_calls) == 3
    assert all(len(call[0]) == 1 for call in worker_calls)
```

- [ ] **Step 2: Run the centralized test to verify it fails**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_pipelines.py::test_centralized_workers_receive_isolated_assignments -q
```

Expected: import fails because `run_centralized_multi_agent` does not exist.

- [ ] **Step 3: Add balancing and centralized prompt builders**

Append these complete helpers to `src/brd_srs_testgen/pipelines.py` before the condition functions:

```python
def _balance(items: list[T], weight: Callable[[T], int]) -> list[list[T]]:
    groups: list[list[T]] = [[] for _ in range(WORKER_COUNT)]
    totals = [0] * WORKER_COUNT
    for item in sorted(items, key=weight, reverse=True):
        index = min(range(WORKER_COUNT), key=totals.__getitem__)
        groups[index].append(item)
        totals[index] += weight(item)
    return groups


def worker_requirements_prompt(
    worker_index: int, chunks: list[DocumentChunk]
) -> str:
    first_id = worker_index * 1000 + 1
    return f"""{RULES}
WORKER REQUIREMENT EXTRACTION {worker_index + 1}/{WORKER_COUNT}
Inspect only this assigned evidence. Return RequirementBatch. Use candidate IDs
starting at REQ-{first_id:03d}. An empty assignment returns an empty list.

EVIDENCE
{render_chunks(chunks)}
"""


def reconcile_requirements_prompt(
    chunks: list[DocumentChunk], candidates: list[Requirement]
) -> str:
    candidate_json = json.dumps(
        [item.model_dump(mode="json") for item in candidates],
        ensure_ascii=False,
    )
    return f"""{RULES}
Reconcile CANDIDATES from isolated workers. Remove duplicates, preserve supported
dependencies, resolve conflicting wording from evidence, and renumber final IDs
contiguously from REQ-001. Return RequirementBatch.

CANDIDATES
{candidate_json}

EVIDENCE
{render_chunks(chunks)}
"""


def worker_cases_prompt(
    worker_index: int,
    requirements: list[Requirement],
    chunks: list[DocumentChunk],
) -> str:
    first_id = worker_index * 1000 + 1
    requirement_json = json.dumps(
        [item.model_dump(mode="json") for item in requirements],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"""{RULES}
WORKER CASE GENERATION {worker_index + 1}/{WORKER_COUNT}
Generate scenarios and detailed test cases only for the assigned requirements.
Use scenario and test-case IDs starting at SCN-{first_id:03d} and TC-{first_id:03d}.
Return GeneratedCases. An empty assignment returns empty lists.

ASSIGNED REQUIREMENTS
{{"requirements":{requirement_json}}}

EVIDENCE
{render_chunks(chunks)}
"""
```

- [ ] **Step 4: Implement the centralized condition**

Append to `src/brd_srs_testgen/pipelines.py`:

```python
def _relevant_chunks(
    requirements: list[Requirement], chunks: list[DocumentChunk]
) -> list[DocumentChunk]:
    chunk_ids = {
        reference.chunk_id
        for requirement in requirements
        for reference in requirement.source_references
    }
    return [item for item in chunks if item.chunk_id in chunk_ids]


def run_centralized_multi_agent(
    context: PipelineContext, chunks: list[DocumentChunk]
) -> ArtifactBundle:
    chunk_groups = _balance(chunks, lambda item: len(item.text))

    def extract(worker_index: int) -> RequirementBatch:
        return context.generate(
            [_user(worker_requirements_prompt(worker_index, chunk_groups[worker_index]))],
            RequirementBatch,
            max_output_tokens=8_000,
        )

    with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
        requirement_batches = list(executor.map(extract, range(WORKER_COUNT)))

    candidates = [
        requirement
        for batch in requirement_batches
        for requirement in batch.requirements
    ]
    requirements = context.generate(
        [_user(reconcile_requirements_prompt(chunks, candidates))],
        RequirementBatch,
        max_output_tokens=12_000,
    )
    requirement_groups = _balance(
        requirements.requirements,
        lambda item: len(item.description) + len(item.source_references) * 100,
    )

    def generate_cases(worker_index: int) -> GeneratedCases:
        assigned = requirement_groups[worker_index]
        evidence = _relevant_chunks(assigned, chunks)
        return context.generate(
            [_user(worker_cases_prompt(worker_index, assigned, evidence))],
            GeneratedCases,
            max_output_tokens=16_000,
        )

    with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
        generated = list(executor.map(generate_cases, range(WORKER_COUNT)))

    bundle = ArtifactBundle(
        requirements=requirements.requirements,
        scenarios=[scenario for batch in generated for scenario in batch.scenarios],
        test_cases=[test_case for batch in generated for test_case in batch.test_cases],
    )
    review = context.generate(
        [_user(review_prompt("centralized artifact bundle", bundle, chunks))],
        ReviewResult,
        max_output_tokens=4_000,
    )
    if review.accepted:
        return bundle
    return context.revise(
        "centralized artifact bundle",
        bundle,
        review,
        chunks,
        ArtifactBundle,
        max_output_tokens=30_000,
    )
```

- [ ] **Step 5: Run all pipeline tests**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_pipelines.py -q
```

Expected: `5 passed`.

- [ ] **Step 6: Commit centralized coordination**

```bash
rtk git add src/brd_srs_testgen/pipelines.py tests/test_pipelines.py
rtk git commit -m "feat: add centralized pipeline"
```

### Task 8: Orchestrate and persist one three-condition comparison

**Files:**
- Create: `src/brd_srs_testgen/runner.py`
- Create: `tests/test_runner.py`
- Modify: `src/brd_srs_testgen/storage.py`

- [ ] **Step 1: Add top-level comparison failure persistence**

Add this method to `RunStore` in `src/brd_srs_testgen/storage.py`:

```python
    def write_comparison_artifact(
        self, comparison_id: str, filename: str, value: Any
    ) -> Path:
        path = self.comparison_dir(comparison_id) / filename
        if path.exists():
            raise ImmutableArtifactError(f"Artifact already exists: {path}")
        _atomic_json(path, value)
        return path
```

- [ ] **Step 2: Write failing orchestration and failure-isolation tests**

Create `tests/test_runner.py`:

```python
from collections import deque
import threading
from unittest.mock import patch

from brd_srs_testgen.documents import DocumentError
from brd_srs_testgen.models import (
    Condition,
    GeneratedCases,
    RequirementBatch,
    ReviewResult,
    RunStatus,
    ScenarioBatch,
    TestCaseBatch,
)
from brd_srs_testgen.providers import (
    BudgetLedger,
    GenerationResult,
    ProviderError,
)
from brd_srs_testgen.runner import ProviderSettings, run_comparison
from brd_srs_testgen.storage import RunStore
from tests.factories import bundle, chunk


class Provider:
    model = "test-model"

    def __init__(self, responses, ledger) -> None:
        self.responses = deque(responses)
        self.ledger = ledger

    def generate(self, messages, schema, *, max_output_tokens):
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return GenerationResult(
            value=schema.model_validate(response.model_dump(mode="json")),
            input_tokens=1,
            output_tokens=1,
            latency_seconds=0.01,
        )


class CentralProvider:
    model = "test-model"

    def __init__(self, ledger) -> None:
        self.ledger = ledger
        self.lock = threading.Lock()
        self.artifacts = bundle()

    def generate(self, messages, schema, *, max_output_tokens):
        content = messages[-1]["content"]
        with self.lock:
            if schema is RequirementBatch:
                value = RequirementBatch(
                    requirements=(
                        self.artifacts.requirements
                        if "p0001-c001" in content or "CANDIDATES" in content
                        else []
                    )
                )
            elif schema is GeneratedCases:
                assigned = '"requirements":[]' not in content.replace(" ", "")
                value = GeneratedCases(
                    scenarios=self.artifacts.scenarios if assigned else [],
                    test_cases=self.artifacts.test_cases if assigned else [],
                )
            else:
                value = accepted()
        return GenerationResult(
            value=schema.model_validate(value.model_dump(mode="json")),
            input_tokens=1,
            output_tokens=1,
            latency_seconds=0.01,
        )


def accepted() -> ReviewResult:
    return ReviewResult(accepted=True)


def scripts():
    artifacts = bundle()
    return {
        Condition.SINGLE_PROMPT: [artifacts],
        Condition.STAGED_SINGLE_AGENT: [
            RequirementBatch(requirements=artifacts.requirements),
            accepted(),
            ScenarioBatch(scenarios=artifacts.scenarios),
            accepted(),
            TestCaseBatch(test_cases=artifacts.test_cases),
            accepted(),
        ],
    }


def settings() -> ProviderSettings:
    return ProviderSettings(provider="ollama", model="gemma4", token_ceiling=100_000)


def test_comparison_runs_and_persists_all_conditions(tmp_path) -> None:
    scripted = scripts()

    def factory(condition, ledger):
        if condition == Condition.CENTRALIZED_MULTI_AGENT:
            return CentralProvider(ledger)
        return Provider(scripted[condition], ledger)

    with patch("brd_srs_testgen.runner.parse_pdf", return_value=[chunk()]):
        result = run_comparison(
            b"pdf",
            settings(),
            store=RunStore(tmp_path),
            provider_factory=factory,
        )

    assert set(result.conditions) == set(Condition)
    assert all(
        item.manifest.status == RunStatus.COMPLETED
        for item in result.conditions.values()
    )
    assert (
        tmp_path
        / result.manifest.comparison_id
        / "conditions"
        / Condition.SINGLE_PROMPT
        / "rtm.json"
    ).exists()


def test_one_failed_condition_does_not_stop_the_others(tmp_path) -> None:
    scripted = scripts()
    scripted[Condition.SINGLE_PROMPT] = [
        ProviderError("rejected", code=400, retryable=False)
    ]

    def factory(condition, ledger):
        if condition == Condition.CENTRALIZED_MULTI_AGENT:
            return CentralProvider(ledger)
        return Provider(scripted[condition], ledger)

    with patch("brd_srs_testgen.runner.parse_pdf", return_value=[chunk()]):
        result = run_comparison(
            b"pdf",
            settings(),
            store=RunStore(tmp_path),
            provider_factory=factory,
        )

    assert result.conditions[Condition.SINGLE_PROMPT].manifest.status == RunStatus.FAILED
    assert (
        result.conditions[Condition.STAGED_SINGLE_AGENT].manifest.status
        == RunStatus.COMPLETED
    )


def test_parsing_failure_is_persisted(tmp_path) -> None:
    with patch(
        "brd_srs_testgen.runner.parse_pdf",
        side_effect=DocumentError("PDF contains insufficient extractable text."),
    ):
        result = run_comparison(b"pdf", settings(), store=RunStore(tmp_path))

    assert result.failure_category == "parsing"
    assert (
        tmp_path / result.manifest.comparison_id / "failure.json"
    ).exists()
```

- [ ] **Step 3: Run runner tests to verify they fail**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_runner.py -q
```

Expected: collection fails because `brd_srs_testgen.runner` does not exist.

- [ ] **Step 4: Implement provider settings and result types**

Create `src/brd_srs_testgen/runner.py` with these definitions:

```python
from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from google import genai

from .documents import DocumentError, parse_pdf
from .models import (
    ArtifactBundle,
    ComparisonManifest,
    Condition,
    ConditionManifest,
    FailureCategory,
    RTMRow,
    RunMetrics,
    RunStatus,
    ValidationReport,
)
from .pipelines import (
    PROMPT_VERSION,
    PipelineContext,
    run_centralized_multi_agent,
    run_single_prompt,
    run_staged_single_agent,
)
from .providers import (
    BudgetExceeded,
    BudgetLedger,
    GeminiProvider,
    OllamaProvider,
    ProviderError,
    StructuredOutputError,
    StructuredProvider,
)
from .storage import RunStore
from .validation import build_rtm, compute_metrics, validate_bundle


SCHEMA_VERSION = "research-core-v1"
ProviderFactory = Callable[[Condition, BudgetLedger], StructuredProvider]
Progress = Callable[[Condition | None, str], None]


@dataclass(frozen=True)
class ProviderSettings:
    provider: str
    model: str
    token_ceiling: int
    api_key: str = field(default="", repr=False)
    base_url: str = "http://localhost:11434"

    def validate(self) -> None:
        if self.provider not in {"gemini", "ollama"}:
            raise ValueError("Provider must be gemini or ollama.")
        if not self.model.strip():
            raise ValueError("Model identifier is required.")
        if self.token_ceiling < 1:
            raise ValueError("Token ceiling must be positive.")
        if self.provider == "gemini" and not self.api_key:
            raise ValueError("Gemini API key is required.")
        if self.provider == "ollama" and not self.base_url:
            raise ValueError("Ollama base URL is required.")


@dataclass
class ConditionResult:
    manifest: ConditionManifest
    bundle: ArtifactBundle | None
    validation: ValidationReport | None
    rtm: list[RTMRow]
    metrics: RunMetrics

    def download_bundle(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.model_dump(mode="json"),
            "requirements": [
                item.model_dump(mode="json")
                for item in (self.bundle.requirements if self.bundle else [])
            ],
            "scenarios": [
                item.model_dump(mode="json")
                for item in (self.bundle.scenarios if self.bundle else [])
            ],
            "test_cases": [
                item.model_dump(mode="json")
                for item in (self.bundle.test_cases if self.bundle else [])
            ],
            "validation": (
                self.validation.model_dump(mode="json") if self.validation else None
            ),
            "rtm": [item.model_dump(mode="json") for item in self.rtm],
            "metrics": self.metrics.model_dump(mode="json"),
        }


@dataclass
class ComparisonResult:
    manifest: ComparisonManifest
    conditions: dict[Condition, ConditionResult]
    failure_category: str | None = None
    failure_message: str | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _comparison_id(document_hash: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{document_hash[:12]}"


def _make_provider(
    settings: ProviderSettings, ledger: BudgetLedger
) -> StructuredProvider:
    if settings.provider == "gemini":
        return GeminiProvider(genai.Client(api_key=settings.api_key), settings.model, ledger)
    return OllamaProvider(settings.base_url, settings.model, ledger)


def _empty_metrics(
    context: PipelineContext, latency: float, *, budget_exhausted: bool
) -> RunMetrics:
    return RunMetrics(
        completion=False,
        schema_valid=False,
        citation_coverage=0.0,
        requirement_scenario_coverage=0.0,
        requirement_test_case_coverage=0.0,
        positive_scenario_coverage=0.0,
        non_positive_scenario_coverage=0.0,
        rtm_completeness=0.0,
        orphan_rate=0.0,
        invalid_reference_rate=0.0,
        duplicate_test_case_rate=0.0,
        requirement_count=0,
        scenario_count=0,
        test_case_count=0,
        input_tokens=context.input_tokens,
        output_tokens=context.output_tokens,
        latency_seconds=latency,
        retries=context.retries,
        schema_repairs=context.schema_repairs,
        semantic_revisions=context.semantic_revisions,
        budget_exhausted=budget_exhausted,
    )


def _failure_category(error: Exception) -> FailureCategory:
    if isinstance(error, BudgetExceeded):
        return FailureCategory.BUDGET_EXHAUSTION
    if isinstance(error, StructuredOutputError):
        return FailureCategory.SCHEMA_FAILURE
    if isinstance(error, ProviderError):
        return (
            FailureCategory.TRANSPORT_EXHAUSTION
            if error.retryable
            else FailureCategory.PROVIDER_REJECTION
        )
    if isinstance(error, TimeoutError):
        return FailureCategory.TIMEOUT
    return FailureCategory.CONFIGURATION
```

- [ ] **Step 5: Implement comparison execution and persistence**

Append to `src/brd_srs_testgen/runner.py`:

```python
PIPELINES = {
    Condition.SINGLE_PROMPT: run_single_prompt,
    Condition.STAGED_SINGLE_AGENT: run_staged_single_agent,
    Condition.CENTRALIZED_MULTI_AGENT: run_centralized_multi_agent,
}


def _write_bundle(
    store: RunStore,
    comparison_id: str,
    condition: Condition,
    bundle: ArtifactBundle,
    validation: ValidationReport,
    rtm: list[RTMRow],
    metrics: RunMetrics,
) -> None:
    values = {
        "requirements.json": [
            item.model_dump(mode="json") for item in bundle.requirements
        ],
        "scenarios.json": [item.model_dump(mode="json") for item in bundle.scenarios],
        "test_cases.json": [
            item.model_dump(mode="json") for item in bundle.test_cases
        ],
        "validation.json": validation.model_dump(mode="json"),
        "rtm.json": [item.model_dump(mode="json") for item in rtm],
        "metrics.json": metrics.model_dump(mode="json"),
    }
    for filename, value in values.items():
        store.write_artifact(comparison_id, condition, filename, value)


def run_comparison(
    pdf_bytes: bytes,
    settings: ProviderSettings,
    *,
    store: RunStore | None = None,
    provider_factory: ProviderFactory | None = None,
    progress: Progress | None = None,
) -> ComparisonResult:
    settings.validate()
    store = store or RunStore()
    provider_factory = provider_factory or (
        lambda _condition, ledger: _make_provider(settings, ledger)
    )
    progress = progress or (lambda _condition, _message: None)
    document_hash = hashlib.sha256(pdf_bytes).hexdigest()
    comparison_id = _comparison_id(document_hash)
    manifest = ComparisonManifest(
        comparison_id=comparison_id,
        document_hash=document_hash,
        provider=settings.provider,
        model=settings.model,
        temperature=0.0,
        token_ceiling=settings.token_ceiling,
        condition_order=list(Condition),
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        started_at=_now(),
    )

    progress(None, "Parsing PDF")
    try:
        chunks = parse_pdf(pdf_bytes)
    except DocumentError as error:
        manifest = manifest.model_copy(update={"completed_at": _now()})
        store.create_comparison(manifest, [])
        store.write_comparison_artifact(
            comparison_id,
            "failure.json",
            {
                "category": FailureCategory.PARSING.value,
                "message": str(error),
            },
        )
        return ComparisonResult(
            manifest=manifest,
            conditions={},
            failure_category=FailureCategory.PARSING.value,
            failure_message=str(error),
        )

    store.create_comparison(manifest, chunks)
    condition_results: dict[Condition, ConditionResult] = {}
    for condition in manifest.condition_order:
        progress(condition, "Starting")
        condition_manifest = ConditionManifest(
            condition=condition,
            status=RunStatus.RUNNING,
            provider=settings.provider,
            model=settings.model,
            temperature=0.0,
            token_ceiling=settings.token_ceiling,
            started_at=_now(),
        )
        store.start_condition(comparison_id, condition_manifest)
        store.append_event(
            comparison_id,
            condition,
            {"timestamp": _now(), "stage": "started"},
        )
        ledger = BudgetLedger(settings.token_ceiling)
        context = PipelineContext(provider=provider_factory(condition, ledger))
        started = time.perf_counter()
        try:
            bundle = PIPELINES[condition](context, chunks)
            validation = validate_bundle(bundle, chunks)
            rtm = build_rtm(bundle)
            latency = time.perf_counter() - started
            metrics = compute_metrics(
                bundle,
                validation,
                input_tokens=context.input_tokens,
                output_tokens=context.output_tokens,
                latency_seconds=latency,
                retries=context.retries,
                schema_repairs=context.schema_repairs,
                semantic_revisions=context.semantic_revisions,
                budget_exhausted=False,
            )
            failed = not validation.valid
            condition_manifest = condition_manifest.model_copy(
                update={
                    "status": RunStatus.FAILED if failed else RunStatus.COMPLETED,
                    "completed_at": _now(),
                    "failure_category": (
                        FailureCategory.SEMANTIC_VALIDATION if failed else None
                    ),
                    "failure_message": (
                        "Deterministic validation failed." if failed else None
                    ),
                }
            )
            _write_bundle(
                store,
                comparison_id,
                condition,
                bundle,
                validation,
                rtm,
                metrics,
            )
            result = ConditionResult(
                manifest=condition_manifest,
                bundle=bundle,
                validation=validation,
                rtm=rtm,
                metrics=metrics,
            )
        except Exception as error:
            category = _failure_category(error)
            metrics = _empty_metrics(
                context,
                time.perf_counter() - started,
                budget_exhausted=category == FailureCategory.BUDGET_EXHAUSTION,
            )
            condition_manifest = condition_manifest.model_copy(
                update={
                    "status": RunStatus.FAILED,
                    "completed_at": _now(),
                    "failure_category": category,
                    "failure_message": str(error),
                }
            )
            store.write_artifact(
                comparison_id,
                condition,
                "metrics.json",
                metrics.model_dump(mode="json"),
            )
            result = ConditionResult(
                manifest=condition_manifest,
                bundle=None,
                validation=None,
                rtm=[],
                metrics=metrics,
            )

        store.update_condition(comparison_id, condition_manifest)
        store.append_event(
            comparison_id,
            condition,
            {
                "timestamp": _now(),
                "stage": "finished",
                "status": condition_manifest.status.value,
                "input_tokens": result.metrics.input_tokens,
                "output_tokens": result.metrics.output_tokens,
                "retries": result.metrics.retries,
                "schema_repairs": result.metrics.schema_repairs,
                "semantic_revisions": result.metrics.semantic_revisions,
            },
        )
        condition_results[condition] = result
        progress(condition, condition_manifest.status.value)

    manifest = manifest.model_copy(update={"completed_at": _now()})
    store.update_comparison(manifest)
    progress(None, "Complete")
    return ComparisonResult(manifest=manifest, conditions=condition_results)
```

- [ ] **Step 6: Run storage and runner tests**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_storage.py tests/test_runner.py -q
```

Expected: `6 passed`.

- [ ] **Step 7: Commit the comparison runner**

```bash
rtk git add src/brd_srs_testgen/storage.py src/brd_srs_testgen/runner.py tests/test_runner.py
rtk git commit -m "feat: run three-condition comparisons"
```

### Task 9: Build the thin English Streamlit interface

**Files:**
- Create: `app.py`
- Create: `tests/test_app.py`

- [ ] **Step 1: Write a failing Streamlit smoke test**

Create `tests/test_app.py`:

```python
from datetime import UTC, datetime

from streamlit.testing.v1 import AppTest

from brd_srs_testgen.models import (
    ComparisonManifest,
    Condition,
    ConditionManifest,
    FailureCategory,
    RunMetrics,
    RunStatus,
)
from brd_srs_testgen.runner import ComparisonResult, ConditionResult
from brd_srs_testgen.validation import build_rtm, compute_metrics, validate_bundle
from tests.factories import bundle, chunk


def successful(condition: Condition) -> ConditionResult:
    artifacts = bundle()
    validation = validate_bundle(artifacts, [chunk()])
    metrics = compute_metrics(
        artifacts,
        validation,
        input_tokens=10,
        output_tokens=5,
        latency_seconds=0.1,
        retries=0,
        schema_repairs=0,
        semantic_revisions=0,
        budget_exhausted=False,
    )
    return ConditionResult(
        manifest=ConditionManifest(
            condition=condition,
            status=RunStatus.COMPLETED,
            provider="ollama",
            model="gemma4",
            temperature=0.0,
            token_ceiling=100_000,
            started_at=datetime.now(UTC).isoformat(),
            completed_at=datetime.now(UTC).isoformat(),
        ),
        bundle=artifacts,
        validation=validation,
        rtm=build_rtm(artifacts),
        metrics=metrics,
    )


def failed(condition: Condition) -> ConditionResult:
    metrics = RunMetrics(
        completion=False,
        schema_valid=False,
        citation_coverage=0.0,
        requirement_scenario_coverage=0.0,
        requirement_test_case_coverage=0.0,
        positive_scenario_coverage=0.0,
        non_positive_scenario_coverage=0.0,
        rtm_completeness=0.0,
        orphan_rate=0.0,
        invalid_reference_rate=0.0,
        duplicate_test_case_rate=0.0,
        requirement_count=0,
        scenario_count=0,
        test_case_count=0,
        input_tokens=0,
        output_tokens=0,
        latency_seconds=0.1,
        retries=0,
        schema_repairs=0,
        semantic_revisions=0,
        budget_exhausted=False,
    )
    return ConditionResult(
        manifest=ConditionManifest(
            condition=condition,
            status=RunStatus.FAILED,
            provider="ollama",
            model="gemma4",
            temperature=0.0,
            token_ceiling=100_000,
            started_at=datetime.now(UTC).isoformat(),
            completed_at=datetime.now(UTC).isoformat(),
            failure_category=FailureCategory.PROVIDER_REJECTION,
            failure_message="Provider rejected the request.",
        ),
        bundle=None,
        validation=None,
        rtm=[],
        metrics=metrics,
    )


def fake_runner(_pdf_bytes, settings, *, progress):
    progress(None, "Parsing PDF")
    progress(Condition.SINGLE_PROMPT, "completed")
    now = datetime.now(UTC).isoformat()
    return ComparisonResult(
        manifest=ComparisonManifest(
            comparison_id="test-comparison",
            document_hash="a" * 64,
            provider=settings.provider,
            model=settings.model,
            temperature=0.0,
            token_ceiling=settings.token_ceiling,
            condition_order=list(Condition),
            prompt_version="research-core-v1",
            schema_version="research-core-v1",
            started_at=now,
            completed_at=now,
        ),
        conditions={
            Condition.SINGLE_PROMPT: successful(Condition.SINGLE_PROMPT),
            Condition.STAGED_SINGLE_AGENT: failed(Condition.STAGED_SINGLE_AGENT),
            Condition.CENTRALIZED_MULTI_AGENT: successful(
                Condition.CENTRALIZED_MULTI_AGENT
            ),
        },
    )


def test_app_uploads_runs_displays_failure_and_downloads() -> None:
    app = AppTest.from_file("app.py", default_timeout=10)
    app.session_state["_runner"] = fake_runner
    app.run()
    app.selectbox(key="provider").set_value("ollama").run()
    app.text_input(key="model").set_value("gemma4").run()
    app.file_uploader(key="pdf").upload("sample.pdf", b"%PDF", "application/pdf").run()
    app.button(key="run").click().run()

    assert not app.exception
    assert any("Provider rejected" in item.value for item in app.error)
    assert len(app.download_button) >= 2
```

- [ ] **Step 2: Run the app test to verify it fails**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_app.py -q
```

Expected: failure because `app.py` does not exist.

- [ ] **Step 3: Implement result rendering and JSON downloads**

Create `app.py`:

```python
from __future__ import annotations

import json

import streamlit as st

from brd_srs_testgen.models import Condition, RunStatus
from brd_srs_testgen.runner import (
    ComparisonResult,
    ConditionResult,
    ProviderSettings,
    run_comparison,
)


st.set_page_config(page_title="BRD/SRS Test-Case Research Core", layout="wide")


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _download(
    label: str, value, filename: str, *, key: str
) -> None:
    st.download_button(
        label,
        data=_json(value),
        file_name=filename,
        mime="application/json",
        key=key,
        on_click="ignore",
    )


def _render_condition(condition: Condition, result: ConditionResult) -> None:
    st.subheader(condition.value.replace("_", " ").title())
    if result.manifest.status == RunStatus.COMPLETED:
        st.success("Completed and validated")
    else:
        message = result.manifest.failure_message or "Condition failed."
        category = result.manifest.failure_category
        prefix = f"{category.value}: " if category else ""
        st.error(f"{prefix}{message}")

    st.metric("Requirements", result.metrics.requirement_count)
    st.metric("Test cases", result.metrics.test_case_count)
    st.metric("RTM completeness", f"{result.metrics.rtm_completeness:.0%}")
    st.metric(
        "Tokens",
        result.metrics.input_tokens + result.metrics.output_tokens,
    )
    st.caption(f"Latency: {result.metrics.latency_seconds:.2f}s")

    prefix = result.manifest.condition.value
    if result.bundle:
        _download(
            "Requirements JSON",
            [item.model_dump(mode="json") for item in result.bundle.requirements],
            f"{prefix}-requirements.json",
            key=f"{prefix}-requirements",
        )
        _download(
            "Scenarios JSON",
            [item.model_dump(mode="json") for item in result.bundle.scenarios],
            f"{prefix}-scenarios.json",
            key=f"{prefix}-scenarios",
        )
        _download(
            "Test cases JSON",
            [item.model_dump(mode="json") for item in result.bundle.test_cases],
            f"{prefix}-test-cases.json",
            key=f"{prefix}-test-cases",
        )
    _download(
        "RTM JSON",
        [item.model_dump(mode="json") for item in result.rtm],
        f"{prefix}-rtm.json",
        key=f"{prefix}-rtm",
    )
    _download(
        "Complete condition bundle",
        result.download_bundle(),
        f"{prefix}-complete.json",
        key=f"{prefix}-complete",
    )


def _render_result(result: ComparisonResult) -> None:
    st.header("Comparison results")
    st.caption(f"Comparison ID: {result.manifest.comparison_id}")
    if result.failure_category:
        st.error(f"{result.failure_category}: {result.failure_message}")
        return
    columns = st.columns(3)
    for column, condition in zip(columns, result.manifest.condition_order):
        with column:
            _render_condition(condition, result.conditions[condition])
```

- [ ] **Step 4: Add provider configuration, upload, and execution UI**

Append to `app.py`:

```python
def main() -> None:
    st.title("BRD/SRS Test-Case Research Core")
    st.write(
        "Compare single-prompt, staged single-agent, and centralized multi-agent "
        "generation on one text-extractable PDF."
    )

    with st.sidebar:
        st.header("Provider")
        provider = st.selectbox(
            "Provider",
            options=["gemini", "ollama"],
            format_func=str.title,
            key="provider",
        )
        default_model = "gemini-3.5-flash" if provider == "gemini" else "gemma4"
        model = st.text_input("Exact model identifier", value=default_model, key="model")
        api_key = (
            st.text_input("Gemini API key", type="password", key="api_key")
            if provider == "gemini"
            else ""
        )
        base_url = (
            st.text_input(
                "Ollama base URL",
                value="http://localhost:11434",
                key="base_url",
            )
            if provider == "ollama"
            else ""
        )
        token_ceiling = int(
            st.number_input(
                "Token ceiling per condition",
                min_value=1_000,
                value=100_000,
                step=1_000,
                key="token_ceiling",
            )
        )
        st.caption("Temperature is fixed at 0.0; centralized workers are fixed at 3.")

    uploaded = st.file_uploader("Upload one BRD/SRS PDF", type=["pdf"], key="pdf")
    if st.button("Run all three conditions", type="primary", key="run"):
        if uploaded is None:
            st.error("Upload a PDF before starting.")
        else:
            settings = ProviderSettings(
                provider=provider,
                model=model,
                token_ceiling=token_ceiling,
                api_key=api_key,
                base_url=base_url,
            )
            try:
                settings.validate()
                with st.status("Running comparison", expanded=True) as status:
                    def progress(condition, message):
                        label = condition.value if condition else "comparison"
                        status.write(f"{label}: {message}")

                    runner = st.session_state.get("_runner", run_comparison)
                    result = runner(uploaded.getvalue(), settings, progress=progress)
                    status.update(label="Comparison finished", state="complete")
                st.session_state["comparison_result"] = result
            except Exception as error:
                st.error(str(error))

    result = st.session_state.get("comparison_result")
    if isinstance(result, ComparisonResult):
        _render_result(result)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the Streamlit smoke test**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_app.py -q
```

Expected: `1 passed`.

- [ ] **Step 6: Commit the Streamlit UI**

```bash
rtk git add app.py tests/test_app.py
rtk git commit -m "feat: add research comparison UI"
```

### Task 10: Document operation and run the full verification gate

**Files:**
- Create: `docs/research-core-operations.md`

- [ ] **Step 1: Write the complete operations guide**

Create `docs/research-core-operations.md`:

````markdown
# Research Core Operations

## Setup

```bash
rtk uv --cache-dir /tmp/citd-final-uv-cache venv --python 3.11 .venv
rtk .venv/bin/python -m pip install -r requirements.txt
```

## Launch

```bash
PYTHONPATH=src rtk .venv/bin/python -m streamlit run app.py
```

Open the local URL printed by Streamlit. The application accepts one
text-extractable PDF and runs the three conditions in this fixed order:

1. single-prompt reference;
2. staged single-agent; and
3. centralized multi-agent.

## Gemini smoke run

1. Select **Gemini**.
2. Enter an API key and an exact model identifier supported by the account.
3. Keep the same model and token ceiling for all three conditions.
4. Upload a small text-extractable PDF and run the comparison.
5. Confirm three condition summaries appear and successful conditions provide
   requirements, scenarios, test cases, RTM, metrics, and complete JSON downloads.

The API key stays in Streamlit session state and is not stored under `runs/`.

## Ollama smoke run

```bash
rtk ollama serve
rtk ollama pull gemma4
```

1. Select **Ollama**.
2. Keep `http://localhost:11434` or enter the local Ollama URL.
3. Enter `gemma4` or another locally installed structured-output-capable model.
4. Upload the same small PDF and run the comparison.
5. Confirm failures, token counts, and downloads are shown without exposing model
   thinking output.

## Persisted files

```text
runs/<comparison_id>/
  manifest.json
  chunks.json
  conditions/<condition>/
    manifest.json
    requirements.json
    scenarios.json
    test_cases.json
    validation.json
    rtm.json
    metrics.json
    events.jsonl
```

A parsing failure writes `failure.json` at the comparison root. Re-running creates
a new comparison ID. Completed or failed semantic runs are never overwritten.

## Automated verification

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest -q
rtk .venv/bin/python -m compileall -q app.py src tests
rtk git diff --check
```

Automated tests use fake providers and do not require network access, credentials,
or a running Ollama server.

## First-slice limits

- No OCR or scanned-PDF recovery.
- No Excel export.
- No 54-run scheduler or resume workflow.
- No provider-comparison statistics.
- No blinded human evaluation.
````

- [ ] **Step 2: Run every automated test from a clean process**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest -q
```

Expected: `27 passed` and no network calls.

- [ ] **Step 3: Verify imports and syntax without starting a server**

Run:

```bash
rtk .venv/bin/python -m compileall -q app.py src tests
PYTHONPATH=src rtk .venv/bin/python -c "from brd_srs_testgen.runner import run_comparison; print('imports ok')"
```

Expected: compileall exits `0` without output; the import command prints `imports ok`.

- [ ] **Step 4: Verify the final diff is scoped and whitespace-clean**

Run:

```bash
rtk git status --short
rtk git diff --check
rtk git diff --stat
```

Expected: only the files listed in this plan plus pre-existing user changes appear; `git diff --check` exits `0` without output. Do not stage or modify the pre-existing prototype changes.

- [ ] **Step 5: Optionally run one live provider smoke check**

Use the operations guide with a small non-sensitive PDF. Record the exact provider/model and outcome in local notes, not in the automated test suite. A live smoke run is optional because credentials, cost, local hardware, and provider availability are external.

- [ ] **Step 6: Commit operations documentation**

```bash
rtk git add docs/research-core-operations.md
rtk git commit -m "docs: add research core operations"
```

## Requirements coverage

| Approved design requirement | Implemented by |
|---|---|
| One English Streamlit comparison page | Task 9 |
| One text-extractable PDF; no silent truncation | Task 2 |
| Gemini and Ollama operational alternatives | Task 3 |
| Fixed provider/model and equal condition budgets | Tasks 3 and 8 |
| Single-prompt condition | Task 6 |
| Staged single-agent condition | Task 6 |
| Three isolated centralized workers | Task 7 |
| One schema repair; two transport retries; bounded semantic revision | Tasks 6 and 7 |
| Deterministic citations, relationships, RTM, coverage, duplicates, usage | Task 4 |
| Failure isolation and categorized failures | Task 8 |
| Atomic immutable JSON runs and downloads | Tasks 5, 8, and 9 |
| Mocked automated checks and optional live smoke checks | Tasks 3, 6-10 |

## Primary API references

- Gemini structured outputs: `https://ai.google.dev/gemini-api/docs/structured-output`
- Gemini token counting and usage: `https://ai.google.dev/gemini-api/docs/tokens`
- Ollama chat endpoint: `https://docs.ollama.com/api/chat`
- Ollama structured outputs: `https://docs.ollama.com/capabilities/structured-outputs`
- Streamlit AppTest: `https://docs.streamlit.io/develop/api-reference/app-testing`
