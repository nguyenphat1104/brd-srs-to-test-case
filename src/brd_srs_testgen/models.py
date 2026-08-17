from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ActivityEvent(str):
    """A display-safe progress update that remains compatible with text callbacks."""

    def __new__(
        cls,
        message: str,
        *,
        agent: str = "",
        role: str = "",
        model: str = "",
        state: str = "",
        task: str = "",
        scope: str = "",
        deliverable: str = "",
        artifact: BaseModel | None = None,
        artifact_label: str = "",
    ) -> "ActivityEvent":
        event = super().__new__(cls, message)
        event.agent = agent
        event.role = role
        event.model = model
        event.state = state
        event.task = task
        event.scope = scope
        event.deliverable = deliverable
        event.artifact = artifact
        event.artifact_label = artifact_label
        return event


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


class RunType(StrEnum):
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
    test_data: dict[str, JsonValue] = Field(default_factory=dict)
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


class CoverageAssignment(StrictModel):
    requirement_id: str = Field(pattern=r"^REQ-\d{3,}$")
    scenario_id: str = Field(pattern=r"^SCN-\d{3,}$")
    test_case_id: str = Field(pattern=r"^TC-\d{3,}$")


class CoverageRepair(StrictModel):
    assignments: list[CoverageAssignment] = Field(min_length=1)


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
    citation_coverage: float = Field(ge=0, le=1)
    requirement_scenario_coverage: float = Field(ge=0, le=1)
    requirement_test_case_coverage: float = Field(ge=0, le=1)
    positive_scenario_coverage: float = Field(ge=0, le=1)
    non_positive_scenario_coverage: float = Field(ge=0, le=1)
    rtm_completeness: float = Field(ge=0, le=1)
    orphan_rate: float = Field(ge=0, le=1)
    invalid_reference_rate: float = Field(ge=0, le=1)
    duplicate_test_case_rate: float = Field(ge=0, le=1)
    requirement_count: int = Field(ge=0)
    scenario_count: int = Field(ge=0)
    test_case_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    charged_tokens: int = Field(default=0, ge=0)
    latency_seconds: float = Field(ge=0)
    retries: int = Field(ge=0)
    schema_repairs: int = Field(ge=0)
    semantic_revisions: int = Field(ge=0)
    budget_exhausted: bool


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
                if bundle
                else []
            ),
            "scenarios": (
                [item.model_dump(mode="json") for item in bundle.scenarios] if bundle else []
            ),
            "test_cases": (
                [item.model_dump(mode="json") for item in bundle.test_cases]
                if bundle
                else []
            ),
            "validation": self.validation.model_dump(mode="json") if self.validation else None,
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
