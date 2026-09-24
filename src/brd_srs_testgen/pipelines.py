from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TypeVar

from pydantic import BaseModel

from .documents import DocumentError, canonicalize_source_references
from .models import (
    AgentSetup,
    AgentStageOutput,
    ActivityEvent,
    ArtifactBundle,
    CandidateRequirement,
    CandidateRequirementBatch,
    CriticFinding,
    CriticReport,
    CriticSeverity,
    DocumentChunk,
    Requirement,
    RequirementBatch,
    RequirementSynthesis,
    ReviewResult,
    Scenario,
    ScenarioBatch,
    TestCaseBatch,
    default_agent_setups,
)
from .providers import (
    BudgetExceeded,
    GenerationResult,
    ProviderError,
    StructuredOutputError,
    StructuredProvider,
)
from .prompts import (
    RULES,
    WORKER_COUNT,
    _assistant,
    _data_block,
    _user,
    critic_prompt,
    requirements_prompt,
    review_prompt,
    repair_prompt,
    revision_prompt,
    scenario_architect_prompt,
    scenarios_prompt,
    curator_prompt,
    scout_prompt,
    single_prompt,
    test_writer_prompt,
    test_cases_prompt,
)


T = TypeVar("T", bound=BaseModel)
I = TypeVar("I")
R = TypeVar("R")
Messages = list[dict[str, str]]
PROMPT_VERSION = "research-core-v4"
MIN_OUTPUT_TOKENS = 1_024
MULTI_AGENT_BUDGET_SHARES = {
    "scout": 0.25,
    "curator": 0.15,
    "scenario_architect": 0.15,
    "test_writer": 0.30,
    "critic": 0.10,
    "repair": 0.05,
}
LOCAL_EVIDENCE_CHARS_PER_TASK = 6_000
STAGED_OUTPUT_TOKEN_DEFAULTS = {
    "requirements": 16_000,
    "scenarios": 24_000,
    "test_cases": 48_000,
}


class PipelineOutputError(ValueError):
    pass


@dataclass
class PipelineContext:
    provider: StructuredProvider
    token_ceiling: int = 100_000
    providers: dict[str, StructuredProvider] = field(default_factory=dict)
    agent_setups: dict[str, AgentSetup] = field(default_factory=default_agent_setups)
    agent_prompts: dict[str, str] = field(default_factory=dict)
    agent_max_output_tokens: dict[str, int] = field(default_factory=dict)
    sleep: Callable[[float], None] = time.sleep
    progress: Callable[[str], None] | None = None
    retries: int = 0
    schema_repairs: int = 0
    semantic_revisions: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    max_request_tokens: int | None = None
    worker_limit: int = WORKER_COUNT
    bounded_tasks: bool = False
    critic_enabled: bool = True
    stage_outputs: list[AgentStageOutput] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def charged_tokens(self) -> int:
        return self.provider.ledger.used

    def _provider_for(self, agent: str) -> StructuredProvider:
        return self.providers.get(agent, self.provider)

    def model_for(self, agent: str) -> str:
        return self._provider_for(agent).model

    def agent_setup(self, agent: str) -> AgentSetup:
        return self.agent_setups.get(agent, default_agent_setups()[agent])

    def prompt_for(self, agent: str) -> str:
        return self.agent_prompts.get(agent, "").strip()

    def _record(self, result: GenerationResult | StructuredOutputError) -> None:
        with self._lock:
            self.input_tokens += result.input_tokens
            self.output_tokens += result.output_tokens
            self.latency_seconds += result.latency_seconds

    def _record_latency(self, latency_seconds: float) -> None:
        with self._lock:
            self.latency_seconds += latency_seconds

    def record_stage(
        self,
        stage: str,
        task_index: int,
        input_ids: list[str],
        output: BaseModel,
        *,
        role: str | None = None,
    ) -> None:
        row = AgentStageOutput(
            stage=stage,
            task_index=task_index,
            role=role or stage,
            input_ids=input_ids,
            output=output.model_dump(mode="json"),
            created_at=datetime.now(UTC),
        )
        with self._lock:
            self.stage_outputs.append(row)
            self.stage_outputs.sort(key=AgentStageOutput.order_key)

    def _output_budget(self, messages: Messages, schema: type[BaseModel], requested: int) -> int:
        if self.max_request_tokens is None:
            return requested
        payload = json.dumps(
            {"messages": messages, "schema": schema.model_json_schema()},
            ensure_ascii=False,
        ).encode("utf-8")
        available = self.max_request_tokens - max(1, (len(payload) + 3) // 4)
        if available < MIN_OUTPUT_TOKENS:
            raise PipelineOutputError(
                "Prompt exceeds the local context budget before generation."
            )
        return min(requested, available)

    def notify(
        self,
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
    ) -> None:
        if self.progress is None:
            return
        try:
            self.progress(
                ActivityEvent(
                    message,
                    agent=agent,
                    role=role,
                    model=model,
                    state=state,
                    task=task,
                    scope=scope,
                    deliverable=deliverable,
                    artifact=artifact,
                    artifact_label=artifact_label,
                )
            )
        except Exception:
            pass

    def generate(
        self,
        messages: Messages,
        schema: type[T],
        max_output_tokens: int,
        allow_schema_repair: bool = True,
        cancellation_event: threading.Event | None = None,
        agent: str = "default",
    ) -> T:
        current_messages = [message.copy() for message in messages]
        if prompt := self.prompt_for(agent):
            current_messages.insert(
                0,
                _user(
                    "Trusted run-specific prompt instructions. Apply these after "
                    "the core evidence, safety, and output-schema rules:\n" + prompt
                ),
            )
        transport_retries = 0
        schema_repair_count = 0
        observed_timeout: ProviderError | None = None
        while True:
            if cancellation_event is not None and cancellation_event.is_set():
                raise CancelledError("A sibling worker failed.")
            output_budget = self._output_budget(
                current_messages,
                schema,
                # Hierarchical calls already apply configured caps to their allocation.
                max_output_tokens
                if agent in MULTI_AGENT_BUDGET_SHARES
                else self.agent_max_output_tokens.get(agent, max_output_tokens),
            )
            started = time.perf_counter()
            try:
                result = self._provider_for(agent).generate(
                    current_messages, schema, max_output_tokens=output_budget
                )
            except ProviderError as error:
                self._record_latency(time.perf_counter() - started)
                observed_timeout = error if error.timed_out else None
                if cancellation_event is not None and cancellation_event.is_set():
                    raise CancelledError("A sibling worker failed.") from error
                if not error.retryable or transport_retries == 2:
                    raise
                delay = 2**transport_retries
                with self._lock:
                    self.retries += 1
                transport_retries += 1
                if cancellation_event is None:
                    self.sleep(delay)
                elif cancellation_event.wait(delay):
                    raise CancelledError("A sibling worker failed.") from error
            except BudgetExceeded as error:
                self._record_latency(time.perf_counter() - started)
                if observed_timeout is not None and error.reservation_blocked:
                    raise observed_timeout from error
                raise
            except StructuredOutputError as error:
                observed_timeout = None
                self._record(error)
                if cancellation_event is not None and cancellation_event.is_set():
                    raise CancelledError("A sibling worker failed.") from error
                if (
                    error.incomplete
                    or not allow_schema_repair
                    or schema_repair_count == 2
                ):
                    raise
                with self._lock:
                    self.schema_repairs += 1
                schema_repair_count += 1
                validation_error = str(error.__cause__ or error)[:500]
                repair_instruction = _user(
                    "The previous response was an invalid response. Return only valid "
                    "JSON matching this schema and all evidence/support constraints.\n"
                    f"Validation error: {validation_error}"
                )
                current_messages.append(repair_instruction)
            except Exception:
                self._record_latency(time.perf_counter() - started)
                raise
            else:
                self._record(result)
                return result.value

    def revise(
        self,
        messages: Messages,
        label: str,
        value: T,
        review: ReviewResult,
        chunks: Iterable[DocumentChunk],
        schema: type[T],
        max_output_tokens: int,
        *,
        use_history: bool = False,
    ) -> T:
        with self._lock:
            self.semantic_revisions += 1
        prompt = _user(
            revision_prompt(label, value, review, chunks, use_history=use_history)
        )
        revised = self.generate(
            [*messages, prompt],
            schema,
            max_output_tokens,
            agent="reviewer",
        )
        if use_history:
            messages.extend((prompt, _assistant(revised)))
        return revised


def stage_output_tokens(
    context: PipelineContext,
    stage: str,
    *,
    token_ceiling: int,
    task_count: int = 1,
    configured_stage: str | None = None,
) -> int:
    allocation = int(token_ceiling * MULTI_AGENT_BUDGET_SHARES[stage] / task_count)
    configured = context.agent_max_output_tokens.get(
        configured_stage or stage, allocation
    )
    return max(MIN_OUTPUT_TOKENS, min(configured, allocation))


def run_single_prompt(
    context: PipelineContext, chunks: Iterable[DocumentChunk]
) -> ArtifactBundle:
    chunks = list(chunks)
    return canonicalize_source_references(
        context.generate(
            [_user(single_prompt(chunks))],
            ArtifactBundle,
            max_output_tokens=16_000,
            agent="single",
        ),
        chunks,
    )


def run_staged_single_agent(
    context: PipelineContext, chunks: Iterable[DocumentChunk]
) -> ArtifactBundle:
    chunks = list(chunks)

    prompt = _user(requirements_prompt(chunks))
    requirements = canonicalize_source_references(
        context.generate(
            [prompt],
            RequirementBatch,
            max_output_tokens=STAGED_OUTPUT_TOKEN_DEFAULTS["requirements"],
            agent="requirements",
        ),
        chunks,
    )
    prompt = _user(scenarios_prompt(requirements, chunks))
    scenarios = canonicalize_source_references(
        context.generate(
            [prompt],
            ScenarioBatch,
            max_output_tokens=STAGED_OUTPUT_TOKEN_DEFAULTS["scenarios"],
            agent="scenarios",
        ),
        chunks,
    )
    prompt = _user(test_cases_prompt(requirements, scenarios, chunks))
    test_cases = canonicalize_source_references(
        context.generate(
            [prompt],
            TestCaseBatch,
            max_output_tokens=STAGED_OUTPUT_TOKEN_DEFAULTS["test_cases"],
            agent="test_cases",
        ),
        chunks,
    )
    return ArtifactBundle(
        requirements=requirements.requirements,
        scenarios=scenarios.scenarios,
        test_cases=test_cases.test_cases,
    )


def _bounded_groups(
    items: list[I],
    weight: Callable[[I], int],
    limit: int,
    *,
    max_items: int | None = None,
) -> list[list[I]]:
    if limit < 1 or (max_items is not None and max_items < 1):
        raise ValueError("limits must be positive")
    groups: list[list[I]] = []
    group: list[I] = []
    total = 0
    for item in items:
        item_weight = max(1, weight(item))
        if group and (
            total + item_weight > limit
            or (max_items is not None and len(group) == max_items)
        ):
            groups.append(group)
            group, total = [], 0
        group.append(item)
        total += item_weight
    if group or not groups:
        groups.append(group)
    return groups


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


def _chunk_scope(chunks: list[DocumentChunk]) -> str:
    pages = ", ".join(
        str(page) for page in sorted({chunk.page_number for chunk in chunks})
    )
    label = "chunks" if len(chunks) != 1 else "chunk"
    return f"{len(chunks)} assigned source {label} · pages {pages or 'none'}"


def _run_parallel_workers(
    groups: list[list[I]],
    worker: Callable[[int, list[I], threading.Event], R],
    *,
    max_workers: int = WORKER_COUNT,
    on_started: Callable[[int], None] | None = None,
    on_completed: Callable[[int, R], None] | None = None,
) -> list[R]:
    cancellation_event = threading.Event()
    if max_workers == 1:
        results = []
        for worker_index, group in enumerate(groups):
            if on_started is not None:
                on_started(worker_index)
            try:
                result = worker(worker_index, group, cancellation_event)
            except Exception:
                cancellation_event.set()
                raise
            results.append(result)
            if on_completed is not None:
                on_completed(worker_index, result)
        return results

    first_error: list[Exception] = []
    error_lock = threading.Lock()

    def invoke(worker_index: int, group: list[I]) -> R:
        try:
            return worker(worker_index, group, cancellation_event)
        except Exception as error:
            with error_lock:
                if not first_error:
                    first_error.append(error)
            cancellation_event.set()
            raise

    results: dict[int, R] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(groups))) as executor:
        futures = {
            executor.submit(invoke, worker_index, group): worker_index
            for worker_index, group in enumerate(groups)
        }
        for worker_index in range(len(groups)):
            if on_started is not None:
                on_started(worker_index)
        for future in as_completed(futures):
            worker_index = futures[future]
            try:
                results[worker_index] = future.result()
            except Exception:
                for pending in futures:
                    pending.cancel()
                wait(futures)
                raise first_error[0]
            if on_completed is not None:
                on_completed(worker_index, results[worker_index])
    if first_error:
        raise first_error[0]
    return [results[worker_index] for worker_index in range(len(groups))]


def _worker_bounds(worker_index: int) -> tuple[int, int]:
    return worker_index * 1000 + 1, (worker_index + 1) * 1000


def _validate_worker_ids(
    worker_index: int, label: str, prefix: str, item_ids: Iterable[str]
) -> None:
    lower, upper = _worker_bounds(worker_index)
    seen: set[str] = set()
    for item_id in item_ids:
        if not lower <= int(item_id.removeprefix(f"{prefix}-")) <= upper:
            raise PipelineOutputError(
                f"{label.title()} ID {item_id} is outside worker "
                f"{worker_index + 1} range {prefix}-{lower:03d} through "
                f"{prefix}-{upper:03d}."
            )
        if item_id in seen:
            raise PipelineOutputError(
                f"Worker {worker_index + 1} returned duplicate {label} ID {item_id}."
            )
        seen.add(item_id)


def _validate_scout_candidates(
    worker_index: int,
    batch: CandidateRequirementBatch,
    group: list[DocumentChunk],
) -> None:
    prefix = f"CAND-{worker_index + 1:03d}-"
    seen: set[str] = set()
    chunk_ids = {chunk.chunk_id for chunk in group}
    for candidate in batch.candidates:
        if (
            not candidate.candidate_id.startswith(prefix)
            or int(candidate.candidate_id.removeprefix(prefix)) < 1
        ):
            raise PipelineOutputError(
                f"Candidate ID {candidate.candidate_id} is outside Scout "
                f"{worker_index + 1} namespace {prefix}001 upward."
            )
        if candidate.candidate_id in seen:
            raise PipelineOutputError(
                f"Scout {worker_index + 1} returned duplicate candidate ID "
                f"{candidate.candidate_id}."
            )
        seen.add(candidate.candidate_id)
        for reference in candidate.source_references:
            if reference.chunk_id not in chunk_ids:
                raise PipelineOutputError(
                    f"Candidate {candidate.candidate_id} cites chunk "
                    f"{reference.chunk_id} outside its assigned evidence group."
                )


def _canonicalize_scout_candidates(
    worker_index: int,
    batch: CandidateRequirementBatch,
    group: list[DocumentChunk],
) -> CandidateRequirementBatch:
    _validate_scout_candidates(worker_index, batch, group)
    return _canonicalize_grounded(batch, group)


def _canonicalize_grounded(value: T, chunks: list[DocumentChunk]) -> T:
    try:
        return canonicalize_source_references(
            value, chunks, repair_excerpt=False, strict=True
        )
    except DocumentError as error:
        raise PipelineOutputError(str(error)) from error


def _validate_synthesis(
    candidates: Iterable[CandidateRequirement], synthesis: RequirementSynthesis
) -> None:
    candidates = list(candidates)
    candidate_ids = {item.candidate_id for item in candidates}
    decision_ids = [item.candidate_id for item in synthesis.decisions]
    if len(decision_ids) != len(set(decision_ids)) or set(decision_ids) != candidate_ids:
        raise PipelineOutputError("Curator must decide every candidate exactly once.")

    requirements_by_id = {
        requirement.requirement_id: requirement for requirement in synthesis.requirements
    }
    canonical_ids = [requirement.requirement_id for requirement in synthesis.requirements]
    if len(canonical_ids) != len(requirements_by_id):
        raise PipelineOutputError("Curator returned duplicate canonical requirement IDs.")
    expected_ids = [f"REQ-{index:03d}" for index in range(1, len(canonical_ids) + 1)]
    if canonical_ids != expected_ids:
        raise PipelineOutputError(
            "Curator canonical requirement IDs must begin REQ-001 and increase by one."
        )

    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    sources_by_requirement: dict[str, set[tuple[str, int, str, str]]] = {}
    for decision in synthesis.decisions:
        if decision.canonical_requirement_id is None:
            continue
        requirement_id = decision.canonical_requirement_id
        if requirement_id not in requirements_by_id:
            raise PipelineOutputError(
                f"Curator decision targets unknown canonical requirement {requirement_id}."
            )
        sources_by_requirement.setdefault(requirement_id, set()).update(
            (
                reference.chunk_id,
                reference.page_number,
                reference.section,
                reference.excerpt,
            )
            for reference in candidates_by_id[decision.candidate_id].source_references
        )
    if set(sources_by_requirement) != set(canonical_ids):
        raise PipelineOutputError(
            "Every canonical requirement must have a retained or merged candidate."
        )
    for requirement_id, requirement in requirements_by_id.items():
        allowed_sources = sources_by_requirement[requirement_id]
        requirement_sources = {
            (
                reference.chunk_id,
                reference.page_number,
                reference.section,
                reference.excerpt,
            )
            for reference in requirement.source_references
        }
        if requirement_sources != allowed_sources:
            raise PipelineOutputError(
                f"Canonical requirement {requirement_id} must preserve citations "
                "from retained or merged candidates exactly."
            )


def _validate_scenario_batch(
    batch: ScenarioBatch, requirements: list[Requirement]
) -> None:
    scenario_ids = [scenario.scenario_id for scenario in batch.scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise PipelineOutputError("Architect returned duplicate canonical scenario IDs.")
    expected_ids = [f"SCN-{index:03d}" for index in range(1, len(scenario_ids) + 1)]
    if scenario_ids != expected_ids:
        raise PipelineOutputError(
            "Architect scenario IDs must begin SCN-001 and increase by one."
        )
    canonical_ids = {requirement.requirement_id for requirement in requirements}
    covered_ids: set[str] = set()
    for scenario in batch.scenarios:
        unknown_ids = set(scenario.requirement_ids) - canonical_ids
        if unknown_ids:
            raise PipelineOutputError(
                f"Scenario {scenario.scenario_id} references unknown canonical "
                f"requirement IDs {sorted(unknown_ids)}."
            )
        covered_ids.update(scenario.requirement_ids)
    if covered_ids != canonical_ids:
        raise PipelineOutputError(
            "Architect must cover every canonical requirement with a scenario."
        )


def _validate_writer_cases(
    worker_index: int,
    batch: TestCaseBatch,
    assigned_scenarios: list[Scenario],
) -> None:
    _validate_worker_ids(
        worker_index,
        "test case",
        "TC",
        (test_case.test_case_id for test_case in batch.test_cases),
    )
    scenarios_by_id = {
        scenario.scenario_id: scenario for scenario in assigned_scenarios
    }
    covered_ids: set[str] = set()
    covered_requirement_ids: set[str] = set()
    for test_case in batch.test_cases:
        scenario = scenarios_by_id.get(test_case.scenario_id)
        if scenario is None:
            raise PipelineOutputError(
                f"Test case {test_case.test_case_id} references scenario "
                f"{test_case.scenario_id} outside its assigned scenarios."
            )
        if not set(test_case.requirement_ids) <= set(scenario.requirement_ids):
            raise PipelineOutputError(
                f"Test case {test_case.test_case_id} references requirement IDs "
                "outside its assigned scenario."
            )
        covered_ids.add(test_case.scenario_id)
        covered_requirement_ids.update(test_case.requirement_ids)
    if covered_ids != set(scenarios_by_id):
        raise PipelineOutputError(
            f"Test Writer {worker_index + 1} must cover every assigned scenario."
        )
    assigned_requirement_ids = {
        requirement_id
        for scenario in assigned_scenarios
        for requirement_id in scenario.requirement_ids
    }
    if covered_requirement_ids != assigned_requirement_ids:
        raise PipelineOutputError(
            f"Test Writer {worker_index + 1} must cover every assigned requirement."
        )


def _dependency_context(
    assigned: list[Requirement], requirements: list[Requirement]
) -> list[Requirement]:
    by_id = {requirement.requirement_id: requirement for requirement in requirements}
    assigned_ids = {requirement.requirement_id for requirement in assigned}
    dependency_ids: set[str] = set()
    pending = [
        dependency_id
        for requirement in assigned
        for dependency_id in requirement.dependency_ids
    ]
    while pending:
        dependency_id = pending.pop()
        if dependency_id in assigned_ids or dependency_id in dependency_ids:
            continue
        dependency = by_id.get(dependency_id)
        if dependency is None:
            continue
        dependency_ids.add(dependency_id)
        pending.extend(dependency.dependency_ids)
    return [
        requirement
        for requirement in requirements
        if requirement.requirement_id in dependency_ids
    ]


def _relevant_chunks(
    artifacts: list[Requirement | Scenario], chunks: list[DocumentChunk]
) -> list[DocumentChunk]:
    chunk_ids = {
        reference.chunk_id
        for artifact in artifacts
        for reference in artifact.source_references
    }
    return [chunk for chunk in chunks if chunk.chunk_id in chunk_ids]


def _semantic_payload(bundle: ArtifactBundle) -> dict[str, list[object]]:
    return {
        "requirements": [
            (item.title, item.description, item.ambiguities)
            for item in sorted(
                bundle.requirements, key=lambda item: item.requirement_id
            )
        ],
        "scenarios": [
            (item.title, item.objective, item.preconditions)
            for item in sorted(bundle.scenarios, key=lambda item: item.scenario_id)
        ],
        "tests": [
            (
                item.title,
                item.preconditions,
                item.test_data,
                [(step.action, step.expected_result) for step in item.steps],
            )
            for item in sorted(bundle.test_cases, key=lambda item: item.test_case_id)
        ],
    }


def _artifacts_by_id(bundle: ArtifactBundle) -> dict[str, BaseModel]:
    return {
        **{item.requirement_id: item for item in bundle.requirements},
        **{item.scenario_id: item for item in bundle.scenarios},
        **{item.test_case_id: item for item in bundle.test_cases},
    }


def _authored_semantic_payload(artifact: BaseModel) -> dict[str, object]:
    payload = artifact.model_dump(mode="json")
    for field in (
        "requirement_id",
        "scenario_id",
        "test_case_id",
        "dependency_ids",
        "requirement_ids",
        "source_references",
    ):
        payload.pop(field, None)
    return payload


def _citation_set(artifact: BaseModel) -> set[tuple[str, int, str, str]]:
    return {
        (
            reference.chunk_id,
            reference.page_number,
            reference.section,
            reference.excerpt,
        )
        for reference in artifact.source_references
    }


def _validate_critic_scope(
    report: CriticReport, bundle: ArtifactBundle
) -> set[str]:
    prefixes = {
        "curator": "REQ-",
        "scenario_architect": "SCN-",
        "test_writer": "TC-",
    }
    existing_ids = set(_artifacts_by_id(bundle))
    target_ids: set[str] = set()
    for finding in report.findings:
        prefix = prefixes[finding.responsible_role]
        for artifact_id in finding.artifact_ids:
            if re.fullmatch(r"(?:REQ|SCN|TC)-\d{3,}", artifact_id) is None:
                raise PipelineOutputError(
                    f"Finding {finding.finding_id} has malformed artifact ID "
                    f"{artifact_id}."
                )
            if not artifact_id.startswith(prefix):
                raise PipelineOutputError(
                    f"Finding {finding.finding_id} assigns {artifact_id} to the "
                    f"wrong responsible role {finding.responsible_role}."
                )
            if artifact_id not in existing_ids:
                raise PipelineOutputError(
                    f"Finding {finding.finding_id} references unknown artifact ID "
                    f"{artifact_id}."
                )
            target_ids.add(artifact_id)
    return target_ids


def _validate_repair_scope(
    original: ArtifactBundle, repaired: ArtifactBundle, target_ids: set[str]
) -> None:
    original_by_id = _artifacts_by_id(original)
    repaired_by_id = _artifacts_by_id(repaired)
    if not set(original_by_id) <= set(repaired_by_id):
        raise PipelineOutputError("Repair must preserve every original artifact ID.")
    added_ids = set(repaired_by_id) - set(original_by_id)
    if not added_ids <= target_ids:
        raise PipelineOutputError("Repair added artifacts outside the findings.")
    changed_ids = {
        artifact_id
        for artifact_id, artifact in original_by_id.items()
        if repaired_by_id[artifact_id] != artifact
    } | added_ids
    if not changed_ids <= target_ids:
        raise PipelineOutputError("Repair changed artifacts outside the findings.")
    if any(
        _authored_semantic_payload(repaired_by_id[artifact_id])
        == _authored_semantic_payload(original_by_id[artifact_id])
        for artifact_id in changed_ids - added_ids
    ):
        raise PipelineOutputError("Repair changed links only.")


def _validate_repair_citations(
    original: ArtifactBundle, repaired: ArtifactBundle
) -> None:
    repaired_requirements = {
        item.requirement_id: item for item in repaired.requirements
    }
    for requirement in original.requirements:
        repaired_requirement = repaired_requirements.get(requirement.requirement_id)
        if (
            repaired_requirement is not None
            and _citation_set(repaired_requirement) != _citation_set(requirement)
        ):
            raise PipelineOutputError(
                f"Canonical requirement {requirement.requirement_id} must preserve "
                "citations from the original accumulated set exactly."
            )


def _repair_role(findings: list[CriticFinding]) -> str:
    severity_order = {
        CriticSeverity.HIGH: 0,
        CriticSeverity.MEDIUM: 1,
        CriticSeverity.LOW: 2,
    }
    return min(
        findings, key=lambda finding: severity_order[finding.severity]
    ).responsible_role


def _critique_bundle(
    context: PipelineContext,
    bundle: ArtifactBundle,
    chunks: list[DocumentChunk],
) -> ArtifactBundle:
    if not context.critic_enabled:
        return bundle
    report = _canonicalize_grounded(
        context.generate(
            [
                _user(
                    critic_prompt(
                        bundle,
                        chunks,
                        setup=context.agent_setup("critic"),
                    )
                )
            ],
            CriticReport,
            stage_output_tokens(context, "critic", token_ceiling=context.token_ceiling),
            agent="critic",
        ),
        chunks,
    )
    target_ids = _validate_critic_scope(report, bundle)
    context.record_stage("critic", 0, list(_artifacts_by_id(bundle)), report)
    if report.accepted:
        return bundle
    role = _repair_role(report.findings)
    repaired = context.generate(
        [
            _user(
                repair_prompt(
                    bundle,
                    report.findings,
                    chunks,
                    setup=context.agent_setup(role),
                )
            )
        ],
        ArtifactBundle,
        stage_output_tokens(
            context,
            "repair",
            token_ceiling=context.token_ceiling,
            configured_stage=role,
        ),
        agent=role,
    )
    repaired = _canonicalize_grounded(repaired, chunks)
    _validate_repair_scope(bundle, repaired, target_ids)
    _validate_repair_citations(bundle, repaired)
    context.record_stage(
        "repair", 0,
        [finding.finding_id for finding in report.findings] + sorted(target_ids),
        repaired, role=role,
    )
    with context._lock:
        context.semantic_revisions += 1
    return repaired


def run_centralized_multi_agent(
    context: PipelineContext, chunks: Iterable[DocumentChunk]
) -> ArtifactBundle:
    chunks = list(chunks)
    chunk_groups = _ordered_evidence_groups(
        chunks, char_limit=LOCAL_EVIDENCE_CHARS_PER_TASK
    )
    if not context.bounded_tasks:
        chunk_groups.extend(
            [] for _ in range(max(0, WORKER_COUNT - len(chunk_groups)))
        )
    context.notify(
        f"Orchestrator: queued {len(chunk_groups)} requirement extraction tasks.",
        agent="Orchestrator",
        role="Policy coordinator",
        state="working",
    )

    def scout(
        worker_index: int,
        group: list[DocumentChunk],
        cancellation_event: threading.Event,
    ) -> CandidateRequirementBatch:
        batch = context.generate(
            [
                _user(
                    scout_prompt(
                        worker_index,
                        group,
                        setup=context.agent_setup("scout"),
                        worker_count=len(chunk_groups),
                    )
                )
            ],
            CandidateRequirementBatch,
            stage_output_tokens(
                context, "scout", token_ceiling=context.token_ceiling,
                task_count=len(chunk_groups),
            ),
            cancellation_event=cancellation_event,
            agent="scout",
        )
        batch = _canonicalize_scout_candidates(worker_index, batch, group)
        context.record_stage(
            "scout", worker_index, [item.chunk_id for item in group], batch
        )
        return batch

    worker_candidates = _run_parallel_workers(
        chunk_groups,
        scout,
        max_workers=min(WORKER_COUNT, context.worker_limit),
        on_started=lambda index: context.notify(
            f"Scout {index + 1}: working — extracting requirements.",
            agent=f"Scout {index + 1}",
            role=context.agent_setup("scout").role,
            model=context.model_for("scout"),
            state="working",
            task=(
                "Extract testable business rules, validations, and exceptions "
                "with source references."
            ),
            scope=_chunk_scope(chunk_groups[index]),
            deliverable="Candidate requirements for reviewer reconciliation.",
        ),
        on_completed=lambda index, batch: context.notify(
            f"Scout {index + 1}: done — handed requirements to the orchestrator.",
            agent=f"Scout {index + 1}",
            role=context.agent_setup("scout").role,
            model=context.model_for("scout"),
            state="complete",
            artifact=batch,
            artifact_label="Candidate requirements",
        ),
    )

    candidates = [
        candidate for batch in worker_candidates for candidate in batch.candidates
    ]
    synthesis = canonicalize_source_references(
        context.generate(
            [
                _user(
                    curator_prompt(
                        candidates,
                        chunks,
                        setup=context.agent_setup("curator"),
                    )
                )
            ],
            RequirementSynthesis,
            stage_output_tokens(context, "curator", token_ceiling=context.token_ceiling),
            agent="curator",
        ),
        chunks,
    )
    _validate_synthesis(candidates, synthesis)
    context.record_stage(
        "curator", 0, [item.candidate_id for item in candidates], synthesis
    )
    requirements = RequirementBatch(requirements=synthesis.requirements)
    context.notify(
        f"Orchestrator: reconciled {len(requirements.requirements)} canonical requirements.",
        agent="Orchestrator",
        role="Policy coordinator",
        state="complete",
        artifact=requirements,
        artifact_label="Canonical requirements",
    )

    scenario_batch = _canonicalize_grounded(
        context.generate(
            [
                _user(
                    scenario_architect_prompt(
                        requirements,
                        chunks,
                        setup=context.agent_setup("scenario_architect"),
                    )
                )
            ],
            ScenarioBatch,
            stage_output_tokens(
                context, "scenario_architect", token_ceiling=context.token_ceiling
            ),
            agent="scenario_architect",
        ),
        chunks,
    )
    _validate_scenario_batch(scenario_batch, requirements.requirements)
    context.record_stage(
        "scenario_architect", 0,
        [item.requirement_id for item in requirements.requirements], scenario_batch,
    )
    context.notify(
        f"Scenario Architect: planned {len(scenario_batch.scenarios)} canonical scenarios.",
        agent="Scenario Architect",
        role=context.agent_setup("scenario_architect").role,
        model=context.model_for("scenario_architect"),
        state="complete",
        artifact=scenario_batch,
        artifact_label="Canonical scenarios",
    )

    scenario_groups = _bounded_groups(
        scenario_batch.scenarios,
        lambda item: len(item.objective),
        LOCAL_EVIDENCE_CHARS_PER_TASK,
        max_items=3,
    )
    context.notify(
        f"Orchestrator: queued {len(scenario_groups)} test writing tasks.",
        agent="Orchestrator",
        role="Policy coordinator",
        state="working",
    )

    def write_tests(
        worker_index: int,
        group: list[Scenario],
        cancellation_event: threading.Event,
    ) -> TestCaseBatch:
        if not group:
            batch = TestCaseBatch(test_cases=[])
            context.record_stage("test_writer", worker_index, [], batch)
            return batch
        assigned_ids = {
            requirement_id
            for scenario in group
            for requirement_id in scenario.requirement_ids
        }
        assigned_requirements = [
            requirement
            for requirement in requirements.requirements
            if requirement.requirement_id in assigned_ids
        ]
        dependencies = _dependency_context(
            assigned_requirements, requirements.requirements
        )
        relevant_chunks = _relevant_chunks(
            [*group, *assigned_requirements, *dependencies], chunks
        )
        batch = _canonicalize_grounded(
            context.generate(
                [
                    _user(
                        test_writer_prompt(
                            worker_index,
                            group,
                            assigned_requirements,
                            relevant_chunks,
                            dependency_context=dependencies,
                            setup=context.agent_setup("test_writer"),
                            worker_count=len(scenario_groups),
                        )
                    )
                ],
                TestCaseBatch,
                stage_output_tokens(
                    context, "test_writer", token_ceiling=context.token_ceiling,
                    task_count=len(scenario_groups),
                ),
                cancellation_event=cancellation_event,
                agent="test_writer",
            ),
            relevant_chunks,
        )
        _validate_writer_cases(worker_index, batch, group)
        context.record_stage(
            "test_writer", worker_index,
            [item.scenario_id for item in group] + [
                item.requirement_id for item in [*assigned_requirements, *dependencies]
            ],
            batch,
        )
        return batch

    worker_cases = _run_parallel_workers(
        scenario_groups,
        write_tests,
        max_workers=min(WORKER_COUNT, context.worker_limit),
        on_started=lambda index: context.notify(
            f"Test Writer {index + 1}: working — expanding canonical scenarios.",
            agent=f"Test Writer {index + 1}",
            role=context.agent_setup("test_writer").role,
            model=context.model_for("test_writer"),
            state="working",
        ),
        on_completed=lambda index, batch: context.notify(
            f"Test Writer {index + 1}: done — handed test cases to the orchestrator.",
            agent=f"Test Writer {index + 1}",
            role=context.agent_setup("test_writer").role,
            model=context.model_for("test_writer"),
            state="complete",
            artifact=batch,
            artifact_label="Test cases",
        ),
    )

    test_cases = [
        test_case.model_copy(update={"test_case_id": f"TC-{index:03d}"})
        for index, test_case in enumerate(
            (
                test_case
                for batch in worker_cases
                for test_case in batch.test_cases
            ),
            1,
        )
    ]
    bundle = ArtifactBundle(
        requirements=synthesis.requirements,
        scenarios=scenario_batch.scenarios,
        test_cases=test_cases,
    )
    context.notify(
        "Orchestrator: merging the generated artifacts.",
        agent="Orchestrator",
        role="Policy coordinator",
        state="complete",
        artifact=bundle,
        artifact_label="Merged artifact bundle",
    )
    return _critique_bundle(context, bundle, chunks)
