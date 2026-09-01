from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from typing import Literal, TypeVar

from pydantic import BaseModel, Field, create_model

from .documents import canonicalize_source_references
from .models import (
    AgentSetup,
    ActivityEvent,
    ArtifactBundle,
    DocumentChunk,
    GeneratedCases,
    Requirement,
    RequirementBatch,
    ReviewResult,
    Scenario,
    ScenarioBatch,
    TestCase,
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
    requirements_prompt,
    review_prompt,
    revision_prompt,
    scenarios_prompt,
    single_prompt,
    test_cases_prompt,
    worker_cases_prompt,
    worker_requirements_prompt,
)


T = TypeVar("T", bound=BaseModel)
I = TypeVar("I")
R = TypeVar("R")
Messages = list[dict[str, str]]
PROMPT_VERSION = "research-core-v4"
MIN_OUTPUT_TOKENS = 1_024
LOCAL_EVIDENCE_CHARS_PER_TASK = 6_000
LOCAL_REQUIREMENTS_PER_TASK = 3


class PipelineOutputError(ValueError):
    pass


class BoundedRequirementBatch(RequirementBatch):
    requirements: list[Requirement] = Field(max_length=20)


class BoundedGeneratedCases(GeneratedCases):
    scenarios: list[Scenario] = Field(max_length=8)
    test_cases: list[TestCase] = Field(max_length=8)


def _scoped_worker_cases_schema(
    requirement_ids: Iterable[str],
) -> type[BoundedGeneratedCases]:
    allowed_ids = tuple(sorted(set(requirement_ids)))
    if not allowed_ids:
        return BoundedGeneratedCases
    requirement_id = Literal.__getitem__(allowed_ids)
    scoped_scenario = create_model(
        "ScopedWorkerScenario",
        __base__=Scenario,
        requirement_ids=(list[requirement_id], Field(min_length=1)),
    )
    scoped_test_case = create_model(
        "ScopedWorkerTestCase",
        __base__=TestCase,
        requirement_ids=(list[requirement_id], Field(min_length=1)),
    )
    return create_model(
        "ScopedWorkerCases",
        __base__=BoundedGeneratedCases,
        scenarios=(list[scoped_scenario], Field(max_length=8)),
        test_cases=(list[scoped_test_case], Field(max_length=8)),
    )


@dataclass
class PipelineContext:
    provider: StructuredProvider
    providers: dict[str, StructuredProvider] = field(default_factory=dict)
    agent_setups: dict[str, AgentSetup] = field(default_factory=default_agent_setups)
    agent_prompts: dict[str, str] = field(default_factory=dict)
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
                current_messages, schema, max_output_tokens
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
                if not allow_schema_repair or schema_repair_count == 2:
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
            max_output_tokens=4_000,
            agent="requirements",
        ),
        chunks,
    )
    prompt = _user(scenarios_prompt(requirements, chunks))
    scenarios = canonicalize_source_references(
        context.generate(
            [prompt],
            ScenarioBatch,
            max_output_tokens=4_000,
            agent="scenarios",
        ),
        chunks,
    )
    prompt = _user(test_cases_prompt(requirements, scenarios, chunks))
    test_cases = canonicalize_source_references(
        context.generate(
            [prompt],
            TestCaseBatch,
            max_output_tokens=8_000,
            agent="test_cases",
        ),
        chunks,
    )
    return ArtifactBundle(
        requirements=requirements.requirements,
        scenarios=scenarios.scenarios,
        test_cases=test_cases.test_cases,
    )


def _balance(
    items: list[I], weight: Callable[[I], int], group_count: int = WORKER_COUNT
) -> list[list[I]]:
    if group_count < 1:
        raise ValueError("group_count must be positive")
    groups: list[list[I]] = [[] for _ in range(group_count)]
    totals = [0] * group_count
    for item in sorted(items, key=weight, reverse=True):
        worker_index = min(range(group_count), key=totals.__getitem__)
        groups[worker_index].append(item)
        totals[worker_index] += weight(item)
    return groups


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


def _validate_worker_requirements(
    worker_index: int, batch: RequirementBatch
) -> None:
    _validate_worker_ids(
        worker_index,
        "requirement",
        "REQ",
        (requirement.requirement_id for requirement in batch.requirements),
    )


def _validate_worker_cases(
    worker_index: int,
    batch: GeneratedCases,
    canonical_requirement_ids: Iterable[str],
) -> None:
    _validate_worker_ids(
        worker_index,
        "scenario",
        "SCN",
        (scenario.scenario_id for scenario in batch.scenarios),
    )
    _validate_worker_ids(
        worker_index,
        "test case",
        "TC",
        (test_case.test_case_id for test_case in batch.test_cases),
    )
    scenario_ids = {scenario.scenario_id for scenario in batch.scenarios}
    for test_case in batch.test_cases:
        if test_case.scenario_id not in scenario_ids:
            raise PipelineOutputError(
                f"Test case {test_case.test_case_id} references unknown worker "
                f"scenario {test_case.scenario_id}."
            )
    allowed_ids = set(canonical_requirement_ids)
    artifacts = [
        ("Scenario", scenario.scenario_id, scenario.requirement_ids)
        for scenario in batch.scenarios
    ] + [
        ("Test case", test_case.test_case_id, test_case.requirement_ids)
        for test_case in batch.test_cases
    ]
    for label, artifact_id, requirement_ids in artifacts:
        unknown_ids = set(requirement_ids) - allowed_ids
        if unknown_ids:
            raise PipelineOutputError(
                f"{label} {artifact_id} references requirement IDs "
                f"{sorted(unknown_ids)} outside the canonical requirement catalog."
            )


def _namespace_worker_case_ids(
    worker_index: int, batch: GeneratedCases
) -> GeneratedCases:
    """Turn a worker's local SCN/TC numbering into its reserved ID range."""
    offset = worker_index * 1000

    def namespaced(prefix: str, item_id: str) -> str:
        number = int(item_id.removeprefix(f"{prefix}-"))
        if 1 <= number <= 1000:
            return f"{prefix}-{number + offset:03d}"
        return item_id

    scenario_ids = {
        scenario.scenario_id: namespaced("SCN", scenario.scenario_id)
        for scenario in batch.scenarios
    }
    return GeneratedCases(
        scenarios=[
            scenario.model_copy(
                update={"scenario_id": scenario_ids[scenario.scenario_id]}
            )
            for scenario in batch.scenarios
        ],
        test_cases=[
            test_case.model_copy(
                update={
                    "test_case_id": namespaced("TC", test_case.test_case_id),
                    "scenario_id": scenario_ids.get(
                        test_case.scenario_id,
                        namespaced("SCN", test_case.scenario_id),
                    ),
                }
            )
            for test_case in batch.test_cases
        ],
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
    requirements: list[Requirement], chunks: list[DocumentChunk]
) -> list[DocumentChunk]:
    chunk_ids = {
        reference.chunk_id
        for requirement in requirements
        for reference in requirement.source_references
    }
    return [chunk for chunk in chunks if chunk.chunk_id in chunk_ids]


def _merge_worker_requirements(
    batches: Iterable[RequirementBatch],
) -> RequirementBatch:
    grouped: dict[tuple[str, str], list[Requirement]] = {}
    for requirement in (
        requirement for batch in batches for requirement in batch.requirements
    ):
        key = (
            requirement.title.strip().casefold(),
            requirement.description.strip().casefold(),
        )
        grouped.setdefault(key, []).append(requirement)

    selected = list(grouped.values())[:20]
    id_map = {
        requirement.requirement_id: f"REQ-{index:03d}"
        for index, duplicates in enumerate(selected, 1)
        for requirement in duplicates
    }
    requirements = []
    for index, duplicates in enumerate(selected, 1):
        primary = duplicates[0]
        references = []
        ambiguities = []
        dependencies = []
        seen_references = set()
        for requirement in duplicates:
            for reference in requirement.source_references:
                key = (
                    reference.chunk_id,
                    reference.page_number,
                    reference.section,
                    reference.excerpt,
                )
                if key not in seen_references:
                    seen_references.add(key)
                    references.append(reference)
            for ambiguity in requirement.ambiguities:
                if ambiguity not in ambiguities:
                    ambiguities.append(ambiguity)
            for dependency_id in requirement.dependency_ids:
                mapped = id_map.get(dependency_id)
                if mapped and mapped != f"REQ-{index:03d}" and mapped not in dependencies:
                    dependencies.append(mapped)
        requirements.append(
            primary.model_copy(
                update={
                    "requirement_id": f"REQ-{index:03d}",
                    "source_references": references,
                    "ambiguities": ambiguities,
                    "dependency_ids": dependencies,
                }
            )
        )
    return RequirementBatch(requirements=requirements)


def _normalize_worker_bundle(bundle: ArtifactBundle) -> ArtifactBundle:
    """Make derived trace links and citations consistent without a model call."""
    requirements_by_id = {
        item.requirement_id: item for item in bundle.requirements
    }
    requirement_ids = set(requirements_by_id)

    def sources(linked_ids: Iterable[str]):
        references = []
        seen = set()
        for requirement_id in linked_ids:
            requirement = requirements_by_id.get(requirement_id)
            if requirement is None:
                continue
            for reference in requirement.source_references:
                key = (
                    reference.chunk_id,
                    reference.page_number,
                    reference.section,
                    reference.excerpt,
                )
                if key not in seen:
                    seen.add(key)
                    references.append(reference)
        return references

    cases_by_scenario: dict[str, list[TestCase]] = {}
    for test_case in bundle.test_cases:
        cases_by_scenario.setdefault(test_case.scenario_id, []).append(test_case)

    scenarios = []
    for scenario in bundle.scenarios:
        test_cases = cases_by_scenario.get(scenario.scenario_id, [])
        if not test_cases:
            continue
        linked_ids = list(scenario.requirement_ids)
        for test_case in test_cases:
            for requirement_id in test_case.requirement_ids:
                if requirement_id in requirement_ids and requirement_id not in linked_ids:
                    linked_ids.append(requirement_id)
        scenarios.append(
            scenario.model_copy(
                update={
                    "requirement_ids": linked_ids,
                    "source_references": sources(linked_ids),
                }
            )
        )
    scenario_ids = {scenario.scenario_id for scenario in scenarios}
    return bundle.model_copy(
        update={
            "scenarios": scenarios,
            "test_cases": [
                test_case.model_copy(
                    update={
                        "source_references": sources(test_case.requirement_ids)
                    }
                )
                for test_case in bundle.test_cases
                if test_case.scenario_id in scenario_ids
            ],
        }
    )


def run_centralized_multi_agent(
    context: PipelineContext, chunks: Iterable[DocumentChunk]
) -> ArtifactBundle:
    chunks = list(chunks)
    if context.bounded_tasks:
        chunk_groups = _bounded_groups(
            chunks, lambda chunk: len(chunk.text), LOCAL_EVIDENCE_CHARS_PER_TASK
        )
    else:
        chunk_groups = _balance(chunks, lambda chunk: len(chunk.text))
    context.notify(
        f"Orchestrator: queued {len(chunk_groups)} requirement extraction tasks.",
        agent="Orchestrator",
        role="Policy coordinator",
        state="working",
    )

    def extract_requirements(
        worker_index: int,
        group: list[DocumentChunk],
        cancellation_event: threading.Event,
    ) -> RequirementBatch:
        batch = canonicalize_source_references(context.generate(
            [
                _user(
                    worker_requirements_prompt(
                        worker_index,
                        group,
                        setup=context.agent_setup("analyst"),
                        worker_count=len(chunk_groups),
                    )
                )
            ],
            BoundedRequirementBatch,
            8_000,
            cancellation_event=cancellation_event,
            agent="analyst",
        ), chunks)
        _validate_worker_requirements(worker_index, batch)
        return batch

    worker_requirements = _run_parallel_workers(
        chunk_groups,
        extract_requirements,
        max_workers=context.worker_limit,
        on_started=lambda index: context.notify(
            f"Analyzer {index + 1}: working — extracting requirements.",
            agent=f"Analyzer {index + 1}",
            role=context.agent_setup("analyst").role,
            model=context.model_for("analyst"),
            state="working",
            task=(
                "Extract testable business rules, validations, and exceptions "
                "with source references."
            ),
            scope=_chunk_scope(chunk_groups[index]),
            deliverable="Candidate requirements for reviewer reconciliation.",
        ),
        on_completed=lambda index, batch: context.notify(
            f"Analyzer {index + 1}: done — handed requirements to the orchestrator.",
            agent=f"Analyzer {index + 1}",
            role=context.agent_setup("analyst").role,
            model=context.model_for("analyst"),
            state="complete",
            artifact=batch,
            artifact_label="Candidate requirements",
        ),
    )

    requirements = _merge_worker_requirements(worker_requirements)
    context.notify(
        f"Orchestrator: reconciled {len(requirements.requirements)} canonical requirements.",
        agent="Orchestrator",
        role="Policy coordinator",
        state="complete",
        artifact=requirements,
        artifact_label="Canonical requirements",
    )

    if context.bounded_tasks:
        chunk_sizes = {chunk.chunk_id: len(chunk.text) for chunk in chunks}

        def requirement_weight(requirement: Requirement) -> int:
            return len(requirement.description) + sum(
                chunk_sizes.get(chunk_id, 0)
                for chunk_id in {
                    reference.chunk_id
                    for reference in requirement.source_references
                }
            )

        requirement_groups = _bounded_groups(
            requirements.requirements,
            requirement_weight,
            LOCAL_EVIDENCE_CHARS_PER_TASK,
            max_items=LOCAL_REQUIREMENTS_PER_TASK,
        )
    else:
        requirement_groups = _balance(
            requirements.requirements,
            lambda requirement: len(requirement.description)
            + 100 * len(requirement.source_references),
        )
    context.notify(
        f"Orchestrator: queued {len(requirement_groups)} test generation tasks.",
        agent="Orchestrator",
        role="Policy coordinator",
        state="working",
    )

    def generate_cases(
        worker_index: int,
        group: list[Requirement],
        cancellation_event: threading.Event,
    ) -> GeneratedCases:
        if not group:
            return GeneratedCases(scenarios=[], test_cases=[])
        dependencies = _dependency_context(group, requirements.requirements)
        case_schema = _scoped_worker_cases_schema(
            requirement.requirement_id for requirement in requirements.requirements
        )
        batch = canonicalize_source_references(context.generate(
            [
                _user(
                    worker_cases_prompt(
                        worker_index,
                        group,
                        _relevant_chunks([*group, *dependencies], chunks),
                        dependency_context=dependencies,
                        setup=context.agent_setup("test_generator"),
                        worker_count=len(requirement_groups),
                    )
                )
            ],
            case_schema,
            8_000,
            cancellation_event=cancellation_event,
            agent="test_generator",
        ), chunks)
        batch = _namespace_worker_case_ids(worker_index, batch)
        _validate_worker_cases(
            worker_index,
            batch,
            (requirement.requirement_id for requirement in requirements.requirements),
        )
        return batch

    worker_cases = _run_parallel_workers(
        requirement_groups,
        generate_cases,
        max_workers=context.worker_limit,
        on_started=lambda index: context.notify(
            f"Test Generator {index + 1}: working — creating scenarios and test cases.",
            agent=f"Test Generator {index + 1}",
            role=context.agent_setup("test_generator").role,
            model=context.model_for("test_generator"),
            state="working",
        ),
        on_completed=lambda index, batch: context.notify(
            f"Test Generator {index + 1}: done — handed artifacts to the orchestrator.",
            agent=f"Test Generator {index + 1}",
            role=context.agent_setup("test_generator").role,
            model=context.model_for("test_generator"),
            state="complete",
            artifact=batch,
            artifact_label="Scenarios and test cases",
        ),
    )

    bundle = _normalize_worker_bundle(
        canonicalize_source_references(
            ArtifactBundle(
            requirements=requirements.requirements,
            scenarios=[scenario for batch in worker_cases for scenario in batch.scenarios],
            test_cases=[test_case for batch in worker_cases for test_case in batch.test_cases],
            ),
            chunks,
        )
    )
    context.notify(
        "Orchestrator: merging the generated artifacts.",
        agent="Orchestrator",
        role="Policy coordinator",
        state="complete",
        artifact=bundle,
        artifact_label="Merged artifact bundle",
    )
    return bundle
