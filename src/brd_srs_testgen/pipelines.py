from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel

from .documents import render_chunks
from .models import (
    ArtifactBundle,
    DocumentChunk,
    RequirementBatch,
    ReviewResult,
    ScenarioBatch,
    TestCaseBatch,
)
from .providers import (
    GenerationResult,
    ProviderError,
    StructuredOutputError,
    StructuredProvider,
)


T = TypeVar("T", bound=BaseModel)
Messages = list[dict[str, str]]
PROMPT_VERSION = "research-core-v1"
WORKER_COUNT = 3

RULES = """Rules:
- Write in English only.
- Return only the requested schema as valid JSON.
- Use unique canonical IDs in increasing order: REQ-001, SCN-001, and TC-001.
- Every artifact must cite a real chunk ID and an exact supporting excerpt.
- Do not invent unsupported requirements, behavior, test data, or expected results."""


def _user(content: str) -> dict[str, str]:
    return {"role": "user", "content": content}


def _assistant(value: BaseModel) -> dict[str, str]:
    return {
        "role": "assistant",
        "content": json.dumps(value.model_dump(mode="json"), ensure_ascii=False),
    }


def _evidence(chunks: Iterable[DocumentChunk]) -> str:
    return f"Evidence:\n{render_chunks(chunks)}"


def single_prompt(chunks: Iterable[DocumentChunk]) -> str:
    return f"""{RULES}

From the complete evidence, exhaustively identify functional, nonfunctional, and business requirements. Create traceable positive, negative, boundary, edge, and state-transition scenarios wherever the evidence supports them. Then create executable manual test cases with ordered actions and observable expected results.
Return one ArtifactBundle containing requirements, scenarios, and test_cases.

{_evidence(chunks)}"""


def requirements_prompt(chunks: Iterable[DocumentChunk]) -> str:
    return f"""{RULES}

Extract and consolidate all supported functional, nonfunctional, and business requirements from the full evidence. Preserve ambiguities and dependencies when supported.
Return one RequirementBatch.

{_evidence(chunks)}"""


def scenarios_prompt(
    requirements: RequirementBatch, chunks: Iterable[DocumentChunk]
) -> str:
    return f"""{RULES}

Using the validated requirements JSON and full evidence, create traceable positive scenarios and every relevant negative, boundary, edge, and state-transition scenario supported by the evidence.
Return one ScenarioBatch.

Validated requirements JSON:
{requirements.model_dump_json()}

{_evidence(chunks)}"""


def test_cases_prompt(
    requirements: RequirementBatch,
    scenarios: ScenarioBatch,
    chunks: Iterable[DocumentChunk],
) -> str:
    return f"""{RULES}

Using the validated requirements and scenarios JSON plus the full evidence, create executable manual test cases. Each case must contain ordered steps whose action is manual and whose expected result is directly observable.
Return one TestCaseBatch.

Validated requirements JSON:
{requirements.model_dump_json()}

Validated scenarios JSON:
{scenarios.model_dump_json()}

{_evidence(chunks)}"""


def review_prompt(
    label: str, value: BaseModel, chunks: Iterable[DocumentChunk]
) -> str:
    return f"""{RULES}

Review the {label} for groundedness, completeness, duplicate IDs, valid relationships, and citations to real chunks with supported excerpts. Return one ReviewResult. Set accepted to false and list every required correction if any issue exists.

Artifact JSON:
{value.model_dump_json()}

{_evidence(chunks)}"""


def revision_prompt(
    label: str,
    value: BaseModel,
    review: ReviewResult,
    chunks: Iterable[DocumentChunk],
) -> str:
    return f"""{RULES}

Revise the {label} to address every review issue exactly once while preserving all supported content. Return the same schema as the artifact.

Artifact JSON:
{value.model_dump_json()}

Review issues JSON:
{json.dumps([issue.model_dump(mode="json") for issue in review.issues], ensure_ascii=False)}

{_evidence(chunks)}"""


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
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _record(self, result: GenerationResult | StructuredOutputError) -> None:
        with self._lock:
            self.input_tokens += result.input_tokens
            self.output_tokens += result.output_tokens
            self.latency_seconds += result.latency_seconds

    def generate(
        self,
        messages: Messages,
        schema: type[T],
        max_output_tokens: int,
        allow_schema_repair: bool = True,
    ) -> T:
        current_messages = [message.copy() for message in messages]
        transport_retries = 0
        schema_repaired = False
        while True:
            try:
                result = self.provider.generate(
                    current_messages, schema, max_output_tokens=max_output_tokens
                )
            except ProviderError as error:
                if not error.retryable or transport_retries == 2:
                    raise
                delay = 2**transport_retries
                with self._lock:
                    self.retries += 1
                transport_retries += 1
                self.sleep(delay)
            except StructuredOutputError as error:
                self._record(error)
                if not allow_schema_repair or schema_repaired:
                    raise
                with self._lock:
                    self.schema_repairs += 1
                schema_repaired = True
                current_messages.append(
                    _user(
                        "The previous response was an invalid response. Return valid "
                        "JSON matching this schema and all evidence/support constraints.\n"
                        f"Schema:\n{json.dumps(schema.model_json_schema(), ensure_ascii=False)}\n"
                        f"Invalid response:\n{error.raw_text}"
                    )
                )
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
    ) -> T:
        with self._lock:
            self.semantic_revisions += 1
        return self.generate(
            [*messages, _user(revision_prompt(label, value, review, chunks))],
            schema,
            max_output_tokens,
        )


def run_single_prompt(
    context: PipelineContext, chunks: Iterable[DocumentChunk]
) -> ArtifactBundle:
    return context.generate(
        [_user(single_prompt(chunks))], ArtifactBundle, max_output_tokens=30_000
    )


def _review_once(
    context: PipelineContext,
    history: Messages,
    label: str,
    value: T,
    chunks: list[DocumentChunk],
    schema: type[T],
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
        history, label, value, review, chunks, schema, max_output_tokens
    )


def run_staged_single_agent(
    context: PipelineContext, chunks: Iterable[DocumentChunk]
) -> ArtifactBundle:
    chunks = list(chunks)
    history: Messages = []

    prompt = _user(requirements_prompt(chunks))
    requirements = context.generate(
        [*history, prompt], RequirementBatch, max_output_tokens=12_000
    )
    history.extend((prompt, _assistant(requirements)))
    requirements = _review_once(
        context, history, "requirements", requirements, chunks, RequirementBatch, 12_000
    )
    history[-1] = _assistant(requirements)

    prompt = _user(scenarios_prompt(requirements, chunks))
    scenarios = context.generate(
        [*history, prompt], ScenarioBatch, max_output_tokens=12_000
    )
    history.extend((prompt, _assistant(scenarios)))
    scenarios = _review_once(
        context, history, "scenarios", scenarios, chunks, ScenarioBatch, 12_000
    )
    history[-1] = _assistant(scenarios)

    prompt = _user(test_cases_prompt(requirements, scenarios, chunks))
    test_cases = context.generate(
        [*history, prompt], TestCaseBatch, max_output_tokens=20_000
    )
    history.extend((prompt, _assistant(test_cases)))
    test_cases = _review_once(
        context, history, "test cases", test_cases, chunks, TestCaseBatch, 20_000
    )
    history[-1] = _assistant(test_cases)

    return ArtifactBundle(
        requirements=requirements.requirements,
        scenarios=scenarios.scenarios,
        test_cases=test_cases.test_cases,
    )
