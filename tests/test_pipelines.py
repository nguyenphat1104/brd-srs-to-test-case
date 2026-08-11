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
