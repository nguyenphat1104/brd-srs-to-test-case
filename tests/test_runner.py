import json
import threading
from collections import deque
from datetime import UTC, datetime

import pytest

from brd_srs_testgen import runner
from brd_srs_testgen.documents import DocumentError
from brd_srs_testgen.models import (
    ArtifactBundle,
    ComparisonManifest,
    Condition,
    GeneratedCases,
    RequirementBatch,
    ReviewResult,
    RunStatus,
    ScenarioBatch,
    TestCaseBatch as ModelTestCaseBatch,
)
from brd_srs_testgen.providers import BudgetLedger, GenerationResult, ProviderError
from brd_srs_testgen.runner import ProviderSettings, run_comparison
from brd_srs_testgen.storage import ImmutableArtifactError, RunStore, StorageError
from tests.factories import bundle, chunk


def _result(value):
    return GenerationResult(
        value=value,
        input_tokens=1,
        output_tokens=1,
        latency_seconds=0.01,
    )


class ScriptedProvider:
    model = "test-model"

    def __init__(self, ledger: BudgetLedger, responses) -> None:
        self.ledger = ledger
        self.responses = deque(responses)
        self.lock = threading.Lock()

    def generate(self, messages, schema, *, max_output_tokens):
        with self.lock:
            response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return _result(schema.model_validate(response.model_dump(mode="json")))


class CentralProvider:
    model = "test-model"

    def __init__(self, ledger: BudgetLedger) -> None:
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
            elif schema is ReviewResult:
                value = ReviewResult(accepted=True)
            else:
                raise AssertionError(f"Unexpected schema: {schema}")
        return _result(schema.model_validate(value.model_dump(mode="json")))


def accepted() -> ReviewResult:
    return ReviewResult(accepted=True)


def provider_factory(*, fail_single: bool = False):
    artifacts = bundle()

    def factory(condition: Condition, ledger: BudgetLedger):
        if condition is Condition.SINGLE_PROMPT:
            responses = (
                [ProviderError("rejected", code=400, retryable=False)]
                if fail_single
                else [artifacts]
            )
            return ScriptedProvider(ledger, responses)
        if condition is Condition.STAGED_SINGLE_AGENT:
            return ScriptedProvider(
                ledger,
                [
                    RequirementBatch(requirements=artifacts.requirements),
                    accepted(),
                    ScenarioBatch(scenarios=artifacts.scenarios),
                    accepted(),
                    ModelTestCaseBatch(test_cases=artifacts.test_cases),
                    accepted(),
                ],
            )
        return CentralProvider(ledger)

    return factory


def settings() -> ProviderSettings:
    return ProviderSettings(
        provider="ollama",
        model="test-model",
        token_ceiling=100_000,
    )


def test_comparison_artifacts_require_an_active_existing_comparison(tmp_path) -> None:
    store = RunStore(tmp_path)
    manifest = ComparisonManifest(
        comparison_id="comparison",
        document_hash="0" * 64,
        provider="ollama",
        model="test-model",
        temperature=0.0,
        token_ceiling=100_000,
        condition_order=list(Condition),
        prompt_version="test",
        schema_version="test",
        started_at=datetime.now(UTC),
    )

    with pytest.raises(StorageError):
        store.write_comparison_artifact("missing", "failure.json", {})
    store.create_comparison(manifest, [])
    path = store.write_comparison_artifact(
        manifest.comparison_id, "failure.json", {"category": "parsing"}
    )
    with pytest.raises(ImmutableArtifactError):
        store.write_comparison_artifact(
            manifest.comparison_id, "failure.json", {"category": "other"}
        )
    store.update_comparison(
        manifest.model_copy(update={"completed_at": datetime.now(UTC)})
    )
    with pytest.raises(ImmutableArtifactError):
        store.write_comparison_artifact(
            manifest.comparison_id, "late.json", {}
        )

    assert json.loads(path.read_text()) == {"category": "parsing"}
    assert not (store.comparison_dir(manifest.comparison_id) / "late.json").exists()


def test_comparison_runs_and_persists_all_conditions(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])
    store = RunStore(tmp_path)

    result = run_comparison(
        b"pdf",
        settings(),
        store=store,
        provider_factory=provider_factory(),
    )

    assert list(result.conditions) == list(Condition)
    assert all(
        item.manifest.status is RunStatus.COMPLETED
        for item in result.conditions.values()
    )
    assert (
        store.condition_dir(
            result.manifest.comparison_id, Condition.SINGLE_PROMPT
        )
        / "rtm.json"
    ).exists()


def test_one_failed_condition_does_not_stop_the_others(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])

    result = run_comparison(
        b"pdf",
        settings(),
        store=RunStore(tmp_path),
        provider_factory=provider_factory(fail_single=True),
    )

    assert result.conditions[Condition.SINGLE_PROMPT].manifest.status is RunStatus.FAILED
    assert (
        result.conditions[Condition.SINGLE_PROMPT].manifest.failure_category.value
        == "provider_rejection"
    )
    assert all(
        result.conditions[condition].manifest.status is RunStatus.COMPLETED
        for condition in (
            Condition.STAGED_SINGLE_AGENT,
            Condition.CENTRALIZED_MULTI_AGENT,
        )
    )


def test_parsing_failure_is_persisted(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "parse_pdf",
        lambda _data: (_ for _ in ()).throw(DocumentError("unreadable")),
    )
    store = RunStore(tmp_path)

    result = run_comparison(b"pdf", settings(), store=store)

    assert result.failure_category == "parsing"
    assert result.conditions == {}
    assert (
        store.comparison_dir(result.manifest.comparison_id) / "failure.json"
    ).exists()
