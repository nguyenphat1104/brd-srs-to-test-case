import json
import threading
from collections import deque
from datetime import UTC, datetime

import pytest

import brd_srs_testgen.providers as providers_module
import brd_srs_testgen.storage as storage_module
from brd_srs_testgen import runner
from brd_srs_testgen.documents import DocumentError
from brd_srs_testgen.models import (
    ArtifactBundle,
    ComparisonManifest,
    Condition,
    ConditionManifest,
    CoverageAssignment,
    CoverageRepair,
    GeneratedCases,
    RequirementBatch,
    ReviewResult,
    RunStatus,
    ScenarioBatch,
    TestCaseBatch as ModelTestCaseBatch,
)
from brd_srs_testgen.providers import (
    BudgetLedger,
    GenerationResult,
    ProviderError,
)
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
            if issubclass(schema, RequirementBatch):
                value = RequirementBatch(
                    requirements=(
                        self.artifacts.requirements
                        if "p0001-c001" in content or "CANDIDATES" in content
                        else []
                    )
                )
            elif issubclass(schema, GeneratedCases):
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
                    ScenarioBatch(scenarios=artifacts.scenarios),
                    ModelTestCaseBatch(test_cases=artifacts.test_cases),
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


@pytest.mark.parametrize(
    "reserved_name",
    ["manifest.json", "chunks.json", "conditions", ".runstore.lock", ".tmp-artifact"],
)
def test_comparison_artifacts_reject_storage_owned_names_before_mutation(
    tmp_path, monkeypatch, reserved_name
) -> None:
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
    directory = store.create_comparison(manifest, [])
    original = {
        path.name: path.read_bytes() for path in directory.iterdir()
    }
    mutation = store._mutation

    def reject_mutation(**_kwargs):
        raise AssertionError("reserved names must be rejected before mutation")

    monkeypatch.setattr(store, "_mutation", reject_mutation)
    with pytest.raises(StorageError):
        store.write_comparison_artifact(
            manifest.comparison_id, reserved_name, {"corrupt": True}
        )
    monkeypatch.setattr(store, "_mutation", mutation)

    assert {path.name: path.read_bytes() for path in directory.iterdir()} == original
    store.write_comparison_artifact(
        manifest.comparison_id, "failure.json", {"category": "parsing"}
    )
    store.start_condition(
        manifest.comparison_id,
        ConditionManifest(
            condition=Condition.SINGLE_PROMPT,
            status=RunStatus.RUNNING,
            provider="ollama",
            model="test-model",
            temperature=0.0,
            token_ceiling=100_000,
            started_at=datetime.now(UTC),
        ),
    )

    assert (directory / "failure.json").exists()
    assert store.condition_dir(
        manifest.comparison_id, Condition.SINGLE_PROMPT
    ).is_dir()


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


class ChargedProvider(ScriptedProvider):
    def generate(self, messages, schema, *, max_output_tokens):
        result = super().generate(
            messages, schema, max_output_tokens=max_output_tokens
        )
        reservation = self.ledger.reserve(7)
        self.ledger.settle(reservation, 7)
        return result


class BudgetFailingProvider:
    model = "test-model"

    def __init__(self, ledger: BudgetLedger) -> None:
        self.ledger = ledger

    def generate(self, messages, schema, *, max_output_tokens):
        reservation = self.ledger.reserve(1)
        self.ledger.settle(reservation, self.ledger.limit + 2)
        raise AssertionError("BudgetExceeded was not raised")


class TimeoutProvider:
    model = "test-model"

    def __init__(self, ledger: BudgetLedger) -> None:
        self.ledger = ledger

    def generate(self, messages, schema, *, max_output_tokens):
        raise ProviderError(
            "request timed out", code=None, retryable=True, timed_out=True
        )


class InvalidCentralProvider(CentralProvider):
    def generate(self, messages, schema, *, max_output_tokens):
        result = super().generate(
            messages, schema, max_output_tokens=max_output_tokens
        )
        if (
            issubclass(schema, RequirementBatch)
            and "WORKER REQUIREMENT EXTRACTION 1/3" in messages[-1]["content"]
            and result.value.requirements
        ):
            invalid = result.value.requirements[0].model_copy(
                update={"requirement_id": "REQ-1001"}
            )
            return _result(RequirementBatch(requirements=[invalid]))
        return result


def _persisted_condition(store, result, condition: Condition):
    directory = store.condition_dir(result.manifest.comparison_id, condition)
    metrics = json.loads((directory / "metrics.json").read_text())
    event = json.loads((directory / "events.jsonl").read_text().splitlines()[-1])
    return metrics, event


def test_charged_tokens_match_metrics_download_persistence_and_event(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])
    base_factory = provider_factory()

    def factory(condition, ledger):
        if condition is Condition.SINGLE_PROMPT:
            return ChargedProvider(ledger, [bundle()])
        return base_factory(condition, ledger)

    store = RunStore(tmp_path)
    result = run_comparison(
        b"pdf", settings(), store=store, provider_factory=factory
    )
    single = result.conditions[Condition.SINGLE_PROMPT]
    persisted, event = _persisted_condition(
        store, result, Condition.SINGLE_PROMPT
    )

    assert single.metrics.charged_tokens == 7
    assert single.download_bundle()["metrics"] == persisted
    assert persisted["charged_tokens"] == event["charged_tokens"] == 7
    assert persisted["input_tokens"] == persisted["output_tokens"] == 1
    assert event["provider"] == "ollama"
    assert event["model"] == "test-model"
    assert event["temperature"] == 0.0
    assert event["token_ceiling"] == 100_000
    assert event["validation"] == single.validation.model_dump(mode="json")
    assert event["failure_category"] is None
    assert event["failure_message"] is None


def test_budget_failure_keeps_charged_tokens_without_reported_usage(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])
    base_factory = provider_factory()

    def factory(condition, ledger):
        if condition is Condition.SINGLE_PROMPT:
            return BudgetFailingProvider(ledger)
        return base_factory(condition, ledger)

    store = RunStore(tmp_path)
    result = run_comparison(
        b"pdf", settings(), store=store, provider_factory=factory
    )
    single = result.conditions[Condition.SINGLE_PROMPT]
    persisted, event = _persisted_condition(
        store, result, Condition.SINGLE_PROMPT
    )

    assert single.manifest.failure_category.value == "budget_exhaustion"
    assert single.metrics.input_tokens == single.metrics.output_tokens == 0
    assert single.metrics.charged_tokens == settings().token_ceiling + 2
    assert single.download_bundle()["metrics"] == persisted
    assert persisted["charged_tokens"] == event["charged_tokens"]
    assert event["provider"] == "ollama"
    assert event["model"] == "test-model"
    assert event["validation"] is None
    assert event["failure_category"] == "budget_exhaustion"
    assert event["failure_message"]


def test_exhausted_provider_timeout_is_persisted_as_timeout(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])
    base_factory = provider_factory()

    def factory(condition, ledger):
        if condition is Condition.SINGLE_PROMPT:
            return TimeoutProvider(ledger)
        return base_factory(condition, ledger)

    def timeout_pipeline(context, _chunks):
        context.sleep = lambda _seconds: None
        return context.generate([], ArtifactBundle, max_output_tokens=1)

    monkeypatch.setitem(
        runner.PIPELINES, Condition.SINGLE_PROMPT, timeout_pipeline
    )
    store = RunStore(tmp_path)

    result = run_comparison(
        b"pdf", settings(), store=store, provider_factory=factory
    )
    single = result.conditions[Condition.SINGLE_PROMPT]
    _metrics, event = _persisted_condition(
        store, result, Condition.SINGLE_PROMPT
    )

    assert single.manifest.failure_category.value == "timeout"
    assert single.metrics.retries == 2
    assert event["failure_category"] == "timeout"
    assert event["failure_message"] == "request timed out"


def test_actual_ollama_timeout_survives_budget_blocked_retry(
    tmp_path, monkeypatch
) -> None:
    evidence = chunk().model_copy(
        update={"text": f"{chunk().text} {'x' * 40_000}"}
    )
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [evidence])
    for condition in (
        Condition.STAGED_SINGLE_AGENT,
        Condition.CENTRALIZED_MULTI_AGENT,
    ):
        monkeypatch.setitem(
            runner.PIPELINES, condition, lambda _context, _chunks: bundle()
        )
    calls = 0

    def timeout(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise TimeoutError("ollama timed out")

    monkeypatch.setattr(providers_module, "urlopen", timeout)
    store = RunStore(tmp_path)

    result = run_comparison(b"pdf", settings(), store=store)
    single = result.conditions[Condition.SINGLE_PROMPT]
    _metrics, event = _persisted_condition(
        store, result, Condition.SINGLE_PROMPT
    )

    assert calls == 1
    assert single.manifest.failure_category.value == "timeout"
    assert single.metrics.charged_tokens > 30_000
    assert single.metrics.budget_exhausted is False
    assert event["failure_category"] == "timeout"
    assert event["failure_message"] == "ollama timed out"


def test_terminal_event_retains_failed_validation_report(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])
    artifacts = bundle()
    requirement = artifacts.requirements[0].model_copy(
        update={
            "source_references": [
                artifacts.requirements[0].source_references[0].model_copy(
                    update={"excerpt": "invented evidence"}
                )
            ]
        }
    )
    invalid = artifacts.model_copy(update={"requirements": [requirement]})
    base_factory = provider_factory()

    def factory(condition, ledger):
        if condition is Condition.SINGLE_PROMPT:
            return ScriptedProvider(ledger, [invalid, invalid])
        return base_factory(condition, ledger)

    store = RunStore(tmp_path)
    result = run_comparison(
        b"pdf", settings(), store=store, provider_factory=factory
    )
    single = result.conditions[Condition.SINGLE_PROMPT]
    _metrics, event = _persisted_condition(
        store, result, Condition.SINGLE_PROMPT
    )

    assert single.validation.valid is False
    assert event["validation"] == single.validation.model_dump(mode="json")
    assert event["failure_category"] == "semantic_validation"
    assert event["failure_message"] == "1 deterministic validation issues."
    assert single.metrics.semantic_revisions == 1


def test_failed_validation_gets_one_deterministic_repair(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])
    artifacts = bundle()
    reference = artifacts.requirements[0].source_references[0].model_copy(
        update={"excerpt": "invented evidence"}
    )
    invalid = artifacts.model_copy(
        update={
            "requirements": [
                artifacts.requirements[0].model_copy(
                    update={"source_references": [reference]}
                )
            ]
        }
    )
    base_factory = provider_factory()

    def factory(condition, ledger):
        if condition is Condition.SINGLE_PROMPT:
            return ScriptedProvider(ledger, [invalid, artifacts])
        return base_factory(condition, ledger)

    result = run_comparison(
        b"pdf", settings(), store=RunStore(tmp_path), provider_factory=factory
    )
    single = result.conditions[Condition.SINGLE_PROMPT]

    assert single.manifest.status is RunStatus.COMPLETED
    assert single.validation.valid is True
    assert single.metrics.semantic_revisions == 1


def test_uncovered_requirement_gets_link_only_repair(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])
    artifacts = bundle()
    uncovered = artifacts.requirements[0].model_copy(
        update={
            "requirement_id": "REQ-002",
            "title": "Second requirement",
            "description": "The same scenario also covers this requirement.",
        }
    )
    invalid = artifacts.model_copy(
        update={"requirements": [*artifacts.requirements, uncovered]}
    )
    repair = CoverageRepair(
        assignments=[
            CoverageAssignment(
                requirement_id="REQ-002",
                scenario_id=artifacts.scenarios[0].scenario_id,
                test_case_id=artifacts.test_cases[0].test_case_id,
            )
        ]
    )
    base_factory = provider_factory()

    def factory(condition, ledger):
        if condition is Condition.SINGLE_PROMPT:
            return ScriptedProvider(ledger, [invalid, repair])
        return base_factory(condition, ledger)

    result = run_comparison(
        b"pdf", settings(), store=RunStore(tmp_path), provider_factory=factory
    )
    single = result.conditions[Condition.SINGLE_PROMPT]

    assert single.manifest.status is RunStatus.COMPLETED
    assert single.validation.valid is True
    assert "REQ-002" in single.bundle.scenarios[0].requirement_ids
    assert "REQ-002" in single.bundle.test_cases[0].requirement_ids
    assert single.metrics.semantic_revisions == 1


def test_progress_callback_exceptions_never_change_the_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])

    def broken_progress(_condition, _message):
        raise RuntimeError("observer broke")

    result = run_comparison(
        b"pdf",
        settings(),
        store=RunStore(tmp_path),
        provider_factory=provider_factory(),
        progress=broken_progress,
    )

    assert all(
        condition.manifest.status is RunStatus.COMPLETED
        for condition in result.conditions.values()
    )


def test_parse_failure_emits_terminal_progress(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "parse_pdf",
        lambda _data: (_ for _ in ()).throw(DocumentError("unreadable")),
    )
    trace = []

    run_comparison(
        b"pdf",
        settings(),
        store=RunStore(tmp_path),
        progress=lambda condition, message: trace.append((condition, message)),
    )

    assert trace == [(None, "Parsing PDF"), (None, "Failed")]


def test_lm_studio_settings_create_authenticated_provider() -> None:
    configured = ProviderSettings(
        provider="lm_studio",
        model="local-model",
        token_ceiling=100_000,
        api_key="local-token",
        base_url="http://localhost:1234/v1",
    )

    configured.validate()
    provider = runner._make_provider(configured, BudgetLedger(100_000))

    assert isinstance(provider, providers_module.LMStudioProvider)
    assert provider.api_key == "local-token"


@pytest.mark.parametrize(
    "base_url",
    [
        "http://user:secret@host",
        "http://host?token=secret",
        "http://host#secret",
        "http://host?",
        "http://host#",
        " http://host",
        "http://ho st",
        123,
    ],
)
def test_ollama_credential_bearing_urls_fail_before_persistence(
    tmp_path, base_url
) -> None:
    configured = ProviderSettings(
        provider="ollama",
        model="test-model",
        token_ceiling=100_000,
        base_url=base_url,
    )

    with pytest.raises(ValueError):
        run_comparison(b"pdf", configured, store=RunStore(tmp_path))

    assert list(tmp_path.iterdir()) == []


def test_ollama_credentials_are_not_exposed_by_settings_repr() -> None:
    configured = ProviderSettings(
        provider="ollama",
        model="test-model",
        token_ceiling=100_000,
        base_url="http://user:secret@host",
    )

    assert "user" not in repr(configured)
    assert "secret" not in repr(configured)


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider": 1},
        {"model": 1},
        {"token_ceiling": "100"},
        {"provider": "gemini", "api_key": 1},
    ],
)
def test_provider_settings_reject_wrong_types(overrides) -> None:
    values = {
        "provider": "ollama",
        "model": "test-model",
        "token_ceiling": 100_000,
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        ProviderSettings(**values).validate()


def test_provider_secrets_are_redacted_from_condition_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])
    secret = "gemini-super-secret"
    configured = ProviderSettings(
        provider="gemini",
        model="test-model",
        token_ceiling=100_000,
        api_key=secret,
    )
    base_factory = provider_factory()

    def factory(condition, ledger):
        if condition is Condition.SINGLE_PROMPT:
            raise ValueError(f"api_key={secret} token=secondary password=tertiary")
        return base_factory(condition, ledger)

    run_comparison(
        b"pdf", configured, store=RunStore(tmp_path), provider_factory=factory
    )
    persisted = "\n".join(
        path.read_text() for path in tmp_path.rglob("*") if path.is_file()
    )

    assert secret not in persisted
    assert "secondary" not in persisted
    assert "tertiary" not in persisted


@pytest.mark.parametrize("mismatch", ["shared_ledger", "wrong_model"])
def test_injected_provider_integrity_failure_is_isolated(
    tmp_path, monkeypatch, mismatch
) -> None:
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])
    base_factory = provider_factory()
    first_ledger = None

    def factory(condition, ledger):
        nonlocal first_ledger
        provider = base_factory(condition, ledger)
        if condition is Condition.SINGLE_PROMPT:
            first_ledger = ledger
        elif condition is Condition.STAGED_SINGLE_AGENT:
            if mismatch == "shared_ledger":
                provider.ledger = first_ledger
            else:
                provider.model = "wrong-model"
        return provider

    result = run_comparison(
        b"pdf", settings(), store=RunStore(tmp_path), provider_factory=factory
    )

    assert (
        result.conditions[Condition.STAGED_SINGLE_AGENT].manifest.failure_category.value
        == "configuration"
    )
    assert (
        result.conditions[Condition.CENTRALIZED_MULTI_AGENT].manifest.status
        is RunStatus.COMPLETED
    )


def test_invalid_central_worker_output_is_semantic_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])
    base_factory = provider_factory()

    def factory(condition, ledger):
        if condition is Condition.CENTRALIZED_MULTI_AGENT:
            return InvalidCentralProvider(ledger)
        return base_factory(condition, ledger)

    result = run_comparison(
        b"pdf", settings(), store=RunStore(tmp_path), provider_factory=factory
    )

    assert (
        result.conditions[
            Condition.CENTRALIZED_MULTI_AGENT
        ].manifest.failure_category.value
        == "semantic_validation"
    )


def test_unexpected_pipeline_defect_is_re_raised(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])

    def defect(_context, _chunks):
        raise AssertionError("programming defect")

    monkeypatch.setitem(runner.PIPELINES, Condition.SINGLE_PROMPT, defect)
    store = RunStore(tmp_path)

    with pytest.raises(AssertionError, match="programming defect"):
        run_comparison(
            b"pdf",
            settings(),
            store=store,
            provider_factory=provider_factory(),
        )

    comparison = next(path for path in tmp_path.iterdir() if path.is_dir())
    condition = comparison / "conditions" / Condition.SINGLE_PROMPT
    assert json.loads((condition / "manifest.json").read_text())["status"] == "running"
    assert {path.name for path in condition.iterdir()} == {
        "manifest.json",
        "events.jsonl",
    }


def test_runner_finalization_failure_rolls_back_and_stops(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])
    original_write = storage_module._atomic_text
    failed = False

    def fail_midway(path, text):
        nonlocal failed
        if path.name == "scenarios.json" and not failed:
            failed = True
            raise OSError("injected finalization failure")
        original_write(path, text)

    monkeypatch.setattr(storage_module, "_atomic_text", fail_midway)
    store = RunStore(tmp_path)

    with pytest.raises(OSError, match="injected finalization failure"):
        run_comparison(
            b"pdf",
            settings(),
            store=store,
            provider_factory=provider_factory(),
        )

    comparison = next(path for path in tmp_path.iterdir() if path.is_dir())
    conditions = comparison / "conditions"
    single = conditions / Condition.SINGLE_PROMPT
    assert (
        json.loads((single / "manifest.json").read_text())["status"] == "running"
    )
    assert [
        json.loads(line)["stage"]
        for line in (single / "events.jsonl").read_text().splitlines()
    ] == ["started"]
    assert {path.name for path in single.iterdir()} == {
        "manifest.json",
        "events.jsonl",
    }
    assert not (conditions / Condition.STAGED_SINGLE_AGENT).exists()


def test_comparison_ids_are_unique_with_frozen_time(monkeypatch) -> None:
    class FrozenDateTime:
        @classmethod
        def now(cls, _timezone):
            return datetime(2026, 8, 11, tzinfo=UTC)

    monkeypatch.setattr(runner, "datetime", FrozenDateTime)

    first = runner._comparison_id("a" * 64)
    second = runner._comparison_id("a" * 64)

    assert first != second
    assert first.startswith("20260811T000000000000Z-aaaaaaaaaaaa-")
