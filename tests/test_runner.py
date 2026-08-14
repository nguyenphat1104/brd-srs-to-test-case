import hashlib
import threading
from collections import deque
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

import brd_srs_testgen.providers as providers_module
from brd_srs_testgen import runner
from brd_srs_testgen.documents import DocumentError
from brd_srs_testgen.models import (
    ArtifactBundle,
    CoverageAssignment,
    CoverageRepair,
    FailureCategory,
    GeneratedCases,
    RequirementBatch,
    ReviewResult,
    RunStatus,
    RunType,
)
from brd_srs_testgen.pipelines import PipelineOutputError
from brd_srs_testgen.providers import (
    BudgetLedger,
    GenerationResult,
    ProviderError,
    StructuredOutputError,
)
from brd_srs_testgen.runner import ProviderSettings, run_generation
from tests.factories import bundle, chunk


class RecordingRepository:
    def __init__(self, fail_at=None) -> None:
        self.fail_at = fail_at
        self.calls = []
        self.created = []
        self.saved = []
        self.events = []
        self.finalized = []

    def _fail(self, method) -> None:
        if self.fail_at == method:
            raise RuntimeError(f"{method} failed")

    def create_run(self, manifest) -> None:
        self.calls.append(("create_run", manifest))
        self.created.append(manifest)
        self._fail("create_run")

    def save_chunks(self, run_id, chunks) -> None:
        items = list(chunks)
        self.calls.append(("save_chunks", run_id, items))
        self.saved.append((run_id, items))
        self._fail("save_chunks")

    def append_event(self, run_id, stage, occurred_at=None) -> None:
        self.calls.append(("append_event", run_id, stage, occurred_at))
        self.events.append((run_id, stage, occurred_at))
        self._fail("append_event")

    def finalize(self, result) -> None:
        self.calls.append(("finalize", result))
        self.finalized.append(result)
        self._fail("finalize")


def _result(value, *, input_tokens=1, output_tokens=1):
    return GenerationResult(
        value=value,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
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


class ChargedProvider(ScriptedProvider):
    def generate(self, messages, schema, *, max_output_tokens):
        result = super().generate(messages, schema, max_output_tokens=max_output_tokens)
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


class InvalidCentralProvider(CentralProvider):
    def generate(self, messages, schema, *, max_output_tokens):
        result = super().generate(messages, schema, max_output_tokens=max_output_tokens)
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


def settings(**overrides) -> ProviderSettings:
    values = {
        "provider": "ollama",
        "model": "test-model",
        "token_ceiling": 100_000,
    }
    values.update(overrides)
    return ProviderSettings(**values)


def _successful_run(
    monkeypatch,
    *,
    run_type=RunType.SINGLE_PROMPT,
    source_filename="sample.pdf",
    pdf_bytes=b"pdf",
    repository=None,
    progress=None,
):
    repository = repository or RecordingRepository()
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])
    monkeypatch.setitem(runner.PIPELINES, run_type, lambda _context, _chunks: bundle())
    result = run_generation(
        pdf_bytes,
        source_filename,
        run_type,
        settings(),
        repository=repository,
        progress=progress,
        provider_factory=lambda _run_type, ledger: ScriptedProvider(ledger, []),
    )
    return result, repository


@pytest.mark.parametrize("run_type", list(RunType))
def test_only_the_selected_pipeline_runs_and_run_type_is_persisted(
    monkeypatch, run_type
) -> None:
    called = []
    seen_factory = []
    artifacts = bundle()
    repository = RecordingRepository()
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])
    for candidate in RunType:
        monkeypatch.setitem(
            runner.PIPELINES,
            candidate,
            lambda _context, _chunks, candidate=candidate: (
                called.append(candidate) or artifacts
            ),
        )

    def factory(selected, ledger):
        seen_factory.append((selected, ledger))
        return ScriptedProvider(ledger, [])

    result = run_generation(
        b"pdf",
        "sample.pdf",
        run_type,
        settings(),
        repository=repository,
        provider_factory=factory,
    )

    assert called == [run_type]
    assert [selected for selected, _ledger in seen_factory] == [run_type]
    assert result.manifest.run_type is run_type
    assert repository.created[0].run_type is run_type
    assert repository.finalized == [result]


@pytest.mark.parametrize(
    ("source_filename", "expected"),
    [
        ("/private/uploads/spec.pdf", "spec.pdf"),
        (r"C:\Users\alice\private\spec.pdf", "spec.pdf"),
        ("spec.pdf", "spec.pdf"),
        ("", "document.pdf"),
        (".", "document.pdf"),
        ("..", "document.pdf"),
        ("../../", "document.pdf"),
        ("/", "document.pdf"),
        ("   ", "document.pdf"),
    ],
)
def test_source_filename_is_a_safe_display_basename_and_hash_is_correct(
    monkeypatch, source_filename, expected
) -> None:
    pdf_bytes = b"private PDF bytes"

    result, repository = _successful_run(
        monkeypatch, source_filename=source_filename, pdf_bytes=pdf_bytes
    )

    assert result.manifest.source_filename == expected
    assert repository.created[0].source_filename == expected
    assert result.manifest.document_hash == hashlib.sha256(pdf_bytes).hexdigest()


def test_settings_are_validated_before_any_run_is_created(monkeypatch) -> None:
    repository = RecordingRepository()
    monkeypatch.setattr(
        runner,
        "parse_pdf",
        lambda _data: (_ for _ in ()).throw(AssertionError("parse must not run")),
    )

    with pytest.raises(ValueError, match="Model must not be blank"):
        run_generation(
            b"pdf",
            "sample.pdf",
            RunType.SINGLE_PROMPT,
            settings(model=""),
            repository=repository,
        )

    assert repository.calls == []


def test_running_record_and_started_event_precede_parse_and_provider(
    monkeypatch,
) -> None:
    repository = RecordingRepository()
    trace = []

    def parse(_data):
        assert repository.created[0].status is RunStatus.RUNNING
        assert [call[0] for call in repository.calls] == [
            "create_run",
            "append_event",
        ]
        assert repository.events[0][1] == "started"
        return [chunk()]

    def factory(selected, ledger):
        assert selected is RunType.SINGLE_PROMPT
        assert repository.saved == [(repository.created[0].run_id, [chunk()])]
        assert [event[1] for event in repository.events] == ["started", "parsed"]
        return ScriptedProvider(ledger, [])

    monkeypatch.setattr(runner, "parse_pdf", parse)
    monkeypatch.setitem(
        runner.PIPELINES, RunType.SINGLE_PROMPT, lambda _context, _chunks: bundle()
    )

    result = run_generation(
        b"pdf",
        "sample.pdf",
        RunType.SINGLE_PROMPT,
        settings(),
        repository=repository,
        provider_factory=factory,
        progress=trace.append,
    )

    assert result.manifest.status is RunStatus.COMPLETED
    assert trace[0] == "Preparing document"
    assert trace[-1] == "Completed"


def test_pipeline_activity_is_forwarded_to_the_progress_callback(monkeypatch) -> None:
    repository = RecordingRepository()
    trace = []
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])

    def generate(context, _chunks):
        context.notify("Orchestrator: spawning Analyzer 1.")
        return bundle()

    monkeypatch.setitem(runner.PIPELINES, RunType.CENTRALIZED_MULTI_AGENT, generate)

    result = run_generation(
        b"pdf",
        "sample.pdf",
        RunType.CENTRALIZED_MULTI_AGENT,
        settings(),
        repository=repository,
        progress=trace.append,
        provider_factory=lambda _run_type, ledger: ScriptedProvider(ledger, []),
    )

    assert result.manifest.status is RunStatus.COMPLETED
    assert trace == [
        "Preparing document",
        "Generating artifacts",
        "Orchestrator: spawning Analyzer 1.",
        "Completed",
    ]


def test_parsing_failure_is_finalized_without_generated_parts(monkeypatch) -> None:
    secret = "gemini-super-secret"
    base_url = "http://private-lm-studio:1234/v1"
    configured = settings(
        provider="lm_studio",
        api_key=secret,
        base_url=base_url,
    )
    repository = RecordingRepository()
    trace = []
    monkeypatch.setattr(
        runner,
        "parse_pdf",
        lambda _data: (_ for _ in ()).throw(
            DocumentError(f"unreadable api_key={secret} at {base_url}")
        ),
    )

    result = run_generation(
        b"private PDF bytes",
        "/private/uploads/spec.pdf",
        RunType.SINGLE_PROMPT,
        configured,
        repository=repository,
        progress=trace.append,
    )

    assert result.manifest.status is RunStatus.FAILED
    assert result.manifest.failure_category is FailureCategory.PARSING
    assert result.manifest.failure_message == (
        "unreadable api_key=[REDACTED] at [REDACTED]"
    )
    assert secret not in result.manifest.failure_message
    assert base_url not in result.manifest.failure_message
    assert result.bundle is result.validation is result.metrics is None
    assert result.rtm == []
    assert repository.finalized[0] is result
    assert secret not in repository.finalized[0].manifest.failure_message
    assert base_url not in repository.finalized[0].manifest.failure_message
    assert repository.saved == []
    assert [event[1] for event in repository.events] == ["started"]
    assert trace == ["Preparing document", "Failed"]


def test_success_finalizes_the_bundle_validation_rtm_and_metrics(monkeypatch) -> None:
    repository = RecordingRepository()
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])

    result = run_generation(
        b"pdf",
        "sample.pdf",
        RunType.SINGLE_PROMPT,
        settings(),
        repository=repository,
        provider_factory=lambda _run_type, ledger: ChargedProvider(ledger, [bundle()]),
    )

    assert result.manifest.status is RunStatus.COMPLETED
    assert result.bundle == bundle()
    assert result.validation.valid is True
    assert result.rtm[0].covered is True
    assert result.metrics.completion is True
    assert result.metrics.input_tokens == result.metrics.output_tokens == 1
    assert result.metrics.charged_tokens == 7
    assert result.download_bundle()["metrics"] == result.metrics.model_dump(mode="json")
    assert repository.finalized[0] is result


@pytest.mark.parametrize(
    ("errors", "category", "retries", "schema_repairs", "message"),
    [
        (
            [ProviderError("rejected", code=400, retryable=False)],
            FailureCategory.PROVIDER_REJECTION,
            0,
            0,
            "rejected",
        ),
        (
            [ProviderError("network down", code=503, retryable=True)] * 3,
            FailureCategory.TRANSPORT_EXHAUSTION,
            2,
            0,
            "network down",
        ),
        (
            [
                ProviderError(
                    "request timed out", code=None, retryable=True, timed_out=True
                )
            ]
            * 3,
            FailureCategory.TIMEOUT,
            2,
            0,
            "request timed out",
        ),
        (
            [
                StructuredOutputError(
                    "not json", input_tokens=2, output_tokens=3, latency_seconds=0.01
                )
            ]
            * 3,
            FailureCategory.SCHEMA_FAILURE,
            0,
            2,
            "Provider returned invalid structured output.",
        ),
    ],
)
def test_provider_failures_keep_category_message_retry_and_repair_counts(
    monkeypatch, errors, category, retries, schema_repairs, message
) -> None:
    repository = RecordingRepository()
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])

    def generate(context, _chunks):
        context.sleep = lambda _seconds: None
        return context.generate([], ArtifactBundle, max_output_tokens=1)

    monkeypatch.setitem(runner.PIPELINES, RunType.SINGLE_PROMPT, generate)

    result = run_generation(
        b"pdf",
        "sample.pdf",
        RunType.SINGLE_PROMPT,
        settings(),
        repository=repository,
        provider_factory=lambda _run_type, ledger: ScriptedProvider(ledger, errors),
    )

    assert result.manifest.status is RunStatus.FAILED
    assert result.manifest.failure_category is category
    assert result.manifest.failure_message == message
    assert result.bundle is result.validation is None
    assert result.rtm == []
    assert result.metrics.retries == retries
    assert result.metrics.schema_repairs == schema_repairs
    assert repository.finalized[0] is result


def test_budget_failure_keeps_charged_tokens(monkeypatch) -> None:
    repository = RecordingRepository()
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])
    monkeypatch.setitem(
        runner.PIPELINES,
        RunType.SINGLE_PROMPT,
        lambda context, _chunks: context.generate(
            [], ArtifactBundle, max_output_tokens=1
        ),
    )

    result = run_generation(
        b"pdf",
        "sample.pdf",
        RunType.SINGLE_PROMPT,
        settings(),
        repository=repository,
        provider_factory=lambda _run_type, ledger: BudgetFailingProvider(ledger),
    )

    assert result.manifest.failure_category is FailureCategory.BUDGET_EXHAUSTION
    assert result.metrics.input_tokens == result.metrics.output_tokens == 0
    assert result.metrics.charged_tokens == settings().token_ceiling + 2
    assert result.metrics.budget_exhausted is True


def test_pipeline_output_failure_is_semantic_and_has_empty_metrics(monkeypatch) -> None:
    repository = RecordingRepository()
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])
    monkeypatch.setitem(
        runner.PIPELINES,
        RunType.SINGLE_PROMPT,
        lambda _context, _chunks: (_ for _ in ()).throw(
            PipelineOutputError("invalid worker output")
        ),
    )

    result = run_generation(
        b"pdf",
        "sample.pdf",
        RunType.SINGLE_PROMPT,
        settings(),
        repository=repository,
        provider_factory=lambda _run_type, ledger: ScriptedProvider(ledger, []),
    )

    assert result.manifest.failure_category is FailureCategory.SEMANTIC_VALIDATION
    assert result.manifest.failure_message == "invalid worker output"
    assert result.metrics.completion is False


def test_provider_factory_errors_are_safe_configuration_failures(monkeypatch) -> None:
    secret = "gemini-super-secret"
    repository = RecordingRepository()
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])

    def reject_factory(_run_type, _ledger):
        raise ValueError(f"api_key={secret} token=secondary password=tertiary")

    result = run_generation(
        b"pdf",
        "sample.pdf",
        RunType.SINGLE_PROMPT,
        settings(provider="gemini", api_key=secret),
        repository=repository,
        provider_factory=reject_factory,
    )

    assert result.manifest.failure_category is FailureCategory.CONFIGURATION
    assert result.manifest.failure_message == (
        "api_key=[REDACTED] token=[REDACTED] password=[REDACTED]"
    )
    assert result.metrics is not None
    assert repository.finalized == [result]


@pytest.mark.parametrize("error_type", [AssertionError, TypeError, AttributeError])
def test_provider_factory_programming_defects_leave_the_run_interrupted(
    monkeypatch, error_type
) -> None:
    repository = RecordingRepository()
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])

    def defective_factory(_run_type, _ledger):
        raise error_type("factory defect")

    with pytest.raises(error_type, match="factory defect"):
        run_generation(
            b"pdf",
            "sample.pdf",
            RunType.SINGLE_PROMPT,
            settings(),
            repository=repository,
            provider_factory=defective_factory,
        )

    assert repository.created[0].status is RunStatus.RUNNING
    assert repository.finalized == []
    assert [event[1] for event in repository.events] == ["started", "parsed"]


@pytest.mark.parametrize("mismatch", ["ledger", "model"])
def test_provider_must_use_the_run_ledger_and_model(monkeypatch, mismatch) -> None:
    repository = RecordingRepository()
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])

    def factory(_run_type, ledger):
        provider = ScriptedProvider(ledger, [])
        if mismatch == "ledger":
            provider.ledger = BudgetLedger(100_000)
        else:
            provider.model = "wrong-model"
        return provider

    result = run_generation(
        b"pdf",
        "sample.pdf",
        RunType.STAGED_SINGLE_AGENT,
        settings(),
        repository=repository,
        provider_factory=factory,
    )

    assert result.manifest.failure_category is FailureCategory.CONFIGURATION
    assert result.metrics.charged_tokens == 0


def test_failed_validation_retains_normalized_artifacts_and_metrics(
    monkeypatch,
) -> None:
    artifacts = bundle()
    invalid_reference = artifacts.requirements[0].source_references[0].model_copy(
        update={"excerpt": "invented evidence"}
    )
    invalid = artifacts.model_copy(
        update={
            "requirements": [
                artifacts.requirements[0].model_copy(
                    update={"source_references": [invalid_reference]}
                )
            ]
        }
    )
    repository = RecordingRepository()
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])

    result = run_generation(
        b"pdf",
        "sample.pdf",
        RunType.SINGLE_PROMPT,
        settings(),
        repository=repository,
        provider_factory=lambda _run_type, ledger: ScriptedProvider(
            ledger, [invalid, invalid]
        ),
    )

    assert result.manifest.status is RunStatus.FAILED
    assert result.manifest.failure_category is FailureCategory.SEMANTIC_VALIDATION
    assert result.manifest.failure_message == "1 deterministic validation issues."
    assert result.bundle == invalid
    assert result.validation.valid is False
    assert result.rtm
    assert result.metrics.semantic_revisions == 1
    assert repository.finalized[0] is result


def test_failed_validation_gets_one_semantic_revision(monkeypatch) -> None:
    artifacts = bundle()
    invalid_reference = artifacts.requirements[0].source_references[0].model_copy(
        update={"excerpt": "invented evidence"}
    )
    invalid = artifacts.model_copy(
        update={
            "requirements": [
                artifacts.requirements[0].model_copy(
                    update={"source_references": [invalid_reference]}
                )
            ]
        }
    )
    repository = RecordingRepository()
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])

    result = run_generation(
        b"pdf",
        "sample.pdf",
        RunType.SINGLE_PROMPT,
        settings(),
        repository=repository,
        provider_factory=lambda _run_type, ledger: ScriptedProvider(
            ledger, [invalid, artifacts]
        ),
    )

    assert result.manifest.status is RunStatus.COMPLETED
    assert result.validation.valid is True
    assert result.metrics.semantic_revisions == 1


def test_uncovered_requirement_gets_link_only_repair(monkeypatch) -> None:
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
    repository = RecordingRepository()
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])

    result = run_generation(
        b"pdf",
        "sample.pdf",
        RunType.SINGLE_PROMPT,
        settings(),
        repository=repository,
        provider_factory=lambda _run_type, ledger: ScriptedProvider(
            ledger, [invalid, repair]
        ),
    )

    assert result.manifest.status is RunStatus.COMPLETED
    assert result.validation.valid is True
    assert "REQ-002" in result.bundle.scenarios[0].requirement_ids
    assert "REQ-002" in result.bundle.test_cases[0].requirement_ids
    assert result.metrics.semantic_revisions == 1


def test_invalid_central_worker_output_is_semantic_failure(monkeypatch) -> None:
    repository = RecordingRepository()
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])

    result = run_generation(
        b"pdf",
        "sample.pdf",
        RunType.CENTRALIZED_MULTI_AGENT,
        settings(),
        repository=repository,
        provider_factory=lambda _run_type, ledger: InvalidCentralProvider(ledger),
    )

    assert result.manifest.failure_category is FailureCategory.SEMANTIC_VALIDATION
    assert "outside worker 1 range" in result.manifest.failure_message


def test_actual_ollama_timeout_survives_budget_blocked_retry(monkeypatch) -> None:
    evidence = chunk().model_copy(update={"text": f"{chunk().text} {'x' * 40_000}"})
    repository = RecordingRepository()
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [evidence])
    calls = 0

    def timeout(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise TimeoutError("ollama timed out")

    monkeypatch.setattr(providers_module, "urlopen", timeout)

    result = run_generation(
        b"pdf",
        "sample.pdf",
        RunType.SINGLE_PROMPT,
        settings(),
        repository=repository,
    )

    assert calls == 1
    assert result.manifest.failure_category is FailureCategory.TIMEOUT
    assert result.manifest.failure_message == "ollama timed out"
    assert result.metrics.charged_tokens > 30_000
    assert result.metrics.budget_exhausted is False


def test_unexpected_internal_failure_is_re_raised_without_finalizing(
    monkeypatch,
) -> None:
    repository = RecordingRepository()
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])
    monkeypatch.setitem(
        runner.PIPELINES,
        RunType.SINGLE_PROMPT,
        lambda _context, _chunks: (_ for _ in ()).throw(
            AssertionError("programming defect")
        ),
    )

    with pytest.raises(AssertionError, match="programming defect"):
        run_generation(
            b"pdf",
            "sample.pdf",
            RunType.SINGLE_PROMPT,
            settings(),
            repository=repository,
            provider_factory=lambda _run_type, ledger: ScriptedProvider(ledger, []),
        )

    assert repository.created[0].status is RunStatus.RUNNING
    assert repository.finalized == []
    assert [event[1] for event in repository.events] == ["started", "parsed"]


def test_repository_finalization_errors_are_not_swallowed(monkeypatch) -> None:
    class FailingRepository(RecordingRepository):
        def finalize(self, result) -> None:
            super().finalize(result)
            raise RuntimeError("database write failed")

    repository = FailingRepository()
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])
    monkeypatch.setitem(
        runner.PIPELINES, RunType.SINGLE_PROMPT, lambda _context, _chunks: bundle()
    )

    with pytest.raises(RuntimeError, match="database write failed"):
        run_generation(
            b"pdf",
            "sample.pdf",
            RunType.SINGLE_PROMPT,
            settings(),
            repository=repository,
            provider_factory=lambda _run_type, ledger: ScriptedProvider(ledger, []),
        )


@pytest.mark.parametrize(
    ("fail_at", "expected_calls"),
    [
        ("create_run", ["create_run"]),
        ("append_event", ["create_run", "append_event"]),
        ("save_chunks", ["create_run", "append_event", "save_chunks"]),
    ],
)
def test_repository_setup_failures_propagate_without_finalization(
    monkeypatch, fail_at, expected_calls
) -> None:
    repository = RecordingRepository(fail_at=fail_at)
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])

    with pytest.raises(RuntimeError, match=rf"{fail_at} failed"):
        run_generation(
            b"pdf",
            "sample.pdf",
            RunType.SINGLE_PROMPT,
            settings(),
            repository=repository,
        )

    assert [call[0] for call in repository.calls] == expected_calls
    assert repository.created[0].status is RunStatus.RUNNING
    assert repository.finalized == []


def test_progress_is_a_single_string_and_observer_errors_are_ignored(
    monkeypatch,
) -> None:
    trace = []

    def progress(message):
        trace.append(message)
        if message == "Generating artifacts":
            raise RuntimeError("observer broke")

    result, _repository = _successful_run(monkeypatch, progress=progress)

    assert result.manifest.status is RunStatus.COMPLETED
    assert trace == ["Preparing document", "Generating artifacts", "Completed"]
    assert all(isinstance(message, str) for message in trace)


def test_credentials_pdf_bytes_and_source_paths_are_not_persisted(monkeypatch) -> None:
    secret = "gemini-super-secret"
    pdf_marker = "PRIVATE-PDF-CONTENT"
    source_path = r"C:\TOP-SECRET-UPLOADS\spec.pdf"
    repository = RecordingRepository()
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])

    def reject_factory(_run_type, _ledger):
        raise ValueError(f"api_key={secret} token=secondary")

    result = run_generation(
        pdf_marker.encode(),
        source_path,
        RunType.SINGLE_PROMPT,
        settings(provider="gemini", api_key=secret),
        repository=repository,
        provider_factory=reject_factory,
    )
    def persisted_value(value):
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, (list, tuple)):
            return [persisted_value(item) for item in value]
        return value

    persisted_arguments = [
        repr(persisted_value(argument))
        for _method, *arguments in repository.calls
        for argument in arguments
    ]

    assert result.manifest.source_filename == "spec.pdf"
    assert persisted_arguments
    assert all(secret not in value for value in persisted_arguments)
    assert all("secondary" not in value for value in persisted_arguments)
    assert all(pdf_marker not in value for value in persisted_arguments)
    assert all("TOP-SECRET-UPLOADS" not in value for value in persisted_arguments)


def test_lm_studio_settings_create_authenticated_provider() -> None:
    configured = settings(
        provider="lm_studio",
        model="local-model",
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
def test_credential_bearing_urls_fail_before_persistence(base_url) -> None:
    repository = RecordingRepository()

    with pytest.raises(ValueError):
        run_generation(
            b"pdf",
            "sample.pdf",
            RunType.SINGLE_PROMPT,
            settings(base_url=base_url),
            repository=repository,
        )

    assert repository.calls == []


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
    with pytest.raises(ValueError):
        settings(**overrides).validate()


def test_run_ids_are_unique_with_frozen_time(monkeypatch) -> None:
    class FrozenDateTime:
        @classmethod
        def now(cls, _timezone):
            return datetime(2026, 8, 11, tzinfo=UTC)

    monkeypatch.setattr(runner, "datetime", FrozenDateTime)
    uuids = iter(
        [SimpleNamespace(hex="1" * 32), SimpleNamespace(hex="2" * 32)]
    )
    monkeypatch.setattr(runner, "uuid4", lambda: next(uuids))

    first = runner._run_id("a" * 64)
    second = runner._run_id("a" * 64)

    assert first == "20260811T000000000000Z-aaaaaaaaaaaa-11111111"
    assert second == "20260811T000000000000Z-aaaaaaaaaaaa-22222222"


def test_rerunning_the_same_document_creates_a_new_run_id(monkeypatch) -> None:
    repository = RecordingRepository()
    uuids = iter(
        [SimpleNamespace(hex="1" * 32), SimpleNamespace(hex="2" * 32)]
    )
    monkeypatch.setattr(runner, "uuid4", lambda: next(uuids))
    monkeypatch.setattr(runner, "parse_pdf", lambda _data: [chunk()])
    monkeypatch.setitem(
        runner.PIPELINES, RunType.SINGLE_PROMPT, lambda _context, _chunks: bundle()
    )
    kwargs = {
        "repository": repository,
        "provider_factory": lambda _run_type, ledger: ScriptedProvider(ledger, []),
    }

    first = run_generation(
        b"pdf", "sample.pdf", RunType.SINGLE_PROMPT, settings(), **kwargs
    )
    second = run_generation(
        b"pdf", "sample.pdf", RunType.SINGLE_PROMPT, settings(), **kwargs
    )

    assert first.manifest.run_id != second.manifest.run_id
    assert len(repository.created) == 2
