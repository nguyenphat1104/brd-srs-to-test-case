from datetime import UTC, datetime
from pathlib import Path

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
from tests.factories import bundle as make_bundle
from tests.factories import chunk


def test_runs_comparison_and_renders_partial_failure() -> None:
    now = datetime.now(UTC)

    def successful(condition: Condition) -> ConditionResult:
        artifacts = make_bundle()
        validation = validate_bundle(artifacts, [chunk()])
        return ConditionResult(
            manifest=ConditionManifest(
                condition=condition,
                status=RunStatus.COMPLETED,
                provider="ollama",
                model="gemma4",
                temperature=0,
                token_ceiling=100_000,
                started_at=now,
                completed_at=now,
            ),
            bundle=artifacts,
            validation=validation,
            rtm=build_rtm(artifacts),
            metrics=compute_metrics(
                artifacts,
                validation,
                input_tokens=10,
                output_tokens=20,
                charged_tokens=30,
                latency_seconds=0.1,
                retries=0,
                schema_repairs=0,
                semantic_revisions=0,
                budget_exhausted=False,
            ),
        )

    def failed(condition: Condition) -> ConditionResult:
        return ConditionResult(
            manifest=ConditionManifest(
                condition=condition,
                status=RunStatus.FAILED,
                provider="ollama",
                model="gemma4",
                temperature=0,
                token_ceiling=100_000,
                started_at=now,
                completed_at=now,
                failure_category=FailureCategory.PROVIDER_REJECTION,
                failure_message="Provider rejected the request.",
            ),
            bundle=None,
            validation=None,
            rtm=[],
            metrics=RunMetrics(
                completion=False,
                schema_valid=False,
                citation_coverage=0,
                requirement_scenario_coverage=0,
                requirement_test_case_coverage=0,
                positive_scenario_coverage=0,
                non_positive_scenario_coverage=0,
                rtm_completeness=0,
                orphan_rate=0,
                invalid_reference_rate=0,
                duplicate_test_case_rate=0,
                requirement_count=0,
                scenario_count=0,
                test_case_count=0,
                input_tokens=0,
                output_tokens=0,
                latency_seconds=0,
                retries=0,
                schema_repairs=0,
                semantic_revisions=0,
                budget_exhausted=False,
            ),
        )

    def fake_runner(pdf_bytes: bytes, settings, *, progress) -> ComparisonResult:
        progress(None, "Parsing PDF")
        return ComparisonResult(
            manifest=ComparisonManifest(
                comparison_id="comparison-id",
                document_hash="a" * 64,
                provider=settings.provider,
                model=settings.model,
                temperature=0,
                token_ceiling=settings.token_ceiling,
                condition_order=list(Condition),
                prompt_version="test",
                schema_version="test",
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

    at = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=10)
    at.session_state["_runner"] = fake_runner
    at.run()
    at.selectbox[0].set_value("ollama")
    at.run()
    at.text_input[0].set_value("gemma4")
    at.file_uploader[0].set_value(
        [("sample.pdf", b"%PDF-1.4\n", "application/pdf")]
    )
    at.button[0].click()
    at.run()

    assert not at.exception
    assert any("Provider rejected" in error.value for error in at.error)
    assert len(at.download_button) >= 2


def test_redacts_api_key_from_runner_error() -> None:
    secret = "not-for-display"

    def fake_runner(pdf_bytes: bytes, settings, *, progress) -> ComparisonResult:
        raise RuntimeError(f"runner stopped: {settings.api_key}")

    at = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=10)
    at.session_state["_runner"] = fake_runner
    at.run()
    at.text_input[1].set_value(secret)
    at.file_uploader[0].set_value(
        [("sample.pdf", b"%PDF-1.4\n", "application/pdf")]
    )
    at.button[0].click()
    at.run()

    errors = "\n".join(error.value for error in at.error)
    assert "Comparison failed: runner stopped" in errors
    assert secret not in errors


def test_clears_previous_result_when_a_later_run_fails() -> None:
    now = datetime.now(UTC)
    artifacts = make_bundle()
    validation = validate_bundle(artifacts, [chunk()])
    metrics = compute_metrics(
        artifacts,
        validation,
        input_tokens=10,
        output_tokens=20,
        latency_seconds=0.1,
        retries=0,
        schema_repairs=0,
        semantic_revisions=0,
        budget_exhausted=False,
    )
    calls = 0

    def fake_runner(pdf_bytes: bytes, settings, *, progress) -> ComparisonResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("new run failed")
        manifest = ComparisonManifest(
            comparison_id="comparison-id",
            document_hash="a" * 64,
            provider=settings.provider,
            model=settings.model,
            temperature=0,
            token_ceiling=settings.token_ceiling,
            condition_order=list(Condition),
            prompt_version="test",
            schema_version="test",
            started_at=now,
            completed_at=now,
        )
        return ComparisonResult(
            manifest=manifest,
            conditions={
                condition: ConditionResult(
                    manifest=ConditionManifest(
                        condition=condition,
                        status=RunStatus.COMPLETED,
                        provider=settings.provider,
                        model=settings.model,
                        temperature=0,
                        token_ceiling=settings.token_ceiling,
                        started_at=now,
                        completed_at=now,
                    ),
                    bundle=artifacts,
                    validation=validation,
                    rtm=build_rtm(artifacts),
                    metrics=metrics,
                )
                for condition in Condition
            },
        )

    at = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=10)
    at.session_state["_runner"] = fake_runner
    at.run()
    at.text_input[1].set_value("key")
    at.file_uploader[0].set_value(
        [("sample.pdf", b"%PDF-1.4\n", "application/pdf")]
    )
    at.button[0].click()
    at.run()
    assert at.download_button

    at.button[0].click()
    at.run()

    assert not at.download_button
    assert any("new run failed" in error.value for error in at.error)


def test_marks_top_level_failure_in_status() -> None:
    now = datetime.now(UTC)

    def fake_runner(pdf_bytes: bytes, settings, *, progress) -> ComparisonResult:
        return ComparisonResult(
            manifest=ComparisonManifest(
                comparison_id="comparison-id",
                document_hash="a" * 64,
                provider=settings.provider,
                model=settings.model,
                temperature=0,
                token_ceiling=settings.token_ceiling,
                condition_order=list(Condition),
                prompt_version="test",
                schema_version="test",
                started_at=now,
                completed_at=now,
            ),
            conditions={},
            failure_category=FailureCategory.PARSING,
            failure_message="PDF text could not be parsed.",
        )

    at = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=10)
    at.session_state["_runner"] = fake_runner
    at.run()
    at.text_input[1].set_value("key")
    at.file_uploader[0].set_value(
        [("sample.pdf", b"%PDF-1.4\n", "application/pdf")]
    )
    at.button[0].click()
    at.run()

    assert at.status[0].label == "Comparison failed"
    assert any("PDF text could not be parsed" in error.value for error in at.error)


def test_resets_model_when_provider_changes() -> None:
    at = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=10)
    at.run()
    assert at.text_input[0].value == "gemini-2.5-flash"

    at.selectbox[0].set_value("ollama")
    at.run()

    assert at.text_input[0].value == "gemma4"
