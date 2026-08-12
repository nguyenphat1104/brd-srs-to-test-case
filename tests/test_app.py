from datetime import UTC, datetime
from pathlib import Path

from streamlit.testing.v1 import AppTest

from brd_srs_testgen.models import (
    FailureCategory,
    RunManifest,
    RunResult,
    RunStatus,
    RunType,
    TestStep as ModelTestStep,
)
from tests.factories import completed_run


APP = Path(__file__).parents[1] / "app.py"


def _element(elements, label: str):
    return next(element for element in elements if element.label == label)


def _upload(at: AppTest) -> None:
    at.file_uploader[0].set_value(
        [("customer-login.pdf", b"%PDF-1.4\n", "application/pdf")]
    )


def _rendered_text(at: AppTest) -> str:
    elements = (
        list(at.markdown)
        + list(at.caption)
        + list(at.info)
        + list(at.success)
        + list(at.warning)
        + list(at.error)
        + list(at.code)
    )
    return "\n".join(
        [str(element.value) for element in elements]
        + [element.label for element in at.expander]
    )


def _failed_run(message: str = "PDF text could not be parsed.") -> RunResult:
    now = datetime.now(UTC)
    return RunResult(
        manifest=RunManifest(
            run_id="failed-run",
            source_filename="sample.pdf",
            document_hash="a" * 64,
            run_type=RunType.SINGLE_PROMPT,
            status=RunStatus.FAILED,
            provider="gemini",
            model="gemini-3.6-flash",
            temperature=0,
            token_ceiling=100_000,
            prompt_version="test",
            schema_version="test",
            started_at=now,
            completed_at=now,
            failure_category=FailureCategory.PARSING,
            failure_message=message,
        )
    )


def _interrupted_run() -> RunResult:
    return RunResult(
        manifest=RunManifest(
            run_id="interrupted-run",
            source_filename="interrupted.pdf",
            document_hash="b" * 64,
            run_type=RunType.CENTRALIZED_MULTI_AGENT,
            status=RunStatus.RUNNING,
            provider="ollama",
            model="gemma4",
            temperature=0,
            token_ceiling=75_000,
            prompt_version="test",
            schema_version="test",
            started_at=datetime.now(UTC),
        )
    )


def _semantic_failure_with_bundle() -> RunResult:
    result = completed_run()
    return result.model_copy(
        update={
            "manifest": result.manifest.model_copy(
                update={
                    "status": RunStatus.FAILED,
                    "provider": "lm_studio",
                    "failure_category": FailureCategory.SEMANTIC_VALIDATION,
                    "failure_message": "One traceability issue remains.",
                }
            )
        }
    )


def _detailed_run() -> RunResult:
    result = completed_run(run_type=RunType.STAGED_SINGLE_AGENT)
    assert result.bundle is not None
    requirement = result.bundle.requirements[0].model_copy(
        update={
            "description": "Registered customers can securely sign in.",
            "dependency_ids": ["REQ-000"],
            "ambiguities": ["Lockout threshold is unspecified."],
        }
    )
    scenario = result.bundle.scenarios[0].model_copy(
        update={"preconditions": ["The customer account is active."]}
    )
    test_case = result.bundle.test_cases[0].model_copy(
        update={
            "preconditions": ["The login page is open."],
            "test_data": {"email": "customer@example.com", "remember_me": True},
            "steps": [
                ModelTestStep(
                    step_number=1,
                    action="Enter valid credentials.",
                    expected_result="The credentials are accepted.",
                ),
                ModelTestStep(
                    step_number=2,
                    action="Submit the login form.",
                    expected_result="The account dashboard is displayed.",
                ),
            ],
        }
    )
    return result.model_copy(
        update={
            "bundle": result.bundle.model_copy(
                update={
                    "requirements": [requirement],
                    "scenarios": [scenario],
                    "test_cases": [test_case],
                }
            )
        }
    )


def test_runs_one_selected_type_and_passes_the_complete_request() -> None:
    captured = {}
    repository = object()
    expected = completed_run(run_type=RunType.STAGED_SINGLE_AGENT)

    def fake_runner(
        pdf_bytes,
        source_filename,
        run_type,
        settings,
        *,
        repository,
        progress,
    ):
        captured.update(
            pdf_bytes=pdf_bytes,
            source_filename=source_filename,
            run_type=run_type,
            settings=settings,
            repository=repository,
        )
        progress("Generating artifacts")
        return expected

    at = AppTest.from_file(APP, default_timeout=10)
    at.session_state["_runner"] = fake_runner
    at.session_state["_repository"] = repository
    at.run()

    run_type = _element(at.selectbox, "Run type")
    assert list(run_type.options) == [
        "Single prompt",
        "Staged single agent",
        "Centralized multi-agent",
    ]
    run_type.set_value("Staged single agent")
    _element(at.text_input, "Gemini API key").set_value("secret")
    _upload(at)
    _element(at.button, "Generate test cases").click()
    at.run()

    assert not at.exception
    assert captured["pdf_bytes"] == b"%PDF-1.4\n"
    assert captured["source_filename"] == "customer-login.pdf"
    assert captured["run_type"] is RunType.STAGED_SINGLE_AGENT
    assert captured["repository"] is repository
    assert captured["settings"].model == "gemini-3.6-flash"
    assert at.session_state["run_result"] == expected


def test_renders_detailed_artifact_content_and_downloads() -> None:
    at = AppTest.from_file(APP, default_timeout=10)
    at.session_state["run_result"] = _detailed_run()

    at.run()

    assert not at.exception
    text = _rendered_text(at)
    for expected in (
        "REQ-001 · Authenticate users",
        "Registered customers can securely sign in.",
        "Functional",
        "High",
        "Authentication",
        "REQ-000",
        "Lockout threshold is unspecified.",
        "SCN-001 · Valid sign in",
        "Verify successful authentication.",
        "Positive",
        "The customer account is active.",
        "TC-001 · Sign in with valid credentials",
        "P1",
        "The login page is open.",
        "customer@example.com",
        "AUTHENTICATION",
    ):
        assert expected in text
    tables = "\n".join(str(table.value) for table in at.table)
    assert "Enter valid credentials." in tables
    assert "The credentials are accepted." in tables
    assert "Submit the login form." in tables
    assert "The account dashboard is displayed." in tables
    assert {button.label for button in at.download_button} >= {
        "Download traceability matrix",
        "Download complete bundle",
    }
    assert {button.key for button in at.download_button} == {
        f"current-{at.session_state['run_result'].manifest.run_id}-rtm",
        f"current-{at.session_state['run_result'].manifest.run_id}-bundle",
    }
    for persisted_setting in (
        "Staged single agent",
        "Ollama",
        "gemma4",
        "sample.pdf",
        "Temperature 0",
        "Token ceiling 100,000",
    ):
        assert persisted_setting in text
    assert "Latency 0.10 s · 0 retries" in text
    assert _element(at.metric, "Charged tokens").value == "30"
    quality = "\n".join(str(table.value) for table in at.table)
    assert "Positive scenario coverage" in quality
    assert "Non-positive scenario coverage" in quality


def test_clears_previous_result_when_a_later_generation_raises() -> None:
    calls = 0

    def fake_runner(*args, progress, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("new run failed")
        return completed_run()

    at = AppTest.from_file(APP, default_timeout=10)
    at.session_state["_runner"] = fake_runner
    at.session_state["_repository"] = object()
    at.run()
    _element(at.text_input, "Gemini API key").set_value("key")
    _upload(at)
    _element(at.button, "Generate test cases").click()
    at.run()
    assert at.session_state["run_result"].manifest.status is RunStatus.COMPLETED

    _element(at.button, "Generate test cases").click()
    at.run()

    assert "run_result" not in at.session_state
    assert "Generation failed: new run failed" in _rendered_text(at)
    assert not at.download_button


def test_clears_previous_result_when_runner_returns_a_failure() -> None:
    at = AppTest.from_file(APP, default_timeout=10)
    at.session_state["run_result"] = completed_run()
    at.session_state["_repository"] = object()
    at.session_state["_runner"] = lambda *args, **kwargs: _failed_run()
    at.run()
    _element(at.text_input, "Gemini API key").set_value("key")
    _upload(at)
    _element(at.button, "Generate test cases").click()
    at.run()

    result = at.session_state["run_result"]
    assert result.manifest.status is RunStatus.FAILED
    assert {button.label for button in at.download_button} == {
        "Download diagnostics"
    }
    assert "PDF text could not be parsed" in _rendered_text(at)


def test_requires_repository_only_when_generation_is_started() -> None:
    at = AppTest.from_file(APP, default_timeout=10)
    at.session_state["run_result"] = completed_run()
    at.run()

    assert "TC-001" in _rendered_text(at)
    _element(at.text_input, "Gemini API key").set_value("key")
    _upload(at)
    _element(at.button, "Generate test cases").click()
    at.run()

    assert "Run history database is unavailable" in _rendered_text(at)
    assert "run_result" not in at.session_state


def test_redacts_credentials_and_base_url_from_runner_error() -> None:
    secret = "not-for-display"

    def fake_runner(pdf_bytes, source_filename, run_type, settings, **kwargs):
        raise RuntimeError(f"runner stopped: {settings.api_key} {settings.base_url}")

    at = AppTest.from_file(APP, default_timeout=10)
    at.session_state["_runner"] = fake_runner
    at.session_state["_repository"] = object()
    at.run()
    _element(at.text_input, "Gemini API key").set_value(secret)
    _upload(at)
    _element(at.button, "Generate test cases").click()
    at.run()

    errors = _rendered_text(at)
    assert "Generation failed: runner stopped" in errors
    assert secret not in errors


def test_failed_result_without_metrics_has_an_actionable_summary() -> None:
    at = AppTest.from_file(APP, default_timeout=10)
    at.session_state["run_result"] = _failed_run()

    at.run()

    assert at.status[0].label == "Generation failed"
    assert "text-extractable PDF" in _rendered_text(at)
    assert _element(at.download_button, "Download diagnostics")


def test_interrupted_result_has_diagnostics_without_a_fake_failure() -> None:
    at = AppTest.from_file(APP, default_timeout=10)
    at.session_state["run_result"] = _interrupted_run()

    at.run()

    text = _rendered_text(at)
    assert at.status[0].label == "Generation interrupted"
    assert "Unknown failure" not in text
    assert "Technical details" not in text
    assert not at.error
    assert not at.success
    diagnostics = _element(at.download_button, "Download diagnostics")
    assert diagnostics.key == "current-interrupted-run-diagnostics"


def test_failed_semantic_result_keeps_artifact_details_and_diagnostics() -> None:
    at = AppTest.from_file(APP, default_timeout=10)
    at.session_state["run_result"] = _semantic_failure_with_bundle()

    at.run()

    text = _rendered_text(at)
    assert "TC-001 · Sign in with valid credentials" in text
    assert "LM Studio" in text
    assert {button.label for button in at.download_button} == {
        "Download diagnostics"
    }


def test_prefills_provider_credentials_and_preserves_lm_studio_controls(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-from-env")
    monkeypatch.setenv("LM_STUDIO_API_TOKEN", "lm-studio-from-env")
    at = AppTest.from_file(APP, default_timeout=10)
    at.session_state["_model_loader"] = lambda *_: [
        "google/gemma-4-26b-a4b-qat"
    ]

    at.run()
    assert _element(at.text_input, "Gemini API key").value == "gemini-from-env"

    _element(at.selectbox, "Provider").set_value("lm_studio")
    at.run()
    assert _element(at.text_input, "LM Studio API token").value == "lm-studio-from-env"
    assert _element(at.selectbox, "Model").value == "google/gemma-4-26b-a4b-qat"
    assert _element(at.text_input, "LM Studio base URL").value == (
        "http://localhost:1234/v1"
    )
    assert _element(at.button, "Load available models")


def test_provider_change_resets_model_and_keeps_three_step_workflow() -> None:
    at = AppTest.from_file(APP, default_timeout=10)

    at.run()
    assert [tab.label for tab in at.tabs[:3]] == [
        "1 · Configure",
        "2 · Run",
        "3 · Results",
    ]
    assert _element(at.text_input, "Model").value == "gemini-3.6-flash"

    _element(at.selectbox, "Provider").set_value("ollama")
    at.run()

    assert _element(at.text_input, "Model").value == "gemma4"
