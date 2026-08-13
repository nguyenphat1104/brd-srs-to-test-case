from datetime import UTC, datetime
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from brd_srs_testgen.browser_settings import AppSettings, BrowserSettingsResult
from brd_srs_testgen.models import (
    FailureCategory,
    RunHistoryItem,
    RunManifest,
    RunResult,
    RunStatus,
    RunType,
    TestStep as ModelTestStep,
)
from brd_srs_testgen.storage import StorageError
from tests.factories import completed_run


APP = Path(__file__).parents[1] / "app.py"


class FakeRepository:
    def __init__(
        self,
        *,
        runs: list[RunHistoryItem] | None = None,
        results: dict[str, RunResult] | None = None,
        initialize_error: StorageError | None = None,
        list_error: StorageError | None = None,
        load_error: StorageError | None = None,
    ) -> None:
        self.runs = runs or []
        self.results = results or {}
        self.initialize_error = initialize_error
        self.list_error = list_error
        self.load_error = load_error
        self.initialize_calls = 0
        self.list_calls = 0
        self.load_calls: list[str] = []

    def initialize(self) -> None:
        self.initialize_calls += 1
        if self.initialize_error:
            raise self.initialize_error

    def list_runs(self) -> list[RunHistoryItem]:
        self.list_calls += 1
        if self.list_error:
            raise self.list_error
        return self.runs

    def load_run(self, run_id: str) -> RunResult:
        self.load_calls.append(run_id)
        if self.load_error:
            raise self.load_error
        try:
            return self.results[run_id]
        except KeyError as error:
            raise StorageError(f"missing secret run {run_id}") from error


class FakeBrowserSettings:
    def __init__(self, payload=None, error=None) -> None:
        self.payload = payload
        self.error = error
        self.saved: list[dict[str, object]] = []

    def __call__(self, *, save, revision) -> BrowserSettingsResult:
        if save is not None:
            self.payload = save
            self.saved.append(save)
        return BrowserSettingsResult(
            self.payload,
            self.error,
            loaded=True,
            revision=revision,
        )


def _saved_settings(**overrides) -> dict[str, object]:
    values: dict[str, object] = {
        "version": 1,
        "provider": "gemini",
        "model": "gemini-3.6-flash",
        "api_key": "browser-secret",
        "base_url": "",
        "token_ceiling": 200_000,
    }
    values.update(overrides)
    return values


def _app_test(
    repository: FakeRepository | None = None,
    browser: FakeBrowserSettings | None = None,
) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=10)
    at.session_state["_repository"] = repository or FakeRepository()
    at.session_state["_browser_settings_sync"] = browser or FakeBrowserSettings(
        _saved_settings()
    )
    return at


def _element(elements, label: str):
    return next(element for element in elements if element.label == label)


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


def _show_detail(at: AppTest, result: RunResult) -> None:
    at.session_state["view"] = "detail"
    at.session_state["selected_run_id"] = result.manifest.run_id
    at.session_state["selected_run"] = result


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


def _history_item(result: RunResult) -> RunHistoryItem:
    manifest = result.manifest
    metrics = result.metrics
    return RunHistoryItem(
        run_id=manifest.run_id,
        source_filename=manifest.source_filename,
        run_type=manifest.run_type,
        status=manifest.status,
        provider=manifest.provider,
        model=manifest.model,
        started_at=manifest.started_at,
        completed_at=manifest.completed_at,
        requirement_count=metrics.requirement_count if metrics else None,
        scenario_count=metrics.scenario_count if metrics else None,
        test_case_count=metrics.test_case_count if metrics else None,
    )


def test_runs_home_restores_settings_and_shows_native_history() -> None:
    completed = _detailed_run()
    interrupted = _interrupted_run()
    repository = FakeRepository(
        runs=[_history_item(interrupted), _history_item(completed)]
    )
    at = _app_test(repository)

    at.run()

    assert not at.exception
    assert at.session_state["view"] == "runs"
    assert at.session_state["app_settings"] == AppSettings(**_saved_settings())
    assert not at.tabs
    assert {button.label for button in at.button} >= {
        "BRD/SRS Test Case",
        "Settings",
        "Create new run",
    }
    history = at.dataframe[0].value
    assert list(history.columns) == [
        "Started",
        "Source",
        "Run type",
        "Provider",
        "Model",
        "Status",
        "Test cases",
    ]
    assert list(history["Source"]) == ["interrupted.pdf", "sample.pdf"]
    assert list(history["Status"]) == ["Interrupted", "Completed"]
    assert list(history["Test cases"]) == ["—", "1"]


def test_settings_save_is_explicit_and_activates_after_storage_confirms() -> None:
    browser = FakeBrowserSettings(_saved_settings())
    at = _app_test(browser=browser)
    at.run()

    _element(at.button, "Settings").click()
    at.run()
    _element(at.text_input, "Model").set_value("gemini-cancelled")
    at.run()
    assert browser.saved == []

    _element(at.button, "Cancel").click()
    at.run()
    assert at.session_state["app_settings"].model == "gemini-3.6-flash"

    _element(at.button, "Settings").click()
    at.run()
    _element(at.text_input, "Model").set_value("gemini-saved")
    at.run()
    _element(at.button, "Save").click()
    at.run()

    assert browser.saved[-1]["model"] == "gemini-saved"
    assert "run_type" not in browser.saved[-1]
    assert at.session_state["app_settings"].model == "gemini-saved"


def test_lm_studio_settings_retain_current_controls(monkeypatch) -> None:
    monkeypatch.setenv("LM_STUDIO_API_TOKEN", "lm-studio-from-env")
    monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://lm-studio:1234/v1")
    at = _app_test()
    at.session_state["_model_loader"] = lambda *_: [
        "google/gemma-4-26b-a4b-qat",
        "qwen/qwen3-4b",
    ]
    at.run()

    _element(at.button, "Settings").click()
    at.run()
    _element(at.selectbox, "Provider").set_value("lm_studio")
    at.run()

    assert _element(at.text_input, "LM Studio API token").value == (
        "lm-studio-from-env"
    )
    assert _element(at.text_input, "LM Studio base URL").value == (
        "http://lm-studio:1234/v1"
    )
    _element(at.button, "Load available models").click()
    at.run()
    assert _element(at.selectbox, "Model").value == "google/gemma-4-26b-a4b-qat"

    _element(at.selectbox, "Model").set_value("qwen/qwen3-4b")
    at.run()
    assert _element(at.selectbox, "Model").value == "qwen/qwen3-4b"


def test_lm_studio_model_error_redacts_token_and_base_url(monkeypatch) -> None:
    token = "lm-secret-token"
    base_url = "http://secret-lm-studio:1234/v1"
    monkeypatch.setenv("LM_STUDIO_API_TOKEN", token)
    monkeypatch.setenv("LM_STUDIO_BASE_URL", base_url)
    at = _app_test()

    def fail_loader(*_):
        raise RuntimeError(f"connection failed: {token} {base_url}")

    at.session_state["_model_loader"] = fail_loader
    at.run()
    _element(at.button, "Settings").click()
    at.run()
    _element(at.selectbox, "Provider").set_value("lm_studio")
    at.run()
    _element(at.button, "Load available models").click()
    at.run()

    text = _rendered_text(at)
    assert "Could not load models: connection failed" in text
    assert token not in text
    assert base_url not in text


def test_missing_settings_can_be_saved_before_create() -> None:
    result = _detailed_run()
    browser = FakeBrowserSettings()
    at = _app_test(
        FakeRepository(runs=[_history_item(result)]),
        browser,
    )
    at.run()

    assert at.dataframe
    _element(at.button, "Create new run").click()
    at.run()
    _element(at.text_input, "Gemini API key").set_value("new-secret")
    at.run()
    _element(at.button, "Save").click()
    at.run()

    assert at.session_state["view"] == "create"
    assert _element(at.file_uploader, "BRD/SRS PDF")
    assert _element(at.selectbox, "Run type")


def test_empty_runs_still_offers_create() -> None:
    at = _app_test()

    at.run()

    assert "No saved runs yet." in _rendered_text(at)
    assert _element(at.button, "Create new run")
    assert not at.dataframe


def test_selecting_a_run_loads_detail_once_and_back_returns_home() -> None:
    result = _detailed_run()
    repository = FakeRepository(
        runs=[_history_item(result)],
        results={result.manifest.run_id: result},
    )
    at = _app_test(repository)
    at.run()

    at.session_state["runs-table"] = {"selection": {"rows": [0]}}
    at.run()

    assert repository.load_calls == [result.manifest.run_id]
    assert at.session_state["view"] == "detail"
    assert "TC-001 · Sign in with valid credentials" in _rendered_text(at)
    _element(at.button, "Back to runs").click()
    at.run()
    assert at.session_state["view"] == "runs"


def test_browser_storage_error_warns_without_blocking_history() -> None:
    result = _detailed_run()
    at = _app_test(
        FakeRepository(runs=[_history_item(result)]),
        FakeBrowserSettings(error="secret browser failure"),
    )

    at.run()

    assert at.warning
    assert "Browser settings storage is unavailable" in _rendered_text(at)
    assert "secret" not in _rendered_text(at)
    assert at.dataframe
    assert _element(at.button, "Create new run")


def test_list_error_is_safe_actionable_and_create_remains_available() -> None:
    repository = FakeRepository(
        list_error=StorageError("postgresql://user:list-secret@localhost/database")
    )
    at = _app_test(repository)

    at.run()

    text = _rendered_text(at)
    assert not at.exception
    assert "Saved run history is unavailable" in text
    assert "DATABASE_URL" in text
    assert "list-secret" not in text
    assert _element(at.button, "Create new run")


def test_missing_selected_run_returns_home_with_safe_error() -> None:
    result = _detailed_run()
    repository = FakeRepository(
        runs=[_history_item(result)],
        load_error=StorageError("postgresql://user:load-secret@localhost/database"),
    )
    at = _app_test(repository)
    at.session_state["view"] = "detail"
    at.session_state["selected_run_id"] = result.manifest.run_id

    at.run()

    text = _rendered_text(at)
    assert not at.exception
    assert repository.load_calls == [result.manifest.run_id]
    assert at.session_state["view"] == "runs"
    assert "Saved run could not be opened" in text
    assert "DATABASE_URL" in text
    assert "load-secret" not in text
    assert _element(at.button, "Create new run")


def test_database_initialization_failure_remains_blocking() -> None:
    repository = FakeRepository(
        initialize_error=StorageError(
            "postgresql://user:secret@localhost/database could not connect"
        )
    )
    at = _app_test(repository)

    at.run()

    text = _rendered_text(at)
    assert "Run history database is unavailable" in text
    assert "docker compose up -d db" in text
    assert "DATABASE_URL" in text
    assert "secret" not in text
    assert not any(button.label == "Create new run" for button in at.button)
    assert not at.dataframe


def test_renders_detailed_artifact_content_and_downloads() -> None:
    result = _detailed_run()
    at = _app_test()
    _show_detail(at, result)

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
    assert {button.key for button in at.download_button} == {
        f"detail-{result.manifest.run_id}-rtm",
        f"detail-{result.manifest.run_id}-bundle",
    }
    assert _element(at.metric, "Charged tokens").value == "30"
    assert "Positive scenario coverage" in tables
    assert "Non-positive scenario coverage" in tables


def test_failed_result_without_metrics_has_an_actionable_summary() -> None:
    result = _failed_run()
    at = _app_test()
    _show_detail(at, result)

    at.run()

    assert at.status[0].label == "Generation failed"
    assert "text-extractable PDF" in _rendered_text(at)
    assert _element(at.download_button, "Download diagnostics")


def test_interrupted_result_has_diagnostics_without_a_fake_failure() -> None:
    result = _interrupted_run()
    at = _app_test()
    _show_detail(at, result)

    at.run()

    text = _rendered_text(at)
    assert at.status[0].label == "Generation interrupted"
    assert "Unknown failure" not in text
    assert "Technical details" not in text
    assert not at.error
    assert not at.success
    assert _element(at.download_button, "Download diagnostics").key == (
        "detail-interrupted-run-diagnostics"
    )


def test_failed_semantic_result_keeps_artifact_details_and_diagnostics() -> None:
    result = _semantic_failure_with_bundle()
    at = _app_test()
    _show_detail(at, result)

    at.run()

    text = _rendered_text(at)
    assert "TC-001 · Sign in with valid credentials" in text
    assert "LM Studio" in text
    assert {button.label for button in at.download_button} == {
        "Download diagnostics"
    }


@pytest.mark.skip(reason="Create flow completed in Task 3")
def test_runs_one_selected_type_and_passes_the_complete_request() -> None:
    pass


@pytest.mark.skip(reason="Create flow completed in Task 3")
def test_clears_previous_result_when_a_later_generation_raises() -> None:
    pass


@pytest.mark.skip(reason="Create flow completed in Task 3")
def test_clears_previous_result_when_runner_returns_a_failure() -> None:
    pass


@pytest.mark.skip(reason="Create flow completed in Task 3")
def test_redacts_credentials_and_base_url_from_runner_error() -> None:
    pass
