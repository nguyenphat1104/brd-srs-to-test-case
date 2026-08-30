from datetime import UTC, datetime
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from brd_srs_testgen.browser_settings import AppSettings, BrowserSettingsResult
from brd_srs_testgen.models import (
    AgentSetup,
    ActivityEvent,
    FailureCategory,
    RequirementBatch,
    RunHistoryItem,
    RunManifest,
    RunResult,
    RunStatus,
    RunType,
    TestStep as ModelTestStep,
    default_agent_setups,
)
from brd_srs_testgen.storage import StorageError
from tests.factories import completed_run


APP = Path(__file__).parents[1] / "app.py"


class FakeRepository:
    def __init__(
        self,
        *,
        runs: list[RunHistoryItem] | None = None,
        run_batches: list[list[RunHistoryItem]] | None = None,
        results: dict[str, RunResult] | None = None,
        initialize_error: StorageError | None = None,
        list_error: StorageError | None = None,
        load_error: StorageError | None = None,
    ) -> None:
        self.runs = runs or []
        self.run_batches = run_batches
        self.results = results or {}
        self.initialize_error = initialize_error
        self.list_error = list_error
        self.load_error = load_error
        self.initialize_calls = 0
        self.list_calls = 0
        self.load_calls: list[str] = []
        self.agent_setups = default_agent_setups()

    def initialize(self) -> None:
        self.initialize_calls += 1
        if self.initialize_error:
            raise self.initialize_error

    def list_runs(self) -> list[RunHistoryItem]:
        self.list_calls += 1
        if self.list_error:
            raise self.list_error
        if self.run_batches:
            return self.run_batches[
                min(self.list_calls - 1, len(self.run_batches) - 1)
            ]
        return self.runs

    def load_run(self, run_id: str) -> RunResult:
        self.load_calls.append(run_id)
        if self.load_error:
            raise self.load_error
        try:
            return self.results[run_id]
        except KeyError as error:
            raise StorageError(f"missing secret run {run_id}") from error

    def load_agent_setups(self) -> dict[str, AgentSetup]:
        return self.agent_setups.copy()

    def save_agent_setups(self, setups) -> None:
        self.agent_setups = {setup.agent: setup for setup in setups}


class FakeBrowserSettings:
    def __init__(
        self,
        payload=None,
        error=None,
        *,
        confirm_saves: bool = True,
        loaded: bool = True,
    ) -> None:
        self.payload = payload
        self.error = error
        self.confirm_saves = confirm_saves
        self.loaded = loaded
        self.pending = None
        self.saved: list[dict[str, object]] = []

    def __call__(self, *, save, revision) -> BrowserSettingsResult:
        if save is not None:
            if save != self.pending:
                self.saved.append(save)
            self.pending = save
            if not self.confirm_saves:
                return BrowserSettingsResult(
                    self.payload,
                    self.error,
                    loaded=True,
                    revision=revision - 1,
                )
            self.payload = self.pending
            self.pending = None
        return BrowserSettingsResult(
            self.payload,
            self.error,
            loaded=self.loaded,
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


def _element_starting(elements, prefix: str):
    return next(element for element in elements if element.label.startswith(prefix))


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


def test_runs_home_restores_settings_and_shows_run_item_list() -> None:
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
        "Settings",
        "Create new run",
    }
    text = _rendered_text(at)
    assert not at.dataframe
    assert {button.label for button in at.button} >= {
        "Open interrupted.pdf",
        "Open sample.pdf",
    }
    for expected in (
        "interrupted.pdf",
        "sample.pdf",
        "Centralized multi-agent",
        "Interrupted",
        "Completed",
        "Not recorded",
        "1 generated",
    ):
        assert expected in text


def test_settings_save_is_explicit_and_activates_after_storage_confirms() -> None:
    browser = FakeBrowserSettings(_saved_settings())
    at = _app_test(browser=browser)
    at.run()

    _element(at.button, "Settings").click()
    at.run()
    assert (
        "Scripts running on the same app origin can read stored credentials."
        in _rendered_text(at)
    )
    _element(at.text_input, "Model").set_value("gemini-cancelled")
    at.run()
    assert browser.saved == []

    _element(at.button, "Cancel").click()
    at.run()
    assert at.session_state["app_settings"].model == "gemini-3.6-flash"

    _element(at.button, "Settings").click()
    at.run()
    _element(at.text_input, "Model").set_value("gemini-saved")
    _element(at.text_input, "Analyst model (optional)").set_value("analyst-model")
    _element(at.text_input, "Test generator model (optional)").set_value(
        "generator-model"
    )
    _element(at.text_input, "Reviewer model (optional)").set_value("reviewer-model")
    at.run()
    _element(at.button, "Save settings").click()
    at.run()

    assert browser.saved[-1]["model"] == "gemini-saved"
    assert browser.saved[-1]["analyst_model"] == "analyst-model"
    assert browser.saved[-1]["test_generator_model"] == "generator-model"
    assert browser.saved[-1]["reviewer_model"] == "reviewer-model"
    assert "run_type" not in browser.saved[-1]
    assert at.session_state["app_settings"].model == "gemini-saved"


def test_settings_wait_for_matching_browser_confirmation() -> None:
    browser = FakeBrowserSettings(_saved_settings(), confirm_saves=False)
    at = _app_test(browser=browser)
    at.run()
    _element(at.button, "Settings").click()
    at.run()
    _element(at.text_input, "Model").set_value("gemini-3.6-pro")
    at.run()
    _element(at.button, "Save settings").click()
    at.run()

    assert at.session_state["app_settings"].model == "gemini-3.6-flash"
    assert at.session_state["settings_save_request"]["model"] == "gemini-3.6-pro"
    _element(at.button, "Create new run").click()
    at.run()

    assert at.session_state["view"] == "runs"
    assert "Saving browser settings…" in _rendered_text(at)

    browser.confirm_saves = True
    at.run()

    assert at.session_state["app_settings"].model == "gemini-3.6-pro"
    assert "settings_save_request" not in at.session_state
    _element(at.button, "Create new run").click()
    at.run()
    assert at.session_state["view"] == "create"


def test_create_waits_for_initial_browser_settings_load() -> None:
    browser = FakeBrowserSettings(_saved_settings(), loaded=False)
    at = _app_test(browser=browser)
    at.run()

    _element(at.button, "Create new run").click()
    at.run()

    assert at.session_state["view"] == "runs"
    assert "Browser settings are still loading." in _rendered_text(at)

    browser.loaded = True
    at.run()
    _element(at.button, "Create new run").click()
    at.run()
    assert at.session_state["view"] == "create"


def test_create_explains_why_settings_open() -> None:
    at = _app_test(browser=FakeBrowserSettings(_saved_settings(api_key="")))
    at.run()

    _element(at.button, "Create new run").click()
    at.run()

    assert at.session_state["settings_after_persist"] == "create"
    assert (
        "Gemini API key is required. Add it in Settings before creating a run."
        in _rendered_text(at)
    )


def test_ollama_settings_use_local_defaults() -> None:
    at = _app_test()
    at.run()
    _element(at.button, "Settings").click()
    at.run()

    _element(at.selectbox, "Provider").set_value("ollama")
    at.run()

    assert _element(at.text_input, "Model").value == "gemma4"
    assert _element(at.text_input, "Ollama base URL").value == (
        "http://localhost:11434"
    )


def test_lm_studio_settings_load_and_assign_models_automatically(monkeypatch) -> None:
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
    assert _element(at.selectbox, "Model").value == "google/gemma-4-26b-a4b-qat"
    assert _element(at.selectbox, "Analyst model").value == (
        "google/gemma-4-26b-a4b-qat"
    )
    assert _element(at.selectbox, "Test generator model").value == "qwen/qwen3-4b"
    assert _element(at.selectbox, "Reviewer model").value == "google/gemma-4-26b-a4b-qat"
    assert "Load available models" not in {button.label for button in at.button}

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

    assert _element_starting(at.button, "Open sample.pdf")
    _element(at.button, "Create new run").click()
    at.run()
    _element(at.text_input, "Gemini API key").set_value("new-secret")
    at.run()
    _element(at.button, "Save settings").click()
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

    _element_starting(at.button, f"Open {result.manifest.source_filename}").click()
    at.run()

    assert repository.load_calls == [result.manifest.run_id]
    assert at.session_state["view"] == "detail"
    assert _element(at.button, "Open test case TC-001 detail")
    _element(at.button, "Back to runs").click()
    at.run()
    assert at.session_state["view"] == "runs"


def test_clicking_a_test_case_item_opens_detail_modal_once() -> None:
    result = _detailed_run()
    repository = FakeRepository(
        runs=[_history_item(result)],
        results={result.manifest.run_id: result},
    )
    at = _app_test(repository)
    at.run()

    _element_starting(at.button, f"Open {result.manifest.source_filename}").click()
    at.run()

    _element(at.button, "Open test case TC-001 detail").click()
    at.run()

    text = _rendered_text(at)
    assert "TC-001 · Sign in with valid credentials" in text
    assert "The login page is open." in text
    assert "customer@example.com" in text
    tables = "\n".join(str(table.value) for table in at.table)
    assert "Enter valid credentials." in tables
    assert "The account dashboard is displayed." in tables

    at.run()
    assert "TC-001 · Sign in with valid credentials" not in _rendered_text(at)


def test_clicking_requirement_and_scenario_items_opens_detail_modals() -> None:
    result = _detailed_run()
    at = _app_test()
    _show_detail(at, result)

    at.run()

    _element(at.button, "Open requirement REQ-001 detail").click()
    at.run()
    requirement_text = _rendered_text(at)
    assert "REQ-001 · Authenticate users" in requirement_text
    assert "Registered customers can securely sign in." in requirement_text
    assert "Lockout threshold is unspecified." in requirement_text

    at.run()
    _element(at.button, "Open scenario SCN-001 detail").click()
    at.run()
    scenario_text = _rendered_text(at)
    assert "SCN-001 · Valid sign in" in scenario_text
    assert "Verify successful authentication." in scenario_text
    assert "The customer account is active." in scenario_text


def test_selecting_a_run_uses_the_displayed_row_snapshot() -> None:
    intended = completed_run(
        run_id="20260812T120000000000Z-ecac9f035813-aaaaaaaa"
    )
    newer = completed_run(
        run_id="20260812T120000000000Z-ecac9f035813-cccccccc"
    )
    repository = FakeRepository(
        run_batches=[
            [_history_item(intended)],
            [_history_item(newer), _history_item(intended)],
        ],
        results={
            intended.manifest.run_id: intended,
            newer.manifest.run_id: newer,
        },
    )
    at = _app_test(repository)
    at.run()

    _element_starting(at.button, f"Open {intended.manifest.source_filename}").click()
    at.run()

    assert repository.load_calls == [intended.manifest.run_id]
    assert at.session_state["selected_run_id"] == intended.manifest.run_id


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
    assert _element_starting(at.button, "Open sample.pdf")
    assert _element(at.button, "Create new run")


def test_list_error_is_safe_actionable_and_create_remains_available() -> None:
    repository = FakeRepository(
        list_error=StorageError("postgresql://user:list-secret@localhost/database")
    )
    at = _app_test(repository)

    at.run()

    text = _rendered_text(at)
    assert not at.exception
    assert "Saved runs are unavailable. Check PostgreSQL and DATABASE_URL, then refresh this page." in text
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
    assert "Runs database is unavailable. Start it with" in text
    assert "docker compose up -d db" in text
    assert "DATABASE_URL" in text
    assert "secret" not in text
    assert not any(button.label == "Create new run" for button in at.button)
    assert not at.dataframe


def test_completed_detail_groups_artifacts_and_configuration() -> None:
    result = _detailed_run()
    secret = "browser-only-secret"
    base_url = "http://browser-only.invalid:43114"
    at = _app_test(
        browser=FakeBrowserSettings(
            _saved_settings(api_key=secret, base_url=base_url)
        )
    )
    _show_detail(at, result)

    at.run()

    assert not at.exception
    text = _rendered_text(at)
    headings = [element.value for element in at.markdown]
    assert headings.index("#### Test cases") < headings.index("#### Requirements")
    assert headings.index("#### Requirements") < headings.index("#### Scenarios")
    assert [tab.label for tab in at.tabs] == [
        "Test cases (1)",
        "Requirements (1)",
        "Scenarios (1)",
    ]
    assert "### Quality and traceability" in headings
    assert not any(expander.label == "Quality and traceability" for expander in at.expander)
    assert _element(at.expander, "Run configuration")
    for expected in (
        "Staged single agent",
        "Ollama",
        "gemma4",
    ):
        assert expected in text
    assert _element(at.button, "Open test case TC-001 detail")
    assert _element(at.button, "Open requirement REQ-001 detail")
    assert _element(at.button, "Open scenario SCN-001 detail")
    snapshot = next(
        table.value
        for table in at.table
        if list(table.value.columns) == ["Setting", "Value"]
    )
    assert snapshot.to_dict("records") == [
        {"Setting": "Run ID", "Value": result.manifest.run_id},
        {"Setting": "Run type", "Value": "Staged single agent"},
        {"Setting": "Provider", "Value": "Ollama"},
        {"Setting": "Model", "Value": "gemma4"},
        {"Setting": "Temperature", "Value": "0"},
        {"Setting": "Token ceiling", "Value": "100,000"},
        {"Setting": "Source filename", "Value": "sample.pdf"},
        {"Setting": "Document hash", "Value": "a" * 64},
        {"Setting": "Prompt version", "Value": "research-core-v1"},
        {"Setting": "Schema version", "Value": "research-core-v1"},
        {"Setting": "Status", "Value": "completed"},
        {"Setting": "Started", "Value": result.manifest.started_at.isoformat()},
        {"Setting": "Completed", "Value": result.manifest.completed_at.isoformat()},
    ]
    tables = "\n".join(str(table.value) for table in at.table)
    assert secret not in text
    assert base_url not in text
    assert secret not in tables
    assert base_url not in tables
    assert {button.key for button in at.download_button} == {
        f"detail-{result.manifest.run_id}-rtm",
        f"detail-{result.manifest.run_id}-bundle",
    }
    assert _element(at.metric, "Charged tokens").value == "30"
    assert "Latency 0.10 s · 0 retries" in text
    assert "Citation coverage" in text
    assert "Positive scenario coverage" in text
    assert "Non-positive scenario coverage" in text


def test_quality_chart_surfaces_the_lowest_coverage_gap() -> None:
    result = _detailed_run()
    assert result.metrics is not None
    result = result.model_copy(
        update={
            "metrics": result.metrics.model_copy(
                update={
                    "citation_coverage": 1,
                    "requirement_scenario_coverage": 0.17,
                    "requirement_test_case_coverage": 0.17,
                    "positive_scenario_coverage": 0,
                    "non_positive_scenario_coverage": 0.17,
                    "rtm_completeness": 0.17,
                }
            )
        }
    )
    at = _app_test()
    _show_detail(at, result)

    at.run()

    text = _rendered_text(at)
    assert "Priority: Positive scenario coverage is 0%. Target: 100%." in text
    assert text.index("Positive scenario coverage") < text.index(
        "Requirement → scenario"
    )
    assert "quality-chart__bar--critical" in text
    assert "quality-chart__bar--partial" in text
    assert "quality-chart__bar--complete" in text


def test_failed_result_without_metrics_has_an_actionable_summary() -> None:
    result = _failed_run()
    at = _app_test()
    _show_detail(at, result)

    at.run()

    text = _rendered_text(at)
    assert "text-extractable PDF" in text
    assert at.error
    assert "Diagnostics and next steps" in text
    assert "Run configuration" in text
    snapshot = next(
        table.value
        for table in at.table
        if list(table.value.columns) == ["Setting", "Value"]
    )
    assert dict(zip(snapshot["Setting"], snapshot["Value"], strict=True))[
        "Status"
    ] == "failed"
    assert _element(at.download_button, "Download diagnostics")


def test_interrupted_result_has_diagnostics_without_a_fake_failure() -> None:
    result = _interrupted_run()
    at = _app_test()
    _show_detail(at, result)

    at.run()

    text = _rendered_text(at)
    assert "Generation was interrupted. Review diagnostics before retrying." in text
    assert "Unknown failure" not in text
    assert "Technical details" not in text
    assert "Diagnostics and next steps" in text
    assert "Run configuration" in text
    assert not at.error
    assert not at.success
    assert at.warning
    snapshot = next(
        table.value
        for table in at.table
        if list(table.value.columns) == ["Setting", "Value"]
    )
    snapshot_values = dict(
        zip(snapshot["Setting"], snapshot["Value"], strict=True)
    )
    assert snapshot_values["Status"] == "running"
    assert snapshot_values["Completed"] == "—"
    assert _element(at.download_button, "Download diagnostics").key == (
        "detail-interrupted-run-diagnostics"
    )


def test_failed_semantic_result_keeps_artifact_details_and_diagnostics() -> None:
    result = _semantic_failure_with_bundle()
    at = _app_test()
    _show_detail(at, result)

    at.run()

    text = _rendered_text(at)
    assert _element(at.button, "Open test case TC-001 detail")
    assert "LM Studio" in text
    assert "Run configuration" in text
    assert [element.value for element in at.markdown].index(
        "#### Test cases"
    ) < [element.value for element in at.markdown].index("#### Requirements")
    assert {button.label for button in at.download_button} == {
        "Download diagnostics"
    }


def test_create_runs_one_selected_type_and_opens_returned_detail() -> None:
    repository = FakeRepository()
    result = _detailed_run()
    calls = []

    def fake_runner(
        pdf_bytes,
        source_filename,
        run_type,
        provider_settings,
        *,
        repository,
        progress,
    ):
        progress(
            ActivityEvent(
                "Analyzer 1: done — handed requirements to the orchestrator.",
                agent="Analyzer 1",
                role="Requirement analyst",
                model="analyst-model",
                state="complete",
                artifact=RequirementBatch(requirements=[]),
                artifact_label="Candidate requirements",
            )
        )
        calls.append(
            (
                pdf_bytes,
                source_filename,
                run_type,
                provider_settings,
                repository,
            )
        )
        return result

    at = _app_test(repository)
    at.session_state["_runner"] = fake_runner
    at.run()
    _element(at.button, "Create new run").click()
    at.run()
    _element(at.selectbox, "Run type").set_value(
        RunType.STAGED_SINGLE_AGENT
    )
    _element(at.file_uploader, "BRD/SRS PDF").set_value(
        [("customer-login.pdf", b"%PDF-1.4\n", "application/pdf")]
    )
    _element(at.button, "Generate test cases").click()
    at.run()

    assert len(calls) == 1
    pdf_bytes, filename, run_type, settings, passed_repository = calls[0]
    assert pdf_bytes == b"%PDF-1.4\n"
    assert filename == "customer-login.pdf"
    assert run_type is RunType.STAGED_SINGLE_AGENT
    assert settings.model == "gemini-3.6-flash"
    assert settings.api_key == "browser-secret"
    assert passed_repository is repository
    assert repository.load_calls == []
    assert at.session_state["view"] == "detail"
    assert at.session_state["selected_run_id"] == result.manifest.run_id
    assert at.session_state["selected_run"] == result
    assert not at.exception
    assert _element(at.button, "Open test case TC-001 detail")


def test_centralized_run_shows_plan_and_agent_activity() -> None:
    detailed = _detailed_run()
    result = detailed.model_copy(
        update={
            "manifest": detailed.manifest.model_copy(
                update={"run_type": RunType.CENTRALIZED_MULTI_AGENT}
            )
        }
    )

    def fake_runner(*args, progress, **kwargs):
        progress(
            ActivityEvent(
                "Analyzer 1: working — extracting requirements.",
                agent="Analyzer 1",
                role="Requirement analyst",
                model="analyst-model",
                state="working",
                task="Extract testable business rules with source references.",
                scope="1 assigned source chunk · pages 1",
                deliverable="Candidate requirements for reviewer reconciliation.",
                artifact=RequirementBatch(requirements=[]),
                artifact_label="Candidate requirements",
            )
        )
        return result

    at = _app_test()
    at.session_state["_runner"] = fake_runner
    at.run()
    _element(at.button, "Create new run").click()
    at.run()
    _element(at.selectbox, "Run type").set_value(
        RunType.CENTRALIZED_MULTI_AGENT
    )
    _element(at.file_uploader, "BRD/SRS PDF").set_value(
        [("customer-login.pdf", b"%PDF-1.4\n", "application/pdf")]
    )
    _element(at.button, "Generate test cases").click()
    at.run()

    text = _rendered_text(at)
    assert "Agent activity" in text
    assert "Plan complete" in text
    assert "Prepare document" in text
    assert "Analyzer 1" in text
    assert "Model: analyst-model" in text
    assert "Extract testable business rules with source references." in text
    assert "1 assigned source chunk · pages 1" in text
    assert "Candidate requirements" in text
    assert "Artifact panel" not in text
    assert at.session_state["timeline_activity"][0].artifact_label == (
        "Candidate requirements"
    )
    assert "TC-001 · Sign in with valid credentials" not in text
    _element(at.button, "View result").click()
    at.run()
    assert _element(at.button, "Open test case TC-001 detail")
    assert at.session_state["view"] == "create"
    assert at.session_state["timeline_result"] == result


def test_centralized_returned_failure_stops_the_plan_and_keeps_diagnostics() -> None:
    failed = _failed_run()
    result = failed.model_copy(
        update={
            "manifest": failed.manifest.model_copy(
                update={"run_type": RunType.CENTRALIZED_MULTI_AGENT}
            )
        }
    )
    at = _app_test()
    at.session_state["_runner"] = lambda *args, **kwargs: result
    at.run()
    _element(at.button, "Create new run").click()
    at.run()
    _element(at.selectbox, "Run type").set_value(
        RunType.CENTRALIZED_MULTI_AGENT
    )
    _element(at.file_uploader, "BRD/SRS PDF").set_value(
        [("customer-login.pdf", b"%PDF-1.4\n", "application/pdf")]
    )
    _element(at.button, "Generate test cases").click()
    at.run()

    text = _rendered_text(at)
    assert "Generation stopped" in text
    assert "Diagnostics available" in text
    assert "Stopped at step 1 of 5" in text
    assert "Review complete; the saved artifacts are ready to inspect." not in text
    assert _element(at.button, "View diagnostics")
    assert at.session_state["timeline_current_step"] == 0

    at.run()
    assert "Generation stopped" in _rendered_text(at)


def test_returned_failed_generation_opens_detail_with_diagnostics_only(
    monkeypatch,
) -> None:
    secret = "runner-secret"
    base_url = "http://private-lm-studio:1234/v1"
    result = _failed_run("runner stopped: [REDACTED] at [REDACTED]")
    downloads = {}
    download_button = st.download_button

    def capture_download(*args, **kwargs):
        downloads[args[0]] = kwargs["data"]
        return download_button(*args, **kwargs)

    monkeypatch.setattr(st, "download_button", capture_download)
    at = _app_test()
    at.session_state["_runner"] = lambda *args, **kwargs: result
    at.run()
    _element(at.button, "Create new run").click()
    at.run()
    _element(at.file_uploader, "BRD/SRS PDF").set_value(
        [("customer-login.pdf", b"%PDF-1.4\n", "application/pdf")]
    )
    _element(at.button, "Generate test cases").click()
    at.run()

    assert at.session_state["view"] == "detail"
    assert at.session_state["selected_run_id"] == result.manifest.run_id
    assert at.session_state["selected_run"] == result
    assert at.error
    assert {button.label for button in at.download_button} == {
        "Download diagnostics"
    }
    text = _rendered_text(at)
    diagnostics = downloads["Download diagnostics"]
    assert "[REDACTED]" in text
    assert "[REDACTED]" in diagnostics
    assert secret not in text
    assert base_url not in text
    assert secret not in diagnostics
    assert base_url not in diagnostics


def test_unexpected_generation_error_stays_in_create_and_redacts_settings() -> None:
    token = "runner-secret"
    base_url = "http://private-lm-studio:1234/v1"
    calls = 0

    def fail_runner(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError(f"runner stopped: {token} {base_url}")

    at = _app_test(
        browser=FakeBrowserSettings(
            _saved_settings(
                provider="lm_studio",
                model="gemma-4",
                api_key=token,
                base_url=base_url,
            )
        )
    )
    at.session_state["_runner"] = fail_runner
    at.run()
    _element(at.button, "Create new run").click()
    at.run()
    _element(at.selectbox, "Run type").set_value(
        RunType.CENTRALIZED_MULTI_AGENT
    )
    _element(at.file_uploader, "BRD/SRS PDF").set_value(
        [("customer-login.pdf", b"%PDF-1.4\n", "application/pdf")]
    )
    _element(at.button, "Generate test cases").click()
    at.run()

    text = _rendered_text(at)
    assert calls == 1
    assert at.session_state["view"] == "create"
    assert "Generation failed: runner stopped" in text
    assert token not in text
    assert base_url not in text
    assert not at.download_button
    assert "selected_run" not in at.session_state
    assert _element(at.file_uploader, "BRD/SRS PDF").value.name == (
        "customer-login.pdf"
    )
    assert _element(at.selectbox, "Run type").value is (
        RunType.CENTRALIZED_MULTI_AGENT
    )


def test_edit_settings_from_create_preserves_upload_and_run_type() -> None:
    at = _app_test()
    at.run()
    _element(at.button, "Create new run").click()
    at.run()
    _element(at.selectbox, "Run type").set_value(
        RunType.STAGED_SINGLE_AGENT
    )
    _element(at.file_uploader, "BRD/SRS PDF").set_value(
        [("customer-login.pdf", b"%PDF-1.4\n", "application/pdf")]
    )
    at.run()
    _element(at.button, "Edit settings").click()
    at.run()
    _element(at.text_input, "Model").set_value("gemini-3.6-pro")
    at.run()
    _element(at.button, "Save settings").click()
    at.run()

    assert at.session_state["view"] == "create"
    assert "Gemini · gemini-3.6-pro · 200,000 token ceiling" in _rendered_text(at)
    assert _element(at.file_uploader, "BRD/SRS PDF").value.name == (
        "customer-login.pdf"
    )
    assert _element(at.selectbox, "Run type").value is RunType.STAGED_SINGLE_AGENT


def test_settings_save_shared_agent_setup_in_postgres_repository() -> None:
    repository = FakeRepository()
    at = _app_test(repository)
    at.run()

    _element(at.button, "Settings").click()
    at.run()
    assert any(
        expander.label.startswith("Analyst · Requirement analyst")
        for expander in at.expander
    )
    _element(at.text_input, "Analyst role").set_value(
        "Payments requirement specialist"
    )
    _element(at.text_area, "Analyst additional instructions").set_value(
        "Prioritize validation and exception rules."
    )
    _element(at.button, "Save settings").click()
    at.run()

    analyst = repository.agent_setups["analyst"]
    assert analyst.role == "Payments requirement specialist"
    assert analyst.instructions == "Prioritize validation and exception rules."


def test_create_waits_for_settings_save_without_losing_inputs() -> None:
    browser = FakeBrowserSettings(_saved_settings())
    calls = 0

    def fake_runner(*args, **kwargs):
        nonlocal calls
        calls += 1

    at = _app_test(browser=browser)
    at.session_state["_runner"] = fake_runner
    at.run()
    _element(at.button, "Create new run").click()
    at.run()
    _element(at.selectbox, "Run type").set_value(
        RunType.CENTRALIZED_MULTI_AGENT
    )
    _element(at.file_uploader, "BRD/SRS PDF").set_value(
        [("customer-login.pdf", b"%PDF-1.4\n", "application/pdf")]
    )
    at.run()
    _element(at.button, "Edit settings").click()
    at.run()
    _element(at.text_input, "Model").set_value("gemini-3.6-pro")
    at.run()
    browser.confirm_saves = False
    _element(at.button, "Save settings").click()
    at.run()

    assert at.session_state["view"] == "create"
    assert "Saving browser settings…" in _rendered_text(at)
    assert not any(button.label == "Generate test cases" for button in at.button)
    assert calls == 0
    assert _element(at.file_uploader, "BRD/SRS PDF").value.name == (
        "customer-login.pdf"
    )
    assert _element(at.selectbox, "Run type").value is (
        RunType.CENTRALIZED_MULTI_AGENT
    )

    browser.confirm_saves = True
    at.run()

    assert "Gemini · gemini-3.6-pro · 200,000 token ceiling" in _rendered_text(at)
    assert _element(at.button, "Generate test cases")
    assert _element(at.file_uploader, "BRD/SRS PDF").value.name == (
        "customer-login.pdf"
    )
    assert _element(at.selectbox, "Run type").value is (
        RunType.CENTRALIZED_MULTI_AGENT
    )


def test_missing_upload_stays_in_create_without_calling_runner() -> None:
    calls = 0

    def fake_runner(*args, **kwargs):
        nonlocal calls
        calls += 1

    at = _app_test()
    at.session_state["_runner"] = fake_runner
    at.run()
    _element(at.button, "Create new run").click()
    at.run()
    _element(at.button, "Generate test cases").click()
    at.run()

    assert calls == 0
    assert at.session_state["view"] == "create"
    assert (
        "Upload one text-extractable PDF before generating test cases."
        in _rendered_text(at)
    )


def test_invalid_settings_during_create_reopens_settings_without_running() -> None:
    calls = 0

    def fake_runner(*args, **kwargs):
        nonlocal calls
        calls += 1

    at = _app_test()
    at.session_state["_runner"] = fake_runner
    at.run()
    _element(at.button, "Create new run").click()
    at.run()
    _element(at.file_uploader, "BRD/SRS PDF").set_value(
        [("customer-login.pdf", b"%PDF-1.4\n", "application/pdf")]
    )
    at.session_state["app_settings"] = AppSettings.model_construct(
        provider="gemini",
        model="gemini-3.6-flash",
        api_key="",
        base_url="",
        token_ceiling=200_000,
    )
    _element(at.button, "Generate test cases").click()
    at.run()

    assert calls == 0
    assert at.session_state["view"] == "create"
    assert "Gemini API key is required." in _rendered_text(at)
    assert at.session_state["settings_after_persist"] == "create"
    assert _element(at.text_input, "Gemini API key")
