# Runs-First Navigation and Browser-Local Settings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the four Streamlit workflow tabs with a runs-first workspace, dedicated create/detail views, and explicitly saved browser-local provider settings.

**Architecture:** Keep PostgreSQL and `RunResult` as the saved-run source of truth. Add one focused `browser_settings.py` module for strict settings data plus Streamlit's built-in bidirectional `localStorage` bridge, while `app.py` owns three session-state views and reuses the current runner, repository, and result renderer. No database migration or third-party frontend dependency is required.

**Tech Stack:** Python 3.11, Streamlit 1.61, Streamlit Components v2, Pydantic 2, PostgreSQL/psycopg, pytest, Streamlit AppTest

---

## Scope and pre-flight

The approved design is `docs/superpowers/specs/2026-08-13-runs-first-navigation-local-settings-design.md`.

The worktree currently contains unrelated user changes in `.env.example`, `app.py`, `compose.yaml`, `tests/test_app.py`, `.dockerignore`, and `Dockerfile`. Before each edit, inspect the current diff and preserve those changes. Stage only the paths named by each commit step; never use `git add .`.

No changes are needed in `models.py`, `schema.sql`, `storage.py`, or `runner.py`. `RunManifest` already stores the safe immutable snapshot, and `RunRepository.list_runs()`/`load_run()` already provide the required data.

## File map

| File | Responsibility |
|---|---|
| `requirements.txt` | Raise the Streamlit floor to the installed Components v2-capable version. |
| `src/brd_srs_testgen/browser_settings.py` | Define strict app settings, provider validation, safe fallback parsing, and the bidirectional `localStorage` component. |
| `tests/test_browser_settings.py` | Test settings validation, version rejection, fallback behavior, and absence of run type. |
| `app.py` | Render top navigation, Settings dialog, Runs/Create/Detail views, generation navigation, snapshots, and existing artifacts. |
| `tests/test_app.py` | Exercise settings restore/save, view navigation, run selection, generation, detail rendering, and contained failures. |
| `README.md` | Describe the runs-first quick-start workflow. |
| `docs/research-core-operations.md` | Document browser-local credentials, creating runs, reopening details, and the manual storage smoke check. |

### Task 1: Add the browser-settings boundary

**Files:**
- Create: `src/brd_srs_testgen/browser_settings.py`
- Create: `tests/test_browser_settings.py`
- Modify: `requirements.txt:5`

- [ ] **Step 1: Write the failing settings tests**

Create `tests/test_browser_settings.py`:

```python
import pytest

from brd_srs_testgen.browser_settings import AppSettings, parse_settings


def fallback_settings() -> AppSettings:
    return AppSettings(
        provider="gemini",
        model="gemini-3.6-flash",
        api_key="",
        base_url="",
        token_ceiling=200_000,
    )


def test_valid_settings_build_provider_settings_without_run_type() -> None:
    settings = AppSettings(
        provider="ollama",
        model="gemma4",
        api_key="",
        base_url="http://localhost:11434",
        token_ceiling=80_000,
    )

    provider = settings.provider_settings()

    assert provider.provider == "ollama"
    assert provider.model == "gemma4"
    assert provider.base_url == "http://localhost:11434"
    assert "run_type" not in settings.model_dump()


def test_gemini_settings_require_a_credential_before_a_run() -> None:
    settings = fallback_settings()

    with pytest.raises(ValueError, match="Gemini API key is required"):
        settings.provider_settings()


@pytest.mark.parametrize(
    "payload",
    [
        {"version": 2},
        {
            "version": 1,
            "provider": "gemini",
            "model": "gemini-3.6-flash",
            "api_key": "",
            "base_url": "",
            "token_ceiling": 200_000,
        },
        {
            "version": 1,
            "provider": "ollama",
            "model": " ",
            "api_key": "",
            "base_url": "http://localhost:11434",
            "token_ceiling": 200_000,
        },
    ],
)
def test_invalid_saved_settings_fall_back_without_exposing_values(payload) -> None:
    fallback = fallback_settings()

    settings, warning = parse_settings(payload, fallback)

    assert settings == fallback
    assert warning == "Saved browser settings were invalid; app defaults were restored."
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_browser_settings.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'brd_srs_testgen.browser_settings'`.

- [ ] **Step 3: Implement strict settings and the `localStorage` bridge**

Create `src/brd_srs_testgen/browser_settings.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import streamlit as st
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .runner import ProviderSettings


SETTINGS_VERSION = 1
STORAGE_KEY = "brd-srs-test-case.settings.v1"


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[1] = SETTINGS_VERSION
    provider: Literal["gemini", "lm_studio", "ollama"]
    model: str
    api_key: str = Field(default="", repr=False)
    base_url: str = ""
    token_ceiling: int = Field(ge=1000)

    @field_validator("model")
    @classmethod
    def model_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Model must not be blank.")
        return value

    def provider_settings(self) -> ProviderSettings:
        settings = ProviderSettings(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            token_ceiling=self.token_ceiling,
        )
        settings.validate()
        return settings


def parse_settings(
    payload: object, fallback: AppSettings
) -> tuple[AppSettings, str | None]:
    if payload is None:
        return fallback, None
    try:
        settings = AppSettings.model_validate(payload)
        settings.provider_settings()
    except (ValidationError, ValueError):
        return (
            fallback,
            "Saved browser settings were invalid; app defaults were restored.",
        )
    return settings, None


@dataclass(frozen=True)
class BrowserSettingsResult:
    payload: object = None
    error: str | None = None
    loaded: bool = False
    revision: int = -1


_BROWSER_SETTINGS = st.components.v2.component(
    "brd_srs_browser_settings",
    js="""
        export default function(component) {
            const {data, setStateValue} = component;
            let payload = null;
            let error = null;
            try {
                if (data.save !== null) {
                    localStorage.setItem(data.storageKey, JSON.stringify(data.save));
                }
                const raw = localStorage.getItem(data.storageKey);
                payload = raw === null ? null : JSON.parse(raw);
            } catch (_) {
                error = "Browser settings storage is unavailable.";
            }
            setStateValue("payload", payload);
            setStateValue("error", error);
            setStateValue("loaded", true);
            setStateValue("revision", data.revision);
        }
    """,
)


def sync_browser_settings(
    *, save: dict[str, object] | None = None, revision: int = 0
) -> BrowserSettingsResult:
    injected = st.session_state.get("_browser_settings_sync")
    if injected is not None:
        return injected(save=save, revision=revision)
    result = _BROWSER_SETTINGS(
        data={"storageKey": STORAGE_KEY, "save": save, "revision": revision},
        default={
            "payload": None,
            "error": None,
            "loaded": False,
            "revision": -1,
        },
        key="browser-settings-storage",
        on_payload_change=lambda: None,
        on_error_change=lambda: None,
        on_loaded_change=lambda: None,
        on_revision_change=lambda: None,
        height=0,
    )
    return BrowserSettingsResult(
        payload=result.payload,
        error=result.error,
        loaded=result.loaded,
        revision=result.revision,
    )
```

Change the Streamlit dependency in `requirements.txt`:

```text
streamlit>=1.61,<2
```

- [ ] **Step 4: Run the focused tests**

Run:

```bash
rtk uv pip install --python .venv/bin/python -r requirements.txt
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_browser_settings.py tests/test_runner.py -q
```

Expected: dependencies are current and both files pass. `tests/test_runner.py` confirms that `AppSettings.provider_settings()` continues to use the existing provider validation contract.

- [ ] **Step 5: Commit only the settings boundary**

```bash
rtk git add requirements.txt src/brd_srs_testgen/browser_settings.py tests/test_browser_settings.py
rtk git commit -m "feat: persist app settings in browser"
```

### Task 2: Replace the four tabs with the runs-first workspace shell

**Files:**
- Modify: `app.py:8-369,655-1074`
- Modify: `tests/test_app.py:18-209,421-568`

- [ ] **Step 1: Add a fake browser adapter and failing workspace tests**

Import the browser-settings types in `tests/test_app.py`:

```python
from brd_srs_testgen.browser_settings import AppSettings, BrowserSettingsResult
```

Add this fake beside `FakeRepository`:

```python
class FakeBrowserSettings:
    def __init__(
        self,
        payload: dict[str, object] | None = None,
        *,
        error: str | None = None,
    ) -> None:
        self.payload = payload
        self.error = error
        self.saved: list[dict[str, object]] = []

    def __call__(
        self, *, save: dict[str, object] | None, revision: int
    ) -> BrowserSettingsResult:
        if save is not None:
            self.payload = save
            self.saved.append(save)
        return BrowserSettingsResult(
            payload=self.payload,
            error=self.error,
            loaded=True,
            revision=revision,
        )


def _saved_settings(**overrides: object) -> dict[str, object]:
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
```

Replace `_app_test` with:

```python
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
```

Replace the old four-tab/provider-history tests with these tests:

```python
def test_runs_home_is_initial_and_restores_browser_settings() -> None:
    completed = _detailed_run()
    interrupted = _interrupted_run()
    repository = FakeRepository(
        runs=[_history_item(interrupted), _history_item(completed)]
    )
    browser = FakeBrowserSettings(
        _saved_settings(
            provider="ollama",
            model="gemma4",
            api_key="",
            base_url="http://localhost:11434",
            token_ceiling=75_000,
        )
    )
    at = _app_test(repository, browser)

    at.run()

    assert not at.exception
    assert not at.tabs
    assert at.session_state["view"] == "runs"
    assert at.session_state["app_settings"].provider == "ollama"
    assert {button.label for button in at.button} >= {
        "BRD/SRS Test Case",
        "Settings",
        "Create new run",
    }
    runs = at.dataframe[0].value
    assert list(runs.columns) == [
        "Started",
        "Source",
        "Run type",
        "Provider",
        "Model",
        "Status",
        "Test cases",
    ]
    assert list(runs["Source"]) == ["interrupted.pdf", "sample.pdf"]
    assert list(runs["Status"]) == ["Interrupted", "Completed"]
    assert list(runs["Test cases"]) == ["—", 1]


def test_settings_save_is_explicit_and_excludes_run_type() -> None:
    browser = FakeBrowserSettings(_saved_settings())
    at = _app_test(browser=browser)
    at.run()

    _element(at.button, "Settings").click()
    at.run()
    _element(at.text_input, "Model").set_value("gemini-3.6-pro")
    at.run()

    assert browser.saved == []
    _element(at.button, "Cancel").click()
    at.run()
    assert at.session_state["app_settings"].model == "gemini-3.6-flash"

    _element(at.button, "Settings").click()
    at.run()
    _element(at.text_input, "Model").set_value("gemini-3.6-pro")
    at.run()

    _element(at.button, "Save settings").click()
    at.run()

    assert browser.saved[-1]["model"] == "gemini-3.6-pro"
    assert "run_type" not in browser.saved[-1]
    assert at.session_state["app_settings"].model == "gemini-3.6-pro"


def test_settings_dialog_keeps_lm_studio_model_loading(monkeypatch) -> None:
    monkeypatch.setenv("LM_STUDIO_API_TOKEN", "lm-token-from-env")
    monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://lm-studio:1234/v1")
    at = _app_test()
    at.session_state["_model_loader"] = lambda *_: [
        "google/gemma-4-26b-a4b-qat",
        "qwen/qwen3-4b",
    ]
    at.run()
    _element(at.button, "Settings").click()
    at.run()

    _element(at.selectbox, "Provider").set_value("LM Studio")
    at.run()

    assert _element(at.text_input, "LM Studio API token").value == (
        "lm-token-from-env"
    )
    assert _element(at.text_input, "LM Studio base URL").value == (
        "http://lm-studio:1234/v1"
    )
    _element(at.button, "Load available models").click()
    at.run()
    model = _element(at.selectbox, "Model")
    assert model.value == "google/gemma-4-26b-a4b-qat"
    model.set_value("qwen/qwen3-4b")
    at.run()
    assert _element(at.selectbox, "Model").value == "qwen/qwen3-4b"


def test_missing_settings_open_dialog_then_continue_to_create() -> None:
    browser = FakeBrowserSettings(None)
    result = _detailed_run()
    at = _app_test(
        FakeRepository(runs=[_history_item(result)]),
        browser,
    )
    at.run()

    assert list(at.dataframe[0].value["Source"]) == ["sample.pdf"]
    _element(at.button, "Create new run").click()
    at.run()

    assert at.get("dialog")
    assert _element(at.text_input, "Gemini API key")
    _element(at.text_input, "Gemini API key").set_value("new-secret")
    _element(at.button, "Save settings").click()
    at.run()

    assert at.session_state["view"] == "create"
    assert _element(at.file_uploader, "BRD/SRS PDF")
    assert _element(at.selectbox, "Run type")


def test_empty_runs_home_keeps_create_available() -> None:
    at = _app_test()

    at.run()

    assert "No saved runs yet." in _rendered_text(at)
    assert _element(at.button, "Create new run")
    assert not at.dataframe


def test_selecting_a_run_opens_dedicated_detail() -> None:
    result = _detailed_run()
    repository = FakeRepository(
        runs=[_history_item(result)],
        results={result.manifest.run_id: result},
    )
    at = _app_test(repository)
    at.run()

    at.session_state["runs-table"] = {
        "selection": {"rows": [0], "columns": [], "cells": []}
    }
    at.run()

    assert repository.load_calls == [result.manifest.run_id]
    assert at.session_state["view"] == "detail"
    assert _element(at.button, "Back to runs")
    assert "TC-001 · Sign in with valid credentials" in _rendered_text(at)


def test_browser_storage_error_does_not_block_run_history() -> None:
    result = _detailed_run()
    repository = FakeRepository(runs=[_history_item(result)])
    browser = FakeBrowserSettings(
        None, error="Browser settings storage is unavailable."
    )
    at = _app_test(repository, browser)

    at.run()

    assert "Browser settings storage is unavailable." in _rendered_text(at)
    assert list(at.dataframe[0].value["Source"]) == ["sample.pdf"]
    assert _element(at.button, "Create new run")


def test_history_list_error_is_contained_and_create_remains_available() -> None:
    repository = FakeRepository(
        list_error=StorageError("postgresql://user:list-secret@localhost/database")
    )
    at = _app_test(repository)

    at.run()

    text = _rendered_text(at)
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
    assert at.session_state["view"] == "runs"
    assert "Saved run could not be opened" in text
    assert "load-secret" not in text
    assert _element(at.button, "Create new run")
```

Delete the superseded tests named `test_prefills_provider_credentials_and_preserves_lm_studio_controls`, `test_provider_change_resets_model_and_keeps_four_step_workflow`, `test_history_table_shows_completed_and_interrupted_runs`, `test_identical_history_rows_have_distinct_run_id_labels_and_load_correctly`, `test_history_empty_state`, `test_selecting_saved_run_reuses_detailed_result_renderer_without_key_collisions`, `test_history_list_error_is_contained_redacted_and_actionable`, and `test_history_load_error_is_contained_redacted_and_actionable`. Keep `test_database_initialization_failure_blocks_generation_and_redacts_detail`, replacing its final assertion with:

```python
assert not any(button.label == "Create new run" for button in at.button)
assert not at.dataframe
```

- [ ] **Step 2: Run the workspace tests to verify they fail**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_app.py -q
```

Expected: failures mention the old tabs and missing `BRD/SRS Test Case`, `Settings`, `Create new run`, and `view` state.

- [ ] **Step 3: Integrate browser settings and dialog-local drafts**

Add this import to `app.py`:

```python
from brd_srs_testgen.browser_settings import (
    AppSettings,
    parse_settings,
    sync_browser_settings,
)
```

Replace `_reset_provider`, `_clear_lm_studio_models`, and `_refresh_lm_studio_models` with dialog-draft versions:

```python
def _reset_settings_provider() -> None:
    provider = st.session_state["settings_provider"]
    st.session_state["settings_model"] = _default_model(provider)
    st.session_state["settings_api_key"] = (
        _env("GEMINI_API_KEY")
        if provider == "gemini"
        else _env("LM_STUDIO_API_TOKEN")
        if provider == "lm_studio"
        else ""
    )
    st.session_state["settings_base_url"] = (
        _base_url(provider) if provider in LOCAL_BASE_URLS else ""
    )
    st.session_state.pop("lm_studio_models", None)
    st.session_state.pop("lm_studio_model_error", None)


def _clear_lm_studio_models() -> None:
    st.session_state["settings_model"] = ""
    st.session_state.pop("lm_studio_models", None)
    st.session_state.pop("lm_studio_model_error", None)


def _refresh_lm_studio_models() -> None:
    api_key = st.session_state.get("settings_api_key", "")
    try:
        loader = st.session_state.get("_model_loader", list_lm_studio_models)
        models = loader(st.session_state["settings_base_url"], api_key)
    except Exception as error:
        message = str(error)
        if api_key:
            message = message.replace(api_key, "[REDACTED]")
        st.session_state["settings_model"] = ""
        st.session_state["lm_studio_models"] = []
        st.session_state["lm_studio_model_error"] = message
    else:
        st.session_state["lm_studio_models"] = models
        st.session_state["settings_model"] = models[0]
        st.session_state.pop("lm_studio_model_error", None)
```

Add the settings-state and dialog functions before `_provider_label`:

```python
def _fallback_settings() -> AppSettings:
    return AppSettings(
        provider="gemini",
        model=GEMINI_DEFAULT_MODEL,
        api_key=_env("GEMINI_API_KEY"),
        base_url="",
        token_ceiling=200_000,
    )


def _sync_app_settings() -> AppSettings:
    fallback = _fallback_settings()
    st.session_state.setdefault("app_settings", fallback)
    revision = st.session_state.get("settings_revision", 0)
    pending = st.session_state.get("settings_save_request")
    result = sync_browser_settings(save=pending, revision=revision)
    should_process = (
        not st.session_state.get("browser_settings_loaded") or pending is not None
    )
    if result.loaded and result.revision == revision and should_process:
        if result.error:
            st.session_state["settings_warning"] = result.error
        else:
            settings, warning = parse_settings(result.payload, fallback)
            if warning:
                st.session_state["settings_warning"] = warning
            else:
                st.session_state["app_settings"] = settings
                if pending is not None and (
                    destination := st.session_state.pop(
                        "settings_after_persist", None
                    )
                ):
                    st.session_state["view"] = destination
        st.session_state["browser_settings_loaded"] = True
        if pending is not None:
            st.session_state.pop("settings_save_request", None)
            st.session_state.pop("settings_after_persist", None)
    return st.session_state["app_settings"]


def _settings_are_ready(settings: AppSettings) -> bool:
    try:
        settings.provider_settings()
    except ValueError:
        return False
    return True


def _open_settings(*, after_save: str | None = None) -> None:
    settings: AppSettings = st.session_state["app_settings"]
    st.session_state["settings_provider"] = settings.provider
    st.session_state["settings_model"] = settings.model
    st.session_state["settings_api_key"] = settings.api_key
    st.session_state["settings_base_url"] = settings.base_url
    st.session_state["settings_token_ceiling"] = settings.token_ceiling
    st.session_state["settings_after_save"] = after_save
    st.session_state["show_settings"] = True


@st.dialog("App settings", width="large")
def _render_settings_dialog() -> None:
    provider_column, credential_column = st.columns(2, gap="large")
    with provider_column:
        provider = st.selectbox(
            "Provider",
            list(PROVIDER_LABELS),
            key="settings_provider",
            on_change=_reset_settings_provider,
            format_func=_provider_label,
        )
        if provider == "lm_studio":
            models = st.session_state.get("lm_studio_models", [])
            st.selectbox(
                "Model",
                models,
                index=0 if models else None,
                key="settings_model",
                placeholder="Load models or enter a model ID",
                accept_new_options=True,
            )
        else:
            st.text_input("Model", key="settings_model")
    with credential_column:
        if provider == "gemini":
            st.text_input(
                "Gemini API key", type="password", key="settings_api_key"
            )
        elif provider == "lm_studio":
            st.text_input(
                "LM Studio API token",
                type="password",
                key="settings_api_key",
                on_change=_clear_lm_studio_models,
            )
        if provider in LOCAL_BASE_URLS:
            st.text_input(
                f"{_provider_label(provider)} base URL",
                key="settings_base_url",
                on_change=(
                    _clear_lm_studio_models if provider == "lm_studio" else None
                ),
            )
        if provider == "lm_studio":
            st.button(
                "Load available models",
                on_click=_refresh_lm_studio_models,
                width="stretch",
            )
            if error := st.session_state.get("lm_studio_model_error"):
                st.error(f"Could not load models: {error}")
    st.number_input(
        "Token ceiling",
        min_value=1000,
        step=1000,
        key="settings_token_ceiling",
    )
    st.warning(
        "Saved credentials are readable by scripts running on this app origin."
    )
    save_column, cancel_column = st.columns(2)
    if save_column.button("Save settings", type="primary", width="stretch"):
        try:
            settings = AppSettings(
                provider=st.session_state["settings_provider"],
                model=st.session_state["settings_model"],
                api_key=st.session_state.get("settings_api_key", ""),
                base_url=st.session_state.get("settings_base_url", ""),
                token_ceiling=st.session_state["settings_token_ceiling"],
            )
            settings.provider_settings()
        except ValueError as error:
            st.error(str(error))
        else:
            st.session_state["settings_revision"] = (
                st.session_state.get("settings_revision", 0) + 1
            )
            st.session_state["settings_save_request"] = settings.model_dump()
            st.session_state["settings_after_persist"] = st.session_state.pop(
                "settings_after_save", None
            )
            st.session_state["show_settings"] = False
            st.rerun()
    if cancel_column.button("Cancel", width="stretch"):
        st.session_state.pop("settings_after_save", None)
        st.session_state["show_settings"] = False
        st.rerun()
```

- [ ] **Step 4: Add top navigation and three view renderers**

Replace `_render_empty_state`, `_history_label`, and `_render_history` with:

```python
def _go_home() -> None:
    st.session_state["view"] = "runs"
    st.session_state.pop("selected_run_id", None)
    st.session_state.pop("selected_run", None)
    st.session_state.pop("runs-table", None)


def _render_top_navigation() -> None:
    home_column, spacer, settings_column = st.columns([1.5, 6, 1])
    if home_column.button("BRD/SRS Test Case", type="tertiary"):
        _go_home()
        st.rerun()
    if settings_column.button("Settings", width="stretch"):
        _open_settings()


def _request_create(settings: AppSettings) -> None:
    if _settings_are_ready(settings):
        st.session_state["view"] = "create"
    else:
        _open_settings(after_save="create")


def _render_runs(repository: RunRepository, settings: AppSettings) -> None:
    title_column, action_column = st.columns([5, 1.4])
    title_column.title("Runs")
    title_column.caption("Open a saved test-case generation run or create a new one.")
    if action_column.button("Create new run", type="primary", width="stretch"):
        _request_create(settings)
        if st.session_state.get("view") == "create":
            st.rerun()
    try:
        runs = repository.list_runs()
    except StorageError:
        st.error(
            "Saved run history is unavailable. Check PostgreSQL and DATABASE_URL, "
            "then refresh this page."
        )
        return
    if not runs:
        st.info("No saved runs yet.")
        return
    rows = [
        {
            "Started": item.started_at.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "Source": item.source_filename,
            "Run type": _run_type_label(item.run_type),
            "Provider": _provider_label(item.provider),
            "Model": item.model,
            "Status": item.display_status,
            "Test cases": item.test_case_count
            if item.test_case_count is not None
            else "—",
        }
        for item in runs
    ]
    selection = st.dataframe(
        rows,
        key="runs-table",
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
    )
    if selection.selection.rows:
        item = runs[selection.selection.rows[0]]
        st.session_state["selected_run_id"] = item.run_id
        st.session_state.pop("selected_run", None)
        st.session_state["view"] = "detail"
        st.rerun()


def _render_create(settings: AppSettings) -> None:
    if st.button("Back to runs"):
        _go_home()
        st.rerun()
    st.title("Create new run")
    st.caption("Upload one BRD/SRS and choose exactly one generation strategy.")
    st.file_uploader("BRD/SRS PDF", type=["pdf"], key="pdf")
    run_type = st.selectbox(
        "Run type",
        list(RunType),
        key="run_type",
        format_func=_run_type_label,
    )
    st.caption(RUN_TYPE_COPY[run_type][1])
    with st.container(border=True):
        st.markdown("#### App settings")
        st.caption(
            f"{_provider_label(settings.provider)} · {settings.model} · "
            f"{settings.token_ceiling:,} token ceiling"
        )
        if st.button("Edit settings"):
            _open_settings(after_save="create")


def _render_detail(repository: RunRepository) -> None:
    if st.button("Back to runs"):
        _go_home()
        st.rerun()
    run_id = st.session_state.get("selected_run_id")
    if not isinstance(run_id, str):
        _go_home()
        st.session_state["flash_error"] = "Select a saved run to open its details."
        st.rerun()
    result = st.session_state.get("selected_run")
    if not isinstance(result, RunResult) or result.manifest.run_id != run_id:
        try:
            result = repository.load_run(run_id)
        except StorageError:
            _go_home()
            st.session_state["flash_error"] = (
                "Saved run could not be opened. Check PostgreSQL and DATABASE_URL, "
                "then try again."
            )
            st.rerun()
        st.session_state["selected_run"] = result
    st.title(result.manifest.source_filename)
    _render_result(result, key_prefix="detail")
```

- [ ] **Step 5: Replace `main()` with state-driven routing**

Replace `main()` with:

```python
def main() -> None:
    st.set_page_config(
        page_title="BRD/SRS Test-Case Research Core",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _apply_theme()
    settings = _sync_app_settings()
    _render_top_navigation()
    if warning := st.session_state.pop("settings_warning", None):
        st.warning(warning)
    if error := st.session_state.pop("flash_error", None):
        st.error(error)
    try:
        repository = _resolve_repository()
    except StorageError:
        st.error(
            "Run history database is unavailable. Start it with "
            "`docker compose up -d db`, verify DATABASE_URL, and refresh this page."
        )
        st.stop()

    view = st.session_state.setdefault("view", "runs")
    if view == "create":
        _render_create(settings)
    elif view == "detail":
        _render_detail(repository)
    else:
        st.session_state["view"] = "runs"
        _render_runs(repository, settings)

    if st.session_state.get("show_settings"):
        _render_settings_dialog()
```

Keep the final `main()` invocation. Delete the old research hero, tab creation, tab-only CSS selectors, step headings, `_render_empty_state`, `_history_label`, and `_render_history`; they have no callers after this replacement.

- [ ] **Step 6: Run the workspace slice**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_app.py -k 'runs_home or settings or empty_runs or selecting_a_run or storage_error or list_error or missing_selected or database_initialization' -q
```

Expected: every selected workspace, settings, navigation, and contained-failure test passes with no Python exception or duplicate widget key.

- [ ] **Step 7: Continue directly to Task 3**

Keep the workspace-shell changes unstaged until Task 3 completes the Generate action and the full Streamlit test file passes. This avoids recording a commit with a deliberately incomplete Create Run view.

### Task 3: Complete the Create Run generation flow

**Files:**
- Modify: `app.py` (`_render_create`)
- Modify: `tests/test_app.py:211-379`

- [ ] **Step 1: Replace generation tests with dedicated-view expectations**

Add this test helper beside `_upload`:

```python
def _show_detail(at: AppTest, result: RunResult) -> None:
    at.session_state["view"] = "detail"
    at.session_state["selected_run_id"] = result.manifest.run_id
    at.session_state["selected_run"] = result
```

Replace `test_runs_one_selected_type_and_passes_the_complete_request`, `test_clears_previous_result_when_a_later_generation_raises`, and `test_clears_previous_result_when_runner_returns_a_failure` with:

```python
def test_create_runs_one_selected_type_and_opens_its_detail() -> None:
    captured = {}
    repository = FakeRepository()
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

    at = _app_test(repository)
    at.session_state["_runner"] = fake_runner
    at.run()
    _element(at.button, "Create new run").click()
    at.run()

    run_type = _element(at.selectbox, "Run type")
    assert list(run_type.options) == [
        "Single prompt",
        "Staged single agent",
        "Centralized multi-agent",
    ]
    run_type.set_value("Staged single agent")
    _upload(at)
    _element(at.button, "Generate test cases").click()
    at.run()

    assert not at.exception
    assert captured["pdf_bytes"] == b"%PDF-1.4\n"
    assert captured["source_filename"] == "customer-login.pdf"
    assert captured["run_type"] is RunType.STAGED_SINGLE_AGENT
    assert captured["repository"] is repository
    assert captured["settings"].api_key == "browser-secret"
    assert at.session_state["view"] == "detail"
    assert at.session_state["selected_run"] == expected
    assert "TC-001 · Sign in with valid credentials" in _rendered_text(at)


def test_returned_failed_run_opens_detail_with_diagnostics() -> None:
    at = _app_test()
    at.session_state["_runner"] = lambda *args, **kwargs: _failed_run()
    at.run()
    _element(at.button, "Create new run").click()
    at.run()
    _upload(at)
    _element(at.button, "Generate test cases").click()
    at.run()

    assert at.session_state["view"] == "detail"
    assert at.status[0].label == "Generation failed"
    assert {button.label for button in at.download_button} == {
        "Download diagnostics"
    }


def test_unexpected_generation_error_stays_in_create_and_redacts_settings() -> None:
    secret = "not-for-display"
    browser = FakeBrowserSettings(_saved_settings(api_key=secret))

    def fake_runner(pdf_bytes, source_filename, run_type, settings, **kwargs):
        raise RuntimeError(f"runner stopped: {settings.api_key} {settings.base_url}")

    at = _app_test(browser=browser)
    at.session_state["_runner"] = fake_runner
    at.run()
    _element(at.button, "Create new run").click()
    at.run()
    _upload(at)
    _element(at.button, "Generate test cases").click()
    at.run()

    assert at.session_state["view"] == "create"
    assert "Generation failed: runner stopped" in _rendered_text(at)
    assert secret not in _rendered_text(at)
    assert not at.download_button


def test_edit_settings_preserves_create_inputs() -> None:
    browser = FakeBrowserSettings(_saved_settings())
    at = _app_test(browser=browser)
    at.run()
    _element(at.button, "Create new run").click()
    at.run()
    _element(at.selectbox, "Run type").set_value("Centralized multi-agent")
    _upload(at)
    _element(at.button, "Edit settings").click()
    at.run()
    _element(at.text_input, "Model").set_value("gemini-3.6-pro")
    _element(at.button, "Save settings").click()
    at.run()

    assert at.session_state["view"] == "create"
    assert _element(at.selectbox, "Run type").value == "Centralized multi-agent"
    assert _element(at.file_uploader, "BRD/SRS PDF").value[0].name == (
        "customer-login.pdf"
    )
    assert "gemini-3.6-pro" in _rendered_text(at)
```

Replace the three direct-result tests with dedicated-view versions:

```python
def test_failed_result_without_metrics_has_an_actionable_summary() -> None:
    at = _app_test()
    _show_detail(at, _failed_run())

    at.run()

    assert at.status[0].label == "Generation failed"
    assert "text-extractable PDF" in _rendered_text(at)
    assert _element(at.download_button, "Download diagnostics")


def test_interrupted_result_has_diagnostics_without_a_fake_failure() -> None:
    at = _app_test()
    _show_detail(at, _interrupted_run())

    at.run()

    text = _rendered_text(at)
    assert at.status[0].label == "Generation interrupted"
    assert "Unknown failure" not in text
    assert "Technical details" not in text
    assert not at.error
    assert not at.success
    diagnostics = _element(at.download_button, "Download diagnostics")
    assert diagnostics.key == "detail-interrupted-run-diagnostics"


def test_failed_semantic_result_keeps_artifact_details_and_diagnostics() -> None:
    at = _app_test()
    _show_detail(at, _semantic_failure_with_bundle())

    at.run()

    text = _rendered_text(at)
    assert "TC-001 · Sign in with valid credentials" in text
    assert "LM Studio" in text
    assert {button.label for button in at.download_button} == {
        "Download diagnostics"
    }
```

Remove the superseded `test_redacts_credentials_and_base_url_from_runner_error`; the new unexpected-generation test covers that behavior in the Create view.

In the existing detailed-artifact test, replace its session setup and download-key assertions with:

```python
at = _app_test()
result = _detailed_run()
_show_detail(at, result)
at.run()

assert {button.key for button in at.download_button} == {
    f"detail-{result.manifest.run_id}-rtm",
    f"detail-{result.manifest.run_id}-bundle",
}
```

Keep its artifact, metric, step, and download-label assertions unchanged until Task 4 adds ordering and snapshot coverage.

- [ ] **Step 2: Run the generation tests to verify they fail**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_app.py -k 'create_runs or returned_failed or unexpected_generation' -q
```

Expected: failures report that `Generate test cases` is absent.

- [ ] **Step 3: Add generation to `_render_create`**

Replace `_render_create` with:

```python
def _render_create(repository: RunRepository, settings: AppSettings) -> None:
    if st.button("Back to runs"):
        _go_home()
        st.rerun()
    st.title("Create new run")
    st.caption("Upload one BRD/SRS and choose exactly one generation strategy.")
    uploaded = st.file_uploader(
        "BRD/SRS PDF",
        type=["pdf"],
        key="pdf",
        help="Use a text-extractable PDF; scanned image-only files are unsupported.",
    )
    run_type = st.selectbox(
        "Run type",
        list(RunType),
        key="run_type",
        format_func=_run_type_label,
    )
    st.caption(RUN_TYPE_COPY[run_type][1])
    with st.container(border=True):
        st.markdown("#### App settings")
        st.caption(
            f"{_provider_label(settings.provider)} · {settings.model} · "
            f"{settings.token_ceiling:,} token ceiling"
        )
        if st.button("Edit settings"):
            _open_settings(after_save="create")

    if not st.button(
        "Generate test cases", type="primary", width="stretch", key="run"
    ):
        return
    if uploaded is None:
        st.error("Upload one text-extractable PDF before generating test cases.")
        return
    try:
        provider_settings = settings.provider_settings()
    except ValueError as error:
        st.error(str(error))
        _open_settings(after_save="create")
        return
    try:
        with st.status("Preparing generation", expanded=True) as status:

            def progress(message: str) -> None:
                status.write(message)

            runner = st.session_state.get("_runner", run_generation)
            result = runner(
                uploaded.getvalue(),
                uploaded.name,
                run_type,
                provider_settings,
                repository=repository,
                progress=progress,
            )
            label, state = _result_status(result)
            status.update(label=label, state=state, expanded=state == "error")
    except Exception as error:
        st.error(f"Generation failed: {_safe_error(error, provider_settings)}")
        return
    st.session_state["selected_run_id"] = result.manifest.run_id
    st.session_state["selected_run"] = result
    st.session_state["view"] = "detail"
    st.rerun()
```

Update the create branch in `main()`:

```python
if view == "create":
    _render_create(repository, settings)
```

- [ ] **Step 4: Run all Streamlit tests**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_app.py tests/test_browser_settings.py -q
```

Expected: every test passes. Confirm `rtk rg -n 'run_result|current-|history-|four_step|four-step' tests/test_app.py` returns no matches; do not keep a second current-result renderer.

- [ ] **Step 5: Commit the create flow**

```bash
rtk git add app.py tests/test_app.py
rtk git commit -m "feat: add runs-first generation workspace"
```

### Task 4: Make test cases primary and show the immutable snapshot

**Files:**
- Modify: `app.py:470-654`
- Modify: `tests/test_app.py` (`test_renders_detailed_artifact_content_and_downloads`)

- [ ] **Step 1: Extend the detail-rendering test**

In `test_renders_detailed_artifact_content_and_downloads`, navigate directly to the cached detail and add ordering/snapshot assertions:

```python
def test_renders_test_cases_first_with_snapshot_and_downloads() -> None:
    secret = "browser-only-secret"
    base_url = "http://localhost:11434"
    at = _app_test(
        browser=FakeBrowserSettings(
            _saved_settings(
                provider="ollama",
                model="gemma4",
                api_key=secret,
                base_url=base_url,
            )
        )
    )
    result = _detailed_run()
    at.session_state["view"] = "detail"
    at.session_state["selected_run_id"] = result.manifest.run_id
    at.session_state["selected_run"] = result

    at.run()

    assert not at.exception
    headings = [str(item.value) for item in at.markdown]
    assert headings.index("#### Test cases") < headings.index("#### Requirements")
    text = _rendered_text(at)
    for expected in (
        "TC-001 · Sign in with valid credentials",
        "REQ-001 · Authenticate users",
        "SCN-001 · Valid sign in",
        "Run configuration snapshot",
        result.manifest.document_hash,
        result.manifest.prompt_version,
        result.manifest.schema_version,
        result.manifest.run_id,
    ):
        assert expected in text or expected in "\n".join(
            str(table.value) for table in at.table
        )
    assert secret not in text
    assert secret not in "\n".join(str(table.value) for table in at.table)
    assert base_url not in text
    assert base_url not in "\n".join(str(table.value) for table in at.table)
    assert {button.key for button in at.download_button} == {
        f"detail-{result.manifest.run_id}-rtm",
        f"detail-{result.manifest.run_id}-bundle",
    }
```

- [ ] **Step 2: Run the detail test to verify it fails**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_app.py -k 'test_cases_first' -q
```

Expected: failure because Requirements currently precede Test cases and no snapshot section exists.

- [ ] **Step 3: Extract artifact renderers and order test cases first**

Split the current `_render_bundle` loops into the following functions, keeping each loop body exactly equivalent to the existing renderer:

```python
def _render_test_cases(result: RunResult, *, key_prefix: str) -> None:
    assert result.bundle is not None
    st.markdown("#### Test cases")
    for position, test_case in enumerate(result.bundle.test_cases):
        with st.expander(
            f"{test_case.test_case_id} · {test_case.title}",
            key=f"{key_prefix}-{result.manifest.run_id}-test-case-{position}",
        ):
            st.markdown(
                f"**Priority:** {test_case.priority.value}  \n"
                f"**Scenario ID:** {test_case.scenario_id}  \n"
                f"**Requirement IDs:** {', '.join(test_case.requirement_ids)}"
            )
            st.markdown("**Preconditions:**")
            st.markdown(
                "\n".join(f"- {item}" for item in test_case.preconditions)
                or "None"
            )
            st.markdown("**Test data**")
            st.code(_json(test_case.test_data), language="json")
            st.markdown("**Steps**")
            st.table(
                [
                    {
                        "Step": step.step_number,
                        "Action": step.action,
                        "Expected result": step.expected_result,
                    }
                    for step in test_case.steps
                ]
            )
            _render_sources(test_case.source_references)


def _render_requirements(result: RunResult, *, key_prefix: str) -> None:
    assert result.bundle is not None
    st.markdown("#### Requirements")
    for position, requirement in enumerate(result.bundle.requirements):
        with st.expander(
            f"{requirement.requirement_id} · {requirement.title}",
            key=f"{key_prefix}-{result.manifest.run_id}-requirement-{position}",
        ):
            st.markdown(requirement.description)
            st.markdown(
                f"**Type:** {requirement.requirement_type.value.replace('_', ' ').title()}  \n"
                f"**Priority:** {requirement.priority.value.title()}  \n"
                f"**Module:** {requirement.module}"
            )
            st.markdown(
                "**Dependency IDs:** "
                + (", ".join(requirement.dependency_ids) or "None")
            )
            st.markdown("**Ambiguities:**")
            st.markdown(
                "\n".join(f"- {item}" for item in requirement.ambiguities)
                or "None"
            )
            _render_sources(requirement.source_references)


def _render_scenarios(result: RunResult, *, key_prefix: str) -> None:
    assert result.bundle is not None
    st.markdown("#### Scenarios")
    for position, scenario in enumerate(result.bundle.scenarios):
        with st.expander(
            f"{scenario.scenario_id} · {scenario.title}",
            key=f"{key_prefix}-{result.manifest.run_id}-scenario-{position}",
        ):
            st.markdown(scenario.objective)
            st.markdown(
                f"**Type:** {scenario.scenario_type.value.replace('_', ' ').title()}  \n"
                f"**Requirement IDs:** {', '.join(scenario.requirement_ids)}"
            )
            st.markdown("**Preconditions:**")
            st.markdown(
                "\n".join(f"- {item}" for item in scenario.preconditions)
                or "None"
            )
            _render_sources(scenario.source_references)


def _render_bundle(result: RunResult, *, key_prefix: str) -> None:
    if result.bundle is None:
        return
    st.markdown("### Generated artifacts")
    _render_test_cases(result, key_prefix=key_prefix)
    _render_requirements(result, key_prefix=key_prefix)
    _render_scenarios(result, key_prefix=key_prefix)
```

- [ ] **Step 4: Add the snapshot renderer**

Add this before `_render_result`:

```python
def _render_snapshot(result: RunResult) -> None:
    manifest = result.manifest
    st.markdown("### Run configuration snapshot")
    st.table(
        [
            {"Setting": "Run ID", "Value": manifest.run_id},
            {"Setting": "Run type", "Value": _run_type_label(manifest.run_type)},
            {"Setting": "Provider", "Value": _provider_label(manifest.provider)},
            {"Setting": "Model", "Value": manifest.model},
            {"Setting": "Temperature", "Value": f"{manifest.temperature:g}"},
            {"Setting": "Token ceiling", "Value": f"{manifest.token_ceiling:,}"},
            {"Setting": "Source filename", "Value": manifest.source_filename},
            {"Setting": "Document hash", "Value": manifest.document_hash},
            {"Setting": "Prompt version", "Value": manifest.prompt_version},
            {"Setting": "Schema version", "Value": manifest.schema_version},
            {"Setting": "Status", "Value": manifest.status.value},
            {"Setting": "Started", "Value": manifest.started_at.isoformat()},
            {
                "Setting": "Completed",
                "Value": manifest.completed_at.isoformat()
                if manifest.completed_at
                else "—",
            },
        ]
    )
```

Call `_render_snapshot(result)` in `_render_result` immediately after the status/failure summary and before metrics. Do not add `api_key` or `base_url` to the table.

- [ ] **Step 5: Run all Streamlit and settings tests**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m pytest tests/test_app.py tests/test_browser_settings.py -q
```

Expected: every test passes for home, settings, create, completed detail, failed detail, interrupted detail, navigation errors, and browser-storage fallback.

- [ ] **Step 6: Commit the detail presentation**

```bash
rtk git add app.py tests/test_app.py
rtk git commit -m "feat: show run test cases and snapshot"
```

### Task 5: Update operations and verify the complete redesign

**Files:**
- Modify: `README.md:17-20`
- Modify: `docs/research-core-operations.md:20-95`

- [ ] **Step 1: Update the quick-start workflow**

Replace the README sentence below the launch commands with:

```markdown
The home page lists PostgreSQL-backed runs newest first. Use **Settings** to save provider defaults in this browser, then choose **Create new run** to upload one PDF and execute exactly one generation type. Selecting a saved row opens its detailed test cases and immutable configuration snapshot.
```

- [ ] **Step 2: Update the operations guide**

Replace **Generate and reopen a run** with:

```markdown
## Generate and reopen a run

1. Open **Settings**, choose the provider and model, enter the applicable credential/base URL, set the token ceiling, and click **Save settings**.
2. From **Runs**, click **Create new run**.
3. Select exactly one run type, upload one text-extractable BRD/SRS PDF, and click **Generate test cases**. Only the selected run type executes.
4. The app opens the completed or failed run automatically. Review test cases first, then supporting requirements/scenarios, metrics, downloads, diagnostics, and the immutable configuration snapshot.
5. Use **Back to runs** and select any saved row to reopen it. A running record left by a stopped process is displayed as **Interrupted**.

Each Generate click creates a new immutable run. Correct a problem and generate again rather than modifying a saved run.
```

Replace the credential paragraph under **Persisted data and security boundary** with:

```markdown
Raw PDF bytes and provider credentials are never stored in PostgreSQL, downloads, URLs, or run snapshots. Clicking **Save settings** stores the active provider credential in this browser's `localStorage` as requested; scripts running on the same app origin can read it. Use a dedicated browser profile/origin and do not save a credential on a shared machine. Known secrets are redacted from displayed failures.
```

Replace **Live smoke tests (optional)** with:

```markdown
## Browser storage smoke test

1. Start the app and open **Settings**.
2. Save a non-production provider credential, model, URL where applicable, and token ceiling.
3. Refresh the page and confirm the saved settings are restored.
4. Create a small run and confirm the app opens its dedicated detail view.
5. Confirm the configuration snapshot and downloaded JSON omit the credential and base URL.
6. Return to **Runs**, select the same row, and confirm its test cases reopen.
```

Keep the provider-specific commands and replace their UI instructions with:

````markdown
### Gemini

In **Settings**, select `gemini`, use a supported model (the current default is `gemini-3.6-flash`), enter its API key, and click **Save settings**. Create one selected run and verify its details and downloads.

### Ollama

Ollama is required only for this smoke path. Start the service and make the model available:

```sh
ollama serve
```

In another terminal:

```sh
ollama pull gemma4
```

In **Settings**, select `ollama`; the editable defaults are `http://localhost:11434` and `gemma4`. Save, create one selected run, and verify the result and saved Runs row. Ollama requests disable thinking output.

### LM Studio

Start the local server from LM Studio's Developer tab and load a model. In **Settings**, select `LM Studio` and keep the default OpenAI-compatible base URL, `http://localhost:1234/v1`. If authentication is enabled, enter a token created in LM Studio Server Settings. Select **Load available models**, choose the loaded model, save, create one selected run, and verify the result and saved Runs row.
````

- [ ] **Step 3: Run the complete automated gate**

Start the dedicated local test database if needed, then run:

```bash
rtk docker compose up -d --wait db
set -a
. ./.env
set +a
PYTHONPATH=src rtk .venv/bin/python -m pytest -q
rtk .venv/bin/python -m compileall -q app.py src tests
PYTHONPATH=src rtk .venv/bin/python -c "from brd_srs_testgen.browser_settings import AppSettings; from brd_srs_testgen.runner import run_generation; print('imports ok')"
rtk git diff --check
```

Expected: the database becomes healthy, the full test suite passes with no skips in the PostgreSQL storage suite, compilation exits 0, the import check prints `imports ok`, and `git diff --check` prints nothing.

- [ ] **Step 4: Perform the browser smoke check**

Run:

```bash
PYTHONPATH=src rtk .venv/bin/python -m streamlit run app.py
```

In the browser, perform the six steps in **Browser storage smoke test**. Also confirm Settings Cancel does not persist edits and Edit settings preserves the selected run type and uploaded PDF.

Expected: settings survive refresh only after Save settings, run generation opens its detail, credentials/base URL are absent from run output, and all three views remain navigable.

- [ ] **Step 5: Commit documentation after the smoke check**

```bash
rtk git add README.md docs/research-core-operations.md
rtk git commit -m "docs: explain runs-first workflow"
```

- [ ] **Step 6: Inspect final scope**

Run:

```bash
rtk git status --short
rtk git log -6 --oneline
```

Expected: only the pre-existing unrelated user changes and this uncommitted plan file remain; the new implementation commits are visible. Do not stage or rewrite the unrelated Docker/LM Studio work.
