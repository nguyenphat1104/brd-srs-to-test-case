from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

from brd_srs_testgen.browser_settings import (
    AppSettings,
    parse_settings,
    sync_browser_settings,
)
from brd_srs_testgen.models import (
    FailureCategory,
    RunResult,
    RunStatus,
    RunType,
)
from brd_srs_testgen.providers import list_lm_studio_models
from brd_srs_testgen.runner import ProviderSettings
from brd_srs_testgen.storage import RunRepository, StorageError


GEMINI_DEFAULT_MODEL = "gemini-3.6-flash"
PROVIDER_LABELS = {
    "gemini": "Gemini",
    "lm_studio": "LM Studio",
    "ollama": "Ollama",
}
LOCAL_BASE_URLS = {
    "lm_studio": "http://localhost:1234/v1",
    "ollama": "http://localhost:11434",
}
RUN_TYPE_COPY = {
    RunType.SINGLE_PROMPT: (
        "Single prompt",
        "One structured generation call from source document to complete test suite.",
    ),
    RunType.STAGED_SINGLE_AGENT: (
        "Staged single agent",
        "One agent generates requirements, scenarios, and cases in sequence before deterministic review.",
    ),
    RunType.CENTRALIZED_MULTI_AGENT: (
        "Centralized multi-agent",
        "A coordinator delegates bounded work to three workers before deterministic review.",
    ),
}


def _env(name: str) -> str:
    if (value := os.getenv(name)) is not None:
        return value
    try:
        lines = Path(__file__).with_name(".env").read_text().splitlines()
    except FileNotFoundError:
        return ""
    prefix = f"{name}="
    return next(
        (line[len(prefix) :].strip().strip("'\"") for line in lines if line.startswith(prefix)),
        "",
    )


def _base_url(provider: str) -> str:
    if provider == "lm_studio":
        return _env("LM_STUDIO_BASE_URL") or LOCAL_BASE_URLS[provider]
    return LOCAL_BASE_URLS[provider]


@st.cache_resource
def _cached_repository(database_url: str) -> RunRepository:
    repository = RunRepository(database_url)
    repository.initialize()
    return repository


def _resolve_repository() -> RunRepository:
    injected = st.session_state.get("_repository")
    if injected is not None:
        if st.session_state.get("_initialized_repository") is not injected:
            injected.initialize()
            st.session_state["_initialized_repository"] = injected
        return injected
    return _cached_repository(_env("DATABASE_URL"))


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --color-accent: #2563eb;
            --color-background: #f8fafc;
            --color-surface: #ffffff;
            --color-foreground: #0f172a;
            --color-muted: #475569;
            --color-border: #dbe3ee;
            --color-focus: #1d4ed8;
        }
        .stApp {
            color: var(--color-foreground);
            font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
                "Segoe UI", sans-serif;
            background:
                radial-gradient(circle at 85% 0%, #eaf2ff 0, transparent 28rem),
                var(--color-background);
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 1180px;
            padding-top: 2.25rem;
            padding-bottom: 4rem;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--color-border);
            border-radius: 1rem;
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 8px 30px rgba(15, 23, 42, 0.04);
        }
        [data-testid="stMetric"] {
            min-height: 7rem;
            padding: 1rem 1.05rem;
            border: 1px solid var(--color-border);
            border-radius: 0.85rem;
            background: var(--color-surface);
        }
        [data-testid="stMetricLabel"] {
            color: var(--color-muted);
            font-weight: 600;
        }
        [data-testid="stMetricValue"] {
            color: var(--color-foreground);
            font-variant-numeric: tabular-nums;
        }
        .stButton > button[kind="primary"] {
            min-height: 3rem;
            border-radius: 0.7rem;
            border-color: var(--color-accent);
            background: var(--color-accent);
            font-weight: 700;
        }
        .stButton > button[kind="primary"]:hover {
            border-color: #1d4ed8;
            background: #1d4ed8;
        }
        .stDownloadButton > button {
            min-height: 2.75rem;
            border-radius: 0.65rem;
        }
        [data-testid="stFileUploader"] button,
        [data-testid="stTextInputRootElement"],
        [data-testid="stTextInputRootElement"] input,
        [data-testid="stSelectbox"] input,
        input[type="number"],
        button[aria-label="Show password"] {
            min-height: 2.75rem;
        }
        [data-testid="stSelectbox"] button,
        button[aria-label="Show password"],
        button[aria-label="Decrement"],
        button[aria-label="Increment"],
        button[aria-label^="Help for"] {
            min-height: 2.75rem;
            min-width: 2.75rem;
        }
        button:focus-visible,
        input:focus-visible {
            outline: 3px solid var(--color-focus) !important;
            outline-offset: 2px;
        }
        @media (max-width: 640px) {
            [data-testid="stMainBlockContainer"] {
                padding-top: 1rem;
            }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                scroll-behavior: auto !important;
                animation-duration: 0.01ms !important;
                transition-duration: 0.01ms !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _default_model(provider: str) -> str:
    if provider == "gemini":
        return GEMINI_DEFAULT_MODEL
    return "gemma4" if provider == "ollama" else ""


def _reset_provider() -> None:
    provider = st.session_state["settings_provider"]
    st.session_state["settings_model"] = _default_model(provider)
    st.session_state["settings_api_key"] = (
        _env("GEMINI_API_KEY")
        if provider == "gemini"
        else _env("LM_STUDIO_API_TOKEN") if provider == "lm_studio" else ""
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
        models = loader(
            st.session_state.get("settings_base_url", _base_url("lm_studio")),
            api_key,
        )
    except Exception as error:
        message = str(error)
        for secret in (
            api_key,
            st.session_state.get("settings_base_url", ""),
        ):
            if secret:
                message = message.replace(secret, "[REDACTED]")
        st.session_state["settings_model"] = ""
        st.session_state["lm_studio_models"] = []
        st.session_state["lm_studio_model_error"] = message
    else:
        st.session_state["lm_studio_models"] = models
        st.session_state["settings_model"] = models[0] if models else ""
        if models:
            st.session_state.pop("lm_studio_model_error", None)
        else:
            st.session_state["lm_studio_model_error"] = "No models were reported."


def _provider_label(provider: str) -> str:
    return PROVIDER_LABELS[provider]


def _run_type_label(run_type: RunType) -> str:
    return RUN_TYPE_COPY[run_type][0]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _download(label: str, value: Any, filename: str, key: str) -> None:
    st.download_button(
        label,
        data=_json(value),
        file_name=filename,
        mime="application/json",
        key=key,
        on_click="ignore",
        width="stretch",
    )


def _safe_error(error: Exception, settings: ProviderSettings) -> str:
    message = str(error)
    for secret in (settings.api_key, settings.base_url):
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message


def _failure_name(category: FailureCategory | None) -> str:
    return (
        category.value.replace("_", " ").title()
        if category is not None
        else "Unknown failure"
    )


def _failure_guidance(
    category: FailureCategory | None,
    message: str | None,
    *,
    provider: str,
) -> str:
    detail = (message or "").lower()
    if category is FailureCategory.PROVIDER_REJECTION:
        if provider == "gemini" and any(
            marker in detail
            for marker in ("404", "not_found", "not found", "no longer available")
        ):
            return (
                "The selected model is unavailable for this API key. "
                f"Use {GEMINI_DEFAULT_MODEL} and generate again."
            )
        if provider == "lm_studio":
            return (
                "LM Studio rejected the request. Check its API token, loaded "
                "model identifier, and server permissions, then run again."
            )
        return "Provider rejected the request. Check model access and credentials, then run again."
    if category is FailureCategory.TIMEOUT:
        return "The provider timed out. Check availability and retry when the service is responsive."
    if category is FailureCategory.TRANSPORT_EXHAUSTION:
        return "The provider could not be reached after retries. Check the connection or local server."
    if category is FailureCategory.BUDGET_EXHAUSTION:
        return "The run reached its token ceiling. Raise the ceiling or use a smaller document."
    if category is FailureCategory.SCHEMA_FAILURE:
        return "The model did not return valid structured data. Retry or choose a stronger model."
    if category is FailureCategory.SEMANTIC_VALIDATION:
        return "Generated artifacts did not pass traceability checks. Review the diagnostics before retrying."
    if category is FailureCategory.PARSING:
        cause = message or "The PDF could not be read."
        return f"{cause} Use a text-extractable PDF rather than a scanned image."
    if category is FailureCategory.CONFIGURATION:
        return "Review the provider settings and run configuration, then try again."
    return "Review the technical details, correct the configuration, and run again."


def _technical_detail(message: str | None) -> str:
    detail = message or "No technical detail was returned."
    return detail if len(detail) <= 800 else f"{detail[:797]}..."


def _result_status(result: RunResult) -> tuple[str, str]:
    if result.manifest.status is RunStatus.COMPLETED:
        return "Generation complete", "complete"
    if result.manifest.status is RunStatus.FAILED:
        return "Generation failed", "error"
    return "Generation interrupted", "error"


def _render_sources(sources) -> None:
    st.markdown("**Source references**")
    for source in sources:
        location = f"Page {source.page_number}"
        if source.section:
            location += f" · {source.section}"
        st.markdown(f"- `{source.chunk_id}` · {location} — {source.excerpt}")


def _render_bundle(result: RunResult, *, key_prefix: str) -> None:
    bundle = result.bundle
    if bundle is None:
        return

    st.markdown("### Generated artifacts")
    st.markdown("#### Requirements")
    for position, requirement in enumerate(bundle.requirements):
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

    st.markdown("#### Scenarios")
    for position, scenario in enumerate(bundle.scenarios):
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
                "\n".join(f"- {item}" for item in scenario.preconditions) or "None"
            )
            _render_sources(scenario.source_references)

    st.markdown("#### Test cases")
    for position, test_case in enumerate(bundle.test_cases):
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
                "\n".join(f"- {item}" for item in test_case.preconditions) or "None"
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


def _render_result(result: RunResult, *, key_prefix: str = "result") -> None:
    manifest = result.manifest
    label, state = _result_status(result)
    with st.status(label, state=state, expanded=False):
        st.write(
            f"{_run_type_label(manifest.run_type)} finished with status: "
            f"{manifest.status.value}."
        )
    st.caption(
        f"{_run_type_label(manifest.run_type)} · {_provider_label(manifest.provider)} · "
        f"{manifest.model} · {manifest.source_filename} · "
        f"Temperature {manifest.temperature:g} · "
        f"Token ceiling {manifest.token_ceiling:,} · Run {manifest.run_id}"
    )

    if manifest.failure_category is not None or manifest.failure_message:
        category = _failure_name(manifest.failure_category)
        st.error(
            f"**{category}.** "
            + _failure_guidance(
                manifest.failure_category,
                manifest.failure_message,
                provider=manifest.provider,
            )
        )
        with st.expander(
            "Technical details",
            key=f"{key_prefix}-{manifest.run_id}-technical-details",
        ):
            st.code(_technical_detail(manifest.failure_message), language=None)
    elif manifest.status is RunStatus.COMPLETED:
        st.success("Completed and validated")

    metrics = result.metrics
    if metrics is not None:
        charged = metrics.charged_tokens
        token_label = "Charged tokens" if charged else "Reported tokens"
        token_value = charged or metrics.input_tokens + metrics.output_tokens
        columns = st.columns(4)
        columns[0].metric("Requirements", metrics.requirement_count)
        columns[1].metric("Scenarios", metrics.scenario_count)
        columns[2].metric("Test cases", metrics.test_case_count)
        columns[3].metric(token_label, f"{token_value:,}")
        with st.expander(
            "Quality and traceability",
            key=f"{key_prefix}-{manifest.run_id}-quality",
        ):
            st.table(
                [
                    {
                        "Measure": "Citation coverage",
                        "Result": f"{metrics.citation_coverage:.0%}",
                    },
                    {
                        "Measure": "Requirement → scenario",
                        "Result": f"{metrics.requirement_scenario_coverage:.0%}",
                    },
                    {
                        "Measure": "Requirement → test case",
                        "Result": f"{metrics.requirement_test_case_coverage:.0%}",
                    },
                    {
                        "Measure": "Positive scenario coverage",
                        "Result": f"{metrics.positive_scenario_coverage:.0%}",
                    },
                    {
                        "Measure": "Non-positive scenario coverage",
                        "Result": f"{metrics.non_positive_scenario_coverage:.0%}",
                    },
                    {
                        "Measure": "RTM completeness",
                        "Result": f"{metrics.rtm_completeness:.0%}",
                    },
                ]
            )
            st.caption(
                f"Latency {metrics.latency_seconds:.2f} s · "
                f"{metrics.retries} retries"
            )

    if manifest.status is not RunStatus.COMPLETED:
        _download(
            "Download diagnostics",
            result.download_bundle(),
            f"{manifest.run_id}-diagnostics.json",
            f"{key_prefix}-{manifest.run_id}-diagnostics",
        )
    elif result.bundle is not None:
        download_columns = st.columns(2)
        with download_columns[0]:
            _download(
                "Download traceability matrix",
                [item.model_dump(mode="json") for item in result.rtm],
                f"{manifest.run_id}-rtm.json",
                f"{key_prefix}-{manifest.run_id}-rtm",
            )
        with download_columns[1]:
            _download(
                "Download complete bundle",
                result.download_bundle(),
                f"{manifest.run_id}-bundle.json",
                f"{key_prefix}-{manifest.run_id}-bundle",
            )

    if result.bundle is not None:
        _render_bundle(result, key_prefix=key_prefix)


def _fallback_settings() -> AppSettings:
    return AppSettings(
        provider="gemini",
        model=GEMINI_DEFAULT_MODEL,
        api_key=_env("GEMINI_API_KEY"),
        base_url="",
        token_ceiling=200_000,
    )


def _warn_once(key: str, message: str) -> None:
    if not st.session_state.get(key):
        st.session_state[key] = True
        st.session_state["flash_warning"] = message


def _sync_app_settings() -> None:
    fallback = _fallback_settings()
    st.session_state.setdefault("app_settings", fallback)
    revision = st.session_state.get("settings_revision", 0)
    pending = st.session_state.get("settings_save_request")
    result = sync_browser_settings(save=pending, revision=revision)
    st.session_state["browser_settings_loaded"] = (
        result.loaded and result.revision == revision
    )
    if not st.session_state["browser_settings_loaded"]:
        return
    if st.session_state.get("settings_loaded") and pending is None:
        return

    if result.error:
        _warn_once(
            "browser_storage_warning_shown",
            "Browser settings storage is unavailable. Using app defaults for this session.",
        )
        st.session_state["settings_loaded"] = True
        st.session_state.pop("settings_save_request", None)
        st.session_state.pop("settings_after_persist", None)
        return

    settings, warning = parse_settings(result.payload, fallback)
    if warning:
        _warn_once("invalid_settings_warning_shown", warning)
    if pending is not None and result.payload != pending:
        _warn_once(
            "browser_save_warning_shown",
            "Browser settings could not be confirmed; the previous settings remain active.",
        )
    else:
        st.session_state["app_settings"] = settings
        if pending is not None and (
            destination := st.session_state.get("settings_after_persist")
        ):
            st.session_state["view"] = destination
    st.session_state["settings_loaded"] = True
    st.session_state.pop("settings_save_request", None)
    st.session_state.pop("settings_after_persist", None)


def _settings_are_ready() -> bool:
    try:
        st.session_state["app_settings"].provider_settings()
    except ValueError:
        return False
    return True


def _open_settings(after_save: str | None = None) -> None:
    settings = st.session_state["app_settings"]
    st.session_state["settings_provider"] = settings.provider
    st.session_state["settings_model"] = settings.model
    st.session_state["settings_api_key"] = settings.api_key
    st.session_state["settings_base_url"] = settings.base_url
    st.session_state["settings_token_ceiling"] = settings.token_ceiling
    st.session_state["show_settings"] = True
    if after_save is None:
        st.session_state.pop("settings_after_persist", None)
    else:
        st.session_state["settings_after_persist"] = after_save


@st.dialog("App settings", width="large")
def _settings_dialog() -> None:
    provider = st.selectbox(
        "Provider",
        list(PROVIDER_LABELS),
        key="settings_provider",
        format_func=_provider_label,
        on_change=_reset_provider,
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
        st.text_input(
            "LM Studio API token",
            type="password",
            key="settings_api_key",
            on_change=_clear_lm_studio_models,
        )
    else:
        st.text_input("Model", key="settings_model")
        if provider == "gemini":
            st.text_input(
                "Gemini API key", type="password", key="settings_api_key"
            )

    if provider in LOCAL_BASE_URLS:
        st.text_input(
            f"{_provider_label(provider)} base URL",
            key="settings_base_url",
            on_change=_clear_lm_studio_models if provider == "lm_studio" else None,
        )
    if provider == "lm_studio":
        st.button(
            "Load available models",
            on_click=_refresh_lm_studio_models,
            width="stretch",
        )
        if error := st.session_state.get("lm_studio_model_error"):
            st.error(f"Could not load models: {error}")
        elif models:
            st.success(f"Loaded {len(models)} models.")

    st.number_input(
        "Token ceiling",
        min_value=1000,
        step=1000,
        key="settings_token_ceiling",
    )
    st.warning(
        "Credentials are stored in this browser's local storage. Use a trusted device."
    )
    save_column, cancel_column = st.columns(2)
    save = save_column.button("Save", type="primary", width="stretch")
    cancel = cancel_column.button("Cancel", width="stretch")
    if cancel:
        st.session_state["show_settings"] = False
        st.session_state.pop("settings_after_persist", None)
        st.rerun()
    if not save:
        return

    try:
        settings = AppSettings(
            provider=provider,
            model=st.session_state["settings_model"],
            api_key=st.session_state.get("settings_api_key", ""),
            base_url=st.session_state.get("settings_base_url", ""),
            token_ceiling=st.session_state["settings_token_ceiling"],
        )
        settings.provider_settings()
    except ValueError as error:
        st.error(str(error))
        return
    st.session_state["settings_revision"] = (
        st.session_state.get("settings_revision", 0) + 1
    )
    st.session_state["settings_save_request"] = settings.model_dump(mode="json")
    st.session_state["show_settings"] = False
    st.rerun()


def _go_home() -> None:
    st.session_state["view"] = "runs"
    st.session_state.pop("selected_run_id", None)
    st.session_state.pop("selected_run", None)
    st.session_state.pop("runs-table", None)
    st.session_state.pop("displayed_run_ids", None)


def _render_top_nav() -> None:
    home, _, settings = st.columns([3, 6, 1])
    home.button("BRD/SRS Test Case", on_click=_go_home)
    settings.button("Settings", on_click=_open_settings, width="stretch")


def _request_create() -> None:
    if st.session_state.get("settings_save_request") is not None:
        st.session_state["runs_notice"] = "Saving browser settings…"
    elif not st.session_state.get("browser_settings_loaded"):
        st.session_state["runs_notice"] = "Browser settings are still loading."
    elif _settings_are_ready():
        st.session_state["view"] = "create"
    else:
        _open_settings("create")


def _render_runs(repository: RunRepository) -> None:
    selection = st.session_state.get("runs-table", {})
    selected_rows = (
        selection.get("selection", {}).get("rows", [])
        if hasattr(selection, "get")
        else []
    )
    displayed_run_ids = st.session_state.get("displayed_run_ids", [])
    if selected_rows and 0 <= selected_rows[0] < len(displayed_run_ids):
        st.session_state["selected_run_id"] = displayed_run_ids[selected_rows[0]]
        st.session_state.pop("selected_run", None)
        st.session_state["view"] = "detail"
        st.rerun()

    st.title("Runs")
    st.caption("Create a run or open a saved result.")
    st.button(
        "Create new run",
        type="primary",
        on_click=_request_create,
    )
    if notice := st.session_state.pop("runs_notice", None):
        st.info(notice)
    try:
        runs = repository.list_runs()
    except StorageError:
        st.session_state.pop("displayed_run_ids", None)
        st.error(
            "Saved run history is unavailable. Check PostgreSQL and DATABASE_URL, "
            "then refresh this page."
        )
        return
    if not runs:
        st.session_state.pop("displayed_run_ids", None)
        st.info("No saved runs yet.")
        return

    st.session_state["displayed_run_ids"] = [item.run_id for item in runs]
    rows = [
        {
            "Started": item.started_at.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "Source": item.source_filename,
            "Run type": _run_type_label(item.run_type),
            "Provider": _provider_label(item.provider),
            "Model": item.model,
            "Status": item.display_status,
            "Test cases": str(item.test_case_count)
            if item.test_case_count is not None
            else "—",
        }
        for item in runs
    ]
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        selection_mode="single-row",
        on_select="rerun",
        key="runs-table",
    )


def _render_create() -> None:
    st.button("Back to runs", on_click=_go_home)
    st.title("Create a run")
    st.caption("Add a source document and choose one generation strategy.")
    st.file_uploader(
        "BRD/SRS PDF",
        type=["pdf"],
        key="pdf",
        help="Use a text-extractable PDF.",
    )
    run_type = st.selectbox(
        "Run type",
        list(RunType),
        key="run_type",
        format_func=_run_type_label,
    )
    st.caption(RUN_TYPE_COPY[run_type][1])
    settings = st.session_state["app_settings"]
    st.markdown("### App settings")
    st.caption(
        f"{_provider_label(settings.provider)} · {settings.model} · "
        f"{settings.token_ceiling:,} token ceiling"
    )
    st.button("Edit settings", on_click=_open_settings, args=("create",))


def _render_detail(repository: RunRepository) -> None:
    st.button("Back to runs", on_click=_go_home)
    run_id = st.session_state.get("selected_run_id")
    if not isinstance(run_id, str) or not run_id:
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


def _render_flashes() -> None:
    if warning := st.session_state.pop("flash_warning", None):
        st.warning(warning)
    if error := st.session_state.pop("flash_error", None):
        st.error(error)


def main() -> None:
    st.set_page_config(
        page_title="BRD/SRS Test Case",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _apply_theme()
    _sync_app_settings()
    _render_top_nav()
    _render_flashes()

    try:
        repository = _resolve_repository()
    except StorageError:
        st.error(
            "Run history database is unavailable. Start it with "
            "`docker compose up -d db`, verify DATABASE_URL, and refresh this page."
        )
        st.stop()

    st.session_state.setdefault("view", "runs")
    view = st.session_state["view"]
    if view == "create":
        _render_create()
    elif view == "detail":
        _render_detail(repository)
    else:
        st.session_state["view"] = "runs"
        _render_runs(repository)

    if st.session_state.get("show_settings"):
        _settings_dialog()


main()
