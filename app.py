from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

from brd_srs_testgen.models import (
    FailureCategory,
    RunHistoryItem,
    RunResult,
    RunStatus,
    RunType,
)
from brd_srs_testgen.providers import list_lm_studio_models
from brd_srs_testgen.runner import ProviderSettings, run_generation
from brd_srs_testgen.storage import RunRepository, StorageError


GEMINI_DEFAULT_MODEL = "gemini-3.6-flash"
RETIRED_GEMINI_DEFAULTS = {"gemini-2.5-flash"}
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
            --color-primary: #1e293b;
            --color-on-primary: #ffffff;
            --color-accent: #2563eb;
            --color-background: #f8fafc;
            --color-surface: #ffffff;
            --color-foreground: #0f172a;
            --color-muted: #475569;
            --color-soft: #eff6ff;
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
        .research-hero {
            padding: 2.25rem;
            margin-bottom: 1.5rem;
            border: 1px solid var(--color-border);
            border-radius: 1.25rem;
            background: linear-gradient(135deg, #ffffff 0%, #f4f8ff 100%);
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
        }
        .research-eyebrow {
            margin: 0 0 0.75rem;
            color: var(--color-accent);
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .research-hero h1 {
            max-width: 760px;
            margin: 0;
            color: var(--color-foreground);
            font-size: clamp(2rem, 4vw, 3.35rem);
            line-height: 1.04;
            letter-spacing: -0.04em;
        }
        .research-hero p {
            max-width: 720px;
            margin: 1rem 0 0;
            color: var(--color-muted);
            font-size: 1.05rem;
            line-height: 1.65;
        }
        .protocol-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1.35rem;
        }
        .protocol-strip span {
            padding: 0.45rem 0.7rem;
            border: 1px solid #cbdcf8;
            border-radius: 999px;
            color: #1e3a5f;
            background: #f7faff;
            font-size: 0.82rem;
            font-weight: 600;
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
        :is(.st-key-workflow_steps_empty, .st-key-workflow_steps_with_result)
        :is([role="tablist"], [data-baseweb="tab-list"]) {
            gap: 0.5rem;
            padding: 0.4rem;
            border: 1px solid var(--color-border);
            border-radius: 1rem;
            background: #eef2f7;
            box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.05);
        }
        :is(.st-key-workflow_steps_empty, .st-key-workflow_steps_with_result)
        :is([role="tab"], [data-baseweb="tab"]) {
            flex: 1 1 0;
            min-height: 3.5rem;
            padding-inline: 1rem;
            border-radius: 0.7rem;
            color: var(--color-muted);
            font-weight: 650;
            transition: color 180ms ease, background 180ms ease,
                box-shadow 180ms ease;
        }
        :is(.st-key-workflow_steps_empty, .st-key-workflow_steps_with_result)
        [aria-selected="true"] {
            color: var(--color-accent) !important;
            background: var(--color-surface);
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.1);
        }
        :is(.st-key-workflow_steps_empty, .st-key-workflow_steps_with_result)
        :is(.react-aria-SelectionIndicator, [data-baseweb="tab-highlight"],
            [data-baseweb="tab-border"]) {
            display: none;
        }
        :is(.st-key-workflow_steps_empty, .st-key-workflow_steps_with_result)
        :is([role="tabpanel"], [data-baseweb="tab-panel"]) {
            padding-top: 1.25rem;
        }
        .step-heading {
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .step-index {
            display: grid;
            flex: 0 0 2.75rem;
            width: 2.75rem;
            height: 2.75rem;
            place-items: center;
            border-radius: 0.8rem;
            color: var(--color-on-primary);
            background: var(--color-accent);
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.22);
            font-size: 1rem;
            font-weight: 800;
        }
        .step-heading h2 {
            margin: 0;
            color: var(--color-foreground);
            font-size: 1.35rem;
            letter-spacing: -0.02em;
        }
        .step-heading p {
            max-width: 700px;
            margin: 0.3rem 0 0;
            color: var(--color-muted);
            line-height: 1.55;
        }
        button:focus-visible,
        input:focus-visible,
        [role="tab"]:focus-visible {
            outline: 3px solid var(--color-focus) !important;
            outline-offset: 2px;
        }
        @media (max-width: 640px) {
            [data-testid="stMainBlockContainer"] {
                padding-top: 1rem;
            }
            .research-hero {
                padding: 1.5rem 1.25rem;
                border-radius: 1rem;
            }
            .research-hero p {
                font-size: 1rem;
            }
            :is(.st-key-workflow_steps_empty, .st-key-workflow_steps_with_result)
            :is([role="tab"], [data-baseweb="tab"]) {
                min-height: 3.25rem;
                padding-inline: 0.45rem;
                font-size: 0.82rem;
            }
            .step-heading {
                gap: 0.75rem;
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
    provider = st.session_state["provider"]
    st.session_state["model"] = _default_model(provider)
    st.session_state.pop("lm_studio_models", None)
    st.session_state.pop("lm_studio_model_error", None)
    if provider in LOCAL_BASE_URLS:
        st.session_state["base_url"] = _base_url(provider)
    if provider == "lm_studio":
        _refresh_lm_studio_models()


def _clear_lm_studio_models() -> None:
    st.session_state["model"] = ""
    st.session_state.pop("lm_studio_models", None)
    st.session_state.pop("lm_studio_model_error", None)


def _refresh_lm_studio_models() -> None:
    api_key = st.session_state.get("lm_studio_api_key", "") or _env(
        "LM_STUDIO_API_TOKEN"
    )
    try:
        loader = st.session_state.get("_model_loader", list_lm_studio_models)
        models = loader(
            st.session_state.get("base_url", _base_url("lm_studio")),
            api_key,
        )
    except Exception as error:
        message = str(error)
        if api_key:
            message = message.replace(api_key, "[REDACTED]")
        st.session_state["model"] = ""
        st.session_state["lm_studio_models"] = []
        st.session_state["lm_studio_model_error"] = message
    else:
        st.session_state["lm_studio_models"] = models
        st.session_state["model"] = models[0]
        st.session_state.pop("lm_studio_model_error", None)


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


def _render_empty_state() -> None:
    st.markdown("### No results yet")
    st.caption("Complete the configuration and run steps to populate this workspace.")
    st.info(
        "Upload one text-extractable PDF, choose one run type, and generate detailed test cases."
    )


def _history_label(item: RunHistoryItem) -> str:
    started = item.started_at.strftime("%Y-%m-%d %H:%M %Z")
    return (
        f"{started} · {item.source_filename} · {_run_type_label(item.run_type)} "
        f"· {item.run_id[-8:]}"
    )


def _render_history(repository: RunRepository) -> None:
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

    st.table(
        [
            {
                "Started": item.started_at.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "Source": item.source_filename,
                "Run type": _run_type_label(item.run_type),
                "Provider": _provider_label(item.provider),
                "Model": item.model,
                "Status": item.display_status,
                "Requirements": str(item.requirement_count)
                if item.requirement_count is not None
                else "—",
                "Scenarios": str(item.scenario_count)
                if item.scenario_count is not None
                else "—",
                "Test cases": str(item.test_case_count)
                if item.test_case_count is not None
                else "—",
            }
            for item in runs
        ]
    )
    selected = st.selectbox(
        "Open saved run",
        runs,
        index=None,
        format_func=_history_label,
        key="history_run_id",
        placeholder="Select a saved run",
    )
    if selected is None:
        return
    try:
        result = repository.load_run(selected.run_id)
    except StorageError:
        st.error(
            "Saved run could not be opened. Check PostgreSQL and DATABASE_URL, "
            "then try again."
        )
        return
    _render_result(result, key_prefix="history")


def main() -> None:
    st.set_page_config(
        page_title="BRD/SRS Test-Case Research Core",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _apply_theme()
    st.markdown(
        """
        <section class="research-hero">
            <p class="research-eyebrow">Traceable test-case generation</p>
            <h1>Generate detailed test cases from one controlled run.</h1>
            <p>
                Turn a text-extractable BRD or SRS into requirements, scenarios,
                test cases, and a traceability matrix with the generation strategy
                that fits this run.
            </p>
            <div class="protocol-strip" aria-label="Generation protocol">
                <span>1 selected run type</span>
                <span>Temperature 0</span>
                <span>Traceability validation</span>
                <span>Detailed artifacts</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    try:
        repository = _resolve_repository()
    except StorageError:
        st.error(
            "Run history database is unavailable. Start it with "
            "`docker compose up -d db`, verify DATABASE_URL, and refresh this page."
        )
        st.stop()

    current_provider = st.session_state.get("provider", "gemini")
    if (
        "model" not in st.session_state
        or (
            current_provider == "gemini"
            and st.session_state["model"] in RETIRED_GEMINI_DEFAULTS
        )
    ):
        st.session_state["model"] = _default_model(current_provider)

    existing_result = st.session_state.get("run_result")
    step_labels = ["1 · Configure", "2 · Run", "3 · Results", "4 · Run history"]
    configure_tab, run_tab, results_tab, history_tab = st.tabs(
        step_labels,
        default=(
            step_labels[2]
            if isinstance(existing_result, RunResult)
            else step_labels[0]
        ),
        key=(
            "workflow_steps_with_result"
            if isinstance(existing_result, RunResult)
            else "workflow_steps_empty"
        ),
    )

    with configure_tab:
        with st.container(border=True):
            st.markdown(
                """
                <div class="step-heading">
                    <span class="step-index" aria-hidden="true">1</span>
                    <div>
                        <h2>Configure the generation engine</h2>
                        <p>Choose the provider, model, credentials, and shared run limits.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Credentials stay in memory and are never written to run artifacts.")
            provider_column, credential_column = st.columns(2, gap="large")
            with provider_column:
                provider = st.selectbox(
                    "Provider",
                    ["gemini", "lm_studio", "ollama"],
                    key="provider",
                    on_change=_reset_provider,
                    format_func=_provider_label,
                )
                if provider == "lm_studio":
                    available_models = st.session_state.get("lm_studio_models", [])
                    model = (
                        st.selectbox(
                            "Model",
                            available_models,
                            index=0 if available_models else None,
                            key="model",
                            placeholder="Load models or enter a model ID",
                            accept_new_options=True,
                            help="Models reported by LM Studio's OpenAI-compatible API.",
                        )
                        or ""
                    )
                    if not available_models:
                        st.caption(
                            "Enter the server token and load available models, "
                            "or type the exact model ID."
                        )
                else:
                    model = st.text_input(
                        "Model",
                        key="model",
                        help=(
                            "Current stable Gemini default."
                            if provider == "gemini"
                            else "The model must already be available in Ollama."
                        ),
                    )

            api_key = ""
            base_url = _base_url(provider) if provider in LOCAL_BASE_URLS else ""
            with credential_column:
                if provider == "gemini":
                    api_key = st.text_input(
                        "Gemini API key",
                        type="password",
                        value=_env("GEMINI_API_KEY"),
                        key="gemini_api_key",
                        help="Used in memory for provider requests and never persisted in run artifacts.",
                    )
                elif provider == "lm_studio":
                    api_key = st.text_input(
                        "LM Studio API token",
                        type="password",
                        value=_env("LM_STUDIO_API_TOKEN"),
                        key="lm_studio_api_key",
                        help="Required only when authentication is enabled in LM Studio Server Settings.",
                        on_change=_clear_lm_studio_models,
                    )

                if provider in LOCAL_BASE_URLS:
                    st.session_state.setdefault("base_url", base_url)
                    base_url = st.text_input(
                        f"{_provider_label(provider)} base URL",
                        key="base_url",
                        help=(
                            "OpenAI-compatible base URL, including /v1."
                            if provider == "lm_studio"
                            else "Start Ollama locally before generating test cases."
                        ),
                        on_change=(
                            _clear_lm_studio_models
                            if provider == "lm_studio"
                            else None
                        ),
                    )
                if provider == "lm_studio":
                    st.button(
                        "Load available models",
                        key="load_lm_studio_models",
                        on_click=_refresh_lm_studio_models,
                        width="stretch",
                    )
                    if error := st.session_state.get("lm_studio_model_error"):
                        st.error(f"Could not load models: {error}")
                    elif st.session_state.get("lm_studio_models"):
                        st.success(
                            f"Loaded {len(st.session_state['lm_studio_models'])} models."
                        )

            with st.expander("Advanced run controls"):
                token_ceiling = st.number_input(
                    "Token ceiling",
                    min_value=1000,
                    value=200_000,
                    step=1000,
                    key="token_ceiling",
                    help="Maximum provider tokens charged to this run.",
                )
                st.caption("Temperature is fixed at 0 for reproducible output.")

            run_type = st.selectbox(
                "Run type",
                list(RunType),
                key="run_type",
                format_func=_run_type_label,
                help=(
                    "Choose one generation strategy for this run. "
                    "Only the selected strategy executes."
                ),
            )
            st.caption(RUN_TYPE_COPY[run_type][1])

    with run_tab:
        with st.container(border=True):
            st.markdown(
                """
                <div class="step-heading">
                    <span class="step-index" aria-hidden="true">2</span>
                    <div>
                        <h2>Add the source and generate</h2>
                        <p>Upload one text-extractable BRD or SRS, then run the selected strategy.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            document_column, protocol_column = st.columns([1.1, 0.9], gap="large")
            with document_column:
                uploaded = st.file_uploader(
                    "BRD/SRS PDF",
                    type=["pdf"],
                    key="pdf",
                    help="Use a text-extractable PDF. Scanned image-only files require OCR and are not supported.",
                )
                if uploaded is not None:
                    st.success(
                        f"Ready: {uploaded.name} · {uploaded.size / 1024:.1f} KB"
                    )

            with protocol_column:
                with st.container(border=True):
                    st.markdown("#### Selected run")
                    st.markdown(f"**{_run_type_label(run_type)}**")
                    st.caption(RUN_TYPE_COPY[run_type][1])
                    st.caption(
                        f"{_provider_label(provider)} · {model} · "
                        f"{token_ceiling:,} token ceiling"
                    )

            run_clicked = st.button(
                "Generate test cases",
                type="primary",
                key="run",
                width="stretch",
            )

        if run_clicked:
            st.session_state.pop("run_result", None)
            if uploaded is None:
                st.error("Upload one text-extractable PDF before generating test cases.")
            elif provider == "gemini" and not api_key.strip():
                st.error("Enter a Gemini API key before generating test cases.")
            else:
                settings = ProviderSettings(
                    provider=provider,
                    model=model,
                    token_ceiling=token_ceiling,
                    api_key=api_key,
                    base_url=base_url,
                )
                try:
                    settings.validate()
                    with st.status("Preparing generation", expanded=True) as status:

                        def progress(message: str) -> None:
                            status.write(message)

                        runner = st.session_state.get("_runner", run_generation)
                        result = runner(
                            uploaded.getvalue(),
                            uploaded.name,
                            run_type,
                            settings,
                            repository=repository,
                            progress=progress,
                        )
                        st.session_state["run_result"] = result
                        label, state = _result_status(result)
                        status.update(
                            label=label,
                            state=state,
                            expanded=state == "error",
                        )
                    st.rerun()
                except Exception as error:
                    st.error(f"Generation failed: {_safe_error(error, settings)}")

    with results_tab:
        with st.container(border=True):
            st.markdown(
                """
                <div class="step-heading">
                    <span class="step-index" aria-hidden="true">3</span>
                    <div>
                        <h2>Review and export generated artifacts</h2>
                        <p>Inspect requirements, scenarios, test cases, traceability, and run diagnostics.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            result = st.session_state.get("run_result")
            if isinstance(result, RunResult):
                if (
                    result.manifest.provider != provider
                    or result.manifest.model != model
                    or result.manifest.token_ceiling != token_ceiling
                    or result.manifest.run_type != run_type
                ):
                    st.info(
                        "The setup has changed since this result was created. "
                        "Generate again to refresh it."
                    )
                _render_result(result, key_prefix="current")
            else:
                _render_empty_state()

    with history_tab:
        with st.container(border=True):
            st.markdown(
                """
                <div class="step-heading">
                    <span class="step-index" aria-hidden="true">4</span>
                    <div>
                        <h2>Browse saved runs</h2>
                        <p>Compare persisted run settings and open complete generated artifacts.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            _render_history(repository)


main()
