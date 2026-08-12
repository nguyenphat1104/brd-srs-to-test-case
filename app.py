from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

from brd_srs_testgen.models import Condition, FailureCategory, RunStatus
from brd_srs_testgen.providers import list_lm_studio_models
from brd_srs_testgen.runner import (
    ComparisonResult,
    ConditionResult,
    ProviderSettings,
    run_comparison,
)


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
CONDITION_COPY = {
    Condition.SINGLE_PROMPT: (
        "Single prompt",
        "One structured generation call from source document to complete test suite.",
    ),
    Condition.STAGED_SINGLE_AGENT: (
        "Staged single agent",
        "One agent generates requirements, scenarios, and cases in sequence before deterministic review.",
    ),
    Condition.CENTRALIZED_MULTI_AGENT: (
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
        st.session_state["base_url"] = LOCAL_BASE_URLS[provider]
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
            st.session_state.get("base_url", LOCAL_BASE_URLS["lm_studio"]),
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


def _condition_label(condition: Condition) -> str:
    return CONDITION_COPY[condition][0]


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
                f"Use {GEMINI_DEFAULT_MODEL} and run the comparison again."
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


def _failed_count(result: ComparisonResult) -> int:
    return sum(
        condition_result is None
        or condition_result.manifest.status is not RunStatus.COMPLETED
        for condition in result.manifest.condition_order
        for condition_result in [result.conditions.get(condition)]
    )


def _result_status(result: ComparisonResult) -> tuple[str, str]:
    if result.failure_category:
        return "Comparison failed", "error"
    failed = _failed_count(result)
    total = len(result.manifest.condition_order)
    if failed == total:
        return f"All {total} conditions failed", "error"
    if failed:
        noun = "condition" if failed == 1 else "conditions"
        return f"Finished with {failed} failed {noun}", "error"
    return "Comparison complete", "complete"


def _summary_rows(result: ComparisonResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in result.manifest.condition_order:
        condition_result = result.conditions.get(condition)
        if condition_result is None:
            rows.append(
                {
                    "Condition": _condition_label(condition),
                    "Status": "No result",
                    "Requirements": None,
                    "Scenarios": None,
                    "Test cases": None,
                    "Citation": "—",
                    "RTM": "—",
                    "Tokens": None,
                    "Latency": "—",
                }
            )
            continue
        manifest = condition_result.manifest
        metrics = condition_result.metrics
        completed = manifest.status is RunStatus.COMPLETED
        tokens = metrics.charged_tokens or metrics.input_tokens + metrics.output_tokens
        rows.append(
            {
                "Condition": _condition_label(condition),
                "Status": "Completed" if completed else _failure_name(manifest.failure_category),
                "Requirements": metrics.requirement_count if completed else None,
                "Scenarios": metrics.scenario_count if completed else None,
                "Test cases": metrics.test_case_count if completed else None,
                "Citation": f"{metrics.citation_coverage:.0%}" if completed else "—",
                "RTM": f"{metrics.rtm_completeness:.0%}" if completed else "—",
                "Tokens": tokens,
                "Latency": f"{metrics.latency_seconds:.2f} s",
            }
        )
    return rows


def _render_condition(result: ConditionResult) -> None:
    manifest = result.manifest
    metrics = result.metrics
    condition = manifest.condition
    label, description = CONDITION_COPY[condition]
    st.markdown(f"### {label}")
    st.caption(description)

    charged = metrics.charged_tokens
    token_label = "Charged tokens" if charged else "Reported tokens"
    token_value = charged or metrics.input_tokens + metrics.output_tokens
    if manifest.status is not RunStatus.COMPLETED:
        category = _failure_name(manifest.failure_category)
        st.error(
            f"**{category}.** "
            + _failure_guidance(
                manifest.failure_category,
                manifest.failure_message,
                provider=manifest.provider,
            )
        )
        with st.expander("Technical details"):
            st.code(_technical_detail(manifest.failure_message), language=None)
        detail_columns = st.columns(3)
        detail_columns[0].metric(token_label, f"{token_value:,}")
        detail_columns[1].metric("Retries", metrics.retries)
        detail_columns[2].metric("Latency", f"{metrics.latency_seconds:.2f} s")
        _download(
            "Download diagnostics",
            result.download_bundle(),
            f"{condition.value}-diagnostics.json",
            f"{condition.value}-diagnostics",
        )
        return

    st.success("Completed and validated")
    volume_columns = st.columns(4)
    volume_columns[0].metric("Requirements", metrics.requirement_count)
    volume_columns[1].metric("Scenarios", metrics.scenario_count)
    volume_columns[2].metric("Test cases", metrics.test_case_count)
    volume_columns[3].metric(token_label, f"{token_value:,}")

    st.markdown("#### Quality and traceability")
    st.table(
        [
            {"Measure": "Citation coverage", "Result": f"{metrics.citation_coverage:.0%}"},
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
            {"Measure": "RTM completeness", "Result": f"{metrics.rtm_completeness:.0%}"},
        ]
    )
    st.caption(f"Latency {metrics.latency_seconds:.2f} s · {metrics.retries} retries")

    with st.expander("Artifacts and downloads"):
        bundle_columns = st.columns(2)
        with bundle_columns[0]:
            _download(
                "Traceability matrix",
                [item.model_dump(mode="json") for item in result.rtm],
                f"{condition.value}-rtm.json",
                f"{condition.value}-rtm",
            )
        with bundle_columns[1]:
            _download(
                "Complete condition bundle",
                result.download_bundle(),
                f"{condition.value}-bundle.json",
                f"{condition.value}-bundle",
            )


def _render_result(result: ComparisonResult) -> None:
    label, state = _result_status(result)
    with st.status(label, state=state, expanded=False):
        st.write("The comparison run has finished. Review the summary and condition details below.")
    st.caption(
        f"{result.manifest.provider.title()} · {result.manifest.model} · "
        f"Run {result.manifest.comparison_id}"
    )
    if result.failure_category:
        category = _failure_name(result.failure_category)
        st.error(
            f"**{category}.** "
            + _failure_guidance(
                result.failure_category,
                result.failure_message,
                provider=result.manifest.provider,
            )
        )
        with st.expander("Technical details"):
            st.code(_technical_detail(result.failure_message), language=None)
        return

    failed = _failed_count(result)
    total = len(result.manifest.condition_order)
    if failed == total:
        first = next(iter(result.conditions.values()), None)
        category = first.manifest.failure_category if first else None
        message = first.manifest.failure_message if first else None
        st.error(
            f"**All {total} conditions failed.** "
            + _failure_guidance(
                category,
                message,
                provider=result.manifest.provider,
            )
        )
    elif failed:
        noun = "condition" if failed == 1 else "conditions"
        st.warning(
            f"{total - failed} of {total} conditions completed; "
            f"{failed} {noun} failed. Successful artifacts remain available."
        )
    else:
        st.success(f"All {total} conditions completed and passed validation.")

    st.markdown("### At a glance")
    st.dataframe(
        _summary_rows(result),
        hide_index=True,
        width="stretch",
    )

    tabs = st.tabs(
        [_condition_label(condition) for condition in result.manifest.condition_order]
    )
    for tab, condition in zip(
        tabs, result.manifest.condition_order, strict=True
    ):
        with tab:
            condition_result = result.conditions.get(condition)
            if condition_result is None:
                st.error("No result was returned for this condition.")
            else:
                _render_condition(condition_result)


def _render_empty_state() -> None:
    st.markdown("### No results yet")
    st.caption("Complete the configuration and run steps to populate this workspace.")
    st.markdown("#### What this comparison produces")
    columns = st.columns(3)
    for column, condition in zip(columns, Condition, strict=True):
        label, description = CONDITION_COPY[condition]
        with column:
            with st.container(border=True):
                st.markdown(f"#### {label}")
                st.caption(description)
    st.info(
        "Upload one text-extractable PDF, confirm the provider, and run the fixed three-condition protocol."
    )


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
            <p class="research-eyebrow">Evidence-led generation study</p>
            <h1>Compare test-case generation strategies with one controlled run.</h1>
            <p>
                Turn a text-extractable BRD or SRS into requirements, scenarios,
                test cases, and a traceability matrix—then compare three fixed
                generation conditions side by side.
            </p>
            <div class="protocol-strip" aria-label="Fixed comparison protocol">
                <span>3 fixed conditions</span>
                <span>Temperature 0</span>
                <span>Traceability validation</span>
                <span>Isolated failures</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    current_provider = st.session_state.get("provider", "gemini")
    if (
        "model" not in st.session_state
        or (
            current_provider == "gemini"
            and st.session_state["model"] in RETIRED_GEMINI_DEFAULTS
        )
    ):
        st.session_state["model"] = _default_model(current_provider)

    existing_result = st.session_state.get("comparison_result")
    step_labels = ["1 · Configure", "2 · Run", "3 · Results"]
    configure_tab, run_tab, results_tab = st.tabs(
        step_labels,
        default=(
            step_labels[2]
            if isinstance(existing_result, ComparisonResult)
            else step_labels[0]
        ),
        key=(
            "workflow_steps_with_result"
            if isinstance(existing_result, ComparisonResult)
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
            base_url = LOCAL_BASE_URLS.get(provider, "")
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
                            else "Start Ollama locally before running the comparison."
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
                    "Token ceiling per condition",
                    min_value=1000,
                    value=200_000,
                    step=1000,
                    key="token_ceiling",
                    help="The same ceiling is enforced independently for all three conditions.",
                )
                st.caption(
                    "Temperature is fixed at 0. The centralized condition uses three workers."
                )

    with run_tab:
        with st.container(border=True):
            st.markdown(
                """
                <div class="step-heading">
                    <span class="step-index" aria-hidden="true">2</span>
                    <div>
                        <h2>Add the source and run the protocol</h2>
                        <p>Upload one text-extractable BRD or SRS, then start all three controlled conditions.</p>
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
                    st.markdown("#### Fixed protocol")
                    st.markdown(
                        "Single prompt  \n"
                        "Staged single agent  \n"
                        "Centralized multi-agent"
                    )
                    st.caption(
                        f"{_provider_label(provider)} · {model} · "
                        f"{token_ceiling:,} tokens per condition"
                    )

            run_clicked = st.button(
                "Run comparison",
                type="primary",
                key="run",
                width="stretch",
            )

        if run_clicked:
            st.session_state.pop("comparison_result", None)
            if uploaded is None:
                st.error("Upload one text-extractable PDF before running the comparison.")
            elif provider == "gemini" and not api_key.strip():
                st.error("Enter a Gemini API key before running the comparison.")
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
                    with st.status("Preparing comparison", expanded=True) as status:

                        def progress(condition: Condition | None, message: str) -> None:
                            name = _condition_label(condition) if condition else "Document"
                            status.write(f"**{name}** — {message}")

                        runner = st.session_state.get("_runner", run_comparison)
                        result = runner(
                            uploaded.getvalue(), settings, progress=progress
                        )
                        st.session_state["comparison_result"] = result
                        label, state = _result_status(result)
                        status.update(
                            label=label,
                            state=state,
                            expanded=state == "error",
                        )
                    st.rerun()
                except Exception as error:
                    st.error(f"Comparison failed: {_safe_error(error, settings)}")

    with results_tab:
        with st.container(border=True):
            st.markdown(
                """
                <div class="step-heading">
                    <span class="step-index" aria-hidden="true">3</span>
                    <div>
                        <h2>Review quality and export artifacts</h2>
                        <p>Compare output volume, traceability, cost, and failures before downloading evidence.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            result = st.session_state.get("comparison_result")
            if isinstance(result, ComparisonResult):
                if (
                    result.manifest.provider != provider
                    or result.manifest.model != model
                    or result.manifest.token_ceiling != token_ceiling
                ):
                    st.info(
                        "The setup has changed since this result was created. "
                        "Run the comparison again to refresh it."
                    )
                _render_result(result)
            else:
                _render_empty_state()


main()
