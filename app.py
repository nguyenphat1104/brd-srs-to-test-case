from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from brd_srs_testgen.models import (
    AgentSetup,
    ActivityEvent,
    ArtifactBundle,
    CoverageScore,
    CoverageUnitBatch,
    FailureCategory,
    GeneratedCases,
    RequirementBatch,
    RunHistoryItem,
    RunMetrics,
    RunResult,
    RunStatus,
    RunType,
    ScenarioBatch,
    TestCaseBatch,
    default_agent_setups,
)
from brd_srs_testgen.prompts import RUN_PROMPT_DEFAULTS
from brd_srs_testgen.providers import list_llama_cpp_models
from brd_srs_testgen.runner import ProviderSettings, run_generation
from brd_srs_testgen.storage import RunRepository, StorageError


GEMINI_DEFAULT_MODEL = "gemini-3.6-flash"
SINGLE_DEFAULT_MODEL = "gemini-3.5-flash"
STAGED_DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_TOKEN_CEILING = 200_000
PROVIDER_LABELS = {
    "gemini": "Gemini",
    "lm_studio": "LM Studio",
    "llama_cpp": "llama.cpp",
    "ollama": "Ollama",
}
RUN_PROVIDERS = ("gemini", "llama_cpp")
PROVIDER_MODELS = {
    "gemini": (
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ),
}
MODEL_LABELS = {
    "gemini-3.7-flash": "Gemini 3.7 Flash",
    "gemini-3.6-flash": "Gemini 3.6 Flash",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
}
LOCAL_BASE_URLS = {
    "llama_cpp": "http://localhost:8080/v1",
}
RUN_TYPE_COPY = {
    RunType.SINGLE_PROMPT: (
        "Single prompt",
        "Fastest. Creates a complete first draft in one pass.",
    ),
    RunType.STAGED_SINGLE_AGENT: (
        "Staged prompt",
        "Builds requirements, scenarios, and test cases step by step before checking them.",
    ),
    RunType.CENTRALIZED_MULTI_AGENT: (
        "Multi agents",
        "Most thorough. Several specialists generate and cross-check the test suite.",
    ),
}
RUN_CONFIG_AGENTS = {
    RunType.SINGLE_PROMPT: ("single",),
    RunType.STAGED_SINGLE_AGENT: ("requirements", "scenarios", "test_cases"),
    RunType.CENTRALIZED_MULTI_AGENT: (
        "analyst",
        "test_generator",
        "reviewer",
        "coverage_analyzer",
    ),
}
AGENT_LABELS = {
    "analyst": "Analyst",
    "test_generator": "Test generator",
    "reviewer": "Reviewer",
    "coverage_analyzer": "Coverage analyzer",
}
LOCAL_AGENT_MODEL_HINTS = {
    "analyst": "qwen",
    "test_generator": "gemma",
    "reviewer": "phi",
    "coverage_analyzer": "qwen",
}
RUN_AGENT_LABELS = {
    "single": "Test suite generator",
    "requirements": "Requirements step",
    "scenarios": "Scenarios step",
    "test_cases": "Test cases step",
    **AGENT_LABELS,
}
ACTIVITY_PLAN = (
    ("Prepare document", "Source ready"),
    ("Extract requirements", "Analyst handoff"),
    ("Reconcile findings", "Reviewer handoff"),
    ("Generate test cases", "Generator handoff"),
    ("Validate and deliver", "Final bundle"),
)


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
    return _env("LLAMA_CPP_BASE_URL") or LOCAL_BASE_URLS[provider]


def _api_key(provider: str) -> str:
    return _env("GEMINI_API_KEY") if provider == "gemini" else ""


@st.cache_data(ttl=30, show_spinner=False)
def _llama_cpp_models(base_url: str) -> tuple[tuple[str, str], ...]:
    try:
        return tuple(list_llama_cpp_models(base_url))
    except Exception:
        return ()


def _models_for_provider(
    provider: str,
) -> tuple[tuple[str, ...], dict[str, str]]:
    if provider == "gemini":
        models = PROVIDER_MODELS[provider]
        return models, {model: _model_label(model) for model in models}
    try:
        loader = st.session_state.get("_model_loader", _llama_cpp_models)
        options = tuple(loader(_base_url(provider)))
    except Exception:
        options = ()
    if not options:
        st.session_state["_llama_cpp_model_error"] = True
    return tuple(model for model, _label in options), dict(options)


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
            --color-accent-strong: #1d4ed8;
            --color-accent-soft: #eff6ff;
            --color-background: #f8fafc;
            --color-surface: #ffffff;
            --color-foreground: #0f172a;
            --color-muted: #475569;
            --color-faint: #64748b;
            --color-border: #dbe3ee;
            --color-focus: #1d4ed8;
            --color-success: #15803d;
            --color-success-soft: #f0fdf4;
            --color-success-border: #bbf7d0;
            --color-danger: #b91c1c;
            --color-danger-soft: #fef2f2;
            --color-danger-border: #fecaca;
            --color-warning: #b45309;
            --color-warning-soft: #fffbeb;
            --color-warning-border: #fde68a;
        }
        .stApp {
            color: var(--color-foreground);
            font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
                "Segoe UI", sans-serif;
            background: var(--color-background);
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 78rem !important;
            width: 100%;
            margin-inline: auto;
            padding-top: 1.5rem;
            padding-bottom: 4rem;
            padding-left: clamp(1.5rem, 5vw, 6rem);
            padding-right: clamp(1.5rem, 5vw, 6rem);
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--color-border);
            border-radius: 1rem;
            background: rgba(255, 255, 255, 0.94);
            box-shadow: none;
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
        .stButton > button {
            min-height: 2.75rem;
            border-radius: 0.65rem;
            font-weight: 600;
        }
        .stButton > button[kind="primary"] {
            min-height: 3rem;
            border-radius: 0.7rem;
            border-color: var(--color-accent);
            background: var(--color-accent);
            font-weight: 700;
            box-shadow: none;
        }
        .stButton > button[kind="primary"]:hover {
            border-color: var(--color-accent-strong);
            background: var(--color-accent-strong);
        }
        .stDownloadButton > button {
            min-height: 2.75rem;
            border-radius: 0.65rem;
        }
        /* ---------- App bar ---------- */
        .app-bar {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            padding: 0.55rem 0 0.9rem;
        }
        .app-bar__rule {
            margin: 0 0 0.75rem;
            border-bottom: 1px solid var(--color-border);
        }
        .app-bar__mark {
            display: grid;
            place-items: center;
            width: 2.15rem;
            height: 2.15rem;
            border-radius: 0.65rem;
            background: var(--color-accent);
            color: #ffffff;
            font-size: 0.75rem;
            font-weight: 850;
            letter-spacing: 0.02em;
            box-shadow: none;
        }
        .app-bar__name {
            color: var(--color-foreground);
            font-size: 0.95rem;
            font-weight: 750;
            letter-spacing: -0.01em;
        }
        .app-bar__tag {
            margin-top: 0.05rem;
            color: var(--color-muted);
            font-size: 0.75rem;
        }
        /* ---------- Runs hero ---------- */
        .runs-hero {
            padding: 1.4rem 0 0.4rem;
        }
        .runs-hero__kicker {
            color: var(--color-accent);
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }
        .runs-hero__title {
            margin-top: 0.3rem;
            color: var(--color-foreground);
            font-size: clamp(1.7rem, 3vw, 2.3rem);
            font-weight: 800;
            letter-spacing: -0.035em;
            line-height: 1.08;
        }
        .runs-hero__sub {
            margin-top: 0.45rem;
            max-width: 44rem;
            color: var(--color-muted);
            font-size: 0.92rem;
            line-height: 1.55;
        }
        .simple-steps {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1.25rem 0 1.5rem;
        }
        .simple-step {
            padding: 0.9rem 1rem;
            border: 1px solid var(--color-border);
            border-radius: 0.8rem;
            background: var(--color-surface);
        }
        .simple-step__number {
            color: var(--color-accent);
            font-size: 0.75rem;
            font-weight: 800;
        }
        .simple-step__title {
            margin-top: 0.15rem;
            color: var(--color-foreground);
            font-size: 0.9rem;
            font-weight: 750;
        }
        .simple-step__detail {
            margin-top: 0.2rem;
            color: var(--color-muted);
            font-size: 0.78rem;
            line-height: 1.45;
        }
        .wizard-steps {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin: 1rem 0 1.5rem;
            border: 1px solid var(--color-border);
            border-radius: 0.8rem;
            overflow: hidden;
            background: var(--color-surface);
        }
        .wizard-step {
            padding: 0.75rem 0.9rem;
            border-right: 1px solid var(--color-border);
            color: var(--color-muted);
            font-size: 0.8rem;
            font-weight: 700;
        }
        .wizard-step:last-child {
            border-right: 0;
        }
        .wizard-step--active {
            color: var(--color-accent-strong);
            background: var(--color-accent-soft);
        }
        .wizard-step--complete {
            color: var(--color-success);
            background: var(--color-success-soft);
        }
        .artifact-reader__empty {
            min-height: 12rem;
            display: grid;
            place-items: center;
            padding: 1.5rem;
            border: 1px dashed var(--color-border);
            border-radius: 0.8rem;
            color: var(--color-muted);
            text-align: center;
        }
        /* ---------- Run list ---------- */
        [class*="st-key-run-item-"],
        [class*="st-key-artifact-item-"] {
            position: relative;
            transition: transform 180ms ease-out;
        }
        [class*="st-key-run-item-"] [data-testid="stVerticalBlockBorderWrapper"],
        [class*="st-key-artifact-item-"] [data-testid="stVerticalBlockBorderWrapper"] {
            transition: border-color 180ms ease-out, box-shadow 180ms ease-out;
        }
        [class*="st-key-run-item-"]:has(.stButton > button:hover),
        [class*="st-key-artifact-item-"]:has(.stButton > button:hover) {
            transform: translateY(-1px);
        }
        [class*="st-key-run-item-"]:has(.stButton > button:hover)
        [data-testid="stVerticalBlockBorderWrapper"],
        [class*="st-key-artifact-item-"]:has(.stButton > button:hover)
        [data-testid="stVerticalBlockBorderWrapper"],
        [class*="st-key-run-item-"]:has(.stButton > button:focus-visible)
        [data-testid="stVerticalBlockBorderWrapper"],
        [class*="st-key-artifact-item-"]:has(.stButton > button:focus-visible)
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #93c5fd;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.1);
        }
        [class*="st-key-run-item-"] .stButton,
        [class*="st-key-artifact-item-"] .stButton {
            position: absolute;
            inset: 0;
            z-index: 1;
        }
        [class*="st-key-run-item-"] > [class*="st-key-open-run-"],
        [class*="st-key-artifact-item-"] > [class*="st-key-open-artifact-"] {
            position: static;
        }
        [class*="st-key-run-item-"] .stButton > button,
        [class*="st-key-artifact-item-"] .stButton > button {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            min-height: 0;
            margin: 0;
            border: 0;
            color: transparent;
            background: transparent;
            box-shadow: none;
            cursor: pointer;
        }
        [class*="st-key-run-item-"] .stButton > button:hover,
        [class*="st-key-run-item-"] .stButton > button:active,
        [class*="st-key-artifact-item-"] .stButton > button:hover,
        [class*="st-key-artifact-item-"] .stButton > button:active {
            border: 0;
            color: transparent;
            background: transparent;
            box-shadow: none;
        }
        .run-list-item__title {
            color: var(--color-foreground);
            font-size: 1rem;
            font-weight: 750;
            overflow-wrap: anywhere;
        }
        .run-list-item__header {
            display: flex;
            align-items: start;
            justify-content: space-between;
            gap: 1rem;
        }
        .run-list-item__facts {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(min(100%, 10rem), 1fr));
            gap: 0.8rem 1.25rem;
            margin-top: 1rem;
        }
        .run-list-item__label {
            color: var(--color-faint);
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }
        .run-list-item__value {
            margin-top: 0.15rem;
            color: var(--color-muted);
            font-size: 0.82rem;
            font-variant-numeric: tabular-nums;
            overflow-wrap: anywhere;
        }
        .run-list-item__status {
            padding: 0.22rem 0.55rem;
            border: 1px solid #bfdbfe;
            border-radius: 999px;
            color: var(--color-accent-strong);
            background: var(--color-accent-soft);
            font-size: 0.75rem;
            font-weight: 750;
            white-space: nowrap;
        }
        .run-list-item__status--completed {
            border-color: var(--color-success-border);
            color: var(--color-success);
            background: var(--color-success-soft);
        }
        .run-list-item__status--failed {
            border-color: var(--color-danger-border);
            color: var(--color-danger);
            background: var(--color-danger-soft);
        }
        .run-list-item__status--running {
            border-color: var(--color-warning-border);
            color: var(--color-warning);
            background: var(--color-warning-soft);
        }
        .artifact-item__title {
            color: var(--color-foreground);
            font-size: 0.92rem;
            font-weight: 750;
            overflow-wrap: anywhere;
        }
        .artifact-item__id {
            color: var(--color-accent-strong);
            font-variant-numeric: tabular-nums;
        }
        .artifact-item__facts {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem 0.85rem;
            margin-top: 0.4rem;
            color: var(--color-muted);
            font-size: 0.78rem;
        }
        @media (max-width: 640px) {
            .simple-steps {
                grid-template-columns: 1fr;
            }
            .wizard-steps {
                grid-template-columns: 1fr;
            }
            .wizard-step {
                border-right: 0;
                border-bottom: 1px solid var(--color-border);
            }
            .wizard-step:last-child {
                border-bottom: 0;
            }
            .run-list-item__header {
                align-items: flex-start;
            }
        }
        .quality-chart {
            display: grid;
            gap: 0.7rem;
        }
        .quality-chart__summary {
            margin: 0;
            color: var(--color-muted);
            font-size: 0.84rem;
        }
        .quality-chart__row {
            display: grid;
            grid-template-columns: minmax(12rem, 1fr) minmax(10rem, 2fr) 3rem;
            gap: 0.75rem;
            align-items: center;
        }
        .quality-chart__measure {
            color: var(--color-foreground);
            font-size: 0.84rem;
            font-weight: 650;
        }
        .quality-chart__track {
            height: 0.8rem;
            overflow: hidden;
            border: 1px solid var(--color-border);
            border-radius: 999px;
            background: #eef2f7;
        }
        .quality-chart__bar {
            display: block;
            width: var(--coverage);
            height: 100%;
            border-radius: inherit;
            background: var(--color-warning);
        }
        .quality-chart__bar--critical {
            background: var(--color-danger);
        }
        .quality-chart__track--critical {
            border-color: var(--color-danger-border);
            background: var(--color-danger-soft);
        }
        .quality-chart__bar--complete {
            background: var(--color-success);
        }
        .quality-chart__value {
            color: var(--color-foreground);
            font-size: 0.84rem;
            font-variant-numeric: tabular-nums;
            text-align: right;
        }
        @media (max-width: 640px) {
            .quality-chart__row {
                grid-template-columns: minmax(0, 1fr) 3rem;
            }
            .quality-chart__track {
                grid-column: 1 / -1;
                grid-row: 2;
            }
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
        .activity-plan {
            display: grid;
            gap: 0.25rem;
        }
        .activity-plan__summary {
            margin-bottom: 0.45rem;
            color: var(--color-muted);
            font-size: 0.76rem;
            font-weight: 700;
        }
        .activity-plan__progress {
            height: 0.42rem;
            margin-bottom: 0.55rem;
            overflow: hidden;
            border-radius: 999px;
            background: #e2e8f0;
        }
        .activity-plan__progress > span {
            display: block;
            width: var(--plan-progress);
            height: 100%;
            border-radius: inherit;
            background: var(--color-accent);
            transition: width 200ms ease-out;
        }
        .activity-plan__step {
            display: grid;
            grid-template-columns: 1.35rem minmax(0, 1fr);
            gap: 0.65rem;
            align-items: start;
            padding: 0.7rem 0;
            border-bottom: 1px solid var(--color-border);
        }
        .activity-plan__step:last-child {
            border-bottom: 0;
        }
        .activity-plan__marker {
            display: grid;
            place-items: center;
            width: 1.25rem;
            height: 1.25rem;
            border: 1px solid var(--color-border);
            border-radius: 999px;
            color: var(--color-muted);
            background: #f1f5f9;
            font-size: 0.7rem;
            font-weight: 800;
        }
        .activity-plan__step--complete .activity-plan__marker {
            border-color: #bbf7d0;
            color: #166534;
            background: #f0fdf4;
        }
        .activity-plan__step--active .activity-plan__marker {
            border-color: #93c5fd;
            color: #1d4ed8;
            background: #eff6ff;
            animation: activity-pulse 1.8s ease-in-out infinite;
        }
        .activity-plan__step--error .activity-plan__marker {
            border-color: var(--color-danger-border);
            color: var(--color-danger);
            background: var(--color-danger-soft);
        }
        .activity-plan__name {
            color: var(--color-foreground);
            font-size: 0.84rem;
            font-weight: 700;
        }
        .activity-plan__detail {
            color: var(--color-muted);
            font-size: 0.75rem;
        }
        .activity-state {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            color: var(--color-muted);
            font-size: 0.75rem;
            font-weight: 700;
        }
        .activity-state::before {
            content: "";
            width: 0.45rem;
            height: 0.45rem;
            border-radius: 999px;
            background: currentColor;
        }
        .activity-state--working {
            color: var(--color-accent);
        }
        .activity-state--working::before {
            animation: activity-pulse 1.8s ease-in-out infinite;
        }
        .activity-state--complete {
            color: #15803d;
        }
        .st-key-live-feed {
            background: #fbfdff;
            border: 1px solid var(--color-border);
            border-radius: 0.9rem;
        }
        .st-key-live-feed [data-testid="stChatMessage"] {
            margin-bottom: 0.75rem;
        }
        .st-key-live-feed [data-testid="stVerticalBlock"] {
            scrollbar-color: #bfdbfe transparent;
        }
        .agent-event__header {
            display: flex;
            align-items: baseline;
            flex-wrap: wrap;
            gap: 0.35rem;
        }
        .agent-event__name {
            color: var(--color-foreground);
            font-size: 0.9rem;
            font-weight: 750;
        }
        .agent-event__role {
            color: var(--color-muted);
            font-size: 0.78rem;
        }
        .agent-event__message {
            margin: 0.45rem 0 0;
            color: var(--color-foreground);
            font-size: 0.82rem;
            line-height: 1.5;
        }
        .timeline-result-card {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 1.1rem 1.2rem;
            border: 1px solid var(--color-border);
            border-radius: 0.9rem;
            background: #fbfdff;
        }
        .timeline-result-card__eyebrow {
            color: var(--color-muted);
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .timeline-result-card__title {
            margin-top: 0.2rem;
            color: var(--color-foreground);
            font-size: 1.05rem;
            font-weight: 750;
        }
        [data-testid="stDialog"]:has(#result-reader-marker) > div[role="dialog"] {
            width: 100vw !important;
            max-width: 100vw !important;
            height: 100vh !important;
            max-height: 100vh !important;
            border-radius: 0 !important;
        }
        [data-testid="stDialog"]:has(#result-reader-marker) [data-testid="stDialogContent"] {
            max-height: none !important;
            height: 100%;
            overflow-y: auto;
        }
        /* ---------- Run timeline spine ---------- */
        .timeline-spine {
            position: relative;
            margin: 1.35rem 0 0.4rem 0.5rem;
            padding-left: 2.1rem;
            border-left: 2px solid var(--color-border);
        }
        .seamless-stage {
            position: relative;
            display: grid;
            grid-template-columns: 2rem minmax(0, 1fr);
            gap: 0.75rem;
            align-items: start;
            margin: 1.25rem 0 0.6rem;
        }
        .seamless-stage__marker {
            display: grid;
            place-items: center;
            width: 2rem;
            height: 2rem;
            margin-left: -3.12rem;
            border: 2px solid var(--color-border);
            border-radius: 999px;
            color: var(--color-faint);
            background: #ffffff;
            font-size: 0.8rem;
            font-weight: 800;
        }
        .seamless-stage--active .seamless-stage__marker {
            border-color: var(--color-accent);
            color: var(--color-accent-strong);
            background: var(--color-accent-soft);
            box-shadow: 0 0 0 5px rgba(37, 99, 235, 0.12);
            animation: activity-pulse 1.8s ease-in-out infinite;
        }
        .seamless-stage--complete .seamless-stage__marker {
            border-color: var(--color-accent);
            color: #ffffff;
            background: var(--color-accent);
        }
        .seamless-stage--error .seamless-stage__marker {
            border-color: var(--color-danger-border);
            color: var(--color-danger);
            background: var(--color-danger-soft);
        }
        .seamless-stage__title {
            color: var(--color-foreground);
            font-size: 1rem;
            font-weight: 750;
        }
        .seamless-stage__eyebrow {
            margin-bottom: 0.12rem;
            color: var(--color-faint);
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .seamless-stage__detail {
            margin-top: 0.1rem;
            color: var(--color-muted);
            font-size: 0.82rem;
        }
        .seamless-stage--pending .seamless-stage__title {
            color: var(--color-faint);
        }
        @keyframes activity-pulse {
            50% {
                opacity: 0.55;
                transform: scale(0.82);
            }
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


def _provider_label(provider: str) -> str:
    return PROVIDER_LABELS[provider]


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


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
    protected = (
        settings.api_key,
        settings.base_url,
        *settings.provider_api_keys.values(),
        *settings.provider_base_urls.values(),
    )
    for secret in protected:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message


def _activity_plan_step(event: str, current_step: int) -> int:
    message = str(event).lower()
    if not isinstance(event, ActivityEvent):
        if "generating artifacts" in message:
            return max(current_step, 1)
        if "completed" in message:
            return len(ACTIVITY_PLAN)
        return current_step
    if event.agent.startswith("Analyzer"):
        return max(current_step, 1)
    if event.agent == "Reviewer":
        return max(current_step, 2)
    if event.agent.startswith("Test Generator"):
        return max(current_step, 3)
    if event.agent == "Orchestrator" and "merging" in message:
        return max(current_step, 4)
    return current_step


def _render_activity_plan(
    placeholder, current_step: int, *, stopped: bool = False
) -> None:
    next_step = min(current_step, len(ACTIVITY_PLAN) - 1)
    summary = (
        f"Stopped at step {next_step + 1} of {len(ACTIVITY_PLAN)} · {ACTIVITY_PLAN[next_step][0]}"
        if stopped
        else "Plan complete"
        if current_step == len(ACTIVITY_PLAN)
        else f"Step {next_step + 1} of {len(ACTIVITY_PLAN)} · {ACTIVITY_PLAN[next_step][0]}"
    )
    steps = []
    for index, (name, detail) in enumerate(ACTIVITY_PLAN):
        if stopped and index == next_step:
            state, marker, status = "error", "!", "Stopped"
        elif index < current_step:
            state, marker, status = "complete", "✓", "Complete"
        elif index == current_step and current_step < len(ACTIVITY_PLAN):
            state, marker, status = "active", str(index + 1), "Working"
        else:
            state, marker, status = "pending", str(index + 1), "Queued"
        steps.append(
            f"<div class='activity-plan__step activity-plan__step--{state}'>"
            f"<span class='activity-plan__marker'>{marker}</span>"
            f"<div><div class='activity-plan__name'>{name}</div>"
            f"<div class='activity-plan__detail'>{detail} · {status}</div>"
            "</div></div>"
        )
    placeholder.markdown(
        "<div class='activity-plan' aria-label='Generation plan'>"
        f"<div class='activity-plan__summary'>{summary}</div>"
        "<div class='activity-plan__progress' role='progressbar' "
        f"aria-valuenow='{next_step if stopped else current_step}' aria-valuemin='0' "
        f"aria-valuemax='{len(ACTIVITY_PLAN)}'><span style='--plan-progress: "
        f"{(next_step if stopped else current_step) / len(ACTIVITY_PLAN):.0%}'></span></div>"
        f"{''.join(steps)}</div>",
        unsafe_allow_html=True,
    )


def _render_artifact_content(artifact: Any) -> None:
    if isinstance(artifact, RequirementBatch):
        for requirement in artifact.requirements:
            st.markdown(f"##### {requirement.requirement_id} · {requirement.title}")
            _render_requirement_detail(requirement)
        return
    if isinstance(artifact, ScenarioBatch):
        for scenario in artifact.scenarios:
            st.markdown(f"##### {scenario.scenario_id} · {scenario.title}")
            _render_scenario_detail(scenario)
        return
    if isinstance(artifact, TestCaseBatch):
        for test_case in artifact.test_cases:
            st.markdown(f"##### {test_case.test_case_id} · {test_case.title}")
            _render_test_case_detail(test_case)
        return
    if isinstance(artifact, GeneratedCases):
        st.markdown(f"##### Scenarios · {len(artifact.scenarios)}")
        for scenario in artifact.scenarios:
            st.markdown(f"**{scenario.scenario_id} · {scenario.title}**")
            _render_scenario_detail(scenario)
        st.markdown(f"##### Test cases · {len(artifact.test_cases)}")
        for test_case in artifact.test_cases:
            st.markdown(f"**{test_case.test_case_id} · {test_case.title}**")
            _render_test_case_detail(test_case)
        return
    if isinstance(artifact, ArtifactBundle):
        _render_artifact_content(
            RequirementBatch(requirements=artifact.requirements)
        )
        _render_artifact_content(ScenarioBatch(scenarios=artifact.scenarios))
        _render_artifact_content(TestCaseBatch(test_cases=artifact.test_cases))
        return
    if isinstance(artifact, CoverageUnitBatch):
        for unit in artifact.units:
            st.markdown(f"##### {unit.unit_id} · {unit.title}")
            st.markdown(unit.description)
            st.caption(
                f"{unit.unit_type.replace('_', ' ').title()} · "
                f"Sources: {', '.join(unit.source_chunk_ids)}"
            )
        return
    if isinstance(artifact, CoverageScore):
        columns = st.columns(3)
        columns[0].metric("F1", f"{artifact.f1:.2f}")
        columns[1].metric("Precision", f"{artifact.precision:.2f}")
        columns[2].metric("Recall", f"{artifact.recall:.2f}")
        st.caption(
            f"{artifact.total_test_cases} test cases · "
            f"{artifact.total_coverage_units} coverage units"
        )


def _render_artifact_reader(target, events: list[str]) -> None:
    published = [
        event
        for event in events
        if isinstance(event, ActivityEvent) and event.artifact is not None
    ]
    with target.container():
        st.markdown("#### Artifacts")
        if not published:
            st.markdown(
                "<div class='artifact-reader__empty'>Generated artifacts will "
                "appear here in full as agents publish them.</div>",
                unsafe_allow_html=True,
            )
            return
        st.caption(f"{len(published)} published · newest last")
        with st.container(height=720):
            for position, event in enumerate(published):
                if position:
                    st.divider()
                st.markdown(
                    f"##### {event.artifact_label or 'Published artifact'}"
                )
                st.caption(event.agent or "Agent")
                _render_artifact_content(event.artifact)


def _render_activity(activity, event: str) -> None:
    if not isinstance(event, ActivityEvent) or not event.agent:
        activity.caption(str(event))
        return

    with activity:
        with st.container(border=True):
            st.markdown(
                "<div class='agent-event__header'>"
                f"<span class='agent-event__name'>{html.escape(event.agent)}</span>"
                f"<span class='agent-event__role'>· {html.escape(event.role)}</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            metadata = []
            if event.model:
                metadata.append(f"Model: {event.model}")
            if metadata:
                st.caption(" · ".join(metadata))
            state = event.state.lower() or "update"
            st.markdown(
                f"<span class='activity-state activity-state--{state}'>"
                f"{event.state.title() or 'Update'}</span>",
                unsafe_allow_html=True,
            )
            if event.task:
                st.markdown(f"**Task:** {event.task}")
            if event.scope:
                st.caption(f"Scope: {event.scope}")
            if event.deliverable:
                st.caption(f"Delivers: {event.deliverable}")
            st.markdown(
                f"<p class='agent-event__message'>{html.escape(str(event))}</p>",
                unsafe_allow_html=True,
            )
            if event.artifact is not None:
                st.caption(f"Published to Artifacts · {event.artifact_label}")


def _scroll_live_feed() -> None:
    components.html(
        """
        <script>
        try {
          const root = window.parent.document.querySelector('.st-key-live-feed');
          const scrollbox = [root, ...(root?.querySelectorAll('*') || [])].find(
            (node) => node && node.scrollHeight > node.clientHeight
              && ['auto', 'scroll'].includes(getComputedStyle(node).overflowY)
          );
          requestAnimationFrame(() => scrollbox?.scrollTo({
            top: scrollbox.scrollHeight, behavior: 'smooth'
          }));
        } catch (_) {}
        </script>
        """,
        height=0,
    )


def _render_timeline_stage(
    number: int,
    title: str,
    detail: str,
    *,
    state: str,
) -> None:
    marker = "✓" if state == "complete" else str(number)
    st.markdown(
        f"<div class='seamless-stage seamless-stage--{state}'>"
        f"<span class='seamless-stage__marker'>{marker}</span>"
        "<div>"
        f"<div class='seamless-stage__eyebrow'>Step {number} of 3</div>"
        f"<div class='seamless-stage__title'>{title}</div>"
        f"<div class='seamless-stage__detail'>{detail}</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


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


def _quality_chart_html(metrics: RunMetrics) -> str:
    measures = sorted(
        (
            ("Citation coverage", metrics.citation_coverage),
            ("Requirement → scenario", metrics.requirement_scenario_coverage),
            ("Requirement → test case", metrics.requirement_test_case_coverage),
            ("Positive scenario coverage", metrics.positive_scenario_coverage),
            ("Non-positive scenario coverage", metrics.non_positive_scenario_coverage),
            ("RTM completeness", metrics.rtm_completeness),
        ),
        key=lambda measure: measure[1],
    )
    weakest_name, weakest_value = measures[0]
    summary = (
        "All quality measures are at 100%."
        if weakest_value == 1
        else f"Priority: {weakest_name} is {weakest_value:.0%}. Target: 100%."
    )
    rows = []
    for label, value in measures:
        state = "complete" if value == 1 else "critical" if value == 0 else "partial"
        rows.append(
            "<div class='quality-chart__row'>"
            f"<span class='quality-chart__measure'>{html.escape(label)}</span>"
            f"<span class='quality-chart__track quality-chart__track--{state}' "
            "aria-hidden='true'>"
            f"<span class='quality-chart__bar quality-chart__bar--{state}' "
            f"style='--coverage: {value:.0%}'></span></span>"
            f"<strong class='quality-chart__value'>{value:.0%}</strong>"
            "</div>"
        )
    return (
        "<div class='quality-chart' role='group' "
        f"aria-label='{html.escape(summary)}'>"
        f"<p class='quality-chart__summary'>{html.escape(summary)}</p>"
        + "".join(rows)
        + "</div>"
    )


def _render_sources(sources) -> None:
    st.markdown("**Source references**")
    for source in sources:
        location = f"Page {source.page_number}"
        if source.section:
            location += f" · {source.section}"
        st.markdown(f"- `{source.chunk_id}` · {location} — {source.excerpt}")


def _render_artifact_item(
    *, item_type: str, item_id: str, title: str, facts: tuple[str, ...], key: str
) -> bool:
    with st.container(border=True, key=f"artifact-item-{key}"):
        st.markdown(
            "<div class='artifact-item__title'>"
            f"<span class='artifact-item__id'>{html.escape(item_id)}</span> · "
            f"{html.escape(title)}</div>"
            "<div class='artifact-item__facts'>"
            + "".join(
                f"<span>{html.escape(fact)}</span>" for fact in facts if fact
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        return st.button(
            f"Open {item_type} {item_id} detail",
            key=f"open-artifact-{key}",
        )


def _render_test_case_detail(test_case) -> None:
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


@st.dialog("Test case detail", width="large")
def _test_case_dialog(test_case) -> None:
    st.markdown(f"### {test_case.test_case_id} · {test_case.title}")
    _render_test_case_detail(test_case)


def _render_requirement_detail(requirement) -> None:
    st.markdown(requirement.description)
    st.markdown(
        f"**Type:** {requirement.requirement_type.value.replace('_', ' ').title()}  \n"
        f"**Priority:** {requirement.priority.value.title()}  \n"
        f"**Module:** {requirement.module}"
    )
    st.markdown(
        "**Dependency IDs:** " + (", ".join(requirement.dependency_ids) or "None")
    )
    st.markdown("**Ambiguities:**")
    st.markdown(
        "\n".join(f"- {item}" for item in requirement.ambiguities) or "None"
    )
    _render_sources(requirement.source_references)


@st.dialog("Requirement detail", width="large")
def _requirement_dialog(requirement) -> None:
    st.markdown(f"### {requirement.requirement_id} · {requirement.title}")
    _render_requirement_detail(requirement)


def _render_scenario_detail(scenario) -> None:
    st.markdown(scenario.objective)
    st.markdown(
        f"**Type:** {scenario.scenario_type.value.replace('_', ' ').title()}  \n"
        f"**Requirement IDs:** {', '.join(scenario.requirement_ids)}"
    )
    st.markdown("**Preconditions:**")
    st.markdown("\n".join(f"- {item}" for item in scenario.preconditions) or "None")
    _render_sources(scenario.source_references)


@st.dialog("Scenario detail", width="large")
def _scenario_dialog(scenario) -> None:
    st.markdown(f"### {scenario.scenario_id} · {scenario.title}")
    _render_scenario_detail(scenario)


def _render_test_cases(
    result: RunResult, *, key_prefix: str, in_dialog: bool = False
) -> None:
    bundle = result.bundle
    assert bundle is not None

    st.markdown("#### Test cases")
    selected_case = None
    for position, test_case in enumerate(bundle.test_cases):
        if _render_artifact_item(
            item_type="test case",
            item_id=test_case.test_case_id,
            title=test_case.title,
            facts=(
                f"Priority · {test_case.priority.value}",
                f"Scenario · {test_case.scenario_id}",
                f"{len(test_case.steps)} steps",
            ),
            key=f"{key_prefix}-{result.manifest.run_id}-test-case-{position}",
        ):
            selected_case = test_case
    st.caption("Click any item to open its detail.")

    if selected_case is None:
        return
    if in_dialog:
        # Dialogs cannot be nested, so render the detail inline when the
        # surrounding result is already displayed inside a dialog.
        with st.container(border=True):
            st.markdown(
                f"**{selected_case.test_case_id} · {selected_case.title}**"
            )
            _render_test_case_detail(selected_case)
    else:
        _test_case_dialog(selected_case)


def _render_requirements(
    result: RunResult, *, key_prefix: str, in_dialog: bool = False
) -> None:
    bundle = result.bundle
    assert bundle is not None

    st.markdown("#### Requirements")
    selected_requirement = None
    for position, requirement in enumerate(bundle.requirements):
        if _render_artifact_item(
            item_type="requirement",
            item_id=requirement.requirement_id,
            title=requirement.title,
            facts=(
                requirement.requirement_type.value.replace("_", " ").title(),
                f"Priority · {requirement.priority.value.title()}",
                requirement.module,
            ),
            key=f"{key_prefix}-{result.manifest.run_id}-requirement-{position}",
        ):
            selected_requirement = requirement
    st.caption("Click any item to open its detail.")

    if selected_requirement is None:
        return
    if in_dialog:
        with st.container(border=True):
            st.markdown(
                f"**{selected_requirement.requirement_id} · "
                f"{selected_requirement.title}**"
            )
            _render_requirement_detail(selected_requirement)
    else:
        _requirement_dialog(selected_requirement)


def _render_scenarios(
    result: RunResult, *, key_prefix: str, in_dialog: bool = False
) -> None:
    bundle = result.bundle
    assert bundle is not None

    st.markdown("#### Scenarios")
    selected_scenario = None
    for position, scenario in enumerate(bundle.scenarios):
        if _render_artifact_item(
            item_type="scenario",
            item_id=scenario.scenario_id,
            title=scenario.title,
            facts=(
                scenario.scenario_type.value.replace("_", " ").title(),
                f"{len(scenario.requirement_ids)} requirements",
            ),
            key=f"{key_prefix}-{result.manifest.run_id}-scenario-{position}",
        ):
            selected_scenario = scenario
    st.caption("Click any item to open its detail.")

    if selected_scenario is None:
        return
    if in_dialog:
        with st.container(border=True):
            st.markdown(
                f"**{selected_scenario.scenario_id} · {selected_scenario.title}**"
            )
            _render_scenario_detail(selected_scenario)
    else:
        _scenario_dialog(selected_scenario)


def _render_coverage(coverage, *, key_prefix: str, run_id: str) -> None:
    st.markdown("### Coverage analysis (F1)")
    columns = st.columns(4)
    columns[0].metric("F1 Score", f"{coverage.f1:.2f}")
    columns[1].metric("Precision", f"{coverage.precision:.2f}")
    columns[2].metric("Recall", f"{coverage.recall:.2f}")
    columns[3].metric("Coverage units", coverage.total_coverage_units)

    chart_column, composition_column = st.columns(2)
    with chart_column:
        st.markdown("**Score breakdown**")
        st.bar_chart(
            {
                "Score": [coverage.precision, coverage.recall, coverage.f1],
            },
            y="Score",
            x_label="Measure",
            y_label="Score (0–1)",
            color="#2563eb",
            horizontal=True,
            height=220,
        )
    with composition_column:
        st.markdown("**Mapping composition**")
        st.bar_chart(
            {
                "Count": [
                    coverage.true_positive_count,
                    coverage.false_positive_count,
                    coverage.false_negative_count,
                ],
            },
            y="Count",
            x_label="Category",
            y_label="Artifacts",
            color="#2563eb",
            horizontal=True,
            height=220,
        )
    st.caption(
        "Scores: precision, recall, F1. "
        "Composition: true positives (mapped test cases), "
        "false positives (unmapped test cases), false negatives (uncovered units)."
    )

    with st.expander(
        "Coverage detail",
        key=f"{key_prefix}-{run_id}-coverage-detail",
    ):
        st.table(
            [
                {"Measure": "True positives (mapped test cases)", "Count": coverage.true_positive_count},
                {"Measure": "False positives (unmapped test cases)", "Count": coverage.false_positive_count},
                {"Measure": "False negatives (uncovered units)", "Count": coverage.false_negative_count},
                {"Measure": "Total coverage units", "Count": coverage.total_coverage_units},
                {"Measure": "Total test cases", "Count": coverage.total_test_cases},
            ]
        )
        if coverage.uncovered_unit_ids:
            st.markdown("**Uncovered coverage units** (false negatives)")
            st.markdown(
                "\n".join(f"- `{uid}`" for uid in coverage.uncovered_unit_ids)
            )
        if coverage.unmapped_test_case_ids:
            st.markdown("**Unmapped test cases** (false positives)")
            st.markdown(
                "\n".join(f"- `{tid}`" for tid in coverage.unmapped_test_case_ids)
            )


def _render_bundle(
    result: RunResult, *, key_prefix: str, in_dialog: bool = False
) -> None:
    bundle = result.bundle
    if bundle is None:
        return

    st.markdown("### Generated artifacts")
    test_cases, requirements, scenarios = st.tabs(
        [
            f"Test cases ({len(bundle.test_cases)})",
            f"Requirements ({len(bundle.requirements)})",
            f"Scenarios ({len(bundle.scenarios)})",
        ]
    )
    with test_cases:
        _render_test_cases(result, key_prefix=key_prefix, in_dialog=in_dialog)
    with requirements:
        _render_requirements(result, key_prefix=key_prefix, in_dialog=in_dialog)
    with scenarios:
        _render_scenarios(result, key_prefix=key_prefix, in_dialog=in_dialog)


def _render_snapshot(result: RunResult) -> None:
    manifest = result.manifest
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
                "Value": (
                    manifest.completed_at.isoformat()
                    if manifest.completed_at is not None
                    else "—"
                ),
            },
        ]
    )
    agents = manifest.configuration.get("agents")
    if not isinstance(agents, dict) or not agents:
        return
    st.markdown("#### Agent settings snapshot")
    st.caption("This is the exact provider, model, and prompt setup used for the run.")
    for agent, raw in agents.items():
        if not isinstance(raw, dict):
            continue
        label = RUN_AGENT_LABELS.get(agent, agent.replace("_", " ").title())
        with st.container(border=True):
            st.markdown(f"**{label}**")
            st.caption(
                f"{_provider_label(str(raw.get('provider', manifest.provider)))} · "
                f"{raw.get('model', manifest.model)}"
            )
            st.text_area(
                f"{label} prompt",
                value=str(raw.get("prompt", "")),
                disabled=True,
                key=f"snapshot-{manifest.run_id}-{agent}",
            )


def _render_result(
    result: RunResult, *, key_prefix: str = "result", in_dialog: bool = False
) -> None:
    manifest = result.manifest
    st.caption(f"Source · {manifest.source_filename}")

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
            "Diagnostics and next steps",
            key=f"{key_prefix}-{manifest.run_id}-technical-details",
        ):
            st.code(_technical_detail(manifest.failure_message), language=None)
            st.caption("Download diagnostics before creating a replacement run.")
            _download(
                "Download diagnostics",
                result.download_bundle(),
                f"{manifest.run_id}-diagnostics.json",
                f"{key_prefix}-{manifest.run_id}-diagnostics",
            )
    elif manifest.status is RunStatus.COMPLETED:
        st.success("Your test suite is ready and passed validation.")
    else:
        st.warning("Generation was interrupted. Review diagnostics before retrying.")
        with st.expander(
            "Diagnostics and next steps",
            key=f"{key_prefix}-{manifest.run_id}-technical-details",
        ):
            st.caption("Download diagnostics before creating a replacement run.")
            _download(
                "Download diagnostics",
                result.download_bundle(),
                f"{manifest.run_id}-diagnostics.json",
                f"{key_prefix}-{manifest.run_id}-diagnostics",
            )

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

    if manifest.status is RunStatus.COMPLETED and result.bundle is not None:
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
        _render_bundle(result, key_prefix=key_prefix, in_dialog=in_dialog)

    if metrics is not None or result.coverage is not None:
        if st.toggle(
            "Show quality details",
            key=f"{key_prefix}-{manifest.run_id}-show-quality",
        ):
            st.markdown("### Quality and traceability")
            if metrics is not None:
                st.markdown(_quality_chart_html(metrics), unsafe_allow_html=True)
                st.caption(
                    f"Latency {metrics.latency_seconds:.2f} s · "
                    f"{metrics.retries} retries"
                )
            if result.coverage is not None:
                _render_coverage(
                    result.coverage,
                    key_prefix=key_prefix,
                    run_id=manifest.run_id,
                )

    with st.expander(
        "Run configuration",
        key=f"{key_prefix}-{manifest.run_id}-configuration",
    ):
        _render_snapshot(result)


@st.dialog("Run result", width="large")
def _result_dialog(result: RunResult) -> None:
    st.markdown("<span id='result-reader-marker'></span>", unsafe_allow_html=True)
    _render_result(result, key_prefix="result-reader", in_dialog=True)


def _render_timeline_result_action(result: RunResult) -> None:
    label, _ = _result_status(result)
    completed = result.manifest.status is RunStatus.COMPLETED
    eyebrow = "Validated result" if completed else "Run outcome"
    st.markdown(
        "<div class='timeline-result-card'>"
        f"<div><div class='timeline-result-card__eyebrow'>{eyebrow}</div>"
        f"<div class='timeline-result-card__title'>{label}</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    action = "View result" if completed else "View diagnostics"
    if st.button(
        action,
        type="primary",
        key=f"timeline-{result.manifest.run_id}-view-result",
        width="stretch",
    ):
        _result_dialog(result)


def _go_home() -> None:
    st.session_state["view"] = "runs"
    st.session_state.pop("selected_run_id", None)
    st.session_state.pop("selected_run", None)
    st.session_state.pop("timeline_result", None)
    st.session_state.pop("timeline_activity", None)
    st.session_state.pop("timeline_current_step", None)
    st.session_state.pop("create_step", None)
    st.session_state.pop("run_provider_settings", None)
    st.session_state.pop("pdf", None)
    st.session_state.pop("retained_pdf", None)


def _render_top_nav() -> None:
    st.markdown(
        "<div class='app-bar'>"
        "<span class='app-bar__mark'>TC</span>"
        "<div>"
        "<div class='app-bar__name'>BRD/SRS Test Case</div>"
        "<div class='app-bar__tag'>Traceable test-case generation</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='app-bar__rule'></div>", unsafe_allow_html=True)


def _open_run(run_id: str) -> None:
    st.session_state["selected_run_id"] = run_id
    st.session_state.pop("selected_run", None)
    st.session_state["view"] = "detail"


def _run_item_html(item: RunHistoryItem) -> str:
    test_cases = (
        "Not recorded"
        if item.test_case_count is None
        else f"{item.test_case_count} generated"
    )
    return (
        "<div class='run-list-item'>"
        "<div class='run-list-item__header'>"
        f"<div class='run-list-item__title'>{html.escape(item.source_filename)}</div>"
        f"<span class='run-list-item__status run-list-item__status--{item.status.value}'>"
        f"{item.display_status}</span>"
        "</div>"
        "<div class='run-list-item__facts'>"
        "<div><div class='run-list-item__label'>Method</div>"
        f"<div class='run-list-item__value'>{html.escape(_run_type_label(item.run_type))}</div></div>"
        "<div><div class='run-list-item__label'>Created</div>"
        f"<div class='run-list-item__value'>{item.started_at.strftime('%Y-%m-%d %H:%M UTC')}</div></div>"
        "<div><div class='run-list-item__label'>Output</div>"
        f"<div class='run-list-item__value'>{test_cases}</div></div>"
        "</div>"
        "</div>"
    )


def _request_create() -> None:
    for key in tuple(st.session_state):
        if key.startswith("run_"):
            st.session_state.pop(key)
    st.session_state["view"] = "create"
    st.session_state["create_step"] = 1
    st.session_state.pop("retained_pdf", None)


def _render_runs(repository: RunRepository) -> None:
    try:
        runs = repository.list_runs()
    except StorageError:
        runs = None

    st.markdown(
        "<div class='runs-hero'>"
        "<div class='runs-hero__kicker'>Document to test cases</div>"
        "<div class='runs-hero__title'>Turn a BRD or SRS into test cases</div>"
        "<div class='runs-hero__sub'>"
        "Add one PDF and get a test suite with requirements, scenarios, source "
        "references, and coverage checks."
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='simple-steps' aria-label='How it works'>"
        "<div class='simple-step'><div class='simple-step__number'>1</div>"
        "<div class='simple-step__title'>Add your PDF</div>"
        "<div class='simple-step__detail'>Use a text-based BRD or SRS.</div></div>"
        "<div class='simple-step'><div class='simple-step__number'>2</div>"
        "<div class='simple-step__title'>Generate</div>"
        "<div class='simple-step__detail'>We extract, organize, and check the coverage.</div></div>"
        "<div class='simple-step'><div class='simple-step__number'>3</div>"
        "<div class='simple-step__title'>Review or download</div>"
        "<div class='simple-step__detail'>Open each test case or export the full bundle.</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.button(
        "Create new run",
        type="primary",
        on_click=_request_create,
    )
    if notice := st.session_state.pop("runs_notice", None):
        st.info(notice)
    if runs is None:
        st.error(
            "Saved runs are unavailable. Check PostgreSQL and DATABASE_URL, "
            "then refresh this page."
        )
        return
    if not runs:
        st.info("No test suites yet. Add your first PDF to get started.")
        return

    st.markdown("### Recent test suites")
    for item in runs:
        with st.container(border=True, key=f"run-item-{item.run_id}"):
            st.markdown(_run_item_html(item), unsafe_allow_html=True)
            st.button(
                f"Open {item.source_filename}",
                key=f"open-run-{item.run_id}",
                on_click=_open_run,
                args=(item.run_id,),
            )


def _render_centralized_create(
    repository: RunRepository,
    settings: ProviderSettings,
    upload,
    generate: bool,
) -> None:
    with st.container():
        st.markdown("#### Progress")
        st.markdown(
            "<div class='timeline-spine' aria-hidden='true'></div>",
            unsafe_allow_html=True,
        )
        source_stage = st.empty()
        with source_stage.container():
            _render_timeline_stage(
                1,
                "Document ready" if upload is not None else "Add a document",
                "Your PDF is ready for generation."
                if upload is not None
                else "Choose a text-based BRD or SRS PDF above.",
                state="complete" if upload is not None else "active",
            )

        live_stage = st.empty()
        with live_stage.container():
            _render_timeline_stage(
                2,
                "Generate and validate",
                "We will extract, organize, and cross-check your test cases.",
                state="pending",
            )
        live_panel = st.empty()

        result_stage = st.empty()
        with result_stage.container():
            _render_timeline_stage(
                3,
                "Validated result",
                "The final test suite stays in this timeline after validation.",
                state="pending",
            )
        result_panel = st.empty()

    if not generate:
        result = st.session_state.get("timeline_result")
        if isinstance(result, RunResult):
            completed = result.manifest.status is RunStatus.COMPLETED
            current_step = (
                len(ACTIVITY_PLAN)
                if completed
                else st.session_state.get("timeline_current_step", 0)
            )
            source_stage.empty()
            with source_stage.container():
                _render_timeline_stage(
                    1,
                    "Document ready",
                    "Source locked for this saved run.",
                    state="complete",
                )
            live_stage.empty()
            with live_stage.container():
                _render_timeline_stage(
                    2,
                    "Generate and validate" if completed else "Generation stopped",
                    "Agent handoffs and review completed."
                    if completed
                    else "Review the diagnostics, update the setup, and try again.",
                    state="complete" if completed else "error",
                )
            result_stage.empty()
            with result_stage.container():
                _render_timeline_stage(
                    3,
                    "Validated result" if completed else "Diagnostics available",
                    "Open or download the completed test suite."
                    if completed
                    else "No validated test suite was produced for this run.",
                    state="complete" if completed else "error",
                )
            activity_events = st.session_state.get("timeline_activity", [])
            if activity_events:
                live_panel.empty()
                with live_panel.container(border=True):
                    progress_column, artifact_column = st.columns(
                        (1, 1.15), gap="large"
                    )
                    with progress_column:
                        st.markdown("#### Generation progress")
                        plan = st.empty()
                        _render_activity_plan(
                            plan, current_step, stopped=not completed
                        )
                        with st.expander("Live agent details"):
                            st.caption("Agent handoffs, tasks, scopes, and checkpoints.")
                            activity = st.container(height=420, key="live-feed")
                            for event in activity_events:
                                _render_activity(activity, event)
                            _scroll_live_feed()
                            st.caption("Private model reasoning is not displayed.")
                    with artifact_column:
                        _render_artifact_reader(st.empty(), activity_events)
            with result_panel.container(border=True):
                _render_timeline_result_action(result)
        else:
            live_panel.empty()
        return

    if upload is None:
        st.error("Upload one text-extractable PDF before generating test cases.")
        return

    provider_settings = settings

    st.session_state.pop("timeline_result", None)
    activity_events: list[str] = []
    st.session_state["timeline_activity"] = activity_events
    st.session_state["timeline_current_step"] = 0
    source_stage.empty()
    with source_stage.container():
        _render_timeline_stage(
            1,
            "Document ready",
            "Source and settings are locked for this run.",
            state="complete",
        )
    live_stage.empty()
    with live_stage.container():
        _render_timeline_stage(
            2,
            "Generate and validate",
            "Agents are working through the visible plan below.",
            state="active",
        )
    live_panel.empty()

    try:
        with live_panel.container(border=True):
            progress_column, artifact_column = st.columns((1, 1.15), gap="large")
            with progress_column:
                st.markdown("#### Generation progress")
                plan = st.empty()
                with st.expander("Live agent details"):
                    st.caption("Agent handoffs, tasks, scopes, and checkpoints.")
                    activity = st.container(height=420, key="live-feed")
                    st.caption("Private model reasoning is not displayed.")
            with artifact_column:
                artifact_reader = st.empty()
                _render_artifact_reader(artifact_reader, activity_events)

            current_step = 0
            _render_activity_plan(plan, current_step)

            def progress(event: str) -> None:
                nonlocal current_step
                current_step = _activity_plan_step(event, current_step)
                st.session_state["timeline_current_step"] = current_step
                _render_activity_plan(plan, current_step)
                _render_activity(activity, event)
                activity_events.append(event)
                _render_artifact_reader(artifact_reader, activity_events)
                _scroll_live_feed()

            runner = st.session_state.get("_runner", run_generation)
            result = runner(
                upload.getvalue(),
                upload.name,
                RunType.CENTRALIZED_MULTI_AGENT,
                provider_settings,
                repository=repository,
                progress=progress,
            )
    except Exception as error:
        live_stage.empty()
        with live_stage.container():
            _render_timeline_stage(
                2,
                "Generate and validate",
                "Generation stopped before the result was validated.",
                state="pending",
            )
        st.error(f"Generation failed: {_safe_error(error, provider_settings)}")
        return

    if result.bundle is not None and not any(
        isinstance(event, ActivityEvent)
        and isinstance(event.artifact, ArtifactBundle)
        for event in activity_events
    ):
        activity_events.append(
            ActivityEvent(
                "Final artifact bundle published.",
                agent="Orchestrator",
                role="Policy coordinator",
                state="complete",
                artifact=result.bundle,
                artifact_label="Final artifact bundle",
            )
        )
        _render_artifact_reader(artifact_reader, activity_events)

    completed = result.manifest.status is RunStatus.COMPLETED
    if completed:
        current_step = len(ACTIVITY_PLAN)
        st.session_state["timeline_current_step"] = current_step
    _render_activity_plan(plan, current_step, stopped=not completed)
    live_stage.empty()
    with live_stage.container():
        _render_timeline_stage(
            2,
            "Generate and validate" if completed else "Generation stopped",
            "Agent handoffs and review completed."
            if completed
            else "Review the diagnostics, update the setup, and try again.",
            state="complete" if completed else "error",
        )
    result_stage.empty()
    with result_stage.container():
        _render_timeline_stage(
            3,
            "Validated result" if completed else "Diagnostics available",
            "Review complete; the saved artifacts are ready to inspect."
            if completed
            else "No validated test suite was produced for this run.",
            state="complete" if completed else "error",
        )
    st.session_state["selected_run_id"] = result.manifest.run_id
    st.session_state["selected_run"] = result
    st.session_state["timeline_result"] = result
    with result_panel.container(border=True):
        _render_timeline_result_action(result)
def _render_create_steps(active_step: int) -> None:
    labels = ("Choose run type", "Configure agents", "Upload and run")
    steps = []
    for position, label in enumerate(labels, 1):
        state = "active" if position == active_step else "complete" if position < active_step else "pending"
        steps.append(
            f"<div class='wizard-step wizard-step--{state}'>"
            f"Step {position} · {label}</div>"
        )
    st.markdown(
        f"<div class='wizard-steps'>{''.join(steps)}</div>",
        unsafe_allow_html=True,
    )


def _default_run_model(provider: str, run_type: RunType) -> str:
    if provider == "gemini":
        return (
            SINGLE_DEFAULT_MODEL
            if run_type is RunType.SINGLE_PROMPT
            else STAGED_DEFAULT_MODEL
        )
    return ""


def _reset_run_model(provider_key: str, model_key: str, run_type: RunType) -> None:
    st.session_state[model_key] = _default_run_model(
        st.session_state[provider_key], run_type
    )


def _initialize_run_settings(
    run_type: RunType,
    repository: RunRepository,
) -> None:
    if st.session_state.get("run_config_type") == run_type.value:
        return
    st.session_state["run_config_type"] = run_type.value
    st.session_state["run_token_ceiling"] = DEFAULT_TOKEN_CEILING

    if run_type is RunType.STAGED_SINGLE_AGENT:
        st.session_state["run_staged_provider"] = "gemini"
        st.session_state["run_staged_model"] = STAGED_DEFAULT_MODEL
    elif run_type is RunType.SINGLE_PROMPT:
        st.session_state["run_single_provider"] = "gemini"
        st.session_state["run_single_model"] = SINGLE_DEFAULT_MODEL
    else:
        try:
            setups = repository.load_agent_setups()
        except StorageError:
            setups = default_agent_setups()
        st.session_state["run_agent_roles"] = {
            agent: setups[agent].role for agent in AGENT_LABELS
        }
        for agent in AGENT_LABELS:
            st.session_state[f"run_{agent}_provider"] = "llama_cpp"
            st.session_state[f"run_{agent}_model"] = ""
            st.session_state[f"run_{agent}_prompt"] = (
                setups[agent].instructions.strip() or RUN_PROMPT_DEFAULTS[agent]
            )

    for agent in RUN_CONFIG_AGENTS[run_type]:
        st.session_state.setdefault(
            f"run_{agent}_prompt", RUN_PROMPT_DEFAULTS[agent]
        )


def _render_provider_model(
    config_key: str, label: str, run_type: RunType
) -> tuple[str, str]:
    provider_key = f"run_{config_key}_provider"
    model_key = f"run_{config_key}_model"
    provider_column, model_column = st.columns(2)
    with provider_column:
        provider = st.selectbox(
            f"{label} provider",
            RUN_PROVIDERS,
            key=provider_key,
            format_func=_provider_label,
            on_change=_reset_run_model,
            args=(provider_key, model_key, run_type),
        )
    with model_column:
        models, labels = _models_for_provider(provider)
        if not models:
            st.session_state[model_key] = ""
            st.selectbox(
                f"{label} model",
                ("No models available",),
                key=f"{model_key}_unavailable",
                disabled=True,
            )
            model = ""
        else:
            if st.session_state.get(model_key) not in models:
                default = _default_run_model(provider, run_type)
                if hint := LOCAL_AGENT_MODEL_HINTS.get(config_key):
                    default = next(
                        (
                            model
                            for model in models
                            if hint in f"{model} {labels[model]}".lower()
                        ),
                        default,
                    )
                st.session_state[model_key] = default if default in models else models[0]
            model = st.selectbox(
                f"{label} model",
                models,
                key=model_key,
                format_func=lambda value: labels[value],
            )
    return provider, model


def _render_run_settings(run_type: RunType) -> None:
    st.session_state.pop("_llama_cpp_model_error", None)
    if run_type is RunType.SINGLE_PROMPT:
        with st.container(border=True):
            st.markdown("#### Test suite generator")
            _render_provider_model("single", "Agent", run_type)
            st.text_area(
                "Agent prompt",
                key="run_single_prompt",
                height=180,
                help="Applied after the core evidence, safety, and output-schema rules.",
            )
    elif run_type is RunType.STAGED_SINGLE_AGENT:
        with st.container(border=True):
            st.markdown("#### Shared generation model")
            _render_provider_model("staged", "Agent", run_type)
        for agent in RUN_CONFIG_AGENTS[run_type]:
            with st.container(border=True):
                st.markdown(f"#### {RUN_AGENT_LABELS[agent]}")
                st.text_area(
                    f"{RUN_AGENT_LABELS[agent]} prompt",
                    key=f"run_{agent}_prompt",
                    height=150,
                    help="Applied after the core evidence, safety, and output-schema rules.",
                )
    else:
        for agent in RUN_CONFIG_AGENTS[run_type]:
            with st.container(border=True):
                st.markdown(f"#### {RUN_AGENT_LABELS[agent]}")
                _render_provider_model(agent, RUN_AGENT_LABELS[agent], run_type)
                st.text_area(
                    f"{RUN_AGENT_LABELS[agent]} prompt",
                    key=f"run_{agent}_prompt",
                    height=150,
                    help="Applied after the core evidence, safety, and output-schema rules.",
                )
    if st.session_state.pop("_llama_cpp_model_error", False):
        st.warning(
            "llama.cpp models are unavailable. Check LLAMA_CPP_BASE_URL and "
            "the backend service."
        )
    st.number_input(
        "Token ceiling",
        min_value=1_000,
        step=1_000,
        key="run_token_ceiling",
    )
    st.caption("Provider access is managed by the deployment environment.")


def _run_provider_settings(run_type: RunType) -> ProviderSettings:
    agents = RUN_CONFIG_AGENTS[run_type]
    if run_type is RunType.STAGED_SINGLE_AGENT:
        provider = st.session_state["run_staged_provider"]
        model = st.session_state["run_staged_model"]
        agent_providers = {agent: provider for agent in agents}
        agent_models = {agent: model for agent in agents}
    else:
        agent_providers = {
            agent: st.session_state[f"run_{agent}_provider"] for agent in agents
        }
        agent_models = {
            agent: st.session_state[f"run_{agent}_model"] for agent in agents
        }
        provider = agent_providers[agents[0]]
        model = agent_models[agents[0]]

    roles = st.session_state.get("run_agent_roles", {})
    agent_setups = {
        agent: AgentSetup(
            agent=agent,
            role=roles.get(agent, default_agent_setups()[agent].role),
        )
        for agent in AGENT_LABELS
    }
    settings = ProviderSettings(
        provider=provider,
        model=model,
        token_ceiling=st.session_state["run_token_ceiling"],
        api_key=_api_key(provider),
        base_url=_base_url(provider) if provider in LOCAL_BASE_URLS else "",
        agent_setups=agent_setups,
        agent_providers=agent_providers,
        agent_models=agent_models,
        agent_prompts={
            agent: st.session_state[f"run_{agent}_prompt"] for agent in agents
        },
        provider_api_keys={"gemini": _api_key("gemini")},
        provider_base_urls={
            provider_name: _base_url(provider_name)
            for provider_name in LOCAL_BASE_URLS
        },
    )
    settings.validate()
    return settings


def _restore_run_settings(run_type: RunType, settings: ProviderSettings) -> None:
    st.session_state["run_token_ceiling"] = settings.token_ceiling
    if run_type is RunType.STAGED_SINGLE_AGENT:
        first_agent = RUN_CONFIG_AGENTS[run_type][0]
        st.session_state["run_staged_provider"] = settings.provider_for(first_agent)
        st.session_state["run_staged_model"] = settings.model_for(first_agent)
    else:
        for agent in RUN_CONFIG_AGENTS[run_type]:
            st.session_state[f"run_{agent}_provider"] = settings.provider_for(agent)
            st.session_state[f"run_{agent}_model"] = settings.model_for(agent)
    for agent in RUN_CONFIG_AGENTS[run_type]:
        st.session_state[f"run_{agent}_prompt"] = settings.prompt_for(agent)


def _render_run_summary(run_type: RunType, settings: ProviderSettings) -> None:
    st.markdown("#### Run settings")
    st.table(
        [
            {
                "Agent / step": RUN_AGENT_LABELS[agent],
                "Provider": _provider_label(settings.provider_for(agent)),
                "Model": settings.model_for(agent),
            }
            for agent in RUN_CONFIG_AGENTS[run_type]
        ]
    )
    st.caption("The exact provider, model, and prompt configuration is saved with the run.")


def _render_create(repository: RunRepository) -> None:
    st.button("Back to runs", on_click=_go_home)
    st.title("Create a test suite")
    active_step = st.session_state.setdefault("create_step", 1)
    _render_create_steps(active_step)

    if active_step == 1:
        st.markdown("### Select a run type")
        run_type = st.radio(
            "Run type",
            list(RunType),
            key="run_type_choice",
            format_func=_run_type_label,
            horizontal=True,
        )
        st.info(RUN_TYPE_COPY[run_type][1])
        if st.button("Continue to settings", type="primary", width="stretch"):
            st.session_state["run_type"] = run_type
            _initialize_run_settings(run_type, repository)
            st.session_state["create_step"] = 2
            st.rerun()
        return

    run_type = st.session_state["run_type"]
    if active_step == 2:
        st.markdown("### Configure this run")
        st.caption("Each prompt starts with a working default and can be edited here.")
        _render_run_settings(run_type)
        back_column, continue_column = st.columns(2)
        if back_column.button("Back to run type", width="stretch"):
            st.session_state["create_step"] = 1
            st.rerun()
        if continue_column.button(
            "Continue to document", type="primary", width="stretch"
        ):
            try:
                st.session_state["run_provider_settings"] = _run_provider_settings(
                    run_type
                )
            except ValueError as error:
                st.error(str(error))
                return
            st.session_state["create_step"] = 3
            st.rerun()
        return

    provider_settings = st.session_state.get("run_provider_settings")
    if not isinstance(provider_settings, ProviderSettings):
        st.session_state["create_step"] = 2
        st.rerun()

    st.markdown("### Add your source document")
    _render_run_summary(run_type, provider_settings)
    if st.button("Edit run settings"):
        if current_upload := st.session_state.get("pdf"):
            st.session_state["retained_pdf"] = current_upload
        _restore_run_settings(run_type, provider_settings)
        st.session_state["create_step"] = 2
        st.rerun()
    selected_upload = st.file_uploader(
        "BRD/SRS PDF",
        type=["pdf"],
        key="pdf",
        help="Use a text-extractable PDF.",
    )
    if selected_upload is not None:
        st.session_state["retained_pdf"] = selected_upload
    upload = selected_upload or st.session_state.get("retained_pdf")
    if upload is None:
        st.caption(
            "Add a PDF to continue. Scanned PDFs without selectable text are not "
            "supported."
        )
    elif selected_upload is None:
        st.caption(f"Using your previously selected file: {upload.name}")
    generate = st.button(
        "Generate test cases",
        type="primary",
        key="run",
        width="stretch",
        disabled=upload is None,
    )

    if run_type is RunType.CENTRALIZED_MULTI_AGENT:
        _render_centralized_create(
            repository, provider_settings, upload, generate
        )
        return
    if not generate:
        return
    if upload is None:
        return

    try:
        with st.status("Live generation activity", expanded=True) as status:
            runner = st.session_state.get("_runner", run_generation)
            result = runner(
                upload.getvalue(),
                upload.name,
                run_type,
                provider_settings,
                repository=repository,
                progress=status.write,
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
    if message := st.session_state.pop("flash_toast", None):
        st.toast(message, icon=":material/warning:")
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
    _render_top_nav()
    _render_flashes()

    try:
        repository = _resolve_repository()
    except StorageError:
        st.error(
            "Runs database is unavailable. Start it with "
            "`docker compose up -d db`, verify DATABASE_URL, and refresh this page."
        )
        st.stop()

    st.session_state.setdefault("view", "runs")
    view = st.session_state["view"]
    if view == "create":
        _render_create(repository)
    elif view == "detail":
        _render_detail(repository)
    else:
        st.session_state["view"] = "runs"
        _render_runs(repository)

main()
