from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from brd_srs_testgen.browser_settings import (
    AppSettings,
    parse_settings,
    sync_browser_settings,
)
from brd_srs_testgen.models import (
    AgentSetup,
    ActivityEvent,
    FailureCategory,
    RunHistoryItem,
    RunMetrics,
    RunResult,
    RunStatus,
    RunType,
    default_agent_setups,
)
from brd_srs_testgen.providers import list_lm_studio_models
from brd_srs_testgen.runner import ProviderSettings, run_generation
from brd_srs_testgen.storage import RunRepository, StorageError


GEMINI_DEFAULT_MODEL = "gemini-3.6-flash"
PROVIDER_LABELS = {
    "gemini": "Gemini",
    "lm_studio": "LM Studio",
    "llama_cpp": "llama.cpp",
    "ollama": "Ollama",
}
LOCAL_BASE_URLS = {
    "lm_studio": "http://localhost:1234/v1",
    "llama_cpp": "http://localhost:8080/v1",
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
AGENT_LABELS = {
    "analyst": "Analyst",
    "test_generator": "Test generator",
    "reviewer": "Reviewer",
    "coverage_analyzer": "Coverage analyzer",
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
    if provider == "lm_studio":
        return _env("LM_STUDIO_BASE_URL") or LOCAL_BASE_URLS[provider]
    if provider == "llama_cpp":
        return _env("LLAMA_CPP_BASE_URL") or LOCAL_BASE_URLS[provider]
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
            --color-accent-strong: #1d4ed8;
            --color-accent-soft: #eff6ff;
            --color-background: #f8fafc;
            --color-surface: #ffffff;
            --color-foreground: #0f172a;
            --color-muted: #475569;
            --color-faint: #94a3b8;
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
            background:
                radial-gradient(circle at 85% 0%, #eaf2ff 0, transparent 28rem),
                var(--color-background);
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stMainBlockContainer"] {
            max-width: none !important;
            width: 100%;
            padding-top: 1.5rem;
            padding-bottom: 4rem;
            padding-left: clamp(1.5rem, 5vw, 6rem);
            padding-right: clamp(1.5rem, 5vw, 6rem);
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--color-border);
            border-radius: 1rem;
            background: rgba(255, 255, 255, 0.94);
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
        .stButton > button {
            border-radius: 0.65rem;
            font-weight: 600;
        }
        .stButton > button[kind="primary"] {
            min-height: 3rem;
            border-radius: 0.7rem;
            border-color: var(--color-accent);
            background: var(--color-accent);
            font-weight: 700;
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.22);
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
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 0.02em;
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.28);
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
            font-size: 0.72rem;
        }
        /* ---------- Runs hero ---------- */
        .runs-hero {
            padding: 1.4rem 0 0.4rem;
        }
        .runs-hero__kicker {
            color: var(--color-accent);
            font-size: 0.72rem;
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
            font-size: 0.68rem;
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
            font-size: 0.74rem;
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
            .run-list-item__header {
                align-items: flex-start;
            }
        }
        /* ---------- Run detail ---------- */
        .run-context {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(min(100%, 10rem), 1fr));
            gap: 0.75rem 1.25rem;
            margin: 0.75rem 0 1.25rem;
            padding: 0.85rem 1rem;
            border: 1px solid var(--color-border);
            border-radius: 0.85rem;
            background: rgba(255, 255, 255, 0.72);
        }
        .run-context__label {
            color: var(--color-faint);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }
        .run-context__value {
            margin-top: 0.15rem;
            color: var(--color-foreground);
            font-size: 0.84rem;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            overflow-wrap: anywhere;
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
        /* ---------- Section heading ---------- */
        .section-heading {
            margin: 1.6rem 0 0.2rem;
            padding-bottom: 0.45rem;
            border-bottom: 1px solid var(--color-border);
            color: var(--color-foreground);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        [data-baseweb="popover"] {
            position: fixed !important;
            top: 1rem !important;
            right: 1rem !important;
            bottom: 1rem !important;
            left: auto !important;
            transform: none !important;
        }
        [data-baseweb="popover"] [data-testid="stPopoverBody"] {
            width: min(34rem, calc(100vw - 2rem));
            max-width: none;
            height: 100%;
            max-height: none;
            overflow-y: auto;
            border: 1px solid var(--color-border);
            border-radius: 1rem;
            box-shadow: 0 24px 70px rgba(15, 23, 42, 0.18);
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
            font-size: 0.72rem;
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
            font-size: 0.68rem;
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
    for agent in ("analyst", "test_generator", "reviewer", "coverage_analyzer"):
        st.session_state[f"settings_{agent}_model"] = ""
    st.session_state.pop("lm_studio_models", None)
    st.session_state.pop("lm_studio_model_error", None)


def _clear_lm_studio_models() -> None:
    st.session_state["settings_model"] = ""
    st.session_state.pop("lm_studio_models", None)
    st.session_state.pop("lm_studio_model_error", None)


def _assign_lm_studio_models(models: list[str]) -> None:
    if not models:
        return
    if st.session_state.get("settings_model") not in models:
        st.session_state["settings_model"] = models[0]
    for index, agent in enumerate(("analyst", "test_generator", "reviewer", "coverage_analyzer")):
        key = f"settings_{agent}_model"
        if st.session_state.get(key) not in models:
            st.session_state[key] = models[index % len(models)]


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
        _assign_lm_studio_models(models)
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


def _artifact_popover_key(event: ActivityEvent) -> str:
    return f"artifact-{event.agent}-{event.artifact_label}"


def _render_artifact_popover(event: ActivityEvent) -> None:
    label = event.artifact_label or "Published artifact"
    with st.popover(
        f"Open artifact · {label}",
        type="primary",
        icon=":material/description:",
        key=_artifact_popover_key(event),
    ):
        st.markdown(f"#### {event.agent or 'Agent'} · {label}")
        if event.task:
            st.caption(f"Produced while: {event.task}")
        st.json(event.artifact.model_dump(mode="json"), expanded=False)


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
                st.caption("Artifact published")
                _render_artifact_popover(event)


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


def _run_context_html(result: RunResult) -> str:
    manifest = result.manifest
    facts = (
        ("Run type", _run_type_label(manifest.run_type)),
        ("Provider", _provider_label(manifest.provider)),
        ("Model", manifest.model),
        ("Started", manifest.started_at.strftime("%Y-%m-%d %H:%M UTC")),
    )
    return "<div class='run-context'>" + "".join(
        "<div>"
        f"<div class='run-context__label'>{label}</div>"
        f"<div class='run-context__value'>{html.escape(value)}</div>"
        "</div>"
        for label, value in facts
    ) + "</div>"


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
            color=["#15803d", "#b91c1c", "#b45309"],
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


def _render_result(
    result: RunResult, *, key_prefix: str = "result", in_dialog: bool = False
) -> None:
    manifest = result.manifest
    st.markdown(_run_context_html(result), unsafe_allow_html=True)

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
        st.success("Completed and validated.")
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
        st.markdown("### Quality and traceability")
        st.markdown(_quality_chart_html(metrics), unsafe_allow_html=True)
        st.caption(
            f"Latency {metrics.latency_seconds:.2f} s · "
            f"{metrics.retries} retries"
        )

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

    if result.coverage is not None:
        _render_coverage(result.coverage, key_prefix=key_prefix, run_id=manifest.run_id)

    if result.bundle is not None:
        _render_bundle(result, key_prefix=key_prefix, in_dialog=in_dialog)

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


def _settings_error() -> str | None:
    try:
        st.session_state["app_settings"].provider_settings()
    except ValueError as error:
        return str(error)
    return None


def _open_settings(after_save: str | None = None) -> None:
    settings = st.session_state["app_settings"]
    st.session_state["settings_provider"] = settings.provider
    st.session_state["settings_model"] = settings.model
    st.session_state["settings_api_key"] = settings.api_key
    st.session_state["settings_base_url"] = settings.base_url
    st.session_state["settings_token_ceiling"] = settings.token_ceiling
    for agent in ("analyst", "test_generator", "reviewer", "coverage_analyzer"):
        st.session_state[f"settings_{agent}_model"] = getattr(
            settings, f"{agent}_model"
        )
    st.session_state.pop("agent_setup_form_loaded", None)
    st.session_state["show_settings"] = True
    if after_save is None:
        st.session_state.pop("settings_after_persist", None)
    else:
        st.session_state["settings_after_persist"] = after_save


def _load_agent_setup_form(repository: RunRepository) -> bool:
    if st.session_state.get("agent_setup_form_loaded"):
        return True
    try:
        setups = repository.load_agent_setups()
    except StorageError:
        st.error("Shared agent setup is unavailable. Check PostgreSQL and try again.")
        return False
    for agent, setup in setups.items():
        st.session_state[f"agent_setup_{agent}_role"] = setup.role
        st.session_state[f"agent_setup_{agent}_instructions"] = setup.instructions
    st.session_state["agent_setup_form_loaded"] = True
    return True


def _agent_setups_from_form() -> dict[str, AgentSetup]:
    return {
        agent: AgentSetup(
            agent=agent,
            role=st.session_state[f"agent_setup_{agent}_role"],
            instructions=st.session_state[f"agent_setup_{agent}_instructions"],
        )
        for agent in default_agent_setups()
    }


@st.dialog("App settings", width="large")
def _settings_dialog(repository: RunRepository) -> None:
    if not _load_agent_setup_form(repository):
        return
    if message := st.session_state.pop("settings_required_message", None):
        st.error(message)
    st.markdown(
        "<div class='section-heading' style='margin-top:0'>Provider</div>",
        unsafe_allow_html=True,
    )
    provider = st.selectbox(
        "Provider",
        list(PROVIDER_LABELS),
        key="settings_provider",
        format_func=_provider_label,
        on_change=_reset_provider,
    )
    if provider == "lm_studio":
        if "lm_studio_models" not in st.session_state:
            _refresh_lm_studio_models()
        models = st.session_state.get("lm_studio_models", [])
        st.selectbox(
            "Model",
            models,
            index=0 if models else None,
            key="settings_model",
            placeholder="Models load automatically from LM Studio",
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
        if error := st.session_state.get("lm_studio_model_error"):
            st.error(f"Could not load models: {error}")
        elif models:
            st.success(f"Loaded {len(models)} models.")

    st.markdown("#### Centralized agents")
    st.caption(
        "Select an agent to edit its model routing, role, and prompt instructions. "
        "Roles and instructions are shared in PostgreSQL; provider credentials stay in this browser."
    )
    if provider == "lm_studio" and models:
        st.caption(
            "LM Studio loads each selected agent model automatically when its task begins."
        )
    else:
        st.caption("Leave a model override blank to use the default Model above.")
    for agent, label in AGENT_LABELS.items():
        configured_model = st.session_state[f"settings_{agent}_model"].strip()
        model_summary = configured_model or "Uses default model"
        role = st.session_state[f"agent_setup_{agent}_role"]
        with st.expander(f"{label} · {role} · {model_summary}"):
            st.text_input(f"{label} role", key=f"agent_setup_{agent}_role")
            if provider == "lm_studio" and models:
                st.selectbox(f"{label} model", models, key=f"settings_{agent}_model")
            else:
                st.text_input(
                    f"{label} model (optional)", key=f"settings_{agent}_model"
                )
            st.text_area(
                f"{label} additional instructions",
                key=f"agent_setup_{agent}_instructions",
                max_chars=4_000,
                placeholder="Optional instructions applied only to this agent's prompt.",
            )

    st.markdown(
        "<div class='section-heading'>Limits</div>",
        unsafe_allow_html=True,
    )
    st.number_input(
        "Token ceiling",
        min_value=1000,
        step=1000,
        key="settings_token_ceiling",
    )
    st.warning(
        "Credentials are stored in this browser's local storage. "
        "Scripts running on the same app origin can read stored credentials. "
        "Use a dedicated browser profile and do not save credentials on a shared machine."
    )
    save_column, cancel_column = st.columns(2)
    save = save_column.button("Save settings", type="primary", width="stretch")
    cancel = cancel_column.button("Cancel", width="stretch")
    if cancel:
        st.session_state["show_settings"] = False
        st.session_state.pop("settings_after_persist", None)
        st.session_state.pop("agent_setup_form_loaded", None)
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
            analyst_model=st.session_state["settings_analyst_model"],
            test_generator_model=st.session_state["settings_test_generator_model"],
            reviewer_model=st.session_state["settings_reviewer_model"],
            coverage_analyzer_model=st.session_state["settings_coverage_analyzer_model"],
        )
        settings.provider_settings()
        agent_setups = _agent_setups_from_form()
        repository.save_agent_setups(agent_setups.values())
    except (StorageError, ValueError) as error:
        st.error(str(error))
        return
    st.session_state["settings_revision"] = (
        st.session_state.get("settings_revision", 0) + 1
    )
    st.session_state["settings_save_request"] = settings.model_dump(mode="json")
    st.session_state["show_settings"] = False
    st.session_state.pop("agent_setup_form_loaded", None)
    st.rerun()


def _go_home() -> None:
    st.session_state["view"] = "runs"
    st.session_state.pop("selected_run_id", None)
    st.session_state.pop("selected_run", None)
    st.session_state.pop("timeline_result", None)
    st.session_state.pop("timeline_activity", None)
    st.session_state.pop("timeline_current_step", None)


def _render_top_nav() -> None:
    brand, settings = st.columns([9, 1], vertical_alignment="center")
    brand.markdown(
        "<div class='app-bar'>"
        "<span class='app-bar__mark'>TC</span>"
        "<div>"
        "<div class='app-bar__name'>BRD/SRS Test Case</div>"
        "<div class='app-bar__tag'>Traceable test-case generation</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    settings.button("Settings", on_click=_open_settings, width="stretch")
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
        "<div><div class='run-list-item__label'>Run type</div>"
        f"<div class='run-list-item__value'>{html.escape(_run_type_label(item.run_type))}</div></div>"
        "<div><div class='run-list-item__label'>Provider</div>"
        f"<div class='run-list-item__value'>{html.escape(_provider_label(item.provider))}</div></div>"
        "<div><div class='run-list-item__label'>Model</div>"
        f"<div class='run-list-item__value'>{html.escape(item.model)}</div></div>"
        "<div><div class='run-list-item__label'>Started</div>"
        f"<div class='run-list-item__value'>{item.started_at.strftime('%Y-%m-%d %H:%M UTC')}</div></div>"
        "<div><div class='run-list-item__label'>Test cases</div>"
        f"<div class='run-list-item__value'>{test_cases}</div></div>"
        "</div>"
        "</div>"
    )


def _request_create() -> None:
    if st.session_state.get("settings_save_request") is not None:
        st.session_state["runs_notice"] = "Saving browser settings…"
    elif not st.session_state.get("browser_settings_loaded"):
        st.session_state["runs_notice"] = "Browser settings are still loading."
    else:
        error = _settings_error()
        if error is None:
            st.session_state["view"] = "create"
            return
        message = f"{error} Add it in Settings before creating a run."
        st.session_state["flash_toast"] = message
        st.session_state["settings_required_message"] = message
        _open_settings("create")


def _render_runs(repository: RunRepository) -> None:
    try:
        runs = repository.list_runs()
    except StorageError:
        runs = None

    st.markdown(
        "<div class='runs-hero'>"
        "<div class='runs-hero__kicker'>Dashboard</div>"
        "<div class='runs-hero__title'>Runs</div>"
        "<div class='runs-hero__sub'>"
        "Upload a BRD or SRS, generate a traceable test suite, and reopen any "
        "saved result. Newest runs appear first."
        "</div>"
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
        st.info("No saved runs yet.")
        return

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
    repository: RunRepository, settings: AppSettings
) -> None:
    try:
        agent_setups = repository.load_agent_setups()
    except StorageError:
        st.error("Shared agent setup is unavailable. Check PostgreSQL and try again.")
        return
    with st.container():
        st.markdown("#### Run timeline")
        st.markdown(
            "<div class='timeline-spine' aria-hidden='true'></div>",
            unsafe_allow_html=True,
        )
        source_stage = st.empty()
        with source_stage.container():
            _render_timeline_stage(
                1,
                "Configure and start",
                "Choose a source document, confirm the run settings, then start generation.",
                state="active",
            )
        with st.container(border=True):
            st.markdown("##### Source document")
            upload = st.file_uploader(
                "BRD/SRS PDF",
                type=["pdf"],
                key="pdf",
                help="Use a text-extractable PDF.",
            )
            st.markdown("##### Run settings")
            st.caption(
                f"{_provider_label(settings.provider)} · {settings.model} · "
                f"{settings.token_ceiling:,} token ceiling"
            )
            if any(
                settings.model_for(agent) != settings.model
                for agent in ("analyst", "test_generator", "reviewer", "coverage_analyzer")
            ):
                st.caption(
                    "Routing: "
                    f"Analyst {settings.model_for('analyst')} · "
                    f"Generator {settings.model_for('test_generator')} · "
                    f"Reviewer {settings.model_for('reviewer')} · "
                    f"Coverage {settings.model_for('coverage_analyzer')}"
                )
            st.caption(
                "Agents: "
                + " · ".join(
                    f"{AGENT_LABELS[agent]} — {setup.role}"
                    for agent, setup in agent_setups.items()
                )
            )
            st.button("Edit settings", on_click=_open_settings, args=("create",))
            settings_pending = st.session_state.get("settings_save_request") is not None
            settings_loaded = st.session_state.get("browser_settings_loaded")
            if settings_pending:
                st.info("Saving browser settings…")
            elif not settings_loaded:
                st.info("Browser settings are still loading.")
            generate = (
                False
                if settings_pending or not settings_loaded
                else st.button(
                    "Generate test cases",
                    type="primary",
                    key="run",
                    width="stretch",
                )
            )

        live_stage = st.empty()
        with live_stage.container():
            _render_timeline_stage(
                2,
                "Generate and validate",
                "Agents will publish their task, scope, checkpoint, and artifact here.",
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
                    1, "Configure and start", "Source locked for this saved run.", state="complete"
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
                    feed_column, plan_column = st.columns((1.7, 0.9), gap="large")
                    with feed_column:
                        st.markdown("#### Agent activity")
                        st.caption("Live handoffs, current tasks, scopes, and checkpoints.")
                        activity = st.container(height=420, key="live-feed")
                        for event in activity_events:
                            _render_activity(activity, event)
                        _scroll_live_feed()
                    with plan_column:
                        st.markdown("#### Plan")
                        plan = st.empty()
                        _render_activity_plan(
                            plan, current_step, stopped=not completed
                        )
                        st.caption("Private model reasoning is not displayed.")
            with result_panel.container(border=True):
                _render_timeline_result_action(result)
        else:
            live_panel.empty()
        return

    if upload is None:
        st.error("Upload one text-extractable PDF before generating test cases.")
        return

    try:
        provider_settings = settings.provider_settings().with_agent_setups(agent_setups)
    except ValueError as error:
        st.error(str(error))
        _open_settings(after_save="create")
        return

    st.session_state.pop("timeline_result", None)
    activity_events: list[str] = []
    st.session_state["timeline_activity"] = activity_events
    st.session_state["timeline_current_step"] = 0
    source_stage.empty()
    with source_stage.container():
        _render_timeline_stage(
            1, "Configure and start", "Source and settings are locked for this run.", state="complete"
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
            feed_column, plan_column = st.columns((1.7, 0.9), gap="large")
            with feed_column:
                st.markdown("#### Agent activity")
                st.caption("Agent handoffs, current tasks, scopes, and checkpoints.")
                activity = st.container(height=420, key="live-feed")
            with plan_column:
                st.markdown("#### Plan")
                plan = st.empty()
                st.caption("Private model reasoning is not displayed.")

            current_step = 0
            _render_activity_plan(plan, current_step)

            def progress(event: str) -> None:
                nonlocal current_step
                current_step = _activity_plan_step(event, current_step)
                st.session_state["timeline_current_step"] = current_step
                _render_activity_plan(plan, current_step)
                _render_activity(activity, event)
                activity_events.append(event)
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


def _render_create(repository: RunRepository, settings: AppSettings) -> None:
    st.button("Back to runs", on_click=_go_home)
    st.title("Create a test suite")
    st.caption(
        "Upload a BRD or SRS, confirm the generation strategy, and track every stage of the run."
    )
    run_type = st.selectbox(
        "Run type",
        list(RunType),
        key="run_type",
        format_func=_run_type_label,
    )
    st.caption(RUN_TYPE_COPY[run_type][1])
    if run_type is RunType.CENTRALIZED_MULTI_AGENT:
        st.caption(
            "Live activity shows orchestrator handoffs and agent status, not private model reasoning."
        )
        _render_centralized_create(repository, settings)
        return

    upload = st.file_uploader(
        "BRD/SRS PDF",
        type=["pdf"],
        key="pdf",
        help="Use a text-extractable PDF.",
    )
    st.markdown("### App settings")
    st.caption(
        f"{_provider_label(settings.provider)} · {settings.model} · "
        f"{settings.token_ceiling:,} token ceiling"
    )
    st.button("Edit settings", on_click=_open_settings, args=("create",))
    if st.session_state.get("settings_save_request") is not None:
        st.info("Saving browser settings…")
        return
    if not st.session_state.get("browser_settings_loaded"):
        st.info("Browser settings are still loading.")
        return
    if not st.button(
        "Generate test cases",
        type="primary",
        key="run",
        width="stretch",
    ):
        return
    if upload is None:
        st.error("Upload one text-extractable PDF before generating test cases.")
        return

    try:
        provider_settings = settings.provider_settings()
    except ValueError as error:
        st.error(str(error))
        _open_settings(after_save="create")
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
        st.toast(message, icon="⚠️")
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
            "Runs database is unavailable. Start it with "
            "`docker compose up -d db`, verify DATABASE_URL, and refresh this page."
        )
        st.stop()

    st.session_state.setdefault("view", "runs")
    view = st.session_state["view"]
    if view == "create":
        _render_create(repository, st.session_state["app_settings"])
    elif view == "detail":
        _render_detail(repository)
    else:
        st.session_state["view"] = "runs"
        _render_runs(repository)

    if st.session_state.get("show_settings"):
        _settings_dialog(repository)


main()
