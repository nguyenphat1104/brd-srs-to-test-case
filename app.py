from __future__ import annotations

import json
from typing import Any

import streamlit as st

from brd_srs_testgen.models import Condition, RunStatus
from brd_srs_testgen.runner import ComparisonResult, ConditionResult, ProviderSettings, run_comparison


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
    )


def _render_condition(result: ConditionResult) -> None:
    manifest = result.manifest
    metrics = result.metrics
    condition = manifest.condition.value
    st.subheader(condition.replace("_", " ").title())
    if manifest.status is RunStatus.COMPLETED:
        st.success("Completed")
    else:
        category = (
            manifest.failure_category.value.replace("_", " ").title()
            if manifest.failure_category
            else "Unknown failure"
        )
        st.error(f"{category}: {manifest.failure_message or 'No failure message provided.'}")

    charged = metrics.charged_tokens
    token_label = "Charged tokens" if charged else "Reported tokens"
    token_value = charged or metrics.input_tokens + metrics.output_tokens
    st.metric("Requirements", metrics.requirement_count)
    st.metric("Test cases", metrics.test_case_count)
    st.metric("RTM completeness", f"{metrics.rtm_completeness:.0%}")
    st.metric(token_label, token_value)
    st.metric("Latency", f"{metrics.latency_seconds:.2f} s")

    if result.bundle:
        _download(
            "Download requirements",
            [item.model_dump(mode="json") for item in result.bundle.requirements],
            f"{condition}-requirements.json",
            f"{condition}-requirements",
        )
        _download(
            "Download scenarios",
            [item.model_dump(mode="json") for item in result.bundle.scenarios],
            f"{condition}-scenarios.json",
            f"{condition}-scenarios",
        )
        _download(
            "Download test cases",
            [item.model_dump(mode="json") for item in result.bundle.test_cases],
            f"{condition}-test-cases.json",
            f"{condition}-test-cases",
        )
    _download(
        "Download RTM",
        [item.model_dump(mode="json") for item in result.rtm],
        f"{condition}-rtm.json",
        f"{condition}-rtm",
    )
    _download(
        "Download complete condition bundle",
        result.download_bundle(),
        f"{condition}-bundle.json",
        f"{condition}-bundle",
    )


def _render_result(result: ComparisonResult) -> None:
    if result.failure_category:
        category = result.failure_category.value.replace("_", " ").title()
        st.error(f"Comparison failed — {category}: {result.failure_message or 'No message provided.'}")
        return
    columns = st.columns(len(result.manifest.condition_order))
    for column, condition in zip(columns, result.manifest.condition_order, strict=True):
        with column:
            condition_result = result.conditions.get(condition)
            if condition_result is None:
                st.error(f"{condition.value.replace('_', ' ').title()}: no result returned.")
            else:
                _render_condition(condition_result)


def main() -> None:
    st.set_page_config(page_title="BRD/SRS Test-Case Research Core", layout="wide")
    st.title("BRD/SRS Test-Case Research Core")
    st.write("Compare three controlled generation conditions for one text-extractable BRD or SRS PDF.")

    with st.sidebar:
        st.header("Provider configuration")
        provider = st.selectbox("Provider", ["gemini", "ollama"])
        model = st.text_input("Model", value="gemini-2.5-flash" if provider == "gemini" else "llama3.2")
        api_key = ""
        base_url = "http://localhost:11434"
        if provider == "gemini":
            api_key = st.text_input("Gemini API key", type="password")
        else:
            base_url = st.text_input("Ollama base URL", value=base_url)
        token_ceiling = st.number_input(
            "Token ceiling", min_value=1000, value=100_000, step=1000
        )
        st.caption("Temperature is fixed at 0.0. Centralized condition uses 3 workers.")

    uploaded_pdf = st.file_uploader("BRD/SRS PDF", type=["pdf"])
    if st.button("Run all three conditions", type="primary"):
        if uploaded_pdf is None:
            st.error("Upload one PDF before running the comparison.")
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
                with st.status("Running comparison", expanded=True) as status:
                    def progress(condition: Condition | None, message: str) -> None:
                        name = condition.value if condition else "comparison"
                        status.write(f"{name}: {message}")

                    runner = st.session_state.get("_runner", run_comparison)
                    st.session_state["comparison_result"] = runner(
                        uploaded_pdf.getvalue(), settings, progress
                    )
                    status.update(label="Comparison complete", state="complete", expanded=False)
            except Exception as error:
                st.error(f"Comparison failed: {error}")

    result = st.session_state.get("comparison_result")
    if result is not None:
        _render_result(result)


main()
