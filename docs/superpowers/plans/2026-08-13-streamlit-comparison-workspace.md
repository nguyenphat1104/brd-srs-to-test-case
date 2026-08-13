# Streamlit Comparison Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the run-centric Streamlit experience with a responsive QA workbench that makes baseline-versus-agentic quality, role activity, traceability, and human-reviewed F1 easy to inspect.

**Architecture:** Keep `app.py` as the single Streamlit composition root and reuse its settings, safe-error, artifact, and repository helpers. Add comparison list/create/detail states, then organize a selected comparison into seven task-oriented tabs. Existing standalone runs remain accessible as a secondary history screen; the default experience becomes comparisons.

**Tech Stack:** Streamlit, existing Pydantic/service APIs, native CSS, Streamlit AppTest, pytest. No new frontend or chart dependency.

---

## Prerequisites

Complete these plans first:

1. `docs/superpowers/plans/2026-08-13-human-gold-f1-evaluation.md`
2. `docs/superpowers/plans/2026-08-13-role-based-agentic-comparison-core.md`

This plan assumes `RunRepository.list_comparisons/load_comparison/list_gold_label_sets/save_gold_label_set`, `run_comparison`, `review_comparison`, and the evaluation candidate APIs exist.

## Experience contract

The primary information architecture is:

```text
Comparisons
  Create comparison
  Saved comparison -> Overview
                      Agent activity
                      Requirements
                      Scenarios & test cases
                      Traceability
                      Evaluation
                      Configuration

Runs
  Existing standalone run history and detail

Settings
  Existing browser-local provider settings
```

The reference app informs the compact workbench density, artifact-first navigation, and coverage visibility. This app does not copy unrelated CI/CD, infrastructure, alerting, or Excel features.

## Visual system

- Canvas `#F6F7FB`; surface `#FFFFFF`; border `#D9DFEA`.
- Primary action `#3157D5`; agentic accent `#6D3BD1`; success `#087A55`; warning `#9A5B00`; danger `#B42318`.
- Foreground `#101828`; secondary text `#475467`.
- System sans stack; no remote font dependency.
- Eight-pixel spacing rhythm; 12px card radius; restrained shadow only on elevated panels.
- Minimum 44px interactive targets, visible keyboard focus, text labels for every control, reduced-motion support.
- Desktop uses side-by-side condition cards and master-detail columns; below 768px everything becomes one column without horizontal scrolling.

## Task 1: Add comparison fixtures and repository test doubles

**Files:**

- Modify: `tests/factories.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Add a reusable completed comparison fixture**

Append to `tests/factories.py`:

```python
from brd_srs_testgen.models import (
    AgentActivity,
    AgentRole,
    AgentState,
    ComparisonManifest,
    ComparisonResult,
    ComparisonStatus,
)


def completed_comparison(comparison_id: str = "cmp-001") -> ComparisonResult:
    now = datetime.now(UTC)
    baseline = completed_run("baseline-run", RunType.SINGLE_PROMPT)
    agentic = completed_run("agentic-run", RunType.ROLE_BASED_AGENTIC)
    return ComparisonResult(
        manifest=ComparisonManifest(
            comparison_id=comparison_id,
            source_filename="sample.pdf",
            document_hash="a" * 64,
            status=ComparisonStatus.COMPLETED,
            provider="gemini",
            model="gemini-3.6-flash",
            temperature=0,
            token_ceiling=100_000,
            research_enabled=True,
            label_set_id="gold-001",
            prompt_version="research-core-v4",
            schema_version="research-core-v2",
            started_at=now,
            completed_at=now,
            baseline_run_id=baseline.manifest.run_id,
            agentic_run_id=agentic.manifest.run_id,
        ),
        baseline=baseline,
        agentic=agentic,
        activity=[
            AgentActivity(
                sequence=1,
                role=AgentRole.REQUIREMENT_ANALYST,
                state=AgentState.COMPLETED,
                summary="Extracted 1 requirement.",
                routing_reason="The analyst always runs first.",
                occurred_at=now,
                details={"clarification_count": 0},
            ),
            AgentActivity(
                sequence=2,
                role=AgentRole.INTERNET_RESEARCHER,
                state=AgentState.SKIPPED,
                summary="No external clarification required.",
                routing_reason="No test-blocking external ambiguity was found.",
                occurred_at=now,
            ),
        ],
    )
```

- [ ] **Step 2: Extend `FakeRepository` with comparison behavior**

Add constructor parameters and methods:

```python
comparisons: list[ComparisonManifest] | None = None,
comparison_results: dict[str, ComparisonResult] | None = None,
```

```python
self.comparisons = comparisons or []
self.comparison_results = comparison_results or {}
self.comparison_load_calls: list[str] = []
self.saved_gold_labels: list[GoldLabelSet] = []
self.saved_evaluations: list[EvaluationRevision] = []

def list_comparisons(self) -> list[ComparisonManifest]:
    return self.comparisons

def load_comparison(self, comparison_id: str) -> ComparisonResult:
    self.comparison_load_calls.append(comparison_id)
    try:
        return self.comparison_results[comparison_id]
    except KeyError as error:
        raise StorageError("Comparison does not exist.") from error

def list_gold_label_sets(self, document_hash: str) -> list[GoldLabelSet]:
    return [
        labels
        for labels in self.saved_gold_labels
        if labels.document_hash == document_hash
    ]

def save_gold_label_set(self, labels: GoldLabelSet) -> None:
    self.saved_gold_labels.append(labels)

def load_gold_label_set(self, label_set_id: str) -> GoldLabelSet:
    return next(
        labels for labels in self.saved_gold_labels
        if labels.label_set_id == label_set_id
    )

def save_evaluation_revision(self, revision: EvaluationRevision) -> None:
    self.saved_evaluations.append(revision)

def list_latest_evaluation_revisions(
    self, research_enabled: bool | None = None
) -> list[EvaluationRevision]:
    latest = {}
    for revision in sorted(self.saved_evaluations, key=lambda item: item.created_at):
        comparison = self.comparison_results.get(revision.comparison_id)
        if (
            research_enabled is not None
            and comparison is not None
            and comparison.manifest.research_enabled is not research_enabled
        ):
            continue
        latest[revision.comparison_id] = revision
    return list(latest.values())
```

Import the new models and `completed_comparison` at the top of `tests/test_app.py`.

- [ ] **Step 3: Run the existing app suite**

Run:

```bash
rtk .venv/bin/pytest tests/test_app.py -q
```

Expected: all existing tests still pass; no app behavior has changed yet.

- [ ] **Step 4: Commit test infrastructure**

```bash
rtk git add tests/factories.py tests/test_app.py
rtk git commit -m "test: add comparison app fixtures"
```

## Task 2: Make comparisons the primary navigation state

**Files:**

- Modify: `app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Add failing navigation and empty-state tests**

Add:

```python
def test_comparisons_are_the_default_home_and_runs_remain_available() -> None:
    at = _app_test().run()

    assert at.title[0].value == "Comparisons"
    assert "No saved comparisons yet" in _rendered_text(at)
    assert _element(at.button, "Create comparison")
    assert _element(at.button, "Runs")


def test_saved_comparison_selection_opens_comparison_detail() -> None:
    comparison = completed_comparison()
    repository = FakeRepository(
        comparisons=[comparison.manifest],
        comparison_results={comparison.manifest.comparison_id: comparison},
    )
    at = _app_test(repository)
    at.session_state["view"] = "comparison_detail"
    at.session_state["selected_comparison_id"] = comparison.manifest.comparison_id

    at.run()

    assert at.title[0].value == "sample.pdf"
    assert repository.comparison_load_calls == [comparison.manifest.comparison_id]
```

- [ ] **Step 2: Confirm the default-home test fails**

Run:

```bash
rtk .venv/bin/pytest tests/test_app.py -q
```

Expected: the app still opens on `Runs` and the new tests fail.

- [ ] **Step 3: Add state transitions and compact top navigation**

Add:

```python
def _go_comparisons() -> None:
    st.session_state["view"] = "comparisons"
    st.session_state.pop("selected_comparison_id", None)
    st.session_state.pop("selected_comparison", None)
    st.session_state.pop("comparisons-table", None)
    st.session_state.pop("displayed_comparison_ids", None)


def _go_runs() -> None:
    _go_home()
```

Change `_go_home` to continue setting `view = "runs"` for backward-compatible run-detail buttons. Replace `_render_top_nav` with three labeled controls:

```python
def _render_top_nav() -> None:
    brand, comparisons, runs, settings = st.columns([5, 1.5, 1, 1])
    brand.markdown("<div class='brand'>BRD/SRS QA Workbench</div>", unsafe_allow_html=True)
    comparisons.button("Comparisons", on_click=_go_comparisons, width="stretch")
    runs.button("Runs", on_click=_go_runs, width="stretch")
    settings.button("Settings", on_click=_open_settings, width="stretch")
```

- [ ] **Step 4: Render comparison history**

Add `_request_create_comparison` mirroring `_request_create`, but set `view = "create_comparison"`. Add:

```python
def _render_comparisons(repository: RunRepository) -> None:
    selection = st.session_state.get("comparisons-table", {})
    rows = selection.get("selection", {}).get("rows", []) if hasattr(selection, "get") else []
    displayed = st.session_state.get("displayed_comparison_ids", [])
    if rows and 0 <= rows[0] < len(displayed):
        st.session_state["selected_comparison_id"] = displayed[rows[0]]
        st.session_state.pop("selected_comparison", None)
        st.session_state["view"] = "comparison_detail"
        st.rerun()

    heading, action = st.columns([4, 1])
    heading.title("Comparisons")
    heading.caption("Measure one-prompt generation against the role-based workflow.")
    action.button(
        "Create comparison",
        type="primary",
        on_click=_request_create_comparison,
        width="stretch",
    )
    comparisons = repository.list_comparisons()
    if not comparisons:
        st.info("No saved comparisons yet. Create one to measure agentic lift.")
        return
    st.session_state["displayed_comparison_ids"] = [
        item.comparison_id for item in comparisons
    ]
    st.dataframe(
        [
            {
                "Started": item.started_at.strftime("%Y-%m-%d %H:%M"),
                "Source": item.source_filename,
                "Provider": _provider_label(item.provider),
                "Model": item.model,
                "Research": "Enabled" if item.research_enabled else "Disabled",
                "Status": item.status.value.replace("_", " ").title(),
            }
            for item in comparisons
        ],
        hide_index=True,
        width="stretch",
        selection_mode="single-row",
        on_select="rerun",
        key="comparisons-table",
    )
```

Wrap `repository.list_comparisons()` with the same safe `StorageError` message pattern used by `_render_runs`.

- [ ] **Step 5: Route new views from `main`**

Change the default and dispatcher:

```python
st.session_state.setdefault("view", "comparisons")
view = st.session_state["view"]
if view == "create_comparison":
    _render_create_comparison(repository, st.session_state["app_settings"])
elif view == "comparison_detail":
    _render_comparison_detail(repository)
elif view == "create":
    _render_create(repository, st.session_state["app_settings"])
elif view == "detail":
    _render_detail(repository)
elif view == "runs":
    _render_runs(repository)
else:
    _go_comparisons()
    st.rerun()
```

Add these initial renderers so the dispatcher always has valid call targets; later tasks extend their bodies:

```python
def _render_create_comparison(
    repository: RunRepository, settings: AppSettings
) -> None:
    st.button("Back to comparisons", on_click=_go_comparisons)
    st.title("Create comparison")


def _render_comparison_detail(repository: RunRepository) -> None:
    st.button("Back to comparisons", on_click=_go_comparisons)
    st.title("Comparison")
```

- [ ] **Step 6: Run navigation tests**

Run:

```bash
rtk .venv/bin/pytest tests/test_app.py -q
```

Expected: new navigation tests and existing run/settings tests pass.

- [ ] **Step 7: Commit information architecture**

```bash
rtk git add app.py tests/test_app.py
rtk git commit -m "feat: make comparisons the qa workbench home"
```

## Task 3: Build the fair-comparison creation flow

**Files:**

- Modify: `app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Add failing creation tests**

Add a fake comparison runner and tests:

```python
def test_create_comparison_requires_pdf_and_research_consent() -> None:
    repository = FakeRepository()
    repository.saved_gold_labels = [
        gold_labels(document_hash=hashlib.sha256(b"%PDF-1.4\n").hexdigest())
    ]
    at = _app_test(repository)
    at.session_state["view"] = "create_comparison"
    at.run()

    _element(at.toggle, "Use grounded internet research").set_value(True)
    at.run()
    _element(at.button, "Run comparison").click().run()

    text = _rendered_text(at)
    assert "Upload one text-extractable PDF" in text
    assert "Confirm consent" in text


def test_create_comparison_requires_an_approved_matching_gold_version() -> None:
    at = _app_test()
    at.session_state["view"] = "create_comparison"
    at.run()

    assert "Upload and save an approved gold-label version" in _rendered_text(at)


def test_create_gold_upload_rejects_wrong_document_hash() -> None:
    repository = FakeRepository()
    at = _app_test(repository)
    at.session_state["view"] = "create_comparison"
    at.run()
    _element(at.file_uploader, "BRD/SRS PDF").set_value(
        ("sample.pdf", b"%PDF-1.4\n", "application/pdf")
    )
    wrong = gold_labels(document_hash="b" * 64).model_dump_json().encode()
    _element(at.file_uploader, "Gold label JSON").set_value(
        ("gold.json", wrong, "application/json")
    )
    _element(at.button, "Save gold labels").click().run()

    assert "does not match the uploaded PDF" in _rendered_text(at)
    assert repository.saved_gold_labels == []


def test_research_toggle_is_disabled_for_local_providers() -> None:
    browser = FakeBrowserSettings(_saved_settings(
        provider="ollama", model="gemma4", api_key="", base_url="http://localhost:11434"
    ))
    at = _app_test(browser=browser)
    at.session_state["view"] = "create_comparison"
    at.run()

    toggle = _element(at.toggle, "Use grounded internet research")
    assert toggle.disabled is True
    assert "Gemini" in _rendered_text(at)


def test_successful_comparison_opens_saved_detail() -> None:
    comparison = completed_comparison()
    calls = []

    def fake_runner(*args, **kwargs):
        calls.append((args, kwargs))
        return comparison

    repository = FakeRepository()
    repository.saved_gold_labels = [
        gold_labels(document_hash=hashlib.sha256(b"%PDF-1.4\n").hexdigest())
    ]
    at = _app_test(repository)
    at.session_state["view"] = "create_comparison"
    at.session_state["_comparison_runner"] = fake_runner
    at.run()
    _element(at.file_uploader, "BRD/SRS PDF").set_value(
        ("sample.pdf", b"%PDF-1.4\n", "application/pdf")
    )
    _element(at.button, "Run comparison").click().run()

    assert calls
    assert at.session_state["selected_comparison_id"] == "cmp-001"
    assert at.session_state["view"] == "comparison_detail"
```

- [ ] **Step 2: Confirm creation tests fail against the shell**

Run:

```bash
rtk .venv/bin/pytest tests/test_app.py -q
```

Expected: the creation controls are absent.

- [ ] **Step 3: Implement the form with explicit fairness copy**

Import `ComparisonSettings` and `run_comparison`, then implement:

```python
def _render_create_comparison(
    repository: RunRepository, settings: AppSettings
) -> None:
    st.button("Back to comparisons", on_click=_go_comparisons)
    st.title("Create comparison")
    st.caption(
        "Run the same document, model, temperature, and token ceiling through "
        "a one-prompt baseline and the role-based workflow."
    )
    upload = st.file_uploader(
        "BRD/SRS PDF", type=["pdf"], key="comparison-pdf",
        help="Use a text-extractable PDF."
    )
    research_supported = settings.provider == "gemini"
    research_enabled = st.toggle(
        "Use grounded internet research",
        value=False,
        disabled=not research_supported,
        help="Sends up to three short generic clarification queries to Google Search.",
    )
    if not research_supported:
        st.caption("Grounded internet research is available only with Gemini.")
    consent = st.checkbox(
        "I confirm consent to send generic clarification queries to Google Search.",
        disabled=not research_enabled,
    )

    document_hash = _document_hash(upload.getvalue()) if upload is not None else None
    labels = (
        repository.list_gold_label_sets(document_hash)
        if document_hash is not None
        else []
    )
    approved_labels = [
        item for item in labels if item.status is GoldLabelStatus.APPROVED
    ]
    selected_labels = st.selectbox(
        "Gold label version",
        approved_labels,
        format_func=lambda item: f"v{item.version} · {item.author}",
        index=0 if approved_labels else None,
        placeholder="Upload and save an approved gold-label version",
    )
    gold_upload = st.file_uploader(
        "Gold label JSON", type=["json"], key="comparison-gold-json"
    )
    if st.button("Save gold labels"):
        try:
            if upload is None or gold_upload is None:
                raise ValueError("Choose both the PDF and gold-label JSON first.")
            new_labels = GoldLabelSet.model_validate_json(gold_upload.getvalue())
            if new_labels.document_hash != document_hash:
                raise ValueError("Gold label set does not match the uploaded PDF.")
            if new_labels.status is not GoldLabelStatus.APPROVED:
                raise ValueError("Only approved gold labels can be used for comparison.")
            repository.save_gold_label_set(new_labels)
        except (ValueError, ValidationError, StorageError) as error:
            st.error(str(error))
        else:
            st.success("Gold labels saved.")
            st.rerun()

    with st.container(border=True):
        st.markdown("#### Locked comparison configuration")
        left, middle, right = st.columns(3)
        left.metric("Provider", _provider_label(settings.provider))
        middle.metric("Model", settings.model)
        right.metric("Per-condition ceiling", f"{settings.token_ceiling:,}")
        st.caption("Temperature 0.0 · same PDF bytes · sequential execution")
        st.button("Edit settings", on_click=_open_settings,
                  args=("create_comparison",))

    if not st.button("Run comparison", type="primary", width="stretch"):
        return
    errors = []
    if upload is None:
        errors.append("Upload one text-extractable PDF before running the comparison.")
    if research_enabled and not consent:
        errors.append("Confirm consent before enabling grounded internet research.")
    if selected_labels is None:
        errors.append("Upload and save an approved gold-label version before running.")
    if errors:
        for error in errors:
            st.error(error)
        return

    provider_settings = settings.provider_settings()
    comparison_settings = ComparisonSettings(
        provider=provider_settings,
        label_set_id=selected_labels.label_set_id,
        research_enabled=research_enabled,
        research_consent=consent,
    )
    try:
        with st.status("Running fair comparison", expanded=True) as status:
            runner = st.session_state.get("_comparison_runner", run_comparison)
            result = runner(
                upload.getvalue(),
                upload.name,
                comparison_settings,
                repository=repository,
                progress=status.write,
            )
            status.update(label="Comparison complete", state="complete")
    except Exception as error:
        st.error(f"Comparison failed: {_safe_error(error, provider_settings)}")
        return
    st.session_state["selected_comparison_id"] = result.manifest.comparison_id
    st.session_state["selected_comparison"] = result
    st.session_state["view"] = "comparison_detail"
    st.rerun()
```

- [ ] **Step 4: Run creation tests**

Run:

```bash
rtk .venv/bin/pytest tests/test_app.py -q
```

Expected: validation, local-provider, success, settings, and legacy creation tests pass.

- [ ] **Step 5: Commit the creation flow**

```bash
rtk git add app.py tests/test_app.py
rtk git commit -m "feat: add fair comparison setup"
```

## Task 4: Build the comparison shell, overview, activity, and configuration

**Files:**

- Modify: `app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Add failing detail-shell tests**

Add:

```python
def test_comparison_detail_exposes_seven_workbench_tabs() -> None:
    comparison = completed_comparison()
    at = _app_test(FakeRepository(
        comparison_results={"cmp-001": comparison}
    ))
    at.session_state["view"] = "comparison_detail"
    at.session_state["selected_comparison_id"] = "cmp-001"

    at.run()

    assert [tab.label for tab in at.tabs] == [
        "Overview",
        "Agent activity",
        "Requirements",
        "Scenarios & test cases",
        "Traceability",
        "Evaluation",
        "Configuration",
    ]


def test_overview_and_activity_show_both_conditions_and_role_states() -> None:
    comparison = reviewed_comparison()
    at = _app_test(FakeRepository(
        comparison_results={"cmp-001": comparison}
    ))
    at.session_state.update(
        view="comparison_detail", selected_comparison_id="cmp-001"
    )

    at.run()

    text = _rendered_text(at)
    assert "Single-prompt baseline" in text
    assert "Role-based agentic" in text
    assert "Requirement analyst" in text
    assert "No external clarification required" in text
    assert "Equal per-condition token ceiling" in text
```

Add `reviewed_comparison()` beside the existing app-test helpers. It calls the real `score_comparison` with `gold_labels()` and accepted `REQ-001/GREQ-001` plus `TC-001/GTI-001` decisions for both conditions, then returns `completed_comparison().model_copy(update={"latest_evaluation": revision})`.

- [ ] **Step 2: Confirm the detail shell is incomplete**

Run:

```bash
rtk .venv/bin/pytest tests/test_app.py -q
```

Expected: tab and content assertions fail.

- [ ] **Step 3: Load selected comparisons safely**

Implement `_render_comparison_detail` using the existing run-detail cache pattern:

```python
comparison_id = st.session_state.get("selected_comparison_id")
if not isinstance(comparison_id, str) or not comparison_id:
    _go_comparisons()
    st.session_state["flash_error"] = "Select a comparison to open its details."
    st.rerun()
comparison = st.session_state.get("selected_comparison")
if not isinstance(comparison, ComparisonResult) or (
    comparison.manifest.comparison_id != comparison_id
):
    try:
        comparison = repository.load_comparison(comparison_id)
    except StorageError:
        _go_comparisons()
        st.session_state["flash_error"] = "Saved comparison could not be opened."
        st.rerun()
    st.session_state["selected_comparison"] = comparison
```

Render a back button, source title, status pill text, then create the seven tabs in the exact order asserted above. Pass `review_locked = comparison.latest_evaluation is not None` to every tab renderer.

- [ ] **Step 4: Render overview condition cards**

Add `_render_comparison_overview(comparison, review_locked)`:

If `review_locked` is false, render only `Operational details are hidden until this review is locked.` and return. Otherwise render:

- Two bordered columns labeled `Single-prompt baseline` and `Role-based agentic`.
- Each shows terminal status, requirement/scenario/test counts, structural requirement-to-test coverage, charged tokens, and latency.
- If a condition failed, show the safe failure category/message instead of blank metrics.
- If `latest_evaluation` exists, show requirement F1 and test-coverage F1 first; otherwise show `Human F1 review pending` and clearly label structural coverage as not F1.
- A final callout states `Equal per-condition token ceiling: N` and `Research enabled/disabled`.

Reuse `_result_status`, `_safe_failure_message`, and existing metric formatting helpers rather than duplicating status logic.

- [ ] **Step 5: Render activity as a semantic timeline**

Add `_render_agent_activity(comparison, review_locked)`. Return the same lock message before iterating when `review_locked` is false:

```python
ROLE_LABELS = {
    AgentRole.ORCHESTRATOR: "Orchestrator",
    AgentRole.REQUIREMENT_ANALYST: "Requirement analyst",
    AgentRole.INTERNET_RESEARCHER: "Internet researcher",
    AgentRole.SCENARIO_GENERATOR: "Scenario generator",
    AgentRole.TEST_GENERATOR: "Test generator",
    AgentRole.VALIDATOR: "Validator",
}

for event in comparison.activity:
    with st.container(border=True):
        role, state, time_column = st.columns([2, 1, 1])
        role.markdown(f"**{ROLE_LABELS[event.role]}**")
        state.caption(event.state.value.title())
        time_column.caption(event.occurred_at.strftime("%H:%M:%S"))
        st.write(event.summary)
        st.caption(f"Routing: {event.routing_reason}")
        st.caption(
            f"Tokens {event.charged_tokens:,} · "
            f"Latency {event.latency_seconds:.2f}s · Retries {event.retries}"
        )
        findings = event.details.get("findings", [])
        for finding in findings if isinstance(findings, list) else []:
            st.markdown(f"**Query:** {finding['query']}")
            for citation in finding.get("citations", []):
                st.link_button(citation["title"], citation["url"])
```

When activity is empty, render `No agent activity was recorded.` Never render raw JSON or secret-bearing exception objects.

- [ ] **Step 6: Render immutable configuration**

When review is unlocked, render only the comparison ID, source filename, document hash prefix, frozen gold-label ID, and lock message. When locked, add a bordered key/value layout showing provider, model, temperature, per-condition ceiling, research state, condition run IDs, start/completion timestamps, and prompt/schema versions from each run. Include the exact fairness statement `Equal per-condition token ceiling`.

- [ ] **Step 7: Run detail-shell tests**

Run:

```bash
rtk .venv/bin/pytest tests/test_app.py -q
```

Expected: overview/activity/configuration tests and all existing tests pass.

- [ ] **Step 8: Commit the workspace shell**

```bash
rtk git add app.py tests/test_app.py
rtk git commit -m "feat: add comparison overview and agent activity"
```

## Task 5: Add artifact master-detail and traceability views

**Files:**

- Modify: `app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Add failing artifact navigation tests**

Add:

```python
def test_artifact_tabs_render_condition_selectors_and_traceability() -> None:
    comparison = completed_comparison()
    at = _app_test(FakeRepository(
        comparison_results={"cmp-001": comparison}
    ))
    at.session_state.update(
        view="comparison_detail", selected_comparison_id="cmp-001"
    )

    at.run()

    labels = [item.label for item in at.selectbox]
    assert "Requirement condition" in labels
    assert "Scenario/test condition" in labels
    assert "Traceability condition" in labels
    text = _rendered_text(at)
    assert "REQ-001" in text
    assert "TC-001" in text
    assert "Source evidence" in text
```

- [ ] **Step 2: Confirm artifact controls are absent**

Run:

```bash
rtk .venv/bin/pytest tests/test_app.py -q
```

Expected: selector and content assertions fail.

- [ ] **Step 3: Add one reusable condition selector**

Add:

```python
CONDITION_LABELS = {
    ComparisonCondition.BASELINE: "Single-prompt baseline",
    ComparisonCondition.AGENTIC: "Role-based agentic",
}


def _condition_result(
    comparison: ComparisonResult, condition: ComparisonCondition
) -> RunResult | None:
    return (
        comparison.baseline
        if condition is ComparisonCondition.BASELINE
        else comparison.agentic
    )


def _select_condition(
    comparison: ComparisonResult, label: str, key: str
) -> RunResult | None:
    aliases = (
        CONDITION_LABELS
        if comparison.latest_evaluation is not None
        else {
            condition: alias
            for alias, condition in _blinded_conditions(
                comparison.manifest.comparison_id
            )
        }
    )
    condition = st.selectbox(
        label,
        list(ComparisonCondition),
        format_func=aliases.__getitem__,
        key=key,
    )
    return _condition_result(comparison, condition)
```

- [ ] **Step 4: Render requirements with evidence**

Requirements tab layout:

- condition selector;
- search input filtering ID/title/description/module using `casefold()`;
- compact dataframe/list of matching requirements;
- selected requirement detail in a bordered panel with type, priority, module, ambiguities, dependencies;
- source evidence expanders using the existing `_render_sources`.

Reuse `_render_requirements` internals, but separate selection from rendering so the same detail panel works for both conditions.

- [ ] **Step 5: Render scenario/test master-detail**

Scenarios & test cases tab layout:

- condition selector;
- scenario selectbox as the master list;
- left column: scenario objective, type, preconditions, requirement links, citations;
- right column: test cases for the selected scenario, with priority, preconditions, data, numbered steps, expected results, and citations;
- explicit empty states for missing bundle, scenarios, and cases.

Reuse the existing `_render_test_cases` step/table formatting; do not add a client-side grid library.

- [ ] **Step 6: Render RTM and structural gaps**

Traceability tab layout:

- condition selector;
- metrics for total requirements, covered requirements, and structural coverage;
- dataframe columns `Requirement`, `Scenarios`, `Test cases`, `Source chunks`, `Covered` using `RunResult.rtm`;
- a warning listing `validation.uncovered_requirement_ids` when present;
- caption: `Structural traceability is not the human-reviewed F1 score.`

- [ ] **Step 7: Run artifact tests**

Run:

```bash
rtk .venv/bin/pytest tests/test_app.py -q
```

Expected: artifact selectors, IDs, evidence, traceability, and existing run detail tests pass.

- [ ] **Step 8: Commit artifact exploration**

```bash
rtk git add app.py tests/test_app.py
rtk git commit -m "feat: add comparison artifact exploration"
```

## Task 6: Add blinded F1 review and identity reveal

**Files:**

- Modify: `app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Add failing blinded-review and lock tests**

Add:

```python
def test_pending_review_uses_blinded_condition_labels() -> None:
    comparison = completed_comparison()
    repository = FakeRepository(
        comparison_results={"cmp-001": comparison},
    )
    repository.saved_gold_labels = [gold_labels()]
    at = _app_test(repository)
    at.session_state.update(
        view="comparison_detail", selected_comparison_id="cmp-001"
    )

    at.run()

    text = _rendered_text(at)
    assert "Condition A" in text and "Condition B" in text
    assert "Review candidates without knowing which workflow produced them" in text
    assert "Operational details are hidden until this review is locked" in text
    assert "Requirement analyst" not in text
    assert "Single-prompt baseline" not in text


def test_saving_review_persists_human_decisions_and_shows_f1() -> None:
    comparison = completed_comparison()
    repository = FakeRepository(
        comparison_results={"cmp-001": comparison},
    )
    repository.saved_gold_labels = [gold_labels()]

    def fake_review(repository, comparison_id, label_set_id, reviewer, decisions):
        revision = scored_revision(comparison, gold_labels(), decisions)
        repository.save_evaluation_revision(revision)
        return revision

    at = _app_test(repository)
    at.session_state.update(
        view="comparison_detail",
        selected_comparison_id="cmp-001",
        _comparison_reviewer=fake_review,
    )
    at.run()
    _element(at.text_input, "Reviewer").set_value("qa@example.com")
    for selector in at.selectbox:
        if selector.label.startswith("Match REQ-"):
            selector.set_value("GREQ-001")
        elif selector.label.startswith("Match TC-"):
            selector.set_value("GTI-001")
    _element(at.button, "Save human review").click().run()

    assert repository.saved_evaluations
    assert "Requirement F1" in _rendered_text(at)
    assert "Test-coverage F1" in _rendered_text(at)
    assert "Single-prompt baseline" in _rendered_text(at)
    assert "Role-based agentic" in _rendered_text(at)
```

Add `scored_revision` as a concrete test factory using `score_comparison`; do not fabricate metric dictionaries.

- [ ] **Step 2: Load the frozen approved label version**

At the start of `_render_evaluation`, load exactly `comparison.manifest.label_set_id`:

```python
try:
    labels = repository.load_gold_label_set(comparison.manifest.label_set_id)
except StorageError:
    st.error("The frozen gold-label version is unavailable.")
    return
if labels.document_hash != comparison.manifest.document_hash:
    st.error("The frozen gold-label version no longer matches this document.")
    return
```

Display its version, approved status, author, and created time. Do not offer a different label version inside an existing immutable comparison.

- [ ] **Step 3: Derive a stable blinded condition order**

Add:

```python
def _blinded_conditions(comparison_id: str) -> list[tuple[str, ComparisonCondition]]:
    conditions = [ComparisonCondition.BASELINE, ComparisonCondition.AGENTIC]
    conditions.sort(
        key=lambda condition: hashlib.sha256(
            f"{comparison_id}:{condition.value}".encode()
        ).hexdigest()
    )
    return [(f"Condition {chr(65 + index)}", condition)
            for index, condition in enumerate(conditions)]
```

Before a saved review, do not render `baseline` or `agentic` labels anywhere in comparison detail. Overview, Agent activity, Configuration, token usage, latency, and architecture copy render only `Operational details are hidden until this review is locked.` Requirements, scenarios/tests, and traceability use `Condition A`/`Condition B` selectors. After a revision is saved, rerender with real condition identities and all operational details.

- [ ] **Step 4: Render candidates and collect one explicit decision per prediction**

For each blinded condition:

1. get its non-failed bundle or an empty bundle;
2. call `propose_requirement_matches(bundle, labels, limit=5)` and `propose_test_intent_matches(bundle, labels, limit=5)`;
3. group candidates by prediction ID;
4. render prediction summary and the top-five candidate scores labeled `review aid`;
5. render one selectbox labeled `Match <prediction ID>` whose choices are `Unmatched`, the ranked five candidates, then every remaining gold ID; this lets the reviewer select a lower-ranked gold item without allowing several accepted edges for one prediction;
6. build one accepted `MatchDecision` when a gold ID is selected, or no accepted edge for `Unmatched`;
7. list missed gold items live from gold IDs not currently selected;
8. require a nonblank reviewer before `Save human review`;
9. call `st.session_state.get("_comparison_reviewer", review_comparison)` and cache the returned revision in `selected_comparison.latest_evaluation` via `model_copy`.

Do not auto-select the top candidate and do not hide unmatched predictions or missed gold items. The deterministic maximum-cardinality matcher remains the final duplicate-gold guard.

- [ ] **Step 5: Render reviewed scores, missed items, and identity reveal**

After a revision exists, render baseline and agentic cards with:

- requirement precision, recall, F1, TP/predicted/gold;
- test-coverage precision, recall, F1, TP/predicted/gold;
- signed F1 delta (`agentic - baseline`);
- reviewed-by and revision timestamp;
- a note that failed conditions score zero.
- unmatched prediction IDs and missed gold IDs for each metric;
- evaluation revision history ordered newest first.

Use `st.metric`; do not add a chart library. After lock, reveal the real condition identities and operational tabs. Continue to offer `Create another revision` so corrections remain append-only while keeping the same frozen gold version.

- [ ] **Step 6: Show macro-primary and micro-secondary cohort results on home**

Add `test_comparison_home_shows_macro_primary_and_micro_secondary`: seed the fake repository with one latest evaluation revision, open the default Comparisons home, and assert `Reviewed benchmark`, `Macro requirement F1`, `Macro test-coverage F1`, `Micro precision`, and `Agentic lift` are rendered.

At the top of `_render_comparisons`, add a cohort selectbox with `Research enabled`, `Research disabled`, and `All reviewed`. Map the first two to `True`/`False`, load latest revisions through `list_latest_evaluation_revisions`, and call `aggregate_evaluations`. When revisions exist, render:

- reviewed document count;
- baseline and agentic macro requirement F1 plus signed lift;
- baseline and agentic macro test-coverage F1 plus signed lift;
- secondary micro precision/recall/F1 for each metric in captions or an expander.

Default to `Research enabled`; this prevents research-disabled comparisons from entering the research-enabled benchmark. When there are no reviewed documents in the cohort, show `No locked human reviews in this cohort.`

- [ ] **Step 7: Run evaluation UI tests**

Run:

```bash
rtk .venv/bin/pytest tests/test_app.py tests/test_evaluation.py tests/test_comparison.py -q
```

Expected: frozen-label loading, blinded labels, hidden operational data, unmatched handling, saved decisions, identity reveal, F1 display, and core evaluation tests pass.

- [ ] **Step 8: Commit evaluation workflow**

```bash
rtk git add app.py tests/test_app.py
rtk git commit -m "feat: add blinded human f1 review"
```

## Task 7: Apply the QA workbench theme and responsive behavior

**Files:**

- Modify: `app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Replace the current decorative theme tokens**

Keep `_apply_theme`, but replace the existing radial gradient and large floating-card treatment with:

```css
:root {
    --qa-primary: #3157d5;
    --qa-agentic: #6d3bd1;
    --qa-bg: #f6f7fb;
    --qa-surface: #ffffff;
    --qa-text: #101828;
    --qa-muted: #475467;
    --qa-border: #d9dfea;
    --qa-success: #087a55;
    --qa-warning: #9a5b00;
    --qa-danger: #b42318;
    --qa-focus: #2e5bff;
}
.stApp {
    color: var(--qa-text);
    background: var(--qa-bg);
    font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
[data-testid="stMainBlockContainer"] {
    max-width: 1440px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}
.brand {
    min-height: 44px;
    display: flex;
    align-items: center;
    color: var(--qa-text);
    font-size: 1.05rem;
    font-weight: 750;
    letter-spacing: -0.01em;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--qa-border);
    border-radius: 12px;
    background: var(--qa-surface);
    box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04);
}
[data-testid="stMetric"] {
    min-height: 6.5rem;
    padding: 0.9rem 1rem;
    border: 1px solid var(--qa-border);
    border-radius: 10px;
    background: var(--qa-surface);
}
.stButton > button,
.stDownloadButton > button,
[data-testid="stSelectbox"] button,
[data-testid="stFileUploader"] button {
    min-height: 44px;
}
button:focus-visible,
input:focus-visible,
textarea:focus-visible,
[role="tab"]:focus-visible {
    outline: 3px solid var(--qa-focus) !important;
    outline-offset: 2px;
}
@media (max-width: 768px) {
    [data-testid="stMainBlockContainer"] { padding: 0.75rem 0.75rem 3rem; }
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
    [data-testid="column"] { min-width: 100% !important; }
    [data-testid="stDataFrame"] { overflow-x: auto; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }
}
```

- [ ] **Step 2: Use semantic text instead of color alone**

Every status, gap, F1 delta, and agent state must include a text label. Use color only as reinforcement. Keep every source/citation link descriptive; never use `click here`. Do not add emoji as structural icons.

- [ ] **Step 3: Add a theme regression assertion**

Add an AppTest assertion that emitted markdown includes `--qa-primary`, the 44px target, the 768px responsive breakpoint, and `prefers-reduced-motion`. This catches accidental removal of the accessibility floor without snapshotting every CSS rule.

- [ ] **Step 4: Run app tests**

Run:

```bash
rtk .venv/bin/pytest tests/test_app.py -q
```

Expected: all app tests pass.

- [ ] **Step 5: Commit the visual system**

```bash
rtk git add app.py tests/test_app.py
rtk git commit -m "style: apply responsive qa workbench system"
```

## Task 8: Verify behavior and visual quality

**Files:**

- No production changes expected unless verification finds a defect.

- [ ] **Step 1: Compile the application**

Run:

```bash
rtk .venv/bin/python -m compileall -q app.py src tests
```

Expected: exit code `0` with no output.

- [ ] **Step 2: Run focused app and comparison suites**

Run:

```bash
rtk .venv/bin/pytest tests/test_app.py tests/test_comparison.py tests/test_evaluation.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run the full regression suite**

Run:

```bash
rtk .venv/bin/pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Start the app for visual QA**

Run:

```bash
rtk .venv/bin/streamlit run app.py --server.headless true --server.port 8501
```

Expected: Streamlit reports `Local URL: http://localhost:8501` and keeps running. Use the browser-control skill to inspect the live app.

- [ ] **Step 5: Verify desktop states at 1440px**

Inspect and capture screenshots of:

- comparison empty/history screen;
- create comparison with Gemini research off and on;
- completed comparison Overview;
- activity with skipped and completed researcher states;
- scenario/test master-detail;
- RTM with an uncovered requirement;
- pending blinded evaluation;
- reviewed F1 results.

Confirm no clipped text, overlapping controls, unexpected horizontal page scroll, unlabeled metric, or color-only state.

- [ ] **Step 6: Verify mobile states at 375px**

Inspect Overview, Scenarios & test cases, Traceability, and Evaluation. Confirm columns stack, tables scroll only inside their container, buttons remain at least 44px high, tabs remain usable, and the top navigation wraps without covering content.

- [ ] **Step 7: Verify keyboard and reduced-motion basics**

Tab through top navigation, upload, research toggle, consent, condition selectors, candidate checkboxes, and submit buttons. Confirm the focus ring is visible. Enable reduced motion in browser emulation and confirm no essential state depends on animation.

- [ ] **Step 8: Stop the dev server and inspect the diff**

Stop the Streamlit process with `Ctrl-C`, then run:

```bash
rtk git diff --check
rtk git status --short
```

Expected: no whitespace errors; only intended UI files remain if commits were skipped, or the worktree is clean after planned commits.

## Acceptance checklist

- [ ] Comparisons are the default experience; saved standalone runs remain reachable.
- [ ] Users see baseline and agentic results side by side without confusing structural coverage for F1.
- [ ] Agent activity exposes role, state, summary, and research citations without raw internal payloads.
- [ ] Requirements, scenarios/test cases, and RTM are inspectable without navigating away.
- [ ] Gold labels are strict, document-bound, and append-only.
- [ ] Candidate review is blinded, explicit, and never auto-accepted.
- [ ] Requirement F1 and test-coverage F1 show precision, recall, counts, and agentic lift.
- [ ] Configuration makes fairness and research state auditable.
- [ ] Desktop and mobile layouts pass visual, keyboard, focus, target-size, contrast, and reduced-motion checks.
- [ ] No unrelated dashboard modules or new UI dependency was added.
