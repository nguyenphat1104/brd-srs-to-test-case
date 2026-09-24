from datetime import UTC, datetime
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from brd_srs_testgen.evaluation import (
    COVERAGE_RATING_RUBRIC_VERSION,
    coverage_rating,
)
from brd_srs_testgen.models import (
    AgentSetup,
    ActivityEvent,
    CoverageCatalog,
    CoverageCatalogStatus,
    CoverageEvaluation,
    CoverageEvaluationStatus,
    CoverageMappingBatch,
    CoverageMappingEntry,
    CoverageScore,
    CoverageUnit,
    FailureCategory,
    HumanAdjudication,
    HumanCoverageRating,
    HumanRating,
    HumanRatingDimension,
    RequirementBatch,
    RunHistoryItem,
    RunManifest,
    RunResult,
    RunStatus,
    RunType,
    TestStep as ModelTestStep,
    default_agent_setups,
)
from brd_srs_testgen.runner import EVALUATOR_VERSION
from brd_srs_testgen.storage import StorageError
from tests.factories import completed_run


APP = Path(__file__).parents[1] / "app.py"
LOCAL_TEST_MODELS = (
    ("gemma", "Gemma 4 26B"),
    ("phi", "Phi 4 Mini Instruct"),
    ("qwen", "Qwen 3 4B"),
)


@pytest.fixture(autouse=True)
def _provider_connections(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("LLAMA_CPP_BASE_URL", "http://llama.cpp:8080/v1")


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
        ratings: dict[str, HumanCoverageRating] | None = None,
        catalogs: dict[tuple[str, str], CoverageCatalog] | None = None,
        human_ratings: list[HumanRating] | None = None,
        adjudications: list[HumanAdjudication] | None = None,
    ) -> None:
        self.runs = runs or []
        self.run_batches = run_batches
        self.results = results or {}
        self.initialize_error = initialize_error
        self.list_error = list_error
        self.load_error = load_error
        self.ratings = ratings or {}
        self.catalogs = catalogs or {}
        self.human_ratings = human_ratings or []
        self.adjudications = adjudications or []
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

    def load_human_coverage_rating(self, run_id: str) -> HumanCoverageRating | None:
        return self.ratings.get(run_id)

    def save_human_coverage_rating(
        self, run_id: str, human_score: int, reason: str = ""
    ) -> HumanCoverageRating:
        result = self.results[run_id]
        assert result.coverage is not None
        rating = HumanCoverageRating(
            run_id=run_id,
            human_score=human_score,
            judge_score=coverage_rating(result.coverage.f1),
            judge_f1=result.coverage.f1,
            reason=reason,
            rubric_version=COVERAGE_RATING_RUBRIC_VERSION,
            created_at=datetime.now(UTC),
        )
        self.ratings[run_id] = rating
        return rating

    def list_human_coverage_ratings(self) -> list[HumanCoverageRating]:
        return list(self.ratings.values())

    def load_coverage_catalog(
        self, document_hash: str, evaluator_version: str
    ) -> CoverageCatalog | None:
        return self.catalogs.get((document_hash, evaluator_version))

    def load_coverage_catalog_by_id(
        self, catalog_id: str
    ) -> CoverageCatalog | None:
        return next(
            (
                catalog
                for catalog in self.catalogs.values()
                if catalog.catalog_id == catalog_id
            ),
            None,
        )

    def approve_coverage_catalog(self, catalog_id: str, *, approved_at=None):
        for key, catalog in self.catalogs.items():
            if catalog.catalog_id == catalog_id:
                approved = catalog.model_copy(
                    update={
                        "status": CoverageCatalogStatus.APPROVED,
                        "approved_at": approved_at or datetime.now(UTC),
                    }
                )
                self.catalogs[key] = approved
                return approved
        raise StorageError("Coverage catalog does not exist.")

    def save_human_rating(self, rating: HumanRating) -> HumanRating:
        if any(
            (
                item.run_id,
                item.rater_id,
                item.dimension,
                item.round,
            )
            == (
                rating.run_id,
                rating.rater_id,
                rating.dimension,
                rating.round,
            )
            for item in self.human_ratings
        ):
            raise StorageError("Human rating already exists.")
        self.human_ratings.append(rating)
        return rating

    def list_human_ratings(self, run_id: str | None = None) -> list[HumanRating]:
        return [
            rating
            for rating in self.human_ratings
            if run_id is None or rating.run_id == run_id
        ]

    def save_human_adjudication(
        self, adjudication: HumanAdjudication
    ) -> HumanAdjudication:
        if any(
            item.run_id == adjudication.run_id
            and item.dimension is adjudication.dimension
            for item in self.adjudications
        ):
            raise StorageError("Human adjudication already exists.")
        self.adjudications.append(adjudication)
        return adjudication

    def list_human_adjudications(
        self, run_id: str | None = None
    ) -> list[HumanAdjudication]:
        return [
            item
            for item in self.adjudications
            if run_id is None or item.run_id == run_id
        ]


def _app_test(repository: FakeRepository | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=10)
    at.session_state["_repository"] = repository or FakeRepository()
    at.session_state["_model_loader"] = lambda _url: LOCAL_TEST_MODELS
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


def _open_settings_step(at: AppTest, run_type: RunType) -> None:
    _element(at.button, "Create new run").click()
    at.run()
    _element(at.radio, "Run type").set_value(run_type)
    _element(at.button, "Continue to settings").click()
    at.run()


def _open_document_step(at: AppTest, run_type: RunType) -> None:
    _open_settings_step(at, run_type)
    _element(at.button, "Continue to document").click()
    at.run()


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


def _versioned_coverage_result(
    *, failed: bool = False
) -> tuple[RunResult, CoverageCatalog]:
    result = _detailed_run()
    source = result.bundle.requirements[0].source_references[0]
    catalog = CoverageCatalog(
        catalog_id="catalog-audit-1",
        document_hash=result.manifest.document_hash,
        evaluator_version=EVALUATOR_VERSION,
        status=CoverageCatalogStatus.MACHINE_FROZEN,
        units=[
            CoverageUnit(
                unit_id="CU-001",
                title="Authenticate registered users",
                description="Registered users can authenticate.",
                unit_type="functional",
                source_references=[source],
            )
        ],
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
    )
    if failed:
        evaluation = CoverageEvaluation(
            run_id=result.manifest.run_id,
            catalog_id=catalog.catalog_id,
            status=CoverageEvaluationStatus.FAILED,
            error="Judge mapping timed out.",
            evaluated_at=datetime(2026, 9, 24, 1, tzinfo=UTC),
        )
        return result.model_copy(
            update={"coverage": None, "coverage_evaluation": evaluation}
        ), catalog
    score = CoverageScore(
        catalog_id=catalog.catalog_id,
        precision=1,
        recall=1,
        f1=1,
        true_positive_count=1,
        false_positive_count=0,
        false_negative_count=0,
        total_coverage_units=1,
        total_test_cases=1,
    )
    evaluation = CoverageEvaluation(
        run_id=result.manifest.run_id,
        catalog_id=catalog.catalog_id,
        status=CoverageEvaluationStatus.COMPLETED,
        mappings=CoverageMappingBatch(
            mappings=[
                CoverageMappingEntry(
                    test_case_id="TC-001", covered_unit_ids=["CU-001"]
                )
            ]
        ),
        score=score,
        evaluated_at=datetime(2026, 9, 24, 1, tzinfo=UTC),
    )
    return result.model_copy(
        update={"coverage": score, "coverage_evaluation": evaluation}
    ), catalog


def _human_rating(
    result: RunResult,
    rater_id: str,
    dimension: HumanRatingDimension,
    score: int,
) -> HumanRating:
    return HumanRating(
        rating_id=f"{result.manifest.run_id}-{rater_id}-{dimension.value}",
        run_id=result.manifest.run_id,
        rater_id=rater_id,
        dimension=dimension,
        score=score,
        reason="Blind review.",
        rubric_version="quality-rubric-v1",
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
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


def test_runs_home_shows_run_item_list_without_global_settings() -> None:
    completed = _detailed_run()
    interrupted = _interrupted_run()
    repository = FakeRepository(
        runs=[_history_item(interrupted), _history_item(completed)]
    )
    at = _app_test(repository)

    at.run()

    assert not at.exception
    assert at.session_state["view"] == "runs"
    assert not at.tabs
    assert "Settings" not in {button.label for button in at.button}
    assert _element(at.button, "Create new run")
    docs_link = at.get("link_button")[0].proto
    assert docs_link.label == "How the system and coverage work"
    assert docs_link.url == "/app/static/system-and-coverage.html"
    text = _rendered_text(at)
    assert not at.dataframe
    assert {button.label for button in at.button} >= {
        "Open interrupted.pdf",
        "Open sample.pdf",
    }
    for expected in (
        "interrupted.pdf",
        "sample.pdf",
        "Multi agents",
        "Interrupted",
        "Completed",
        "Not recorded",
        "1 generated",
    ):
        assert expected in text


def test_create_starts_with_run_type_before_settings_or_upload() -> None:
    at = _app_test()
    at.run()

    _element(at.button, "Create new run").click()
    at.run()

    assert _element(at.radio, "Run type").options == [
        "Single prompt",
        "Staged prompt",
        "Multi agents",
    ]
    assert not at.file_uploader
    assert not at.text_area


def test_single_settings_default_to_one_gemini_35_agent() -> None:
    at = _app_test()
    at.run()
    _open_settings_step(at, RunType.SINGLE_PROMPT)

    assert len(at.selectbox) == 3
    assert _element(at.selectbox, "Agent provider").value == "gemini"
    assert _element(at.selectbox, "Agent provider").options == [
        "Gemini",
        "llama.cpp",
    ]
    assert _element(at.selectbox, "Agent model").value == "gemini-3.5-flash"
    assert _element(at.selectbox, "Agent model").options == [
        "Gemini 3.7 Flash",
        "Gemini 3.6 Flash",
        "Gemini 3.5 Flash",
        "Gemini 2.5 Flash",
        "Gemini 2.5 Pro",
    ]
    assert _element(at.selectbox, "Agent thinking level").value == "minimal"
    assert len(at.text_area) == 1
    assert _element(at.text_area, "Agent prompt").value
    assert "Gemini API key" not in {item.label for item in at.text_input}
    assert "Provider access is managed by the deployment environment" in _rendered_text(at)

    _element(at.selectbox, "Agent thinking level").set_value("low")
    _element(at.button, "Continue to document").click()
    at.run()

    assert at.session_state["run_provider_settings"].thinking_level_for("single") == "low"


def test_staged_settings_share_gemini_36_model_and_have_three_prompts() -> None:
    at = _app_test()
    at.run()
    _open_settings_step(at, RunType.STAGED_SINGLE_AGENT)

    assert len(at.selectbox) == 5
    assert _element(at.selectbox, "Agent provider").value == "gemini"
    assert _element(at.selectbox, "Agent model").value == "gemini-3.6-flash"
    assert {
        item.label: item.value
        for item in at.selectbox
        if item.label.endswith("thinking level")
    } == {
        "Requirements step thinking level": "minimal",
        "Scenarios step thinking level": "minimal",
        "Test cases step thinking level": "minimal",
    }
    assert {
        item.label: item.value
        for item in at.number_input
        if item.label.endswith("output tokens")
    } == {
        "Requirements step output tokens": 16_000,
        "Scenarios step output tokens": 24_000,
        "Test cases step output tokens": 48_000,
    }
    assert {
        item.min for item in at.number_input if item.label.endswith("output tokens")
    } == {1_000}
    assert {area.label for area in at.text_area} == {
        "Requirements step prompt",
        "Scenarios step prompt",
        "Test cases step prompt",
    }
    assert "Gemini API key" not in {item.label for item in at.text_input}


def test_multi_agent_settings_keep_local_defaults_for_every_agent() -> None:
    at = _app_test()
    at.run()
    _open_settings_step(at, RunType.CENTRALIZED_MULTI_AGENT)

    assert {
        item.value for item in at.selectbox if item.label.endswith(" provider")
    } == {"llama_cpp"}
    assert {
        item.label: item.value
        for item in at.selectbox
        if item.label.endswith(" model")
    } == {
        "Scout model": "qwen",
        "Curator model": "phi",
        "Scenario architect model": "gemma",
        "Test writer model": "gemma",
        "Critic model": "phi",
    }
    assert {
        tuple(item.options)
        for item in at.selectbox
        if item.label.endswith(" model")
    } == {tuple(label for _model, label in LOCAL_TEST_MODELS)}
    assert len(at.text_area) == 5
    assert len([item for item in at.number_input if item.label.endswith("output tokens")]) == 5
    assert not at.checkbox
    assert not at.toggle
    text = _rendered_text(at)
    assert "Judge" in text
    assert "Gemini 3.6 Flash · Medium thinking" in text
    assert "separate evaluation tokens · Fixed" in text
    assert "llama.cpp base URL" not in {item.label for item in at.text_input}


@pytest.mark.parametrize("requested_cap", [1_000, 1_024])
def test_hierarchical_output_tokens_cannot_submit_below_stage_minimum(requested_cap):
    at = _app_test()
    at.run()
    _open_settings_step(at, RunType.CENTRALIZED_MULTI_AGENT)
    controls = [
        item for item in at.number_input if item.label.endswith("output tokens")
    ]
    assert len(controls) == 5
    assert {item.min for item in controls} == {1_024}
    for control in controls:
        control.set_value(requested_cap)
    _element(at.button, "Continue to document").click()
    at.run()

    assert not at.exception
    assert at.session_state["create_step"] == 3
    assert at.session_state["run_provider_settings"].agent_max_output_tokens == {
        agent: 1_024
        for agent in ("scout", "curator", "scenario_architect", "test_writer", "critic")
    }


def test_multi_agent_gemini_settings_include_thinking_levels() -> None:
    at = _app_test()
    at.run()
    _open_settings_step(at, RunType.CENTRALIZED_MULTI_AGENT)

    for selector in at.selectbox:
        if selector.label.endswith(" provider"):
            selector.set_value("gemini")
    at.run()

    assert {
        item.label: item.value
        for item in at.selectbox
        if item.label.endswith("thinking level")
    } == {
        "Scout thinking level": "minimal",
        "Curator thinking level": "minimal",
        "Scenario architect thinking level": "minimal",
        "Test writer thinking level": "minimal",
        "Critic thinking level": "minimal",
    }
    _element(at.selectbox, "Critic thinking level").set_value("high")
    _element(at.selectbox, "Test writer model").set_value("gemini-2.5-pro")
    _element(at.text_area, "Scout prompt").set_value("Prioritize exceptions.")
    _element(at.number_input, "Curator output tokens").set_value(12_000)
    _element(at.button, "Continue to document").click()
    at.run()

    settings = at.session_state["run_provider_settings"]
    assert {
        agent: settings.thinking_level_for(agent)
        for agent in ("scout", "curator", "scenario_architect", "test_writer", "critic")
    } == {
        "scout": "minimal",
        "curator": "minimal",
        "scenario_architect": "minimal",
        "test_writer": None,
        "critic": "high",
    }
    assert settings.critic_enabled is True
    assert settings.model_for("test_writer") == "gemini-2.5-pro"
    assert settings.prompt_for("scout") == "Prioritize exceptions."
    assert settings.agent_max_output_tokens["curator"] == 12_000
    snapshot = settings.snapshot(RunType.CENTRALIZED_MULTI_AGENT)
    _element(at.button, "Edit run settings").click()
    at.run()
    assert not at.exception
    assert "Test writer thinking level" not in {item.label for item in at.selectbox}
    assert _element(at.number_input, "Curator output tokens").value == 12_000
    _element(at.button, "Continue to document").click()
    at.run()
    assert at.session_state["run_provider_settings"].snapshot(RunType.CENTRALIZED_MULTI_AGENT) == snapshot


def test_llama_cpp_model_api_error_disables_model_selection() -> None:
    def unavailable(_url):
        raise ConnectionError("private backend detail")

    at = _app_test()
    at.session_state["_model_loader"] = unavailable
    at.run()
    _open_settings_step(at, RunType.CENTRALIZED_MULTI_AGENT)

    model_selects = [
        item for item in at.selectbox if item.label.endswith(" model")
    ]
    assert model_selects
    assert all(item.disabled for item in model_selects)
    assert "llama.cpp models are unavailable" in _rendered_text(at)
    assert "private backend detail" not in _rendered_text(at)


def test_empty_runs_still_offers_create() -> None:
    at = _app_test()

    at.run()

    assert "No test suites yet. Add your first PDF to get started." in _rendered_text(at)
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
    at = _app_test()
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
        "Run configuration",
    ]
    assert headings.index("### Quality and traceability") < headings.index(
        "### Generated artifacts"
    )
    assert not any(toggle.label == "Show quality details" for toggle in at.toggle)
    assert not any(item.label == "Run configuration" for item in at.expander)
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
        {"Setting": "Run type", "Value": "Staged prompt"},
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
    assert "test-gemini-key" not in text
    assert "test-gemini-key" not in tables
    assert {button.key for button in at.download_button} == {
        f"detail-{result.manifest.run_id}-rtm",
        f"detail-{result.manifest.run_id}-bundle",
    }
    assert _element(at.metric, "Charged tokens").value == "30"
    assert "Latency 0.10 s · 0 retries" in text
    assert "Citation coverage" in text
    assert "Positive scenario coverage" in text
    assert "Non-positive scenario coverage" in text


def test_completed_detail_renders_coverage_charts() -> None:
    result = _detailed_run().model_copy(
        update={
            "coverage": CoverageScore(
                catalog_id="legacy-unversioned",
                precision=0,
                recall=0,
                f1=0,
                true_positive_count=0,
                false_positive_count=1,
                false_negative_count=1,
                total_coverage_units=1,
                total_test_cases=1,
                uncovered_unit_ids=["CU-001"],
                unmapped_test_case_ids=["TC-001"],
            )
        }
    )
    repository = FakeRepository(results={result.manifest.run_id: result})
    at = _app_test(repository)
    _show_detail(at, result)

    at.run()

    assert not at.exception
    assert "### Independent human evaluation" in [item.value for item in at.markdown]
    assert "### Coverage analysis (F1)" in [item.value for item in at.markdown]
    assert _element(at.metric, "F1 Score").value == "0.00"
    assert "Citation coverage" in _rendered_text(at)
    assert _element(at.text_input, "Rater ID")
    assert "### Inter-rater agreement" in [item.value for item in at.markdown]


def test_versioned_coverage_renders_catalog_evidence_and_mappings() -> None:
    result, catalog = _versioned_coverage_result()
    repository = FakeRepository(
        results={result.manifest.run_id: result},
        catalogs={(catalog.document_hash, catalog.evaluator_version): catalog},
    )
    at = _app_test(repository)
    _show_detail(at, result)

    at.run()

    text = _rendered_text(at)
    assert "### Coverage catalog and mapping evidence" in [
        item.value for item in at.markdown
    ]
    assert catalog.catalog_id in text
    assert "Machine Frozen" in text
    assert "The system shall authenticate registered users." in text
    tables = "\n".join(str(frame.value) for frame in at.dataframe)
    assert "CU-001" in tables
    assert "TC-001" in tables
    _element(at.button, "Approve coverage catalog").click()
    at.run()

    assert repository.catalogs[
        (catalog.document_hash, catalog.evaluator_version)
    ].status is CoverageCatalogStatus.APPROVED
    assert not any(
        button.label == "Approve coverage catalog" for button in at.button
    )


def test_evaluator_failure_is_shown_without_a_zero_f1() -> None:
    result, catalog = _versioned_coverage_result(failed=True)
    repository = FakeRepository(
        results={result.manifest.run_id: result},
        catalogs={(catalog.document_hash, catalog.evaluator_version): catalog},
    )
    at = _app_test(repository)
    _show_detail(at, result)

    at.run()

    text = _rendered_text(at)
    assert _element(at.metric, "F1 Score").value == "—"
    assert "Judge mapping timed out." in text
    assert "No F1 score was assigned" in text
    assert "F1 Score</div><div>0.00" not in text


def test_blinded_rating_form_requires_identity_and_all_dimensions() -> None:
    result, catalog = _versioned_coverage_result()
    repository = FakeRepository(
        results={result.manifest.run_id: result},
        catalogs={(catalog.document_hash, catalog.evaluator_version): catalog},
    )
    at = _app_test(repository)
    _show_detail(at, result)

    at.run()
    _element(at.button, "Save blinded rating").click()
    at.run()

    assert "Enter a rater ID and score all four dimensions." in _rendered_text(at)
    _element(at.text_input, "Rater ID").set_value("rater-a")
    for label in (
        "Coverage score",
        "Groundedness score",
        "Executability score",
        "Redundancy control score",
    ):
        _element(at.selectbox, label).set_value(3)
    _element(at.button, "Save blinded rating").click()
    at.run()

    assert len(repository.human_ratings) == 4
    assert {item.dimension for item in repository.human_ratings} == set(
        HumanRatingDimension
    )
    assert "Saved ratings are append-only" in _rendered_text(at)


def test_agreement_and_adjudication_are_separate_per_dimension() -> None:
    result, catalog = _versioned_coverage_result()
    ratings = [
        _human_rating(
            result,
            rater_id,
            dimension,
            3 if rater_id == "rater-a" else (4 if dimension is HumanRatingDimension.COVERAGE else 3),
        )
        for rater_id in ("rater-a", "rater-b")
        for dimension in HumanRatingDimension
    ]
    repository = FakeRepository(
        results={result.manifest.run_id: result},
        catalogs={(catalog.document_hash, catalog.evaluator_version): catalog},
        human_ratings=ratings,
    )
    at = _app_test(repository)
    _show_detail(at, result)

    at.run()

    text = _rendered_text(at)
    assert "### Inter-rater agreement" in [item.value for item in at.markdown]
    rendered = text + "\n" + "\n".join(
        str(frame.value) for frame in at.dataframe
    )
    labels = (
        "Coverage",
        "Groundedness",
        "Executability",
        "Redundancy control",
    )
    assert not [label for label in labels if label not in rendered], rendered
    assert any(
        {"Exact", "Adjacent", "Quadratic weighted κ"}.issubset(frame.value.columns)
        for frame in at.dataframe
    )
    assert "Adjudication" in [item.label for item in at.expander]
    assert not any(input.value == "rater-a" for input in at.text_input)


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
    assert "### Coverage analysis (F1)" in [item.value for item in at.markdown]
    assert _element(at.metric, "F1 Score").value == "—"
    assert "did not finish judge analysis" in text
    assert [tab.label for tab in at.tabs] == ["Run configuration"]
    snapshot = next(
        table.value
        for table in at.table
        if list(table.value.columns) == ["Setting", "Value"]
    )
    assert dict(zip(snapshot["Setting"], snapshot["Value"], strict=True))[
        "Status"
    ] == "failed"
    assert _element(at.download_button, "Download diagnostics")


def test_run_list_paginates_ten_items_at_a_time() -> None:
    base = _history_item(completed_run())
    runs = [
        base.model_copy(
            update={
                "run_id": f"run-{index:02d}",
                "source_filename": f"suite-{index:02d}.pdf",
            }
        )
        for index in range(12)
    ]
    at = _app_test(FakeRepository(runs=runs))

    at.run()

    assert "Page 1 of 2 · 12 runs" in _rendered_text(at)
    assert {button.label for button in at.button if button.label.startswith("Open ")} == {
        f"Open suite-{index:02d}.pdf" for index in range(10)
    }
    assert _element(at.button, "Previous").disabled
    _element(at.button, "Next").click()
    at.run()

    assert "Page 2 of 2 · 12 runs" in _rendered_text(at)
    assert {button.label for button in at.button if button.label.startswith("Open ")} == {
        "Open suite-10.pdf",
        "Open suite-11.pdf",
    }
    assert _element(at.button, "Next").disabled


def test_incomplete_schema_failure_recommends_more_output_tokens() -> None:
    failed = _failed_run()
    result = failed.model_copy(
        update={
            "manifest": failed.manifest.model_copy(
                update={
                    "failure_category": FailureCategory.SCHEMA_FAILURE,
                    "failure_message": (
                        "Provider stopped at the output token limit before "
                        "completing structured data."
                    ),
                }
            )
        }
    )
    at = _app_test()
    _show_detail(at, result)

    at.run()

    assert "Increase its output tokens and generate again" in _rendered_text(at)


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
    assert [tab.label for tab in at.tabs] == ["Run configuration"]
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

    assert _element(at.button, "Open test case TC-001 detail")
    assert at.tabs[-1].label == "Run configuration"
    snapshot = next(
        table.value
        for table in at.table
        if list(table.value.columns) == ["Setting", "Value"]
    )
    assert dict(zip(snapshot["Setting"], snapshot["Value"], strict=True))[
        "Provider"
    ] == "LM Studio"
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
    _open_document_step(at, RunType.STAGED_SINGLE_AGENT)
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
    assert settings.api_key == "test-gemini-key"
    assert set(settings.agent_prompts) == {
        "requirements",
        "scenarios",
        "test_cases",
    }
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
    _open_document_step(at, RunType.CENTRALIZED_MULTI_AGENT)
    _element(at.file_uploader, "BRD/SRS PDF").set_value(
        [("customer-login.pdf", b"%PDF-1.4\n", "application/pdf")]
    )
    _element(at.button, "Generate test cases").click()
    at.run()

    text = _rendered_text(at)
    assert "Live agent details" in text
    assert "Generation progress" in text
    assert "Plan complete" in text
    assert "Prepare document" in text
    assert "Analyzer 1" in text
    assert "Model: analyst-model" in text
    assert "Extract testable business rules with source references." in text
    assert "1 assigned source chunk · pages 1" in text
    assert "Candidate requirements" in text
    assert "Artifacts" in text
    assert at.session_state["timeline_activity"][0].artifact_label == (
        "Candidate requirements"
    )
    assert "TC-001 · Sign in with valid credentials" in text
    assert "Open artifact" not in {button.label for button in at.button}
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
    _open_document_step(at, RunType.CENTRALIZED_MULTI_AGENT)
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
    _open_document_step(at, RunType.SINGLE_PROMPT)
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


def test_unexpected_generation_error_stays_in_create_and_redacts_settings(
    monkeypatch,
) -> None:
    token = "runner-secret"
    base_url = "http://private-llama:8080/v1"
    calls = 0
    monkeypatch.setenv("GEMINI_API_KEY", token)
    monkeypatch.setenv("LLAMA_CPP_BASE_URL", base_url)

    def fail_runner(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError(f"runner stopped: {token} {base_url}")

    at = _app_test()
    at.session_state["_runner"] = fail_runner
    at.run()
    _open_settings_step(at, RunType.SINGLE_PROMPT)
    _element(at.selectbox, "Agent provider").set_value("llama_cpp")
    at.run()
    assert _element(at.selectbox, "Agent model").value == LOCAL_TEST_MODELS[0][0]
    _element(at.button, "Continue to document").click()
    at.run()
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
    assert at.session_state["run_type"] is RunType.SINGLE_PROMPT


def test_edit_run_settings_preserves_upload_and_run_type() -> None:
    at = _app_test()
    at.run()
    _open_document_step(at, RunType.STAGED_SINGLE_AGENT)
    _element(at.file_uploader, "BRD/SRS PDF").set_value(
        [("customer-login.pdf", b"%PDF-1.4\n", "application/pdf")]
    )
    at.run()
    _element(at.button, "Edit run settings").click()
    at.run()
    _element(at.selectbox, "Agent model").set_value("gemini-2.5-pro")
    _element(at.button, "Continue to document").click()
    at.run()

    assert at.session_state["view"] == "create"
    assert at.session_state["run_provider_settings"].model == "gemini-2.5-pro"
    assert at.session_state["retained_pdf"].name == "customer-login.pdf"
    assert not _element(at.button, "Generate test cases").disabled
    assert at.session_state["run_type"] is RunType.STAGED_SINGLE_AGENT


def test_multi_agent_prompts_start_from_shared_agent_setup() -> None:
    repository = FakeRepository()
    repository.agent_setups["scout"] = AgentSetup(
        agent="scout",
        role="Payments requirement specialist",
        instructions="Prioritize validation and exception rules.",
    )
    at = _app_test(repository)
    at.run()
    _open_settings_step(at, RunType.CENTRALIZED_MULTI_AGENT)

    assert _element(at.text_area, "Scout prompt").value == (
        "Prioritize validation and exception rules."
    )
    original = repository.agent_setups.copy()
    _element(at.text_area, "Scout prompt").set_value("Run-specific prompt.")
    _element(at.button, "Continue to document").click()
    at.run()
    assert repository.agent_setups == original


def test_historical_agent_configuration_renders_without_migration() -> None:
    result = _detailed_run()
    configuration = {
        "analyst_model": "old-analyst",
        "agents": {
            agent: {"model": f"old-{agent}", "prompt": f"Historical {agent}."}
            for agent in ("analyst", "test_generator", "reviewer")
        },
    }
    result = result.model_copy(update={"manifest": result.manifest.model_copy(
        update={"run_type": RunType.CENTRALIZED_MULTI_AGENT, "configuration": configuration}
    )})
    at = _app_test()
    _show_detail(at, result)
    at.run()
    assert not at.exception
    assert {item.value for item in at.text_area if item.disabled} >= {
        "Historical analyst.", "Historical test_generator.", "Historical reviewer."
    }
    assert result.manifest.configuration == configuration


def test_run_settings_capture_custom_prompts_without_credentials() -> None:
    at = _app_test()
    at.run()
    _open_settings_step(at, RunType.STAGED_SINGLE_AGENT)
    _element(at.text_area, "Scenarios step prompt").set_value("Focus on edge cases.")
    _element(at.selectbox, "Scenarios step thinking level").set_value("low")
    _element(at.number_input, "Scenarios step output tokens").set_value(12_000)
    _element(at.button, "Continue to document").click()
    at.run()

    settings = at.session_state["run_provider_settings"]
    assert settings.agent_prompts["scenarios"] == "Focus on edge cases."
    assert settings.thinking_level_for("scenarios") == "low"
    assert settings.agent_max_output_tokens["scenarios"] == 12_000
    snapshot = settings.snapshot(RunType.STAGED_SINGLE_AGENT)
    assert snapshot["agents"]["scenarios"]["prompt"] == "Focus on edge cases."
    assert snapshot["agents"]["scenarios"]["thinking_level"] == "low"
    assert snapshot["agents"]["scenarios"]["max_output_tokens"] == 12_000
    assert "test-gemini-key" not in str(snapshot)


def test_missing_upload_stays_in_create_without_calling_runner() -> None:
    calls = 0

    def fake_runner(*args, **kwargs):
        nonlocal calls
        calls += 1

    at = _app_test()
    at.session_state["_runner"] = fake_runner
    at.run()
    _open_document_step(at, RunType.SINGLE_PROMPT)

    assert calls == 0
    assert at.session_state["view"] == "create"
    assert _element(at.button, "Generate test cases").disabled
    assert "Add a PDF to continue." in _rendered_text(at)


def test_invalid_run_settings_stay_in_settings_without_running(monkeypatch) -> None:
    calls = 0

    def fake_runner(*args, **kwargs):
        nonlocal calls
        calls += 1

    monkeypatch.setenv("GEMINI_API_KEY", "")
    at = _app_test()
    at.session_state["_runner"] = fake_runner
    at.run()
    _open_settings_step(at, RunType.SINGLE_PROMPT)
    _element(at.button, "Continue to document").click()
    at.run()

    assert calls == 0
    assert at.session_state["view"] == "create"
    assert at.session_state["create_step"] == 2
    assert "Gemini API key is required." in _rendered_text(at)
    assert "Gemini API key" not in {item.label for item in at.text_input}
