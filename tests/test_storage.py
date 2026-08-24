from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import os
from threading import Barrier

import psycopg
import pytest

from brd_srs_testgen.models import (
    AgentSetup,
    ArtifactBundle,
    DocumentChunk,
    FailureCategory,
    RunManifest,
    RunResult,
    RunStatus,
    RunType,
    SourceReference,
)
from brd_srs_testgen.storage import ImmutableRunError, RunRepository, StorageError
from brd_srs_testgen.validation import build_rtm, compute_metrics, validate_bundle
from tests.conftest import _is_local_address
from tests.factories import chunk as factory_chunk
from tests.factories import completed_run


def manifest(run_id: str = "run-1", **changes: object) -> RunManifest:
    values = {
        "run_id": run_id,
        "source_filename": "requirements.pdf",
        "document_hash": "a" * 64,
        "run_type": RunType.SINGLE_PROMPT,
        "status": RunStatus.RUNNING,
        "provider": "openai",
        "model": "gpt-test",
        "temperature": 0.0,
        "token_ceiling": 10_000,
        "prompt_version": "1",
        "schema_version": "1",
        "started_at": datetime.now(UTC),
    }
    values.update(changes)
    return RunManifest(**values)


def chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="p0002-c001",
            page_number=2,
            section="Second",
            text="Second page.",
            content_hash="b" * 64,
        ),
        DocumentChunk(
            chunk_id="p0001-c002",
            page_number=1,
            section="First",
            text="Later first-page chunk.",
            content_hash="c" * 64,
        ),
        DocumentChunk(
            chunk_id="p0001-c001",
            page_number=1,
            section="First",
            text="First page.",
            content_hash="d" * 64,
        ),
    ]


def evidence_chunks() -> list[DocumentChunk]:
    return [
        factory_chunk(),
        DocumentChunk(
            chunk_id="p0002-c001-a0c949e7ef48",
            page_number=2,
            section="SESSION MANAGEMENT",
            text="The system shall end an authenticated session on request.",
            content_hash="a0c949e7ef48bf263dfb2bdc23736541783096018c60449464f7496f2ae6a992",
        ),
    ]


def rich_completed_run(run_id: str = "normalized") -> RunResult:
    result = completed_run(run_id)
    assert result.bundle is not None
    first, second = (
        SourceReference(
            chunk_id=item.chunk_id,
            page_number=item.page_number,
            section=item.section,
            excerpt=item.text,
        )
        for item in evidence_chunks()
    )
    requirement = result.bundle.requirements[0].model_copy(
        update={
            "ambiguities": ["Lockout policy is unspecified.", "MFA is unspecified."],
            "dependency_ids": ["REQ-002", "REQ-001"],
            "source_references": [second, first],
        }
    )
    second_requirement = requirement.model_copy(
        update={
            "requirement_id": "REQ-002",
            "title": "End sessions",
            "description": "Authenticated users can sign out.",
            "ambiguities": ["Timeout behavior is unspecified."],
            "dependency_ids": ["REQ-001"],
            "source_references": [second],
        }
    )
    scenario = result.bundle.scenarios[0].model_copy(
        update={
            "preconditions": ["The user is registered.", "The account is enabled."],
            "requirement_ids": ["REQ-002", "REQ-001"],
            "source_references": [first, second],
        }
    )
    second_scenario = scenario.model_copy(
        update={
            "scenario_id": "SCN-002",
            "title": "Sign out",
            "objective": "Verify successful sign out.",
            "preconditions": ["The user is authenticated."],
            "requirement_ids": ["REQ-001"],
            "source_references": [second],
        }
    )
    test_case = result.bundle.test_cases[0].model_copy(
        update={
            "requirement_ids": ["REQ-002", "REQ-001"],
            "preconditions": ["Open the sign-in page.", "Use an enabled account."],
            "test_data": {
                "string": "user@example.com",
                "number": 7,
                "boolean": True,
                "null": None,
                "list": ["one", 2, False, None],
                "object": {"nested": "value", "enabled": True},
            },
            "steps": [
                result.bundle.test_cases[0].steps[0],
                result.bundle.test_cases[0].steps[0].model_copy(
                    update={
                        "step_number": 2,
                        "action": "Sign out.",
                        "expected_result": "The sign-in page is displayed.",
                    }
                ),
            ],
            "source_references": [second, first],
        }
    )
    second_test_case = test_case.model_copy(
        update={
            "test_case_id": "TC-002",
            "scenario_id": "SCN-002",
            "requirement_ids": ["REQ-001"],
            "title": "Sign out of an authenticated session",
            "preconditions": ["The user is authenticated."],
            "test_data": {"redirect": "/sign-in"},
            "steps": [test_case.steps[1]],
            "source_references": [second],
        }
    )
    bundle = ArtifactBundle(
        requirements=[second_requirement, requirement],
        scenarios=[second_scenario, scenario],
        test_cases=[second_test_case, test_case],
    )
    validation = validate_bundle(bundle, evidence_chunks())
    metrics = compute_metrics(
        bundle,
        validation,
        input_tokens=10,
        output_tokens=20,
        charged_tokens=30,
        latency_seconds=0.1,
        retries=1,
        schema_repairs=2,
        semantic_revisions=3,
        budget_exhausted=False,
    )
    return RunResult(
        manifest=result.manifest,
        bundle=bundle,
        validation=validation,
        rtm=build_rtm(bundle),
        metrics=metrics,
    )


def semantic_failure(run_id: str = "semantic-failure") -> RunResult:
    result = rich_completed_run(run_id)
    assert result.bundle is not None
    requirement = result.bundle.requirements[1].model_copy(
        update={
            "dependency_ids": ["REQ-999", "REQ-002"],
            "source_references": [
                result.bundle.requirements[1].source_references[0].model_copy(
                    update={"chunk_id": "missing-requirement-chunk"}
                ),
                result.bundle.requirements[1].source_references[1],
            ],
        }
    )
    scenario = result.bundle.scenarios[1].model_copy(
        update={
            "requirement_ids": ["REQ-998", "REQ-001"],
            "source_references": [
                result.bundle.scenarios[1].source_references[0],
                result.bundle.scenarios[1].source_references[1].model_copy(
                    update={"chunk_id": "missing-scenario-chunk"}
                ),
            ],
        }
    )
    test_case = result.bundle.test_cases[1].model_copy(
        update={
            "scenario_id": "SCN-999",
            "requirement_ids": ["REQ-997", "REQ-001"],
            "source_references": [
                result.bundle.test_cases[1].source_references[0].model_copy(
                    update={"chunk_id": "missing-test-case-chunk"}
                ),
                result.bundle.test_cases[1].source_references[1],
            ],
        }
    )
    bundle = result.bundle.model_copy(
        update={
            "requirements": [result.bundle.requirements[0], requirement],
            "scenarios": [result.bundle.scenarios[0], scenario],
            "test_cases": [result.bundle.test_cases[0], test_case],
        }
    )
    validation = validate_bundle(bundle, evidence_chunks())
    metrics = compute_metrics(
        bundle,
        validation,
        input_tokens=11,
        output_tokens=22,
        charged_tokens=33,
        latency_seconds=0.2,
        retries=1,
        schema_repairs=2,
        semantic_revisions=3,
        budget_exhausted=False,
    )
    return RunResult(
        manifest=result.manifest.model_copy(
            update={
                "status": RunStatus.FAILED,
                "failure_category": FailureCategory.SEMANTIC_VALIDATION,
                "failure_message": "Generated artifacts failed semantic validation.",
            }
        ),
        bundle=bundle,
        validation=validation,
        rtm=build_rtm(bundle),
        metrics=metrics,
    )


def duplicate_id_failure(run_id: str = "duplicate-ids") -> RunResult:
    result = rich_completed_run(run_id)
    assert result.bundle is not None
    first_case, second_case = result.bundle.test_cases
    step = second_case.steps[0]
    ordered_steps = [
        step.model_copy(
            update={"step_number": 2, "action": "Run first declared step."}
        ),
        step.model_copy(
            update={"step_number": 1, "action": "Run second declared step."}
        ),
        step.model_copy(
            update={"step_number": 1, "action": "Run third declared step."}
        ),
    ]
    bundle = ArtifactBundle(
        requirements=[
            result.bundle.requirements[0].model_copy(
                update={"requirement_id": "REQ-001"}
            ),
            result.bundle.requirements[1].model_copy(
                update={"dependency_ids": ["REQ-001", "REQ-001"]}
            ),
        ],
        scenarios=[
            result.bundle.scenarios[0].model_copy(update={"scenario_id": "SCN-001"}),
            result.bundle.scenarios[1].model_copy(
                update={"requirement_ids": ["REQ-001", "REQ-001"]}
            ),
        ],
        test_cases=[
            first_case.model_copy(
                update={"test_case_id": "TC-001", "scenario_id": "SCN-001"}
            ),
            second_case.model_copy(
                update={
                    "requirement_ids": ["REQ-001", "REQ-001"],
                    "steps": ordered_steps,
                }
            ),
        ],
    )
    validation = validate_bundle(bundle, evidence_chunks())
    metrics = compute_metrics(
        bundle,
        validation,
        input_tokens=12,
        output_tokens=24,
        charged_tokens=36,
        latency_seconds=0.3,
        retries=0,
        schema_repairs=0,
        semantic_revisions=1,
        budget_exhausted=False,
    )
    return RunResult(
        manifest=result.manifest.model_copy(
            update={
                "status": RunStatus.FAILED,
                "failure_category": FailureCategory.SEMANTIC_VALIDATION,
                "failure_message": "Generated IDs are duplicated.",
            }
        ),
        bundle=bundle,
        validation=validation,
        rtm=build_rtm(bundle),
        metrics=metrics,
    )


def start_run(repository: RunRepository, result: RunResult, *, save_chunks: bool = True) -> None:
    repository.create_run(
        result.manifest.model_copy(
            update={
                "status": RunStatus.RUNNING,
                "completed_at": None,
                "failure_category": None,
                "failure_message": None,
            }
        )
    )
    if save_chunks:
        repository.save_chunks(result.manifest.run_id, evidence_chunks())


def test_create_run_is_listed_as_interrupted_without_metrics(
    repository: RunRepository,
) -> None:
    repository.create_run(manifest())

    history = repository.list_runs()

    assert len(history) == 1
    assert history[0].display_status == "Interrupted"
    assert history[0].requirement_count is None
    assert history[0].scenario_count is None
    assert history[0].test_case_count is None


def test_fixture_uses_exact_local_test_database(
    repository: RunRepository, test_database_connection: psycopg.Connection
) -> None:
    database, address = test_database_connection.execute(
        "SELECT current_database(), inet_server_addr()"
    ).fetchone()

    assert database == "brd_srs_test"
    assert _is_local_address(address)


def test_local_database_address_predicate() -> None:
    assert _is_local_address(None)
    assert _is_local_address("127.0.0.1")
    assert _is_local_address("::1")
    assert not _is_local_address("203.0.113.1")


def test_list_runs_orders_newest_then_run_id_descending(
    repository: RunRepository,
) -> None:
    now = datetime.now(UTC)
    for run_id, started_at in (
        ("older", now - timedelta(seconds=1)),
        ("same-a", now),
        ("same-b", now),
    ):
        repository.create_run(manifest(run_id, started_at=started_at))

    assert [item.run_id for item in repository.list_runs()] == [
        "same-b",
        "same-a",
        "older",
    ]


def test_chunks_round_trip_in_page_and_chunk_order(repository: RunRepository) -> None:
    repository.create_run(manifest())
    repository.save_chunks("run-1", chunks())

    assert repository.load_chunks("run-1") == [chunks()[2], chunks()[1], chunks()[0]]


def test_events_load_in_sequence_order(repository: RunRepository) -> None:
    repository.create_run(manifest())
    occurred_at = datetime(2026, 8, 12, tzinfo=UTC)

    repository.append_event("run-1", "started", occurred_at)
    repository.append_event("run-1", "finished", occurred_at + timedelta(seconds=1))

    assert repository.load_events("run-1") == [
        {"sequence": 1, "occurred_at": occurred_at, "stage": "started"},
        {
            "sequence": 2,
            "occurred_at": occurred_at + timedelta(seconds=1),
            "stage": "finished",
        },
    ]


def test_duplicate_run_id_is_immutable(repository: RunRepository) -> None:
    repository.create_run(manifest())

    with pytest.raises(ImmutableRunError, match=r"^Run already exists\.$"):
        repository.create_run(manifest())


@pytest.mark.parametrize("operation", ["chunks", "event"])
def test_writes_require_an_existing_run(
    repository: RunRepository, operation: str
) -> None:
    with pytest.raises(StorageError, match=r"^Run does not exist\.$"):
        if operation == "chunks":
            repository.save_chunks("missing", chunks())
        else:
            repository.append_event("missing", "started")


def test_chunks_cannot_be_written_twice(repository: RunRepository) -> None:
    repository.create_run(manifest())
    repository.save_chunks("run-1", chunks())

    with pytest.raises(ImmutableRunError):
        repository.save_chunks("run-1", chunks())


def test_chunks_cannot_be_empty(repository: RunRepository) -> None:
    repository.create_run(manifest())

    with pytest.raises(StorageError, match=r"^Chunks cannot be empty\.$"):
        repository.save_chunks("run-1", [])

    assert repository.load_chunks("run-1") == []


def test_failed_chunk_batch_rolls_back_and_preserves_database_cause(
    repository: RunRepository,
) -> None:
    repository.create_run(manifest())
    duplicate = chunks()[0].model_copy(update={"chunk_id": "duplicate"})

    with pytest.raises(StorageError) as raised:
        repository.save_chunks("run-1", [duplicate, duplicate])

    assert isinstance(raised.value.__cause__, psycopg.Error)
    assert repository.load_chunks("run-1") == []


def test_concurrent_event_appends_have_unique_sequences(
    repository: RunRepository,
) -> None:
    repository.create_run(manifest())

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(
            executor.map(
                lambda number: repository.append_event("run-1", f"stage-{number}"),
                range(6),
            )
        )

    assert [event["sequence"] for event in repository.load_events("run-1")] == list(
        range(1, 7)
    )


def test_only_one_concurrent_chunk_batch_wins(repository: RunRepository) -> None:
    repository.create_run(manifest())
    batches = [
        [chunks()[0].model_copy(update={"chunk_id": "batch-a"})],
        [chunks()[0].model_copy(update={"chunk_id": "batch-b"})],
    ]
    barrier = Barrier(2)

    def save(batch: list[DocumentChunk]) -> str | ImmutableRunError:
        barrier.wait(timeout=5)
        try:
            repository.save_chunks("run-1", batch)
        except ImmutableRunError as error:
            return error
        return "saved"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(save, batches))

    assert results.count("saved") == 1
    assert sum(isinstance(result, ImmutableRunError) for result in results) == 1
    assert repository.load_chunks("run-1") in batches


@pytest.mark.parametrize("operation", ["chunks", "event"])
def test_terminal_runs_are_immutable(
    repository: RunRepository, operation: str
) -> None:
    repository.create_run(manifest())
    with psycopg.connect(os.environ["TEST_DATABASE_URL"]) as connection:
        connection.execute(
            "UPDATE runs SET status = 'completed', completed_at = now() WHERE run_id = %s",
            ("run-1",),
        )

    with pytest.raises(ImmutableRunError, match=r"^Terminal runs are immutable\.$"):
        if operation == "chunks":
            repository.save_chunks("run-1", chunks())
        else:
            repository.append_event("run-1", "late event")


def test_schema_has_no_raw_pdf_or_credential_columns(
    repository: RunRepository,
) -> None:
    with psycopg.connect(os.environ["TEST_DATABASE_URL"]) as connection:
        names = {
            row[0]
            for row in connection.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema()"
            )
        }

    assert names.isdisjoint({"pdf", "pdf_bytes", "api_key", "credential", "token"})


@pytest.mark.parametrize(
    ("run_id", "status", "started_at", "completed_at"),
    [
        ("", "running", datetime(2026, 8, 12, tzinfo=UTC), None),
        (
            "backwards-time",
            "completed",
            datetime(2026, 8, 12, tzinfo=UTC),
            datetime(2026, 8, 11, tzinfo=UTC),
        ),
    ],
)
def test_run_database_constraints_reject_invalid_inserts(
    repository: RunRepository,
    run_id: str,
    status: str,
    started_at: datetime,
    completed_at: datetime | None,
) -> None:
    with psycopg.connect(
        os.environ["TEST_DATABASE_URL"], autocommit=True
    ) as connection:
        with pytest.raises(psycopg.errors.CheckViolation):
            connection.execute(
                "INSERT INTO runs "
                "(run_id, source_filename, document_hash, run_type, status, provider, "
                "model, temperature, token_ceiling, prompt_version, schema_version, "
                "started_at, completed_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    run_id,
                    "requirements.pdf",
                    "a" * 64,
                    "single_prompt",
                    status,
                    "openai",
                    "gpt-test",
                    0,
                    10_000,
                    "1",
                    "1",
                    started_at,
                    completed_at,
                ),
            )


def test_create_run_rejects_non_running_manifest(repository: RunRepository) -> None:
    now = datetime.now(UTC)
    completed = manifest(
        status=RunStatus.COMPLETED,
        completed_at=now,
        started_at=now,
    )

    with pytest.raises(ImmutableRunError):
        repository.create_run(completed)


def test_blank_event_stage_is_rejected(repository: RunRepository) -> None:
    repository.create_run(manifest())

    with pytest.raises(StorageError):
        repository.append_event("run-1", "  ")


def test_completed_run_round_trips_all_normalized_artifacts(
    repository: RunRepository,
) -> None:
    result = completed_run()
    repository.create_run(
        result.manifest.model_copy(
            update={"status": RunStatus.RUNNING, "completed_at": None}
        )
    )
    repository.save_chunks(result.manifest.run_id, [factory_chunk()])

    repository.finalize(result)

    assert repository.load_run(result.manifest.run_id) == result
    assert [event["stage"] for event in repository.load_events(result.manifest.run_id)] == [
        "finished"
    ]


def test_normalized_graph_preserves_all_data_and_list_positions(
    repository: RunRepository,
) -> None:
    result = rich_completed_run()
    start_run(repository, result)

    repository.finalize(result)

    loaded = repository.load_run(result.manifest.run_id)
    assert loaded == result
    assert loaded.bundle is not None
    assert [item.requirement_id for item in loaded.bundle.requirements] == [
        "REQ-002",
        "REQ-001",
    ]
    assert [item.scenario_id for item in loaded.bundle.scenarios] == [
        "SCN-002",
        "SCN-001",
    ]
    assert [item.test_case_id for item in loaded.bundle.test_cases] == [
        "TC-002",
        "TC-001",
    ]

    with psycopg.connect(repository.database_url) as connection:
        tables = (
            "artifact_bundles",
            "requirements",
            "requirement_ambiguities",
            "requirement_dependencies",
            "requirement_sources",
            "scenarios",
            "scenario_preconditions",
            "scenario_requirements",
            "scenario_sources",
            "test_cases",
            "test_case_preconditions",
            "test_case_requirements",
            "test_case_data",
            "test_steps",
            "test_case_sources",
            "run_metrics",
            "validation_reports",
            "validation_issues",
            "validation_uncovered_requirements",
            "validation_orphan_scenarios",
            "validation_orphan_test_cases",
        )
        counts = {
            table: connection.execute(
                f"SELECT count(*) FROM {table} WHERE run_id = %s",
                (result.manifest.run_id,),
            ).fetchone()[0]
            for table in tables
        }
        root_orders = {
            table: connection.execute(
                f"SELECT {identifier}, position FROM {table} "
                "WHERE run_id = %s ORDER BY position",
                (result.manifest.run_id,),
            ).fetchall()
            for table, identifier in (
                ("requirements", "requirement_id"),
                ("scenarios", "scenario_id"),
                ("test_cases", "test_case_id"),
            )
        }
        ordered_children = {
            table: connection.execute(
                f"SELECT {owner}, position FROM {table} WHERE run_id = %s "
                f"ORDER BY {owner}, position",
                (result.manifest.run_id,),
            ).fetchall()
            for table, owner in (
                ("requirement_ambiguities", "requirement_position"),
                ("requirement_dependencies", "requirement_position"),
                ("requirement_sources", "requirement_position"),
                ("scenario_preconditions", "scenario_position"),
                ("scenario_requirements", "scenario_position"),
                ("scenario_sources", "scenario_position"),
                ("test_case_preconditions", "test_case_position"),
                ("test_case_requirements", "test_case_position"),
                ("test_case_sources", "test_case_position"),
            )
        }
        test_data = dict(
            connection.execute(
                "SELECT key, value FROM test_case_data "
                "WHERE run_id = %s AND test_case_position = 2 ORDER BY key",
                (result.manifest.run_id,),
            ).fetchall()
        )
        steps = connection.execute(
            "SELECT position, step_number FROM test_steps "
            "WHERE run_id = %s AND test_case_position = 2 ORDER BY position",
            (result.manifest.run_id,),
        ).fetchall()

    assert counts == {
        "artifact_bundles": 1,
        "requirements": 2,
        "requirement_ambiguities": 3,
        "requirement_dependencies": 3,
        "requirement_sources": 3,
        "scenarios": 2,
        "scenario_preconditions": 3,
        "scenario_requirements": 3,
        "scenario_sources": 3,
        "test_cases": 2,
        "test_case_preconditions": 3,
        "test_case_requirements": 3,
        "test_case_data": 7,
        "test_steps": 3,
        "test_case_sources": 3,
        "run_metrics": 1,
        "validation_reports": 1,
        "validation_issues": 0,
        "validation_uncovered_requirements": 0,
        "validation_orphan_scenarios": 0,
        "validation_orphan_test_cases": 0,
    }
    assert root_orders == {
        "requirements": [("REQ-002", 1), ("REQ-001", 2)],
        "scenarios": [("SCN-002", 1), ("SCN-001", 2)],
        "test_cases": [("TC-002", 1), ("TC-001", 2)],
    }
    assert all(
        positions == list(range(1, len(positions) + 1))
        for rows in ordered_children.values()
        for owner in {row[0] for row in rows}
        for positions in [[row[1] for row in rows if row[0] == owner]]
    )
    assert steps == [(1, 1), (2, 2)]
    assert test_data == {
        "boolean": True,
        "list": ["one", 2, False, None],
        "null": None,
        "number": 7,
        "object": {"nested": "value", "enabled": True},
        "string": "user@example.com",
    }


def test_failed_parsing_run_without_optional_results_round_trips(
    repository: RunRepository,
) -> None:
    running = manifest("parse-failure")
    repository.create_run(running)
    failed = RunResult(
        manifest=running.model_copy(
            update={
                "status": RunStatus.FAILED,
                "completed_at": datetime.now(UTC),
                "failure_category": FailureCategory.PARSING,
                "failure_message": "PDF contains insufficient extractable text.",
            }
        )
    )

    repository.finalize(failed)

    assert repository.load_run(failed.manifest.run_id) == failed


@pytest.mark.parametrize(
    "failure_category",
    [FailureCategory.PROVIDER_REJECTION, FailureCategory.BUDGET_EXHAUSTION],
)
def test_failed_run_with_metrics_but_no_bundle_round_trips(
    repository: RunRepository, failure_category: FailureCategory
) -> None:
    completed = completed_run(f"failed-{failure_category.value}")
    assert completed.metrics is not None
    running = completed.manifest.model_copy(
        update={"status": RunStatus.RUNNING, "completed_at": None}
    )
    repository.create_run(running)
    failed = RunResult(
        manifest=completed.manifest.model_copy(
            update={
                "status": RunStatus.FAILED,
                "failure_category": failure_category,
                "failure_message": "Generation stopped.",
            }
        ),
        metrics=completed.metrics.model_copy(
            update={
                "completion": False,
                "budget_exhausted": failure_category
                is FailureCategory.BUDGET_EXHAUSTION,
            }
        ),
    )

    repository.finalize(failed)

    assert repository.load_run(failed.manifest.run_id) == failed


def test_failed_semantic_validation_bundle_keeps_invalid_declared_references(
    repository: RunRepository,
) -> None:
    failed = semantic_failure()
    start_run(repository, failed)

    repository.finalize(failed)

    assert repository.load_run(failed.manifest.run_id) == failed
    with psycopg.connect(repository.database_url) as connection:
        validation_positions = {
            table: [
                row[0]
                for row in connection.execute(
                    f"SELECT position FROM {table} WHERE run_id = %s ORDER BY position",
                    (failed.manifest.run_id,),
                )
            ]
            for table in (
                "validation_issues",
                "validation_uncovered_requirements",
                "validation_orphan_scenarios",
                "validation_orphan_test_cases",
            )
        }
    assert validation_positions["validation_issues"]
    assert all(
        positions == list(range(1, len(positions) + 1))
        for positions in validation_positions.values()
    )


def test_failed_duplicate_ids_and_step_numbers_round_trip_exactly(
    repository: RunRepository,
) -> None:
    failed = duplicate_id_failure()
    start_run(repository, failed)

    repository.finalize(failed)

    loaded = repository.load_run(failed.manifest.run_id)
    assert loaded == failed
    assert loaded.bundle is not None
    assert [item.requirement_id for item in loaded.bundle.requirements] == [
        "REQ-001",
        "REQ-001",
    ]
    assert [item.scenario_id for item in loaded.bundle.scenarios] == [
        "SCN-001",
        "SCN-001",
    ]
    assert [item.test_case_id for item in loaded.bundle.test_cases] == [
        "TC-001",
        "TC-001",
    ]
    assert [step.step_number for step in loaded.bundle.test_cases[1].steps] == [
        2,
        1,
        1,
    ]
    assert loaded.validation is not None
    assert [issue.code for issue in loaded.validation.issues] == [
        "duplicate_id",
        "duplicate_id",
        "duplicate_id",
    ]


def test_terminal_run_cannot_be_finalized_twice(repository: RunRepository) -> None:
    result = completed_run("immutable")
    start_run(repository, result)
    repository.finalize(result)

    with pytest.raises(ImmutableRunError, match=r"^Terminal runs are immutable\.$"):
        repository.finalize(result)


def test_finalize_rejects_running_result(repository: RunRepository) -> None:
    with pytest.raises(
        ImmutableRunError, match=r"^Finalization requires a terminal run\.$"
    ):
        repository.finalize(RunResult(manifest=manifest()))


@pytest.mark.parametrize(
    "change",
    [
        {"source_filename": "changed.pdf"},
        {"document_hash": "b" * 64},
        {"run_type": RunType.STAGED_SINGLE_AGENT},
        {"provider": "changed-provider"},
        {"model": "changed-model"},
        {"temperature": 0.5},
        {"token_ceiling": 200_000},
        {"prompt_version": "changed-prompt"},
        {"schema_version": "changed-schema"},
        {"started_at": datetime(2026, 8, 11, tzinfo=UTC)},
    ],
)
def test_finalize_rejects_changed_immutable_configuration_and_rolls_back(
    repository: RunRepository, change: dict[str, object]
) -> None:
    result = completed_run("changed-config")
    running = result.manifest.model_copy(
        update={"status": RunStatus.RUNNING, "completed_at": None}
    )
    repository.create_run(running)
    changed = result.model_copy(
        update={"manifest": result.manifest.model_copy(update=change)}
    )

    with pytest.raises(
        ImmutableRunError, match=r"^Run configuration cannot be changed\.$"
    ):
        repository.finalize(changed)

    assert repository.load_run(running.run_id) == RunResult(manifest=running)
    assert repository.load_events(running.run_id) == []


def test_failed_finalization_rolls_back_the_complete_transaction(
    repository: RunRepository,
) -> None:
    result = rich_completed_run("rollback")
    start_run(repository, result)
    repository.append_event(result.manifest.run_id, "started")
    assert result.bundle is not None
    case = result.bundle.test_cases[1]
    overflowing_steps = [
        case.steps[0],
        case.steps[1].model_copy(update={"step_number": 2**40}),
    ]
    invalid_bundle = result.bundle.model_copy(
        update={
            "test_cases": [
                result.bundle.test_cases[0],
                case.model_copy(update={"steps": overflowing_steps}),
            ]
        }
    )
    invalid = result.model_copy(update={"bundle": invalid_bundle})

    with pytest.raises(StorageError) as raised:
        repository.finalize(invalid)

    assert isinstance(raised.value.__cause__, psycopg.Error)
    running = result.manifest.model_copy(
        update={"status": RunStatus.RUNNING, "completed_at": None}
    )
    assert repository.load_run(result.manifest.run_id) == RunResult(manifest=running)
    assert [event["stage"] for event in repository.load_events(result.manifest.run_id)] == [
        "started"
    ]
    with psycopg.connect(repository.database_url) as connection:
        assert all(
            connection.execute(
                f"SELECT count(*) FROM {table} WHERE run_id = %s",
                (result.manifest.run_id,),
            ).fetchone()[0]
            == 0
            for table in (
                "artifact_bundles",
                "requirements",
                "scenarios",
                "test_cases",
                "validation_reports",
                "run_metrics",
            )
        )


def test_load_run_requires_an_existing_run(repository: RunRepository) -> None:
    with pytest.raises(StorageError, match=r"^Run does not exist\.$"):
        repository.load_run("missing")


def test_interrupted_running_run_loads_without_optional_results(
    repository: RunRepository,
) -> None:
    running = manifest("interrupted")
    repository.create_run(running)
    repository.save_chunks(running.run_id, chunks())

    assert repository.load_run(running.run_id) == RunResult(manifest=running)


def test_load_run_rejects_roots_without_a_bundle_marker(
    repository: RunRepository,
) -> None:
    running = manifest("partial-graph")
    repository.create_run(running)
    with psycopg.connect(repository.database_url) as connection:
        connection.execute(
            "INSERT INTO requirements "
            "(run_id, requirement_id, position, title, description, "
            "requirement_type, module, priority) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (
                running.run_id,
                "REQ-001",
                1,
                "Stored requirement",
                "A root was stored without a bundle marker.",
                "functional",
                "Storage",
                "high",
            ),
        )

    with pytest.raises(
        StorageError, match=r"^Stored artifact graph is incomplete\.$"
    ):
        repository.load_run(running.run_id)


def test_partial_artifact_bundle_round_trips(repository: RunRepository) -> None:
    completed = completed_run("partial-bundle")
    assert completed.bundle is not None
    bundle = ArtifactBundle(
        requirements=[completed.bundle.requirements[0]],
        scenarios=[],
        test_cases=[],
    )
    result = RunResult(
        manifest=completed.manifest,
        bundle=bundle,
        rtm=build_rtm(bundle),
    )
    start_run(repository, result)

    repository.finalize(result)

    assert repository.load_run(result.manifest.run_id) == result


def test_empty_artifact_bundle_round_trips(repository: RunRepository) -> None:
    completed = completed_run("empty-bundle")
    bundle = ArtifactBundle(requirements=[], scenarios=[], test_cases=[])
    result = RunResult(
        manifest=completed.manifest,
        bundle=bundle,
        rtm=[],
    )
    start_run(repository, result, save_chunks=False)

    repository.finalize(result)

    assert repository.load_run(result.manifest.run_id) == result


def test_list_runs_receives_metric_counts_after_finalization(
    repository: RunRepository,
) -> None:
    result = rich_completed_run("history-counts")
    start_run(repository, result)
    repository.finalize(result)

    [history] = repository.list_runs()

    assert (
        history.requirement_count,
        history.scenario_count,
        history.test_case_count,
    ) == (2, 2, 2)


def test_terminal_chunks_events_and_finalization_are_immutable(
    repository: RunRepository,
) -> None:
    result = completed_run("terminal-writes")
    start_run(repository, result)
    repository.append_event(result.manifest.run_id, "started")
    repository.finalize(result)
    original_chunks = repository.load_chunks(result.manifest.run_id)
    original_events = repository.load_events(result.manifest.run_id)

    for operation in (
        lambda: repository.save_chunks(result.manifest.run_id, [factory_chunk()]),
        lambda: repository.append_event(result.manifest.run_id, "late"),
        lambda: repository.finalize(result),
    ):
        with pytest.raises(ImmutableRunError):
            operation()

    assert repository.load_chunks(result.manifest.run_id) == original_chunks
    assert repository.load_events(result.manifest.run_id) == original_events


def test_agent_setups_round_trip_as_shared_configuration(
    repository: RunRepository,
) -> None:
    setups = repository.load_agent_setups()
    assert setups["analyst"].role == "Requirement analyst"
    assert setups["analyst"].instructions == (
        "Extract supported functional, nonfunctional, and business requirements "
        "from assigned evidence. Preserve dependencies, ambiguities, and exact "
        "citations. Do not infer unsupported behavior."
    )

    setups["analyst"] = AgentSetup(
        agent="analyst",
        role="Payments requirement specialist",
        instructions="Prioritize validation and exception rules.",
    )
    repository.save_agent_setups(setups.values())

    assert repository.load_agent_setups() == setups
