from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import os
from threading import Barrier

import psycopg
import pytest

from brd_srs_testgen.models import DocumentChunk, RunManifest, RunStatus, RunType
from brd_srs_testgen.storage import ImmutableRunError, RunRepository, StorageError
from tests.conftest import _is_local_address


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
