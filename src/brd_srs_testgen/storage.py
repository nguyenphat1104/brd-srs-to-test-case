from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import psycopg
from psycopg.rows import dict_row

from .models import DocumentChunk, RunHistoryItem, RunManifest, RunStatus


class StorageError(RuntimeError):
    pass


class ImmutableRunError(StorageError):
    pass


class RunRepository:
    def __init__(self, database_url: str) -> None:
        if not database_url or not database_url.strip():
            raise StorageError("DATABASE_URL is required.")
        self.database_url = database_url

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def initialize(self) -> None:
        try:
            schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
            with self._connect() as connection:
                connection.execute(schema)
        except (OSError, psycopg.Error) as error:
            raise StorageError("Database initialization failed.") from error

    def create_run(self, manifest: RunManifest) -> None:
        if manifest.status is not RunStatus.RUNNING:
            raise ImmutableRunError("Runs must start in running state.")
        fields = tuple(RunManifest.model_fields)
        values = tuple(getattr(manifest, field) for field in fields)
        try:
            with self._connect() as connection:
                connection.execute(
                    f"INSERT INTO runs ({', '.join(fields)}) VALUES ({', '.join(['%s'] * len(fields))})",
                    values,
                )
        except psycopg.errors.UniqueViolation as error:
            raise ImmutableRunError("Run already exists.") from error
        except psycopg.Error as error:
            raise StorageError("Database operation failed.") from error

    @staticmethod
    def _require_running(connection: psycopg.Connection, run_id: str) -> None:
        row = connection.execute(
            "SELECT status FROM runs WHERE run_id = %s FOR UPDATE", (run_id,)
        ).fetchone()
        if row is None:
            raise StorageError("Run does not exist.")
        if row["status"] != RunStatus.RUNNING:
            raise ImmutableRunError("Terminal runs are immutable.")

    def save_chunks(self, run_id: str, chunks: Iterable[DocumentChunk]) -> None:
        items = list(chunks)
        try:
            with self._connect() as connection:
                self._require_running(connection, run_id)
                if not items:
                    raise StorageError("Chunks cannot be empty.")
                if connection.execute(
                    "SELECT 1 FROM document_chunks WHERE run_id = %s LIMIT 1", (run_id,)
                ).fetchone():
                    raise ImmutableRunError("Chunks already exist.")
                with connection.cursor() as cursor:
                    cursor.executemany(
                        "INSERT INTO document_chunks "
                        "(run_id, chunk_id, page_number, section, text, content_hash) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        (
                            (
                                run_id,
                                chunk.chunk_id,
                                chunk.page_number,
                                chunk.section,
                                chunk.text,
                                chunk.content_hash,
                            )
                            for chunk in items
                        ),
                    )
        except StorageError:
            raise
        except psycopg.Error as error:
            raise StorageError("Database operation failed.") from error

    def load_chunks(self, run_id: str) -> list[DocumentChunk]:
        try:
            with self._connect() as connection:
                if connection.execute(
                    "SELECT 1 FROM runs WHERE run_id = %s", (run_id,)
                ).fetchone() is None:
                    raise StorageError("Run does not exist.")
                rows = connection.execute(
                    "SELECT chunk_id, page_number, section, text, content_hash "
                    "FROM document_chunks WHERE run_id = %s "
                    "ORDER BY page_number, chunk_id",
                    (run_id,),
                ).fetchall()
            return [DocumentChunk.model_validate(row, strict=True) for row in rows]
        except StorageError:
            raise
        except psycopg.Error as error:
            raise StorageError("Database operation failed.") from error

    def append_event(
        self, run_id: str, stage: str, occurred_at: datetime | None = None
    ) -> None:
        if not isinstance(stage, str) or not stage.strip():
            raise StorageError("Event stage is required.")
        try:
            with self._connect() as connection:
                self._require_running(connection, run_id)
                connection.execute(
                    "INSERT INTO run_events (run_id, sequence, occurred_at, stage) "
                    "SELECT %s, COALESCE(MAX(sequence), 0) + 1, %s, %s "
                    "FROM run_events WHERE run_id = %s",
                    (run_id, occurred_at or datetime.now(UTC), stage, run_id),
                )
        except StorageError:
            raise
        except psycopg.Error as error:
            raise StorageError("Database operation failed.") from error

    def load_events(self, run_id: str) -> list[dict[str, object]]:
        try:
            with self._connect() as connection:
                if connection.execute(
                    "SELECT 1 FROM runs WHERE run_id = %s", (run_id,)
                ).fetchone() is None:
                    raise StorageError("Run does not exist.")
                return connection.execute(
                    "SELECT sequence, occurred_at, stage FROM run_events "
                    "WHERE run_id = %s ORDER BY sequence",
                    (run_id,),
                ).fetchall()
        except StorageError:
            raise
        except psycopg.Error as error:
            raise StorageError("Database operation failed.") from error

    def list_runs(self) -> list[RunHistoryItem]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT r.run_id, r.source_filename, r.run_type, r.status, "
                    "r.provider, r.model, r.started_at, r.completed_at, "
                    "m.requirement_count, m.scenario_count, m.test_case_count "
                    "FROM runs r LEFT JOIN run_metrics m USING (run_id) "
                    "ORDER BY r.started_at DESC, r.run_id DESC"
                ).fetchall()
            return [RunHistoryItem.model_validate(row) for row in rows]
        except psycopg.Error as error:
            raise StorageError("Database operation failed.") from error
