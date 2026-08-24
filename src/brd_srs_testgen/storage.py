from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import psycopg
from pydantic import ValidationError
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .models import (
    AgentSetup,
    ArtifactBundle,
    CoverageScore,
    DocumentChunk,
    Requirement,
    RunHistoryItem,
    RunManifest,
    RunMetrics,
    RunResult,
    RunStatus,
    Scenario,
    SourceReference,
    TestCase,
    TestStep,
    ValidationIssue,
    ValidationReport,
    default_agent_setups,
)
from .validation import build_rtm


class StorageError(RuntimeError):
    pass


class ImmutableRunError(StorageError):
    pass


def _manifest(row: dict[str, object]) -> RunManifest:
    return RunManifest(
        run_id=row["run_id"],
        source_filename=row["source_filename"],
        document_hash=row["document_hash"],
        run_type=row["run_type"],
        status=row["status"],
        provider=row["provider"],
        model=row["model"],
        temperature=row["temperature"],
        token_ceiling=row["token_ceiling"],
        prompt_version=row["prompt_version"],
        schema_version=row["schema_version"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        failure_category=row["failure_category"],
        failure_message=row["failure_message"],
    )


def _source(row: dict[str, object]) -> SourceReference:
    return SourceReference(
        chunk_id=row["chunk_id"],
        page_number=row["page_number"],
        section=row["section"],
        excerpt=row["excerpt"],
    )


def _group(
    rows: list[dict[str, object]], owner: str
) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row[owner]), []).append(row)
    return grouped


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

    def load_agent_setups(self) -> dict[str, AgentSetup]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT agent, role, instructions FROM agent_setups ORDER BY agent"
                ).fetchall()
            setups = default_agent_setups()
            setups.update(
                {
                    row["agent"]: AgentSetup.model_validate(row, strict=True)
                    for row in rows
                }
            )
            return setups
        except (psycopg.Error, ValidationError) as error:
            raise StorageError("Agent setup could not be loaded.") from error

    def save_agent_setups(self, setups: Iterable[AgentSetup]) -> None:
        items = list(setups)
        expected = set(default_agent_setups())
        if {item.agent for item in items} != expected:
            raise StorageError("All centralized agent setups are required.")
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.executemany(
                        "INSERT INTO agent_setups (agent, role, instructions, updated_at) "
                        "VALUES (%s, %s, %s, %s) "
                        "ON CONFLICT (agent) DO UPDATE SET role = EXCLUDED.role, "
                        "instructions = EXCLUDED.instructions, updated_at = EXCLUDED.updated_at",
                        (
                            (
                                item.agent,
                                item.role,
                                item.instructions,
                                datetime.now(UTC),
                            )
                            for item in items
                        ),
                    )
        except psycopg.Error as error:
            raise StorageError("Agent setup could not be saved.") from error

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
    def _require_running(
        connection: psycopg.Connection, run_id: str
    ) -> dict[str, object]:
        row = connection.execute(
            "SELECT run_id, source_filename, document_hash, run_type, status, "
            "provider, model, temperature, token_ceiling, prompt_version, "
            "schema_version, started_at, completed_at, failure_category, "
            "failure_message FROM runs WHERE run_id = %s FOR UPDATE",
            (run_id,),
        ).fetchone()
        if row is None:
            raise StorageError("Run does not exist.")
        if row["status"] != RunStatus.RUNNING.value:
            raise ImmutableRunError("Terminal runs are immutable.")
        return row

    @staticmethod
    def _append_event(
        connection: psycopg.Connection,
        run_id: str,
        stage: str,
        occurred_at: datetime,
    ) -> None:
        connection.execute(
            "INSERT INTO run_events (run_id, sequence, occurred_at, stage) "
            "SELECT %s, COALESCE(MAX(sequence), 0) + 1, %s, %s "
            "FROM run_events WHERE run_id = %s",
            (run_id, occurred_at, stage, run_id),
        )

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
                self._append_event(
                    connection, run_id, stage, occurred_at or datetime.now(UTC)
                )
        except StorageError:
            raise
        except psycopg.Error as error:
            raise StorageError("Database operation failed.") from error

    def finalize(self, result: RunResult) -> None:
        manifest = result.manifest
        if manifest.status is RunStatus.RUNNING:
            raise ImmutableRunError("Finalization requires a terminal run.")
        try:
            with self._connect() as connection:
                persisted = _manifest(
                    self._require_running(connection, manifest.run_id)
                )
                immutable_fields = (
                    "run_id",
                    "source_filename",
                    "document_hash",
                    "run_type",
                    "provider",
                    "model",
                    "temperature",
                    "token_ceiling",
                    "prompt_version",
                    "schema_version",
                    "started_at",
                )
                if any(
                    getattr(persisted, field) != getattr(manifest, field)
                    for field in immutable_fields
                ):
                    raise ImmutableRunError("Run configuration cannot be changed.")
                if result.bundle is not None:
                    self._insert_bundle(connection, manifest.run_id, result.bundle)
                if result.validation is not None:
                    self._insert_validation(
                        connection, manifest.run_id, result.validation
                    )
                if result.metrics is not None:
                    self._insert_metrics(connection, manifest.run_id, result.metrics)
                if result.coverage is not None:
                    self._insert_coverage(connection, manifest.run_id, result.coverage)
                self._append_event(
                    connection, manifest.run_id, "finished", manifest.completed_at
                )
                updated = connection.execute(
                    "UPDATE runs SET status = %s, completed_at = %s, "
                    "failure_category = %s, failure_message = %s "
                    "WHERE run_id = %s AND status = %s",
                    (
                        manifest.status.value,
                        manifest.completed_at,
                        manifest.failure_category.value
                        if manifest.failure_category is not None
                        else None,
                        manifest.failure_message,
                        manifest.run_id,
                        RunStatus.RUNNING.value,
                    ),
                )
                if updated.rowcount != 1:
                    raise ImmutableRunError("Terminal runs are immutable.")
        except StorageError:
            raise
        except (psycopg.Error, ValidationError) as error:
            raise StorageError("Database operation failed.") from error

    @staticmethod
    def _insert_bundle(
        connection: psycopg.Connection, run_id: str, bundle: ArtifactBundle
    ) -> None:
        connection.execute(
            "INSERT INTO artifact_bundles (run_id) VALUES (%s)", (run_id,)
        )
        with connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO requirements "
                "(run_id, requirement_id, position, title, description, "
                "requirement_type, module, priority) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    (
                        run_id,
                        item.requirement_id,
                        requirement_position,
                        item.title,
                        item.description,
                        item.requirement_type.value,
                        item.module,
                        item.priority.value,
                    )
                    for requirement_position, item in enumerate(bundle.requirements, 1)
                ),
            )
            cursor.executemany(
                "INSERT INTO requirement_ambiguities "
                "(run_id, requirement_position, position, value) "
                "VALUES (%s, %s, %s, %s)",
                (
                    (run_id, requirement_position, position, value)
                    for requirement_position, item in enumerate(
                        bundle.requirements, 1
                    )
                    for position, value in enumerate(item.ambiguities, 1)
                ),
            )
            cursor.executemany(
                "INSERT INTO requirement_dependencies "
                "(run_id, requirement_position, position, dependency_id) "
                "VALUES (%s, %s, %s, %s)",
                (
                    (run_id, requirement_position, position, value)
                    for requirement_position, item in enumerate(
                        bundle.requirements, 1
                    )
                    for position, value in enumerate(item.dependency_ids, 1)
                ),
            )
            cursor.executemany(
                "INSERT INTO requirement_sources "
                "(run_id, requirement_position, position, chunk_id, page_number, "
                "section, excerpt) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    (
                        run_id,
                        requirement_position,
                        position,
                        source.chunk_id,
                        source.page_number,
                        source.section,
                        source.excerpt,
                    )
                    for requirement_position, item in enumerate(
                        bundle.requirements, 1
                    )
                    for position, source in enumerate(item.source_references, 1)
                ),
            )
            cursor.executemany(
                "INSERT INTO scenarios "
                "(run_id, scenario_id, position, title, objective, scenario_type) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    (
                        run_id,
                        item.scenario_id,
                        scenario_position,
                        item.title,
                        item.objective,
                        item.scenario_type.value,
                    )
                    for scenario_position, item in enumerate(bundle.scenarios, 1)
                ),
            )
            cursor.executemany(
                "INSERT INTO scenario_preconditions "
                "(run_id, scenario_position, position, value) "
                "VALUES (%s, %s, %s, %s)",
                (
                    (run_id, scenario_position, position, value)
                    for scenario_position, item in enumerate(bundle.scenarios, 1)
                    for position, value in enumerate(item.preconditions, 1)
                ),
            )
            cursor.executemany(
                "INSERT INTO scenario_requirements "
                "(run_id, scenario_position, position, requirement_id) "
                "VALUES (%s, %s, %s, %s)",
                (
                    (run_id, scenario_position, position, value)
                    for scenario_position, item in enumerate(bundle.scenarios, 1)
                    for position, value in enumerate(item.requirement_ids, 1)
                ),
            )
            cursor.executemany(
                "INSERT INTO scenario_sources "
                "(run_id, scenario_position, position, chunk_id, page_number, "
                "section, excerpt) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    (
                        run_id,
                        scenario_position,
                        position,
                        source.chunk_id,
                        source.page_number,
                        source.section,
                        source.excerpt,
                    )
                    for scenario_position, item in enumerate(bundle.scenarios, 1)
                    for position, source in enumerate(item.source_references, 1)
                ),
            )
            cursor.executemany(
                "INSERT INTO test_cases "
                "(run_id, test_case_id, position, scenario_id, title, priority) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    (
                        run_id,
                        item.test_case_id,
                        test_case_position,
                        item.scenario_id,
                        item.title,
                        item.priority.value,
                    )
                    for test_case_position, item in enumerate(bundle.test_cases, 1)
                ),
            )
            cursor.executemany(
                "INSERT INTO test_case_preconditions "
                "(run_id, test_case_position, position, value) "
                "VALUES (%s, %s, %s, %s)",
                (
                    (run_id, test_case_position, position, value)
                    for test_case_position, item in enumerate(bundle.test_cases, 1)
                    for position, value in enumerate(item.preconditions, 1)
                ),
            )
            cursor.executemany(
                "INSERT INTO test_case_requirements "
                "(run_id, test_case_position, position, requirement_id) "
                "VALUES (%s, %s, %s, %s)",
                (
                    (run_id, test_case_position, position, value)
                    for test_case_position, item in enumerate(bundle.test_cases, 1)
                    for position, value in enumerate(item.requirement_ids, 1)
                ),
            )
            cursor.executemany(
                "INSERT INTO test_case_data "
                "(run_id, test_case_position, key, value) VALUES (%s, %s, %s, %s)",
                (
                    (run_id, test_case_position, key, Jsonb(value))
                    for test_case_position, item in enumerate(bundle.test_cases, 1)
                    for key, value in item.test_data.items()
                ),
            )
            cursor.executemany(
                "INSERT INTO test_steps "
                "(run_id, test_case_position, position, step_number, action, "
                "expected_result) VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    (
                        run_id,
                        test_case_position,
                        position,
                        step.step_number,
                        step.action,
                        step.expected_result,
                    )
                    for test_case_position, item in enumerate(bundle.test_cases, 1)
                    for position, step in enumerate(item.steps, 1)
                ),
            )
            cursor.executemany(
                "INSERT INTO test_case_sources "
                "(run_id, test_case_position, position, chunk_id, page_number, "
                "section, excerpt) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    (
                        run_id,
                        test_case_position,
                        position,
                        source.chunk_id,
                        source.page_number,
                        source.section,
                        source.excerpt,
                    )
                    for test_case_position, item in enumerate(bundle.test_cases, 1)
                    for position, source in enumerate(item.source_references, 1)
                ),
            )

    @staticmethod
    def _insert_validation(
        connection: psycopg.Connection, run_id: str, report: ValidationReport
    ) -> None:
        connection.execute(
            "INSERT INTO validation_reports (run_id, valid) VALUES (%s, %s)",
            (run_id, report.valid),
        )
        with connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO validation_issues "
                "(run_id, position, code, artifact_id, message) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    (
                        run_id,
                        position,
                        issue.code,
                        issue.artifact_id,
                        issue.message,
                    )
                    for position, issue in enumerate(report.issues, 1)
                ),
            )
            for table, values in (
                (
                    "validation_uncovered_requirements",
                    report.uncovered_requirement_ids,
                ),
                ("validation_orphan_scenarios", report.orphan_scenario_ids),
                ("validation_orphan_test_cases", report.orphan_test_case_ids),
            ):
                cursor.executemany(
                    f"INSERT INTO {table} VALUES (%s, %s, %s)",
                    (
                        (run_id, position, value)
                        for position, value in enumerate(values, 1)
                    ),
                )

    @staticmethod
    def _insert_metrics(
        connection: psycopg.Connection, run_id: str, metrics: RunMetrics
    ) -> None:
        connection.execute(
            "INSERT INTO run_metrics "
            "(run_id, completion, schema_valid, citation_coverage, "
            "requirement_scenario_coverage, requirement_test_case_coverage, "
            "positive_scenario_coverage, non_positive_scenario_coverage, "
            "rtm_completeness, orphan_rate, invalid_reference_rate, "
            "duplicate_test_case_rate, requirement_count, scenario_count, "
            "test_case_count, input_tokens, output_tokens, charged_tokens, "
            "latency_seconds, retries, schema_repairs, semantic_revisions, "
            "budget_exhausted) VALUES ("
            "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
            "%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                run_id,
                metrics.completion,
                metrics.schema_valid,
                metrics.citation_coverage,
                metrics.requirement_scenario_coverage,
                metrics.requirement_test_case_coverage,
                metrics.positive_scenario_coverage,
                metrics.non_positive_scenario_coverage,
                metrics.rtm_completeness,
                metrics.orphan_rate,
                metrics.invalid_reference_rate,
                metrics.duplicate_test_case_rate,
                metrics.requirement_count,
                metrics.scenario_count,
                metrics.test_case_count,
                metrics.input_tokens,
                metrics.output_tokens,
                metrics.charged_tokens,
                metrics.latency_seconds,
                metrics.retries,
                metrics.schema_repairs,
                metrics.semantic_revisions,
                metrics.budget_exhausted,
            ),
        )

    @staticmethod
    def _insert_coverage(
        connection: psycopg.Connection, run_id: str, coverage: CoverageScore
    ) -> None:
        connection.execute(
            "INSERT INTO coverage_scores "
            "(run_id, precision, recall, f1, true_positive_count, "
            "false_positive_count, false_negative_count, total_coverage_units, "
            "total_test_cases, uncovered_unit_ids, unmapped_test_case_ids) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                run_id,
                coverage.precision,
                coverage.recall,
                coverage.f1,
                coverage.true_positive_count,
                coverage.false_positive_count,
                coverage.false_negative_count,
                coverage.total_coverage_units,
                coverage.total_test_cases,
                Jsonb(coverage.uncovered_unit_ids),
                Jsonb(coverage.unmapped_test_case_ids),
            ),
        )

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

    def load_run(self, run_id: str) -> RunResult:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT run_id, source_filename, document_hash, run_type, "
                    "status, provider, model, temperature, token_ceiling, "
                    "prompt_version, schema_version, started_at, completed_at, "
                    "failure_category, failure_message FROM runs WHERE run_id = %s",
                    (run_id,),
                ).fetchone()
                if row is None:
                    raise StorageError("Run does not exist.")
                manifest = _manifest(row)
                bundle = self._load_bundle(connection, run_id)
                validation = self._load_validation(connection, run_id)
                metrics = self._load_metrics(connection, run_id)
                coverage = self._load_coverage(connection, run_id)
                return RunResult(
                    manifest=manifest,
                    bundle=bundle,
                    validation=validation,
                    rtm=build_rtm(bundle) if bundle is not None else [],
                    metrics=metrics,
                    coverage=coverage,
                )
        except StorageError:
            raise
        except (psycopg.Error, ValidationError, KeyError, TypeError) as error:
            raise StorageError("Stored run data is invalid.") from error

    @staticmethod
    def _load_metrics(
        connection: psycopg.Connection, run_id: str
    ) -> RunMetrics | None:
        row = connection.execute(
            "SELECT completion, schema_valid, citation_coverage, "
            "requirement_scenario_coverage, requirement_test_case_coverage, "
            "positive_scenario_coverage, non_positive_scenario_coverage, "
            "rtm_completeness, orphan_rate, invalid_reference_rate, "
            "duplicate_test_case_rate, requirement_count, scenario_count, "
            "test_case_count, input_tokens, output_tokens, charged_tokens, "
            "latency_seconds, retries, schema_repairs, semantic_revisions, "
            "budget_exhausted FROM run_metrics WHERE run_id = %s",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return RunMetrics(
            completion=row["completion"],
            schema_valid=row["schema_valid"],
            citation_coverage=row["citation_coverage"],
            requirement_scenario_coverage=row["requirement_scenario_coverage"],
            requirement_test_case_coverage=row["requirement_test_case_coverage"],
            positive_scenario_coverage=row["positive_scenario_coverage"],
            non_positive_scenario_coverage=row["non_positive_scenario_coverage"],
            rtm_completeness=row["rtm_completeness"],
            orphan_rate=row["orphan_rate"],
            invalid_reference_rate=row["invalid_reference_rate"],
            duplicate_test_case_rate=row["duplicate_test_case_rate"],
            requirement_count=row["requirement_count"],
            scenario_count=row["scenario_count"],
            test_case_count=row["test_case_count"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            charged_tokens=row["charged_tokens"],
            latency_seconds=row["latency_seconds"],
            retries=row["retries"],
            schema_repairs=row["schema_repairs"],
            semantic_revisions=row["semantic_revisions"],
            budget_exhausted=row["budget_exhausted"],
        )

    @staticmethod
    def _load_coverage(
        connection: psycopg.Connection, run_id: str
    ) -> CoverageScore | None:
        row = connection.execute(
            "SELECT precision, recall, f1, true_positive_count, "
            "false_positive_count, false_negative_count, total_coverage_units, "
            "total_test_cases, uncovered_unit_ids, unmapped_test_case_ids "
            "FROM coverage_scores WHERE run_id = %s",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return CoverageScore(
            precision=row["precision"],
            recall=row["recall"],
            f1=row["f1"],
            true_positive_count=row["true_positive_count"],
            false_positive_count=row["false_positive_count"],
            false_negative_count=row["false_negative_count"],
            total_coverage_units=row["total_coverage_units"],
            total_test_cases=row["total_test_cases"],
            uncovered_unit_ids=row["uncovered_unit_ids"],
            unmapped_test_case_ids=row["unmapped_test_case_ids"],
        )

    @staticmethod
    def _load_validation(
        connection: psycopg.Connection, run_id: str
    ) -> ValidationReport | None:
        row = connection.execute(
            "SELECT valid FROM validation_reports WHERE run_id = %s", (run_id,)
        ).fetchone()
        if row is None:
            return None
        issues = connection.execute(
            "SELECT code, artifact_id, message FROM validation_issues "
            "WHERE run_id = %s ORDER BY position",
            (run_id,),
        ).fetchall()

        def values(table: str, column: str) -> list[str]:
            return [
                item[column]
                for item in connection.execute(
                    f"SELECT {column} FROM {table} WHERE run_id = %s "
                    "ORDER BY position",
                    (run_id,),
                ).fetchall()
            ]

        return ValidationReport(
            valid=row["valid"],
            issues=[
                ValidationIssue(
                    code=item["code"],
                    artifact_id=item["artifact_id"],
                    message=item["message"],
                )
                for item in issues
            ],
            uncovered_requirement_ids=values(
                "validation_uncovered_requirements", "requirement_id"
            ),
            orphan_scenario_ids=values(
                "validation_orphan_scenarios", "scenario_id"
            ),
            orphan_test_case_ids=values(
                "validation_orphan_test_cases", "test_case_id"
            ),
        )

    @staticmethod
    def _load_bundle(
        connection: psycopg.Connection, run_id: str
    ) -> ArtifactBundle | None:
        stored = connection.execute(
            "SELECT 1 FROM artifact_bundles WHERE run_id = %s", (run_id,)
        ).fetchone()
        requirement_rows = connection.execute(
            "SELECT position, requirement_id, title, description, requirement_type, "
            "module, priority FROM requirements WHERE run_id = %s ORDER BY position",
            (run_id,),
        ).fetchall()
        scenario_rows = connection.execute(
            "SELECT position, scenario_id, title, objective, scenario_type "
            "FROM scenarios WHERE run_id = %s ORDER BY position",
            (run_id,),
        ).fetchall()
        test_case_rows = connection.execute(
            "SELECT position, test_case_id, scenario_id, title, priority "
            "FROM test_cases WHERE run_id = %s ORDER BY position",
            (run_id,),
        ).fetchall()
        roots = (requirement_rows, scenario_rows, test_case_rows)
        if stored is None:
            if any(roots):
                raise StorageError("Stored artifact graph is incomplete.")
            return None

        requirement_ambiguities = _group(
            connection.execute(
                "SELECT requirement_position, value FROM requirement_ambiguities "
                "WHERE run_id = %s ORDER BY requirement_position, position",
                (run_id,),
            ).fetchall(),
            "requirement_position",
        )
        requirement_dependencies = _group(
            connection.execute(
                "SELECT requirement_position, dependency_id "
                "FROM requirement_dependencies WHERE run_id = %s "
                "ORDER BY requirement_position, position",
                (run_id,),
            ).fetchall(),
            "requirement_position",
        )
        requirement_sources = _group(
            connection.execute(
                "SELECT requirement_position, chunk_id, page_number, section, excerpt "
                "FROM requirement_sources WHERE run_id = %s "
                "ORDER BY requirement_position, position",
                (run_id,),
            ).fetchall(),
            "requirement_position",
        )
        scenario_preconditions = _group(
            connection.execute(
                "SELECT scenario_position, value FROM scenario_preconditions "
                "WHERE run_id = %s ORDER BY scenario_position, position",
                (run_id,),
            ).fetchall(),
            "scenario_position",
        )
        scenario_requirements = _group(
            connection.execute(
                "SELECT scenario_position, requirement_id FROM scenario_requirements "
                "WHERE run_id = %s ORDER BY scenario_position, position",
                (run_id,),
            ).fetchall(),
            "scenario_position",
        )
        scenario_sources = _group(
            connection.execute(
                "SELECT scenario_position, chunk_id, page_number, section, excerpt "
                "FROM scenario_sources WHERE run_id = %s "
                "ORDER BY scenario_position, position",
                (run_id,),
            ).fetchall(),
            "scenario_position",
        )
        test_case_preconditions = _group(
            connection.execute(
                "SELECT test_case_position, value FROM test_case_preconditions "
                "WHERE run_id = %s ORDER BY test_case_position, position",
                (run_id,),
            ).fetchall(),
            "test_case_position",
        )
        test_case_requirements = _group(
            connection.execute(
                "SELECT test_case_position, requirement_id "
                "FROM test_case_requirements WHERE run_id = %s "
                "ORDER BY test_case_position, position",
                (run_id,),
            ).fetchall(),
            "test_case_position",
        )
        test_case_data = _group(
            connection.execute(
                "SELECT test_case_position, key, value FROM test_case_data "
                "WHERE run_id = %s ORDER BY test_case_position, key",
                (run_id,),
            ).fetchall(),
            "test_case_position",
        )
        test_steps = _group(
            connection.execute(
                "SELECT test_case_position, step_number, action, expected_result "
                "FROM test_steps WHERE run_id = %s "
                "ORDER BY test_case_position, position",
                (run_id,),
            ).fetchall(),
            "test_case_position",
        )
        test_case_sources = _group(
            connection.execute(
                "SELECT test_case_position, chunk_id, page_number, section, excerpt "
                "FROM test_case_sources WHERE run_id = %s "
                "ORDER BY test_case_position, position",
                (run_id,),
            ).fetchall(),
            "test_case_position",
        )

        requirements = [
            Requirement(
                requirement_id=row["requirement_id"],
                title=row["title"],
                description=row["description"],
                requirement_type=row["requirement_type"],
                module=row["module"],
                priority=row["priority"],
                ambiguities=[
                    item["value"]
                    for item in requirement_ambiguities.get(row["position"], [])
                ],
                dependency_ids=[
                    item["dependency_id"]
                    for item in requirement_dependencies.get(row["position"], [])
                ],
                source_references=[
                    _source(item)
                    for item in requirement_sources.get(row["position"], [])
                ],
            )
            for row in requirement_rows
        ]
        scenarios = [
            Scenario(
                scenario_id=row["scenario_id"],
                title=row["title"],
                objective=row["objective"],
                scenario_type=row["scenario_type"],
                preconditions=[
                    item["value"]
                    for item in scenario_preconditions.get(row["position"], [])
                ],
                requirement_ids=[
                    item["requirement_id"]
                    for item in scenario_requirements.get(row["position"], [])
                ],
                source_references=[
                    _source(item)
                    for item in scenario_sources.get(row["position"], [])
                ],
            )
            for row in scenario_rows
        ]
        test_cases = [
            TestCase(
                test_case_id=row["test_case_id"],
                scenario_id=row["scenario_id"],
                requirement_ids=[
                    item["requirement_id"]
                    for item in test_case_requirements.get(row["position"], [])
                ],
                title=row["title"],
                priority=row["priority"],
                preconditions=[
                    item["value"]
                    for item in test_case_preconditions.get(row["position"], [])
                ],
                test_data={
                    item["key"]: item["value"]
                    for item in test_case_data.get(row["position"], [])
                },
                steps=[
                    TestStep(
                        step_number=item["step_number"],
                        action=item["action"],
                        expected_result=item["expected_result"],
                    )
                    for item in test_steps.get(row["position"], [])
                ],
                source_references=[
                    _source(item)
                    for item in test_case_sources.get(row["position"], [])
                ],
            )
            for row in test_case_rows
        ]
        return ArtifactBundle(
            requirements=requirements,
            scenarios=scenarios,
            test_cases=test_cases,
        )

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
