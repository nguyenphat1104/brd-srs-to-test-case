from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Any, Iterator

from .models import (
    ComparisonManifest,
    Condition,
    ConditionManifest,
    DocumentChunk,
    RunStatus,
)


class StorageError(RuntimeError):
    pass


class ImmutableArtifactError(StorageError):
    pass


_MUTATION_LOCK = threading.RLock()


def _component(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Path must be a single normal component.")
    path = Path(value)
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or path.is_absolute()
        or len(path.parts) != 1
    ):
        raise ValueError("Path must be a single normal component.")
    return value


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_text(path: Path, text: str) -> None:
    if not path.parent.is_dir():
        raise StorageError("Artifact parent directory does not exist.")
    if path.is_symlink():
        raise StorageError("Artifact path cannot be a symlink.")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(
            value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
    )


class RunStore:
    def __init__(self, root: str | Path = "runs") -> None:
        self.root = Path(root)

    def comparison_dir(self, comparison_id: str) -> Path:
        return self.root / _component(comparison_id)

    def condition_dir(self, comparison_id: str, condition: Condition) -> Path:
        return self.comparison_dir(comparison_id) / "conditions" / condition.value

    @contextmanager
    def _mutation(self, *, create_root: bool = False) -> Iterator[None]:
        with _MUTATION_LOCK:
            if create_root:
                self.root.mkdir(parents=True, exist_ok=True)
            elif not self.root.is_dir():
                raise StorageError("Run-store root directory does not exist.")
            lock_path = self.root / ".runstore.lock"
            if lock_path.is_symlink():
                raise StorageError("Run-store lock cannot be a symlink.")
            with lock_path.open("a+") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                try:
                    # ponytail: global serialization; use per-run locks if throughput matters.
                    yield
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _assert_no_symlinks(self, path: Path) -> None:
        try:
            relative = path.relative_to(self.root)
        except ValueError as error:
            raise StorageError("Run path must remain under the configured root.") from error
        current = self.root
        for component in relative.parts:
            current /= component
            if current.is_symlink():
                raise StorageError("Run path cannot contain a symlink.")

    def _require_comparison(self, comparison_id: str) -> tuple[Path, Path]:
        directory = self.comparison_dir(comparison_id)
        self._assert_no_symlinks(directory)
        if not directory.is_dir():
            raise StorageError("Comparison directory does not exist.")
        manifest = directory / "manifest.json"
        self._assert_no_symlinks(manifest)
        if not manifest.is_file():
            raise StorageError("Comparison manifest does not exist.")
        return directory, manifest

    def _require_condition(
        self, comparison_id: str, condition: Condition
    ) -> tuple[Path, Path]:
        self._require_comparison(comparison_id)
        directory = self.condition_dir(comparison_id, condition)
        self._assert_no_symlinks(directory)
        if not directory.is_dir():
            raise StorageError("Condition directory does not exist.")
        manifest = directory / "manifest.json"
        self._assert_no_symlinks(manifest)
        if not manifest.is_file():
            raise StorageError("Condition manifest does not exist.")
        return directory, manifest

    def _comparison_manifest(self, comparison_id: str) -> ComparisonManifest:
        _, path = self._require_comparison(comparison_id)
        try:
            return ComparisonManifest.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as error:
            raise StorageError("Comparison manifest is invalid.") from error

    def _condition_manifest(
        self, comparison_id: str, condition: Condition
    ) -> ConditionManifest:
        _, path = self._require_condition(comparison_id, condition)
        try:
            return ConditionManifest.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as error:
            raise StorageError("Condition manifest is invalid.") from error

    def _validated_comparison(
        self, manifest: ComparisonManifest
    ) -> ComparisonManifest:
        try:
            return ComparisonManifest.model_validate(manifest.model_dump(mode="json"))
        except ValueError as error:
            raise StorageError("Comparison manifest is invalid.") from error

    def _validated_condition(self, manifest: ConditionManifest) -> ConditionManifest:
        try:
            return ConditionManifest.model_validate(manifest.model_dump(mode="json"))
        except ValueError as error:
            raise StorageError("Condition manifest is invalid.") from error

    def _cleanup_created(self, directory: Path) -> None:
        self._assert_no_symlinks(directory)
        shutil.rmtree(directory)

    def create_comparison(
        self, manifest: ComparisonManifest, chunks: list[DocumentChunk]
    ) -> Path:
        validated = self._validated_comparison(manifest)
        if validated.completed_at is not None:
            raise ImmutableArtifactError("Comparisons must start unfinished.")
        directory = self.comparison_dir(validated.comparison_id)
        with self._mutation(create_root=True):
            self._assert_no_symlinks(directory)
            try:
                directory.mkdir(parents=True, exist_ok=False)
            except FileExistsError as error:
                raise ImmutableArtifactError("Comparison already exists.") from error
            try:
                _atomic_json(directory / "manifest.json", validated.model_dump(mode="json"))
                _atomic_json(
                    directory / "chunks.json",
                    [chunk.model_dump(mode="json") for chunk in chunks],
                )
            except Exception:
                self._cleanup_created(directory)
                raise
        return directory

    def update_comparison(self, manifest: ComparisonManifest) -> Path:
        path = self.comparison_dir(manifest.comparison_id) / "manifest.json"
        with self._mutation():
            current = self._comparison_manifest(manifest.comparison_id)
            if current.completed_at is not None:
                raise ImmutableArtifactError("Completed comparisons cannot be updated.")
            updated = self._validated_comparison(manifest)
            if any(
                getattr(updated, field) != getattr(current, field)
                for field in (
                    "comparison_id",
                    "document_hash",
                    "provider",
                    "model",
                    "temperature",
                    "token_ceiling",
                    "condition_order",
                    "prompt_version",
                    "schema_version",
                    "started_at",
                )
            ):
                raise ImmutableArtifactError("Comparison configuration cannot be updated.")
            if updated.completed_at is None:
                raise ImmutableArtifactError("Comparison updates must set completed_at.")
            self._assert_no_symlinks(path)
            _atomic_json(path, updated.model_dump(mode="json"))
        return path

    def start_condition(
        self, comparison_id: str, manifest: ConditionManifest
    ) -> Path:
        validated = self._validated_condition(manifest)
        if validated.status is not RunStatus.RUNNING:
            raise ImmutableArtifactError("Conditions must start running.")
        directory = self.condition_dir(comparison_id, validated.condition)
        with self._mutation():
            if self._comparison_manifest(comparison_id).completed_at is not None:
                raise ImmutableArtifactError("Completed comparisons cannot start conditions.")
            self._assert_no_symlinks(directory)
            try:
                directory.mkdir(parents=True, exist_ok=False)
            except FileExistsError as error:
                raise ImmutableArtifactError("Condition already exists.") from error
            try:
                _atomic_json(directory / "manifest.json", validated.model_dump(mode="json"))
            except Exception:
                self._cleanup_created(directory)
                raise
        return directory

    def update_condition(self, comparison_id: str, manifest: ConditionManifest) -> Path:
        path = self.condition_dir(comparison_id, manifest.condition) / "manifest.json"
        with self._mutation():
            if self._comparison_manifest(comparison_id).completed_at is not None:
                raise ImmutableArtifactError("Completed comparisons cannot update conditions.")
            current = self._condition_manifest(comparison_id, manifest.condition)
            if current.status is not RunStatus.RUNNING:
                raise ImmutableArtifactError("Terminal conditions cannot be updated.")
            updated = self._validated_condition(manifest)
            if any(
                getattr(updated, field) != getattr(current, field)
                for field in (
                    "condition",
                    "provider",
                    "model",
                    "temperature",
                    "token_ceiling",
                    "started_at",
                )
            ):
                raise ImmutableArtifactError("Condition configuration cannot be updated.")
            if updated.status not in {RunStatus.COMPLETED, RunStatus.FAILED}:
                raise ImmutableArtifactError("Condition updates must be terminal.")
            self._assert_no_symlinks(path)
            _atomic_json(path, updated.model_dump(mode="json"))
        return path

    def write_artifact(
        self, comparison_id: str, condition: Condition, filename: str, value: Any
    ) -> Path:
        filename = _component(filename)
        path = self.condition_dir(comparison_id, condition) / filename
        with self._mutation():
            if self._comparison_manifest(comparison_id).completed_at is not None:
                raise ImmutableArtifactError("Completed comparisons cannot write artifacts.")
            if self._condition_manifest(comparison_id, condition).status is not RunStatus.RUNNING:
                raise ImmutableArtifactError("Terminal conditions cannot write artifacts.")
            self._assert_no_symlinks(path)
            if path.exists():
                raise ImmutableArtifactError("Artifact already exists.")
            _atomic_json(path, value)
        return path

    def write_comparison_artifact(
        self, comparison_id: str, filename: str, value: Any
    ) -> Path:
        filename = _component(filename)
        if filename in {
            "manifest.json",
            "chunks.json",
            "conditions",
            ".runstore.lock",
        } or filename.startswith(".tmp-"):
            raise StorageError("Artifact filename is reserved by the run store.")
        path = self.comparison_dir(comparison_id) / filename
        with self._mutation():
            directory, _ = self._require_comparison(comparison_id)
            if self._comparison_manifest(comparison_id).completed_at is not None:
                raise ImmutableArtifactError(
                    "Completed comparisons cannot write artifacts."
                )
            chunks = directory / "chunks.json"
            self._assert_no_symlinks(chunks)
            if not chunks.is_file():
                raise StorageError("Comparison chunks do not exist.")
            self._assert_no_symlinks(path)
            if path.exists():
                raise ImmutableArtifactError("Artifact already exists.")
            _atomic_json(path, value)
        return path

    def append_event(
        self, comparison_id: str, condition: Condition, event: Any
    ) -> Path:
        path = self.condition_dir(comparison_id, condition) / "events.jsonl"
        event_text = json.dumps(event, allow_nan=False, ensure_ascii=False, sort_keys=True)
        with self._mutation():
            if self._comparison_manifest(comparison_id).completed_at is not None:
                raise ImmutableArtifactError("Completed comparisons cannot append events.")
            if self._condition_manifest(comparison_id, condition).status is not RunStatus.RUNNING:
                raise ImmutableArtifactError("Terminal conditions cannot append events.")
            self._assert_no_symlinks(path)
            existing = path.read_text(encoding="utf-8") if path.exists() else ""
            if existing and not existing.endswith("\n"):
                existing += "\n"
            _atomic_text(path, existing + event_text + "\n")
        return path
