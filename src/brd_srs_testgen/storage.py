from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .models import ComparisonManifest, Condition, ConditionManifest, DocumentChunk


class ImmutableArtifactError(RuntimeError):
    pass


def _component(value: str) -> str:
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


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


class RunStore:
    def __init__(self, root: str | Path = "runs") -> None:
        self.root = Path(root)

    def comparison_dir(self, comparison_id: str) -> Path:
        return self.root / _component(comparison_id)

    def condition_dir(self, comparison_id: str, condition: Condition) -> Path:
        return self.comparison_dir(comparison_id) / "conditions" / condition.value

    def create_comparison(
        self, manifest: ComparisonManifest, chunks: list[DocumentChunk]
    ) -> Path:
        directory = self.comparison_dir(manifest.comparison_id)
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise ImmutableArtifactError("Comparison already exists.") from error
        _atomic_json(directory / "manifest.json", manifest.model_dump(mode="json"))
        _atomic_json(
            directory / "chunks.json", [chunk.model_dump(mode="json") for chunk in chunks]
        )
        return directory

    def update_comparison(self, manifest: ComparisonManifest) -> Path:
        path = self.comparison_dir(manifest.comparison_id) / "manifest.json"
        _atomic_json(path, manifest.model_dump(mode="json"))
        return path

    def start_condition(
        self, comparison_id: str, manifest: ConditionManifest
    ) -> Path:
        directory = self.condition_dir(comparison_id, manifest.condition)
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise ImmutableArtifactError("Condition already exists.") from error
        _atomic_json(directory / "manifest.json", manifest.model_dump(mode="json"))
        return directory

    def update_condition(self, comparison_id: str, manifest: ConditionManifest) -> Path:
        path = self.condition_dir(comparison_id, manifest.condition) / "manifest.json"
        _atomic_json(path, manifest.model_dump(mode="json"))
        return path

    def write_artifact(
        self, comparison_id: str, condition: Condition, filename: str, value: Any
    ) -> Path:
        path = self.condition_dir(comparison_id, condition) / _component(filename)
        if path.exists():
            raise ImmutableArtifactError("Artifact already exists.")
        _atomic_json(path, value)
        return path

    def append_event(
        self, comparison_id: str, condition: Condition, event: Any
    ) -> Path:
        path = self.condition_dir(comparison_id, condition) / "events.jsonl"
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        if existing and not existing.endswith("\n"):
            existing += "\n"
        _atomic_text(path, existing + json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        return path
