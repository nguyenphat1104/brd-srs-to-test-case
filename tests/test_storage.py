from datetime import UTC, datetime
from concurrent.futures import ThreadPoolExecutor
import json
import os

import pytest

import brd_srs_testgen.storage as storage
from brd_srs_testgen.models import (
    ComparisonManifest,
    Condition,
    ConditionManifest,
    RunStatus,
)
from brd_srs_testgen.storage import ImmutableArtifactError, RunStore, StorageError
from tests.factories import bundle, chunk


def comparison_manifest() -> ComparisonManifest:
    return ComparisonManifest(
        comparison_id="20260811T000000Z-ecac9f035813",
        document_hash="ecac9f0358134f174bcbf0d60ddbc7c25bcb4f812ea8e4c57bfbd8c02edaa274",
        provider="ollama",
        model="gemma4",
        temperature=0.0,
        token_ceiling=1000,
        condition_order=list(Condition),
        prompt_version="1",
        schema_version="1",
        started_at=datetime.now(UTC).isoformat(),
    )


def condition_manifest() -> ConditionManifest:
    return ConditionManifest(
        condition=Condition.SINGLE_PROMPT,
        status=RunStatus.RUNNING,
        provider="ollama",
        model="gemma4",
        temperature=0.0,
        token_ceiling=1000,
        started_at=datetime.now(UTC).isoformat(),
    )


def test_create_comparison_writes_manifest_and_chunks(tmp_path) -> None:
    store = RunStore(tmp_path)
    manifest = comparison_manifest()

    directory = store.create_comparison(manifest, [chunk()])

    assert (directory / "manifest.json").exists()
    assert (directory / "chunks.json").exists()


def test_artifacts_cannot_be_overwritten(tmp_path) -> None:
    store = RunStore(tmp_path)
    manifest = comparison_manifest()
    store.create_comparison(manifest, [chunk()])
    store.start_condition(manifest.comparison_id, condition_manifest())
    store.write_artifact(
        manifest.comparison_id,
        Condition.SINGLE_PROMPT,
        "requirements.json",
        [item.model_dump(mode="json") for item in bundle().requirements],
    )

    with pytest.raises(ImmutableArtifactError):
        store.write_artifact(
            manifest.comparison_id,
            Condition.SINGLE_PROMPT,
            "requirements.json",
            [],
        )


def test_events_are_appended_with_atomic_replacement(tmp_path) -> None:
    store = RunStore(tmp_path)
    manifest = comparison_manifest()
    store.create_comparison(manifest, [chunk()])
    store.start_condition(manifest.comparison_id, condition_manifest())

    store.append_event(
        manifest.comparison_id, Condition.SINGLE_PROMPT, {"stage": "start"}
    )
    store.append_event(
        manifest.comparison_id, Condition.SINGLE_PROMPT, {"stage": "finish"}
    )

    events = (
        tmp_path
        / manifest.comparison_id
        / "conditions"
        / Condition.SINGLE_PROMPT
        / "events.jsonl"
    ).read_text().splitlines()
    assert len(events) == 2


@pytest.mark.parametrize("unsafe_id", ["../escaped", "absolute"])
def test_unsafe_comparison_ids_cannot_escape_root(tmp_path, unsafe_id) -> None:
    root = tmp_path / "runs"
    store = RunStore(root)
    comparison_id = (
        "../escaped" if unsafe_id == "../escaped" else str(tmp_path / "absolute-id")
    )
    manifest = comparison_manifest().model_copy(
        update={"comparison_id": comparison_id}
    )
    escaped = tmp_path / "escaped" if unsafe_id == "../escaped" else tmp_path / "absolute-id"

    with pytest.raises(ValueError):
        store.create_comparison(manifest, [chunk()])

    assert not root.exists()
    assert not escaped.exists()


@pytest.mark.parametrize("unsafe_name", ["../artifact.json", "absolute"])
def test_unsafe_artifact_names_cannot_escape_root(tmp_path, unsafe_name) -> None:
    root = tmp_path / "runs"
    store = RunStore(root)
    manifest = comparison_manifest()
    store.create_comparison(manifest, [chunk()])
    store.start_condition(manifest.comparison_id, condition_manifest())
    filename = (
        "../artifact.json"
        if unsafe_name == "../artifact.json"
        else str(tmp_path / "artifact.json")
    )
    escaped = tmp_path / "artifact.json"

    with pytest.raises(ValueError):
        store.write_artifact(
            manifest.comparison_id, Condition.SINGLE_PROMPT, filename, []
        )

    assert not escaped.exists()


def test_concurrent_artifact_writes_keep_the_original(tmp_path) -> None:
    store = RunStore(tmp_path)
    manifest = comparison_manifest()
    store.create_comparison(manifest, [chunk()])
    store.start_condition(manifest.comparison_id, condition_manifest())

    def write(writer: int) -> str:
        try:
            store.write_artifact(
                manifest.comparison_id,
                Condition.SINGLE_PROMPT,
                "requirements.json",
                {"writer": writer},
            )
        except ImmutableArtifactError:
            return "immutable"
        return "written"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(write, range(2)))

    path = store.condition_dir(manifest.comparison_id, Condition.SINGLE_PROMPT) / "requirements.json"
    assert results.count("written") == 1
    assert results.count("immutable") == 1
    assert json.loads(path.read_text()) in ({"writer": 0}, {"writer": 1})


def test_concurrent_event_appends_keep_every_event(tmp_path) -> None:
    store = RunStore(tmp_path)
    manifest = comparison_manifest()
    store.create_comparison(manifest, [chunk()])
    store.start_condition(manifest.comparison_id, condition_manifest())

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda number: store.append_event(
                    manifest.comparison_id,
                    Condition.SINGLE_PROMPT,
                    {"number": number},
                ),
                range(20),
            )
        )

    path = store.condition_dir(manifest.comparison_id, Condition.SINGLE_PROMPT) / "events.jsonl"
    assert {json.loads(line)["number"] for line in path.read_text().splitlines()} == set(
        range(20)
    )


def test_symlinked_paths_cannot_escape_root(tmp_path) -> None:
    root = tmp_path / "runs"
    store = RunStore(root)
    manifest = comparison_manifest()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "manifest.json"
    sentinel.write_text(json.dumps(manifest.model_dump(mode="json")))
    root.mkdir()
    os.symlink(outside, root / manifest.comparison_id)

    with pytest.raises(StorageError):
        store.update_comparison(manifest)

    assert sentinel.read_text() == json.dumps(manifest.model_dump(mode="json"))


def test_symlinked_condition_cannot_escape_root(tmp_path) -> None:
    root = tmp_path / "runs"
    store = RunStore(root)
    manifest = comparison_manifest()
    store.create_comparison(manifest, [chunk()])
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "manifest.json").write_text(
        json.dumps(condition_manifest().model_dump(mode="json"))
    )
    conditions = root / manifest.comparison_id / "conditions"
    conditions.mkdir()
    os.symlink(outside, conditions / Condition.SINGLE_PROMPT)

    with pytest.raises(StorageError):
        store.write_artifact(
            manifest.comparison_id,
            Condition.SINGLE_PROMPT,
            "artifact.json",
            {},
        )

    assert not (outside / "artifact.json").exists()


def test_terminal_manifests_cannot_be_updated(tmp_path) -> None:
    store = RunStore(tmp_path)
    manifest = comparison_manifest()
    store.create_comparison(manifest, [chunk()])
    condition = condition_manifest()
    store.start_condition(manifest.comparison_id, condition)
    completed_condition = condition.model_copy(
        update={
            "status": RunStatus.COMPLETED,
            "completed_at": datetime.now(UTC),
        }
    )
    store.update_condition(manifest.comparison_id, completed_condition)

    with pytest.raises(ImmutableArtifactError):
        store.update_condition(manifest.comparison_id, completed_condition)

    completed_comparison = manifest.model_copy(
        update={"completed_at": datetime.now(UTC)}
    )
    store.update_comparison(completed_comparison)

    with pytest.raises(ImmutableArtifactError):
        store.update_comparison(completed_comparison)


def test_comparison_configuration_cannot_be_updated(tmp_path) -> None:
    store = RunStore(tmp_path)
    manifest = comparison_manifest()
    store.create_comparison(manifest, [chunk()])
    changed = manifest.model_copy(update={"provider": "different"})

    with pytest.raises(ImmutableArtifactError):
        store.update_comparison(changed)


def test_condition_updates_must_be_valid_terminal_transitions(tmp_path) -> None:
    store = RunStore(tmp_path)
    manifest = comparison_manifest()
    condition = condition_manifest()
    store.create_comparison(manifest, [chunk()])
    store.start_condition(manifest.comparison_id, condition)

    with pytest.raises(ImmutableArtifactError):
        store.update_condition(manifest.comparison_id, condition)
    with pytest.raises(ImmutableArtifactError):
        store.update_condition(
            manifest.comparison_id, condition.model_copy(update={"model": "other"})
        )
    with pytest.raises(StorageError):
        store.update_condition(
            manifest.comparison_id,
            condition.model_copy(update={"status": RunStatus.COMPLETED}),
        )


def test_completed_conditions_reject_artifacts_and_events(tmp_path) -> None:
    store = RunStore(tmp_path)
    manifest = comparison_manifest()
    condition = condition_manifest()
    store.create_comparison(manifest, [chunk()])
    store.start_condition(manifest.comparison_id, condition)
    store.append_event(manifest.comparison_id, Condition.SINGLE_PROMPT, {"stage": "final"})
    completed = condition.model_copy(
        update={
            "status": RunStatus.COMPLETED,
            "completed_at": datetime.now(UTC),
        }
    )
    store.update_condition(manifest.comparison_id, completed)
    events = store.condition_dir(manifest.comparison_id, Condition.SINGLE_PROMPT) / "events.jsonl"
    original = events.read_text()

    with pytest.raises(ImmutableArtifactError):
        store.write_artifact(
            manifest.comparison_id,
            Condition.SINGLE_PROMPT,
            "after-completion.json",
            {},
        )
    with pytest.raises(ImmutableArtifactError):
        store.append_event(
            manifest.comparison_id,
            Condition.SINGLE_PROMPT,
            {"stage": "late"},
        )

    assert not (
        store.condition_dir(manifest.comparison_id, Condition.SINGLE_PROMPT)
        / "after-completion.json"
    ).exists()
    assert events.read_text() == original


def test_completed_comparisons_reject_all_child_mutations(tmp_path) -> None:
    store = RunStore(tmp_path)
    manifest = comparison_manifest()
    condition = condition_manifest()
    store.create_comparison(manifest, [chunk()])
    store.start_condition(manifest.comparison_id, condition)
    store.append_event(manifest.comparison_id, Condition.SINGLE_PROMPT, {"stage": "final"})
    condition_path = store.condition_dir(manifest.comparison_id, Condition.SINGLE_PROMPT)
    original_manifest = (condition_path / "manifest.json").read_text()
    events = condition_path / "events.jsonl"
    original_events = events.read_text()
    store.update_comparison(
        manifest.model_copy(update={"completed_at": datetime.now(UTC)})
    )
    other_condition = condition.model_copy(
        update={"condition": Condition.STAGED_SINGLE_AGENT}
    )
    completed_condition = condition.model_copy(
        update={
            "status": RunStatus.COMPLETED,
            "completed_at": datetime.now(UTC),
        }
    )

    with pytest.raises(ImmutableArtifactError):
        store.start_condition(manifest.comparison_id, other_condition)
    with pytest.raises(ImmutableArtifactError):
        store.update_condition(manifest.comparison_id, completed_condition)
    with pytest.raises(ImmutableArtifactError):
        store.write_artifact(
            manifest.comparison_id,
            Condition.SINGLE_PROMPT,
            "after-completion.json",
            {},
        )
    with pytest.raises(ImmutableArtifactError):
        store.append_event(
            manifest.comparison_id,
            Condition.SINGLE_PROMPT,
            {"stage": "late"},
        )

    assert not store.condition_dir(
        manifest.comparison_id, Condition.STAGED_SINGLE_AGENT
    ).exists()
    assert (condition_path / "manifest.json").read_text() == original_manifest
    assert events.read_text() == original_events
    assert not (condition_path / "after-completion.json").exists()


def test_terminal_manifests_cannot_start_persistence(tmp_path) -> None:
    root = tmp_path / "runs"
    store = RunStore(root)
    comparison = comparison_manifest().model_copy(
        update={"completed_at": datetime.now(UTC)}
    )

    with pytest.raises(ImmutableArtifactError):
        store.create_comparison(comparison, [chunk()])

    assert not root.exists()
    manifest = comparison_manifest()
    store.create_comparison(manifest, [chunk()])
    terminal_condition = condition_manifest().model_copy(
        update={
            "status": RunStatus.COMPLETED,
            "completed_at": datetime.now(UTC),
        }
    )

    with pytest.raises(ImmutableArtifactError):
        store.start_condition(manifest.comparison_id, terminal_condition)

    assert not store.condition_dir(
        manifest.comparison_id, Condition.SINGLE_PROMPT
    ).exists()


@pytest.mark.parametrize("failed_file", ["manifest.json", "chunks.json"])
def test_comparison_creation_failure_is_cleaned_up(tmp_path, monkeypatch, failed_file) -> None:
    store = RunStore(tmp_path)
    manifest = comparison_manifest()
    original = storage._atomic_json

    def fail(path, value) -> None:
        if path.name == failed_file:
            raise OSError("write failed")
        original(path, value)

    monkeypatch.setattr(storage, "_atomic_json", fail)
    with pytest.raises(OSError, match="write failed"):
        store.create_comparison(manifest, [chunk()])

    assert not store.comparison_dir(manifest.comparison_id).exists()
    monkeypatch.setattr(storage, "_atomic_json", original)
    store.create_comparison(manifest, [chunk()])


def test_condition_creation_failure_is_cleaned_up(tmp_path, monkeypatch) -> None:
    store = RunStore(tmp_path)
    manifest = comparison_manifest()
    store.create_comparison(manifest, [chunk()])
    original = storage._atomic_json
    monkeypatch.setattr(storage, "_atomic_json", lambda path, value: (_ for _ in ()).throw(OSError("write failed")))

    with pytest.raises(OSError, match="write failed"):
        store.start_condition(manifest.comparison_id, condition_manifest())

    path = store.condition_dir(manifest.comparison_id, Condition.SINGLE_PROMPT)
    assert not path.exists()
    monkeypatch.setattr(storage, "_atomic_json", original)
    store.start_condition(manifest.comparison_id, condition_manifest())


def test_lifecycle_requires_existing_manifests(tmp_path) -> None:
    store = RunStore(tmp_path)
    manifest = comparison_manifest()

    with pytest.raises(StorageError):
        store.update_comparison(manifest)
    with pytest.raises(StorageError):
        store.write_artifact(
            manifest.comparison_id, Condition.SINGLE_PROMPT, "artifact.json", {}
        )
    with pytest.raises(StorageError):
        store.append_event(manifest.comparison_id, Condition.SINGLE_PROMPT, {})


def test_non_finite_json_is_rejected_without_corruption(tmp_path) -> None:
    store = RunStore(tmp_path)
    manifest = comparison_manifest()
    store.create_comparison(manifest, [chunk()])
    store.start_condition(manifest.comparison_id, condition_manifest())
    store.append_event(manifest.comparison_id, Condition.SINGLE_PROMPT, {"stage": "ok"})
    events = store.condition_dir(manifest.comparison_id, Condition.SINGLE_PROMPT) / "events.jsonl"
    original = events.read_text()

    with pytest.raises(ValueError):
        store.write_artifact(
            manifest.comparison_id,
            Condition.SINGLE_PROMPT,
            "invalid.json",
            {"value": float("nan")},
        )
    with pytest.raises(ValueError):
        store.append_event(
            manifest.comparison_id,
            Condition.SINGLE_PROMPT,
            {"value": float("inf")},
        )

    assert not (
        store.condition_dir(manifest.comparison_id, Condition.SINGLE_PROMPT)
        / "invalid.json"
    ).exists()
    assert events.read_text() == original
