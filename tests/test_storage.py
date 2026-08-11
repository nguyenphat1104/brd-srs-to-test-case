from datetime import UTC, datetime

import pytest

from brd_srs_testgen.models import (
    ComparisonManifest,
    Condition,
    ConditionManifest,
    RunStatus,
)
from brd_srs_testgen.storage import ImmutableArtifactError, RunStore
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
