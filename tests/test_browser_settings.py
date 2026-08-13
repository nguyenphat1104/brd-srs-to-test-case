from types import SimpleNamespace
from typing import get_type_hints

import pytest

import brd_srs_testgen.browser_settings as browser_settings
from brd_srs_testgen.browser_settings import (
    AppSettings,
    BrowserSettingsResult,
    parse_settings,
    sync_browser_settings,
)


def app_settings(**overrides) -> AppSettings:
    values = {
        "provider": "ollama",
        "model": "llama3.2",
        "base_url": "http://localhost:11434",
        "token_ceiling": 100_000,
    }
    values.update(overrides)
    return AppSettings(**values)


def test_ollama_settings_build_provider_settings_without_run_type():
    settings = app_settings()

    provider_settings = settings.provider_settings()

    assert provider_settings.provider == "ollama"
    assert provider_settings.model == "llama3.2"
    assert "run_type" not in settings.model_dump()


def test_gemini_requires_a_credential_when_building_provider_settings():
    settings = app_settings(provider="gemini", base_url="")

    with pytest.raises(ValueError, match="Gemini API key is required"):
        settings.provider_settings()


@pytest.mark.parametrize(
    "payload",
    [
        {"version": 2, "provider": "ollama", "model": "llama3.2", "base_url": "http://localhost:11434", "token_ceiling": 100_000},
        {"version": 1},
        {"version": 1, "provider": "gemini", "model": "gemini-2.5-flash", "token_ceiling": 100_000},
        {"version": 1, "provider": "ollama", "model": " ", "base_url": "http://localhost:11434", "token_ceiling": 100_000},
    ],
)
def test_invalid_saved_settings_restore_defaults_without_echoing_payload(payload):
    fallback = app_settings()

    settings, warning = parse_settings(payload, fallback)

    assert settings == fallback
    assert warning == "Saved browser settings were invalid; app defaults were restored."


def test_no_saved_settings_keeps_defaults_without_warning():
    fallback = app_settings()

    settings, warning = parse_settings(None, fallback)

    assert settings == fallback
    assert warning is None


def test_browser_settings_result_has_safe_adapter_defaults():
    assert BrowserSettingsResult() == BrowserSettingsResult(
        payload=None, error=None, loaded=False, revision=-1
    )


def test_sync_browser_settings_forwards_to_the_injected_apptest_adapter(
    monkeypatch,
):
    expected = BrowserSettingsResult(payload={"provider": "ollama"}, loaded=True)
    calls = []

    def injected(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(
        browser_settings.st, "session_state", {"_browser_settings_sync": injected}
    )

    assert sync_browser_settings(save={"model": "llama3.2"}, revision=3) == expected
    assert calls == [{"save": {"model": "llama3.2"}, "revision": 3}]


def test_sync_browser_settings_mounts_and_maps_the_storage_component(monkeypatch):
    calls = []

    def renderer(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            payload={"provider": "ollama"},
            error=None,
            loaded=True,
            revision=4,
        )

    monkeypatch.setattr(browser_settings.st, "session_state", {})
    monkeypatch.setattr(browser_settings, "brd_srs_browser_settings", renderer)

    result = sync_browser_settings(save={"model": "llama3.2"}, revision=4)

    assert result == BrowserSettingsResult(
        payload={"provider": "ollama"}, error=None, loaded=True, revision=4
    )
    assert calls[0]["data"] == {
        "storageKey": "brd-srs-test-case.settings.v1",
        "save": {"model": "llama3.2"},
        "revision": 4,
    }
    assert calls[0]["default"] == {
        "payload": None,
        "error": None,
        "loaded": False,
        "revision": -1,
    }
    assert calls[0]["key"] == "browser-settings-storage"
    assert calls[0]["height"] == 0
    for name in ("on_payload_change", "on_error_change", "on_loaded_change", "on_revision_change"):
        assert calls[0][name]() is None


def test_browser_settings_adapter_uses_json_object_boundary_types():
    result_hints = get_type_hints(BrowserSettingsResult)
    sync_hints = get_type_hints(sync_browser_settings)

    assert result_hints["payload"] == object
    assert sync_hints["save"] == dict[str, object] | None
