import pytest

from brd_srs_testgen.browser_settings import (
    AppSettings,
    BrowserSettingsResult,
    parse_settings,
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
