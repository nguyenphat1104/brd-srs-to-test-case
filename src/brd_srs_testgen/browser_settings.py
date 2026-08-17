from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import streamlit as st
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .runner import ProviderSettings


_INVALID_SETTINGS_WARNING = (
    "Saved browser settings were invalid; app defaults were restored."
)
_STORAGE_KEY = "brd-srs-test-case.settings.v1"


class AppSettings(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    version: Literal[1] = 1
    provider: Literal["gemini", "lm_studio", "ollama"]
    model: str
    api_key: str = Field(default="", repr=False)
    base_url: str = ""
    token_ceiling: int = Field(strict=True, ge=1000)
    analyst_model: str = ""
    test_generator_model: str = ""
    reviewer_model: str = ""

    @field_validator("model")
    @classmethod
    def model_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Model must not be blank.")
        return value

    def model_for(self, agent: str) -> str:
        configured = getattr(self, f"{agent}_model", "")
        return configured.strip() or self.model

    def provider_settings(self) -> ProviderSettings:
        settings = ProviderSettings(
            provider=self.provider,
            model=self.model,
            token_ceiling=self.token_ceiling,
            api_key=self.api_key,
            base_url=self.base_url,
            analyst_model=self.analyst_model,
            test_generator_model=self.test_generator_model,
            reviewer_model=self.reviewer_model,
        )
        settings.validate()
        return settings


def parse_settings(
    payload: object, fallback: AppSettings
) -> tuple[AppSettings, str | None]:
    if payload is None:
        return fallback, None
    try:
        settings = AppSettings.model_validate(payload)
        settings.provider_settings()
    except (ValidationError, ValueError):
        return fallback, _INVALID_SETTINGS_WARNING
    return settings, None


@dataclass(frozen=True)
class BrowserSettingsResult:
    payload: object = None
    error: str | None = None
    loaded: bool = False
    revision: int = -1


brd_srs_browser_settings = st.components.v2.component(
    "brd_srs_browser_settings",
    js="""
export default function(component) {
    const { storageKey, save, revision } = component.data;
    try {
        if (save !== null) {
            localStorage.setItem(storageKey, JSON.stringify(save));
        }
        const saved = localStorage.getItem(storageKey);
        component.setStateValue("payload", saved === null ? null : JSON.parse(saved));
        component.setStateValue("error", null);
    } catch (error) {
        component.setStateValue("payload", null);
        component.setStateValue("error", "Browser settings storage is unavailable.");
    }
    component.setStateValue("loaded", true);
    component.setStateValue("revision", revision);
}
""",
)


def sync_browser_settings(
    *, save: dict[str, object] | None = None, revision: int = 0
) -> BrowserSettingsResult:
    injected = st.session_state.get("_browser_settings_sync")
    if injected is not None:
        return injected(save=save, revision=revision)

    result = brd_srs_browser_settings(
        data={"storageKey": _STORAGE_KEY, "save": save, "revision": revision},
        default={"payload": None, "error": None, "loaded": False, "revision": -1},
        on_payload_change=lambda: None,
        on_error_change=lambda: None,
        on_loaded_change=lambda: None,
        on_revision_change=lambda: None,
        key="browser-settings-storage",
        height=0,
    )
    return BrowserSettingsResult(
        payload=result.payload,
        error=result.error,
        loaded=result.loaded,
        revision=result.revision,
    )
