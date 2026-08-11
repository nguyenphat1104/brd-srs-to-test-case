from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import parse_qsl, urlsplit
from uuid import uuid4

import google.genai as genai

from .documents import DocumentError, parse_pdf
from .models import (
    ArtifactBundle,
    ComparisonManifest,
    Condition,
    ConditionManifest,
    FailureCategory,
    RTMRow,
    RunMetrics,
    RunStatus,
    ValidationReport,
)
from .pipelines import (
    PROMPT_VERSION,
    PipelineContext,
    PipelineOutputError,
    run_centralized_multi_agent,
    run_single_prompt,
    run_staged_single_agent,
)
from .providers import (
    BudgetExceeded,
    BudgetLedger,
    GeminiProvider,
    OllamaProvider,
    ProviderError,
    StructuredOutputError,
    StructuredProvider,
)
from .storage import RunStore
from .validation import build_rtm, compute_metrics, validate_bundle


SCHEMA_VERSION = "research-core-v1"
ProviderFactory = Callable[[Condition, BudgetLedger], StructuredProvider]
Progress = Callable[[Condition | None, str], None]


class ConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderSettings:
    provider: str
    model: str
    token_ceiling: int
    api_key: str = field(default="", repr=False)
    base_url: str = field(default="http://localhost:11434", repr=False)

    def validate(self) -> None:
        if not isinstance(self.provider, str) or self.provider not in {
            "gemini",
            "ollama",
        }:
            raise ValueError("Provider must be gemini or ollama.")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Model must not be blank.")
        if (
            isinstance(self.token_ceiling, bool)
            or not isinstance(self.token_ceiling, int)
            or self.token_ceiling < 1
        ):
            raise ValueError("Token ceiling must be positive.")
        if not isinstance(self.api_key, str):
            raise ValueError("API key must be a string.")
        if self.provider == "gemini" and not self.api_key.strip():
            raise ValueError("Gemini API key is required.")
        if self.provider == "ollama":
            if not isinstance(self.base_url, str) or any(
                character.isspace() for character in self.base_url
            ):
                raise ValueError("Ollama base URL must be an HTTP(S) URL.")
            try:
                parsed = urlsplit(self.base_url)
                hostname = parsed.hostname
                parsed.port
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Ollama base URL must be an HTTP(S) URL."
                ) from error
            if (
                parsed.scheme not in {"http", "https"}
                or not hostname
                or parsed.username is not None
                or parsed.password is not None
                or "?" in self.base_url
                or "#" in self.base_url
            ):
                raise ValueError(
                    "Ollama base URL cannot contain credentials, query, or fragment."
                )


@dataclass(frozen=True)
class ConditionResult:
    manifest: ConditionManifest
    bundle: ArtifactBundle | None
    validation: ValidationReport | None
    rtm: list[RTMRow]
    metrics: RunMetrics

    def download_bundle(self) -> dict:
        bundle = self.bundle
        return {
            "manifest": self.manifest.model_dump(mode="json"),
            "requirements": (
                [item.model_dump(mode="json") for item in bundle.requirements]
                if bundle
                else []
            ),
            "scenarios": (
                [item.model_dump(mode="json") for item in bundle.scenarios]
                if bundle
                else []
            ),
            "test_cases": (
                [item.model_dump(mode="json") for item in bundle.test_cases]
                if bundle
                else []
            ),
            "validation": (
                self.validation.model_dump(mode="json") if self.validation else None
            ),
            "rtm": [item.model_dump(mode="json") for item in self.rtm],
            "metrics": self.metrics.model_dump(mode="json"),
        }


@dataclass(frozen=True)
class ComparisonResult:
    manifest: ComparisonManifest
    conditions: dict[Condition, ConditionResult]
    failure_category: FailureCategory | None = None
    failure_message: str | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _comparison_id(document_hash: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{document_hash[:12]}-{uuid4().hex[:8]}"


def _make_provider(
    settings: ProviderSettings, ledger: BudgetLedger
) -> StructuredProvider:
    if settings.provider == "gemini":
        return GeminiProvider(
            genai.Client(api_key=settings.api_key), settings.model, ledger
        )
    return OllamaProvider(settings.base_url, settings.model, ledger)


def _empty_metrics(
    context: PipelineContext | None,
    *,
    latency_seconds: float,
    budget_exhausted: bool,
    charged_tokens: int,
) -> RunMetrics:
    return RunMetrics(
        completion=False,
        schema_valid=False,
        citation_coverage=0,
        requirement_scenario_coverage=0,
        requirement_test_case_coverage=0,
        positive_scenario_coverage=0,
        non_positive_scenario_coverage=0,
        rtm_completeness=0,
        orphan_rate=0,
        invalid_reference_rate=0,
        duplicate_test_case_rate=0,
        requirement_count=0,
        scenario_count=0,
        test_case_count=0,
        input_tokens=context.input_tokens if context else 0,
        output_tokens=context.output_tokens if context else 0,
        charged_tokens=charged_tokens,
        latency_seconds=latency_seconds,
        retries=context.retries if context else 0,
        schema_repairs=context.schema_repairs if context else 0,
        semantic_revisions=context.semantic_revisions if context else 0,
        budget_exhausted=budget_exhausted,
    )


def _failure_category(error: Exception) -> FailureCategory | None:
    if isinstance(error, BudgetExceeded):
        return FailureCategory.BUDGET_EXHAUSTION
    if isinstance(error, StructuredOutputError):
        return FailureCategory.SCHEMA_FAILURE
    if isinstance(error, ProviderError):
        return (
            FailureCategory.TRANSPORT_EXHAUSTION
            if error.retryable
            else FailureCategory.PROVIDER_REJECTION
        )
    if isinstance(error, TimeoutError):
        return FailureCategory.TIMEOUT
    if isinstance(error, PipelineOutputError):
        return FailureCategory.SEMANTIC_VALIDATION
    if isinstance(error, ConfigurationError):
        return FailureCategory.CONFIGURATION
    return None


PIPELINES = {
    Condition.SINGLE_PROMPT: run_single_prompt,
    Condition.STAGED_SINGLE_AGENT: run_staged_single_agent,
    Condition.CENTRALIZED_MULTI_AGENT: run_centralized_multi_agent,
}


def _bundle_artifacts(
    bundle: ArtifactBundle,
    validation: ValidationReport,
    rtm: list[RTMRow],
    metrics: RunMetrics,
) -> dict[str, object]:
    return {
        "requirements.json": [
            item.model_dump(mode="json") for item in bundle.requirements
        ],
        "scenarios.json": [item.model_dump(mode="json") for item in bundle.scenarios],
        "test_cases.json": [
            item.model_dump(mode="json") for item in bundle.test_cases
        ],
        "validation.json": validation.model_dump(mode="json"),
        "rtm.json": [item.model_dump(mode="json") for item in rtm],
        "metrics.json": metrics.model_dump(mode="json"),
    }


def _safe_message(error: Exception, settings: ProviderSettings) -> str:
    message = str(error)
    secrets = [settings.api_key]
    try:
        parsed = urlsplit(settings.base_url)
        secrets.extend(
            [
                parsed.username or "",
                parsed.password or "",
                *(value for _key, value in parse_qsl(parsed.query)),
                parsed.fragment,
            ]
        )
    except (TypeError, ValueError):
        pass
    for secret in sorted(filter(None, secrets), key=len, reverse=True):
        message = message.replace(secret, "[REDACTED]")
    return re.sub(
        r"(?i)\b(api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        message,
    )


def _notify(progress: Progress, condition: Condition | None, message: str) -> None:
    try:
        progress(condition, message)
    except Exception:
        pass


def run_comparison(
    pdf_bytes: bytes,
    settings: ProviderSettings,
    store: RunStore | None = None,
    provider_factory: ProviderFactory | None = None,
    progress: Progress | None = None,
) -> ComparisonResult:
    settings.validate()
    store = store or RunStore()
    provider_factory = provider_factory or (
        lambda _condition, ledger: _make_provider(settings, ledger)
    )
    progress = progress or (lambda _condition, _message: None)
    document_hash = hashlib.sha256(pdf_bytes).hexdigest()
    manifest = ComparisonManifest(
        comparison_id=_comparison_id(document_hash),
        document_hash=document_hash,
        provider=settings.provider,
        model=settings.model,
        temperature=0.0,
        token_ceiling=settings.token_ceiling,
        condition_order=list(Condition),
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        started_at=_now(),
    )

    _notify(progress, None, "Parsing PDF")
    try:
        chunks = parse_pdf(pdf_bytes)
    except DocumentError as error:
        message = _safe_message(error, settings)
        store.create_comparison(manifest, [])
        store.write_comparison_artifact(
            manifest.comparison_id,
            "failure.json",
            {"category": FailureCategory.PARSING.value, "message": message},
        )
        manifest = manifest.model_copy(
            update={"completed_at": datetime.fromisoformat(_now())}
        )
        store.update_comparison(manifest)
        _notify(progress, None, "Failed")
        return ComparisonResult(
            manifest=manifest,
            conditions={},
            failure_category=FailureCategory.PARSING,
            failure_message=message,
        )

    store.create_comparison(manifest, chunks)
    results: dict[Condition, ConditionResult] = {}
    for condition in manifest.condition_order:
        _notify(progress, condition, "Starting")
        condition_manifest = ConditionManifest(
            condition=condition,
            status=RunStatus.RUNNING,
            provider=settings.provider,
            model=settings.model,
            temperature=0.0,
            token_ceiling=settings.token_ceiling,
            started_at=_now(),
        )
        store.start_condition(manifest.comparison_id, condition_manifest)
        store.append_event(
            manifest.comparison_id,
            condition,
            {"timestamp": _now(), "stage": "started"},
        )

        ledger = BudgetLedger(settings.token_ceiling)
        context: PipelineContext | None = None
        bundle: ArtifactBundle | None = None
        validation: ValidationReport | None = None
        rtm: list[RTMRow] = []
        started = time.perf_counter()
        try:
            try:
                provider = provider_factory(condition, ledger)
            except Exception as error:
                raise ConfigurationError(str(error)) from error
            if getattr(provider, "ledger", None) is not ledger:
                raise ConfigurationError(
                    "Provider must use the condition budget ledger."
                )
            if getattr(provider, "model", None) != settings.model:
                raise ConfigurationError(
                    "Provider model must match the comparison model."
                )
            context = PipelineContext(provider=provider)
            bundle = PIPELINES[condition](context, chunks)
            validation = validate_bundle(bundle, chunks)
            rtm = build_rtm(bundle)
            latency = time.perf_counter() - started
            metrics = compute_metrics(
                bundle,
                validation,
                input_tokens=context.input_tokens,
                output_tokens=context.output_tokens,
                charged_tokens=context.charged_tokens,
                latency_seconds=latency,
                retries=context.retries,
                schema_repairs=context.schema_repairs,
                semantic_revisions=context.semantic_revisions,
                budget_exhausted=False,
            )
            if validation.valid:
                condition_manifest = condition_manifest.model_copy(
                    update={
                        "status": RunStatus.COMPLETED,
                        "completed_at": datetime.fromisoformat(_now()),
                    }
                )
            else:
                condition_manifest = condition_manifest.model_copy(
                    update={
                        "status": RunStatus.FAILED,
                        "completed_at": datetime.fromisoformat(_now()),
                        "failure_category": FailureCategory.SEMANTIC_VALIDATION,
                        "failure_message": (
                            f"{len(validation.issues)} deterministic validation issues."
                        ),
                    }
                )
        except Exception as error:
            category = _failure_category(error)
            if category is None:
                raise
            metrics = _empty_metrics(
                context,
                latency_seconds=time.perf_counter() - started,
                budget_exhausted=category is FailureCategory.BUDGET_EXHAUSTION,
                charged_tokens=ledger.used,
            )
            bundle = None
            validation = None
            rtm = []
            condition_manifest = condition_manifest.model_copy(
                update={
                    "status": RunStatus.FAILED,
                    "completed_at": datetime.fromisoformat(_now()),
                    "failure_category": category,
                    "failure_message": _safe_message(error, settings),
                }
            )

        artifacts = (
            {"metrics.json": metrics.model_dump(mode="json")}
            if bundle is None
            else _bundle_artifacts(bundle, validation, rtm, metrics)
        )
        finished_event = {
            "timestamp": _now(),
            "stage": "finished",
            "status": condition_manifest.status.value,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "retries": metrics.retries,
            "schema_repairs": metrics.schema_repairs,
            "semantic_revisions": metrics.semantic_revisions,
            "charged_tokens": metrics.charged_tokens,
        }
        store.finalize_condition(
            manifest.comparison_id,
            condition_manifest,
            artifacts,
            finished_event,
        )
        results[condition] = ConditionResult(
            manifest=condition_manifest,
            bundle=bundle,
            validation=validation,
            rtm=rtm,
            metrics=metrics,
        )
        _notify(progress, condition, condition_manifest.status.value.title())

    manifest = manifest.model_copy(
        update={"completed_at": datetime.fromisoformat(_now())}
    )
    store.update_comparison(manifest)
    _notify(progress, None, "Complete")
    return ComparisonResult(manifest=manifest, conditions=results)
