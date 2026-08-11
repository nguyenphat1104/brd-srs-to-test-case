from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlsplit

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


@dataclass(frozen=True)
class ProviderSettings:
    provider: str
    model: str
    token_ceiling: int
    api_key: str = field(default="", repr=False)
    base_url: str = "http://localhost:11434"

    def validate(self) -> None:
        if self.provider not in {"gemini", "ollama"}:
            raise ValueError("Provider must be gemini or ollama.")
        if not self.model.strip():
            raise ValueError("Model must not be blank.")
        if (
            isinstance(self.token_ceiling, bool)
            or not isinstance(self.token_ceiling, int)
            or self.token_ceiling < 1
        ):
            raise ValueError("Token ceiling must be positive.")
        if self.provider == "gemini" and not self.api_key.strip():
            raise ValueError("Gemini API key is required.")
        if self.provider == "ollama":
            parsed = urlsplit(self.base_url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or any(character.isspace() for character in parsed.netloc)
            ):
                raise ValueError("Ollama base URL must be an HTTP(S) URL.")


@dataclass(frozen=True)
class ConditionResult:
    manifest: ConditionManifest
    bundle: ArtifactBundle | None
    validation: ValidationReport | None
    rtm: list[RTMRow]
    metrics: RunMetrics

    @property
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
    return f"{timestamp}-{document_hash[:12]}"


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
        latency_seconds=latency_seconds,
        retries=context.retries if context else 0,
        schema_repairs=context.schema_repairs if context else 0,
        semantic_revisions=context.semantic_revisions if context else 0,
        budget_exhausted=budget_exhausted,
    )


def _failure_category(error: Exception) -> FailureCategory:
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
    return FailureCategory.CONFIGURATION


PIPELINES = {
    Condition.SINGLE_PROMPT: run_single_prompt,
    Condition.STAGED_SINGLE_AGENT: run_staged_single_agent,
    Condition.CENTRALIZED_MULTI_AGENT: run_centralized_multi_agent,
}


def _write_bundle(
    store: RunStore,
    comparison_id: str,
    condition: Condition,
    bundle: ArtifactBundle,
    validation: ValidationReport,
    rtm: list[RTMRow],
    metrics: RunMetrics,
) -> None:
    values = {
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
    for filename, value in values.items():
        store.write_artifact(comparison_id, condition, filename, value)


def _safe_message(error: Exception, settings: ProviderSettings) -> str:
    message = str(error)
    return message.replace(settings.api_key, "[REDACTED]") if settings.api_key else message


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

    progress(None, "Parsing PDF")
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
        return ComparisonResult(
            manifest=manifest,
            conditions={},
            failure_category=FailureCategory.PARSING,
            failure_message=message,
        )

    store.create_comparison(manifest, chunks)
    results: dict[Condition, ConditionResult] = {}
    for condition in manifest.condition_order:
        progress(condition, "Starting")
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
            provider = provider_factory(condition, ledger)
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
            metrics = _empty_metrics(
                context,
                latency_seconds=time.perf_counter() - started,
                budget_exhausted=category is FailureCategory.BUDGET_EXHAUSTION,
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

        if bundle is None:
            store.write_artifact(
                manifest.comparison_id,
                condition,
                "metrics.json",
                metrics.model_dump(mode="json"),
            )
        else:
            _write_bundle(
                store,
                manifest.comparison_id,
                condition,
                bundle,
                validation,
                rtm,
                metrics,
            )
        store.append_event(
            manifest.comparison_id,
            condition,
            {
                "timestamp": _now(),
                "stage": "finished",
                "status": condition_manifest.status.value,
                "input_tokens": metrics.input_tokens,
                "output_tokens": metrics.output_tokens,
                "retries": metrics.retries,
                "schema_repairs": metrics.schema_repairs,
                "semantic_revisions": metrics.semantic_revisions,
                "charged_tokens": ledger.used,
            },
        )
        store.update_condition(manifest.comparison_id, condition_manifest)
        results[condition] = ConditionResult(
            manifest=condition_manifest,
            bundle=bundle,
            validation=validation,
            rtm=rtm,
            metrics=metrics,
        )
        progress(condition, condition_manifest.status.value.title())

    manifest = manifest.model_copy(
        update={"completed_at": datetime.fromisoformat(_now())}
    )
    store.update_comparison(manifest)
    progress(None, "Complete")
    return ComparisonResult(manifest=manifest, conditions=results)
