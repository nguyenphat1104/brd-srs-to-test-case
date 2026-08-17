from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import PurePosixPath
from urllib.parse import parse_qsl, urlsplit
from uuid import uuid4

import google.genai as genai

from .documents import DocumentError, canonicalize_source_references, parse_pdf
from .models import (
    ArtifactBundle,
    CoverageRepair,
    FailureCategory,
    ReviewIssue,
    ReviewResult,
    RTMRow,
    RunManifest,
    RunMetrics,
    RunResult,
    RunStatus,
    RunType,
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
    LMStudioProvider,
    OllamaProvider,
    ProviderError,
    StructuredOutputError,
    StructuredProvider,
)
from .storage import RunRepository
from .validation import build_rtm, compute_metrics, validate_bundle


SCHEMA_VERSION = "research-core-v1"
ProviderFactory = Callable[[RunType, BudgetLedger], StructuredProvider]
Progress = Callable[[str], None]


class ConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderSettings:
    provider: str
    model: str
    token_ceiling: int
    api_key: str = field(default="", repr=False)
    base_url: str = field(default="http://localhost:11434", repr=False)
    analyst_model: str = ""
    test_generator_model: str = ""
    reviewer_model: str = ""

    def model_for(self, agent: str) -> str:
        configured = getattr(self, f"{agent}_model", "")
        return configured.strip() or self.model

    def with_model(self, model: str) -> ProviderSettings:
        return replace(self, model=model)

    def validate(self) -> None:
        if not isinstance(self.provider, str) or self.provider not in {
            "gemini",
            "lm_studio",
            "ollama",
        }:
            raise ValueError("Provider must be gemini, LM Studio, or ollama.")
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
        for agent in ("analyst", "test_generator", "reviewer"):
            if not isinstance(getattr(self, f"{agent}_model"), str):
                raise ValueError(
                    f"{agent.replace('_', ' ').title()} model must be a string."
                )
        if self.provider == "gemini" and not self.api_key.strip():
            raise ValueError("Gemini API key is required.")
        if self.provider in {"lm_studio", "ollama"}:
            provider_name = "LM Studio" if self.provider == "lm_studio" else "Ollama"
            if not isinstance(self.base_url, str) or any(
                character.isspace() for character in self.base_url
            ):
                raise ValueError(
                    f"{provider_name} base URL must be an HTTP(S) URL."
                )
            try:
                parsed = urlsplit(self.base_url)
                hostname = parsed.hostname
                parsed.port
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{provider_name} base URL must be an HTTP(S) URL."
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
                    f"{provider_name} base URL cannot contain credentials, "
                    "query, or fragment."
                )


def _now() -> datetime:
    return datetime.now(UTC)


def _run_id(document_hash: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{document_hash[:12]}-{uuid4().hex[:8]}"


def _make_provider(
    settings: ProviderSettings, ledger: BudgetLedger
) -> StructuredProvider:
    if settings.provider == "gemini":
        return GeminiProvider(
            genai.Client(api_key=settings.api_key), settings.model, ledger
        )
    if settings.provider == "lm_studio":
        return LMStudioProvider(
            settings.base_url,
            settings.model,
            ledger,
            api_key=settings.api_key,
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
        if error.timed_out:
            return FailureCategory.TIMEOUT
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
    RunType.SINGLE_PROMPT: run_single_prompt,
    RunType.STAGED_SINGLE_AGENT: run_staged_single_agent,
    RunType.CENTRALIZED_MULTI_AGENT: run_centralized_multi_agent,
}


def _safe_message(error: Exception, settings: ProviderSettings) -> str:
    message = str(error)
    secrets = [settings.api_key, settings.base_url]
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


def _notify(progress: Progress | None, message: str) -> None:
    if progress is None:
        return
    try:
        progress(message)
    except Exception:
        pass


def _repair_coverage(
    context: PipelineContext,
    bundle: ArtifactBundle,
    uncovered_requirement_ids: list[str],
) -> ArtifactBundle:
    requirements = [
        {
            "requirement_id": item.requirement_id,
            "title": item.title,
            "description": item.description,
            "module": item.module,
        }
        for item in bundle.requirements
        if item.requirement_id in uncovered_requirement_ids
    ]
    scenarios = [
        {
            "scenario_id": item.scenario_id,
            "title": item.title,
            "objective": item.objective,
            "requirement_ids": item.requirement_ids,
        }
        for item in bundle.scenarios
    ]
    test_cases = [
        {
            "test_case_id": item.test_case_id,
            "scenario_id": item.scenario_id,
            "title": item.title,
            "requirement_ids": item.requirement_ids,
        }
        for item in bundle.test_cases
    ]
    context.semantic_revisions += 1
    repair = context.generate(
        [
            {
                "role": "user",
                "content": (
                    "Return one coverage assignment for each uncovered requirement. "
                    "Use only IDs in these catalogs. Choose the existing scenario and "
                    "that scenario's existing test case that most directly cover the "
                    "requirement. Do not create artifacts or IDs.\n"
                    f"UNCOVERED REQUIREMENTS:\n{json.dumps(requirements, ensure_ascii=False)}\n"
                    f"SCENARIOS:\n{json.dumps(scenarios, ensure_ascii=False)}\n"
                    f"TEST CASES:\n{json.dumps(test_cases, ensure_ascii=False)}"
                ),
            }
        ],
        CoverageRepair,
        max_output_tokens=2_000,
        agent="reviewer",
    )
    scenario_links = {
        item.scenario_id: list(item.requirement_ids) for item in bundle.scenarios
    }
    test_links = {
        item.test_case_id: list(item.requirement_ids) for item in bundle.test_cases
    }
    scenarios_by_id = {item.scenario_id: item for item in bundle.scenarios}
    tests_by_id = {item.test_case_id: item for item in bundle.test_cases}
    uncovered = set(uncovered_requirement_ids)
    for assignment in repair.assignments:
        scenario = scenarios_by_id.get(assignment.scenario_id)
        test_case = tests_by_id.get(assignment.test_case_id)
        if (
            assignment.requirement_id not in uncovered
            or scenario is None
            or test_case is None
            or test_case.scenario_id != scenario.scenario_id
        ):
            continue
        if assignment.requirement_id not in scenario_links[scenario.scenario_id]:
            scenario_links[scenario.scenario_id].append(assignment.requirement_id)
        if assignment.requirement_id not in test_links[test_case.test_case_id]:
            test_links[test_case.test_case_id].append(assignment.requirement_id)
    return bundle.model_copy(
        update={
            "scenarios": [
                item.model_copy(
                    update={"requirement_ids": scenario_links[item.scenario_id]}
                )
                for item in bundle.scenarios
            ],
            "test_cases": [
                item.model_copy(
                    update={"requirement_ids": test_links[item.test_case_id]}
                )
                for item in bundle.test_cases
            ],
        }
    )


def run_generation(
    pdf_bytes: bytes,
    source_filename: str,
    run_type: RunType,
    settings: ProviderSettings,
    *,
    repository: RunRepository,
    progress: Progress | None = None,
    provider_factory: ProviderFactory | None = None,
) -> RunResult:
    settings.validate()
    display_name = (
        PurePosixPath(source_filename.replace("\\", "/")).name.strip()
        if isinstance(source_filename, str)
        else ""
    )
    if display_name in {"", ".", ".."} or any(
        ord(character) < 32 for character in display_name
    ):
        display_name = "document.pdf"

    document_hash = hashlib.sha256(pdf_bytes).hexdigest()
    manifest = RunManifest(
        run_id=_run_id(document_hash),
        source_filename=display_name,
        document_hash=document_hash,
        run_type=run_type,
        status=RunStatus.RUNNING,
        provider=settings.provider,
        model=settings.model,
        temperature=0.0,
        token_ceiling=settings.token_ceiling,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        started_at=_now(),
    )
    repository.create_run(manifest)
    repository.append_event(manifest.run_id, "started")
    _notify(progress, "Preparing document")

    try:
        chunks = parse_pdf(pdf_bytes)
    except DocumentError as error:
        manifest = manifest.model_copy(
            update={
                "status": RunStatus.FAILED,
                "completed_at": _now(),
                "failure_category": FailureCategory.PARSING,
                "failure_message": _safe_message(error, settings),
            }
        )
        result = RunResult(manifest=manifest)
        repository.finalize(result)
        _notify(progress, "Failed")
        return result

    repository.save_chunks(manifest.run_id, chunks)
    repository.append_event(manifest.run_id, "parsed")
    _notify(progress, "Generating artifacts")

    provider_factory = provider_factory or (
        lambda _run_type, ledger: _make_provider(settings, ledger)
    )
    ledger = BudgetLedger(settings.token_ceiling)
    context: PipelineContext | None = None
    bundle: ArtifactBundle | None = None
    validation: ValidationReport | None = None
    rtm: list[RTMRow] = []
    started = time.perf_counter()
    try:
        try:
            provider = provider_factory(run_type, ledger)
        except ValueError as error:
            raise ConfigurationError(str(error)) from error
        if getattr(provider, "ledger", None) is not ledger:
            raise ConfigurationError("Provider must use the run budget ledger.")
        if getattr(provider, "model", None) != settings.model:
            raise ConfigurationError("Provider model must match the run model.")

        providers: dict[str, StructuredProvider] = {}
        if run_type is RunType.CENTRALIZED_MULTI_AGENT:
            for agent in ("analyst", "test_generator", "reviewer"):
                model = settings.model_for(agent)
                if model == settings.model:
                    continue
                agent_provider = _make_provider(settings.with_model(model), ledger)
                if getattr(agent_provider, "ledger", None) is not ledger:
                    raise ConfigurationError("Provider must use the run budget ledger.")
                if getattr(agent_provider, "model", None) != model:
                    raise ConfigurationError(
                        "Agent provider model must match its configured model."
                    )
                providers[agent] = agent_provider

        context = PipelineContext(
            provider=provider, providers=providers, progress=progress
        )
        bundle = canonicalize_source_references(
            PIPELINES[run_type](context, chunks), chunks
        )
        validation = validate_bundle(bundle, chunks)
        if validation.issues and all(
            issue.code == "uncovered_requirement" for issue in validation.issues
        ):
            bundle = _repair_coverage(
                context, bundle, validation.uncovered_requirement_ids
            )
            validation = validate_bundle(bundle, chunks)
        if not validation.valid:
            bundle = context.revise(
                [],
                "artifact bundle",
                bundle,
                ReviewResult(
                    accepted=False,
                    issues=[
                        ReviewIssue(
                            artifact_id=issue.artifact_id,
                            reason=f"{issue.code}: {issue.message}",
                        )
                        for issue in validation.issues
                    ],
                ),
                chunks,
                ArtifactBundle,
                16_000,
            )
            bundle = canonicalize_source_references(bundle, chunks)
            validation = validate_bundle(bundle, chunks)

        rtm = build_rtm(bundle)
        metrics = compute_metrics(
            bundle,
            validation,
            input_tokens=context.input_tokens,
            output_tokens=context.output_tokens,
            charged_tokens=context.charged_tokens,
            latency_seconds=time.perf_counter() - started,
            retries=context.retries,
            schema_repairs=context.schema_repairs,
            semantic_revisions=context.semantic_revisions,
            budget_exhausted=False,
        )
        if validation.valid:
            manifest = manifest.model_copy(
                update={"status": RunStatus.COMPLETED, "completed_at": _now()}
            )
        else:
            manifest = manifest.model_copy(
                update={
                    "status": RunStatus.FAILED,
                    "completed_at": _now(),
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
        manifest = manifest.model_copy(
            update={
                "status": RunStatus.FAILED,
                "completed_at": _now(),
                "failure_category": category,
                "failure_message": _safe_message(error, settings),
            }
        )

    result = RunResult(
        manifest=manifest,
        bundle=bundle,
        validation=validation,
        rtm=rtm,
        metrics=metrics,
    )
    repository.finalize(result)
    _notify(progress, manifest.status.value.title())
    return result
