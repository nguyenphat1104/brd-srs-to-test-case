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
from pydantic import ValidationError

from .documents import DocumentError, canonicalize_source_references, parse_pdf
from .coverage import evaluate_against_catalog, extract_coverage_catalog
from .models import (
    AgentSetup,
    AgentStageOutput,
    ArtifactBundle,
    CoverageEvaluation,
    CoverageEvaluationStatus,
    CoverageRepair,
    CoverageScore,
    DocumentChunk,
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
    STAGED_OUTPUT_TOKEN_DEFAULTS,
    WORKER_COUNT,
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
LOCAL_REQUEST_TOKEN_BUDGET = 12_000
LOCAL_PROVIDERS = {"lm_studio", "llama_cpp", "ollama"}
THINKING_LEVELS = {"minimal", "low", "medium", "high"}
JUDGE_PROVIDER = "gemini"
JUDGE_MODEL = "gemini-3.6-flash"
JUDGE_THINKING_LEVEL = "medium"
JUDGE_TOKEN_CEILING = 100_000
COVERAGE_PROMPT_VERSION = "coverage-v2"
COVERAGE_SCHEMA_VERSION = "coverage-catalog-v1"
EVALUATOR_VERSION = (
    f"{COVERAGE_PROMPT_VERSION}:{COVERAGE_SCHEMA_VERSION}:"
    f"{JUDGE_PROVIDER}:{JUDGE_MODEL}:{JUDGE_THINKING_LEVEL}"
)
JUDGE_PURPOSE = (
    "Extract atomic source coverage units and strictly map generated test cases "
    "for precision, recall, and F1 scoring."
)
RUN_AGENTS = {
    RunType.SINGLE_PROMPT: ("single",),
    RunType.STAGED_SINGLE_AGENT: ("requirements", "scenarios", "test_cases"),
    RunType.CENTRALIZED_MULTI_AGENT: (
        "analyst",
        "test_generator",
        "reviewer",
    ),
}
ProviderFactory = Callable[[RunType, BudgetLedger], StructuredProvider]
JudgeProviderFactory = Callable[[BudgetLedger], StructuredProvider]
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
    coverage_analyzer_model: str = ""
    agent_setups: dict[str, AgentSetup] = field(default_factory=dict)
    agent_providers: dict[str, str] = field(default_factory=dict)
    agent_models: dict[str, str] = field(default_factory=dict)
    agent_prompts: dict[str, str] = field(default_factory=dict)
    thinking_level: str | None = None
    agent_thinking_levels: dict[str, str] = field(default_factory=dict)
    agent_max_output_tokens: dict[str, int] = field(default_factory=dict)
    critic_enabled: bool = True
    provider_api_keys: dict[str, str] = field(default_factory=dict, repr=False)
    provider_base_urls: dict[str, str] = field(default_factory=dict, repr=False)

    def provider_for(self, agent: str) -> str:
        return self.agent_providers.get(agent, self.provider).strip()

    def model_for(self, agent: str) -> str:
        if configured := self.agent_models.get(agent, "").strip():
            return configured
        configured = getattr(self, f"{agent}_model", "")
        return configured.strip() or self.model

    def prompt_for(self, agent: str) -> str:
        return self.agent_prompts.get(agent, "").strip()

    def thinking_level_for(self, agent: str) -> str | None:
        configured = self.agent_thinking_levels.get(agent)
        if configured is not None:
            return configured
        return self.thinking_level if self.provider_for(agent) == self.provider else None

    def api_key_for(self, provider: str) -> str:
        return self.provider_api_keys.get(
            provider, self.api_key if provider == self.provider else ""
        )

    def base_url_for(self, provider: str) -> str:
        return self.provider_base_urls.get(
            provider, self.base_url if provider == self.provider else ""
        )

    def for_agent(self, agent: str) -> ProviderSettings:
        provider = self.provider_for(agent)
        return replace(
            self,
            provider=provider,
            model=self.model_for(agent),
            api_key=self.api_key_for(provider),
            base_url=self.base_url_for(provider),
            thinking_level=self.thinking_level_for(agent),
        )

    def snapshot(self, run_type: RunType) -> dict[str, object]:
        agents = {
            agent: {
                "provider": self.provider_for(agent),
                "model": self.model_for(agent),
                "prompt": self.prompt_for(agent),
                "thinking_level": self.thinking_level_for(agent),
                "max_output_tokens": self.agent_max_output_tokens.get(agent),
            }
            for agent in RUN_AGENTS[run_type]
        }
        agents["judge"] = {
            "provider": JUDGE_PROVIDER,
            "model": JUDGE_MODEL,
            "prompt": JUDGE_PURPOSE,
            "thinking_level": JUDGE_THINKING_LEVEL,
            "token_ceiling": JUDGE_TOKEN_CEILING,
            "fixed": True,
        }
        return {
            "agents": agents,
            "token_ceiling": self.token_ceiling,
            "critic_enabled": self.critic_enabled,
        }

    def with_model(self, model: str) -> ProviderSettings:
        return replace(self, model=model)

    def with_agent_setups(
        self, agent_setups: dict[str, AgentSetup]
    ) -> ProviderSettings:
        return replace(self, agent_setups=agent_setups)

    def validate(self) -> None:
        if not isinstance(self.critic_enabled, bool):
            raise ValueError("Critic enabled must be a boolean.")
        if (
            not isinstance(self.token_ceiling, int)
            or isinstance(self.token_ceiling, bool)
            or self.token_ceiling < 1
        ):
            raise ValueError("Token ceiling must be positive.")
        agent_names = (
            set(self.agent_providers)
            | set(self.agent_models)
            | set(self.agent_prompts)
            | set(self.agent_thinking_levels)
            | set(self.agent_max_output_tokens)
        )
        if any(not isinstance(agent, str) or not agent for agent in agent_names):
            raise ValueError("Agent configuration keys must be non-empty strings.")
        for agent in ("default", *sorted(agent_names)):
            provider = self.provider if agent == "default" else self.provider_for(agent)
            model = self.model if agent == "default" else self.model_for(agent)
            self._validate_provider(provider, model)
            thinking_level = (
                self.thinking_level
                if agent == "default"
                else self.thinking_level_for(agent)
            )
            if thinking_level is not None:
                if not isinstance(thinking_level, str) or thinking_level not in THINKING_LEVELS:
                    raise ValueError("Thinking level must be minimal, low, medium, or high.")
                if provider != "gemini" or not model.startswith("gemini-3"):
                    raise ValueError("Thinking level requires a Gemini 3 model.")
            if agent != "default" and not isinstance(self.agent_prompts.get(agent, ""), str):
                raise ValueError("Agent prompt must be a string.")
            max_output_tokens = self.agent_max_output_tokens.get(agent)
            if max_output_tokens is not None and (
                not isinstance(max_output_tokens, int)
                or isinstance(max_output_tokens, bool)
                or max_output_tokens < 1
            ):
                raise ValueError("Step output tokens must be positive integers.")

    def _validate_provider(self, provider: str, model: str) -> None:
        if not isinstance(provider, str) or provider not in {
            "gemini",
            *LOCAL_PROVIDERS,
        }:
            raise ValueError("Provider must be gemini, LM Studio, llama.cpp, or ollama.")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Model must not be blank.")
        api_key = self.api_key_for(provider)
        base_url = self.base_url_for(provider)
        if not isinstance(api_key, str):
            raise ValueError("API key must be a string.")
        for agent in ("analyst", "test_generator", "reviewer", "coverage_analyzer"):
            if not isinstance(getattr(self, f"{agent}_model"), str):
                raise ValueError(
                    f"{agent.replace('_', ' ').title()} model must be a string."
                )
        if provider == "gemini" and not api_key.strip():
            raise ValueError("Gemini API key is required.")
        if provider in LOCAL_PROVIDERS:
            provider_name = {
                "lm_studio": "LM Studio",
                "llama_cpp": "llama.cpp",
                "ollama": "Ollama",
            }[provider]
            if not isinstance(base_url, str) or any(
                character.isspace() for character in base_url
            ):
                raise ValueError(
                    f"{provider_name} base URL must be an HTTP(S) URL."
                )
            try:
                parsed = urlsplit(base_url)
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
                or "?" in base_url
                or "#" in base_url
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
            genai.Client(api_key=settings.api_key),
            settings.model,
            ledger,
            thinking_level=settings.thinking_level,
        )
    if settings.provider == "lm_studio":
        return LMStudioProvider(
            settings.base_url,
            settings.model,
            ledger,
            api_key=settings.api_key,
        )
    if settings.provider == "llama_cpp":
        return LMStudioProvider(
            settings.base_url,
            settings.model,
            ledger,
            api_key=settings.api_key,
            auto_load=False,
        )
    return OllamaProvider(settings.base_url, settings.model, ledger)


def _make_judge_provider(
    settings: ProviderSettings, ledger: BudgetLedger
) -> StructuredProvider:
    return _make_provider(
        replace(
            settings,
            provider=JUDGE_PROVIDER,
            model=JUDGE_MODEL,
            api_key=settings.api_key_for(JUDGE_PROVIDER),
            base_url="",
            thinking_level=JUDGE_THINKING_LEVEL,
        ),
        ledger,
    )


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
    secrets = [
        settings.api_key,
        settings.base_url,
        *settings.provider_api_keys.values(),
        *settings.provider_base_urls.values(),
    ]
    for base_url in [settings.base_url, *settings.provider_base_urls.values()]:
        try:
            parsed = urlsplit(base_url)
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


def _revision_chunks(
    bundle: ArtifactBundle,
    validation: ValidationReport,
    chunks: list[DocumentChunk],
) -> list[DocumentChunk]:
    artifact_ids = {issue.artifact_id for issue in validation.issues}
    artifacts = [
        *(
            item for item in bundle.requirements if item.requirement_id in artifact_ids
        ),
        *(item for item in bundle.scenarios if item.scenario_id in artifact_ids),
        *(item for item in bundle.test_cases if item.test_case_id in artifact_ids),
    ]
    source_chunk_ids = {
        reference.chunk_id
        for artifact in artifacts
        for reference in artifact.source_references
    }
    return [chunk for chunk in chunks if chunk.chunk_id in source_chunk_ids]


def run_generation(
    pdf_bytes: bytes,
    source_filename: str,
    run_type: RunType,
    settings: ProviderSettings,
    *,
    repository: RunRepository,
    progress: Progress | None = None,
    provider_factory: ProviderFactory | None = None,
    judge_provider_factory: JudgeProviderFactory | None = None,
) -> RunResult:
    settings.validate()
    primary_agent = RUN_AGENTS[run_type][0]
    primary_settings = settings.for_agent(primary_agent)
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
        provider=primary_settings.provider,
        model=primary_settings.model,
        temperature=0.0,
        token_ceiling=settings.token_ceiling,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        configuration=settings.snapshot(run_type),
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
        lambda _run_type, ledger: _make_provider(primary_settings, ledger)
    )
    ledger = BudgetLedger(settings.token_ceiling)
    context: PipelineContext | None = None
    bundle: ArtifactBundle | None = None
    validation: ValidationReport | None = None
    coverage: CoverageScore | None = None
    coverage_evaluation: CoverageEvaluation | None = None
    judge_charged_tokens = 0
    rtm: list[RTMRow] = []
    started = time.perf_counter()
    try:
        try:
            provider = provider_factory(run_type, ledger)
        except ValueError as error:
            raise ConfigurationError(str(error)) from error
        if getattr(provider, "ledger", None) is not ledger:
            raise ConfigurationError("Provider must use the run budget ledger.")
        if getattr(provider, "model", None) != primary_settings.model:
            raise ConfigurationError("Provider model must match the run model.")

        providers: dict[str, StructuredProvider] = {}
        for agent in RUN_AGENTS[run_type][1:]:
            agent_settings = settings.for_agent(agent)
            if (
                agent_settings.provider == primary_settings.provider
                and agent_settings.model == primary_settings.model
                and agent_settings.thinking_level == primary_settings.thinking_level
            ):
                continue
            agent_provider = _make_provider(agent_settings, ledger)
            if getattr(agent_provider, "ledger", None) is not ledger:
                raise ConfigurationError("Provider must use the run budget ledger.")
            if getattr(agent_provider, "model", None) != agent_settings.model:
                raise ConfigurationError(
                    "Agent provider model must match its configured model."
                )
            providers[agent] = agent_provider

        configured_providers = {
            settings.provider_for(agent) for agent in RUN_AGENTS[run_type]
        }
        local_provider = bool(configured_providers & LOCAL_PROVIDERS)
        context = PipelineContext(
            provider=provider,
            token_ceiling=settings.token_ceiling,
            critic_enabled=settings.critic_enabled,
            providers=providers,
            agent_setups=settings.agent_setups,
            agent_prompts=settings.agent_prompts,
            agent_max_output_tokens=settings.agent_max_output_tokens,
            progress=progress,
            max_request_tokens=(
                LOCAL_REQUEST_TOKEN_BUDGET
                if "llama_cpp" in configured_providers
                else None
            ),
            worker_limit=1 if local_provider else WORKER_COUNT,
            bounded_tasks=local_provider,
        )
        bundle = canonicalize_source_references(
            PIPELINES[run_type](context, chunks), chunks
        )
        validation = validate_bundle(bundle, chunks)
        if (
            run_type is not RunType.CENTRALIZED_MULTI_AGENT
            and validation.issues
            and all(
                issue.code == "uncovered_requirement"
                for issue in validation.issues
            )
        ):
            bundle = _repair_coverage(
                context, bundle, validation.uncovered_requirement_ids
            )
            validation = validate_bundle(bundle, chunks)
        if (
            run_type is not RunType.CENTRALIZED_MULTI_AGENT
            and not validation.valid
            and context.max_request_tokens is None
        ):
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
                _revision_chunks(bundle, validation, chunks),
                ArtifactBundle,
                (
                    settings.agent_max_output_tokens.get(
                        "test_cases", STAGED_OUTPUT_TOKEN_DEFAULTS["test_cases"]
                    )
                    if run_type is RunType.STAGED_SINGLE_AGENT
                    else 16_000
                ),
            )
            bundle = canonicalize_source_references(bundle, chunks)
            validation = validate_bundle(bundle, chunks)

        rtm = build_rtm(bundle)
        if bundle is not None:
            _notify(progress, "Analyzing coverage")
            judge_ledger = BudgetLedger(JUDGE_TOKEN_CEILING)
            judge_context: PipelineContext | None = None
            catalog = repository.load_coverage_catalog(
                document_hash, EVALUATOR_VERSION
            )
            try:
                if judge_provider_factory is not None:
                    judge_provider = judge_provider_factory(judge_ledger)
                elif not settings.api_key_for(JUDGE_PROVIDER).strip():
                    raise ConfigurationError(
                        "Gemini API key is required for the fixed judge."
                    )
                else:
                    judge_provider = _make_judge_provider(settings, judge_ledger)
                if getattr(judge_provider, "ledger", None) is not judge_ledger:
                    raise ConfigurationError(
                        "Judge provider must use the judge budget ledger."
                    )
                if getattr(judge_provider, "model", None) != JUDGE_MODEL:
                    raise ConfigurationError(
                        "Judge provider model must be Gemini 3.6 Flash."
                    )
                judge_context = PipelineContext(
                    provider=judge_provider,
                    token_ceiling=JUDGE_TOKEN_CEILING,
                    agent_setups={
                        "coverage_analyzer": AgentSetup(
                            agent="coverage_analyzer",
                            role="Independent quality judge",
                        )
                    },
                    progress=progress,
                )
                if catalog is None:
                    _notify(
                        progress,
                        "Judge: extracting coverage units from the source document.",
                    )
                    catalog = extract_coverage_catalog(
                        judge_context,
                        chunks,
                        document_hash=document_hash,
                        evaluator_version=EVALUATOR_VERSION,
                        catalog_id=(
                            f"{document_hash[:16]}-"
                            f"{hashlib.sha256(EVALUATOR_VERSION.encode()).hexdigest()[:12]}"
                        ),
                        created_at=_now(),
                    )
                    catalog = repository.save_coverage_catalog(catalog)
                    _notify(
                        progress,
                        f"Judge: extracted {len(catalog.units)} coverage units. "
                        "Mapping test cases...",
                    )
                else:
                    _notify(
                        progress,
                        f"Judge: reusing {len(catalog.units)} frozen coverage units. "
                        "Mapping test cases...",
                    )
                coverage_evaluation = evaluate_against_catalog(
                    judge_context,
                    run_id=manifest.run_id,
                    bundle=bundle,
                    catalog=catalog,
                    evaluated_at=_now(),
                )
                coverage = coverage_evaluation.score
                if coverage is not None:
                    _notify(
                        progress,
                        f"Judge: F1={coverage.f1:.2f} "
                        f"(precision={coverage.precision:.2f}, "
                        f"recall={coverage.recall:.2f}).",
                    )
            except (
                ConfigurationError,
                ProviderError,
                PipelineOutputError,
                ValueError,
                ValidationError,
            ) as error:
                message = _safe_message(error, settings)
                if catalog is not None:
                    coverage_evaluation = CoverageEvaluation(
                        run_id=manifest.run_id,
                        catalog_id=catalog.catalog_id,
                        status=CoverageEvaluationStatus.FAILED,
                        error=message,
                        evaluated_at=_now(),
                    )
                _notify(progress, f"Coverage evaluation failed: {message}")
            finally:
                judge_charged_tokens = judge_ledger.used
                if judge_context is not None:
                    context.input_tokens += judge_context.input_tokens
                    context.output_tokens += judge_context.output_tokens
                    context.retries += judge_context.retries
                    context.schema_repairs += judge_context.schema_repairs
                    context.semantic_revisions += judge_context.semantic_revisions
        metrics = compute_metrics(
            bundle,
            validation,
            input_tokens=context.input_tokens,
            output_tokens=context.output_tokens,
            charged_tokens=context.charged_tokens + judge_charged_tokens,
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
        stage_outputs=(
            sorted(context.stage_outputs, key=AgentStageOutput.order_key)
            if context else []
        ),
        bundle=bundle,
        validation=validation,
        rtm=rtm,
        metrics=metrics,
        coverage=coverage,
        coverage_evaluation=coverage_evaluation,
    )
    repository.finalize(result)
    _notify(progress, manifest.status.value.title())
    return result
