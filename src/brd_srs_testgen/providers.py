from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)
Messages = list[dict[str, str]]
RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}
TIMEOUT_CODES = {408, 504}
LM_STUDIO_MIN_CONTEXT = 32_768


class BudgetExceeded(RuntimeError):
    def __init__(self, message: str, *, reservation_blocked: bool = False) -> None:
        super().__init__(message)
        self.reservation_blocked = reservation_blocked


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: int | None,
        retryable: bool,
        timed_out: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.timed_out = timed_out


class StructuredOutputError(RuntimeError):
    def __init__(
        self,
        raw_text: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_seconds: float = 0.0,
    ) -> None:
        super().__init__("Provider returned invalid structured output.")
        self.raw_text = raw_text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_seconds = latency_seconds


@dataclass(frozen=True)
class Reservation:
    tokens: int
    _ledger: object | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _identity: object = field(
        default_factory=object, init=False, repr=False, compare=False
    )


@dataclass
class BudgetLedger:
    limit: int
    used: int = 0
    reserved: int = field(default=0, init=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _active: dict[object, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if _token_count(self.limit, "Token limit") < 1:
            raise ValueError("Token limit must be positive.")
        _token_count(self.used, "Used token count")
        if self.used > self.limit:
            raise ValueError("Used tokens cannot exceed the token limit.")

    @property
    def remaining(self) -> int:
        with self._lock:
            return self.limit - self.used - self.reserved

    def reserve(self, tokens: int) -> Reservation:
        _positive_token_count(tokens, "Reservation")
        with self._lock:
            remaining = self.limit - self.used - self.reserved
            if tokens > remaining:
                raise BudgetExceeded(
                    f"Need {tokens} tokens; {remaining} remain.",
                    reservation_blocked=True,
                )
            self.reserved += tokens
            reservation = Reservation(tokens)
            object.__setattr__(reservation, "_ledger", self)
            self._active[reservation._identity] = tokens
        return reservation

    def _release(self, reservation: Reservation) -> int:
        if (
            reservation._ledger is not self
            or reservation._identity not in self._active
        ):
            raise ValueError("Reservation is not active for this ledger.")
        return self._active.pop(reservation._identity)

    def cancel(self, reservation: Reservation) -> None:
        with self._lock:
            self.reserved -= self._release(reservation)

    def settle(self, reservation: Reservation, actual_tokens: int) -> None:
        _token_count(actual_tokens, "Actual token count")
        with self._lock:
            self.reserved -= self._release(reservation)
            self.used += actual_tokens
            over = self.used + self.reserved > self.limit
        if over:
            raise BudgetExceeded(
                f"Actual usage {self.used} exceeded token limit {self.limit}."
            )


@dataclass(frozen=True)
class GenerationResult(Generic[T]):
    value: T
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    billed_tokens: int | None = None

    @property
    def total_tokens(self) -> int:
        return (
            self.billed_tokens
            if self.billed_tokens is not None
            else self.input_tokens + self.output_tokens
        )


class StructuredProvider(Protocol):
    model: str
    ledger: BudgetLedger

    def generate(
        self,
        messages: Messages,
        schema: type[T],
        *,
        max_output_tokens: int,
    ) -> GenerationResult[T]:
        pass


def _prompt(messages: Messages) -> str:
    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}" for message in messages
    )


def _error_code(error: Exception) -> int | None:
    code = getattr(error, "code", None)
    if isinstance(code, int):
        return code
    status = getattr(error, "status_code", None)
    return status if isinstance(status, int) else None


def _token_count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


def _positive_token_count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _max_output_tokens(value: object) -> int:
    return _positive_token_count(value, "max_output_tokens")


def _is_transient(error: Exception) -> bool:
    error_type = type(error)
    return isinstance(error, (TimeoutError, ConnectionError)) or any(
        base.__module__ in {"httpx", "httpx._exceptions"}
        and base.__name__ == "TransportError"
        for base in error_type.__mro__
    ) or (
        error_type.__module__.startswith("google.genai.")
        and error_type.__name__ in {"APIConnectionError", "APITimeoutError"}
    )


def _is_timeout(error: Exception) -> bool:
    error_type = type(error)
    return _error_code(error) in TIMEOUT_CODES or isinstance(
        error, TimeoutError
    ) or any(
        (
            base.__module__ in {"httpx", "httpx._exceptions"}
            and base.__name__ == "TimeoutException"
        )
        or (
            base.__module__.startswith("google.genai.")
            and base.__name__ == "APITimeoutError"
        )
        for base in error_type.__mro__
    )


def _provider_error(error: Exception) -> ProviderError:
    code = _error_code(error)
    return ProviderError(
        str(error),
        code=code,
        retryable=code in RETRYABLE_CODES or _is_transient(error),
        timed_out=_is_timeout(error),
    )


def _url_error_retryable(error: URLError) -> bool:
    reason = error.reason
    return isinstance(reason, (TimeoutError, ConnectionError)) or (
        isinstance(reason, socket.gaierror) and reason.errno == socket.EAI_AGAIN
    )


def _ollama_url(base_url: str) -> str:
    url = f"{base_url}/api/chat"
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or any(char.isspace() for char in parsed.netloc)
    ):
        raise ValueError("Invalid Ollama URL.")
    return url


def _lm_studio_url(base_url: str, endpoint: str) -> str:
    url = f"{base_url.rstrip('/')}/{endpoint}"
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or any(char.isspace() for char in parsed.netloc)
    ):
        raise ValueError("Invalid LM Studio URL.")
    return url


def _lm_studio_native_url(base_url: str, endpoint: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3].rstrip("/")
    return _lm_studio_url(base_url, f"api/v1/{endpoint}")


def _lm_studio_error_message(error: HTTPError) -> str:
    try:
        body = error.read(2_000).decode("utf-8", "replace").strip()
        detail = json.loads(body).get("error", {}).get("message", body)
    except (AttributeError, TypeError, ValueError, OSError):
        return str(error)
    return f"{error}: {detail}" if isinstance(detail, str) and detail else str(error)


def _lm_studio_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def load_lm_studio_model(
    base_url: str,
    model: str,
    api_key: str = "",
    *,
    context_length: int | None = None,
    timeout: int = 120,
) -> None:
    headers = {"Content-Type": "application/json"}
    headers.update(_lm_studio_headers(api_key))
    payload: dict[str, object] = {"model": model}
    if context_length is not None:
        payload["context_length"] = context_length
    try:
        request = Request(
            _lm_studio_native_url(base_url, "models/load"),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=timeout):
            pass
    except HTTPError as error:
        raise ProviderError(
            _lm_studio_error_message(error),
            code=error.code,
            retryable=error.code in RETRYABLE_CODES,
            timed_out=error.code in TIMEOUT_CODES,
        ) from error
    except URLError as error:
        raise ProviderError(
            str(error),
            code=None,
            retryable=_url_error_retryable(error),
            timed_out=_is_timeout(error.reason),
        ) from error
    except (TimeoutError, ConnectionError) as error:
        raise ProviderError(
            str(error), code=None, retryable=True, timed_out=_is_timeout(error)
        ) from error
    except (TypeError, ValueError) as error:
        raise ProviderError(
            "Invalid LM Studio URL.", code=None, retryable=False
        ) from error
    except Exception as error:
        raise ProviderError(str(error), code=None, retryable=False) from error


def _lm_studio_model_state(
    base_url: str, model: str, api_key: str, *, timeout: int
) -> tuple[list[str], int, int]:
    try:
        request = Request(
            _lm_studio_native_url(base_url, "models"),
            headers=_lm_studio_headers(api_key),
            method="GET",
        )
        with urlopen(request, timeout=timeout) as response:
            models = json.load(response)["models"]
        selected = next(
            item for item in models if isinstance(item, dict) and item.get("key") == model
        )
        maximum = _positive_token_count(
            selected["max_context_length"], "LM Studio model context length"
        )
        instances = selected.get("loaded_instances", [])
        instance_ids = [
            item["id"]
            for item in instances
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
        loaded_context = max(
            (
                item.get("config", {}).get("context_length", 0)
                for item in instances
                if isinstance(item, dict)
                and isinstance(item.get("config"), dict)
                and isinstance(item["config"].get("context_length"), int)
            ),
            default=0,
        )
        return instance_ids, loaded_context, maximum
    except HTTPError as error:
        raise ProviderError(
            _lm_studio_error_message(error),
            code=error.code,
            retryable=error.code in RETRYABLE_CODES,
            timed_out=error.code in TIMEOUT_CODES,
        ) from error
    except (KeyError, StopIteration, TypeError, ValueError) as error:
        raise ProviderError(
            f"LM Studio could not find configured model '{model}'.",
            code=None,
            retryable=False,
        ) from error
    except URLError as error:
        raise ProviderError(
            str(error),
            code=None,
            retryable=_url_error_retryable(error),
            timed_out=_is_timeout(error.reason),
        ) from error
    except (TimeoutError, ConnectionError) as error:
        raise ProviderError(
            str(error), code=None, retryable=True, timed_out=_is_timeout(error)
        ) from error
    except Exception as error:
        raise ProviderError(str(error), code=None, retryable=False) from error


def _unload_lm_studio_model(
    base_url: str, instance_id: str, api_key: str, *, timeout: int
) -> None:
    headers = {"Content-Type": "application/json"}
    headers.update(_lm_studio_headers(api_key))
    try:
        request = Request(
            _lm_studio_native_url(base_url, "models/unload"),
            data=json.dumps({"instance_id": instance_id}).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=timeout):
            pass
    except HTTPError as error:
        raise ProviderError(
            _lm_studio_error_message(error),
            code=error.code,
            retryable=error.code in RETRYABLE_CODES,
            timed_out=error.code in TIMEOUT_CODES,
        ) from error
    except Exception as error:
        raise _provider_error(error) from error


def list_lm_studio_models(
    base_url: str, api_key: str = "", *, timeout: int = 10
) -> list[str]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        request = Request(
            _lm_studio_url(base_url, "models"),
            headers=headers,
            method="GET",
        )
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        models = sorted(
            {
                item["id"]
                for item in result["data"]
                if isinstance(item, dict)
                and isinstance(item.get("id"), str)
                and item["id"].strip()
            }
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ProviderError(
            "LM Studio returned an invalid model list.",
            code=None,
            retryable=False,
        ) from error
    except Exception as error:
        raise _provider_error(error) from error
    if not models:
        raise ProviderError(
            "LM Studio returned no available models.", code=None, retryable=False
        )
    return models


class GeminiProvider:
    def __init__(self, client, model: str, ledger: BudgetLedger) -> None:
        self.client = client
        self.model = model
        self.ledger = ledger

    def generate(
        self,
        messages: Messages,
        schema: type[T],
        *,
        max_output_tokens: int,
    ) -> GenerationResult[T]:
        max_output_tokens = _max_output_tokens(max_output_tokens)
        prompt = _prompt(messages)
        try:
            input_estimate = _token_count(
                self.client.models.count_tokens(
                    model=self.model, contents=prompt
                ).total_tokens,
                "Input token count",
            )
        except Exception as error:
            raise _provider_error(error) from error

        reservation = self.ledger.reserve(input_estimate + max_output_tokens)
        started = time.perf_counter()
        try:
            interaction = self.client.interactions.create(
                model=self.model,
                input=prompt,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema.model_json_schema(),
                },
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": max_output_tokens,
                },
            )
        except Exception as error:
            if isinstance(error, (ValueError, TypeError)):
                self.ledger.cancel(reservation)
            else:
                self.ledger.settle(reservation, reservation.tokens)
            raise _provider_error(error) from error

        try:
            input_tokens = _token_count(
                interaction.usage.total_input_tokens, "Input token count"
            )
            output_tokens = _token_count(
                interaction.usage.total_output_tokens, "Output token count"
            )
            reported_total = getattr(interaction.usage, "total_tokens", None)
            total_tokens = (
                input_tokens + output_tokens
                if reported_total is None
                else max(
                    _token_count(reported_total, "Total token count"),
                    input_tokens + output_tokens,
                )
            )
        except Exception as error:
            self.ledger.settle(reservation, reservation.tokens)
            raise ProviderError(
                "Gemini returned an incomplete response.", code=None, retryable=False
            ) from error
        self.ledger.settle(reservation, total_tokens)
        raw_text = getattr(interaction, "output_text", None)
        if not isinstance(raw_text, str):
            raise ProviderError(
                "Gemini returned an incomplete response.", code=None, retryable=False
            )
        try:
            value = schema.model_validate_json(raw_text)
        except Exception as error:
            raise StructuredOutputError(
                raw_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_seconds=time.perf_counter() - started,
            ) from error
        return GenerationResult(
            value=value,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=time.perf_counter() - started,
            billed_tokens=total_tokens,
        )


class OllamaProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        ledger: BudgetLedger,
        *,
        timeout: int = 600,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.ledger = ledger
        self.timeout = timeout

    def generate(
        self,
        messages: Messages,
        schema: type[T],
        *,
        max_output_tokens: int,
    ) -> GenerationResult[T]:
        max_output_tokens = _max_output_tokens(max_output_tokens)
        payload = {
            "model": self.model,
            "messages": messages,
            "format": schema.model_json_schema(),
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0, "num_predict": max_output_tokens},
        }
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            request = Request(
                _ollama_url(self.base_url),
                data=encoded,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        except (TypeError, ValueError) as error:
            raise ProviderError(
                "Invalid Ollama URL.", code=None, retryable=False
            ) from error
        reservation = self.ledger.reserve(len(encoded) + max_output_tokens)
        started = time.perf_counter()
        try:
            response = urlopen(request, timeout=self.timeout)
        except ValueError as error:
            self.ledger.cancel(reservation)
            raise ProviderError("Invalid Ollama URL.", code=None, retryable=False) from error
        except HTTPError as error:
            self.ledger.settle(reservation, reservation.tokens)
            raise ProviderError(
                str(error),
                code=error.code,
                retryable=error.code in RETRYABLE_CODES,
                timed_out=error.code in TIMEOUT_CODES,
            ) from error
        except URLError as error:
            self.ledger.settle(reservation, reservation.tokens)
            raise ProviderError(
                str(error),
                code=None,
                retryable=_url_error_retryable(error),
                timed_out=_is_timeout(error.reason),
            ) from error
        except (TimeoutError, ConnectionError) as error:
            self.ledger.settle(reservation, reservation.tokens)
            raise ProviderError(
                str(error),
                code=None,
                retryable=True,
                timed_out=_is_timeout(error),
            ) from error
        except Exception as error:
            self.ledger.settle(reservation, reservation.tokens)
            raise ProviderError(str(error), code=None, retryable=False) from error

        try:
            with response:
                result = json.load(response)
        except Exception as error:
            self.ledger.settle(reservation, reservation.tokens)
            raise ProviderError(str(error), code=None, retryable=False) from error

        try:
            input_tokens = _token_count(result["prompt_eval_count"], "Input token count")
            output_tokens = _token_count(
                result["eval_count"], "Output token count"
            )
            raw_text = result["message"]["content"]
        except (KeyError, TypeError, ValueError) as error:
            self.ledger.settle(reservation, reservation.tokens)
            raise ProviderError(
                "Ollama returned an incomplete response.", code=None, retryable=False
            ) from error

        self.ledger.settle(reservation, input_tokens + output_tokens)
        try:
            value = schema.model_validate_json(raw_text)
        except Exception as error:
            raise StructuredOutputError(
                raw_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_seconds=time.perf_counter() - started,
            ) from error
        return GenerationResult(
            value=value,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=time.perf_counter() - started,
        )


class LMStudioProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        ledger: BudgetLedger,
        *,
        api_key: str = "",
        auto_load: bool = True,
        timeout: int = 600,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.ledger = ledger
        self.api_key = api_key
        self.auto_load = auto_load
        self.timeout = timeout
        self._model_load_lock = threading.Lock()

    def _ensure_model_loaded(self, required_context: int) -> None:
        with self._model_load_lock:
            instances, loaded_context, maximum = _lm_studio_model_state(
                self.base_url, self.model, self.api_key, timeout=self.timeout
            )
            if required_context > maximum:
                raise ProviderError(
                    f"LM Studio model '{self.model}' supports {maximum:,} tokens, "
                    f"but this task needs about {required_context:,}.",
                    code=None,
                    retryable=False,
                )
            target_context = min(maximum, max(required_context, LM_STUDIO_MIN_CONTEXT))
            if loaded_context >= target_context:
                return
            for instance_id in instances:
                _unload_lm_studio_model(
                    self.base_url, instance_id, self.api_key, timeout=self.timeout
                )
            load_lm_studio_model(
                self.base_url,
                self.model,
                self.api_key,
                context_length=target_context,
                timeout=self.timeout,
            )

    def generate(
        self,
        messages: Messages,
        schema: type[T],
        *,
        max_output_tokens: int,
    ) -> GenerationResult[T]:
        max_output_tokens = _max_output_tokens(max_output_tokens)
        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": schema.model_json_schema(),
                },
            },
            "temperature": 0.0,
            "reasoning_effort": "none",
            "max_tokens": max_output_tokens,
            "stream": False,
        }
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        # ponytail: byte estimate; use a model tokenizer if it ever rejects a valid request.
        input_estimate = max(1, (len(encoded) + 3) // 4)
        required_context = ((input_estimate + max_output_tokens + 1_023) // 1_024) * 1_024
        if self.auto_load:
            self._ensure_model_loaded(required_context)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            request = Request(
                _lm_studio_url(self.base_url, "chat/completions"),
                data=encoded,
                headers=headers,
                method="POST",
            )
        except (TypeError, ValueError) as error:
            raise ProviderError(
                "Invalid LM Studio URL.", code=None, retryable=False
            ) from error

        reservation = self.ledger.reserve(input_estimate + max_output_tokens)
        started = time.perf_counter()
        try:
            response = urlopen(request, timeout=self.timeout)
        except ValueError as error:
            self.ledger.cancel(reservation)
            raise ProviderError(
                "Invalid LM Studio URL.", code=None, retryable=False
            ) from error
        except HTTPError as error:
            self.ledger.settle(reservation, reservation.tokens)
            raise ProviderError(
                _lm_studio_error_message(error),
                code=error.code,
                retryable=error.code in RETRYABLE_CODES,
                timed_out=error.code in TIMEOUT_CODES,
            ) from error
        except URLError as error:
            self.ledger.settle(reservation, reservation.tokens)
            raise ProviderError(
                str(error),
                code=None,
                retryable=_url_error_retryable(error),
                timed_out=_is_timeout(error.reason),
            ) from error
        except (TimeoutError, ConnectionError) as error:
            self.ledger.settle(reservation, reservation.tokens)
            raise ProviderError(
                str(error),
                code=None,
                retryable=True,
                timed_out=_is_timeout(error),
            ) from error
        except Exception as error:
            self.ledger.settle(reservation, reservation.tokens)
            raise ProviderError(str(error), code=None, retryable=False) from error

        try:
            with response:
                result = json.load(response)
            usage = result["usage"]
            input_tokens = _token_count(usage["prompt_tokens"], "Input token count")
            output_tokens = _token_count(
                usage["completion_tokens"], "Output token count"
            )
            raw_text = result["choices"][0]["message"]["content"]
        except (IndexError, KeyError, TypeError, ValueError) as error:
            self.ledger.settle(reservation, reservation.tokens)
            raise ProviderError(
                "LM Studio returned an incomplete response.",
                code=None,
                retryable=False,
            ) from error

        self.ledger.settle(reservation, input_tokens + output_tokens)
        try:
            value = schema.model_validate_json(raw_text)
        except Exception as error:
            raise StructuredOutputError(
                raw_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_seconds=time.perf_counter() - started,
            ) from error
        return GenerationResult(
            value=value,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=time.perf_counter() - started,
        )
