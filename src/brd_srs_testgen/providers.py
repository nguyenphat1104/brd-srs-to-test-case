from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)
Messages = list[dict[str, str]]
RETRYABLE_CODES = {429, 500, 502, 503, 504}


class BudgetExceeded(RuntimeError):
    pass


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, code: int | None, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


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


@dataclass
class BudgetLedger:
    limit: int
    used: int = 0
    reserved: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("Token limit must be positive.")

    @property
    def remaining(self) -> int:
        with self._lock:
            return self.limit - self.used - self.reserved

    def reserve(self, tokens: int) -> Reservation:
        if tokens < 1:
            raise ValueError("Reservation must be positive.")
        with self._lock:
            remaining = self.limit - self.used - self.reserved
            if tokens > remaining:
                raise BudgetExceeded(f"Need {tokens} tokens; {remaining} remain.")
            self.reserved += tokens
        return Reservation(tokens)

    def cancel(self, reservation: Reservation) -> None:
        with self._lock:
            self.reserved -= reservation.tokens

    def settle(self, reservation: Reservation, actual_tokens: int) -> None:
        with self._lock:
            self.reserved -= reservation.tokens
            self.used += actual_tokens
            over = self.used > self.limit
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

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


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
        prompt = _prompt(messages)
        try:
            input_estimate = int(
                self.client.models.count_tokens(
                    model=self.model, contents=prompt
                ).total_tokens
            )
        except Exception as error:
            code = _error_code(error)
            raise ProviderError(
                str(error), code=code, retryable=code in RETRYABLE_CODES or code is None
            ) from error

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
            self.ledger.cancel(reservation)
            code = _error_code(error)
            raise ProviderError(
                str(error), code=code, retryable=code in RETRYABLE_CODES or code is None
            ) from error

        input_tokens = int(interaction.usage.total_input_tokens)
        output_tokens = int(interaction.usage.total_output_tokens)
        self.ledger.settle(reservation, input_tokens + output_tokens)
        raw_text = interaction.output_text
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
        payload = {
            "model": self.model,
            "messages": messages,
            "format": schema.model_json_schema(),
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0, "num_predict": max_output_tokens},
        }
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        reservation = self.ledger.reserve(len(encoded) + max_output_tokens)
        request = Request(
            f"{self.base_url}/api/chat",
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            self.ledger.cancel(reservation)
            raise ProviderError(
                str(error), code=error.code, retryable=error.code in RETRYABLE_CODES
            ) from error
        except (URLError, TimeoutError) as error:
            self.ledger.cancel(reservation)
            raise ProviderError(str(error), code=None, retryable=True) from error
        except Exception as error:
            self.ledger.cancel(reservation)
            raise ProviderError(str(error), code=None, retryable=False) from error

        try:
            input_tokens = int(result["prompt_eval_count"])
            output_tokens = int(result["eval_count"])
            raw_text = result["message"]["content"]
        except (KeyError, TypeError, ValueError) as error:
            self.ledger.cancel(reservation)
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
