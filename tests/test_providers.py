import io
import json
import socket
import ssl
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

import httpx
import pytest

from brd_srs_testgen.models import RequirementBatch
from brd_srs_testgen.providers import (
    BudgetExceeded,
    BudgetLedger,
    GeminiProvider,
    LMStudioProvider,
    OllamaProvider,
    ProviderError,
    StructuredOutputError,
    list_lm_studio_models,
)


class FakeModels:
    def count_tokens(self, **_kwargs):
        return SimpleNamespace(total_tokens=10)


class FakeInteractions:
    def __init__(
        self,
        text: str = '{"requirements": []}',
        usage: SimpleNamespace | None = None,
    ) -> None:
        self.text = text
        self.usage = usage or SimpleNamespace(
            total_input_tokens=10,
            total_output_tokens=5,
            total_tokens=15,
        )
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            output_text=self.text,
            usage=self.usage,
        )


class StatusError(Exception):
    def __init__(self, code: int) -> None:
        self.code = code


APIConnectionError = type(
    "APIConnectionError",
    (Exception,),
    {"__module__": "google.genai._gaos.lib.compat_errors"},
)
APITimeoutError = type(
    "APITimeoutError",
    (Exception,),
    {"__module__": "google.genai._gaos.lib.compat_errors"},
)


class RaisingModels:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def count_tokens(self, **_kwargs):
        raise self.error


class RaisingInteractions:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def create(self, **_kwargs):
        raise self.error


def transient_errors() -> list[Exception]:
    request = httpx.Request("GET", "https://example.com")
    return [
        TimeoutError("timed out"),
        ConnectionError("disconnected"),
        httpx.ConnectError("connection failed", request=request),
        APIConnectionError(),
        APITimeoutError(),
        StatusError(408),
        StatusError(503),
    ]


def test_ledger_prevents_over_reservation() -> None:
    ledger = BudgetLedger(limit=100)
    reservation = ledger.reserve(80)

    with pytest.raises(BudgetExceeded):
        ledger.reserve(21)

    ledger.settle(reservation, actual_tokens=50)
    assert ledger.used == 50
    assert ledger.remaining == 50


@pytest.mark.parametrize(
    "kwargs",
    [
        {"limit": True},
        {"limit": 1.5},
        {"limit": 0},
        {"used": True},
        {"used": 1.5},
        {"used": -1},
        {"limit": 1, "used": 2},
    ],
)
def test_ledger_rejects_invalid_initial_values(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {"limit": 10}
    values.update(kwargs)
    with pytest.raises(ValueError):
        BudgetLedger(**values)


@pytest.mark.parametrize("field", ("reserved", "_lock", "_active"))
def test_ledger_internal_state_cannot_be_injected(field: str) -> None:
    with pytest.raises(TypeError):
        BudgetLedger(limit=10, **{field: object()})


@pytest.mark.parametrize("tokens", (True, 1.5, 0, -1))
def test_ledger_rejects_invalid_reservations(tokens: object) -> None:
    ledger = BudgetLedger(limit=10)

    with pytest.raises(ValueError, match="positive integer"):
        ledger.reserve(tokens)

    assert ledger.used == 0
    assert ledger.reserved == 0


def test_ledger_uses_authoritative_amount_for_tampered_reservations() -> None:
    ledger = BudgetLedger(limit=100)
    reservation = ledger.reserve(10)

    with pytest.raises(ValueError, match="active"):
        ledger.settle(replace(reservation, tokens=99), 5)

    assert ledger.used == 0
    assert ledger.reserved == 10
    ledger.cancel(reservation)


def test_ledger_rejects_reused_and_wrong_ledger_reservations() -> None:
    ledger = BudgetLedger(limit=100)
    other = BudgetLedger(limit=100)
    reservation = ledger.reserve(10)

    with pytest.raises(ValueError, match="active"):
        other.cancel(reservation)

    assert ledger.reserved == 10
    ledger.cancel(reservation)
    with pytest.raises(ValueError, match="active"):
        ledger.cancel(reservation)
    assert ledger.used == 0
    assert ledger.reserved == 0


def test_ledger_rejects_reused_settlement_and_invalid_actual_tokens() -> None:
    ledger = BudgetLedger(limit=100)
    reservation = ledger.reserve(10)

    for actual_tokens in (True, 1.5, -1):
        with pytest.raises(ValueError, match="nonnegative integer"):
            ledger.settle(reservation, actual_tokens)
        assert ledger.used == 0
        assert ledger.reserved == 10

    ledger.settle(reservation, 5)
    with pytest.raises(ValueError, match="active"):
        ledger.settle(reservation, 5)
    assert ledger.used == 5
    assert ledger.reserved == 0


def test_ledger_settlement_preserves_concurrent_reservations() -> None:
    ledger = BudgetLedger(limit=100)
    first = ledger.reserve(60)
    second = ledger.reserve(40)

    with pytest.raises(BudgetExceeded):
        ledger.settle(first, 70)

    assert ledger.used == 70
    assert ledger.reserved == 40
    assert ledger.remaining == -10
    ledger.cancel(second)


def test_gemini_uses_structured_output_and_records_usage() -> None:
    interactions = FakeInteractions()
    client = SimpleNamespace(models=FakeModels(), interactions=interactions)
    ledger = BudgetLedger(limit=100)
    provider = GeminiProvider(client, "gemini-test", ledger)

    result = provider.generate(
        [{"role": "user", "content": "Extract requirements"}],
        RequirementBatch,
        max_output_tokens=40,
    )

    assert result.value.requirements == []
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert ledger.used == 15
    assert interactions.kwargs["response_format"]["mime_type"] == "application/json"
    assert interactions.kwargs["generation_config"]["temperature"] == 0.0


def test_gemini_charges_reported_total_tokens() -> None:
    interactions = FakeInteractions(
        usage=SimpleNamespace(
            total_input_tokens=10,
            total_output_tokens=5,
            total_tokens=21,
        )
    )
    client = SimpleNamespace(models=FakeModels(), interactions=interactions)
    ledger = BudgetLedger(limit=100)
    provider = GeminiProvider(client, "gemini-test", ledger)

    result = provider.generate(
        [{"role": "user", "content": "Extract requirements"}],
        RequirementBatch,
        max_output_tokens=40,
    )

    assert result.total_tokens == 21
    assert ledger.used == 21


def test_gemini_does_not_undercharge_reported_total_tokens() -> None:
    interactions = FakeInteractions(
        usage=SimpleNamespace(
            total_input_tokens=10,
            total_output_tokens=5,
            total_tokens=2,
        )
    )
    client = SimpleNamespace(models=FakeModels(), interactions=interactions)
    ledger = BudgetLedger(limit=100)
    provider = GeminiProvider(client, "gemini-test", ledger)

    result = provider.generate(
        [{"role": "user", "content": "Extract requirements"}],
        RequirementBatch,
        max_output_tokens=40,
    )

    assert result.total_tokens == 15
    assert ledger.used == 15


def test_gemini_uses_input_and_output_when_total_is_absent() -> None:
    interactions = FakeInteractions(
        usage=SimpleNamespace(total_input_tokens=10, total_output_tokens=5)
    )
    client = SimpleNamespace(models=FakeModels(), interactions=interactions)
    ledger = BudgetLedger(limit=100)
    provider = GeminiProvider(client, "gemini-test", ledger)

    result = provider.generate(
        [{"role": "user", "content": "Extract requirements"}],
        RequirementBatch,
        max_output_tokens=40,
    )

    assert result.total_tokens == 15
    assert ledger.used == 15


@pytest.mark.parametrize("max_output_tokens", (0, -1))
def test_providers_reject_invalid_max_before_adapter_action(
    max_output_tokens: int,
) -> None:
    models = SimpleNamespace(count_tokens=Mock())
    gemini = GeminiProvider(
        SimpleNamespace(models=models, interactions=FakeInteractions()),
        "gemini-test",
        BudgetLedger(limit=100),
    )
    ollama = OllamaProvider("http://localhost:11434", "gemma4", BudgetLedger(100))
    lm_studio = LMStudioProvider(
        "http://localhost:1234/v1", "local-model", BudgetLedger(100)
    )
    messages = [{"role": "user", "content": "Extract requirements"}]

    with pytest.raises(ValueError, match="positive integer"):
        gemini.generate(messages, RequirementBatch, max_output_tokens=max_output_tokens)
    with patch("brd_srs_testgen.providers.urlopen") as opened:
        with pytest.raises(ValueError, match="positive integer"):
            ollama.generate(messages, RequirementBatch, max_output_tokens=max_output_tokens)
        with pytest.raises(ValueError, match="positive integer"):
            lm_studio.generate(
                messages, RequirementBatch, max_output_tokens=max_output_tokens
            )

    models.count_tokens.assert_not_called()
    opened.assert_not_called()


def test_ollama_posts_schema_and_reads_token_counts() -> None:
    response = io.BytesIO(
        json.dumps(
            {
                "message": {"content": '{"requirements": []}'},
                "prompt_eval_count": 8,
                "eval_count": 4,
            }
        ).encode()
    )
    ledger = BudgetLedger(limit=10_000)
    provider = OllamaProvider("http://localhost:11434/", "gemma4", ledger)

    with patch("brd_srs_testgen.providers.urlopen", return_value=response) as opened:
        result = provider.generate(
            [{"role": "user", "content": "Extract requirements"}],
            RequirementBatch,
            max_output_tokens=40,
        )

    request = opened.call_args.args[0]
    payload = json.loads(request.data)
    assert request.full_url == "http://localhost:11434/api/chat"
    assert payload["stream"] is False
    assert payload["format"]["type"] == "object"
    assert result.total_tokens == 12
    assert ledger.used == 12


def test_lm_studio_posts_openai_schema_auth_and_reads_usage() -> None:
    models = io.BytesIO(
        b'{"models":[{"key":"local-model","max_context_length":4096,"loaded_instances":[]}]}'
    )
    loaded = io.BytesIO(b'{"status":"loaded"}')
    response = io.BytesIO(
        json.dumps(
            {
                "choices": [
                    {"message": {"content": '{"requirements": []}'}}
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4},
            }
        ).encode()
    )
    ledger = BudgetLedger(limit=10_000)
    provider = LMStudioProvider(
        "http://localhost:1234/v1/",
        "local-model",
        ledger,
        api_key="local-token",
    )

    with patch(
        "brd_srs_testgen.providers.urlopen", side_effect=[models, loaded, response]
    ) as opened:
        result = provider.generate(
            [{"role": "user", "content": "Extract requirements"}],
            RequirementBatch,
            max_output_tokens=40,
        )

    models_request = opened.call_args_list[0].args[0]
    assert models_request.full_url == "http://localhost:1234/api/v1/models"

    load_request = opened.call_args_list[1].args[0]
    assert load_request.full_url == "http://localhost:1234/api/v1/models/load"
    assert load_request.get_header("Authorization") == "Bearer local-token"
    assert json.loads(load_request.data)["model"] == "local-model"
    assert json.loads(load_request.data)["context_length"] >= 40

    request = opened.call_args_list[2].args[0]
    payload = json.loads(request.data)
    assert request.full_url == "http://localhost:1234/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer local-token"
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert payload["reasoning_effort"] == "none"
    assert payload["max_tokens"] == 40
    assert result.total_tokens == 12
    assert ledger.used == 12


def test_llama_cpp_posts_openai_schema_without_lm_studio_model_management() -> None:
    response = io.BytesIO(
        b'{"choices":[{"message":{"content":"{\\"requirements\\":[]}"}}],"usage":{"prompt_tokens":8,"completion_tokens":4}}'
    )
    provider = LMStudioProvider(
        "http://localhost:8080/v1",
        "local",
        BudgetLedger(10_000),
        auto_load=False,
    )

    with patch("brd_srs_testgen.providers.urlopen", return_value=response) as opened:
        result = provider.generate(
            [{"role": "user", "content": "Extract requirements"}],
            RequirementBatch,
            max_output_tokens=100,
        )

    assert result.value.requirements == []
    assert opened.call_count == 1
    assert opened.call_args.args[0].full_url == "http://localhost:8080/v1/chat/completions"


def test_lm_studio_reloads_an_insufficient_context_once() -> None:
    models = io.BytesIO(
        b'{"models":[{"key":"local-model","max_context_length":4096,"loaded_instances":[{"id":"local-model:1","config":{"context_length":512}}]}]}'
    )
    unloaded = io.BytesIO(b'{"instance_id":"local-model:1"}')
    loaded = io.BytesIO(b'{"status":"loaded"}')
    response = io.BytesIO(
        b'{"choices":[{"message":{"content":"{\\"requirements\\":[]}"}}],"usage":{"prompt_tokens":8,"completion_tokens":4}}'
    )
    provider = LMStudioProvider(
        "http://localhost:1234/v1", "local-model", BudgetLedger(10_000)
    )

    with patch(
        "brd_srs_testgen.providers.urlopen",
        side_effect=[models, unloaded, loaded, response],
    ) as opened:
        provider.generate(
            [{"role": "user", "content": "Extract requirements"}],
            RequirementBatch,
            max_output_tokens=40,
        )

    requests = [call.args[0] for call in opened.call_args_list]
    assert [request.full_url for request in requests] == [
        "http://localhost:1234/api/v1/models",
        "http://localhost:1234/api/v1/models/unload",
        "http://localhost:1234/api/v1/models/load",
        "http://localhost:1234/v1/chat/completions",
    ]
    assert json.loads(requests[1].data) == {"instance_id": "local-model:1"}


def test_lm_studio_model_load_error_keeps_server_message() -> None:
    error = HTTPError(
        "http://localhost:1234/api/v1/models/load",
        400,
        "Bad Request",
        None,
        io.BytesIO(b'{"error":{"message":"Model needs 8 GB free RAM."}}'),
    )
    ledger = BudgetLedger(limit=100)
    provider = LMStudioProvider(
        "http://localhost:1234/v1", "local-model", ledger
    )

    with patch("brd_srs_testgen.providers.urlopen", side_effect=error):
        with pytest.raises(ProviderError, match="8 GB free RAM"):
            provider.generate(
                [{"role": "user", "content": "Extract requirements"}],
                RequirementBatch,
                max_output_tokens=40,
            )

    assert ledger.used == 0


def test_lm_studio_lists_models_with_auth() -> None:
    response = io.BytesIO(
        json.dumps(
            {"data": [{"id": "gemma-3-12b"}, {"id": "gemma-3-4b"}]}
        ).encode()
    )

    with patch("brd_srs_testgen.providers.urlopen", return_value=response) as opened:
        models = list_lm_studio_models(
            "http://localhost:1234/v1", "local-token"
        )

    request = opened.call_args.args[0]
    assert request.full_url == "http://localhost:1234/v1/models"
    assert request.get_header("Authorization") == "Bearer local-token"
    assert models == ["gemma-3-12b", "gemma-3-4b"]


def test_invalid_json_is_charged_before_schema_error() -> None:
    interactions = FakeInteractions("not-json")
    client = SimpleNamespace(models=FakeModels(), interactions=interactions)
    ledger = BudgetLedger(limit=100)
    provider = GeminiProvider(client, "gemini-test", ledger)

    with pytest.raises(StructuredOutputError, match="structured output"):
        provider.generate(
            [{"role": "user", "content": "Extract"}],
            RequirementBatch,
            max_output_tokens=40,
        )

    assert ledger.used == 15


def test_gemini_missing_usage_charges_reservation() -> None:
    client = SimpleNamespace(
        models=FakeModels(),
        interactions=SimpleNamespace(create=lambda **_kwargs: SimpleNamespace()),
    )
    ledger = BudgetLedger(limit=100)
    provider = GeminiProvider(client, "gemini-test", ledger)

    with pytest.raises(ProviderError, match="incomplete response") as error:
        provider.generate(
            [{"role": "user", "content": "Extract"}],
            RequirementBatch,
            max_output_tokens=40,
        )

    assert error.value.retryable is False
    assert ledger.used == 50
    assert ledger.reserved == 0


def test_gemini_charges_usage_for_missing_output() -> None:
    client = SimpleNamespace(
        models=FakeModels(),
        interactions=SimpleNamespace(
            create=lambda **_kwargs: SimpleNamespace(
                usage=SimpleNamespace(
                    total_input_tokens=10,
                    total_output_tokens=5,
                    total_tokens=15,
                )
            )
        ),
    )
    ledger = BudgetLedger(limit=100)
    provider = GeminiProvider(client, "gemini-test", ledger)

    with pytest.raises(ProviderError, match="incomplete response") as error:
        provider.generate(
            [{"role": "user", "content": "Extract"}],
            RequirementBatch,
            max_output_tokens=40,
        )

    assert error.value.retryable is False
    assert ledger.used == 15
    assert ledger.reserved == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_input_tokens", -1),
        ("total_output_tokens", 1.5),
        ("total_tokens", True),
    ],
)
def test_gemini_invalid_usage_charges_reservation(field: str, value: object) -> None:
    usage = SimpleNamespace(
        total_input_tokens=10,
        total_output_tokens=5,
        total_tokens=15,
    )
    setattr(usage, field, value)
    client = SimpleNamespace(
        models=FakeModels(), interactions=FakeInteractions(usage=usage)
    )
    ledger = BudgetLedger(limit=100)
    provider = GeminiProvider(client, "gemini-test", ledger)

    with pytest.raises(ProviderError, match="incomplete response") as error:
        provider.generate(
            [{"role": "user", "content": "Extract"}],
            RequirementBatch,
            max_output_tokens=40,
        )

    assert error.value.retryable is False
    assert ledger.used == 50
    assert ledger.reserved == 0


@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        (ValueError("bad input"), False),
        *((error, True) for error in transient_errors()),
    ],
)
def test_gemini_count_token_errors_are_classified(
    error: Exception, retryable: bool
) -> None:
    client = SimpleNamespace(models=RaisingModels(error), interactions=FakeInteractions())
    provider = GeminiProvider(client, "gemini-test", BudgetLedger(limit=100))

    with pytest.raises(ProviderError) as raised:
        provider.generate(
            [{"role": "user", "content": "Extract"}],
            RequirementBatch,
            max_output_tokens=40,
        )

    assert raised.value.retryable is retryable


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("timed out"),
        httpx.ReadTimeout(
            "timed out", request=httpx.Request("GET", "https://example.com")
        ),
        APITimeoutError("timed out"),
        StatusError(408),
    ],
)
def test_gemini_timeout_errors_preserve_timeout_identity(error: Exception) -> None:
    client = SimpleNamespace(
        models=RaisingModels(error), interactions=FakeInteractions()
    )
    provider = GeminiProvider(client, "gemini-test", BudgetLedger(limit=100))

    with pytest.raises(ProviderError) as raised:
        provider.generate(
            [{"role": "user", "content": "Extract"}],
            RequirementBatch,
            max_output_tokens=40,
        )

    assert raised.value.retryable is True
    assert raised.value.timed_out is True


@pytest.mark.parametrize(
    ("error", "retryable", "used"),
    [
        (TypeError("bad response"), False, 0),
        *((error, True, 50) for error in transient_errors()),
    ],
)
def test_gemini_interaction_errors_are_classified(
    error: Exception, retryable: bool, used: int
) -> None:
    client = SimpleNamespace(
        models=FakeModels(), interactions=RaisingInteractions(error)
    )
    ledger = BudgetLedger(limit=100)
    provider = GeminiProvider(client, "gemini-test", ledger)

    with pytest.raises(ProviderError) as raised:
        provider.generate(
            [{"role": "user", "content": "Extract"}],
            RequirementBatch,
            max_output_tokens=40,
        )

    assert raised.value.retryable is retryable
    assert ledger.used == used
    assert ledger.reserved == 0


@pytest.mark.parametrize(
    ("reason", "retryable"),
    [
        (TimeoutError("timed out"), True),
        (socket.gaierror(socket.EAI_AGAIN, "DNS lookup failed"), True),
        (socket.gaierror(socket.EAI_NONAME, "DNS name not found"), False),
        (ssl.SSLError("certificate verify failed"), False),
    ],
)
def test_ollama_url_errors_charge_reservation_and_classify_reason(
    reason: Exception, retryable: bool
) -> None:
    ledger = BudgetLedger(limit=10_000)
    provider = OllamaProvider("http://localhost:11434", "gemma4", ledger)

    with patch("brd_srs_testgen.providers.urlopen", side_effect=URLError(reason)) as opened:
        with pytest.raises(ProviderError) as error:
            provider.generate(
                [{"role": "user", "content": "Extract"}],
                RequirementBatch,
                max_output_tokens=40,
            )

    assert error.value.retryable is retryable
    assert error.value.timed_out is isinstance(reason, TimeoutError)
    assert ledger.used == len(opened.call_args.args[0].data) + 40
    assert ledger.reserved == 0


def test_ollama_missing_usage_charges_reservation() -> None:
    ledger = BudgetLedger(limit=10_000)
    provider = OllamaProvider("http://localhost:11434", "gemma4", ledger)

    with patch(
        "brd_srs_testgen.providers.urlopen", return_value=io.BytesIO(b"{}")
    ) as opened:
        with pytest.raises(ProviderError, match="incomplete response"):
            provider.generate(
                [{"role": "user", "content": "Extract"}],
                RequirementBatch,
                max_output_tokens=40,
            )

    assert ledger.used == len(opened.call_args.args[0].data) + 40
    assert ledger.reserved == 0


def test_ollama_invalid_url_does_not_reserve_or_open() -> None:
    ledger = BudgetLedger(limit=10_000)
    provider = OllamaProvider("http://bad host", "gemma4", ledger)

    with patch("brd_srs_testgen.providers.urlopen") as opened:
        with pytest.raises(ProviderError, match="Invalid Ollama URL") as error:
            provider.generate(
                [{"role": "user", "content": "Extract"}],
                RequirementBatch,
                max_output_tokens=40,
            )

    assert error.value.retryable is False
    assert ledger.used == 0
    assert ledger.reserved == 0
    opened.assert_not_called()
