import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from brd_srs_testgen.models import RequirementBatch
from brd_srs_testgen.providers import (
    BudgetExceeded,
    BudgetLedger,
    GeminiProvider,
    OllamaProvider,
    StructuredOutputError,
)


class FakeModels:
    def count_tokens(self, **_kwargs):
        return SimpleNamespace(total_tokens=10)


class FakeInteractions:
    def __init__(self, text: str = '{"requirements": []}') -> None:
        self.text = text
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            output_text=self.text,
            usage=SimpleNamespace(
                total_input_tokens=10,
                total_output_tokens=5,
                total_tokens=15,
            ),
        )


def test_ledger_prevents_over_reservation() -> None:
    ledger = BudgetLedger(limit=100)
    reservation = ledger.reserve(80)

    with pytest.raises(BudgetExceeded):
        ledger.reserve(21)

    ledger.settle(reservation, actual_tokens=50)
    assert ledger.used == 50
    assert ledger.remaining == 50


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
