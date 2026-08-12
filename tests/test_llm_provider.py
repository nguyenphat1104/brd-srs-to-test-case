import io
import json
import unittest
from unittest.mock import patch

from llm_provider import generate_with_ollama


class OllamaProviderTests(unittest.TestCase):
    def test_uses_ollama_generate_endpoint_without_streaming(self) -> None:
        with patch("llm_provider.urlopen", return_value=io.BytesIO(b'{"response": "ok"}')) as mock:
            self.assertEqual(
                generate_with_ollama("prompt", "gemma4", "http://localhost:11434/"),
                "ok",
            )

        request = mock.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:11434/api/generate")
        self.assertEqual(json.loads(request.data), {"model": "gemma4", "prompt": "prompt", "stream": False})
