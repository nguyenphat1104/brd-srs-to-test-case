"""Local LLM provider helpers."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def generate_with_ollama(prompt: str, model: str, base_url: str) -> str:
    """Generate text from a locally running Ollama model."""
    request = Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps({"model": model, "prompt": prompt, "stream": False}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=600) as response:
            result = json.load(response)
    except (HTTPError, URLError) as error:
        raise RuntimeError(
            f"Không thể kết nối Ollama tại {base_url}. Hãy chạy `ollama serve`."
        ) from error

    text = result.get("response")
    if not isinstance(text, str):
        raise RuntimeError("Ollama không trả về nội dung văn bản hợp lệ.")
    return text
