# brd-srs-to-test-case

Research core for generating traceable test cases from a BRD or SRS.

[Operations guide](docs/research-core-operations.md)

## Research core quick start

Create `.env` only when it does not already exist; preserve existing local credentials and settings.

```sh
test -f .env || cp .env.example .env
docker compose up -d --wait db
uv --cache-dir /tmp/citd-final-uv-cache venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
env PYTHONPATH=src .venv/bin/python -m streamlit run app.py
```

The home page lists PostgreSQL-backed runs newest first. Choose **Create new run**, select a run type, configure its agents, then upload one PDF. The UI offers Gemini and a local llama.cpp backend with provider-aware model dropdowns. Single-prompt runs default to Gemini 3.5 Flash, staged runs default to Gemini 2.5 Flash, and multi-agent runs default to llama.cpp using the first model reported by its API. Provider credentials and base URLs come from `.env`; users adjust only the provider, model, prompts, and token ceiling. Every run stores an immutable settings snapshot without connection details.

## Existing prototypes

`app-ba.py` and `app-ba-sys-architect.py` are the earlier standalone prototypes; they do not provide the PostgreSQL-backed research-core workflow above.

```sh
pip install streamlit pandas pypdf openpyxl
ollama serve
ollama pull gemma4
python -m streamlit run app-ba.py
```

Both prototypes support Gemini and local Gemma 4 through Ollama.
