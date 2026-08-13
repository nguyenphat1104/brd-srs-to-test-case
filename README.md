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

The home page lists PostgreSQL-backed runs newest first. Use **Settings** to save provider defaults in this browser, then choose **Create new run** to upload one PDF and execute exactly one generation type. Selecting a saved row opens its detailed test cases and immutable configuration snapshot.

## Existing prototypes

`app-ba.py` and `app-ba-sys-architect.py` are the earlier standalone prototypes; they do not provide the PostgreSQL-backed research-core workflow above.

```sh
pip install streamlit pandas pypdf openpyxl
ollama serve
ollama pull gemma4
python -m streamlit run app-ba.py
```

Both prototypes support Gemini and local Gemma 4 through Ollama.
