# Research Core Operations

## Setup and launch

Use Python 3.11 and install `uv`. Preserve an existing `.env`; the conditional copy creates it only on first setup. Ensure it contains the `DATABASE_URL` from `.env.example`.

```sh
test -f .env || cp .env.example .env
docker compose up -d --wait db
uv --cache-dir /tmp/citd-final-uv-cache venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
env PYTHONPATH=src .venv/bin/python -m streamlit run app.py
```

Open the local URL Streamlit prints, normally `http://localhost:8501`.

To stop the database without deleting run history:

```sh
docker compose down
```

PostgreSQL data remains in the Compose-managed `postgres_data` named volume.

## Generate and reopen a run

1. Configure a provider, model, credentials, and token ceiling.
2. Select exactly one run type: `single_prompt`, `staged_single_agent`, or `centralized_multi_agent`.
3. Upload one text-extractable BRD/SRS PDF and click **Generate test cases**. Only the selected run type executes.
4. Review the full requirements, scenarios, test cases, steps, expected results, test data, citations, metrics, and downloads in **Results**. Validation details are included in the diagnostics or complete-bundle download.
5. Open **Run history** to reopen completed or failed runs. A running record left by a stopped process is displayed as **Interrupted** and can also be reopened for its available diagnostics.

Each click creates a new run. Correct a problem and generate again rather than modifying a saved run.

## Persisted data and security boundary

The application stores normalized run data in local PostgreSQL: the source basename and document hash, extracted text chunks, run configuration and lifecycle events, metrics, validation, requirements, scenarios, test cases, citations, and traceability data.

Raw PDF bytes and provider credentials are never stored in PostgreSQL or **Run history**. Credentials configured in `.env` remain in that local disk file; values entered in the UI are transient. Known secrets are redacted from displayed failures. Do not put secrets in the source PDF or in a local-provider URL.

The legacy `runs/` directory is ignored and left untouched. There is no import from that filesystem format.

## Failures and recovery

Reported failure categories are parsing, configuration, provider rejection, transport exhaustion, timeout, budget exhaustion, schema failure, and semantic validation. Failed and interrupted records remain visible in **Run history**.

- Parsing: use a text-extractable PDF; scanned image-only PDFs require OCR and are not supported.
- Configuration or provider rejection: check the selected provider, model access, base URL, and credentials, then generate a new run.
- Transport exhaustion or timeout: restore connectivity or the local model server, then generate a new run.
- Budget exhaustion: raise the token ceiling or use a smaller source, then generate a new run.
- Schema or semantic validation: inspect the saved diagnostics and retry with the corrected provider/model setup.

An unexpected internal exception can leave a running database record. The history UI labels that record **Interrupted** so its saved metadata remains inspectable.

## Live smoke tests (optional)

Use a small, non-sensitive, text-extractable PDF. Select one run type and confirm only that type runs. After generation, verify the detailed result, then reopen the same row from **Run history**.

### Gemini

Select `gemini`, use a supported model (the current default is `gemini-3.6-flash`), and enter its API key. Generate one selected run and verify its details and downloads.

### Ollama

Ollama is required only for this smoke path. Start the service and make the model available:

```sh
ollama serve
```

In another terminal:

```sh
ollama pull gemma4
```

Select `ollama`; the editable defaults are `http://localhost:11434` and `gemma4`. Generate one selected run and verify the result and saved history row. Ollama requests disable thinking output.

### LM Studio

Start the local server from LM Studio's Developer tab and load a model. Select `LM Studio` and keep the default OpenAI-compatible base URL, `http://localhost:1234/v1`. If authentication is enabled, enter a token created in LM Studio Server Settings. Select **Load available models**, choose the loaded model, generate one selected run, and verify the result and saved history row.

## Offline verification

Tests use fake providers and make no live provider calls. `TEST_DATABASE_URL` must target the dedicated local `brd_srs_test` database; never point it at the application database or a remote database. Source the existing `.env` so the current local DSN is exported:

```sh
set -a
. ./.env
set +a
env PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q app.py src tests
env PYTHONPATH=src .venv/bin/python -c "from brd_srs_testgen.runner import run_generation; print('imports ok')"
git diff --check
git status --short
```

The full gate requires all tests to pass and the PostgreSQL storage suite to run with no skips. Compilation must exit successfully, the import check must print `imports ok`, `git diff --check` must be empty, and status must contain only the intended changes before commit.

## First-slice limits

This slice does not support OCR/scanned PDFs, Excel, a 54-run scheduler or resume flow, provider statistics, or blinded human evaluation.
