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

To stop the database without deleting saved runs:

```sh
docker compose down
```

PostgreSQL data remains in the Compose-managed `postgres_data` named volume.

## Generate and reopen a run

1. Open **Settings** to set the provider, model, applicable credential, base URL, and token ceiling, then explicitly select **Save settings**.
2. From **Runs**, select **Create new run**.
3. Select exactly one run type: `single_prompt`, `staged_single_agent`, or `centralized_multi_agent`; upload one text-extractable BRD/SRS PDF; then select **Generate test cases**. Only the chosen type executes.
4. A completed or failed run opens automatically. Review its test cases first, then supporting requirements and scenarios, metrics, downloads, diagnostics, and immutable configuration snapshot.
5. Use **Back to runs** and select a row to reopen it. A running record left by a stopped process displays as **Interrupted**.

Each accepted generation request creates a new immutable run. Correct a problem and retry rather than editing a saved run.

## Persisted data and security boundary

The application stores normalized run data in local PostgreSQL: the source basename and document hash, extracted text chunks, run configuration and lifecycle events, metrics, validation, requirements, scenarios, test cases, citations, and traceability data.

Raw PDF bytes and provider credentials are never stored in PostgreSQL, downloads, URLs, or run snapshots. **Save settings** stores the active credential in this browser's `localStorage` only when explicitly selected; same-origin scripts can read it. Use a dedicated browser profile and origin, and do not use a shared machine. Known secrets and base URLs are redacted from displayed and persisted failures. Do not put secrets in the source PDF or in a local-provider URL.

The legacy `runs/` directory is ignored and left untouched. There is no import from that filesystem format.

## Failures and recovery

Reported failure categories are parsing, configuration, provider rejection, transport exhaustion, timeout, budget exhaustion, schema failure, and semantic validation. Failed and interrupted records remain visible in **Runs**.

- Parsing: use a text-extractable PDF; scanned image-only PDFs require OCR and are not supported.
- Configuration or provider rejection: check the selected provider, model access, base URL, and credentials, then generate a new run.
- Transport exhaustion or timeout: restore connectivity or the local model server, then generate a new run.
- Budget exhaustion: raise the token ceiling or use a smaller source, then generate a new run.
- Schema or semantic validation: inspect the saved diagnostics and retry with the corrected provider/model setup.

An unexpected internal exception can leave a running database record. The **Runs** view labels that record **Interrupted** so its saved metadata remains inspectable.

## Browser storage smoke test

1. Start the app and open **Settings**.
2. Save a non-production credential, model, applicable URL, and token ceiling.
3. Refresh and confirm the settings are restored.
4. Create a small run and confirm its dedicated detail page opens.
5. Confirm the snapshot and downloaded JSON omit the credential and base URL.
6. Select **Back to runs**, select the same row, and confirm its test cases reopen.

### Gemini

In **Settings**, select `gemini`, use a supported model (the current default is `gemini-3.6-flash`), enter its API key, and select **Save settings**. Create one run, then verify its detail page and download.

### Ollama

Ollama is required only for this smoke path. Start the service and make the model available:

```sh
ollama serve
```

In another terminal:

```sh
ollama pull gemma4
```

In **Settings**, select `ollama`; the editable defaults are `http://localhost:11434` and `gemma4`. Select **Save settings**, create one run, and verify its detail page and saved **Runs** row. Ollama requests disable thinking output.

### LM Studio

Start the local server from LM Studio's Developer tab and load a model. In **Settings**, select `LM Studio` and keep the default OpenAI-compatible base URL, `http://localhost:1234/v1`. If authentication is enabled, enter a token created in LM Studio Server Settings. Select **Load available models**, choose the loaded model, select **Save settings**, create one run, and verify its detail page and saved **Runs** row.

For `centralized_multi_agent`, optional Analyst, Test generator, and Reviewer model IDs route each role to a different model; blank fields use the primary Model. LM Studio does not load these IDs on demand, so load every selected model before generating. All roles share the configured token ceiling.

## Offline verification

Tests use fake providers and make no live provider calls. `TEST_DATABASE_URL` must target the dedicated local `brd_srs_test` database; never point it at the application database or a remote database. Source the existing `.env` so the current local DSN is exported:

```sh
set -a
. ./.env
set +a
env PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q app.py src tests
env PYTHONPATH=src .venv/bin/python -c "from brd_srs_testgen.browser_settings import AppSettings; from brd_srs_testgen.runner import run_generation; print('imports ok')"
git diff --check
git status --short
```

The full gate requires all tests to pass and the PostgreSQL storage suite to run with no skips. Compilation must exit successfully, the import check must print `imports ok`, `git diff --check` must be empty, and status must contain only the intended changes before commit. The browser storage smoke test above is manual; it is not an automated verification.

## First-slice limits

This slice does not support OCR/scanned PDFs, Excel, a 54-run scheduler or resume flow, provider statistics, or blinded human evaluation.
