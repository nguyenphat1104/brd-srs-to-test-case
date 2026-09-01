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

1. From **Runs**, select **Create new run**.
2. Select exactly one run type: `single_prompt`, `staged_single_agent`, or `centralized_multi_agent`.
3. Configure the provider and model, review or edit each agent prompt, adjust the token ceiling, and select **Continue to document**. Provider connections come from the deployment environment and are not shown in the UI.
4. Upload one text-extractable BRD/SRS PDF, then select **Generate test cases**. Only the chosen type executes.
5. A completed or failed run opens automatically. Review its test cases first, then supporting requirements and scenarios, metrics, downloads, diagnostics, and immutable per-agent settings snapshot.
6. Use **Back to runs** and select a row to reopen it. A running record left by a stopped process displays as **Interrupted**.

Each accepted generation request creates a new immutable run. Correct a problem and retry rather than editing a saved run.

## Persisted data and security boundary

The application stores normalized run data in local PostgreSQL: the source basename and document hash, extracted text chunks, run configuration and lifecycle events, metrics, validation, requirements, scenarios, test cases, citations, and traceability data.

Raw PDF bytes, provider credentials, and base URLs are never stored in PostgreSQL, downloads, URLs, or run snapshots. Provider, model, custom prompt, and token-ceiling values are stored with each run. Credentials and base URLs are loaded from deployment environment variables and never rendered in the UI. Known secrets and base URLs are redacted from displayed and persisted failures. Do not put secrets in the source PDF or in a local-provider URL.

The legacy `runs/` directory is ignored and left untouched. There is no import from that filesystem format.

## Failures and recovery

Reported failure categories are parsing, configuration, provider rejection, transport exhaustion, timeout, budget exhaustion, schema failure, and semantic validation. Failed and interrupted records remain visible in **Runs**.

- Parsing: use a text-extractable PDF; scanned image-only PDFs require OCR and are not supported.
- Configuration or provider rejection: check the selected provider, model access, base URL, and credentials, then generate a new run.
- Transport exhaustion or timeout: restore connectivity or the local model server, then generate a new run.
- Budget exhaustion: raise the token ceiling or use a smaller source, then generate a new run.
- Schema or semantic validation: inspect the saved diagnostics and retry with the corrected provider/model setup.

An unexpected internal exception can leave a running database record. The **Runs** view labels that record **Interrupted** so its saved metadata remains inspectable.

## Run configuration smoke test

1. Start the app and create a new run.
2. Select the run type before any settings or upload control appears.
3. Confirm connection fields are absent; select a model and token ceiling, then edit one prompt.
4. Create a small run and confirm its dedicated detail page opens.
5. Confirm the snapshot includes the selected provider, model, and edited prompt but omits the credential and base URL.
6. Select **Back to runs**, select the same row, and confirm its test cases and settings snapshot reopen.

### Gemini

Set `GEMINI_API_KEY` in `.env`. In the run settings step, select Gemini and choose a model from the dropdown. Single-prompt runs default to `gemini-3.5-flash`; staged runs default to `gemini-3.6-flash` because Gemini no longer accepts `gemini-2.5-flash` for new users.

### llama.cpp

Run an OpenAI-compatible llama.cpp backend that exposes `/v1/models` and `/v1/chat/completions`. Set `LLAMA_CPP_BASE_URL` in `.env` when the default `http://localhost:8080/v1` is unsuitable. The deployed Compose app defaults to `http://host.docker.internal:8081/v1`.

In the run settings step, select llama.cpp and choose one of the models reported by `/v1/models`. Multi-agent runs prefer Qwen for analysis and coverage, Gemma for test generation, and Phi for review when those models are available.

### Local multi-model execution

Local `centralized_multi_agent` runs keep requests small and predictable: requirement extraction uses consecutive evidence batches targeting about 6,000 characters, and test generation handles at most three consecutive canonical requirements per task. Local requests run one at a time so llama.cpp does not receive three simultaneous generations for the same role model. Role-specific models are still used at their respective pipeline phase.

On a 32 GB Mac, run llama.cpp in router mode with `--models-max 1 --parallel 1 --ctx-size 16384`. The agents are queued together, but only one model and one inference request are active at a time.

A run remains one immutable transaction; completed task outputs are not resumable after a later task fails. Retry/resume storage should be added only if bounded local tasks still fail often enough to justify the extra schema and UI state.

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

The full gate requires all tests to pass and the PostgreSQL storage suite to run with no skips. Compilation must exit successfully, the import check must print `imports ok`, `git diff --check` must be empty, and status must contain only the intended changes before commit. The run configuration smoke test above is manual; it is not an automated verification.

## First-slice limits

This slice does not support OCR/scanned PDFs, Excel, a 54-run scheduler or resume flow, provider statistics, or blinded human evaluation.
