# Research Core Operations

## Setup and launch

Use Python 3.11 and install `uv`. The `uv` virtual environment does not include `pip`, so install dependencies with `uv`:

```sh
uv --cache-dir /tmp/citd-final-uv-cache venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
env PYTHONPATH=src .venv/bin/python -m streamlit run app.py
```

Open the local URL Streamlit prints (normally `http://localhost:8501`). Upload one text-extractable BRD/SRS PDF and run the three fixed conditions, in order: `single_prompt`, `staged_single_agent`, and `centralized_multi_agent`.

## Live smoke tests (optional; not part of the automated gate)

Use a small, non-sensitive, text-extractable PDF. Keep the selected model and token ceiling identical for every condition. The API key is transient UI input only; it is not stored under `runs/`.

### Gemini

1. Select `gemini`, use a supported Gemini model (the default is `gemini-3.6-flash`), and enter its API key.
2. Upload the PDF, set one token ceiling, and run all three conditions.
3. Confirm three condition summaries and their requirements, scenarios, test cases, RTM, and complete-bundle downloads.

### Ollama

Ollama is required only for this smoke path. Use two terminals:

Terminal 1:
```sh
ollama serve
```

After it is ready, use Terminal 2:

```sh
ollama pull gemma4
```

Leave Terminal 1 running. Select `ollama`; the local URL and model are editable (defaults: `http://localhost:11434` and `gemma4`). Run the same PDF with the same ceiling. Confirm isolated condition failures are shown when applicable, charged tokens when supplied (otherwise reported input plus output tokens), and all downloads. Ollama requests disable thinking output.

### LM Studio

Start the local server from LM Studio's Developer tab and load a model. Select
`LM Studio` and keep the default OpenAI-compatible base URL
(`http://localhost:1234/v1`). If authentication is enabled, enter an API token
created in LM Studio Server Settings, select **Load available models**, then
choose the loaded model from the dropdown.

Do not put credentials in a URL's userinfo or query string, and do not put secrets in the PDF. The UI masks Gemini and LM Studio credentials and redacts known secrets from errors.

## Persisted runs

Each run is stored under `runs/<comparison-id>/`:

```text
manifest.json
chunks.json
conditions/<condition>/manifest.json
conditions/<condition>/events.jsonl
```

Successful and deterministic-semantic-validation outputs include `requirements.json`, `scenarios.json`, `test_cases.json`, `validation.json`, `rtm.json`, and `metrics.json`. Configuration, provider rejection, transport exhaustion, timeout, budget exhaustion, and schema failures contain `metrics.json` only. Configuration failures here are errors raised after condition startup (for example, an injected provider mismatch); `ProviderSettings` preflight failures create no run directory. A PDF parsing failure is recorded at the comparison root as `failure.json`. Re-running creates a new collision-resistant comparison ID.

Terminal conditions and completed comparisons are immutable. Each finalization file is atomically replaced, and caught finalization-write failures are rolled back. This is not a full crash transaction: a process or host crash can leave an incomplete `RUNNING` condition. The complete condition bundle is an on-demand UI download, not a separately persisted file.

## Failures and recovery

The reported categories are parsing, configuration, provider rejection, transport exhaustion, timeout, budget exhaustion, schema failure, and semantic validation. A condition failure does not prevent the remaining conditions from running. Correct the configuration, model availability, or PDF and rerun; this produces a new ID. For budget exhaustion, raise the per-condition ceiling or use a smaller input within the comparison protocol. For transport exhaustion or timeout, restore connectivity, the provider, or the local model server, then rerun. Unexpected internal defects fail loudly; fix the code and rerun to a new ID, retaining any stranded `RUNNING` directory for diagnosis.

## Offline verification

Tests use fake providers and do not make live provider calls. The optional live smoke tests above are deliberately skipped by this gate:

```sh
env PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q app.py src tests
env PYTHONPATH=src .venv/bin/python -c "from brd_srs_testgen.runner import run_comparison; print('imports ok')"
git diff --check
git status --short
```

## First-slice limits

This slice does not support OCR/scanned PDFs, Excel, a 54-run scheduler or resume flow, provider statistics, or blinded human evaluation.
