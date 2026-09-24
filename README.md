# brd-srs-to-test-case

Research core for generating traceable test cases from a BRD or SRS.

[Operations guide](docs/research-core-operations.md) · [Research methodology](docs/research-methodology.md) · [System and coverage guide](static/system-and-coverage.html)

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

## Hierarchical multi-agent architecture

The coverage-first pipeline runs ordered Scouts, then one Curator, one global Scenario Architect, parallel Test Writers, an always-on Critic, and at most one targeted repair routed to the role responsible for the highest-priority finding. Each typed handoff is saved to a persisted blackboard and shown in the run detail, including task scope, role, model, summary, and raw JSON output.

Provider, model, prompt, thinking level, and output-token settings are configurable per role and saved with each run. For example:

```json
{
  "agents": {
    "scout": {
      "provider": "llama_cpp",
      "model": "qwen3",
      "prompt": "Prioritize business rules and exceptions.",
      "thinking_level": null,
      "max_output_tokens": 8000
    },
    "critic": {
      "provider": "gemini",
      "model": "gemini-3.6-flash",
      "prompt": "Find material coverage and traceability gaps.",
      "thinking_level": "high",
      "max_output_tokens": 12000
    }
  }
}
```

Changing the current defaults affects future runs only; historical manifests and their saved role settings remain unchanged.

## Reproducible coverage evaluation

For an official comparison, extract one versioned coverage catalog per source
document, inspect its quoted evidence, approve it, and reuse that same catalog
for every run type. Each run persists its test-case mappings and linked score;
an evaluator failure is stored and excluded from score comparisons rather than
treated as F1 zero. Two blinded raters score coverage, groundedness,
executability, and redundancy control before separate adjudication. The
operational reliability gate is quadratic-weighted Cohen's kappa ≥ 0.70 in
every dimension. See the [Vietnamese research methodology](docs/research-methodology.md)
for metric definitions, limitations, and public research sources.

## Existing prototypes

`app-ba.py` and `app-ba-sys-architect.py` are the earlier standalone prototypes; they do not provide the PostgreSQL-backed research-core workflow above.

```sh
pip install streamlit pandas pypdf openpyxl
ollama serve
ollama pull gemma4
python -m streamlit run app-ba.py
```

Both prototypes support Gemini and local Gemma 4 through Ollama.
