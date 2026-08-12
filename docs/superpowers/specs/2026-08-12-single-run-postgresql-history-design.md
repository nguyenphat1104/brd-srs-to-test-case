# Single-Run PostgreSQL History Design

**Date:** 2026-08-12
**Status:** Approved

## 1. Purpose

Replace the fixed three-condition comparison workflow with independent generation runs. A user selects exactly one generation type, executes it against one text-extractable PDF, and can later reopen the complete result from local PostgreSQL history.

The three available run types remain:

- `single_prompt`;
- `staged_single_agent`; and
- `centralized_multi_agent`.

Each run executes only the selected pipeline. It does not start either of the other pipelines.

## 2. Success criteria

The feature is complete when:

1. the run form requires exactly one of the three run types;
2. one submission invokes only the selected pipeline;
3. every started run is persisted in local PostgreSQL as running, completed, or failed;
4. a persisted running record is shown as interrupted when reopened;
5. history survives Streamlit and machine restarts through a Docker Compose PostgreSQL volume;
6. selecting a history entry reconstructs and displays its complete result;
7. the result view shows detailed requirements, scenarios, and test cases, including every ordered test step and source citation;
8. credentials and original PDF bytes are never stored;
9. new runs are not written to the legacy `runs/` directory; and
10. existing filesystem histories are neither imported nor displayed.

## 3. Scope

### In scope

- Local PostgreSQL supplied through Docker Compose.
- `DATABASE_URL` application configuration.
- One run type per execution.
- Fully normalized storage for run configuration, extracted evidence, generated artifacts, validation, metrics, and lifecycle events.
- Newest-first run history.
- Reopening a saved run in the existing result workspace.
- Detailed on-screen test-case inspection and existing JSON downloads.
- Completed, failed, and interrupted history entries.

### Out of scope

- Running all three types from one action.
- Comparing multiple historical runs.
- Importing existing `runs/` directories.
- Continuing filesystem writes for new runs.
- Storing raw PDF bytes or provider credentials.
- Editing generated artifacts.
- Deleting runs from the UI.
- Pagination, search, tags, or retention policies.
- A general-purpose database migration framework.

## 4. User experience

### 4.1 Configure and run

The existing provider, model, credentials, base URL, and token-ceiling controls remain. Configuration adds one required **Run type** selector using the three existing generation types.

The run action is renamed from **Run comparison** to **Generate test cases**. After basic form validation, the app passes the uploaded filename, PDF bytes, provider settings, and selected run type to the runner.

The runner creates the database run before parsing or provider work begins. Parsing, provider, budget, schema, timeout, transport, and semantic-validation failures therefore remain visible in history. Invalid form input is not a started run and is not persisted.

### 4.2 Current result

The Results tab displays the newly completed or failed run. It no longer renders three condition tabs or a cross-condition summary table.

For a successful run, the result view shows:

- run type and configuration;
- volume, token, latency, quality, and traceability metrics;
- requirements and their descriptions, priority, dependencies, ambiguities, and citations;
- scenarios and their objective, type, preconditions, requirement links, and citations; and
- test cases with title, priority, scenario, requirement links, preconditions, test data, ordered action/expected-result steps, and citations.

Long artifact collections use collapsed expanders so the result remains scannable. Existing traceability-matrix and complete-bundle downloads remain available.

### 4.3 Run history

A fourth workflow tab, **Run history**, lists PostgreSQL records newest first. The list shows:

- start time;
- source filename;
- run type;
- provider and model;
- displayed status; and
- requirement, scenario, and test-case counts when metrics exist.

The persisted statuses remain `running`, `completed`, and `failed`. A history record still marked `running` is labeled **Interrupted** because this application has no background resume worker.

Selecting one row loads that run and renders the same complete result component used by the Results tab. History does not modify the current form configuration.

## 5. Domain model and runner

The comparison aggregate is removed from the active workflow:

- `Condition` becomes the user-selected `RunType` with the same three values.
- `ComparisonManifest` is replaced by `RunManifest` containing one run type and one lifecycle status.
- `ComparisonResult` and its condition map are replaced by `RunResult`, which directly contains one manifest, bundle, validation report, RTM, and metrics.
- `run_comparison` is replaced by a single-run entry point that receives `run_type` and calls only `PIPELINES[run_type]`.

The pipeline implementations, provider adapters, validation, RTM construction, metrics calculation, and failure categorization are reused.

The run manifest contains:

- run ID;
- source filename and SHA-256 document hash;
- run type;
- provider, exact model, temperature, and token ceiling;
- prompt and schema versions;
- lifecycle status and timestamps; and
- optional failure category and safe failure message.

The raw PDF and credentials are excluded from every model passed to storage.

## 6. PostgreSQL schema

Text columns with check constraints represent the existing enumerations. This avoids PostgreSQL enum migrations while preserving allowed values. All timestamps use `timestamptz`. Child tables reference `runs(run_id)` with cascading foreign keys so no rows can outlive their run.

### 6.1 Run and evidence tables

`runs`

- `run_id` primary key;
- source filename and document hash;
- run type, status, provider, model, temperature, and token ceiling;
- prompt and schema versions;
- start and completion timestamps; and
- optional failure category and failure message.

Status checks enforce that completed runs have a completion timestamp and no failure, failed runs have a completion timestamp and failure category, and running runs have no terminal fields.

`document_chunks`

- run ID and chunk ID composite primary key;
- page number, section, extracted text, and content hash.

`run_events`

- run ID and sequence number composite primary key;
- event timestamp and lifecycle stage.

Events record lifecycle order. Configuration, metrics, validation, and failures are stored in their canonical normalized tables rather than duplicated in event payloads.

`run_metrics`

- one row per run;
- every field currently defined by `RunMetrics`, using numeric, integer, and boolean columns.

Metrics are optional for parsing failures and interrupted runs.

### 6.2 Requirement tables

`requirements` stores the run-scoped requirement ID, title, description, type, module, and priority.

`requirement_ambiguities` stores ordered ambiguity strings.

`requirement_dependencies` stores each declared dependency ID.

`requirement_sources` stores each ordered citation's declared chunk ID, page, section, and excerpt.

### 6.3 Scenario tables

`scenarios` stores the run-scoped scenario ID, title, objective, and type.

`scenario_preconditions` stores ordered precondition strings.

`scenario_requirements` stores each declared requirement ID.

`scenario_sources` stores each ordered citation's declared chunk ID, page, section, and excerpt.

### 6.4 Test-case tables

`test_cases` stores the run-scoped test-case ID, scenario ID, title, and priority.

`test_case_preconditions` stores ordered precondition strings.

`test_case_requirements` stores each declared requirement ID.

`test_case_data` stores each arbitrary test-data key separately; its value uses JSONB because the model explicitly permits any JSON value.

`test_steps` stores step number, action, and expected result.

`test_case_sources` stores each ordered citation's declared chunk ID, page, section, and excerpt.

Declared dependency, cross-artifact, and citation chunk IDs are stored without target foreign keys. A semantically invalid generated bundle may contain a missing artifact or chunk reference, and that exact failed output must remain persistable for diagnosis. Every child row still has a cascading foreign key to its owning run.

### 6.5 Validation and RTM

`validation_reports` stores the run-level valid flag.

`validation_issues` stores ordered issue code, artifact ID, and message rows.

`validation_uncovered_requirements`, `validation_orphan_scenarios`, and `validation_orphan_test_cases` retain the report's ordered ID lists.

The RTM is reconstructed from requirements, scenario-requirement links, test-case-requirement links, and source citations. It is not stored as a second copy of the same relationships.

## 7. Persistence and data flow

1. The app validates required form fields and constructs provider settings.
2. The runner hashes the PDF bytes and inserts a `running` run with the safe source filename.
3. PDF parsing succeeds and document chunks are inserted, or the run is marked failed with the parsing category.
4. The runner creates the provider and invokes only the selected pipeline.
5. Validation, optional repair/revision, RTM construction, and metrics calculation reuse the existing logic.
6. Finalization uses one database transaction to insert the normalized result graph and metrics, append the finished event, and change the run to completed or failed.
7. History performs a lightweight list query. Loading one run queries its child rows and reconstructs the strict domain models used by the renderer and downloads.

Terminal runs are read-only through the repository. No update or delete operation is exposed for them.

If finalization rolls back, the original `running` record remains and is presented as interrupted. This makes an incomplete database write visible without publishing a partial result graph.

## 8. Local database setup

Docker Compose defines one PostgreSQL service with:

- a pinned PostgreSQL image;
- a named data volume;
- a health check;
- credentials supplied through local environment values; and
- a host binding limited to `127.0.0.1`.

The application reads `DATABASE_URL`. A committed example file documents safe local defaults without real credentials.

The Python application uses `psycopg` directly. An idempotent schema initializer executes `CREATE TABLE IF NOT EXISTS` statements at startup. A migration framework is deferred until an incompatible schema change exists.

The legacy `runs/` directory remains untouched and ignored by Git, but the application stops reading and writing it.

## 9. Error handling and security

- Database connection or schema initialization failure produces an actionable application error and blocks generation; there is no silent filesystem fallback.
- A run row is committed before parsing or provider calls so started failures are retained.
- Artifact finalization is atomic.
- Provider credentials remain transient UI/provider values and never enter logs, database columns, failure messages, or downloads.
- Stored source filenames are display metadata only and are never interpreted as filesystem paths.
- Existing safe-error redaction remains in force before failure messages are persisted.
- The original PDF is never persisted; extracted chunks are retained because they are required research evidence and citation context.

## 10. Testing and verification

Automated checks cover:

1. selecting each run type invokes only its mapped pipeline;
2. a completed run round-trips through the normalized PostgreSQL schema into an equivalent `RunResult`;
3. parsing and provider failures persist with the correct status and safe failure details;
4. an unfinalized running record appears as interrupted in history;
5. history is newest first and loads the selected run;
6. test-case details render preconditions, test data, ordered steps, requirement/scenario links, and citations;
7. credentials and PDF bytes are absent from stored records; and
8. the legacy filesystem store is not called for new runs.

PostgreSQL integration tests run against the local Compose test database and isolate their rows. Provider tests remain offline and deterministic.

Fresh verification includes the focused storage, runner, and Streamlit tests, followed by the full offline test suite, Python compilation, and `git diff --check`.
