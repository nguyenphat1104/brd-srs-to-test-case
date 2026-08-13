# Runs-First Navigation and Browser-Local Settings Design

**Date:** 2026-08-13
**Status:** Approved

## 1. Purpose

Replace the current four-step tab workflow with a runs-first workspace. The home view lists saved generation runs, each run opens in a dedicated detail view, and a dedicated create view owns file upload and run-type selection. Provider settings move to a top-navigation dialog and persist in the current browser's `localStorage`.

The existing PostgreSQL run lifecycle, immutable run snapshots, generation pipelines, result models, validation, downloads, and detailed artifact rendering remain the system of record and are reused.

## 2. Success criteria

The redesign is complete when:

1. the four Configure, Run, Results, and Run history tabs are removed;
2. the initial view lists persisted runs newest first;
3. the top navigation exposes Home and Settings;
4. selecting a run opens its test cases and immutable configuration snapshot in a dedicated detail view;
5. Create new run opens a dedicated view with PDF upload and run-type selection;
6. run type is selected per new run and is not stored as an app setting;
7. app settings persist only after Save settings is clicked;
8. saved settings are restored from browser `localStorage` after a refresh;
9. successful and persisted failed generations automatically open the resulting run detail;
10. existing runs remain browsable when app settings are missing or browser storage is unavailable; and
11. provider credentials never enter PostgreSQL, downloads, logs, errors, URLs, or run snapshots.

## 3. Scope

### In scope

- A persistent top navigation with an app/Home action and Settings action.
- Runs, Create Run, and Run Detail views in one Streamlit application.
- A Settings dialog for provider, model, credential, applicable base URL, and token ceiling.
- Explicit browser-local settings load and save through Streamlit's built-in component API.
- Provider-specific settings validation.
- A newest-first, selectable run list.
- Reuse of the existing PostgreSQL repository and result renderer.
- Test cases as the primary run-detail content, with supporting artifacts and diagnostics retained.
- Automatic navigation from generation to the persisted result.

### Out of scope

- Multipage routing and shareable view URLs.
- Search, filtering, pagination, tags, deletion, editing, or retention controls for runs.
- Storing run type as an app default.
- Changing pipeline, provider, validation, domain-model, or database behavior unrelated to navigation.
- Storing credentials or raw PDF bytes with a run.
- Adding a third-party local-storage or navigation dependency.

## 4. User experience

### 4.1 Top navigation

A persistent top bar replaces the step tabs. The app name/Home action appears on the left and Settings appears on the right.

Home always returns to the Runs view. Settings opens a modal dialog without changing the current view. The top bar remains available from Runs, Create Run, and Run Detail.

### 4.2 Runs home

Runs is the initial view. Its header contains Create new run. The list queries the existing repository and displays records newest first with:

- start time;
- source filename;
- run type;
- provider and model;
- displayed status; and
- test-case count when metrics exist.

The list uses Streamlit's native single-row selection. Selecting a row loads the complete run and opens Run Detail. An empty list shows an empty state with the same Create new run action.

Users can browse completed, failed, and interrupted runs without configuring a provider.

### 4.3 Create Run

Create Run contains only run-specific input and context:

- one text-extractable PDF upload;
- one required run-type selector;
- a read-only summary of the saved provider, model, and token ceiling;
- Edit settings; and
- Generate test cases.

Run type is selected here for every run. It is not present in Settings and is not restored from `localStorage` as an app default.

If required app settings are missing, Create new run opens Settings before continuing. Existing history remains accessible. A successful Save settings action then opens Create Run. Opening Edit settings from Create Run preserves the PDF-upload and run-type widget state.

Generate reuses the existing progress reporting. A returned completed or failed `RunResult` becomes the selected run and opens in Run Detail automatically. An unexpected exception that does not return a run ID leaves the user in Create Run with a safe error.

### 4.4 Run Detail

Run Detail is a dedicated view with Back to runs. Test cases are its primary artifact section. Existing requirements and scenarios remain available as supporting context, along with:

- lifecycle status and safe diagnostics;
- volume, token, latency, quality, and traceability metrics;
- traceability and complete-bundle downloads; and
- the immutable run configuration snapshot.

The snapshot displays:

- run type;
- provider and exact model;
- fixed temperature;
- token ceiling;
- source filename and document hash;
- prompt and schema versions;
- lifecycle status and timestamps; and
- run ID.

Credentials and base URL are not shown or persisted in the snapshot. Failed and interrupted runs open through the same view and expose the diagnostics available for their lifecycle state.

## 5. App settings and browser persistence

### 5.1 Stored settings

The Settings dialog contains:

- provider;
- model;
- the selected provider's credential, when applicable;
- base URL for providers that use one; and
- token ceiling.

Run type is explicitly excluded.

Settings are stored as one versioned JSON object under one application-specific `localStorage` key. Versioning permits invalidating an incompatible object without attempting a migration framework before one is needed.

### 5.2 Loading

On browser startup, a small Streamlit component reads the stored object and returns it to Python. The application validates its version, shape, provider, model, applicable URL, credential requirements, and token-ceiling bounds before copying values into session state.

If no valid stored object exists, the application uses the current `.env` values and existing defaults. These fallback values remain session values until the user explicitly saves them.

The Runs view does not wait for settings to become valid. Settings validity gates only starting a new run.

### 5.3 Saving and cancellation

Settings fields use dialog-local draft values. Save settings validates the draft, writes the complete object to `localStorage`, updates the active session values, and closes the dialog. Settings do not persist field-by-field.

Cancel closes the dialog and discards unsaved edits. Saved changes affect future runs only and never mutate persisted run snapshots.

The dialog warns that the requested browser-local credential is readable by scripts running on the same application origin. Credentials are masked in the UI after entry.

### 5.4 Streamlit integration

The application uses Streamlit's built-in bidirectional component API for the minimal JavaScript required to read and write `localStorage`. No third-party dependency is added. The declared Streamlit requirement becomes `streamlit>=1.61,<2`, matching the component API used by the implementation.

The component accepts only application-produced settings data and does not render untrusted content. Python remains responsible for all validation before settings reach provider construction.

## 6. Application structure and state

The redesign remains a single Streamlit entry point. Existing rendering and provider helpers are reused. Focused render functions own the three views and Settings dialog rather than introducing a routing framework or new application layer.

Session state holds:

- the active view: `runs`, `create`, or `detail`;
- the selected run ID/result for Run Detail;
- the validated active app settings;
- settings-loading state; and
- existing transient upload, model-loading, and generation state.

Navigation rules are:

1. startup selects Runs;
2. Home or Back to runs selects Runs and clears only detail selection;
3. Create new run validates active settings, opening Settings when incomplete, otherwise selecting Create Run;
4. selecting a history row loads it and selects Run Detail; and
5. generation selects Run Detail when the runner returns a persisted result.

Returning to Runs triggers a fresh repository list, so a newly generated run is visible without adding a second client-side cache.

## 7. Data flow

### 7.1 Settings

1. The browser-storage component reads the versioned settings object.
2. Python validates it and places valid values in session state, or applies `.env`/defaults.
3. The Settings dialog edits a draft.
4. Save settings validates the draft.
5. The component writes the complete object to `localStorage`.
6. Python promotes the draft to active session settings for future runs.

### 7.2 Existing-run selection

1. Runs queries `RunRepository.list_runs()`.
2. Native row selection yields one run ID.
3. The app calls `RunRepository.load_run(run_id)`.
4. The reconstructed strict `RunResult` is passed to the shared detail renderer.

### 7.3 New run

1. Create Run validates the active settings, uploaded PDF, and selected run type.
2. The app constructs the existing `ProviderSettings` without persisting the credential.
3. The existing runner creates the PostgreSQL run and executes only the selected pipeline.
4. The runner returns a completed or failed `RunResult` whose manifest contains the immutable safe snapshot.
5. The app selects that run and opens Run Detail.

The PostgreSQL repository remains the source of truth for saved runs. Browser storage contains app defaults only and is never used as run history.

## 8. Error handling and security

- Database connection or initialization failure remains a blocking, actionable application error because history and generation both require PostgreSQL.
- A run that disappears or cannot be loaded returns the user to Runs with an error.
- Missing settings, an unsupported provider, a missing model/credential, an invalid URL, or an invalid token ceiling is rejected before generation.
- Missing or invalid PDF input stays in Create Run with an actionable message.
- A returned failed or interrupted result opens normally in Run Detail.
- An unexpected generation exception without a returned result stays in Create Run and uses existing credential/base-URL redaction.
- Missing, malformed, version-incompatible, quota-exceeded, or inaccessible `localStorage` falls back to `.env`/defaults and shows a warning without blocking run history.
- Credentials never enter database rows, run models, downloads, logs, errors, query parameters, or the visible snapshot.
- The original PDF remains transient and is not added to browser storage.
- Existing safe-error redaction and terminal-run immutability remain unchanged.

## 9. Testing and verification

Automated checks cover:

1. Runs is the initial view and the four workflow tabs are absent;
2. empty history exposes Create new run;
3. selecting a history record loads its dedicated detail view and snapshot;
4. Create Run contains PDF upload, run-type selection, saved-settings summary, and generation action;
5. run type is absent from Settings and browser-persisted settings;
6. missing required settings open Settings before Create Run;
7. Save settings validates and updates Python-side active settings, while cancel preserves prior values;
8. successful and returned failed generations automatically open Run Detail;
9. result artifacts, downloads, diagnostics, and safe redaction continue to work;
10. malformed or missing browser settings fall back safely; and
11. credentials are absent from snapshot rendering and persisted run structures.

Streamlit's Python application test runner does not execute the browser JavaScript. One manual browser smoke check therefore verifies the storage boundary:

1. save settings, including a provider credential;
2. refresh the browser and confirm the settings are restored;
3. create a run and confirm it opens automatically; and
4. inspect the run snapshot and downloads to confirm the credential is absent.

Fresh verification runs the focused Streamlit tests, the full offline test suite, Python compilation, and `git diff --check`.
