CREATE TABLE IF NOT EXISTS runs (
    run_id text PRIMARY KEY CHECK (run_id <> ''),
    source_filename text NOT NULL CHECK (source_filename <> ''),
    document_hash text NOT NULL CHECK (document_hash ~ '^[0-9a-f]{64}$'),
    run_type text NOT NULL CHECK (run_type IN ('single_prompt', 'staged_single_agent', 'centralized_multi_agent')),
    status text NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    provider text NOT NULL CHECK (provider <> ''),
    model text NOT NULL CHECK (model <> ''),
    temperature double precision NOT NULL CHECK (temperature >= 0),
    token_ceiling integer NOT NULL CHECK (token_ceiling > 0),
    prompt_version text NOT NULL CHECK (prompt_version <> ''),
    schema_version text NOT NULL CHECK (schema_version <> ''),
    started_at timestamptz NOT NULL,
    completed_at timestamptz,
    failure_category text CHECK (failure_category IN (
        'parsing', 'configuration', 'provider_rejection', 'transport_exhaustion',
        'timeout', 'budget_exhaustion', 'schema_failure', 'semantic_validation'
    )),
    failure_message text,
    CHECK (completed_at IS NULL OR completed_at >= started_at),
    CHECK (
        (status = 'running' AND completed_at IS NULL AND failure_category IS NULL AND failure_message IS NULL)
        OR (status = 'completed' AND completed_at IS NOT NULL AND failure_category IS NULL AND failure_message IS NULL)
        OR (status = 'failed' AND completed_at IS NOT NULL AND failure_category IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS runs_started_at_idx ON runs (started_at DESC);

CREATE TABLE IF NOT EXISTS document_chunks (
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    chunk_id text NOT NULL CHECK (chunk_id <> ''),
    page_number integer NOT NULL CHECK (page_number > 0),
    section text NOT NULL,
    text text NOT NULL CHECK (text <> ''),
    content_hash text NOT NULL CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    PRIMARY KEY (run_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS run_events (
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    sequence integer NOT NULL CHECK (sequence > 0),
    occurred_at timestamptz NOT NULL,
    stage text NOT NULL CHECK (stage <> ''),
    PRIMARY KEY (run_id, sequence)
);

CREATE TABLE IF NOT EXISTS run_metrics (
    run_id text PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    completion boolean NOT NULL,
    schema_valid boolean NOT NULL,
    citation_coverage double precision NOT NULL CHECK (citation_coverage BETWEEN 0 AND 1),
    requirement_scenario_coverage double precision NOT NULL CHECK (requirement_scenario_coverage BETWEEN 0 AND 1),
    requirement_test_case_coverage double precision NOT NULL CHECK (requirement_test_case_coverage BETWEEN 0 AND 1),
    positive_scenario_coverage double precision NOT NULL CHECK (positive_scenario_coverage BETWEEN 0 AND 1),
    non_positive_scenario_coverage double precision NOT NULL CHECK (non_positive_scenario_coverage BETWEEN 0 AND 1),
    rtm_completeness double precision NOT NULL CHECK (rtm_completeness BETWEEN 0 AND 1),
    orphan_rate double precision NOT NULL CHECK (orphan_rate BETWEEN 0 AND 1),
    invalid_reference_rate double precision NOT NULL CHECK (invalid_reference_rate BETWEEN 0 AND 1),
    duplicate_test_case_rate double precision NOT NULL CHECK (duplicate_test_case_rate BETWEEN 0 AND 1),
    requirement_count integer NOT NULL CHECK (requirement_count >= 0),
    scenario_count integer NOT NULL CHECK (scenario_count >= 0),
    test_case_count integer NOT NULL CHECK (test_case_count >= 0),
    input_tokens integer NOT NULL CHECK (input_tokens >= 0),
    output_tokens integer NOT NULL CHECK (output_tokens >= 0),
    charged_tokens integer NOT NULL CHECK (charged_tokens >= 0),
    latency_seconds double precision NOT NULL CHECK (latency_seconds >= 0),
    retries integer NOT NULL CHECK (retries >= 0),
    schema_repairs integer NOT NULL CHECK (schema_repairs >= 0),
    semantic_revisions integer NOT NULL CHECK (semantic_revisions >= 0),
    budget_exhausted boolean NOT NULL
);

CREATE TABLE IF NOT EXISTS requirements (
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    requirement_id text NOT NULL CHECK (requirement_id ~ '^REQ-[0-9]{3,}$'),
    title text NOT NULL CHECK (title <> ''),
    description text NOT NULL CHECK (description <> ''),
    requirement_type text NOT NULL CHECK (requirement_type IN ('functional', 'non_functional', 'business')),
    module text NOT NULL CHECK (module <> ''),
    priority text NOT NULL CHECK (priority IN ('high', 'medium', 'low')),
    PRIMARY KEY (run_id, requirement_id)
);

CREATE TABLE IF NOT EXISTS requirement_ambiguities (
    run_id text NOT NULL,
    requirement_id text NOT NULL,
    position integer NOT NULL CHECK (position > 0),
    value text NOT NULL,
    PRIMARY KEY (run_id, requirement_id, position),
    FOREIGN KEY (run_id, requirement_id) REFERENCES requirements(run_id, requirement_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS requirement_dependencies (
    run_id text NOT NULL,
    requirement_id text NOT NULL,
    position integer NOT NULL CHECK (position > 0),
    dependency_id text NOT NULL,
    PRIMARY KEY (run_id, requirement_id, position),
    FOREIGN KEY (run_id, requirement_id) REFERENCES requirements(run_id, requirement_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS requirement_sources (
    run_id text NOT NULL,
    requirement_id text NOT NULL,
    position integer NOT NULL CHECK (position > 0),
    chunk_id text NOT NULL CHECK (chunk_id <> ''),
    page_number integer NOT NULL CHECK (page_number > 0),
    section text NOT NULL,
    excerpt text NOT NULL CHECK (excerpt <> ''),
    PRIMARY KEY (run_id, requirement_id, position),
    FOREIGN KEY (run_id, requirement_id) REFERENCES requirements(run_id, requirement_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scenarios (
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    scenario_id text NOT NULL CHECK (scenario_id ~ '^SCN-[0-9]{3,}$'),
    title text NOT NULL CHECK (title <> ''),
    objective text NOT NULL CHECK (objective <> ''),
    scenario_type text NOT NULL CHECK (scenario_type IN ('positive', 'negative', 'boundary', 'edge', 'state_transition')),
    PRIMARY KEY (run_id, scenario_id)
);

CREATE TABLE IF NOT EXISTS scenario_preconditions (
    run_id text NOT NULL,
    scenario_id text NOT NULL,
    position integer NOT NULL CHECK (position > 0),
    value text NOT NULL,
    PRIMARY KEY (run_id, scenario_id, position),
    FOREIGN KEY (run_id, scenario_id) REFERENCES scenarios(run_id, scenario_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scenario_requirements (
    run_id text NOT NULL,
    scenario_id text NOT NULL,
    position integer NOT NULL CHECK (position > 0),
    requirement_id text NOT NULL,
    PRIMARY KEY (run_id, scenario_id, position),
    FOREIGN KEY (run_id, scenario_id) REFERENCES scenarios(run_id, scenario_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scenario_sources (
    run_id text NOT NULL,
    scenario_id text NOT NULL,
    position integer NOT NULL CHECK (position > 0),
    chunk_id text NOT NULL CHECK (chunk_id <> ''),
    page_number integer NOT NULL CHECK (page_number > 0),
    section text NOT NULL,
    excerpt text NOT NULL CHECK (excerpt <> ''),
    PRIMARY KEY (run_id, scenario_id, position),
    FOREIGN KEY (run_id, scenario_id) REFERENCES scenarios(run_id, scenario_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS test_cases (
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    test_case_id text NOT NULL CHECK (test_case_id ~ '^TC-[0-9]{3,}$'),
    scenario_id text NOT NULL CHECK (scenario_id ~ '^SCN-[0-9]{3,}$'),
    title text NOT NULL CHECK (title <> ''),
    priority text NOT NULL CHECK (priority IN ('P1', 'P2', 'P3')),
    PRIMARY KEY (run_id, test_case_id)
);

CREATE TABLE IF NOT EXISTS test_case_preconditions (
    run_id text NOT NULL,
    test_case_id text NOT NULL,
    position integer NOT NULL CHECK (position > 0),
    value text NOT NULL,
    PRIMARY KEY (run_id, test_case_id, position),
    FOREIGN KEY (run_id, test_case_id) REFERENCES test_cases(run_id, test_case_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS test_case_requirements (
    run_id text NOT NULL,
    test_case_id text NOT NULL,
    position integer NOT NULL CHECK (position > 0),
    requirement_id text NOT NULL,
    PRIMARY KEY (run_id, test_case_id, position),
    FOREIGN KEY (run_id, test_case_id) REFERENCES test_cases(run_id, test_case_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS test_case_data (
    run_id text NOT NULL,
    test_case_id text NOT NULL,
    key text NOT NULL,
    value jsonb NOT NULL,
    PRIMARY KEY (run_id, test_case_id, key),
    FOREIGN KEY (run_id, test_case_id) REFERENCES test_cases(run_id, test_case_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS test_steps (
    run_id text NOT NULL,
    test_case_id text NOT NULL,
    step_number integer NOT NULL CHECK (step_number > 0),
    action text NOT NULL CHECK (action <> ''),
    expected_result text NOT NULL CHECK (expected_result <> ''),
    PRIMARY KEY (run_id, test_case_id, step_number),
    FOREIGN KEY (run_id, test_case_id) REFERENCES test_cases(run_id, test_case_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS test_case_sources (
    run_id text NOT NULL,
    test_case_id text NOT NULL,
    position integer NOT NULL CHECK (position > 0),
    chunk_id text NOT NULL CHECK (chunk_id <> ''),
    page_number integer NOT NULL CHECK (page_number > 0),
    section text NOT NULL,
    excerpt text NOT NULL CHECK (excerpt <> ''),
    PRIMARY KEY (run_id, test_case_id, position),
    FOREIGN KEY (run_id, test_case_id) REFERENCES test_cases(run_id, test_case_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS validation_reports (
    run_id text PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    valid boolean NOT NULL
);

CREATE TABLE IF NOT EXISTS validation_issues (
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    position integer NOT NULL CHECK (position > 0),
    code text NOT NULL,
    artifact_id text NOT NULL,
    message text NOT NULL,
    PRIMARY KEY (run_id, position)
);

CREATE TABLE IF NOT EXISTS validation_uncovered_requirements (
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    position integer NOT NULL CHECK (position > 0),
    requirement_id text NOT NULL,
    PRIMARY KEY (run_id, position)
);

CREATE TABLE IF NOT EXISTS validation_orphan_scenarios (
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    position integer NOT NULL CHECK (position > 0),
    scenario_id text NOT NULL,
    PRIMARY KEY (run_id, position)
);

CREATE TABLE IF NOT EXISTS validation_orphan_test_cases (
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    position integer NOT NULL CHECK (position > 0),
    test_case_id text NOT NULL,
    PRIMARY KEY (run_id, position)
);
