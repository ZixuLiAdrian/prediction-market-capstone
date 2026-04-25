-- Prediction Market Pipeline Database Schema
-- Run this file to initialize all tables: psql -d prediction_markets -f db/schema.sql

-- ============================================================
-- FR1: Event Ingestion
-- ============================================================
CREATE TABLE IF NOT EXISTS events (
    id              SERIAL PRIMARY KEY,
    title           TEXT NOT NULL DEFAULT '',
    content         TEXT NOT NULL,
    source          VARCHAR(100) NOT NULL,    -- e.g. "reuters", "bbc", "gdelt", "polymarket", "reddit", "sec_edgar"
    source_type     VARCHAR(20) NOT NULL,     -- "rss", "gdelt", "market", "social", "official"
    url             TEXT DEFAULT '',
    entities        TEXT DEFAULT '',           -- comma-separated entity names
    content_hash    VARCHAR(64) UNIQUE,       -- SHA256 for deduplication
    signal_role     VARCHAR(20) DEFAULT 'discovery',  -- discovery | resolution | benchmark | attention
    timestamp       TIMESTAMP NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Migration guard: older databases may have an events table without signal_role.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='events' AND column_name='signal_role') THEN
        ALTER TABLE events ADD COLUMN signal_role VARCHAR(20) DEFAULT 'discovery';
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS idx_events_source_type ON events(source_type);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_content_hash ON events(content_hash);
CREATE INDEX IF NOT EXISTS idx_events_signal_role ON events(signal_role);

-- ============================================================
-- FR2: Event Clustering
-- ============================================================
CREATE TABLE IF NOT EXISTS clusters (
    id                  SERIAL PRIMARY KEY,
    label               INTEGER NOT NULL,          -- DBSCAN cluster label
    mention_velocity    FLOAT DEFAULT 0.0,         -- mentions per hour
    source_diversity    INTEGER DEFAULT 0,         -- unique source count
    recency             FLOAT DEFAULT 0.0,         -- hours since most recent event
    size                INTEGER DEFAULT 0,         -- number of events in cluster
    source_role_mix     JSONB DEFAULT '{}',        -- {"discovery": 3, "attention": 5, ...}
    coherence_score     FLOAT DEFAULT 0.0,         -- avg pairwise embedding similarity
    weighted_mention_velocity FLOAT DEFAULT 0.0,   -- source-weighted mentions per hour
    created_at          TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Many-to-many mapping between clusters and events
CREATE TABLE IF NOT EXISTS cluster_events (
    cluster_id  INTEGER REFERENCES clusters(id) ON DELETE CASCADE,
    event_id    INTEGER REFERENCES events(id) ON DELETE CASCADE,
    PRIMARY KEY (cluster_id, event_id)
);

-- ============================================================
-- FR3: LLM Event Extraction
-- ============================================================
CREATE TABLE IF NOT EXISTS extracted_events (
    id                    SERIAL PRIMARY KEY,
    cluster_id            INTEGER REFERENCES clusters(id) ON DELETE CASCADE,
    event_summary         TEXT NOT NULL,
    entities              JSONB DEFAULT '[]',
    event_type            VARCHAR(50) DEFAULT '',
    outcome_variable      TEXT DEFAULT '',
    candidate_deadlines   JSONB DEFAULT '[]',
    resolution_sources    JSONB DEFAULT '[]',
    tradability           VARCHAR(20) DEFAULT 'suitable',
    rejection_reason      TEXT DEFAULT '',
    confidence            FLOAT DEFAULT 0.5,
    market_angle          TEXT DEFAULT '',
    contradiction_flag    BOOLEAN DEFAULT FALSE,
    contradiction_details TEXT DEFAULT '',
    time_horizon          VARCHAR(200) DEFAULT '',
    resolution_hints      JSONB DEFAULT '[]',
    raw_llm_response      TEXT DEFAULT '',
    created_at            TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_extracted_cluster ON extracted_events(cluster_id);

-- ============================================================
-- FR4: LLM Question Generation
-- ============================================================
CREATE TABLE IF NOT EXISTS candidate_questions (
    id                  SERIAL PRIMARY KEY,
    extracted_event_id  INTEGER REFERENCES extracted_events(id) ON DELETE CASCADE,
    repair_parent_question_id INTEGER REFERENCES candidate_questions(id) ON DELETE SET NULL,
    question_text       TEXT NOT NULL,
    category            VARCHAR(50) DEFAULT '',       -- e.g. politics, finance, health
    question_type       VARCHAR(20) DEFAULT '',       -- "binary" or "multiple_choice"
    options             JSONB DEFAULT '[]',           -- list of option strings
    deadline            VARCHAR(200) DEFAULT '',      -- ISO date string
    deadline_source     TEXT DEFAULT '',              -- URL confirming the deadline date
    resolution_source   TEXT DEFAULT '',              -- authoritative source org + URL
    resolution_criteria TEXT DEFAULT '',              -- per-option resolution logic
    rationale               TEXT DEFAULT '',              -- why this question is interesting
    resolution_confidence   FLOAT DEFAULT 0.0,           -- LLM self-score: how cleanly outcome can be confirmed
    resolution_confidence_reason TEXT DEFAULT '',         -- one-sentence explanation
    source_independence     FLOAT DEFAULT 0.0,           -- LLM self-score: source neutrality
    timing_reliability      FLOAT DEFAULT 0.0,           -- LLM self-score: publication reliability by deadline
    already_resolved        BOOLEAN DEFAULT FALSE,        -- True = event already concluded, auto-rejected
    raw_llm_response        TEXT DEFAULT '',              -- full LLM JSON for debugging
    created_at              TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cq_extracted_event ON candidate_questions(extracted_event_id);
CREATE INDEX IF NOT EXISTS idx_cq_category ON candidate_questions(category);
CREATE INDEX IF NOT EXISTS idx_cq_question_type ON candidate_questions(question_type);

-- Migration guard: add new columns to existing deployments without dropping data
DO $$
BEGIN
    -- clusters table
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='clusters' AND column_name='source_role_mix') THEN
        ALTER TABLE clusters ADD COLUMN source_role_mix JSONB DEFAULT '{}';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='clusters' AND column_name='coherence_score') THEN
        ALTER TABLE clusters ADD COLUMN coherence_score FLOAT DEFAULT 0.0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='clusters' AND column_name='weighted_mention_velocity') THEN
        ALTER TABLE clusters ADD COLUMN weighted_mention_velocity FLOAT DEFAULT 0.0;
    END IF;
    -- extracted_events table
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='extracted_events' AND column_name='event_type') THEN
        ALTER TABLE extracted_events ADD COLUMN event_type VARCHAR(50) DEFAULT '';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='extracted_events' AND column_name='outcome_variable') THEN
        ALTER TABLE extracted_events ADD COLUMN outcome_variable TEXT DEFAULT '';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='extracted_events' AND column_name='candidate_deadlines') THEN
        ALTER TABLE extracted_events ADD COLUMN candidate_deadlines JSONB DEFAULT '[]';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='extracted_events' AND column_name='resolution_sources') THEN
        ALTER TABLE extracted_events ADD COLUMN resolution_sources JSONB DEFAULT '[]';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='extracted_events' AND column_name='tradability') THEN
        ALTER TABLE extracted_events ADD COLUMN tradability VARCHAR(20) DEFAULT 'suitable';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='extracted_events' AND column_name='rejection_reason') THEN
        ALTER TABLE extracted_events ADD COLUMN rejection_reason TEXT DEFAULT '';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='extracted_events' AND column_name='confidence') THEN
        ALTER TABLE extracted_events ADD COLUMN confidence FLOAT DEFAULT 0.5;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='extracted_events' AND column_name='market_angle') THEN
        ALTER TABLE extracted_events ADD COLUMN market_angle TEXT DEFAULT '';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='extracted_events' AND column_name='contradiction_flag') THEN
        ALTER TABLE extracted_events ADD COLUMN contradiction_flag BOOLEAN DEFAULT FALSE;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='extracted_events' AND column_name='contradiction_details') THEN
        ALTER TABLE extracted_events ADD COLUMN contradiction_details TEXT DEFAULT '';
    END IF;
    -- candidate_questions table
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='candidate_questions' AND column_name='repair_parent_question_id') THEN
        ALTER TABLE candidate_questions ADD COLUMN repair_parent_question_id INTEGER REFERENCES candidate_questions(id) ON DELETE SET NULL;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='candidate_questions' AND column_name='category') THEN
        ALTER TABLE candidate_questions ADD COLUMN category VARCHAR(50) DEFAULT '';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='candidate_questions' AND column_name='question_type') THEN
        ALTER TABLE candidate_questions ADD COLUMN question_type VARCHAR(20) DEFAULT '';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='candidate_questions' AND column_name='options') THEN
        ALTER TABLE candidate_questions ADD COLUMN options JSONB DEFAULT '[]';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='candidate_questions' AND column_name='deadline_source') THEN
        ALTER TABLE candidate_questions ADD COLUMN deadline_source TEXT DEFAULT '';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='candidate_questions' AND column_name='rationale') THEN
        ALTER TABLE candidate_questions ADD COLUMN rationale TEXT DEFAULT '';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='candidate_questions' AND column_name='raw_llm_response') THEN
        ALTER TABLE candidate_questions ADD COLUMN raw_llm_response TEXT DEFAULT '';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='candidate_questions' AND column_name='resolution_confidence') THEN
        ALTER TABLE candidate_questions ADD COLUMN resolution_confidence FLOAT DEFAULT 0.0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='candidate_questions' AND column_name='resolution_confidence_reason') THEN
        ALTER TABLE candidate_questions ADD COLUMN resolution_confidence_reason TEXT DEFAULT '';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='candidate_questions' AND column_name='source_independence') THEN
        ALTER TABLE candidate_questions ADD COLUMN source_independence FLOAT DEFAULT 0.0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='candidate_questions' AND column_name='timing_reliability') THEN
        ALTER TABLE candidate_questions ADD COLUMN timing_reliability FLOAT DEFAULT 0.0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='candidate_questions' AND column_name='already_resolved') THEN
        ALTER TABLE candidate_questions ADD COLUMN already_resolved BOOLEAN DEFAULT FALSE;
    END IF;
    -- validation_results table (only ALTER if table already existed without the column)
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='validation_results')
       AND NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='validation_results' AND column_name='clarity_score') THEN
        ALTER TABLE validation_results ADD COLUMN clarity_score FLOAT DEFAULT 1.0;
    END IF;
    -- scored_candidates table
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='scored_candidates')
       AND NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='scored_candidates' AND column_name='market_interest_score') THEN
        ALTER TABLE scored_candidates ADD COLUMN market_interest_score FLOAT DEFAULT 0.0;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='scored_candidates')
       AND NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='scored_candidates' AND column_name='resolution_strength_score') THEN
        ALTER TABLE scored_candidates ADD COLUMN resolution_strength_score FLOAT DEFAULT 0.0;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='scored_candidates')
       AND NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='scored_candidates' AND column_name='time_horizon_score') THEN
        ALTER TABLE scored_candidates ADD COLUMN time_horizon_score FLOAT DEFAULT 0.0;
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS idx_cq_repair_parent ON candidate_questions(repair_parent_question_id);

CREATE INDEX IF NOT EXISTS idx_extracted_tradability ON extracted_events(tradability);

-- ============================================================
-- FR5: Rule Validation
-- ============================================================
CREATE TABLE IF NOT EXISTS validation_results (
    id                  SERIAL PRIMARY KEY,
    question_id         INTEGER REFERENCES candidate_questions(id) ON DELETE CASCADE,
    is_valid            BOOLEAN DEFAULT FALSE,
    flags               JSONB DEFAULT '[]',        -- list of validation flag strings
    clarity_score       FLOAT DEFAULT 1.0,         -- 1.0 - 0.2 * len(flags), clamped [0,1]
    created_at          TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ============================================================
-- FR6: Heuristic Scoring
-- ============================================================
CREATE TABLE IF NOT EXISTS scored_candidates (
    id                      SERIAL PRIMARY KEY,
    question_id             INTEGER REFERENCES candidate_questions(id) ON DELETE CASCADE,
    total_score             FLOAT DEFAULT 0.0,
    mention_velocity_score  FLOAT DEFAULT 0.0,
    source_diversity_score  FLOAT DEFAULT 0.0,
    clarity_score           FLOAT DEFAULT 0.0,
    novelty_score           FLOAT DEFAULT 0.0,
    market_interest_score   FLOAT DEFAULT 0.0,
    resolution_strength_score FLOAT DEFAULT 0.0,
    time_horizon_score      FLOAT DEFAULT 0.0,
    rank                    INTEGER DEFAULT 0,
    created_at              TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Pipeline Runs: background execution tracking for Streamlit
-- ============================================================
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id                  SERIAL PRIMARY KEY,
    status              VARCHAR(20) NOT NULL DEFAULT 'queued'
                        CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    stage_start         INTEGER NOT NULL DEFAULT 1,
    stage_end           INTEGER NOT NULL DEFAULT 6,
    fr3_limit_mode      VARCHAR(20) NOT NULL DEFAULT 'default'
                        CHECK (fr3_limit_mode IN ('default', 'custom', 'all')),
    fr3_limit_value     INTEGER,
    fr4_limit_mode      VARCHAR(20) NOT NULL DEFAULT 'default'
                        CHECK (fr4_limit_mode IN ('default', 'custom', 'all')),
    fr4_limit_value     INTEGER,
    log_mode            VARCHAR(20) NOT NULL DEFAULT 'normal'
                        CHECK (log_mode IN ('normal', 'debug')),
    fr3_model           TEXT DEFAULT '',
    fr4_model           TEXT DEFAULT '',
    subprocess_pid      INTEGER,
    error_message       TEXT DEFAULT '',
    started_at          TIMESTAMP,
    finished_at         TIMESTAMP,
    created_at          TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pipeline_run_stages (
    id                  SERIAL PRIMARY KEY,
    run_id              INTEGER NOT NULL REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    stage_number        INTEGER NOT NULL,
    stage_name          VARCHAR(100) NOT NULL,
    status              VARCHAR(20) NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    summary             JSONB DEFAULT '{}'::jsonb,
    error_message       TEXT DEFAULT '',
    started_at          TIMESTAMP,
    finished_at         TIMESTAMP,
    created_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, stage_number)
);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status ON pipeline_runs(status);
CREATE INDEX IF NOT EXISTS idx_pipeline_run_stages_run_id ON pipeline_run_stages(run_id);

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='pipeline_runs')
       AND NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='pipeline_runs' AND column_name='fr3_limit_mode') THEN
        ALTER TABLE pipeline_runs ADD COLUMN fr3_limit_mode VARCHAR(20) DEFAULT 'default';
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='pipeline_runs')
       AND NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='pipeline_runs' AND column_name='fr3_limit_value') THEN
        ALTER TABLE pipeline_runs ADD COLUMN fr3_limit_value INTEGER;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='pipeline_runs')
       AND NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='pipeline_runs' AND column_name='fr3_model') THEN
        ALTER TABLE pipeline_runs ADD COLUMN fr3_model TEXT DEFAULT '';
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='pipeline_runs')
       AND NOT EXISTS (SELECT 1 FROM information_schema.columns
                       WHERE table_name='pipeline_runs' AND column_name='fr4_model') THEN
        ALTER TABLE pipeline_runs ADD COLUMN fr4_model TEXT DEFAULT '';
    END IF;
END
$$;

-- ============================================================
-- FR7 Review State: Human selection / removal lifecycle
-- ============================================================
CREATE TABLE IF NOT EXISTS question_review_state (
    question_id          INTEGER PRIMARY KEY REFERENCES candidate_questions(id) ON DELETE CASCADE,
    status               VARCHAR(20) NOT NULL CHECK (status IN ('selected', 'removed')),
    reason               TEXT DEFAULT '',
    notes                TEXT DEFAULT '',
    changed_at           TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_question_review_state_status ON question_review_state(status);
