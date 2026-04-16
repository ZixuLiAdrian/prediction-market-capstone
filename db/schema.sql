-- Prediction Market Pipeline Database Schema
-- Run this file to initialize all tables: psql -d prediction_markets -f db/schema.sql

-- ============================================================
-- FR1: Event Ingestion
-- ============================================================
CREATE TABLE IF NOT EXISTS events (
    id              SERIAL PRIMARY KEY,
    title           TEXT NOT NULL DEFAULT '',
    content         TEXT NOT NULL,
    source          VARCHAR(100) NOT NULL,    -- e.g. "reuters", "bbc", "gdelt", "polymarket"
    source_type     VARCHAR(20) NOT NULL,     -- "rss", "gdelt", "market"
    url             TEXT DEFAULT '',
    entities        TEXT DEFAULT '',           -- comma-separated entity names
    content_hash    VARCHAR(64) UNIQUE,       -- SHA256 for deduplication
    timestamp       TIMESTAMP NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_source_type ON events(source_type);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_content_hash ON events(content_hash);

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
    id                  SERIAL PRIMARY KEY,
    cluster_id          INTEGER REFERENCES clusters(id) ON DELETE CASCADE,
    event_summary       TEXT NOT NULL,
    entities            JSONB DEFAULT '[]',        -- list of entity strings
    time_horizon        VARCHAR(200) DEFAULT '',
    resolution_hints    JSONB DEFAULT '[]',        -- list of hint strings
    raw_llm_response    TEXT DEFAULT '',            -- full LLM response for debugging
    created_at          TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_extracted_cluster ON extracted_events(cluster_id);

-- ============================================================
-- FR4-FR7: Placeholder tables for teammates
-- These tables are stubs; teammates should modify as needed.
-- ============================================================

-- FR4: LLM Question Generation
CREATE TABLE IF NOT EXISTS candidate_questions (
    id                  SERIAL PRIMARY KEY,
    extracted_event_id  INTEGER REFERENCES extracted_events(id) ON DELETE CASCADE,
    question_text       TEXT NOT NULL,
    category            VARCHAR(100) DEFAULT '',   -- e.g. politics, finance, technology
    question_type       VARCHAR(50)  DEFAULT 'binary',  -- binary | multiple_choice
    options             JSONB        DEFAULT '[]', -- ordered list of answer option strings
    deadline            VARCHAR(200) DEFAULT '',
    deadline_source     TEXT         DEFAULT '',   -- official schedule/calendar confirming the deadline
    resolution_source   TEXT         DEFAULT '',
    resolution_criteria TEXT         DEFAULT '',
    rationale           TEXT         DEFAULT '',   -- why this is a good market question
    raw_llm_response    TEXT         DEFAULT '',   -- full LLM output for debugging
    created_at          TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cq_extracted_event ON candidate_questions(extracted_event_id);
CREATE INDEX IF NOT EXISTS idx_cq_category        ON candidate_questions(category);
CREATE INDEX IF NOT EXISTS idx_cq_question_type   ON candidate_questions(question_type);

-- Migration guard: add new columns if this table already existed without them.
-- Safe to run repeatedly — IF NOT EXISTS prevents duplicate-column errors.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'candidate_questions' AND column_name = 'category'
    ) THEN
        ALTER TABLE candidate_questions ADD COLUMN category VARCHAR(100) DEFAULT '';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'candidate_questions' AND column_name = 'question_type'
    ) THEN
        ALTER TABLE candidate_questions ADD COLUMN question_type VARCHAR(50) DEFAULT 'binary';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'candidate_questions' AND column_name = 'options'
    ) THEN
        ALTER TABLE candidate_questions ADD COLUMN options JSONB DEFAULT '[]';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'candidate_questions' AND column_name = 'rationale'
    ) THEN
        ALTER TABLE candidate_questions ADD COLUMN rationale TEXT DEFAULT '';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'candidate_questions' AND column_name = 'deadline_source'
    ) THEN
        ALTER TABLE candidate_questions ADD COLUMN deadline_source TEXT DEFAULT '';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'candidate_questions' AND column_name = 'raw_llm_response'
    ) THEN
        ALTER TABLE candidate_questions ADD COLUMN raw_llm_response TEXT DEFAULT '';
    END IF;
END $$;

-- FR5: Rule Validation
CREATE TABLE IF NOT EXISTS validation_results (
    id                  SERIAL PRIMARY KEY,
    question_id         INTEGER REFERENCES candidate_questions(id) ON DELETE CASCADE,
    is_valid            BOOLEAN DEFAULT FALSE,
    flags               JSONB DEFAULT '[]',        -- list of validation flag strings
    created_at          TIMESTAMP NOT NULL DEFAULT NOW()
);

-- FR6: Heuristic Scoring
CREATE TABLE IF NOT EXISTS scored_candidates (
    id                  SERIAL PRIMARY KEY,
    question_id         INTEGER REFERENCES candidate_questions(id) ON DELETE CASCADE,
    total_score         FLOAT DEFAULT 0.0,
    mention_velocity_score  FLOAT DEFAULT 0.0,
    source_diversity_score  FLOAT DEFAULT 0.0,
    clarity_score       FLOAT DEFAULT 0.0,
    novelty_score       FLOAT DEFAULT 0.0,
    rank                INTEGER DEFAULT 0,
    created_at          TIMESTAMP NOT NULL DEFAULT NOW()
);
