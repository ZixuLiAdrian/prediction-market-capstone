# Prediction market pipeline

Batch system that turns ingested news and market signals into candidate prediction-market questions, validates them with fixed rules, scores and ranks them, and persists everything in PostgreSQL. Stages **FR1–FR6** run in `pipeline.py`; a **Streamlit** UI reads ranked results from the database (it is not a pipeline stage).

## Architecture

```
[RSS | GDELT | Polymarket] → FR1 ingest → FR2 cluster → FR3 extract → FR4 generate
  → FR5 validate → FR6 score & rank → PostgreSQL → Streamlit (read-only UI)
```

- **Pipeline (FR1–FR6):** writes intermediate and final rows through `db/connection.py` into the schema in `db/schema.sql`.
- **Database:** source of truth between runs.
- **Streamlit:** optional viewer on top of stored scores; does not execute FR1–FR6.

## Functional stages

| Stage | Role |
|--------|------|
| **FR1** | Ingest events from RSS, GDELT, and Polymarket; dedupe by content hash; store `Event` rows. |
| **FR2** | Embed event text, cluster with DBSCAN, compute cluster features (velocity, source diversity, recency), persist `Cluster` rows. |
| **FR3** | Call the configured LLM to extract structured fields per cluster; validate JSON against the extraction schema; store `ExtractedEvent` rows. |
| **FR4** | Call the LLM to generate candidate questions from extracted events; schema-validated output; store `CandidateQuestion` rows. |
| **FR5** | Run deterministic checks on each candidate; attach flags, validity, and a clarity score; persist `ValidationResult`. |
| **FR6** | Score and rank validated questions with deterministic heuristics; persist ranked rows in `scored_candidates` and log component breakdowns. |

## FR5: Deterministic validation

FR5 evaluates each `CandidateQuestion` with explicit rules (no LLM): wording and resolution clarity, resolution source and criteria strength, deadline parsing and plausibility, and binary question shape where applicable. It **flags invalid or weak questions and filters them from downstream scoring**—invalid rows remain in the database with their validation record; FR6 only consumes questions marked valid.

## FR6: Scoring and ranking

FR6 ranks questions that passed FR5 using **deterministic heuristics** (fixed formulas and weights, not learned models). Scores combine signals such as cluster-derived features, resolution-source strength, time-to-deadline, market-interest-style signals, and **penalties** for low-value or noisy patterns. Each scored row carries a **total score**, **rank**, and enough structure to produce **component breakdowns** and short **explanation text** in logs and in the UI. The intent is **heuristic ordering aligned with how prediction-market listings are typically prioritized**, not a claim of parity with any live marketplace.

## Streamlit UI

`streamlit_app.py` loads top-N **ranked scored questions** from PostgreSQL, shows question text, category, deadline, source, total score, a **plain-language explanation**, and an expandable **numeric score breakdown** (clarity, cluster-style components, market interest, resolution strength, time horizon, quality flags). Install Streamlit if needed (`pip install streamlit`); the core pipeline does not require it.

## Setup

**Prerequisites:** Python 3.10+, PostgreSQL.

```bash
git clone <repo-url>
cd prediction-market-capstone

pip install -r requirements.txt

cp .env.example .env
# Then edit .env with your real local DB credentials and LLM_API_KEY (see below).

# Create the database if it does not exist yet (name must match DB_NAME in .env).
createdb prediction_markets

psql -d prediction_markets -f db/schema.sql
```

**Secrets and local configuration**

- Copy the template: `cp .env.example .env`
- Edit `.env` and set your **local** PostgreSQL settings (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`) and your **LLM provider API key** (`LLM_API_KEY`, plus `LLM_PROVIDER` / `LLM_MODEL` if you change them). Values in `.env.example` are placeholders for documentation only.
- **`.env` is local-only and must not be committed** (it is gitignored). Only `.env.example` should live in the repository as the shared template.

Runtime tuning for ingestion, clustering, and LLM routing lives in **`.env`** (see `.env.example`). Pipeline stage wiring is in `pipeline.py` (`STAGES` list).

## How to run

**Tests**

```bash
python -m pytest tests/ -q
```

**Pipeline**

```bash
# FR1 → FR6 (all stages)
python pipeline.py

# Selected stages (examples)
python pipeline.py --stage 4    # FR4 only
python pipeline.py --stage 5    # FR5 only
python pipeline.py --stage 6    # FR6 only
python pipeline.py --stage 1-2  # inclusive range
```

Logs go to stdout and `pipeline.log`.

**Streamlit (after FR6 has written scores)**

```bash
python -m streamlit run streamlit_app.py
```

## Configurable event cap (`MAX_EVENTS`)

In `pipeline.py`, **`MAX_EVENTS`** limits how many **extracted events** are passed into **FR4** question generation (`events[:MAX_EVENTS]` after loading from the DB). Use a **small integer (e.g. 3)** for quick end-to-end runs; set to **`None`** to process **all** pending extracted events. Adjust the constant in code before running—useful for demos versus full backfills.

## Project structure

```
prediction-market-capstone/
├── config.py
├── models.py
├── pipeline.py
├── streamlit_app.py
├── db/
│   ├── schema.sql
│   └── connection.py
├── ingestion/           # FR1
├── clustering/          # FR2
├── extraction/          # FR3 (LLM)
├── generation/          # FR4 (LLM)
├── validation/          # FR5
├── scoring/             # FR6
├── tests/
└── sample_outputs/
```

## Known limitations

- GDELT DOC responses are headline-oriented; body text is not ingested from that API.
- Cluster quality depends on embedding model and DBSCAN settings for the current corpus.
- LLM stages (FR3–FR4) depend on provider rate limits and model behavior.
- Processing is **batch**, not streaming.
- PostgreSQL must be available for pipeline and UI reads.

## Extending the pipeline

Append a `(name, callable)` entry to **`STAGES`** in `pipeline.py`, add any new tables in `db/schema.sql`, and wire persistence in `db/connection.py` as needed.
