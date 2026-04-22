"""
FR7 Demo — runs without a database or pipeline.
Loads realistic sample questions so you can see the full UI.

Run:  streamlit run demo_app.py
"""
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import streamlit as st

UTC = timezone.utc

st.set_page_config(
    page_title="Prediction Market Intelligence · Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Sample data
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_ROWS = [
    {
        "rank": 1, "total_score": 0.91,
        "question_text": "Will the Federal Reserve cut interest rates at the June 2025 FOMC meeting?",
        "category": "finance", "question_type": "binary",
        "options": ["Yes", "No"],
        "deadline": "2025-06-18",
        "resolution_source": "Federal Reserve — https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        "resolution_criteria": "Resolves YES if the FOMC announces a reduction in the federal funds rate target range at the conclusion of the June 17–18, 2025 meeting. Resolves NO otherwise.",
        "rationale": "Rate decisions are among the most actively traded prediction markets globally. The June meeting follows May CPI data and will be closely watched.",
        "market_interest_score": 1.0, "resolution_strength_score": 1.0,
        "clarity_score": 1.0, "mention_velocity_score": 0.9,
        "novelty_score": 1.0, "time_horizon_score": 0.9, "source_diversity_score": 0.8,
        "resolution_confidence": 0.97, "source_independence": 0.95, "timing_reliability": 0.99,
    },
    {
        "rank": 2, "total_score": 0.87,
        "question_text": "Will Bitcoin (BTC) close above $100,000 on any day before July 1, 2025?",
        "category": "crypto", "question_type": "binary",
        "options": ["Yes", "No"],
        "deadline": "2025-07-01",
        "resolution_source": "CoinGecko — https://www.coingecko.com/en/coins/bitcoin",
        "resolution_criteria": "Resolves YES if Bitcoin's daily closing price (UTC 00:00) exceeds $100,000 on any calendar day on or before June 30, 2025 per CoinGecko. Resolves NO otherwise.",
        "rationale": "The $100k milestone is a major psychological level with enormous retail and institutional interest. High search volume and social media attention.",
        "market_interest_score": 1.0, "resolution_strength_score": 0.9,
        "clarity_score": 1.0, "mention_velocity_score": 1.0,
        "novelty_score": 0.9, "time_horizon_score": 0.85, "source_diversity_score": 0.9,
        "resolution_confidence": 0.95, "source_independence": 0.90, "timing_reliability": 0.98,
    },
    {
        "rank": 3, "total_score": 0.83,
        "question_text": "Will the FDA approve Eli Lilly's orforglipron oral GLP-1 drug before the end of 2025?",
        "category": "health", "question_type": "binary",
        "options": ["Yes", "No"],
        "deadline": "2025-12-31",
        "resolution_source": "FDA Drug Approvals — https://www.fda.gov/drugs/development-approval-process-drugs",
        "resolution_criteria": "Resolves YES if the FDA grants New Drug Application (NDA) approval for orforglipron (LY3502970) on or before December 31, 2025. Resolves NO otherwise.",
        "rationale": "First oral GLP-1 receptor agonist. Approval would be a landmark event in obesity treatment and would massively affect Eli Lilly's market cap.",
        "market_interest_score": 1.0, "resolution_strength_score": 1.0,
        "clarity_score": 1.0, "mention_velocity_score": 0.75,
        "novelty_score": 1.0, "time_horizon_score": 0.6, "source_diversity_score": 0.7,
        "resolution_confidence": 0.88, "source_independence": 0.96, "timing_reliability": 0.92,
    },
    {
        "rank": 4, "total_score": 0.79,
        "question_text": "Who will win the 2025 UEFA Champions League?",
        "category": "sports", "question_type": "multiple_choice",
        "options": ["Real Madrid", "Manchester City", "Bayern Munich", "Arsenal", "PSG"],
        "deadline": "2025-05-31",
        "resolution_source": "UEFA Official — https://www.uefa.com/uefachampionsleague/",
        "resolution_criteria": "Resolves to whichever club wins the 2024–25 UEFA Champions League Final played in Munich on May 31, 2025.",
        "rationale": "The Champions League final is the most-watched annual sporting event in the world and generates enormous prediction market volume.",
        "market_interest_score": 1.0, "resolution_strength_score": 1.0,
        "clarity_score": 1.0, "mention_velocity_score": 0.8,
        "novelty_score": 0.85, "time_horizon_score": 1.0, "source_diversity_score": 0.75,
        "resolution_confidence": 0.99, "source_independence": 0.97, "timing_reliability": 1.0,
    },
    {
        "rank": 5, "total_score": 0.75,
        "question_text": "Will the US Congress pass comprehensive AI regulation legislation before January 1, 2026?",
        "category": "technology", "question_type": "binary",
        "options": ["Yes", "No"],
        "deadline": "2025-12-31",
        "resolution_source": "Congress.gov — https://www.congress.gov/search?q=%22artificial+intelligence%22",
        "resolution_criteria": "Resolves YES if a bill specifically establishing binding federal AI regulations is signed into law by the President on or before December 31, 2025. Executive orders do not count.",
        "rationale": "Major legislative uncertainty following EU AI Act passage. High political salience with bipartisan interest but significant gridlock risk.",
        "market_interest_score": 1.0, "resolution_strength_score": 0.9,
        "clarity_score": 0.8, "mention_velocity_score": 0.7,
        "novelty_score": 1.0, "time_horizon_score": 0.6, "source_diversity_score": 0.8,
        "resolution_confidence": 0.82, "source_independence": 0.94, "timing_reliability": 0.88,
    },
    {
        "rank": 6, "total_score": 0.71,
        "question_text": "Will US CPI inflation fall below 2.5% year-over-year by September 2025?",
        "category": "economics", "question_type": "binary",
        "options": ["Yes", "No"],
        "deadline": "2025-10-15",
        "resolution_source": "Bureau of Labor Statistics — https://www.bls.gov/cpi/",
        "resolution_criteria": "Resolves YES if the BLS CPI report released in October 2025 (covering September 2025) shows a year-over-year headline CPI change strictly below 2.5%. Resolves NO otherwise.",
        "rationale": "Inflation trajectory is the primary driver of Fed policy expectations and bond markets. CPI releases are among the highest-impact scheduled economic events.",
        "market_interest_score": 1.0, "resolution_strength_score": 1.0,
        "clarity_score": 1.0, "mention_velocity_score": 0.65,
        "novelty_score": 0.7, "time_horizon_score": 0.7, "source_diversity_score": 0.6,
        "resolution_confidence": 0.94, "source_independence": 0.98, "timing_reliability": 0.99,
    },
    {
        "rank": 7, "total_score": 0.67,
        "question_text": "Will Apple announce a foldable iPhone at WWDC 2025?",
        "category": "technology", "question_type": "binary",
        "options": ["Yes", "No"],
        "deadline": "2025-06-13",
        "resolution_source": "Apple Newsroom — https://www.apple.com/newsroom/",
        "resolution_criteria": "Resolves YES if Apple officially announces or previews a foldable iPhone form factor at the WWDC 2025 keynote (June 9–13, 2025). Leaks or supply chain reports do not count.",
        "rationale": "Foldable iPhone has been rumoured for years. A WWDC announcement would be a major product milestone driving significant stock and consumer interest.",
        "market_interest_score": 0.8, "resolution_strength_score": 0.7,
        "clarity_score": 1.0, "mention_velocity_score": 0.6,
        "novelty_score": 1.0, "time_horizon_score": 0.9, "source_diversity_score": 0.6,
        "resolution_confidence": 0.78, "source_independence": 0.72, "timing_reliability": 0.95,
    },
    {
        "rank": 8, "total_score": 0.62,
        "question_text": "Will SpaceX successfully land Starship's Super Heavy booster in its next orbital test flight?",
        "category": "science", "question_type": "binary",
        "options": ["Yes", "No"],
        "deadline": "2025-08-31",
        "resolution_source": "SpaceX — https://www.spacex.com/launches/",
        "resolution_criteria": "Resolves YES if the Super Heavy booster is caught by the mechazilla arms or successfully soft-lands during the next Starship integrated flight test conducted before August 31, 2025.",
        "rationale": "Starship development is a top-tier science and tech story with massive public interest. Each flight test generates significant media coverage.",
        "market_interest_score": 0.8, "resolution_strength_score": 0.6,
        "clarity_score": 0.8, "mention_velocity_score": 0.55,
        "novelty_score": 0.9, "time_horizon_score": 0.75, "source_diversity_score": 0.5,
        "resolution_confidence": 0.75, "source_independence": 0.68, "timing_reliability": 0.70,
    },
    {
        "rank": 9, "total_score": 0.57,
        "question_text": "Will Germany hold a federal election re-run before September 2025?",
        "category": "politics", "question_type": "binary",
        "options": ["Yes", "No"],
        "deadline": "2025-09-01",
        "resolution_source": "Bundeswahlleiter — https://www.bundeswahlleiter.de/",
        "resolution_criteria": "Resolves YES if a federal Bundestag election takes place in Germany before September 1, 2025 following the collapse of the Scholz coalition. Resolves NO if no election is held by that date.",
        "rationale": "German snap election following coalition collapse. Outcome affects EU stability, energy policy, and NATO commitments.",
        "market_interest_score": 1.0, "resolution_strength_score": 0.9,
        "clarity_score": 0.8, "mention_velocity_score": 0.5,
        "novelty_score": 0.85, "time_horizon_score": 0.8, "source_diversity_score": 0.55,
        "resolution_confidence": 0.86, "source_independence": 0.92, "timing_reliability": 0.90,
    },
    {
        "rank": 10, "total_score": 0.51,
        "question_text": "Will Tesla's robotaxi service launch commercially in at least one US city by December 31, 2025?",
        "category": "technology", "question_type": "binary",
        "options": ["Yes", "No"],
        "deadline": "2025-12-31",
        "resolution_source": "Tesla Newsroom — https://www.tesla.com/blog",
        "resolution_criteria": "Resolves YES if Tesla begins paid commercial robotaxi rides (no safety driver) available to the general public in at least one US city on or before December 31, 2025.",
        "rationale": "Elon Musk has repeatedly promised robotaxi launch dates. High public interest and significant scepticism creates genuine market uncertainty.",
        "market_interest_score": 0.8, "resolution_strength_score": 0.4,
        "clarity_score": 0.8, "mention_velocity_score": 0.55,
        "novelty_score": 0.7, "time_horizon_score": 0.6, "source_diversity_score": 0.5,
        "resolution_confidence": 0.71, "source_independence": 0.65, "timing_reliability": 0.72,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens  (dark + orange)
# ─────────────────────────────────────────────────────────────────────────────
# Category: (text colour, bg tint hex for dark cards)
CAT_COLORS: dict = {
    "politics":      ("#60A5FA", "#1e3a5f"),
    "finance":       ("#34D399", "#064e3b"),
    "crypto":        ("#FBBF24", "#451a03"),
    "sports":        ("#F87171", "#450a0a"),
    "health":        ("#F472B6", "#500724"),
    "technology":    ("#A78BFA", "#2e1065"),
    "environment":   ("#4ADE80", "#052e16"),
    "science":       ("#38BDF8", "#082f49"),
    "legal":         ("#FB923C", "#431407"),
    "entertainment": ("#C084FC", "#3b0764"),
    "international": ("#818CF8", "#1e1b4b"),
    "economics":     ("#2DD4BF", "#042f2e"),
    "other":         ("#94A3B8", "#1e293b"),
}

COMPONENT_META = [
    ("market_interest_score",     "Market Interest",     "25%"),
    ("resolution_strength_score", "Resolution Strength", "25%"),
    ("clarity_score",             "Clarity",             "20%"),
    ("mention_velocity_score",    "Mention Velocity",    "10%"),
    ("novelty_score",             "Novelty",             "10%"),
    ("time_horizon_score",        "Time Horizon",         "5%"),
    ("source_diversity_score",    "Source Diversity",     "5%"),
]

SORT_OPTIONS = {
    "Rank (Best First)":     lambda r: r["rank"],
    "Score (Highest First)": lambda r: -float(r["total_score"]),
    "Deadline (Soonest)":    lambda r: r.get("deadline") or "9999-99-99",
    "Category (A–Z)":        lambda r: (r.get("category") or "").lower(),
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_color(s: float) -> str:
    """Bright colours readable on a dark background."""
    if s >= 0.75: return "#34D399"   # emerald
    if s >= 0.55: return "#60A5FA"   # sky blue
    if s >= 0.40: return "#F97316"   # orange  ← brand accent
    return "#F87171"                  # red


def _days_until(deadline_str: str) -> Optional[int]:
    if not deadline_str: return None
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%Y/%m/%d"):
        try:
            d = datetime.strptime(deadline_str.strip(), fmt).date()
            return (d - datetime.now(UTC).date()).days
        except ValueError:
            continue
    return None


def _cat_badge(cat: str) -> str:
    text, bg = CAT_COLORS.get((cat or "other").lower(), ("#94A3B8", "#1e293b"))
    label = (cat or "other").replace("_", " ").title()
    return (
        f'<span class="badge" style="'
        f'color:{text};background:{bg}CC;'
        f'border:1px solid {text}55;">'
        f'{label}</span>'
    )


def _type_badge(qtype: str) -> str:
    if (qtype or "").lower() == "binary":
        return (
            '<span class="badge" style="'
            'color:#F97316;background:#431407CC;'
            'border:1px solid #F9731655;">⊙ Binary</span>'
        )
    return (
        '<span class="badge" style="'
        'color:#A78BFA;background:#2e1065CC;'
        'border:1px solid #A78BFA55;">⊞ Multiple Choice</span>'
    )


def _rank_badge(rank: int) -> str:
    if rank == 1:
        style = "background:linear-gradient(135deg,#F97316,#EA580C);"
    elif rank == 2:
        style = "background:linear-gradient(135deg,#FB923C,#C2410C);"
    elif rank == 3:
        style = "background:linear-gradient(135deg,#FDBA74,#EA580C);"
    else:
        style = "background:#1E293B;border:1px solid #334155;"
    return f'<div class="rank-badge" style="{style}">{rank}</div>'


def _score_circle(score: float) -> str:
    color = _score_color(score)
    pct = int(score * 100)
    return (
        f'<div class="score-circle" style="'
        f'border:3px solid {color};color:{color};">'
        f'<span class="sc-num">{pct}</span>'
        f'<span class="sc-sub">/ 100</span>'
        f'</div>'
    )


def _days_chip(days: Optional[int]) -> str:
    if days is None:  return ""
    if days < 0:      return '<span class="days-chip days-urgent">Expired</span>'
    if days == 0:     return '<span class="days-chip days-urgent">Today</span>'
    if days <= 7:     return f'<span class="days-chip days-urgent">{days}d left</span>'
    if days <= 30:    return f'<span class="days-chip days-soon">{days}d left</span>'
    return                   f'<span class="days-chip days-ok">{days}d left</span>'


def _source_link(source: str) -> str:
    m = re.search(r"https?://[^\s)]+", source or "")
    if m:
        url = m.group(0).rstrip(".,;")
        domain = urlparse(url).netloc.replace("www.", "")
        return f'<a href="{url}" target="_blank" rel="noopener">{domain} ↗</a>'
    s = (source or "—").strip()
    return s[:55] + ("…" if len(s) > 55 else "")


def _breakdown_row(label: str, weight: str, value: float) -> str:
    color = _score_color(value)
    pct = int(value * 100)
    return (
        f'<div class="br-row">'
        f'<span class="br-label">{label}</span>'
        f'<span class="br-weight">{weight}</span>'
        f'<div class="br-track">'
        f'<div class="br-fill" style="width:{pct}%;background:{color};"></div>'
        f'</div>'
        f'<span class="br-val" style="color:{color};">{pct}%</span>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
def _inject_css() -> None:
    st.markdown("""
<style>
/* ── Fonts ───────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Hide Streamlit chrome ───────────────────────────────── */
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── Page ────────────────────────────────────────────────── */
.main { background: #0F1117; }
.main .block-container {
    padding: 2rem 2.5rem 4rem 2.5rem;
    max-width: 1300px;
}

/* ── Sidebar ─────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #14161E !important;
    border-right: 1px solid #1E2535;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 1.75rem 1.25rem 1.25rem 1.25rem;
}

/* ── Demo banner ─────────────────────────────────────────── */
.demo-banner {
    background: #1A1400;
    border: 1px solid #F9731640;
    border-left: 3px solid #F97316;
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 24px;
    font-size: 14px;
    color: #FDBA74;
    font-weight: 500;
    line-height: 1.5;
}
.demo-banner strong { color: #FED7AA; }

/* ── App header ──────────────────────────────────────────── */
.app-header { margin-bottom: 4px; }
.app-title {
    font-size: 30px;
    font-weight: 800;
    color: #F1F5F9;
    letter-spacing: -0.8px;
    line-height: 1.15;
    margin: 0;
}
.app-title span { color: #F97316; }   /* orange accent on icon/word */
.app-subtitle {
    font-size: 15px;
    color: #94A3B8;
    margin-top: 6px;
    font-weight: 400;
    line-height: 1.6;
}

/* ── Stat strip ──────────────────────────────────────────── */
.stat-grid {
    display: flex;
    gap: 12px;
    margin: 22px 0 30px 0;
}
.stat-card {
    flex: 1;
    background: #14161E;
    border: 1px solid #1E2535;
    border-radius: 14px;
    padding: 18px 20px;
    transition: border-color 0.15s, box-shadow 0.15s;
}
.stat-card:hover {
    border-color: #F9731640;
    box-shadow: 0 0 0 1px #F9731620;
}
.stat-value {
    font-size: 30px;
    font-weight: 800;
    color: #F1F5F9;
    letter-spacing: -0.5px;
    line-height: 1;
}
.stat-label {
    font-size: 11px;
    color: #64748B;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 6px;
}

/* ── Question card ───────────────────────────────────────── */
.q-card {
    background: #14161E;
    border: 1px solid #1E2535;
    border-radius: 16px;
    padding: 24px 26px 20px 26px;
    margin-bottom: 10px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
    position: relative;
}
.q-card:hover {
    border-color: #F9731660;
    box-shadow: 0 0 0 1px #F9731625, 0 12px 40px rgba(0,0,0,0.5);
    transform: translateY(-2px);
}

/* Card header row */
.q-header { display: flex; align-items: flex-start; gap: 14px; }

/* Rank circle */
.rank-badge {
    min-width: 36px;
    height: 36px;
    border-radius: 50%;
    color: #FFFFFF;
    font-size: 13px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 3px;
    letter-spacing: -0.3px;
}

/* Question text */
.q-text {
    font-size: 17px;
    font-weight: 600;
    color: #F1F5F9;
    line-height: 1.5;
    flex: 1;
}

/* Score circle — outline style, no fill */
.score-circle {
    min-width: 54px;
    height: 54px;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    line-height: 1;
    background: transparent;
}
.sc-num { font-size: 17px; font-weight: 800; }
.sc-sub { font-size: 9px;  font-weight: 500; opacity: 0.6; margin-top: 1px; }

/* Badges row */
.badges-row {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin: 12px 0 0 50px;
}
.badge {
    font-size: 12px;
    font-weight: 600;
    padding: 4px 11px;
    border-radius: 20px;
    white-space: nowrap;
}

/* Score bar */
.score-bar-wrap {
    margin: 16px 0 12px 0;
    background: #1E293B;
    border-radius: 6px;
    height: 5px;
    overflow: hidden;
}
.score-bar-fill { height: 100%; border-radius: 6px; }

/* Meta row */
.q-meta {
    display: flex;
    gap: 22px;
    flex-wrap: wrap;
    font-size: 13.5px;
    color: #64748B;
    margin-top: 4px;
    align-items: center;
}
.meta-item { display: flex; align-items: center; gap: 6px; }
.meta-item a {
    color: #F97316;
    text-decoration: none;
    font-weight: 500;
}
.meta-item a:hover { color: #FB923C; text-decoration: underline; }

/* Days chip */
.days-chip {
    font-size: 11.5px;
    font-weight: 600;
    padding: 2px 9px;
    border-radius: 10px;
    margin-left: 5px;
}
.days-urgent { background: #450a0a; color: #F87171; border: 1px solid #F8717140; }
.days-soon   { background: #1c1400; color: #FBBF24; border: 1px solid #FBBF2440; }
.days-ok     { background: #052e16; color: #4ADE80; border: 1px solid #4ADE8040; }

/* Options pills */
.options-row { display: flex; gap: 7px; flex-wrap: wrap; margin-top: 12px; }
.option-pill {
    font-size: 13px;
    font-weight: 500;
    padding: 5px 14px;
    border-radius: 20px;
    background: #1E293B;
    border: 1px solid #334155;
    color: #CBD5E1;
}

/* ── Score breakdown (inside expander) ──────────────────── */
.br-section-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #475569;
    margin: 0 0 14px 0;
}
.br-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 11px;
}
.br-label {
    font-size: 13px;
    color: #CBD5E1;
    min-width: 175px;
    flex-shrink: 0;
    font-weight: 500;
}
.br-weight {
    font-size: 11px;
    color: #475569;
    width: 32px;
    text-align: right;
    flex-shrink: 0;
    font-weight: 600;
}
.br-track {
    flex: 1;
    background: #1E293B;
    border-radius: 5px;
    height: 9px;
    overflow: hidden;
}
.br-fill { height: 100%; border-radius: 5px; }
.br-val {
    font-size: 13px;
    font-weight: 700;
    width: 38px;
    text-align: right;
    flex-shrink: 0;
}

/* LLM quality chips */
.llm-grid {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin: 18px 0 4px 0;
}
.llm-chip {
    font-size: 13px;
    padding: 8px 14px;
    border-radius: 10px;
    background: #1A1D27;
    border: 1px solid #2D3348;
    color: #94A3B8;
    line-height: 1.4;
}
.llm-chip strong { color: #E2E8F0; font-weight: 700; font-size: 14px; }
.llm-label {
    font-size: 10px;
    color: #475569;
    display: block;
    margin-bottom: 2px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Detail blocks */
.detail-header {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    color: #475569;
    margin: 20px 0 8px 0;
}
.detail-text {
    font-size: 14px;
    color: #CBD5E1;
    line-height: 1.7;
    padding: 14px 16px;
    background: #1A1D27;
    border-radius: 10px;
    border: 1px solid #2D3348;
}
.detail-text.muted {
    color: #94A3B8;
    font-style: italic;
}

/* ── Sidebar labels ──────────────────────────────────────── */
.sb-section {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    color: #475569;
    margin: 22px 0 8px 0;
}

/* ── Results bar ─────────────────────────────────────────── */
.results-bar {
    font-size: 14px;
    color: #64748B;
    font-weight: 500;
    margin-bottom: 18px;
    padding-bottom: 14px;
    border-bottom: 1px solid #1E2535;
}
.results-bar strong { color: #94A3B8; font-weight: 600; }

/* ── Empty state ─────────────────────────────────────────── */
.empty-state { text-align: center; padding: 100px 20px; }
.empty-icon  { font-size: 56px; margin-bottom: 18px; }
.empty-title { font-size: 22px; font-weight: 700; color: #94A3B8; margin-bottom: 10px; }
.empty-sub   { font-size: 15px; color: #475569; line-height: 1.7; max-width: 380px; margin: 0 auto; }

/* ── Streamlit expander override ─────────────────────────── */
details summary p {
    font-size: 13px !important;
    color: #64748B !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Card renderer
# ─────────────────────────────────────────────────────────────────────────────
def _render_card(row: dict) -> None:
    rank  = int(row["rank"])
    score = float(row["total_score"])
    color = _score_color(score)
    pct   = int(score * 100)
    days  = _days_until(row.get("deadline", ""))

    options   = row.get("options", [])
    opts_html = ""
    if options:
        pills = "".join(f'<span class="option-pill">{o}</span>' for o in options)
        opts_html = f'<div class="options-row">{pills}</div>'

    st.markdown(f"""
<div class="q-card">
  <div class="q-header">
    {_rank_badge(rank)}
    <div class="q-text">{row['question_text']}</div>
    {_score_circle(score)}
  </div>
  <div class="badges-row">
    {_cat_badge(row.get('category', 'other'))}
    {_type_badge(row.get('question_type', 'binary'))}
  </div>
  <div class="score-bar-wrap">
    <div class="score-bar-fill" style="width:{pct}%;background:{color};"></div>
  </div>
  <div class="q-meta">
    <span class="meta-item">📅&nbsp;{row.get('deadline', '—')}{_days_chip(days)}</span>
    <span class="meta-item">🔗&nbsp;{_source_link(row.get('resolution_source', ''))}</span>
  </div>
  {opts_html}
</div>""", unsafe_allow_html=True)

    with st.expander("Score breakdown & details", expanded=False):
        # Component bars
        bars = '<div class="br-section-title">Score Components</div>'
        for key, label, weight in COMPONENT_META:
            bars += _breakdown_row(label, weight, float(row.get(key, 0)))

        # LLM quality chips
        rc = row.get("resolution_confidence", 0)
        si = row.get("source_independence", 0)
        tr = row.get("timing_reliability", 0)
        llm = f"""
<div class="llm-grid">
  <div class="llm-chip">
    <span class="llm-label">Resolution Confidence</span>
    <strong style="color:{_score_color(rc)};">{int(rc*100)}%</strong>
  </div>
  <div class="llm-chip">
    <span class="llm-label">Source Independence</span>
    <strong style="color:{_score_color(si)};">{int(si*100)}%</strong>
  </div>
  <div class="llm-chip">
    <span class="llm-label">Timing Reliability</span>
    <strong style="color:{_score_color(tr)};">{int(tr*100)}%</strong>
  </div>
</div>"""
        st.markdown(bars + llm, unsafe_allow_html=True)

        if row.get("resolution_criteria"):
            st.markdown(
                '<div class="detail-header">Resolution Criteria</div>'
                f'<div class="detail-text">{row["resolution_criteria"]}</div>',
                unsafe_allow_html=True,
            )
        if row.get("rationale"):
            st.markdown(
                '<div class="detail-header">Why This Market?</div>'
                f'<div class="detail-text muted">{row["rationale"]}</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    _inject_css()

    all_rows = SAMPLE_ROWS

    # ── Header ───────────────────────────────────────────────
    st.markdown(
        '<div class="app-header">'
        '<div class="app-title"><span>🎯</span> Prediction Market Intelligence</div>'
        '<div class="app-subtitle">'
        'AI-curated prediction markets — ranked by resolvability, market interest &amp; quality'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Demo banner ──────────────────────────────────────────
    st.markdown(
        '<div class="demo-banner">'
        '🧪 <strong>Demo mode</strong> — 10 sample markets shown. '
        'Run the full pipeline (FR1–FR6) to populate with live AI-generated questions.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Stats strip ──────────────────────────────────────────
    cats      = set(r["category"] for r in all_rows)
    avg_score = sum(r["total_score"] for r in all_rows) / len(all_rows)
    top_score = max(r["total_score"] for r in all_rows)
    binary_n  = sum(1 for r in all_rows if r["question_type"] == "binary")
    mc_n      = len(all_rows) - binary_n

    st.markdown(f"""
<div class="stat-grid">
  <div class="stat-card">
    <div class="stat-value">{len(all_rows)}</div>
    <div class="stat-label">Ranked Markets</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{len(cats)}</div>
    <div class="stat-label">Categories</div>
  </div>
  <div class="stat-card">
    <div class="stat-value" style="color:#F97316;">{int(avg_score*100)}</div>
    <div class="stat-label">Avg Score</div>
  </div>
  <div class="stat-card">
    <div class="stat-value" style="color:#34D399;">{int(top_score*100)}</div>
    <div class="stat-label">Top Score</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{binary_n}<span style="color:#475569;font-size:18px;"> / </span>{mc_n}</div>
    <div class="stat-label">Binary / Multi</div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Sidebar filters ──────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div style="font-size:20px;font-weight:800;color:#F1F5F9;letter-spacing:-0.4px;">'
            '🔧 Filters</div>'
            '<div style="font-size:13px;color:#64748B;margin-top:4px;margin-bottom:20px;">'
            'Refine the market list</div>',
            unsafe_allow_html=True,
        )

        search = st.text_input(
            "Search", placeholder="e.g. Fed, Bitcoin, election…",
            label_visibility="collapsed",
        )
        st.caption("🔍  Search question text")

        st.markdown('<div class="sb-section">Category</div>', unsafe_allow_html=True)
        all_cats = sorted(set(r["category"] for r in all_rows))
        selected_cats = st.multiselect(
            "cat", options=all_cats, default=all_cats,
            format_func=lambda x: x.replace("_", " ").title(),
            label_visibility="collapsed",
        )

        st.markdown('<div class="sb-section">Question Type</div>', unsafe_allow_html=True)
        type_choice = st.radio(
            "type", ["All", "Binary", "Multiple Choice"],
            horizontal=True, label_visibility="collapsed",
        )

        st.markdown('<div class="sb-section">Minimum Score</div>', unsafe_allow_html=True)
        min_score_pct = st.slider(
            "min", 0, 100, 0, step=5,
            label_visibility="collapsed",
            help="Hide markets with a total score below this threshold",
        )

        st.markdown('<div class="sb-section">Sort By</div>', unsafe_allow_html=True)
        sort_choice = st.selectbox(
            "sort", list(SORT_OPTIONS.keys()),
            label_visibility="collapsed",
        )

        st.markdown(
            '<div style="margin-top:32px;padding-top:20px;border-top:1px solid #1E2535;">'
            '<div style="font-size:11px;color:#334155;font-weight:500;line-height:1.6;">'
            'Scores are weighted composites of market interest, resolution quality, '
            'clarity, velocity, novelty, time horizon, and source diversity.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    # ── Filter & sort ────────────────────────────────────────
    filtered = list(all_rows)

    if search:
        q = search.strip().lower()
        filtered = [r for r in filtered if q in r["question_text"].lower()]

    if selected_cats:
        filtered = [r for r in filtered if r["category"] in selected_cats]

    if type_choice != "All":
        target = "binary" if type_choice == "Binary" else "multiple_choice"
        filtered = [r for r in filtered if r["question_type"] == target]

    if min_score_pct > 0:
        filtered = [r for r in filtered if r["total_score"] >= min_score_pct / 100]

    filtered.sort(key=SORT_OPTIONS[sort_choice])

    # ── Empty state ──────────────────────────────────────────
    if not filtered:
        st.markdown("""
<div class="empty-state">
  <div class="empty-icon">🔍</div>
  <div class="empty-title">No matches found</div>
  <div class="empty-sub">Try broadening your search or relaxing the filters.</div>
</div>""", unsafe_allow_html=True)
        return

    # ── Results bar ──────────────────────────────────────────
    suffix = (
        f" &nbsp;·&nbsp; filtered from <strong>{len(all_rows)}</strong>"
        if len(filtered) < len(all_rows) else ""
    )
    st.markdown(
        f'<div class="results-bar">'
        f'<strong>{len(filtered)}</strong> market{"s" if len(filtered) != 1 else ""}'
        f'{suffix}</div>',
        unsafe_allow_html=True,
    )

    # ── Cards ────────────────────────────────────────────────
    for row in filtered:
        _render_card(row)


if __name__ == "__main__":
    main()
