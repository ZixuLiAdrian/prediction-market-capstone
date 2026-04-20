"""
FR7: Prediction Market Intelligence Dashboard

A modern dark-themed Streamlit UI for browsing, filtering, and inspecting
AI-curated prediction markets ranked by FR6.
"""
import json
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import streamlit as st

from db.connection import get_ranked_scored_questions

UTC = timezone.utc

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediction Market Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
# Category: (text colour, dark bg tint)
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
    if s >= 0.75: return "#34D399"   # emerald
    if s >= 0.55: return "#60A5FA"   # sky blue
    if s >= 0.40: return "#F97316"   # orange — brand accent
    return "#F87171"                  # red


def _days_until(deadline_str: str) -> Optional[int]:
    if not deadline_str:
        return None
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%Y/%m/%d", "%d %B %Y"):
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
        f'border:1px solid {text}55;">{label}</span>'
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
        f'<div class="score-circle" style="border:3px solid {color};color:{color};">'
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


def _parse_options(raw) -> list:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return []
    return []


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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
.main { background: #0F1117; }
.main .block-container { padding: 2rem 2.5rem 4rem 2.5rem; max-width: 1300px; }
section[data-testid="stSidebar"] { background: #14161E !important; border-right: 1px solid #1E2535; }
section[data-testid="stSidebar"] > div:first-child { padding: 1.75rem 1.25rem 1.25rem 1.25rem; }

.app-header { margin-bottom: 4px; }
.app-title { font-size: 30px; font-weight: 800; color: #F1F5F9; letter-spacing: -0.8px; line-height: 1.15; margin: 0; }
.app-title span { color: #F97316; }
.app-subtitle { font-size: 15px; color: #94A3B8; margin-top: 6px; font-weight: 400; line-height: 1.6; }

.stat-grid { display: flex; gap: 12px; margin: 22px 0 30px 0; }
.stat-card { flex: 1; background: #14161E; border: 1px solid #1E2535; border-radius: 14px; padding: 18px 20px; transition: border-color 0.15s, box-shadow 0.15s; }
.stat-card:hover { border-color: #F9731640; box-shadow: 0 0 0 1px #F9731620; }
.stat-value { font-size: 30px; font-weight: 800; color: #F1F5F9; letter-spacing: -0.5px; line-height: 1; }
.stat-label { font-size: 11px; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 6px; }

.q-card { background: #14161E; border: 1px solid #1E2535; border-radius: 16px; padding: 24px 26px 20px 26px; margin-bottom: 10px; transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease; position: relative; }
.q-card:hover { border-color: #F9731660; box-shadow: 0 0 0 1px #F9731625, 0 12px 40px rgba(0,0,0,0.5); transform: translateY(-2px); }
.q-header { display: flex; align-items: flex-start; gap: 14px; }
.rank-badge { min-width: 36px; height: 36px; border-radius: 50%; color: #FFFFFF; font-size: 13px; font-weight: 700; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 3px; letter-spacing: -0.3px; }
.q-text { font-size: 17px; font-weight: 600; color: #F1F5F9; line-height: 1.5; flex: 1; }
.score-circle { min-width: 54px; height: 54px; border-radius: 50%; display: flex; flex-direction: column; align-items: center; justify-content: center; flex-shrink: 0; line-height: 1; background: transparent; }
.sc-num { font-size: 17px; font-weight: 800; }
.sc-sub { font-size: 9px; font-weight: 500; opacity: 0.6; margin-top: 1px; }
.badges-row { display: flex; gap: 6px; flex-wrap: wrap; margin: 12px 0 0 50px; }
.badge { font-size: 12px; font-weight: 600; padding: 4px 11px; border-radius: 20px; white-space: nowrap; }
.score-bar-wrap { margin: 16px 0 12px 0; background: #1E293B; border-radius: 6px; height: 5px; overflow: hidden; }
.score-bar-fill { height: 100%; border-radius: 6px; }
.q-meta { display: flex; gap: 22px; flex-wrap: wrap; font-size: 13.5px; color: #64748B; margin-top: 4px; align-items: center; }
.meta-item { display: flex; align-items: center; gap: 6px; }
.meta-item a { color: #F97316; text-decoration: none; font-weight: 500; }
.meta-item a:hover { color: #FB923C; text-decoration: underline; }
.days-chip { font-size: 11.5px; font-weight: 600; padding: 2px 9px; border-radius: 10px; margin-left: 5px; }
.days-urgent { background: #450a0a; color: #F87171; border: 1px solid #F8717140; }
.days-soon   { background: #1c1400; color: #FBBF24; border: 1px solid #FBBF2440; }
.days-ok     { background: #052e16; color: #4ADE80; border: 1px solid #4ADE8040; }
.options-row { display: flex; gap: 7px; flex-wrap: wrap; margin-top: 12px; }
.option-pill { font-size: 13px; font-weight: 500; padding: 5px 14px; border-radius: 20px; background: #1E293B; border: 1px solid #334155; color: #CBD5E1; }

.br-section-title { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #475569; margin: 0 0 14px 0; }
.br-row { display: flex; align-items: center; gap: 12px; margin-bottom: 11px; }
.br-label  { font-size: 13px; color: #CBD5E1; min-width: 175px; flex-shrink: 0; font-weight: 500; }
.br-weight { font-size: 11px; color: #475569; width: 32px; text-align: right; flex-shrink: 0; font-weight: 600; }
.br-track  { flex: 1; background: #1E293B; border-radius: 5px; height: 9px; overflow: hidden; }
.br-fill   { height: 100%; border-radius: 5px; }
.br-val    { font-size: 13px; font-weight: 700; width: 38px; text-align: right; flex-shrink: 0; }

.llm-grid { display: flex; gap: 10px; flex-wrap: wrap; margin: 18px 0 4px 0; }
.llm-chip { font-size: 13px; padding: 8px 14px; border-radius: 10px; background: #1A1D27; border: 1px solid #2D3348; color: #94A3B8; line-height: 1.4; }
.llm-chip strong { color: #E2E8F0; font-weight: 700; font-size: 14px; }
.llm-label { font-size: 10px; color: #475569; display: block; margin-bottom: 2px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }

.detail-header { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.9px; color: #475569; margin: 20px 0 8px 0; }
.detail-text { font-size: 14px; color: #CBD5E1; line-height: 1.7; padding: 14px 16px; background: #1A1D27; border-radius: 10px; border: 1px solid #2D3348; }
.detail-text.muted { color: #94A3B8; font-style: italic; }

.sb-section { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.9px; color: #475569; margin: 22px 0 8px 0; }
.results-bar { font-size: 14px; color: #64748B; font-weight: 500; margin-bottom: 18px; padding-bottom: 14px; border-bottom: 1px solid #1E2535; }
.results-bar strong { color: #94A3B8; font-weight: 600; }
.empty-state { text-align: center; padding: 100px 20px; }
.empty-icon  { font-size: 56px; margin-bottom: 18px; }
.empty-title { font-size: 22px; font-weight: 700; color: #94A3B8; margin-bottom: 10px; }
.empty-sub   { font-size: 15px; color: #475569; line-height: 1.7; max-width: 380px; margin: 0 auto; }

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
    score = float(row.get("total_score") or 0)
    color = _score_color(score)
    pct   = int(score * 100)
    days  = _days_until((row.get("deadline") or "").strip())

    options  = _parse_options(row.get("options"))
    opts_html = ""
    if options:
        pills = "".join(f'<span class="option-pill">{o}</span>' for o in options)
        opts_html = f'<div class="options-row">{pills}</div>'

    res_conf = float(row.get("resolution_confidence") or 0)
    src_ind  = float(row.get("source_independence")   or 0)
    timing   = float(row.get("timing_reliability")    or 0)

    st.markdown(f"""
<div class="q-card">
  <div class="q-header">
    {_rank_badge(rank)}
    <div class="q-text">{row.get('question_text','')}</div>
    {_score_circle(score)}
  </div>
  <div class="badges-row">
    {_cat_badge(row.get('category','other'))}
    {_type_badge(row.get('question_type','binary'))}
  </div>
  <div class="score-bar-wrap">
    <div class="score-bar-fill" style="width:{pct}%;background:{color};"></div>
  </div>
  <div class="q-meta">
    <span class="meta-item">📅&nbsp;{row.get('deadline','—') or '—'}{_days_chip(days)}</span>
    <span class="meta-item">🔗&nbsp;{_source_link(row.get('resolution_source',''))}</span>
  </div>
  {opts_html}
</div>""", unsafe_allow_html=True)

    with st.expander("Score breakdown & details", expanded=False):
        bars = '<div class="br-section-title">Score Components</div>'
        for key, label, weight in COMPONENT_META:
            bars += _breakdown_row(label, weight, float(row.get(key) or 0))

        llm = f"""
<div class="llm-grid">
  <div class="llm-chip">
    <span class="llm-label">Resolution Confidence</span>
    <strong style="color:{_score_color(res_conf)};">{int(res_conf*100)}%</strong>
  </div>
  <div class="llm-chip">
    <span class="llm-label">Source Independence</span>
    <strong style="color:{_score_color(src_ind)};">{int(src_ind*100)}%</strong>
  </div>
  <div class="llm-chip">
    <span class="llm-label">Timing Reliability</span>
    <strong style="color:{_score_color(timing)};">{int(timing*100)}%</strong>
  </div>
</div>"""
        st.markdown(bars + llm, unsafe_allow_html=True)

        resolution_crit = (row.get("resolution_criteria") or "").strip()
        rationale       = (row.get("rationale") or "").strip()

        if resolution_crit:
            st.markdown(
                '<div class="detail-header">Resolution Criteria</div>'
                f'<div class="detail-text">{resolution_crit}</div>',
                unsafe_allow_html=True,
            )
        if rationale:
            st.markdown(
                '<div class="detail-header">Why This Market?</div>'
                f'<div class="detail-text muted">{rationale}</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    _inject_css()

    all_rows = [dict(r) for r in get_ranked_scored_questions()]

    # Header
    hdr_left, hdr_right = st.columns([5, 1])
    with hdr_left:
        st.markdown(
            '<div class="app-header">'
            '<div class="app-title"><span>🎯</span> Prediction Market Intelligence</div>'
            '<div class="app-subtitle">'
            'AI-curated prediction markets — ranked by resolvability, market interest &amp; quality'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with hdr_right:
        st.write("")
        if st.button("↻  Refresh", use_container_width=True):
            st.rerun()

    # Stats strip
    if all_rows:
        cats      = set((r.get("category") or "other").lower() for r in all_rows)
        avg_score = sum(float(r["total_score"]) for r in all_rows) / len(all_rows)
        top_score = max(float(r["total_score"]) for r in all_rows)
        binary_n  = sum(1 for r in all_rows if (r.get("question_type") or "") == "binary")
        mc_n      = len(all_rows) - binary_n

        st.markdown(f"""
<div class="stat-grid">
  <div class="stat-card"><div class="stat-value">{len(all_rows)}</div><div class="stat-label">Ranked Markets</div></div>
  <div class="stat-card"><div class="stat-value">{len(cats)}</div><div class="stat-label">Categories</div></div>
  <div class="stat-card"><div class="stat-value" style="color:#F97316;">{int(avg_score*100)}</div><div class="stat-label">Avg Score</div></div>
  <div class="stat-card"><div class="stat-value" style="color:#34D399;">{int(top_score*100)}</div><div class="stat-label">Top Score</div></div>
  <div class="stat-card"><div class="stat-value">{binary_n}<span style="color:#475569;font-size:18px;"> / </span>{mc_n}</div><div class="stat-label">Binary / Multi</div></div>
</div>""", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown(
            '<div style="font-size:20px;font-weight:800;color:#F1F5F9;letter-spacing:-0.4px;">🔧 Filters</div>'
            '<div style="font-size:13px;color:#64748B;margin-top:4px;margin-bottom:20px;">Refine the market list</div>',
            unsafe_allow_html=True,
        )

        search = st.text_input("Search", placeholder="e.g. Fed, Bitcoin, election…", label_visibility="collapsed")
        st.caption("🔍  Search question text")

        st.markdown('<div class="sb-section">Category</div>', unsafe_allow_html=True)
        all_cats = sorted(set((r.get("category") or "other").lower() for r in all_rows))
        selected_cats = st.multiselect(
            "cat", options=all_cats, default=all_cats,
            format_func=lambda x: x.replace("_", " ").title(),
            label_visibility="collapsed",
        )

        st.markdown('<div class="sb-section">Question Type</div>', unsafe_allow_html=True)
        type_choice = st.radio("type", ["All", "Binary", "Multiple Choice"], horizontal=True, label_visibility="collapsed")

        st.markdown('<div class="sb-section">Minimum Score</div>', unsafe_allow_html=True)
        min_score_pct = st.slider("min_score", 0, 100, 0, step=5, label_visibility="collapsed",
                                  help="Hide markets with a total score below this threshold")

        st.markdown('<div class="sb-section">Sort By</div>', unsafe_allow_html=True)
        sort_choice = st.selectbox("sort", list(SORT_OPTIONS.keys()), label_visibility="collapsed")

        st.markdown(
            '<div style="margin-top:32px;padding-top:20px;border-top:1px solid #1E2535;">'
            '<div style="font-size:11px;color:#334155;font-weight:500;line-height:1.6;">'
            'Scores are weighted composites of market interest, resolution quality, '
            'clarity, velocity, novelty, time horizon, and source diversity.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    # Filter & sort
    filtered = list(all_rows)
    if search:
        q = search.strip().lower()
        filtered = [r for r in filtered if q in (r.get("question_text") or "").lower()]
    if selected_cats:
        filtered = [r for r in filtered if (r.get("category") or "other").lower() in selected_cats]
    if type_choice != "All":
        target = "binary" if type_choice == "Binary" else "multiple_choice"
        filtered = [r for r in filtered if (r.get("question_type") or "") == target]
    if min_score_pct > 0:
        filtered = [r for r in filtered if float(r.get("total_score") or 0) >= min_score_pct / 100]
    filtered.sort(key=SORT_OPTIONS[sort_choice])

    # Empty states
    if not all_rows:
        st.markdown("""
<div class="empty-state">
  <div class="empty-icon">🔬</div>
  <div class="empty-title">No markets scored yet</div>
  <div class="empty-sub">Run the full pipeline (FR1–FR6) to generate, validate, and score prediction markets.</div>
</div>""", unsafe_allow_html=True)
        return

    if not filtered:
        st.markdown("""
<div class="empty-state">
  <div class="empty-icon">🔍</div>
  <div class="empty-title">No matches found</div>
  <div class="empty-sub">Try broadening your search or relaxing the filters.</div>
</div>""", unsafe_allow_html=True)
        return

    # Results bar
    suffix = f" &nbsp;·&nbsp; filtered from <strong>{len(all_rows)}</strong>" if len(filtered) < len(all_rows) else ""
    st.markdown(
        f'<div class="results-bar"><strong>{len(filtered)}</strong> market{"s" if len(filtered)!=1 else ""}{suffix}</div>',
        unsafe_allow_html=True,
    )

    for row in filtered:
        _render_card(row)


if __name__ == "__main__":
    main()
