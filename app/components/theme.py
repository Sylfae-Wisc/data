"""Shared visual theme helpers for the Streamlit app."""

import streamlit as st


def apply_global_styles() -> None:
    """Apply the VCT dashboard theme on every Streamlit page."""
    st.markdown(
        """
<style>
    :root {
        --vct-red: #ff4655;
        --vct-red-soft: rgba(255, 70, 85, 0.16);
        --vct-cyan: #22d3ee;
        --vct-blue: #38bdf8;
        --vct-green: #86efac;
        --vct-orange: #fbbf24;
        --vct-purple: #c084fc;
        --vct-bg: #080b14;
        --vct-bg-2: #0c1220;
        --vct-surface: rgba(15, 23, 42, 0.92);
        --vct-surface-2: rgba(24, 34, 56, 0.9);
        --vct-border: rgba(226, 232, 240, 0.12);
        --vct-border-strong: rgba(255, 70, 85, 0.36);
        --vct-text: #f8fafc;
        --vct-text-secondary: #aab7cc;
        --vct-text-muted: #75839a;
        --vct-shadow: 0 18px 44px rgba(0, 0, 0, 0.34);
        --vct-radius: 8px;
    }

    .stApp {
        color: var(--vct-text);
        background:
            linear-gradient(135deg, rgba(255, 70, 85, 0.12), transparent 26%),
            linear-gradient(215deg, rgba(34, 211, 238, 0.08), transparent 30%),
            repeating-linear-gradient(90deg, rgba(255,255,255,0.025) 0 1px, transparent 1px 80px),
            var(--vct-bg);
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
            linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
        background-size: 36px 36px;
        mask-image: linear-gradient(to bottom, black 0%, transparent 72%);
    }

    .main .block-container {
        max-width: 1280px;
        padding-top: 1.35rem;
        padding-bottom: 3rem;
    }

    p, li, span, div, label {
        color: inherit;
    }

    a {
        color: var(--vct-cyan) !important;
        text-decoration: none !important;
    }

    h1, h2, h3, h4 {
        color: var(--vct-text) !important;
        letter-spacing: 0 !important;
    }

    .vct-hero {
        position: relative;
        overflow: hidden;
        padding: 1.35rem 1.45rem;
        margin: 0 0 1.25rem;
        border: 1px solid var(--vct-border);
        border-left: 3px solid var(--vct-red);
        border-radius: var(--vct-radius);
        background:
            linear-gradient(135deg, rgba(255,70,85,0.18), transparent 42%),
            linear-gradient(180deg, rgba(20,31,54,0.96), rgba(11,17,30,0.94));
        box-shadow: var(--vct-shadow);
    }

    .vct-hero::after {
        content: "";
        position: absolute;
        right: -15%;
        top: 0;
        width: 42%;
        height: 100%;
        transform: skewX(-18deg);
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06));
        pointer-events: none;
    }

    .vct-header {
        position: relative;
        z-index: 1;
        color: var(--vct-text);
        font-size: clamp(1.85rem, 2.7vw, 3rem);
        font-weight: 850;
        line-height: 1.05;
        margin: 0;
        letter-spacing: 0;
    }

    .vct-header-red {
        color: var(--vct-red);
    }

    .vct-subtitle {
        position: relative;
        z-index: 1;
        max-width: 760px;
        color: var(--vct-text-secondary);
        font-size: 0.98rem;
        line-height: 1.65;
        margin-top: 0.7rem;
    }

    .vct-accent-line {
        width: 74px;
        height: 3px;
        margin-top: 1rem;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--vct-red), var(--vct-cyan));
    }

    .glass-card,
    .pred-card,
    [data-testid="stMetric"],
    div[data-testid="stExpander"],
    .stDataFrame {
        border: 1px solid var(--vct-border) !important;
        border-radius: var(--vct-radius) !important;
        background: var(--vct-surface) !important;
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.22);
    }

    .glass-card {
        padding: 1rem;
    }

    .pred-card {
        padding: 1.15rem;
        margin: 0.8rem 0 1rem;
    }

    .pred-card .card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }

    .pred-card .card-header span.label {
        color: var(--vct-text-secondary);
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .pred-card .card-header span.badge {
        color: #07111f;
        background: var(--vct-red);
        border-radius: 6px;
        padding: 0.22rem 0.6rem;
        font-size: 0.72rem;
        font-weight: 800;
    }

    .matchup-card {
        padding: 1.2rem;
        border: 1px solid var(--vct-border-strong);
        border-radius: var(--vct-radius);
        background:
            linear-gradient(90deg, rgba(56,189,248,0.12), transparent 45%, rgba(255,70,85,0.12)),
            var(--vct-surface);
        box-shadow: var(--vct-shadow);
    }

    .matchup-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1.1rem;
    }

    .matchup-label {
        color: var(--vct-text-secondary);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .matchup-mode {
        border-radius: 6px;
        padding: 0.25rem 0.65rem;
        color: #08111f;
        font-size: 0.76rem;
        font-weight: 850;
    }

    .matchup-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
        align-items: center;
        gap: 1rem;
    }

    .match-team {
        min-width: 0;
    }

    .match-team.right {
        text-align: right;
    }

    .team-name {
        overflow-wrap: anywhere;
        color: var(--vct-text);
        font-size: 1rem;
        font-weight: 800;
        line-height: 1.25;
    }

    .prob-value {
        margin-top: 0.25rem;
        font-size: clamp(2.25rem, 4vw, 3.8rem);
        font-weight: 900;
        line-height: 0.95;
        letter-spacing: 0;
    }

    .prob-track {
        height: 10px;
        margin-top: 0.85rem;
        overflow: hidden;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.16);
    }

    .prob-fill {
        height: 100%;
        border-radius: inherit;
    }

    .versus {
        display: grid;
        place-items: center;
        width: 54px;
        height: 54px;
        color: var(--vct-text);
        border: 1px solid var(--vct-border);
        border-radius: 50%;
        background: rgba(255,255,255,0.05);
        font-weight: 900;
    }

    [data-testid="stMetric"] {
        position: relative;
        overflow: hidden;
        padding: 0.9rem 1rem;
    }

    [data-testid="stMetric"]::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 3px;
        background: linear-gradient(180deg, var(--vct-red), var(--vct-cyan));
    }

    [data-testid="stMetric"] label,
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: var(--vct-text-secondary) !important;
        font-size: 0.78rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--vct-text) !important;
        font-size: 1.72rem !important;
        font-weight: 850 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        padding: 0.35rem;
        border: 1px solid var(--vct-border);
        border-radius: var(--vct-radius);
        background: rgba(15, 23, 42, 0.78);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: var(--vct-text-secondary);
        font-size: 0.9rem;
        font-weight: 750;
    }

    .stTabs [aria-selected="true"] {
        color: var(--vct-text) !important;
        background: rgba(255, 70, 85, 0.16);
    }

    div.stButton > button {
        width: 100%;
        min-height: 2.85rem;
        border: 1px solid rgba(255, 70, 85, 0.55) !important;
        border-radius: var(--vct-radius) !important;
        background: linear-gradient(135deg, #ff4655, #d72f3e) !important;
        color: #fff !important;
        font-size: 0.98rem !important;
        font-weight: 850 !important;
        box-shadow: 0 12px 30px rgba(255, 70, 85, 0.28);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 34px rgba(255, 70, 85, 0.34) !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border: 1px solid var(--vct-border) !important;
        border-radius: var(--vct-radius) !important;
        background: rgba(8, 13, 24, 0.84) !important;
        color: var(--vct-text) !important;
    }

    div[data-baseweb="popover"] {
        color: var(--vct-text) !important;
    }

    .stDataFrame {
        overflow: hidden;
    }

    .stDataFrame div {
        color: var(--vct-text) !important;
    }

    .stAlert {
        border-radius: var(--vct-radius) !important;
        border: 1px solid var(--vct-border) !important;
        background: rgba(15, 23, 42, 0.92) !important;
        color: var(--vct-text) !important;
    }

    hr {
        border-color: var(--vct-border) !important;
        margin: 1.35rem 0 !important;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--vct-border);
        background:
            linear-gradient(180deg, rgba(255,70,85,0.12), transparent 32%),
            #070b13;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 1.4rem;
    }

    .sidebar-brand {
        padding: 1rem 0.85rem 1.1rem;
        text-align: center;
        border: 1px solid var(--vct-border);
        border-radius: var(--vct-radius);
        background: rgba(15, 23, 42, 0.72);
    }

    .sidebar-mark {
        display: grid;
        place-items: center;
        width: 44px;
        height: 44px;
        margin: 0 auto 0.5rem;
        color: #fff;
        border-radius: var(--vct-radius);
        background: linear-gradient(135deg, var(--vct-red), #a51220);
        font-size: 1.35rem;
        font-weight: 900;
    }

    .sidebar-title {
        margin: 0;
        color: var(--vct-text);
        font-size: 1.08rem;
        font-weight: 900;
        letter-spacing: 0;
    }

    .sidebar-subtitle,
    .sidebar-footer {
        color: var(--vct-text-muted);
        font-size: 0.72rem;
        line-height: 1.6;
    }

    .sidebar-subtitle {
        margin: 0.25rem 0 0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .sidebar-footer {
        margin-top: 1rem;
        text-align: center;
    }

    div[data-testid="stSidebarNav"] {
        display: none !important;
    }

    section[data-testid="stSidebar"] a {
        border-radius: var(--vct-radius);
        color: var(--vct-text-secondary) !important;
        font-weight: 750;
    }

    section[data-testid="stSidebar"] a:hover {
        color: var(--vct-text) !important;
        background: rgba(255, 70, 85, 0.13);
    }

    .bp-map,
    .veto-chip,
    .final-map-card {
        border-radius: var(--vct-radius);
    }

    .veto-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.42rem 0.7rem;
        margin: 0.22rem 0.35rem 0.22rem 0;
        border: 1px solid var(--chip-border);
        background: var(--chip-bg);
        color: var(--vct-text);
        font-size: 0.84rem;
        font-weight: 700;
    }

    .veto-step {
        display: inline-grid;
        place-items: center;
        width: 22px;
        height: 22px;
        color: #07111f;
        border-radius: 50%;
        background: var(--chip-color);
        font-size: 0.72rem;
        font-weight: 900;
    }

    .final-map-card {
        padding: 1rem;
        text-align: center;
        border: 1px solid var(--card-border);
        background: var(--card-bg);
    }

    .final-map-label {
        color: var(--vct-text-muted);
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .final-map-name {
        color: var(--card-color);
        font-size: 1.38rem;
        font-weight: 900;
        line-height: 1.2;
    }

    .final-map-note {
        color: var(--vct-text-secondary);
        font-size: 0.76rem;
    }

    @media (max-width: 760px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .vct-hero {
            padding: 1.1rem;
        }

        .matchup-grid {
            grid-template-columns: 1fr;
        }

        .match-team.right {
            text-align: left;
        }

        .versus {
            width: 44px;
            height: 44px;
            margin: 0.1rem 0;
        }
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title_html: str, subtitle: str) -> None:
    """Render the shared hero header for a page."""
    st.markdown(
        f"""
<section class="vct-hero">
    <div class="vct-header">{title_html}</div>
    <div class="vct-subtitle">{subtitle}</div>
    <div class="vct-accent-line"></div>
</section>
        """,
        unsafe_allow_html=True,
    )
