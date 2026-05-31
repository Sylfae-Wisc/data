"""VCT 预测分析平台 — 主入口"""

import streamlit as st

# ==== Page Config — 必须在最前面 ====
st.set_page_config(
    page_title="VCT Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==== 共享 CSS ====
st.markdown("""
<style>
    /* VCT 主题色 */
    :root {
        --vct-red: #FF4655;
        --vct-dark: #0F1923;
        --vct-light: #ECE8E1;
        --vct-bg: #1a1a2e;
    }

    /* 首页卡片 */
    .nav-card {
        background: #16213e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s;
    }
    .nav-card:hover {
        border-color: var(--vct-red);
    }
    .nav-card h3 {
        margin: 0 0 0.5rem;
        color: var(--vct-light);
    }
    .nav-card p {
        margin: 0;
        color: #888;
        font-size: 0.9rem;
    }

    /* 预测卡片 */
    .pred-card {
        background: #16213e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.5rem;
    }
    .prob-label {
        font-size: 0.85rem;
        color: #888;
    }
    .prob-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .team-name {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .bp-map {
        display: inline-block;
        background: #0f3460;
        color: #e0e0e0;
        border-radius: 6px;
        padding: 0.25rem 0.75rem;
        margin: 0.2rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ==== Sidebar ====
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <span style="font-size: 2rem;">🏆</span>
        <h2 style="margin: 0; color: #FF4655;">VCT Predictor</h2>
        <p style="color: #888; font-size: 0.8rem; margin: 0;">Valorant Champion Tour</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.page_link("pages/1_match_predictor.py", label=" Match Predictor", icon="🎮")
    st.page_link("pages/2_team_analysis.py", label=" Team Dashboard", icon="📊")
    st.page_link("pages/3_masters_london.py", label=" Masters London", icon="🏅")
    st.page_link("pages/4_player_analyzer.py", label=" Player Analyzer", icon="👤")

    st.divider()
    st.caption("VCT 2021–2026 · Phase 1")


# ==== 首页内容 ====
col_title, _ = st.columns([3, 1])
with col_title:
    st.title("VCT 比赛预测分析平台")
    st.markdown(
        "基于 Kaggle VCT 2021–2026 数据集，对《无畏契约》全球冠军赛进行数据分析和胜负预测。"
    )

st.divider()

# 四个功能入口卡片
st.subheader("功能导航")

c1, c2 = st.columns(2)

with c1:
    st.page_link("pages/1_match_predictor.py", label="🎮 Match Predictor", help="赛前/赛中胜负概率预测", use_container_width=True)
    st.page_link("pages/3_masters_london.py", label="🏅 Masters London", help="2026 大师赛专题页", use_container_width=True)

with c2:
    st.page_link("pages/2_team_analysis.py", label="📊 Team Dashboard", help="战队数据概览、趋势分析", use_container_width=True)
    st.page_link("pages/4_player_analyzer.py", label="👤 Player Analyzer", help="选手表现分析、数据对比", use_container_width=True)

st.divider()

# 底部数据概览
st.subheader("数据概览")
m1, m2, m3, m4 = st.columns(4)

try:
    import pandas as pd
    features = pd.read_parquet("data/features/match_features.parquet")
    draft = pd.read_parquet("data/processed/draft_dataset.parquet")

    m1.metric("比赛场次", f"{len(features):,}")
    m2.metric("战队数量", f"{pd.unique(pd.concat([features['Team A'], features['Team B']])).shape[0]:,}")
    m3.metric("BP 记录", f"{len(draft):,}")
    m4.metric("覆盖年份", "2021–2026")
except Exception:
    m1.metric("比赛场次", "—")
    m2.metric("战队数量", "—")
    m3.metric("BP 记录", "—")
    m4.metric("覆盖年份", "2021–2026")
