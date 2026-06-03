"""VCT 预测分析平台 — 主入口"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit runs this file with data/app on sys.path; app.* imports need data.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.components.sidebar import show_sidebar
from app.components.theme import apply_global_styles, page_header


st.set_page_config(
    page_title="VCT Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_styles()
show_sidebar()

page_header(
    'VCT <span class="vct-header-red">比赛预测分析平台</span>',
    "基于 Kaggle VCT 2021–2026 数据集，对《无畏契约》全球冠军赛进行数据分析和胜负预测。",
)

st.subheader("功能导航")

nav_left, nav_right = st.columns(2)
with nav_left:
    st.page_link(
        "pages/1_match_predictor.py",
        label="🎮 Match Predictor",
        help="赛前/赛中胜负概率预测",
        use_container_width=True,
    )
    st.page_link(
        "pages/3_masters_london.py",
        label="🏅 Masters London",
        help="2026 大师赛专题页",
        use_container_width=True,
    )

with nav_right:
    st.page_link(
        "pages/2_team_analysis.py",
        label="📊 Team Dashboard",
        help="战队数据概览、趋势分析",
        use_container_width=True,
    )
    st.page_link(
        "pages/4_player_analyzer.py",
        label="👤 Player Analyzer",
        help="选手表现分析、数据对比",
        use_container_width=True,
    )

st.divider()

st.subheader("数据概览")


@st.cache_resource(show_spinner=False)
def load_home_data():
    features = pd.read_parquet("data/features/match_features.parquet")
    draft = pd.read_parquet("data/processed/draft_dataset.parquet")
    n_teams = pd.unique(pd.concat([features["Team A"], features["Team B"]])).shape[0]
    return len(features), n_teams, len(draft)


try:
    n_matches, n_teams, n_bp = load_home_data()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("比赛场次", f"{n_matches:,}")
    m2.metric("战队数量", f"{n_teams:,}")
    m3.metric("BP 记录", f"{n_bp:,}")
    m4.metric("覆盖年份", "2021–2026")
except Exception:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("比赛场次", "—")
    m2.metric("战队数量", "—")
    m3.metric("BP 记录", "—")
    m4.metric("覆盖年份", "2021–2026")
    st.info("数据尚未生成。请在终端执行 `python src/run_pipeline.py` 完成数据处理后刷新页面。")
