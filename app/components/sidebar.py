"""共享侧边栏 — 品牌展示 + 导航链接"""

import streamlit as st


def show_sidebar():
    """在所有页面显示统一的侧边栏"""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-mark">V</div>
            <h2 class="sidebar-title">VCT Predictor</h2>
            <p class="sidebar-subtitle">Valorant Champion Tour</p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.page_link("pages/1_match_predictor.py", label="Match Predictor", icon="🎮")
        st.page_link("pages/2_team_analysis.py", label="Team Dashboard", icon="📊")
        st.page_link("pages/3_masters_london.py", label="Masters London", icon="🏅")
        st.page_link("pages/4_player_analyzer.py", label="Player Analyzer", icon="👤")

        st.divider()

        st.markdown("""
        <div class="sidebar-footer">
            VCT 2021–2026 · v1.0<br>
            基于 XGBoost + Streamlit
        </div>
        """, unsafe_allow_html=True)
