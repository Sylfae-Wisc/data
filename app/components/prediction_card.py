"""可复用组件：预测结果展示卡片"""

import streamlit as st
import plotly.graph_objects as go
from typing import Optional


def show_win_prob(result: dict, team1: str, team2: str):
    """展示胜负概率预测结果

    Args:
        result: predict_match() 返回的 dict
        team1: 队伍1简称
        team2: 队伍2简称
    """
    p1 = result["team1_win_prob"]
    p2 = result["team2_win_prob"]
    mode = result.get("mode", "pre_match")

    # 模式标签
    mode_label = "赛中预测" if mode == "in_match" else "赛前预测"
    mode_color = "#4CAF50" if mode == "in_match" else "#FF9800"

    st.markdown(f"""
    <div class="pred-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <span style="color: #888; font-size: 0.85rem;">胜负概率预测</span>
            <span style="background: {mode_color}; color: #fff; border-radius: 4px; padding: 0.15rem 0.6rem; font-size: 0.75rem;">{mode_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 概率横向对比
    c1, c2, c3 = st.columns([2, 1, 2])

    with c1:
        st.markdown(f'<div class="team-name" style="color: #4FC3F7;">{team1}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="prob-value" style="color: #4FC3F7;">{p1:.1%}</div>', unsafe_allow_html=True)
        st.progress(p1)

    with c2:
        st.markdown('<div style="text-align: center; padding-top: 1.5rem; color: #888;">VS</div>', unsafe_allow_html=True)

    with c3:
        st.markdown(f'<div class="team-name" style="text-align: right; color: #FF8A80;">{team2}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="prob-value" style="text-align: right; color: #FF8A80;">{p2:.1%}</div>', unsafe_allow_html=True)
        st.progress(p2)


def show_bp(result: dict, team1: str, team2: str):
    """展示 BP（地图 Ban/Pick）预测结果

    Args:
        result: predict_bp() 返回的 dict
    """
    st.markdown("""
    <div class="pred-card">
        <span style="color: #888; font-size: 0.85rem;">地图 Ban / Pick 预测</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f'<span style="color: #4FC3F7;">{team1}</span>', unsafe_allow_html=True)
        st.markdown("**Bans**")
        for item in result.get("team1_bans", []):
            st.markdown(f'<span class="bp-map">{item["map"]} ({item["prob"]:.1%})</span>', unsafe_allow_html=True,)
        st.markdown("**Picks**")
        for item in result.get("team1_picks", []):
            st.markdown(f'<span class="bp-map">{item["map"]} ({item["prob"]:.1%})</span>', unsafe_allow_html=True,)

    with c2:
        st.markdown(f'<span style="color: #FF8A80;">{team2}</span>', unsafe_allow_html=True)
        st.markdown("**Bans**")
        for item in result.get("team2_bans", []):
            st.markdown(f'<span class="bp-map">{item["map"]} ({item["prob"]:.1%})</span>', unsafe_allow_html=True,)
        st.markdown("**Picks**")
        for item in result.get("team2_picks", []):
            st.markdown(f'<span class="bp-map">{item["map"]} ({item["prob"]:.1%})</span>', unsafe_allow_html=True,)

    st.caption("基于历史 BP 统计的条件概率")


def show_bo3(result: dict):
    """展示 BO3 比分预测

    Args:
        result: predict_bo3_score() 返回的 dict
    """
    st.markdown("""
    <div class="pred-card">
        <span style="color: #888; font-size: 0.85rem;">BO3 比分预测</span>
    </div>
    """, unsafe_allow_html=True)

    # 用 Plotly 横向柱状图展示
    outcomes = ["2-0", "2-1", "1-2", "0-2"]
    probs = [result.get(o, 0) for o in outcomes]
    colors = ["#4FC3F7", "#81D4FA", "#FF8A80", "#EF5350"]

    fig = go.Figure(go.Bar(
        x=probs,
        y=outcomes,
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1%}" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(range=[0, max(probs) * 1.3], showgrid=False, visible=False),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("链式乘法：每图独立同分布，P(2-0) = p²")


def show_match_summary(result: dict, team1: str, team2: str):
    """组合展示完整的比赛预测结果

    Args:
        result: predict_match() 的完整返回
        team1: 队伍1简称
        team2: 队伍2简称
    """
    tab1, tab2, tab3 = st.tabs(["胜负概率", "BP 预测", "BO3 比分"])

    with tab1:
        show_win_prob(result, team1, team2)

    with tab2:
        if "team1_bans" in result:
            show_bp(result, team1, team2)
        else:
            st.info("BP 数据仅部分页面可用")

    with tab3:
        if "2-0" in result:
            show_bo3(result)
        else:
            st.info("BO3 比分仅部分页面可用")
