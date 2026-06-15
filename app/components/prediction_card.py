"""可复用组件：预测结果展示卡片（VCT 主题）"""

from html import escape
import streamlit as st
import plotly.graph_objects as go


def show_win_prob(result: dict, team1: str, team2: str):
    """展示胜负概率预测结果"""
    p1 = result["team1_win_prob"]
    p2 = result["team2_win_prob"]
    mode = result.get("mode", "pre_match")

    if mode == "in_match":
        mode_label = "赛中预测"
    elif result.get("feature_mode") == "team_strength_blend":
        mode_label = f"赛前预测"
    else:
        mode_label = "赛前预测"
    mode_color = "#86EFAC" if mode == "in_match" else "#FBBF24"
    team1_safe = escape(team1)
    team2_safe = escape(team2)
    p1_width = max(0, min(p1, 1)) * 100
    p2_width = max(0, min(p2, 1)) * 100
    h2h_matches = result.get("h2h_matches", 0)
    h2h_ratio = result.get("h2h_ratio", 0.5)
    h2h_weight = result.get("h2h_weight", 0.0)
    if h2h_matches:
        h2h_note = (
            f"历史交手时间衰减参考：{team1_safe} {h2h_ratio:.1%} · "
            f"{h2h_matches} 场 · 校准权重 {h2h_weight:.0%}"
        )
    else:
        h2h_note = "暂无历史交手记录，本场预测不做 H2H 校准。"

    st.markdown(
        f"""
<div class="matchup-card">
    <div class="matchup-head">
        <span class="matchup-label">胜负概率预测</span>
        <span class="matchup-mode" style="background:{mode_color};">{mode_label}</span>
    </div>
    <div class="matchup-grid">
        <div class="match-team">
            <div class="team-name" style="color:var(--vct-blue);">{team1_safe}</div>
            <div class="prob-value" style="color:var(--vct-blue);">{p1:.1%}</div>
            <div class="prob-track">
                <div class="prob-fill" style="width:{p1_width:.1f}%; background:linear-gradient(90deg, var(--vct-blue), var(--vct-cyan));"></div>
            </div>
        </div>
        <div class="versus">VS</div>
        <div class="match-team right">
            <div class="team-name" style="color:var(--vct-red);">{team2_safe}</div>
            <div class="prob-value" style="color:var(--vct-red);">{p2:.1%}</div>
            <div class="prob-track">
                <div class="prob-fill" style="width:{p2_width:.1f}%; margin-left:auto; background:linear-gradient(90deg, #fb7185, var(--vct-red));"></div>
            </div>
        </div>
    </div>
    <div class="matchup-meta">{h2h_note}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def show_bp(result: dict, team1: str, team2: str):
    """展示 BP（地图 Ban/Pick）预测 — VCT 7 图 → 3 图禁选过程"""
    team1_safe = escape(team1)
    team2_safe = escape(team2)

    st.markdown("""
    <div class="pred-card">
        <div class="card-header">
            <span class="label">地图 Ban / Pick 预测</span>
            <span class="badge">禁选模拟</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    sequence = result.get("veto_sequence", [])
    if not sequence:
        st.info("暂无 BP 预测数据")
        return

    final = result.get("final_maps", [])
    if len(final) == 3:
        st.markdown("**最终地图**")
        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(
                f'<div class="final-map-card" style="--card-bg:linear-gradient(135deg,rgba(134,239,172,0.16),rgba(15,23,42,0.88)); --card-border:#86EFAC66; --card-color:#86EFAC;">'
                f'<div class="final-map-label">Map 1</div>'
                f'<div class="final-map-name">{escape(final[0])}</div>'
                f'<div class="final-map-note">{team1_safe} 选图</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="final-map-card" style="--card-bg:linear-gradient(135deg,rgba(34,211,238,0.16),rgba(15,23,42,0.88)); --card-border:#22D3EE66; --card-color:#22D3EE;">'
                f'<div class="final-map-label">Map 2</div>'
                f'<div class="final-map-name">{escape(final[1])}</div>'
                f'<div class="final-map-note">{team2_safe} 选图</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="final-map-card" style="--card-bg:linear-gradient(135deg,rgba(251,191,36,0.16),rgba(15,23,42,0.88)); --card-border:#FBBF2466; --card-color:#FBBF24;">'
                f'<div class="final-map-label">Map 3</div>'
                f'<div class="final-map-name">{escape(final[2])}</div>'
                f'<div class="final-map-note">决胜图</div></div>',
                unsafe_allow_html=True,
            )

        st.divider()

    # BP 过程
    st.markdown("**BP 过程（地图禁选顺序）**")
    st.caption("Ban 表示禁用地图，Pick 表示选择地图，决胜图是前面禁选结束后自动留下的最后一张地图。")
    for s in sequence:
        step = s["step"]
        if s["team"] != "—":
            actor = team1 if s["team"] == "A" else team2
        else:
            actor = "自动"

        action = s["action"]
        map_name = s["map"]

        if action == "ban":
            icon, action_label, color, bg = "🚫", "禁用", "#FB7185", "rgba(251,113,133,0.14)"
        elif action == "pick":
            icon, action_label, color, bg = "✅", "选择", "#86EFAC", "rgba(134,239,172,0.12)"
        else:
            icon, action_label, color, bg = "🎲", "决胜", "#FBBF24", "rgba(251,191,36,0.12)"

        st.markdown(
            f'<span class="veto-chip" style="--chip-bg:{bg}; --chip-border:{color}55; --chip-color:{color};">'
            f'<span class="veto-step">{step}</span>'
            f'{icon} {escape(actor)} <span style="color:{color};font-weight:850;">{action_label}</span> → '
            f'<b>{escape(map_name)}</b></span>',
            unsafe_allow_html=True,
        )

    st.caption(
        f"基于两队历史 BP 数据的贪心博弈模拟 · "
        f"置信度: {team1}={result.get('team_a_pick_prob', 0):.1%} / "
        f"{team2}={result.get('team_b_pick_prob', 0):.1%}"
    )


def show_bo3(result: dict):
    """展示 BO3 比分预测"""
    st.markdown("""
    <div class="pred-card">
        <div class="card-header">
            <span class="label">BO3 比分预测</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    outcomes = ["2-0", "2-1", "1-2", "0-2"]
    probs = [result.get(o, 0) for o in outcomes]
    colors = ["#38BDF8", "#86EFAC", "#FBBF24", "#FF4655"]
    max_prob = max(max(probs), 0.01)

    fig = go.Figure(go.Bar(
        x=probs,
        y=outcomes,
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1%}" for p in probs],
        textposition="outside",
        textfont=dict(size=13, color="#F8FAFC"),
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(range=[0, max_prob * 1.35], showgrid=False, visible=False),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#AAB7CC"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("链式乘法：每图独立同分布，P(2-0) = p²")


def show_match_summary(result: dict, team1: str, team2: str):
    """组合展示完整的比赛预测结果"""
    tab1, tab2, tab3 = st.tabs(["📊 胜负概率", "🗺️ BP 预测", "🏆 BO3 比分"])

    with tab1:
        show_win_prob(result, team1, team2)

    with tab2:
        if "veto_sequence" in result:
            show_bp(result, team1, team2)
        else:
            st.info("BP 数据仅部分页面可用")

    with tab3:
        if "2-0" in result:
            show_bo3(result)
        else:
            st.info("BO3 比分仅部分页面可用")
