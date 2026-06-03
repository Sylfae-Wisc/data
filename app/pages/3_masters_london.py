"""Masters London — 2026 大师赛专题"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from app.components import FEATURES_DIR, PROCESSED_DIR
from app.components.sidebar import show_sidebar
from app.components.theme import apply_global_styles, page_header
from src.predictor import predict_match, predict_bo3_score

st.set_page_config(page_title="Masters London", page_icon="🏅", layout="wide")

apply_global_styles()
show_sidebar()

page_header("🏅 Masters London 2026", "VCT 2026 大师赛专题页 — 赛程回顾、数据分析、胜负复盘。")

# ==== 数据加载 ====
MASTERS_NAME = "Valorant Masters Santiago 2026"  # 最新 Masters 数据

@st.cache_resource
def load_match_data():
    return pd.read_parquet(FEATURES_DIR / "match_features.parquet")

@st.cache_resource
def load_player_stats():
    return pd.read_csv("data/raw/vct_2026/players_stats/players_stats.csv")

features = load_match_data()
masters = features[features["Tournament"] == MASTERS_NAME].copy()

if len(masters) == 0:
    st.warning(f"暂无 {MASTERS_NAME} 数据。")
    st.stop()

ps = load_player_stats()
ps_masters = ps[ps["Tournament"] == MASTERS_NAME]

# ==== 计算 ====
teams_in_masters = sorted(
    set(masters["Team A"]).union(set(masters["Team B"]))
)
stages = masters["Stage"].unique()

# 按阶段分组
stage_order = ["Swiss Stage", "Playoffs"]
masters["stage_order"] = masters["Stage"].map(
    {s: i for i, s in enumerate(stage_order)}
).fillna(99)
masters_sorted = masters.sort_values(["stage_order", "tord"])

# 队伍战绩
team_records = {}
for team in teams_in_masters:
    as_a = masters[masters["Team A"] == team]
    as_b = masters[masters["Team B"] == team]
    wins = (as_a["Team A Score"] > as_a["Team B Score"]).sum() + \
           (as_b["Team B Score"] > as_b["Team A Score"]).sum()
    losses = (as_a["Team A Score"] < as_a["Team B Score"]).sum() + \
             (as_b["Team B Score"] < as_b["Team A Score"]).sum()
    team_records[team] = {"W": wins, "L": losses}

# 选手 MVP 排行
if len(ps_masters) > 0:
    ps_agg = ps_masters.groupby("Player").agg(
        Rating=("Rating", "mean"),
        ACS=("Average Combat Score", "mean"),
        KAST=("Kill, Assist, Trade, Survive %", lambda x: x.str.rstrip("%").astype(float).mean() / 100),
        ADR=("Average Damage Per Round", "mean"),
        KPR=("Kills Per Round", "mean"),
        HS=("Headshot %", lambda x: x.str.rstrip("%").astype(float).mean() / 100),
        Teams=("Teams", "first"),
        Matches=("Tournament", "count"),
    ).reset_index()
    ps_agg["Rating"] = ps_agg["Rating"].fillna(0)
    ps_agg = ps_agg.sort_values("Rating", ascending=False)

# 地图数据
maps_played = pd.read_csv("data/raw/vct_2026/matches/maps_played.csv")
maps_played = maps_played[maps_played["Tournament"] == MASTERS_NAME]
map_counts = maps_played["Map"].value_counts()

# Agent 数据
agents = pd.read_csv("data/raw/vct_2026/agents/agents_pick_rates.csv")
agents = agents[agents["Tournament"] == MASTERS_NAME]

# ==== 渲染 ====
c1, c2, c3 = st.columns(3)
c1.metric("参赛战队", f"{len(teams_in_masters)}")
c2.metric("比赛场次", f"{len(masters)}")
c3.metric("比赛阶段", f"{', '.join(stages)}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["比赛列表", "战队战绩", "选手排行", "地图/英雄"])

with tab1:
    for _, match in masters_sorted.iterrows():
        t1, t2 = match["Team A"], match["Team B"]
        s1, s2 = match["Team A Score"], match["Team B Score"]
        stage = match["Stage"]
        match_name = match["Match Type"]

        winner = t1 if s1 > s2 else t2 if s2 > s1 else "平局"
        winner_color = "#38BDF8" if winner == t1 else "#FB7185"

        with st.container():
            cols = st.columns([1, 3, 1, 1])
            cols[0].markdown(f"**{stage}**")
            cols[1].markdown(
                f"<span style='color:{winner_color}'>{match_name}</span>",
                unsafe_allow_html=True,
            )
            score_str = f"{int(s1)} - {int(s2)}"
            cols[2].markdown(f"**{score_str}**")
            # 预测该场比赛的复盘
            with cols[3]:
                st.markdown(f"🏆 {winner}")

        st.divider()

with tab2:
    # 战绩表
    records_df = pd.DataFrame([
        {"战队": t, "胜": r["W"], "负": r["L"],
         "胜率": r["W"] / (r["W"] + r["L"]) if (r["W"] + r["L"]) > 0 else 0}
        for t, r in team_records.items()
    ]).sort_values("胜", ascending=False)

    fig = go.Figure(go.Bar(
        x=records_df["战队"],
        y=records_df["胜"],
        name="胜",
        marker_color="#38BDF8",
        text=records_df["胜"],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=records_df["战队"],
        y=records_df["负"],
        name="负",
        marker_color="#FF4655",
        text=records_df["负"],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#AAB7CC",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width='stretch')

    st.dataframe(
        records_df.style.format({"胜率": "{:.0%}"}),
        width='stretch',
        hide_index=True,
    )

with tab3:
    if len(ps_agg) > 0:
        top_n = st.slider("显示前 N 名", 5, 30, 10, key="mvp_n")
        top_players = ps_agg.head(top_n)

        fig = go.Figure(go.Bar(
            x=top_players["Rating"],
            y=top_players["Player"],
            orientation="h",
            marker_color="#FBBF24",
            text=top_players["Rating"].round(2),
            textposition="outside",
        ))
        fig.update_layout(
            height=50 * min(top_n, 30) + 80,
            xaxis=dict(range=[0, top_players["Rating"].max() * 1.2]),
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#AAB7CC",
            margin=dict(l=10, r=60, t=10, b=10),
        )
        st.plotly_chart(fig, width='stretch')

        # 详细数据表
        display = top_players.copy()
        display["KAST"] = display["KAST"].apply(lambda x: f"{x:.1%}")
        display["HS"] = display["HS"].apply(lambda x: f"{x:.1%}")
        display["ACS"] = display["ACS"].round(0).astype(int)
        display["ADR"] = display["ADR"].round(0).astype(int)
        display["KPR"] = display["KPR"].round(2)
        st.dataframe(
            display[["Player", "Teams", "Rating", "ACS", "KAST", "ADR", "KPR", "HS"]],
            width='stretch',
            hide_index=True,
            column_config={
                "Player": "选手",
                "Teams": "战队",
                "Rating": st.column_config.NumberColumn("Rating", format="%.2f"),
                "ACS": "ACS",
                "KAST": "KAST",
                "ADR": "ADR",
                "KPR": "KPR",
                "HS": "HS%",
            },
        )
    else:
        st.info("暂无选手数据。")

with tab4:
    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown("**地图出场次数**")
        if len(map_counts) > 0:
            fig = go.Figure(go.Bar(
                x=map_counts.values,
                y=map_counts.index,
                orientation="h",
                marker_color="#86EFAC",
                text=map_counts.values,
                textposition="outside",
            ))
            fig.update_layout(
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#AAB7CC",
                margin=dict(l=10, r=40, t=10, b=10),
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("无地图数据。")

    with mc2:
        st.markdown("**英雄选取率 Top 10**")
        if len(agents) > 0:
            agent_agg = agents.groupby("Agent")["Pick Rate"].apply(
                lambda x: x.str.rstrip("%").astype(float).mean()
            ).sort_values(ascending=False).head(10)
            fig = go.Figure(go.Bar(
                x=agent_agg.values,
                y=agent_agg.index,
                orientation="h",
                marker_color="#C084FC",
                text=[f"{v:.0f}%" for v in agent_agg.values],
                textposition="outside",
            ))
            fig.update_layout(
                height=300,
                xaxis=dict(range=[0, 100]),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#AAB7CC",
                margin=dict(l=10, r=40, t=10, b=10),
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("无英雄数据。")
