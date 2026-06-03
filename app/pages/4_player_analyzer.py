"""Player Analyzer — 选手分析"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from app.components.sidebar import show_sidebar
from app.components.theme import apply_global_styles, page_header

st.set_page_config(page_title="Player Analyzer", page_icon="👤", layout="wide")

apply_global_styles()
show_sidebar()

page_header("👤 Player Analyzer", "选手表现分析、数据对比、状态追踪。")

# ==== 数据加载 ====
@st.cache_resource
def load_all_players():
    """合并所有年份的选手数据"""
    dfs = []
    for year in range(2021, 2027):
        df = pd.read_csv(f"data/raw/vct_{year}/players_stats/players_stats.csv")
        df["Year"] = year
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

@st.cache_resource
def get_player_list(ps):
    """获取所有选手名称列表（按年份降序，最近活跃的排前面）"""
    # 按最近活跃年份降序排列：2026 选手 → 2025 → ... → 2021
    player_last_year = ps.groupby("Player")["Year"].max().sort_values(ascending=False)
    return player_last_year.index.tolist()

@st.cache_resource
def load_overview_data():
    """加载 overview 数据用于 map 级选手统计"""
    dfs = []
    for year in range(2021, 2027):
        try:
            df = pd.read_csv(f"data/raw/vct_{year}/matches/overview.csv")
            df["Year"] = year
            dfs.append(df)
        except FileNotFoundError:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

ps = load_all_players()
player_list = get_player_list(ps)

# ==== 计算 ====
selected_player = st.selectbox(
    "选择选手", options=player_list,
    placeholder="搜索选手名称…",
    index=None,
    key="player_select",
)

if not selected_player:
    st.info("请先选择一名选手查看分析。")
    st.stop()

# 筛选该选手的所有记录
player_data = ps[ps["Player"] == selected_player].copy()
player_data = player_data.sort_values(["Year", "Tournament"])

# 基础字段转换
player_data["KAST"] = player_data["Kill, Assist, Trade, Survive %"].str.rstrip("%").astype(float) / 100
player_data["HS%"] = player_data["Headshot %"].str.rstrip("%").astype(float) / 100
player_data["ACS"] = player_data["Average Combat Score"]
player_data["ADR"] = player_data["Average Damage Per Round"]
player_data["KPR"] = player_data["Kills Per Round"]
player_data["APR"] = player_data["Assists Per Round"]
player_data["FKPR"] = player_data["First Kills Per Round"]
player_data["FDPR"] = player_data["First Deaths Per Round"]
player_data["KD"] = player_data["Kills:Deaths"]

# 战队历史
teams_played = player_data.groupby("Teams").agg(
    Tournaments=("Tournament", "nunique"),
    Matches=("Year", "count"),
).sort_values("Matches", ascending=False)

# 按赛事聚合
tournament_agg = player_data.groupby(["Year", "Tournament", "Teams"]).agg(
    Rating=("Rating", "mean"),
    ACS=("ACS", "mean"),
    KAST=("KAST", "mean"),
    ADR=("ADR", "mean"),
    KPR=("KPR", "mean"),
    HS=("HS%", "mean"),
    KD=("KD", "mean"),
    Rounds=("Rounds Played", "sum"),
).reset_index()
tournament_agg["Rating"] = tournament_agg["Rating"].fillna(0)
tournament_agg = tournament_agg.rename(columns={"HS": "HS%"})

# Agent 使用
agent_data = player_data.groupby("Agents").agg(
    Rounds=("Rounds Played", "sum"),
    Rating=("Rating", "mean"),
).sort_values("Rounds", ascending=False)

# ==== 渲染 ====
c1, c2, c3, c4 = st.columns(4)
c1.metric("效力战队数", f"{teams_played.shape[0]}")
c2.metric("参赛赛事数", f"{player_data['Tournament'].nunique()}")
c3.metric("覆盖年份", f"{int(player_data['Year'].min())}–{int(player_data['Year'].max())}")
c4.metric("总回合数", f"{player_data['Rounds Played'].sum():,}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["赛事表现", "数据总览", "英雄池", "选手对比"])

with tab1:
    if len(tournament_agg) > 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tournament_agg["Tournament"],
            y=tournament_agg["Rating"],
            mode="lines+markers",
            name="Rating",
            line=dict(color="#FBBF24", width=2),
            marker=dict(size=8),
            text=tournament_agg["Teams"],
        ))
        fig.update_layout(
            height=350,
            xaxis_tickangle=-45,
            yaxis=dict(range=[0, max(tournament_agg["Rating"].max() * 1.2, 1.5)]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#AAB7CC",
            margin=dict(b=120),
        )
        st.plotly_chart(fig, width='stretch')
        st.caption("每届赛事的平均 Rating，括号内为当时所属战队。")

        # 详细赛事数据表
        display = tournament_agg.copy()
        display["KAST"] = display["KAST"].apply(lambda x: f"{x:.1%}")
        display["HS%"] = display["HS%"].apply(lambda x: f"{x:.1%}")
        st.dataframe(
            display[["Year", "Tournament", "Teams", "Rating", "ACS", "KAST", "ADR", "KPR", "HS%", "KD", "Rounds"]],
            width='stretch',
            hide_index=True,
            column_config={
                "Year": "年份",
                "Tournament": "赛事",
                "Teams": "战队",
                "Rating": st.column_config.NumberColumn("Rating", format="%.2f"),
                "ACS": st.column_config.NumberColumn("ACS", format="%.0f"),
                "KAST": "KAST",
                "ADR": st.column_config.NumberColumn("ADR", format="%.0f"),
                "KPR": st.column_config.NumberColumn("KPR", format="%.2f"),
                "HS%": "HS%",
                "KD": st.column_config.NumberColumn("K/D", format="%.2f"),
                "Rounds": "回合数",
            },
        )
    else:
        st.info("无赛事数据。")

with tab2:
    # 核心指标雷达图
    latest = tournament_agg.sort_values("Year", ascending=False).iloc[0] if len(tournament_agg) > 0 else None
    career_avg = player_data[["Rating", "ACS", "KAST", "ADR", "KPR", "HS%"]].mean()

    if latest is not None:
        radar_cats = ["Rating", "ACS", "KAST", "ADR", "KPR", "HS%"]
        max_vals = {"Rating": 1.5, "ACS": 350, "KAST": 1.0, "ADR": 250, "KPR": 1.5, "HS%": 0.5}
        career_norm = [min(career_avg[c] / max_vals[c], 1.0) for c in radar_cats]
        latest_norm = [min(latest[c] / max_vals[c], 1.0) for c in radar_cats]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=career_norm, theta=radar_cats,
            fill="toself", name="生涯均值",
            line_color="#38BDF8", opacity=0.6,
        ))
        fig.add_trace(go.Scatterpolar(
            r=latest_norm, theta=radar_cats,
            fill="toself", name="最近赛事",
            line_color="#FBBF24", opacity=0.6,
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 1], showticklabels=False),
                bgcolor="rgba(0,0,0,0)",
            ),
            height=400,
            paper_bgcolor="rgba(0,0,0,0)", font_color="#AAB7CC",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, width='stretch')

    # 生涯统计卡片
    st.subheader("生涯数据")
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("生涯 Rating", f"{career_avg['Rating']:.2f}")
    rc1.metric("ACS", f"{career_avg['ACS']:.0f}")
    rc2.metric("KAST", f"{career_avg['KAST']:.1%}")
    rc2.metric("ADR", f"{career_avg['ADR']:.0f}")
    rc3.metric("KPR", f"{career_avg['KPR']:.2f}")
    rc3.metric("HS%", f"{career_avg['HS%']:.1%}")
    rc4.metric("效力战队", f"{teams_played.shape[0]}")
    rc4.metric("总赛事", f"{player_data['Tournament'].nunique()}")

with tab3:
    mc1, mc2 = st.columns([3, 2])

    with mc1:
        st.markdown("**英雄选用回合数**")
        if len(agent_data) > 0:
            top_agents = agent_data.head(10)
            fig = go.Figure(go.Bar(
                x=top_agents["Rounds"].values,
                y=top_agents.index,
                orientation="h",
                marker_color="#C084FC",
                text=top_agents["Rounds"].values,
                textposition="outside",
            ))
            fig.update_layout(
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#AAB7CC",
                margin=dict(l=10, r=40, t=10, b=10),
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("无英雄数据。")

    with mc2:
        st.markdown("**英雄 Rating 排行**")
        if len(agent_data) > 0:
            top_rating = agent_data[agent_data["Rounds"] >= agent_data["Rounds"].quantile(0.3)]\
                .sort_values("Rating", ascending=False).head(8)
            fig = go.Figure(go.Bar(
                x=top_rating["Rating"].values,
                y=top_rating.index,
                orientation="h",
                marker_color="#86EFAC",
                text=top_rating["Rating"].round(2),
                textposition="outside",
            ))
            fig.update_layout(
                height=300,
                xaxis=dict(range=[0, top_rating["Rating"].max() * 1.2]),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#AAB7CC",
                margin=dict(l=10, r=40, t=10, b=10),
            )
            st.plotly_chart(fig, width='stretch')
            st.caption("已过滤回合数过少的英雄（P30 以下）。")

with tab4:
    st.markdown("**选择对比选手**")
    compare_player = st.selectbox(
        "对比选手", options=[p for p in player_list if p != selected_player],
        index=None, placeholder="搜索对比选手…",
        key="compare_select",
    )

    if compare_player:
        compare_data = ps[ps["Player"] == compare_player].copy()
        compare_data["KAST"] = compare_data["Kill, Assist, Trade, Survive %"].str.rstrip("%").astype(float) / 100
        compare_data["HS%"] = compare_data["Headshot %"].str.rstrip("%").astype(float) / 100
        compare_data["ACS"] = compare_data["Average Combat Score"]
        compare_data["ADR"] = compare_data["Average Damage Per Round"]
        compare_data["KPR"] = compare_data["Kills Per Round"]
        compare_data["KD"] = compare_data["Kills:Deaths"]

        # 对比表格
        p1_career = player_data[["Rating", "ACS", "KAST", "ADR", "KPR", "HS%", "Rounds Played"]].mean()
        p2_career = compare_data[["Rating", "ACS", "KAST", "ADR", "KPR", "HS%", "Rounds Played"]].mean()

        compare_df = pd.DataFrame({
            "指标": ["Rating", "ACS", "KAST", "ADR", "KPR", "HS%", "场均回合"],
            selected_player: [
                f"{p1_career['Rating']:.2f}", f"{p1_career['ACS']:.0f}",
                f"{p1_career['KAST']:.1%}", f"{p1_career['ADR']:.0f}",
                f"{p1_career['KPR']:.2f}", f"{p1_career['HS%']:.1%}",
                f"{p1_career['Rounds Played']:.0f}",
            ],
            compare_player: [
                f"{p2_career['Rating']:.2f}", f"{p2_career['ACS']:.0f}",
                f"{p2_career['KAST']:.1%}", f"{p2_career['ADR']:.0f}",
                f"{p2_career['KPR']:.2f}", f"{p2_career['HS%']:.1%}",
                f"{p2_career['Rounds Played']:.0f}",
            ],
        })
        st.dataframe(compare_df, width='stretch', hide_index=True)

        # 雷达对比
        radar_cats = ["Rating", "ACS", "KAST", "ADR", "KPR", "HS%"]
        max_vals = {"Rating": 1.5, "ACS": 350, "KAST": 1.0, "ADR": 250, "KPR": 1.5, "HS%": 0.5}
        p1_norm = [min(p1_career[c] / max_vals[c], 1.0) for c in radar_cats]
        p2_norm = [min(p2_career[c] / max_vals[c], 1.0) for c in radar_cats]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=p1_norm, theta=radar_cats, fill="toself",
            name=selected_player, line_color="#38BDF8", opacity=0.6,
        ))
        fig.add_trace(go.Scatterpolar(
            r=p2_norm, theta=radar_cats, fill="toself",
            name=compare_player, line_color="#FB7185", opacity=0.6,
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 1], showticklabels=False),
                bgcolor="rgba(0,0,0,0)",
            ),
            height=400,
            paper_bgcolor="rgba(0,0,0,0)", font_color="#AAB7CC",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, width='stretch')

    else:
        st.info("选择一名选手进行对比分析。")
