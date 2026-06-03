"""Team Dashboard — 战队数据分析"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from app.components.team_selector import select_team
from app.components import FEATURES_DIR, PROCESSED_DIR
from app.components.sidebar import show_sidebar
from app.components.theme import apply_global_styles, page_header

st.set_page_config(page_title="Team Dashboard", page_icon="📊", layout="wide")

apply_global_styles()
show_sidebar()

page_header("📊 Team Dashboard", "战队数据概览、趋势分析、地图池深度。")

# ==== 数据加载 ====
@st.cache_resource
def load_features():
    return pd.read_parquet(FEATURES_DIR / "match_features.parquet")

@st.cache_resource
def load_draft():
    return pd.read_parquet(PROCESSED_DIR / "draft_dataset.parquet")

features = load_features()
draft = load_draft()

# ==== 计算 ====
team = select_team("选择战队", key="team_analysis")

if not team:
    st.info("请先选择一支战队查看分析。")
    st.stop()

# 筛选该战队所有比赛记录（作为 Team A 或 Team B）
as_a = features[features["Team A"] == team].copy()
as_b = features[features["Team B"] == team].copy()

# 统一视角：从该战队视角看
def normalize_view(df, as_team_a: bool):
    """将 df 统一为战队视角"""
    if as_team_a:
        df = df.rename(columns={
            "Team A": "Team", "Team B": "Opponent",
            "Team A Score": "Team Score", "Team B Score": "Opponent Score",
        })
        df["Win"] = df["Team Score"] > df["Opponent Score"]
    else:
        df = df.rename(columns={
            "Team B": "Team", "Team A": "Opponent",
            "Team B Score": "Team Score", "Team A Score": "Opponent Score",
        })
        df["Win"] = df["Team Score"] > df["Opponent Score"]
    return df

a_view = normalize_view(as_a.copy(), True)
b_view = normalize_view(as_b.copy(), False)

all_matches = pd.concat([a_view, b_view], ignore_index=True)
all_matches = all_matches.sort_values(["Year", "tord"])

total_matches = len(all_matches)
total_wins = all_matches["Win"].sum()
win_rate = total_wins / total_matches if total_matches > 0 else 0

# 按年份聚合
yearly = all_matches.groupby("Year").agg(
    Matches=("Win", "count"),
    Wins=("Win", "sum"),
).reset_index()
yearly["Win Rate"] = yearly["Wins"] / yearly["Matches"]

# 按对手聚合 H2H
h2h = all_matches.groupby("Opponent").agg(
    Matches=("Win", "count"),
    Wins=("Win", "sum"),
).reset_index()
h2h["Win Rate"] = h2h["Wins"] / h2h["Matches"]
h2h = h2h.sort_values("Matches", ascending=False)

# Form 数据
team_forms = []
for _, row in all_matches.iterrows():
    team_forms.append({
        "Match": f"{row['Team']} vs {row['Opponent']}",
        "Year": row["Year"],
        "tord": row["tord"],
        "form_5": row.get("team_a_form_5" if row.get("Team") != "Team A"
                          else "team_a_form_5", None),
    })
# 更直接地获取 form
form_cols_a = ["team_a_form_5", "team_a_form_10", "team_a_form_20"]
form_cols_b = ["team_b_form_5", "team_b_form_10", "team_b_form_20"]
a_forms = as_a[["Tournament", "Year", "tord"] + form_cols_a].rename(
    columns=dict(zip(form_cols_a, ["form_5", "form_10", "form_20"])))
a_forms["Side"] = "A"
b_forms = as_b[["Tournament", "Year", "tord"] + form_cols_b].rename(
    columns=dict(zip(form_cols_b, ["form_5", "form_10", "form_20"])))
b_forms["Side"] = "B"
all_forms = pd.concat([a_forms, b_forms], ignore_index=True).sort_values(["Year", "tord"])

# 地图 BP 数据
team_draft = draft[draft["Team"] == team]
ban_counts = team_draft[team_draft["Action"] == "ban"]["Map"].value_counts()
pick_counts = team_draft[team_draft["Action"] == "pick"]["Map"].value_counts()

# 统计均值（该战队作为 Team A 时的均值）
team_a_stats = features[features["Team A"] == team][
    ["team_a_acs", "team_a_kast", "team_a_adr", "team_a_hs_pct",
     "team_a_rating", "team_a_kd"]
].mean()
team_b_stats = features[features["Team B"] == team][
    ["team_b_acs", "team_b_kast", "team_b_adr", "team_b_hs_pct",
     "team_b_rating", "team_b_kd"]
].mean()

# ==== 渲染 ====

# --- 顶栏 KPI ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("总场次", f"{total_matches}")
c2.metric("胜场", f"{total_wins}")
c3.metric("胜率", f"{win_rate:.1%}")
avg_score_a = all_matches["Team Score"].mean()
avg_score_b = all_matches["Opponent Score"].mean()
c4.metric("场均得分", f"{avg_score_a:.1f} : {avg_score_b:.1f}")

st.divider()

# --- 胜率趋势 ---
st.subheader("📈 胜率趋势")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "年度趋势", "近期 Form", "地图池", "对战记录", "数据雷达",
])

with tab1:
    fig = px.bar(
        yearly, x="Year", y="Win Rate",
        text_auto=".0%",
        color_discrete_sequence=["#38BDF8"],
        labels={"Win Rate": "胜率"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=300,
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#AAB7CC",
    )
    st.plotly_chart(fig, width='stretch')

with tab2:
    if len(all_forms) > 0:
        fig = go.Figure()
        for col, name, color in [
            ("form_5", "近 5 场", "#38BDF8"),
            ("form_10", "近 10 场", "#FBBF24"),
            ("form_20", "近 20 场", "#86EFAC"),
        ]:
            valid = all_forms[col].dropna()
            fig.add_trace(go.Scatter(
                x=list(range(len(valid))),
                y=valid.values,
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=4),
            ))
        fig.update_layout(
            height=350,
            yaxis=dict(range=[0, 1], tickformat=".0%"),
            xaxis_title="比赛场次（按时间）",
            yaxis_title="胜率",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#AAB7CC",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1,
            ),
        )
        st.plotly_chart(fig, width='stretch')
        st.caption("滚动窗口胜率：近 N 场比赛的移动平均胜率。")
    else:
        st.info("无可用 form 数据。")

with tab3:
    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown("**Ban 地图分布**")
        if len(ban_counts) > 0:
            fig = go.Figure(go.Bar(
                x=ban_counts.values,
                y=ban_counts.index,
                orientation="h",
                marker_color="#FF4655",
                text=ban_counts.values,
                textposition="outside",
            ))
            fig.update_layout(
                height=300,
                xaxis_title="次数",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#AAB7CC",
                margin=dict(l=10, r=40, t=10, b=10),
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("无 ban 记录。")

    with mc2:
        st.markdown("**Pick 地图分布**")
        if len(pick_counts) > 0:
            fig = go.Figure(go.Bar(
                x=pick_counts.values,
                y=pick_counts.index,
                orientation="h",
                marker_color="#38BDF8",
                text=pick_counts.values,
                textposition="outside",
            ))
            fig.update_layout(
                height=300,
                xaxis_title="次数",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#AAB7CC",
                margin=dict(l=10, r=40, t=10, b=10),
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("无 pick 记录。")

with tab4:
    top_h2h = h2h.head(15)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top_h2h["Opponent"],
        x=top_h2h["Win Rate"],
        orientation="h",
        marker_color="#38BDF8",
        text=[f"{w}/{m}" for w, m in zip(top_h2h["Wins"], top_h2h["Matches"])],
        textposition="outside",
        name="胜率",
    ))
    fig.update_layout(
        height=400,
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#AAB7CC",
        margin=dict(l=10, r=80, t=10, b=10),
    )
    st.plotly_chart(fig, width='stretch')
    st.caption("对阵历史（胜场/总场），仅展示交手最多的 15 支战队。")

with tab5:
    stats_labels = ["ACS", "KAST", "ADR", "HS%", "Rating", "K/D"]
    # 融合两个方向的均值
    if len(team_a_stats) > 0 or len(team_b_stats) > 0:
        combined = {}
        for col, label in [
            ("team_a_acs", "ACS"), ("team_a_kast", "KAST"),
            ("team_a_adr", "ADR"), ("team_a_hs_pct", "HS%"),
            ("team_a_rating", "Rating"), ("team_a_kd", "K/D"),
        ]:
            vals = []
            if col in team_a_stats and pd.notna(team_a_stats[col]):
                vals.append(team_a_stats[col])
            b_col = col.replace("team_a_", "team_b_")
            if b_col in team_b_stats and pd.notna(team_b_stats[b_col]):
                vals.append(team_b_stats[b_col])
            combined[label] = sum(vals) / len(vals) if vals else 0

        # 归一化到 0-1 范围做雷达
        max_vals = {
            "ACS": 300, "KAST": 1.0, "ADR": 200,
            "HS%": 0.5, "Rating": 1.5, "K/D": 1.5,
        }
        norm = {}
        for k in combined:
            norm[k] = min(combined[k] / max_vals.get(k, 1), 1.0)

        fig = go.Figure(go.Scatterpolar(
            r=[norm[k] for k in stats_labels],
            theta=stats_labels,
            fill="toself",
            line_color="#38BDF8",
            marker_color="#38BDF8",
            opacity=0.7,
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 1], showticklabels=False),
                bgcolor="rgba(0,0,0,0)",
            ),
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#AAB7CC",
            margin=dict(l=40, r=40, t=10, b=10),
        )
        st.plotly_chart(fig, width='stretch')

        # 实际数值展示
        vc1, vc2, vc3 = st.columns(3)
        vc1.metric("ACS", f"{combined.get('ACS', 0):.0f}")
        vc1.metric("ADR", f"{combined.get('ADR', 0):.0f}")
        vc2.metric("KAST", f"{combined.get('KAST', 0):.1%}")
        vc2.metric("Rating", f"{combined.get('Rating', 0):.2f}")
        vc3.metric("HS%", f"{combined.get('HS%', 0):.1%}")
        vc3.metric("K/D", f"{combined.get('K/D', 0):.2f}")
    else:
        st.info("无统计数据。")
