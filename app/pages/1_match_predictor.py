"""Match Predictor — 比赛胜负预测（模板页面）"""

import streamlit as st
import sys
from pathlib import Path

# 确保项目根目录在 PATH 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.components.team_selector import select_team
from app.components.prediction_card import show_match_summary
from src.predictor import predict_match, predict_bp, predict_bo3_score
from app.components.sidebar import show_sidebar
from app.components.theme import apply_global_styles, page_header

st.set_page_config(page_title="Match Predictor", page_icon="🎮", layout="wide")

apply_global_styles()
show_sidebar()

page_header("🎮 Match Predictor", "赛前/赛中胜负概率预测，基于 XGBoost + 特征工程。")

# ==== 数据加载 ====

# ==== 计算 ====
c1, c2 = st.columns(2)
with c1:
    team1 = select_team("选择战队 A", key="team1", index=None)
with c2:
    team2 = select_team("选择战队 B", key="team2", index=None)

predict_btn = st.button("预测比赛", type="primary", use_container_width=True)

# ==== 渲染 ====
if predict_btn and team1 and team2:
    if team1 == team2:
        st.error("请选择两支不同的战队")
    else:
        with st.spinner("预测中…"):
            match_result = predict_match(team1, team2)
            bp_result = predict_bp(team1, team2)
            bo3_result = predict_bo3_score(team1, team2)

        # 合并结果为完整 dict
        full_result = {**match_result, **bp_result, **bo3_result}

        show_match_summary(full_result, team1, team2)
elif predict_btn:
    st.warning("请先选择两支战队")
