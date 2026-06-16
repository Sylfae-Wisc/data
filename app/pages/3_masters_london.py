"""Masters London 2026 — 瑞士轮模型回测专题页"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, accuracy_score

from app.components.sidebar import show_sidebar
from app.components.theme import apply_global_styles, page_header
from app.components.bracket import render_swiss_bracket, render_metrics_bar
from src.predictor import predict_match

st.set_page_config(page_title="Masters London", page_icon="🏅", layout="wide")

apply_global_styles()
show_sidebar()

page_header(
    "🏅 Masters London 2026",
    "瑞士轮模型回测 — 预测 vs 实际结果对照 · 准确度评估 · 冷门检测",
)

# ============================================================
# 数据加载
# ============================================================

ROUND_DATA = Path("data/masters_london_2026/matches.csv")

if not ROUND_DATA.exists():
    st.error(f"数据文件不存在：{ROUND_DATA}")
    st.stop()

raw = pd.read_csv(ROUND_DATA)

# 逐场预测
@st.cache_data(show_spinner="正在运行模型预测…")
def run_predictions(_raw: pd.DataFrame) -> list[dict]:
    results = []
    for _, row in _raw.iterrows():
        r = predict_match(str(row["team_a"]), str(row["team_b"]))
        prob_a = float(r["team1_win_prob"])
        pred_winner = row["team_a"] if prob_a > 0.5 else row["team_b"]
        is_upset = (pred_winner != row["winner"]) and (max(prob_a, 1 - prob_a) > 0.6)

        results.append({
            "round": int(row["round"]),
            "group": str(row["group"]),
            "team_a": str(row["team_a"]),
            "team_b": str(row["team_b"]),
            "score_a": int(row["score_a"]),
            "score_b": int(row["score_b"]),
            "winner": str(row["winner"]),
            "prob_a": prob_a,
            "predicted_winner": pred_winner,
            "is_upset": is_upset,
            "details": {
                "mode": r.get("mode", "pre_match"),
                "h2h_ratio": r.get("h2h_ratio", 0.5),
                "h2h_matches": r.get("h2h_matches", 0),
                "h2h_weight": r.get("h2h_weight", 0.0),
                "h2h_effective_weight": r.get("h2h_effective_weight", 0.0),
                "raw_prob": r.get("raw_prob", prob_a),
            },
        })
    return results


matches = run_predictions(raw)

# ============================================================
# 计算指标
# ============================================================

def compute_metrics(matches: list[dict]) -> dict:
    y_true = np.array([1.0 if m["winner"] == m["team_a"] else 0.0 for m in matches], dtype=float)
    y_prob = np.array([m["prob_a"] for m in matches], dtype=float)

    n = len(matches)
    n_correct = sum(1 for m in matches if m["predicted_winner"] == m["winner"])
    n_upsets = sum(1 for m in matches if m.get("is_upset"))

    def _safe(fn, *args):
        try:
            return fn(*args)
        except ValueError:
            return float("nan")

    return {
        "n_matches": n,
        "n_correct": n_correct,
        "n_upsets": n_upsets,
        "accuracy": n_correct / n if n else 0.0,
        "brier": _safe(brier_score_loss, y_true, y_prob),
        "log_loss": _safe(log_loss, y_true, y_prob),
        "roc_auc": _safe(roc_auc_score, y_true, y_prob),
    }


metrics = compute_metrics(matches)

# ============================================================
# 渲染
# ============================================================

# --- 概览 KPI ---
st.markdown(render_metrics_bar(metrics), unsafe_allow_html=True)

st.divider()

tabs = st.tabs(["📊 瑞士轮对阵图", "📈 准确度报告", "⚡ 冷门场次"])

# ── Tab 1: Swiss bracket ───────────────────────────────────────────────
with tabs[0]:
    st.subheader("London 2026 · 瑞士轮对阵（R1 → R2 → R3）")
    st.caption(
        "每张比赛卡片可点开展开详情面板。"
        "蓝色 = 冷门预测正确，红色 = 冷门预测错误。⚡ 标记为冷门场次。"
    )

    bracket_html = render_swiss_bracket(matches)
    st.html(bracket_html)

# ── Tab 2: Accuracy report ────────────────────────────────────────────
with tabs[1]:
    st.subheader("模型准确度报告")

    # Per-match table
    st.markdown("**逐场预测明细**")
    rows = []
    for m in matches:
        pa = m["prob_a"]
        pb = 1.0 - pa
        correct = m["predicted_winner"] == m["winner"]
        rows.append({
            "轮次": f"R{m['round']} ({m['group']})",
            "队伍 A": m["team_a"],
            "队伍 B": m["team_b"],
            "比分": f"{m['score_a']} – {m['score_b']}",
            "胜者": m["winner"],
            f"A 胜率": f"{pa:.1%}",
            f"B 胜率": f"{pb:.1%}",
            "预测正确": "✅" if correct else "❌",
            "冷门": "⚡" if m.get("is_upset") else "",
        })
    report_df = pd.DataFrame(rows)
    st.dataframe(
        report_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Per-round metrics
    st.markdown("**按轮次细分**")
    round_metrics = []
    for rnd in sorted(set(m["round"] for m in matches)):
        rnd_matches = [m for m in matches if m["round"] == rnd]
        rm = compute_metrics(rnd_matches)
        round_metrics.append({
            "轮次": f"R{rnd}",
            "场次": rm["n_matches"],
            "正确": f"{rm['n_correct']}/{rm['n_matches']}",
            "准确率": f"{rm['accuracy']:.0%}",
            "Brier": f"{rm['brier']:.4f}",
            "冷门": rm["n_upsets"],
        })
    st.dataframe(
        pd.DataFrame(round_metrics),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Calibration plot
    st.markdown("**校准曲线（预测概率 vs 实际频率）**")
    y_true = np.array([1.0 if m["winner"] == m["team_a"] else 0.0 for m in matches], dtype=float)
    y_prob = np.array([m["prob_a"] for m in matches], dtype=float)

    prob_true, prob_pred = [], []
    try:
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5)
    except Exception:
        # manual bins
        bins = np.linspace(0, 1, 6)
        for i in range(5):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            if mask.sum() > 0:
                prob_pred.append(y_prob[mask].mean())
                prob_true.append(y_true[mask].mean())

    if len(prob_true) > 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=prob_pred, y=prob_true,
            mode="lines+markers",
            name="模型校准",
            line=dict(color="#38BDF8", width=2),
            marker=dict(size=10, color="#38BDF8"),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="完美校准",
            line=dict(color="#75839A", width=1, dash="dash"),
        ))
        fig.update_layout(
            height=350,
            xaxis=dict(title="预测概率", range=[0, 1], tickformat=".0%"),
            yaxis=dict(title="实际频率", range=[0, 1], tickformat=".0%"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#AAB7CC",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(
            fig,
            use_container_width=True
        )
        st.caption(
            "点越靠近虚线 → 模型概率校准越好。10 场样本量较小，曲线仅供参考。"
        )
    else:
        st.info("样本不足，无法绘制校准曲线。")

# ── Tab 3: Upsets ──────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("冷门场次")

    upsets = [m for m in matches if m.get("is_upset")]
    close = [m for m in matches if m["predicted_winner"] != m["winner"] and not m.get("is_upset")]

    if not upsets and not close:
        st.success("🎉 模型在所有场次中预测方向正确！")
    else:
        if upsets:
            st.markdown(
                f'<div style="font-size:0.82rem;color:#aab7cc;margin-bottom:10px;">'
                f'以下 {len(upsets)} 场比赛模型给出较高信心（&gt;60%）但预测方向错误：</div>',
                unsafe_allow_html=True,
            )
            for m in upsets:
                pa = m["prob_a"]
                pb = 1.0 - pa
                model_confidence = max(pa, pb)
                d = m["details"]
                h2h_w = d.get("h2h_weight", 0.0)
                h2h_eff = d.get("h2h_effective_weight", 0.0)
                h2h_n = d.get("h2h_matches", 0)
                raw_p = d.get("raw_prob", pa)
                if h2h_w > 0.005:
                    h2h_status = f"H2H 校准已启用（累积权重 {h2h_eff:.2f}，校准权重 {h2h_w:.0%}）"
                    h2h_color = "#86efac"
                elif h2h_n > 0:
                    h2h_status = f"H2H 数据不足（{h2h_n} 场，累积权重 {h2h_eff:.2f} &lt; 2.0），未启用校准"
                    h2h_color = "#fbbf24"
                else:
                    h2h_status = "无历史交手记录，H2H 校准不适用"
                    h2h_color = "#75839a"
                calib_note = ""
                if abs(pa - raw_p) > 0.005:
                    arrow = "↑" if pa > raw_p else "↓"
                    calib_note = f" · H2H 校准 {arrow}{abs(pa - raw_p):.1%}（原始 {raw_p:.1%} → {pa:.1%}）"
                st.markdown(f"""
<div style="background:rgba(15,23,42,0.92);border:2px solid rgba(255,70,85,0.50);border-radius:8px;padding:16px 20px;margin:12px 0;">
<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
<div>
    <span style="font-weight:850;font-size:0.85rem;">R{m['round']} ({m['group']})</span>
    <span style="color:#aab7cc;margin-left:8px;">{m['team_a']} vs {m['team_b']}</span>
</div>
<div style="font-weight:850;font-size:1.1rem;">{m['score_a']} : {m['score_b']}</div>
<div>
    <span style="background:rgba(255,70,85,0.18);color:#FF4655;padding:3px 10px;border-radius:4px;font-weight:850;font-size:0.75rem;">
    模型信心 {model_confidence:.0%} → {m['predicted_winner']}
    </span>
    <span style="margin-left:6px;color:#86efac;font-weight:750;">实际 {m['winner']}</span>
</div>
</div>
<div style="margin-top:10px;font-size:0.78rem;color:#aab7cc;">
模型预测概率：{m['team_a']} {pa:.1%} — {m['team_b']} {pb:.1%}{calib_note}
</div>
<div style="margin-top:4px;font-size:0.72rem;color:{h2h_color};">{h2h_status}</div>
</div>""", unsafe_allow_html=True)

        if close:
            st.markdown(
                f'<div style="font-size:0.82rem;color:#aab7cc;margin:20px 0 10px;">'
                f'另有 {len(close)} 场预测接近但方向错误（信心 ≤60%，不算冷门）：</div>',
                unsafe_allow_html=True,
            )
            for m in close:
                st.markdown(f"""
<div style="background:rgba(15,23,42,0.92);border:1px solid rgba(251,191,36,0.25);border-radius:8px;padding:12px 20px;margin:8px 0;">
<span style="color:#fbbf24;">⚠</span>
R{m['round']} ({m['group']}): {m['team_a']} vs {m['team_b']}
→ {m['score_a']} : {m['score_b']}（胜者 {m['winner']}）
<span style="color:#75839a;">· 模型概率：{m['prob_a']:.1%} / {1 - m['prob_a']:.1%}</span>
</div>""", unsafe_allow_html=True)
