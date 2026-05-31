"""预测推理接口：胜负/BP(MAP)/BO3 比分

供给前端三个函数：
    predict_match(team1, team2, stats_a=None, stats_b=None)
    predict_bp(team1, team2, n_top=5)
    predict_bo3_score(team1, team2)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Optional

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = Path("models")

# 17 个训练特征列（顺序与模型一致）
FEATURE_COLS = [
    "team_a_form_5", "team_b_form_5", "form_diff_5",
    "team_a_form_10", "team_b_form_10", "form_diff_10",
    "team_a_form_20", "team_b_form_20", "form_diff_20",
    "h2h_ratio",
    "diff_acs", "diff_kast", "diff_adr", "diff_hs_pct",
    "diff_rating", "diff_kd", "diff_fk",
]

# 只需 form + H2H 即可推理的特征索引（其余会被 imputer 填充）
PREMATCH_FEATURE_IDX = list(range(10))  # 前 10 个是 form + h2h

# 缓存
_model: any = None
_imputer: any = None
_feature_df: Optional[pd.DataFrame] = None
_draft_df: Optional[pd.DataFrame] = None
_team_form_cache: dict = {}
_team_ban_cache: dict = {}
_team_pick_cache: dict = {}


def _load_resources():
    global _model, _imputer, _feature_df, _draft_df
    if _model is None:
        _model = joblib.load(MODELS_DIR / "xgb_model.pkl")
        _imputer = joblib.load(MODELS_DIR / "imputer.pkl")
        _feature_df = pd.read_parquet(FEATURES_DIR / "match_features.parquet")
        _draft_df = pd.read_parquet(PROCESSED_DIR / "draft_dataset.parquet")


def _get_team_form(team: str) -> dict:
    """获取某支队伍最新的 form 值（滚动窗口 5/10/20）"""
    if team in _team_form_cache:
        return _team_form_cache[team]

    df = _feature_df
    as_a = df[df["Team A"] == team].sort_values(["Year", "tord"])
    as_b = df[df["Team B"] == team].sort_values(["Year", "tord"])

    if len(as_a) > 0:
        last = as_a.iloc[-1]
        result = {
            "form_5": last["team_a_form_5"],
            "form_10": last["team_a_form_10"],
            "form_20": last["team_a_form_20"],
        }
    elif len(as_b) > 0:
        last = as_b.iloc[-1]
        result = {
            "form_5": last["team_b_form_5"],
            "form_10": last["team_b_form_10"],
            "form_20": last["team_b_form_20"],
        }
    else:
        result = {"form_5": 0.5, "form_10": 0.5, "form_20": 0.5}

    # 处理 NaN（新队伍没有足够历史）
    for k in result:
        if pd.isna(result[k]):
            result[k] = 0.5

    _team_form_cache[team] = result
    return result


def _get_h2h(team1: str, team2: str) -> float:
    """获取两队历史交手 ratio（从 team1 视角）"""
    df = _feature_df
    mask_a = (df["Team A"] == team1) & (df["Team B"] == team2)
    mask_b = (df["Team A"] == team2) & (df["Team B"] == team1)

    if mask_a.any():
        last = df[mask_a].iloc[-1]
        return float(last["h2h_ratio"])
    elif mask_b.any():
        last = df[mask_b].iloc[-1]
        return 1.0 - float(last["h2h_ratio"])
    return 0.5


def predict_match(
    team1: str,
    team2: str,
    stats_a: Optional[dict] = None,
    stats_b: Optional[dict] = None,
) -> dict:
    """预测 team1 vs team2 的胜负概率

    Args:
        team1: 队伍1名称
        team2: 队伍2名称
        stats_a: 赛中模式 — team1 实时统计
                 含 {acs, kast, adr, hs_pct, rating, kd, fk}
        stats_b: 赛中模式 — team2 实时统计（同上字段）

    Returns:
        {"team1_win_prob": 0.62, "team2_win_prob": 0.38, "mode": "pre_match"}
    """
    _load_resources()

    f1 = _get_team_form(team1)
    f2 = _get_team_form(team2)

    feature_values = {
        "team_a_form_5": f1["form_5"],
        "team_b_form_5": f2["form_5"],
        "form_diff_5": f1["form_5"] - f2["form_5"],
        "team_a_form_10": f1["form_10"],
        "team_b_form_10": f2["form_10"],
        "form_diff_10": f1["form_10"] - f2["form_10"],
        "team_a_form_20": f1["form_20"],
        "team_b_form_20": f2["form_20"],
        "form_diff_20": f1["form_20"] - f2["form_20"],
        "h2h_ratio": _get_h2h(team1, team2),
    }

    # Diff 特征：赛中模式用实时数据计算，赛前模式置空
    if stats_a is not None and stats_b is not None:
        diff_keys = ["acs", "kast", "adr", "hs_pct", "rating", "kd", "fk"]
        diff = {k: stats_a.get(k, np.nan) - stats_b.get(k, np.nan) for k in diff_keys}
        mode = "in_match"
    else:
        diff = {k: np.nan for k in ["acs", "kast", "adr", "hs_pct", "rating", "kd", "fk"]}
        mode = "pre_match"

    feature_values.update({
        "diff_acs": diff["acs"],
        "diff_kast": diff["kast"],
        "diff_adr": diff["adr"],
        "diff_hs_pct": diff["hs_pct"],
        "diff_rating": diff["rating"],
        "diff_kd": diff["kd"],
        "diff_fk": diff["fk"],
    })

    X = pd.DataFrame([feature_values])[FEATURE_COLS].values
    X = _imputer.transform(X)
    prob = float(_model.predict_proba(X)[0, 1])

    return {
        "team1_win_prob": round(prob, 4),
        "team2_win_prob": round(1 - prob, 4),
        "mode": mode,
    }


def _team_ban_pick(team: str):
    """获取队伍的历史 ban/pick 统计"""
    if team in _team_ban_cache and team in _team_pick_cache:
        return _team_ban_cache[team], _team_pick_cache[team]

    df = _draft_df
    team_df = df[df["Team"] == team]

    if len(team_df) == 0:
        return {}, {}

    total = len(team_df)
    bans = team_df[team_df["Action"] == "ban"]["Map"].value_counts()
    picks = team_df[team_df["Action"] == "pick"]["Map"].value_counts()

    ban_dist = {map: round(count / total, 4) for map, count in bans.items()}
    pick_dist = {map: round(count / total, 4) for map, count in picks.items()}

    _team_ban_cache[team] = ban_dist
    _team_pick_cache[team] = pick_dist
    return ban_dist, pick_dist


def predict_bp(team1: str, team2: str, n_top: int = 5) -> dict:
    """预测 BP（地图 Ban/Pick）概率

    基于两队历史 BP 记录，返回最可能的 ban/pick 分布。

    Returns:
        {
            "team1_bans": [{"map": "Ascent", "prob": 0.15}, ...],
            "team1_picks": [{"map": "Bind", "prob": 0.12}, ...],
            "team2_bans": [...],
            "team2_picks": [...],
            "global_ban_rate": {"Ascent": 0.08, ...},
            "global_pick_rate": {"Ascent": 0.10, ...},
        }
    """
    _load_resources()

    b1, p1 = _team_ban_pick(team1)
    b2, p2 = _team_ban_pick(team2)

    def _top_n(dist: dict, n: int) -> list:
        return [{"map": k, "prob": v} for k, v in
                sorted(dist.items(), key=lambda x: -x[1])[:n]]

    # 全局统计
    total = len(_draft_df)
    all_bans = _draft_df[_draft_df["Action"] == "ban"]["Map"].value_counts()
    all_picks = _draft_df[_draft_df["Action"] == "pick"]["Map"].value_counts()

    global_ban_rate = {m: round(c / total, 4) for m, c in all_bans.items()}
    global_pick_rate = {m: round(c / total, 4) for m, c in all_picks.items()}

    return {
        "team1_bans": _top_n(b1, n_top),
        "team1_picks": _top_n(p1, n_top),
        "team2_bans": _top_n(b2, n_top),
        "team2_picks": _top_n(p2, n_top),
        "global_ban_rate": dict(sorted(global_ban_rate.items(), key=lambda x: -x[1])[:n_top]),
        "global_pick_rate": dict(sorted(global_pick_rate.items(), key=lambda x: -x[1])[:n_top]),
    }


def predict_bo3_score(team1: str, team2: str) -> dict:
    """预测 BO3 比分概率

    链式乘法：每张地图独立且概率相同（近似）。

    Returns:
        {"2-0": 0.25, "2-1": 0.30, "1-2": 0.25, "0-2": 0.20}
    """
    result = predict_match(team1, team2)
    p = result["team1_win_prob"]
    q = 1 - p

    return {
        "2-0": round(p * p, 4),
        "2-1": round(2 * p * p * q, 4),
        "1-2": round(2 * p * q * q, 4),
        "0-2": round(q * q, 4),
    }


# ---- CLI 测试 ----
if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        t1, t2 = sys.argv[1], sys.argv[2]
    else:
        t1, t2 = "FNATIC", "LOUD"

    print(f"\n=== predict_match({t1}, {t2}) ===")
    print(predict_match(t1, t2))

    print(f"\n=== predict_match({t1}, {t2}, in-match) ===")
    sample_stats = {
        "acs": 250, "kast": 0.75, "adr": 150, "hs_pct": 0.22,
        "rating": 1.15, "kd": 8, "fk": 12,
    }
    print(predict_match(t1, t2, sample_stats, sample_stats))

    print(f"\n=== predict_bp({t1}, {t2}) ===")
    bp = predict_bp(t1, t2)
    for k, v in bp.items():
        print(f"  {k}: {v}")

    print(f"\n=== predict_bo3_score({t1}, {t2}) ===")
    print(predict_bo3_score(t1, t2))
