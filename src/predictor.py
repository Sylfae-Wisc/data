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

# 赛前预测使用最近 N 场队伍表现均值，填充模型最重要的统计差值特征。
RECENT_STAT_WINDOW = 10
LONG_TERM_STAT_WEIGHT = 0.85
STAT_KEYS = ["acs", "kast", "adr", "hs_pct", "rating", "kd", "fk"]
FORM_WINDOWS = [5, 10, 20]
FORM_KEYS = [f"form_{window}" for window in FORM_WINDOWS]
DEFAULT_FORM_VALUE = 0.5
DEFAULT_H2H_RATIO = 0.5
MIN_PROBABILITY = 0.01
MAX_PROBABILITY = 0.99
H2H_MATCH_DECAY = 0.65
H2H_YEAR_DECAY = 0.8
H2H_FULL_WEIGHT_AT = 2.0
H2H_MAX_BLEND_WEIGHT = 0.18
IN_MATCH_H2H_WEIGHT_MULTIPLIER = 0.45

TEAM_FORM_COLUMN_MAP = {
    "Team A": {f"team_a_form_{window}": f"form_{window}" for window in FORM_WINDOWS},
    "Team B": {f"team_b_form_{window}": f"form_{window}" for window in FORM_WINDOWS},
}

TEAM_STAT_COLUMN_MAP = {
    "Team A": {f"team_a_{key}": key for key in STAT_KEYS},
    "Team B": {f"team_b_{key}": key for key in STAT_KEYS},
}

EMPTY_H2H_RESULT = {
    "ratio": DEFAULT_H2H_RATIO,
    "matches": 0,
    "effective_weight": 0.0,
    "blend_weight": 0.0,
}

# 缓存
_model: any = None
_imputer: any = None
_feature_df: Optional[pd.DataFrame] = None
_draft_df: Optional[pd.DataFrame] = None
_team_alias_map: dict[str, str] = {}
_team_form_cache: dict = {}
_team_long_term_stats_cache: dict = {}
_team_recent_stats_cache: dict = {}
_team_strength_stats_cache: dict = {}
_decayed_h2h_cache: dict = {}
_team_ban_cache: dict = {}
_team_pick_cache: dict = {}


def _canonical_team_names(feature_df: pd.DataFrame) -> list[str]:
    return sorted(set(feature_df["Team A"]).union(set(feature_df["Team B"])))


def _add_team_alias(alias_map: dict[str, str], alias: str, canonical: str) -> None:
    alias = str(alias).strip()
    if alias:
        alias_map[alias.lower()] = canonical


def _load_team_alias_map(teams: list[str]) -> dict[str, str]:
    alias_map = {team.strip().lower(): team for team in teams}
    mapping_path = DATA_DIR / "raw" / "all_ids" / "all_teams_mapping.csv"
    if not mapping_path.exists():
        return alias_map

    mapping = pd.read_csv(mapping_path)
    for _, row in mapping.iterrows():
        full_name = str(row.get("Full Name", "")).strip()
        abbreviated = str(row.get("Abbreviated", "")).strip()
        if full_name in teams:
            _add_team_alias(alias_map, full_name, full_name)
            _add_team_alias(alias_map, abbreviated, full_name)

    return alias_map


def _default_form() -> dict:
    return {key: DEFAULT_FORM_VALUE for key in FORM_KEYS}


def _sanitize_form_values(values: dict) -> dict:
    return {
        key: DEFAULT_FORM_VALUE
        if pd.isna(value)
        else value
        for key, value in values.items()
    }


def _team_form_slice(team: str, side: str) -> pd.DataFrame:
    column_map = TEAM_FORM_COLUMN_MAP[side]
    return _feature_df[_feature_df[side] == team][
        ["Year", "tord"] + list(column_map.keys())
    ].rename(columns=column_map)


def _team_stat_slice(team: str, side: str) -> pd.DataFrame:
    column_map = TEAM_STAT_COLUMN_MAP[side]
    return _feature_df[_feature_df[side] == team][
        ["Year", "tord"] + list(column_map.keys())
    ].rename(columns=column_map)


def _chronological_history(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True).sort_values(["Year", "tord"])


def _empty_h2h_result() -> dict:
    return dict(EMPTY_H2H_RESULT)


def _h2h_cache_key(team1: str, team2: str) -> tuple[str, str]:
    return team1, team2


def _h2h_match_mask(df: pd.DataFrame, team1: str, team2: str) -> pd.Series:
    return (
        ((df["Team A"] == team1) & (df["Team B"] == team2)) |
        ((df["Team A"] == team2) & (df["Team B"] == team1))
    )


def _decayed_match_weight(match_age: int, year_age: int) -> float:
    return (H2H_MATCH_DECAY ** match_age) * (H2H_YEAR_DECAY ** year_age)


def _clamp_probability(value: float) -> float:
    return min(max(value, MIN_PROBABILITY), MAX_PROBABILITY)


def _load_resources():
    global _model, _imputer, _feature_df, _draft_df, _team_alias_map
    if _model is None:
        _model = joblib.load(MODELS_DIR / "xgb_model.pkl")
        _imputer = joblib.load(MODELS_DIR / "imputer.pkl")
        _feature_df = pd.read_parquet(FEATURES_DIR / "match_features.parquet")
        _draft_df = pd.read_parquet(PROCESSED_DIR / "draft_dataset.parquet")
        _team_alias_map = _load_team_alias_map(_canonical_team_names(_feature_df))


def _normalize_team_name(team: str) -> str:
    """Map UI labels/abbreviations to the canonical names used in feature data."""
    _load_resources()
    team = str(team).strip()
    return _team_alias_map.get(team.lower(), team)


def _get_team_form(team: str) -> dict:
    """获取某支队伍最新的 form 值（滚动窗口 5/10/20）"""
    team = _normalize_team_name(team)
    if team in _team_form_cache:
        return _team_form_cache[team]

    history = _chronological_history([
        _team_form_slice(team, "Team A"),
        _team_form_slice(team, "Team B"),
    ])
    if history.empty:
        result = _default_form()
    else:
        last = history.iloc[-1]
        result = {
            key: last[key]
            for key in FORM_KEYS
        }

    # 处理 NaN（新队伍没有足够历史）
    result = _sanitize_form_values(result)

    _team_form_cache[team] = result
    return result


def _get_h2h(team1: str, team2: str) -> float:
    """获取两队历史交手 ratio（从 team1 视角）"""
    team1 = _normalize_team_name(team1)
    team2 = _normalize_team_name(team2)
    df = _feature_df
    mask_a = (df["Team A"] == team1) & (df["Team B"] == team2)
    mask_b = (df["Team A"] == team2) & (df["Team B"] == team1)

    if mask_a.any():
        last = df[mask_a].iloc[-1]
        return float(last["h2h_ratio"])
    elif mask_b.any():
        last = df[mask_b].iloc[-1]
        return 1.0 - float(last["h2h_ratio"])
    return DEFAULT_H2H_RATIO


def _match_result_for_team(row: pd.Series, team: str) -> float:
    """Return 1/0.5/0 from team's perspective for one historical match."""
    team_a_score = row["Team A Score"]
    team_b_score = row["Team B Score"]
    if pd.isna(team_a_score) or pd.isna(team_b_score):
        return 0.5
    if team_a_score == team_b_score:
        return 0.5

    team_a_won = team_a_score > team_b_score
    if row["Team A"] == team:
        return 1.0 if team_a_won else 0.0
    return 0.0 if team_a_won else 1.0


def _get_decayed_h2h(team1: str, team2: str) -> dict:
    """时间衰减 H2H：最近交手影响最大，久远交手逐步变弱。"""
    team1 = _normalize_team_name(team1)
    team2 = _normalize_team_name(team2)
    cache_key = _h2h_cache_key(team1, team2)
    if cache_key in _decayed_h2h_cache:
        return _decayed_h2h_cache[cache_key]

    df = _feature_df
    mask = _h2h_match_mask(df, team1, team2)
    matches = df[mask].sort_values(["Year", "tord"])
    if matches.empty:
        result = _empty_h2h_result()
        _decayed_h2h_cache[cache_key] = result
        return result

    latest_year = int(df["Year"].max())
    weighted_wins = 0.0
    total_weight = 0.0

    for match_age, (_, row) in enumerate(matches.iloc[::-1].iterrows()):
        year_age = max(0, latest_year - int(row["Year"]))
        weight = _decayed_match_weight(match_age, year_age)
        weighted_wins += _match_result_for_team(row, team1) * weight
        total_weight += weight

    ratio = weighted_wins / total_weight if total_weight else DEFAULT_H2H_RATIO
    confidence = min(total_weight / H2H_FULL_WEIGHT_AT, 1.0)
    result = {
        "ratio": float(ratio),
        "matches": int(len(matches)),
        "effective_weight": float(total_weight),
        "blend_weight": float(H2H_MAX_BLEND_WEIGHT * confidence),
    }
    _decayed_h2h_cache[cache_key] = result
    return result


def _blend_with_decayed_h2h(prob: float, team1: str, team2: str, mode: str) -> tuple[float, dict]:
    """Use decayed H2H as a bounded post-model calibration signal."""
    h2h = _get_decayed_h2h(team1, team2)
    blend_weight = h2h["blend_weight"]
    if mode == "in_match":
        blend_weight *= IN_MATCH_H2H_WEIGHT_MULTIPLIER

    adjusted = prob * (1 - blend_weight) + h2h["ratio"] * blend_weight
    return float(_clamp_probability(adjusted)), {**h2h, "blend_weight": blend_weight}


def _get_team_stat_history(team: str) -> pd.DataFrame:
    """Return team stat rows from both sides in chronological order."""
    team = _normalize_team_name(team)
    history = _chronological_history([
        _team_stat_slice(team, "Team A"),
        _team_stat_slice(team, "Team B"),
    ])
    if history.empty:
        return history

    history[STAT_KEYS] = history[STAT_KEYS].apply(pd.to_numeric, errors="coerce")
    return history


def _mean_stats(rows: pd.DataFrame) -> dict:
    """Convert stat rows into a plain mean-stat dict."""
    if rows.empty:
        return {key: np.nan for key in STAT_KEYS}

    return {
        key: float(value) if not pd.isna(value) else np.nan
        for key, value in rows[STAT_KEYS].mean().items()
    }


def _get_team_long_term_stats(team: str) -> dict:
    """获取队伍全历史平均表现，作为基础实力画像。"""
    team = _normalize_team_name(team)
    if team in _team_long_term_stats_cache:
        return _team_long_term_stats_cache[team]

    result = _mean_stats(_get_team_stat_history(team))
    _team_long_term_stats_cache[team] = result
    return result


def _get_team_recent_stats(team: str, n: int = RECENT_STAT_WINDOW) -> dict:
    """获取队伍最近 N 场的平均表现，用作赛前实力对比特征。"""
    team = _normalize_team_name(team)
    cache_key = (team, n)
    if cache_key in _team_recent_stats_cache:
        return _team_recent_stats_cache[cache_key]

    result = _mean_stats(_get_team_stat_history(team).tail(n))
    _team_recent_stats_cache[cache_key] = result
    return result


def _blend_stat_value(long_value: float, recent_value: float) -> float:
    """Blend long-term strength with recent form while handling sparse data."""
    if pd.isna(long_value) and pd.isna(recent_value):
        return np.nan
    if pd.isna(long_value):
        return recent_value
    if pd.isna(recent_value):
        return long_value
    return LONG_TERM_STAT_WEIGHT * long_value + (1 - LONG_TERM_STAT_WEIGHT) * recent_value


def _get_team_strength_stats(team: str) -> dict:
    """长期实力为主、近期状态为辅的赛前统计画像。"""
    team = _normalize_team_name(team)
    if team in _team_strength_stats_cache:
        return _team_strength_stats_cache[team]

    long_term = _get_team_long_term_stats(team)
    recent = _get_team_recent_stats(team)
    result = {
        key: _blend_stat_value(long_term.get(key, np.nan), recent.get(key, np.nan))
        for key in STAT_KEYS
    }

    _team_strength_stats_cache[team] = result
    return result


def _form_feature_values(form_a: dict, form_b: dict) -> dict:
    return {
        **{
            f"team_a_form_{window}": form_a[f"form_{window}"]
            for window in FORM_WINDOWS
        },
        **{
            f"team_b_form_{window}": form_b[f"form_{window}"]
            for window in FORM_WINDOWS
        },
        **{
            f"form_diff_{window}": (
                form_a[f"form_{window}"] - form_b[f"form_{window}"]
            )
            for window in FORM_WINDOWS
        },
    }


def _stat_diff_values(stats_a: dict, stats_b: dict) -> dict:
    return {
        f"diff_{key}": stats_a.get(key, np.nan) - stats_b.get(key, np.nan)
        for key in STAT_KEYS
    }


def _resolve_match_stats(
    team1: str,
    team2: str,
    stats_a: Optional[dict],
    stats_b: Optional[dict],
) -> tuple[dict, dict]:
    if stats_a is None:
        stats_a = _get_team_strength_stats(team1)
    if stats_b is None:
        stats_b = _get_team_strength_stats(team2)
    return stats_a, stats_b


def _build_match_feature_values(
    team1: str,
    team2: str,
    stats_a: Optional[dict] = None,
    stats_b: Optional[dict] = None,
) -> dict:
    """Build model features with team1 occupying the Team A side."""
    f1 = _get_team_form(team1)
    f2 = _get_team_form(team2)

    feature_values = {
        **_form_feature_values(f1, f2),
        "h2h_ratio": _get_h2h(team1, team2),
    }

    stats_a, stats_b = _resolve_match_stats(team1, team2, stats_a, stats_b)
    feature_values.update(_stat_diff_values(stats_a, stats_b))

    return feature_values


def _raw_match_prob(
    team1: str,
    team2: str,
    stats_a: Optional[dict] = None,
    stats_b: Optional[dict] = None,
) -> float:
    """Raw model probability that team1 wins as the Team A side."""
    feature_values = _build_match_feature_values(team1, team2, stats_a, stats_b)
    X = pd.DataFrame([feature_values])[FEATURE_COLS].values
    X = _imputer.transform(X)
    return float(_model.predict_proba(X)[0, 1])


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
    team1 = _normalize_team_name(team1)
    team2 = _normalize_team_name(team2)

    # The trained XGBoost model has a Team A position prior. Run both directions
    # and combine them so swapping left/right returns complementary probabilities.
    p_forward = _raw_match_prob(team1, team2, stats_a, stats_b)
    p_reverse = _raw_match_prob(team2, team1, stats_b, stats_a)
    prob = (p_forward + (1 - p_reverse)) / 2
    raw_prob = prob
    mode = "in_match" if stats_a is not None and stats_b is not None else "pre_match"
    prob, h2h = _blend_with_decayed_h2h(prob, team1, team2, mode)

    return {
        "team1_win_prob": round(prob, 4),
        "team2_win_prob": round(1 - prob, 4),
        "mode": mode,
        "feature_mode": "team_strength_blend" if mode == "pre_match" else "live_stats",
        "stat_window": RECENT_STAT_WINDOW if mode == "pre_match" else None,
        "long_term_weight": LONG_TERM_STAT_WEIGHT if mode == "pre_match" else None,
        "h2h_ratio": round(h2h["ratio"], 4),
        "h2h_matches": h2h["matches"],
        "h2h_weight": round(h2h["blend_weight"], 4),
        "h2h_effective_weight": round(h2h["effective_weight"], 4),
        "raw_prob": round(raw_prob, 4),
    }


# 当前比赛图池
VETO_MAPS = ["Split", "Pearl", "Haven", "Breeze", "Fracture", "Ascent", "Lotus"]
TEAM_A = "A"
TEAM_B = "B"
DECIDER_TEAM = "—"
BAN_ACTION = "ban"
PICK_ACTION = "pick"
DECIDER_ACTION = "decider"


def _team_ban_pick(team: str):
    """获取队伍的历史 ban/pick 统计（归一化到 VETO_MAPS）"""
    team = _normalize_team_name(team)
    if team in _team_ban_cache and team in _team_pick_cache:
        return _team_ban_cache[team], _team_pick_cache[team]

    df = _draft_df
    team_df = df[df["Team"] == team]

    if len(team_df) == 0:
        empty = {m: 0.0 for m in VETO_MAPS}
        return empty, empty

    total = len(team_df)
    bans = team_df[team_df["Action"] == "ban"]["Map"].value_counts()
    picks = team_df[team_df["Action"] == "pick"]["Map"].value_counts()

    ban_dist = {m: round(bans.get(m, 0) / total, 4) for m in VETO_MAPS}
    pick_dist = {m: round(picks.get(m, 0) / total, 4) for m in VETO_MAPS}

    _team_ban_cache[team] = ban_dist
    _team_pick_cache[team] = pick_dist
    return ban_dist, pick_dist


def _compute_map_score(team: str) -> dict[str, float]:
    """计算队伍每张地图的偏好分（越高 = 队伍越擅长/喜欢此图）

    pick_rate - 0.7 * ban_rate，无队伍数据时用全局先验。
    """
    ban_dist, pick_dist = _team_ban_pick(team)

    # 全局先验
    total = len(_draft_df)
    global_bans = _draft_df[_draft_df["Action"] == "ban"]["Map"].value_counts()
    global_picks = _draft_df[_draft_df["Action"] == "pick"]["Map"].value_counts()

    scores = {}
    for m in VETO_MAPS:
        p = pick_dist.get(m, 0.0)
        b = ban_dist.get(m, 0.0)

        if p == 0.0 and b == 0.0:
            gp = global_picks.get(m, 0) / max(total, 1)
            gb = global_bans.get(m, 0) / max(total, 1)
            scores[m] = round(gp - 0.5 * gb, 4)
        else:
            scores[m] = round(p - 0.7 * b, 4)

    return scores


def _map_advantage(
    actor: str,
    map_name: str,
    score_a: dict[str, float],
    score_b: dict[str, float],
) -> float:
    """地图 map_name 对 actor 方的相对优势（越高 = 对 actor 越有利）"""
    s_actor = score_a if actor == TEAM_A else score_b
    s_opponent = score_b if actor == TEAM_A else score_a
    return s_actor[map_name] - s_opponent[map_name]


def _choose_map(
    remaining: set[str],
    actor: str,
    score_a: dict[str, float],
    score_b: dict[str, float],
) -> str:
    return max(
        remaining,
        key=lambda map_name: _map_advantage(actor, map_name, score_a, score_b),
    )


def _record_veto_step(
    sequence: list[dict],
    step: int,
    team: str,
    action: str,
    map_name: str,
) -> None:
    sequence.append({
        "step": step,
        "team": team,
        "action": action,
        "map": map_name,
    })


def _take_veto_map(
    remaining: set[str],
    sequence: list[dict],
    step: int,
    team: str,
    action: str,
    advantage_actor: str,
    score_a: dict[str, float],
    score_b: dict[str, float],
) -> str:
    map_name = _choose_map(remaining, advantage_actor, score_a, score_b)
    remaining.remove(map_name)
    _record_veto_step(sequence, step, team, action, map_name)
    return map_name


def _simulate_veto(team1: str, team2: str) -> dict:
    """模拟 VCT 地图 BP 流程（贪心博弈），返回最可能的 veto 结果

    流程（7 图 → 3 图）：
      Step 1: Team A ban  → 6 剩
      Step 2: Team B ban  → 5 剩
      Step 3: Team A pick → Map 1（A 选图）
      Step 4: Team B pick → Map 2（B 选图）
      Step 5: Team A ban  → 2 剩
      Step 6: Team B ban  → 1 剩 → Map 3（决胜图）
    """
    score_a = _compute_map_score(team1)
    score_b = _compute_map_score(team2)
    remaining = set(VETO_MAPS)
    sequence = []

    # Step 1: A ban — 移除对 A 最不利（对 B 最有利）的图
    ban1 = _take_veto_map(
        remaining, sequence, 1, TEAM_A, BAN_ACTION, TEAM_B, score_a, score_b
    )

    # Step 2: B ban — 移除对 B 最不利（对 A 最有利）的图
    ban2 = _take_veto_map(
        remaining, sequence, 2, TEAM_B, BAN_ACTION, TEAM_A, score_a, score_b
    )

    # Step 3: A pick — 选对 A 最有利的图
    pick_a = _take_veto_map(
        remaining, sequence, 3, TEAM_A, PICK_ACTION, TEAM_A, score_a, score_b
    )

    # Step 4: B pick — 选对 B 最有利的图
    pick_b = _take_veto_map(
        remaining, sequence, 4, TEAM_B, PICK_ACTION, TEAM_B, score_a, score_b
    )

    # Step 5: A ban
    ban3 = _take_veto_map(
        remaining, sequence, 5, TEAM_A, BAN_ACTION, TEAM_B, score_a, score_b
    )

    # Step 6: B ban
    ban4 = _take_veto_map(
        remaining, sequence, 6, TEAM_B, BAN_ACTION, TEAM_A, score_a, score_b
    )

    # 最后一张自动成为决胜图
    decider = remaining.pop()
    _record_veto_step(sequence, 7, DECIDER_TEAM, DECIDER_ACTION, decider)

    return {
        "veto_sequence": sequence,
        "final_maps": [pick_a, pick_b, decider],
        "team_a_bans": [ban1, ban3],
        "team_b_bans": [ban2, ban4],
        "team_a_pick": pick_a,
        "team_b_pick": pick_b,
        "decider": decider,
    }


def predict_bp(team1: str, team2: str) -> dict:
    """预测 BP（地图 Ban/Pick）— 模拟 VCT 7 图 → 3 图 veto 流程

    Args:
        team1: Team A 名称
        team2: Team B 名称

    Returns:
        {
            "veto_sequence": [{"step": 1, "team": "A", "action": "ban", "map": "Sunset"}, ...],
            "final_maps": ["Ascent", "Haven", "Breeze"],
            "team_a_bans": ["Sunset", "Lotus"],
            "team_b_bans": ["Split", "Pearl"],
            "team_a_pick": "Ascent",
            "team_b_pick": "Haven",
            "decider": "Breeze",
            "team_a_pick_prob": 0.12,
            "team_b_pick_prob": 0.15,
        }
    """
    _load_resources()
    team1 = _normalize_team_name(team1)
    team2 = _normalize_team_name(team2)

    result = _simulate_veto(team1, team2)

    # 附上历史 pick 概率作为置信度参考
    _, pa = _team_ban_pick(team1)
    _, pb = _team_ban_pick(team2)
    result["team_a_pick_prob"] = pa.get(result["team_a_pick"], 0.0)
    result["team_b_pick_prob"] = pb.get(result["team_b_pick"], 0.0)

    return result


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
    print(f"  BP 过程:")
    for step in bp["veto_sequence"]:
        s = step
        tag = f"Team {s['team']}" if s["team"] != "—" else "自动"
        print(f"    Step {s['step']}: {tag} {s['action']} → {s['map']}")
    print(f"  最终地图: {bp['final_maps']}")
    print(f"  Team A pick 置信度: {bp['team_a_pick_prob']:.1%}")
    print(f"  Team B pick 置信度: {bp['team_b_pick_prob']:.1%}")

    print(f"\n=== predict_bo3_score({t1}, {t2}) ===")
    print(predict_bo3_score(t1, t2))
