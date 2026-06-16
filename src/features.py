"""特征构建：队伍 form、滚动统计、对战历史"""

import re

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")

# 每个队伍近 N 场作为 form 窗口
FORM_WINDOWS = [5, 10, 20]

SORT_COLUMNS = [
    "tord",
    "Stage",
    "Match Name",
]

TOURNAMENT_TIER_PRIORITY = {
    "Kickoff": 0,
    "Challengers": 1,
    "League": 2,
    "Masters": 3,
    "Champions": 4,
    "Last Chance": 5,
    "Qualifier": 6,
}

STAT_COLUMNS = [
    "acs",
    "kast",
    "adr",
    "hs_pct",
    "rating",
    "kd",
    "fk",
]

BASE_MATCH_COLUMNS = [
    "Tournament",
    "Stage",
    "Match Type",
    "Match Name",
    "Team A",
    "Team B",
    "Team A Score",
    "Team B Score",
    "Match Result",
    "Year",
    "target",
    "total_score",
]

TEAM_STAT_COLUMNS = [
    "acs",
    "kast",
    "adr",
    "hs_pct",
    "rating",
    "kills",
    "deaths",
    "assists",
    "kd",
    "fk",
    "fd",
]


def _tier_rank(name: str) -> int:
    name_lower = name.lower()
    for keyword, rank in TOURNAMENT_TIER_PRIORITY.items():
        if keyword.lower() in name_lower:
            return rank
    return 5  # 默认中间


def _extract_year(name: str) -> int:
    match = re.search(r"(20\d{2})", name)
    return int(match.group(1)) if match else 0


def _sort_matches_by_time(match_df: pd.DataFrame) -> pd.DataFrame:
    return match_df.sort_values(SORT_COLUMNS).reset_index(drop=True)


def _team_record(
    match_idx: int,
    team: str,
    opponent: str,
    win: float,
    tord: int,
) -> dict:
    return {
        "match_idx": match_idx,
        "team": team,
        "opponent": opponent,
        "win": win,
        "tord": tord,
    }


def _build_team_match_records(match_df: pd.DataFrame) -> list[dict]:
    records = []
    for match_idx, row in match_df.iterrows():
        records.append(
            _team_record(
                match_idx=match_idx,
                team=row["Team A"],
                opponent=row["Team B"],
                win=row["target"],
                tord=row["tord"],
            )
        )
        records.append(
            _team_record(
                match_idx=match_idx,
                team=row["Team B"],
                opponent=row["Team A"],
                win=1 - row["target"],
                tord=row["tord"],
            )
        )
    return records


def _add_form_windows(team_log: pd.DataFrame) -> pd.DataFrame:
    for window in FORM_WINDOWS:
        team_log[f"form_{window}"] = team_log.groupby("team")["win"].transform(
            lambda x: x.rolling(window, min_periods=1).mean().shift(1)
        )
    return team_log


def _add_form_columns(
    result: pd.DataFrame,
    team_a_log: pd.DataFrame,
    team_b_log: pd.DataFrame,
) -> pd.DataFrame:
    for window in FORM_WINDOWS:
        team_a_col = f"team_a_form_{window}"
        team_b_col = f"team_b_form_{window}"
        form_col = f"form_{window}"

        result[team_a_col] = team_a_log[form_col]
        result[team_b_col] = team_b_log[form_col]
        result[f"form_diff_{window}"] = team_a_log[form_col] - team_b_log[form_col]

    return result


def _h2h_key(team_a: str, team_b: str) -> tuple[str, str]:
    return tuple(sorted([team_a, team_b]))


def _append_h2h_snapshot(
    row: pd.Series,
    h2h_map: dict,
    a_wins: list[int],
    b_wins: list[int],
) -> tuple[int, int]:
    key = _h2h_key(row["Team A"], row["Team B"])
    wins = h2h_map.get(key, (0, 0))
    a_wins.append(wins[0])
    b_wins.append(wins[1])
    return wins


def _update_h2h_record(
    row: pd.Series,
    h2h_map: dict,
    wins: tuple[int, int],
) -> None:
    key = _h2h_key(row["Team A"], row["Team B"])
    w_a, w_b = wins

    # 保持原有语义：target=1 时累计第一个计数字段，否则累计第二个。
    if row["target"] == 1:
        h2h_map[key] = (w_a + 1, w_b)
    else:
        h2h_map[key] = (w_a, w_b + 1)


def _excluded_feature_columns() -> list[str]:
    team_a_stats = [f"team_a_{column}" for column in TEAM_STAT_COLUMNS]
    team_b_stats = [f"team_b_{column}" for column in TEAM_STAT_COLUMNS]
    return BASE_MATCH_COLUMNS + team_a_stats + team_b_stats


def _add_tournament_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """按年份 + 赛事层级提取时间顺序"""
    df = df.copy()
    df["_year_extracted"] = df["Tournament"].apply(_extract_year)
    df["_tier"] = df["Tournament"].apply(_tier_rank)
    # 按 (年份, 赛事层级, Stage) 排序
    df["tord"] = df.groupby(["Year", "_year_extracted"], group_keys=False).cumcount()
    df = df.sort_values(["Year", "_tier", "tord"]).reset_index(drop=True)
    df["tord"] = range(len(df))
    return df.drop(columns=["_year_extracted", "_tier"])


def build_team_form_features(match_df: pd.DataFrame) -> pd.DataFrame:
    """为每场比赛构建队伍近期 form 特征"""
    match_df = _add_tournament_ordinal(match_df)
    match_df = _sort_matches_by_time(match_df)

    # 构建 team-match 映射表
    team_log = pd.DataFrame(_build_team_match_records(match_df))

    # 滚动统计
    team_log = _add_form_windows(team_log)

    # 按 match_idx 拆回 team_a / team_b
    team_a_log = team_log.iloc[::2].reset_index(drop=True)
    team_b_log = team_log.iloc[1::2].reset_index(drop=True)

    result = match_df.copy()
    return _add_form_columns(result, team_a_log, team_b_log)


def build_h2h_features(match_df: pd.DataFrame) -> pd.DataFrame:
    """构建历史对战特征"""
    match_df = _add_tournament_ordinal(match_df)
    match_df = _sort_matches_by_time(match_df)

    a_wins = []
    b_wins = []
    h2h_map = {}

    for _, row in match_df.iterrows():
        wins = _append_h2h_snapshot(row, h2h_map, a_wins, b_wins)
        _update_h2h_record(row, h2h_map, wins)

    match_df["team_a_h2h_wins"] = a_wins
    match_df["team_b_h2h_wins"] = b_wins
    match_df["h2h_total"] = match_df["team_a_h2h_wins"] + match_df["team_b_h2h_wins"]
    match_df["h2h_ratio"] = np.where(
        match_df["h2h_total"] > 0,
        match_df["team_a_h2h_wins"] / match_df["h2h_total"],
        0.5,
    )
    return match_df


def build_stat_diff_features(match_df: pd.DataFrame) -> pd.DataFrame:
    """构建双方统计量差值特征"""
    for col in STAT_COLUMNS:
        a_col = f"team_a_{col}"
        b_col = f"team_b_{col}"
        if a_col in match_df.columns and b_col in match_df.columns:
            match_df[f"diff_{col}"] = match_df[a_col] - match_df[b_col]
    return match_df


def build_all_features(match_df: pd.DataFrame) -> pd.DataFrame:
    """执行全部特征构建"""
    df = match_df.copy()

    print("Building form features...")
    df = build_team_form_features(df)

    print("Building H2H features...")
    df = build_h2h_features(df)

    print("Building stat diff features...")
    df = build_stat_diff_features(df)

    return df


def run_feature_engineering():
    """读取 clean data → 构建特征 → 保存"""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    match_df = pd.read_parquet(PROCESSED_DIR / "match_dataset.parquet")
    print(f"Loaded {len(match_df)} matches")

    df = build_all_features(match_df)
    df.to_parquet(FEATURES_DIR / "match_features.parquet", index=False)
    print(f"Saved {len(df)} rows with {len(df.columns)} columns")

    # 打印特征列
    feature_cols = [
        column
        for column in df.columns
        if column not in _excluded_feature_columns()
    ]
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")


if __name__ == "__main__":
    run_feature_engineering()
