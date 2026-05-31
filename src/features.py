"""特征构建：队伍 form、滚动统计、对战历史"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")

# 每个队伍近 N 场作为 form 窗口
FORM_WINDOWS = [5, 10, 20]


def _add_tournament_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """按年份 + 赛事层级提取时间顺序"""
    # 赛事层级优先级（数值越小越早）
    TIER_PRIORITY = {
        "Kickoff": 0,
        "Challengers": 1,
        "League": 2,
        "Masters": 3,
        "Champions": 4,
        "Last Chance": 5,
        "Qualifier": 6,
    }

    def _tier_rank(name: str) -> int:
        name_lower = name.lower()
        for keyword, rank in TIER_PRIORITY.items():
            if keyword.lower() in name_lower:
                return rank
        return 5  # 默认中间

    # 直接从 Tournament 名提取年份
    def _extract_year(name: str) -> int:
        import re
        match = re.search(r"(20\d{2})", name)
        return int(match.group(1)) if match else 0

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
    match_df = match_df.sort_values(["tord", "Stage", "Match Name"]).reset_index(drop=True)

    # 构建 team-match 映射表
    records = []
    for _, row in match_df.iterrows():
        records.append({"match_idx": _, "team": row["Team A"], "opponent": row["Team B"],
                        "win": row["target"], "tord": row["tord"]})
        records.append({"match_idx": _, "team": row["Team B"], "opponent": row["Team A"],
                        "win": 1 - row["target"], "tord": row["tord"]})

    team_log = pd.DataFrame(records)

    # 滚动统计
    for w in FORM_WINDOWS:
        team_log[f"form_{w}"] = team_log.groupby("team")["win"].transform(
            lambda x: x.rolling(w, min_periods=1).mean().shift(1)
        )

    # 按 match_idx 拆回 team_a / team_b
    a = team_log.iloc[::2].reset_index(drop=True)
    b = team_log.iloc[1::2].reset_index(drop=True)

    result = match_df.copy()
    for w in FORM_WINDOWS:
        result[f"team_a_form_{w}"] = a[f"form_{w}"]
        result[f"team_b_form_{w}"] = b[f"form_{w}"]
        result[f"form_diff_{w}"] = a[f"form_{w}"] - b[f"form_{w}"]

    return result


def build_h2h_features(match_df: pd.DataFrame) -> pd.DataFrame:
    """构建历史对战特征"""
    match_df = _add_tournament_ordinal(match_df)
    match_df = match_df.sort_values(["tord", "Stage", "Match Name"]).reset_index(drop=True)

    a_wins = []
    b_wins = []
    h2h_map = {}

    for _, row in match_df.iterrows():
        ta, tb = row["Team A"], row["Team B"]
        key = tuple(sorted([ta, tb]))
        w_a, w_b = h2h_map.get(key, (0, 0))
        a_wins.append(w_a)
        b_wins.append(w_b)

        # 更新
        if row["target"] == 1:
            h2h_map[key] = (w_a + 1, w_b)
        else:
            h2h_map[key] = (w_a, w_b + 1)

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
    stat_cols = ["acs", "kast", "adr", "hs_pct", "rating", "kd", "fk"]
    for col in stat_cols:
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
    feature_cols = [c for c in df.columns if c not in [
        "Tournament", "Stage", "Match Type", "Match Name",
        "Team A", "Team B", "Team A Score", "Team B Score",
        "Match Result", "Year", "target", "total_score",
        "team_a_acs", "team_a_kast", "team_a_adr", "team_a_hs_pct",
        "team_a_rating", "team_a_kills", "team_a_deaths", "team_a_assists",
        "team_a_kd", "team_a_fk", "team_a_fd",
        "team_b_acs", "team_b_kast", "team_b_adr", "team_b_hs_pct",
        "team_b_rating", "team_b_kills", "team_b_deaths", "team_b_assists",
        "team_b_kd", "team_b_fk", "team_b_fd",
    ]]
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")


if __name__ == "__main__":
    run_feature_engineering()
