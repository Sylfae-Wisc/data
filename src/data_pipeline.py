"""数据加载与清洗管道：raw CSV → 合并主表 → data/processed/"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

YEARS = [2021, 2022, 2023, 2024, 2025, 2026]

# scores.csv 是 match 级（无 Map 列），overview.csv 是 player-match-map 级
# 合并时需将 overview 聚合到 match-team 级别


def load_year_scores(year: int) -> pd.DataFrame:
    path = RAW_DIR / f"vct_{year}" / "matches" / "scores.csv"
    df = pd.read_csv(path, low_memory=False)
    df["Year"] = year
    return df


def load_year_overview(year: int) -> pd.DataFrame:
    path = RAW_DIR / f"vct_{year}" / "matches" / "overview.csv"
    df = pd.read_csv(path, low_memory=False)
    df["Year"] = year
    return df


def load_year_draft_phase(year: int) -> pd.DataFrame:
    path = RAW_DIR / f"vct_{year}" / "matches" / "draft_phase.csv"
    df = pd.read_csv(path, low_memory=False)
    df["Year"] = year
    return df


def _parse_pct(series: pd.Series) -> pd.Series:
    """将 '78%' → 0.78"""
    return pd.to_numeric(series.astype(str).str.rstrip("%"), errors="coerce") / 100.0


def clean_overview(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Average Combat Score": "ACS",
        "Kills - Deaths (KD)": "KD",
        "Kill, Assist, Trade, Survive %": "KAST",
        "Average Damage Per Round": "ADR",
        "Headshot %": "HS_Pct",
        "First Kills": "FK",
        "First Deaths": "FD",
        "Kills - Deaths (FKD)": "FKD",
    }
    df = df.rename(columns=rename_map)

    # 百分比字段需先去除 % 再转数值
    pct_cols = ["KAST", "HS_Pct"]
    for col in pct_cols:
        if col in df.columns:
            df[col] = _parse_pct(df[col])

    # Rating 可能有 "1.23" 之类，直接转
    numeric_cols = ["Rating", "ACS", "Kills", "Deaths", "Assists",
                    "KD", "ADR", "FK", "FD", "FKD"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def aggregate_team_stats(overview: pd.DataFrame) -> pd.DataFrame:
    """将 overview 从 player-match-map 级聚合到 match-team 级（跨地图平均）"""
    id_cols = ["Tournament", "Stage", "Match Type", "Match Name", "Team", "Year"]
    agg_dict = {
        "ACS": "mean", "KAST": "mean", "ADR": "mean", "HS_Pct": "mean",
        "Rating": "mean", "Kills": "sum", "Deaths": "sum", "Assists": "sum",
        "KD": "mean", "FK": "sum", "FD": "sum",
    }
    available = {k: v for k, v in agg_dict.items() if k in overview.columns}
    team_stats = overview.groupby(id_cols, as_index=False).agg(available)
    team_stats.columns = [c if c in id_cols else f"team_{c.lower()}" for c in team_stats.columns]
    return team_stats


def build_match_dataset(years: Optional[list[int]] = None) -> pd.DataFrame:
    if years is None:
        years = YEARS

    scores_list, stats_list = [], []
    for year in years:
        scores = load_year_scores(year)
        scores["Team A Score"] = pd.to_numeric(scores["Team A Score"], errors="coerce")
        scores["Team B Score"] = pd.to_numeric(scores["Team B Score"], errors="coerce")
        scores_list.append(scores)

        overview = load_year_overview(year)
        overview = clean_overview(overview)
        stats_list.append(aggregate_team_stats(overview))

    scores_all = pd.concat(scores_list, ignore_index=True)
    stats_all = pd.concat(stats_list, ignore_index=True)

    merge_keys = ["Tournament", "Stage", "Match Type", "Match Name", "Year"]

    # 分别重命名 stats 中的 Team 和特征列，避免列冲突
    stats_all_renamed = stats_all.rename(columns={"Team": "tmp_team"})
    stats_a = stats_all_renamed.rename(
        columns={c: c.replace("team_", "team_a_") if c.startswith("team_") else c
                 for c in stats_all_renamed.columns if c != "tmp_team"}
    )
    stats_b = stats_all_renamed.rename(
        columns={c: c.replace("team_", "team_b_") if c.startswith("team_") else c
                 for c in stats_all_renamed.columns if c != "tmp_team"}
    )

    merged = scores_all.merge(
        stats_a,
        left_on=merge_keys + ["Team A"],
        right_on=merge_keys + ["tmp_team"],
        how="left",
    )
    merged = merged.drop(columns=["tmp_team"])

    merged = merged.merge(
        stats_b,
        left_on=merge_keys + ["Team B"],
        right_on=merge_keys + ["tmp_team"],
        how="left",
        suffixes=("", "_dup"),
    )
    merged = merged.drop(columns=["tmp_team"])

    dup_cols = [c for c in merged.columns if c.endswith("_dup")]
    merged = merged.drop(columns=dup_cols)

    merged["target"] = (merged["Team A Score"] > merged["Team B Score"]).astype(int)
    merged["total_score"] = merged["Team A Score"] + merged["Team B Score"]
    return merged


def build_draft_dataset(years: Optional[list[int]] = None) -> pd.DataFrame:
    if years is None:
        years = YEARS
    return pd.concat([load_year_draft_phase(y) for y in years], ignore_index=True)


def run_pipeline(years: Optional[list[int]] = None):
    if years is None:
        years = YEARS
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Building match dataset...")
    match_data = build_match_dataset(years)
    match_data.to_parquet(PROCESSED_DIR / "match_dataset.parquet", index=False)
    print(f"  → {len(match_data)} matches, {len(match_data.columns)} columns")

    print("Building BP dataset...")
    draft_data = build_draft_dataset(years)
    draft_data.to_parquet(PROCESSED_DIR / "draft_dataset.parquet", index=False)
    print(f"  → {len(draft_data)} draft records")

    print("Done.")


if __name__ == "__main__":
    run_pipeline()
