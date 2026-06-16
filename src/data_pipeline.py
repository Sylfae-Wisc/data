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
MATCH_DIR = "matches"
SCORES_FILE = "scores.csv"
OVERVIEW_FILE = "overview.csv"
DRAFT_PHASE_FILE = "draft_phase.csv"

MATCH_ID_COLUMNS = [
    "Tournament",
    "Stage",
    "Match Type",
    "Match Name",
    "Year",
]

TEAM_STATS_ID_COLUMNS = MATCH_ID_COLUMNS[:4] + ["Team", "Year"]

SCORE_COLUMNS = [
    "Team A Score",
    "Team B Score",
]

OVERVIEW_RENAME_MAP = {
    "Average Combat Score": "ACS",
    "Kills - Deaths (KD)": "KD",
    "Kill, Assist, Trade, Survive %": "KAST",
    "Average Damage Per Round": "ADR",
    "Headshot %": "HS_Pct",
    "First Kills": "FK",
    "First Deaths": "FD",
    "Kills - Deaths (FKD)": "FKD",
}

PERCENTAGE_COLUMNS = [
    "KAST",
    "HS_Pct",
]

NUMERIC_OVERVIEW_COLUMNS = [
    "Rating",
    "ACS",
    "Kills",
    "Deaths",
    "Assists",
    "KD",
    "ADR",
    "FK",
    "FD",
    "FKD",
]

TEAM_STATS_AGGREGATIONS = {
    "ACS": "mean",
    "KAST": "mean",
    "ADR": "mean",
    "HS_Pct": "mean",
    "Rating": "mean",
    "Kills": "sum",
    "Deaths": "sum",
    "Assists": "sum",
    "KD": "mean",
    "FK": "sum",
    "FD": "sum",
}

SIDE_TO_TEAM_COLUMN = {
    "a": "Team A",
    "b": "Team B",
}


def _year_match_file(year: int, filename: str) -> Path:
    return RAW_DIR / f"vct_{year}" / MATCH_DIR / filename


def _load_year_csv(year: int, filename: str) -> pd.DataFrame:
    df = pd.read_csv(_year_match_file(year, filename), low_memory=False)
    df["Year"] = year
    return df


def _coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _available_aggregations(df: pd.DataFrame) -> dict:
    return {
        column: aggregation
        for column, aggregation in TEAM_STATS_AGGREGATIONS.items()
        if column in df.columns
    }


def _rename_team_stat_columns(stats: pd.DataFrame, side: str) -> pd.DataFrame:
    renamed = stats.rename(columns={"Team": "tmp_team"})
    prefix = f"team_{side}_"
    return renamed.rename(
        columns={
            column: column.replace("team_", prefix)
            if column.startswith("team_")
            else column
            for column in renamed.columns
            if column != "tmp_team"
        }
    )


def _merge_team_stats(
    matches: pd.DataFrame,
    stats: pd.DataFrame,
    side: str,
) -> pd.DataFrame:
    team_column = SIDE_TO_TEAM_COLUMN[side]
    side_stats = _rename_team_stat_columns(stats, side)

    merged = matches.merge(
        side_stats,
        left_on=MATCH_ID_COLUMNS + [team_column],
        right_on=MATCH_ID_COLUMNS + ["tmp_team"],
        how="left",
        suffixes=("", "_dup"),
    )
    merged = merged.drop(columns=["tmp_team"])

    duplicate_columns = [
        column
        for column in merged.columns
        if column.endswith("_dup")
    ]
    if duplicate_columns:
        merged = merged.drop(columns=duplicate_columns)

    return merged


def load_year_scores(year: int) -> pd.DataFrame:
    return _load_year_csv(year, SCORES_FILE)


def load_year_overview(year: int) -> pd.DataFrame:
    return _load_year_csv(year, OVERVIEW_FILE)


def load_year_draft_phase(year: int) -> pd.DataFrame:
    return _load_year_csv(year, DRAFT_PHASE_FILE)


def _parse_pct(series: pd.Series) -> pd.Series:
    """将 '78%' → 0.78"""
    return pd.to_numeric(series.astype(str).str.rstrip("%"), errors="coerce") / 100.0


def clean_overview(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=OVERVIEW_RENAME_MAP)

    # 百分比字段需先去除 % 再转数值
    for col in PERCENTAGE_COLUMNS:
        if col in df.columns:
            df[col] = _parse_pct(df[col])

    # Rating 可能有 "1.23" 之类，直接转
    return _coerce_numeric_columns(df, NUMERIC_OVERVIEW_COLUMNS)


def aggregate_team_stats(overview: pd.DataFrame) -> pd.DataFrame:
    """将 overview 从 player-match-map 级聚合到 match-team 级（跨地图平均）"""
    team_stats = overview.groupby(
        TEAM_STATS_ID_COLUMNS,
        as_index=False,
    ).agg(_available_aggregations(overview))
    team_stats.columns = [
        column
        if column in TEAM_STATS_ID_COLUMNS
        else f"team_{column.lower()}"
        for column in team_stats.columns
    ]
    return team_stats


def build_match_dataset(years: Optional[list[int]] = None) -> pd.DataFrame:
    if years is None:
        years = YEARS

    scores_list, stats_list = [], []
    for year in years:
        scores = load_year_scores(year)
        scores = _coerce_numeric_columns(scores, SCORE_COLUMNS)
        scores_list.append(scores)

        overview = load_year_overview(year)
        overview = clean_overview(overview)
        stats_list.append(aggregate_team_stats(overview))

    scores_all = pd.concat(scores_list, ignore_index=True)
    stats_all = pd.concat(stats_list, ignore_index=True)

    merged = _merge_team_stats(scores_all, stats_all, "a")
    merged = _merge_team_stats(merged, stats_all, "b")

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
