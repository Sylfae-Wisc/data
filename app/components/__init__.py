"""VCT 组件包 — 共享工具与常量"""

from pathlib import Path
from typing import Optional
import streamlit as st
import pandas as pd

DATA_DIR = Path("data")
FEATURES_DIR = DATA_DIR / "features"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("models")


@st.cache_resource
def get_teams() -> list[str]:
    """获取所有战队名称列表（缓存）"""
    df = pd.read_parquet(FEATURES_DIR / "match_features.parquet")
    teams = sorted(set(df["Team A"]).union(set(df["Team B"])))
    return teams


@st.cache_resource
def get_team_mapping() -> dict[str, str]:
    """获取战队简称→全称映射"""
    try:
        mapping = pd.read_csv(DATA_DIR / "raw" / "all_ids" / "all_teams_mapping.csv")
        return dict(zip(mapping["Abbreviated"], mapping["Full Name"]))
    except FileNotFoundError:
        return {}


@st.cache_resource
def get_team_options() -> list[str]:
    """获取带完整名称的选项列表（用于 selectbox）"""
    mapping = get_team_mapping()
    teams = get_teams()
    # 仅保留映射中有全称的队伍（核心战队）
    mapped = {k: v for k, v in mapping.items() if k in set(teams)}
    # 格式: "FNC — FNATIC"
    options = sorted(f"{k} — {v}" for k, v in mapped.items())
    if not options:
        options = sorted(teams)
    return options


def parse_team_option(option: str) -> str:
    """从选项文本解析出简称"""
    if " — " in option:
        return option.split(" — ")[0]
    return option
