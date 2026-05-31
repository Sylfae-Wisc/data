"""可复用组件：战队搜索选择器"""

from typing import Optional
import streamlit as st
from app.components import get_team_options, parse_team_option


def select_team(
    label: str = "选择战队",
    key: str = "team",
    index: Optional[int] = None,
    placeholder: str = "搜索战队名称…",
) -> str:
    """搜索式战队选择器

    Args:
        label: 选择框标签
        key: Streamlit widget key
        index: 默认选中索引
        placeholder: 搜索占位文本

    Returns:
        选择的战队简称（如 "FNATIC" → "FNC"）
    """
    options = get_team_options()

    selected = st.selectbox(
        label,
        options=options,
        index=index,
        key=key,
        placeholder=placeholder,
    )

    if selected:
        return parse_team_option(selected)
    return ""
