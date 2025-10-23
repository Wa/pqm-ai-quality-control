"""Placeholder tab for special symbols workflow rebuild."""

import streamlit as st


def render_special_symbols_check_tab(session_id):
    """Temporary placeholder view while the special symbols workflow is rebuilt."""
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return

    st.info("特殊特性符号检查功能正在升级，稍后会提供全新的工作流体验。")
