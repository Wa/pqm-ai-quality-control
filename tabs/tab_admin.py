import streamlit as st


def render_admin_tab(session_id):
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return

    st.subheader("🛡️ 管理员面板")
    st.info("仅管理员可见。你可以在此放置管理功能。")






