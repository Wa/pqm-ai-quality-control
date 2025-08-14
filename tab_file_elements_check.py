import streamlit as st
from config import CONFIG

def render_file_elements_check_tab(session_id):
    # Handle None session_id (user not logged in)
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return
    
    # Page subheader
    st.subheader("✅ 文件要素检查")