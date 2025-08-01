import streamlit as st
import pandas as pd
from config import CONFIG

def render_history_issues_avoidance_tab(session_id):
    # Handle None session_id (user not logged in)
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return
    
    st.title("历史问题规避")
    st.info("此功能暂未开放。下方为占位符内容，后续将替换为实际历史问题规避数据。")
    # Display a subset of the Excel sheet as a placeholder
    excel_path = str(CONFIG["files"]["history_excel"])
    try:
        df = pd.read_excel(excel_path, sheet_name="乘用车", usecols="A:V", nrows=20)
        
        # Fix DataFrame serialization issues by converting all data to strings
        # This prevents PyArrow conversion errors with mixed data types
        df = df.astype(str)
        
        # Replace NaN values with empty strings
        df = df.fillna('')
        
        st.dataframe(df)
    except Exception as e:
        st.warning(f"无法加载Excel内容: {e}") 