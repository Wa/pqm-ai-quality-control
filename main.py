import streamlit as st
from tab_consistency_check import render_consistency_check_tab
from tab_file_elements_check import render_file_elements_check_tab
from tab_file_completeness_check import render_file_completeness_check_tab
from tab_history_issues_avoidance import render_history_issues_avoidance_tab
from tab_settings import render_settings_tab
from tab_help_documentation import render_help_documentation_tab
from util import render_login_widget, get_user_session_id

st.set_page_config(layout="wide")

# Login system
username = render_login_widget()

# Only show main app if user is logged in
if username:
    # Generate session ID based on username for persistence
    session_id = get_user_session_id(username)
    
    # Main application tabs
    一致性审查_tab, 文件要素审查_tab, 文件齐套性审查_tab, 历史问题规避_tab, 设置_tab, 帮助文档_tab = st.tabs(["一致性审查", "文件要素审查", "文件齐套性审查", "历史问题规避", "设置", "帮助文档"])

    with 一致性审查_tab:
        render_consistency_check_tab(session_id)
    with 文件要素审查_tab:
        render_file_elements_check_tab(session_id)
    with 文件齐套性审查_tab:
        render_file_completeness_check_tab(session_id)
    with 历史问题规避_tab:
        render_history_issues_avoidance_tab(session_id)
    with 设置_tab:
        render_settings_tab(session_id)
    with 帮助文档_tab:
        render_help_documentation_tab(session_id)
else:
    # User is not logged in - login widget is already shown by render_login_widget()
    # No need to show additional message since the login form is already displayed
    pass 