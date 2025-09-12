import streamlit as st
from tab_special_symbols_check import render_special_symbols_check_tab
from tab_parameters_check import render_parameters_check_tab
from tab_file_elements_check import render_file_elements_check_tab
from tab_file_completeness_check import render_file_completeness_check_tab
from tab_enterprise_standard_check import render_enterprise_standard_check_tab
from tab_history_issues_avoidance import render_history_issues_avoidance_tab
from tab_settings import render_settings_tab
from tab_ai_agent import render_ai_agent_tab
from tab_help_documentation import render_help_documentation_tab
from tab_home import render_home_tab
from tab_admin import render_admin_tab
from util import render_login_widget, get_user_session_id
from util import is_admin

st.set_page_config(layout="wide")

# Login system
username = render_login_widget()

# Only show main app if user is logged in
if username:
    # Generate session ID based on username for persistence
    session_id = get_user_session_id(username)
    
    # Labels and conditional admin tab
    HOME = "🏠 首页"
    SPECIAL = "🔍 特殊特性符号检查"
    PARAMETERS = "📊 设计制程检查"
    ELEMENTS = "✅ 文件要素检查"
    COMPLETE = "📁 文件齐套性检查"
    ENTERPRISE = "🏢 企业标准检查"
    HISTORY = "📋 历史问题规避"
    AI = "🤖 AI智能体"
    SETTINGS = "⚙️ 设置"
    HELP = "📖 帮助文档"
    ADMIN = "🛡️ 网站管理"

    tab_labels = [HOME, COMPLETE, SPECIAL, PARAMETERS, ELEMENTS, ENTERPRISE, HISTORY, AI, SETTINGS, HELP]
    if is_admin(username):
        tab_labels.insert(8, ADMIN)

    tabs = st.tabs(tab_labels)
    idx = {label: i for i, label in enumerate(tab_labels)}

    with tabs[idx[HOME]]:
        render_home_tab(session_id)
    with tabs[idx[COMPLETE]]:
        render_file_completeness_check_tab(session_id)
    with tabs[idx[SPECIAL]]:
        render_special_symbols_check_tab(session_id)
    with tabs[idx[PARAMETERS]]:
        render_parameters_check_tab(session_id)
    with tabs[idx[ELEMENTS]]:
        render_file_elements_check_tab(session_id)
    with tabs[idx[ENTERPRISE]]:
        render_enterprise_standard_check_tab(session_id)
    with tabs[idx[HISTORY]]:
        render_history_issues_avoidance_tab(session_id)
    with tabs[idx[AI]]:
        render_ai_agent_tab(session_id)
    if ADMIN in idx:
        with tabs[idx[ADMIN]]:
            render_admin_tab(session_id)
    with tabs[idx[SETTINGS]]:
        render_settings_tab(session_id)
    with tabs[idx[HELP]]:
        render_help_documentation_tab(session_id)
else:
    pass 