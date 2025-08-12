import streamlit as st
from tab_special_symbols_check import render_special_symbols_check_tab
from tab_parameters_check import render_parameters_check_tab
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
    
    # Login-related UI moved to Settings tab
    
    # Main app tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔍 特殊符号检查", "📊 参数检查", "📁 文件要素检查", 
        "✅ 文件完整性检查", "📚 历史问题规避", "⚙️ 设置", "❓ 帮助文档"
    ])

    with tab1:
        render_special_symbols_check_tab(session_id)
    with tab2:
        render_parameters_check_tab(session_id)
    with tab3:
        render_file_elements_check_tab(session_id)
    with tab4:
        render_file_completeness_check_tab(session_id)
    with tab5:
        render_history_issues_avoidance_tab(session_id)
    with tab6:
        render_settings_tab(session_id)
    with tab7:
        render_help_documentation_tab(session_id)
else:
    pass 