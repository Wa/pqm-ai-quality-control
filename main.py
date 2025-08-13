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
    
    # Main app header with logout button
    st.title("🤖 PQM AI 质量控制系统")
    
    # Show active users (multi-user support)
    from util import get_active_users
    active_users = get_active_users()
    if len(active_users) > 1:
        st.info(f"👥 当前在线用户: {', '.join(active_users)}")
    
    # Simple logout button
    if st.button("🚪 退出登录", type="secondary", key="logout_button"):
        st.write("🔍 退出登录按钮被点击，正在执行登出...")  # Debug message
        # Deactivate user session for multi-user support
        current_username = st.session_state.get('username')
        if current_username:
            from util import deactivate_user_session
            deactivate_user_session(current_username)
            st.write(f"✅ 已停用用户会话: {current_username}")  # Debug
        
        # Clear all session state immediately
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.write("✅ 会话状态已清除，正在重新加载...")  # Debug message
        st.rerun()
    else:
        st.write("🔍 退出登录按钮未被点击")  # Debug: button not clicked
    
    st.divider()
    
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