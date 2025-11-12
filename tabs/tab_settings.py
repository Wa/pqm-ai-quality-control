import streamlit as st
import requests
from config import CONFIG

@st.cache_data(ttl=30)  # Cache for 30 seconds
def test_ollama_connection(host: str):
    """Test connection to the specified Ollama server."""
    try:
        response = requests.get(f"{host}/api/tags", timeout=3)
        return response.status_code == 200
    except Exception:
        return False

@st.cache_data(ttl=30)  # Cache for 30 seconds
def test_openai_connection():
    """Test connection to OpenAI API."""
    try:
        headers = {
            "Authorization": f"Bearer {CONFIG['llm']['openai_api_key']}",
            "Content-Type": "application/json"
        }
        response = requests.get(
            f"{CONFIG['llm']['openai_base_url']}/models",
            headers=headers,
            timeout=5
        )
        return response.status_code == 200
    except Exception:
        return False

def render_settings_tab(session_id):
    """Render the settings tab using native Streamlit components."""
    
    # Handle None session_id (user not logged in)
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return
    
    # Create a container with constrained width for the settings page
    with st.container(key="settings-container"):
        # Add CSS to constrain the width of this specific container
        st.markdown("""
        <style>
        .st-key-settings-container {
            max-width: 800px !important;
            margin: 0 auto !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # User section: current user, active users, logout, and clear saved username
        st.header("👤 用户与会话")
        st.write(f"**当前用户:** {session_id}")

        cols = st.columns(1)
        with cols[0]:
            if st.button("🚪 退出登录", key=f"logout_btn_{session_id}"):
                try:
                    from util import deactivate_user_session
                    deactivate_user_session(session_id)
                except Exception:
                    pass
                # Clear session state
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                # Keep 'user' to remember username on this PC; remove only auth flag
                try:
                    if "auth" in st.query_params:
                        del st.query_params["auth"]
                except Exception:
                    pass
                st.rerun()

        st.divider()
        
        # Connection Status
        st.header("🔗 连接状态")
        ollama_host = CONFIG['llm']['ollama_host'].replace("10.31.60.127", "10.31.60.9")
        ollama_display = "Ollama (10.31.60.9:11434)"
        
        col1, col2 = st.columns(2)
        with col1:
            if test_ollama_connection(ollama_host):
                st.success(f"✅ {ollama_display} 在线")
            else:
                st.error(f"❌ {ollama_display} 无法连接")
        
        with col2:
            if CONFIG['llm'].get("openai_api_key"):
                if test_openai_connection():
                    st.success("✅ OpenAI API 在线")
                else:
                    st.error("❌ OpenAI API 无法连接")
            else:
                st.info("OpenAI API 未配置")

        st.divider()
        # Documentation Links
        st.header("📚 相关文档")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Ollama文档")
            st.link_button("GitHub Repository", "https://github.com/ollama/ollama")
            st.link_button("API文档", "https://github.com/ollama/ollama/blob/main/docs/api.md")
            st.link_button("模型参数", "https://github.com/ollama/ollama/blob/main/docs/modelfile.md")
        
        with col2:
            st.subheader("OpenAI文档")
            st.link_button("API文档", "https://platform.openai.com/docs/api-reference")
            st.link_button("模型参数", "https://platform.openai.com/docs/api-reference/chat/create")
            st.link_button("UIUIApi", "https://sg.uiuiapi.com/") 
