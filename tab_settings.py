import streamlit as st
import requests
import json
import os
from pathlib import Path
from config import CONFIG

def get_user_settings_file(session_id):
    """Get the path to the user's settings file."""
    settings_dir = Path("user_settings")
    settings_dir.mkdir(exist_ok=True)
    return settings_dir / f"user_{session_id}_settings.json"

def save_user_settings(session_id, settings):
    """Save user settings to a JSON file."""
    try:
        settings_file = get_user_settings_file(session_id)
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"保存设置失败: {e}")
        return False

def load_user_settings(session_id):
    """Load user settings from JSON file."""
    try:
        settings_file = get_user_settings_file(session_id)
        if settings_file.exists():
            with open(settings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Return default settings if file doesn't exist
            return {
                'llm_backend': 'ollama',
                'ollama_model': CONFIG['llm']['ollama_model'],
                'openai_model': CONFIG['llm']['openai_model'],
                'ollama_temperature': 0.7,
                'ollama_top_p': 0.9,
                'ollama_top_k': 40,
                'ollama_repeat_penalty': 1.1,
                'openai_temperature': 0.7,
                'openai_top_p': 1.0,
                'openai_max_tokens': 2048,
                'openai_presence_penalty': 0.0,
                'openai_frequency_penalty': 0.0
            }
    except Exception as e:
        st.error(f"加载设置失败: {e}")
        # Return default settings on error
        return {
            'llm_backend': 'ollama',
            'ollama_model': CONFIG['llm']['ollama_model'],
            'openai_model': CONFIG['llm']['openai_model'],
            'ollama_temperature': 0.7,
            'ollama_top_p': 0.9,
            'ollama_top_k': 40,
            'ollama_repeat_penalty': 1.1,
            'openai_temperature': 0.7,
            'openai_top_p': 1.0,
            'openai_max_tokens': 2048,
            'openai_presence_penalty': 0.0,
            'openai_frequency_penalty': 0.0
        }

def save_current_settings(session_id):
    """Save all current session state settings to file."""
    current_settings = {
        'llm_backend': st.session_state.get(f'llm_backend_{session_id}', 'ollama'),
        'ollama_model': st.session_state.get(f'ollama_model_{session_id}', CONFIG['llm']['ollama_model']),
        'openai_model': st.session_state.get(f'openai_model_{session_id}', CONFIG['llm']['openai_model']),
        'ollama_temperature': st.session_state.get(f'ollama_temperature_{session_id}', 0.7),
        'ollama_top_p': st.session_state.get(f'ollama_top_p_{session_id}', 0.9),
        'ollama_top_k': st.session_state.get(f'ollama_top_k_{session_id}', 40),
        'ollama_repeat_penalty': st.session_state.get(f'ollama_repeat_penalty_{session_id}', 1.1),
        'openai_temperature': st.session_state.get(f'openai_temperature_{session_id}', 0.7),
        'openai_top_p': st.session_state.get(f'openai_top_p_{session_id}', 1.0),
        'openai_max_tokens': st.session_state.get(f'openai_max_tokens_{session_id}', 2048),
        'openai_presence_penalty': st.session_state.get(f'openai_presence_penalty_{session_id}', 0.0),
        'openai_frequency_penalty': st.session_state.get(f'openai_frequency_penalty_{session_id}', 0.0)
    }
    return save_user_settings(session_id, current_settings)

@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_ollama_models():
    """Get available Ollama models from the local server."""
    try:
        response = requests.get(f"{CONFIG['llm']['ollama_host']}/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        else:
            return []
    except Exception as e:
        st.warning(f"无法连接到Ollama服务器: {e}")
        return []

@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_ollama_model_info(model_name):
    """Get detailed information about a specific Ollama model."""
    try:
        response = requests.post(
            f"{CONFIG['llm']['ollama_host']}/api/show",
            json={"name": model_name},
            timeout=3
        )
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception:
        return None

@st.cache_data(ttl=30)  # Cache for 30 seconds
def test_ollama_connection():
    """Test connection to Ollama server."""
    try:
        response = requests.get(f"{CONFIG['llm']['ollama_host']}/api/tags", timeout=3)
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

@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_openai_models():
    """Get available OpenAI models."""
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
        if response.status_code == 200:
            models = response.json().get('data', [])
            # Filter for chat completion models
            chat_models = [model['id'] for model in models if 'gpt' in model['id'].lower()]
            return chat_models
        else:
            return []
    except Exception as e:
        st.warning(f"无法连接到OpenAI API: {e}")
        return []

def render_settings_tab(session_id):
    """Render the settings tab using native Streamlit components."""
    
    # Handle None session_id (user not logged in)
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return
    
    # Load user settings from file (persistent across restarts)
    user_settings = load_user_settings(session_id)
    
    # Initialize session state from file settings
    if f'llm_backend_{session_id}' not in st.session_state:
        st.session_state[f'llm_backend_{session_id}'] = user_settings['llm_backend']
    if f'ollama_model_{session_id}' not in st.session_state:
        st.session_state[f'ollama_model_{session_id}'] = user_settings['ollama_model']
    if f'openai_model_{session_id}' not in st.session_state:
        st.session_state[f'openai_model_{session_id}'] = user_settings['openai_model']
    
    # Initialize other parameters from file settings
    for param, value in user_settings.items():
        if param not in ['llm_backend', 'ollama_model', 'openai_model']:
            session_key = f'{param}_{session_id}'
            if session_key not in st.session_state:
                st.session_state[session_key] = value
    
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
        
        # Page header
        st.title("设置")
        st.caption("配置大语言模型参数和连接设置")
        st.divider()
        
        # LLM Backend Selection
        st.header("🤖 大语言模型选择")
        
        llm_options = {
            "Ollama (local)": "ollama",
            "OpenAI (sg.uiuiapi.com)": "openai"
        }
        
        # Get current LLM choice from session state
        current_llm_backend = st.session_state.get(f'llm_backend_{session_id}', 'ollama')
        
        # Find the display name for current backend
        current_display_name = None
        for display_name, backend in llm_options.items():
            if backend == current_llm_backend:
                current_display_name = display_name
                break
        
        # Create the selectbox with current selection
        selected_display_name = st.selectbox(
            "选择大语言模型", 
            list(llm_options.keys()), 
            index=list(llm_options.keys()).index(current_display_name) if current_display_name else 0,
            key=f"settings_llm_select_{session_id}"
        )
        
        # Get the backend value from the selection
        selected_backend = llm_options[selected_display_name]
        
        # Update session state when selection changes (this is appropriate since it's not a widget value)
        if selected_backend != current_llm_backend:
            st.session_state[f'llm_backend_{session_id}'] = selected_backend
            st.success(f"已切换到: {selected_display_name}")
            
            # Save settings to file for persistence
            save_current_settings(session_id)
        
        st.divider()
        
        # Connection Status
        st.header("🔗 连接状态")
        
        col1, col2 = st.columns(2)
        with col1:
            if selected_backend == "ollama":
                if test_ollama_connection():
                    st.success("✅ Ollama服务器连接正常")
                else:
                    st.error("❌ 无法连接到Ollama服务器")
            elif selected_backend == "openai":
                if test_openai_connection():
                    st.success("✅ OpenAI API连接正常")
                else:
                    st.error("❌ 无法连接到OpenAI API")
        
        with col2:
            st.info(f"""
            **当前后端:** {selected_display_name}  
            **状态:** {'在线' if (selected_backend == "ollama" and test_ollama_connection()) or (selected_backend == "openai" and test_openai_connection()) else '离线'}
            """)
        
        st.divider()
        
        # Model Configuration
        st.header("⚙️ 模型配置")
        
        if selected_backend == "ollama":
            # Initialize selected_model variable from session state
            selected_model = st.session_state.get(f'ollama_model_{session_id}', CONFIG['llm']['ollama_model'])
            
            # Ollama Model Selection
            with st.spinner("正在获取可用模型列表..."):
                available_models = get_ollama_models()
            
            if available_models:
                current_model = st.session_state.get(f'ollama_model_{session_id}', CONFIG['llm']['ollama_model'])
                
                # Ensure current model is in the list, otherwise use first available
                if current_model not in available_models:
                    current_model = available_models[0]
                    st.session_state[f'ollama_model_{session_id}'] = current_model
                    selected_model = current_model
                
                selected_model = st.selectbox(
                    "选择Ollama模型",
                    available_models,
                    index=available_models.index(current_model),
                    key=f"ollama_model_select_{session_id}"
                )
                
                # Update the model session state when selection changes
                if selected_model != current_model:
                    st.session_state[f'ollama_model_{session_id}'] = selected_model
                    st.success(f"✅ 已切换到: {selected_model} (将在下次运行时生效)")
                    
                    # Save settings to file for persistence
                    save_current_settings(session_id)
                    
                    # No st.rerun() needed - settings are saved and will apply to future runs
                    # This prevents interrupting any currently running analysis
            
            # Model Information - with error handling
            try:
                model_info = get_ollama_model_info(selected_model)
                if model_info:
                    with st.expander("模型详细信息", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**模型名称:** {model_info.get('name', 'N/A')}")
                            st.write(f"**模型大小:** {model_info.get('size', 'N/A')} bytes")
                            st.write(f"**修改时间:** {model_info.get('modified_at', 'N/A')}")
                        with col2:
                            st.write(f"**参数数量:** {model_info.get('parameter_size', 'N/A')}")
                            st.write(f"**量化级别:** {model_info.get('quantization_level', 'N/A')}")
            except Exception as e:
                st.warning(f"无法获取模型详细信息: {e}")
            else:
                st.warning("无法获取可用模型列表，请检查Ollama服务器连接")
                # Don't override the user's selection when model list can't be fetched
                # Keep the current selected_model value
            
            # Ollama Parameters
            st.subheader("Ollama参数设置")
            
            col1, col2 = st.columns(2)
            with col1:
                temperature = st.slider(
                    "Temperature (温度)",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.get(f'ollama_temperature_{session_id}', 0.7),
                    step=0.1,
                    help="控制输出的随机性。较低的值产生更确定性的输出，较高的值产生更创造性的输出。",
                    key=f"ollama_temperature_{session_id}"
                )
                
                top_p = st.slider(
                    "Top-p (核采样)",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get(f'ollama_top_p_{session_id}', 0.9),
                    step=0.1,
                    help="控制词汇选择的多样性。",
                    key=f"ollama_top_p_{session_id}"
                )
            
            with col2:
                top_k = st.slider(
                    "Top-k",
                    min_value=1,
                    max_value=100,
                    value=st.session_state.get(f'ollama_top_k_{session_id}', 40),
                    step=1,
                    help="限制每次选择时考虑的词汇数量。",
                    key=f"ollama_top_k_{session_id}"
                )
                
                repeat_penalty = st.slider(
                    "Repeat Penalty (重复惩罚)",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.get(f'ollama_repeat_penalty_{session_id}', 1.1),
                    step=0.1,
                    help="减少重复内容的生成。",
                    key=f"ollama_repeat_penalty_{session_id}"
                )
            
            # Advanced Ollama Settings
            with st.expander("高级设置", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    num_ctx = st.number_input(
                        "上下文窗口大小",
                        min_value=512,
                        max_value=8192,
                        value=st.session_state.get(f'ollama_num_ctx_{session_id}', 4096),
                        step=512,
                        help="模型可以处理的最大上下文长度。",
                        key=f"ollama_num_ctx_{session_id}"
                    )
                
                with col2:
                    num_thread = st.number_input(
                        "线程数",
                        min_value=1,
                        max_value=16,
                        value=st.session_state.get(f'ollama_num_thread_{session_id}', 4),
                        step=1,
                        help="用于推理的CPU线程数。",
                        key=f"ollama_num_thread_{session_id}"
                    )
        
        elif selected_backend == "openai":
            # OpenAI Model Selection
            available_models = get_openai_models()
            if available_models:
                current_model = st.session_state.get(f'openai_model_{session_id}', CONFIG['llm']['openai_model'])
                
                selected_model = st.selectbox(
                    "选择OpenAI模型",
                    available_models,
                    index=available_models.index(current_model) if current_model in available_models else 0,
                    key=f"openai_model_select_{session_id}"
                )
                
                # Update the model session state when selection changes
                if selected_model != current_model:
                    st.session_state[f'openai_model_{session_id}'] = selected_model
                    st.success(f"已切换到: {selected_model}")
            else:
                st.warning("无法获取可用模型列表")
                st.session_state[f'openai_model_{session_id}'] = CONFIG['llm']['openai_model']
            
            # OpenAI Parameters
            st.subheader("OpenAI参数设置")
            
            col1, col2 = st.columns(2)
            with col1:
                temperature = st.slider(
                    "Temperature (温度)",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.get(f'openai_temperature_{session_id}', 0.7),
                    step=0.1,
                    help="控制输出的随机性。0表示完全确定性，2表示最大随机性。",
                    key=f"openai_temperature_{session_id}"
                )
                
                top_p = st.slider(
                    "Top-p (核采样)",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get(f'openai_top_p_{session_id}', 1.0),
                    step=0.1,
                    help="控制词汇选择的多样性。",
                    key=f"openai_top_p_{session_id}"
                )
            
            with col2:
                max_tokens = st.number_input(
                    "最大输出长度",
                    min_value=1,
                    max_value=4096,
                    value=st.session_state.get(f'openai_max_tokens_{session_id}', 2048),
                    step=1,
                    help="生成响应的最大token数量。",
                    key=f"openai_max_tokens_{session_id}"
                )
                
                presence_penalty = st.slider(
                    "Presence Penalty (存在惩罚)",
                    min_value=-2.0,
                    max_value=2.0,
                    value=st.session_state.get(f'openai_presence_penalty_{session_id}', 0.0),
                    step=0.1,
                    help="减少模型重复相同主题的倾向。",
                    key=f"openai_presence_penalty_{session_id}"
                )
            
            # Advanced OpenAI Settings
            with st.expander("高级设置", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    frequency_penalty = st.slider(
                        "Frequency Penalty (频率惩罚)",
                        min_value=-2.0,
                        max_value=2.0,
                        value=st.session_state.get(f'openai_frequency_penalty_{session_id}', 0.0),
                        step=0.1,
                        help="减少模型重复相同词汇的倾向。",
                        key=f"openai_frequency_penalty_{session_id}"
                    )
                
                with col2:
                    logit_bias = st.text_input(
                        "Logit Bias (词汇偏好)",
                        value=st.session_state.get(f'openai_logit_bias_{session_id}', '{}'),
                        help="JSON格式的词汇偏好设置，例如: {\"word\": 0.5}",
                        key=f"openai_logit_bias_{session_id}"
                    )
        
        st.divider()
        
        # Current Configuration Overview
        st.header("📋 当前配置概览")
        
        if selected_backend == "ollama":
            # Display configuration in a more compact format
            st.write("**后端:**", selected_display_name)
            st.write("**主机:**", CONFIG['llm']['ollama_host'])
            st.write("**当前模型:**", selected_model)
            st.write("**Temperature:**", st.session_state.get(f'ollama_temperature_{session_id}', 0.7))
            st.write("**Top-p:**", st.session_state.get(f'ollama_top_p_{session_id}', 0.9))
            st.write("**Top-k:**", st.session_state.get(f'ollama_top_k_{session_id}', 40))
            
            with st.expander("完整配置JSON", expanded=False):
                config_data = {
                    "backend": "ollama",
                    "host": CONFIG['llm']['ollama_host'],
                    "model": selected_model,
                    "temperature": st.session_state.get(f'ollama_temperature_{session_id}', 0.7),
                    "top_p": st.session_state.get(f'ollama_top_p_{session_id}', 0.9),
                    "top_k": st.session_state.get(f'ollama_top_k_{session_id}', 40),
                    "repeat_penalty": st.session_state.get(f'ollama_repeat_penalty_{session_id}', 1.1),
                    "num_ctx": st.session_state.get(f'ollama_num_ctx_{session_id}', 4096),
                    "num_thread": st.session_state.get(f'ollama_num_thread_{session_id}', 4)
                }
                st.json(config_data)
        
        elif selected_backend == "openai":
            # Display configuration in a more compact format
            st.write("**后端:**", selected_display_name)
            st.write("**API地址:**", CONFIG['llm']['openai_base_url'])
            st.write("**当前模型:**", st.session_state.get(f'openai_model_{session_id}', CONFIG['llm']['openai_model']))
            st.write("**Temperature:**", st.session_state.get(f'openai_temperature_{session_id}', 0.7))
            st.write("**Top-p:**", st.session_state.get(f'openai_top_p_{session_id}', 1.0))
            st.write("**Max Tokens:**", st.session_state.get(f'openai_max_tokens_{session_id}', 2048))
            
            with st.expander("完整配置JSON", expanded=False):
                config_data = {
                    "backend": "openai",
                    "base_url": CONFIG['llm']['openai_base_url'],
                    "model": st.session_state.get(f'openai_model_{session_id}', CONFIG['llm']['openai_model']),
                    "temperature": st.session_state.get(f'openai_temperature_{session_id}', 0.7),
                    "top_p": st.session_state.get(f'openai_top_p_{session_id}', 1.0),
                    "max_tokens": st.session_state.get(f'openai_max_tokens_{session_id}', 2048),
                    "presence_penalty": st.session_state.get(f'openai_presence_penalty_{session_id}', 0.0),
                    "frequency_penalty": st.session_state.get(f'openai_frequency_penalty_{session_id}', 0.0),
                    "logit_bias": st.session_state.get(f'openai_logit_bias_{session_id}', '{}')
                }
                st.json(config_data)
        
        st.divider()
        
        # User Account Management
        st.header("👤 用户账户")
        
        # Get current username from session state
        current_username = st.session_state.get('username', 'Unknown')
        
        st.write(f"**当前用户:** {current_username}")
        st.write("**会话ID:**", session_id)
        
        # Logout button
        if st.button("🚪 退出登录", key=f"logout_button_{session_id}", type="secondary"):
            st.session_state['logged_in'] = False
            st.session_state['username'] = None
            st.session_state['user_session_id'] = None
            st.success("✅ 已退出登录，正在返回登录页面...")
            st.rerun()  # Necessary to return to login screen
        
        st.divider()
        
        # Action Buttons
        st.header("⚡ 快速操作")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("重置为默认设置", key=f"reset_settings_{session_id}"):
                # Reset Ollama settings
                st.session_state[f'ollama_model_{session_id}'] = CONFIG['llm']['ollama_model']
                st.session_state[f'ollama_temperature_{session_id}'] = 0.7
                st.session_state[f'ollama_top_p_{session_id}'] = 0.9
                st.session_state[f'ollama_top_k_{session_id}'] = 40
                st.session_state[f'ollama_repeat_penalty_{session_id}'] = 1.1
                st.session_state[f'ollama_num_ctx_{session_id}'] = 4096
                st.session_state[f'ollama_num_thread_{session_id}'] = 4
                
                # Reset OpenAI settings
                st.session_state[f'openai_model_{session_id}'] = CONFIG['llm']['openai_model']
                st.session_state[f'openai_temperature_{session_id}'] = 0.7
                st.session_state[f'openai_top_p_{session_id}'] = 1.0
                st.session_state[f'openai_max_tokens_{session_id}'] = 2048
                st.session_state[f'openai_presence_penalty_{session_id}'] = 0.0
                st.session_state[f'openai_frequency_penalty_{session_id}'] = 0.0
                st.session_state[f'openai_logit_bias_{session_id}'] = '{}'
                
                st.success("已重置为默认设置")
                # No st.rerun() needed - Streamlit will update automatically
        
        with col2:
            if st.button("刷新连接状态", key=f"refresh_connection_{session_id}"):
                # Clear the cache to force fresh API calls
                get_ollama_models.clear()
                get_ollama_model_info.clear()
                test_ollama_connection.clear()
                test_openai_connection.clear()
                get_openai_models.clear()
                
                st.success("缓存已清除，正在刷新连接状态...")
                # No st.rerun() needed - Streamlit will update automatically
        
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