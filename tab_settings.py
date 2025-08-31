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
                'ollama_num_ctx': 65536,
                'ollama_num_thread': 4,
                'openai_temperature': 0.7,
                'openai_top_p': 1.0,
                'openai_max_tokens': 65536,
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
            'ollama_num_ctx': 65536,
            'ollama_num_thread': 4,
            'openai_temperature': 0.7,
            'openai_top_p': 1.0,
            'openai_max_tokens': 65536,
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
        'ollama_num_ctx': st.session_state.get(f'ollama_num_ctx_{session_id}', 65536),
        'ollama_num_thread': st.session_state.get(f'ollama_num_thread_{session_id}', 4),
        'openai_temperature': st.session_state.get(f'openai_temperature_{session_id}', 0.7),
        'openai_top_p': st.session_state.get(f'openai_top_p_{session_id}', 1.0),
        'openai_max_tokens': st.session_state.get(f'openai_max_tokens_{session_id}', 65536),
        'openai_presence_penalty': st.session_state.get(f'openai_presence_penalty_{session_id}', 0.0),
        'openai_frequency_penalty': st.session_state.get(f'openai_frequency_penalty_{session_id}', 0.0)
    }
    return save_user_settings(session_id, current_settings)

@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_ollama_models(host: str):
    """Get available Ollama models from the specified server."""
    try:
        response = requests.get(f"{host}/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [m.get('name') or m.get('model') for m in models]
        else:
            return []
    except Exception as e:
        st.warning(f"无法连接到Ollama服务器: {e}")
        return []

@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_ollama_model_info(model_name, host: str):
    """Get detailed information about a specific Ollama model on the specified server."""
    try:
        response = requests.post(
            f"{host}/api/show",
            json={"name": model_name},
            timeout=3
        )
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception:
        return None

@st.cache_data(ttl=60)
def get_ollama_tags_map(host: str):
    """Return a mapping of model name -> tag info (size, modified_at, digest, etc.) from the specified server."""
    try:
        response = requests.get(f"{host}/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return {(m.get('name') or m.get('model')): m for m in models}
    except Exception:
        pass
    return {}

def _human_size(num_bytes: int) -> str:
    try:
        num = int(num_bytes)
    except Exception:
        return "N/A"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while num >= 1024 and i < len(units) - 1:
        num /= 1024.0
        i += 1
    return f"{num:.1f} {units[i]}"

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
        
        # LLM Backend Selection
        st.header("🤖 大语言模型选择")
        
        llm_options = {
            "Ollama (10.31.60.127:11434)": "ollama_127",
            "Ollama (10.31.60.9:11434)": "ollama_9",
            "OpenAI (sg.uiuiapi.com)": "openai"
        }
        
        # Get current LLM choice from session state
        current_llm_backend = st.session_state.get(f'llm_backend_{session_id}', 'ollama_127')
        
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

        # Resolve host for selected backend
        host = CONFIG['llm']['ollama_host']
        if selected_backend == "ollama_9":
            host = host.replace("10.31.60.127", "10.31.60.9")
        
        col1, col2 = st.columns(2)
        with col1:
            if selected_backend in ("ollama_127", "ollama_9"):
                if test_ollama_connection(host):
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
            **状态:** {'在线' if (selected_backend in ("ollama_127", "ollama_9") and test_ollama_connection(host)) or (selected_backend == "openai" and test_openai_connection()) else '离线'}
            """)

        st.divider()
        
        # Model Configuration
        st.header("⚙️ 模型配置")
        
        if selected_backend in ("ollama_127", "ollama_9"):
            # Initialize selected_model variable from session state
            selected_model = st.session_state.get(f'ollama_model_{session_id}', CONFIG['llm']['ollama_model'])
            
            # Ollama Model Selection
            with st.spinner("正在获取可用模型列表..."):
                available_models = get_ollama_models(host)
            
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
            
                # Model Information - display always, enrich from /api/tags when /api/show lacks fields
                try:
                    show_info = get_ollama_model_info(selected_model, host) or {}
                    tags_map = get_ollama_tags_map(host) or {}
                    tag_info = tags_map.get(selected_model, {})

                    name = show_info.get('name') or tag_info.get('name') or selected_model
                    size_val = show_info.get('size') or tag_info.get('size')
                    size_h = _human_size(size_val) if size_val is not None else 'N/A'
                    modified = show_info.get('modified_at') or tag_info.get('modified_at') or 'N/A'
                    param_sz = (show_info.get('parameter_size')
                                or (show_info.get('model_info') or {}).get('parameter_size')
                                or 'N/A')
                    quant_lvl = (show_info.get('quantization_level')
                                 or (show_info.get('model_info') or {}).get('quantization_level')
                                 or 'N/A')

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**模型名称:** {name}")
                        st.write(f"**模型大小:** {size_h}")
                        st.write(f"**修改时间:** {modified}")
                    with col2:
                        st.write(f"**参数数量:** {param_sz}")
                        st.write(f"**量化级别:** {quant_lvl}")
                except Exception as e:
                    st.warning(f"无法获取模型详细信息: {e}")

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
            
            # Advanced Ollama Settings (always visible)
            col1, col2 = st.columns(2)
            with col1:
                # Determine dynamic max context length from model info
                dynamic_max_ctx = 8192
                try:
                    info = get_ollama_model_info(selected_model) or {}
                    mi = info.get('model_info', {}) or {}
                    # Try common keys first
                    for key in [
                        'gptoss.context_length',
                        'qwen3.context_length',
                        'llama.context_length',
                        'general.context_length'
                    ]:
                        if key in mi and isinstance(mi[key], int):
                            dynamic_max_ctx = int(mi[key])
                            break
                    else:
                        # Fallback: search for any *.context_length field
                        for k, v in mi.items():
                            if isinstance(k, str) and k.endswith('context_length') and isinstance(v, int):
                                dynamic_max_ctx = int(v)
                                break
                except Exception:
                    dynamic_max_ctx = 8192
                # Default to 65536; allow values beyond model-reported max (for RoPE scaling / custom builds)
                _default_ctx = st.session_state.get(f'ollama_num_ctx_{session_id}', 65536)
                num_ctx = st.number_input(
                    "上下文窗口大小",
                    min_value=512,
                    max_value=131072,
                    value=_default_ctx,
                    step=512,
                    help="模型可以处理的最大上下文长度。",
                    key=f"ollama_num_ctx_{session_id}"
                )
                # Hint if chosen value exceeds the model-reported max
                try:
                    if num_ctx > int(dynamic_max_ctx):
                        st.caption(f"提示: 当前模型建议最大为 {dynamic_max_ctx}。较大的上下文可能依赖RoPE缩放/自定义模型或导致内存压力。")
                except Exception:
                    pass
            
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
                    value=st.session_state.get(f'openai_max_tokens_{session_id}', 65536),
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

        if selected_backend in ("ollama_127", "ollama_9"):
            # Two-column compact overview (no JSON toggle)
            col_left, col_right = st.columns(2)
            with col_left:
                st.write("**后端:**", selected_display_name)
                st.write("**主机:**", host)
                st.write("**当前模型:**", selected_model)
            with col_right:
                st.write("**Temperature:**", st.session_state.get(f'ollama_temperature_{session_id}', 0.7))
                st.write("**Top-p:**", st.session_state.get(f'ollama_top_p_{session_id}', 0.9))
                st.write("**Top-k:**", st.session_state.get(f'ollama_top_k_{session_id}', 40))
                st.write("**Repeat Penalty:**", st.session_state.get(f'ollama_repeat_penalty_{session_id}', 1.1))
                st.write("**num_ctx:**", st.session_state.get(f'ollama_num_ctx_{session_id}', 65536))
                st.write("**num_thread:**", st.session_state.get(f'ollama_num_thread_{session_id}', 4))

        elif selected_backend == "openai":
            # Two-column compact overview (no JSON toggle)
            col_left, col_right = st.columns(2)
            with col_left:
                st.write("**后端:**", selected_display_name)
                st.write("**API地址:**", CONFIG['llm']['openai_base_url'])
                st.write("**当前模型:**", st.session_state.get(f'openai_model_{session_id}', CONFIG['llm']['openai_model']))
            with col_right:
                st.write("**Temperature:**", st.session_state.get(f'openai_temperature_{session_id}', 0.7))
                st.write("**Top-p:**", st.session_state.get(f'openai_top_p_{session_id}', 1.0))
                st.write("**Max Tokens:**", st.session_state.get(f'openai_max_tokens_{session_id}', 65536))
                st.write("**Presence Penalty:**", st.session_state.get(f'openai_presence_penalty_{session_id}', 0.0))
                st.write("**Frequency Penalty:**", st.session_state.get(f'openai_frequency_penalty_{session_id}', 0.0))
                st.write("**Logit Bias:**", st.session_state.get(f'openai_logit_bias_{session_id}', '{}'))
        
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

        # Persist any changes made during this run (sliders, inputs, etc.)
        # Model selection already saves explicitly above; this ensures other parameters are saved too.
        save_current_settings(session_id)