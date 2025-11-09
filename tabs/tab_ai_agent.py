import os
import streamlit as st
from config import CONFIG
from util import resolve_ollama_host, ensure_session_dirs, handle_file_upload
from typing import Dict
from tabs.ai_agent.background import run_ai_agent_job


def render_ai_agent_tab(session_id):
    # Handle None session_id (user not logged in)
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return

    with st.container(key="ai_agent_root"):
        st.subheader("🤖 AI智能体")

        # Ensure the global chat input is visible (other tabs may hide it via CSS)
        st.markdown(
            """
            <style>
            .st-key-ai_agent_root [data-testid=\"stChatInput\"] { display: flex; }
            # [data-testid=\"stChatInput\"] { display: flex; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # New agent workflow UI (LangGraph + MCP tools)
        base_dirs = {
            "uploads": str(CONFIG["directories"]["uploads"]),
            "generated_files": str(CONFIG["directories"]["generated_files"]),
        }
        session_dirs: Dict[str, str] = ensure_session_dirs(base_dirs, session_id)

        # Create main and right column layout (like other tabs)
        col_main, col_info = st.columns([2, 1])
        
        with col_info:
            st.subheader("📁 文件管理")
            
            files = st.file_uploader("上传文件", type=None, accept_multiple_files=True, key=f"file_upload_{session_id}")
            if files:
                names = tuple(sorted([f.name for f in files]))
                state_key = f"ai_agent_last_uploaded_{session_id}"
                if st.session_state.get(state_key) != names:
                    saved = handle_file_upload(files, session_dirs.get("ai_agent_inputs", ""))
                    st.session_state[state_key] = names
                    st.success(f"已保存 {saved} 个文件")
                else:
                    st.caption(f"已保存 {len(files)} 个文件")
            
            # Show uploaded files
            try:
                uploads_dir = session_dirs.get("ai_agent_inputs", "")
                if uploads_dir and os.path.exists(uploads_dir):
                    uploaded_files = [f for f in os.listdir(uploads_dir) if os.path.isfile(os.path.join(uploads_dir, f))]
                    if uploaded_files:
                        st.caption(f"已上传文件 ({len(uploaded_files)}):")
                        for fname in uploaded_files[:5]:  # Show first 5
                            st.text(f"• {fname}")
                        if len(uploaded_files) > 5:
                            st.caption(f"... 还有 {len(uploaded_files) - 5} 个文件")
            except:
                pass
            
            st.divider()
            
            st.subheader("⚙️ 设置")
            
            # Turbo mode with tooltip (using Streamlit's built-in help parameter)
            turbo_enabled = st.checkbox(
                "高性能模式",
                key=f"turbo_mode_{session_id}",
                help="用阿里云服务器，速度提升10倍以上，涉密文件勿勾选此模式。"
            )
            
            st.divider()
            
            if st.button("🗑️ 清空对话历史", use_container_width=True):
                history_key = f'ai_agent_chat_history_{session_id}'
                st.session_state[history_key] = []
                # Clear step tracking keys to prevent duplicates in new conversation
                for key in list(st.session_state.keys()):
                    if key.startswith(f"reasoning_step_") and key.endswith(f"_{session_id}"):
                        del st.session_state[key]
                    if key.startswith(f"tool_step_") and key.endswith(f"_{session_id}"):
                        del st.session_state[key]
                st.rerun()
        
        with col_main:
            # Initialize conversation history
            history_key = f'ai_agent_chat_history_{session_id}'
            if history_key not in st.session_state:
                st.session_state[history_key] = []
            
            chat_history = st.session_state[history_key]
            
            # Running state
            running_key = f"ai_agent_running_{session_id}"
            if running_key not in st.session_state:
                st.session_state[running_key] = False
            
            # Debug messages
            debug_key = f"ai_agent_debug_{session_id}"
            if debug_key not in st.session_state:
                st.session_state[debug_key] = []
            
            # Display chat history
            for msg in chat_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                metadata = msg.get("metadata", {})
                
                with st.chat_message(role):
                    if role == "assistant" and metadata.get("type") == "reasoning":
                        # Show reasoning steps in expandable sections
                        with st.expander(f"💭 思考过程 (步骤 {metadata.get('step', '?')})", expanded=False):
                            st.json(metadata.get("action", {}))
                        st.write(content)
                    elif role == "assistant" and metadata.get("type") == "tool_use":
                        # Show tool usage
                        tool_name = metadata.get("tool", "unknown")
                        st.caption(f"🔧 使用工具: {tool_name}")
                        with st.expander("查看详情", expanded=False):
                            st.json(metadata.get("result", {}))
                        st.write(content)
                    else:
                        st.write(content)
            
            # Chat input
            user_input = st.chat_input("输入问题或任务，AI智能体会帮你完成...")
            
            if user_input:
                # Add user message to history
                chat_history.append({"role": "user", "content": user_input})
                st.session_state[history_key] = chat_history
                st.rerun()
            
            # Process the last user message if agent is not running
            if chat_history and chat_history[-1].get("role") == "user" and not st.session_state.get(running_key, False):
                last_user_msg = chat_history[-1]["content"]
                last_msg_processed_key = f"last_msg_processed_{session_id}"
                
                # Check if we've already processed this message
                if st.session_state.get(last_msg_processed_key) != last_user_msg:
                    # Mark this message as being processed
                    st.session_state[last_msg_processed_key] = last_user_msg
                    
                    # Mark as running
                    st.session_state[running_key] = True
                    
                    # Create a placeholder for assistant response
                    assistant_msg_placeholder = None
                    
                    def _publish(event: Dict[str, object]) -> None:
                        nonlocal assistant_msg_placeholder
                        
                        kind = event.get("status") or (event.get("stream") and (event["stream"].get("kind")))
                        stage = event.get("stage", "")
                        message = event.get("message", "")
                        
                        if kind == "running":
                            if stage == "initializing":
                                # Show initialization message
                                if assistant_msg_placeholder is None:
                                    assistant_msg_placeholder = st.chat_message("assistant").empty()
                                assistant_msg_placeholder.info(f"🚀 {message}")
                            elif stage in ("llm_call", "graph_execution"):
                                # Debug messages - can be shown in expander if needed
                                st.session_state[debug_key].append(f"[{stage}] {message}")
                        elif kind == "failed":
                            if assistant_msg_placeholder:
                                assistant_msg_placeholder.error(f"❌ {message}")
                        elif kind == "succeeded":
                            if assistant_msg_placeholder:
                                assistant_msg_placeholder.success(f"✅ {message}")
                        
                        # Handle step updates
                        stream = event.get("stream")
                        if isinstance(stream, dict) and stream.get("kind") == "step":
                            step_num = stream.get("step", 0)
                            last_action = stream.get("last_action", {})
                            
                            # Add reasoning step to chat history (prevent duplicates)
                            if isinstance(last_action, dict):
                                thought = last_action.get("thought", "")
                                tool = last_action.get("tool", "")
                                
                                # Check if we've already added a reasoning message for this step
                                step_key = f"reasoning_step_{step_num}_{session_id}"
                                if thought and not st.session_state.get(step_key, False):
                                    reasoning_msg = {
                                        "role": "assistant",
                                        "content": thought,
                                        "metadata": {
                                            "type": "reasoning",
                                            "step": step_num,
                                            "action": last_action
                                        }
                                    }
                                    chat_history.append(reasoning_msg)
                                    st.session_state[history_key] = chat_history
                                    st.session_state[step_key] = True
                                    
                                    # Show tool usage if applicable
                                    tool_step_key = f"tool_step_{step_num}_{tool}_{session_id}"
                                    if tool and tool != "none" and not st.session_state.get(tool_step_key, False):
                                        tool_msg = {
                                            "role": "assistant",
                                            "content": f"正在使用工具: {tool}",
                                            "metadata": {
                                                "type": "tool_use",
                                                "step": step_num,
                                                "tool": tool,
                                                "result": stream.get("artifacts", [{}])[-1] if stream.get("artifacts") else {}
                                            }
                                        }
                                        chat_history.append(tool_msg)
                                        st.session_state[history_key] = chat_history
                                        st.session_state[tool_step_key] = True
                    
                    def _check_control() -> Dict[str, bool]:
                        return {"paused": False, "stopped": not st.session_state.get(running_key, False)}
                    
                    # Run agent job - always use local as primary, cloud as fallback
                    try:
                        # Pass conversation history for context
                        conversation_context = [
                            msg for msg in chat_history 
                            if msg.get("role") in ("user", "assistant") 
                            and msg.get("metadata", {}).get("type") not in ("reasoning", "tool_use")
                        ]
                        
                        res = run_ai_agent_job(
                            session_id=session_id,
                            goal=last_user_msg,
                            publish=_publish,
                            check_control=_check_control,
                            primary="local",  # Always use local as primary
                            turbo_mode=bool(turbo_enabled),
                            max_steps=20,  # Fixed default value
                            conversation_history=conversation_context,
                        )
                        
                        # Get final result
                        final_list = res.get("final_results") or []
                        final_text = final_list[0] if final_list else "任务已完成，但没有返回结果。"
                        
                        # Add final response to chat history
                        final_msg = {
                            "role": "assistant",
                            "content": final_text,
                            "metadata": {"type": "final_response"}
                        }
                        chat_history.append(final_msg)
                        st.session_state[history_key] = chat_history
                        
                    except Exception as e:
                        error_msg = {
                            "role": "assistant",
                            "content": f"❌ 执行出错: {str(e)}",
                            "metadata": {"type": "error"}
                        }
                        chat_history.append(error_msg)
                        st.session_state[history_key] = chat_history
                    finally:
                        st.session_state[running_key] = False
                        st.rerun()
            
            # Show stop button if running
            if st.session_state.get(running_key, False):
                if st.button("⏹️ 停止执行", use_container_width=True):
                    st.session_state[running_key] = False
                    st.rerun()
            
            # Debug area (collapsible)
            with st.expander("🔍 调试信息", expanded=False):
                for msg in st.session_state.get(debug_key, [])[-20:]:
                    st.text(msg)
        
        return

