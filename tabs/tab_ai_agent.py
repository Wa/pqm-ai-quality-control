import json
import os
from typing import Dict, List

import streamlit as st

from config import CONFIG
from util import ensure_session_dirs, handle_file_upload

from tabs.ai_agent.background import generate_agent_plan, run_ai_agent_job
from tabs.ai_agent.mcp_tools import get_agent_paths


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
        agent_paths = get_agent_paths(session_dirs)
        history_file = None
        logs_dir = agent_paths.get("logs")
        if logs_dir:
            os.makedirs(logs_dir, exist_ok=True)
            history_file = os.path.join(logs_dir, "chat_history.json")

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
                if history_file and os.path.exists(history_file):
                    try:
                        os.remove(history_file)
                    except Exception:
                        pass
                plan_related_keys = [
                    f"ai_agent_plan_{session_id}",
                    f"ai_agent_plan_status_{session_id}",
                    f"ai_agent_plan_error_{session_id}",
                    f"ai_agent_pending_goal_{session_id}",
                    f"ai_agent_plan_message_idx_{session_id}",
                    f"last_msg_processed_{session_id}",
                ]
                for key in plan_related_keys:
                    st.session_state.pop(key, None)
                # Clear step tracking keys to prevent duplicates in new conversation
                for key in list(st.session_state.keys()):
                    if key.startswith(f"reasoning_step_") and key.endswith(f"_{session_id}"):
                        del st.session_state[key]
                    if key.startswith(f"tool_step_") and key.endswith(f"_{session_id}"):
                        del st.session_state[key]
                st.rerun()
        
        with col_main:
            history_key = f'ai_agent_chat_history_{session_id}'
            running_key = f"ai_agent_running_{session_id}"
            debug_key = f"ai_agent_debug_{session_id}"
            plan_key = f"ai_agent_plan_{session_id}"
            plan_status_key = f"ai_agent_plan_status_{session_id}"
            plan_error_key = f"ai_agent_plan_error_{session_id}"
            pending_key = f"ai_agent_pending_goal_{session_id}"
            plan_msg_idx_key = f"ai_agent_plan_message_idx_{session_id}"
            last_msg_processed_key = f"last_msg_processed_{session_id}"

            if history_key not in st.session_state:
                loaded: List[Dict[str, object]] = []
                if history_file and os.path.exists(history_file):
                    try:
                        with open(history_file, "r", encoding="utf-8") as handle:
                            data = json.load(handle)
                            if isinstance(data, list):
                                loaded = data
                    except Exception:
                        loaded = []
                st.session_state[history_key] = loaded

            if running_key not in st.session_state:
                st.session_state[running_key] = False
            if debug_key not in st.session_state:
                st.session_state[debug_key] = []
            if plan_key not in st.session_state:
                st.session_state[plan_key] = None
            if plan_status_key not in st.session_state:
                st.session_state[plan_status_key] = "idle"
            if plan_error_key not in st.session_state:
                st.session_state[plan_error_key] = None
            if pending_key not in st.session_state:
                st.session_state[pending_key] = None

            chat_history: List[Dict[str, object]] = st.session_state[history_key]

            def _persist_history(history: List[Dict[str, object]]) -> None:
                if not history_file:
                    return
                try:
                    with open(history_file, "w", encoding="utf-8") as handle:
                        json.dump(history, handle, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            plan_data = st.session_state.get(plan_key)
            plan_status = st.session_state.get(plan_status_key, "idle")
            plan_error = st.session_state.get(plan_error_key)
            pending_goal = st.session_state.get(pending_key)
            running = st.session_state.get(running_key, False)

            base_context = [
                {"role": msg.get("role"), "content": msg.get("content")}
                for msg in chat_history
                if msg.get("role") in ("user", "assistant")
                and msg.get("content")
                and msg.get("metadata", {}).get("type") not in ("reasoning", "tool_use")
            ]

            if pending_goal and plan_status == "pending" and not running:
                st.session_state[plan_status_key] = "generating"
                st.session_state[plan_error_key] = None
                with st.spinner("智能体正在制定执行计划..."):
                    try:
                        plan_result = generate_agent_plan(
                            session_id=session_id,
                            goal=pending_goal,
                            turbo_mode=bool(turbo_enabled),
                            primary="local",
                            conversation_history=base_context,
                        )
                    except Exception as exc:
                        st.session_state[plan_error_key] = str(exc)
                        st.session_state[plan_status_key] = "error"
                    else:
                        st.session_state[plan_key] = plan_result
                        st.session_state[plan_status_key] = "ready"
                st.rerun()

            def _ensure_plan_message() -> None:
                current_plan = st.session_state.get(plan_key)
                current_status = st.session_state.get(plan_status_key, "idle")
                if not current_plan or current_status not in ("ready", "approved", "running", "completed"):
                    return
                lines = []
                for step in current_plan.get("plan", []):
                    if not isinstance(step, dict):
                        continue
                    step_no = step.get("step", "?")
                    title = step.get("title") or "(未命名步骤)"
                    detail = step.get("details") or ""
                    tool_hints = step.get("tool_hints") or []
                    success = step.get("success_criteria") or ""
                    segment = [f"步骤 {step_no}: {title}"]
                    if detail:
                        segment.append(f"- 行动: {detail}")
                    if tool_hints:
                        segment.append(f"- 工具: {', '.join(tool_hints)}")
                    if success:
                        segment.append(f"- 完成标准: {success}")
                    lines.append("\n".join(segment))
                risks = current_plan.get("risks") or []
                if risks:
                    lines.append("风险: " + "；".join(str(r) for r in risks if r))
                guidelines = current_plan.get("final_answer_guidelines")
                if guidelines:
                    lines.append("最终答复提示: " + str(guidelines))
                plan_text = "\n\n".join(lines) or "已生成执行计划。"
                message = {
                    "role": "assistant",
                    "content": plan_text,
                    "metadata": {
                        "type": "plan",
                        "plan": current_plan,
                        "status": current_status,
                    },
                }
                idx = st.session_state.get(plan_msg_idx_key)
                if isinstance(idx, int) and 0 <= idx < len(chat_history):
                    if chat_history[idx] != message:
                        chat_history[idx] = message
                        st.session_state[history_key] = chat_history
                        _persist_history(chat_history)
                else:
                    chat_history.append(message)
                    st.session_state[plan_msg_idx_key] = len(chat_history) - 1
                    st.session_state[history_key] = chat_history
                    _persist_history(chat_history)

            _ensure_plan_message()

            conversation_context = [
                {"role": msg.get("role"), "content": msg.get("content")}
                for msg in chat_history
                if msg.get("role") in ("user", "assistant")
                and msg.get("content")
                and msg.get("metadata", {}).get("type") not in ("reasoning", "tool_use")
            ]

            for msg in chat_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                metadata = msg.get("metadata", {})

                with st.chat_message(role):
                    if role == "assistant" and metadata.get("type") == "reasoning":
                        with st.expander(f"💭 思考过程 (步骤 {metadata.get('step', '?')})", expanded=False):
                            st.json(metadata.get("action", {}))
                        st.write(content)
                    elif role == "assistant" and metadata.get("type") == "tool_use":
                        tool_name = metadata.get("tool", "unknown")
                        st.caption(f"🔧 使用工具: {tool_name}")
                        with st.expander("查看详情", expanded=False):
                            st.json(metadata.get("result", {}))
                        st.write(content)
                    else:
                        st.write(content)

            if pending_goal or plan_status in ("generating", "ready", "approved", "running", "completed", "error"):
                with st.container():
                    st.markdown("### 🧭 执行计划")
                    if plan_status == "generating":
                        st.info("智能体正在制定执行计划，请稍候…")
                    elif plan_status == "error":
                        st.error(f"规划失败：{plan_error or '未知错误'}")
                        if st.button("重新规划", key=f"retry_plan_{session_id}", use_container_width=True):
                            st.session_state[plan_status_key] = "pending"
                            st.session_state[plan_key] = None
                            st.session_state[plan_error_key] = None
                            st.session_state[plan_msg_idx_key] = None
                            st.rerun()
                    elif plan_status in ("ready", "approved", "running", "completed") and plan_data:
                        for step in plan_data.get("plan", []):
                            if not isinstance(step, dict):
                                continue
                            header = f"步骤 {step.get('step', '?')} · {step.get('title') or '未命名步骤'}"
                            with st.expander(header, expanded=False):
                                if step.get("details"):
                                    st.write(step["details"])
                                if step.get("tool_hints"):
                                    st.caption(f"建议工具：{', '.join(step['tool_hints'])}")
                                if step.get("success_criteria"):
                                    st.caption(f"完成标准：{step['success_criteria']}")
                        if plan_data.get("risks"):
                            st.warning("风险提示：" + "；".join(str(r) for r in plan_data["risks"] if r))
                        if plan_data.get("final_answer_guidelines"):
                            st.info("最终答复提示：" + str(plan_data["final_answer_guidelines"]))
                        files_info = plan_data.get("files", {}) or {}
                        txt_dir = files_info.get("txt_dir")
                        if txt_dir:
                            st.caption(f"文本目录：{txt_dir}")
                        converted = files_info.get("converted") or files_info.get("existing_txts") or []
                        if converted:
                            preview = ", ".join(str(name) for name in converted[:5])
                            if len(converted) > 5:
                                preview += " 等"
                            st.caption(f"可读文本：{preview}")
                        execute_disabled = plan_status in ("approved", "running")
                        regenerate_disabled = plan_status == "running"
                        action_cols = st.columns(2)
                        if action_cols[0].button(
                            "✅ 执行计划",
                            key=f"confirm_plan_{session_id}",
                            use_container_width=True,
                            disabled=execute_disabled,
                        ):
                            st.session_state[plan_status_key] = "approved"
                            st.rerun()
                        if action_cols[1].button(
                            "♻️ 重新规划",
                            key=f"regen_plan_{session_id}",
                            use_container_width=True,
                            disabled=regenerate_disabled,
                        ):
                            st.session_state[plan_status_key] = "pending"
                            st.session_state[plan_key] = None
                            st.session_state[plan_error_key] = None
                            st.session_state[plan_msg_idx_key] = None
                            st.rerun()
                        if plan_status == "approved":
                            st.info("计划已确认，智能体即将开始执行。")
                        elif plan_status == "running":
                            st.info("计划执行中，可使用下方的停止按钮终止任务。")
                        elif plan_status == "completed":
                            st.success("计划执行完成，欢迎继续提问。")

            user_input = st.chat_input("输入问题或任务，AI智能体会帮你完成...")

            if user_input:
                chat_history.append({"role": "user", "content": user_input})
                st.session_state[history_key] = chat_history
                _persist_history(chat_history)
                st.session_state[pending_key] = user_input
                st.session_state[plan_status_key] = "pending"
                st.session_state[plan_key] = None
                st.session_state[plan_error_key] = None
                st.session_state[plan_msg_idx_key] = None
                st.session_state[last_msg_processed_key] = None
                st.rerun()

            if pending_goal and st.session_state.get(plan_status_key) == "approved" and not running:
                st.session_state[last_msg_processed_key] = pending_goal
                st.session_state[running_key] = True
                st.session_state[plan_status_key] = "running"
                running = True
                plan_status = "running"
                _ensure_plan_message()

                assistant_msg_placeholder = None

                def _publish(event: Dict[str, object]) -> None:
                    nonlocal assistant_msg_placeholder

                    kind = event.get("status") or (event.get("stream") and (event["stream"].get("kind")))
                    stage = event.get("stage", "")
                    message = event.get("message", "")

                    if kind == "running":
                        if stage == "initializing":
                            if assistant_msg_placeholder is None:
                                assistant_msg_placeholder = st.chat_message("assistant").empty()
                            assistant_msg_placeholder.info(f"🚀 {message}")
                        elif stage in ("llm_call", "graph_execution"):
                            st.session_state[debug_key].append(f"[{stage}] {message}")
                    elif kind == "failed":
                        if assistant_msg_placeholder:
                            assistant_msg_placeholder.error(f"❌ {message}")
                    elif kind == "succeeded":
                        if assistant_msg_placeholder:
                            assistant_msg_placeholder.success(f"✅ {message}")

                    stream = event.get("stream")
                    if isinstance(stream, dict) and stream.get("kind") == "step":
                        step_num = stream.get("step", 0)
                        last_action = stream.get("last_action", {})

                        if isinstance(last_action, dict):
                            thought = last_action.get("thought", "")
                            tool = last_action.get("tool", "")

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
                                _persist_history(chat_history)

                                tool_step_key = f"tool_step_{step_num}_{tool}_{session_id}"
                                if tool and tool != "none" and not st.session_state.get(tool_step_key, False):
                                    tool_msg = {
                                        "role": "assistant",
                                        "content": f"正在使用工具: {tool}",
                                        "metadata": {
                                            "type": "tool_use",
                                            "step": step_num,
                                            "tool": tool,
                                            "result": stream.get("artifacts", [{}])[-1] if stream.get("artifacts") else {},
                                        }
                                    }
                                    chat_history.append(tool_msg)
                                    st.session_state[history_key] = chat_history
                                    st.session_state[tool_step_key] = True
                                    _persist_history(chat_history)

                def _check_control() -> Dict[str, bool]:
                    return {"paused": False, "stopped": not st.session_state.get(running_key, False)}

                try:
                    res = run_ai_agent_job(
                        session_id=session_id,
                        goal=pending_goal,
                        publish=_publish,
                        check_control=_check_control,
                        primary="local",
                        turbo_mode=bool(turbo_enabled),
                        max_steps=20,
                        conversation_history=conversation_context,
                    )

                    final_list = res.get("final_results") or []
                    final_text = final_list[0] if final_list else "任务已完成，但没有返回结果。"

                    final_msg = {
                        "role": "assistant",
                        "content": final_text,
                        "metadata": {"type": "final_response"}
                    }
                    chat_history.append(final_msg)
                    st.session_state[history_key] = chat_history
                    _persist_history(chat_history)

                except Exception as e:
                    error_msg = {
                        "role": "assistant",
                        "content": f"❌ 执行出错: {str(e)}",
                        "metadata": {"type": "error"}
                    }
                    chat_history.append(error_msg)
                    st.session_state[history_key] = chat_history
                    _persist_history(chat_history)
                finally:
                    st.session_state[running_key] = False
                    st.session_state[plan_status_key] = "completed"
                    st.session_state[pending_key] = None
                    running = False
                    plan_status = "completed"
                    _ensure_plan_message()
                    st.rerun()

            if st.session_state.get(running_key, False):
                if st.button("⏹️ 停止执行", use_container_width=True):
                    st.session_state[running_key] = False
                    st.session_state[plan_status_key] = "completed"
                    st.session_state[pending_key] = None
                    st.rerun()

            with st.expander("🔍 调试信息", expanded=False):
                for msg in st.session_state.get(debug_key, [])[-20:]:
                    st.text(msg)

        return
