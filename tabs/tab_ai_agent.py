import json
import os
import re
import shutil
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

import streamlit as st

from config import CONFIG
from util import ensure_session_dirs, handle_file_upload

from tabs.ai_agent.background import generate_agent_plan, run_ai_agent_job
from tabs.ai_agent.mcp_tools import get_agent_paths, prepare_conversation_dirs


def render_ai_agent_tab(session_id):
    # Handle None session_id (user not logged in)
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return

    with st.container(key="ai_agent_root"):
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
        history_dir: Optional[str] = None
        logs_dir = agent_paths.get("logs")
        if logs_dir:
            os.makedirs(logs_dir, exist_ok=True)
            history_dir = os.path.join(logs_dir, "chat_sessions")
            os.makedirs(history_dir, exist_ok=True)

        def _conversation_path(conversation_id: str) -> Optional[str]:
            if not history_dir:
                return None
            return os.path.join(history_dir, f"{conversation_id}.json")

        def _load_conversation(conversation_id: str) -> Dict[str, object]:
            path = _conversation_path(conversation_id)
            if not path or not os.path.exists(path):
                return {"id": conversation_id, "messages": [], "title": "新对话"}
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except Exception:
                return {"id": conversation_id, "messages": [], "title": "新对话"}
            data.setdefault("id", conversation_id)
            data.setdefault("messages", [])
            data.setdefault("title", "新对话")
            return data

        def _save_conversation(
            conversation_id: str,
            messages: List[Dict[str, object]],
            *,
            title: Optional[str] = None,
        ) -> None:
            path = _conversation_path(conversation_id)
            if not path:
                return
            now = datetime.utcnow().isoformat()
            existing = _load_conversation(conversation_id)
            created_at = existing.get("created_at") or now
            payload = {
                "id": conversation_id,
                "title": title or existing.get("title") or "新对话",
                "created_at": created_at,
                "updated_at": now,
                "messages": messages,
            }
            try:
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2)
            except Exception:
                pass

        def _list_conversations() -> List[Dict[str, object]]:
            items: List[Dict[str, object]] = []
            if not history_dir or not os.path.isdir(history_dir):
                return items
            for name in sorted(os.listdir(history_dir)):
                if not name.endswith(".json"):
                    continue
                cid = name[:-5]
                data = _load_conversation(cid)
                items.append(
                    {
                        "id": cid,
                        "title": data.get("title") or "新对话",
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                    }
                )
            items.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
            return items

        def _create_conversation() -> str:
            conversation_id = datetime.utcnow().strftime("%Y%m%d%H%M%S") + f"-{uuid4().hex[:6]}"
            _save_conversation(conversation_id, [], title="新对话")
            prepare_conversation_dirs(session_dirs, conversation_id)
            return conversation_id

        def _delete_conversation(conversation_id: str) -> None:
            path = _conversation_path(conversation_id)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
            uploads_root = session_dirs.get("ai_agent_inputs", "")
            if uploads_root:
                shutil.rmtree(os.path.join(uploads_root, conversation_id), ignore_errors=True)
            generated_root = session_dirs.get("generated_ai_agent", "")
            if generated_root:
                shutil.rmtree(os.path.join(generated_root, conversation_id), ignore_errors=True)
            suffix = f"_{session_id}_{conversation_id}"
            for key in list(st.session_state.keys()):
                if key.endswith(suffix):
                    st.session_state.pop(key, None)

        conversations = _list_conversations()
        active_key = f"ai_agent_active_conversation_{session_id}"

        if not conversations:
            new_id = _create_conversation()
            conversations = _list_conversations()
            st.session_state[active_key] = new_id

        active_conversation_id = st.session_state.get(active_key)
        if not active_conversation_id or all(c["id"] != active_conversation_id for c in conversations):
            active_conversation_id = conversations[0]["id"]
            st.session_state[active_key] = active_conversation_id

        conversation_data = _load_conversation(active_conversation_id)

        # Prepare per-conversation directories and paths
        conversation_dirs = prepare_conversation_dirs(session_dirs, active_conversation_id)

        def _state_key(name: str) -> str:
            return f"{name}_{session_id}_{active_conversation_id}"

        # Create chat history, main chat, and file management columns
        col_history, col_main, col_info = st.columns([1, 2, 1])

        with col_history:
            st.markdown("🗂️ 对话历史")
            if st.button("➕ 新建对话", use_container_width=True):
                new_id = _create_conversation()
                st.session_state[active_key] = new_id
                st.rerun()

            st.divider()

            if not conversations:
                st.info("暂无历史对话")
            else:
                for conv in conversations:
                    is_active = conv["id"] == active_conversation_id
                    display_title = str(conv.get("title") or "新对话").strip() or "新对话"
                    if len(display_title) > 20:
                        display_title = display_title[:20] + "…"
                    prefix = "👉 " if is_active else ""
                    row = st.container()
                    left_col, right_col = row.columns([5, 1])
                    if left_col.button(
                        f"{prefix}{display_title}",
                        key=f"select_conv_{conv['id']}",
                        use_container_width=True,
                    ):
                        st.session_state[active_key] = conv["id"]
                        st.rerun()
                    if right_col.button(
                        "🗑️",
                        key=f"delete_conv_{conv['id']}",
                        use_container_width=True,
                    ):
                        _delete_conversation(conv["id"])
                        remaining = [c for c in conversations if c["id"] != conv["id"]]
                        if remaining:
                            st.session_state[active_key] = remaining[0]["id"]
                        else:
                            st.session_state[active_key] = _create_conversation()
                        st.rerun()
                    timestamp = conv.get("updated_at")
                    if timestamp:
                        row.caption(f"更新于 {str(timestamp)[:19]}")

        uploaded_files_list: List[str] = []

        with col_info:
            st.markdown("📁 文件管理")

            uploads_dir = conversation_dirs.get("ai_agent_inputs", "")
            files = st.file_uploader(
                "上传文件",
                type=None,
                accept_multiple_files=True,
                key=f"file_upload_{session_id}_{active_conversation_id}",
            )
            if files and uploads_dir:
                names = tuple(sorted([f.name for f in files]))
                state_key = _state_key("ai_agent_last_uploaded")
                if st.session_state.get(state_key) != names:
                    saved = handle_file_upload(files, uploads_dir)
                    st.session_state[state_key] = names
                    st.success(f"已保存 {saved} 个文件")
                else:
                    st.caption(f"已保存 {len(files)} 个文件")

            # Show uploaded files
            try:
                if uploads_dir and os.path.exists(uploads_dir):
                    uploaded_files_list = [
                        f for f in os.listdir(uploads_dir)
                        if os.path.isfile(os.path.join(uploads_dir, f))
                    ]
                    if uploaded_files_list:
                        st.caption(f"已上传文件 ({len(uploaded_files_list)}):")
                        for fname in uploaded_files_list[:5]:  # Show first 5
                            st.text(f"• {fname}")
                        if len(uploaded_files_list) > 5:
                            st.caption(f"... 还有 {len(uploaded_files_list) - 5} 个文件")
                    else:
                        st.caption("当前对话暂无上传文件")
            except Exception:
                st.caption("文件列表读取失败")

        with col_main:
            st.subheader("🤖 AI智能体")
            
            history_key = _state_key("ai_agent_chat_history")
            running_key = _state_key("ai_agent_running")
            debug_key = _state_key("ai_agent_debug")
            plan_key = _state_key("ai_agent_plan")
            plan_status_key = _state_key("ai_agent_plan_status")
            plan_error_key = _state_key("ai_agent_plan_error")
            pending_key = _state_key("ai_agent_pending_goal")
            plan_msg_idx_key = _state_key("ai_agent_plan_message_idx")
            last_msg_processed_key = _state_key("last_msg_processed")
            title_key = _state_key("ai_agent_conversation_title")
            auto_run_key = _state_key("ai_agent_auto_run")

            if history_key not in st.session_state:
                st.session_state[history_key] = list(conversation_data.get("messages") or [])

            if title_key not in st.session_state:
                st.session_state[title_key] = str(conversation_data.get("title") or "新对话")

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
            if auto_run_key not in st.session_state:
                st.session_state[auto_run_key] = False
            
            streaming_text_key = _state_key("ai_agent_streaming_text")
            if streaming_text_key not in st.session_state:
                st.session_state[streaming_text_key] = ""

            chat_history: List[Dict[str, object]] = st.session_state[history_key]

            def _persist_history(history: List[Dict[str, object]]) -> None:
                title = st.session_state.get(title_key) or "新对话"
                _save_conversation(active_conversation_id, history, title=title)

            plan_data = st.session_state.get(plan_key)
            plan_status = st.session_state.get(plan_status_key, "idle")
            plan_error = st.session_state.get(plan_error_key)
            pending_goal = st.session_state.get(pending_key)
            running = st.session_state.get(running_key, False)
            auto_run_active = st.session_state.get(auto_run_key, False)

            turbo_checkbox_key = f"turbo_mode_{session_id}"
            if turbo_checkbox_key not in st.session_state:
                st.session_state[turbo_checkbox_key] = False
            turbo_enabled = st.session_state[turbo_checkbox_key]

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
                            conversation_id=active_conversation_id,
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

            # Create boxed chat container
            chat_container = st.container(height=500)
            
            with chat_container:
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
                
                # Show streaming text if available
                if st.session_state.get(streaming_text_key):
                    with st.chat_message("assistant"):
                        st.write(st.session_state[streaming_text_key])
                
                # Chat input inside the container
                user_input = st.chat_input("输入问题或任务，AI智能体会帮你完成...")

            show_plan_panel = (
                not auto_run_active
                and (
                    pending_goal
                    or plan_status in ("generating", "ready", "approved", "running", "completed", "error")
                    or plan_data
                )
            )

            if show_plan_panel:
                with st.container():
                    st.markdown("### 🧭 执行计划")
                    if plan_status == "generating":
                        st.info("智能体正在制定执行计划，请稍候…")
                    elif plan_status == "error":
                        st.error(f"规划失败：{plan_error or '未知错误'}")
                        if st.button("重新规划", key=f"retry_plan_{session_id}_{active_conversation_id}", use_container_width=True):
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
                            key=f"confirm_plan_{session_id}_{active_conversation_id}",
                            use_container_width=True,
                            disabled=execute_disabled,
                        ):
                            st.session_state[plan_status_key] = "approved"
                            st.rerun()
                        if action_cols[1].button(
                            "♻️ 重新规划",
                            key=f"regen_plan_{session_id}_{active_conversation_id}",
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

            # Turbo mode checkbox placed after chat container
            turbo_enabled = st.checkbox(
                "高性能模式",
                key=turbo_checkbox_key,
                help="用阿里云服务器，速度提升10倍以上，涉密文件勿勾选此模式。",
            )

            def _should_auto_run(goal: str) -> bool:
                text = (goal or "").strip()
                if not text:
                    return False
                if len(text) > 120:
                    return False
                if "\n" in text or "\r" in text:
                    return False
                lowered = text.lower()
                complex_keywords = (
                    "analysis",
                    "analyze",
                    "analyse",
                    "summarize",
                    "summary",
                    "plan",
                    "workflow",
                    "steps",
                    "generate",
                    "implement",
                    "write",
                    "code",
                    "script",
                    "translate",
                    "compare",
                    "explain",
                    "document",
                    "file",
                    "pdf",
                    "upload",
                )
                if any(keyword in lowered for keyword in complex_keywords):
                    return False
                complex_keywords_zh = (
                    "分析",
                    "总结",
                    "计划",
                    "流程",
                    "步骤",
                    "方案",
                    "生成",
                    "编写",
                    "写",
                    "代码",
                    "脚本",
                    "翻译",
                    "比较",
                    "解释",
                    "报告",
                    "文档",
                    "文件",
                )
                if any(keyword in text for keyword in complex_keywords_zh):
                    return False
                if uploaded_files_list:
                    return False
                if re.search(r"[。！？!?]{2,}$", text):
                    return False
                return True

            if user_input:
                chat_history.append({"role": "user", "content": user_input})
                if not st.session_state.get(title_key) or st.session_state[title_key] == "新对话":
                    summary = user_input.strip().splitlines()[0][:20]
                    if summary:
                        st.session_state[title_key] = summary
                auto_mode = _should_auto_run(user_input)
                st.session_state[auto_run_key] = auto_mode
                if auto_mode:
                    assistant_note = {
                        "role": "assistant",
                        "content": "这个问题较为简单，我会直接处理并给出答案，无需生成详细执行计划。",
                        "metadata": {"type": "note", "auto_run": True},
                    }
                    chat_history.append(assistant_note)
                st.session_state[history_key] = chat_history
                _persist_history(chat_history)
                st.session_state[pending_key] = user_input
                st.session_state[plan_status_key] = "approved" if auto_mode else "pending"
                st.session_state[plan_key] = None
                st.session_state[plan_error_key] = None
                st.session_state[plan_msg_idx_key] = None
                st.session_state[last_msg_processed_key] = None
                st.session_state[streaming_text_key] = ""
                st.rerun()

            if pending_goal and st.session_state.get(plan_status_key) == "approved" and not running:
                st.session_state[last_msg_processed_key] = pending_goal
                st.session_state[running_key] = True
                st.session_state[plan_status_key] = "running"
                running = True
                plan_status = "running"
                _ensure_plan_message()

                assistant_msg_placeholder = None
                streaming_placeholder = None

                def _publish(event: Dict[str, object]) -> None:
                    nonlocal assistant_msg_placeholder, streaming_placeholder

                    kind = event.get("status") or (event.get("stream") and (event["stream"].get("kind")))
                    stage = event.get("stage", "")
                    message = event.get("message", "")

                    # Handle streaming chunks
                    if "stream_chunk" in event:
                        if streaming_placeholder is None:
                            streaming_placeholder = st.chat_message("assistant").empty()
                        current_text = st.session_state.get(streaming_text_key, "")
                        current_text += event["stream_chunk"]
                        st.session_state[streaming_text_key] = current_text
                        streaming_placeholder.write(current_text)
                        return

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

                            step_key = f"reasoning_step_{step_num}_{session_id}_{active_conversation_id}"
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

                                tool_step_key = f"tool_step_{step_num}_{tool}_{session_id}_{active_conversation_id}"
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
                        conversation_id=active_conversation_id,
                    )

                    final_list = res.get("final_results") or []
                    final_text = final_list[0] if final_list else "任务已完成，但没有返回结果。"

                    # Clear streaming text and add final message
                    st.session_state[streaming_text_key] = ""
                    final_msg = {
                        "role": "assistant",
                        "content": final_text,
                        "metadata": {"type": "final_response"}
                    }
                    chat_history.append(final_msg)
                    st.session_state[history_key] = chat_history
                    _persist_history(chat_history)

                except Exception as e:
                    st.session_state[streaming_text_key] = ""
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
                    st.session_state[pending_key] = None
                    if st.session_state.get(auto_run_key):
                        st.session_state[plan_status_key] = "idle"
                        st.session_state[plan_key] = None
                        st.session_state[plan_error_key] = None
                        st.session_state[plan_msg_idx_key] = None
                        st.session_state[auto_run_key] = False
                    else:
                        st.session_state[plan_status_key] = "completed"
                        _ensure_plan_message()
                    running = False
                    plan_status = st.session_state.get(plan_status_key, "idle")
                    st.rerun()

            if st.session_state.get(running_key, False):
                if st.button(
                    "⏹️ 停止执行",
                    key=f"stop_agent_{session_id}_{active_conversation_id}",
                    use_container_width=True,
                ):
                    st.session_state[running_key] = False
                    st.session_state[plan_status_key] = "completed"
                    st.session_state[pending_key] = None
                    st.rerun()

            with st.expander("🔍 调试信息", expanded=False):
                for msg in st.session_state.get(debug_key, [])[-20:]:
                    st.text(msg)

        return
