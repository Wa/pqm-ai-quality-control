"""Streamlit UI for the file completeness check workflow."""
from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import streamlit as st

from backend_client import get_backend_client, is_backend_available
from config import CONFIG
from tabs.file_completeness import STAGE_ORDER, STAGE_REQUIREMENTS, STAGE_SLUG_MAP
from tabs.shared import stream_text
from util import (
    ensure_session_dirs,
    get_file_list,
    get_user_session,
    handle_file_upload,
)


def clear_folder_contents(folder_path: str) -> int:
    """Remove all files and folders inside ``folder_path`` and return removed count."""

    if not os.path.isdir(folder_path):
        return 0
    removed = 0
    for entry in os.listdir(folder_path):
        entry_path = os.path.join(folder_path, entry)
        try:
            if os.path.isfile(entry_path) or os.path.islink(entry_path):
                os.remove(entry_path)
                removed += 1
            elif os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
                removed += 1
        except Exception:
            continue
    return removed


def format_file_size(size_bytes: int) -> str:
    if size_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    idx = 0
    size = float(size_bytes)
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.1f} {units[idx]}"


def format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def truncate_filename(filename: str, max_length: int = 40) -> str:
    if len(filename) <= max_length:
        return filename
    name, ext = os.path.splitext(filename)
    available = max_length - len(ext) - 3
    if available <= 0:
        return filename[: max_length - 3] + "..."
    return name[:available] + "..." + ext


@st.fragment()
def _render_file_completeness_file_lists(
    *, stage_dirs: Dict[str, str], final_results_dir: str, session_id: str
) -> None:
    """Render the right-hand file listing column as a fragment."""

    st.subheader("📁 文件管理")
    tab_labels = list(STAGE_ORDER) + ["分析结果"]
    tabs = st.tabs(tab_labels)
    for idx, stage_name in enumerate(STAGE_ORDER):
        with tabs[idx]:
            folder = stage_dirs.get(stage_name, "")
            files = get_file_list(folder)
            if files:
                for info in files:
                    display_name = truncate_filename(info["name"])
                    with st.expander(f"📄 {display_name}", expanded=False):
                        col_i, col_a = st.columns([3, 1])
                        with col_i:
                            st.write(f"**文件名:** {info['name']}")
                            st.write(f"**大小:** {format_file_size(int(info['size']))}")
                            st.write(f"**修改时间:** {format_timestamp(float(info['modified']))}")
                        with col_a:
                            delete_key = f"delete_{stage_name}_{info['name'].replace(' ', '_')}_{session_id}"
                            if st.button("🗑️ 删除", key=delete_key):
                                try:
                                    os.remove(info["path"])
                                    st.success(f"已删除: {info['name']}")
                                    st.rerun()
                                except Exception as error:
                                    st.error(f"删除失败: {error}")
            else:
                st.write("（未上传）")
    with tabs[-1]:
        result_files = get_file_list(final_results_dir)
        if result_files:
            for idx, info in enumerate(result_files):
                display_name = truncate_filename(info["name"])
                with st.expander(f"📊 {display_name}", expanded=False):
                    st.write(f"**文件名:** {info['name']}")
                    st.write(f"**大小:** {format_file_size(int(info['size']))}")
                    st.write(f"**修改时间:** {format_timestamp(float(info['modified']))}")
                    try:
                        with open(info["path"], "rb") as handle:
                            data = handle.read()
                        st.download_button(
                            label="⬇️ 下载",
                            data=data,
                            file_name=info["name"],
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"download_result_{session_id}_{idx}",
                        )
                    except Exception as error:
                        st.warning(f"无法提供下载: {error}")
        else:
            st.write("（暂无分析结果）")


def _copy_tree(src: str, dst: str) -> int:
    count = 0
    if not os.path.isdir(src):
        return count
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name == ".gitkeep":
            continue
        src_path = os.path.join(src, name)
        dst_path = os.path.join(dst, name)
        if os.path.isdir(src_path):
            count += _copy_tree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)
            count += 1
    return count


def prepare_demo_run(
    session_id: str,
    stage_dirs: Dict[str, str],
    initial_results_dir: str,
    final_results_dir: str,
) -> List[Dict[str, object]]:
    """Populate session directories with demonstration files and build stream events."""

    demo_root = os.path.join(str(CONFIG["directories"]["demonstration"]), "file_completeness_files")
    demo_initial = os.path.join(demo_root, "initial_results")
    demo_final = os.path.join(demo_root, "final_results")
    demo_apqp_root = os.path.join(demo_root, "APQP_files")

    clear_folder_contents(initial_results_dir)

    for stage_name in STAGE_ORDER:
        stage_slug = STAGE_SLUG_MAP.get(stage_name, stage_name)
        demo_stage = os.path.join(demo_apqp_root, stage_slug)
        target_stage = stage_dirs.get(stage_name)
        if not target_stage:
            continue
        clear_folder_contents(target_stage)
        _copy_tree(demo_stage, target_stage)

    _copy_tree(demo_initial, initial_results_dir)

    if os.path.isdir(demo_final):
        os.makedirs(final_results_dir, exist_ok=True)
        for name in os.listdir(demo_final):
            if name == ".gitkeep":
                continue
            src_path = os.path.join(demo_final, name)
            dst_path = os.path.join(final_results_dir, name)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)

    events: List[Dict[str, object]] = []
    sequence = 0
    timestamp = datetime.now().isoformat(timespec="seconds")
    for stage_name in STAGE_ORDER:
        prompt_path = os.path.join(initial_results_dir, f"prompt_{stage_name}.txt")
        if os.path.isfile(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as handle:
                prompt_text = handle.read()
            sequence += 1
            events.append(
                {
                    "sequence": sequence,
                    "kind": "prompt",
                    "file": stage_name,
                    "part": 1,
                    "total_parts": 1,
                    "engine": "demo",
                    "text": prompt_text,
                    "ts": timestamp,
                }
            )
        response_path = os.path.join(initial_results_dir, f"response_{stage_name}.txt")
        if os.path.isfile(response_path):
            with open(response_path, "r", encoding="utf-8") as handle:
                response_text = handle.read()
            sequence += 1
            events.append(
                {
                    "sequence": sequence,
                    "kind": "response",
                    "file": stage_name,
                    "part": 1,
                    "total_parts": 1,
                    "engine": "demo",
                    "text": response_text,
                    "ts": timestamp,
                }
            )
    return events


def _event_sequence_order(event: Dict[str, object]) -> int:
    try:
        value = event.get("sequence") if isinstance(event, dict) else None
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def render_stage_events(
    session_id: str,
    events: Iterable[Dict[str, object]],
    source: str,
    *,
    state_token: Optional[str] = None,
) -> None:
    if not events:
        return
    token_value = state_token if state_token is not None else source
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for event in events:
        stage = str(event.get("file") or event.get("file_name") or "")
        grouped.setdefault(stage, []).append(event)

    for stage_name in STAGE_ORDER:
        stage_events = grouped.get(stage_name, [])
        if not stage_events:
            continue
        with st.expander(f"{stage_name} · 分析记录", expanded=False):
            state_key = f"file_completeness_stream_state_{session_id}_{source}_{stage_name}"
            state = st.session_state.get(state_key)
            if not isinstance(state, dict) or state.get("token") != token_value:
                state = {"token": token_value, "messages": {}}
            messages_state = state.get("messages")
            if not isinstance(messages_state, dict) or not all(
                isinstance(key, str) for key in messages_state.keys()
            ):
                messages_state = {}

            grouped_messages: List[Dict[str, object]] = []
            group_index: Dict[str, int] = {}

            sorted_events = sorted(stage_events, key=_event_sequence_order)
            for event in sorted_events:
                kind = str(event.get("kind") or "info")
                part = int(event.get("part") or 0)
                total_parts = int(event.get("total_parts") or 0)
                message_key = f"{kind}:{part}:{total_parts}"
                role = "user" if kind == "prompt" else "assistant"
                idx = group_index.get(message_key)
                if idx is None:
                    idx = len(grouped_messages)
                    group_index[message_key] = idx
                    grouped_messages.append(
                        {
                            "key": message_key,
                            "role": role,
                            "first_sequence": int(event.get("sequence", 0)) or (len(grouped_messages) + 1),
                            "timestamp": event.get("ts"),
                            "text": str(event.get("text") or ""),
                        }
                    )
                else:
                    group = grouped_messages[idx]
                    sequence_value = int(event.get("sequence", 0))
                    if sequence_value and sequence_value < int(group.get("first_sequence", sequence_value)):
                        group["first_sequence"] = sequence_value
                    timestamp_value = event.get("ts")
                    if timestamp_value:
                        group["timestamp"] = timestamp_value
                    text_value = str(event.get("text") or "")
                    if text_value:
                        group["text"] = text_value

            grouped_messages.sort(key=lambda item: int(item.get("first_sequence", 0)))

            active_keys: List[str] = []
            for message in grouped_messages:
                key = str(message.get("key") or "")
                role = str(message.get("role") or "assistant")
                timestamp = message.get("timestamp")
                message_text = str(message.get("text") or "")
                previous_state = messages_state.get(key)
                previous_text = ""
                if isinstance(previous_state, dict):
                    previous_text = str(previous_state.get("text") or "")
                is_new = not isinstance(previous_state, dict)
                text_changed = message_text and message_text != previous_text

                with st.chat_message(role):
                    if timestamp:
                        st.caption(str(timestamp))
                    if message_text:
                        if is_new or text_changed:
                            placeholder = st.empty()
                            render_method = "text" if role == "user" else "write"
                            stream_text(
                                placeholder,
                                message_text,
                                render_method=render_method,
                                delay=0.02,
                            )
                        else:
                            if role == "user":
                                st.text(message_text)
                            else:
                                st.write(message_text)
                    else:
                        st.write("(无内容)")

                messages_state[key] = {
                    "text": message_text,
                    "timestamp": timestamp,
                    "role": role,
                }
                active_keys.append(key)

            for existing_key in list(messages_state.keys()):
                if existing_key not in active_keys:
                    messages_state.pop(existing_key, None)

            state["messages"] = messages_state
            state["token"] = token_value
            st.session_state[state_key] = state


def render_file_completeness_check_tab(session_id: Optional[str]) -> None:
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return

    uploads_root = str(CONFIG["directories"]["uploads"])
    base_dirs: Dict[str, object] = {}
    for stage_name in STAGE_ORDER:
        slug = STAGE_SLUG_MAP.get(stage_name, stage_name)
        base_dirs[slug] = os.path.join(uploads_root, "{session_id}", "file_completeness", slug)
    base_dirs["generated"] = str(CONFIG["directories"]["generated_files"])
    session_dirs = ensure_session_dirs(base_dirs, session_id)
    generated_session_dir = session_dirs.get("generated") or session_dirs.get("generated_files")
    completeness_dir = session_dirs.get(
        "generated_file_completeness_check",
        os.path.join(generated_session_dir, "file_completeness_check"),
    )
    os.makedirs(completeness_dir, exist_ok=True)
    initial_results_dir = os.path.join(completeness_dir, "initial_results")
    final_results_dir = os.path.join(completeness_dir, "final_results")
    os.makedirs(initial_results_dir, exist_ok=True)
    os.makedirs(final_results_dir, exist_ok=True)

    stage_dirs = {name: session_dirs.get(STAGE_SLUG_MAP.get(name, name), "") for name in STAGE_ORDER}

    custom_requirements_path = os.path.join(completeness_dir, "custom_requirements.json")
    existing_custom_requirements: Dict[str, List[str]] = {}
    if os.path.isfile(custom_requirements_path):
        try:
            with open(custom_requirements_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            stages_data = data.get("stages") if isinstance(data, dict) else None
            if isinstance(stages_data, dict):
                for stage_name, items in stages_data.items():
                    if isinstance(stage_name, str):
                        if isinstance(items, list):
                            cleaned = [str(item).strip() for item in items if str(item).strip()]
                        elif isinstance(items, str):
                            cleaned = [
                                piece.strip()
                                for piece in str(items).splitlines()
                                if piece.strip()
                            ]
                        else:
                            cleaned = []
                        if cleaned:
                            existing_custom_requirements[stage_name] = cleaned
                        elif stage_name in STAGE_REQUIREMENTS:
                            existing_custom_requirements[stage_name] = []
        except Exception:
            existing_custom_requirements = {}

    get_user_session(session_id, "completeness")

    backend_ready = is_backend_available()
    backend_client = get_backend_client() if backend_ready else None
    job_state_key = f"file_completeness_job_id_{session_id}"
    demo_state_key = f"file_completeness_demo_events_{session_id}"
    stream_events_state_key = f"file_completeness_stream_events_{session_id}"

    job_status: Optional[Dict[str, object]] = None
    job_error: Optional[str] = None

    if backend_ready and backend_client is not None:
        stored_job_id = st.session_state.get(job_state_key)
        if stored_job_id:
            result = backend_client.get_file_completeness_job(stored_job_id)
            if isinstance(result, dict) and result.get("job_id"):
                job_status = result
            elif isinstance(result, dict) and result.get("status") == "error":
                job_error = str(result.get("message") or "后台任务查询失败")
            elif isinstance(result, dict) and result.get("detail") == "未找到任务":
                st.session_state.pop(job_state_key, None)
        if job_status is None:
            result = backend_client.list_file_completeness_jobs(session_id)
            if isinstance(result, list) and result:
                job_status = result[0]
                if isinstance(job_status, dict) and job_status.get("job_id"):
                    st.session_state[job_state_key] = job_status.get("job_id")
            elif isinstance(result, dict) and result.get("status") == "error":
                job_error = str(result.get("message") or "后台任务列表查询失败")
    elif not backend_ready:
        job_error = "后台服务未连接"

    status_value = str(job_status.get("status")) if isinstance(job_status, dict) else ""
    job_running = status_value in {"queued", "running"}
    job_paused = status_value == "paused"

    if job_status:
        st.session_state.pop(demo_state_key, None)

    col_main, col_info = st.columns([2, 1])

    with col_info:
        _render_file_completeness_file_lists(
            stage_dirs=stage_dirs,
            final_results_dir=final_results_dir,
            session_id=session_id,
        )

    with col_main:

        st.subheader("📁 文件齐套性检查")
        st.markdown(
            "第1步：重要！在右边文件列表处清空上次任务的文件（不需要清空分析结果）。  \n"
            "第2步：上传每个阶段的文件。  \n"
            "第3步：点击开始，AI会根据预设的清单检查并输出结果。  \n"
            "第4步：在右边文件列表处下载结果。  \n"
            "审核时间取决于文件数量和长度，一般在1分钟到10分钟之间。"
        )
        upload_cols = st.columns(2)
        for index, stage_name in enumerate(STAGE_ORDER):
            uploader_key = f"uploader_{stage_name}_{session_id}"
            target_dir = stage_dirs.get(stage_name)
            column = upload_cols[index % len(upload_cols)]
            with column:
                uploaded = st.file_uploader(
                    f"上传{stage_name}文件",
                    accept_multiple_files=True,
                    key=uploader_key,
                )
                if uploaded:
                    if target_dir:
                        handle_file_upload(uploaded, target_dir)
                        st.rerun()
                    else:
                        st.error("未找到对应的上传目录，请稍后重试。")

                expander_label = f"点击填写{stage_name}清单"
                requirements_key = f"requirements_{STAGE_SLUG_MAP.get(stage_name, stage_name)}_{session_id}"
                override_items = existing_custom_requirements.get(stage_name)
                if override_items is None:
                    default_items = list(STAGE_REQUIREMENTS.get(stage_name, ()))
                    default_text = "\n".join(default_items)
                else:
                    default_text = "\n".join(override_items)
                with st.expander(expander_label, expanded=False):
                    st.caption("每行填写一个应包含的文件名称。")
                    st.text_area(
                        "文件清单（每行一个）",
                        value=default_text,
                        key=requirements_key,
                        height=180,
                    )

        btn_row1 = st.columns([1, 1, 1])
        with btn_row1[0]:
            if st.button("开始分析", key=f"start_completeness_{session_id}"):
                if not backend_ready or backend_client is None:
                    st.error("后台服务不可用，无法启动任务。")
                else:
                    custom_requirements_payload: Dict[str, List[str]] = {}
                    for stage_name in STAGE_ORDER:
                        slug = STAGE_SLUG_MAP.get(stage_name, stage_name)
                        requirements_key = f"requirements_{slug}_{session_id}"
                        raw_value = st.session_state.get(requirements_key)
                        lines: List[str] = []
                        if isinstance(raw_value, str):
                            lines = [line.strip() for line in raw_value.splitlines() if line.strip()]
                        elif isinstance(raw_value, list):
                            lines = [str(item).strip() for item in raw_value if str(item).strip()]
                        default_items = list(STAGE_REQUIREMENTS.get(stage_name, ()))
                        if lines != default_items:
                            custom_requirements_payload[stage_name] = lines
                    try:
                        if custom_requirements_payload:
                            os.makedirs(os.path.dirname(custom_requirements_path), exist_ok=True)
                            with open(custom_requirements_path, "w", encoding="utf-8") as handle:
                                json.dump(
                                    {"stages": custom_requirements_payload},
                                    handle,
                                    ensure_ascii=False,
                                    indent=2,
                                )
                        elif os.path.isfile(custom_requirements_path):
                            os.remove(custom_requirements_path)
                    except Exception as error:
                        st.warning(f"保存自定义清单失败：{error}")
                    clear_folder_contents(initial_results_dir)
                    st.session_state.pop(demo_state_key, None)
                    st.session_state.pop(stream_events_state_key, None)
                    response = backend_client.start_file_completeness_job(session_id)
                    if isinstance(response, dict) and response.get("job_id"):
                        st.session_state[job_state_key] = response["job_id"]
                        st.success("已提交后台任务。")
                        st.rerun()
                    else:
                        detail = ""
                        if isinstance(response, dict):
                            detail = str(response.get("detail") or response.get("message") or "")
                        st.error(f"提交任务失败：{detail or response}")
        with btn_row1[1]:
            if st.button("暂停", key=f"pause_completeness_{session_id}", disabled=not (backend_ready and job_running)):
                if backend_ready and backend_client and job_status:
                    resp = backend_client.pause_file_completeness_job(job_status.get("job_id"))
                    if isinstance(resp, dict) and (resp.get("status") in {"paused", "running", "stopping"} or resp.get("job_id")):
                        st.success("已请求暂停任务。")
                        st.rerun()
                    else:
                        st.error(f"暂停失败：{resp}")
        with btn_row1[2]:
            if st.button("继续", key=f"resume_completeness_{session_id}", disabled=not (backend_ready and job_paused)):
                if backend_ready and backend_client and job_status:
                    resp = backend_client.resume_file_completeness_job(job_status.get("job_id"))
                    if isinstance(resp, dict) and (resp.get("status") in {"running", "queued"} or resp.get("job_id")):
                        st.success("已请求恢复任务。")
                        st.rerun()
                    else:
                        st.error(f"恢复失败：{resp}")

        btn_row2 = st.columns([1, 1, 1])
        with btn_row2[0]:
            if st.button("停止", key=f"stop_completeness_{session_id}", disabled=not (backend_ready and (job_running or job_paused))):
                if backend_ready and backend_client and job_status:
                    resp = backend_client.stop_file_completeness_job(job_status.get("job_id"))
                    if isinstance(resp, dict) and (resp.get("status") in {"stopping", "failed", "succeeded"} or resp.get("job_id")):
                        st.success("已请求停止任务。")
                        st.rerun()
                    else:
                        st.error(f"停止失败：{resp}")
        with btn_row2[1]:
            if st.button("演示", key=f"demo_completeness_{session_id}"):
                events = prepare_demo_run(session_id, stage_dirs, initial_results_dir, final_results_dir)
                st.session_state.pop(job_state_key, None)
                st.session_state.pop(stream_events_state_key, None)
                st.session_state[demo_state_key] = events
                st.success("已加载演示数据。")
                st.rerun()

        if backend_ready:
            if job_status:
                st.info(
                    f"后台任务状态：{status_value}，阶段：{job_status.get('stage') or ''}，"
                    f"提示：{job_status.get('message') or ''}"
                )
                progress = job_status.get("progress")
                if isinstance(progress, (int, float)):
                    clamped = min(100.0, max(0.0, float(progress)))
                    st.progress(clamped / 100.0)
                    st.caption(f"进度：{clamped:.0f}%")
                result_files = job_status.get("result_files") or []
                if status_value == "succeeded" and result_files:
                    first_result = str(result_files[0])
                    if os.path.isfile(first_result):
                        try:
                            with open(first_result, "rb") as result_handle:
                                result_data = result_handle.read()
                            st.download_button(
                                label="⬇️ 下载最新分析结果",
                                data=result_data,
                                file_name=os.path.basename(first_result),
                                mime="application/octet-stream",
                                key=f"download_latest_{session_id}",
                            )
                        except Exception as error:
                            st.warning(f"无法提供结果下载：{error}")
                job_id = str(job_status.get("job_id") or "")
                stream_state = st.session_state.get(stream_events_state_key)
                if not isinstance(stream_state, dict) or stream_state.get("job_id") != job_id:
                    stream_state = {"job_id": job_id, "events": {}}
                events_state = stream_state.get("events")
                if not isinstance(events_state, dict):
                    events_state = {}
                    stream_state["events"] = events_state

                events = job_status.get("stream_events")
                if isinstance(events, list):
                    for event in events:
                        if not isinstance(event, dict):
                            continue
                        sequence_value = str(event.get("sequence") or "")
                        if not sequence_value:
                            fallback_key = "::".join(
                                [
                                    str(event.get("kind") or ""),
                                    str(event.get("file") or ""),
                                    str(event.get("part") or ""),
                                    str(event.get("total_parts") or ""),
                                    str(event.get("ts") or ""),
                                ]
                            )
                            sequence_value = fallback_key or str(len(events_state) + 1)
                        existing_event = events_state.get(sequence_value)
                        if isinstance(existing_event, dict):
                            merged_event = dict(existing_event)
                            merged_event.update(event)
                            events_state[sequence_value] = merged_event
                        else:
                            events_state[sequence_value] = dict(event)

                st.session_state[stream_events_state_key] = stream_state

                aggregated_events = list(events_state.values())
                if aggregated_events:
                    aggregated_events.sort(key=_event_sequence_order)
                    job_state_token = f"job_{job_id}"
                    render_stage_events(
                        session_id,
                        aggregated_events,
                        source="job",
                        state_token=job_state_token,
                    )
                logs = job_status.get("logs")
                if isinstance(logs, list) and logs:
                    with st.expander("点击查看后台日志", expanded=False):
                        for entry in logs[-50:]:
                            if isinstance(entry, dict):
                                ts = entry.get("ts") or ""
                                level = entry.get("level") or "info"
                                message = entry.get("message") or ""
                                st.write(f"[{ts}] {level}: {message}")
                            else:
                                st.write(str(entry))
            elif job_error:
                st.error(job_error)
            else:
                st.info("后台服务已连接，点击开始即可在后台运行齐套性检查。")
        else:
            st.warning("后台服务不可用，演示模式仍可体验功能。")

        demo_events = st.session_state.get(demo_state_key)
        if demo_events and not job_status:
            st.info("演示模式：下方展示预先生成的提示词与响应。")
            demo_state_token = f"demo_{session_id}"
            render_stage_events(
                session_id,
                demo_events,
                source="demo",
                state_token=demo_state_token,
            )
        elif not job_status:
            stream_state = st.session_state.get(stream_events_state_key)
            if isinstance(stream_state, dict):
                cached_events = stream_state.get("events")
                if isinstance(cached_events, dict) and cached_events:
                    cached_list = list(cached_events.values())
                    cached_list.sort(key=_event_sequence_order)
                    job_state_token = f"job_{stream_state.get('job_id') or ''}"
                    render_stage_events(
                        session_id,
                        cached_list,
                        source="job",
                        state_token=job_state_token,
                    )

        if backend_ready and job_status and status_value in {"queued", "running"}:
            st.caption("页面将在 5 秒后自动刷新以更新后台任务进度…")
            time.sleep(5)
            st.rerun()
