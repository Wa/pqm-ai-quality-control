"""Streamlit UI for the file completeness check workflow."""
from __future__ import annotations

import os
import shutil
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import streamlit as st

from backend_client import get_backend_client, is_backend_available
from config import CONFIG
from tabs.file_completeness import STAGE_ORDER, STAGE_SLUG_MAP
from tabs.shared import stream_text
from util import ensure_session_dirs, handle_file_upload, get_user_session


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


def get_file_list(folder: str) -> List[Dict[str, object]]:
    if not os.path.exists(folder):
        return []
    files: List[Dict[str, object]] = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            stat = os.stat(path)
            files.append(
                {
                    "name": name,
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "path": path,
                }
            )
    return sorted(files, key=lambda item: (item["name"].lower(), item["modified"]))


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


def render_stage_events(
    session_id: str,
    events: Iterable[Dict[str, object]],
    source: str,
) -> None:
    if not events:
        return
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
            if not isinstance(state, dict):
                state = {"rendered": set()}
            rendered = set(state.get("rendered") or [])
            for event in sorted(stage_events, key=lambda item: int(item.get("sequence", 0))):
                sequence = int(event.get("sequence", 0))
                is_new = sequence not in rendered
                rendered.add(sequence)
                role = "user" if event.get("kind") == "prompt" else "assistant"
                text = str(event.get("text") or "")
                timestamp = event.get("ts")
                with st.chat_message(role):
                    if timestamp:
                        st.caption(str(timestamp))
                    if text:
                        if is_new:
                            placeholder = st.empty()
                            render_method = "text" if role == "user" else "write"
                            stream_text(placeholder, text, render_method=render_method, delay=0.02)
                        else:
                            if role == "user":
                                st.text(text)
                            else:
                                st.write(text)
                    else:
                        st.write("(无内容)")
            state["rendered"] = sorted(rendered)
            st.session_state[state_key] = state


def render_file_completeness_check_tab(session_id: Optional[str]) -> None:
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return

    st.subheader("📁 文件齐套性检查")
    st.markdown("上传每个阶段的文件后点击开始，AI会根据预设的清单检查并输出结果。")

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

    get_user_session(session_id, "completeness")

    backend_ready = is_backend_available()
    backend_client = get_backend_client() if backend_ready else None
    job_state_key = f"file_completeness_job_id_{session_id}"
    demo_state_key = f"file_completeness_demo_events_{session_id}"

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
        st.markdown("### 文件管理")
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
                st.markdown("---")
                uploader_key = f"uploader_{stage_name}_{session_id}"
                uploaded = st.file_uploader(f"选择{stage_name}文件", accept_multiple_files=True, key=uploader_key)
                if uploaded:
                    target_dir = stage_dirs.get(stage_name)
                    handle_file_upload(uploaded, target_dir)
                    st.rerun()
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

    with col_main:
        st.markdown("### 分析控制")
        btn_row1 = st.columns([1, 1, 1])
        with btn_row1[0]:
            if st.button("开始分析", key=f"start_completeness_{session_id}"):
                if not backend_ready or backend_client is None:
                    st.error("后台服务不可用，无法启动任务。")
                else:
                    clear_folder_contents(initial_results_dir)
                    st.session_state.pop(demo_state_key, None)
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
                    st.progress(min(100, max(0, float(progress))) / 100.0)
                events = job_status.get("stream_events")
                if isinstance(events, list) and events:
                    render_stage_events(session_id, events, source="job")
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
            render_stage_events(session_id, demo_events, source="demo")
