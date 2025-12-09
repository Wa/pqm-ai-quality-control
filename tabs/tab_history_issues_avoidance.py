"""Streamlit tab for history issue avoidance workflow."""
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Iterable

import streamlit as st

from backend_client import get_backend_client, is_backend_available
from config import CONFIG
from util import (
    ensure_session_dirs,
    get_directory_refresh_token,
    handle_file_upload,
    list_directory_contents,
)


def _format_file_size(size_bytes: int) -> str:
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    value = float(size_bytes)
    index = 0
    while value >= 1024 and index < len(units) - 1:
        value /= 1024.0
        index += 1
    return f"{value:.1f} {units[index]}"


def _format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def _truncate_filename(filename: str, max_length: int = 40) -> str:
    if len(filename) <= max_length:
        return filename
    name, ext = os.path.splitext(filename)
    available = max_length - len(ext) - 3
    if available <= 0:
        return filename[: max_length - 3] + "..."
    return name[:available] + "..." + ext


def _collect_files(folder: str | None) -> list[dict[str, object]]:
    if not folder:
        return []
    token = get_directory_refresh_token(folder)
    items = [dict(item) for item in list_directory_contents(folder, token)]
    for item in items:
        item.setdefault("path", os.path.join(folder, item["name"]))
    items.sort(key=lambda info: info["modified"], reverse=True)
    return items


def _latest_file(paths: Iterable[str]) -> str | None:
    candidates = [p for p in paths if os.path.isfile(p)]
    if not candidates:
        return None
    candidates.sort(key=lambda value: os.path.getmtime(value))
    return candidates[-1]


@st.fragment()
def _render_history_file_lists(
    *,
    upload_targets: list[dict[str, str]],
    final_results_dir: str,
    session_id: str,
) -> None:
    """Render the right-hand history issues file listings as a fragment."""

    st.subheader("📁 文件管理")
    col_clear1, col_clear2 = st.columns(2)
    col_clear3, col_clear4 = st.columns(2)
    clear_buttons = [col_clear1, col_clear2, col_clear3, col_clear4]
    for column, target in zip(clear_buttons, upload_targets):
        with column:
            if st.button(
                f"🗑️ 清空{target['label']}",
                key=f"history_clear_{target['key']}_{session_id}",
            ):
                try:
                    if target["dir"] and os.path.isdir(target["dir"]):
                        for name in os.listdir(target["dir"]):
                            path = os.path.join(target["dir"], name)
                            if os.path.isfile(path):
                                os.remove(path)
                    st.success(f"已清空 {target['label']} 文件")
                    st.rerun()
                except Exception as error:
                    st.error(f"清空失败: {error}")

    tabs = st.tabs([target["label"] for target in upload_targets])
    for tab, target in zip(tabs, upload_targets):
        with tab:
            files = _collect_files(target["dir"])
            if not files:
                st.info("暂无文件")
                continue
            for info in files:
                display = _truncate_filename(str(info["name"]))
                with st.expander(f"📄 {display}", expanded=False):
                    col_meta, col_actions = st.columns([3, 1])
                    with col_meta:
                        st.write(f"**文件名：** {info['name']}")
                        st.write(f"**大小：** {_format_file_size(int(info['size']))}")
                        st.write(f"**修改时间：** {_format_timestamp(float(info['modified']))}")
                    with col_actions:
                        delete_key = (
                            f"history_delete_{target['key']}_{info['name'].replace(' ', '_').replace('.', '_')}_{session_id}"
                        )
                        if st.button("🗑️ 删除", key=delete_key):
                            try:
                                os.remove(str(info["path"]))
                                st.success(f"已删除 {info['name']}")
                                st.rerun()
                            except Exception as error:
                                st.error(f"删除失败: {error}")

    st.markdown("---")
    result_files = _collect_files(final_results_dir)
    if result_files:
        st.markdown("**下载最新结果**")
        csv_paths = [str(info["path"]) for info in result_files if str(info["name"]).lower().endswith(".csv")]
        xlsx_paths = [str(info["path"]) for info in result_files if str(info["name"]).lower().endswith(".xlsx")]
        latest_csv = _latest_file(csv_paths)
        latest_xlsx = _latest_file(xlsx_paths)
        if latest_csv:
            with open(latest_csv, "rb") as handle:
                st.download_button(
                    label=f"下载CSV（{os.path.basename(latest_csv)}）",
                    data=handle.read(),
                    file_name=os.path.basename(latest_csv),
                    mime="text/csv",
                    key=f"history_download_csv_{session_id}",
                )
        if latest_xlsx:
            with open(latest_xlsx, "rb") as handle:
                st.download_button(
                    label=f"下载Excel（{os.path.basename(latest_xlsx)}）",
                    data=handle.read(),
                    file_name=os.path.basename(latest_xlsx),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"history_download_xlsx_{session_id}",
                )


def render_history_issues_avoidance_tab(session_id: str | None) -> None:
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return

    base_dirs = {"generated": str(CONFIG["directories"]["generated_files"])}
    session_dirs = ensure_session_dirs(base_dirs, session_id)

    issue_lists_dir = session_dirs.get("history_issue_lists")
    dfmea_dir = session_dirs.get("history_dfmea")
    pfmea_dir = session_dirs.get("history_pfmea")
    cp_dir = session_dirs.get("history_cp")
    generated_root = session_dirs.get("generated_history_issues_avoidance")
    final_results_dir = os.path.join(generated_root, "final_results") if generated_root else ""

    backend_ready = is_backend_available()
    backend_client = get_backend_client() if backend_ready else None

    job_state_key = f"history_job_id_{session_id}"
    last_status_key = f"history_last_status_{session_id}"
    stream_state_key = f"history_stream_state_{session_id}"
    job_status: dict[str, object] | None = None
    job_error: str | None = None

    stored_job_id = st.session_state.get(job_state_key)

    if backend_ready and backend_client is not None:
        if stored_job_id:
            result = backend_client.get_history_job(stored_job_id)
            if isinstance(result, dict) and result.get("job_id"):
                job_status = result
            elif isinstance(result, dict) and result.get("status") == "error":
                job_error = str(result.get("message", "后台任务查询失败"))
            elif isinstance(result, dict) and result.get("detail"):
                detail = str(result.get("detail"))
                job_error = detail
                if detail == "未找到任务":
                    st.session_state.pop(job_state_key, None)
        if job_status is None:
            result = backend_client.list_history_jobs(session_id)
            if isinstance(result, list) and result:
                job_status = result[0]
                if isinstance(job_status, dict) and job_status.get("job_id"):
                    st.session_state[job_state_key] = job_status.get("job_id")
                    stored_job_id = job_status.get("job_id")
            elif isinstance(result, dict) and result.get("status") == "error":
                job_error = str(result.get("message", "后台任务列表查询失败"))
    else:
        job_error = "后台服务未连接"

    if job_status:
        st.session_state[last_status_key] = job_status
    elif stored_job_id:
        cached_status = st.session_state.get(last_status_key)
        if isinstance(cached_status, dict):
            job_status = cached_status
            job_error = None
    else:
        st.session_state.pop(last_status_key, None)

    status_str = str(job_status.get("status")) if job_status else ""
    job_running = bool(job_status and status_str in {"queued", "running"})
    job_paused = bool(job_status and status_str == "paused")

    upload_targets = [
        {"label": "历史问题清单", "key": "issue_lists", "dir": issue_lists_dir},
        {"label": "DFMEA", "key": "dfmea", "dir": dfmea_dir},
        {"label": "PFMEA", "key": "pfmea", "dir": pfmea_dir},
        {"label": "控制计划 (CP)", "key": "cp", "dir": cp_dir},
    ]

    col_main, col_info = st.columns([2, 1])

    with col_info:
        _render_history_file_lists(
            upload_targets=upload_targets,
            final_results_dir=final_results_dir,
            session_id=session_id,
        )

    with col_main:
        st.subheader("📋 历史问题规避")
        st.markdown(
            "第1步：在右侧文件列表中清理上一轮任务遗留的文件。  \n"
            "第2步：在下方上传本次需要处理的历史问题清单与DFMEA/PFMEA/控制计划文件。  \n"
            "第3步：使用下方的控制按钮启动或管理任务并关注进度提示。  \n"
            "第4步：任务结束后在右侧的分析结果列表中下载输出文件。  \n"
        )

        st.markdown("**上传文件**")
        upload_columns = st.columns(2)
        for index, target in enumerate(upload_targets):
            column = upload_columns[index % len(upload_columns)]
            with column:
                uploads = st.file_uploader(
                    f"上传{target['label']}",
                    type=None,
                    accept_multiple_files=True,
                    key=f"history_upload_{target['key']}_{session_id}",
                )
                if uploads:
                    handle_file_upload(uploads, target["dir"])
                    st.success(f"已上传 {len(uploads)} 个 {target['label']} 文件")

        st.markdown("---")

        col_controls = st.container()
        with col_controls:
            col_start, col_pause, col_resume, col_stop = st.columns(4)
            with col_start:
                start_disabled = not backend_ready or job_running or job_paused
                if st.button("▶️ 开始", disabled=start_disabled, key=f"history_start_{session_id}"):
                    if backend_client is not None:
                        result = backend_client.start_history_job(session_id)
                        job_id = result.get("job_id") if isinstance(result, dict) else None
                        if not job_id:
                            fallback = backend_client.list_history_jobs(session_id)
                            if isinstance(fallback, list) and fallback:
                                latest_job = fallback[0]
                                if isinstance(latest_job, dict) and latest_job.get("job_id"):
                                    job_id = latest_job.get("job_id")
                                    result = latest_job
                        if job_id:
                            st.session_state[job_state_key] = job_id
                            st.success("已启动后台任务")
                            st.rerun()
                        else:
                            message = "任务启动失败"
                            if isinstance(result, dict):
                                message = str(result.get("message", message))
                            st.error(message)
            with col_pause:
                if st.button("⏸ 暂停", disabled=not job_running, key=f"history_pause_{session_id}"):
                    if backend_client is not None and job_status:
                        job_id = str(job_status.get("job_id"))
                        backend_client.pause_history_job(job_id)
                        st.rerun()
            with col_resume:
                if st.button("▶️ 继续", disabled=not job_paused, key=f"history_resume_{session_id}"):
                    if backend_client is not None and job_status:
                        job_id = str(job_status.get("job_id"))
                        backend_client.resume_history_job(job_id)
                        st.rerun()
            with col_stop:
                if st.button(
                    "⏹ 停止",
                    disabled=not (job_running or job_paused),
                    key=f"history_stop_{session_id}",
                ):
                    if backend_client is not None and job_status:
                        job_id = str(job_status.get("job_id"))
                        backend_client.stop_history_job(job_id)
                        st.rerun()

        st.markdown("---")

        STATUS_LABELS = {
            "queued": "排队中",
            "running": "运行中",
            "paused": "已暂停",
            "stopping": "停止中",
            "stopped": "已停止",
            "succeeded": "已完成",
            "finished": "已完成",
            "completed": "已完成",
            "failed": "失败",
            "error": "出错",
            "canceled": "已取消",
            "cancelled": "已取消",
        }
        STAGE_LABELS = {
            "conversion": "文件解析",
            "kb_sync": "同步知识库",
            "warmup": "预热流程",
            "compare": "历史问题比对",
            "completed": "已完成",
        }

        if job_status:
            status_label = STATUS_LABELS.get(status_str.lower(), status_str or "未知状态")
            st.markdown(f"**任务状态：** {status_label}")
            stage = job_status.get("stage") or ""
            stage_label = STAGE_LABELS.get(str(stage).lower(), stage)
            if stage_label:
                st.markdown(f"**阶段：** {stage_label}")
            message = job_status.get("message") or ""
            if message:
                st.info(str(message))

            progress_value = float(job_status.get("progress") or 0.0)
            progress_label = f"处理进度：{progress_value * 100:.0f}%"
            st.progress(max(0.0, min(progress_value, 1.0)), text=progress_label)

            stream_events = job_status.get("stream_events") or []
            if isinstance(stream_events, list):
                stream_state = st.session_state.get(stream_state_key)
                job_id = job_status.get("job_id")
                if not isinstance(stream_state, dict) or stream_state.get("job_id") != job_id:
                    stream_state = {"job_id": job_id, "events": [], "rendered": []}
                stored_events = stream_state.get("events") or []
                known_sequences = {
                    int(event.get("sequence") or 0)
                    for event in stored_events
                    if isinstance(event, dict)
                }
                rendered_raw = stream_state.get("rendered") or []
                rendered_sequences: set[int] = set()
                for value in rendered_raw:
                    try:
                        rendered_sequences.add(int(value))
                    except (TypeError, ValueError):
                        continue
                new_events = []
                for event in stream_events:
                    if not isinstance(event, dict):
                        continue
                    try:
                        sequence = int(event.get("sequence") or 0)
                    except (TypeError, ValueError):
                        sequence = 0
                    if sequence in known_sequences:
                        continue
                    known_sequences.add(sequence)
                    new_events.append(event)
                if new_events:
                    stored_events.extend(new_events)
                    stored_events.sort(key=lambda item: int(item.get("sequence") or 0))
                    stream_state["events"] = stored_events
                    st.session_state[stream_state_key] = stream_state
                elif stream_state.get("events") is None:
                    stream_state["events"] = []
                    st.session_state[stream_state_key] = stream_state

                events_to_render = st.session_state.get(stream_state_key, {}).get("events", [])
                if events_to_render:
                    with st.expander("点击查看详细提示与回答", expanded=status_str in {"queued", "running"}):
                        current_group: tuple[str, int] | None = None
                        for event in events_to_render:
                            if not isinstance(event, dict):
                                continue
                            try:
                                sequence = int(event.get("sequence") or 0)
                            except (TypeError, ValueError):
                                sequence = 0
                            kind = event.get("kind")
                            file_name = event.get("file") or ""
                            part = int(event.get("part") or 0)
                            total_parts = int(event.get("total_parts") or 0)
                            timestamp = event.get("ts")
                            message_text = str(event.get("text") or "")
                            if file_name and part:
                                header_key = (file_name, part)
                                if header_key != current_group:
                                    label = f"{file_name} · 第{part}条"
                                    if total_parts:
                                        label = f"{file_name} · 第{part}/{total_parts}条"
                                    st.markdown(f"**{label}**")
                                    current_group = header_key
                            role = "user" if kind == "prompt" else "assistant"
                            is_new = sequence not in rendered_sequences
                            with st.chat_message(role):
                                if timestamp:
                                    st.caption(str(timestamp))
                                if message_text:
                                    if is_new and role == "assistant":
                                        placeholder = st.empty()
                                        placeholder.write(message_text)
                                    elif is_new and role == "user":
                                        st.text(message_text)
                                    else:
                                        if role == "user":
                                            st.text(message_text)
                                        else:
                                            st.write(message_text)
                                else:
                                    st.write("(无内容)")
                            rendered_sequences.add(sequence)
                    stream_state["rendered"] = sorted(rendered_sequences)
                    st.session_state[stream_state_key] = stream_state

            logs = job_status.get("logs") or []
            if isinstance(logs, list) and logs:
                with st.expander("后台日志", expanded=False):
                    for entry in logs[-100:]:
                        if not isinstance(entry, dict):
                            st.write(entry)
                            continue
                        ts = entry.get("ts") or ""
                        level = entry.get("level") or "info"
                        msg = entry.get("message") or ""
                        st.write(f"[{ts}] {level}: {msg}")
        elif job_error:
            st.warning(job_error)
            st.session_state.pop(stream_state_key, None)
        elif backend_ready:
            st.info("后台服务已连接，上传文件后点击开始即可进行历史问题规避分析。")
            st.session_state.pop(stream_state_key, None)
        else:
            st.error("后台服务不可用，请稍后重试。")
            st.session_state.pop(stream_state_key, None)

    if job_running:
        st.caption("页面将在 5 秒后自动刷新以更新后台任务进度…")
        time.sleep(5)
        st.rerun()
