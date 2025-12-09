"""Streamlit tab for parameters consistency checks."""
from __future__ import annotations

import os
import re
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

from .parameters import PARAMETERS_WORKFLOW_SURFACE, stream_text


def _format_file_size(size_bytes: int) -> str:
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    index = 0
    value = float(size_bytes)
    while value >= 1024 and index < len(size_names) - 1:
        value /= 1024.0
        index += 1
    return f"{value:.1f} {size_names[index]}"


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


def _collect_files(folder: str) -> list[dict[str, object]]:
    if not folder:
        return []
    token = get_directory_refresh_token(folder)
    entries = [dict(entry) for entry in list_directory_contents(folder, token)]
    for entry in entries:
        entry.setdefault("path", os.path.join(folder, entry["name"]))
    entries.sort(key=lambda item: item["modified"], reverse=True)
    return entries


def _latest_file(paths: Iterable[str]) -> str | None:
    candidates = [p for p in paths if os.path.isfile(p)]
    if not candidates:
        return None
    candidates.sort(key=lambda value: os.path.getmtime(value))
    return candidates[-1]


@st.fragment()
def _render_parameters_file_lists(
    *,
    reference_dir: str,
    target_dir: str,
    graph_dir: str,
    final_results_dir: str,
    session_id: str,
) -> None:
    """Render the right-hand file management column as a fragment."""

    st.subheader("📁 文件管理")

    reference_files = _collect_files(reference_dir)
    target_files = _collect_files(target_dir)
    graph_files = _collect_files(graph_dir)
    result_files = _collect_files(final_results_dir)

    col_clear1, col_clear2, col_clear3, col_clear4 = st.columns(4)
    with col_clear1:
        if st.button("🗑️ 清空基准文件", key=f"parameters_clear_reference_{session_id}"):
            try:
                for info in reference_files:
                    os.remove(info["path"])
                st.success("已清空基准文件")
                st.rerun(scope="fragment")
            except Exception as error:
                st.error(f"清空失败: {error}")
    with col_clear2:
        if st.button("🗑️ 清空待检查", key=f"parameters_clear_target_{session_id}"):
            try:
                for info in target_files:
                    os.remove(info["path"])
                st.success("已清空待检查文件")
                st.rerun(scope="fragment")
            except Exception as error:
                st.error(f"清空失败: {error}")
    with col_clear3:
        if st.button("🗑️ 清空图纸", key=f"parameters_clear_graph_{session_id}"):
            try:
                for info in graph_files:
                    os.remove(info["path"])
                st.success("已清空图纸文件")
                st.rerun(scope="fragment")
            except Exception as error:
                st.error(f"清空失败: {error}")
    with col_clear4:
        if st.button("🗑️ 清空结果", key=f"parameters_clear_results_{session_id}"):
            try:
                for info in result_files:
                    os.remove(info["path"])
                st.success("已清空分析结果")
                st.rerun(scope="fragment")
            except Exception as error:
                st.error(f"清空失败: {error}")

    tab_reference, tab_target, tab_graph, tab_results = st.tabs([
        "基准文件",
        "待检查文件",
        "图纸文件",
        "分析结果",
    ])

    def _render_file_list(tab, entries, delete_prefix: str) -> None:
        with tab:
            if not entries:
                st.info("暂无文件")
                return
            for info in entries:
                display_name = _truncate_filename(str(info["name"]))
                with st.expander(f"📄 {display_name}", expanded=False):
                    col_meta, col_action = st.columns([3, 1])
                    with col_meta:
                        st.write(f"**文件名：** {info['name']}")
                        st.write(f"**大小：** {_format_file_size(int(info['size']))}")
                        st.write(f"**修改时间：** {_format_timestamp(float(info['modified']))}")
                    with col_action:
                        delete_key = f"{delete_prefix}_{info['name'].replace(' ', '_').replace('.', '_')}_{session_id}"
                        if st.button("🗑️ 删除", key=delete_key):
                            try:
                                os.remove(str(info["path"]))
                                st.success(f"已删除 {info['name']}")
                                st.rerun(scope="fragment")
                            except Exception as error:
                                st.error(f"删除失败: {error}")

    _render_file_list(tab_reference, reference_files, "parameters_delete_reference")
    _render_file_list(tab_target, target_files, "parameters_delete_target")
    _render_file_list(tab_graph, graph_files, "parameters_delete_graph")
    _render_file_list(tab_results, result_files, "parameters_delete_result")

    st.markdown("---")
    try:
        result_names = os.listdir(final_results_dir)
    except Exception:
        result_names = []
    csv_paths = [os.path.join(final_results_dir, name) for name in result_names if name.lower().endswith(".csv")]
    xlsx_paths = [os.path.join(final_results_dir, name) for name in result_names if name.lower().endswith(".xlsx")]
    latest_csv = _latest_file(csv_paths)
    latest_xlsx = _latest_file(xlsx_paths)
    if latest_csv or latest_xlsx:
        st.markdown("**下载最新结果**")
    if latest_csv:
        with open(latest_csv, "rb") as handle:
            st.download_button(
                label=f"下载CSV（{os.path.basename(latest_csv)}）",
                data=handle.read(),
                file_name=os.path.basename(latest_csv),
                mime="text/csv",
                key=f"parameters_download_csv_{session_id}",
            )
    if latest_xlsx:
        with open(latest_xlsx, "rb") as handle:
            st.download_button(
                label=f"下载Excel（{os.path.basename(latest_xlsx)}）",
                data=handle.read(),
                file_name=os.path.basename(latest_xlsx),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"parameters_download_xlsx_{session_id}",
            )


def render_parameters_check_tab(session_id: str | None) -> None:
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return

    uploads_root = str(CONFIG["directories"]["uploads"])
    parameters_base = os.path.join(uploads_root, "{session_id}", "parameters")
    base_dirs = {
        "reference": os.path.join(parameters_base, "reference"),
        "target": os.path.join(parameters_base, "target"),
        "graph": os.path.join(parameters_base, "graph"),
        "generated": str(CONFIG["directories"]["generated_files"]),
    }
    session_dirs = ensure_session_dirs(base_dirs, session_id)
    try:
        paths = PARAMETERS_WORKFLOW_SURFACE.prepare_paths(session_dirs)
    except KeyError as error:
        st.error(f"初始化会话目录失败：{error}")
        return

    reference_dir = paths.standards_dir or session_dirs.get("reference", "")
    target_dir = paths.examined_dir or session_dirs.get("target", "")
    graph_dir = session_dirs.get("graph", "")
    parameters_out_root = paths.output_root
    initial_results_dir = paths.initial_results_dir
    final_results_dir = paths.final_results_dir
    checkpoint_dir = paths.checkpoint_dir

    backend_ready = is_backend_available()
    backend_client = get_backend_client() if backend_ready else None
    job_state_key = f"parameters_job_id_{session_id}"
    last_status_key = f"parameters_last_status_{session_id}"
    stream_state_key = f"parameters_stream_state_{session_id}"
    job_status: dict[str, object] | None = None
    job_error: str | None = None

    stored_job_id = st.session_state.get(job_state_key)

    if backend_ready and backend_client is not None:
        if stored_job_id:
            result = backend_client.get_parameters_job(stored_job_id)
            if isinstance(result, dict) and result.get("job_id"):
                job_status = result
            elif isinstance(result, dict) and result.get("status") == "error":
                job_error = str(result.get("message", "后台任务查询失败"))
            elif isinstance(result, dict) and result.get("detail"):
                job_error = str(result.get("detail"))
                if result.get("detail") == "未找到任务":
                    st.session_state.pop(job_state_key, None)
        if job_status is None:
            result = backend_client.list_parameters_jobs(session_id)
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

    col_main, col_info = st.columns([2, 1])

    with col_info:
        _render_parameters_file_lists(
            reference_dir=reference_dir,
            target_dir=target_dir,
            graph_dir=graph_dir,
            final_results_dir=final_results_dir,
            session_id=session_id,
        )

    with col_main:
        st.subheader("⚙️ 参数一致性检查")
        st.markdown(
            "第1步：在右侧文件列表中清理上一轮任务遗留的文件。  \n"
            "第2步：在下方上传本次需要比对的基准、待检查和图纸文件。  \n"
            "第3步：使用下方的控制按钮启动或管理任务并关注进度提示。  \n"
            "第4步：任务结束后在右侧的分析结果列表中下载输出文件。  \n"
        )

        st.markdown("**上传文件**")
        col_reference, col_target = st.columns(2)
        with col_reference:
            reference_uploads = st.file_uploader(
                "上传基准文件",
                type=None,
                accept_multiple_files=True,
                key=f"parameters_reference_uploader_{session_id}",
            )
            if reference_uploads:
                handle_file_upload(reference_uploads, reference_dir)
                st.success(f"已上传 {len(reference_uploads)} 份基准文件")
        with col_target:
            target_uploads = st.file_uploader(
                "上传待检查文件",
                type=None,
                accept_multiple_files=True,
                key=f"parameters_target_uploader_{session_id}",
            )
            if target_uploads:
                handle_file_upload(target_uploads, target_dir)
                st.success(f"已上传 {len(target_uploads)} 份待检查文件")

        graph_uploads = st.file_uploader(
            "上传图纸文件",
            type=None,
            accept_multiple_files=True,
            key=f"parameters_graph_uploader_{session_id}",
        )
        if graph_uploads and graph_dir:
            handle_file_upload(graph_uploads, graph_dir)
            st.success(f"已上传 {len(graph_uploads)} 份图纸文件")
        elif graph_uploads:
            st.warning("未配置图纸目录，无法保存图纸文件。")

        st.markdown("---")

        col_controls = st.container()
        with col_controls:
            col_start, col_pause, col_resume, col_stop = st.columns(4)
            with col_start:
                start_disabled = not backend_ready or job_running or job_paused
                if st.button("▶️ 开始", disabled=start_disabled, key=f"parameters_start_{session_id}"):
                    if backend_client is not None:
                        result = backend_client.start_parameters_job(session_id)
                        job_id = result.get("job_id") if isinstance(result, dict) else None
                        if not job_id:
                            fallback = backend_client.list_parameters_jobs(session_id)
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
                if st.button("⏸ 暂停", disabled=not job_running, key=f"parameters_pause_{session_id}"):
                    if backend_client is not None and job_status:
                        job_id = str(job_status.get("job_id"))
                        backend_client.pause_parameters_job(job_id)
                        st.rerun()
            with col_resume:
                if st.button("▶️ 继续", disabled=not job_paused, key=f"parameters_resume_{session_id}"):
                    if backend_client is not None and job_status:
                        job_id = str(job_status.get("job_id"))
                        backend_client.resume_parameters_job(job_id)
                        st.rerun()
            with col_stop:
                if st.button("⏹ 停止", disabled=not (job_running or job_paused), key=f"parameters_stop_{session_id}"):
                    if backend_client is not None and job_status:
                        job_id = str(job_status.get("job_id"))
                        backend_client.stop_parameters_job(job_id)
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
            "conversion": "文件转换",
            "kb_sync": "同步知识库",
            "warmup": "准备比对",
            "compare": "参数比对",
            "aggregate": "汇总结果",
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

            processed = int(job_status.get("processed_chunks") or 0)
            total = int(job_status.get("total_chunks") or 0)
            conversion_weight = 10.0
            bisheng_weight = 85.0
            max_before_results = conversion_weight + bisheng_weight
            stage_name = str(stage).lower()
            status_lower = status_str.lower()
            bisheng_ratio = 0.0
            if total:
                bisheng_ratio = min(max(processed / total, 0.0), 1.0)

            progress_percent = 0.0
            if stage_name == "conversion":
                progress_percent = conversion_weight
            elif stage_name in {"kb_sync", "warmup", "compare", "aggregate"}:
                progress_percent = conversion_weight + bisheng_weight * bisheng_ratio
            else:
                progress_percent = bisheng_weight * bisheng_ratio

            progress_percent = min(progress_percent, max_before_results)

            result_ready = status_lower in {"succeeded", "finished", "completed", "success"}
            if result_ready:
                progress_percent = 100.0

            progress_value = max(0.0, min(progress_percent / 100.0, 1.0))
            progress_label = f"处理进度：{progress_percent:.0f}%"
            st.progress(progress_value, text=progress_label)

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
                    with st.expander(
                        "点击查看具体进展",
                        expanded=status_str in {"queued", "running"},
                    ):
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
                                    label = f"{file_name} · 第{part}段"
                                    if total_parts:
                                        label = f"{file_name} · 第{part}/{total_parts}段"
                                    st.markdown(f"**{label}**")
                                    current_group = header_key
                            role = "user" if kind == "prompt" else "assistant"
                            is_new = sequence not in rendered_sequences
                            with st.chat_message(role):
                                if timestamp:
                                    st.caption(str(timestamp))
                                if message_text:
                                    if is_new:
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
                            rendered_sequences.add(sequence)
                    stream_state["rendered"] = sorted(rendered_sequences)
                    st.session_state[stream_state_key] = stream_state

            logs = job_status.get("logs") or []
            if isinstance(logs, list) and logs:
                with st.expander("后台日志", expanded=False):
                    for entry in logs[-50:]:
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
            st.info("后台服务已连接，上传文件后点击开始即可进行参数一致性检查。")
            st.session_state.pop(stream_state_key, None)
        else:
            st.error("后台服务不可用，请稍后重试。")
            st.session_state.pop(stream_state_key, None)

    if job_running:
        st.caption("页面将在 5 秒后自动刷新以更新后台任务进度…")
        time.sleep(5)
        st.rerun()
