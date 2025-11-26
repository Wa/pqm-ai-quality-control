"""Streamlit UI for the APQP one-click deliverable check (upload management phase)."""
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

from backend_client import get_backend_client, is_backend_available
from config import CONFIG
from tabs.file_completeness import STAGE_ORDER, STAGE_REQUIREMENTS, STAGE_SLUG_MAP
from tabs.shared.file_conversion import (
    process_excel_folder,
    process_pdf_folder,
    process_textlike_folder,
    process_word_ppt_folder,
)


def _format_file_size(size_bytes: int) -> str:
    if size_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    idx = 0
    size = float(size_bytes)
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.1f} {units[idx]}"


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


def _recover_apqp_job_status(backend_client, session_id: str, job_state_key: str):
    job_status: Optional[Dict[str, Any]] = None
    job_error: Optional[str] = None

    stored_job_id = st.session_state.get(job_state_key)
    if stored_job_id:
        result = backend_client.get_apqp_job_status(stored_job_id)
        if isinstance(result, dict) and result.get("job_id"):
            job_status = result
        elif isinstance(result, dict) and result.get("detail") == "未找到解析任务":
            st.session_state.pop(job_state_key, None)
        elif isinstance(result, dict) and result.get("status") == "error":
            job_error = str(result.get("message") or "后台任务查询失败")

    if job_status is None:
        result = backend_client.list_apqp_jobs(session_id)
        if isinstance(result, list) and result:
            job_status = result[0]
            if isinstance(job_status, dict) and job_status.get("job_id"):
                st.session_state[job_state_key] = job_status.get("job_id")
        elif isinstance(result, dict) and result.get("status") == "error":
            job_error = str(result.get("message") or "后台任务列表查询失败")

    return job_status, job_error


def _fetch_stage_files(backend_client, session_id: str, stage_name: str) -> List[Dict[str, object]]:
    stage_slug = STAGE_SLUG_MAP.get(stage_name, stage_name)
    response = backend_client.list_apqp_files(session_id, stage_slug)
    if not isinstance(response, dict) or response.get("status") != "success":
        return []
    files_by_stage = response.get("files") or {}
    entries = files_by_stage.get(stage_name) or []
    normalized: List[Dict[str, object]] = []
    for entry in entries:
        normalized.append(
            {
                "name": entry.get("name"),
                "size": int(entry.get("size", 0)),
                "modified": float(entry.get("modified", 0.0)),
                "path": entry.get("path") or "",
            }
        )
    return sorted(normalized, key=lambda item: (item["name"] or "").lower())


def _render_classification_results(summary: Dict[str, Any]) -> None:
    """Render APQP classification summary in the UI."""

    stage_order = summary.get("stage_order") or []
    stages = summary.get("stages") or {}
    if not stage_order:
        st.info("暂无分类结果。")
        return

    stage_tabs = st.tabs(stage_order)
    for idx, stage_name in enumerate(stage_order):
        stage_data = stages.get(stage_name) or {}
        with stage_tabs[idx]:
            stats = stage_data.get("stats") or {}
            reqs = stage_data.get("requirements") or []
            docs = stage_data.get("documents") or []

            cols = st.columns(4)
            cols[0].metric("应交付物", stats.get("total_requirements", 0))
            cols[1].metric("已覆盖", stats.get("present", 0))
            cols[2].metric("缺失", stats.get("missing", 0))
            cols[3].metric("已分类文件", stats.get("files_classified", 0))

            if stage_data.get("warning"):
                st.warning(stage_data.get("warning"))

            present = [item for item in reqs if item.get("status") == "present"]
            missing = [item for item in reqs if item.get("status") != "present"]

            st.markdown("### 交付物覆盖情况")
            if present:
                st.success(
                    "\n".join(
                        f"✅ {item['name']}（来源: {', '.join(item.get('sources') or ['LLM判定'])}; 置信度: {item.get('confidence', 0):.2f})"
                        for item in present
                    )
                )
            if missing:
                st.error("\n".join(f"⚠️ {item['name']} (未匹配)" for item in missing))
            if not present and not missing:
                st.write("暂无覆盖数据。")

            st.markdown("### 文件分类详情")
            if not docs:
                st.write("暂无文件分类结果。")
            for doc in docs:
                title = doc.get("file_name") or os.path.basename(doc.get("path", ""))
                status = doc.get("status") or ""
                suffix = "" if status == "success" else "（失败）"
                with st.expander(f"📄 {title}{suffix}", expanded=False):
                    if status != "success":
                        st.error(doc.get("error") or "分类失败")
                        continue
                    primary = doc.get("primary_type")
                    additional = doc.get("additional_types") or []
                    matched = doc.get("matched_requirements") or []
                    suggested = doc.get("suggested_types") or []
                    st.write(f"**主匹配:** {primary or 'none'}  ·  置信度 {doc.get('confidence', 0):.2f}")
                    if additional:
                        st.write(f"**额外匹配:** {', '.join(additional)}")
                    if matched:
                        st.caption(f"命中的应交付物：{', '.join(matched)}")
                    if suggested:
                        st.caption(f"未在清单中的候选：{', '.join(suggested)}")
                    st.write(f"**理由:** {doc.get('rationale') or '无'}")
                    st.caption(f"预览字符数：{doc.get('preview_length', 0)}")


def render_apqp_one_click_check_tab(session_id: Optional[str]) -> None:
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return

    uploads_root = str(CONFIG["directories"]["uploads"])
    generated_root = str(CONFIG["directories"]["generated_files"])
    stage_slugs = {stage_name: STAGE_SLUG_MAP.get(stage_name, stage_name) for stage_name in STAGE_ORDER}
    apqp_parsed_root = os.path.join(generated_root, session_id, "APQP_one_click_check", "parsed_files")

    backend_ready = is_backend_available()
    backend_client = get_backend_client() if backend_ready else None

    col_main, col_info = st.columns([2, 1])

    with col_main:
        st.subheader("⚡ APQP交付物一键检查")
        if not backend_ready:
            st.warning("后台服务未连接，解析和删除操作暂不可用。")
        st.markdown(
            "• 第1步：按阶段上传交付物，系统会单独保存各阶段文件。  \n"
            "• 第2步：右侧可以查看、确认或删除已上传文件。  \n"
            "• 第3步：文件分类与齐套性自动分析功能正在开发中，敬请期待。"
        )
        upload_columns = st.columns(2)
        for index, stage_name in enumerate(STAGE_ORDER):
            uploader_key = f"apqp_one_click_uploader_{stage_name}_{session_id}"
            column = upload_columns[index % len(upload_columns)]
            with column:
                uploaded_files = st.file_uploader(
                    f"上传{stage_name}文件",
                    accept_multiple_files=True,
                    key=uploader_key,
                )
                if uploaded_files:
                    if not backend_ready or backend_client is None:
                        st.error("后台服务不可用，无法上传文件。")
                    else:
                        success = 0
                        for file in uploaded_files:
                            resp = backend_client.upload_apqp_file(
                                session_id, stage_slugs.get(stage_name, stage_name), file
                            )
                            if isinstance(resp, dict) and resp.get("status") == "success":
                                success += 1
                            else:
                                detail = ""
                                if isinstance(resp, dict):
                                    detail = str(resp.get("detail") or resp.get("message") or "")
                                st.warning(f"上传 {file.name} 失败：{detail or resp}")
                        if success:
                            st.success(f"已上传 {success} 个文件到 {stage_name}")
                            st.rerun()

                requirements = STAGE_REQUIREMENTS.get(stage_name, ())
                with st.expander(f"{stage_name}应交付物清单", expanded=False):
                    if requirements:
                        st.markdown("\n".join(f"- {item}" for item in requirements))
                    else:
                        st.write("暂无预设清单。")

        st.info("提示：上传的文件会保存到您的专属目录，后续会自动解析并进行齐套性识别。")
        if apqp_parsed_root:
            st.caption(f"解析后的文本文件将保存至 `{apqp_parsed_root}`。")

        classification_state_key = f"apqp_classification_summary_{session_id}"
        turbo_state_key = f"apqp_one_click_turbo_mode_{session_id}"
        job_state_key = f"apqp_one_click_job_id_{session_id}"
        pending_state_key = f"apqp_one_click_pending_{session_id}"
        classified_job_key = f"apqp_one_click_classified_job_{session_id}"

        job_status: Optional[Dict[str, Any]] = None
        job_error: Optional[str] = None
        if backend_ready and backend_client is not None:
            job_status, job_error = _recover_apqp_job_status(backend_client, session_id, job_state_key)
        elif not backend_ready:
            job_error = "后台服务未连接"

        status_str = str(job_status.get("status")) if job_status else ""
        job_running = status_str in {"queued", "running"}

        classify_log_container = st.container()
        turbo_checkbox = st.checkbox(
            "高性能模式",
            key=turbo_state_key,
            disabled=not backend_ready,
            help="并行调用 ModelScope/云端模型加速分类，涉密文件请谨慎使用。",
        )
        classify_button = st.button(
            "运行智能齐套性识别",
            key=f"apqp_classify_{session_id}",
            disabled=not backend_ready or job_running,
            help="调用大模型基于内容进行归类，支持1对多、多对一匹配。",
        )
        if job_running:
            st.info("后台解析任务正在运行，稍后将自动更新进度。")
        if classify_button:
            with classify_log_container:
                if not backend_ready or backend_client is None:
                    st.error("后台服务不可用，无法进行齐套性识别。")
                else:
                    selected_turbo = bool(st.session_state.get(turbo_state_key, turbo_checkbox))
                    if selected_turbo:
                        st.caption("高性能模式将并行提交至 ModelScope/云端通道，若不可用会自动回退到本地串行。")
                    with st.spinner("正在提交解析任务..."):
                        parse_job = backend_client.start_apqp_parse_job(session_id)

                    job_id = parse_job.get("job_id") if isinstance(parse_job, dict) else None
                    if not job_id:
                        detail = ""
                        message = ""
                        if isinstance(parse_job, dict):
                            detail = str(parse_job.get("detail") or "")
                            message = str(parse_job.get("message") or "")
                        st.error(f"无法启动解析：{detail or message or parse_job}")
                    else:
                        st.session_state[job_state_key] = job_id
                        st.session_state[pending_state_key] = {
                            "job_id": job_id,
                            "turbo_mode": selected_turbo,
                        }
                        st.session_state.pop(classification_state_key, None)
                        st.session_state.pop(classified_job_key, None)
                        st.success("已提交后台解析任务，稍后将自动更新进度并分类。")
                        st.rerun()

        pending_info = st.session_state.get(pending_state_key)
        with classify_log_container:
            if job_status:
                progress_bar = st.progress(float(job_status.get("progress") or 0.0))
                stage_label = job_status.get("stage") or "运行中"
                message = job_status.get("message") or "正在解析上传文件..."
                st.info(f"{stage_label} · {message}")
                logs = job_status.get("logs") or []
                if logs:
                    last_log = logs[-1]
                    st.caption(
                        f"{last_log.get('ts', '')} [{last_log.get('level', '')}] {last_log.get('message', '')}"
                    )
                    with st.expander("点击查看后台日志", expanded=False):
                        for entry in logs[-100:]:
                            if not isinstance(entry, dict):
                                st.write(entry)
                                continue
                            ts = entry.get("ts") or ""
                            level = entry.get("level") or "info"
                            log_msg = entry.get("message") or ""
                            st.write(f"[{ts}] {level}: {log_msg}")
                if status_str == "failed":
                    err = job_status.get("error") or job_status.get("message") or "解析任务失败"
                    st.error(err)
                    st.session_state.pop(pending_state_key, None)
                elif status_str == "succeeded" and pending_info and pending_info.get("job_id") == job_status.get("job_id"):
                    already_classified = st.session_state.get(classified_job_key) == job_status.get("job_id")
                    if not already_classified:
                        with st.spinner("解析完成，正在调用大模型分类..."):
                            response = backend_client.classify_apqp_files(
                                session_id, turbo_mode=bool(pending_info.get("turbo_mode"))
                            )
                        if isinstance(response, dict) and response.get("status") == "success":
                            summary = response.get("summary") or {}
                            st.session_state[classification_state_key] = summary
                            st.session_state[classified_job_key] = job_status.get("job_id")
                            st.success("分类完成，结果如下。")
                        else:
                            detail = ""
                            message = ""
                            if isinstance(response, dict):
                                detail = str(response.get("detail") or "")
                                message = str(response.get("message") or "")
                            st.error(f"分类失败：{detail or message or response}")
                        st.session_state.pop(pending_state_key, None)
                elif status_str == "succeeded":
                    st.success("解析已完成，可重新运行分类或查看结果。")
                st.divider()
            elif job_error:
                st.warning(job_error)

        classification_summary = st.session_state.get(classification_state_key)
        if classification_summary:
            st.divider()
            st.subheader("🤖 LLM 文件归类与齐套性判断")
            _render_classification_results(classification_summary)

        if job_running:
            st.caption("页面将在 3 秒后自动刷新以更新后台任务进度…")
            time.sleep(3)
            st.rerun()

    with col_info:
        st.subheader("📁 文件管理")
        st.caption("如果上传的文件没有在此显示，可点击 Ctrl + R 刷新页面。")
        clear_disabled = not backend_ready
        if st.button(
            "🗑️ 删除全部上传文件",
            key=f"apqp_clear_all_{session_id}",
            disabled=clear_disabled,
        ):
            if not backend_ready or backend_client is None:
                st.error("后台服务不可用，无法删除文件。")
            else:
                response = backend_client.clear_apqp_files(session_id, target="all")
                if isinstance(response, dict) and response.get("status") == "success":
                    deleted = int(response.get("deleted") or 0)
                    st.success(f"已清空上传及解析文件（共删除 {deleted} 个条目）。")
                    st.rerun()
                else:
                    detail = ""
                    message = ""
                    if isinstance(response, dict):
                        detail = str(response.get("detail") or "")
                        message = str(response.get("message") or "")
                    st.error(f"删除失败：{detail or message or response}")
        stage_tabs = st.tabs(list(STAGE_ORDER))
        for idx, stage_name in enumerate(STAGE_ORDER):
            with stage_tabs[idx]:
                files = _fetch_stage_files(backend_client, session_id, stage_name) if backend_client else []
                if not files:
                    st.write("（未上传）")
                    continue
                for info in files:
                    display_name = _truncate_filename(info["name"])
                    with st.expander(f"📄 {display_name}", expanded=False):
                        st.write(f"**文件名:** {info['name']}")
                        st.write(f"**大小:** {_format_file_size(int(info['size']))}")
                        st.write(f"**修改时间:** {_format_timestamp(float(info['modified']))}")
                        delete_key = f"apqp_delete_{stage_name}_{info['name'].replace(' ', '_')}_{session_id}"
                        if st.button(
                            "🗑️ 删除",
                            key=delete_key,
                            disabled=not backend_ready,
                        ):
                            if not backend_ready or backend_client is None:
                                st.error("后台服务不可用，无法删除文件。")
                            else:
                                response = backend_client.delete_file(session_id, info["path"])
                                if isinstance(response, dict) and response.get("status") == "success":
                                    st.success(f"已删除: {info['name']}")
                                    st.rerun()
                                else:
                                    detail = ""
                                    message = ""
                                    if isinstance(response, dict):
                                        detail = str(response.get("detail") or "")
                                        message = str(response.get("message") or "")
                                    st.error(f"删除失败：{detail or message or response}")

