"""Streamlit UI for the APQP one-click deliverable check (upload management phase)."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

from backend_client import get_backend_client, is_backend_available
from config import CONFIG
from tabs.file_completeness import STAGE_ORDER, STAGE_REQUIREMENTS, STAGE_SLUG_MAP
from util import (
    ensure_session_dirs,
    get_directory_refresh_token,
    handle_file_upload,
    list_directory_contents,
)
from tabs.shared.file_conversion import (
    process_excel_folder,
    process_pdf_folder,
    process_textlike_folder,
    process_word_ppt_folder,
)


def _list_files(folder: str) -> List[Dict[str, object]]:
    if not folder:
        return []
    token = get_directory_refresh_token(folder)
    entries = [dict(entry) for entry in list_directory_contents(folder, token)]
    for entry in entries:
        entry.setdefault("path", os.path.join(folder, entry["name"]))
        entry["size"] = int(entry.get("size", 0))
        entry["modified"] = float(entry.get("modified", 0.0))
    return sorted(entries, key=lambda item: (item["name"].lower(), item["modified"]))


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
    base_dirs: Dict[str, str] = {
        "generated": str(CONFIG["directories"]["generated_files"]),
    }
    for stage_name in STAGE_ORDER:
        slug = STAGE_SLUG_MAP.get(stage_name, stage_name)
        base_dirs[slug] = os.path.join(uploads_root, "{session_id}", "APQP_one_click_check", slug)
    session_dirs = ensure_session_dirs(base_dirs, session_id)
    stage_dirs = {
        stage_name: session_dirs.get(STAGE_SLUG_MAP.get(stage_name, stage_name), "")
        for stage_name in STAGE_ORDER
    }
    generated_root = session_dirs.get("generated") or session_dirs.get("generated_files") or ""
    apqp_parsed_root = os.path.join(generated_root, "APQP_one_click_check") if generated_root else ""
    if apqp_parsed_root:
        os.makedirs(apqp_parsed_root, exist_ok=True)

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
            target_dir = stage_dirs.get(stage_name)
            column = upload_columns[index % len(upload_columns)]
            with column:
                uploaded_files = st.file_uploader(
                    f"上传{stage_name}文件",
                    accept_multiple_files=True,
                    key=uploader_key,
                )
                if uploaded_files:
                    if target_dir:
                        handle_file_upload(uploaded_files, target_dir)
                        st.rerun()
                    else:
                        st.error("未找到对应的上传目录，请稍后重试。")

                requirements = STAGE_REQUIREMENTS.get(stage_name, ())
                with st.expander(f"{stage_name}应交付物清单", expanded=False):
                    if requirements:
                        st.markdown("\n".join(f"- {item}" for item in requirements))
                    else:
                        st.write("暂无预设清单。")

        st.info("提示：上传的文件会保存到您的专属目录，后续会自动解析并进行齐套性识别。")
        if apqp_parsed_root:
            st.caption(f"解析后的文本文件将保存至 `{apqp_parsed_root}`。")

        parse_log_container = st.container()
        parse_button = st.button(
            "解析所有阶段文件",
            key=f"apqp_parse_all_{session_id}",
            disabled=not backend_ready,
        )
        if parse_button:
            with parse_log_container:
                if not backend_ready or backend_client is None:
                    st.error("后台服务不可用，无法解析文件。")
                else:
                    with st.spinner("正在解析上传的文件，请稍候……"):
                        response = backend_client.parse_apqp_files(session_id)
                    if isinstance(response, dict) and response.get("status") == "success":
                        summary = response.get("summary") or {}
                        stage_order = summary.get("stage_order") or list(STAGE_ORDER)
                        stage_results = summary.get("stages") or {}
                        total_created = int(summary.get("total_created") or 0)
                        if apqp_parsed_root:
                            st.info(f"解析输出根目录：`{apqp_parsed_root}`")
                        if total_created:
                            st.success(f"解析完成，本次共生成 {total_created} 个文本文件。")
                        else:
                            st.info("解析完成，本次未生成新的文本文件。")
                        for stage_name in stage_order:
                            stage_data = stage_results.get(stage_name)
                            if not stage_data:
                                continue
                            with st.expander(f"{stage_name} · 解析日志", expanded=False):
                                upload_dir = stage_data.get("upload_dir") or ""
                                parsed_dir = stage_data.get("parsed_dir") or ""
                                st.write(f"- 上传目录：`{upload_dir}`")
                                st.write(f"- 解析目标目录：`{parsed_dir}`")
                                pdf_count = int(stage_data.get("pdf_created") or 0)
                                office_count = int(stage_data.get("word_ppt_created") or 0)
                                excel_count = int(stage_data.get("excel_created") or 0)
                                text_count = int(stage_data.get("text_created") or 0)
                                total_count = int(stage_data.get("total_created") or 0)
                                files_found = int(stage_data.get("files_found") or 0)
                                st.caption(
                                    "解析统计："
                                    f"PDF {pdf_count} · Word/PPT {office_count} · "
                                    f"Excel {excel_count} · 文本 {text_count} · 总计 {total_count}"
                                )
                                if files_found == 0:
                                    st.info("当前阶段没有上传文件，跳过解析。")
                                messages = stage_data.get("messages") or []
                                for message in messages:
                                    level = str((message or {}).get("level") or "info").lower()
                                    text = str((message or {}).get("text") or "").strip()
                                    if not text:
                                        continue
                                    if level == "warning":
                                        st.warning(text)
                                    elif level == "error":
                                        st.error(text)
                                    elif level == "success":
                                        st.success(text)
                                    else:
                                        st.info(text)
                                if stage_data.get("error"):
                                    st.error(f"阶段解析失败：{stage_data['error']}")
                    else:
                        detail = ""
                        message = ""
                        if isinstance(response, dict):
                            detail = str(response.get("detail") or "")
                            message = str(response.get("message") or "")
                        st.error(f"解析失败：{detail or message or response}")

        classification_state_key = f"apqp_classification_summary_{session_id}"
        classify_log_container = st.container()
        classify_button = st.button(
            "运行智能齐套性识别",
            key=f"apqp_classify_{session_id}",
            disabled=not backend_ready,
            help="调用大模型基于内容进行归类，支持1对多、多对一匹配。",
        )
        if classify_button:
            with classify_log_container:
                if not backend_ready or backend_client is None:
                    st.error("后台服务不可用，无法进行齐套性识别。")
                else:
                    with st.spinner("正在调用大模型分类，请稍候……"):
                        response = backend_client.classify_apqp_files(session_id)
                    if isinstance(response, dict) and response.get("status") == "success":
                        summary = response.get("summary") or {}
                        st.session_state[classification_state_key] = summary
                        st.success("分类完成，结果如下。")
                    else:
                        detail = ""
                        message = ""
                        if isinstance(response, dict):
                            detail = str(response.get("detail") or "")
                            message = str(response.get("message") or "")
                        st.error(f"分类失败：{detail or message or response}")

        classification_summary = st.session_state.get(classification_state_key)
        if classification_summary:
            st.divider()
            st.subheader("🤖 LLM 文件归类与齐套性判断")
            _render_classification_results(classification_summary)

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
                folder = stage_dirs.get(stage_name, "")
                files = _list_files(folder)
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

