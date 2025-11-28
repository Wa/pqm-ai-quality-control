"""Streamlit UI for the APQP one-click deliverable check (upload management phase)."""
from __future__ import annotations

import os
import re
import shutil
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import streamlit as st

from backend_client import get_backend_client, is_backend_available
from config import CONFIG
from tabs.file_completeness import STAGE_ORDER, STAGE_REQUIREMENTS, STAGE_SLUG_MAP
from tabs.file_elements import (
    DeliverableProfile,
    EvaluationResult,
    PHASE_TO_DELIVERABLES,
    SEVERITY_LABELS,
    SEVERITY_ORDER,
)
from tabs.tab_file_elements_check import (
    _compose_table_key,
    _load_result_from_file,
    _profile_to_payload,
    _render_file_elements_job_fragment,
)
from tabs.shared.file_conversion import (
    process_excel_folder,
    process_pdf_folder,
    process_textlike_folder,
    process_word_ppt_folder,
)
from util import ensure_session_dirs


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


def _recover_apqp_job_status(backend_client, session_id: str, job_state_key: str, job_type: str):
    job_status: Optional[Dict[str, Any]] = None
    job_error: Optional[str] = None

    stored_job_id = st.session_state.get(job_state_key)
    fetch_status = backend_client.get_apqp_job_status if job_type == "parse" else backend_client.get_apqp_classify_job
    list_jobs = backend_client.list_apqp_jobs if job_type == "parse" else backend_client.list_apqp_classify_jobs
    not_found_detail = "未找到解析任务" if job_type == "parse" else "未找到任务"

    if stored_job_id:
        result = fetch_status(stored_job_id)
        if isinstance(result, dict) and result.get("job_id"):
            job_status = result
        elif isinstance(result, dict) and result.get("detail") == not_found_detail:
            st.session_state.pop(job_state_key, None)
        elif isinstance(result, dict) and result.get("status") == "error":
            job_error = str(result.get("message") or "后台任务查询失败")

    if job_status is None:
        result = list_jobs(session_id)
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


def _fetch_result_files(backend_client, session_id: str) -> List[Dict[str, object]]:
    response = backend_client.list_apqp_results(session_id)
    if not isinstance(response, dict) or response.get("status") != "success":
        return []
    entries = response.get("files") or []
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
    return sorted(normalized, key=lambda item: item["modified"], reverse=True)


def _load_file_bytes(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except Exception:
        return None


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
                title = doc.get("source_label") or doc.get("file_name") or os.path.basename(doc.get("path", ""))
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


DELIVERABLE_PROFILE_ALIASES = {
    "初始DFMEA": "DFMEA",
    "更新DFMEA": "DFMEA",
    "初始过程流程图": "过程流程图",
    "更新过程流程图": "过程流程图",
    "初版CP": "控制计划",
    "更新CP": "控制计划",
}


def _resolve_stage_profile(stage_name: str) -> Tuple[Optional[DeliverableProfile], str]:
    stage_profiles = PHASE_TO_DELIVERABLES.get(stage_name, ())
    if not stage_profiles:
        return None, ""

    requirement_candidates = list(STAGE_REQUIREMENTS.get(stage_name, ()))
    profile_pool = {profile.name: profile for profile in stage_profiles}
    selected_profile: Optional[DeliverableProfile] = None
    display_name = ""

    for candidate in requirement_candidates:
        alias = DELIVERABLE_PROFILE_ALIASES.get(candidate, candidate)
        if alias in profile_pool:
            selected_profile = profile_pool[alias]
            display_name = candidate
            break

    if selected_profile is None:
        selected_profile = stage_profiles[0]
        display_name = selected_profile.name

    return selected_profile, display_name


def _gather_stage_sources(stage_summary: Dict[str, Any]) -> List[str]:
    seen: set[str] = set()
    collected: List[str] = []

    def _add(path: str | None) -> None:
        if not path:
            return
        normalized = os.path.normpath(path)
        if normalized in seen:
            return
        if os.path.isfile(normalized):
            seen.add(normalized)
            collected.append(normalized)

    for doc in stage_summary.get("documents") or []:
        paths: Sequence[str] = []
        doc_paths = doc.get("paths") or []
        if isinstance(doc_paths, Sequence) and not isinstance(doc_paths, (str, bytes)):
            paths = [str(item) for item in doc_paths]
        elif doc.get("path"):
            paths = [str(doc.get("path"))]
        for path in paths:
            _add(path)

    parsed_dir = stage_summary.get("parsed_dir")
    if parsed_dir and os.path.isdir(parsed_dir):
        entries = [
            os.path.join(parsed_dir, name)
            for name in os.listdir(parsed_dir)
            if os.path.isfile(os.path.join(parsed_dir, name))
        ]
        entries.sort(key=os.path.getmtime, reverse=True)
        for path in entries:
            _add(path)

    upload_dir = stage_summary.get("upload_dir")
    if upload_dir and os.path.isdir(upload_dir):
        uploads = [
            os.path.join(upload_dir, name)
            for name in os.listdir(upload_dir)
            if os.path.isfile(os.path.join(upload_dir, name)) and not name.startswith(".")
        ]
        uploads.sort(key=os.path.getmtime, reverse=True)
        for path in uploads:
            _add(path)

    return collected


def _slugify_label(label: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z]+", "_", str(label)).strip("_").lower()
    return token or "file"


def _gather_stage_files(stage_summary: Dict[str, Any], encrypted_files: set[str]) -> List[Dict[str, object]]:
    groups: List[Dict[str, object]] = []
    seen_tokens: set[str] = set()
    upload_dir = stage_summary.get("upload_dir")

    documents = stage_summary.get("documents") or []
    encrypted_set = set(encrypted_files or set())

    for doc in documents:
        paths: List[str] = []
        doc_paths = doc.get("paths") or []
        if isinstance(doc_paths, Sequence) and not isinstance(doc_paths, (str, bytes)):
            paths = [str(item) for item in doc_paths]
        elif doc.get("path"):
            paths = [str(doc.get("path"))]

        label = (
            doc.get("source_label")
            or doc.get("file_name")
            or (os.path.basename(paths[0]) if paths else "")
        )
        if not label:
            continue

        token_base = _slugify_label(label)
        token = token_base
        suffix = 1
        while token in seen_tokens:
            suffix += 1
            token = f"{token_base}_{suffix}"
        seen_tokens.add(token)

        sources: List[str] = []
        if upload_dir:
            raw_path = os.path.join(upload_dir, label)
            if os.path.isfile(raw_path):
                sources.append(raw_path)

        for path in paths:
            normalized = os.path.normpath(path)
            if os.path.isfile(normalized) and normalized not in sources:
                sources.append(normalized)

        skip_reason: Optional[str] = None
        if label in encrypted_set:
            skip_reason = "文件已加密，无法评估。"
        elif str(doc.get("status")) != "success":
            skip_reason = str(doc.get("error") or "文件无法参与评估。")
        elif not sources:
            skip_reason = "未找到可用的解析文本，已跳过该文件。"

        groups.append({
            "label": str(label),
            "token": token,
            "sources": sources,
            "skip_reason": skip_reason,
        })

    return groups


def _render_elements_overview(result: EvaluationResult) -> None:
    summary = result.summary_counts
    col_total, col_pass, col_missing = st.columns(3)
    col_total.metric("要素总数", summary.get("total", 0))
    col_pass.metric("已满足", summary.get("pass", 0))
    col_missing.metric("待补充", summary.get("missing", 0))

    severity_stats: Dict[str, Dict[str, int]] = {}
    for item in result.evaluations:
        bucket = severity_stats.setdefault(item.severity, {"total": 0, "pass": 0, "missing": 0})
        bucket["total"] += 1
        if item.status == "pass":
            bucket["pass"] += 1
        else:
            bucket["missing"] += 1

    ordered_levels = [level for level in SEVERITY_ORDER if level in severity_stats]
    if ordered_levels:
        st.markdown("#### 严重度拆解")
        for start in range(0, len(ordered_levels), 3):
            chunk = ordered_levels[start : start + 3]
            cols = st.columns(len(chunk))
            for col, level in zip(cols, chunk):
                data = severity_stats[level]
                label = SEVERITY_LABELS.get(level, level)
                value = f"{data['missing']} 待补 / {data['total']} 项"
                delta = f"已满足 {data['pass']}"
                col.metric(label, value, delta=delta)

    pending_items = [item for item in result.evaluations if item.status != "pass"]
    st.markdown("#### 要素指导")
    if pending_items:
        for item in pending_items:
            severity_label = SEVERITY_LABELS.get(item.severity, item.severity)
            with st.expander(f"⚠️ {item.requirement.name} · {severity_label}", expanded=False):
                st.markdown(f"**要素描述：** {item.requirement.description}")
                st.markdown(f"**当前判断：** {item.message}")
                st.markdown(f"**整改指导：** {item.requirement.guidance or '—'}")
                if item.keyword:
                    st.caption(f"检测关键字：{item.keyword}")
                if item.snippet:
                    st.code(item.snippet, language="text")
    else:
        st.success("所有要素均已满足，无需整改。")


def render_apqp_one_click_check_tab(session_id: Optional[str]) -> None:
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return

    uploads_root = str(CONFIG["directories"]["uploads"])
    generated_root = str(CONFIG["directories"]["generated_files"])
    stage_slugs = {stage_name: STAGE_SLUG_MAP.get(stage_name, stage_name) for stage_name in STAGE_ORDER}
    apqp_parsed_root = os.path.join(generated_root, session_id, "APQP_one_click_check", "parsed_files")

    session_dirs = ensure_session_dirs(
        {
            "elements": os.path.join(uploads_root, "{session_id}", "elements"),
            "generated": str(CONFIG["directories"]["generated_files"]),
        },
        session_id,
    )
    elements_source_dir = session_dirs.get("elements", "")
    elements_parsed_dir = session_dirs.get("generated_file_elements_check_parsed", "")

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
        parse_job_state_key = f"apqp_one_click_job_id_{session_id}"
        classify_job_state_key = f"apqp_one_click_classify_job_id_{session_id}"
        elements_job_state_key = f"apqp_one_click_elements_job_id_{session_id}"
        elements_fragment_state_key = f"apqp_one_click_elements_job_fragment_{session_id}"
        elements_status_cache_key = f"{elements_fragment_state_key}_status"
        elements_debug_cache_key = f"{elements_fragment_state_key}_details"
        elements_result_state_key = f"apqp_one_click_elements_result_{session_id}"
        elements_loaded_path_key = f"apqp_one_click_elements_loaded_{session_id}"
        elements_source_state_key = f"apqp_one_click_elements_sources_{session_id}"
        elements_result_cache_key = f"apqp_one_click_elements_result_cache_{session_id}"
        elements_autorun_key = f"apqp_one_click_elements_autorun_{session_id}"
        pending_state_key = f"apqp_one_click_pending_{session_id}"
        classified_job_key = f"apqp_one_click_classified_job_{session_id}"

        parse_status: Optional[Dict[str, Any]] = None
        classify_status: Optional[Dict[str, Any]] = None
        parse_error: Optional[str] = None
        classify_error: Optional[str] = None
        if backend_ready and backend_client is not None:
            parse_status, parse_error = _recover_apqp_job_status(
                backend_client, session_id, parse_job_state_key, "parse"
            )
            classify_status, classify_error = _recover_apqp_job_status(
                backend_client, session_id, classify_job_state_key, "classify"
            )
        elif not backend_ready:
            parse_error = classify_error = "后台服务未连接"

        active_status = classify_status if (classify_status and classify_status.get("status") in {"queued", "running", "paused"}) else parse_status
        status_str = str(active_status.get("status")) if active_status else ""
        job_paused = status_str == "paused"
        job_active = status_str in {"queued", "running", "paused"}

        classify_log_container = st.container()
        turbo_checkbox = st.checkbox(
            "高性能模式",
            key=turbo_state_key,
            disabled=not backend_ready,
            help="并行调用 ModelScope/云端模型加速分类，涉密文件请谨慎使用。",
        )
        action_cols = st.columns([1, 0.6, 0.6])
        with action_cols[0]:
            classify_button = st.button(
                "运行",
                key=f"apqp_classify_{session_id}",
                disabled=not backend_ready or job_active,
                help="调用大模型基于内容进行归类，支持1对多、多对一匹配。",
            )
        active_job_id = active_status.get("job_id") if active_status else None
        active_job_type = "classify" if active_status is classify_status else ("parse" if active_status is parse_status else None)
        with action_cols[1]:
            pause_disabled = (not backend_ready) or (not active_status) or job_paused
            if st.button(
                "暂停解析",
                key=f"apqp_pause_{session_id}",
                disabled=pause_disabled,
                help="请求后台暂停当前解析/分类任务。",
            ):
                if not backend_ready or backend_client is None or not active_status or not active_job_id:
                    st.error("后台服务不可用或暂无运行中的任务。")
                else:
                    if active_job_type == "classify":
                        resp = backend_client.pause_apqp_classify_job(active_job_id)
                    else:
                        resp = backend_client.pause_apqp_job(active_job_id)
                    if isinstance(resp, dict) and resp.get("job_id"):
                        st.success("已请求暂停任务。")
                        st.rerun()
                    else:
                        st.error(f"暂停失败：{resp}")
        with action_cols[2]:
            resume_disabled = (not backend_ready) or (not active_status) or (not job_paused)
            if st.button(
                "继续解析",
                key=f"apqp_resume_{session_id}",
                disabled=resume_disabled,
                help="恢复已暂停的解析/分类任务。",
            ):
                if not backend_ready or backend_client is None or not active_status or not active_job_id:
                    st.error("后台服务不可用或暂无解析任务。")
                else:
                    if active_job_type == "classify":
                        resp = backend_client.resume_apqp_classify_job(active_job_id)
                    else:
                        resp = backend_client.resume_apqp_job(active_job_id)
                    if isinstance(resp, dict) and resp.get("job_id"):
                        st.success("已请求继续任务。")
                        st.rerun()
                    else:
                        st.error(f"继续失败：{resp}")

        if job_active and not job_paused:
            st.info("后台解析任务正在运行，稍后将自动更新进度。")
        elif job_paused:
            st.info("解析已暂停，可点击继续解析恢复。")
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
                        st.session_state[parse_job_state_key] = job_id
                        st.session_state[pending_state_key] = {
                            "job_id": job_id,
                            "turbo_mode": selected_turbo,
                        }
                        st.session_state.pop(classification_state_key, None)
                        st.session_state.pop(classified_job_key, None)
                        st.session_state[parse_job_state_key] = job_id
                        st.success("已提交后台解析任务，稍后将自动更新进度并分类。")
                        st.rerun()

        pending_info = st.session_state.get(pending_state_key)
        with classify_log_container:
            display_status = active_status or classify_status or parse_status
            if display_status:
                current_status = str(display_status.get("status"))
                progress_val = float(display_status.get("progress") or 0.0)
                progress_pct = int(progress_val * 100)
                bar_col, pct_col = st.columns([9, 1])
                with bar_col:
                    st.progress(progress_val)
                with pct_col:
                    st.markdown(f"**{progress_pct}%**")
                stage_label = display_status.get("stage") or "运行中"
                message = display_status.get("message") or "正在处理..."
                st.info(f"{stage_label} · {message}")
                logs = display_status.get("logs") or []
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
                if current_status == "failed":
                    err = display_status.get("error") or display_status.get("message") or "任务失败"
                    st.error(err)
                    st.session_state.pop(pending_state_key, None)
                elif current_status == "succeeded":
                    if (
                        classify_status
                        and classify_status.get("status") == "succeeded"
                        and st.session_state.get(classified_job_key) != classify_status.get("job_id")
                    ):
                        checkpoint = (classify_status.get("metadata") or {}).get("checkpoint") or {}
                        summary = checkpoint.get("summary") if isinstance(checkpoint, dict) else None
                        if summary:
                            st.session_state[classification_state_key] = summary
                            st.session_state[classified_job_key] = classify_status.get("job_id")
                            st.success("分类完成，结果如下。")
                        else:
                            st.success("分类完成，可在结果文件夹查看详情。")
                        st.session_state.pop(pending_state_key, None)
                    elif (
                        parse_status
                        and display_status.get("job_id") == parse_status.get("job_id")
                        and pending_info
                        and pending_info.get("job_id") == parse_status.get("job_id")
                        and not st.session_state.get(classify_job_state_key)
                    ):
                        with st.spinner("解析完成，正在创建分类任务…"):
                            classify_resp = backend_client.start_apqp_classify_job(
                                session_id,
                                turbo_mode=bool(pending_info.get("turbo_mode")),
                                control_job_id=parse_status.get("job_id"),
                            )
                        if isinstance(classify_resp, dict) and classify_resp.get("job_id"):
                            st.session_state[classify_job_state_key] = classify_resp.get("job_id")
                            st.info("已启动分类任务，稍后将更新进度…")
                            st.rerun()
                        else:
                            detail = ""
                            if isinstance(classify_resp, dict):
                                detail = str(classify_resp.get("detail") or classify_resp.get("message") or "")
                            st.error(f"无法启动分类：{detail or classify_resp}")
                        st.session_state.pop(pending_state_key, None)
                    else:
                        st.success("任务已完成。")
                st.divider()
            else:
                if parse_error:
                    st.warning(parse_error)
                if classify_error:
                    st.warning(classify_error)

        classification_summary = st.session_state.get(classification_state_key)
        if classification_summary:
            st.divider()
            st.subheader("🤖 LLM 文件归类与齐套性判断")
            with st.expander("查看齐套性判断结果", expanded=False):
                _render_classification_results(classification_summary)

            st.divider()
            st.subheader("🧩 交付物要素自动评估")

            elements_turbo = bool(classification_summary.get("turbo_mode"))
            elements_initial_results_dir = os.path.join(
                generated_root, session_id, "APQP_one_click_check", "initial_results_element"
            )
            os.makedirs(elements_initial_results_dir, exist_ok=True)
            stage_options = classification_summary.get("stage_order") or list(STAGE_ORDER)
            if not stage_options:
                stage_options = list(PHASE_TO_DELIVERABLES.keys())

            if stage_options:
                with st.expander("查看要素评估结果", expanded=False):
                    st.caption("分类完成后会按阶段自动启动要素评估，无需额外点击。")
                    auto_run_tokens = st.session_state.setdefault(elements_autorun_key, {})
                    source_map = st.session_state.setdefault(elements_source_state_key, {})
                    result_cache: Dict[str, EvaluationResult] = st.session_state.setdefault(
                        elements_result_cache_key, {}
                    )
                    trigger_token = str(
                        classification_summary.get("timestamp_label")
                        or (classify_status or {}).get("job_id")
                        or (parse_status or {}).get("job_id")
                        or ""
                    )
                    parse_summary = (parse_status or {}).get("summary") if isinstance(parse_status, dict) else None

                    def _stage_encrypted(stage: str) -> set[str]:
                        if not isinstance(parse_summary, dict):
                            return set()
                        return set(
                            (parse_summary.get("stages") or {})
                            .get(stage, {})
                            .get("encrypted_files", [])
                        )

                    for stage_name in stage_options:
                        stage_slug = stage_slugs.get(stage_name, stage_name)
                        stage_summary = (classification_summary.get("stages") or {}).get(stage_name, {})
                        profile, profile_label = _resolve_stage_profile(stage_name)

                        st.markdown(f"#### {stage_name}")
                        if profile:
                            st.caption(
                                f"使用交付物【{profile_label or profile.name}】的要素清单自动评估。"
                            )
                        else:
                            st.warning("当前阶段未配置要素模板，暂无法运行评估。")

                        file_groups = _gather_stage_files(stage_summary, _stage_encrypted(stage_name))
                        if not file_groups:
                            st.info("当前阶段没有可评估的文件。")
                            continue

                        for group in file_groups:
                            label = group["label"]
                            token = group["token"]
                            stage_file_key = f"{stage_slug}_{token}"
                            st.markdown(f"##### 文件：{label}")
                            if group.get("skip_reason"):
                                st.warning(group.get("skip_reason"))
                                continue

                            source_paths = group.get("sources") or []
                            if elements_parsed_dir and elements_parsed_dir not in source_paths:
                                os.makedirs(elements_parsed_dir, exist_ok=True)
                            source_map.setdefault(stage_slug, {})[token] = source_paths

                            job_state_key = f"{elements_job_state_key}_{stage_file_key}"
                            fragment_state_key = f"{elements_fragment_state_key}_{stage_file_key}"
                            status_cache_key = f"{elements_status_cache_key}_{stage_file_key}"
                            _debug_cache_key = f"{elements_debug_cache_key}_{stage_file_key}"
                            result_state_key = f"{elements_result_state_key}_{stage_file_key}"
                            loaded_path_key = f"{elements_loaded_path_key}_{stage_file_key}"

                            elements_job_status: Optional[Dict[str, Any]] = None
                            elements_job_error: Optional[str] = None
                            if backend_ready and backend_client is not None:
                                stored_elements_job = st.session_state.get(job_state_key)
                                if stored_elements_job:
                                    resp = backend_client.get_file_elements_job(stored_elements_job)
                                    if isinstance(resp, dict) and resp.get("job_id"):
                                        elements_job_status = resp
                                    elif isinstance(resp, dict) and resp.get("detail") == "未找到任务":
                                        st.session_state.pop(job_state_key, None)
                                    elif isinstance(resp, dict) and resp.get("status") == "error":
                                        elements_job_error = str(resp.get("message") or "后台任务查询失败")

                                if elements_job_status is None:
                                    resp = backend_client.list_file_elements_jobs(session_id)
                                    if isinstance(resp, list) and resp:
                                        for status in resp:
                                            if not isinstance(status, dict):
                                                continue
                                            metadata = status.get("metadata") or {}
                                            if str(metadata.get("stage")) == str(profile.stage if profile else stage_name) and str(metadata.get("source_file")) == label:
                                                elements_job_status = status
                                                break
                                        if elements_job_status is None:
                                            elements_job_status = resp[0]
                                        if isinstance(elements_job_status, dict) and elements_job_status.get("job_id"):
                                            st.session_state[job_state_key] = elements_job_status.get("job_id")
                                    elif isinstance(resp, dict) and resp.get("status") == "error":
                                        elements_job_error = str(resp.get("message") or "后台任务列表查询失败")
                                elif not backend_ready or backend_client is None:
                                    elements_job_error = "后台服务未连接"

                            status_value = str(elements_job_status.get("status")) if elements_job_status else ""
                            job_running = status_value in {"queued", "running"}

                            stage_tokens = auto_run_tokens.setdefault(stage_slug, {}) if isinstance(auto_run_tokens.get(stage_slug), dict) else {}
                            auto_run_tokens[stage_slug] = stage_tokens
                            auto_token = stage_tokens.get(token)
                            should_submit = False
                            if auto_token != trigger_token and not job_running and not elements_job_status:
                                should_submit = True

                            if should_submit:
                                if backend_ready and backend_client is not None:
                                    payload = _profile_to_payload(profile)
                                    payload.update(
                                        {
                                            "session_id": session_id,
                                            "source_paths": source_paths,
                                            "turbo_mode": elements_turbo,
                                            "initial_results_dir": elements_initial_results_dir,
                                            "result_root_dir": os.path.join(
                                                generated_root, session_id, "APQP_one_click_check"
                                            ),
                                        }
                                    )
                                    response = backend_client.start_file_elements_job(payload)
                                    if isinstance(response, dict) and response.get("job_id"):
                                        st.session_state[job_state_key] = response.get("job_id")
                                        stage_tokens[token] = trigger_token
                                        st.info("已自动提交要素评估任务。")
                                    else:
                                        detail = ""
                                        if isinstance(response, dict):
                                            detail = str(response.get("detail") or response.get("message") or "")
                                        elements_job_error = detail or str(response)
                                else:
                                    elements_job_error = "后台服务未连接"

                            _render_file_elements_job_fragment(
                                backend_ready=backend_ready,
                                backend_client=backend_client,
                                job_state_key=job_state_key,
                                job_status=elements_job_status,
                                job_error=elements_job_error,
                                fragment_state_key=fragment_state_key,
                                status_cache_key=status_cache_key,
                            )

                            live_status = st.session_state.get(status_cache_key)
                            if isinstance(live_status, dict):
                                elements_job_status = live_status
                            status_value = str(elements_job_status.get("status")) if elements_job_status else ""

                            table_key = _compose_table_key(
                                stage_name, f"{profile_label or (profile.name if profile else '')}::{label}"
                            )

                            def _load_latest_result() -> Optional[EvaluationResult]:
                                result_files = elements_job_status.get("result_files") if elements_job_status else None
                                if not result_files:
                                    return None
                                latest_path = str(result_files[0])
                                if not latest_path:
                                    return None
                                loaded_path = st.session_state.get(loaded_path_key)
                                if loaded_path == latest_path and st.session_state.get(result_state_key):
                                    return st.session_state.get(result_state_key)
                                loaded = _load_result_from_file(latest_path)
                                if loaded:
                                    st.session_state[loaded_path_key] = latest_path
                                    st.session_state[result_state_key] = loaded
                                    if table_key:
                                        result_cache[table_key] = loaded
                                return loaded

                            if elements_job_status and status_value in {"succeeded", "failed"}:
                                st.session_state.pop(job_state_key, None)

                            if elements_job_error:
                                st.warning(elements_job_error)

                            active_result: Optional[EvaluationResult] = None
                            if table_key:
                                active_result = result_cache.get(table_key)
                            if not active_result:
                                active_result = st.session_state.get(result_state_key)
                            if status_value == "succeeded":
                                loaded = _load_latest_result()
                                if loaded:
                                    active_result = loaded

                            if active_result:
                                _render_elements_overview(active_result)

                                download_targets = (elements_job_status or {}).get("result_files") or []
                                csv_target = next(
                                    (path for path in download_targets if str(path).lower().endswith(".csv")),
                                    None,
                                )
                                xlsx_target = next(
                                    (path for path in download_targets if str(path).lower().endswith(".xlsx")),
                                    None,
                                )

                                def _download(label: str, path: Optional[str], key_suffix: str, mime: str) -> None:
                                    if path and os.path.isfile(path):
                                        st.download_button(
                                            label,
                                            data=_load_file_bytes(path) or b"",
                                            file_name=os.path.basename(path),
                                            mime=mime,
                                            key=f"apqp_elements_{key_suffix}_{stage_slug}_{token}_{session_id}",
                                        )
                                    else:
                                        st.download_button(
                                            label,
                                            data=b"",
                                            file_name=f"file_elements_result.{key_suffix}",
                                            key=f"apqp_elements_{key_suffix}_{stage_slug}_{token}_{session_id}",
                                            disabled=True,
                                        )

                                col_csv, col_xlsx = st.columns(2)
                                with col_csv:
                                    _download("📥 导出CSV", csv_target, "csv", "text/csv")
                                with col_xlsx:
                                    _download(
                                        "📥 导出Excel",
                                        xlsx_target,
                                        "xlsx",
                                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    )
                            else:
                                st.caption("完成评估后将在此展示要素覆盖情况与整改建议。")
            else:
                st.info("暂无阶段可选，无法发起要素评估。")

        if job_active:
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
        if st.button(
            "🗑️ 删除全部分析结果",
            key=f"apqp_clear_results_{session_id}",
            disabled=not backend_ready,
        ):
            if not backend_ready or backend_client is None:
                st.error("后台服务不可用，无法删除分析结果。")
            else:
                response = backend_client.clear_apqp_results(session_id)
                if isinstance(response, dict) and response.get("status") == "success":
                    deleted = int(response.get("deleted") or 0)
                    st.success(f"已清空分析结果，共删除 {deleted} 个文件。")
                    st.rerun()
                else:
                    detail = ""
                    message = ""
                    if isinstance(response, dict):
                        detail = str(response.get("detail") or "")
                        message = str(response.get("message") or "")
                    st.error(f"删除失败：{detail or message or response}")
        tab_labels = list(STAGE_ORDER) + ["分析结果"]
        stage_tabs = st.tabs(tab_labels)
        for idx, stage_name in enumerate(STAGE_ORDER):
            with stage_tabs[idx]:
                files = _fetch_stage_files(backend_client, session_id, stage_name) if backend_client else []
                if not files:
                    st.write("（未上传）")
                    continue
                for info in files:
                    cols = st.columns([4, 1])
                    cols[0].write(f"📄 {info['name']}")
                    delete_key = f"apqp_delete_{stage_name}_{info['name'].replace(' ', '_')}_{session_id}"
                    if cols[1].button(
                        "删除",
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

        with stage_tabs[-1]:
            result_files = _fetch_result_files(backend_client, session_id) if backend_client else []
            if not result_files:
                st.write("（暂无分析结果）")
            else:
                for info in result_files:
                    cols = st.columns([6, 2])
                    cols[0].write(f"📑 {info['name']}")
                    data = _load_file_bytes(info.get("path") or "")
                    disabled = data is None or not backend_ready
                    cols[1].download_button(
                        "⬇️ 下载",
                        data=data or b"",
                        file_name=info.get("name") or "result",
                        mime="application/octet-stream",
                        disabled=disabled,
                        key=f"apqp_result_download_{info.get('name')}_{session_id}",
                    )

