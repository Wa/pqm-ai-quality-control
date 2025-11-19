"""Streamlit tab for文件要素检查."""
from __future__ import annotations

import json
import os
from io import BytesIO
from datetime import datetime
from typing import Dict, List

import pandas as pd
import streamlit as st

from config import CONFIG
from tabs.file_completeness import STAGE_ORDER, STAGE_REQUIREMENTS
from util import (
    ensure_session_dirs,
    get_directory_refresh_token,
    handle_file_upload,
    list_directory_contents,
)

from .file_elements import (
    EvaluationOrchestrator,
    PHASE_TO_DELIVERABLES,
    SEVERITY_LABELS,
    SEVERITY_ORDER,
    auto_convert_sources,
    parse_deliverable_stub,
    save_result_payload,
)
from .file_elements.requirement_overview import get_deliverable_overview
from .shared.file_conversion import (
    process_excel_folder,
    process_pdf_folder,
    process_textlike_folder,
    process_word_ppt_folder,
)


CUSTOM_DELIVERABLE_OPTION = "其它（自定义）"
DELIVERABLE_PROFILE_ALIASES = {
    "初始DFMEA": "DFMEA",
    "更新DFMEA": "DFMEA",
    "初始过程流程图": "过程流程图",
    "更新过程流程图": "过程流程图",
    "初版CP": "控制计划",
    "更新CP": "控制计划",
}


def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ["KB", "MB", "GB"]
    size = float(num_bytes)
    idx = -1
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.1f} {units[idx]}"


def _format_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def _collect_files(folder: str) -> List[Dict[str, object]]:
    if not folder:
        return []
    token = get_directory_refresh_token(folder)
    entries = [dict(entry) for entry in list_directory_contents(folder, token)]
    for entry in entries:
        entry.setdefault("path", os.path.normpath(os.path.join(folder, entry["name"])))
    entries.sort(key=lambda item: item["modified"], reverse=True)
    return entries


def _extract_paths(entries: List[Dict[str, object]]) -> List[str]:
    return [item["path"] for item in entries if item.get("path")]


def render_file_elements_check_tab(session_id: str | None) -> None:
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return

    st.subheader("✅ 文件要素检查")
    st.caption(
        "基于APQP标准梳理关键文档要素，帮助识别缺失项并给出整改建议。上传交付物文本后即可启动评估。"
    )

    uploads_root = str(CONFIG["directories"]["uploads"])
    base_dirs = {
        "elements": os.path.join(uploads_root, "{session_id}", "elements"),
        "generated": str(CONFIG["directories"]["generated_files"]),
    }
    session_dirs = ensure_session_dirs(base_dirs, session_id)
    source_dir = session_dirs.get("elements", "")
    parsed_dir = session_dirs.get("generated_file_elements_check", "")
    export_dir = session_dirs.get("generated_file_elements_check", session_dirs.get("generated", ""))

    result_state_key = f"file_elements_result_{session_id}"
    severity_state_key = f"file_elements_severity_{session_id}"
    issue_state_key = f"file_elements_issue_{session_id}"
    paths_state_key = f"file_elements_source_paths_{session_id}"
    export_state_key = f"file_elements_export_path_{session_id}"

    stage_options = list(STAGE_ORDER or [])
    if not stage_options:
        stage_options = list(PHASE_TO_DELIVERABLES.keys())
    for stage_name in PHASE_TO_DELIVERABLES.keys():
        if stage_name not in stage_options:
            stage_options.append(stage_name)
    if not stage_options:
        st.error("未配置APQP阶段，请联系系统管理员。")
        return

    profile = None
    selected_deliverable_display = ""
    overview_metadata: Dict[str, object] | None = None

    with st.container():
        st.markdown("### 1. 阶段与交付物选择")
        col_stage, col_deliverable = st.columns(2)
        with col_stage:
            selected_stage = st.selectbox(
                "选择APQP阶段",
                stage_options,
                key=f"file_elements_stage_{session_id}",
                help="阶段列表来源于AIAG APQP流程，可根据项目推进选择。",
            )

        stage_deliverables = PHASE_TO_DELIVERABLES.get(selected_stage, ())
        completeness_candidates = list(STAGE_REQUIREMENTS.get(selected_stage, ()))
        deliverable_options: List[str] = []
        seen: set[str] = set()
        alias_coverage: set[str] = set()

        def _add_option(name: str) -> None:
            if not name or name in seen:
                return
            alias_name = DELIVERABLE_PROFILE_ALIASES.get(name, name)
            if alias_name in alias_coverage:
                return
            deliverable_options.append(name)
            seen.add(name)
            alias_coverage.add(alias_name)

        for candidate in completeness_candidates:
            _add_option(candidate)
        for profile_candidate in stage_deliverables:
            _add_option(profile_candidate.name)

        if not deliverable_options:
            st.info("该阶段尚未配置交付物，请先在文件完整性台账中维护清单。")
        if CUSTOM_DELIVERABLE_OPTION not in deliverable_options:
            deliverable_options.append(CUSTOM_DELIVERABLE_OPTION)

        profile_name_pool = {item.name for item in stage_deliverables}
        default_index = 0
        for idx, option in enumerate(deliverable_options):
            if option == CUSTOM_DELIVERABLE_OPTION:
                continue
            alias_name = DELIVERABLE_PROFILE_ALIASES.get(option, option)
            if alias_name in profile_name_pool:
                default_index = idx
                break

        with col_deliverable:
            selected_option = st.selectbox(
                "选择交付物",
                deliverable_options,
                index=default_index if deliverable_options else 0,
                key=f"file_elements_deliverable_{session_id}",
                help="交付物要求将用于生成要素清单与评估标准。",
            )

        if selected_option == CUSTOM_DELIVERABLE_OPTION:
            custom_name = st.text_input(
                "输入交付物名称",
                key=f"file_elements_custom_deliverable_{session_id}",
                placeholder="例如：客户特殊要求对照表",
            ).strip()
            selected_deliverable_display = custom_name or "自定义交付物"
        else:
            selected_deliverable_display = selected_option

        normalized_name = DELIVERABLE_PROFILE_ALIASES.get(
            selected_deliverable_display, selected_deliverable_display
        )
        profile = next(
            (item for item in stage_deliverables if item.name == normalized_name),
            None,
        )
        overview_metadata = get_deliverable_overview(selected_deliverable_display)
        references = profile.references if profile and profile.references else overview_metadata[
            "references"
        ]
        summary_text = profile.description if profile else overview_metadata["summary"]
        st.markdown(
            f"**交付物说明：** {summary_text}\n\n"
            f"**标准参考：** {'，'.join(references) if references else '—'}"
        )

    with st.container():
        st.markdown("### 2. 要素要求概览")
        if profile:
            requirement_rows = [
                {
                    "要素": req.name,
                    "严重度": SEVERITY_LABELS.get(req.severity, req.severity),
                    "描述": req.description,
                    "核查要点": req.guidance,
                }
                for req in profile.requirements
            ]
        else:
            requirement_rows = [
                {
                    "要素": row.get("element", ""),
                    "严重度": SEVERITY_LABELS.get(row.get("severity", "major"), row.get("severity", "major")),
                    "描述": row.get("description", ""),
                    "核查要点": row.get("guidance", ""),
                }
                for row in (overview_metadata or {}).get("requirements", [])
            ]
        overview_df = pd.DataFrame(requirement_rows or [{}])
        st.data_editor(
            overview_df,
            use_container_width=True,
            num_rows="dynamic",
            key=f"file_elements_overview_editor_{session_id}_{selected_deliverable_display}",
        )
        st.caption("严重度标签参考AIAG APQP要求，表格可直接编辑并可根据项目自定义补充。")
        if not profile:
            st.info("该交付物尚未配置AI自动评估，目前提供模板以便人工核对。")
            return

    with st.container():
        st.markdown("### 3. 评估执行与结果")

        existing_files = _collect_files(source_dir)
        st.session_state[paths_state_key] = _extract_paths(existing_files)
        uploaded = st.file_uploader(
            "上传交付物（支持TXT/MD/CSV/TSV，PDF/Word/PPT/Excel将自动解析为文本）",
            accept_multiple_files=True,
            key=f"file_elements_upload_{session_id}",
        )
        if uploaded:
            saved = handle_file_upload(uploaded, source_dir)
            if saved:
                st.success(f"已保存 {saved} 个文件至 {source_dir}")
                conversion_area = st.container()
                created, _ = auto_convert_sources(
                    source_dir,
                    parsed_dir,
                    progress_area=conversion_area,
                    annotate_sources=True,
                )
                if created:
                    conversion_area.success(
                        f"已自动解析生成 {len(created)} 个文本文件，供后续评估使用。"
                    )
                existing_files = _collect_files(source_dir)
                st.session_state[paths_state_key] = _extract_paths(existing_files)

        if existing_files:
            file_info_rows = [
                {
                    "文件名": item["name"],
                    "大小": _format_size(int(item["size"])) if isinstance(item["size"], int) else "-",
                    "上传时间": _format_time(float(item["modified"])),
                }
                for item in existing_files
            ]
            st.table(pd.DataFrame(file_info_rows))
        else:
            st.info("暂无上传文件，请先上传交付物文本或对应解析结果。")

        with st.expander("📎 非文本文件解析助手", expanded=False):
            st.markdown(
                "系统会在上传后自动尝试解析PDF、Office与Excel文件。如需手动重试，"
                "或查看进度，可使用下列工具。解析结果统一存入"
                "`generated_files/{session_id}/file_elements_check/`。"
            )
            pdf_status = st.container()
            office_status = st.container()
            excel_status = st.container()
            col_pdf, col_office, col_excel = st.columns(3)
            with col_pdf:
                if st.button(
                    "解析PDF", key=f"file_elements_convert_pdf_{session_id}", help="调用MinerU解析当前目录下的PDF文件。"
                ):
                    if not source_dir or not parsed_dir:
                        pdf_status.warning("目录尚未初始化，请刷新页面或重新登录后重试。")
                    else:
                        created = process_pdf_folder(source_dir, parsed_dir, pdf_status, annotate_sources=True)
                        if created:
                            pdf_status.success(f"成功生成 {len(created)} 个文本文件。")
                        else:
                            pdf_status.info("未解析出新的文本，请确认PDF文件是否存在或已处理。")
            with col_office:
                if st.button(
                    "解析Word/PPT",
                    key=f"file_elements_convert_office_{session_id}",
                    help="调用Unstructured服务解析Word/PPT文件为文本。",
                ):
                    if not source_dir or not parsed_dir:
                        office_status.warning("目录尚未初始化，请刷新页面或重新登录后重试。")
                    else:
                        created = process_word_ppt_folder(source_dir, parsed_dir, office_status, annotate_sources=True)
                        if created:
                            office_status.success(f"成功生成 {len(created)} 个文本文件。")
                        else:
                            office_status.info("未解析出新的文本，请确认文件格式并重试。")
            with col_excel:
                if st.button(
                    "解析Excel/CSV",
                    key=f"file_elements_convert_excel_{session_id}",
                    help="使用pandas将Excel与CSV内容展平为文本。",
                ):
                    if not source_dir or not parsed_dir:
                        excel_status.warning("目录尚未初始化，请刷新页面或重新登录后重试。")
                    else:
                        created = process_excel_folder(source_dir, parsed_dir, excel_status, annotate_sources=True)
                        created.extend(process_textlike_folder(source_dir, parsed_dir, excel_status))
                        if created:
                            excel_status.success(f"成功生成 {len(created)} 个文本文件。")
                        else:
                            excel_status.info("未转换出新的文本，请确认Excel/CSV文件是否存在或已处理。")

        orchestrator = EvaluationOrchestrator(profile)

        def run_evaluation() -> None:
            current_files = _collect_files(source_dir)
            normalized_paths = _extract_paths(current_files)
            st.session_state[paths_state_key] = normalized_paths
            text, source_file, warnings = parse_deliverable_stub(
                profile,
                source_dir,
                parsed_dir,
                source_paths=normalized_paths,
            )
            result = orchestrator.evaluate(text, source_file=source_file, warnings=warnings)
            st.session_state[result_state_key] = result
            available_levels = [
                level
                for level in SEVERITY_ORDER
                if any(item.severity == level for item in result.evaluations)
            ]
            st.session_state[severity_state_key] = available_levels or list(SEVERITY_ORDER)
            st.session_state.pop(issue_state_key, None)
            st.session_state[export_state_key] = None
            if export_dir:
                try:
                    saved_path = save_result_payload(result, export_dir)
                    st.session_state[export_state_key] = saved_path
                except OSError as error:
                    st.warning(f"结果保存失败：{error}")

        col_run, col_rerun, col_export = st.columns([1, 1, 1])
        with col_run:
            if st.button("🚀 运行评估", key=f"file_elements_run_{session_id}"):
                run_evaluation()
        with col_rerun:
            if st.button(
                "🔄 重新评估",
                key=f"file_elements_rerun_{session_id}",
                help="重新加载最新上传的交付物，并刷新评估结果。",
            ):
                run_evaluation()
        with col_export:
            result = st.session_state.get(result_state_key)
            export_path = st.session_state.get(export_state_key)
            if result and export_path and os.path.isfile(export_path):
                with open(export_path, "r", encoding="utf-8") as handle:
                    payload = handle.read()
                st.download_button(
                    "📥 导出JSON",
                    payload.encode("utf-8"),
                    file_name=os.path.basename(export_path),
                    mime="application/json",
                    key=f"file_elements_export_{session_id}",
                    help="导出评估结果以便归档或共享。",
                )
            elif result:
                export_content = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
                st.download_button(
                    "📥 导出JSON",
                    export_content.encode("utf-8"),
                    file_name=f"{profile.id}_file_elements.json",
                    mime="application/json",
                    key=f"file_elements_export_{session_id}",
                    help="导出评估结果以便归档或共享。",
                )
            else:
                st.download_button(
                    "📥 导出JSON",
                    data="",
                    file_name="file_elements.json",
                    disabled=True,
                    key=f"file_elements_export_{session_id}",
                )

        result = st.session_state.get(result_state_key)
        if result:
            for message in result.warnings:
                st.warning(message)

            summary = result.summary_counts
            col_total, col_pass, col_missing = st.columns(3)
            with col_total:
                st.metric("要素总数", summary.get("total", 0))
            with col_pass:
                st.metric("已满足", summary.get("pass", 0))
            with col_missing:
                st.metric("待补充", summary.get("missing", 0))

            severity_stats: Dict[str, Dict[str, int]] = {}
            for item in result.evaluations:
                bucket = severity_stats.setdefault(
                    item.severity, {"total": 0, "pass": 0, "missing": 0}
                )
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

            severity_options = ordered_levels or [level for level in SEVERITY_ORDER if level in SEVERITY_LABELS]
            current_selection = st.session_state.get(severity_state_key, severity_options)
            if not current_selection:
                current_selection = severity_options
            selected_levels = st.multiselect(
                "按严重度筛选",
                options=severity_options,
                default=current_selection,
                format_func=lambda level: SEVERITY_LABELS.get(level, level),
                key=f"file_elements_severity_selector_{session_id}",
            )
            st.session_state[severity_state_key] = selected_levels

            filtered_items = [item for item in result.evaluations if item.severity in selected_levels]
            table_rows = [
                {
                    "要素": item.requirement.name,
                    "严重度": SEVERITY_LABELS.get(item.severity, item.severity),
                    "状态": "✅ 已满足" if item.status == "pass" else "⚠️ 待补充",
                    "说明": item.message,
                }
                for item in filtered_items
            ]
            if table_rows:
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
            else:
                st.info("当前筛选条件下暂无项目。")
        else:
            st.caption("运行评估后将展示结果和指标。")

    with st.container():
        st.markdown("### 4. 问题详情与整改建议")
        result = st.session_state.get(result_state_key)
        if not result:
            st.info("暂无评估结果，请先运行要素评估。")
            return

        severity_filter = st.session_state.get(severity_state_key, list(SEVERITY_ORDER))
        candidates = [item for item in result.evaluations if item.severity in severity_filter]
        missing_items = [item for item in candidates if item.status != "pass"]
        detail_pool = missing_items or candidates

        selected_item = None
        if detail_pool:
            index_to_item = {idx: item for idx, item in enumerate(detail_pool)}
            default_issue = st.session_state.get(issue_state_key, next(iter(index_to_item), 0))
            selected_index = st.selectbox(
                "选择要素",
                options=list(index_to_item.keys()),
                index=0
                if default_issue not in index_to_item
                else list(index_to_item.keys()).index(default_issue),
                format_func=lambda idx: f"{index_to_item[idx].requirement.name}（{SEVERITY_LABELS.get(index_to_item[idx].severity, index_to_item[idx].severity)}）",
                key=f"file_elements_issue_selector_{session_id}",
            )
            st.session_state[issue_state_key] = selected_index
            selected_item = index_to_item[selected_index]
        else:
            st.session_state.pop(issue_state_key, None)
            st.success("所有要素均已满足，无需额外整改。")

        if selected_item:
            detail_col, snippet_col = st.columns((1.4, 1))
            with detail_col:
                severity_label = SEVERITY_LABELS.get(selected_item.severity, selected_item.severity)
                status_prefix = "✅" if selected_item.status == "pass" else "⚠️"
                status_text = "已满足" if selected_item.status == "pass" else "待补充"
                st.markdown(f"{status_prefix} **{selected_item.requirement.name}** — {severity_label} · {status_text}")
                st.markdown(f"**要素描述：** {selected_item.requirement.description}")
                st.markdown(f"**当前判断：** {selected_item.message}")
                st.markdown(f"**整改指导：** {selected_item.requirement.guidance or '—'}")
                if selected_item.keyword:
                    st.caption(f"检测关键字：{selected_item.keyword}")
            with snippet_col:
                if selected_item.snippet:
                    st.markdown("**上下文摘录：**")
                    st.code(selected_item.snippet, language="text")
                else:
                    st.caption("未检索到上下文片段，请在源文档中补充证据。")

        rectify_rows = [
            {
                "要素": item.requirement.name,
                "严重度": SEVERITY_LABELS.get(item.severity, item.severity),
                "当前状态": item.message,
                "整改指导": item.requirement.guidance,
                "建议完成日期": "",
            }
            for item in result.evaluations
            if item.status != "pass"
        ]

        st.markdown("#### 整改跟踪表（自动草稿）")
        if rectify_rows:
            rectify_df = pd.DataFrame(rectify_rows)
            st.dataframe(rectify_df, use_container_width=True)
            csv_data = rectify_df.to_csv(index=False).encode("utf-8-sig")
            excel_buffer = BytesIO()
            rectify_df.to_excel(excel_buffer, index=False, sheet_name="整改清单")
            excel_buffer.seek(0)
            col_csv, col_excel = st.columns(2)
            with col_csv:
                st.download_button(
                    "导出CSV",
                    csv_data,
                    file_name=f"{profile.id}_file_elements_rectify.csv",
                    mime="text/csv",
                    key=f"file_elements_rectify_csv_{session_id}",
                )
            with col_excel:
                st.download_button(
                    "导出Excel",
                    excel_buffer.getvalue(),
                    file_name=f"{profile.id}_file_elements_rectify.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"file_elements_rectify_excel_{session_id}",
                )
        else:
            st.success("暂无需整改项目，所有要素均已满足。")

        st.caption("如需再次分析，请使用上方“重新评估”按钮；若需归档，可结合整改清单与JSON导出共享。")
