"""Streamlit tab for文件要素检查."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List

import pandas as pd
import streamlit as st

from config import CONFIG
from util import ensure_session_dirs, handle_file_upload

from .file_elements import (
    EvaluationOrchestrator,
    PHASE_TO_DELIVERABLES,
    SEVERITY_LABELS,
    SEVERITY_ORDER,
    parse_deliverable_stub,
)


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
    entries: List[Dict[str, object]] = []
    if not folder or not os.path.isdir(folder):
        return entries
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        stat = os.stat(path)
        entries.append(
            {
                "name": name,
                "path": path,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            }
        )
    entries.sort(key=lambda item: item["modified"], reverse=True)
    return entries


def render_file_elements_check_tab(session_id: str | None) -> None:
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return

    st.subheader("✅ 文件要素检查")
    st.caption(
        "基于APQP标准梳理关键文档要素，帮助识别缺失项并给出整改建议。上传交付物文本后即可启动评估。"
    )

    uploads_root = str(CONFIG["directories"]["uploads"])
    elements_base = os.path.join(uploads_root, "{session_id}", "file_elements")
    base_dirs = {
        "source": os.path.join(elements_base, "source"),
        "parsed": os.path.join(elements_base, "parsed"),
        "generated": str(CONFIG["directories"]["generated_files"]),
    }
    session_dirs = ensure_session_dirs(base_dirs, session_id)
    source_dir = session_dirs.get("source", "")
    parsed_dir = session_dirs.get("parsed", "")
    export_dir = session_dirs.get("generated_file_elements_check", session_dirs.get("generated", ""))

    result_state_key = f"file_elements_result_{session_id}"
    severity_state_key = f"file_elements_severity_{session_id}"
    issue_state_key = f"file_elements_issue_{session_id}"

    stage_options = list(PHASE_TO_DELIVERABLES.keys())
    if not stage_options:
        st.error("未配置APQP阶段，请联系系统管理员。")
        return

    with st.container():
        st.markdown("### 1. 阶段与交付物选择")
        selected_stage = st.selectbox(
            "选择APQP阶段",
            stage_options,
            key=f"file_elements_stage_{session_id}",
            help="阶段列表来源于AIAG APQP流程，可根据项目推进选择。",
        )
        stage_deliverables = PHASE_TO_DELIVERABLES.get(selected_stage, ())
        deliverable_names = [item.name for item in stage_deliverables]
        if not deliverable_names:
            st.info("该阶段尚未配置交付物。请在配置文件中维护后重试。")
            return
        default_index = 0
        selected_deliverable_name = st.selectbox(
            "选择交付物",
            deliverable_names,
            index=default_index,
            key=f"file_elements_deliverable_{session_id}",
            help="交付物要求将用于生成要素清单与评估标准。",
        )
        profile = next(item for item in stage_deliverables if item.name == selected_deliverable_name)
        st.markdown(
            f"**交付物说明：** {profile.description}\n\n"
            f"**标准参考：** {'，'.join(profile.references) if profile.references else '—'}"
        )

    with st.container():
        st.markdown("### 2. 要素要求概览")
        requirement_rows = [
            {
                "要素": req.name,
                "严重度": SEVERITY_LABELS.get(req.severity, req.severity),
                "描述": req.description,
                "核查要点": req.guidance,
            }
            for req in profile.requirements
        ]
        overview_df = pd.DataFrame(requirement_rows)
        st.dataframe(overview_df, use_container_width=True)
        st.caption("严重度标签参考AIAG APQP要求：关键项需优先闭环，提示项用于完善文档可追溯性。")

    with st.container():
        st.markdown("### 3. 评估执行与结果")

        existing_files = _collect_files(source_dir)
        uploaded = st.file_uploader(
            "上传交付物（支持TXT/MD，若为其他格式请提供同名文本解析文件）",
            accept_multiple_files=True,
            key=f"file_elements_upload_{session_id}",
        )
        if uploaded:
            saved = handle_file_upload(uploaded, source_dir)
            if saved:
                st.success(f"已保存 {saved} 个文件至 {source_dir}")
                existing_files = _collect_files(source_dir)

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

        orchestrator = EvaluationOrchestrator(profile)

        def run_evaluation() -> None:
            text, source_file, warnings = parse_deliverable_stub(profile, source_dir, parsed_dir)
            result = orchestrator.evaluate(text, source_file=source_file, warnings=warnings)
            st.session_state[result_state_key] = result
            st.session_state[severity_state_key] = list(SEVERITY_ORDER)
            st.session_state.pop(issue_state_key, None)
            if export_dir:
                try:
                    os.makedirs(export_dir, exist_ok=True)
                    payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
                    target_path = os.path.join(export_dir, "file_elements_evaluation.json")
                    with open(target_path, "w", encoding="utf-8") as handle:
                        handle.write(payload)
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
            if result:
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

            severity_options = [level for level in SEVERITY_ORDER if level in SEVERITY_LABELS]
            current_selection = st.session_state.get(severity_state_key, severity_options)
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
        if missing_items:
            detail_pool = missing_items
        else:
            detail_pool = candidates

        if not detail_pool:
            st.success("所有要素均已满足，无需额外整改。")
            return

        index_to_item = {idx: item for idx, item in enumerate(detail_pool)}
        default_issue = st.session_state.get(issue_state_key, next(iter(index_to_item), 0))
        selected_index = st.selectbox(
            "选择要素",
            options=list(index_to_item.keys()),
            index=0 if default_issue not in index_to_item else list(index_to_item.keys()).index(default_issue),
            format_func=lambda idx: f"{index_to_item[idx].requirement.name}（{SEVERITY_LABELS.get(index_to_item[idx].severity, index_to_item[idx].severity)}）",
            key=f"file_elements_issue_selector_{session_id}",
        )
        st.session_state[issue_state_key] = selected_index
        selected_item = index_to_item[selected_index]

        if selected_item.status == "pass":
            st.success(f"✅ {selected_item.requirement.name}：{selected_item.message}")
        else:
            st.error(f"⚠️ {selected_item.requirement.name}：{selected_item.message}")

        st.markdown(f"**要素描述：** {selected_item.requirement.description}")
        st.markdown(f"**整改指导：** {selected_item.requirement.guidance}")
        if selected_item.keyword:
            st.caption(f"检测关键字：{selected_item.keyword}")
        if selected_item.snippet:
            st.markdown("**上下文摘录：**")
            st.code(selected_item.snippet, language="text")
        else:
            st.caption("未获取到上下文，请在源文件中补充相关内容。")

        st.markdown(
            "如需再次分析，请使用上方“重新评估”按钮；若需对外共享，可导出JSON文件或将结果复制至整改清单。"
        )
