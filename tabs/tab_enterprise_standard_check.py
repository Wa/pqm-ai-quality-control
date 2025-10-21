"""Streamlit tab for enterprise standard checks."""
from __future__ import annotations

import os
import re
import shutil
import time
from datetime import datetime

import streamlit as st

from backend_client import get_backend_client, is_backend_available
from config import CONFIG
from util import ensure_session_dirs, handle_file_upload

from .enterprise_standard import ENTERPRISE_WORKFLOW_SURFACE, stream_text


def render_enterprise_standard_check_tab(session_id):
    # Handle None session_id (user not logged in)
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return

    st.subheader("🏢 企业标准检查")

    # No CSS width overrides; rely on Streamlit columns like special symbols tab
    # Ensure enterprise directories and a generated output root exist
    base_dirs = {
        "generated": str(CONFIG["directories"]["generated_files"]),
    }
    session_dirs = ensure_session_dirs(base_dirs, session_id)
    try:
        workflow_paths = ENTERPRISE_WORKFLOW_SURFACE.prepare_paths(session_dirs)
    except KeyError as error:
        st.error(f"初始化会话目录失败：{error}")
        return

    standards_dir = workflow_paths.standards_dir
    examined_dir = workflow_paths.examined_dir
    enterprise_out_root = workflow_paths.output_root
    standards_txt_dir = workflow_paths.standards_txt_dir
    examined_txt_dir = workflow_paths.examined_txt_dir
    initial_results_dir = workflow_paths.initial_results_dir
    final_results_dir = workflow_paths.final_results_dir
    checkpoint_dir = workflow_paths.checkpoint_dir

    backend_ready = is_backend_available()
    backend_client = get_backend_client() if backend_ready else None
    job_state_key = f"enterprise_job_id_{session_id}"
    job_status: dict[str, object] | None = None
    job_error: str | None = None
    stream_state_key = f"enterprise_stream_state_{session_id}"
    if backend_ready and backend_client is not None:
        stored_job_id = st.session_state.get(job_state_key)
        if stored_job_id:
            result = backend_client.get_enterprise_job(stored_job_id)
            if isinstance(result, dict) and result.get("job_id"):
                job_status = result
            elif isinstance(result, dict) and result.get("status") == "error":
                job_error = str(result.get("message", "后台任务查询失败"))
            elif isinstance(result, dict) and result.get("detail"):
                job_error = str(result.get("detail"))
                if result.get("detail") == "未找到任务":
                    st.session_state.pop(job_state_key, None)
        if job_status is None:
            result = backend_client.list_enterprise_jobs(session_id)
            if isinstance(result, list) and result:
                job_status = result[0]
                if isinstance(job_status, dict) and job_status.get("job_id"):
                    st.session_state[job_state_key] = job_status.get("job_id")
            elif isinstance(result, dict) and result.get("status") == "error":
                job_error = str(result.get("message", "后台任务列表查询失败"))
    else:
        job_error = "后台服务未连接"

    job_running = bool(job_status and str(job_status.get("status")) in {"queued", "running"})

    # Layout: right column for info, left for main content
    col_main, col_info = st.columns([2, 1])

    with col_info:
        # Right column intentionally limited to file manager and utilities only
        # File manager utilities (mirroring completeness tab behavior)
        def get_file_list(folder):
            if not folder or not os.path.exists(folder):
                return []
            files = []
            for f in os.listdir(folder):
                file_path = os.path.join(folder, f)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    files.append({
                        'name': f,
                        'size': stat.st_size,
                        'modified': stat.st_mtime,
                        'path': file_path
                    })
            # Sort by name then modified time for stability
            return sorted(files, key=lambda x: (x['name'].lower(), x['modified']))

        def format_file_size(size_bytes):
            if size_bytes == 0:
                return "0 B"
            size_names = ["B", "KB", "MB", "GB"]
            i = 0
            while size_bytes >= 1024 and i < len(size_names) - 1:
                size_bytes /= 1024.0
                i += 1
            return f"{size_bytes:.1f} {size_names[i]}"

        def format_timestamp(timestamp):
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')

        def truncate_filename(filename, max_length=40):
            if len(filename) <= max_length:
                return filename
            name, ext = os.path.splitext(filename)
            available_length = max_length - len(ext) - 3
            if available_length <= 0:
                return filename[:max_length-3] + "..."
            truncated_name = name[:available_length] + "..."
            return truncated_name + ext

        # Clear buttons
        col_clear1, col_clear2, col_clear3 = st.columns(3)
        with col_clear1:
            if st.button("🗑️ 清空企业标准文件", key=f"clear_enterprise_std_{session_id}"):
                try:
                    for file in os.listdir(standards_dir):
                        file_path = os.path.join(standards_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    st.success("已清空企业标准文件")
                except Exception as e:
                    st.error(f"清空失败: {e}")
        with col_clear2:
            if st.button("🗑️ 清空待检查文件", key=f"clear_enterprise_exam_{session_id}"):
                try:
                    for file in os.listdir(examined_dir):
                        file_path = os.path.join(examined_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    st.success("已清空待检查文件")
                except Exception as e:
                    st.error(f"清空失败: {e}")
        with col_clear3:
            if st.button("🗑️ 清空分析结果", key=f"clear_enterprise_results_{session_id}"):
                try:
                    deleted_count = 0
                    if os.path.isdir(final_results_dir):
                        for fname in os.listdir(final_results_dir):
                            fpath = os.path.join(final_results_dir, fname)
                            if os.path.isfile(fpath):
                                os.remove(fpath)
                                deleted_count += 1
                    st.success(f"已清空分析结果（{deleted_count} 个文件）")
                except Exception as e:
                    st.error(f"清空失败: {e}")

        # File lists in tabs (fixed order)
        tab_std, tab_exam, tab_results = st.tabs(["企业标准文件", "待检查文件", "分析结果"])
        with tab_std:
            std_files = get_file_list(standards_dir)
            if std_files:
                for file_info in std_files:
                    display_name = truncate_filename(file_info['name'])
                    with st.expander(f"📄 {display_name}", expanded=False):
                        col_i, col_a = st.columns([3, 1])
                        with col_i:
                            st.write(f"**文件名:** {file_info['name']}")
                            st.write(f"**大小:** {format_file_size(file_info['size'])}")
                            st.write(f"**修改时间:** {format_timestamp(file_info['modified'])}")
                        with col_a:
                            delete_key = f"del_std_{file_info['name'].replace(' ', '_').replace('.', '_')}_{session_id}"
                            if st.button("🗑️ 删除", key=delete_key):
                                try:
                                    os.remove(file_info['path'])
                                    st.success(f"已删除: {file_info['name']}")
                                except Exception as e:
                                    st.error(f"删除失败: {e}")
            else:
                st.write("（未上传）")

        with tab_exam:
            exam_files = get_file_list(examined_dir)
            if exam_files:
                for file_info in exam_files:
                    display_name = truncate_filename(file_info['name'])
                    with st.expander(f"📄 {display_name}", expanded=False):
                        col_i, col_a = st.columns([3, 1])
                        with col_i:
                            st.write(f"**文件名:** {file_info['name']}")
                            st.write(f"**大小:** {format_file_size(file_info['size'])}")
                            st.write(f"**修改时间:** {format_timestamp(file_info['modified'])}")
                        with col_a:
                            delete_key = f"del_exam_{file_info['name'].replace(' ', '_').replace('.', '_')}_{session_id}"
                            if st.button("🗑️ 删除", key=delete_key):
                                try:
                                    os.remove(file_info['path'])
                                    st.success(f"已删除: {file_info['name']}")
                                except Exception as e:
                                    st.error(f"删除失败: {e}")
            else:
                st.write("（未上传）")

        with tab_results:
            # List files under generated/<session>/enterprise_standard_check/final_results
            final_dir = final_results_dir
            if os.path.isdir(final_dir):
                final_files = get_file_list(final_dir)
                if final_files:
                    for file_info in final_files:
                        display_name = truncate_filename(file_info['name'])
                        with st.expander(f"📄 {display_name}", expanded=False):
                            col_i, col_a = st.columns([4, 1])
                            with col_i:
                                st.write(f"**文件名:** {file_info['name']}")
                                st.write(f"**大小:** {format_file_size(file_info['size'])}")
                                st.write(f"**修改时间:** {format_timestamp(file_info['modified'])}")
                            with col_a:
                                try:
                                    with open(file_info['path'], 'rb') as _fbin:
                                        _data = _fbin.read()
                                    st.download_button(
                                        label="⬇️ 下载",
                                        data=_data,
                                        file_name=file_info['name'],
                                        mime='application/octet-stream',
                                        key=f"dl_final_{file_info['name'].replace(' ', '_').replace('.', '_')}_{session_id}"
                                    )
                                except Exception as e:
                                    st.error(f"下载失败: {e}")
                                delete_key = f"del_final_{file_info['name'].replace(' ', '_').replace('.', '_')}_{session_id}"
                                if st.button("🗑️ 删除", key=delete_key):
                                    try:
                                        os.remove(file_info['path'])
                                        st.success(f"已删除: {file_info['name']}")
                                    except Exception as e:
                                        st.error(f"删除失败: {e}")
                else:
                    st.write("（暂无分析结果）")
            else:
                st.write("（暂无分析结果目录）")




    with col_main:
        status_area = st.container()
        with status_area:
            if job_status:
                status_value = str(job_status.get("status", "unknown"))
                status_labels = {
                    "queued": "排队中",
                    "running": "运行中",
                    "succeeded": "已完成",
                    "failed": "失败",
                }
                stage = str(job_status.get("stage") or "")
                message = str(job_status.get("message") or "")
                pid = job_status.get("pid")
                st.markdown(f"**后台任务状态：{status_labels.get(status_value, status_value)}**")
                if stage:
                    st.caption(f"当前阶段：{stage}")
                if message:
                    st.write(message)
                if pid:
                    st.caption(f"后台进程ID：{pid}")
                total_chunks = int(job_status.get("total_chunks") or 0)
                processed_chunks = int(job_status.get("processed_chunks") or 0)
                if total_chunks > 0:
                    progress_value = min(max(processed_chunks / total_chunks, 0.0), 1.0)
                    st.progress(progress_value)
                    st.caption(f"进度：{processed_chunks}/{total_chunks} 段")
                elif status_value in {"queued", "running"}:
                    st.progress(0.0)
                result_files = job_status.get("result_files") or []
                if result_files and status_value == "succeeded":
                    st.write("已生成结果文件：")
                    for path in result_files:
                        try:
                            st.write(f"- {os.path.basename(str(path))}")
                        except Exception:
                            st.write(f"- {path}")
                logs = job_status.get("logs")
                if isinstance(logs, list) and logs:
                    expanded = status_value in {"queued", "running"}
                    with st.expander("后台日志", expanded=expanded):
                        for entry in logs[-50:]:
                            if not isinstance(entry, dict):
                                st.write(entry)
                                continue
                            ts = entry.get("ts") or ""
                            level = entry.get("level") or "info"
                            message = entry.get("message") or ""
                            st.write(f"[{ts}] {level}: {message}")
                stream_events = job_status.get("stream_events")
                if isinstance(stream_events, list) and stream_events:
                    stream_state = st.session_state.get(stream_state_key)
                    if not isinstance(stream_state, dict) or stream_state.get("job_id") != job_status.get("job_id"):
                        stream_state = {"job_id": job_status.get("job_id"), "rendered": []}
                    rendered_set = set(stream_state.get("rendered") or [])
                    events_sorted = sorted(
                        [event for event in stream_events if isinstance(event, dict)],
                        key=lambda item: int(item.get("sequence") or 0),
                    )
                    with st.expander("运行输出", expanded=status_value in {"queued", "running"}):
                        current_group: tuple[str, int] | None = None
                        for event in events_sorted:
                            seq = int(event.get("sequence") or 0)
                            is_new = seq not in rendered_set
                            rendered_set.add(seq)
                            file_name = str(event.get("file") or event.get("file_name") or "")
                            part = int(event.get("part") or 0)
                            total_parts = int(event.get("total_parts") or 0)
                            kind = str(event.get("kind") or "info")
                            header_key: tuple[str, int] | None = None
                            if file_name and part:
                                header_key = (file_name, part)
                            if header_key and header_key != current_group:
                                if total_parts > 0:
                                    st.markdown(f"**{file_name} · 第{part}/{total_parts}段**")
                                else:
                                    st.markdown(f"**{file_name} · 第{part}段**")
                                current_group = header_key
                            role = "user" if kind == "prompt" else "assistant"
                            message_text = str(event.get("text") or "")
                            timestamp = event.get("ts")
                            with st.chat_message(role):
                                if timestamp:
                                    st.caption(str(timestamp))
                                if message_text:
                                    if is_new:
                                        placeholder = st.empty()
                                        render_method = "text" if role == "user" else "write"
                                        stream_text(placeholder, message_text, render_method=render_method, delay=0.02)
                                    else:
                                        if role == "user":
                                            st.text(message_text)
                                        else:
                                            st.write(message_text)
                                else:
                                    st.write("(无内容)")
                    stream_state["rendered"] = sorted(rendered_set)
                    st.session_state[stream_state_key] = stream_state
            elif job_error:
                st.warning(job_error)
                st.session_state.pop(stream_state_key, None)
            elif backend_ready:
                st.info("后台服务已连接，点击开始即可在后台运行企业标准检查。")
                st.session_state.pop(stream_state_key, None)
            else:
                st.warning("后台服务不可用，请稍后重试。")
                st.session_state.pop(stream_state_key, None)

        # Two uploaders side by side
        col_std, col_exam = st.columns(2)
        with col_std:
            files_std = st.file_uploader("点击上传企业标准文件", type=None, accept_multiple_files=True, key=f"enterprise_std_{session_id}")
            if files_std:
                handle_file_upload(files_std, standards_dir)
                st.success(f"已上传 {len(files_std)} 个企业标准文件")
        with col_exam:
            files_exam = st.file_uploader("点击上传待检查文件", type=None, accept_multiple_files=True, key=f"enterprise_exam_{session_id}")
            if files_exam:
                handle_file_upload(files_exam, examined_dir)
                st.success(f"已上传 {len(files_exam)} 个待检查文件")

        # Start / Stop / Demo buttons
        btn_col1, btn_col_stop, btn_col2 = st.columns([1, 1, 1])
        with btn_col1:
            start_disabled = (not backend_ready) or job_running
            if st.button("开始", key=f"enterprise_start_button_{session_id}", disabled=start_disabled):
                if not backend_ready or backend_client is None:
                    st.error("后台服务不可用，无法启动企业标准检查。")
                else:
                    response = backend_client.start_enterprise_job(session_id)
                    if isinstance(response, dict) and response.get("job_id"):
                        st.session_state[job_state_key] = response["job_id"]
                        st.success("已提交后台任务，刷新或稍后查看进度。")
                        if hasattr(st, "rerun"):
                            st.rerun()
                        else:
                            st.experimental_rerun()
                    else:
                        detail = ""
                        if isinstance(response, dict):
                            detail = str(response.get("detail") or response.get("message") or "")
                        if not detail:
                            detail = str(response)
                        st.error(f"提交任务失败：{detail}")
                    
        with btn_col_stop:
            if st.button("停止", key=f"enterprise_stop_button_{session_id}"):
                st.info("后台任务在服务器中执行，当前版本暂不支持在此处直接停止。请联系管理员或等待任务完成。")

            if st.button("继续", key=f"enterprise_continue_button_{session_id}"):
                st.info("后台任务会自动继续执行，无需手动恢复。")

        with btn_col2:
            if st.button("演示", key=f"enterprise_demo_button_{session_id}"):
                # Copy demonstration files into the user's enterprise folders (no processing here)
                try:
                    # Locate demonstration root (same convention as other tabs)
                    demo_base_dir = CONFIG["directories"]["cp_files"].parent / "demonstration"
                    demo_enterprise = os.path.join(str(demo_base_dir), "enterprise_standard_files")
                    # Subfolders to copy from → to
                    pairs = [
                        (os.path.join(demo_enterprise, "standards"), standards_dir),
                        (os.path.join(demo_enterprise, "examined_files"), examined_dir),
                        # New: copy demonstration prompt/response chunks into session enterprise output
                        # Entire folders copied under enterprise_out_root
                        (os.path.join(demo_enterprise, "prompt_text_chunks"), os.path.join(enterprise_out_root, "prompt_text_chunks")),
                        (os.path.join(demo_enterprise, "llm responses"), os.path.join(enterprise_out_root, "llm responses")),
                        # New: copy final_results for demo summary
                        (os.path.join(demo_enterprise, "final_results"), final_results_dir),
                        # New: copy pre-made prompted responses and json outputs for demo
                        (os.path.join(demo_enterprise, "prompted_llm responses_and_json"), os.path.join(enterprise_out_root, "prompted_llm responses_and_json")),
                    ]
                    files_copied = 0
                    for src, dst in pairs:
                        if not os.path.exists(src):
                            continue
                    # If source is a directory that we want to mirror (prompt_text_chunks / llm responses / final_results / prompted_llm responses_and_json)
                        if os.path.isdir(src) and (src.endswith("prompt_text_chunks") or src.endswith("llm responses") or src.endswith("final_results") or src.endswith("prompted_llm responses_and_json")):
                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            # Copy whole directory tree into enterprise_out_root subfolder
                            shutil.copytree(src, dst, dirs_exist_ok=True)
                            for root, _, files in os.walk(src):
                                files_copied += len([f for f in files if os.path.isfile(os.path.join(root, f))])
                            continue
                        # Otherwise treat as file list copy (standards / examined_files)
                        for name in os.listdir(src):
                            src_path = os.path.join(src, name)
                            dst_path = os.path.join(dst, name)
                            if os.path.isfile(src_path):
                                os.makedirs(dst, exist_ok=True)
                                shutil.copy2(src_path, dst_path)
                                files_copied += 1
                    # Trigger demo streaming phase
                    st.session_state[f"enterprise_demo_{session_id}"] = True
                    st.success(f"已复制演示文件：{files_copied} 个，开始演示…")
                except Exception as e:
                    st.error(f"演示文件复制失败: {e}")
                # Immediately rerun to render the demo streaming phase in main column
                st.rerun()

        # Demo streaming phase (reads from prepared prompt/response chunks; no LLM calls)
        if st.session_state.get(f"enterprise_demo_{session_id}"):
            # Directories prepared by demo button copy
            prompt_dir = os.path.join(enterprise_out_root, 'prompt_text_chunks')
            resp_dir = os.path.join(enterprise_out_root, 'llm responses')
            final_dir = final_results_dir
            prompted_and_json_dir = os.path.join(enterprise_out_root, 'prompted_llm responses_and_json')
            # Collect prompt chunk files
            prompt_files = []
            try:
                if os.path.isdir(prompt_dir):
                    for f in os.listdir(prompt_dir):
                        if f.lower().endswith('.txt'):
                            prompt_files.append(f)
            except Exception:
                prompt_files = []
            # Natural sort by base name and numeric part index
            _prompt_entries = []
            for _f in prompt_files:
                _m = re.match(r"^(?P<base>.+)_pt(?P<idx>\d+)\.txt$", _f)
                if _m:
                    _prompt_entries.append((_m.group('base').lower(), int(_m.group('idx')), _f))
                else:
                    _prompt_entries.append(("", 0, _f))
            _prompt_entries.sort(key=lambda t: (t[0], t[1]))
            # Render each prompt/response pair in UI (original demo)
            for _, _, fname in _prompt_entries:
                m = re.match(r"^(?P<base>.+)_pt(?P<idx>\d+)\.txt$", fname)
                if not m:
                    continue
                base = m.group('base')
                idx = m.group('idx')
                prompt_path = os.path.join(prompt_dir, fname)
                resp_name = f"response_{base}_pt{idx}.txt"
                resp_path = os.path.join(resp_dir, resp_name)
                # Read prompt content
                try:
                    with open(prompt_path, 'r', encoding='utf-8') as f:
                        prompt_text = f.read()
                except Exception:
                    prompt_text = ""
                # Read response content (optional)
                resp_text = None
                if os.path.isfile(resp_path):
                    try:
                        with open(resp_path, 'r', encoding='utf-8') as f:
                            resp_text = f.read()
                    except Exception:
                        resp_text = None
                col_prompt, col_response = st.columns([1, 1])
                with col_prompt:
                    st.markdown(f"提示词（{base} - 第{idx}部分）")
                    prompt_container = st.container(height=400)
                    with prompt_container:
                        with st.chat_message("user"):
                            prompt_placeholder = st.empty()
                            stream_text(prompt_placeholder, prompt_text, render_method="text", delay=0.1)
                        st.chat_input(placeholder="", disabled=True, key=f"enterprise_demo_prompt_{session_id}_{base}_{idx}")
                with col_response:
                    st.markdown(f"示例比对结果（{base} - 第{idx}部分）")
                    response_container = st.container(height=400)
                    with response_container:
                        with st.chat_message("assistant"):
                            resp_placeholder = st.empty()
                            if resp_text is None:
                                resp_placeholder.info("未找到对应示例结果。")
                            else:
                                stream_text(resp_placeholder, resp_text, render_method="write", delay=0.1)
                            st.chat_input(placeholder="", disabled=True, key=f"enterprise_demo_resp_{session_id}_{base}_{idx}")
            # (Removed hardcoded final report section for demo)
            # End of demo streaming pass; reset the flag
            st.session_state[f"enterprise_demo_{session_id}"] = False

            # New demo rendering: prompted_response_* and corresponding json_* from pre-made folder
            try:
                if os.path.isdir(prompted_and_json_dir):
                    # Find prompted_response_*.txt files and for each, display prompt and its json_ response
                    demo_parts = [f for f in os.listdir(prompted_and_json_dir) if f.startswith('prompted_response_') and f.lower().endswith('.txt')]
                    # Natural sort by extracted name and numeric part index
                    _entries = []
                    for _pf in demo_parts:
                        _m = re.match(r"^prompted_response_(?P<name>.+)_pt(?P<idx>\d+)\.txt$", _pf)
                        if _m:
                            _entries.append((_m.group('name').lower(), int(_m.group('idx')), _pf))
                        else:
                            _entries.append(("", 0, _pf))
                    _entries.sort(key=lambda t: (t[0], t[1]))
                    total_parts = len(_entries)
                    for idx, (_, _, pf) in enumerate(_entries, start=1):
                        p_path = os.path.join(prompted_and_json_dir, pf)
                        # Map to json_<name>.txt in same folder
                        json_name = f"json_{pf[len('prompted_response_'):] }"
                        j_path = os.path.join(prompted_and_json_dir, json_name)
                        # Read prompt
                        try:
                            with open(p_path, 'r', encoding='utf-8') as f:
                                ptext = f.read()
                        except Exception:
                            ptext = ""
                        # Read json response (text form)
                        try:
                            with open(j_path, 'r', encoding='utf-8') as f:
                                jtext = f.read()
                        except Exception:
                            jtext = ""
                        # Two column display
                        col_lp, col_lr = st.columns([1, 1])
                        with col_lp:
                            st.markdown(f"生成汇总表格提示词（第{idx}部分，共{total_parts}部分）")
                            pc = st.container(height=400)
                            with pc:
                                with st.chat_message("user"):
                                    ph = st.empty()
                                    stream_text(ph, ptext, render_method="text", delay=0.1)
                            st.chat_input(placeholder="", disabled=True, key=f"enterprise_demo_prompted_prompt_{session_id}_{idx}")
                        with col_lr:
                            st.markdown(f"生成汇总表格结果（第{idx}部分，共{total_parts}部分）")
                            rc = st.container(height=400)
                            with rc:
                                with st.chat_message("assistant"):
                                    ph2 = st.empty()
                                    stream_text(ph2, jtext, render_method="write", delay=0.1)
                            st.chat_input(placeholder="", disabled=True, key=f"enterprise_demo_prompted_resp_{session_id}_{idx}")
            except Exception:
                pass

            # Add demo download buttons for CSV/XLSX in final_results
            try:
                if os.path.isdir(final_dir):
                    csv_files = [f for f in os.listdir(final_dir) if f.lower().endswith('.csv')]
                    xlsx_files = [f for f in os.listdir(final_dir) if f.lower().endswith('.xlsx')]
                    def _latest(path_list):
                        if not path_list:
                            return None
                        paths = [os.path.join(final_dir, f) for f in path_list]
                        paths.sort(key=lambda p: os.path.getmtime(p))
                        return paths[-1]
                    latest_csv = _latest(csv_files)
                    latest_xlsx = _latest(xlsx_files)
                    if latest_csv:
                        with open(latest_csv, 'rb') as fcsv:
                            st.download_button(label="下载CSV结果(演示)", data=fcsv.read(), file_name=os.path.basename(latest_csv), mime='text/csv', key=f"demo_download_csv_{session_id}")
                    if latest_xlsx:
                        with open(latest_xlsx, 'rb') as fxlsx:
                            st.download_button(label="下载Excel结果(演示)", data=fxlsx.read(), file_name=os.path.basename(latest_xlsx), mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key=f"demo_download_xlsx_{session_id}")
            except Exception:
                pass

    if job_running:
        st.caption("页面将在 5 秒后自动刷新以更新后台任务进度…")
        time.sleep(5)
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

# The end
