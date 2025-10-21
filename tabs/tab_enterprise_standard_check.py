"""Streamlit tab for enterprise standard checks."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections import OrderedDict
from datetime import datetime

import streamlit as st

from bisheng_client import (
    call_flow_process,
    create_knowledge,
    find_knowledge_id_by_name,
    kb_sync_folder,
    parse_flow_answer,
    split_to_chunks,
    stop_workflow,
)
from config import CONFIG
from util import ensure_session_dirs, handle_file_upload

from .enterprise_standard import (
    KB_MODEL_ID,
    aggregate_outputs,
    cleanup_orphan_txts,
    ENTERPRISE_WORKFLOW_SURFACE,
    estimate_tokens,
    get_bisheng_settings,
    log_llm_metrics,
    persist_compare_outputs,
    preprocess_txt_directories,
    process_archives,
    process_excel_folder,
    process_pdf_folder,
    process_textlike_folder,
    process_word_ppt_folder,
    report_exception,
    stream_text,
    summarize_with_ollama,
)


SETTINGS = get_bisheng_settings()
BISHENG_BASE_URL = SETTINGS.base_url
BISHENG_INVOKE_PATH = SETTINGS.invoke_path
BISHENG_STOP_PATH = SETTINGS.stop_path
BISHENG_WORKFLOW_ID = SETTINGS.workflow_id
BISHENG_FLOW_ID = SETTINGS.flow_id
FLOW_INPUT_NODE_ID = SETTINGS.flow_input_node_id
FLOW_MILVUS_NODE_ID = SETTINGS.flow_milvus_node_id
FLOW_ES_NODE_ID = SETTINGS.flow_es_node_id
BISHENG_API_KEY = SETTINGS.api_key
BISHENG_MAX_WORDS = SETTINGS.max_words
BISHENG_TIMEOUT_S = SETTINGS.timeout_s


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
            if st.button("开始", key=f"enterprise_start_button_{session_id}"):
                # Process PDFs (MinerU) and Word/PPT (Unstructured) into plain text
                area = st.container()
                with area:
                    # Step 0: Clean orphan .txt files that don't correspond to current uploads
                    try:
                        removed_std = cleanup_orphan_txts(standards_dir, standards_txt_dir, st)
                        removed_exam = cleanup_orphan_txts(examined_dir, examined_txt_dir, st)
                        if removed_std or removed_exam:
                            st.info(f"已清理无关文本 {removed_std + removed_exam} 个")
                    except Exception:
                        pass
                    # Step 0b: Clear previous run results so current run writes fresh outputs
                    try:
                        initial_dir = initial_results_dir
                        os.makedirs(initial_dir, exist_ok=True)
                        cleared = 0
                        for fname in os.listdir(initial_dir):
                            fpath = os.path.join(initial_dir, fname)
                            if os.path.isfile(fpath):
                                try:
                                    os.remove(fpath)
                                    cleared += 1
                                except Exception:
                                    pass
                        if cleared:
                            st.info(f"已清空上次运行结果 {cleared} 个文件")
                    except Exception:
                        pass
                    st.markdown("**阅读企业标准文件中，10分钟左右，请等待...**")
                    created_std_pdf = process_pdf_folder(standards_dir, standards_txt_dir, st, annotate_sources=True)
                    created_std_wp = process_word_ppt_folder(standards_dir, standards_txt_dir, st, annotate_sources=True)
                    created_std_xls = process_excel_folder(standards_dir, standards_txt_dir, st, annotate_sources=True)
                    # New: copy text-like files directly to standards_txt
                    created_std_text = process_textlike_folder(standards_dir, standards_txt_dir, st)
                    # New: extract archives and process recursively
                    _ = process_archives(standards_dir, standards_txt_dir, st)
                    st.markdown("**阅读待检查文件中，10分钟左右，请等待...**")
                    created_exam_pdf = process_pdf_folder(examined_dir, examined_txt_dir, st, annotate_sources=False)
                    created_exam_wp = process_word_ppt_folder(examined_dir, examined_txt_dir, st, annotate_sources=False)
                    created_exam_xls = process_excel_folder(examined_dir, examined_txt_dir, st, annotate_sources=False)
                    # New: copy text-like files directly to examined_txt
                    created_exam_text = process_textlike_folder(examined_dir, examined_txt_dir, st)
                    # New: extract archives and process recursively
                    _ = process_archives(examined_dir, examined_txt_dir, st)

                    try:
                        updated_txts = preprocess_txt_directories(standards_txt_dir, examined_txt_dir)
                        if updated_txts:
                            for _updated_path in updated_txts[:5]:
                                st.info(f"已优化文本：{_updated_path.name}")
                            if len(updated_txts) > 5:
                                st.info(f"还有 {len(updated_txts) - 5} 个文本已优化…")
                    except Exception as error:
                        report_exception("文本预处理失败", error, level="warning")

                    # If we have any txt, switch to running phase and rerun so streaming renders in main column
                    try:
                        std_txt_files = [f for f in os.listdir(standards_txt_dir) if f.lower().endswith('.txt')] if os.path.isdir(standards_txt_dir) else []
                        exam_txt_files = [f for f in os.listdir(examined_txt_dir) if f.lower().endswith('.txt')] if os.path.isdir(examined_txt_dir) else []
                        if not exam_txt_files:
                            st.warning("未发现待检查的 .txt 文本，跳过企业标准比对。")
                        else:
                            # --- Checkpoint preparation: generate prompts for all chunks and manifest ---
                            try:
                                os.makedirs(checkpoint_dir, exist_ok=True)
                                # If previous manifest exists and all entries are done, clear checkpoint files
                                try:
                                    _manifest_path = os.path.join(checkpoint_dir, 'manifest.json')
                                    if os.path.isfile(_manifest_path):
                                        with open(_manifest_path, 'r', encoding='utf-8') as _mf:
                                            _prev_manifest = json.load(_mf) or {}
                                        _entries_prev = _prev_manifest.get('entries') or []
                                        _all_done = bool(_entries_prev) and all((str(e.get('status')) == 'done') for e in _entries_prev)
                                        if _all_done:
                                            for _fn in os.listdir(checkpoint_dir):
                                                _fp = os.path.join(checkpoint_dir, _fn)
                                                try:
                                                    if os.path.isfile(_fp):
                                                        os.remove(_fp)
                                                except Exception:
                                                    pass
                                except Exception:
                                    pass
                                # Build run_id based on current examined_txt files (name|size|mtime)
                                try:
                                    infos = []
                                    for _n in sorted(exam_txt_files, key=lambda x: x.lower()):
                                        _pp = os.path.join(examined_txt_dir, _n)
                                        try:
                                            _st = os.stat(_pp)
                                            infos.append(f"{_n}|{_st.st_size}|{int(_st.st_mtime)}")
                                        except Exception:
                                            infos.append(f"{_n}|0|0")
                                    run_id = hashlib.sha1("\n".join(infos).encode('utf-8', errors='ignore')).hexdigest()
                                except Exception:
                                    run_id = ""
                                manifest = {"run_id": run_id, "entries": []}
                                # The same prompt prefix used later in streaming phase
                                entry_id = 0
                                for _fname in sorted(exam_txt_files, key=lambda x: x.lower()):
                                    _src = os.path.join(examined_txt_dir, _fname)
                                    try:
                                        with open(_src, 'r', encoding='utf-8') as _f:
                                            _doc_text = _f.read()
                                    except Exception:
                                        _doc_text = ""
                                    _chunks = split_to_chunks(_doc_text, int(BISHENG_MAX_WORDS))
                                    for _ci, _piece in enumerate(_chunks):
                                        entry_id += 1
                                        _prompt_text = ENTERPRISE_WORKFLOW_SURFACE.build_chunk_prompt(_piece)
                                        # Use 1-based numbering per chunk within file for filenames, new convention
                                        _num = _ci + 1
                                        _prompt_name = f"checkpoint_prompt_{_fname}_pt{_num}.txt"
                                        _resp_name = f"checkpoint_response_{_fname}_pt{_num}.txt"
                                        _prompt_path = os.path.join(checkpoint_dir, _prompt_name)
                                        _response_path = os.path.join(checkpoint_dir, _resp_name)
                                        try:
                                            with open(_prompt_path, 'w', encoding='utf-8') as _pf:
                                                _pf.write(_prompt_text)
                                        except Exception:
                                            pass
                                        manifest["entries"].append({
                                            "id": entry_id,
                                            "file_name": _fname,
                                            "chunk_index": _ci,
                                            "prompt_file": _prompt_name,
                                            "response_file": _resp_name,
                                            "status": "not_done"
                                        })
                                # Write manifest.json atomically (paths stored as relative filenames for portability)
                                try:
                                    _manifest_path = os.path.join(checkpoint_dir, 'manifest.json')
                                    with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8', dir=checkpoint_dir) as _tf:
                                        _tf.write(json.dumps(manifest, ensure_ascii=False, indent=2))
                                        _tmpname = _tf.name
                                    shutil.move(_tmpname, _manifest_path)
                                except Exception:
                                    pass
                            except Exception:
                                pass

                            st.session_state[f"enterprise_continue_running_{session_id}"] = False
                            st.session_state[f"enterprise_running_{session_id}"] = True
                            st.session_state[f"enterprise_std_txt_files_{session_id}"] = std_txt_files
                            st.session_state[f"enterprise_exam_txt_files_{session_id}"] = exam_txt_files
                            st.rerun()
                    except Exception as e:
                        st.error(f"企业标准比对流程异常：{e}")
                    
        with btn_col_stop:
            if st.button("停止", key=f"enterprise_stop_button_{session_id}"):
                st.session_state[f"enterprise_running_{session_id}"] = False
                try:
                    # Load current bisheng session id if any
                    bs_key = f"bisheng_session_{session_id}"
                    bisheng_sid = st.session_state.get(bs_key)
                    bs_cfg = CONFIG.get('bisheng', {})
                    base_url = (st.session_state.get(f"bisheng_{session_id}_base_url")
                                or os.getenv('BISHENG_BASE_URL') or bs_cfg.get('base_url') or 'http://10.31.60.11:3001')
                    stop_path = (st.session_state.get(f"bisheng_{session_id}_stop_path")
                                or os.getenv('BISHENG_STOP_PATH') or bs_cfg.get('stop_path') or '/api/v2/workflow/stop')
                    api_key = (st.session_state.get(f"bisheng_{session_id}_api_key")
                                or os.getenv('BISHENG_API_KEY') or bs_cfg.get('api_key') or '')
                    if not bisheng_sid:
                        st.info("当前无活动会话可停止。")
                    else:
                        res = stop_workflow(base_url, stop_path, bisheng_sid, api_key or None)
                        st.success(f"已请求停止，响应：{res}")
                except Exception as e:
                    st.error(f"停止失败：{e}")

            # Continue 按钮：始终可见。若无可续条目，提示信息；否则切换到运行分支消费 checkpoint。
            if st.button("继续", key=f"enterprise_continue_button_{session_id}"):
                if st.session_state.get(f"enterprise_running_{session_id}"):
                    st.info("当前流程正在运行，请先停止再继续。")
                else:
                    manifest_path = os.path.join(checkpoint_dir, 'manifest.json')
                    _has_entries = False
                    try:
                        if os.path.isfile(manifest_path):
                            with open(manifest_path, 'r', encoding='utf-8') as _mf:
                                _m = json.load(_mf) or {}
                                _ents = _m.get('entries') or []
                                _has_entries = len(_ents) > 0
                    except Exception:
                        _has_entries = False
                    if not _has_entries:
                        st.info("未发现跑到一半的项目")
                    else:
                        st.session_state[f"enterprise_running_{session_id}"] = False
                        st.session_state[f"enterprise_continue_running_{session_id}"] = True
                        st.rerun()

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

        # Render streaming phase in main column after rerun (mirrors special_symbols pattern)
        if st.session_state.get(f"enterprise_running_{session_id}"):
            # Retrieve context saved before rerun
            std_txt_files = st.session_state.get(f"enterprise_std_txt_files_{session_id}") or []
            exam_txt_files = st.session_state.get(f"enterprise_exam_txt_files_{session_id}") or []

            # Upload standards once (optional)
            if std_txt_files:
                with st.status("Sync standards to KB...", expanded=False) as status:
                    try:
                        kb_name_dyn = ENTERPRISE_WORKFLOW_SURFACE.knowledge_base_name(session_id)
                        kid = find_knowledge_id_by_name(BISHENG_BASE_URL, BISHENG_API_KEY or None, kb_name_dyn)
                        if not kid:
                            kid = create_knowledge(BISHENG_BASE_URL, BISHENG_API_KEY or None, kb_name_dyn, model=str(KB_MODEL_ID))
                        if kid:
                            res = kb_sync_folder(
                                base_url=BISHENG_BASE_URL,
                                api_key=BISHENG_API_KEY or None,
                                knowledge_id=int(kid),
                                folder_path=standards_txt_dir,
                                clear_first=False,
                                chunk_size=1000,
                                chunk_overlap=0,
                                separators=["\n\n", "\n"],
                                separator_rule=["after", "after"],
                            )
                            status.update(label=f"KB sync: uploaded {len(res.get('uploaded', []))}, deleted {len(res.get('deleted', []))}, skipped {len(res.get('skipped', []))}", state="complete")
                        else:
                            status.update(label="KB create/lookup failed (check server auth)", state="error")
                    except Exception as e:
                        status.update(label=f"KB sync failed: {e}", state="error")

            # Iterate examined texts
            exam_txt_files.sort(key=lambda x: x.lower())
            bisheng_session_id = st.session_state.get(f"bisheng_session_{session_id}")
            initial_dir = initial_results_dir
            os.makedirs(initial_dir, exist_ok=True)
            # 预热步骤：在并发开始前，串行对 Bisheng Flow 发起一次极短请求，促使检索/LLM 初始化与缓存
            # 说明：首次请求常见的冷启动（模型加载、连接池、检索索引唤醒）会导致首批并发请求失败率上升；
            # 通过一次轻量的预热，可以显著降低“第一批全挂”的概率。返回内容无需使用。
            try:
                if not st.session_state.get(f"enterprise_warmup_done_{session_id}"):
                    warmup_prompt = ENTERPRISE_WORKFLOW_SURFACE.warmup_prompt or "预热：请简短回复 'gotcha' 即可。"
                    _ = call_flow_process(
                        base_url=BISHENG_BASE_URL,
                        flow_id=BISHENG_FLOW_ID,
                        question=warmup_prompt,
                        kb_id=None,
                        input_node_id=FLOW_INPUT_NODE_ID,
                        api_key=BISHENG_API_KEY or None,
                        session_id=None,
                        history_count=0,
                        extra_tweaks={"CombineDocsChain-520ca": {"token_max": 5000}},
                        milvus_node_id=FLOW_MILVUS_NODE_ID,
                        es_node_id=FLOW_ES_NODE_ID,
                        timeout_s=60,
                        max_retries=0,
                        # clear_cache=True,
                    )
                    st.session_state[f"enterprise_warmup_done_{session_id}"] = True
            except Exception:
                pass
            # Start fresh: original per-file splitting and live prompting
            for idx_file, name in enumerate(exam_txt_files, start=1):
                src_path = os.path.join(examined_txt_dir, name)
                st.markdown(f"**📄 正在比对第{idx_file}个文件，共{len(exam_txt_files)}个：{name}**")
                try:
                    with open(src_path, 'r', encoding='utf-8') as f:
                        doc_text = f.read()
                except Exception as e:
                    st.error(f"读取失败：{e}")
                    continue
                if not doc_text.strip():
                    st.info("文件为空，跳过。")
                    continue
                chunks = split_to_chunks(doc_text, int(BISHENG_MAX_WORDS))
                full_out_text = ""
                prompt_texts = []
                for i, piece in enumerate(chunks, start=1):
                    col_prompt, col_response = st.columns([1, 1])
                    prompt_text = ENTERPRISE_WORKFLOW_SURFACE.build_chunk_prompt(piece)
                    prompt_texts.append(prompt_text)
                    with col_prompt:
                        st.markdown(f"提示词（第{i}部分，共{len(chunks)}部分）")
                        prompt_container = st.container(height=400)
                        with prompt_container:
                            with st.chat_message("user"):
                                prompt_placeholder = st.empty()
                                stream_text(prompt_placeholder, prompt_text, render_method="text")
                            st.chat_input(placeholder="", disabled=True, key=f"enterprise_prompt_{session_id}_{idx_file}_{i}")
                    with col_response:
                        st.markdown(f"AI比对结果（第{i}部分，共{len(chunks)}部分）")
                        response_container = st.container(height=400)
                        with response_container:
                            with st.chat_message("assistant"):
                                response_placeholder = st.empty()
                                try:
                                    # Determine per-user KB name and call Flow with tweaks for per-run KB binding
                                    kb_name_dyn = ENTERPRISE_WORKFLOW_SURFACE.knowledge_base_name(session_id)
                                    kid = find_knowledge_id_by_name(BISHENG_BASE_URL, BISHENG_API_KEY or None, kb_name_dyn)
                                    start_ts = time.time()
                                    res = call_flow_process(
                                        base_url=BISHENG_BASE_URL,
                                        flow_id=BISHENG_FLOW_ID,
                                        question=prompt_text,
                                        kb_id=kid,
                                        input_node_id=FLOW_INPUT_NODE_ID,
                                        api_key=BISHENG_API_KEY or None,
                                        session_id=bisheng_session_id,
                                        history_count=0,
                                        extra_tweaks=None,
                                        milvus_node_id=FLOW_MILVUS_NODE_ID,
                                        es_node_id=FLOW_ES_NODE_ID,
                                        timeout_s=180,        # 新增
                                        max_retries=2,        # 新增
                                        # clear_cache=True,
                                        )
                                    ans_text, new_sid = parse_flow_answer(res)
                                    dur_ms = int((time.time() - start_ts) * 1000)
                                    try:
                                                                                log_llm_metrics(
                                            enterprise_out_root,
                                            session_id,
                                            {
                                                "ts": datetime.now().isoformat(timespec="seconds"),
                                                "engine": "bisheng",
                                                "model": "qwen3",
                                                "session_id": bisheng_session_id or "",
                                                "file": name,
                                                "part": i,
                                                "phase": "compare",
                                                "prompt_chars": len(prompt_text or ""),
                                                "prompt_tokens": estimate_tokens(prompt_text or ""),
                                                "output_chars": len(ans_text or ""),
                                                "output_tokens": estimate_tokens(ans_text or ""),
                                                "duration_ms": dur_ms,
                                                "success": 1 if (ans_text or "").strip() else 0,
                                                "error": res.get("error") if isinstance(res, dict) else "",
                                            },
                                        )
                                    except Exception:
                                        pass
                                    if new_sid:
                                        bisheng_session_id = new_sid
                                        st.session_state[f"bisheng_session_{session_id}"] = bisheng_session_id
                                    response_placeholder.write(ans_text or "")
                                    full_out_text += ("\n\n" if full_out_text else "") + (ans_text or "")
                                    # Mark progress in checkpoint only if Bisheng output contains <think>
                                    if '<think>' in (ans_text or ''):
                                        try:
                                            os.makedirs(checkpoint_dir, exist_ok=True)
                                            resp_fname = f"checkpoint_response_{name}_pt{i}.txt"
                                            resp_path = os.path.join(checkpoint_dir, resp_fname)
                                            # atomic write: temp file + move
                                            with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8', dir=checkpoint_dir) as _tf:
                                                _tf.write(ans_text or "")
                                                _tmpname = _tf.name
                                            shutil.move(_tmpname, resp_path)
                                            # load and update manifest
                                            m_path = os.path.join(checkpoint_dir, 'manifest.json')
                                            _m = None
                                            try:
                                                with open(m_path, 'r', encoding='utf-8') as _mf:
                                                    _m = json.load(_mf) or {}
                                            except Exception:
                                                _m = None
                                            if isinstance(_m, dict) and isinstance(_m.get('entries'), list):
                                                for __e in _m['entries']:
                                                    if __e.get('file_name') == name and int(__e.get('chunk_index', -1)) == (i - 1):
                                                        __e['status'] = 'done'
                                                        break
                                                with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8', dir=checkpoint_dir) as _tf:
                                                    _tf.write(json.dumps(_m, ensure_ascii=False, indent=2))
                                                    _tmpname = _tf.name
                                                shutil.move(_tmpname, m_path)
                                        except Exception:
                                            pass
                                except Exception as e:
                                    response_placeholder.error(f"调用失败：{e}")
                            st.chat_input(placeholder="", disabled=True, key=f"enterprise_response_{session_id}_{idx_file}_{i}")
                # Persist per-file combined output
                try:
                    name_no_ext = os.path.splitext(name)[0]
                    persist_compare_outputs(initial_dir, name_no_ext, prompt_texts, full_out_text)
                    summarize_with_ollama(initial_dir, enterprise_out_root, session_id, name_no_ext, full_out_text)
                except Exception as e:
                    st.error(f"保存结果失败：{e}")


            # End of current run; clear running flag (no final aggregation here)
            try:
                st.session_state[f"enterprise_running_{session_id}"] = False
            except Exception as e:
                st.error(f"流程收尾失败：{e}")

            # After LLM step: aggregate json_*_ptN.txt into CSV and XLSX in final_results
            try:
                aggregate_outputs(initial_dir, enterprise_out_root, session_id)
            except Exception as e:
                st.error(f"汇总导出失败：{e}")

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

        # Continue streaming/processing section (consume checkpoint: done → playback; not_done → run)
        if st.session_state.get(f"enterprise_continue_running_{session_id}"):
            # Retrieve context and init dirs
            std_txt_files = st.session_state.get(f"enterprise_std_txt_files_{session_id}") or []
            exam_txt_files = st.session_state.get(f"enterprise_exam_txt_files_{session_id}") or []
            initial_dir = initial_results_dir
            os.makedirs(initial_dir, exist_ok=True)
            os.makedirs(checkpoint_dir, exist_ok=True)
            manifest_path = os.path.join(checkpoint_dir, 'manifest.json')
            _manifest = None
            try:
                with open(manifest_path, 'r', encoding='utf-8') as _mf:
                    _manifest = json.load(_mf) or {}
            except Exception:
                _manifest = None
            if not _manifest or not isinstance(_manifest.get('entries'), list) or not _manifest['entries']:
                st.info("未发现跑到一半的项目")
                st.session_state[f"enterprise_continue_running_{session_id}"] = False
            else:
                # group by file in entry order
                _entries_sorted = sorted((_manifest.get('entries') or []), key=lambda e: int(e.get('id', 0)))
                _groups = OrderedDict()
                for _e in _entries_sorted:
                    fname = str(_e.get('file_name', ''))
                    _groups.setdefault(fname, []).append(_e)
                bisheng_session_id = st.session_state.get(f"bisheng_session_{session_id}")
                for name, _elist in _groups.items():
                    st.markdown(f"**📄 继续比对：{name}**")
                    full_out_text = ""
                    prompt_texts = []
                    _total_parts = len(_elist)
                    try:
                        _elist.sort(key=lambda e: (int(e.get('chunk_index', 0)), int(e.get('id', 0))))
                    except Exception:
                        pass
                    for _i, _entry in enumerate(_elist, start=1):
                        _prompt_file = str(_entry.get('prompt_file', ''))
                        _response_file = str(_entry.get('response_file', ''))
                        _status = str(_entry.get('status', 'not_done'))
                        _prompt_path = os.path.join(checkpoint_dir, _prompt_file)
                        _response_path = os.path.join(checkpoint_dir, _response_file)
                        # load prompt
                        try:
                            with open(_prompt_path, 'r', encoding='utf-8') as _pf:
                                prompt_text = _pf.read()
                        except Exception:
                            prompt_text = ""
                        prompt_texts.append(prompt_text)
                        col_prompt, col_response = st.columns([1, 1])
                        with col_prompt:
                            st.markdown(f"提示词（第{_i}部分，共{_total_parts}部分）")
                            prompt_container = st.container(height=400)
                            with prompt_container:
                                with st.chat_message("user"):
                                    prompt_placeholder = st.empty()
                                    stream_text(prompt_placeholder, prompt_text, render_method="text")
                            st.chat_input(placeholder="", disabled=True, key=f"enterprise_continue_prompt_{session_id}_{name}_{_i}")
                        with col_response:
                            st.markdown(f"AI比对结果（第{_i}部分，共{_total_parts}部分）")
                            response_container = st.container(height=400)
                            with response_container:
                                with st.chat_message("assistant"):
                                    ph = st.empty()
                                    if _status == 'done':
                                        # playback response
                                        try:
                                            with open(_response_path, 'r', encoding='utf-8') as _rf:
                                                _resp_text = _rf.read()
                                        except Exception:
                                            _resp_text = ""
                                        stream_text(ph, _resp_text, render_method="write", delay=0.1)
                                        full_out_text += ("\n\n" if full_out_text else "") + (_resp_text or "")
                                    else:
                                        # run LLM and update manifest
                                        try:
                                            kb_name_dyn = ENTERPRISE_WORKFLOW_SURFACE.knowledge_base_name(session_id)
                                            kid = find_knowledge_id_by_name(BISHENG_BASE_URL, BISHENG_API_KEY or None, kb_name_dyn)
                                            response_text = ""
                                            res = call_flow_process(
                                                base_url=BISHENG_BASE_URL,
                                                flow_id=BISHENG_FLOW_ID,
                                                question=prompt_text,
                                                kb_id=kid,
                                                input_node_id=FLOW_INPUT_NODE_ID,
                                                api_key=BISHENG_API_KEY or None,
                                                session_id=bisheng_session_id,
                                                history_count=0,
                                                extra_tweaks=None,
                                                milvus_node_id=FLOW_MILVUS_NODE_ID,
                                                es_node_id=FLOW_ES_NODE_ID,
                                                timeout_s=180,
                                                max_retries=2,
                                            )
                                            ans_text, new_sid = parse_flow_answer(res)
                                            if new_sid:
                                                bisheng_session_id = new_sid
                                                st.session_state[f"bisheng_session_{session_id}"] = bisheng_session_id
                                            ph.write(ans_text or "")
                                            full_out_text += ("\n\n" if full_out_text else "") + (ans_text or "")
                                            # save resp and mark done (atomic writes like Start) only if Bisheng output contains <think>
                                            if '<think>' in (ans_text or ''):
                                                try:
                                                    with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8', dir=checkpoint_dir) as _tf:
                                                        _tf.write(ans_text or "")
                                                        _tmpname = _tf.name
                                                    shutil.move(_tmpname, _response_path)
                                                except Exception:
                                                    pass
                                                try:
                                                    for __e in _manifest.get('entries', []):
                                                        if (
                                                            str(__e.get('file_name')) == str(name)
                                                            and int(__e.get('chunk_index', -1)) == int(_entry.get('chunk_index', -1))
                                                        ):
                                                            __e['status'] = 'done'
                                                            break
                                                    with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8', dir=checkpoint_dir) as _tf:
                                                        _tf.write(json.dumps(_manifest, ensure_ascii=False, indent=2))
                                                        _tmpname = _tf.name
                                                    shutil.move(_tmpname, manifest_path)
                                                except Exception:
                                                    pass
                                        except Exception as e:
                                            ph.error(f"调用失败：{e}")
                            st.chat_input(placeholder="", disabled=True, key=f"enterprise_continue_response_{session_id}_{name}_{_i}")
                # write combined prompt/response for this file
                    name_no_ext = os.path.splitext(name)[0]
                    try:
                        persist_compare_outputs(initial_dir, name_no_ext, prompt_texts, full_out_text)
                    except Exception:
                        pass
                    # summarize for this file as well
                    try:
                        summarize_with_ollama(initial_dir, enterprise_out_root, session_id, name_no_ext, full_out_text)
                    except Exception:
                        pass
            # End continue branch
            # After continue: aggregate all outputs to CSV/XLSX/Word like Start does
            try:
                aggregate_outputs(initial_dir, enterprise_out_root, session_id)
            except Exception as e:
                st.error(f"汇总导出失败：{e}")
# The end
