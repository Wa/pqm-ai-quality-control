import streamlit as st
import pandas as pd
import os
import io
import zipfile
import json
import re
import requests
import shutil
from util import ensure_session_dirs, handle_file_upload
from config import CONFIG


def _list_pdfs(folder: str):
	"""Return absolute paths for all PDF files in a folder (non-recursive)."""
	try:
		return [
			os.path.join(folder, f)
			for f in os.listdir(folder)
			if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith('.pdf')
		]
	except Exception:
		return []


def _mineru_parse_pdf(pdf_path: str) -> bytes:
	"""Call MinerU API to parse a single PDF and return ZIP bytes on success.

	Raises an exception on failure.
	"""
	api_url = "http://10.31.60.127:8000/file_parse"
	data = {
		'backend': 'vlm-sglang-engine',
		'response_format_zip': 'true',
		# Enable richer outputs; we will primarily consume the .md text for now
		'formula_enable': 'true',
		'table_enable': 'true',
		'return_images': 'false',
		'return_middle_json': 'true',
		'return_model_output': 'false',
		'return_content_list': 'true',
	}
	with open(pdf_path, 'rb') as f:
		files = {'files': (os.path.basename(pdf_path), f, 'application/pdf')}
		resp = requests.post(api_url, data=data, files=files, timeout=300)
		if resp.status_code != 200:
			raise RuntimeError(f"MinerU API error {resp.status_code}: {resp.text[:200]}")
		return resp.content


def _zip_to_txts(zip_bytes: bytes, target_txt_path: str) -> bool:
	"""Extract first .md file from ZIP bytes and save as plain text (.txt).

	Returns True if a .txt was written, False otherwise.
	"""
	# MinerU returns a ZIP archive for each PDF containing: a Markdown file (extracted
	# plain text content), JSONs (structured intermediates), and optionally images.
	# For now we only need plain text for LLM prompts, so we take the first .md file
	# and write it out as a .txt. The images are intentionally ignored here, but they
	# are valuable for future RAG/Q&A over figures or diagrams. We will revisit image
	# handling later to index them alongside text for multimodal retrieval.
	bio = io.BytesIO(zip_bytes)
	try:
		with zipfile.ZipFile(bio) as zf:
			# Prefer top-level or nested .md
			md_members = [n for n in zf.namelist() if n.lower().endswith('.md')]
			if not md_members:
				return False
			# Use the first .md
			name = md_members[0]
			content = zf.read(name)
			# Ensure output directory exists
			os.makedirs(os.path.dirname(target_txt_path), exist_ok=True)
			with open(target_txt_path, 'wb') as out_f:
				out_f.write(content)
			return True
	except zipfile.BadZipFile:
		return False


def _process_pdf_folder(input_dir: str, output_dir: str, progress_area):
	"""Process all PDFs in input_dir via MinerU and write .txts into output_dir."""
	pdf_paths = _list_pdfs(input_dir)
	if not pdf_paths:
		progress_area.info("（无PDF文件可处理）")
		return []
	created = []
	for pdf_path in pdf_paths:
		orig_name = os.path.basename(pdf_path)
		# Preserve original extension in output filename, e.g., name.pdf.txt
		out_txt = os.path.join(output_dir, f"{orig_name}.txt")
		try:
			# Skip if parsed file already exists and is non-empty
			if os.path.exists(out_txt) and os.path.getsize(out_txt) > 0:
				progress_area.info(f"已存在（跳过）: {os.path.basename(out_txt)}")
				continue
			progress_area.write(f"解析: {os.path.basename(pdf_path)} …")
			zip_bytes = _mineru_parse_pdf(pdf_path)
			ok = _zip_to_txts(zip_bytes, out_txt)
			if ok:
				created.append(out_txt)
				progress_area.success(f"已生成: {os.path.basename(out_txt)}")
			else:
				progress_area.warning(f"未发现可用的 .md 内容，跳过: {os.path.basename(pdf_path)}")
		except Exception as e:
			progress_area.error(f"失败: {os.path.basename(pdf_path)} → {e}")
	return created


def _list_word_ppt(folder: str):
	"""Return absolute paths for .doc, .docx, .ppt, .pptx in a folder (non-recursive)."""
	try:
		return [
			os.path.join(folder, f)
			for f in os.listdir(folder)
			if os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[1].lower() in {'.doc', '.docx', '.ppt', '.pptx'}
		]
	except Exception:
		return []


def _unstructured_partition_to_txt(file_path: str, target_txt_path: str) -> bool:
	"""Send a single Word/PPT file to Unstructured API and write plain text (.txt).

	The Unstructured server is expected at 10.31.60.11 running the API. We call the
	"general" endpoint and extract text fields. Table-like data, if present, are
	flattened with tab separators for readability in plain text.

	Future plan (RAG-focused tables):
	- Keep Unstructured for narrative text/structure.
	- Extract tables directly from the original DOCX/PPTX using python-docx/python-pptx.
	- Convert those tables to TSV (one row per line, cells separated by a single tab).
	- Replace or insert TSV blocks into the final output .txt in place of flattened table text.
	This will improve recall/precision on numeric lookups. Not implemented yet; will
	be added later when schedule allows.
	"""
	# Resolve API URL: env var first, then CONFIG.services.unstructured_api_url
	api_url = os.getenv('UNSTRUCTURED_API_URL') or CONFIG.get('services', {}).get('unstructured_api_url') or 'http://10.31.60.11:8000/general/v0/general'
	try:
		with open(file_path, 'rb') as f:
			files = {'files': (os.path.basename(file_path), f)}
			# RAG-optimized defaults: structured tables, auto strategy, Chinese+English OCR support
			form = {
				"strategy": "auto",
				"ocr_languages": "chi_sim,eng",
				"infer_table_structure": "true",
			}
			resp = requests.post(api_url, files=files, data=form, timeout=300)
			if resp.status_code != 200:
				raise RuntimeError(f"Unstructured API {resp.status_code}: {resp.text[:200]}")
			data = resp.json()
			# data is expected to be a list of elements; each may have 'text' or table-like content
			lines = []
			if isinstance(data, list):
				for el in data:
					# Prefer 'text'
					text = None
					if isinstance(el, dict):
						# Common key is 'text'
						text = el.get('text')
						# Some table extractions might be under 'data' (list of rows)
						if not text and isinstance(el.get('data'), list):
							for row in el['data']:
								if isinstance(row, list):
									lines.append('\t'.join(str(c) for c in row))
							# Continue to next element after adding table rows
							continue
					if isinstance(text, str) and text.strip():
						lines.append(text.strip())
			# Write as UTF-8 plain text
			os.makedirs(os.path.dirname(target_txt_path), exist_ok=True)
			with open(target_txt_path, 'w', encoding='utf-8') as out_f:
				out_f.write('\n'.join(lines))
			return True
	except Exception as e:
		# Surface errors to caller via return False; logging via progress UI
		return False


def _process_word_ppt_folder(input_dir: str, output_dir: str, progress_area):
	"""Process .doc/.docx/.ppt/.pptx via Unstructured API and write .txts."""
	paths = _list_word_ppt(input_dir)
	if not paths:
		progress_area.info("（无Word/PPT文件可处理）")
		return []
	created = []
	for p in paths:
		orig_name = os.path.basename(p)
		# Preserve original extension in output filename, e.g., name.docx.txt / name.ppt.txt
		out_txt = os.path.join(output_dir, f"{orig_name}.txt")
		try:
			# Skip if parsed file already exists and is non-empty
			if os.path.exists(out_txt) and os.path.getsize(out_txt) > 0:
				progress_area.info(f"已存在（跳过）: {os.path.basename(out_txt)}")
				continue
			progress_area.write(f"解析(Word/PPT): {os.path.basename(p)} …")
			ok = _unstructured_partition_to_txt(p, out_txt)
			if ok:
				created.append(out_txt)
				progress_area.success(f"已生成: {os.path.basename(out_txt)}")
			else:
				progress_area.warning(f"未能从文件中生成文本，跳过: {os.path.basename(p)}")
		except Exception as e:
			progress_area.error(f"失败: {os.path.basename(p)} → {e}")
	return created


def _list_excels(folder: str):
	"""Return absolute paths for .xls/.xlsx/.xlsm in a folder (non-recursive)."""
	try:
		return [
			os.path.join(folder, f)
			for f in os.listdir(folder)
			if os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[1].lower() in {'.xls', '.xlsx', '.xlsm'}
		]
	except Exception:
		return []


def _sanitize_sheet_name(name: str) -> str:
	"""Sanitize sheet names for filenames: keep readable, remove path-forbidden chars."""
	bad = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
	for ch in bad:
		name = name.replace(ch, '_')
	return '_'.join(name.strip().split())[:80] or 'Sheet'


def _process_excel_folder(input_dir: str, output_dir: str, progress_area):
	"""Convert each Excel sheet to CSV text and save as <file>_SHEET_<sheet>.txt.

	Note: We intentionally save CSV content with a .txt extension for uniform LLM
	consumption. This is technically fine: the content is plain text CSV and the
	file extension does not affect parsing for our use case.
	"""
	paths = _list_excels(input_dir)
	if not paths:
		progress_area.info("（无Excel文件可处理）")
		return []
	created = []
	import pandas as pd
	for excel_path in paths:
		orig_name = os.path.basename(excel_path)  # keep extension in base name per spec
		try:
			xls = pd.ExcelFile(excel_path)
			for sheet in xls.sheet_names:
				safe_sheet = _sanitize_sheet_name(sheet)
				out_txt = os.path.join(output_dir, f"{orig_name}_SHEET_{safe_sheet}.txt")
				# Skip if exists and non-empty
				if os.path.exists(out_txt) and os.path.getsize(out_txt) > 0:
					progress_area.info(f"已存在（跳过）: {os.path.basename(out_txt)}")
					continue
				progress_area.write(f"转换(Excel→CSV): {orig_name} / {sheet} …")
				df = xls.parse(sheet)
				# Write CSV content into .txt
				df.to_csv(out_txt, index=False, encoding='utf-8')
				created.append(out_txt)
				progress_area.success(f"已生成: {os.path.basename(out_txt)}")
		except Exception as e:
			progress_area.error(f"失败: {orig_name} → {e}")
	return created


def render_history_issues_avoidance_tab(session_id):
    # Handle None session_id (user not logged in)
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return
    
    st.subheader("📋 历史问题规避")
    
    # Ensure history issues avoidance directories exist
    base_dirs = {
        "generated": str(CONFIG["directories"]["generated_files"]),
    }
    session_dirs = ensure_session_dirs(base_dirs, session_id)
    issue_lists_dir = session_dirs.get("history_issue_lists")
    target_files_dir = session_dirs.get("history_target_files")
    generated_session_dir = session_dirs.get("generated")
    
    # Layout similar to enterprise standard check: left main content, right file manager
    col_main, col_info = st.columns([2, 1])
    
    with col_main:
        # Two uploaders side by side
        col_issues, col_targets = st.columns(2)
        with col_issues:
            files_issues = st.file_uploader("点击上传历史问题清单", type=None, accept_multiple_files=True, key=f"history_issues_{session_id}")
            if files_issues:
                handle_file_upload(files_issues, issue_lists_dir)
                st.success(f"已上传 {len(files_issues)} 个历史问题清单文件")
        with col_targets:
            files_targets = st.file_uploader("点击上传待检查文件", type=None, accept_multiple_files=True, key=f"history_targets_{session_id}")
            if files_targets:
                handle_file_upload(files_targets, target_files_dir)
                st.success(f"已上传 {len(files_targets)} 个待检查文件")
        
        # Start and Demo buttons
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            if st.button("开始", key=f"history_start_button_{session_id}"):
                # Process PDFs (MinerU) and Word/PPT (Unstructured) into plain text
                st.info("开始处理文件：PDF 使用 MinerU，Word/PPT 使用 Unstructured…")
                area = st.container()
                with area:
                    # Create output directories for parsed text files
                    history_out_root = os.path.join(generated_session_dir, "history_issues_avoidance")
                    issue_lists_txt_dir = os.path.join(history_out_root, "issue_lists_txt")
                    target_files_txt_dir = os.path.join(history_out_root, "target_files_txt")
                    os.makedirs(issue_lists_txt_dir, exist_ok=True)
                    os.makedirs(target_files_txt_dir, exist_ok=True)
                    
                    st.markdown("**历史问题清单 → 文本**")
                    created_issues_pdf = _process_pdf_folder(issue_lists_dir, issue_lists_txt_dir, st)
                    created_issues_wp = _process_word_ppt_folder(issue_lists_dir, issue_lists_txt_dir, st)
                    created_issues_xls = _process_excel_folder(issue_lists_dir, issue_lists_txt_dir, st)
                    st.markdown("**待检查文件 → 文本**")
                    created_targets_pdf = _process_pdf_folder(target_files_dir, target_files_txt_dir, st)
                    created_targets_wp = _process_word_ppt_folder(target_files_dir, target_files_txt_dir, st)
                    created_targets_xls = _process_excel_folder(target_files_dir, target_files_txt_dir, st)
                    if any([created_issues_pdf, created_issues_wp, created_issues_xls, created_targets_pdf, created_targets_wp, created_targets_xls]):
                        st.success("处理完成。")
                    else:
                        st.info("未生成任何文本文件，请确认已上传 PDF、Word/PPT 或 Excel。")
        with btn_col2:
            if st.button("演示", key=f"history_demo_button_{session_id}"):
                # Copy demonstration file to issue_lists directory
                try:
                    # Locate demonstration root (same convention as other tabs)
                    demo_base_dir = CONFIG["directories"]["demonstration"]
                    demo_file_path = os.path.join(str(demo_base_dir), "副本LL-lesson learn-历史问题规避-V9.4.xlsx")
                    if os.path.exists(demo_file_path):
                        dest_path = os.path.join(issue_lists_dir, "副本LL-lesson learn-历史问题规避-V9.4.xlsx")
                        shutil.copy2(demo_file_path, dest_path)
                        st.session_state[f"history_demo_{session_id}"] = True
                        st.success("已复制演示文件：副本LL-lesson learn-历史问题规避-V9.4.xlsx")
                    else:
                        st.error("演示文件不存在，请检查路径是否正确。")
                except Exception as e:
                    st.error(f"演示文件复制失败: {e}")

    with col_info:
        # File manager utilities (mirroring enterprise standard tab behavior)
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
            from datetime import datetime
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
        col_clear1, col_clear2 = st.columns(2)
        with col_clear1:
            if st.button("🗑️ 清空历史问题清单", key=f"clear_history_issues_{session_id}"):
                try:
                    for file in os.listdir(issue_lists_dir):
                        file_path = os.path.join(issue_lists_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    st.success("已清空历史问题清单文件")
                except Exception as e:
                    st.error(f"清空失败: {e}")
        with col_clear2:
            if st.button("🗑️ 清空待检查文件", key=f"clear_history_targets_{session_id}"):
                try:
                    for file in os.listdir(target_files_dir):
                        file_path = os.path.join(target_files_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    st.success("已清空待检查文件")
                except Exception as e:
                    st.error(f"清空失败: {e}")

        # File lists in tabs
        tab_issues, tab_targets = st.tabs(["历史问题清单", "待检查文件"])
        with tab_issues:
            issue_files = get_file_list(issue_lists_dir)
            if issue_files:
                for file_info in issue_files:
                    display_name = truncate_filename(file_info['name'])
                    with st.expander(f"📄 {display_name}", expanded=False):
                        col_i, col_a = st.columns([3, 1])
                        with col_i:
                            st.write(f"**文件名:** {file_info['name']}")
                            st.write(f"**大小:** {format_file_size(file_info['size'])}")
                            st.write(f"**修改时间:** {format_timestamp(file_info['modified'])}")
                        with col_a:
                            delete_key = f"del_issue_{file_info['name'].replace(' ', '_').replace('.', '_')}_{session_id}"
                            if st.button("🗑️ 删除", key=delete_key):
                                try:
                                    os.remove(file_info['path'])
                                    st.success(f"已删除: {file_info['name']}")
                                except Exception as e:
                                    st.error(f"删除失败: {e}")
            else:
                st.write("（未上传）")

        with tab_targets:
            target_files = get_file_list(target_files_dir)
            if target_files:
                for file_info in target_files:
                    display_name = truncate_filename(file_info['name'])
                    with st.expander(f"📄 {display_name}", expanded=False):
                        col_i, col_a = st.columns([3, 1])
                        with col_i:
                            st.write(f"**文件名:** {file_info['name']}")
                            st.write(f"**大小:** {format_file_size(file_info['size'])}")
                            st.write(f"**修改时间:** {format_timestamp(file_info['modified'])}")
                        with col_a:
                            delete_key = f"del_target_{file_info['name'].replace(' ', '_').replace('.', '_')}_{session_id}"
                            if st.button("🗑️ 删除", key=delete_key):
                                try:
                                    os.remove(file_info['path'])
                                    st.success(f"已删除: {file_info['name']}")
                                except Exception as e:
                                    st.error(f"删除失败: {e}")
            else:
                st.write("（未上传）")