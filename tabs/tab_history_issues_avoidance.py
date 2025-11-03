from __future__ import annotations

import streamlit as st
import pandas as pd
import os
import io
import zipfile
import json
import requests
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


def _collect_files(folder: str) -> list[dict[str, object]]:
    if not folder or not os.path.isdir(folder):
        return []
    items = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        stat = os.stat(path)
        items.append(
            {
                "name": name,
                "path": path,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            }
        )
    items.sort(key=lambda info: info["modified"], reverse=True)
    return items


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
    from datetime import datetime

    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def _truncate_filename(filename: str, max_length: int = 40) -> str:
    if len(filename) <= max_length:
        return filename
    name, ext = os.path.splitext(filename)
    available = max_length - len(ext) - 3
    if available <= 0:
        return filename[: max_length - 3] + "..."
    return name[:available] + "..." + ext


def _process_category(
    label: str,
    source_dir: str | None,
    output_dir: str,
    progress_area,
):
    os.makedirs(output_dir, exist_ok=True)
    if not source_dir or not os.path.isdir(source_dir):
        progress_area.warning(f"未找到 {label} 上传目录，已跳过。")
        return []

    progress_area.markdown(f"**{label} → 文本**")
    created = []
    created.extend(_process_pdf_folder(source_dir, output_dir, progress_area))
    created.extend(_process_word_ppt_folder(source_dir, output_dir, progress_area))
    created.extend(_process_excel_folder(source_dir, output_dir, progress_area))
    if not created:
        progress_area.info(f"{label} 未生成任何文本文件，请确认已上传 PDF、Word/PPT 或 Excel。")
    return created


def render_history_issues_avoidance_tab(session_id):
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return

    st.subheader("📋 历史问题规避")

    base_dirs = {
        "generated": str(CONFIG["directories"]["generated_files"]),
    }
    session_dirs = ensure_session_dirs(base_dirs, session_id)

    issue_lists_dir = session_dirs.get("history_issue_lists")
    dfmea_dir = session_dirs.get("history_dfmea")
    pfmea_dir = session_dirs.get("history_pfmea")
    cp_dir = session_dirs.get("history_cp")
    generated_root = session_dirs.get("generated_history_issues_avoidance")
    if not generated_root:
        generated_base = session_dirs.get("generated")
        if generated_base:
            generated_root = os.path.join(generated_base, "history_issues_avoidance")
            os.makedirs(generated_root, exist_ok=True)

    upload_targets = [
        {"label": "历史问题清单", "key": "issue_lists", "dir": issue_lists_dir},
        {"label": "DFMEA", "key": "dfmea", "dir": dfmea_dir},
        {"label": "PFMEA", "key": "pfmea", "dir": pfmea_dir},
        {"label": "控制计划 (CP)", "key": "cp", "dir": cp_dir},
    ]

    col_main, col_info = st.columns([2, 1])

    with col_main:
        st.markdown("请上传历史问题清单、DFMEA、PFMEA 与控制计划文件。支持 PDF、Word/PPT、Excel 等格式。")

        upload_columns = st.columns(2)
        for index, target in enumerate(upload_targets):
            column = upload_columns[index % len(upload_columns)]
            with column:
                uploaded_files = st.file_uploader(
                    f"点击上传 {target['label']}",
                    type=None,
                    accept_multiple_files=True,
                    key=f"history_upload_{target['key']}_{session_id}",
                )
                if uploaded_files:
                    handle_file_upload(uploaded_files, target["dir"])
                    st.success(f"已上传 {len(uploaded_files)} 个 {target['label']} 文件")

        st.divider()

        if st.button("开始解析", key=f"history_start_{session_id}"):
            area = st.container()
            with area:
                if not generated_root:
                    st.error("未能初始化生成文件目录，请检查配置。")
                else:
                    st.info("开始处理文件：PDF 使用 MinerU，Word/PPT 使用 Unstructured…")
                    total_created = []
                    for target in upload_targets:
                        output_dir = os.path.join(generated_root, f"{target['key']}_txt")
                        created = _process_category(
                            target["label"],
                            target["dir"],
                            output_dir,
                            area,
                        )
                        total_created.extend(created)
                    if total_created:
                        st.success("处理完成。")
                    else:
                        st.info("未生成任何文本文件，请确认上传内容后重试。")

    with col_info:
        st.subheader("📁 上传文件")

        clear_columns = st.columns(2)
        for index, target in enumerate(upload_targets):
            column = clear_columns[index % len(clear_columns)]
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
                    st.write("（未上传）")
                    continue
                for info in files:
                    display = _truncate_filename(info["name"])
                    with st.expander(f"📄 {display}", expanded=False):
                        col_meta, col_actions = st.columns([3, 1])
                        with col_meta:
                            st.write(f"**文件名:** {info['name']}")
                            st.write(f"**大小:** {_format_file_size(info['size'])}")
                            st.write(f"**修改时间:** {_format_timestamp(info['modified'])}")
                        with col_actions:
                            delete_key = f"history_delete_{target['key']}_{info['name'].replace(' ', '_').replace('.', '_')}_{session_id}"
                            if st.button("🗑️ 删除", key=delete_key):
                                try:
                                    os.remove(info["path"])
                                    st.success(f"已删除: {info['name']}")
                                    st.rerun()
                                except Exception as error:
                                    st.error(f"删除失败: {error}")
