import streamlit as st
import os
from util import ensure_session_dirs, handle_file_upload


def render_enterprise_standard_check_tab(session_id):
	# Handle None session_id (user not logged in)
	if session_id is None:
		st.warning("请先登录以使用此功能。")
		return

	st.subheader("🏢 企业标准检查")

	# Ensure enterprise directories exist and get paths
	# Passing empty base_dirs is fine; util.ensure_session_dirs will still create enterprise directories
	session_dirs = ensure_session_dirs({}, session_id)
	standards_dir = session_dirs.get("enterprise_standards")
	examined_dir = session_dirs.get("enterprise_examined")

	# Layout similar to 文件齐套性检查: left main content, right file manager
	col_main, col_info = st.columns([2, 1])

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

		# Start and Demo buttons similar to 文件齐套性检查
		btn_col1, btn_col2 = st.columns([1, 1])
		with btn_col1:
			if st.button("开始", key=f"enterprise_start_button_{session_id}"):
				st.session_state[f"enterprise_started_{session_id}"] = True
				st.info("即将上线，预计在9月5日前准备就绪。")
		with btn_col2:
			if st.button("演示", key=f"enterprise_demo_button_{session_id}"):
				st.session_state[f"enterprise_demo_{session_id}"] = True
				st.info("即将上线，预计在9月5日前准备就绪。")

		# If started previously, keep showing placeholder message
		if st.session_state.get(f"enterprise_started_{session_id}") or st.session_state.get(f"enterprise_demo_{session_id}"):
			st.info("即将上线，预计在9月5日前准备就绪。")

	with col_info:
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

		# File lists in tabs
		tab_std, tab_exam = st.tabs(["企业标准文件", "待检查文件"])
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


