import streamlit as st
import os
from util import ensure_session_dirs, handle_file_upload, get_user_session, start_analysis, reset_user_session, complete_analysis, resolve_ollama_host
from config import CONFIG
from ollama import Client as OllamaClient
import openai
import re
import pandas as pd
from datetime import datetime
import json

def parse_llm_table_response(response_text):
    """Parse LLM response to extract table data."""
    if not response_text:
        return []
    # Look for table patterns in the response
    # Common patterns: "应包含的交付物文件清单" followed by "是" or "否"
    table_data = []
    # Split response into lines and look for table-like patterns
    lines = response_text.split('\n')
    in_table = False
    for line in lines:
        line = line.strip()
        
        # Check if we're entering a table section
        if '应包含的交付物文件清单' in line and ('存在' in line or '是' in line or '否' in line):
            in_table = True
            continue
            
        # Skip empty lines and non-table content
        if not line or not in_table:
            continue
            
        # Look for table row patterns
        # Pattern: filename followed by "是" or "否"
        if '|' in line:
            # Handle pipe-separated format
            parts = [part.strip() for part in line.split('|')]
            if len(parts) >= 2:
                filename = parts[0].strip()
                status = parts[1].strip()
                if filename and status in ['是', '否']:
                    table_data.append({'filename': filename, 'status': status})
        else:
            # Handle other formats - look for filename and status
            # Try to find patterns like "filename: 是" or "filename: 否"
            status_match = re.search(r'[：:]\s*(是|否)', line)
            if status_match:
                status = status_match.group(1)
                filename = line[:status_match.start()].strip()
                if filename:
                    table_data.append({'filename': filename, 'status': status})
    return table_data

def get_stage_requirements(stage_name):
    """Get hardcoded requirements for each stage."""
    stage_requirements = {
        "立项阶段": [
            "项目立项报告", "项目可行性分析报告", "项目风险评估报告", "项目计划书",
            "项目团队组建方案", "项目预算方案", "项目时间计划", "项目质量目标",
            "项目成本目标", "项目交付物清单"
        ],
        "A样阶段": [
            "电芯规格书", "尺寸链公差计算书", "初始DFMEA", "初始特殊特性清单",
            "三新清单", "制程标准", "开模清单", "3D数模", "2D图纸", "BOM清单",
            "仿真报告", "测试大纲", "专利挖掘清单", "初版PFMEA", "产线规划方案",
            "过程设计初始方案", "产品可制造性分析及风险应对报告", "初始过程流程图",
            "初始过程特殊特性", "初版CP", "初版SOP", "工艺验证计划", "样品包装方案"
        ],
        "B样阶段": [
            "设计变更履历表", "更新电芯规格书", "更新DFMEA", "更新特殊特性清单",
            "制程标准", "更新3D数模", "更新2D图纸", "尺寸链公差计算书",
            "更新BOM清单", "更新开模清单", "更新三新清单", "仿真报告",
            "DV测试报告"
        ],
        "C样阶段": [
            "更新PFMEA", "量产产线开发进展报告", "更新样品包装方案",
            "更新过程特殊特性清单", "更新CP", "更新SOP", "工艺验证计划",
            "样品历史问题清单", "CMK分析报告", "CPK分析报告",
            "工程变更履历表", "产品可制造性分析与风险应对报告", "更新过程流程图", "更新过程特殊特性清单",
            "设备停机率统计表&设备故障记录表", "工艺验证报告", "外观标准书", "PV测试报告"
        ]
    }
    
    return stage_requirements.get(stage_name, [])

def create_completeness_excel(all_stage_data, session_id, generated_session_dir):
    """Create and save Excel file with completeness results in normalized format.
    Columns: [Stage, Deliverable, Exists, FileName, Notes]
    """
    try:
        rows = []
        # Keep a deterministic stage order
        ordered_stages = ['立项阶段', 'A样阶段', 'B样阶段', 'C样阶段']
        for stage_name in ordered_stages:
            for item in all_stage_data.get(stage_name, []):
                rows.append({
                    'Stage': stage_name,
                    'Deliverable': item.get('filename', ''),
                    'Exists': item.get('status', ''),  # '是' / '否'
                    'FileName': item.get('matched_file', ''),
                    'Notes': item.get('note', '')
                })

        df = pd.DataFrame(rows, columns=['Stage', 'Deliverable', 'Exists', 'FileName', 'Notes'])
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"file_completeness_results_{session_id}_{timestamp}.xlsx"
        filepath = os.path.join(generated_session_dir, filename)
        
        # Save to Excel
        df.to_excel(filepath, index=False, engine='openpyxl')
        
        return filepath, filename
        
    except Exception as e:
        st.error(f"Excel导出失败: {e}")
        return None, None

def export_completeness_results(session_id, stage_responses, generated_session_dir):
    """Main function to export completeness results to Excel."""
    try:
        all_stage_data = {}
        
        # Process each stage response
        for stage_name, response_text in stage_responses.items():
            if response_text:
                # Prefer strict JSON, fallback to legacy text parsing
                parsed_ok = False
                try:
                    # Try direct JSON first
                    data = json.loads(response_text)
                    if isinstance(data, dict) and isinstance(data.get('items'), list):
                        table_data = []
                        for item in data['items']:
                            name = str(item.get('name', '')).strip()
                            if not name:
                                continue
                            # Robust boolean coercion for various model outputs
                            exists_raw = item.get('exists')
                            if isinstance(exists_raw, bool):
                                exists = exists_raw
                            else:
                                s = str(exists_raw).strip().lower()
                                exists = s in ("true", "1", "yes", "y", "是", "存在")
                            matched_file = str(item.get('matched_file', '') or '').strip()
                            note = str(item.get('note', '') or '').strip()
                            if not exists:
                                matched_file = ''
                            table_data.append({
                                'filename': name,
                                'status': '是' if exists else '否',
                                'matched_file': matched_file,
                                'note': note
                            })
                        all_stage_data[stage_name] = table_data
                        parsed_ok = True
                except Exception:
                    # Try to extract JSON object from code fences or extra text
                    try:
                        # Remove markdown fences if present
                        cleaned = response_text.strip()
                        if cleaned.startswith("```"):
                            cleaned = cleaned.strip('`')
                            idx = cleaned.find("{")
                            if idx >= 0:
                                cleaned = cleaned[idx:]
                        # Fallback: slice first {...} block
                        start = cleaned.find('{')
                        end = cleaned.rfind('}')
                        if start >= 0 and end > start:
                            cleaned = cleaned[start:end+1]
                        data = json.loads(cleaned)
                        if isinstance(data, dict) and isinstance(data.get('items'), list):
                            table_data = []
                            for item in data['items']:
                                name = str(item.get('name', '')).strip()
                                if not name:
                                    continue
                                exists_raw = item.get('exists')
                                if isinstance(exists_raw, bool):
                                    exists = exists_raw
                                else:
                                    s = str(exists_raw).strip().lower()
                                    exists = s in ("true", "1", "yes", "y", "是", "存在")
                                matched_file = str(item.get('matched_file', '') or '').strip()
                                note = str(item.get('note', '') or '').strip()
                                if not exists:
                                    matched_file = ''
                                table_data.append({
                                    'filename': name,
                                    'status': '是' if exists else '否',
                                    'matched_file': matched_file,
                                    'note': note
                                })
                            all_stage_data[stage_name] = table_data
                            parsed_ok = True
                        else:
                            parsed_ok = False
                    except Exception:
                        parsed_ok = False
                
                if not parsed_ok:
                    # Parse LLM response to extract table data (Markdown/loose text)
                    table_data = parse_llm_table_response(response_text)
                    # ensure keys for downstream export
                    for it in table_data:
                        it.setdefault('matched_file', '')
                        it.setdefault('note', '')
                    all_stage_data[stage_name] = table_data
            else:
                # For empty stages, create "否" entries for all requirements
                stage_requirements = get_stage_requirements(stage_name)
                all_stage_data[stage_name] = [
                    {'filename': req, 'status': '否', 'matched_file': '', 'note': ''} for req in stage_requirements
                ]
        
        # Create Excel file
        filepath, filename = create_completeness_excel(all_stage_data, session_id, generated_session_dir)
        
        if filepath:
            st.success(f"✅ 文件齐套性检查结果已导出到: {filename}")
            # Display the exported Excel content as a table preview
            try:
                df_preview = pd.read_excel(filepath)
                st.dataframe(df_preview, use_container_width=True)
                # Provide a download button for the exported Excel file
                try:
                    with open(filepath, "rb") as f:
                        file_bytes = f.read()
                    st.download_button(
                        label="⬇️ 下载Excel结果",
                        data=file_bytes,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"download_completeness_{session_id}"
                    )
                except Exception as e:
                    st.warning(f"无法提供下载按钮: {e}")
            except Exception as e:
                st.warning(f"无法预览导出的Excel文件: {e}")
            return filepath
        else:
            st.error("❌ Excel导出失败")
            return None
            
    except Exception as e:
        st.error(f"导出过程中发生错误: {e}")
        return None

def generate_stage_prompt(stage_name, stage_folder, stage_requirements):
    """Generate prompt for a specific stage based on requirements and actual files."""
    if not os.path.exists(stage_folder):
        return f"{stage_name}文件夹不存在"
    
    # Get actual files in the stage folder
    actual_files = []
    if os.path.exists(stage_folder):
        files = [f for f in os.listdir(stage_folder) if os.path.isfile(os.path.join(stage_folder, f))]
        actual_files = files
    
    # Create the prompt
    prompt = f"""{stage_name}应包含的文件包括
{stage_requirements}

{stage_name}文件夹中已有的文件清单包括
{chr(10).join(actual_files) if actual_files else "（无文件）"}

对比{stage_name}应包含的文件清单和{stage_name}文件夹中已有的文件清单，做匹配判断（允许合理的名称近似，例如“历史问题规避清单”≈“副本 LL-lesson learn-历史问题规避-V9.4.xlsx”）。

请只输出一个JSON对象，严格符合以下结构，不要输出任何额外文本（不要有解释、markdown或其他字符）：
{{
  "stage": "{stage_name}",
  "items": [
    {{
      "name": "<应包含的交付物文件名>",
      "exists": true|false,
      "matched_file": "<若exists=true，请填写在该阶段文件夹中匹配到的实际文件名；若不存在则填空字符串>",
      "note": "<可选：关于该行的说明/备注；若无则填空字符串>"
    }}
    // 针对应包含清单中的每一项都输出一条
  ]
}}

要求：
- items必须覆盖“应包含的交付物文件清单”中的每一项，且只出现一次；
- exists为布尔类型；
- 仅输出上述JSON对象本身。"""
    
    return prompt

def render_file_completeness_check_tab(session_id):
    # Handle None session_id (user not logged in)
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return
    
    # Page subheader
    st.subheader("📁 文件齐套性检查")
    st.markdown("上传每个阶段的文件后点击开始，AI会根据预设的清单检查并输出结果，预设清单的具体条目见帮助文档。")
    
    # Add CSS to hide chat input (required for auto-scroll to work)
    st.markdown("""
    <style>
    [data-testid="stChatInput"] { display: none; }
    </style>
    """, unsafe_allow_html=True)
    
    
    
    # Define APQP stage directories (with session subfolders) - using centralized config
    base_dir = str(CONFIG["directories"]["apqp_files"])
    base_dirs = {
        "Stage_Initial": os.path.join(base_dir, "Stage_Initial"),
        "Stage_A": os.path.join(base_dir, "Stage_A"),
        "Stage_B": os.path.join(base_dir, "Stage_B"),
        "Stage_C": os.path.join(base_dir, "Stage_C"),
        "generated": str(CONFIG["directories"]["generated_files"])
    }
    session_dirs = ensure_session_dirs(base_dirs, session_id)
    generated_session_dir = session_dirs["generated"]
    completeness_dir = session_dirs.get("generated_file_completeness_check", os.path.join(generated_session_dir, "file_completeness_check"))
    os.makedirs(completeness_dir, exist_ok=True)

    # Get structured user session
    session = get_user_session(session_id, 'completeness')
    
    # Initialize LLM clients
    llm_backend = st.session_state.get(f'llm_backend_{session_id}', 'ollama_9')
    if llm_backend in ("ollama_127","ollama_9"):
        host = resolve_ollama_host(llm_backend)
        ollama_client = OllamaClient(host=host)
    elif llm_backend == "openai":
        openai.api_key = CONFIG["llm"]["openai_api_key"]
        openai.base_url = CONFIG["llm"]["openai_base_url"]

    # Layout: right column for info, left for main content
    col_main, col_info = st.columns([2, 1])

    # Render the info/file column FIRST so file lists appear immediately when demo starts
    with col_info:
        # Early bulk operations: handle clear-all before listing so UI updates immediately
        if st.button("🗑️ 清空所有文件", key=f"clear_all_files_completeness_{session_id}"):
            try:
                for dir_path in [session_dirs["Stage_Initial"], session_dirs["Stage_A"], session_dirs["Stage_B"], session_dirs["Stage_C"]]:
                    for file in os.listdir(dir_path):
                        file_path = os.path.join(dir_path, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                st.success("已清空所有文件")
            except Exception as e:
                st.error(f"清空失败: {e}")
        # --- File Manager Module ---
        def get_file_list(folder):
            if not os.path.exists(folder):
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
            # Use stable sorting by name first, then by modification time
            return sorted(files, key=lambda x: (x['name'].lower(), x['modified']), reverse=False)

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

        # File Manager Tabs
        tab_initial, tab_a, tab_b, tab_c = st.tabs(["立项阶段", "A样阶段", "B样阶段", "C样阶段"])
        
        with tab_initial:
            initial_files_list = get_file_list(session_dirs["Stage_Initial"])
            if initial_files_list:
                for i, file_info in enumerate(initial_files_list):
                    display_name = truncate_filename(file_info['name'])
                    with st.expander(f"📄 {display_name}", expanded=False):
                        col_info, col_action = st.columns([3, 1])
                        with col_info:
                            st.write(f"**文件名:** {file_info['name']}")
                            st.write(f"**大小:** {format_file_size(file_info['size'])}")
                            st.write(f"**修改时间:** {format_timestamp(file_info['modified'])}")
                        with col_action:
                            delete_key = f"delete_initial_{file_info['name'].replace(' ', '_').replace('.', '_')}_{session_id}"
                            if st.button("🗑️ 删除", key=delete_key):
                                try:
                                    os.remove(file_info['path'])
                                    st.success(f"已删除: {file_info['name']}")
                                except Exception as e:
                                    st.error(f"删除失败: {e}")
            else:
                st.write("（未上传）")
            st.markdown("---")
            st.markdown("**上传新文件:**")
            new_initial_files = st.file_uploader("选择立项阶段文件", type=None, accept_multiple_files=True, key=f"initial_uploader_tab_{session_id}")
            if new_initial_files:
                handle_file_upload(new_initial_files, session_dirs["Stage_Initial"])

        with tab_a:
            a_files_list = get_file_list(session_dirs["Stage_A"])
            if a_files_list:
                for i, file_info in enumerate(a_files_list):
                    display_name = truncate_filename(file_info['name'])
                    with st.expander(f"📄 {display_name}", expanded=False):
                        col_info, col_action = st.columns([3, 1])
                        with col_info:
                            st.write(f"**文件名:** {file_info['name']}")
                            st.write(f"**大小:** {format_file_size(file_info['size'])}")
                            st.write(f"**修改时间:** {format_timestamp(file_info['modified'])}")
                        with col_action:
                            delete_key = f"delete_a_{file_info['name'].replace(' ', '_').replace('.', '_')}_{session_id}"
                            if st.button("🗑️ 删除", key=delete_key):
                                try:
                                    os.remove(file_info['path'])
                                    st.success(f"已删除: {file_info['name']}")
                                except Exception as e:
                                    st.error(f"删除失败: {e}")
            else:
                st.write("（未上传）")
            st.markdown("---")
            st.markdown("**上传新文件:**")
            new_a_files = st.file_uploader("选择A样阶段文件", type=None, accept_multiple_files=True, key=f"a_uploader_tab_{session_id}")
            if new_a_files:
                handle_file_upload(new_a_files, session_dirs["Stage_A"])

        with tab_b:
            b_files_list = get_file_list(session_dirs["Stage_B"])
            if b_files_list:
                for i, file_info in enumerate(b_files_list):
                    display_name = truncate_filename(file_info['name'])
                    with st.expander(f"📄 {display_name}", expanded=False):
                        col_info, col_action = st.columns([3, 1])
                        with col_info:
                            st.write(f"**文件名:** {file_info['name']}")
                            st.write(f"**大小:** {format_file_size(file_info['size'])}")
                            st.write(f"**修改时间:** {format_timestamp(file_info['modified'])}")
                        with col_action:
                            delete_key = f"delete_b_{file_info['name'].replace(' ', '_').replace('.', '_')}_{session_id}"
                            if st.button("🗑️ 删除", key=delete_key):
                                try:
                                    os.remove(file_info['path'])
                                    st.success(f"已删除: {file_info['name']}")
                                except Exception as e:
                                    st.error(f"删除失败: {e}")
            else:
                st.write("（未上传）")
            st.markdown("---")
            st.markdown("**上传新文件:**")
            new_b_files = st.file_uploader("选择B样阶段文件", type=None, accept_multiple_files=True, key=f"b_uploader_tab_{session_id}")
            if new_b_files:
                handle_file_upload(new_b_files, session_dirs["Stage_B"])

        with tab_c:
            c_files_list = get_file_list(session_dirs["Stage_C"])
            if c_files_list:
                for i, file_info in enumerate(c_files_list):
                    display_name = truncate_filename(file_info['name'])
                    with st.expander(f"📄 {display_name}", expanded=False):
                        col_info, col_action = st.columns([3, 1])
                        with col_info:
                            st.write(f"**文件名:** {file_info['name']}")
                            st.write(f"**大小:** {format_file_size(file_info['size'])}")
                            st.write(f"**修改时间:** {format_timestamp(file_info['modified'])}")
                        with col_action:
                            delete_key = f"delete_c_{file_info['name'].replace(' ', '_').replace('.', '_')}_{session_id}"
                            if st.button("🗑️ 删除", key=delete_key):
                                try:
                                    os.remove(file_info['path'])
                                    st.success(f"已删除: {file_info['name']}")
                                except Exception as e:
                                    st.error(f"删除失败: {e}")
            else:
                st.write("（未上传）")
            st.markdown("---")
            st.markdown("**上传新文件:**")
            new_c_files = st.file_uploader("选择C样阶段文件", type=None, accept_multiple_files=True, key=f"c_uploader_tab_{session_id}")
            if new_c_files:
                handle_file_upload(new_c_files, session_dirs["Stage_C"])
    # Render MAIN column content: uploaders and controls
    with col_main:
        # File uploads directly in col_main (no nested columns)
        col_initial, col_a, col_b, col_c = st.columns(4)
        with col_initial:
            files_initial = st.file_uploader("点击上传立项阶段文件", type=None, accept_multiple_files=True, key="stage_initial")
            if files_initial:
                handle_file_upload(files_initial, session_dirs["Stage_Initial"])
                st.success(f"已上传 {len(files_initial)} 个立项阶段文件")
        with col_a:
            files_a = st.file_uploader("点击上传A样阶段文件", type=None, accept_multiple_files=True, key="stage_a")
            if files_a:
                handle_file_upload(files_a, session_dirs["Stage_A"])
                st.success(f"已上传 {len(files_a)} 个A样阶段文件")
        with col_b:
            files_b = st.file_uploader("点击上传B样阶段文件", type=None, accept_multiple_files=True, key="stage_b")
            if files_b:
                handle_file_upload(files_b, session_dirs["Stage_B"])
                st.success(f"已上传 {len(files_b)} 个B样阶段文件")
        with col_c:
            files_c = st.file_uploader("点击上传C样阶段文件", type=None, accept_multiple_files=True, key="stage_c")
            if files_c:
                handle_file_upload(files_c, session_dirs["Stage_C"])
                st.success(f"已上传 {len(files_c)} 个C样阶段文件")

        # Start button - only show if process hasn't started
        if not session['process_started']:
            col_buttons = st.columns([1, 1])
            with col_buttons[0]:
                if st.button("开始", key=f"file_completeness_start_button_{session_id}"):
                    # Start the analysis process
                    start_analysis(session_id, 'completeness')
                    st.rerun()
            with col_buttons[1]:
                if st.button("演示", key=f"file_completeness_demo_button_{session_id}"):
                    # Demo feature: copy demonstration files to current session
                    demo_base_dir = CONFIG["directories"]["apqp_files"].parent / "demonstration"
                    
                    # Copy files from demonstration APQP_files to session folders
                    demo_apqp_path = os.path.join(demo_base_dir, "APQP_files")
                    if os.path.exists(demo_apqp_path):
                        import shutil
                        for stage_folder in ["Stage_Initial", "Stage_A", "Stage_B", "Stage_C"]:
                            demo_stage_path = os.path.join(demo_apqp_path, stage_folder)
                            session_stage_path = session_dirs[stage_folder]
                            
                            if os.path.exists(demo_stage_path):
                                # Copy all files from demo stage folder to session stage folder
                                for file_name in os.listdir(demo_stage_path):
                                    demo_file_path = os.path.join(demo_stage_path, file_name)
                                    session_file_path = os.path.join(session_stage_path, file_name)
                                    
                                    if os.path.isfile(demo_file_path):
                                        shutil.copy2(demo_file_path, session_file_path)
                        
                        start_analysis(session_id, 'completeness')
                        st.success("演示已开始！正在分析演示文件...")
                        st.rerun()
                    else:
                        st.error("演示文件不存在，请检查演示文件夹")
        
        # Show results if process has started
        if session['process_started']:
            st.divider()
            
            # Add a button to reset and clear history with status message
            col_reset, col_status = st.columns([1, 2])
            with col_reset:
                if st.button("重新开始", key=f"file_completeness_reset_button_{session_id}"):
                    reset_user_session(session_id, 'completeness')
                    st.rerun()
            
            with col_status:
                if not session['analysis_completed']:
                    st.info("🤖 分析进行中...")
                else:
                    st.success("✅ 分析完成")
                    # Add a more prominent reset option when analysis is completed
                    st.info('💡 如需重新开始分析，请点击左侧的"重新开始"按钮')
            
            # Define stage requirements
            stage_requirements = {
                "立项阶段": """1. 项目立项报告
2. 项目可行性分析报告
3. 项目风险评估报告
4. 项目计划书
5. 项目团队组建方案
6. 项目预算方案
7. 项目时间计划
8. 项目质量目标
9. 项目成本目标
10. 项目交付物清单""",
                
                "A样阶段": """1. 电芯规格书
2. 尺寸链公差计算书
3. 初始DFMEA
4. 初始特殊特性清单
5. 三新清单
6. 制程标准
7. 开模清单
8. 3D数模
9. 2D图纸
10. BOM清单
11. 仿真报告
12. 测试大纲
13. 专利挖掘清单
14. 初版PFMEA
15. 产线规划方案
16. 过程设计初始方案
17. 产品可制造性分析及风险应对报告
18. 初始过程流程图
19. 初始过程特殊特性
20. 初版CP
21. 初版SOP
22. 工艺验证计划
23. 样品包装方案""",
                
                "B样阶段": """1. 设计变更履历表
2. 更新电芯规格书
3. 更新DFMEA
4. 更新特殊特性清单
5. 制程标准
6. 更新3D数模
7. 更新2D图纸
8. 尺寸链公差计算书
9. 更新BOM清单
10. 更新开模清单
11. 更新三新清单
12. 仿真报告
13. DV测试报告""",
                
                "C样阶段": """1. 更新PFMEA
2. 量产产线开发进展报告
3. 更新样品包装方案
4. 更新过程流程图
5. 更新过程特殊特性清单
6. 更新CP
7. 更新SOP
8. 工艺验证计划
9. 样品历史问题清单
10. CMK分析报告
11. CPK分析报告
12. 工程变更履历表
13. 产品可制造性分析及风险应对报告
14. 设备停机率统计表&设备故障记录表
15. 工艺验证报告
16. 外观标准书
17. PV测试报告"""
            }
            
            # Generate prompts and run analysis for each stage
            stages = [
                ("立项阶段", session_dirs["Stage_Initial"]),
                ("A样阶段", session_dirs["Stage_A"]),
                ("B样阶段", session_dirs["Stage_B"]),
                ("C样阶段", session_dirs["Stage_C"])
            ]
            
            # Dictionary to store all stage responses for Excel export
            stage_responses = {}
            
            for stage_name, stage_folder in stages:
                if os.path.exists(stage_folder):
                    if any(os.listdir(stage_folder)):
                        # Stage has files - run full LLM analysis
                        # Generate prompt for this stage
                        prompt = generate_stage_prompt(stage_name, stage_folder, stage_requirements[stage_name])
                        
                        # Save prompt to file
                        prompt_file = os.path.join(completeness_dir, f"prompt_{stage_name}.txt")
                        with open(prompt_file, "w", encoding="utf-8") as f:
                            f.write(prompt)
                        
                        # Display the prompt and response side by side
                        col_prompt, col_response = st.columns([1, 1])
                        with col_prompt:
                            st.subheader(f"{stage_name} - 提示词:")
                            prompt_container = st.container(height=400)
                            with prompt_container:
                                with st.chat_message("user"):
                                    prompt_placeholder = st.empty()
                                    prompt_placeholder.text(prompt)
                                
                                st.chat_input(placeholder="", disabled=True, key=f"file_completeness_prompt_chat_input_{stage_name}_{session_id}")
                        
                        with col_response:
                            st.subheader(f"{stage_name} - 检查结果:")
                            response_container = st.container(height=400)
                            with response_container:
                                with st.chat_message("assistant"):
                                    response_placeholder = st.empty()
                                    
                                    # Stream the response using selected LLM
                                    response_text = ""
                                    if llm_backend in ("ollama_127", "ollama_9"):
                                        for chunk in ollama_client.chat(
                                            model=st.session_state.get(f'ollama_model_{session_id}', CONFIG["llm"]["ollama_model"]),
                                            messages=[{"role": "user", "content": prompt}],
                                            stream=True,
                                            options={
                                                "temperature": st.session_state.get(f'ollama_temperature_{session_id}', 0.7),
                                                "top_p": st.session_state.get(f'ollama_top_p_{session_id}', 0.9),
                                                "top_k": st.session_state.get(f'ollama_top_k_{session_id}', 40),
                                                "repeat_penalty": st.session_state.get(f'ollama_repeat_penalty_{session_id}', 1.1),
                                                "num_ctx": st.session_state.get(f'ollama_num_ctx_{session_id}', 40001),
                                                "num_thread": st.session_state.get(f'ollama_num_thread_{session_id}', 4),
                                                "format": "json"
                                            }
                                        ):
                                            new_text = chunk['message']['content']
                                            response_text += new_text
                                            response_placeholder.write(response_text)
                                    elif llm_backend == "openai":
                                        stream = openai.chat.completions.create(
                                            model=st.session_state.get(f'openai_model_{session_id}', CONFIG["llm"]["openai_model"]),
                                            messages=[{"role": "user", "content": prompt}],
                                            stream=True,
                                            temperature=st.session_state.get(f'openai_temperature_{session_id}', 0.7),
                                            top_p=st.session_state.get(f'openai_top_p_{session_id}', 1.0),
                                            max_tokens=st.session_state.get(f'openai_max_tokens_{session_id}', 2048),
                                            presence_penalty=st.session_state.get(f'openai_presence_penalty_{session_id}', 0.0),
                                            frequency_penalty=st.session_state.get(f'openai_frequency_penalty_{session_id}', 0.0),
                                            response_format={"type": "json_object"}
                                        )
                                        for chunk in stream:
                                            delta = chunk.choices[0].delta.content or ""
                                            response_text += delta
                                            response_placeholder.write(response_text)
                                    
                                    # Store the response for Excel export
                                    stage_responses[stage_name] = response_text
                                    
                                    st.chat_input(placeholder="", disabled=True, key=f"file_completeness_response_chat_input_{stage_name}_{session_id}")
                    else:
                        # Stage has no files - show simple message
                        st.info(f"📁 {stage_name}文件夹为空，因此该阶段的所有必需文件均缺失。")
                        # Store empty response for Excel export (will be handled as "否" for all requirements)
                        stage_responses[stage_name] = ""
            
            # Mark analysis as completed and export Excel
            if not session['analysis_completed']:
                complete_analysis(session_id, 'completeness')
                
                # Export results to Excel after all stages are processed
                if stage_responses:
                    export_completeness_results(session_id, stage_responses, completeness_dir)


        # (Bulk operations moved earlier to avoid duplicate keys and to update UI promptly)