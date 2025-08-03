import streamlit as st
import os
from util import ensure_session_dirs, handle_file_upload, get_user_session, start_analysis, reset_user_session, complete_analysis
from config import CONFIG
from ollama import Client as OllamaClient
import openai
import re

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

对比{stage_name}应包含的文件清单和{stage_name}文件夹中已有的文件清单，并以表格的形式给出对比结果。表格的第一列是需要的交付物文件清单，第二列写"是"或者"否"，如果该文件能在{stage_name}文件夹里找到，则写"是"，如果该文件不能在{stage_name}文件夹里找到，"否"。注意，文件名不一定完全一致，所以需要你通过常识判断。例如应包含的交付物文件清单中的一个文件为"历史问题规避清单"，而{stage_name}文件夹中有一个文件为"副本 LL-lesson learn-历史问题规避-V9.4.xlsx"，虽然文件名不完全一致，但通过常识可判断这两个指的是同一个文件，所以判断"历史问题规避清单"已经存在。最后，如果一个文件出现在{stage_name}文件夹中里，但没出现在应包含的清单里，将这些文件单独罗列出来。"""
    
    return prompt

def render_file_completeness_check_tab(session_id):
    # Handle None session_id (user not logged in)
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return
    
    # Add CSS to hide chat input (required for auto-scroll to work)
    st.markdown("""
    <style>
    [data-testid="stChatInput"] { display: none; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("文件齐套性检查")
    
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

    # Get structured user session
    session = get_user_session(session_id, 'completeness')
    
    # Initialize LLM clients
    llm_backend = st.session_state.get(f'llm_backend_{session_id}', 'ollama')
    if llm_backend == "ollama":
        ollama_client = OllamaClient(host=CONFIG["llm"]["ollama_host"])
    elif llm_backend == "openai":
        openai.api_key = CONFIG["llm"]["openai_api_key"]
        openai.base_url = CONFIG["llm"]["openai_base_url"]

    # Layout: right column for info, left for main content
    col_main, col_info = st.columns([2, 1])
    
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
                
                "B样阶段": """1. 产品设计验证报告
2. 过程设计验证报告
3. 产品设计确认报告
4. 过程设计确认报告
5. 产品设计评审报告
6. 过程设计评审报告
7. 产品设计变更记录
8. 过程设计变更记录
9. 产品设计问题清单
10. 过程设计问题清单
11. 产品设计改进方案
12. 过程设计改进方案
13. 产品设计风险评估
14. 过程设计风险评估
15. 产品设计成本分析
16. 过程设计成本分析
17. 产品设计质量分析
18. 过程设计质量分析
19. 产品设计进度分析
20. 过程设计进度分析""",
                
                "C样阶段": """1. 产品设计冻结报告
2. 过程设计冻结报告
3. 产品设计发布报告
4. 过程设计发布报告
5. 产品设计归档报告
6. 过程设计归档报告
7. 产品设计总结报告
8. 过程设计总结报告
9. 产品设计经验总结
10. 过程设计经验总结
11. 产品设计教训总结
12. 过程设计教训总结
13. 产品设计改进建议
14. 过程设计改进建议
15. 产品设计标准化建议
16. 过程设计标准化建议
17. 产品设计培训材料
18. 过程设计培训材料
19. 产品设计文档清单
20. 过程设计文档清单"""
            }
            
            # Generate prompts and run analysis for each stage
            stages = [
                ("立项阶段", session_dirs["Stage_Initial"]),
                ("A样阶段", session_dirs["Stage_A"]),
                ("B样阶段", session_dirs["Stage_B"]),
                ("C样阶段", session_dirs["Stage_C"])
            ]
            
            for stage_name, stage_folder in stages:
                if os.path.exists(stage_folder):
                    if any(os.listdir(stage_folder)):
                        # Stage has files - run full LLM analysis
                        # Generate prompt for this stage
                        prompt = generate_stage_prompt(stage_name, stage_folder, stage_requirements[stage_name])
                        
                        # Save prompt to file
                        prompt_file = os.path.join(generated_session_dir, f"prompt_{stage_name}.txt")
                        with open(prompt_file, "w", encoding="utf-8") as f:
                            f.write(prompt)
                        
                        st.divider()
                        
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
                            st.subheader(f"{stage_name} - AI回复:")
                            response_container = st.container(height=400)
                            with response_container:
                                with st.chat_message("assistant"):
                                    response_placeholder = st.empty()
                                    
                                    # Stream the response using selected LLM
                                    response_text = ""
                                    if llm_backend == "ollama":
                                        for chunk in ollama_client.chat(
                                            model=st.session_state.get(f'ollama_model_{session_id}', CONFIG["llm"]["ollama_model"]),
                                            messages=[{"role": "user", "content": prompt}],
                                            stream=True,
                                            options={
                                                "temperature": st.session_state.get(f'ollama_temperature_{session_id}', 0.7),
                                                "top_p": st.session_state.get(f'ollama_top_p_{session_id}', 0.9),
                                                "top_k": st.session_state.get(f'ollama_top_k_{session_id}', 40),
                                                "repeat_penalty": st.session_state.get(f'ollama_repeat_penalty_{session_id}', 1.1),
                                                "num_ctx": st.session_state.get(f'ollama_num_ctx_{session_id}', 4096),
                                                "num_thread": st.session_state.get(f'ollama_num_thread_{session_id}', 4)
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
                                            frequency_penalty=st.session_state.get(f'openai_frequency_penalty_{session_id}', 0.0)
                                        )
                                        for chunk in stream:
                                            delta = chunk.choices[0].delta.content or ""
                                            response_text += delta
                                            response_placeholder.write(response_text)
                                    
                                    st.chat_input(placeholder="", disabled=True, key=f"file_completeness_response_chat_input_{stage_name}_{session_id}")
                    else:
                        # Stage has no files - show simple message
                        st.divider()
                        st.info(f"📁 {stage_name}文件夹为空，因此该阶段的所有必需文件均缺失。")
            
            # Mark analysis as completed
            if not session['analysis_completed']:
                complete_analysis(session_id, 'completeness')

    with col_info:
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
            available_length = max_length - len(ext) - 3  # 3 for "..."
            
            if available_length <= 0:
                # If extension is too long, just truncate the whole thing
                return filename[:max_length-3] + "..."
            
            # Truncate name part and add ellipsis
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
                            st.write(f"**文件名:** {file_info['name']}")  # Show full name inside
                            st.write(f"**大小:** {format_file_size(file_info['size'])}")
                            st.write(f"**修改时间:** {format_timestamp(file_info['modified'])}")
                        with col_action:
                            # Use a more stable key for delete button
                            delete_key = f"delete_initial_{file_info['name'].replace(' ', '_').replace('.', '_')}_{session_id}"
                            if st.button("🗑️ 删除", key=delete_key):
                                try:
                                    os.remove(file_info['path'])
                                    st.success(f"已删除: {file_info['name']}")
                                except Exception as e:
                                    st.error(f"删除失败: {e}")
            else:
                st.write("（未上传）")
                
            # Upload new files directly in this tab
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
                            st.write(f"**文件名:** {file_info['name']}")  # Show full name inside
                            st.write(f"**大小:** {format_file_size(file_info['size'])}")
                            st.write(f"**修改时间:** {format_timestamp(file_info['modified'])}")
                        with col_action:
                            # Use a more stable key for delete button
                            delete_key = f"delete_a_{file_info['name'].replace(' ', '_').replace('.', '_')}_{session_id}"
                            if st.button("🗑️ 删除", key=delete_key):
                                try:
                                    os.remove(file_info['path'])
                                    st.success(f"已删除: {file_info['name']}")
                                except Exception as e:
                                    st.error(f"删除失败: {e}")
            else:
                st.write("（未上传）")
                
            # Upload new files directly in this tab
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
                            st.write(f"**文件名:** {file_info['name']}")  # Show full name inside
                            st.write(f"**大小:** {format_file_size(file_info['size'])}")
                            st.write(f"**修改时间:** {format_timestamp(file_info['modified'])}")
                        with col_action:
                            # Use a more stable key for delete button
                            delete_key = f"delete_b_{file_info['name'].replace(' ', '_').replace('.', '_')}_{session_id}"
                            if st.button("🗑️ 删除", key=delete_key):
                                try:
                                    os.remove(file_info['path'])
                                    st.success(f"已删除: {file_info['name']}")
                                except Exception as e:
                                    st.error(f"删除失败: {e}")
            else:
                st.write("（未上传）")
                
            # Upload new files directly in this tab
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
                            st.write(f"**文件名:** {file_info['name']}")  # Show full name inside
                            st.write(f"**大小:** {format_file_size(file_info['size'])}")
                            st.write(f"**修改时间:** {format_timestamp(file_info['modified'])}")
                        with col_action:
                            # Use a more stable key for delete button
                            delete_key = f"delete_c_{file_info['name'].replace(' ', '_').replace('.', '_')}_{session_id}"
                            if st.button("🗑️ 删除", key=delete_key):
                                try:
                                    os.remove(file_info['path'])
                                    st.success(f"已删除: {file_info['name']}")
                                except Exception as e:
                                    st.error(f"删除失败: {e}")
            else:
                st.write("（未上传）")
                
            # Upload new files directly in this tab
            st.markdown("---")
            st.markdown("**上传新文件:**")
            new_c_files = st.file_uploader("选择C样阶段文件", type=None, accept_multiple_files=True, key=f"c_uploader_tab_{session_id}")
            if new_c_files:
                handle_file_upload(new_c_files, session_dirs["Stage_C"])

        # Bulk operations
        st.markdown("---")
        st.markdown("### 批量操作")
        
        if st.button("🗑️ 清空所有文件", key=f"clear_all_files_completeness_{session_id}"):
            try:
                # Clear all session directories
                for dir_path in [session_dirs["Stage_Initial"], session_dirs["Stage_A"], session_dirs["Stage_B"], session_dirs["Stage_C"]]:
                    for file in os.listdir(dir_path):
                        file_path = os.path.join(dir_path, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                st.success("已清空所有文件")
            except Exception as e:
                st.error(f"清空失败: {e}") 