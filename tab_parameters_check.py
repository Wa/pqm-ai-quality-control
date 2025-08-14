import streamlit as st
import os
from util import ensure_session_dirs, handle_file_upload, get_user_session, start_analysis, reset_user_session
from config import CONFIG

class ParametersPromptGenerator:
    """Generate prompts for LLM to perform parameters check on Excel files."""
    
    def __init__(self):
        """Initialize the ParametersPromptGenerator."""
        pass
    
    def generate_prompt(self, cp_session_dir, target_file_path, output_file):
        """
        Generate prompts for parameters check analysis.
        
        Args:
            cp_session_dir: Directory containing control plan files
            target_file_path: Path to the target file to be checked
            output_file: Output file path for generated prompts
        """
        # TODO: Implement prompt generation logic
        # This will read Excel files and generate prompts for parameters check
        pass

def render_parameters_file_upload_section(session_dirs, session_id):
    """Render the file upload section for parameters check with unique keys."""
    col_cp, col_target, col_graph = st.columns([1, 1, 1])

    with col_cp:
        cp_files = st.file_uploader("点击上传控制计划文件", type=None, accept_multiple_files=True, key=f"parameters_cp_uploader_{session_id}")
        if cp_files:
            handle_file_upload(cp_files, session_dirs["cp"])

    with col_target:
        target_files = st.file_uploader("点击上传待检查文件", type=None, accept_multiple_files=True, key=f"parameters_target_uploader_{session_id}")
        if target_files:
            handle_file_upload(target_files, session_dirs["target"])

    with col_graph:
        graph_files = st.file_uploader("点击上传图纸文件", type=None, accept_multiple_files=True, key=f"parameters_graph_uploader_{session_id}")
        if graph_files:
            handle_file_upload(graph_files, session_dirs["graph"])

def run_parameters_analysis_workflow(session_id, session_dirs, prompt_generator):
    """Run the complete parameters analysis workflow."""
    # Get tab-specific session state
    session = get_user_session(session_id, 'parameters')
    cp_session_dir = session_dirs["cp"]
    target_session_dir = session_dirs["target"]
    generated_session_dir = session_dirs["generated"]
    
    st.info("🔍 开始设计制程检查分析，请稍候...")
    
    # Get target files
    target_files_list = [f for f in os.listdir(target_session_dir) if os.path.isfile(os.path.join(target_session_dir, f))]
    if not target_files_list:
        st.warning("请先上传待检查文件")
        return
    
    # TODO: Implement the actual parameters analysis workflow
    # This will be similar to the special symbols check but for parameters
    st.info("🚧 设计制程检查功能正在开发中，敬请期待！")

def render_parameters_check_tab(session_id):
    """Render the design process parameters check tab."""
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
    
    st.title("设计制程检查")
    st.caption("检查设计参数和制程参数的合理性")
    
    # Base directories for each upload box - using centralized config
    BASE_DIRS = {
        "cp": str(CONFIG["directories"]["cp_files"]),
        "target": str(CONFIG["directories"]["target_files"]),
        "graph": str(CONFIG["directories"]["graph_files"]),
        "generated": str(CONFIG["directories"]["generated_files"])
    }
    session_dirs = ensure_session_dirs(BASE_DIRS, session_id)
    cp_session_dir = session_dirs["cp"]
    target_session_dir = session_dirs["target"]
    graph_session_dir = session_dirs["graph"]
    generated_session_dir = session_dirs["generated"]

    # Initialize ParametersPromptGenerator
    prompt_generator = ParametersPromptGenerator()

    # Layout: right column for info, left for main content
    col_main, col_info = st.columns([2, 1])

    # Render the info/file column FIRST so lists appear immediately when demo starts
    with col_info:
        # Early bulk operations: handle clear-all before listing so UI updates immediately
        if st.button("🗑️ 清空所有文件", key=f"parameters_clear_all_files_{session_id}"):
            try:
                for dir_path in [cp_session_dir, target_session_dir, graph_session_dir]:
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
            """Convert bytes to human readable format."""
            if size_bytes == 0:
                return "0 B"
            size_names = ["B", "KB", "MB", "GB"]
            i = 0
            while size_bytes >= 1024 and i < len(size_names) - 1:
                size_bytes /= 1024.0
                i += 1
            return f"{size_bytes:.1f} {size_names[i]}"

        def format_timestamp(timestamp):
            """Convert timestamp to readable date."""
            from datetime import datetime
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')

        def truncate_filename(filename, max_length=40):
            """Truncate filename if too long, preserving extension."""
            if len(filename) <= max_length:
                return filename
            name, ext = os.path.splitext(filename)
            available_length = max_length - len(ext) - 3
            if available_length <= 0:
                return filename[:max_length-3] + "..."
            truncated_name = name[:available_length] + "..."
            return truncated_name + ext

        # File Manager Tabs
        tab_cp, tab_target, tab_graph = st.tabs(["控制计划文件", "待检查文件", "图纸文件"])
        
        with tab_cp:
            cp_files_list = get_file_list(cp_session_dir)
            if cp_files_list:
                for i, file_info in enumerate(cp_files_list):
                    display_name = truncate_filename(file_info['name'])
                    with st.expander(f"📄 {display_name}", expanded=False):
                        col_info, col_action = st.columns([3, 1])
                        with col_info:
                            st.write(f"**文件名:** {file_info['name']}")
                            st.write(f"**大小:** {format_file_size(file_info['size'])}")
                            st.write(f"**修改时间:** {format_timestamp(file_info['modified'])}")
                        with col_action:
                            delete_key = f"parameters_delete_cp_{file_info['name'].replace(' ', '_').replace('.', '_')}_{session_id}"
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
            new_cp_files = st.file_uploader("选择控制计划文件", type=None, accept_multiple_files=True, key=f"parameters_cp_uploader_tab_{session_id}")
            if new_cp_files:
                handle_file_upload(new_cp_files, cp_session_dir)

        with tab_target:
            target_files_list = get_file_list(target_session_dir)
            if target_files_list:
                for i, file_info in enumerate(target_files_list):
                    display_name = truncate_filename(file_info['name'])
                    with st.expander(f"📄 {display_name}", expanded=False):
                        col_info, col_action = st.columns([3, 1])
                        with col_info:
                            st.write(f"**文件名:** {file_info['name']}")
                            st.write(f"**大小:** {format_file_size(file_info['size'])}")
                            st.write(f"**修改时间:** {format_timestamp(file_info['modified'])}")
                        with col_action:
                            delete_key = f"parameters_delete_target_{file_info['name'].replace(' ', '_').replace('.', '_')}_{session_id}"
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
            new_target_files = st.file_uploader("选择待检查文件", type=None, accept_multiple_files=True, key=f"parameters_target_uploader_tab_{session_id}")
            if new_target_files:
                handle_file_upload(new_target_files, target_session_dir)

        with tab_graph:
            graph_files_list = get_file_list(graph_session_dir)
            if graph_files_list:
                for i, file_info in enumerate(graph_files_list):
                    display_name = truncate_filename(file_info['name'])
                    with st.expander(f"📄 {display_name}", expanded=False):
                        col_info, col_action = st.columns([3, 1])
                        with col_info:
                            st.write(f"**文件名:** {file_info['name']}")
                            st.write(f"**大小:** {format_file_size(file_info['size'])}")
                            st.write(f"**修改时间:** {format_timestamp(file_info['modified'])}")
                        with col_action:
                            delete_key = f"parameters_delete_graph_{file_info['name'].replace(' ', '_').replace('.', '_')}_{session_id}"
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
            new_graph_files = st.file_uploader("选择图纸文件", type=None, accept_multiple_files=True, key=f"parameters_graph_uploader_tab_{session_id}")
            if new_graph_files:
                handle_file_upload(new_graph_files, graph_session_dir)
    # Render MAIN column content: uploaders and controls
    with col_main:
        # Get structured user session
        session = get_user_session(session_id, 'parameters')

        # Always show file upload section
        render_parameters_file_upload_section(session_dirs, session_id)
        
        # Start button - only show if process hasn't started
        if not session['process_started']:
            col_buttons = st.columns([1, 1])
            with col_buttons[0]:
                if st.button("开始", key=f"parameters_start_button_{session_id}"):
                    # Clear any existing generated files to ensure fresh generation
                    output_file = os.path.join(generated_session_dir, "parameters_prompt_output.txt")
                    result_file = os.path.join(generated_session_dir, "parameters_check_result.txt")
                    
                    if os.path.exists(output_file):
                        os.remove(output_file)
                    if os.path.exists(result_file):
                        os.remove(result_file)
                    
                    # Clear chat history for fresh analysis
                    session['ollama_history'] = []
                    session['openai_history'] = []
                    session['analysis_completed'] = False
                    
                    # Start the analysis process
                    start_analysis(session_id, 'parameters')
                    st.rerun()
            with col_buttons[1]:
                if st.button("演示", key=f"parameters_demo_button_{session_id}"):
                    # TODO: Implement demo functionality
                    st.info("🚧 演示功能正在开发中，敬请期待！")
        
        # Show status and reset button if process has started
        if session['process_started']:
            # Add a button to reset and clear history
            if st.button("重新开始", key=f"parameters_reset_button_{session_id}"):
                reset_user_session(session_id, 'parameters')
                st.rerun()
            
            # Check if we need to run analysis
            target_files_list = [f for f in os.listdir(target_session_dir) if os.path.isfile(os.path.join(target_session_dir, f))]
            if target_files_list:
                if session['process_started'] and not session['analysis_completed']:
                    # Run the analysis workflow
                    run_parameters_analysis_workflow(session_id, session_dirs, prompt_generator)
                    
                    # Mark as completed
                    session['analysis_completed'] = True
                else:
                    # Files exist but process wasn't explicitly started
                    st.info("检测到待检查文件，请点击\"开始\"按钮开始分析，或点击\"演示\"按钮使用演示文件。")
            else:
                st.warning("请先上传待检查文件")


        # (Bulk operations moved earlier to avoid duplicate keys and to update UI promptly)