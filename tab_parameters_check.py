import streamlit as st
import os
import json
import pandas as pd
from util import ensure_session_dirs, handle_file_upload, get_user_session, start_analysis, reset_user_session, resolve_ollama_host
from config import CONFIG
from util import extract_parameters_to_json
from ollama import Client as OllamaClient
import openai

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

def run_parameters_analysis_workflow(session_id, session_dirs):
    """Run the complete parameters analysis workflow."""
    # Get tab-specific session state
    session = get_user_session(session_id, 'parameters')
    cp_session_dir = session_dirs["cp"]
    target_session_dir = session_dirs["target"]
    generated_session_dir = session_dirs["generated"]
    parameters_dir = session_dirs.get("generated_parameters_check", os.path.join(generated_session_dir, "parameters_check"))
    
    st.info("🔍 正在进行设计制程参数一致性分析…")
    
    # Get target files
    target_files_list = [f for f in os.listdir(target_session_dir) if os.path.isfile(os.path.join(target_session_dir, f))]
    if not target_files_list:
        st.warning("请先上传待检查文件")
        return
    
    # Initialize LLM backend (default to ollama)
    llm_backend = st.session_state.get(f'llm_backend_{session_id}', 'ollama_127')
    if llm_backend in ("ollama_127", "ollama_9"):
        host = resolve_ollama_host(llm_backend)
        ollama_client = OllamaClient(host=host)
    elif llm_backend == "openai":
        openai.base_url = CONFIG["llm"]["openai_base_url"]
        openai.api_key = CONFIG["llm"]["openai_api_key"]

    # Load prompt from parameters_llm_prompt.txt
    prompt_path = os.path.join(parameters_dir, "parameters_llm_prompt.txt")
    if not os.path.exists(prompt_path):
        st.warning("未找到提示词文件，请先点击“开始”生成 JSON 与提示词后再试。")
        return
    with open(prompt_path, 'r', encoding='utf-8') as f:
        full_prompt_text = f.read()

    # Display prompt and stream LLM response side by side
    col_prompt, col_response = st.columns([1, 1])
    with col_prompt:
        st.subheader("目标参数评审 - 提示词")
        prompt_container = st.container(height=400)
        with prompt_container:
            with st.chat_message("user"):
                prompt_placeholder = st.empty()
                streamed = ""
                for line in full_prompt_text.splitlines(True):
                    streamed += line
                    prompt_placeholder.text(streamed)
        st.chat_input(placeholder="", disabled=True, key=f"parameters_prompt_input_{session_id}")

    with col_response:
        st.subheader("目标参数评审 - AI回复")
        response_container = st.container(height=400)
        with response_container:
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response_text = ""

                if llm_backend in ("ollama_127", "ollama_9"):
                    for chunk in ollama_client.chat(
                        model=st.session_state.get(f'ollama_model_{session_id}', CONFIG["llm"]["ollama_model"]),
                        messages=[{"role": "user", "content": full_prompt_text}],
                        stream=True,
                        options={
                            "temperature": st.session_state.get(f'ollama_temperature_{session_id}', 0.7),
                            "top_p": st.session_state.get(f'ollama_top_p_{session_id}', 0.9),
                            "top_k": st.session_state.get(f'ollama_top_k_{session_id}', 40),
                            "repeat_penalty": st.session_state.get(f'ollama_repeat_penalty_{session_id}', 1.1),
                            "num_ctx": st.session_state.get(f'ollama_num_ctx_{session_id}', 40001),
                            "num_thread": st.session_state.get(f'ollama_num_thread_{session_id}', 4),
                        }
                    ):
                        new_text = chunk['message']['content']
                        response_text += new_text
                        response_placeholder.write(response_text)
                elif llm_backend == "openai":
                    stream = openai.chat.completions.create(
                        model=st.session_state.get(f'openai_model_{session_id}', CONFIG["llm"]["openai_model"]),
                        messages=[{"role": "user", "content": full_prompt_text}],
                        stream=True,
                        temperature=st.session_state.get(f'openai_temperature_{session_id}', 0.7),
                        top_p=st.session_state.get(f'openai_top_p_{session_id}', 1.0),
                        max_tokens=st.session_state.get(f'openai_max_tokens_{session_id}', 2048),
                        presence_penalty=st.session_state.get(f'openai_presence_penalty_{session_id}', 0.0),
                        frequency_penalty=st.session_state.get(f'openai_frequency_penalty_{session_id}', 0.0),
                    )
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content or ""
                        response_text += delta
                        response_placeholder.write(response_text)

                # Save LLM response to file in parameters_check subfolder
                try:
                    result_path = os.path.join(parameters_dir, "parameters_check_result.txt")
                    with open(result_path, 'w', encoding='utf-8') as f:
                        f.write(response_text)
                    st.success(f"已保存评审结果: {result_path}")
                except Exception as e:
                    st.warning(f"评审结果保存失败: {e}")

        st.chat_input(placeholder="", disabled=True, key=f"parameters_response_input_{session_id}")

    # --- 第二阶段：仅提取“不一致项”并输出为JSON ---
    st.divider()
    st.subheader("不一致项提取（JSON）")

    # Load prior free-form results (may exist from one or multiple runs)
    result_paths = [
        os.path.join(parameters_dir, "parameters_check_result.txt"),
        os.path.join(parameters_dir, "parameters_check_result2.txt"),
    ]
    prior_texts = []
    for p in result_paths:
        if os.path.exists(p):
            try:
                prior_texts.append(open(p, 'r', encoding='utf-8').read())
            except Exception:
                pass
    prior_merged = "\n\n---\n\n".join(prior_texts) if prior_texts else ""

    if not prior_merged:
        st.info("未找到先前的评审结果文本（parameters_check_result*.txt），请先完成上一阶段评审。")
        return

    # Build JSON-only extraction prompt
    extraction_prompt = (
        "你是一名 APQP 专家。现在请从以下评审文本中“只提取不一致项”，忽略所有“一致项”和“控制计划中无对应项/缺失项”的描述。\n"
        "请将不一致项整理为统一的 JSON 对象并严格只输出 JSON（不要输出解释/Markdown）。\n"
        "在每条不一致项中，location 字段必须提供明确定位：‘目标文件文件名 + 目标文件Sheet名称；控制计划文件文件名 + 控制计划Sheet名称’，"
        "若无法确定其中任一项，请以空字符串代替。不要提供行号。\n\n"
        "JSON 结构要求如下：\n"
        "{\n"
        "  \"items\": [\n"
        "    {\n"
        "      \"parameter\": \"参数名称\",\n"
        "      \"target_value\": \"目标文件中的取值/范围（若无法确定则空字符串）\",\n"
        "      \"cp_value\": \"控制计划中的取值/范围（若无法确定则空字符串）\",\n"
        "      \"location\": \"目标文件：<文件名>/<Sheet名>；控制计划：<文件名>/<Sheet名>\",\n"
        "      \"issue\": \"一句话说明不一致点\",\n"
        "      \"suggestion\": \"简短的修订建议\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "评审文本如下：\n"
        f"{prior_merged}"
    )

    col_json_prompt, col_json_response = st.columns([1, 1])
    with col_json_prompt:
        st.markdown("**JSON 提取 - 提示词**")
        prompt_container2 = st.container(height=400)
        with prompt_container2:
            with st.chat_message("user"):
                ph = st.empty()
                streamed = ""
                for line in extraction_prompt.splitlines(True):
                    streamed += line
                    ph.text(streamed)
        st.chat_input(placeholder="", disabled=True, key=f"parameters_json_prompt_input_{session_id}")

    with col_json_response:
        st.markdown("**JSON 提取 - AI回复**")
        response_container2 = st.container(height=400)
        with response_container2:
            with st.chat_message("assistant"):
                ph2 = st.empty()
                json_response_text = ""

                if llm_backend in ("ollama_127", "ollama_9"):
                    for chunk in ollama_client.chat(
                        model=st.session_state.get(f'ollama_model_{session_id}', CONFIG["llm"]["ollama_model"]),
                        messages=[{"role": "user", "content": extraction_prompt}],
                        stream=True,
                        options={
                            "temperature": st.session_state.get(f'ollama_temperature_{session_id}', 0.7),
                            "top_p": st.session_state.get(f'ollama_top_p_{session_id}', 0.9),
                            "top_k": st.session_state.get(f'ollama_top_k_{session_id}', 40),
                            "repeat_penalty": st.session_state.get(f'ollama_repeat_penalty_{session_id}', 1.1),
                            "num_ctx": st.session_state.get(f'ollama_num_ctx_{session_id}', 40001),
                            "num_thread": st.session_state.get(f'ollama_num_thread_{session_id}', 4),
                            "format": "json",
                        }
                    ):
                        new_text = chunk['message']['content']
                        json_response_text += new_text
                        ph2.write(json_response_text)
                elif llm_backend == "openai":
                    stream = openai.chat.completions.create(
                        model=st.session_state.get(f'openai_model_{session_id}', CONFIG["llm"]["openai_model"]),
                        messages=[{"role": "user", "content": extraction_prompt}],
                        stream=True,
                        temperature=st.session_state.get(f'openai_temperature_{session_id}', 0.7),
                        top_p=st.session_state.get(f'openai_top_p_{session_id}', 1.0),
                        max_tokens=st.session_state.get(f'openai_max_tokens_{session_id}', 2048),
                        presence_penalty=st.session_state.get(f'openai_presence_penalty_{session_id}', 0.0),
                        frequency_penalty=st.session_state.get(f'openai_frequency_penalty_{session_id}', 0.0),
                        response_format={"type": "json_object"},
                    )
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content or ""
                        json_response_text += delta
                        ph2.write(json_response_text)

        st.chat_input(placeholder="", disabled=True, key=f"parameters_json_response_input_{session_id}")

    # Parse and persist JSON
    parsed = None
    try:
        parsed = json.loads(json_response_text)
    except Exception:
        try:
            cleaned = json_response_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip('`')
                idx = cleaned.find("{")
                if idx >= 0:
                    cleaned = cleaned[idx:]
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            if start >= 0 and end > start:
                cleaned = cleaned[start:end+1]
            parsed = json.loads(cleaned)
        except Exception:
            parsed = None

    if parsed and isinstance(parsed, dict):
        save_path = os.path.join(parameters_dir, "parameters_check_result.json")
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(parsed, f, ensure_ascii=False, indent=2)
            st.success(f"已保存不一致项JSON: {save_path}")
        except Exception as e:
            st.warning(f"保存JSON失败: {e}")

        # Show as table
        items = parsed.get('items') if isinstance(parsed.get('items'), list) else []
        # Ensure column order, fill missing keys
        norm_rows = []
        for it in items:
            norm_rows.append({
                'parameter': str(it.get('parameter', '')),
                'target_value': str(it.get('target_value', '')),
                'cp_value': str(it.get('cp_value', '')),
                'location': str(it.get('location', '')),
                'issue': str(it.get('issue', '')),
                'suggestion': str(it.get('suggestion', '')),
            })
        if norm_rows:
            df = pd.DataFrame(norm_rows, columns=['parameter', 'target_value', 'cp_value', 'location', 'issue', 'suggestion'])
            st.dataframe(df, use_container_width=True)
            # Save CSV to parameters_check folder
            try:
                csv_path = os.path.join(parameters_dir, 'parameters_check_result.csv')
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                st.success(f"已保存不一致项表格: {csv_path}")
            except Exception as e:
                st.warning(f"保存CSV失败: {e}")
        else:
            st.info("未解析到任何不一致项。")
    else:
        st.warning("未能解析为有效的JSON，请检查上方AI回复内容。")

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
    # Page subheader
    st.subheader("📊 设计制程检查")
    
    
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
    parameters_dir = session_dirs.get("generated_parameters_check", os.path.join(generated_session_dir, "parameters_check"))
    os.makedirs(parameters_dir, exist_ok=True)

    # Layout: right column for info, left for main content
    col_main, col_info = st.columns([2, 1])

    # Render the info/file column FIRST so lists appear immediately when demo starts
    with col_info:
        # Early scoped clear operations: three per-bucket buttons
        col_clear_cp, col_clear_target, col_clear_graph = st.columns(3)
        with col_clear_cp:
            if st.button("🗑️ 清空控制计划文件", key=f"parameters_clear_cp_files_{session_id}"):
                try:
                    for file in os.listdir(cp_session_dir):
                        file_path = os.path.join(cp_session_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    st.success("已清空控制计划文件")
                except Exception as e:
                    st.error(f"清空失败: {e}")
        with col_clear_target:
            if st.button("🗑️ 清空待检查文件", key=f"parameters_clear_target_files_{session_id}"):
                try:
                    for file in os.listdir(target_session_dir):
                        file_path = os.path.join(target_session_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    st.success("已清空待检查文件")
                except Exception as e:
                    st.error(f"清空失败: {e}")
        with col_clear_graph:
            if st.button("🗑️ 清空图纸文件", key=f"parameters_clear_graph_files_{session_id}"):
                try:
                    for file in os.listdir(graph_session_dir):
                        file_path = os.path.join(graph_session_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    st.success("已清空图纸文件")
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
                    output_file = os.path.join(parameters_dir, "parameters_prompt_output.txt")
                    result_file = os.path.join(parameters_dir, "parameters_check_result.txt")
                    
                    if os.path.exists(output_file):
                        os.remove(output_file)
                    if os.path.exists(result_file):
                        os.remove(result_file)
                    
                    # Extract parameters into JSON for this tab
                    try:
                        # Extract CP parameters JSON
                        json_output_path = os.path.join(parameters_dir, "extracted_data.json")
                        summary = extract_parameters_to_json(
                            cp_session_dir=cp_session_dir,
                            output_json_path=json_output_path,
                            # Read config from parameters subfolder instead of CP_files/<user>
                            config_csv_path=os.path.join(parameters_dir, "excel_sheets.csv"),
                        )
                        st.success(f"已生成参数JSON: {summary['output']} (表: {summary['sheets']}, 行: {summary['rows']})")

                        # Extract Target parameters JSON
                        json_output_path_t = os.path.join(parameters_dir, "extracted_target_data.json")
                        summary_t = extract_parameters_to_json(
                            cp_session_dir=target_session_dir,
                            output_json_path=json_output_path_t,
                            config_csv_path=os.path.join(parameters_dir, "excel_sheets.csv"),
                        )
                        st.success(f"已生成目标参数JSON: {summary_t['output']} (表: {summary_t['sheets']}, 行: {summary_t['rows']})")

                        # Build LLM prompt that embeds both JSON payloads and save to file
                        try:
                            with open(json_output_path_t, 'r', encoding='utf-8') as f:
                                target_json_obj = json.load(f)
                            with open(json_output_path, 'r', encoding='utf-8') as f:
                                cp_json_obj = json.load(f)
                            target_files = sorted({str(item.get('File')) for item in target_json_obj if item.get('File')}) or ["目标文件"]
                            target_json_str = json.dumps(target_json_obj, ensure_ascii=False, indent=2)
                            cp_json_str = json.dumps(cp_json_obj, ensure_ascii=False, indent=2)
                            prompt_text = (
                                "你是一名 APQP 专家，需要对应用交付物进行设计与制程参数一致性评审。\n"
                                "请对比目标文件与控制计划中的参数名称、单位、取值/公差范围等是否一致，逐项给出：是否一致、不一致项、缺失项、疑似问题，"
                                "并提供引用依据（段落/数据），且在引用中务必标明“文件名 + Sheet 名称”（不要提供行号）；给出简明的改进建议；最后给出总体结论。\n\n"
                                f"以下为目标文件（提取自：{', '.join(target_files)}）的参数数据（JSON）：\n"
                                f"{target_json_str}\n\n"
                                "以下为相关控制计划文件汇总得到的参数数据（JSON）：\n"
                                f"{cp_json_str}\n\n"
                                "请基于上述数据完成评审，并按参数项分组输出。"
                            )
                            prompt_path = os.path.join(parameters_dir, "parameters_llm_prompt.txt")
                            with open(prompt_path, 'w', encoding='utf-8') as f:
                                f.write(prompt_text)
                            st.success(f"已生成评审提示词: {prompt_path}")
                        except Exception as e:
                            st.warning(f"提示词生成失败（不会影响后续JSON结果）：{e}")

                    except Exception as e:
                        st.error(f"参数提取失败: {e}")
                        return
                    
                    # Clear chat history for fresh analysis
                    session['ollama_history'] = []
                    session['openai_history'] = []
                    session['analysis_completed'] = False
                    
                    # Start the analysis process
                    start_analysis(session_id, 'parameters')
                    st.rerun()
            with col_buttons[1]:
                if st.button("演示", key=f"parameters_demo_button_{session_id}"):
                    # Copy demo files into this tab's session directories only (isolated)
                    demo_base_dir = CONFIG["directories"]["cp_files"].parent / "demonstration"
                    # Map demonstration folders to this tab's session directories
                    demo_folder_mapping = {
                        "CP_files": "cp",
                        "graph_files": "graph",
                        "target_files": "target",
                    }
                    files_copied = False
                    for demo_folder, session_key in demo_folder_mapping.items():
                        demo_folder_path = os.path.join(demo_base_dir, demo_folder)
                        session_folder_path = session_dirs[session_key]

                        if os.path.exists(demo_folder_path):
                            for file_name in os.listdir(demo_folder_path):
                                demo_file_path = os.path.join(demo_folder_path, file_name)
                                session_file_path = os.path.join(session_folder_path, file_name)
                                if os.path.isfile(demo_file_path):
                                    import shutil
                                    shutil.copy2(demo_file_path, session_file_path)
                                    files_copied = True

                    # Also copy excel_sheets.csv config into parameters_dir for this tab
                    demo_config_file = os.path.join(demo_base_dir, "excel_sheets.csv")
                    if os.path.exists(demo_config_file):
                        import shutil
                        shutil.copy2(demo_config_file, os.path.join(parameters_dir, "excel_sheets.csv"))

                    if files_copied and os.path.exists(os.path.join(parameters_dir, "excel_sheets.csv")):
                        # Auto-generate JSONs and prompt so the analysis view works immediately
                        try:
                            # CP JSON
                            json_output_path = os.path.join(parameters_dir, "extracted_data.json")
                            extract_parameters_to_json(
                                cp_session_dir=cp_session_dir,
                                output_json_path=json_output_path,
                                config_csv_path=os.path.join(parameters_dir, "excel_sheets.csv"),
                            )
                            # Target JSON
                            json_output_path_t = os.path.join(parameters_dir, "extracted_target_data.json")
                            extract_parameters_to_json(
                                cp_session_dir=target_session_dir,
                                output_json_path=json_output_path_t,
                                config_csv_path=os.path.join(parameters_dir, "excel_sheets.csv"),
                            )
                            # Prompt
                            with open(json_output_path_t, 'r', encoding='utf-8') as f:
                                target_json_obj = json.load(f)
                            with open(json_output_path, 'r', encoding='utf-8') as f:
                                cp_json_obj = json.load(f)
                            target_files = sorted({str(item.get('File')) for item in target_json_obj if item.get('File')}) or ["目标文件"]
                            target_json_str = json.dumps(target_json_obj, ensure_ascii=False, indent=2)
                            cp_json_str = json.dumps(cp_json_obj, ensure_ascii=False, indent=2)
                            prompt_text = (
                                "你是一名 APQP 专家，需要对应用交付物进行设计与制程参数一致性评审。\n"
                                "请对比目标文件与控制计划中的参数名称、单位、取值/公差范围等是否一致，逐项给出：是否一致、不一致项、缺失项、疑似问题，"
                                "并提供引用依据（段落/数据），且在引用中务必标明“文件名 + Sheet 名称”（不要提供行号）；给出简明的改进建议；最后给出总体结论。\n\n"
                                f"以下为目标文件（提取自：{', '.join(target_files)}）的参数数据（JSON）：\n"
                                f"{target_json_str}\n\n"
                                "以下为相关控制计划文件汇总得到的参数数据（JSON）：\n"
                                f"{cp_json_str}\n\n"
                                "请基于上述数据完成评审，并按参数项分组输出。"
                            )
                            prompt_path = os.path.join(parameters_dir, "parameters_llm_prompt.txt")
                            with open(prompt_path, 'w', encoding='utf-8') as f:
                                f.write(prompt_text)
                        except Exception as e:
                            st.warning(f"演示准备提示词失败：{e}")

                        # Prepare this tab's session state and start analysis lifecycle
                        session['analysis_completed'] = False
                        session['process_started'] = True
                        session['ollama_history'] = []
                        session['openai_history'] = []
                        st.rerun()
                    else:
                        st.info("未找到演示文件，请检查 demonstration 目录。")

            
        
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
                    run_parameters_analysis_workflow(session_id, session_dirs)
                    
                    # Mark as completed
                    session['analysis_completed'] = True
                else:
                    # Files exist but process wasn't explicitly started
                    st.info("检测到待检查文件，请点击\"开始\"按钮开始分析，或点击\"演示\"按钮使用演示文件。")
            else:
                st.warning("请先上传待检查文件")


        # (Bulk operations moved earlier to avoid duplicate keys and to update UI promptly)