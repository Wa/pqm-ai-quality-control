import streamlit as st
import os
import json
import time
from datetime import datetime
from pathlib import Path
from util import get_user_session, reset_user_session, start_analysis, complete_analysis, SimplePromptGenerator, ensure_session_dirs, handle_file_upload, resolve_ollama_host
from config import CONFIG
from backend_client import get_backend_client, is_backend_available
from ollama import Client as OllamaClient
import openai
import re
import pandas as pd

def render_file_upload_section(session_dirs, session_id):
    """Render the file upload section with proper keys to prevent duplication."""
    col_cp, col_target, col_graph = st.columns([1, 1, 1])

    with col_cp:
        cp_files = st.file_uploader("点击上传控制计划文件", type=None, accept_multiple_files=True, key=f"cp_uploader_{session_id}")
        if cp_files:
            handle_file_upload(cp_files, session_dirs["cp"])

    with col_target:
        target_files = st.file_uploader("点击上传待检查文件", type=None, accept_multiple_files=True, key=f"target_uploader_{session_id}")
        if target_files:
            handle_file_upload(target_files, session_dirs["target"])

    with col_graph:
        graph_files = st.file_uploader("点击上传图纸文件", type=None, accept_multiple_files=True, key=f"graph_uploader_{session_id}")
        if graph_files:
            handle_file_upload(graph_files, session_dirs["graph"])

def parse_prompts(output_file):
    """Parse prompt_output.txt into individual prompts."""
    if not os.path.exists(output_file):
        return []
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
    prompts = [p.strip() for p in content.split('='*80) if p.strip()]
    return prompts

def remove_think_blocks(text):
    """Remove all <think>...</think> blocks, including the tags."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

def run_analysis_workflow(session_id, session_dirs, prompt_generator):
    """Run the complete analysis workflow."""
    # Get tab-specific session state
    session = get_user_session(session_id, 'special_symbols')
    cp_session_dir = session_dirs["cp"]
    target_session_dir = session_dirs["target"]
    generated_session_dir = session_dirs["generated"]
    special_dir = session_dirs.get("generated_special_symbols_check", os.path.join(generated_session_dir, "special_symbols_check"))
    os.makedirs(special_dir, exist_ok=True)
    st.info("🔍 开始特殊特性符号检查分析，请稍候...")
    
    # Get target files
    target_files_list = [f for f in os.listdir(target_session_dir) if os.path.isfile(os.path.join(target_session_dir, f))]
    if not target_files_list:
        st.warning("请先上传待检查文件")
        return
    
    target_file_path = os.path.join(target_session_dir, target_files_list[0])
    output_file = os.path.join(special_dir, "prompt_output.txt")
    
    # Check if prompt_output.txt already exists (from demo or previous run)
    if not os.path.exists(output_file):
        # Generate prompt only if it doesn't exist
        prompt_generator.generate_prompt(cp_session_dir, target_file_path, output_file)
    
    prompts = parse_prompts(output_file)
    result_file = os.path.join(special_dir, "2_symbol_check_result.txt")
    
    # Get LLM backend from session state (default to ollama)
    llm_backend = st.session_state.get(f'llm_backend_{session_id}', 'ollama_127')
    
    # Initialize LLM clients based on selected backend
    if llm_backend in ("ollama_127","ollama_9"):
        host = resolve_ollama_host(llm_backend)
        ollama_client = OllamaClient(host=host)
        
        # Streaming generator for Ollama (stateless per call)
        def llm_stream_chat(prompt):
            response_text = ""
            
            # Get Ollama parameters from session state
            model = st.session_state.get(f'ollama_model_{session_id}', CONFIG["llm"]["ollama_model"])
            temperature = st.session_state.get(f'ollama_temperature_{session_id}', 0.7)
            top_p = st.session_state.get(f'ollama_top_p_{session_id}', 0.9)
            top_k = st.session_state.get(f'ollama_top_k_{session_id}', 40)
            repeat_penalty = st.session_state.get(f'ollama_repeat_penalty_{session_id}', 1.1)
            num_ctx = st.session_state.get(f'ollama_num_ctx_{session_id}', 65536)
            num_thread = st.session_state.get(f'ollama_num_thread_{session_id}', 4)
            
            for chunk in ollama_client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options={
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repeat_penalty": repeat_penalty,
                    "num_ctx": num_ctx,
                    "num_thread": num_thread
                }
            ):
                new_text = chunk['message']['content']
                response_text += new_text
                yield new_text

    elif llm_backend == "openai":
        openai.base_url = CONFIG["llm"]["openai_base_url"]
        openai.api_key = CONFIG["llm"]["openai_api_key"]
        
        # Streaming generator for OpenAI (stateless per call)
        def llm_stream_chat(prompt):
            response_text = ""
            
            # Get OpenAI parameters from session state
            model = st.session_state.get(f'openai_model_{session_id}', CONFIG["llm"]["openai_model"])
            temperature = st.session_state.get(f'openai_temperature_{session_id}', 0.7)
            top_p = st.session_state.get(f'openai_top_p_{session_id}', 1.0)
            max_tokens = st.session_state.get(f'openai_max_tokens_{session_id}', 2048)
            presence_penalty = st.session_state.get(f'openai_presence_penalty_{session_id}', 0.0)
            frequency_penalty = st.session_state.get(f'openai_frequency_penalty_{session_id}', 0.0)
            logit_bias_str = st.session_state.get(f'openai_logit_bias_{session_id}', '{}')
            
            # Parse logit_bias if provided
            logit_bias = {}
            try:
                if logit_bias_str and logit_bias_str != '{}':
                    logit_bias = json.loads(logit_bias_str)
            except json.JSONDecodeError:
                pass
            
            stream = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias if logit_bias else None
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                response_text += delta
                yield delta

    # Add timestamp to make keys even more unique
    import time
    timestamp = int(time.time())
    
    # Display analysis results
    for prompt_idx, prompt in enumerate(prompts, 1):
        col_prompt, col_response = st.columns([1, 1])
        with col_prompt:
            # Create bounded container for auto-scrolling prompt
            prompt_container = st.container(height=400)
            
            with prompt_container:
                # Use chat message structure for prompt display
                with st.chat_message("user"):
                    prompt_placeholder = st.empty()
                    
                    # Simulated streaming for prompt (10 words at a time)
                    words = prompt.split()
                    streamed_prompt = ""
                    for chunk_idx in range(0, len(words), 10):
                        chunk_words = words[chunk_idx:chunk_idx + 10]
                        streamed_prompt += " ".join(chunk_words) + " "
                        prompt_placeholder.text(streamed_prompt.strip())
                
                # Add disabled chat input for auto-scroll with unique key including timestamp
                st.chat_input(placeholder="", disabled=True, key=f"workflow_prompt_{timestamp}_{prompt_idx}_{session_id}")
        
        with col_response:
            # Create bounded container for auto-scrolling response
            response_container = st.container(height=400)
            
            with response_container:
                # Use chat message structure for response display
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    
                    # Stream the response using selected LLM
                    response_text = ""
                    for chunk in llm_stream_chat(prompt):
                        response_text += chunk
                        response_placeholder.write(response_text)
                    
                    # Write response_text to file (overwrite for first, append for others), filtering <think>...</think>
                    cleaned_response = remove_think_blocks(response_text)
                    if prompt_idx == 1:
                        with open(result_file, "w", encoding="utf-8") as f:
                            f.write('提取以下文字中有关"特殊特性符号（/、 ★和☆）不一致"的地方，如果下文字未提及任何"特殊特性符号（/、 ★和☆）不一致"的地方，则输出"比对结果为全部一致，但是有些参数的特殊性作符号在控制计划中没有定义"\n')
                            f.write(cleaned_response)
                    else:
                        with open(result_file, "a", encoding="utf-8") as f:
                            f.write(cleaned_response)
                
                # Add disabled chat input for auto-scroll with unique key including timestamp
                st.chat_input(placeholder="", disabled=True, key=f"workflow_response_{timestamp}_{prompt_idx}_{session_id}")
    
    # --- 特殊特性符号不一致结论 (symbol_check_final) ---
    # Read the content of 2_symbol_check_result.txt as the new prompt
    symbol_check_final_file = os.path.join(special_dir, "2_symbol_check_result.txt")
    if os.path.exists(symbol_check_final_file):
        with open(symbol_check_final_file, "r", encoding="utf-8") as f:
            symbol_check_final_prompt = f.read()
        # Augment prompt to require including an explicit parameter field per item
        _parameter_requirement = "\n\n请在每条结论中明确提供一个\"参数\"字段（可称为\"参数\"/\"目标参数\"/\"零件参数\"等），用于标示相关的参数名称。"
        # Also require explicit file references (目标文件与控制计划文件)，包含文件名与工作表（若已知）
        _file_fields_requirement = (
            "\n请在每条结论中同时给出对应的\"目标文件\"与\"控制计划文件\"来源（若可判定），"
            "格式为：目标文件：<文件名>/<Sheet名>；控制计划：<文件名>/<Sheet名>。若未知请留空字符串。"
        )
        symbol_check_final_prompt_aug = symbol_check_final_prompt + _parameter_requirement + _file_fields_requirement
        
        # Display the prompt and response side by side
        col_final_prompt, col_final_response = st.columns([1, 1])
        with col_final_prompt:
            st.subheader("特殊特性符号不一致结论 - 提示词:")
            prompt_container = st.container(height=400)
            with prompt_container:
                with st.chat_message("user"):
                    prompt_placeholder = st.empty()
                    prompt_placeholder.text(symbol_check_final_prompt_aug)
                
                st.chat_input(placeholder="", disabled=True, key=f"workflow_final_prompt_{timestamp}_{session_id}")
        
        with col_final_response:
            st.subheader("特殊特性符号不一致结论 - AI回复:")
            response_container = st.container(height=400)
            with response_container:
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    
                    # Stream the final response using selected LLM
                    symbol_check_final_response = ""
                    if llm_backend in ("ollama_127", "ollama_9"):
                        for chunk in ollama_client.chat(
                            model=st.session_state.get(f'ollama_model_{session_id}', CONFIG["llm"]["ollama_model"]),
                            messages=[{"role": "user", "content": symbol_check_final_prompt_aug}],
                            stream=True,
                            options={
                                "temperature": st.session_state.get(f'ollama_temperature_{session_id}', 0.7),
                                "top_p": st.session_state.get(f'ollama_top_p_{session_id}', 0.9),
                                "top_k": st.session_state.get(f'ollama_top_k_{session_id}', 40),
                                "repeat_penalty": st.session_state.get(f'ollama_repeat_penalty_{session_id}', 1.1),
                                "num_ctx": st.session_state.get(f'ollama_num_ctx_{session_id}', 65536),
                                "num_thread": st.session_state.get(f'ollama_num_thread_{session_id}', 4)
                            }
                        ):
                            new_text = chunk['message']['content']
                            symbol_check_final_response += new_text
                            response_placeholder.write(symbol_check_final_response)
                    elif llm_backend == "openai":
                        stream = openai.chat.completions.create(
                            model=st.session_state.get(f'openai_model_{session_id}', CONFIG["llm"]["openai_model"]),
                            messages=[{"role": "user", "content": symbol_check_final_prompt_aug}],
                            stream=True,
                            temperature=st.session_state.get(f'openai_temperature_{session_id}', 0.7),
                            top_p=st.session_state.get(f'openai_top_p_{session_id}', 1.0),
                            max_tokens=st.session_state.get(f'openai_max_tokens_{session_id}', 2048),
                            presence_penalty=st.session_state.get(f'openai_presence_penalty_{session_id}', 0.0),
                            frequency_penalty=st.session_state.get(f'openai_frequency_penalty_{session_id}', 0.0)
                        )
                        for chunk in stream:
                            delta = chunk.choices[0].delta.content or ""
                            symbol_check_final_response += delta
                            response_placeholder.write(symbol_check_final_response)
                    
                    # Save only the AI response to the subfolder
                    result_path = os.path.join(special_dir, "3_special_symbols_check_result.txt")
                    with open(result_path, "w", encoding="utf-8") as f:
                        f.write(symbol_check_final_response)
                
                st.chat_input(placeholder="", disabled=True, key=f"workflow_final_response_{timestamp}_{session_id}")

    # --- JSON 转换步骤（无标题、无分割线）---
    # 读取上一步保存的文本并构建“转换为 JSON”的提示词
    json_source_file = os.path.join(special_dir, "3_special_symbols_check_result.txt")
    if os.path.exists(json_source_file):
        try:
            with open(json_source_file, "r", encoding="utf-8") as f:
                source_text_for_json = f.read()
        except Exception:
            source_text_for_json = ""
        
        # 参考文件齐套性检查中的严格 JSON 要求，构造仅输出 JSON 的提示词
        # 更新字段：
        # - issue: 要求为“参数”（如 参数/目标参数/零件参数），而非问题描述
        # - location: 拆分为 target_file 与 control_plan_file 两列
        json_conversion_prompt = (
            "请将以下内容转换为严格的 JSON 对象，并且仅输出 JSON（不要输出解释、Markdown 或其他文本）。\n"
            "请将结果组织为：{\n"
            "  \"items\": [\n"
            "    {\n"
            "      \"issue\": \"参数名称（如 参数/目标参数/零件参数 等）\",\n"
            "      \"target_file\": \"目标文件：<文件名>/<Sheet名>，未知则空字符串\",\n"
            "      \"control_plan_file\": \"控制计划：<文件名>/<Sheet名>，未知则空字符串\",\n"
            "      \"note\": \"可选的补充说明，没有则空字符串\",\n"
            "      \"suggestion\": \"简短的修订建议，没有则空字符串\"\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "请确保 issue 字段填写为参数名称，而非问题描述。\n\n"
            "内容如下：\n"
            f"{source_text_for_json}"
        )
        
        col_json_left, col_json_right = st.columns([1, 1])
        with col_json_left:
            # 左侧显示提示词（无标题）
            left_container = st.container(height=400)
            with left_container:
                with st.chat_message("user"):
                    ph_left = st.empty()
                    ph_left.text(json_conversion_prompt)
            st.chat_input(placeholder="", disabled=True, key=f"special_symbols_json_prompt_{timestamp}_{session_id}")
        
        with col_json_right:
            # 右侧流式显示 LLM 回复（无标题）
            right_container = st.container(height=400)
            with right_container:
                with st.chat_message("assistant"):
                    ph_right = st.empty()
                    json_response_text = ""
                    if llm_backend in ("ollama_127", "ollama_9"):
                        for chunk in ollama_client.chat(
                            model=st.session_state.get(f'ollama_model_{session_id}', CONFIG["llm"]["ollama_model"]),
                            messages=[{"role": "user", "content": json_conversion_prompt}],
                            stream=True,
                            options={
                                "temperature": st.session_state.get(f'ollama_temperature_{session_id}', 0.7),
                                "top_p": st.session_state.get(f'ollama_top_p_{session_id}', 0.9),
                                "top_k": st.session_state.get(f'ollama_top_k_{session_id}', 40),
                                "repeat_penalty": st.session_state.get(f'ollama_repeat_penalty_{session_id}', 1.1),
                                "num_ctx": st.session_state.get(f'ollama_num_ctx_{session_id}', 100000),
                                "num_thread": st.session_state.get(f'ollama_num_thread_{session_id}', 4),
                                "format": "json",
                            }
                        ):
                            new_text = chunk['message']['content']
                            json_response_text += new_text
                            ph_right.write(json_response_text)
                    elif llm_backend == "openai":
                        stream = openai.chat.completions.create(
                            model=st.session_state.get(f'openai_model_{session_id}', CONFIG["llm"]["openai_model"]),
                            messages=[{"role": "user", "content": json_conversion_prompt}],
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
                            ph_right.write(json_response_text)
            st.chat_input(placeholder="", disabled=True, key=f"special_symbols_json_response_{timestamp}_{session_id}")

        # 解析 JSON 并导出为 Excel，显示在页面
        parsed_json = None
        try:
            parsed_json = json.loads(json_response_text)
        except Exception:
            # 尝试从代码块/杂文中提取 JSON 对象
            try:
                cleaned = (json_response_text or "").strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.strip('`')
                    idx = cleaned.find("{")
                    if idx >= 0:
                        cleaned = cleaned[idx:]
                start = cleaned.find('{')
                end = cleaned.rfind('}')
                if start >= 0 and end > start:
                    cleaned = cleaned[start:end+1]
                parsed_json = json.loads(cleaned)
            except Exception:
                parsed_json = None

        if isinstance(parsed_json, dict):
            items = parsed_json.get('items')
            if isinstance(items, list) and items:
                # 归一化为表格
                try:
                    df = pd.DataFrame(items)
                except Exception:
                    df = pd.DataFrame()
                if not df.empty:
                    try:
                        from datetime import datetime as _dt
                        ts = _dt.now().strftime('%Y%m%d_%H%M%S')
                        excel_path = os.path.join(special_dir, f"special_symbols_json_result_{ts}.xlsx")
                        df.to_excel(excel_path, index=False, engine='openpyxl')
                        st.success(f"已导出 JSON 结果为 Excel: {os.path.basename(excel_path)}")
                        # 展示在页面
                        st.dataframe(df, use_container_width=True)
                    except Exception as e:
                        st.warning(f"导出或展示 Excel 失败: {e}")
            else:
                st.info("JSON 中未找到可用的 items 列表。")

    st.info("✅ 分析完成")

def render_special_symbols_check_tab(session_id):
    # Handle None session_id (user not logged in)
    if session_id is None:
        st.warning("请先登录以使用此功能。")
        return
    
    # Add CSS to hide chat input scoped ONLY to this tab's container
    with st.container(key="special_symbols_root"):
        st.markdown("""
        <style>
        .st-key-special_symbols_root [data-testid="stChatInput"] { display: none; }
        </style>
        """, unsafe_allow_html=True)
    
    # Page subheader
    st.subheader("🔍 特殊特性符号检查")
    
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
    special_dir = session_dirs.get("generated_special_symbols_check", os.path.join(generated_session_dir, "special_symbols_check"))
    os.makedirs(special_dir, exist_ok=True)

    # Initialize prompt generator
    prompt_generator = SimplePromptGenerator(session_id)

    # Layout: right column for info, left for main content
    col_main, col_info = st.columns([2, 1])

    # Render the file/info column FIRST so file lists appear immediately after clicking buttons
    with col_info:
        # Early bulk operations: handle clear-all before listing files so UI updates immediately
        backend_available = is_backend_available()
        sess_for_clear = get_user_session(session_id, 'special_symbols')
        workflow_safe_for_clear = not sess_for_clear['process_started'] or sess_for_clear['analysis_completed']
        if workflow_safe_for_clear:
            col_clear_cp, col_clear_target, col_clear_graph = st.columns(3)
            with col_clear_cp:
                if st.button("🗑️ 清空控制计划文件", key=f"clear_cp_files_{session_id}"):
                    if backend_available:
                        try:
                            client = get_backend_client()
                            # No dedicated endpoint per bucket; list and delete
                            result = client.list_files(session_id, file_type="cp")
                            deleted = 0
                            for fi in result.get("cp", []):
                                del_res = client.delete_file(session_id, os.path.join(cp_session_dir, fi["name"]))
                                if del_res.get("status") == "success":
                                    deleted += 1
                            st.success(f"已清空控制计划文件（{deleted} 个）")
                        except Exception as e:
                            st.error(f"清空失败: {e}")
                    else:
                        try:
                            for file in os.listdir(cp_session_dir):
                                file_path = os.path.join(cp_session_dir, file)
                                if os.path.isfile(file_path):
                                    os.remove(file_path)
                            st.success("已清空控制计划文件")
                        except Exception as e:
                            st.error(f"清空失败: {e}")
            with col_clear_target:
                if st.button("🗑️ 清空待检查文件", key=f"clear_target_files_{session_id}"):
                    if backend_available:
                        try:
                            client = get_backend_client()
                            result = client.list_files(session_id, file_type="target")
                            deleted = 0
                            for fi in result.get("target", []):
                                del_res = client.delete_file(session_id, os.path.join(target_session_dir, fi["name"]))
                                if del_res.get("status") == "success":
                                    deleted += 1
                            st.success(f"已清空待检查文件（{deleted} 个）")
                        except Exception as e:
                            st.error(f"清空失败: {e}")
                    else:
                        try:
                            for file in os.listdir(target_session_dir):
                                file_path = os.path.join(target_session_dir, file)
                                if os.path.isfile(file_path):
                                    os.remove(file_path)
                            st.success("已清空待检查文件")
                        except Exception as e:
                            st.error(f"清空失败: {e}")
            with col_clear_graph:
                if st.button("🗑️ 清空图纸文件", key=f"clear_graph_files_{session_id}"):
                    if backend_available:
                        try:
                            client = get_backend_client()
                            result = client.list_files(session_id, file_type="graph")
                            deleted = 0
                            for fi in result.get("graph", []):
                                del_res = client.delete_file(session_id, os.path.join(graph_session_dir, fi["name"]))
                                if del_res.get("status") == "success":
                                    deleted += 1
                            st.success(f"已清空图纸文件（{deleted} 个）")
                        except Exception as e:
                            st.error(f"清空失败: {e}")
                    else:
                        try:
                            for file in os.listdir(graph_session_dir):
                                file_path = os.path.join(graph_session_dir, file)
                                if os.path.isfile(file_path):
                                    os.remove(file_path)
                            st.success("已清空图纸文件")
                        except Exception as e:
                            st.error(f"清空失败: {e}")
        else:
            st.info("🔄 分析进行中，请等待完成后再清空文件")
            st.button("🗑️ 清空控制计划文件", key=f"clear_cp_files_disabled_{session_id}", disabled=True)
            st.button("🗑️ 清空待检查文件", key=f"clear_target_files_disabled_{session_id}", disabled=True)
            st.button("🗑️ 清空图纸文件", key=f"clear_graph_files_disabled_{session_id}", disabled=True)

        # --- File Manager Module ---
        def get_file_list(folder):
            # Always use FastAPI backend
            try:
                client = get_backend_client()
                # Determine file type from folder path
                if "CP_files" in folder:
                    file_type = "cp"
                elif "target_files" in folder:
                    file_type = "target"
                elif "graph_files" in folder:
                    file_type = "graph"
                else:
                    # For other directories, return empty list
                    return []
                
                # Get files from backend
                result = client.list_files(session_id, file_type)
                if file_type in result:
                    files = []
                    for file_info in result[file_type]:
                        files.append({
                            'name': file_info['name'],
                            'size': file_info['size'],
                            'modified': float(file_info['modified_time']),
                            'path': os.path.join(folder, file_info['name'])
                        })
                    # Use stable sorting by name first, then by modification time
                    return sorted(files, key=lambda x: (x['name'].lower(), x['modified']), reverse=False)
                else:
                    return []
            except Exception as e:
                st.error(f"后端连接失败: {e}")
                return []
        
        def get_file_list_direct(folder):
            """Direct file system access as fallback"""
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
            
            # Split filename and extension
            name, ext = os.path.splitext(filename)
            # Calculate how much space we have for the name part
            available_length = max_length - len(ext) - 3  # 3 for "..."
            
            if available_length <= 0:
                # If extension is too long, just truncate the whole thing
                return filename[:max_length-3] + "..."
            
            # Truncate name part and add ellipsis
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
                            st.write(f"**文件名:** {file_info['name']}")  # Show full name inside
                            st.write(f"**大小:** {format_file_size(file_info['size'])}")
                            st.write(f"**修改时间:** {format_timestamp(file_info['modified'])}")
                        with col_action:
                            # Use a more stable key for delete button
                            delete_key = f"delete_cp_{file_info['name'].replace(' ', '_').replace('.', '_')}_{session_id}"
                            
                            # Check if workflow is safe to interrupt
                            session = get_user_session(session_id, 'special_symbols')
                            workflow_safe = not session['process_started'] or session['analysis_completed']
                            
                            if workflow_safe:
                                if st.button("🗑️ 删除", key=delete_key):
                                    try:
                                        if is_backend_available():
                                            # Use FastAPI backend
                                            client = get_backend_client()
                                            result = client.delete_file(session_id, file_info['path'])
                                            if result.get("status") == "success":
                                                st.success(f"已删除: {file_info['name']}")
                                            else:
                                                st.error(f"删除失败: {result.get('message', '未知错误')}")
                                        else:
                                            # Fallback to direct file system access
                                            os.remove(file_info['path'])
                                            st.success(f"已删除: {file_info['name']}")
                                    except Exception as e:
                                        st.error(f"删除失败: {e}")
                            else:
                                # Show disabled button during workflow
                                st.button("🗑️ 删除", key=f"{delete_key}_disabled", disabled=True)
            else:
                st.write("（未上传）")
                
            # Upload new files directly in this tab
            st.markdown("---")
            st.markdown("**上传新文件:**")
            new_cp_files = st.file_uploader("选择控制计划文件", type=None, accept_multiple_files=True, key=f"cp_uploader_tab_{session_id}")
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
                            st.write(f"**文件名:** {file_info['name']}")  # Show full name inside
                            st.write(f"**大小:** {format_file_size(file_info['size'])}")
                            st.write(f"**修改时间:** {format_timestamp(file_info['modified'])}")
                        with col_action:
                            # Use a more stable key for delete button
                            delete_key = f"delete_target_{file_info['name'].replace(' ', '_').replace('.', '_')}_{session_id}"
                            
                            # Check if workflow is safe to interrupt
                            session = get_user_session(session_id, 'special_symbols')
                            workflow_safe = not session['process_started'] or session['analysis_completed']
                            
                            if workflow_safe:
                                if st.button("🗑️ 删除", key=delete_key):
                                    try:
                                        if is_backend_available():
                                            # Use FastAPI backend
                                            client = get_backend_client()
                                            result = client.delete_file(session_id, file_info['path'])
                                            if result.get("status") == "success":
                                                st.success(f"已删除: {file_info['name']}")
                                            else:
                                                st.error(f"删除失败: {result.get('message', '未知错误')}")
                                        else:
                                            # Fallback to direct file system access
                                            os.remove(file_info['path'])
                                            st.success(f"已删除: {file_info['name']}")
                                    except Exception as e:
                                        st.error(f"删除失败: {e}")
                            else:
                                # Show disabled button during workflow
                                st.button("🗑️ 删除", key=f"{delete_key}_disabled", disabled=True)
            else:
                st.write("（未上传）")
                
            # Upload new files directly in this tab
            st.markdown("---")
            st.markdown("**上传新文件:**")
            new_target_files = st.file_uploader("选择待检查文件", type=None, accept_multiple_files=True, key=f"target_uploader_tab_{session_id}")
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
                            st.write(f"**文件名:** {file_info['name']}")  # Show full name inside
                            st.write(f"**大小:** {format_file_size(file_info['size'])}")
                            st.write(f"**修改时间:** {format_timestamp(file_info['modified'])}")
                        with col_action:
                            # Use a more stable key for delete button
                            delete_key = f"delete_graph_{file_info['name'].replace(' ', '_').replace('.', '_')}_{session_id}"
                            
                            # Check if workflow is safe to interrupt
                            session = get_user_session(session_id, 'special_symbols')
                            workflow_safe = not session['process_started'] or session['analysis_completed']
                            
                            if workflow_safe:
                                if st.button("🗑️ 删除", key=delete_key):
                                    try:
                                        if is_backend_available():
                                            # Use FastAPI backend
                                            client = get_backend_client()
                                            result = client.delete_file(session_id, file_info['path'])
                                            if result.get("status") == "success":
                                                st.success(f"已删除: {file_info['name']}")
                                            else:
                                                st.error(f"删除失败: {result.get('message', '未知错误')}")
                                        else:
                                            # Fallback to direct file system access
                                            os.remove(file_info['path'])
                                            st.success(f"已删除: {file_info['name']}")
                                    except Exception as e:
                                        st.error(f"删除失败: {e}")
                            else:
                                # Show disabled button during workflow
                                st.button("🗑️ 删除", key=f"{delete_key}_disabled", disabled=True)
            else:
                st.write("（未上传）")
                
            # Upload new files directly in this tab
            st.markdown("---")
            st.markdown("**上传新文件:**")
            new_graph_files = st.file_uploader("选择图纸文件", type=None, accept_multiple_files=True, key=f"graph_uploader_tab_{session_id}")
            if new_graph_files:
                handle_file_upload(new_graph_files, graph_session_dir)

        # (Bulk operations moved earlier to avoid UI lag)

    # Now render the MAIN column containing uploads, demo/start, and streaming analysis
    with col_main:
        # Get structured user session
        session = get_user_session(session_id, 'special_symbols')

        # Always show file upload section
        render_file_upload_section(session_dirs, session_id)
        
        # Start button - only show if process hasn't started
        if not session['process_started']:
            col_buttons = st.columns([1, 1])
            with col_buttons[0]:
                if st.button("开始", key=f"start_button_{session_id}"):
                    # Clear any existing generated files to ensure fresh generation
                    output_file = os.path.join(special_dir, "prompt_output.txt")
                    result_file = os.path.join(special_dir, "2_symbol_check_result.txt")
                    
                    if os.path.exists(output_file):
                        os.remove(output_file)
                    if os.path.exists(result_file):
                        os.remove(result_file)
                    
                    # Clear chat history for fresh analysis
                    session['ollama_history'] = []
                    session['openai_history'] = []
                    session['analysis_completed'] = False
                    
                    # Start the analysis process
                    start_analysis(session_id, 'special_symbols')
                    st.rerun()
            with col_buttons[1]:
                if st.button("演示", key=f"demo_button_{session_id}"):
                    # Workflow-based approach: Start → Run → Finish
                    
                    # 2. Run demo workflow
                    demo_base_dir = CONFIG["directories"]["cp_files"].parent / "demonstration"
                    
                    # Copy files from demonstration folders to session folders
                    demo_folder_mapping = {
                        "CP_files": "cp",
                        "graph_files": "graph", 
                        "target_files": "target"
                    }
                    
                    files_copied = False
                    for demo_folder, session_key in demo_folder_mapping.items():
                        demo_folder_path = os.path.join(demo_base_dir, demo_folder)
                        session_folder_path = session_dirs[session_key]
                        
                        if os.path.exists(demo_folder_path):
                            # Copy all files from demo folder to session folder
                            for file_name in os.listdir(demo_folder_path):
                                demo_file_path = os.path.join(demo_folder_path, file_name)
                                session_file_path = os.path.join(session_folder_path, file_name)
                                
                                if os.path.isfile(demo_file_path):
                                    import shutil
                                    shutil.copy2(demo_file_path, session_file_path)
                                    files_copied = True
                    
                    # Copy pre-generated prompt file (but not result file)
                    demo_prompt_file = os.path.join(demo_base_dir, "generated_files", "prompt_output.txt")
                    session_prompt_file = os.path.join(special_dir, "prompt_output.txt")
                    
                    if os.path.exists(demo_prompt_file):
                        import shutil
                        shutil.copy2(demo_prompt_file, session_prompt_file)
                    
                    if files_copied:
                        # Set up session for analysis
                        session['analysis_completed'] = False
                        session['process_started'] = True
                        session['ollama_history'] = []
                        session['openai_history'] = []
                        
                        # Force page refresh to hide buttons and show analysis
                        st.rerun()
                    else:
                        st.error("演示文件不存在，请检查演示文件夹")
        
        # Show status and reset button if process has started
        if session['process_started']:
            # Add a button to reset and clear history
            if st.button("重新开始", key=f"reset_button_{session_id}"):
                reset_user_session(session_id, 'special_symbols')
                st.rerun()
            
            # Check if we need to run analysis
            target_files_list = [f for f in os.listdir(target_session_dir) if os.path.isfile(os.path.join(target_session_dir, f))]
            if target_files_list:
                if session['process_started'] and not session['analysis_completed']:
                    # Run the analysis workflow in full width within the main column
                    run_analysis_workflow(session_id, session_dirs, prompt_generator)
                    
                    # Mark as completed
                    session['analysis_completed'] = True
                else:
                    # Files exist but process wasn't explicitly started
                    st.info("检测到待检查文件，请点击\"开始\"按钮开始分析，或点击\"演示\"按钮使用演示文件。")
            else:
                st.warning("请先上传待检查文件")
