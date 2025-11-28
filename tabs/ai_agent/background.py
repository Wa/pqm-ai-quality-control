"""Background runner for AI Agent (LangGraph + MCP tools)."""
from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

from ollama import Client as OllamaClient

from config import CONFIG
from tabs.shared.modelscope_client import ModelScopeClient
from util import ensure_session_dirs, resolve_ollama_host

from .graph import AgentState, build_agent_graph
from .mcp_tools import (
    get_agent_paths,
    prepare_conversation_dirs,
    tool_convert_to_text,
    tool_filesystem,
    tool_http_fetch,
    tool_python_exec,
    tool_web_search,
)


def _extract_message_text(resp: Any) -> str:
    if resp is None:
        return ""

    message_obj = getattr(resp, "message", None)
    if message_obj is not None:
        content = getattr(message_obj, "content", None)
        if content:
            return content
        if isinstance(message_obj, dict):
            content = message_obj.get("content")
            if content:
                return str(content)

    data: Optional[Dict[str, Any]] = None
    if isinstance(resp, dict):
        data = resp
    else:
        for attr in ("model_dump", "dict"):
            if hasattr(resp, attr):
                try:
                    data = getattr(resp, attr)()
                    break
                except Exception:
                    data = None

    if data is not None:
        return str((data.get("message") or {}).get("content") or data.get("response") or data.get("text") or "")

    return ""


def _extract_tool_calls_from_chunk(chunk) -> Optional[List[Dict[str, Any]]]:
    """Extract tool_calls from a ChatResponse chunk if available."""
    if not hasattr(chunk, 'message'):
        return None
    message_obj = chunk.message
    if not hasattr(message_obj, 'tool_calls') or not message_obj.tool_calls:
        return None
    return message_obj.tool_calls


def _parse_json_block(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        import re

        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                return None
    return None


def _build_provider_sequence(primary: str, turbo_mode: bool) -> List[Tuple[str, Callable[[], Any], str, str]]:
    provider_sequence: List[Tuple[str, Callable[[], Any], str, str]] = []

    modelscope_api_key = CONFIG["llm"].get("modelscope_api_key")
    modelscope_model = CONFIG["llm"].get("modelscope_model", "deepseek-ai/DeepSeek-V3.1")
    modelscope_base = CONFIG["llm"].get("modelscope_base_url", "https://api-inference.modelscope.cn/v1")

    if turbo_mode and modelscope_api_key:
        def _mk_modelscope() -> ModelScopeClient:
            return ModelScopeClient(api_key=modelscope_api_key, model=modelscope_model, base_url=modelscope_base)

        provider_sequence.append(("ModelScope", _mk_modelscope, modelscope_model, "modelscope"))

    local_model = CONFIG["llm"].get("ollama_model") or "gpt-oss:latest"
    cloud_model = "gpt-oss:20b-cloud"

    cloud_host = CONFIG["llm"].get("ollama_cloud_host")
    cloud_key = CONFIG["llm"].get("ollama_cloud_api_key")

    def _mk_local() -> OllamaClient:
        host = resolve_ollama_host("ollama_9")
        return OllamaClient(host=host)

    def _mk_cloud() -> OllamaClient:
        if not cloud_host or not cloud_key:
            raise RuntimeError("Cloud provider not configured")
        return OllamaClient(host=cloud_host, headers={"Authorization": f"Bearer {cloud_key}"})

    if primary == "local":
        provider_sequence.append(("本地 gpt-oss", _mk_local, local_model, "ollama"))
        if cloud_host and cloud_key:
            provider_sequence.append(("云端 gpt-oss", _mk_cloud, cloud_model, "ollama"))
    else:
        if cloud_host and cloud_key:
            provider_sequence.append(("云端 gpt-oss", _mk_cloud, cloud_model, "ollama"))
        provider_sequence.append(("本地 gpt-oss", _mk_local, local_model, "ollama"))

    return provider_sequence


def _make_chat_completion(
    provider_sequence: List[Tuple[str, Callable[[], Any], str, str]],
    publish: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Callable[[List[Dict[str, str]]], str]:
    def _emit(stage: str, message: str) -> None:
        if publish:
            payload = {"status": "running", "stage": stage, "message": message}
            publish(payload)

    def chat_completion(messages: List[Dict[str, str]]) -> str:
        last_err: Optional[Exception] = None
        last_empty_response_label: Optional[str] = None

        _emit("llm_call", f"开始尝试LLM调用，提供商序列: {[label for label, _, _, _ in provider_sequence]}")
        
        # Debug: Log message structure
        _emit("llm_call", f"[DEBUG] 消息总数: {len(messages)}")
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))
            content_preview = content[:200] + ("..." if len(content) > 200 else "")
            has_web_search = "web_search" in content.lower() or "搜索" in content or "search" in content.lower()
            _emit("llm_call", f"[DEBUG] 消息[{i}] role={role}, len={len(content)}, has_web_search={has_web_search}, preview={content_preview}")

        for label, factory, model, engine in provider_sequence:
            _emit("llm_call", f"尝试提供商: {label} (模型: {model}, 引擎: {engine})")
            try:
                _emit("llm_call", f"正在创建 {label} 客户端...")
                client = factory()
                _emit("llm_call", f"{label} 客户端创建成功")

                if engine == "modelscope":
                    _emit("llm_call", f"调用 {label} (非流式)...")
                    resp = client.chat(model=model, messages=messages, stream=False, options={"num_ctx": 40001})
                    _emit(
                        "llm_call",
                        f"{label} 响应接收完成，响应类型: {type(resp)}, 响应键: {list(resp.keys()) if isinstance(resp, dict) else 'N/A'}",
                    )
                    text = _extract_message_text(resp)
                    _emit("llm_call", f"{label} 提取的文本长度: {len(text)}, 前100字符: {text[:100] if text else '(空)'}")
                    if text and str(text).strip():
                        _emit("llm_call", f"✓ 成功调用 {label}，返回文本长度: {len(text)}")
                        return text
                    _emit("llm_call", f"✗ {label} 返回空响应")
                    last_empty_response_label = label
                else:
                    _emit("llm_call", f"调用 {label} (流式)...")
                    _emit("llm_call", f"[DEBUG] {label} 开始流式调用，模型={model}, 消息数={len(messages)}")
                    response_text = ""
                    chunk_count = 0
                    first_chunk_logged = False
                    last_chunk_sample = None
                    empty_chunk_count = 0
                    non_empty_chunk_count = 0
                    
                    try:
                        stream_iterator = client.chat(
                            model=model,
                            messages=messages,
                            stream=True,
                            options={"num_ctx": 40001},
                        )
                        _emit("llm_call", f"[DEBUG] {label} 流式迭代器已创建，开始迭代...")
                        
                        for chunk in stream_iterator:
                            chunk_count += 1
                            
                            # Log first chunk structure in detail
                            if not first_chunk_logged:
                                first_chunk_logged = True
                                chunk_type = type(chunk).__name__
                                _emit("llm_call", f"[DEBUG] {label} 第一个chunk: type={chunk_type}")
                                if isinstance(chunk, dict):
                                    chunk_keys = list(chunk.keys())
                                    _emit("llm_call", f"[DEBUG] {label} 第一个chunk键: {chunk_keys}")
                                    # Log all key-value pairs (truncated)
                                    for key in chunk_keys:
                                        value = chunk[key]
                                        value_str = str(value)
                                        if len(value_str) > 200:
                                            value_str = value_str[:200] + "..."
                                        _emit("llm_call", f"[DEBUG] {label} chunk['{key}'] = {value_str}")
                                elif hasattr(chunk, 'message'):
                                    # ChatResponse object
                                    _emit("llm_call", f"[DEBUG] {label} 第一个chunk是ChatResponse对象")
                                    _emit("llm_call", f"[DEBUG] {label} ChatResponse属性: {[attr for attr in dir(chunk) if not attr.startswith('_')]}")
                                    message_obj = chunk.message
                                    if message_obj:
                                        _emit("llm_call", f"[DEBUG] {label} message对象属性: {[attr for attr in dir(message_obj) if not attr.startswith('_')]}")
                                        # Check all possible content fields
                                        content_val = getattr(message_obj, 'content', None)
                                        thinking_val = getattr(message_obj, 'thinking', None)
                                        text_val = getattr(message_obj, 'text', None)
                                        _emit("llm_call", f"[DEBUG] {label} message.content = {repr(content_val)}")
                                        _emit("llm_call", f"[DEBUG] {label} message.thinking = {repr(thinking_val)}")
                                        _emit("llm_call", f"[DEBUG] {label} message.text = {repr(text_val)}")
                                        _emit("llm_call", f"[DEBUG] {label} message.role = {getattr(message_obj, 'role', 'N/A')}")
                                        # Check if message_obj itself is a string or has other attributes
                                        if isinstance(message_obj, str):
                                            _emit("llm_call", f"[DEBUG] {label} message对象本身是字符串: {message_obj[:100]}")
                                        # Try to see all attributes and their values
                                        for attr in ['content', 'thinking', 'text', 'response', 'role', 'tool_calls']:
                                            if hasattr(message_obj, attr):
                                                val = getattr(message_obj, attr)
                                                _emit("llm_call", f"[DEBUG] {label} message.{attr} = {repr(val)[:200]}")
                                else:
                                    chunk_str = str(chunk)
                                    _emit("llm_call", f"[DEBUG] {label} 第一个chunk值: {chunk_str[:500]}")
                            
                            # Try to extract content
                            piece = None
                            if isinstance(chunk, dict):
                                # Try multiple extraction methods
                                msg_content = chunk.get("message", {}).get("content") if isinstance(chunk.get("message"), dict) else None
                                response_val = chunk.get("response")
                                content_val = chunk.get("content")
                                text_val = chunk.get("text")
                                
                                piece = msg_content or response_val or content_val or text_val or ""
                                
                                # Debug: log what we found
                                if chunk_count <= 3:  # Log first 3 chunks
                                    _emit("llm_call", f"[DEBUG] {label} chunk[{chunk_count}]: msg.content={msg_content}, response={response_val}, content={content_val}, text={text_val}, piece={piece[:50] if piece else '(empty)'}")
                                
                                # Check for done flag
                                if chunk.get("done") is True:
                                    _emit("llm_call", f"[DEBUG] {label} chunk[{chunk_count}] 包含 done=True")
                                
                                last_chunk_sample = {
                                    "chunk_num": chunk_count,
                                    "keys": list(chunk.keys()),
                                    "has_content": bool(piece),
                                    "done": chunk.get("done"),
                                }
                            elif isinstance(chunk, str):
                                piece = chunk
                                if chunk_count <= 3:
                                    _emit("llm_call", f"[DEBUG] {label} chunk[{chunk_count}] 是字符串类型，长度={len(chunk)}, 内容={chunk[:100]}")
                            else:
                                # Handle ChatResponse objects from Ollama
                                # Check if it's a ChatResponse-like object with message attribute
                                if hasattr(chunk, 'message'):
                                    message_obj = chunk.message
                                    
                                    # Try multiple ways to extract content
                                    # 1. Try content field (standard)
                                    if hasattr(message_obj, 'content'):
                                        piece = message_obj.content or ""
                                        if piece is None:
                                            piece = ""
                                    
                                    # 2. If content is empty, check if thinking contains the actual response
                                    # Some models stream to thinking field instead of content
                                    if not piece and hasattr(message_obj, 'thinking'):
                                        thinking_val = message_obj.thinking or ""
                                        if thinking_val:
                                            # For some models, thinking might be the actual response stream
                                            piece = thinking_val
                                            if chunk_count <= 3:
                                                _emit("llm_call", f"[DEBUG] {label} chunk[{chunk_count}] 使用thinking字段作为内容: {repr(piece[:50])}")
                                    
                                    # 3. Try other possible fields
                                    if not piece:
                                        for field_name in ['text', 'response', 'output']:
                                            if hasattr(message_obj, field_name):
                                                field_val = getattr(message_obj, field_name)
                                                if field_val:
                                                    piece = str(field_val)
                                                    if chunk_count <= 3:
                                                        _emit("llm_call", f"[DEBUG] {label} chunk[{chunk_count}] 使用{field_name}字段作为内容: {repr(piece[:50])}")
                                                    break
                                    
                                    if chunk_count <= 3:
                                        content_val = getattr(message_obj, 'content', None) if hasattr(message_obj, 'content') else None
                                        thinking_val = getattr(message_obj, 'thinking', None) if hasattr(message_obj, 'thinking') else None
                                        _emit("llm_call", f"[DEBUG] {label} chunk[{chunk_count}] ChatResponse: content={repr(content_val)}, thinking={repr(thinking_val)}, piece={repr(piece[:50]) if piece else '(empty)'}")
                                    
                                    # Check for done flag and tool_calls
                                    tool_calls_detected = False
                                    json_extracted = False
                                    if hasattr(chunk, 'done') and chunk.done is True:
                                        _emit("llm_call", f"[DEBUG] {label} chunk[{chunk_count}] ChatResponse包含 done=True")
                                        # On final chunk, log all available fields
                                        _emit("llm_call", f"[DEBUG] {label} 最终chunk的所有message字段:")
                                        for attr in ['content', 'thinking', 'text', 'response', 'role', 'tool_calls']:
                                            if hasattr(message_obj, attr):
                                                val = getattr(message_obj, attr)
                                                if val:
                                                    _emit("llm_call", f"[DEBUG] {label}   message.{attr} = {repr(str(val)[:200])}")
                                        
                                        # Check for tool_calls - if present, we'll convert them to JSON format
                                        if hasattr(message_obj, 'tool_calls') and message_obj.tool_calls:
                                            tool_calls_detected = True
                                            _emit("llm_call", f"[DEBUG] {label} 检测到tool_calls，将转换为JSON格式")
                                            tool_calls = message_obj.tool_calls
                                            # Convert tool_calls to our JSON format
                                            if tool_calls and len(tool_calls) > 0:
                                                first_call = tool_calls[0]
                                                # Extract function name and arguments
                                                if hasattr(first_call, 'function'):
                                                    func = first_call.function
                                                    func_name = getattr(func, 'name', '') if hasattr(func, 'name') else ''
                                                    func_args = getattr(func, 'arguments', {}) if hasattr(func, 'arguments') else {}
                                                    
                                                    # Convert arguments if it's a string (JSON)
                                                    if isinstance(func_args, str):
                                                        try:
                                                            import json
                                                            func_args = json.loads(func_args)
                                                        except Exception:
                                                            func_args = {}
                                                    
                                                    # Build our expected JSON format using accumulated thinking as thought
                                                    thought_text = response_text.strip() if response_text.strip() else "使用工具完成任务"
                                                    converted_json = {
                                                        "thought": thought_text,
                                                        "tool": func_name,
                                                        "input": func_args
                                                    }
                                                    # Replace response_text with the converted JSON
                                                    import json
                                                    response_text = json.dumps(converted_json, ensure_ascii=False)
                                                    _emit("llm_call", f"[DEBUG] {label} 已将tool_calls转换为JSON: {response_text[:200]}")
                                                    piece = ""  # Don't add thinking content since we have JSON now
                                        else:
                                            # No tool_calls - try to extract JSON from thinking text or convert to JSON
                                            thinking_text = response_text.strip()
                                            if thinking_text:
                                                # First, try to parse as JSON (might be embedded)
                                                import json
                                                import re
                                                # Try to find JSON in the text
                                                json_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', thinking_text)
                                                if json_match:
                                                    try:
                                                        parsed = json.loads(json_match.group(0))
                                                        if isinstance(parsed, dict) and "tool" in parsed:
                                                            response_text = json_match.group(0)
                                                            _emit("llm_call", f"[DEBUG] {label} 从thinking文本中提取到JSON: {response_text[:200]}")
                                                            piece = ""
                                                            json_extracted = True
                                                    except Exception:
                                                        pass
                                                
                                                # If no JSON found, try to infer tool from text
                                                if not json_extracted:
                                                    tool_name = "none"
                                                    # Try to infer tool from thinking text
                                                    thinking_lower = thinking_text.lower()
                                                    if "web_search" in thinking_lower or "搜索" in thinking_text or "search" in thinking_lower:
                                                        tool_name = "web_search"
                                                        # Try to extract query
                                                        query_match = re.search(r'(?:query|搜索|search)[:：\s]+["\']?([^"\']+)["\']?', thinking_text, re.IGNORECASE)
                                                        query = query_match.group(1) if query_match else "latest information"
                                                        converted_json = {
                                                            "thought": thinking_text,
                                                            "tool": tool_name,
                                                            "input": {"query": query}
                                                        }
                                                        response_text = json.dumps(converted_json, ensure_ascii=False)
                                                        _emit("llm_call", f"[DEBUG] {label} 从thinking文本推断并转换为JSON: {response_text[:200]}")
                                                        piece = ""
                                                        json_extracted = True
                                                    elif "http_fetch" in thinking_lower or "fetch" in thinking_lower or "获取网页" in thinking_text:
                                                        tool_name = "http_fetch"
                                                        url_match = re.search(r'(?:url|网址)[:：\s]+(https?://[^\s]+)', thinking_text, re.IGNORECASE)
                                                        url = url_match.group(1) if url_match else "https://www.example.com"
                                                        converted_json = {
                                                            "thought": thinking_text,
                                                            "tool": tool_name,
                                                            "input": {"url": url}
                                                        }
                                                        response_text = json.dumps(converted_json, ensure_ascii=False)
                                                        _emit("llm_call", f"[DEBUG] {label} 从thinking文本推断并转换为JSON: {response_text[:200]}")
                                                        piece = ""
                                                        json_extracted = True
                                                    elif "完成" in thinking_text or "done" in thinking_lower or "结论" in thinking_text or "cannot" in thinking_lower or "unable" in thinking_lower:
                                                        # Task is complete or cannot proceed
                                                        converted_json = {
                                                            "thought": thinking_text,
                                                            "tool": "none",
                                                            "input": {}
                                                        }
                                                        response_text = json.dumps(converted_json, ensure_ascii=False)
                                                        _emit("llm_call", f"[DEBUG] {label} 从thinking文本推断任务完成: {response_text[:200]}")
                                                        piece = ""
                                                        json_extracted = True
                                    
                                    # If we detected tool_calls or converted to JSON, skip adding piece to avoid double accumulation
                                    if tool_calls_detected or json_extracted:
                                        piece = ""
                                    
                                    last_chunk_sample = {
                                        "chunk_num": chunk_count,
                                        "type": "ChatResponse",
                                        "has_content": bool(piece),
                                        "done": getattr(chunk, 'done', None),
                                        "message_content": getattr(message_obj, 'content', None),
                                        "message_thinking": getattr(message_obj, 'thinking', None) if hasattr(message_obj, 'thinking') else None,
                                        "message_content_len": len(getattr(message_obj, 'content', '')) if hasattr(message_obj, 'content') and getattr(message_obj, 'content') else 0,
                                        "message_thinking_len": len(getattr(message_obj, 'thinking', '')) if hasattr(message_obj, 'thinking') and getattr(message_obj, 'thinking') else 0,
                                    }
                                else:
                                    # Unknown type - don't convert to string, just skip
                                    _emit("llm_call", f"[DEBUG] {label} chunk[{chunk_count}] 未知类型且无message属性: {type(chunk)}, 跳过")
                                    piece = ""
                            
                            if piece:
                                response_text += piece
                                non_empty_chunk_count += 1
                                # Publish streaming chunk for UI updates
                                if publish:
                                    publish({"stream_chunk": piece})
                            else:
                                empty_chunk_count += 1
                                if chunk_count <= 5:  # Log first 5 empty chunks
                                    _emit("llm_call", f"[DEBUG] {label} chunk[{chunk_count}] 为空")
                        
                        _emit("llm_call", f"[DEBUG] {label} 流式迭代完成: 总chunks={chunk_count}, 空chunks={empty_chunk_count}, 有内容chunks={non_empty_chunk_count}")
                        if last_chunk_sample:
                            _emit("llm_call", f"[DEBUG] {label} 最后一个chunk样本: {last_chunk_sample}")
                        
                    except StopIteration:
                        _emit("llm_call", f"[DEBUG] {label} 迭代器抛出 StopIteration")
                    except Exception as stream_error:
                        _emit("llm_call", f"[DEBUG] {label} 流式迭代异常: {type(stream_error).__name__}: {str(stream_error)[:300]}")
                        raise
                    
                    _emit("llm_call", f"[DEBUG] {label} 最终response_text长度: {len(response_text)}, 内容预览: {response_text[:200] if response_text else '(empty)'}")
                    
                    if response_text and response_text.strip():
                        _emit("llm_call", f"✓ 成功调用 {label}，返回文本长度: {len(response_text)}")
                        return response_text
                    _emit("llm_call", f"✗ {label} 返回空响应")
                    _emit("llm_call", f"[DEBUG] {label} 空响应详情: chunk_count={chunk_count}, empty_chunks={empty_chunk_count}, non_empty_chunks={non_empty_chunk_count}, response_text_len={len(response_text)}")
                    last_empty_response_label = label
            except Exception as error:  # pragma: no cover - defensive
                last_err = error
                error_msg = str(error)[:500]
                error_type = type(error).__name__
                _emit("llm_call", f"✗ 调用 {label} 异常: {error_type}: {error_msg}")
                tb_str = traceback.format_exc()[:500]
                _emit("llm_call", f"{label} 异常堆栈: {tb_str}")
                continue

        error_summary = "所有LLM提供商均失败"
        if last_empty_response_label:
            error_summary += f"。{last_empty_response_label} 返回了空响应"
        if last_err:
            error_summary += f"。最后异常: {type(last_err).__name__}: {str(last_err)[:300]}"
        _emit("llm_call", f"✗ {error_summary}")
        raise RuntimeError(
            f"无法完成LLM调用: {error_summary}。已尝试的提供商: {[label for label, _, _, _ in provider_sequence]}"
        )

    return chat_completion


def generate_agent_plan(
    *,
    session_id: str,
    goal: str,
    turbo_mode: bool = False,
    primary: str = "local",
    conversation_history: Optional[List[Dict[str, str]]] = None,
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a structured execution plan for the agent."""

    base_dirs = {
        "uploads": str(CONFIG["directories"]["uploads"]),
        "generated_files": str(CONFIG["directories"]["generated_files"]),
    }
    session_dirs = ensure_session_dirs(base_dirs, session_id)
    session_dirs = prepare_conversation_dirs(session_dirs, conversation_id)
    paths = get_agent_paths(session_dirs)
    if paths.get("logs"):
        os.makedirs(paths["logs"], exist_ok=True)

    try:
        conversion = tool_convert_to_text(session_dirs)
    except Exception as exc:
        conversion = {"error": str(exc)}

    created_full = conversion.get("files", []) or []
    created_files = [os.path.basename(p) for p in created_full]
    existing_txts = conversion.get("existing_txts", []) or []
    uploads_listing = conversion.get("uploads_files", []) or []
    txt_dir_hint = conversion.get("examined_txt", "")

    provider_sequence = _build_provider_sequence(primary, turbo_mode)
    if not provider_sequence:
        raise RuntimeError("没有可用的LLM提供商用于规划")

    chat_completion = _make_chat_completion(provider_sequence, None)

    system_prompt = (
        "你是一名高级AI项目规划师，负责为执行智能体制定详尽计划。"
        "当前智能体可以使用的工具包括：filesystem（读写文件）、http_fetch（联网获取页面）、"
        "convert_to_text（将上传文档转为文本）、web_search（DuckDuckGo 搜索）、python_exec（在沙箱中执行 Python 代码）。"
        "你的计划需要覆盖关键步骤、每步期望产出以及建议使用的工具。"
        "请严格输出 JSON，格式如下：\n"
        "{\n"
        "  \"plan\": [\n"
        "    {\n"
        "      \"step\": 1,\n"
        "      \"title\": \"概述步骤\",\n"
        "      \"details\": \"具体行动说明\",\n"
        "      \"tool_hints\": [\"filesystem\"],\n"
        "      \"success_criteria\": \"完成判定标准\"\n"
        "    }\n"
        "  ],\n"
        "  \"risks\": [\"潜在风险与缓解措施\"],\n"
        "  \"final_answer_guidelines\": \"输出格式或注意事项\"\n"
        "}\n"
        "如无法提供某项内容，请使用空数组或空字符串，不要添加额外文本。"
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if conversation_history:
        for msg in conversation_history:
            role = msg.get("role")
            content = msg.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    files_hint = "\n".join(created_files or [os.path.basename(x) for x in existing_txts]) if (created_files or existing_txts) else "（无可用文本文件）"
    uploads_hint = "，".join(uploads_listing) if uploads_listing else "（无原始上传记录）"

    plan_request = (
        f"最新任务：{goal}\n"
        f"文本目录：{txt_dir_hint or '（未创建）'}\n"
        f"可用文本文件：{files_hint}\n"
        f"原始上传文件：{uploads_hint}\n"
        "请生成三至六步的行动计划，每一步说明要点、建议工具及成功标准。"
    )
    messages.append({"role": "user", "content": plan_request})

    reply = chat_completion(messages)
    data = _parse_json_block(reply or "")
    if not isinstance(data, dict):
        raise ValueError("规划模型返回了非JSON或空响应")

    raw_plan = data.get("plan") or []
    plan_steps: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_plan):
        if not isinstance(item, dict):
            continue
        step_number = item.get("step")
        try:
            step_number = int(step_number)
        except Exception:
            step_number = idx + 1
        tool_hints = item.get("tool_hints") or item.get("tools") or []
        if isinstance(tool_hints, str):
            tool_hints = [tool_hints]
        if not isinstance(tool_hints, list):
            tool_hints = []
        plan_steps.append({
            "step": step_number,
            "title": str(item.get("title") or item.get("name") or f"步骤 {idx + 1}"),
            "details": str(item.get("details") or item.get("description") or ""),
            "tool_hints": [str(t) for t in tool_hints if t],
            "success_criteria": str(item.get("success_criteria") or item.get("criteria") or ""),
        })

    risks = [str(r) for r in data.get("risks", []) if r]
    guidelines = str(data.get("final_answer_guidelines") or "")

    return {
        "plan": plan_steps,
        "risks": risks,
        "final_answer_guidelines": guidelines,
        "files": {
            "converted": created_files,
            "existing_txts": [os.path.basename(x) for x in existing_txts],
            "uploads": uploads_listing,
            "txt_dir": txt_dir_hint,
        },
        "provider_sequence": [label for label, _, _, _ in provider_sequence],
        "raw_response": reply,
    }


def run_ai_agent_job(
    *,
    session_id: str,
    goal: str,
    publish: Callable[[Dict[str, object]], None],
    check_control: Optional[Callable[[], Dict[str, bool]]] = None,
    primary: str = "local",  # "local" | "cloud"
    turbo_mode: bool = False,
    max_steps: int = 20,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run an agent session until completion or stop.

    publish: callable receiving structured progress updates for Streamlit.
    """

    # Initialize directories for this session
    base_dirs = {
        "uploads": str(CONFIG["directories"]["uploads"]),
        "generated_files": str(CONFIG["directories"]["generated_files"]),
    }
    session_dirs = ensure_session_dirs(base_dirs, session_id)
    session_dirs = prepare_conversation_dirs(session_dirs, conversation_id)
    paths = get_agent_paths(session_dirs)
    os.makedirs(paths.get("logs", "") or ".", exist_ok=True)
    actions_log = os.path.join(paths["logs"], "actions.jsonl")

    def _publish(stage: str, message: str, **extra: Any) -> None:
        payload = {"status": "running", "stage": stage, "message": message}
        payload.update(extra)
        publish(payload)

    def _ensure_running(stage: str, detail: str) -> bool:
        if not check_control:
            _publish(stage, detail)
            return True
        while True:
            state = check_control() or {"paused": False, "stopped": False}
            if state.get("stopped"):
                publish({"status": "failed", "stage": "stopped", "message": "任务已被用户停止"})
                return False
            if state.get("paused"):
                publish({"status": "paused", "stage": "paused", "message": f"暂停中：等待恢复（{detail}）"})
                time.sleep(1)
                continue
            _publish(stage, detail)
            return True

    # Convert any uploaded files to text upfront so the agent can read them
    try:
        conv = tool_convert_to_text(session_dirs)
        created_full = conv.get("files", []) or []
        created_files = [os.path.basename(p) for p in created_full]
        uploads_listing = conv.get("uploads_files", []) or []
        txt_dir_hint = conv.get("examined_txt", "")
        existing_txts = conv.get("existing_txts", []) or []
        publish({
            "stream": {
                "kind": "step",
                "step": 0,
                "last_action": {"tool": "convert_to_text", "examined_txt": txt_dir_hint},
                "artifacts": [{
                    "uploads_listing": uploads_listing,
                    "converted": created_files,
                    "existing_txts": [os.path.basename(x) for x in existing_txts],
                }],
            }
        })
    except Exception:
        created_full = []
        created_files = []
        uploads_listing = []
        txt_dir_hint = paths.get("examined_txt", "")
        existing_txts = []

    provider_sequence = _build_provider_sequence(primary, turbo_mode)
    if not provider_sequence:
        raise RuntimeError("没有可用的LLM提供商")

    chat_completion = _make_chat_completion(provider_sequence, publish)

    def tool_router(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if name == "filesystem":
            return tool_filesystem(payload.get("action"), payload.get("path"), session_dirs=session_dirs, content=payload.get("content"))
        if name == "http_fetch":
            return tool_http_fetch(payload.get("url"), timeout=30.0, max_bytes=2_000_000)
        if name == "convert_to_text":
            return tool_convert_to_text(session_dirs)
        if name == "web_search":
            max_results = payload.get("max_results", 5)
            try:
                max_results = int(max_results)
            except Exception:
                max_results = 5
            return tool_web_search(payload.get("query", ""), max_results=max_results)
        if name == "python_exec":
            timeout = payload.get("timeout", 8.0)
            try:
                timeout = float(timeout)
            except Exception:
                timeout = 8.0
            return tool_python_exec(payload.get("code", ""), inputs=payload.get("inputs"), timeout=timeout)
        raise ValueError(f"Unknown tool: {name}")

    graph = build_agent_graph(
        chat_completion=chat_completion,
        tool_router=tool_router,
    )
    app = graph.compile()

    # Initialize conversation
    files_hint = "\n".join(created_files or [os.path.basename(x) for x in existing_txts]) if (created_files or existing_txts) else "（无）"
    
    # Build initial messages with conversation context
    messages: List[Dict[str, str]] = []
    
    # Add conversation history if provided (for multi-turn conversations)
    if conversation_history:
        # Filter out metadata and keep only role/content
        for msg in conversation_history:
            if isinstance(msg, dict) and msg.get("role") and msg.get("content"):
                # Only include user and assistant messages, skip reasoning/tool messages
                metadata = msg.get("metadata", {})
                if metadata.get("type") not in ("reasoning", "tool_use"):
                    # Normalize the message format
                    content = msg["content"]
                    # Don't add the current goal if it's already the last user message
                    if not (msg["role"] == "user" and content == goal):
                        messages.append({
                            "role": msg["role"],
                            "content": content
                        })
    
    # Add current task with file context (only if not already added)
    if not messages or messages[-1].get("role") != "user" or not messages[-1].get("content", "").startswith("任务："):
        messages.append({"role": "user", "content": f"任务：{goal}"})
    
    messages.append({"role": "user", "content": (
        "已完成必要的文本转换。你可以使用 filesystem 工具读取这些文本并继续：\n"
        f"文本目录: {txt_dir_hint}\n"
        f"文件列表:\n{files_hint}"
    )})
    
    state = AgentState(goal=goal, messages=messages, step=0, max_steps=max_steps)

    _publish("initializing", f"启动智能体，任务: {goal}")
    _publish("initializing", f"初始消息数: {len(messages)}, 文件数: {len(created_files) + len(existing_txts)}")

    # Run the compiled graph and stream step updates from the 'act' node
    last_state = state
    try:
        for update in app.stream(state, {"recursion_limit": max_steps}):  # type: ignore[call-arg]
            if not _ensure_running("running", "执行中…"):
                return {"final_results": []}
            if not isinstance(update, dict):
                continue
            # Take any node update; prefer 'act' for step completion
            step_state = update.get("act") or next(iter(update.values()), None)
            if step_state is None:
                continue
            last_state = step_state
            # Support both dataclass and plain dict state
            if isinstance(step_state, dict):
                step_num = int(step_state.get("step", 0) or 0)
                last_action_val = step_state.get("last_action")
                artifacts_val = step_state.get("artifacts") or []
                error_val = step_state.get("error")
                done_val = step_state.get("done", False)
            else:
                step_num = int(getattr(step_state, "step", 0) or 0)
                last_action_val = getattr(step_state, "last_action", None)
                artifacts_val = getattr(step_state, "artifacts", [])
                error_val = getattr(step_state, "error", None)
                done_val = getattr(step_state, "done", False)
            
            # Debug: show error state if present
            if error_val:
                _publish("graph_execution", f"步骤 {step_num} 检测到错误: {error_val}")
            if done_val:
                _publish("graph_execution", f"步骤 {step_num} 标记为完成")
            
            record = {
                "stream": {
                    "kind": "step",
                    "step": step_num,
                    "last_action": last_action_val,
                    "artifacts": artifacts_val[-1:] if artifacts_val else [],
                }
            }
            publish(record)
            try:
                with open(actions_log, "a", encoding="utf-8") as logf:
                    logf.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                pass
    except Exception as error:
        error_msg = str(error)[:500]
        import traceback
        tb_str = traceback.format_exc()[:1000]
        _publish("graph_execution", f"图执行异常: {type(error).__name__}: {error_msg}")
        _publish("graph_execution", f"异常堆栈: {tb_str}")
        publish({"status": "failed", "stage": "error", "message": f"图执行失败: {error_msg}"})
        return {"final_results": []}

    status = "succeeded" if not getattr(last_state, "error", None) else "failed"
    final_message = next((m["content"] for m in reversed(getattr(last_state, "messages", [])) if m.get("role") == "assistant"), "")

    # Persist final output
    try:
        os.makedirs(paths["final_results"], exist_ok=True)
        output_name = time.strftime("agent_result_%Y%m%d_%H%M%S.txt")
        with open(os.path.join(paths["final_results"], output_name), "w", encoding="utf-8") as handle:
            handle.write(final_message or "")
    except Exception:
        pass

    publish({"status": status, "stage": "done", "message": "任务完成", "result": final_message})
    return {"final_results": [final_message]}


__all__ = ["run_ai_agent_job", "generate_agent_plan"]


