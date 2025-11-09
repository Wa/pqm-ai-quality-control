"""Background runner for AI Agent (LangGraph + MCP tools)."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ollama import Client as OllamaClient

from config import CONFIG
from tabs.shared.modelscope_client import ModelScopeClient
from util import ensure_session_dirs, resolve_ollama_host

from .graph import AgentState, build_agent_graph
from .mcp_tools import get_agent_paths, tool_convert_to_text, tool_filesystem, tool_http_fetch


def _extract_message_text(resp: Dict[str, Any]) -> str:
    return (
        (resp.get("message") or {}).get("content")
        or resp.get("response")
        or resp.get("text")
        or ""
    )


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

    # Prepare providers (ModelScope → Cloud → Local, depending on user choice)
    provider_sequence: List[Tuple[str, Callable[[], Any], str, str]] = []  # (label, factory, model, engine)

    modelscope_api_key = CONFIG["llm"].get("modelscope_api_key")
    modelscope_model = CONFIG["llm"].get("modelscope_model", "deepseek-ai/DeepSeek-V3.1")
    modelscope_base = CONFIG["llm"].get("modelscope_base_url", "https://api-inference.modelscope.cn/v1")
    if turbo_mode and modelscope_api_key:
        def _mk_modelscope() -> ModelScopeClient:
            return ModelScopeClient(api_key=modelscope_api_key, model=modelscope_model, base_url=modelscope_base)
        provider_sequence.append(("ModelScope", _mk_modelscope, modelscope_model, "modelscope"))

    # Local/Cloud model selection (use configured local model; cloud uses 20b)
    local_model = CONFIG["llm"].get("ollama_model") or "gpt-oss:latest"
    cloud_model = "gpt-oss:20b-cloud"

    cloud_host = CONFIG["llm"].get("ollama_cloud_host")
    cloud_key = CONFIG["llm"].get("ollama_cloud_api_key")

    def _mk_local() -> OllamaClient:
        # Use office network host (10.31.60.9) via resolver
        host = resolve_ollama_host("ollama_9")
        return OllamaClient(host=host)

    def _mk_cloud() -> OllamaClient:
        return OllamaClient(host=cloud_host, headers={"Authorization": f"Bearer {cloud_key}"})

    # Build provider order per user choice
    if primary == "local":
        if turbo_mode and modelscope_api_key:
            provider_sequence.append(("ModelScope", _mk_modelscope, modelscope_model, "modelscope"))
        provider_sequence.append(("本地 gpt-oss", _mk_local, local_model, "ollama"))
        if cloud_host and cloud_key:
            provider_sequence.append(("云端 gpt-oss", _mk_cloud, cloud_model, "ollama"))
    else:  # primary == "cloud"
        if turbo_mode and modelscope_api_key:
            provider_sequence.append(("ModelScope", _mk_modelscope, modelscope_model, "modelscope"))
        if cloud_host and cloud_key:
            provider_sequence.append(("云端 gpt-oss", _mk_cloud, cloud_model, "ollama"))
        provider_sequence.append(("本地 gpt-oss", _mk_local, local_model, "ollama"))

    def chat_completion(messages: List[Dict[str, str]]) -> str:
        last_err: Optional[Exception] = None
        last_empty_response_label: Optional[str] = None
        
        _publish("llm_call", f"开始尝试LLM调用，提供商序列: {[label for label, _, _, _ in provider_sequence]}")
        
        for label, factory, model, engine in provider_sequence:
            _publish("llm_call", f"尝试提供商: {label} (模型: {model}, 引擎: {engine})")
            try:
                _publish("llm_call", f"正在创建 {label} 客户端...")
                client = factory()
                _publish("llm_call", f"{label} 客户端创建成功")
                
                if engine == "modelscope":
                    # ModelScope: non-streaming
                    _publish("llm_call", f"调用 {label} (非流式)...")
                    resp = client.chat(model=model, messages=messages, stream=False, options={"num_ctx": 40001})
                    _publish("llm_call", f"{label} 响应接收完成，响应类型: {type(resp)}, 响应键: {list(resp.keys()) if isinstance(resp, dict) else 'N/A'}")
                    text = _extract_message_text(resp)
                    _publish("llm_call", f"{label} 提取的文本长度: {len(text)}, 前100字符: {text[:100] if text else '(空)'}")
                    if text and str(text).strip():
                        _publish("llm_call", f"✓ 成功调用 {label}，返回文本长度: {len(text)}")
                        return text
                    else:
                        _publish("llm_call", f"✗ {label} 返回空响应")
                        last_empty_response_label = label
                else:  # ollama
                    # Ollama: use streaming like special symbols tab does
                    _publish("llm_call", f"调用 {label} (流式)...")
                    response_text = ""
                    for chunk in client.chat(
                        model=model,
                        messages=messages,
                        stream=True,
                        options={"num_ctx": 40001},
                    ):
                        piece = (
                            chunk.get("message", {}).get("content")
                            or chunk.get("response")
                            or ""
                        )
                        if piece:
                            response_text += piece
                    if response_text and response_text.strip():
                        _publish("llm_call", f"✓ 成功调用 {label}，返回文本长度: {len(response_text)}")
                        return response_text
                    else:
                        _publish("llm_call", f"✗ {label} 返回空响应")
                        last_empty_response_label = label
            except Exception as error:
                last_err = error
                error_msg = str(error)[:500]
                error_type = type(error).__name__
                _publish("llm_call", f"✗ 调用 {label} 异常: {error_type}: {error_msg}")
                import traceback
                tb_str = traceback.format_exc()[:500]
                _publish("llm_call", f"{label} 异常堆栈: {tb_str}")
                continue
        
        # If all providers failed, raise an error instead of falling back
        error_summary = f"所有LLM提供商均失败"
        if last_empty_response_label:
            error_summary += f"。{last_empty_response_label} 返回了空响应"
        if last_err:
            error_summary += f"。最后异常: {type(last_err).__name__}: {str(last_err)[:300]}"
        _publish("llm_call", f"✗ {error_summary}")
        raise RuntimeError(
            f"无法完成LLM调用: {error_summary}。"
            f"已尝试的提供商: {[label for label, _, _, _ in provider_sequence]}"
        )

    def tool_router(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if name == "filesystem":
            return tool_filesystem(payload.get("action"), payload.get("path"), session_dirs=session_dirs, content=payload.get("content"))
        if name == "http_fetch":
            return tool_http_fetch(payload.get("url"), timeout=30.0, max_bytes=2_000_000)
        if name == "convert_to_text":
            return tool_convert_to_text(session_dirs)
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


__all__ = ["run_ai_agent_job"]


