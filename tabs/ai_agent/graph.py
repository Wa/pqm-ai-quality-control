"""LangGraph-based agent graph with simple ReAct-style JSON actions."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import END, StateGraph


Action = Dict[str, Any]


@dataclass
class AgentState:
    goal: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    step: int = 0
    max_steps: int = 20
    last_action: Optional[Action] = None
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    done: bool = False
    error: Optional[str] = None


def _build_system_prompt() -> str:
    return (
        "你是一名能够使用工具完成复杂文档处理任务的AI智能体。\n"
        "按照 ReAct 风格逐步思考，但只输出严格的 JSON：\n"
        "{\n"
        "  \"thought\": \"你在想什么，中文\",\n"
        "  \"tool\": \"filesystem|http_fetch|convert_to_text|web_search|python_exec|none\",\n"
        "  \"input\": { \"...\": \"...\" }\n"
        "}\n"
        "- 当需要读取/写入文件时，使用 filesystem 工具，input 包含 action、path、content（写入时）。\n"
        "- 当需要从网络获取页面时，使用 http_fetch 工具，input 包含 url。\n"
        "- 当需要将已上传文件转为文本时，使用 convert_to_text 工具，input 为空对象。\n"
        "- 当需要查找开放信息时，使用 web_search 工具，input 包含 query 与可选 max_results。\n"
        "- 当需要运行受限的Python代码时，使用 python_exec 工具，input 包含 code、可选 inputs 与 timeout。\n"
        "- 当你认为任务已完成或已给出最终答案时，tool=none，并在 thought 中给出结论。\n"
        "- 只输出 JSON，不要输出任何非JSON字符；如失败请重试直至给出合法JSON。\n"
    )


def _safe_json_loads(text: str) -> Optional[Action]:
    try:
        return json.loads(text)
    except Exception:
        # try code fence wrapped
        import re

        m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
    return None


def build_agent_graph(
    *,
    chat_completion: Callable[[List[Dict[str, str]]], str],
    tool_router: Callable[[str, Dict[str, Any]], Dict[str, Any]],
) -> StateGraph:
    """Construct a simple agent graph using LangGraph."""

    graph = StateGraph(AgentState)

    def node_plan(state: AgentState) -> AgentState:
        if state.step >= state.max_steps:
            state.done = True
            state.error = "max steps reached"
            return state
        system = {"role": "system", "content": _build_system_prompt()}
        messages = [system] + state.messages
        # Note: We can't use _publish here as it's not in scope, but errors will be caught and set in state.error
        try:
            reply = chat_completion(messages)
            # Log successful reply (truncated for debugging)
            print(f"[DEBUG graph] node_plan step={state.step}: 收到LLM回复，长度={len(reply) if reply else 0}")
            if reply:
                print(f"[DEBUG graph] node_plan step={state.step}: 回复前500字符: {reply[:500]}")
            if not reply or not reply.strip():
                print(f"[DEBUG graph] node_plan step={state.step}: 回复为空或仅空白")
                state.error = f"步骤 {state.step}: LLM返回了空响应"
                state.done = True
                return state
        except Exception as error:
            error_msg = str(error)[:500]
            print(f"[DEBUG graph] node_plan step={state.step}: LLM调用异常: {type(error).__name__}: {error_msg}")
            import traceback
            print(f"[DEBUG graph] node_plan step={state.step}: 异常堆栈: {traceback.format_exc()[:500]}")
            state.error = f"步骤 {state.step}: LLM调用失败: {error_msg}"
            state.done = True
            return state
        print(f"[DEBUG graph] node_plan step={state.step}: 开始解析JSON，回复长度={len(reply)}")
        action = _safe_json_loads(reply)
        print(f"[DEBUG graph] node_plan step={state.step}: JSON解析结果: {action}")
        # Retry once with stricter instruction if parsing failed or tool invalid
        allowed = {"filesystem", "http_fetch", "convert_to_text", "web_search", "python_exec", "none"}
        tool_name = str(action.get("tool", "")).lower() if action else None
        print(f"[DEBUG graph] node_plan step={state.step}: 工具名称={tool_name}, 是否允许={tool_name in allowed if tool_name else False}")
        if not action or tool_name not in allowed:
            print(f"[DEBUG graph] node_plan step={state.step}: JSON解析失败或工具无效，开始重试")
            try:
                messages = messages + [{"role": "user", "content": "只输出JSON，严格遵循字段与取值，不要其它字符。"}]
                reply = chat_completion(messages)
                print(f"[DEBUG graph] node_plan step={state.step}: 重试收到回复，长度={len(reply) if reply else 0}")
                if reply:
                    print(f"[DEBUG graph] node_plan step={state.step}: 重试回复前500字符: {reply[:500]}")
                if not reply or not reply.strip():
                    print(f"[DEBUG graph] node_plan step={state.step}: 重试回复为空")
                    state.error = f"步骤 {state.step}: LLM重试返回了空响应"
                    state.done = True
                    return state
            except Exception as error:
                print(f"[DEBUG graph] node_plan step={state.step}: 重试调用异常: {type(error).__name__}: {str(error)[:500]}")
                state.error = f"步骤 {state.step}: LLM重试调用失败: {str(error)[:500]}"
                state.done = True
                return state
            action = _safe_json_loads(reply)
            print(f"[DEBUG graph] node_plan step={state.step}: 重试后JSON解析结果: {action}")
        # If still invalid after retry, set error state instead of falling back
        final_tool = str(action.get("tool", "")).lower() if action else None
        if not action or final_tool not in allowed:
            error_detail = f"工具: {action.get('tool') if action else 'None'}, 原始回复前500字符: {reply[:500] if reply else '(无回复)'}"
            print(f"[DEBUG graph] node_plan step={state.step}: 最终验证失败: {error_detail}")
            state.error = f"步骤 {state.step}: LLM返回了无效的JSON或工具名称。{error_detail}"
            state.done = True
            return state
        # Use the parsed action (or fallback to none if somehow still invalid)
        action = action or {"tool": "none", "thought": reply[:500] if reply else "无响应", "input": {}}
        print(f"[DEBUG graph] node_plan step={state.step}: 最终action设置: tool={action.get('tool')}, thought长度={len(str(action.get('thought', '')))}")
        state.last_action = action
        return state

    def node_act(state: AgentState) -> AgentState:
        action = state.last_action or {"tool": "none", "input": {}}
        tool = str(action.get("tool") or "none").lower()
        if tool == "none":
            state.done = True
            # Final answer appended
            thought = str(action.get("thought") or "完成")
            state.messages.append({"role": "assistant", "content": thought})
            return state

        try:
            result = tool_router(tool, action.get("input") or {})
        except Exception as error:  # defensive
            result = {"error": str(error)}
        state.artifacts.append({"step": state.step, "tool": tool, "result": result})
        # Provide observation back to the model
        obs = json.dumps({"tool": tool, "result": result}, ensure_ascii=False)
        state.messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
        state.messages.append({"role": "user", "content": f"观察: {obs}. 继续，若完成请 tool=none 并给出结论。"})
        state.step += 1
        return state

    def should_end(state: AgentState) -> bool:
        return bool(state.done or state.error)

    graph.add_node("plan", node_plan)
    graph.add_node("act", node_act)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "act")
    graph.add_conditional_edges("act", lambda s: END if should_end(s) else "plan")

    return graph


__all__ = ["AgentState", "build_agent_graph"]


