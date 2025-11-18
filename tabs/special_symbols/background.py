"""Background worker for the special symbols workflow."""

from __future__ import annotations

import ast
import json
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import requests

from config import CONFIG

from ollama import Client as OllamaClient
from tabs.shared.modelscope_client import ModelScopeClient

from util import ensure_session_dirs, resolve_ollama_host

from .file_filter import run_filtering

from . import (
    SPECIAL_SYMBOLS_WORKFLOW_SURFACE,
    aggregate_outputs,
    cleanup_orphan_txts,
    estimate_tokens,
    log_llm_metrics,
    preprocess_txt_directories,
    process_archives,
    process_excel_folder,
    process_pdf_folder,
    process_textlike_folder,
    process_word_ppt_folder,
    report_exception,
    summarize_with_ollama,
)
from .summaries import persist_compare_outputs
from .workflow import SPECIAL_SYMBOLS_CHUNK_PROMPT_PREFIX

GPT_OSS_PROMPT_PREFIX = (
    "你是一名质量工程专家。任务：从文本中找出在“特殊特性”分类列中被标记的项目，并输出清单。\n\n"
    "严格规则：\n"
    "1) 识别表头中与“特殊特性分类”含义相同的列（可能写作：特殊特性/特殊特性分类/特性分类/Classification/Special characteristics 等）。\n"
    "2) 只选择该分类列的取值为以下符号之一的行：★、☆、/。\n"
    "   注意：其它列里出现的“/”（如频率“每班/每6h”、文本分隔符等）一律忽略，不算有效标记。\n"
    "3) 在输出中给出每条记录的核心信息：工序编号（如有，OP号）、产品/过程/工序/设备名称（如有）、项目/特性名称，以及特殊特性分类符号（★/☆/ /）。\n"
    "   如果缺少某些字段，则尽量从上下文补充；无法确定时仅给出项目/特性名称+标记符号。\n"
    "4) 仅输出项目清单，每行一个条目，不要解释、不要编号。\n"
    "5) 若整段文本中没有任何被上述分类列标记为★/☆/ /的行，则仅回复：无。\n\n"
    "以下提供表头参考与数据片段。请务必依据表头中的“特殊特性分类/Classification”列来判断：\n"
)

OLLAMA_CALL_TIMEOUT = 600  # seconds
MODELSCOPE_COMPARISON_MODELS: List[Tuple[str, str]] = [
    ("ModelScope DeepSeek-V3.2-Exp", "deepseek-ai/DeepSeek-V3.2-Exp"),
    ("ModelScope DeepSeek-V3.1", "deepseek-ai/DeepSeek-V3.1"),
    ("ModelScope Qwen3-235B", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
]

CHUNK_SUBMISSION_INTERVAL_SECONDS = 10.0

_MODELSCOPE_THROTTLE_LOCK = threading.Lock()
_LAST_MODELSCOPE_CALL_TS = 0.0


def _await_chunk_submission_window(last_submission_ts: float) -> float:
    """Wait until the 10s gap between chunk submissions is satisfied."""

    if last_submission_ts <= 0:
        return time.time()

    now = time.time()
    elapsed = now - last_submission_ts
    if elapsed < CHUNK_SUBMISSION_INTERVAL_SECONDS:
        time.sleep(CHUNK_SUBMISSION_INTERVAL_SECONDS - elapsed)
    return time.time()


def _execute_ollama_chat_with_timeout(
    client: OllamaClient,
    model_name: str,
    prompt_text: str,
    provider_label: str,
    timeout_s: int = OLLAMA_CALL_TIMEOUT,
) -> Tuple[str, Dict[str, object], str, int]:
    """Invoke an Ollama client with a hard timeout and return (text, stats, model, duration_ms)."""

    start_ts = time.time()

    def _invoke_call() -> Dict[str, object]:
        return client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt_text}],
            stream=False,
            options={"num_ctx": 40001},
        )

    with ThreadPoolExecutor(max_workers=1) as single_executor:
        future = single_executor.submit(_invoke_call)
        try:
            response = future.result(timeout=timeout_s)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"{provider_label} 调用超时（超过{timeout_s}秒未返回）") from exc

    duration_ms = int((time.time() - start_ts) * 1000)
    text = (
        response.get("message", {}).get("content")
        or response.get("response")
        or ""
    )
    stats = response.get("eval_info") or response.get("stats") or {}
    used_model = response.get("model") or model_name
    return text, stats, used_model, duration_ms


class _ModelScopeClientLegacy:
    """Minimal HTTP client for invoking ModelScope chat completions."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api-inference.modelscope.cn/v1",
        temperature: float = 0.3,
        timeout: float = 120.0,
    ) -> None:
        self._session = requests.Session()
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._timeout = timeout

    def chat(
        self,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, object]]] = None,
        stream: bool = False,
        options: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        if stream:
            raise NotImplementedError("ModelScopeClient does not support streaming responses")

        target_model = model or self._model
        payload: Dict[str, object] = {
            "model": target_model,
            "messages": messages or [],
            "temperature": self._temperature,
            "stream": False,
        }

        num_ctx: Optional[int] = None
        if options and isinstance(options, dict):
            num_ctx_value = options.get("num_ctx")
            if isinstance(num_ctx_value, int):
                num_ctx = num_ctx_value
        payload["max_context_length"] = num_ctx or 40001

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        response = self._session.post(url, json=payload, headers=headers, timeout=self._timeout)
        response.raise_for_status()
        data = response.json()

        message_content = ""
        if isinstance(data, dict):
            choices = data.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    message = choice.get("message")
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        message_content = message["content"]
                        break
                    if isinstance(choice.get("text"), str):
                        message_content = str(choice["text"])
                        break
            if not message_content and isinstance(data.get("text"), str):
                message_content = str(data["text"])
            if not message_content:
                output_text = data.get("output_text") or data.get("output")
                if isinstance(output_text, str):
                    message_content = output_text
            if not message_content:
                data_field = data.get("data")
                if isinstance(data_field, list) and data_field:
                    first_item = data_field[0]
                    if isinstance(first_item, dict):
                        for key in ("text", "output_text", "content"):
                            value = first_item.get(key)
                            if isinstance(value, str) and value:
                                message_content = value
                                break

        if not message_content:
            message_content = ""

        usage = data.get("usage") if isinstance(data, dict) else None
        stats: Dict[str, object] = {}
        if isinstance(usage, dict):
            stats = usage

        return {
            "message": {"content": message_content},
            "model": data.get("model") if isinstance(data, dict) else target_model,
            "stats": stats,
        }


class ProgressEmitter:
    """Adapter exposing Streamlit-like logging surface."""

    def __init__(self, publish: Callable[[Dict[str, object]], None], stage: str) -> None:
        self._publish = publish
        self._stage = stage

    def set_stage(self, stage: str) -> None:
        self._stage = stage

    def _emit(self, level: str, message: str) -> None:
        self._publish(
            {
                "stage": self._stage,
                "log": {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "level": level,
                    "message": message,
                },
            }
        )

    def info(self, message: str) -> None:
        self._emit("info", message)

    def warning(self, message: str) -> None:
        self._emit("warning", message)

    def error(self, message: str) -> None:
        self._emit("error", message)

    def write(self, message: str) -> None:
        self._emit("info", str(message))


def _list_txt_files(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return [
        name
        for name in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, name)) and name.lower().endswith(".txt")
    ]


def _clear_directory(directory: str) -> int:
    if not os.path.isdir(directory):
        return 0
    cleared = 0
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
                cleared += 1
            except Exception:
                continue
    return cleared


def _strip_code_fences(value: str) -> str:
    value = (value or "").strip()
    if value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
        if value.lower().startswith("json"):
            value = value[4:].strip()
    return value


def _extract_json_text(text: str) -> str:
    s = (text or "").strip().lstrip("\ufeff")
    s = _strip_code_fences(s)
    try:
        json.loads(s)
        return s
    except json.JSONDecodeError:
        pass
    start_arr = s.find("[")
    end_arr = s.rfind("]")
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        candidate = s[start_arr : end_arr + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            merged = re.sub(r"\]\s*,?\s*\[", ",", candidate)
            try:
                json.loads(merged)
                return merged
            except json.JSONDecodeError:
                pass
    start_obj = s.find("{")
    end_obj = s.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        candidate = s[start_obj : end_obj + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    return s


def _is_rate_limit_error(error: Exception) -> bool:
    status_code = getattr(getattr(error, "response", None), "status_code", None)
    if status_code == 429:
        return True
    text = str(error)
    return "429" in text and "Too Many Requests" in text


def _should_retry_response(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    if lowered.startswith("[error") or lowered.startswith("error"):
        return True
    extracted = _extract_json_text(stripped)
    return extracted in ("[]", "")


def _is_plain_symbol(value: str) -> bool:
    return value in {"★", "☆", "/"}


def _filter_identical_matches(rows: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], int]:
    filtered: List[Dict[str, object]] = []
    removed = 0

    for row in rows:
        if not isinstance(row, dict):
            filtered.append(row)
            continue
        base_symbol = str(row.get("基准特殊特性分类") or "").strip()
        exam_symbol = str(row.get("待检查特殊特性分类") or "").strip()
        if _is_plain_symbol(base_symbol) and base_symbol == exam_symbol:
            removed += 1
            continue
        filtered.append(row)

    return filtered, removed


def _filter_missing_symbol_rows(rows: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], int]:
    filtered: List[Dict[str, object]] = []
    removed = 0

    for row in rows:
        if not isinstance(row, dict):
            filtered.append(row)
            continue
        base_symbol = str(row.get("基准特殊特性分类") or "").strip()
        exam_symbol = str(row.get("待检查特殊特性分类") or "").strip()
        if not base_symbol or not exam_symbol:
            removed += 1
            continue
        filtered.append(row)

    return filtered, removed


def _await_modelscope_window() -> None:
    """Ensure there is a 10s gap between ModelScope invocations."""

    global _LAST_MODELSCOPE_CALL_TS
    with _MODELSCOPE_THROTTLE_LOCK:
        now = time.time()
        elapsed = now - _LAST_MODELSCOPE_CALL_TS
        if elapsed < CHUNK_SUBMISSION_INTERVAL_SECONDS:
            time.sleep(CHUNK_SUBMISSION_INTERVAL_SECONDS - elapsed)
        _LAST_MODELSCOPE_CALL_TS = time.time()


def _strip_trailing_commas(value: str) -> str:
    pattern = re.compile(r",(\s*[}\]])")
    previous = None
    current = value
    while previous != current:
        previous = current
        current = pattern.sub(r"\1", current)
    return current


def _loose_json_loads(text: str) -> Optional[object]:
    candidate = _extract_json_text(text)
    if not candidate:
        return None
    attempts = [candidate]
    stripped_commas = _strip_trailing_commas(candidate)
    if stripped_commas not in attempts:
        attempts.append(stripped_commas)
    for attempt in attempts:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    for attempt in attempts:
        try:
            return ast.literal_eval(attempt)
        except (ValueError, SyntaxError):
            continue
    return None


def _normalize_json_rows(obj: object) -> Optional[List[Dict[str, object]]]:
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        normalized: List[Dict[str, object]] = []
        for item in obj:
            if isinstance(item, dict):
                normalized.append(item)
        if normalized or obj == []:
            return normalized
    return None


def _parse_json_rows_loose(text: str) -> Optional[List[Dict[str, object]]]:
    data = _loose_json_loads(text)
    if data is None:
        return None
    return _normalize_json_rows(data)


def _build_exam_chunks(exam_text: str) -> List[Dict[str, object]]:
    """Split exam content by source file for chunked comparison."""

    parsed_rows = _parse_json_rows_loose(exam_text)
    if not parsed_rows:
        fallback = exam_text.strip() or "无"
        return [
            {
                "file_name": "全部待检内容",
                "sheet_count": 0,
                "payload_text": fallback,
            }
        ]

    grouped: OrderedDict[str, List[Dict[str, object]]] = OrderedDict()
    for record in parsed_rows:
        file_key = str(record.get("文件名") or record.get("待检查文件名") or "未命名文件")
        grouped.setdefault(file_key, []).append(record)

    chunks: List[Dict[str, object]] = []
    for file_name, records in grouped.items():
        payload_text = json.dumps(records, ensure_ascii=False, indent=2)
        chunks.append(
            {
                "file_name": file_name or "未命名文件",
                "sheet_count": len(records),
                "payload_text": payload_text,
            }
        )

    if not chunks:
        fallback = exam_text.strip() or "无"
        return [
            {
                "file_name": "全部待检内容",
                "sheet_count": 0,
                "payload_text": fallback,
            }
        ]
    return chunks


def _make_chunk_stream_name(base: str, chunk_index: int, file_name: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z]+", "_", file_name or "chunk").strip("_")
    if not safe:
        safe = "chunk"
    if len(safe) > 32:
        safe = safe[:32]
    return f"{base}_chunk_{chunk_index:02d}_{safe}"


def _run_gpt_extraction(
    emitter: ProgressEmitter,
    publish: Callable[[Dict[str, object]], None],
    primary_client: Optional[OllamaClient],
    primary_model_name: str,
    session_id: str,
    output_root: str,
    src_dir: str,
    file_names: Iterable[str],
    dest_dir: str,
    stage_label: str,
    stage_message: str,
    clear_message_template: Optional[str],
    client_unavailable_message: str,
    combined_prefix: str,
    combined_log_template: str = "已生成汇总结果 {name}",
    progress_callback: Optional[Callable[[], None]] = None,
    control_handler: Optional[Callable[[str, str], bool]] = None,
    fallback_client: Optional[OllamaClient] = None,
    fallback_model_name: Optional[str] = None,
    primary_label: str = "本地 gpt-oss",
    fallback_label: str = "云端 gpt-oss",
) -> List[str]:
    names = sorted(file_names, key=lambda value: value.lower())
    if not names:
        return []

    emitter.set_stage(stage_label)
    if primary_client is None and fallback_client is None:
        emitter.warning(client_unavailable_message)
        return []

    os.makedirs(dest_dir, exist_ok=True)
    cleared = _clear_directory(dest_dir)
    if cleared and clear_message_template:
        try:
            emitter.info(clear_message_template.format(cleared=cleared))
        except Exception:
            emitter.info(clear_message_template)  # type: ignore[arg-type]

    outputs: List[str] = []
    emitter.info(stage_message)

    aggregate_data: List[Dict[str, object]] = []

    for name in names:
        if control_handler and not control_handler(stage_label, f"{stage_message}（{name}）"):
            return outputs
        src_path = os.path.join(src_dir, name)
        try:
            with open(src_path, "r", encoding="utf-8") as handle:
                doc_text = handle.read()
        except Exception as error:
            report_exception(f"读取文本失败({stage_label}:{name})", error, level="warning")
            if progress_callback:
                progress_callback()
            continue

        prompt_text = f"{GPT_OSS_PROMPT_PREFIX}{doc_text}"
        publish(
            {
                "stream": {
                    "kind": "prompt",
                    "file": name,
                    "part": 1,
                    "total_parts": 1,
                    "engine": "gpt-oss",
                    "text": prompt_text,
                }
            }
        )
        attempts: List[Tuple[str, Optional[OllamaClient], Optional[str]]] = [
            (primary_label, primary_client, primary_model_name),
            (fallback_label, fallback_client, fallback_model_name or primary_model_name),
        ]
        response_text = ""
        last_stats = None
        start_ts = 0.0
        used_model_name = primary_model_name
        call_success = False

        for attempt_label, attempt_client, attempt_model in attempts:
            if attempt_client is None or not attempt_model:
                continue

            publish({"stage": stage_label, "message": f"调用 {attempt_label} ({name})"})
            start_ts = time.time()
            response_text = ""
            last_stats = None
            duration_ms = 0
            is_cloud_fallback = (
                fallback_client is not None and attempt_client is fallback_client
            )

            try:
                if is_cloud_fallback:
                    response_text_raw, last_stats, used_model_name, duration_ms = _execute_ollama_chat_with_timeout(
                        attempt_client,
                        attempt_model,
                        prompt_text,
                        attempt_label,
                    )
                    response_text = response_text_raw
                else:
                    for chunk in attempt_client.chat(
                        model=attempt_model,
                        messages=[{"role": "user", "content": prompt_text}],
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
                        last_stats = chunk.get("eval_info") or chunk.get("stats") or last_stats
                        if control_handler and not control_handler(stage_label, f"{attempt_label} 流式响应（{name}）"):
                            return outputs
                    used_model_name = attempt_model
                    duration_ms = int((time.time() - start_ts) * 1000)
                call_success = True
                break
            except TimeoutError as timeout_error:
                report_exception(
                    f"调用 {attempt_label} 失败({stage_label}:{name})",
                    timeout_error,
                    level="warning",
                )
                publish(
                    {
                        "stream": {
                            "kind": "response",
                            "file": name,
                            "part": 1,
                            "total_parts": 1,
                            "engine": attempt_model,
                            "text": f"调用 {attempt_label} 超时：{timeout_error}",
                        }
                    }
                )
                if fallback_client and attempt_client is primary_client:
                    emitter.warning(f"{attempt_label} 调用失败，尝试 {fallback_label}")
                continue
            except Exception as error:
                report_exception(
                    f"调用 {attempt_label} 失败({stage_label}:{name})",
                    error,
                    level="warning",
                )
                publish(
                    {
                        "stream": {
                            "kind": "response",
                            "file": name,
                            "part": 1,
                            "total_parts": 1,
                            "engine": attempt_model,
                            "text": f"调用 {attempt_label} 失败：{error}",
                        }
                    }
                )
                if fallback_client and attempt_client is primary_client:
                    emitter.warning(f"{attempt_label} 调用失败，尝试 {fallback_label}")
                continue

        if not call_success:
            if progress_callback:
                progress_callback()
            if control_handler and not control_handler(stage_label, f"已跳过（{name}）"):
                return outputs
            continue

        if not duration_ms:
            duration_ms = int((time.time() - start_ts) * 1000)
        response_clean = response_text.strip() or "无"
        publish(
            {
                "stream": {
                    "kind": "response",
                    "file": name,
                    "part": 1,
                    "total_parts": 1,
                    "engine": used_model_name,
                    "text": response_clean,
                }
            }
        )

        log_llm_metrics(
            output_root,
            session_id,
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "engine": "ollama",
                "model": used_model_name,
                "session_id": session_id,
                "file": name,
                "part": 1,
                "phase": stage_label,
                "prompt_chars": len(prompt_text),
                "prompt_tokens": estimate_tokens(prompt_text, used_model_name),
                "output_chars": len(response_clean),
                "output_tokens": estimate_tokens(response_clean, used_model_name),
                "duration_ms": duration_ms,
                "success": 1 if response_clean else 0,
                "stats": last_stats or {},
                "error": "",
            },
        )

        dst_path = os.path.join(dest_dir, name)
        entries = [line.strip() for line in response_clean.splitlines() if line.strip()]
        if not entries and response_clean.strip():
            entries = [response_clean.strip()]
        stem = os.path.splitext(name)[0]
        file_name_value = stem
        sheet_name_value: Optional[str] = None
        if "_SHEET_" in stem:
            file_name_value, sheet_name_value = stem.split("_SHEET_", 1)
        record: Dict[str, object] = {
            "文件名": file_name_value,
            "特殊特性符号": entries,
        }
        if sheet_name_value:
            record["工作表名"] = sheet_name_value
        try:
            with open(dst_path, "w", encoding="utf-8") as writer:
                json.dump(record, writer, ensure_ascii=False, indent=2)
                writer.write("\n")
            outputs.append(dst_path)
            if not (len(entries) == 1 and entries[0] == "无"):
                aggregate_data.append(record)
        except Exception as error:
            report_exception(f"写入 gpt-oss 结果失败({stage_label}:{name})", error, level="warning")

        if progress_callback:
            progress_callback()
        if control_handler and not control_handler(stage_label, f"完成写入（{name}）"):
            return outputs

    if aggregate_data or outputs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_name = f"{combined_prefix}_{timestamp}.txt"
        combined_path = os.path.join(dest_dir, combined_name)
        try:
            with open(combined_path, "w", encoding="utf-8") as handle:
                json.dump(aggregate_data, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            outputs.append(combined_path)
            emitter.info(combined_log_template.format(name=combined_name))
            if control_handler and not control_handler(stage_label, f"生成汇总（{combined_name}）"):
                return outputs
        except Exception as error:
            report_exception(f"汇总 gpt-oss 结果失败({stage_label})", error, level="warning")

    return outputs


def _run_gpt_extraction_parallel(
    emitter: ProgressEmitter,
    publish: Callable[[Dict[str, object]], None],
    client_sequence: Optional[List[Tuple[str, Callable[[], Any], str, str]]],
    session_id: str,
    output_root: str,
    src_dir: str,
    file_names: Iterable[str],
    dest_dir: str,
    stage_label: str,
    stage_message: str,
    clear_message_template: Optional[str],
    client_unavailable_message: str,
    combined_prefix: str,
    combined_log_template: str = "已生成汇总结果 {name}",
    progress_callback: Optional[Callable[[], None]] = None,
    max_workers: int = 6,
) -> List[str]:
    names = sorted(file_names, key=lambda value: value.lower())
    if not names:
        return []

    emitter.set_stage(stage_label)
    if not client_sequence:
        emitter.warning(client_unavailable_message)
        return []

    os.makedirs(dest_dir, exist_ok=True)
    cleared = _clear_directory(dest_dir)
    if cleared and clear_message_template:
        try:
            emitter.info(clear_message_template.format(cleared=cleared))
        except Exception:
            emitter.info(clear_message_template)

    emitter.info(f"{stage_message}（高性能模式）")

    outputs: List[str] = []
    aggregate_data: List[Dict[str, object]] = []
    prompts: Dict[str, str] = {}
    tasks: List[str] = []

    # Use the first provider info for prompt engine labeling
    first_label, _, first_model_name, _ = client_sequence[0]

    for name in names:
        src_path = os.path.join(src_dir, name)
        try:
            with open(src_path, "r", encoding="utf-8") as handle:
                doc_text = handle.read()
        except Exception as error:
            report_exception(f"读取文本失败({stage_label}:{name})", error, level="warning")
            if progress_callback:
                progress_callback()
            continue

        prompt_text = f"{GPT_OSS_PROMPT_PREFIX}{doc_text}"
        prompts[name] = prompt_text
        publish(
            {
                "stream": {
                    "kind": "prompt",
                    "file": name,
                    "part": 1,
                    "total_parts": 1,
                    "engine": first_model_name,
                    "text": prompt_text,
                }
            }
        )
        publish({"stage": stage_label, "message": f"调用 {first_label} ({name})"})
        tasks.append(name)

    if not tasks:
        return outputs

    worker_limit = max(1, min(max_workers, len(tasks)))

    def _invoke(name: str, prompt_text: str) -> Dict[str, object]:
        last_error: Optional[Exception] = None
        for provider_label, provider_factory, provider_model, provider_engine in client_sequence or []:
            attempts = 3 if provider_engine == "modelscope" else 1
            for _ in range(attempts):
                try:
                    client = provider_factory()
                except Exception as client_error:
                    last_error = client_error
                    continue

                if provider_engine == "modelscope":
                    start_ts = time.time()
                    _await_modelscope_window()
                    try:
                        response = client.chat(
                            model=provider_model,
                            messages=[{"role": "user", "content": prompt_text}],
                            stream=False,
                            options={"num_ctx": 40001},
                        )
                    except Exception as call_error:
                        last_error = call_error
                        continue

                    message = (
                        response.get("message", {}).get("content")
                        or response.get("response")
                        or ""
                    )
                    stats = response.get("eval_info") or response.get("stats") or {}
                    used_model = response.get("model") or provider_model
                    duration_ms = int((time.time() - start_ts) * 1000)
                    if not message.strip():
                        last_error = RuntimeError(f"{provider_label} 未返回有效内容")
                        continue
                    return {
                        "name": name,
                        "text": message,
                        "stats": stats,
                        "model": used_model,
                        "engine_tag": provider_engine,
                        "duration_ms": duration_ms,
                    }
                try:
                    message, stats, used_model, duration_ms = _execute_ollama_chat_with_timeout(
                        client,
                        provider_model,
                        prompt_text,
                        provider_label,
                    )
                except TimeoutError as timeout_error:
                    last_error = timeout_error
                    continue
                except Exception as call_error:
                    last_error = call_error
                    continue
                if not message.strip():
                    last_error = RuntimeError(f"{provider_label} 未返回有效内容")
                    continue
                return {
                    "name": name,
                    "text": message,
                    "stats": stats,
                    "model": used_model,
                    "engine_tag": provider_engine,
                    "duration_ms": duration_ms,
                }

        return {"name": name, "error": last_error or RuntimeError("no available provider")}

    futures: Dict[object, str] = {}
    with ThreadPoolExecutor(max_workers=worker_limit) as executor:
        for name in tasks:
            prompt_text = prompts[name]
            future = executor.submit(_invoke, name, prompt_text)
            futures[future] = name

        for future in as_completed(futures):
            name = futures[future]
            prompt_text = prompts.get(name, "")
            try:
                result = future.result()
            except Exception as error:
                report_exception(f"调用高性能通道失败({stage_label}:{name})", error, level="warning")
                if progress_callback:
                    progress_callback()
                continue

            error_obj = result.get("error") if isinstance(result, dict) else None
            if error_obj:
                report_exception(f"调用高性能通道失败({stage_label}:{name})", error_obj, level="warning")
                publish(
                    {
                        "stream": {
                            "kind": "response",
                            "file": name,
                            "part": 1,
                            "total_parts": 1,
                            "engine": first_model_name,
                            "text": f"调用高性能通道失败：{error_obj}",
                        }
                    }
                )
                if progress_callback:
                    progress_callback()
                continue

            response_clean = (result.get("text") or "").strip() or "无"
            used_model_name = result.get("model") or first_model_name
            duration_ms = int(result.get("duration_ms") or 0)
            stats = result.get("stats") or {}
            used_engine_tag = result.get("engine_tag") or "ollama"

            publish(
                {
                    "stream": {
                        "kind": "response",
                        "file": name,
                        "part": 1,
                        "total_parts": 1,
                        "engine": used_model_name,
                        "text": response_clean,
                    }
                }
            )

            log_llm_metrics(
                output_root,
                session_id,
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "engine": used_engine_tag,
                    "model": used_model_name,
                    "session_id": session_id,
                    "file": name,
                    "part": 1,
                    "phase": stage_label,
                    "prompt_chars": len(prompt_text),
                    "prompt_tokens": estimate_tokens(prompt_text, used_model_name),
                    "output_chars": len(response_clean),
                    "output_tokens": estimate_tokens(response_clean, used_model_name),
                    "duration_ms": duration_ms,
                    "success": 1 if response_clean else 0,
                    "stats": stats,
                    "error": "",
                },
            )

            dst_path = os.path.join(dest_dir, name)
            entries = [line.strip() for line in response_clean.splitlines() if line.strip()]
            if not entries and response_clean.strip():
                entries = [response_clean.strip()]
            stem = os.path.splitext(name)[0]
            file_name_value = stem
            sheet_name_value: Optional[str] = None
            if "_SHEET_" in stem:
                file_name_value, sheet_name_value = stem.split("_SHEET_", 1)
            record: Dict[str, object] = {
                "文件名": file_name_value,
                "特殊特性符号": entries,
            }
            if sheet_name_value:
                record["工作表名"] = sheet_name_value
            try:
                with open(dst_path, "w", encoding="utf-8") as writer:
                    json.dump(record, writer, ensure_ascii=False, indent=2)
                    writer.write("\n")
                outputs.append(dst_path)
                if not (len(entries) == 1 and entries[0] == "无"):
                    aggregate_data.append(record)
            except Exception as error:
                report_exception(f"写入 gpt-oss 结果失败({stage_label}:{name})", error, level="warning")

            if progress_callback:
                progress_callback()

    if aggregate_data or outputs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_name = f"{combined_prefix}_{timestamp}.txt"
        combined_path = os.path.join(dest_dir, combined_name)
        try:
            with open(combined_path, "w", encoding="utf-8") as handle:
                json.dump(aggregate_data, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            outputs.append(combined_path)
            emitter.info(combined_log_template.format(name=combined_name))
        except Exception as error:
            report_exception(f"汇总 gpt-oss 结果失败({stage_label})", error, level="warning")

    return outputs


def run_special_symbols_job(
    session_id: str,
    publish: Callable[[Dict[str, object]], None],
    check_control: Optional[Callable[[], Dict[str, bool]]] = None,
    turbo_mode: bool = False,
) -> Dict[str, List[str]]:
    """Run the special symbols workflow headlessly and report progress via ``publish``."""

    publish({"status": "running", "stage": "initializing", "message": "准备特殊特性符号会话目录"})

    progress_value = 0.0
    job_start_time = time.time()
    turbo_mode_enabled = bool(turbo_mode)
    if turbo_mode_enabled:
        check_control = None

    def publish_progress(value: float) -> None:
        nonlocal progress_value
        progress_value = max(0.0, min(value, 100.0))
        publish({"progress": progress_value})

    processed_chunks = 0
    total_chunks = 0
    stop_announced = False
    control_state = "running"

    def announce_stop(message: str) -> None:
        nonlocal stop_announced
        if stop_announced:
            return
        stop_announced = True
        publish(
            {
                "status": "failed",
                "stage": "stopped",
                "message": message,
                "processed_chunks": processed_chunks,
                "total_chunks": total_chunks,
                "progress": progress_value,
            }
        )

    def ensure_running(stage: str, detail: str) -> bool:
        nonlocal control_state
        if not check_control:
            return True
        stage_value = stage or "running"
        while True:
            try:
                status = check_control()
            except Exception:
                status = None
            if not status:
                if control_state != "running":
                    control_state = "running"
                    publish({"status": "running", "stage": stage_value, "message": detail})
                return True
            if status.get("stopped"):
                announce_stop("任务已被用户停止")
                return False
            if status.get("paused"):
                if control_state != "paused":
                    control_state = "paused"
                publish(
                    {
                        "status": "paused",
                        "stage": "paused",
                        "message": f"暂停中：等待恢复（{detail}）",
                        "processed_chunks": processed_chunks,
                        "total_chunks": total_chunks,
                        "progress": progress_value,
                    }
                )
                time.sleep(1)
                continue
            if control_state != "running":
                control_state = "running"
                publish({"status": "running", "stage": stage_value, "message": detail})
            return True

    publish_progress(1.0)
    if not ensure_running("initializing", "初始化特殊特性符号检查"):
        return {"final_results": []}

    base_dirs = {"generated": str(CONFIG["directories"]["generated_files"])}
    session_dirs = ensure_session_dirs(base_dirs, session_id)
    paths = SPECIAL_SYMBOLS_WORKFLOW_SURFACE.prepare_paths(session_dirs)

    emitter = ProgressEmitter(publish, stage="preparing")

    reference_dir = paths.standards_dir
    examined_dir = paths.examined_dir
    output_root = paths.output_root
    reference_txt_dir = paths.standards_txt_dir
    examined_txt_dir = paths.examined_txt_dir
    initial_results_dir = paths.initial_results_dir
    final_results_dir = paths.final_results_dir

    os.makedirs(initial_results_dir, exist_ok=True)
    os.makedirs(final_results_dir, exist_ok=True)

    preexisting_final_files: set[str] = set()
    try:
        if os.path.isdir(final_results_dir):
            for existing_name in os.listdir(final_results_dir):
                existing_path = os.path.join(final_results_dir, existing_name)
                if os.path.isfile(existing_path):
                    preexisting_final_files.add(existing_path)
    except Exception:
        preexisting_final_files = set()

    try:
        removed_ref = cleanup_orphan_txts(reference_dir, reference_txt_dir, emitter)
        removed_exam = cleanup_orphan_txts(examined_dir, examined_txt_dir, emitter)
        if removed_ref or removed_exam:
            emitter.info(f"已清理无关文本 {removed_ref + removed_exam} 个")
    except Exception as error:
        report_exception("清理无关文本失败", error, level="warning")

    try:
        cleared = 0
        for name in os.listdir(initial_results_dir):
            path = os.path.join(initial_results_dir, name)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                    cleared += 1
                except Exception:
                    continue
        if cleared:
            emitter.info(f"已清空上次运行结果 {cleared} 个文件")
    except Exception:
        pass

    emitter.set_stage("conversion")
    if not ensure_running("conversion", "准备解析基准文件"):
        return {"final_results": []}
    emitter.info("正在解析基准文件")
    process_pdf_folder(reference_dir, reference_txt_dir, emitter, annotate_sources=True)
    if not ensure_running("conversion", "解析基准文件（PDF）"):
        return {"final_results": []}
    process_word_ppt_folder(reference_dir, reference_txt_dir, emitter, annotate_sources=True)
    if not ensure_running("conversion", "解析基准文件（Word/PPT）"):
        return {"final_results": []}
    process_excel_folder(reference_dir, reference_txt_dir, emitter, annotate_sources=True)
    if not ensure_running("conversion", "解析基准文件（Excel）"):
        return {"final_results": []}
    process_textlike_folder(reference_dir, reference_txt_dir, emitter)
    if not ensure_running("conversion", "解析基准文件（文本类）"):
        return {"final_results": []}
    process_archives(reference_dir, reference_txt_dir, emitter)
    if not ensure_running("conversion", "解析基准文件（压缩包）"):
        return {"final_results": []}

    if not ensure_running("conversion", "准备解析待检查文件"):
        return {"final_results": []}
    emitter.info("正在解析待检查文件")
    process_pdf_folder(examined_dir, examined_txt_dir, emitter, annotate_sources=False)
    if not ensure_running("conversion", "解析待检查文件（PDF）"):
        return {"final_results": []}
    process_word_ppt_folder(examined_dir, examined_txt_dir, emitter, annotate_sources=False)
    if not ensure_running("conversion", "解析待检查文件（Word/PPT）"):
        return {"final_results": []}
    process_excel_folder(examined_dir, examined_txt_dir, emitter, annotate_sources=False)
    if not ensure_running("conversion", "解析待检查文件（Excel）"):
        return {"final_results": []}
    process_textlike_folder(examined_dir, examined_txt_dir, emitter)
    if not ensure_running("conversion", "解析待检查文件（文本类）"):
        return {"final_results": []}
    process_archives(examined_dir, examined_txt_dir, emitter)
    if not ensure_running("conversion", "解析待检查文件（压缩包）"):
        return {"final_results": []}

    try:
        updated_txts = preprocess_txt_directories(reference_txt_dir, examined_txt_dir)
        if updated_txts:
            emitter.info(f"文本预处理完成 {len(updated_txts)} 个")
    except Exception as error:
        report_exception("文本预处理失败", error, level="warning")

    publish_progress(10.0)

    standards_txt_filtered_dir = os.path.join(output_root, "standards_txt_filtered")
    if not ensure_running("filter", "准备过滤基准文本"):
        return {"final_results": []}
    try:
        standards_summary = run_filtering(
            reference_txt_dir,
            standards_txt_filtered_dir,
            config_path=os.path.join(os.path.dirname(__file__), "filter_config.yml"),
            name_exclude_substrings=["标准选项", "特殊特性符号对照表", "变更履历"],
        )
        emitter.info(
            f"基准过滤完成 保留{standards_summary.get('kept', 0)} 排除{standards_summary.get('dropped', 0)} 清空{standards_summary.get('empty_after_filter', 0)}"
        )
    except Exception as error:
        report_exception("过滤基准文本失败", error, level="warning")

    standards_txt_filtered_files = _list_txt_files(standards_txt_filtered_dir)

    # Filter examined .txt files into a separate folder to reduce irrelevant content
    examined_txt_filtered_dir = os.path.join(output_root, "examined_txt_filtered")
    if not ensure_running("filter", "准备过滤待检文本"):
        return {"final_results": []}
    try:
        summary = run_filtering(
            examined_txt_dir,
            examined_txt_filtered_dir,
            config_path=os.path.join(os.path.dirname(__file__), "filter_config.yml"),
            name_exclude_substrings=["变更履历", "附图、附表", "封面"],
        )
        emitter.info(
            f"待检过滤完成 保留{summary.get('kept', 0)} 排除{summary.get('dropped', 0)} 清空{summary.get('empty_after_filter', 0)}"
        )
    except Exception as error:
        report_exception("过滤待检文本失败", error, level="warning")

    exam_src_dir = examined_txt_filtered_dir if _list_txt_files(examined_txt_filtered_dir) else examined_txt_dir
    exam_txt_files = _list_txt_files(exam_src_dir)

    local_model_name = CONFIG["llm"].get("ollama_model") or "gpt-oss:latest"
    cloud_extraction_model = "gpt-oss:20b-cloud"
    cloud_comparison_model = "deepseek-v3.1:671b-cloud"
    local_ollama_client: Optional[OllamaClient] = None
    cloud_ollama_client: Optional[OllamaClient] = None
    cloud_client_factory: Optional[Callable[[], OllamaClient]] = None
    local_client_factory: Optional[Callable[[], OllamaClient]] = None
    cloud_host = CONFIG["llm"].get("ollama_cloud_host")
    cloud_api_key = CONFIG["llm"].get("ollama_cloud_api_key")
    modelscope_api_key = os.getenv("MODELSCOPE_API_KEY") or CONFIG["llm"].get("modelscope_api_key")
    modelscope_model_name = CONFIG["llm"].get("modelscope_model") or "deepseek-ai/DeepSeek-V3.1"
    modelscope_base_url = CONFIG["llm"].get("modelscope_base_url") or "https://api-inference.modelscope.cn/v1"
    modelscope_client_factory: Optional[Callable[[], ModelScopeClient]] = None

    if standards_txt_filtered_files or exam_txt_files:
        try:
            host = resolve_ollama_host("ollama_9")
            local_ollama_client = OllamaClient(host=host)
        except Exception as error:
            report_exception("初始化本地 gpt-oss 客户端失败", error, level="warning")
        else:
            def _make_local_client() -> OllamaClient:
                return OllamaClient(host=resolve_ollama_host("ollama_9"))
            local_client_factory = _make_local_client
        if cloud_host and cloud_api_key:
            try:
                cloud_ollama_client = OllamaClient(
                    host=cloud_host,
                    headers={"Authorization": f"Bearer {cloud_api_key}"},
                )
            except Exception as error:
                report_exception("初始化云端 gpt-oss 客户端失败", error, level="warning")
            else:
                def _make_cloud_client() -> OllamaClient:
                    return OllamaClient(
                        host=cloud_host,
                        headers={"Authorization": f"Bearer {cloud_api_key}"},
                    )

                cloud_client_factory = _make_cloud_client
        else:
            emitter.warning("未配置云端 gpt-oss，无法启用云端备份")

        if modelscope_api_key:
            try:
                def _make_modelscope_client() -> ModelScopeClient:
                    return ModelScopeClient(
                        api_key=modelscope_api_key,
                        model=modelscope_model_name,
                        base_url=modelscope_base_url,
                        timeout=900.0,
                    )

                _make_modelscope_client()
            except Exception as error:
                report_exception("初始化 ModelScope DeepSeek 客户端失败", error, level="warning")
                if turbo_mode_enabled:
                    emitter.warning("ModelScope DeepSeek 初始化失败，将尝试使用其它高性能通道。")
            else:
                modelscope_client_factory = _make_modelscope_client
        elif turbo_mode_enabled:
            emitter.warning("未配置 ModelScope API Key，无法启用 ModelScope 高性能通道")

    turbo_parallel_factory: Optional[Callable[[], Any]] = None
    turbo_parallel_model_name = cloud_extraction_model
    turbo_parallel_label = "云端 gpt-oss"
    turbo_unavailable_message = "云端 gpt-oss 客户端不可用，已跳过符号提取"
    turbo_enabled_description = "将使用云端 Ollama 并行处理待检文本。"
    turbo_engine_tag = "ollama"

    if turbo_mode_enabled:
        if modelscope_client_factory is not None:
            turbo_parallel_factory = modelscope_client_factory
            first_model = MODELSCOPE_COMPARISON_MODELS[0]
            turbo_parallel_model_name = first_model[1]
            turbo_parallel_label = first_model[0]
            turbo_unavailable_message = "ModelScope DeepSeek 客户端不可用，已跳过符号提取"
            turbo_enabled_description = f"将依次使用 {', '.join(label for label, _ in MODELSCOPE_COMPARISON_MODELS)} 并行处理待检文本。"
            turbo_engine_tag = "modelscope"
        elif cloud_client_factory is not None:
            turbo_parallel_factory = cloud_client_factory
        else:
            emitter.warning("未检测到 ModelScope 或云端 gpt-oss 配置，高性能模式已回退为标准模式。")
            turbo_mode_enabled = False

    if turbo_mode_enabled and turbo_parallel_factory is None:
        emitter.warning("高性能模式的所有通道均不可用，已回退为标准模式。")
        turbo_mode_enabled = False

    if turbo_mode_enabled and turbo_parallel_factory is not None:
        emitter.info(f"已启用高性能模式，{turbo_enabled_description}")

    standards_txt_filtered_further_dir = os.path.join(output_root, "standards_txt_filtered_further")
    standards_outputs: List[str] = []
    progress_target_after_conversion = 10.0
    ollama_progress_share = 85.0
    ollama_target = progress_target_after_conversion + ollama_progress_share
    gpt_task_total = len(standards_txt_filtered_files) + len(exam_txt_files)
    if exam_txt_files:
        gpt_task_total += 1  # final comparison
    increment = ollama_progress_share / gpt_task_total if gpt_task_total else 0.0

    def advance_ollama_progress() -> None:
        if increment <= 0:
            return
        publish_progress(min(ollama_target, progress_value + increment))

    if standards_txt_filtered_files:
        if not ensure_running("standards_gpt_oss", "开始基准文件符号提取"):
            return {"final_results": []}
        if turbo_mode_enabled:
            # Build provider sequence: ModelScope (3 attempts) -> Cloud Ollama -> Local Ollama
            provider_sequence: List[Tuple[str, Callable[[], Any], str, str]] = []
            if modelscope_client_factory is not None:
                for provider_label, provider_model in MODELSCOPE_COMPARISON_MODELS:
                    provider_sequence.append((
                        provider_label,
                        modelscope_client_factory,
                        provider_model,
                        "modelscope",
                    ))
            if cloud_client_factory is not None:
                provider_sequence.append((
                    "云端 gpt-oss",
                    cloud_client_factory,
                    cloud_extraction_model,
                    "ollama",
                ))
            if local_client_factory is not None:
                provider_sequence.append((
                    "本地 gpt-oss",
                    local_client_factory,
                    local_model_name,
                    "ollama",
                ))

            standards_outputs = _run_gpt_extraction_parallel(
                emitter=emitter,
                publish=publish,
                client_sequence=provider_sequence,
                session_id=session_id,
                output_root=output_root,
                src_dir=standards_txt_filtered_dir,
                file_names=standards_txt_filtered_files,
                dest_dir=standards_txt_filtered_further_dir,
                stage_label="standards_gpt_oss",
                stage_message="正在调用高性能通道提取基准文件特殊特性标记",
                clear_message_template="已清空上次基准 GPT-OSS 结果 {cleared} 个文件",
                client_unavailable_message=turbo_unavailable_message,
                combined_prefix="standards_txt_filtered_final",
                progress_callback=advance_ollama_progress if increment else None,
            )
        else:
            standards_outputs = _run_gpt_extraction(
                emitter=emitter,
                publish=publish,
                primary_client=local_ollama_client,
                primary_model_name=local_model_name,
                session_id=session_id,
                output_root=output_root,
                src_dir=standards_txt_filtered_dir,
                file_names=standards_txt_filtered_files,
                dest_dir=standards_txt_filtered_further_dir,
                stage_label="standards_gpt_oss",
                stage_message="正在调用 gpt-oss 提取基准文件特殊特性标记",
                clear_message_template="已清空上次基准 GPT-OSS 结果 {cleared} 个文件",
                client_unavailable_message="gpt-oss 客户端不可用，已跳过基准文件符号提取",
                combined_prefix="standards_txt_filtered_final",
                progress_callback=advance_ollama_progress if increment else None,
                control_handler=ensure_running,
                fallback_client=cloud_ollama_client,
                fallback_model_name=cloud_extraction_model,
            )
        if stop_announced:
            return {"final_results": []}

    if not exam_txt_files:
        publish_progress(99.0)
        publish(
            {
                "status": "succeeded",
                "stage": "completed",
                "message": "未发现待检查文本",
                "progress": progress_value,
            }
        )
        return {"final_results": []}

    examined_txt_filtered_further_dir = os.path.join(output_root, "examined_txt_filtered_further")
    if not ensure_running("gpt_oss", "开始待检文件符号提取"):
        return {"final_results": []}
    if turbo_mode_enabled:
        provider_sequence: List[Tuple[str, Callable[[], Any], str, str]] = []
        if modelscope_client_factory is not None:
            for provider_label, provider_model in MODELSCOPE_COMPARISON_MODELS:
                provider_sequence.append((
                    provider_label,
                    modelscope_client_factory,
                    provider_model,
                    "modelscope",
                ))
        if cloud_client_factory is not None:
            provider_sequence.append((
                "云端 gpt-oss",
                cloud_client_factory,
                cloud_extraction_model,
                "ollama",
            ))
        if local_client_factory is not None:
            provider_sequence.append((
                "本地 gpt-oss",
                local_client_factory,
                local_model_name,
                "ollama",
            ))

        exam_outputs = _run_gpt_extraction_parallel(
            emitter=emitter,
            publish=publish,
            client_sequence=provider_sequence,
            session_id=session_id,
            output_root=output_root,
            src_dir=exam_src_dir,
            file_names=exam_txt_files,
            dest_dir=examined_txt_filtered_further_dir,
            stage_label="gpt_oss",
            stage_message="正在调用高性能通道提取特殊特性标记",
            clear_message_template="已清空上次 GPT-OSS 结果 {cleared} 个文件",
            client_unavailable_message=turbo_unavailable_message,
            combined_prefix="examined_txt_filtered_final",
            progress_callback=advance_ollama_progress if increment else None,
        )
    else:
        exam_outputs = _run_gpt_extraction(
            emitter=emitter,
            publish=publish,
            primary_client=local_ollama_client,
            primary_model_name=local_model_name,
            session_id=session_id,
            output_root=output_root,
            src_dir=exam_src_dir,
            file_names=exam_txt_files,
            dest_dir=examined_txt_filtered_further_dir,
            stage_label="gpt_oss",
            stage_message="正在调用 gpt-oss 提取特殊特性标记",
            clear_message_template="已清空上次 GPT-OSS 结果 {cleared} 个文件",
            client_unavailable_message="gpt-oss 客户端不可用，已跳过符号提取",
            combined_prefix="examined_txt_filtered_final",
            progress_callback=advance_ollama_progress if increment else None,
            control_handler=ensure_running,
            fallback_client=cloud_ollama_client,
            fallback_model_name=cloud_extraction_model,
        )
    if stop_announced:
        return {"final_results": []}

    exam_combined_path = next(
        (
            path
            for path in exam_outputs
            if os.path.basename(path).startswith("examined_txt_filtered_final_")
        ),
        None,
    )
    standards_combined_path = next(
        (
            path
            for path in standards_outputs
            if os.path.basename(path).startswith("standards_txt_filtered_final_")
        ),
        None,
    )

    emitter.set_stage("compare")

    if cloud_ollama_client is None and local_ollama_client is None and modelscope_client_factory is None:
        emitter.error("无可用比对引擎，无法执行对比分析")
        publish(
            {
                "status": "failed",
                "stage": "compare",
                "message": "无可用比对引擎，无法执行对比分析",
            }
        )
        return {"final_results": []}

    exam_content = ""
    if exam_combined_path and os.path.isfile(exam_combined_path):
        try:
            with open(exam_combined_path, "r", encoding="utf-8") as handle:
                exam_content = handle.read()
        except Exception as error:
            report_exception("读取待检聚合文本失败", error, level="warning")
    else:
        emitter.warning("未生成待检聚合文本，使用空内容继续执行")

    standards_content = ""
    if standards_combined_path and os.path.isfile(standards_combined_path):
        try:
            with open(standards_combined_path, "r", encoding="utf-8") as handle:
                standards_content = handle.read()
        except Exception as error:
            report_exception("读取基准聚合文本失败", error, level="warning")
    elif standards_txt_filtered_files:
        emitter.warning("未生成基准聚合文本，使用空内容继续执行")

    exam_section = exam_content.strip()
    standards_section = standards_content.strip()
    if not standards_section:
        standards_section = "无"

    exam_chunks_payloads = _build_exam_chunks(exam_section)
    total_chunks = max(1, len(exam_chunks_payloads))
    processed_chunks = 0
    publish({"total_chunks": total_chunks, "processed_chunks": processed_chunks})

    output_requirements = (
        "\n\n输出要求（严格遵守）：\n"
        "- 仅输出一个 JSON 数组（UTF-8，无额外文本或说明）。\n"
        '- 数组中的每个对象必须包含以下键："条目", "基准文件名", "基准工作表名", "基准特殊特性分类", "待检查文件名", "待检查工作表名", "待检查特殊特性分类"。\n'
        '- 如果某个字段在源材料中缺失，请使用空字符串 ""。\n'
        "- 若未发现任何符号不一致，请输出空数组 []。\n"
    )
    comparison_base = os.path.basename(exam_combined_path) if exam_combined_path else "examined_txt_filtered_final"
    comparison_name = os.path.splitext(comparison_base)[0] + "_comparison"

    if not ensure_running("compare", "准备执行对比分析"):
        return {"final_results": []}
    # Prefer ModelScope for final comparison, then cloud Ollama, then local Ollama
    modelscope_attempts: List[Tuple[str, Optional[ModelScopeClient], Optional[str], str]] = []
    if modelscope_client_factory is not None:
        for label, model_id in MODELSCOPE_COMPARISON_MODELS:
            try:
                client_instance = modelscope_client_factory()
            except Exception as error:
                report_exception(f"初始化 {label} 客户端失败", error, level="warning")
                client_instance = None
            modelscope_attempts.append((label, client_instance, model_id, "modelscope"))

    # attempt tuple: (label, client, model, engine_tag)
    attempt_sequence: List[Tuple[str, object, Optional[str], str]] = [
        *modelscope_attempts,
        ("云端 deepseek", cloud_ollama_client, cloud_comparison_model, "ollama"),
        ("本地 gpt-oss", local_ollama_client, local_model_name, "ollama"),
    ]

    first_engine = next(
        (
            attempt_model
            for _, attempt_client, attempt_model, _ in attempt_sequence
            if attempt_client and attempt_model
        ),
        cloud_comparison_model,
    )

    chunk_prompts: List[str] = []
    chunk_response_sections: List[str] = []
    aggregated_rows: List[Dict[str, object]] = []
    parsed_chunk_count = 0
    failed_chunk_indices: List[int] = []
    last_error: Optional[Exception] = None

    last_chunk_submission_ts = 0.0

    for chunk_index, chunk_payload in enumerate(exam_chunks_payloads, start=1):
        if not ensure_running("compare", f"第{chunk_index}组对比分析"):
            return {"final_results": []}

        chunk_file_display = str(chunk_payload.get("file_name") or f"待检文件{chunk_index}")
        sheet_count = int(chunk_payload.get("sheet_count") or 0)
        chunk_intro = (
            f"（提示：本提示仅包含待检文本的第{chunk_index}/{total_chunks}组，源文件：{chunk_file_display}，"
            f"包含 {sheet_count} 条记录。请独立完成本组比对。）\n"
        )
        chunk_exam_section = str(chunk_payload.get("payload_text") or exam_section or "无")
        chunk_prompt = (
            f"{SPECIAL_SYMBOLS_CHUNK_PROMPT_PREFIX}{chunk_intro}{chunk_exam_section}\n\n以下是企业基准文件：\n{standards_section}"
            f"{output_requirements}"
        )
        chunk_prompts.append(chunk_prompt)
        chunk_stream_name = _make_chunk_stream_name(comparison_name, chunk_index, chunk_file_display)
        publish(
            {
                "stream": {
                    "kind": "prompt",
                    "file": f"{chunk_stream_name}.txt",
                    "part": chunk_index,
                    "total_parts": total_chunks,
                    "engine": first_engine,
                    "text": chunk_prompt,
                }
            }
        )

        if chunk_index == 1:
            last_chunk_submission_ts = time.time()
        else:
            last_chunk_submission_ts = _await_chunk_submission_window(last_chunk_submission_ts)

        chunk_start_ts = last_chunk_submission_ts
        chunk_response_text = ""
        chunk_last_stats = None
        chunk_used_model_name: Optional[str] = None
        chunk_used_engine_tag = "ollama"
        chunk_success = False

        for attempt_label, attempt_client, attempt_model, attempt_engine_tag in attempt_sequence:
            if attempt_client is None or not attempt_model:
                continue

            emitter.info(
                f"正在调用 {attempt_label} 处理第{chunk_index}组 ({chunk_file_display})"
            )
            try:
                if isinstance(attempt_client, ModelScopeClient):
                    throttled = False
                    ms_attempt_error: Optional[Exception] = None
                    chunk_response_text = ""
                    chunk_used_model_candidate: Optional[str] = None

                    for attempt_retry in range(3):
                        try:
                            _await_modelscope_window()
                            resp = attempt_client.chat(
                                model=attempt_model,
                                messages=[{"role": "user", "content": chunk_prompt}],
                                stream=False,
                                options={"num_ctx": 40001},
                            )
                        except Exception as call_error:
                            ms_attempt_error = call_error
                            if _is_rate_limit_error(call_error):
                                throttled = True
                                emitter.warning(
                                    f"{attempt_label} 返回 429（第{attempt_retry + 1}次尝试），切换至下一个模型。"
                                )
                                break
                            report_exception(
                                f"调用 {attempt_label} 对比分析失败",
                                call_error,
                                level="warning",
                            )
                            publish(
                                {
                                    "stage": "compare",
                                    "message": f"调用 {attempt_label} 失败: {call_error}",
                                    "error": str(call_error),
                                }
                            )
                            if not ensure_running(
                                "compare", f"{attempt_label} 对比分析（{chunk_stream_name}）"
                            ):
                                return {"final_results": []}
                            continue

                        piece = (
                            (resp.get("message", {}) or {}).get("content")
                            or resp.get("response")
                            or ""
                        )
                        candidate_text = piece or ""
                        publish(
                            {
                                "stream": {
                                    "kind": "response",
                                    "file": f"{chunk_stream_name}.txt",
                                    "part": chunk_index,
                                    "total_parts": total_chunks,
                                    "engine": resp.get("model") or attempt_model,
                                    "text": candidate_text,
                                }
                            }
                        )
                        chunk_last_stats = resp.get("eval_info") or resp.get("stats") or None
                        chunk_used_model_candidate = resp.get("model") or attempt_model
                        chunk_used_engine_tag = attempt_engine_tag
                        last_error = None
                        if not ensure_running(
                            "compare", f"{attempt_label} 对比分析（{chunk_stream_name}）"
                        ):
                            return {"final_results": []}
                        if _should_retry_response(candidate_text):
                            emitter.warning(
                                f"{attempt_label} 第{attempt_retry + 1}次返回为空或无效结果，准备重试。"
                            )
                            if attempt_retry < 2:
                                time.sleep(1.0)
                                continue
                            candidate_text = "[]"
                        chunk_response_text = candidate_text
                        chunk_used_model_name = chunk_used_model_candidate or attempt_model
                        chunk_success = True
                        break

                    if throttled:
                        continue
                    if not chunk_success:
                        if ms_attempt_error is not None:
                            last_error = ms_attempt_error
                        continue
                    break
                else:
                    for stream_chunk in attempt_client.chat(
                        model=attempt_model,
                        messages=[{"role": "user", "content": chunk_prompt}],
                        stream=True,
                        options={"num_ctx": 40001},
                    ):
                        piece = (
                            stream_chunk.get("message", {}).get("content")
                            or stream_chunk.get("response")
                            or ""
                        )
                        if piece:
                            chunk_response_text += piece
                        publish(
                            {
                                "stream": {
                                    "kind": "response",
                                    "file": f"{chunk_stream_name}.txt",
                                    "part": chunk_index,
                                    "total_parts": total_chunks,
                                    "engine": attempt_model,
                                    "text": chunk_response_text,
                                }
                            }
                        )
                        chunk_last_stats = (
                            stream_chunk.get("eval_info")
                            or stream_chunk.get("stats")
                            or chunk_last_stats
                        )
                        if not ensure_running(
                            "compare", f"{attempt_label} 对比分析（{chunk_stream_name}）"
                        ):
                            return {"final_results": []}
                    chunk_used_model_name = attempt_model
                    chunk_used_engine_tag = attempt_engine_tag
                    last_error = None
                    chunk_success = True
                    break
            except Exception as error:
                last_error = error
                report_exception(f"调用 {attempt_label} 对比分析失败", error, level="warning")
                publish(
                    {
                        "stage": "compare",
                        "message": f"调用 {attempt_label} 失败: {error}",
                        "error": str(error),
                    }
                )
                emitter.warning(f"{attempt_label} 调用失败，尝试备用通道")
                continue

        if not chunk_success:
            message = f"第{chunk_index}组对比失败，无法执行对比分析"
            if last_error:
                message = f"{message}：{last_error}"
            publish({"status": "failed", "stage": "compare", "message": message})
            return {"final_results": []}

        duration_ms = int((time.time() - chunk_start_ts) * 1000)
        chunk_response_clean = chunk_response_text.strip() or "[]"
        chunk_response_sections.append(
            f"【第{chunk_index}/{total_chunks}组：{chunk_file_display}】\n{chunk_response_clean}"
        )

        if chunk_used_model_name:
            log_llm_metrics(
                output_root,
                session_id,
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "engine": chunk_used_engine_tag,
                    "model": chunk_used_model_name,
                    "session_id": session_id,
                    "file": chunk_stream_name,
                    "part": chunk_index,
                    "phase": "compare",
                    "prompt_chars": len(chunk_prompt),
                    "prompt_tokens": estimate_tokens(chunk_prompt, chunk_used_model_name),
                    "output_chars": len(chunk_response_clean),
                    "output_tokens": estimate_tokens(chunk_response_clean, chunk_used_model_name),
                    "duration_ms": duration_ms,
                    "success": 1 if chunk_response_clean else 0,
                    "stats": chunk_last_stats or {},
                    "error": "",
                },
            )

        parsed_chunk = _parse_json_rows_loose(chunk_response_clean)
        if parsed_chunk is None:
            failed_chunk_indices.append(chunk_index)
        else:
            aggregated_rows.extend(parsed_chunk)
            parsed_chunk_count += 1

        processed_chunks = chunk_index
        publish({"processed_chunks": processed_chunks, "total_chunks": total_chunks})

    if not chunk_response_sections:
        publish(
            {
                "status": "failed",
                "stage": "compare",
                "message": "未能获得任何对比结果",
            }
        )
        return {"final_results": []}

    if failed_chunk_indices:
        emitter.warning(
            "以下分组未能解析为有效JSON："
            + ", ".join(str(idx) for idx in failed_chunk_indices)
        )

    response_clean = "\n\n".join(chunk_response_sections).strip()
    if not response_clean:
        response_clean = "[]"

    if increment:
        advance_ollama_progress()

    json_output_written = False
    response_to_save = response_clean
    parsed_rows: Optional[List[Dict[str, object]]] = None
    if parsed_chunk_count > 0:
        parsed_rows = aggregated_rows
    else:
        parsed_rows = _parse_json_rows_loose(response_clean)

    if parsed_rows is not None:
        filtered_rows, removed_matches = _filter_identical_matches(parsed_rows)
        parsed_rows = filtered_rows
        if removed_matches:
            emitter.info(f"已过滤掉 {removed_matches} 条符号完全一致的记录")
        response_to_save = json.dumps(parsed_rows, ensure_ascii=False, indent=2)
        json_output_path = os.path.join(initial_results_dir, f"json_{comparison_name}.txt")
        try:
            with open(json_output_path, "w", encoding="utf-8") as handle:
                handle.write(response_to_save)
        except Exception as error:
            parsed_rows = None
            report_exception("写入比对JSON结果失败", error, level="warning")
        else:
            json_output_written = True
            emitter.info(f"比对结果已输出为JSON：{os.path.basename(json_output_path)}")
    else:
        emitter.warning("比对结果未生成有效JSON，回退至二次LLM转换流程")

    try:
        if not ensure_running("aggregate", "保存对比与摘要结果"):
            return {"final_results": []}
        persist_compare_outputs(initial_results_dir, comparison_name, chunk_prompts, response_to_save)
        if not json_output_written:
            summarize_with_ollama(initial_results_dir, output_root, session_id, comparison_name, response_clean)
    except Exception as error:
        report_exception("保存比对结果失败", error, level="warning")

    emitter.set_stage("aggregate")
    try:
        if not ensure_running("aggregate", "生成导出文件"):
            return {"final_results": []}
        has_comparison_rows = aggregate_outputs(initial_results_dir, output_root, session_id)
    except Exception as error:
        report_exception("汇总导出失败", error, level="warning")
        has_comparison_rows = False

    result_files: List[str] = []
    try:
        if os.path.isdir(final_results_dir):
            for fname in os.listdir(final_results_dir):
                fpath = os.path.join(final_results_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                if fpath in preexisting_final_files:
                    continue
                try:
                    mtime = os.path.getmtime(fpath)
                except OSError:
                    mtime = None
                if mtime is not None and mtime < job_start_time:
                    continue
                result_files.append(fpath)
    except Exception:
        pass

    has_csv = any(path.lower().endswith(".csv") for path in result_files)
    has_xlsx = any(path.lower().endswith(".xlsx") for path in result_files)
    no_differences = not has_comparison_rows
    if has_csv and has_xlsx:
        publish_progress(100.0)
        progress_value = 100.0
    elif no_differences:
        publish_progress(100.0)
        progress_value = 100.0
    elif progress_value < ollama_target:
        publish_progress(min(ollama_target, progress_value))

    final_message = "特殊特性符号检查完成"
    if no_differences:
        final_message = "已完成比对，但未发现独特性符号不一致的地方。点击下方下载分析过程。"

    if not ensure_running("completed", "准备发布结果"):
        return {"final_results": []}
    publish(
        {
            "status": "succeeded",
            "stage": "completed",
            "message": final_message,
            "processed_chunks": processed_chunks,
            "total_chunks": total_chunks,
            "result_files": result_files,
            "progress": progress_value,
            "no_differences": no_differences,
        }
    )
    return {"final_results": result_files}


__all__ = ["run_special_symbols_job"]

