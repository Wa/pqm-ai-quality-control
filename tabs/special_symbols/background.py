"""Background worker for the special symbols workflow."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from config import CONFIG

from ollama import Client as OllamaClient

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
    "3) 在输出中给出每条记录的核心信息：工序编号（如有，OP号）、工序/设备/过程名称（可用）、项目/特性名称，以及标记符号（★/☆/ /）。\n"
    "   如果缺少某些字段，则尽量从上下文补充；无法确定时仅给出项目/特性名称+标记符号。\n"
    "4) 仅输出项目清单，每行一个条目，不要解释、不要编号。\n"
    "5) 若整段文本中没有任何被上述分类列标记为★/☆/ /的行，则仅回复：无。\n\n"
    "以下提供表头参考与数据片段。请务必依据表头中的“特殊特性分类/Classification”列来判断：\n"
)


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

            try:
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
                call_success = True
                break
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


def run_special_symbols_job(
    session_id: str,
    publish: Callable[[Dict[str, object]], None],
    check_control: Optional[Callable[[], Dict[str, bool]]] = None,
) -> Dict[str, List[str]]:
    """Run the special symbols workflow headlessly and report progress via ``publish``."""

    publish({"status": "running", "stage": "initializing", "message": "准备特殊特性符号会话目录"})

    progress_value = 0.0
    job_start_time = time.time()

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

    special_uploads = CONFIG["uploads"]["special_symbols"]
    base_dirs = {
        "special_symbols_reference": special_uploads["reference"],
        "special_symbols_inspected": special_uploads["inspected"],
        "generated": {"path": CONFIG["directories"]["generated_files"]},
    }
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
    if standards_txt_filtered_files or exam_txt_files:
        try:
            host = resolve_ollama_host("ollama_9")
            local_ollama_client = OllamaClient(host=host)
        except Exception as error:
            report_exception("初始化本地 gpt-oss 客户端失败", error, level="warning")
        cloud_host = CONFIG["llm"].get("ollama_cloud_host")
        cloud_api_key = CONFIG["llm"].get("ollama_cloud_api_key")
        if cloud_host and cloud_api_key:
            try:
                cloud_ollama_client = OllamaClient(
                    host=cloud_host,
                    headers={"Authorization": f"Bearer {cloud_api_key}"},
                )
            except Exception as error:
                report_exception("初始化云端 gpt-oss 客户端失败", error, level="warning")
        else:
            emitter.warning("未配置云端 gpt-oss，无法启用云端备份")

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

    total_chunks = 1
    processed_chunks = 0
    publish({"total_chunks": total_chunks, "processed_chunks": processed_chunks})

    emitter.set_stage("compare")

    if cloud_ollama_client is None and local_ollama_client is None:
        emitter.error("gpt-oss 客户端不可用，无法执行对比分析")
        publish(
            {
                "status": "failed",
                "stage": "compare",
                "message": "gpt-oss 客户端不可用，无法执行对比分析",
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

    combined_prompt = (
        f"{SPECIAL_SYMBOLS_CHUNK_PROMPT_PREFIX}{exam_section}\n\n以下是企业基准文件：\n{standards_section}"
    )

    comparison_base = os.path.basename(exam_combined_path) if exam_combined_path else "examined_txt_filtered_final"
    comparison_name = os.path.splitext(comparison_base)[0] + "_comparison"

    if not ensure_running("compare", "准备执行对比分析"):
        return {"final_results": []}
    attempt_sequence: List[Tuple[str, Optional[OllamaClient], Optional[str]]] = [
        ("云端 deepseek", cloud_ollama_client, cloud_comparison_model),
        ("本地 gpt-oss", local_ollama_client, local_model_name),
    ]

    first_engine = next(
        (attempt_model for _, attempt_client, attempt_model in attempt_sequence if attempt_client and attempt_model),
        cloud_comparison_model,
    )
    publish(
        {
            "stream": {
                "kind": "prompt",
                "file": f"{comparison_name}.txt",
                "part": 1,
                "total_parts": 1,
                "engine": first_engine,
                "text": combined_prompt,
            }
        }
    )

    response_text = ""
    last_stats = None
    used_model_name: Optional[str] = None
    start_ts = 0.0
    last_error: Optional[Exception] = None

    for attempt_label, attempt_client, attempt_model in attempt_sequence:
        if attempt_client is None or not attempt_model:
            continue

        emitter.info(f"正在调用 {attempt_label} 进行对比分析")
        start_ts = time.time()
        response_text = ""
        last_stats = None
        try:
            for chunk in attempt_client.chat(
                model=attempt_model,
                messages=[{"role": "user", "content": combined_prompt}],
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
                publish(
                    {
                        "stream": {
                            "kind": "response",
                            "file": f"{comparison_name}.txt",
                            "part": 1,
                            "total_parts": 1,
                            "engine": attempt_model,
                            "text": response_text,
                        }
                    }
                )
                last_stats = chunk.get("eval_info") or chunk.get("stats") or last_stats
                if not ensure_running("compare", f"{attempt_label} 对比分析（{comparison_name}）"):
                    return {"final_results": []}
            used_model_name = attempt_model
            last_error = None
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

    if used_model_name is None:
        message = "云端与本地 gpt-oss 均不可用，无法执行对比分析"
        if last_error:
            message = f"{message}：{last_error}"
        publish({"status": "failed", "stage": "compare", "message": message})
        return {"final_results": []}

    duration_ms = int((time.time() - start_ts) * 1000)
    if not ensure_running("compare", "对比分析完成"):
        return {"final_results": []}
    response_clean = response_text.strip() or "无相关发现"

    log_llm_metrics(
        output_root,
        session_id,
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "engine": "ollama",
                "model": used_model_name,
                "session_id": session_id,
                "file": comparison_name,
                "part": 1,
                "phase": "compare",
                "prompt_chars": len(combined_prompt),
                "prompt_tokens": estimate_tokens(combined_prompt, used_model_name),
                "output_chars": len(response_clean),
                "output_tokens": estimate_tokens(response_clean, used_model_name),
                "duration_ms": duration_ms,
                "success": 1 if response_clean else 0,
                "stats": last_stats or {},
                "error": "",
            },
    )

    if increment:
        advance_ollama_progress()

    processed_chunks = 1
    publish({"processed_chunks": processed_chunks, "total_chunks": total_chunks})

    try:
        if not ensure_running("aggregate", "保存对比与摘要结果"):
            return {"final_results": []}
        persist_compare_outputs(initial_results_dir, comparison_name, [combined_prompt], response_clean)
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

