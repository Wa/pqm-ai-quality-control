"""Background worker for the special symbols workflow."""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional

from config import CONFIG

from ollama import Client as OllamaClient

from util import ensure_session_dirs, resolve_ollama_host

from tabs.enterprise_standard.summaries import persist_compare_outputs
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

GPT_OSS_HEADER_TEMPLATE = "以下是《{base}》文件包含的特殊特性符号："


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
    client: Optional[OllamaClient],
    model_name: str,
    session_id: str,
    output_root: str,
    src_dir: str,
    file_names: Iterable[str],
    dest_dir: str,
    stage_label: str,
    stage_message: str,
    clear_message_template: Optional[str],
    client_unavailable_message: str,
    header_builder: Callable[[str], str],
    combined_prefix: str,
    combined_log_template: str = "已生成汇总结果 {name}",
    progress_callback: Optional[Callable[[], None]] = None,
) -> List[str]:
    names = sorted(file_names, key=lambda value: value.lower())
    if not names:
        return []

    emitter.set_stage(stage_label)
    if client is None:
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
        publish({"stage": stage_label, "message": f"调用 gpt-oss ({name})"})

        start_ts = time.time()
        response_text = ""
        last_stats = None
        try:
            for chunk in client.chat(
                model=model_name,
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
        except Exception as error:
            report_exception(f"调用 gpt-oss 失败({stage_label}:{name})", error, level="warning")
            publish(
                {
                    "stream": {
                        "kind": "response",
                        "file": name,
                        "part": 1,
                        "total_parts": 1,
                        "engine": "gpt-oss",
                        "text": f"调用 gpt-oss 失败：{error}",
                    }
                }
            )
            if progress_callback:
                progress_callback()
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
                    "engine": "gpt-oss",
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
                "model": model_name,
                "session_id": session_id,
                "file": name,
                "part": 1,
                "phase": stage_label,
                "prompt_chars": len(prompt_text),
                "prompt_tokens": estimate_tokens(prompt_text, model_name),
                "output_chars": len(response_clean),
                "output_tokens": estimate_tokens(response_clean, model_name),
                "duration_ms": duration_ms,
                "success": 1 if response_clean else 0,
                "stats": last_stats or {},
                "error": "",
            },
        )

        base_name = os.path.splitext(name)[0]
        dst_path = os.path.join(dest_dir, name)
        header = header_builder(base_name)
        try:
            with open(dst_path, "w", encoding="utf-8") as writer:
                if header:
                    writer.write(header.rstrip("\n") + "\n")
                writer.write(response_clean)
            outputs.append(dst_path)
        except Exception as error:
            report_exception(f"写入 gpt-oss 结果失败({stage_label}:{name})", error, level="warning")

        if progress_callback:
            progress_callback()

    if outputs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_name = f"{combined_prefix}_{timestamp}.txt"
        combined_path = os.path.join(dest_dir, combined_name)
        try:
            with open(combined_path, "w", encoding="utf-8") as handle:
                for idx, path in enumerate(sorted(outputs, key=lambda value: os.path.basename(value).lower())):
                    try:
                        with open(path, "r", encoding="utf-8") as reader:
                            content = reader.read().rstrip()
                    except Exception as error:
                        report_exception(
                            f"读取 gpt-oss 结果失败({stage_label}:{os.path.basename(path)})",
                            error,
                            level="warning",
                        )
                        continue
                    if idx:
                        handle.write("\n\n")
                    handle.write(content)
            outputs.append(combined_path)
            emitter.info(combined_log_template.format(name=combined_name))
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

    def publish_progress(value: float) -> None:
        nonlocal progress_value
        progress_value = max(0.0, min(value, 100.0))
        publish({"progress": progress_value})

    publish_progress(1.0)

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
    emitter.info("正在解析基准文件")
    process_pdf_folder(reference_dir, reference_txt_dir, emitter, annotate_sources=True)
    process_word_ppt_folder(reference_dir, reference_txt_dir, emitter, annotate_sources=True)
    process_excel_folder(reference_dir, reference_txt_dir, emitter, annotate_sources=True)
    process_textlike_folder(reference_dir, reference_txt_dir, emitter)
    process_archives(reference_dir, reference_txt_dir, emitter)

    emitter.info("正在解析待检查文件")
    process_pdf_folder(examined_dir, examined_txt_dir, emitter, annotate_sources=False)
    process_word_ppt_folder(examined_dir, examined_txt_dir, emitter, annotate_sources=False)
    process_excel_folder(examined_dir, examined_txt_dir, emitter, annotate_sources=False)
    process_textlike_folder(examined_dir, examined_txt_dir, emitter)
    process_archives(examined_dir, examined_txt_dir, emitter)

    try:
        updated_txts = preprocess_txt_directories(reference_txt_dir, examined_txt_dir)
        if updated_txts:
            emitter.info(f"文本预处理完成 {len(updated_txts)} 个")
    except Exception as error:
        report_exception("文本预处理失败", error, level="warning")

    publish_progress(10.0)

    standards_txt_filtered_dir = os.path.join(output_root, "standards_txt_filtered")
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

    model_name = CONFIG["llm"].get("ollama_model") or "gpt-oss:latest"
    ollama_client: Optional[OllamaClient] = None
    if standards_txt_filtered_files or exam_txt_files:
        try:
            host = resolve_ollama_host("ollama_9")
            ollama_client = OllamaClient(host=host)
        except Exception as error:
            report_exception("初始化 gpt-oss 客户端失败", error, level="warning")

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
        standards_outputs = _run_gpt_extraction(
            emitter=emitter,
            publish=publish,
            client=ollama_client,
            model_name=model_name,
            session_id=session_id,
            output_root=output_root,
            src_dir=standards_txt_filtered_dir,
            file_names=standards_txt_filtered_files,
            dest_dir=standards_txt_filtered_further_dir,
            stage_label="standards_gpt_oss",
            stage_message="正在调用 gpt-oss 提取基准文件特殊特性标记",
            clear_message_template="已清空上次基准 GPT-OSS 结果 {cleared} 个文件",
            client_unavailable_message="gpt-oss 客户端不可用，已跳过基准文件符号提取",
            header_builder=lambda base: GPT_OSS_HEADER_TEMPLATE.format(base=base),
            combined_prefix="standards_txt_filtered_final",
            progress_callback=advance_ollama_progress if increment else None,
        )

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
    exam_outputs = _run_gpt_extraction(
        emitter=emitter,
        publish=publish,
        client=ollama_client,
        model_name=model_name,
        session_id=session_id,
        output_root=output_root,
        src_dir=exam_src_dir,
        file_names=exam_txt_files,
        dest_dir=examined_txt_filtered_further_dir,
        stage_label="gpt_oss",
        stage_message="正在调用 gpt-oss 提取特殊特性标记",
        clear_message_template="已清空上次 GPT-OSS 结果 {cleared} 个文件",
        client_unavailable_message="gpt-oss 客户端不可用，已跳过符号提取",
        header_builder=lambda base: GPT_OSS_HEADER_TEMPLATE.format(base=base),
        combined_prefix="examined_txt_filtered_final",
        progress_callback=advance_ollama_progress if increment else None,
    )

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

    if ollama_client is None:
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

    publish(
        {
            "stream": {
                "kind": "prompt",
                "file": f"{comparison_name}.txt",
                "part": 1,
                "total_parts": 1,
                "engine": "gpt-oss",
                "text": combined_prompt,
            }
        }
    )
    emitter.info("正在调用 gpt-oss 进行对比分析")

    start_ts = time.time()
    response_text = ""
    last_stats = None
    try:
        for chunk in ollama_client.chat(
            model=model_name,
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
                        "engine": "gpt-oss",
                        "text": response_text,
                    }
                }
            )
            last_stats = chunk.get("eval_info") or chunk.get("stats") or last_stats
    except Exception as error:
        publish(
            {
                "stage": "compare",
                "message": f"调用 gpt-oss 失败: {error}",
                "error": str(error),
            }
        )
        raise

    duration_ms = int((time.time() - start_ts) * 1000)
    response_clean = response_text.strip() or "无相关发现"

    log_llm_metrics(
        output_root,
        session_id,
        {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "engine": "ollama",
            "model": model_name,
            "session_id": session_id,
            "file": comparison_name,
            "part": 1,
            "phase": "compare",
            "prompt_chars": len(combined_prompt),
            "prompt_tokens": estimate_tokens(combined_prompt, model_name),
            "output_chars": len(response_clean),
            "output_tokens": estimate_tokens(response_clean, model_name),
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
        persist_compare_outputs(initial_results_dir, comparison_name, [combined_prompt], response_clean)
        summarize_with_ollama(initial_results_dir, output_root, session_id, comparison_name, response_clean)
    except Exception as error:
        report_exception("保存比对结果失败", error, level="warning")

    emitter.set_stage("aggregate")
    try:
        aggregate_outputs(initial_results_dir, output_root, session_id)
    except Exception as error:
        report_exception("汇总导出失败", error, level="warning")

    result_files: List[str] = []
    try:
        if os.path.isdir(final_results_dir):
            for fname in os.listdir(final_results_dir):
                if os.path.isfile(os.path.join(final_results_dir, fname)):
                    result_files.append(os.path.join(final_results_dir, fname))
    except Exception:
        pass

    has_csv = any(path.lower().endswith(".csv") for path in result_files)
    has_xlsx = any(path.lower().endswith(".xlsx") for path in result_files)
    if has_csv and has_xlsx:
        publish_progress(100.0)
    elif progress_value < ollama_target:
        publish_progress(min(ollama_target, progress_value))

    publish(
        {
            "status": "succeeded",
            "stage": "completed",
            "message": "特殊特性符号检查完成",
            "processed_chunks": processed_chunks,
            "total_chunks": total_chunks,
            "result_files": result_files,
            "progress": progress_value,
        }
    )
    return {"final_results": result_files}


__all__ = ["run_special_symbols_job"]

