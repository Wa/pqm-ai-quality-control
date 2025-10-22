"""Background execution helpers for enterprise standard checks."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional

from bisheng_client import (
    call_flow_process,
    create_knowledge,
    find_knowledge_id_by_name,
    kb_sync_folder,
    parse_flow_answer,
    split_to_chunks,
)
from config import CONFIG

from util import ensure_session_dirs

from . import (
    ENTERPRISE_WORKFLOW_SURFACE,
    KB_MODEL_ID,
    aggregate_outputs,
    cleanup_orphan_txts,
    estimate_tokens,
    get_bisheng_settings,
    log_llm_metrics,
    persist_compare_outputs,
    preprocess_txt_directories,
    process_archives,
    process_excel_folder,
    process_pdf_folder,
    process_textlike_folder,
    process_word_ppt_folder,
    report_exception,
    summarize_with_ollama,
)


@dataclass
class BackgroundJobContext:
    """Mutable context passed through background execution."""

    session_id: str
    base_dirs: Dict[str, str]
    paths: object
    bisheng_session_id: Optional[str] = None
    total_chunks: int = 0
    processed_chunks: int = 0


class ProgressEmitter:
    """Adapter exposing Streamlit-like logging surface."""

    def __init__(self, publish: Callable[[Dict[str, object]], None], stage: str) -> None:
        self._publish = publish
        self._stage = stage

    def set_stage(self, stage: str) -> None:
        self._stage = stage

    def _emit(self, level: str, message: str) -> None:
        self._publish({
            "stage": self._stage,
            "log": {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "level": level,
                "message": message,
            },
        })

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


def _build_run_manifest(
    checkpoint_dir: str,
    examined_txt_dir: str,
    exam_txt_files: Iterable[str],
    publish: Callable[[Dict[str, object]], None],
) -> Dict[str, object]:
    os.makedirs(checkpoint_dir, exist_ok=True)
    manifest_path = os.path.join(checkpoint_dir, "manifest.json")
    manifest = {"entries": []}
    try:
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as handle:
                existing = json.load(handle) or {}
            entries = existing.get("entries") or []
            all_done = bool(entries) and all(str(e.get("status")) == "done" for e in entries)
            if all_done:
                for name in os.listdir(checkpoint_dir):
                    try:
                        os.remove(os.path.join(checkpoint_dir, name))
                    except Exception:
                        continue
    except Exception:
        pass

    try:
        infos: List[str] = []
        for name in sorted(exam_txt_files, key=lambda value: value.lower()):
            src = os.path.join(examined_txt_dir, name)
            try:
                stat = os.stat(src)
                infos.append(f"{name}|{stat.st_size}|{int(stat.st_mtime)}")
            except Exception:
                infos.append(f"{name}|0|0")
        run_id = hashlib.sha1("\n".join(infos).encode("utf-8", errors="ignore")).hexdigest()
    except Exception:
        run_id = ""

    manifest = {"run_id": run_id, "entries": []}
    entry_id = 0
    for fname in sorted(exam_txt_files, key=lambda value: value.lower()):
        src = os.path.join(examined_txt_dir, fname)
        try:
            with open(src, "r", encoding="utf-8") as handle:
                doc_text = handle.read()
        except Exception:
            doc_text = ""
        chunks = split_to_chunks(doc_text, int(get_bisheng_settings().max_words))
        for chunk_index, piece in enumerate(chunks):
            entry_id += 1
            prompt_text = ENTERPRISE_WORKFLOW_SURFACE.build_chunk_prompt(piece)
            num = chunk_index + 1
            prompt_name = f"checkpoint_prompt_{fname}_pt{num}.txt"
            prompt_path = os.path.join(checkpoint_dir, prompt_name)
            try:
                with open(prompt_path, "w", encoding="utf-8") as handle:
                    handle.write(prompt_text)
            except Exception:
                pass
            manifest["entries"].append(
                {
                    "id": entry_id,
                    "file_name": fname,
                    "chunk_index": chunk_index,
                    "prompt_file": prompt_name,
                    "response_file": f"checkpoint_response_{fname}_pt{num}.txt",
                    "status": "not_done",
                }
            )
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=checkpoint_dir) as tmp:
            tmp.write(json.dumps(manifest, ensure_ascii=False, indent=2))
            tmp_name = tmp.name
        shutil.move(tmp_name, manifest_path)
    except Exception:
        pass
    publish({"checkpoint": {"entries": len(manifest["entries"])}})
    return manifest


def _update_manifest_entry(checkpoint_dir: str, file_name: str, chunk_index: int) -> None:
    manifest_path = os.path.join(checkpoint_dir, "manifest.json")
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle) or {}
    except Exception:
        return
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        return
    changed = False
    for entry in entries:
        if entry.get("file_name") == file_name and int(entry.get("chunk_index", -1)) == chunk_index:
            entry["status"] = "done"
            changed = True
            break
    if not changed:
        return
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=checkpoint_dir) as tmp:
            tmp.write(json.dumps(manifest, ensure_ascii=False, indent=2))
            tmp_name = tmp.name
        shutil.move(tmp_name, manifest_path)
    except Exception:
        pass


def run_enterprise_standard_job(
    session_id: str,
    publish: Callable[[Dict[str, object]], None],
    check_control: Optional[Callable[[], Dict[str, bool]]] = None,
) -> Dict[str, List[str]]:
    """Run the enterprise standard workflow headlessly and report progress via ``publish``."""

    settings = get_bisheng_settings()
    publish({"status": "running", "stage": "initializing", "message": "准备会话目录"})

    base_dirs = {"generated": str(CONFIG["directories"]["generated_files"])}
    session_dirs = ensure_session_dirs(base_dirs, session_id)
    paths = ENTERPRISE_WORKFLOW_SURFACE.prepare_paths(session_dirs)

    context = BackgroundJobContext(session_id=session_id, base_dirs=base_dirs, paths=paths)
    emitter = ProgressEmitter(publish, stage="preparing")

    standards_dir = paths.standards_dir
    examined_dir = paths.examined_dir
    enterprise_out_root = paths.output_root
    standards_txt_dir = paths.standards_txt_dir
    examined_txt_dir = paths.examined_txt_dir
    initial_results_dir = paths.initial_results_dir
    final_results_dir = paths.final_results_dir
    checkpoint_dir = paths.checkpoint_dir

    os.makedirs(initial_results_dir, exist_ok=True)
    os.makedirs(final_results_dir, exist_ok=True)

    try:
        removed_std = cleanup_orphan_txts(standards_dir, standards_txt_dir, emitter)
        removed_exam = cleanup_orphan_txts(examined_dir, examined_txt_dir, emitter)
        if removed_std or removed_exam:
            emitter.info(f"已清理无关文本 {removed_std + removed_exam} 个")
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
    emitter.info("正在解析企业标准文件")
    process_pdf_folder(standards_dir, standards_txt_dir, emitter, annotate_sources=True)
    process_word_ppt_folder(standards_dir, standards_txt_dir, emitter, annotate_sources=True)
    process_excel_folder(standards_dir, standards_txt_dir, emitter, annotate_sources=True)
    process_textlike_folder(standards_dir, standards_txt_dir, emitter)
    process_archives(standards_dir, standards_txt_dir, emitter)

    emitter.info("正在解析待检查文件")
    process_pdf_folder(examined_dir, examined_txt_dir, emitter, annotate_sources=False)
    process_word_ppt_folder(examined_dir, examined_txt_dir, emitter, annotate_sources=False)
    process_excel_folder(examined_dir, examined_txt_dir, emitter, annotate_sources=False)
    process_textlike_folder(examined_dir, examined_txt_dir, emitter)
    process_archives(examined_dir, examined_txt_dir, emitter)

    try:
        updated_txts = preprocess_txt_directories(standards_txt_dir, examined_txt_dir)
        if updated_txts:
            emitter.info(f"文本预处理完成 {len(updated_txts)} 个")
    except Exception as error:
        report_exception("文本预处理失败", error, level="warning")

    std_txt_files = _list_txt_files(standards_txt_dir)
    exam_txt_files = _list_txt_files(examined_txt_dir)
    if not exam_txt_files:
        publish({"status": "succeeded", "stage": "completed", "message": "未发现待检查文本"})
        return {"final_results": []}

    manifest = _build_run_manifest(checkpoint_dir, examined_txt_dir, exam_txt_files, publish)
    total_chunks = len(manifest.get("entries", []))
    context.total_chunks = total_chunks
    publish({"total_chunks": total_chunks, "processed_chunks": 0})

    emitter.set_stage("kb_sync")
    kb_name_dyn = ENTERPRISE_WORKFLOW_SURFACE.knowledge_base_name(session_id)
    kid = None
    try:
        kid = find_knowledge_id_by_name(settings.base_url, settings.api_key or None, kb_name_dyn)
        if not kid:
            kid = create_knowledge(settings.base_url, settings.api_key or None, kb_name_dyn, model=str(KB_MODEL_ID))
        if kid:
            res = kb_sync_folder(
                base_url=settings.base_url,
                api_key=settings.api_key or None,
                knowledge_id=int(kid),
                folder_path=standards_txt_dir,
                clear_first=False,
                chunk_size=1000,
                chunk_overlap=0,
                separators=["\n\n", "\n"],
                separator_rule=["after", "after"],
            )
            emitter.info(
                "KB同步完成 上传{} 删除{} 跳过{}".format(
                    len(res.get("uploaded", [])),
                    len(res.get("deleted", [])),
                    len(res.get("skipped", [])),
                )
            )
        else:
            emitter.warning("KB 创建或查找失败，继续执行但可能影响检索效果")
    except Exception as error:
        report_exception("企业标准KB同步失败", error, level="warning")

    emitter.set_stage("warmup")
    try:
        warmup_prompt = ENTERPRISE_WORKFLOW_SURFACE.warmup_prompt or "预热：请简短回复 'gotcha' 即可。"
        call_flow_process(
            base_url=settings.base_url,
            flow_id=settings.flow_id,
            question=warmup_prompt,
            kb_id=None,
            input_node_id=settings.flow_input_node_id,
            api_key=settings.api_key or None,
            session_id=None,
            history_count=0,
            extra_tweaks={"CombineDocsChain-520ca": {"token_max": 5000}},
            milvus_node_id=settings.flow_milvus_node_id,
            es_node_id=settings.flow_es_node_id,
            timeout_s=60,
            max_retries=0,
        )
    except Exception:
        pass

    emitter.set_stage("compare")
    processed_chunks = 0
    for name in sorted(exam_txt_files, key=lambda value: value.lower()):
        # Check control flags before starting each file
        try:
            status = check_control() if check_control else None
        except Exception:
            status = None
        if status:
            if status.get("stopped"):
                publish({
                    "status": "failed",
                    "stage": "stopped",
                    "message": "任务已被用户停止",
                    "processed_chunks": processed_chunks,
                    "total_chunks": total_chunks,
                })
                return {"final_results": []}
            while status.get("paused") and not status.get("stopped"):
                publish({"status": "paused", "stage": "paused", "message": f"暂停中：等待恢复（当前文件 {name}）"})
                time.sleep(1)
                try:
                    status = check_control() if check_control else None
                except Exception:
                    status = None
            if status and status.get("stopped"):
                publish({
                    "status": "failed",
                    "stage": "stopped",
                    "message": "任务已被用户停止",
                    "processed_chunks": processed_chunks,
                    "total_chunks": total_chunks,
                })
                return {"final_results": []}
        publish({"current_file": name, "stage": "compare", "message": f"比对 {name}"})
        try:
            with open(os.path.join(examined_txt_dir, name), "r", encoding="utf-8") as handle:
                doc_text = handle.read()
        except Exception as error:
            report_exception(f"读取待检查文本失败({name})", error, level="warning")
            continue
        if not doc_text.strip():
            emitter.warning(f"文件为空，跳过 {name}")
            continue
        chunks = split_to_chunks(doc_text, int(settings.max_words))
        full_out_text = ""
        prompt_texts: List[str] = []
        total_parts = len(chunks)
        for index, piece in enumerate(chunks, start=1):
            # Check control flags before each chunk
            try:
                status = check_control() if check_control else None
            except Exception:
                status = None
            if status:
                if status.get("stopped"):
                    publish({
                        "status": "failed",
                        "stage": "stopped",
                        "message": "任务已被用户停止",
                        "processed_chunks": processed_chunks,
                        "total_chunks": total_chunks,
                    })
                    return {"final_results": []}
                while status.get("paused") and not status.get("stopped"):
                    publish({"status": "paused", "stage": "paused", "message": f"暂停中：等待恢复（{name} 第{index}/{total_parts}）"})
                    time.sleep(1)
                    try:
                        status = check_control() if check_control else None
                    except Exception:
                        status = None
                if status and status.get("stopped"):
                    publish({
                        "status": "failed",
                        "stage": "stopped",
                        "message": "任务已被用户停止",
                        "processed_chunks": processed_chunks,
                        "total_chunks": total_chunks,
                    })
                    return {"final_results": []}
            prompt_text = ENTERPRISE_WORKFLOW_SURFACE.build_chunk_prompt(piece)
            prompt_texts.append(prompt_text)
            publish(
                {
                    "stream": {
                        "kind": "prompt",
                        "file": name,
                        "part": index,
                        "total_parts": total_parts,
                        "text": prompt_text,
                    }
                }
            )
            publish({
                "stage": "compare",
                "message": f"调用 Bisheng ({name} 第{index}/{len(chunks)}段)",
                "processed_chunks": processed_chunks,
                "total_chunks": total_chunks,
            })
            start_ts = time.time()
            try:
                response = call_flow_process(
                    base_url=settings.base_url,
                    flow_id=settings.flow_id,
                    question=prompt_text,
                    kb_id=kid,
                    input_node_id=settings.flow_input_node_id,
                    api_key=settings.api_key or None,
                    session_id=context.bisheng_session_id,
                    history_count=0,
                    extra_tweaks=None,
                    milvus_node_id=settings.flow_milvus_node_id,
                    es_node_id=settings.flow_es_node_id,
                    timeout_s=settings.timeout_s or 180,
                    max_retries=2,
                )
                ans_text, new_sid = parse_flow_answer(response)
            except Exception as error:
                publish({
                    "stage": "compare",
                    "message": f"调用 Bisheng 失败: {error}",
                    "error": str(error),
                })
                raise
            duration_ms = int((time.time() - start_ts) * 1000)
            context.bisheng_session_id = new_sid or context.bisheng_session_id
            log_llm_metrics(
                enterprise_out_root,
                session_id,
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "engine": "bisheng",
                    "model": "qwen3",
                    "session_id": context.bisheng_session_id or "",
                    "file": name,
                    "part": index,
                    "phase": "compare",
                    "prompt_chars": len(prompt_text or ""),
                    "prompt_tokens": estimate_tokens(prompt_text or ""),
                    "output_chars": len(ans_text or ""),
                    "output_tokens": estimate_tokens(ans_text or ""),
                    "duration_ms": duration_ms,
                    "success": 1 if (ans_text or "").strip() else 0,
                    "error": response.get("error") if isinstance(response, dict) else "",
                },
            )
            full_out_text += ("\n\n" if full_out_text else "") + (ans_text or "")
            if "<think>" in (ans_text or ""):
                try:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    resp_name = f"checkpoint_response_{name}_pt{index}.txt"
                    resp_path = os.path.join(checkpoint_dir, resp_name)
                    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=checkpoint_dir) as tmp:
                        tmp.write(ans_text or "")
                        tmp_name = tmp.name
                    shutil.move(tmp_name, resp_path)
                    _update_manifest_entry(checkpoint_dir, name, index - 1)
                except Exception:
                    pass
            processed_chunks += 1
            publish({"processed_chunks": processed_chunks, "total_chunks": total_chunks})
            publish(
                {
                    "stream": {
                        "kind": "response",
                        "file": name,
                        "part": index,
                        "total_parts": total_parts,
                        "text": ans_text or "",
                    }
                }
            )
        try:
            name_no_ext = os.path.splitext(name)[0]
            persist_compare_outputs(initial_results_dir, name_no_ext, prompt_texts, full_out_text)
            summarize_with_ollama(initial_results_dir, enterprise_out_root, session_id, name_no_ext, full_out_text)
        except Exception as error:
            report_exception(f"保存比对结果失败({name})", error, level="warning")

    emitter.set_stage("aggregate")
    try:
        aggregate_outputs(initial_results_dir, enterprise_out_root, session_id)
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

    publish({
        "status": "succeeded",
        "stage": "completed",
        "message": "企业标准检查完成",
        "processed_chunks": processed_chunks,
        "total_chunks": total_chunks,
        "result_files": result_files,
    })
    return {"final_results": result_files}
