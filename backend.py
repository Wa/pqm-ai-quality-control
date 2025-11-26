from dataclasses import dataclass, field
import json
import multiprocessing
from multiprocessing import Process, Queue
from queue import Full
import os
from pathlib import Path
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ollama import Client as OllamaClient
from pydantic import BaseModel, Field

from config import CONFIG
from tabs.enterprise_standard.background import run_enterprise_standard_job
from tabs.file_completeness import (
    run_file_completeness_job,
    STAGE_ORDER,
    STAGE_SLUG_MAP,
    STAGE_REQUIREMENTS,
    CANONICAL_APQP_DESCRIPTORS,
    descriptor_for,
)
from tabs.file_elements.background import run_file_elements_job
from tabs.parameters.background import run_parameters_job
from tabs.history.background import run_history_job
from tabs.special_symbols.background import run_special_symbols_job
from tabs.shared.file_conversion import (
    cleanup_orphan_txts,
    process_excel_folder,
    process_pdf_folder,
    process_textlike_folder,
    process_word_ppt_folder,
)
from tabs.shared.modelscope_client import ModelScopeClient
from util import resolve_ollama_host


JOB_DEFINITIONS = {
    "enterprise_standard": {
        "runner": run_enterprise_standard_job,
        "label": "企业标准检查",
    },
    "special_symbols": {
        "runner": run_special_symbols_job,
        "label": "特殊特性符号检查",
    },
    "parameters": {
        "runner": run_parameters_job,
        "label": "参数一致性检查",
    },
    "file_completeness": {
        "runner": run_file_completeness_job,
        "label": "文件齐套性检查",
    },
    "history": {
        "runner": run_history_job,
        "label": "历史问题规避",
    },
    "file_elements": {
        "runner": run_file_elements_job,
        "label": "文件要素检查",
    },
    "apqp_one_click_parse": {
        "runner": None,  # Placeholder; assigned after function definition
        "label": "APQP一键解析",
    },
}


def _job_label(job_type: str) -> str:
    return JOB_DEFINITIONS.get(job_type, {}).get("label", job_type)

app = FastAPI(title="PQM AI Backend", description="File operations backend for PQM AI Quality Control")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    # Allow common dev origins; for portability, enable all during development
    allow_origins=["*"],  # Consider restricting in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class FileOperation(BaseModel):
    session_id: str
    file_path: str = None

class ClearFilesRequest(BaseModel):
    session_id: str

class FileInfo(BaseModel):
    name: str
    size: int
    modified_time: str


class EnterpriseJobRequest(BaseModel):
    session_id: str


class SpecialSymbolsJobRequest(BaseModel):
    session_id: str
    turbo_mode: bool = False


class FileElementsRequirementPayload(BaseModel):
    key: Optional[str] = None
    name: str
    severity: Optional[str] = None
    description: Optional[str] = None
    guidance: Optional[str] = None


class FileElementsProfilePayload(BaseModel):
    id: Optional[str] = None
    stage: Optional[str] = None
    name: str
    description: Optional[str] = None
    references: Optional[List[str]] = None
    requirements: List[FileElementsRequirementPayload]


class FileElementsJobRequest(BaseModel):
    session_id: str
    profile: FileElementsProfilePayload
    source_paths: Optional[List[str]] = None


class ApqpParseRequest(BaseModel):
    session_id: str
    stages: Optional[List[str]] = None


class ApqpParseJobRequest(BaseModel):
    session_id: str
    stages: Optional[List[str]] = None


class ApqpClearRequest(BaseModel):
    session_id: str
    target: Optional[str] = "all"


class ApqpClassifyRequest(BaseModel):
    session_id: str
    stages: Optional[List[str]] = None
    head_chars: int = 3200
    tail_chars: int = 2000
    turbo_mode: bool = False


class EnterpriseJobStatus(BaseModel):
    job_id: str
    session_id: str
    status: str
    stage: Optional[str] = None
    message: Optional[str] = None
    processed_chunks: int = 0
    total_chunks: int = 0
    progress: float = 0.0
    result_files: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    logs: List[Dict[str, Any]] = Field(default_factory=list)
    stream_events: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float
    updated_at: float
    pid: Optional[int] = None
    job_type: str = Field(default="enterprise_standard")


@dataclass
class JobRecord:
    job_id: str
    session_id: str
    job_type: str = "enterprise_standard"
    status: str = "queued"
    stage: Optional[str] = None
    message: Optional[str] = None
    processed_chunks: int = 0
    total_chunks: int = 0
    progress: float = 0.0
    result_files: List[str] = field(default_factory=list)
    error: Optional[str] = None
    logs: List[Dict[str, Any]] = field(default_factory=list)
    stream_events: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    process: Optional[Process] = None
    pid: Optional[int] = None


jobs_lock = Lock()
jobs: Dict[str, JobRecord] = {}

# IPC queue for background worker processes to push incremental updates
_update_queue: "Queue[Tuple[str, Dict[str, Any]]]" = multiprocessing.Queue()


def _update_listener_loop(queue: "Queue[Tuple[str, Dict[str, Any]]]") -> None:
    """Drain the multiprocessing queue and apply updates to job records."""

    while True:
        try:
            job_id, payload = queue.get()
        except Exception:
            continue
        if job_id is None:
            break
        try:
            if isinstance(payload, dict):
                _update_job(job_id, payload)
        except Exception:
            continue


if multiprocessing.current_process().name == "MainProcess":
    _listener_thread = Thread(
        target=_update_listener_loop,
        args=(_update_queue,),
        daemon=True,
    )
    _listener_thread.start()


def _record_to_status(record: JobRecord) -> EnterpriseJobStatus:
    return EnterpriseJobStatus(
        job_id=record.job_id,
        session_id=record.session_id,
        status=record.status,
        stage=record.stage,
        message=record.message,
        processed_chunks=record.processed_chunks,
        total_chunks=record.total_chunks,
        progress=record.progress,
        result_files=record.result_files,
        error=record.error,
        logs=record.logs,
        stream_events=record.stream_events,
        metadata=record.metadata,
        created_at=record.created_at,
        updated_at=record.updated_at,
        pid=record.pid,
        job_type=record.job_type,
    )


def _update_job(job_id: str, update: Dict[str, Any]) -> None:
    with jobs_lock:
        record = jobs.get(job_id)
        if not record:
            return
        record.updated_at = time.time()
        if "status" in update:
            record.status = str(update["status"])
        if "stage" in update:
            record.stage = str(update["stage"]) if update["stage"] is not None else None
        if "message" in update:
            record.message = str(update["message"]) if update["message"] is not None else None
        if "processed_chunks" in update:
            try:
                record.processed_chunks = int(update["processed_chunks"])  # type: ignore[arg-type]
            except (ValueError, TypeError):
                pass
        if "total_chunks" in update:
            try:
                record.total_chunks = int(update["total_chunks"])  # type: ignore[arg-type]
            except (ValueError, TypeError):
                pass
        if "progress" in update:
            try:
                record.progress = float(update["progress"])  # type: ignore[arg-type]
            except (ValueError, TypeError):
                pass
        if "result_files" in update:
            try:
                record.result_files = [str(path) for path in update["result_files"]]
            except Exception:
                record.result_files = []
        if "error" in update and update["error"]:
            record.error = str(update["error"])
        if "log" in update:
            log_entry = update["log"]
            if isinstance(log_entry, dict):
                record.logs.append(log_entry)
                if len(record.logs) > 200:
                    record.logs = record.logs[-200:]
        if "stream" in update:
            stream_entry = update["stream"]
            if isinstance(stream_entry, dict):
                entry = dict(stream_entry)
                seq_counter = int(record.metadata.get("_stream_seq", 0))
                seq_counter += 1
                record.metadata["_stream_seq"] = seq_counter
                entry.setdefault("sequence", seq_counter)
                entry.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
                record.stream_events.append(entry)
                if len(record.stream_events) > 120:
                    record.stream_events = record.stream_events[-120:]
        if "checkpoint" in update:
            record.metadata["checkpoint"] = update["checkpoint"]
        if "pid" in update:
            try:
                record.pid = int(update["pid"]) if update["pid"] is not None else None
            except (TypeError, ValueError):
                record.pid = None


def _prune_jobs(max_age_seconds: float = 6 * 3600) -> None:
    cutoff = time.time() - max_age_seconds
    with jobs_lock:
        expired = [job_id for job_id, record in jobs.items() if record.updated_at < cutoff and record.status in {"succeeded", "failed"}]
        for job_id in expired:
            jobs.pop(job_id, None)


def _job_process_entry(
    job_id: str,
    session_id: str,
    queue: "Queue[Tuple[str, Dict[str, Any]]]",
    job_type: str,
    runner_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Entry point executed within a separate process for long-running jobs."""
    # Backward-compat signature retained; control_dir resolved from metadata if available.

    def publish(update: Dict[str, Any]) -> None:
        payload = dict(update or {})
        try:
            queue.put_nowait((job_id, payload))
        except Full:
            pass
        except Exception:
            pass

    # Attempt to locate control_dir from environment variable set by parent
    _ctrl_key = f"PQM_JOB_CONTROL_DIR_{job_id}"
    control_dir = os.environ.get(_ctrl_key, "")

    def check_control() -> Dict[str, bool]:
        if not control_dir:
            return {"paused": False, "stopped": False}
        try:
            paused = os.path.isfile(os.path.join(control_dir, "pause"))
            stopped = os.path.isfile(os.path.join(control_dir, "stop"))
            return {"paused": paused, "stopped": stopped}
        except Exception:
            return {"paused": False, "stopped": False}

    runner_kwargs = dict(runner_kwargs or {})

    runner_def = JOB_DEFINITIONS.get(job_type)
    if not runner_def:
        publish(
            {
                "status": "failed",
                "stage": "completed",
                "error": f"未知的任务类型: {job_type}",
                "message": "任务初始化失败",
            }
        )
        return

    try:
        runner_def["runner"](session_id, publish, check_control=check_control, **runner_kwargs)
    except Exception as error:
        publish(
            {
                "status": "failed",
                "stage": "completed",
                "error": str(error),
                "message": "后台任务执行失败",
            }
        )
        try:
            queue.put_nowait(
                (
                    job_id,
                    {
                        "log": {
                            "ts": datetime.now().isoformat(timespec="seconds"),
                            "level": "error",
                            "message": traceback.format_exc(limit=10),
                        }
                    },
                )
            )
        except Full:
            pass
        except Exception:
            pass
        raise


def _monitor_process(job_id: str, process: Process) -> None:
    """Wait for a child process to exit and reconcile final status."""

    try:
        process.join()
    except Exception:
        return

    exit_code = process.exitcode
    with jobs_lock:
        record = jobs.get(job_id)
        if not record:
            return
        record.process = None
        record.pid = None
        record.updated_at = time.time()
        if record.status not in {"failed", "succeeded"}:
            if exit_code == 0:
                record.status = "succeeded"
                record.stage = "completed"
            else:
                record.status = "failed"
                record.stage = "completed"
                record.error = f"任务进程异常退出(code={exit_code})"


def _start_job(
    job_type: str,
    session_id: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    runner_kwargs: Optional[Dict[str, Any]] = None,
) -> JobRecord:
    if job_type not in JOB_DEFINITIONS:
        raise HTTPException(status_code=404, detail="未知任务类型")

    _prune_jobs()
    with jobs_lock:
        for record in jobs.values():
            if (
                record.session_id == session_id
                and record.job_type == job_type
                and record.status in {"queued", "running"}
            ):
                raise HTTPException(status_code=409, detail=f"已有{_job_label(job_type)}任务正在运行")
        job_id = uuid4().hex
        record = JobRecord(job_id=job_id, session_id=session_id, job_type=job_type)
        record.stage = "queued"
        record.message = "任务已加入队列"
        record.created_at = record.updated_at = time.time()
        if metadata:
            try:
                record.metadata.update(dict(metadata))
            except Exception:
                record.metadata.update({})
        jobs[job_id] = record

    try:
        base_ctrl = os.path.join(tempfile.gettempdir(), "pqm_backend", "jobs")
        os.makedirs(base_ctrl, exist_ok=True)
        control_dir = os.path.join(base_ctrl, job_id)
        os.makedirs(control_dir, exist_ok=True)
        with jobs_lock:
            jobs[job_id].metadata["control_dir"] = control_dir
            jobs[job_id].metadata["job_type"] = job_type
        
        _ctrl_key = f"PQM_JOB_CONTROL_DIR_{job_id}"
        _prev_env = os.environ.get(_ctrl_key)
        os.environ[_ctrl_key] = control_dir
        try:
            process = Process(
                target=_job_process_entry,
                args=(job_id, session_id, _update_queue, job_type, dict(runner_kwargs or {})),
                daemon=False,
            )
            process.start()
        finally:
            if _prev_env is None:
                os.environ.pop(_ctrl_key, None)
            else:
                os.environ[_ctrl_key] = _prev_env
    except Exception as error:
        with jobs_lock:
            jobs.pop(job_id, None)
        raise HTTPException(status_code=500, detail=f"后台任务启动失败: {error}")

    with jobs_lock:
        record = jobs[job_id]
        record.process = process
        record.pid = process.pid
        record.status = "running"
        record.stage = "initializing"
        record.message = "任务已启动"
        record.updated_at = time.time()

    try:
        _update_queue.put_nowait((job_id, {"pid": process.pid}))
    except Full:
        pass
    monitor_thread = Thread(target=_monitor_process, args=(job_id, process), daemon=True)
    monitor_thread.start()
    return record


def _get_job_status(job_id: str, expected_type: Optional[str] = None) -> EnterpriseJobStatus:
    _prune_jobs()
    with jobs_lock:
        record = jobs.get(job_id)
    if not record or (expected_type and record.job_type != expected_type):
        raise HTTPException(status_code=404, detail="未找到任务")
    return _record_to_status(record)


def _list_jobs(job_type: str, session_id: Optional[str] = None) -> List[EnterpriseJobStatus]:
    _prune_jobs()
    with jobs_lock:
        filtered = [
            _record_to_status(record)
            for record in jobs.values()
            if record.job_type == job_type
            and (session_id is None or record.session_id == session_id)
        ]
    filtered.sort(key=lambda status: status.created_at, reverse=True)
    return filtered


def _pause_resume_stop_job(job_id: str, action: str, job_type: str) -> EnterpriseJobStatus:
    with jobs_lock:
        record = jobs.get(job_id)
        if not record or record.job_type != job_type:
            raise HTTPException(status_code=404, detail="未找到任务")
        control_dir = record.metadata.get("control_dir")
        if not control_dir:
            raise HTTPException(status_code=400, detail="任务不支持此操作")
        pause_flag = os.path.join(control_dir, "pause")
        stop_flag = os.path.join(control_dir, "stop")
        if action == "pause":
            try:
                open(pause_flag, "a").close()
            except Exception as error:
                raise HTTPException(status_code=500, detail=f"设置暂停失败: {error}")
            record.status = "paused"
            record.stage = "paused"
        elif action == "resume":
            try:
                if os.path.isfile(pause_flag):
                    os.remove(pause_flag)
            except Exception as error:
                raise HTTPException(status_code=500, detail=f"取消暂停失败: {error}")
            record.status = "running"
            record.stage = record.stage or "running"
        elif action == "stop":
            try:
                open(stop_flag, "a").close()
                if os.path.isfile(pause_flag):
                    os.remove(pause_flag)
            except Exception as error:
                raise HTTPException(status_code=500, detail=f"请求停止失败: {error}")
            record.status = "stopping"
            record.stage = "stopping"
        else:
            raise HTTPException(status_code=400, detail="未知操作")
        record.updated_at = time.time()
        return _record_to_status(record)

# Base directories
BASE_DIR = str(CONFIG["directories"]["uploads"])
os.makedirs(BASE_DIR, exist_ok=True)

def get_session_dirs(session_id: str) -> Dict[str, str]:
    """Get session directories for a user"""
    session_root = os.path.join(BASE_DIR, session_id, "parameters")
    return {
        "reference": os.path.join(session_root, "reference"),
        "target": os.path.join(session_root, "target"),
        "graph": os.path.join(session_root, "graph"),
    }

def ensure_session_dirs(session_id: str):
    """Ensure session directories exist"""
    dirs = get_session_dirs(session_id)
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)


def _apqp_stage_layout(session_id: str) -> Dict[str, Dict[str, str]]:
    """Return upload and parsed directories for each APQP stage for a session."""

    uploads_root = str(CONFIG["directories"]["uploads"])
    generated_root = str(CONFIG["directories"]["generated_files"])
    stage_layout: Dict[str, Dict[str, str]] = {}
    for stage_name in STAGE_ORDER:
        slug = STAGE_SLUG_MAP.get(stage_name, stage_name)
        upload_dir = os.path.join(uploads_root, session_id, "APQP_one_click_check", slug)
        parsed_dir = os.path.join(generated_root, session_id, "APQP_one_click_check", slug)
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(parsed_dir, exist_ok=True)
        stage_layout[stage_name] = {
            "slug": slug,
            "upload_dir": upload_dir,
            "parsed_dir": parsed_dir,
        }
    return stage_layout


def _normalize_apqp_stages(
    layout: Dict[str, Dict[str, str]], requested: Optional[List[str]]
) -> List[str]:
    """Normalize requested stage identifiers to canonical stage names."""

    if not requested:
        return list(layout.keys())

    name_map = {name.lower(): name for name in layout}
    slug_map = {info["slug"].lower(): name for name, info in layout.items()}
    result: List[str] = []
    for item in requested:
        token = str(item or "").strip().lower()
        if not token:
            continue
        stage_name = name_map.get(token) or slug_map.get(token)
        if not stage_name:
            raise HTTPException(status_code=400, detail=f"未知阶段: {item}")
        if stage_name not in result:
            result.append(stage_name)
    if not result:
        raise HTTPException(status_code=400, detail="未提供有效的阶段名称")
    return result


def _collect_parsed_txt_files(folder: str) -> List[str]:
    """Return sorted .txt files from a parsed folder."""

    if not folder or not os.path.isdir(folder):
        return []
    paths = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, name)) and name.lower().endswith(".txt")
    ]
    return sorted(paths)


def _parse_apqp_stages(
    layout: Dict[str, Dict[str, str]],
    stages: List[str],
    publish: Optional[Callable[[Dict[str, Any]], None]] = None,
    check_control: Optional[Callable[[], Dict[str, bool]]] = None,
) -> Dict[str, Any]:
    """Parse uploaded APQP files into text, optionally emitting progress updates."""

    summary: Dict[str, Any] = {
        "stage_order": stages,
        "stages": {},
        "total_created": 0,
    }
    total_stages = max(len(stages), 1)

    def _emit(update: Dict[str, Any], *, log_message: Optional[str] = None, level: str = "info") -> None:
        if not publish:
            return
        payload = dict(update or {})
        if log_message:
            payload["log"] = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "level": level,
                "message": log_message,
            }
        publish(payload)

    for idx, stage_name in enumerate(stages):
        if check_control:
            status = check_control() or {}
            if status.get("stopped"):
                raise RuntimeError("解析已被停止")

        info = layout[stage_name]
        upload_dir = info["upload_dir"]
        parsed_dir = info["parsed_dir"]
        logger = _ParseLogger()
        stage_report: Dict[str, Any] = {
            "slug": info["slug"],
            "upload_dir": upload_dir,
            "parsed_dir": parsed_dir,
            "files_found": 0,
            "pdf_created": 0,
            "word_ppt_created": 0,
            "excel_created": 0,
            "text_created": 0,
            "total_created": 0,
        }

        step_weight = 1.0 / max(total_stages, 1)
        stage_base_progress = idx * step_weight
        _emit(
            {
                "status": "running",
                "stage": stage_name,
                "progress": stage_base_progress,
                "message": f"开始解析 {stage_name}",
            },
            log_message=f"开始解析 {stage_name}",
        )

        try:
            files_found = [
                name
                for name in os.listdir(upload_dir)
                if os.path.isfile(os.path.join(upload_dir, name)) and name != ".gitkeep"
            ]
            stage_report["files_found"] = len(files_found)

            removed_txts = cleanup_orphan_txts(upload_dir, parsed_dir, logger)
            if removed_txts:
                logger.info(f"已清理无关文本 {removed_txts} 个")
            if not files_found:
                logger.info("当前阶段没有上传文件，跳过解析。")
            else:
                substeps = 4
                progress_increment = step_weight / max(substeps, 1)
                subprogress = 0.0

                pdf_created = process_pdf_folder(
                    upload_dir, parsed_dir, logger, annotate_sources=True
                )
                stage_report["pdf_created"] = len(pdf_created)
                subprogress += progress_increment
                _emit(
                    {
                        "status": "running",
                        "stage": stage_name,
                        "progress": stage_base_progress + subprogress,
                        "message": f"已处理PDF，共{len(pdf_created)}个",
                    },
                    log_message=f"已处理PDF，共{len(pdf_created)}个",
                )

                office_created = process_word_ppt_folder(
                    upload_dir, parsed_dir, logger, annotate_sources=True
                )
                stage_report["word_ppt_created"] = len(office_created)
                subprogress += progress_increment
                _emit(
                    {
                        "status": "running",
                        "stage": stage_name,
                        "progress": stage_base_progress + subprogress,
                        "message": f"已处理Word/PPT，共{len(office_created)}个",
                    },
                    log_message=f"已处理Word/PPT，共{len(office_created)}个",
                )

                excel_created = process_excel_folder(
                    upload_dir, parsed_dir, logger, annotate_sources=True
                )
                stage_report["excel_created"] = len(excel_created)
                subprogress += progress_increment
                _emit(
                    {
                        "status": "running",
                        "stage": stage_name,
                        "progress": stage_base_progress + subprogress,
                        "message": f"已处理Excel，共{len(excel_created)}个",
                    },
                    log_message=f"已处理Excel，共{len(excel_created)}个",
                )

                text_created = process_textlike_folder(upload_dir, parsed_dir, logger)
                stage_report["text_created"] = len(text_created)
                subprogress += progress_increment
                _emit(
                    {
                        "status": "running",
                        "stage": stage_name,
                        "progress": stage_base_progress + subprogress,
                        "message": f"已处理文本类文件，共{len(text_created)}个",
                    },
                    log_message=f"已处理文本类文件，共{len(text_created)}个",
                )

                total_created = (
                    len(pdf_created)
                    + len(office_created)
                    + len(excel_created)
                    + len(text_created)
                )
                stage_report["total_created"] = total_created
                summary["total_created"] += total_created
        except Exception as error:
            logger.error(f"解析阶段失败: {error}")
            stage_report["error"] = str(error)

            _emit(
                {
                    "status": "running",
                    "stage": stage_name,
                    "message": f"解析阶段失败: {error}",
                },
                log_message=f"解析阶段失败: {error}",
                level="error",
            )

        stage_report["messages"] = logger.messages
        summary["stages"][stage_name] = stage_report

        _emit(
            {
                "status": "running",
                "stage": stage_name,
                "progress": min(1.0, stage_base_progress + step_weight),
                "message": f"完成 {stage_name} 解析",
            },
            log_message=f"完成 {stage_name} 解析",
        )

    _emit({"status": "succeeded", "stage": "completed", "progress": 1.0}, log_message="解析任务完成")

    return summary


def _load_text_preview(file_path: str, head_chars: int = 3200, tail_chars: int = 2000) -> str:
    """Return a head/tail preview from a text file to control token usage."""

    try:
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    if head_chars <= 0 and tail_chars <= 0:
        return text
    head = text[: max(head_chars, 0)]
    tail = text[-max(tail_chars, 0) :] if tail_chars > 0 else ""
    if tail and head:
        return f"【开头】\n{head}\n\n【结尾】\n{tail}"
    return head or tail


def _apqp_candidate_definitions(stage_name: str) -> List[Dict[str, str]]:
    """Build canonical deliverable definitions for a stage."""

    requirements = STAGE_REQUIREMENTS.get(stage_name) or tuple()
    result: List[Dict[str, str]] = []
    for item in requirements:
        description = CANONICAL_APQP_DESCRIPTORS.get(item) or descriptor_for(item)
        result.append({"name": item, "description": description})
    return result


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Best-effort JSON decoding that tolerates fenced blocks."""

    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return {}


@dataclass
class _LLMProvider:
    label: str
    engine: str  # "ollama" or "modelscope"
    model: str
    client_factory: Callable[[], object]
    max_retries: int = 1
    supports_parallel: bool = True


def _prepare_apqp_llm_providers(turbo_mode: bool) -> Tuple[List[_LLMProvider], List[_LLMProvider], List[str]]:
    """Build provider lists for APQP classification.

    Returns:
        fast_providers: providers eligible for parallel/turbo usage (non-local).
        serial_chain: ordered providers for serial fallback.
        warnings: any initialization warnings.
    """

    warnings: List[str] = []
    llm_settings = CONFIG.get("llm", {})

    modelscope_api_key = os.getenv("MODELSCOPE_API_KEY") or llm_settings.get("modelscope_api_key")
    modelscope_base = llm_settings.get("modelscope_base_url") or "https://api-inference.modelscope.cn/v1"
    ms_models: List[Tuple[str, str]] = [
        ("ModelScope DeepSeek-V3.2-Exp", "deepseek-ai/DeepSeek-V3.2-Exp"),
        ("ModelScope DeepSeek-V3.1", "deepseek-ai/DeepSeek-V3.1"),
        ("ModelScope Qwen3-235B", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
    ]

    def _mk_modelscope_provider(label: str, model_id: str) -> _LLMProvider:
        return _LLMProvider(
            label=label,
            engine="modelscope",
            model=model_id,
            client_factory=lambda model=model_id: ModelScopeClient(
                api_key=modelscope_api_key or "", model=model, base_url=modelscope_base, timeout=900.0
            ),
            max_retries=2,
            supports_parallel=True,
        )

    fast_providers: List[_LLMProvider] = []
    serial_chain: List[_LLMProvider] = []

    local_model = llm_settings.get("ollama_model") or "gpt-oss:latest"
    local_provider: Optional[_LLMProvider] = None
    try:
        host = resolve_ollama_host("ollama_9")
        local_provider = _LLMProvider(
            label="本地 gpt-oss",
            engine="ollama",
            model=local_model,
            client_factory=lambda host=host: OllamaClient(host=host),
            max_retries=1,
            supports_parallel=False,
        )
    except Exception as error:
        warnings.append(f"初始化本地 Ollama 失败：{error}")

    cloud_host = llm_settings.get("ollama_cloud_host")
    cloud_api_key = llm_settings.get("ollama_cloud_api_key")
    cloud_provider: Optional[_LLMProvider] = None
    if cloud_host and cloud_api_key:
        try:
            def _make_cloud() -> OllamaClient:
                return OllamaClient(host=cloud_host, headers={"Authorization": f"Bearer {cloud_api_key}"})

            cloud_provider = _LLMProvider(
                label="云端 gpt-oss-20b",
                engine="ollama",
                model="gpt-oss:20b-cloud",
                client_factory=_make_cloud,
                max_retries=2,
                supports_parallel=True,
            )
        except Exception as error:
            warnings.append(f"初始化云端 Ollama 失败：{error}")

    modelscope_providers: List[_LLMProvider] = []
    if modelscope_api_key:
        for label, model_id in ms_models:
            try:
                provider = _mk_modelscope_provider(label, model_id)
                # Validate factory lazily by instantiating once
                provider.client_factory()
                modelscope_providers.append(provider)
            except Exception as error:
                warnings.append(f"{label} 初始化失败：{error}")
    elif turbo_mode:
        warnings.append("未配置 ModelScope API Key，无法使用 ModelScope 高性能通道")

    if turbo_mode:
        fast_providers.extend(modelscope_providers)
        if cloud_provider:
            fast_providers.append(cloud_provider)
        serial_chain.extend(fast_providers)
        if local_provider:
            serial_chain.append(local_provider)
    else:
        if local_provider:
            serial_chain.append(local_provider)
        serial_chain.extend(modelscope_providers)

    return fast_providers, serial_chain, warnings


def _invoke_with_providers(providers: List[_LLMProvider], messages: List[Dict[str, str]]) -> Tuple[str, str, str]:
    """Try providers in order; returns (raw_content, provider_label, model)."""

    if not providers:
        raise RuntimeError("无可用模型通道")

    attempts: List[str] = []
    for provider in providers:
        tries = max(1, provider.max_retries)
        for attempt in range(tries):
            try:
                client = provider.client_factory()
                if provider.engine == "ollama":
                    response = client.chat(
                        model=provider.model,
                        messages=messages,
                        options={"num_ctx": 40001, "temperature": 0, "top_p": 0},
                    )
                    content = ""
                    if isinstance(response, dict):
                        content = (response.get("message") or {}).get("content") or ""
                    if content:
                        return content, provider.label, provider.model
                    attempts.append(f"{provider.label} 响应为空")
                elif provider.engine == "modelscope":
                    response = provider.client_factory().chat(
                        model=provider.model,
                        messages=messages,
                        options={"num_ctx": 40001},
                    )
                    content = ""
                    if isinstance(response, dict):
                        content = (response.get("message") or {}).get("content") or ""
                    if content:
                        return content, provider.label, provider.model
                    attempts.append(f"{provider.label} 响应为空")
                else:
                    attempts.append(f"未知引擎 {provider.engine}")
            except Exception as error:
                attempts.append(f"{provider.label}# {attempt + 1} 失败: {error}")
                time.sleep(1.0)

    raise RuntimeError("；".join(attempts) or "无法完成请求")


def _classify_document(
    *,
    invoker: Callable[[List[Dict[str, str]]], Tuple[str, str, str]],
    stage_name: str,
    file_path: str,
    candidates: List[Dict[str, str]],
    head_chars: int,
    tail_chars: int,
) -> Dict[str, Any]:
    """Call LLM to classify a parsed document into canonical APQP types."""

    preview = _load_text_preview(file_path, head_chars=head_chars, tail_chars=tail_chars)
    file_name = os.path.basename(file_path)
    payload = {
        "file_name": file_name,
        "path": file_path,
        "preview_length": len(preview),
        "primary_type": None,
        "additional_types": [],
        "confidence": 0.0,
        "rationale": "",
        "raw_response": None,
        "status": "pending",
    }

    if not preview.strip():
        payload.update(
            {
                "status": "error",
                "error": "文本为空或无法读取",
            }
        )
        return payload

    candidates_text = "\n".join(
        f"{idx+1}. {item['name']}：{item['description']}" for idx, item in enumerate(candidates)
    )
    prompt = f"""
你是一名 APQP 交付物分类助手，请基于文档内容判断文件最符合的交付物类型。
阶段：{stage_name}
候选交付物列表（仅可从中选择或回答 none）：
{candidates_text}

请阅读以下文件内容片段（已截取开头和结尾，以避免过长）：
{preview}

输出严格的 JSON（不含多余文字），字段要求：
{{
  "primary_type": "<从候选列表选择的名称，若无法匹配请填 none>",
  "additional_types": ["<可选的额外交付物名称，用于表示1个文件覆盖多个交付物，可为空列表>"],
  "confidence": <0到1之间的小数，表示匹配置信度>,
  "rationale": "简要中文理由，说明为何匹配或为何选择 none"
}}

规则：
- primary_type 必须是候选列表中的名称或 "none"，不得编造。
- 如果文件同时覆盖多个交付物，可在 additional_types 中列出额外交付物名称（必须来自候选列表且不重复）。
- 若文本与候选内容明显无关或信息太少，请输出 primary_type="none"，并给出低置信度原因。
- 仅依据文本内容做判断，避免依赖文件名。
"""

    try:
        messages = [
            {"role": "system", "content": "你是严谨的APQP文件分类专家，回答请使用简体中文。"},
            {"role": "user", "content": prompt},
        ]
        raw_content, provider_label, model_name = invoker(messages)
        payload["raw_response"] = raw_content
        payload["provider"] = provider_label
        payload["model"] = model_name
        data = _extract_json_object(raw_content)
    except Exception as error:
        payload.update({"status": "error", "error": str(error)})
        return payload

    primary_type = str(data.get("primary_type") or "").strip()
    additional = data.get("additional_types") or []
    if isinstance(additional, str):
        additional = [additional]
    additional_types = [str(item).strip() for item in additional if str(item).strip()]
    rationale = str(data.get("rationale") or "").strip()
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))

    payload.update(
        {
            "primary_type": primary_type if primary_type else None,
            "additional_types": additional_types,
            "confidence": confidence,
            "rationale": rationale,
            "status": "success",
        }
    )
    return payload


def _clear_directory_contents(path: str) -> int:
    """Remove all files/directories inside `path`, returning number of items removed."""

    if not path:
        return 0
    removed = 0
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        return removed
    for name in os.listdir(path):
        item_path = os.path.join(path, name)
        try:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
            removed += 1
        except Exception:
            continue
    return removed


class _ParseLogger:
    """Collect progress messages from parsing helpers."""

    def __init__(self) -> None:
        self.messages: List[Dict[str, str]] = []

    def _add(self, level: str, message: str) -> None:
        self.messages.append({"level": level, "text": str(message)})

    def write(self, message: str) -> None:
        self._add("info", message)

    def info(self, message: str) -> None:
        self._add("info", message)

    def warning(self, message: str) -> None:
        self._add("warning", message)

    def error(self, message: str) -> None:
        self._add("error", message)

    def success(self, message: str) -> None:
        self._add("success", message)


@app.get("/")
async def root():
    return {"message": "PQM AI Backend is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/upload-file")
async def upload_file(
    session_id: str,
    file_type: str,  # "reference", "target", or "graph"
    file: UploadFile = File(...)
):
    """Upload a file to the specified session directory"""
    try:
        ensure_session_dirs(session_id)
        dirs = get_session_dirs(session_id)
        
        if file_type not in dirs:
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file_type}")
        
        target_dir = dirs[file_type]
        file_path = os.path.join(target_dir, file.filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "status": "success",
            "message": f"File {file.filename} uploaded successfully",
            "file_path": file_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.delete("/delete-file")
async def delete_file(operation: FileOperation):
    """Delete a specific file"""
    try:
        if not os.path.exists(operation.file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        os.remove(operation.file_path)
        return {
            "status": "success",
            "message": f"File deleted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@app.get("/list-files/{session_id}")
async def list_files(session_id: str, file_type: str = None):
    """List files in session directories"""
    try:
        ensure_session_dirs(session_id)
        dirs = get_session_dirs(session_id)
        
        if file_type and file_type not in dirs:
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file_type}")
        
        result = {}
        
        for dir_name, dir_path in dirs.items():
            if file_type and dir_name != file_type:
                continue
                
            files = []
            if os.path.exists(dir_path):
                for filename in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, filename)
                    if os.path.isfile(file_path):
                        stat = os.stat(file_path)
                        files.append(FileInfo(
                            name=filename,
                            size=stat.st_size,
                            modified_time=str(stat.st_mtime)
                        ))
            
            result[dir_name] = files
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List files failed: {str(e)}")


@app.post("/apqp-one-click/upload")
async def apqp_upload_file(
    session_id: str = Form(...),
    stage: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload an APQP file to the specified stage directory."""

    layout = _apqp_stage_layout(session_id)
    stages = _normalize_apqp_stages(layout, [stage])
    stage_name = stages[0]
    target_dir = layout[stage_name]["upload_dir"]
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, file.filename)

    try:
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"上传失败: {error}")

    return {
        "status": "success",
        "stage": stage_name,
        "path": target_path,
        "name": file.filename,
    }


@app.get("/apqp-one-click/files/{session_id}")
async def apqp_list_files(session_id: str, stage: Optional[str] = None):
    """List APQP uploaded files (optionally filtered by stage)."""

    layout = _apqp_stage_layout(session_id)
    if stage:
        stages = _normalize_apqp_stages(layout, [stage])
    else:
        stages = list(layout.keys())

    result: Dict[str, List[Dict[str, object]]] = {}
    for stage_name in stages:
        info = layout[stage_name]
        entries: List[Dict[str, object]] = []
        try:
            for name in os.listdir(info["upload_dir"]):
                file_path = os.path.join(info["upload_dir"], name)
                if os.path.isfile(file_path) and name != ".gitkeep":
                    stat = os.stat(file_path)
                    entries.append(
                        {
                            "name": name,
                            "size": stat.st_size,
                            "modified": stat.st_mtime,
                            "path": file_path,
                        }
                    )
        except Exception:
            entries = []
        result[stage_name] = sorted(entries, key=lambda item: item["name"].lower())

    return {
        "status": "success",
        "stage_order": stages,
        "files": result,
    }

@app.post("/clear-files")
async def clear_files(request: ClearFilesRequest):
    """Clear all files for a session"""
    try:
        dirs = get_session_dirs(request.session_id)
        deleted_count = 0
        
        for dir_path in dirs.values():
            if os.path.exists(dir_path):
                for filename in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        deleted_count += 1
        
        return {
            "status": "success",
            "message": f"Cleared {deleted_count} files",
            "deleted_count": deleted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear files failed: {str(e)}")


@app.post("/apqp-one-click/parse")
async def apqp_parse(request: ApqpParseRequest):
    """Parse APQP uploads into text using MinerU/Unstructured pipeline."""

    layout = _apqp_stage_layout(request.session_id)
    stages = _normalize_apqp_stages(layout, request.stages)
    summary = _parse_apqp_stages(layout, stages)

    return {
        "status": "success",
        "summary": summary,
    }


def run_apqp_one_click_parse_job(
    session_id: str,
    publish: Callable[[Dict[str, Any]], None],
    *,
    check_control: Optional[Callable[[], Dict[str, bool]]] = None,
    stages: Optional[List[str]] = None,
) -> None:
    """Background runner for APQP一键解析任务。"""

    layout = _apqp_stage_layout(session_id)
    stage_list = _normalize_apqp_stages(layout, stages)

    def _publish(update: Dict[str, Any]) -> None:
        payload = dict(update or {})
        payload.setdefault("message", "")
        publish(payload)

    _publish({"status": "running", "stage": "init", "message": "开始准备解析任务"})
    try:
        _parse_apqp_stages(layout, stage_list, publish=_publish, check_control=check_control)
    except Exception as error:
        _publish(
            {
                "status": "failed",
                "stage": "completed",
                "error": str(error),
                "message": f"解析失败: {error}",
            }
        )
        raise


# Bind runner after definition to avoid forward reference issues
JOB_DEFINITIONS["apqp_one_click_parse"]["runner"] = run_apqp_one_click_parse_job



@app.post("/apqp-one-click/classify")
async def apqp_classify(request: ApqpClassifyRequest):
    """Classify parsed APQP documents via LLM to assess completeness."""

    layout = _apqp_stage_layout(request.session_id)
    stages = _normalize_apqp_stages(layout, request.stages)
    summary: Dict[str, Any] = {"stage_order": stages, "stages": {}, "turbo_mode": bool(request.turbo_mode)}
    fast_providers, serial_chain, provider_warnings = _prepare_apqp_llm_providers(bool(request.turbo_mode))
    summary_warnings: List[str] = []
    summary_warnings.extend(provider_warnings)

    if not serial_chain and not fast_providers:
        raise HTTPException(status_code=500, detail="未找到可用的模型通道，请检查模型配置或网络。")

    generated_root = Path(CONFIG["directories"]["generated_files"])
    summary_dir = generated_root / request.session_id / "APQP_one_click_check"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "classification_summary.json"

    serial_invoker = lambda messages: _invoke_with_providers(serial_chain or fast_providers, messages)
    fast_invoker = lambda messages: _invoke_with_providers(fast_providers, messages)
    parallel_enabled = bool(request.turbo_mode and fast_providers)

    for stage_name in stages:
        info = layout[stage_name]
        parsed_dir = info["parsed_dir"]
        upload_dir = info["upload_dir"]
        candidates = _apqp_candidate_definitions(stage_name)
        txt_files = _collect_parsed_txt_files(parsed_dir)

        requirement_state: Dict[str, Dict[str, Any]] = {
            name: {"status": "missing", "sources": [], "confidence": 0.0}
            for name in (STAGE_REQUIREMENTS.get(stage_name) or tuple())
        }

        documents: List[Dict[str, Any]] = []
        pending_retry: List[Tuple[str, Dict[str, Any]]] = []

        def _classify_path(path: str, *, use_fast: bool) -> Dict[str, Any]:
            invoker = fast_invoker if use_fast else serial_invoker
            return _classify_document(
                invoker=invoker,
                stage_name=stage_name,
                file_path=path,
                candidates=candidates,
                head_chars=max(0, int(request.head_chars)),
                tail_chars=max(0, int(request.tail_chars)),
            )

        if parallel_enabled and len(txt_files) > 1:
            max_workers = min(4, len(txt_files))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(_classify_path, path, use_fast=True): path for path in txt_files}
                for future in as_completed(future_map):
                    path = future_map[future]
                    try:
                        result = future.result()
                    except Exception as error:
                        result = {
                            "status": "error",
                            "error": str(error),
                            "file_name": os.path.basename(path),
                            "path": path,
                        }
                    if result.get("status") == "success":
                        documents.append(result)
                    else:
                        pending_retry.append((path, result))
        else:
            for file_path in txt_files:
                result = _classify_path(file_path, use_fast=False)
                if result.get("status") == "success":
                    documents.append(result)
                else:
                    pending_retry.append((file_path, result))

        if pending_retry and serial_chain:
            for path, first_result in pending_retry:
                fallback_result = _classify_path(path, use_fast=False)
                target_result = fallback_result
                if fallback_result.get("status") != "success":
                    if first_result.get("error"):
                        fallback_result.setdefault("previous_errors", []).append(first_result.get("error"))
                    merged_error = "; ".join(
                        item
                        for item in [first_result.get("error"), fallback_result.get("error")]
                        if item
                    )
                    fallback_result["error"] = merged_error or fallback_result.get("error")
                documents.append(target_result)

        for result in documents:
            matched: List[str] = []
            suggested: List[str] = []
            file_path = result.get("path", "")
            if result.get("status") == "success":
                names = []
                primary = result.get("primary_type") or ""
                if primary and primary.lower() != "none":
                    names.append(primary)
                for extra in result.get("additional_types") or []:
                    if extra not in names:
                        names.append(extra)
                for name in names:
                    if name in requirement_state:
                        matched.append(name)
                    else:
                        suggested.append(name)
                for req in matched:
                    state = requirement_state[req]
                    state["status"] = "present"
                    state["confidence"] = max(state.get("confidence", 0.0), result.get("confidence") or 0.0)
                    state.setdefault("sources", []).append(result.get("file_name") or os.path.basename(file_path))
            result["matched_requirements"] = matched
            if suggested:
                result["suggested_types"] = suggested

        present_count = sum(1 for item in requirement_state.values() if item.get("status") == "present")
        missing_count = len(requirement_state) - present_count

        stage_summary = {
            "slug": info["slug"],
            "upload_dir": upload_dir,
            "parsed_dir": parsed_dir,
            "llm_mode": "turbo" if request.turbo_mode else "standard",
            "requirements": [
                {"name": name, **requirement_state[name]} for name in requirement_state
            ],
            "documents": documents,
            "stats": {
                "total_requirements": len(requirement_state),
                "present": present_count,
                "missing": max(missing_count, 0),
                "files_classified": len(documents),
                "parsed_files_found": len(txt_files),
            },
        }

        if not txt_files:
            stage_summary["warning"] = "未找到解析后的文本文件，请先执行解析。"
        if not candidates:
            extra = "当前阶段未配置应交付物列表。"
            if stage_summary.get("warning"):
                stage_summary["warning"] = f"{stage_summary['warning']} {extra}"
            else:
                stage_summary["warning"] = extra
        summary["stages"][stage_name] = stage_summary

    if summary_warnings:
        summary["warnings"] = summary_warnings

    try:
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["summary_file"] = str(summary_path)
    except Exception:
        summary["summary_file"] = None

    return {"status": "success", "summary": summary}


@app.post("/apqp-one-click/clear")
async def apqp_clear(request: ApqpClearRequest):
    """Clear APQP uploads and/or parsed outputs for a session."""

    layout = _apqp_stage_layout(request.session_id)
    target = (request.target or "all").lower()
    if target not in {"uploads", "parsed", "all"}:
        raise HTTPException(status_code=400, detail="target 必须是 uploads、parsed 或 all")

    details: Dict[str, Dict[str, int]] = {}
    total_deleted = 0
    for stage_name, info in layout.items():
        uploads_deleted = 0
        parsed_deleted = 0
        if target in {"uploads", "all"}:
            uploads_deleted = _clear_directory_contents(info["upload_dir"])
            total_deleted += uploads_deleted
        if target in {"parsed", "all"}:
            parsed_deleted = _clear_directory_contents(info["parsed_dir"])
            total_deleted += parsed_deleted
        details[stage_name] = {
            "uploads_deleted": uploads_deleted,
            "parsed_deleted": parsed_deleted,
        }

    return {
        "status": "success",
        "target": target,
        "deleted": total_deleted,
        "details": details,
        "stage_order": list(layout.keys()),
    }


@app.get("/file-exists/{session_id}")
async def file_exists(session_id: str, file_path: str):
    """Check if a file exists"""
    try:
        exists = os.path.exists(file_path)
        return {
            "exists": exists,
            "file_path": file_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Check file failed: {str(e)}")


@app.post("/apqp-one-click/jobs", response_model=EnterpriseJobStatus)
async def start_apqp_one_click_job(request: ApqpParseJobRequest):
    """Start a background APQP parse job."""

    record = _start_job(
        "apqp_one_click_parse",
        request.session_id,
        runner_kwargs={"stages": request.stages},
    )
    return _record_to_status(record)


@app.get("/apqp-one-click/jobs/{job_id}", response_model=EnterpriseJobStatus)
async def get_apqp_one_click_job(job_id: str):
    _prune_jobs()
    with jobs_lock:
        record = jobs.get(job_id)
        if not record or record.job_type != "apqp_one_click_parse":
            raise HTTPException(status_code=404, detail="未找到解析任务")
        return _record_to_status(record)


@app.get("/apqp-one-click/jobs", response_model=List[EnterpriseJobStatus])
async def list_apqp_one_click_jobs(session_id: Optional[str] = None):
    """List APQP one-click parsing jobs for a session (or all sessions)."""

    _prune_jobs()
    with jobs_lock:
        records = [
            _record_to_status(record)
            for record in jobs.values()
            if record.job_type == "apqp_one_click_parse"
            and (session_id is None or record.session_id == session_id)
        ]
    records.sort(key=lambda status: status.created_at, reverse=True)
    return records


@app.post("/enterprise-standard/jobs", response_model=EnterpriseJobStatus)
async def start_enterprise_standard_job(request: EnterpriseJobRequest):
    record = _start_job("enterprise_standard", request.session_id)
    return _record_to_status(record)


@app.get("/enterprise-standard/jobs/{job_id}", response_model=EnterpriseJobStatus)
async def get_enterprise_standard_job(job_id: str):
    _prune_jobs()
    with jobs_lock:
        record = jobs.get(job_id)
    if not record or record.job_type != "enterprise_standard":
        raise HTTPException(status_code=404, detail="未找到任务")
    return _record_to_status(record)


@app.get("/enterprise-standard/jobs", response_model=List[EnterpriseJobStatus])
async def list_enterprise_standard_jobs(session_id: Optional[str] = None):
    _prune_jobs()
    with jobs_lock:
        records = [
            _record_to_status(record)
            for record in jobs.values()
            if record.job_type == "enterprise_standard"
            and (session_id is None or record.session_id == session_id)
        ]
    records.sort(key=lambda status: status.created_at, reverse=True)
    return records


@app.post("/special-symbols/jobs", response_model=EnterpriseJobStatus)
async def start_special_symbols_job(request: SpecialSymbolsJobRequest):
    turbo_mode = bool(request.turbo_mode)
    record = _start_job(
        "special_symbols",
        request.session_id,
        metadata={"turbo_mode": turbo_mode},
        runner_kwargs={"turbo_mode": turbo_mode},
    )
    return _record_to_status(record)


@app.get("/special-symbols/jobs/{job_id}", response_model=EnterpriseJobStatus)
async def get_special_symbols_job(job_id: str):
    _prune_jobs()
    with jobs_lock:
        record = jobs.get(job_id)
    if not record or record.job_type != "special_symbols":
        raise HTTPException(status_code=404, detail="未找到任务")
    return _record_to_status(record)


@app.get("/special-symbols/jobs", response_model=List[EnterpriseJobStatus])
async def list_special_symbols_jobs(session_id: Optional[str] = None):
    _prune_jobs()
    with jobs_lock:
        records = [
            _record_to_status(record)
            for record in jobs.values()
            if record.job_type == "special_symbols"
            and (session_id is None or record.session_id == session_id)
        ]
    records.sort(key=lambda status: status.created_at, reverse=True)
    return records


@app.post("/special-symbols/jobs/{job_id}/pause", response_model=EnterpriseJobStatus)
async def pause_special_symbols_job(job_id: str):
    with jobs_lock:
        record = jobs.get(job_id)
        if not record or record.job_type != "special_symbols":
            raise HTTPException(status_code=404, detail="未找到任务")
        if record.metadata.get("turbo_mode"):
            raise HTTPException(status_code=400, detail="高性能模式任务不支持暂停")
        control_dir = record.metadata.get("control_dir")
        if not control_dir:
            raise HTTPException(status_code=400, detail="任务不支持暂停")
        try:
            open(os.path.join(control_dir, "pause"), "a").close()
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"设置暂停失败: {error}")
        record.status = "paused"
        record.stage = "paused"
        record.updated_at = time.time()
        return _record_to_status(record)


@app.post("/special-symbols/jobs/{job_id}/resume", response_model=EnterpriseJobStatus)
async def resume_special_symbols_job(job_id: str):
    with jobs_lock:
        record = jobs.get(job_id)
        if not record or record.job_type != "special_symbols":
            raise HTTPException(status_code=404, detail="未找到任务")
        if record.metadata.get("turbo_mode"):
            raise HTTPException(status_code=400, detail="高性能模式任务不支持恢复")
        control_dir = record.metadata.get("control_dir")
        if not control_dir:
            raise HTTPException(status_code=400, detail="任务不支持恢复")
        try:
            pause_flag = os.path.join(control_dir, "pause")
            if os.path.isfile(pause_flag):
                os.remove(pause_flag)
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"取消暂停失败: {error}")
        record.status = "running"
        record.stage = record.stage or "running"
        record.updated_at = time.time()
        return _record_to_status(record)


@app.post("/special-symbols/jobs/{job_id}/stop", response_model=EnterpriseJobStatus)
async def stop_special_symbols_job(job_id: str):
    with jobs_lock:
        record = jobs.get(job_id)
        if not record or record.job_type != "special_symbols":
            raise HTTPException(status_code=404, detail="未找到任务")
        if record.metadata.get("turbo_mode"):
            raise HTTPException(status_code=400, detail="高性能模式任务不支持停止")
        control_dir = record.metadata.get("control_dir")
        if not control_dir:
            raise HTTPException(status_code=400, detail="任务不支持停止")
        try:
            open(os.path.join(control_dir, "stop"), "a").close()
            pause_flag = os.path.join(control_dir, "pause")
            if os.path.isfile(pause_flag):
                os.remove(pause_flag)
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"请求停止失败: {error}")
        record.status = "stopping"
        record.stage = "stopping"
        record.updated_at = time.time()
        return _record_to_status(record)


@app.post("/file-elements/jobs", response_model=EnterpriseJobStatus)
async def start_file_elements_job(request: FileElementsJobRequest):
    profile_payload = request.profile.dict()
    metadata = {
        "stage": request.profile.stage or "",
        "deliverable": request.profile.name,
    }
    record = _start_job(
        "file_elements",
        request.session_id,
        metadata=metadata,
        runner_kwargs={
            "profile_payload": profile_payload,
            "source_paths": request.source_paths or [],
        },
    )
    return _record_to_status(record)


@app.get("/file-elements/jobs/{job_id}", response_model=EnterpriseJobStatus)
async def get_file_elements_job(job_id: str):
    _prune_jobs()
    with jobs_lock:
        record = jobs.get(job_id)
    if not record or record.job_type != "file_elements":
        raise HTTPException(status_code=404, detail="未找到任务")
    return _record_to_status(record)


@app.get("/file-elements/jobs", response_model=List[EnterpriseJobStatus])
async def list_file_elements_jobs(session_id: Optional[str] = None):
    _prune_jobs()
    with jobs_lock:
        records = [
            _record_to_status(record)
            for record in jobs.values()
            if record.job_type == "file_elements"
            and (session_id is None or record.session_id == session_id)
        ]
    records.sort(key=lambda status: status.created_at, reverse=True)
    return records


@app.post("/parameters/jobs", response_model=EnterpriseJobStatus)
def start_parameters_job(request: EnterpriseJobRequest) -> EnterpriseJobStatus:
    record = _start_job("parameters", request.session_id)
    return _record_to_status(record)


@app.get("/parameters/jobs/{job_id}", response_model=EnterpriseJobStatus)
def get_parameters_job(job_id: str) -> EnterpriseJobStatus:
    return _get_job_status(job_id, expected_type="parameters")


@app.get("/parameters/jobs", response_model=List[EnterpriseJobStatus])
def list_parameters_jobs(session_id: Optional[str] = None) -> List[EnterpriseJobStatus]:
    return _list_jobs("parameters", session_id)


@app.post("/parameters/jobs/{job_id}/pause", response_model=EnterpriseJobStatus)
def pause_parameters_job(job_id: str) -> EnterpriseJobStatus:
    return _pause_resume_stop_job(job_id, action="pause", job_type="parameters")


@app.post("/parameters/jobs/{job_id}/resume", response_model=EnterpriseJobStatus)
def resume_parameters_job(job_id: str) -> EnterpriseJobStatus:
    return _pause_resume_stop_job(job_id, action="resume", job_type="parameters")


@app.post("/parameters/jobs/{job_id}/stop", response_model=EnterpriseJobStatus)
def stop_parameters_job(job_id: str) -> EnterpriseJobStatus:
    return _pause_resume_stop_job(job_id, action="stop", job_type="parameters")


@app.post("/file-completeness/jobs", response_model=EnterpriseJobStatus)
def start_file_completeness_job(request: EnterpriseJobRequest) -> EnterpriseJobStatus:
    record = _start_job("file_completeness", request.session_id)
    return _record_to_status(record)


@app.get("/file-completeness/jobs/{job_id}", response_model=EnterpriseJobStatus)
def get_file_completeness_job(job_id: str) -> EnterpriseJobStatus:
    return _get_job_status(job_id, expected_type="file_completeness")


@app.get("/file-completeness/jobs", response_model=List[EnterpriseJobStatus])
def list_file_completeness_jobs(session_id: Optional[str] = None) -> List[EnterpriseJobStatus]:
    return _list_jobs("file_completeness", session_id)


@app.post("/file-completeness/jobs/{job_id}/pause", response_model=EnterpriseJobStatus)
def pause_file_completeness_job(job_id: str) -> EnterpriseJobStatus:
    return _pause_resume_stop_job(job_id, action="pause", job_type="file_completeness")


@app.post("/file-completeness/jobs/{job_id}/resume", response_model=EnterpriseJobStatus)
def resume_file_completeness_job(job_id: str) -> EnterpriseJobStatus:
    return _pause_resume_stop_job(job_id, action="resume", job_type="file_completeness")


@app.post("/file-completeness/jobs/{job_id}/stop", response_model=EnterpriseJobStatus)
def stop_file_completeness_job(job_id: str) -> EnterpriseJobStatus:
    return _pause_resume_stop_job(job_id, action="stop", job_type="file_completeness")


@app.post("/history/jobs", response_model=EnterpriseJobStatus)
def start_history_job(request: EnterpriseJobRequest) -> EnterpriseJobStatus:
    record = _start_job("history", request.session_id)
    return _record_to_status(record)


@app.get("/history/jobs/{job_id}", response_model=EnterpriseJobStatus)
def get_history_job(job_id: str) -> EnterpriseJobStatus:
    return _get_job_status(job_id, expected_type="history")


@app.get("/history/jobs", response_model=List[EnterpriseJobStatus])
def list_history_jobs(session_id: Optional[str] = None) -> List[EnterpriseJobStatus]:
    return _list_jobs("history", session_id)


@app.post("/history/jobs/{job_id}/pause", response_model=EnterpriseJobStatus)
def pause_history_job(job_id: str) -> EnterpriseJobStatus:
    return _pause_resume_stop_job(job_id, action="pause", job_type="history")


@app.post("/history/jobs/{job_id}/resume", response_model=EnterpriseJobStatus)
def resume_history_job(job_id: str) -> EnterpriseJobStatus:
    return _pause_resume_stop_job(job_id, action="resume", job_type="history")


@app.post("/history/jobs/{job_id}/stop", response_model=EnterpriseJobStatus)
def stop_history_job(job_id: str) -> EnterpriseJobStatus:
    return _pause_resume_stop_job(job_id, action="stop", job_type="history")


@app.post("/enterprise-standard/jobs/{job_id}/pause", response_model=EnterpriseJobStatus)
async def pause_enterprise_standard_job(job_id: str):
    with jobs_lock:
        record = jobs.get(job_id)
        if not record or record.job_type != "enterprise_standard":
            raise HTTPException(status_code=404, detail="未找到任务")
        control_dir = record.metadata.get("control_dir")
        if not control_dir:
            raise HTTPException(status_code=400, detail="任务不支持暂停")
        try:
            open(os.path.join(control_dir, "pause"), "a").close()
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"设置暂停失败: {error}")
        record.status = "paused"
        record.stage = "paused"
        record.updated_at = time.time()
        return _record_to_status(record)


@app.post("/enterprise-standard/jobs/{job_id}/resume", response_model=EnterpriseJobStatus)
async def resume_enterprise_standard_job(job_id: str):
    with jobs_lock:
        record = jobs.get(job_id)
        if not record or record.job_type != "enterprise_standard":
            raise HTTPException(status_code=404, detail="未找到任务")
        control_dir = record.metadata.get("control_dir")
        if not control_dir:
            raise HTTPException(status_code=400, detail="任务不支持恢复")
        try:
            pause_flag = os.path.join(control_dir, "pause")
            if os.path.isfile(pause_flag):
                os.remove(pause_flag)
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"取消暂停失败: {error}")
        record.status = "running"
        record.stage = record.stage or "running"
        record.updated_at = time.time()
        return _record_to_status(record)


@app.post("/enterprise-standard/jobs/{job_id}/stop", response_model=EnterpriseJobStatus)
async def stop_enterprise_standard_job(job_id: str):
    with jobs_lock:
        record = jobs.get(job_id)
        if not record or record.job_type != "enterprise_standard":
            raise HTTPException(status_code=404, detail="未找到任务")
        control_dir = record.metadata.get("control_dir")
        if not control_dir:
            raise HTTPException(status_code=400, detail="任务不支持停止")
        try:
            open(os.path.join(control_dir, "stop"), "a").close()
            pause_flag = os.path.join(control_dir, "pause")
            if os.path.isfile(pause_flag):
                os.remove(pause_flag)
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"请求停止失败: {error}")
        record.status = "stopping"
        record.stage = "stopping"
        record.updated_at = time.time()
        return _record_to_status(record)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
