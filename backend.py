from dataclasses import dataclass, field
import multiprocessing
from multiprocessing import Process, Queue
from queue import Full
import os
import shutil
import time
import traceback
from datetime import datetime
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import CONFIG
from tabs.enterprise_standard.background import run_enterprise_standard_job
from tabs.file_completeness import (
    run_file_completeness_job,
    STAGE_ORDER,
    STAGE_SLUG_MAP,
)
from tabs.parameters.background import run_parameters_job
from tabs.history.background import run_history_job
from tabs.special_symbols.background import run_special_symbols_job
from tabs.shared.file_conversion import (
    process_excel_folder,
    process_pdf_folder,
    process_textlike_folder,
    process_word_ppt_folder,
)


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


class ApqpParseRequest(BaseModel):
    session_id: str
    stages: Optional[List[str]] = None


class ApqpClearRequest(BaseModel):
    session_id: str
    target: Optional[str] = "all"


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
    summary: Dict[str, Any] = {
        "stage_order": stages,
        "stages": {},
        "total_created": 0,
    }

    for stage_name in stages:
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

        try:
            files_found = [
                name
                for name in os.listdir(upload_dir)
                if os.path.isfile(os.path.join(upload_dir, name)) and name != ".gitkeep"
            ]
            stage_report["files_found"] = len(files_found)
            if not files_found:
                logger.info("当前阶段没有上传文件，跳过解析。")
            else:
                pdf_created = process_pdf_folder(
                    upload_dir, parsed_dir, logger, annotate_sources=True
                )
                stage_report["pdf_created"] = len(pdf_created)

                office_created = process_word_ppt_folder(
                    upload_dir, parsed_dir, logger, annotate_sources=True
                )
                stage_report["word_ppt_created"] = len(office_created)

                excel_created = process_excel_folder(
                    upload_dir, parsed_dir, logger, annotate_sources=True
                )
                stage_report["excel_created"] = len(excel_created)

                text_created = process_textlike_folder(upload_dir, parsed_dir, logger)
                stage_report["text_created"] = len(text_created)

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

        stage_report["messages"] = logger.messages
        summary["stages"][stage_name] = stage_report

    return {
        "status": "success",
        "summary": summary,
    }


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
