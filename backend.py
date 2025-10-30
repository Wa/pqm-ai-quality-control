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
from tabs.parameters.background import run_parameters_job
from tabs.special_symbols.background import run_special_symbols_job


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
        runner_def["runner"](session_id, publish, check_control=check_control)
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


def _start_job(job_type: str, session_id: str) -> JobRecord:
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
                args=(job_id, session_id, _update_queue, job_type),
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
async def start_special_symbols_job(request: EnterpriseJobRequest):
    record = _start_job("special_symbols", request.session_id)
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
    return _get_job_status(job_id)


@app.get("/parameters/jobs", response_model=List[EnterpriseJobStatus])
def list_parameters_jobs(session_id: Optional[str] = None) -> List[EnterpriseJobStatus]:
    return _list_jobs("parameters", session_id)


@app.post("/parameters/jobs/{job_id}/pause", response_model=EnterpriseJobStatus)
def pause_parameters_job(job_id: str) -> EnterpriseJobStatus:
    return _pause_resume_stop_job(job_id, action="pause")


@app.post("/parameters/jobs/{job_id}/resume", response_model=EnterpriseJobStatus)
def resume_parameters_job(job_id: str) -> EnterpriseJobStatus:
    return _pause_resume_stop_job(job_id, action="resume")


@app.post("/parameters/jobs/{job_id}/stop", response_model=EnterpriseJobStatus)
def stop_parameters_job(job_id: str) -> EnterpriseJobStatus:
    return _pause_resume_stop_job(job_id, action="stop")


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
