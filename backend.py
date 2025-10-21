from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import os
import shutil
import time
from threading import Lock
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tabs.enterprise_standard.background import run_enterprise_standard_job

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
    result_files: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    logs: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float
    updated_at: float


@dataclass
class JobRecord:
    job_id: str
    session_id: str
    status: str = "queued"
    stage: Optional[str] = None
    message: Optional[str] = None
    processed_chunks: int = 0
    total_chunks: int = 0
    result_files: List[str] = field(default_factory=list)
    error: Optional[str] = None
    logs: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    future: Optional[Future] = None


executor = ThreadPoolExecutor(max_workers=2)
jobs_lock = Lock()
jobs: Dict[str, JobRecord] = {}


def _record_to_status(record: JobRecord) -> EnterpriseJobStatus:
    return EnterpriseJobStatus(
        job_id=record.job_id,
        session_id=record.session_id,
        status=record.status,
        stage=record.stage,
        message=record.message,
        processed_chunks=record.processed_chunks,
        total_chunks=record.total_chunks,
        result_files=record.result_files,
        error=record.error,
        logs=record.logs,
        metadata=record.metadata,
        created_at=record.created_at,
        updated_at=record.updated_at,
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
        if "checkpoint" in update:
            record.metadata["checkpoint"] = update["checkpoint"]


def _prune_jobs(max_age_seconds: float = 6 * 3600) -> None:
    cutoff = time.time() - max_age_seconds
    with jobs_lock:
        expired = [job_id for job_id, record in jobs.items() if record.updated_at < cutoff and record.status in {"succeeded", "failed"}]
        for job_id in expired:
            jobs.pop(job_id, None)
# Base directories
BASE_DIR = ""  # Use root directory instead of "user_sessions"

def get_session_dirs(session_id: str) -> Dict[str, str]:
    """Get session directories for a user"""
    return {
        "cp": os.path.join(BASE_DIR, "CP_files", session_id),
        "target": os.path.join(BASE_DIR, "target_files", session_id), 
        "graph": os.path.join(BASE_DIR, "graph_files", session_id)
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
    file_type: str,  # "cp", "target", or "graph"
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
    _prune_jobs()
    with jobs_lock:
        for record in jobs.values():
            if record.session_id == request.session_id and record.status in {"queued", "running"}:
                raise HTTPException(status_code=409, detail="已有企业标准检查任务正在运行")
        job_id = uuid4().hex
        record = JobRecord(job_id=job_id, session_id=request.session_id)
        record.stage = "queued"
        record.message = "任务已加入队列"
        record.created_at = record.updated_at = time.time()
        jobs[job_id] = record

    def publish(update: Dict[str, Any]) -> None:
        _update_job(job_id, update)

    future = executor.submit(run_enterprise_standard_job, request.session_id, publish)
    with jobs_lock:
        record = jobs[job_id]
        record.future = future
        record.status = "running"
        record.stage = "initializing"
        record.message = "任务已启动"
        record.updated_at = time.time()

    def finalize(fut: Future) -> None:
        exc = fut.exception()
        if exc is not None:
            _update_job(job_id, {"status": "failed", "stage": "completed", "error": str(exc)})
        else:
            with jobs_lock:
                rec = jobs.get(job_id)
                if rec and rec.status not in {"succeeded", "failed"}:
                    rec.status = "succeeded"
                    rec.stage = "completed"
                    rec.updated_at = time.time()
        with jobs_lock:
            rec = jobs.get(job_id)
            if rec:
                rec.future = None

    future.add_done_callback(finalize)
    return _record_to_status(record)


@app.get("/enterprise-standard/jobs/{job_id}", response_model=EnterpriseJobStatus)
async def get_enterprise_standard_job(job_id: str):
    _prune_jobs()
    with jobs_lock:
        record = jobs.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="未找到任务")
    return _record_to_status(record)


@app.get("/enterprise-standard/jobs", response_model=List[EnterpriseJobStatus])
async def list_enterprise_standard_jobs(session_id: Optional[str] = None):
    _prune_jobs()
    with jobs_lock:
        records = [
            _record_to_status(record)
            for record in jobs.values()
            if session_id is None or record.session_id == session_id
        ]
    records.sort(key=lambda status: status.created_at, reverse=True)
    return records

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 