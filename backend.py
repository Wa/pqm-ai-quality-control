from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from typing import List, Dict
import json

app = FastAPI(title="PQM AI Backend", description="File operations backend for PQM AI Quality Control")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8888", "http://127.0.0.1:8888", "http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit default ports
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

# Base directories
BASE_DIR = "user_sessions"

def get_session_dirs(session_id: str) -> Dict[str, str]:
    """Get session directories for a user"""
    return {
        "cp": os.path.join(BASE_DIR, session_id, "CP_files"),
        "target": os.path.join(BASE_DIR, session_id, "target_files"), 
        "graph": os.path.join(BASE_DIR, session_id, "graph_files")
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
                            modified_time=stat.st_mtime
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 