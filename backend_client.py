import requests
import os
from typing import Dict, List, Optional
import streamlit as st

class BackendClient:
    """Client for communicating with the FastAPI backend"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
    
    def health_check(self) -> bool:
        """Check if backend is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def upload_file(self, session_id: str, file_type: str, file_path: str) -> Dict:
        """Upload a file to the backend"""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                data = {'session_id': session_id, 'file_type': file_type}
                response = requests.post(f"{self.base_url}/upload-file", files=files, data=data)
                return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def delete_file(self, session_id: str, file_path: str) -> Dict:
        """Delete a file via the backend"""
        try:
            data = {"session_id": session_id, "file_path": file_path}
            response = requests.delete(f"{self.base_url}/delete-file", json=data)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def list_files(self, session_id: str, file_type: Optional[str] = None) -> Dict:
        """List files via the backend"""
        try:
            params = {"file_type": file_type} if file_type else {}
            response = requests.get(f"{self.base_url}/list-files/{session_id}", params=params)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def clear_files(self, session_id: str) -> Dict:
        """Clear all files via the backend"""
        try:
            data = {"session_id": session_id}
            response = requests.post(f"{self.base_url}/clear-files", json=data)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def file_exists(self, session_id: str, file_path: str) -> Dict:
        """Check if a file exists via the backend"""
        try:
            params = {"file_path": file_path}
            response = requests.get(f"{self.base_url}/file-exists/{session_id}", params=params)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def start_enterprise_job(self, session_id: str) -> Dict:
        """Start an enterprise standard check job."""
        try:
            payload = {"session_id": session_id}
            response = requests.post(f"{self.base_url}/enterprise-standard/jobs", json=payload)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_enterprise_job(self, job_id: str) -> Dict:
        """Fetch a specific enterprise job status."""
        try:
            response = requests.get(f"{self.base_url}/enterprise-standard/jobs/{job_id}")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_enterprise_jobs(self, session_id: Optional[str] = None) -> Dict:
        """List enterprise jobs for a session (or all sessions)."""
        try:
            params = {"session_id": session_id} if session_id else {}
            response = requests.get(f"{self.base_url}/enterprise-standard/jobs", params=params)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def pause_enterprise_job(self, job_id: str) -> Dict:
        """Pause a running enterprise job."""
        try:
            response = requests.post(f"{self.base_url}/enterprise-standard/jobs/{job_id}/pause")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def resume_enterprise_job(self, job_id: str) -> Dict:
        """Resume a paused enterprise job."""
        try:
            response = requests.post(f"{self.base_url}/enterprise-standard/jobs/{job_id}/resume")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def stop_enterprise_job(self, job_id: str) -> Dict:
        """Request a running enterprise job to stop as soon as possible."""
        try:
            response = requests.post(f"{self.base_url}/enterprise-standard/jobs/{job_id}/stop")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def start_special_symbols_job(self, session_id: str) -> Dict:
        """Start a special symbols check job."""
        try:
            payload = {"session_id": session_id}
            response = requests.post(f"{self.base_url}/special-symbols/jobs", json=payload)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_special_symbols_job(self, job_id: str) -> Dict:
        """Fetch a specific special symbols job status."""
        try:
            response = requests.get(f"{self.base_url}/special-symbols/jobs/{job_id}")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_special_symbols_jobs(self, session_id: Optional[str] = None) -> Dict:
        """List special symbols jobs for a session (or all sessions)."""
        try:
            params = {"session_id": session_id} if session_id else {}
            response = requests.get(f"{self.base_url}/special-symbols/jobs", params=params)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def pause_special_symbols_job(self, job_id: str) -> Dict:
        """Pause a running special symbols job."""
        try:
            response = requests.post(f"{self.base_url}/special-symbols/jobs/{job_id}/pause")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def resume_special_symbols_job(self, job_id: str) -> Dict:
        """Resume a paused special symbols job."""
        try:
            response = requests.post(f"{self.base_url}/special-symbols/jobs/{job_id}/resume")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def stop_special_symbols_job(self, job_id: str) -> Dict:
        """Request a running special symbols job to stop."""
        try:
            response = requests.post(f"{self.base_url}/special-symbols/jobs/{job_id}/stop")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

# Global backend client instance
def get_backend_client():
    """Get backend client instance"""
    return BackendClient()

def is_backend_available() -> bool:
    """Check if FastAPI backend is available"""
    client = get_backend_client()
    return client.health_check() 
