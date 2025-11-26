import os
import time
from typing import Any, Dict, List, Optional

import requests
import streamlit as st


@st.cache_data(show_spinner=False, ttl=5)
def _cached_health_check(base_url: str) -> bool:
    """Return backend health status with a short cache to avoid repeated calls."""

    try:
        response = requests.get(f"{base_url}/health", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False

class BackendClient:
    """Client for communicating with the FastAPI backend"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
    
    def health_check(self) -> bool:
        """Check if backend is running"""
        return _cached_health_check(self.base_url)
    
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

    def upload_apqp_file(self, session_id: str, stage: str, file_obj) -> Dict:
        """Upload an APQP file to backend"""

        try:
            files = {"file": (file_obj.name, file_obj, getattr(file_obj, "type", None) or "application/octet-stream")}
            data = {"session_id": session_id, "stage": stage}
            response = requests.post(f"{self.base_url}/apqp-one-click/upload", files=files, data=data, timeout=60)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_apqp_files(self, session_id: str, stage: Optional[str] = None) -> Dict:
        """List APQP files for a session/stage."""

        try:
            params = {"stage": stage} if stage else None
            response = requests.get(f"{self.base_url}/apqp-one-click/files/{session_id}", params=params, timeout=10)
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
    
    def parse_apqp_files(self, session_id: str, stages: Optional[List[str]] = None) -> Dict:
        """Request backend to parse APQP uploads into text."""
        try:
            payload: Dict[str, Any] = {"session_id": session_id}
            if stages:
                payload["stages"] = stages
            response = requests.post(
                f"{self.base_url}/apqp-one-click/parse",
                json=payload,
                timeout=600,
            )
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def start_apqp_parse_job(self, session_id: str, stages: Optional[List[str]] = None) -> Dict:
        """Start background APQP parsing job."""

        try:
            payload: Dict[str, Any] = {"session_id": session_id}
            if stages:
                payload["stages"] = stages
            response = requests.post(
                f"{self.base_url}/apqp-one-click/jobs", json=payload, timeout=30
            )
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_apqp_job_status(self, job_id: str) -> Dict:
        """Fetch status for APQP parsing job."""

        try:
            response = requests.get(f"{self.base_url}/apqp-one-click/jobs/{job_id}", timeout=10)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def pause_apqp_job(self, job_id: str) -> Dict:
        """Pause an APQP one-click parsing job."""

        try:
            response = requests.post(
                f"{self.base_url}/apqp-one-click/jobs/{job_id}/pause", timeout=10
            )
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def resume_apqp_job(self, job_id: str) -> Dict:
        """Resume a paused APQP one-click parsing job."""

        try:
            response = requests.post(
                f"{self.base_url}/apqp-one-click/jobs/{job_id}/resume", timeout=10
            )
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_apqp_jobs(self, session_id: Optional[str] = None) -> Dict | List[Dict]:
        """List APQP one-click parsing jobs for a session (or all sessions)."""

        try:
            params = {"session_id": session_id} if session_id else None
            response = requests.get(
                f"{self.base_url}/apqp-one-click/jobs", params=params, timeout=10
            )
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def classify_apqp_files(
        self,
        session_id: str,
        stages: Optional[List[str]] = None,
        head_chars: int = 3200,
        tail_chars: int = 2000,
        turbo_mode: bool = False,
    ) -> Dict:
        """Request backend to classify parsed APQP documents via LLM."""

        payload: Dict[str, Any] = {
            "session_id": session_id,
            "head_chars": head_chars,
            "tail_chars": tail_chars,
            "turbo_mode": turbo_mode,
        }
        if stages:
            payload["stages"] = stages

        last_error: Optional[str] = None
        for attempt in range(2):
            try:
                response = requests.post(
                    f"{self.base_url}/apqp-one-click/classify",
                    json=payload,
                    timeout=900,
                )
                return response.json()
            except Exception as e:
                last_error = str(e)
                time.sleep(2)
        return {"status": "error", "message": last_error or "未知错误"}
    
    def clear_apqp_files(self, session_id: str, target: str = "all") -> Dict:
        """Clear APQP uploads and/or parsed outputs via the backend."""
        try:
            payload = {"session_id": session_id, "target": target}
            response = requests.post(f"{self.base_url}/apqp-one-click/clear", json=payload, timeout=120)
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

    def start_special_symbols_job(self, session_id: str, turbo_mode: bool = False) -> Dict:
        """Start a special symbols check job."""
        try:
            payload = {"session_id": session_id, "turbo_mode": bool(turbo_mode)}
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

    def start_file_elements_job(self, payload: Dict[str, Any]) -> Dict:
        """Start a file elements evaluation job."""
        try:
            response = requests.post(f"{self.base_url}/file-elements/jobs", json=payload, timeout=60)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_file_elements_job(self, job_id: str) -> Dict:
        """Fetch a file elements job status."""
        try:
            response = requests.get(f"{self.base_url}/file-elements/jobs/{job_id}")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_file_elements_jobs(self, session_id: Optional[str] = None) -> Dict:
        """List file elements jobs for a session."""
        try:
            params = {"session_id": session_id} if session_id else {}
            response = requests.get(f"{self.base_url}/file-elements/jobs", params=params)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def start_parameters_job(self, session_id: str) -> Dict:
        """Start a parameters consistency check job."""
        try:
            payload = {"session_id": session_id}
            response = requests.post(f"{self.base_url}/parameters/jobs", json=payload)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_parameters_job(self, job_id: str) -> Dict:
        """Fetch a specific parameters job status."""
        try:
            response = requests.get(f"{self.base_url}/parameters/jobs/{job_id}")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_parameters_jobs(self, session_id: Optional[str] = None) -> Dict:
        """List parameters jobs for a session (or all sessions)."""
        try:
            params = {"session_id": session_id} if session_id else {}
            response = requests.get(f"{self.base_url}/parameters/jobs", params=params)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def pause_parameters_job(self, job_id: str) -> Dict:
        """Pause a running parameters job."""
        try:
            response = requests.post(f"{self.base_url}/parameters/jobs/{job_id}/pause")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def resume_parameters_job(self, job_id: str) -> Dict:
        """Resume a paused parameters job."""
        try:
            response = requests.post(f"{self.base_url}/parameters/jobs/{job_id}/resume")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def stop_parameters_job(self, job_id: str) -> Dict:
        """Request a running parameters job to stop."""
        try:
            response = requests.post(f"{self.base_url}/parameters/jobs/{job_id}/stop")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def start_file_completeness_job(self, session_id: str) -> Dict:
        """Start a file completeness check job."""
        try:
            payload = {"session_id": session_id}
            response = requests.post(f"{self.base_url}/file-completeness/jobs", json=payload)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_file_completeness_job(self, job_id: str) -> Dict:
        """Fetch a specific file completeness job status."""
        try:
            response = requests.get(f"{self.base_url}/file-completeness/jobs/{job_id}")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_file_completeness_jobs(self, session_id: Optional[str] = None) -> Dict:
        """List file completeness jobs for a session (or all sessions)."""
        try:
            params = {"session_id": session_id} if session_id else {}
            response = requests.get(f"{self.base_url}/file-completeness/jobs", params=params)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def pause_file_completeness_job(self, job_id: str) -> Dict:
        """Pause a running file completeness job."""
        try:
            response = requests.post(f"{self.base_url}/file-completeness/jobs/{job_id}/pause")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def resume_file_completeness_job(self, job_id: str) -> Dict:
        """Resume a paused file completeness job."""
        try:
            response = requests.post(f"{self.base_url}/file-completeness/jobs/{job_id}/resume")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def stop_file_completeness_job(self, job_id: str) -> Dict:
        """Request a running file completeness job to stop."""
        try:
            response = requests.post(f"{self.base_url}/file-completeness/jobs/{job_id}/stop")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def start_history_job(self, session_id: str) -> Dict:
        """Start a history issue avoidance job."""
        try:
            payload = {"session_id": session_id}
            response = requests.post(f"{self.base_url}/history/jobs", json=payload)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_history_job(self, job_id: str) -> Dict:
        """Fetch a specific history job status."""
        try:
            response = requests.get(f"{self.base_url}/history/jobs/{job_id}")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_history_jobs(self, session_id: Optional[str] = None) -> Dict:
        """List history jobs for a session (or all sessions)."""
        try:
            params = {"session_id": session_id} if session_id else {}
            response = requests.get(f"{self.base_url}/history/jobs", params=params)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def pause_history_job(self, job_id: str) -> Dict:
        """Pause a running history job."""
        try:
            response = requests.post(f"{self.base_url}/history/jobs/{job_id}/pause")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def resume_history_job(self, job_id: str) -> Dict:
        """Resume a paused history job."""
        try:
            response = requests.post(f"{self.base_url}/history/jobs/{job_id}/resume")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def stop_history_job(self, job_id: str) -> Dict:
        """Request a running history job to stop."""
        try:
            response = requests.post(f"{self.base_url}/history/jobs/{job_id}/stop")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

# Global backend client instance
def get_backend_client():
    """Get backend client instance"""
    return BackendClient()

def is_backend_available(*, force_refresh: bool = False) -> bool:
    """Check if FastAPI backend is available, optionally forcing a refresh."""

    if force_refresh:
        _cached_health_check.clear()
    client = get_backend_client()
    return client.health_check()
