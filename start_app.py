#!/usr/bin/env python3
"""
Startup script for PQM AI Application
Runs both FastAPI backend and Streamlit frontend
"""

import subprocess
import sys
import time
import os
from pathlib import Path

import socket


def check_dependencies():
    """Check if required packages are installed"""
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
        import streamlit  # noqa: F401
        import requests  # noqa: F401
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("❌ Dependencies not found. Please install them first:")
        print("pip install -r requirements.txt")
        return False


def is_backend_running():
    """Check backend health endpoint."""
    try:
        import requests
        r = requests.get("http://localhost:8001/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def is_frontend_running():
    """Check if Streamlit is responding on port 8888."""
    try:
        import requests
        r = requests.get("http://localhost:8888", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def start_backend():
    """Start the FastAPI backend if not already running.
    Returns (process, used_existing: bool)
    """
    if is_backend_running():
        print("ℹ️ Detected existing FastAPI backend at http://localhost:8001 (reusing)")
        return None, True

    print("🚀 Starting FastAPI backend...")
    try:
        # Start backend in a subprocess; redirect output to avoid buffer blocking
        backend_process = subprocess.Popen(
            [sys.executable, "backend.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        # Wait up to 10s for health
        for _ in range(20):
            if is_backend_running():
                print("✅ FastAPI backend is running on http://localhost:8001")
                return backend_process, False
            time.sleep(0.5)

        print("❌ Backend failed to start properly")
        return None, False

    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None, False


def start_frontend():
    """Start the Streamlit frontend if not already running.
    Returns (process, used_existing: bool)
    """
    if is_frontend_running():
        print("ℹ️ Detected existing Streamlit frontend at http://localhost:8888 (reusing)")
        return None, True

    print("🚀 Starting Streamlit frontend...")
    try:
        frontend_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "main.py",
                "--server.port",
                "8888",
                "--server.address",
                "0.0.0.0",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        # Wait up to 12s for it to respond
        for _ in range(24):
            if is_frontend_running():
                print("✅ Streamlit frontend is running on http://localhost:8888")
                return frontend_process, False
            time.sleep(0.5)

        print("❌ Frontend failed to start properly")
        return None, False

    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        return None, False


def check_restart_signal():
    """Check if restart signal file exists and remove it.
    Returns True if restart signal was detected, False otherwise.
    """
    restart_file = Path(__file__).parent / ".restart_signal"
    if restart_file.exists():
        try:
            restart_file.unlink()  # Delete the signal file
            return True
        except Exception as e:
            print(f"⚠️ Warning: Failed to delete restart signal file: {e}")
            return True  # Still return True to trigger restart
    return False


def main():
    """Main startup function"""
    print("🎯 PQM AI Application Startup")
    print("=" * 40)

    # Check dependencies
    if not check_dependencies():
        return

    # Start backend (or reuse existing)
    backend_process, backend_existing = start_backend()
    if not backend_existing and not backend_process:
        print("❌ Failed to start backend. Exiting.")
        return

    # Start frontend (or reuse existing)
    frontend_process, frontend_existing = start_frontend()
    if not frontend_existing and not frontend_process:
        print("❌ Failed to start frontend.")
        # Stop backend we just started (do not stop existing)
        if backend_process:
            backend_process.terminate()
            print("✅ Backend stopped")
        return

    print("\n🎉 Application started successfully!")
    print("📱 Frontend: http://localhost:8888")
    print("🔧 Backend:  http://localhost:8001")
    print("📚 API Docs: http://localhost:8001/docs")
    print("\nPress Ctrl+C to stop both services...")

    try:
        # Keep both services healthy
        while True:
            time.sleep(1)

            # Check for restart signal
            if check_restart_signal():
                print("\n🔄 Restart signal detected. Restarting services...")
                # Gracefully stop services we started
                if backend_process:
                    try:
                        backend_process.terminate()
                        backend_process.wait(timeout=5)
                        print("✅ Backend stopped")
                    except subprocess.TimeoutExpired:
                        backend_process.kill()
                        print("⚠️ Backend force-killed")
                    except Exception as e:
                        print(f"⚠️ Error stopping backend: {e}")
                
                if frontend_process:
                    try:
                        frontend_process.terminate()
                        frontend_process.wait(timeout=5)
                        print("✅ Frontend stopped")
                    except subprocess.TimeoutExpired:
                        frontend_process.kill()
                        print("⚠️ Frontend force-killed")
                    except Exception as e:
                        print(f"⚠️ Error stopping frontend: {e}")
                
                # Wait a moment for ports to be released
                time.sleep(1)
                
                # Restart services
                backend_process_new, backend_existing_new = start_backend()
                frontend_process_new, frontend_existing_new = start_frontend()
                
                if backend_process_new or backend_existing_new:
                    if frontend_process_new or frontend_existing_new:
                        backend_process = backend_process_new
                        frontend_process = frontend_process_new
                        print("✅ Services restarted successfully!")
                        continue
                
                print("❌ Restart failed. Exiting.")
                break

            # Monitor backend: if we started it, prefer process; otherwise health
            if backend_process is not None:
                if backend_process.poll() is not None and not is_backend_running():
                    print("❌ Backend process stopped and health check failed")
                    break
            else:
                if not is_backend_running():
                    print("❌ Backend health check failed")
                    break

            # Monitor frontend similarly
            if frontend_process is not None:
                if frontend_process.poll() is not None and not is_frontend_running():
                    print("❌ Frontend process stopped and health check failed")
                    break
            else:
                if not is_frontend_running():
                    print("❌ Frontend health check failed")
                    break

    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")

    finally:
        # Terminate only processes we started
        if backend_process:
            backend_process.terminate()
            print("✅ Backend stopped")
        if frontend_process:
            frontend_process.terminate()
            print("✅ Frontend stopped")
        print("👋 Goodbye!")


if __name__ == "__main__":
    main() 