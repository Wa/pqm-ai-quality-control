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

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import fastapi
        import uvicorn
        import streamlit
        import requests
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("❌ Dependencies not found. Please install them first:")
        print("pip install -r requirements.txt")
        return False

def start_backend():
    """Start the FastAPI backend"""
    print("🚀 Starting FastAPI backend...")
    try:
        # Start backend in a subprocess
        backend_process = subprocess.Popen([
            sys.executable, "backend.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for backend to start
        time.sleep(3)
        
        # Check if backend is running
        import requests
        try:
            response = requests.get("http://localhost:8001/health", timeout=5)
            if response.status_code == 200:
                print("✅ FastAPI backend is running on http://localhost:8001")
                return backend_process
            else:
                print("❌ Backend failed to start properly")
                return None
        except:
            print("❌ Backend failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def start_frontend():
    """Start the Streamlit frontend"""
    print("🚀 Starting Streamlit frontend...")
    try:
        # Start Streamlit in a subprocess
        frontend_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "main.py",
            "--server.port", "8888",
            "--server.address", "localhost"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for frontend to start
        time.sleep(5)
        
        print("✅ Streamlit frontend is running on http://localhost:8888")
        return frontend_process
        
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        return None

def main():
    """Main startup function"""
    print("🎯 PQM AI Application Startup")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Failed to start backend. Exiting.")
        return
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Failed to start frontend. Stopping backend.")
        backend_process.terminate()
        return
    
    print("\n🎉 Application started successfully!")
    print("📱 Frontend: http://localhost:8501")
    print("🔧 Backend:  http://localhost:8001")
    print("📚 API Docs: http://localhost:8001/docs")
    print("\nPress Ctrl+C to stop both services...")
    
    try:
        # Keep both processes running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("❌ Backend process stopped unexpectedly")
                break
                
            if frontend_process.poll() is not None:
                print("❌ Frontend process stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        
        # Terminate processes
        if backend_process:
            backend_process.terminate()
            print("✅ Backend stopped")
            
        if frontend_process:
            frontend_process.terminate()
            print("✅ Frontend stopped")
            
        print("👋 Goodbye!")

if __name__ == "__main__":
    main() 