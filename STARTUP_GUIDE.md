# 🚀 PQM AI Application Startup Guide

## 📋 Prerequisites

- **Anaconda** installed on your system
- **Python 3.12** (recommended)
- **Git** (for version control)

## 🛠️ Environment Setup

### 1. Create and Activate Conda Environment

```bash
# Create new environment
conda create -n PQM_AI python=3.12 -y

# Activate environment
conda activate PQM_AI
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

## 🎯 How to Start the Application

### Option 1: Manual Start (Recommended for Development)

**Terminal 1 - Start FastAPI Backend:**
```bash
conda activate PQM_AI
python backend.py
```
✅ Backend will be available at: http://localhost:8001

**Terminal 2 - Start Streamlit Frontend:**
```bash
conda activate PQM_AI
streamlit run main.py --server.port 8888
```
✅ Frontend will be available at: http://localhost:8888

### Option 2: Automated Start (All-in-One)

```bash
conda activate PQM_AI
python start_app.py
```
✅ This will start both services automatically

## 🔍 Verification

### Check Backend Health:
```bash
curl http://localhost:8001/health
```
Expected response: `{"status":"healthy"}`

### Check Frontend:
Open browser and go to: http://localhost:8888

## 📚 API Documentation

Once the backend is running, you can view the interactive API documentation at:
http://localhost:8001/docs

## 🛑 Stopping the Application

### If using Manual Start:
- Press `Ctrl+C` in each terminal

### If using Automated Start:
- Press `Ctrl+C` in the startup script terminal

## 🔧 Troubleshooting

### Port Already in Use
If you get "address already in use" errors:
```bash
# Find processes using the ports
lsof -i :8001
lsof -i :8888

# Kill the processes
pkill -f "python backend.py"
pkill -f "streamlit"
```

### Environment Issues
If you encounter import errors:
```bash
# Ensure you're in the correct environment
conda activate PQM_AI

# Reinstall dependencies
pip install -r requirements.txt
```

### Backend Not Responding
```bash
# Check if backend is running
curl http://localhost:8001/health

# Restart backend if needed
python backend.py
```

## 📁 Project Structure

```
PQM_AI/
├── main.py                          # Streamlit main application
├── backend.py                       # FastAPI backend server
├── backend_client.py                # Backend communication client
├── start_app.py                     # Automated startup script
├── requirements.txt                 # All dependencies with exact versions
├── config.py                        # Application configuration
├── util.py                          # Utility functions
├── tab_*.py                         # Individual tab implementations
└── user_sessions/                   # User session data (auto-created)
```

## 🎉 Success Indicators

✅ **Backend Running:** You see "FastAPI backend is running on http://localhost:8001"

✅ **Frontend Running:** You can access http://localhost:8888 in your browser

✅ **FastAPI Integration:** You see "🔧 使用 FastAPI 后端处理文件操作" in the app

✅ **File Operations:** "清空所有文件" button works without page refresh

## 🚨 Important Notes

- **Always activate the conda environment** before running the application
- **Keep both services running** for full functionality
- **Backend must be running** for FastAPI file operations to work
- **Fallback mode** will be used if backend is unavailable

## 🔄 Development Workflow

1. **Activate environment:** `conda activate PQM_AI`
2. **Start backend:** `python backend.py`
3. **Start frontend:** `streamlit run main.py --server.port 8888`
4. **Make changes** to your code
5. **Test changes** in the browser
6. **Stop services:** `Ctrl+C` when done

---

**Happy coding! 🎯** 