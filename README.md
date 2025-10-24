# PQM AI Quality Control Assistant

A comprehensive AI-powered application for APQP (Advanced Product Quality Planning) document analysis and quality control. Built with Streamlit frontend and FastAPI backend for optimal performance and user experience.

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (3.12 recommended)
- **Anaconda** (recommended for environment management)
- **Git** (for version control)

### Installation

#### Option 1: Automated Setup (Recommended)
```bash
# Clone the repository
git clone <your-repo-url>
cd PQM_AI

# Create and activate conda environment
conda create -n PQM_AI python=3.12 -y
conda activate PQM_AI

# Install dependencies
pip install -r requirements.txt

# Start the application
python start_app.py
```

#### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start backend (Terminal 1)
python backend.py

# Start frontend (Terminal 2)
streamlit run main.py --server.port 8888
```

### Access the Application
- **Frontend:** http://localhost:8888
- **Backend API:** http://localhost:8001
- **API Documentation:** http://localhost:8001/docs

## 🎯 Features

### Core Analysis Tabs
1. **特殊特性符号检查** (Special Symbols Check)
   - Workflow currently undergoing a major rebuild to align with the enterprise standard experience
   - Placeholder interface keeps the tab visible while the new implementation is prepared

2. **设计制程检查** (Design Process Check)
   - Parameter validation and process design analysis
   - Independent session state management

3. **文件齐套性检查** (File Completeness Check)
   - APQP stage-based file completeness analysis
   - Multi-stage comparison (立项阶段, A样阶段, B样阶段, C样阶段)
   - Smart empty stage handling

4. **文件要素检查** (File Elements Check)
   - Integration with external APQP platforms
   - Demo video and documentation access

5. **历史问题规避** (Historical Issues Avoidance)
   - Historical problem tracking and avoidance
   - Excel-based data management

### Advanced Features
- **Multi-User Support:** Complete session isolation with tab-specific state management
- **Dual LLM Backend:** Support for both Ollama (local) and OpenAI (cloud) models
- **FastAPI Integration:** Efficient file operations with real-time updates
- **Workflow Protection:** Prevents interruptions during analysis
- **Demo Mode:** Pre-configured demonstration files for testing

## 🏗️ Architecture

### Frontend (Streamlit)
- **Main Application:** `main.py`
- **Tab Modules:** `tab_*.py` files for each analysis type
- **Utilities:** `util.py` for shared functions
- **Configuration:** `config.py` for centralized settings

### Backend (FastAPI)
- **File Operations:** Upload, delete, list, clear files
- **Health Monitoring:** Service status and connectivity
- **CORS Support:** Cross-origin request handling

### Session Management
```python
# Tab-specific session state structure
session = {
    'tabs': {
        'completeness': { ... },
        'parameters': { ... }
    },
    # Shared global settings
    'llm_backend': 'ollama',
    'ollama_model': 'llama3.1',
    # ... other settings
}
```

## 📁 Project Structure

```
PQM_AI/
├── main.py                          # Streamlit main application
├── backend.py                       # FastAPI backend server
├── backend_client.py                # Backend communication client
├── start_app.py                     # Automated startup script
├── config.py                        # Centralized configuration
├── util.py                          # Utility functions and session management
├── tab_special_symbols_check.py     # Special symbols tab (placeholder during rebuild)
├── tab_parameters_check.py          # Design process analysis
├── tab_file_completeness_check.py   # File completeness analysis
├── tab_file_elements_check.py       # File elements integration
├── tab_history_issues_avoidance.py  # Historical issues tracking
├── tab_settings.py                  # Application settings
├── tab_help_documentation.py        # Help and documentation
├── requirements.txt                 # Python dependencies
├── demonstration/                   # Demo files (Git LFS tracked)
├── CP_files/                        # Control plan files
├── target_files/                    # Target files for analysis
├── graph_files/                     # Graph/drawing files
├── APQP_files/                      # APQP stage files
├── generated_files/                 # Generated outputs
└── media/                          # Media files (images, videos)
```

## ⚙️ Configuration

### Centralized Configuration (`config.py`)
All settings are managed in `config.py` for easy maintenance:

```python
CONFIG = {
    "directories": {
        "cp_files": PROJECT_ROOT / "CP_files",
        "target_files": PROJECT_ROOT / "target_files",
        "graph_files": PROJECT_ROOT / "graph_files",
        "apqp_files": PROJECT_ROOT / "APQP_files",
        "generated_files": PROJECT_ROOT / "generated_files",
        "media": PROJECT_ROOT / "media"
    },
    "llm": {
        "ollama_host": "http://localhost:11434",
        "ollama_model": "llama3.1",
        "openai_base_url": "https://api.openai.com/v1",
        "openai_api_key": "your-api-key",
        "openai_model": "gpt-4"
    }
}
```

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key
- `OLLAMA_HOST`: Ollama server host (default: localhost:11434)

## 🔧 Usage

### 1. User Authentication
- Simple username/password login system
- Session persistence across browser refreshes
- Cross-computer file access using username

### 2. File Upload
- Drag-and-drop or browse file upload
- Support for multiple file types
- Real-time file management with FastAPI backend

### 3. Analysis Workflow
1. **Upload Files:** Add control plans, target files, and related documents
2. **Start Analysis:** Click "开始" to begin AI-powered analysis
3. **Demo Mode:** Click "演示" to use pre-configured demo files
4. **View Results:** Real-time streaming analysis results
5. **Reset:** Click "重新开始" to clear and start over

### 4. Settings Management
- **LLM Backend Selection:** Choose between Ollama and OpenAI
- **Model Configuration:** Adjust temperature, top-p, and other parameters
- **Connection Testing:** Verify LLM service connectivity

## 🛠️ Development

### Adding New Tabs
The application uses a modular tab system. To add a new analysis tab:

1. **Create tab file:** `tab_new_analysis.py`
2. **Add to main.py:** Import and render the new tab
3. **Update session state:** Add tab-specific state in `util.py`
4. **Follow naming conventions:** Use consistent key patterns

### Session State Management
```python
# Get tab-specific session
session = get_user_session(session_id, 'tab_name')

# Start analysis
start_analysis(session_id, 'tab_name')

# Complete analysis
complete_analysis(session_id, 'tab_name')

# Reset session
reset_user_session(session_id, 'tab_name')
```

## 🚨 Troubleshooting

### Common Issues

#### Backend Connection Issues
```bash
# Check backend health
curl http://localhost:8001/health

# Restart backend
python backend.py
```

#### Port Conflicts
```bash
# Find processes using ports
lsof -i :8001
lsof -i :8888

# Kill conflicting processes
pkill -f "python backend.py"
pkill -f "streamlit"
```

#### Environment Issues
```bash
# Ensure correct environment
conda activate PQM_AI

# Reinstall dependencies
pip install -r requirements.txt
```

#### File Upload Issues
- **Large Files:** Use Git LFS for files > 50MB
- **Permission Errors:** Check file permissions and directory access
- **Duplicate Uploads:** Ensure unique session keys (automatically handled)

### Performance Optimization
- **Large Files:** Consider splitting files > 100MB
- **Multiple Users:** Each user has isolated session state
- **LLM Responses:** Streaming responses for better UX

## 📚 API Documentation

### Backend Endpoints
- `GET /health` - Service health check
- `POST /upload-file` - File upload
- `DELETE /delete-file` - File deletion
- `GET /list-files` - File listing
- `POST /clear-files` - Clear all files

### Frontend Integration
```python
from backend_client import get_backend_client, is_backend_available

# Check backend availability
if is_backend_available():
    client = get_backend_client()
    result = client.list_files(session_id, file_type)
```

## 🤝 Contributing

### Development Workflow
1. **Fork the repository**
2. **Create feature branch:** `git checkout -b feature/new-analysis`
3. **Make changes:** Follow existing code patterns
4. **Test thoroughly:** Ensure multi-user compatibility
5. **Submit pull request:** Include detailed description

### Code Standards
- **Session Isolation:** Always use session-specific keys and state
- **Error Handling:** Graceful fallbacks for backend unavailability
- **Documentation:** Update README for new features
- **Testing:** Test with multiple concurrent users

## 📄 License

This project is proprietary software developed for CALB quality control processes.

## 🆘 Support

For technical support or feature requests:
1. Check the troubleshooting section above
2. Review the help documentation in the application
3. Contact the development team

---

**Built with ❤️ for CALB Quality Control Excellence** 