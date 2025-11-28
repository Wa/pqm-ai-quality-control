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
cd pqm-ai-quality-control

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
1. **⚡ APQP交付物一键检查** (APQP Deliverables One-Click Check)
   - Stage-based upload of APQP deliverables (立项, A样, B样, C样等)
   - One-click parsing pipeline that converts PDFs/Office/Excel/text into normalized text using the backend
   - Designed to power downstream completeness and consistency checks

2. **文件齐套性检查** (File Completeness Check)
   - APQP stage-based file completeness analysis
   - Multi-stage comparison (立项阶段, A样阶段, B样阶段, C样阶段)
   - Smart handling of empty or missing stages

3. **特殊特性符号检查** (Special Symbols Check)
   - Full special-symbol parsing and checking workflow (no longer a placeholder)
   - Supports long-running background jobs with progress tracking and pause/stop controls
   - Can run in regular or high-performance modes depending on deployment

4. **设计制程检查** (Design Process / Parameter Check)
   - Parameter validation and process design analysis against reference documents
   - Independent session state management per user

5. **文件要素检查** (File Elements Check)
   - Checks whether key document “elements” are present for each APQP stage/deliverable
   - Integrates with external APQP platforms and internal standards

6. **企业标准检查** (Enterprise Standard Check)
   - Compares target documents against enterprise standards
   - Streams detailed LLM prompts/responses and generates structured result files (CSV/XLSX)
   - Includes a full “演示” workflow using pre-generated prompt/response chunks

7. **历史问题规避** (Historical Issues Avoidance)
   - Historical problem tracking and avoidance based on Excel knowledge base
   - Helps engineers avoid repeating known issues during new projects

8. **🤖 AI智能体** (AI Agent)
   - LangGraph-based ReAct-style agent with tools for file operations, HTTP, web search, and text conversion
   - Multi-step planning UI with explicit execution plan, streaming progress, and debug panel
   - Per-user, per-conversation history with file uploads scoped to each conversation

### Advanced Features
- **Multi-User Support:** Complete session isolation with tab-specific state management
- **Backend Job Orchestration:** FastAPI manages long-running analysis jobs with progress, pause/resume, and log streaming
- **Hybrid LLM Backend:** Supports local Ollama and cloud providers (OpenAI-compatible gateways, ModelScope, etc.) via `config.py`
- **Workflow Protection:** Prevents interruptions during analysis and protects long-running jobs
- **Demo Mode:** Pre-configured demonstration files and prompt/response traces for safe testing

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
pqm-ai-quality-control/
├── main.py                       # Streamlit main application (tab container + login)
├── backend.py                    # FastAPI backend (file ops + long-running jobs)
├── backend_client.py             # Backend communication client used by tabs
├── start_app.py                  # Automated startup script (starts backend + frontend)
├── config.py                     # Centralized configuration (paths, LLM, Bisheng, services)
├── util.py                       # Utility functions and session management helpers
├── tabs/
│   ├── tab_home.py               # 首页
│   ├── tab_apqp_one_click_check.py
│   ├── tab_file_completeness_check.py
│   ├── tab_special_symbols_check.py
│   ├── tab_parameters_check.py
│   ├── tab_file_elements_check.py
│   ├── tab_enterprise_standard_check.py
│   ├── tab_history_issues_avoidance.py
│   ├── tab_ai_agent.py           # AI智能体主界面
│   ├── tab_settings.py
│   ├── tab_help_documentation.py
│   ├── tab_admin.py
│   ├── ai_agent/                 # LangGraph agent + MCP tools implementation
│   ├── enterprise_standard/      # Enterprise standard comparison workflow
│   ├── file_completeness/        # APQP completeness logic
│   ├── file_elements/            # File elements evaluation logic
│   ├── history/                  # Historical issues workflows
│   ├── parameters/               # Design parameter workflows
│   ├── special_symbols/          # Special-symbol parsing & checking
│   └── shared/                   # Shared text/file conversion & utilities
├── demonstration/                # Demo files used by “演示” buttons
├── uploads/                      # User uploads (per user/session)
├── generated_files/              # Generated outputs, parsed text, reports
├── user_sessions/                # Persisted per-user session metadata
├── requirements.txt              # Python dependencies
└── README.md
```

## ⚙️ Configuration

### Centralized Configuration (`config.py`)
All core paths and service settings are managed in `config.py`:

- **Directories** (`CONFIG["directories"]`): `uploads`, `generated_files`, `demonstration`, optional `media` assets.
- **Files** (`CONFIG["files"]`): Paths to demo images/videos and reference Excel workbooks.
- **LLM settings** (`CONFIG["llm"]`): Local Ollama hosts/models, OpenAI-compatible base URL/model, ModelScope API, etc.
- **Bisheng** (`CONFIG["bisheng"]`): Internal workflow service used for some enterprise checks.
- **Services** (`CONFIG["services"]`): External services such as Unstructured/MinerU-style document parsing.

Before deploying outside your internal environment, review and update `config.py` to point to your own endpoints and API keys, and move any secrets into environment variables or a secure config mechanism (do not commit real keys to a public repo).

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

### Backend Endpoints (selected)
- `GET /health` - Service health check
- `POST /upload-file` - File upload for parameter/reference/graph files
- `DELETE /delete-file` - File deletion
- `GET /list-files/{session_id}` - File listing per session
- `POST /clear-files` - Clear all parameter/reference/graph files for a session
- `POST /apqp-one-click/parse` - Parse APQP uploads into normalized text by stage
- `POST /apqp-one-click/classify` - Run LLM-based classification to map parsed texts to APQP deliverables
- `POST /apqp-one-click/clear` - Clear APQP uploads/parsed outputs
- `POST /enterprise-standard/jobs` - Start an enterprise-standard comparison job
- `GET /enterprise-standard/jobs` / `GET /enterprise-standard/jobs/{job_id}` - Inspect enterprise jobs
- `POST /special-symbols/jobs` - Start a special-symbols analysis job
- `POST /file-completeness/jobs` / `POST /parameters/jobs` / `POST /history/jobs` / `POST /file-elements/jobs` - Start other analysis jobs

All endpoints are fully documented in the built-in FastAPI docs at `http://localhost:8001/docs`.

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