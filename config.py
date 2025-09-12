from pathlib import Path

# Get the project root directory (where this script is located)
PROJECT_ROOT = Path(__file__).parent.absolute()

# Define all file paths relative to project root
CONFIG = {
    "directories": {
        "cp_files": PROJECT_ROOT / "CP_files",
        "target_files": PROJECT_ROOT / "target_files", 
        "graph_files": PROJECT_ROOT / "graph_files",
        "generated_files": PROJECT_ROOT / "generated_files",
        "apqp_files": PROJECT_ROOT / "APQP_files",
        "media": PROJECT_ROOT / "media"
    },
    "files": {
        "apqp_image": PROJECT_ROOT / "media" / "APQP3.png",
        "demo_video": PROJECT_ROOT / "media" / "bisheng_demo_1.webm",
        "history_excel": PROJECT_ROOT / "demonstration" / "副本LL-lesson learn-历史问题规避-V9.4.xlsx"
    },
    "llm": {
        "ollama_host": "http://10.31.60.9:11434",
        "ollama_host_127": "http://10.31.60.127:11434",
        "ollama_host_9": "http://10.31.60.9:11434",
        "ollama_model": "gpt-oss:latest",
        "openai_base_url": "https://sg.uiuiapi.com/v1",
        "openai_api_key": "sk-dDG9UBQHLshfb8Z5FYQQXFOZAe6FtUxltMwIxg0KNCSsGKjh",
        "openai_model": "gpt-3.5-turbo"
    },
    "bisheng": {
        # Bisheng workflow service defaults; can be overridden by environment variables
        # BISHENG_BASE_URL, BISHENG_INVOKE_PATH, BISHENG_STOP_PATH, BISHENG_WORKFLOW_ID, BISHENG_API_KEY
        "base_url": "http://10.31.60.11:3001",
        "invoke_path": "/api/v2/workflow/invoke",
        "stop_path": "/api/v2/workflow/stop",
        "workflow_id": "31208af992c94e9fb56b759ebff2f242",
        "api_key": "",
        # Chunking default
        "max_words": 2000,
        # Request timeout seconds
        "timeout_s": 90
    },
    # Central service endpoints/switches
    "services": {
        # Unstructured API endpoint (fallback if env UNSTRUCTURED_API_URL not set)
        "unstructured_api_url": "http://10.31.60.11:8000/general/v0/general"
    }
}

# Helper function to get a specific path
def get_path(category, name):
    """Get a specific path from the configuration."""
    if category in CONFIG and name in CONFIG[category]:
        return CONFIG[category][name]
    raise KeyError(f"Path not found: {category}.{name}")

# Helper function to get a directory path
def get_directory(name):
    """Get a directory path from the configuration."""
    return get_path("directories", name)

# Helper function to get a file path
def get_file(name):
    """Get a file path from the configuration."""
    return get_path("files", name) 