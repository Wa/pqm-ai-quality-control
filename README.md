# Consistency Check Application

A Streamlit-based application for checking consistency between control plan files and target files using large language models.

## Installation

### Using pip (recommended)

```bash
# Clone or copy the application files
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main.py
```

### Manual installation

```bash
# Install required packages
pip install streamlit>=1.28.0 pandas>=2.0.0 openpyxl>=3.1.0 pyarrow>=10.0.0 requests>=2.31.0 ollama>=0.1.0 openai>=1.0.0

# Run the application
streamlit run main.py
```

## Configuration

All file paths and settings are centralized in `config.py` for easy maintenance and portability.

### Session Management

The application uses a user login system for session management:
- **User Authentication**: Simple username/password login system
- **Session Persistence**: Username is saved locally for auto-login
- **Session Directories**: Each user gets their own subdirectories for file uploads
- **Cross-computer Access**: Users can access their files from any computer using the same username

### File Structure

```
Quality_control_assistant/
├── config.py              # Centralized configuration
├── util.py                # Utility functions and PromptGenerator class
├── main.py                # Main Streamlit application
├── consistency_check.py   # Consistency check tab
├── file_completeness_check.py  # File completeness check tab
├── file_elements_check.py      # File elements check tab
├── history_issues_avoidance.py # History issues avoidance tab
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── CP_files/             # Control plan files directory
├── target_files/         # Target files directory
├── graph_files/          # Graph files directory
├── generated_files/      # Generated output files
├── APQP_files/          # APQP stage files
└── media/               # Media files (images, videos)
```

### Configuration Options

The `config.py` file contains:

#### Directories
- `cp_files`: Control plan files directory
- `target_files`: Target files to be checked
- `graph_files`: Graph/drawing files
- `generated_files`: Generated output files
- `apqp_files`: APQP stage files
- `media`: Media files (images, videos)

#### Files
- `apqp_image`: APQP image file
- `demo_video`: Demo video file
- `history_excel`: History issues Excel file

#### LLM Settings
- `ollama_host`: Ollama server host
- `ollama_model`: Ollama model name
- `openai_base_url`: OpenAI API base URL
- `openai_api_key`: OpenAI API key
- `openai_model`: OpenAI model name

### Usage

To modify paths or settings, simply edit the `config.py` file:

```python
# Example: Change the control plan files directory
CONFIG["directories"]["cp_files"] = PROJECT_ROOT / "new_cp_files"

# Example: Change the Ollama host
CONFIG["llm"]["ollama_host"] = "http://localhost:11434"
```

### Benefits

1. **Portability**: All paths are relative to the project root
2. **Maintainability**: Centralized configuration makes updates easy
3. **Flexibility**: Easy to change settings for different environments
4. **No Hardcoded Paths**: All file paths are configurable

### Running the Application

```bash
cd /path/to/Quality_control_assistant
streamlit run main.py
```

The application will automatically use the paths defined in `config.py` and create any missing directories as needed.

## Dependencies

### Core Dependencies
- `streamlit>=1.28.0`: Web application framework
- `pandas>=2.0.0`: Data manipulation and analysis
- `openpyxl>=3.1.0`: Excel file reading and writing
- `requests>=2.31.0`: HTTP requests for API calls
- `pyarrow>=10.0.0`: DataFrame serialization (for Streamlit compatibility)

### LLM Dependencies
- `ollama>=0.1.0`: Local LLM client
- `openai>=1.0.0`: OpenAI API client

### System Requirements
- Python 3.8 or higher
- Internet connection (for LLM APIs)
- Optional: Ollama server running locally

## Troubleshooting

### DataFrame Serialization Issues
If you encounter PyArrow serialization errors when displaying DataFrames, the application automatically handles mixed data types by converting all data to strings. This prevents compatibility issues between pandas DataFrames and Streamlit's display system.

### Duplicate Upload Boxes
If file upload boxes appear duplicated after clicking "开始", this was caused by `st.rerun()` calls that have been removed. The application now properly handles file uploads without page refreshes.

### Duplicate Element ID Errors
If you encounter `StreamlitDuplicateElementId` errors with text areas, all text areas now have unique keys to prevent conflicts:
- Prompt text areas: `prompt_1`, `prompt_2`, etc.
- Response text areas: `ollama_response_1`, `openai_response_1`, etc.
- Final section: `final_prompt`, `final_ollama_response`, `final_openai_response`

**Note**: For streaming responses, the application uses `st.empty()` with `placeholder.write()` instead of `text_area()` to avoid creating multiple widgets with the same key during streaming updates.

### Known Issues
- The `st.cache` deprecation warning may appear but doesn't affect functionality
- DataFrame display automatically converts mixed data types to strings for compatibility

## Accessibility

The application follows accessibility best practices:
- All form widgets have proper labels for screen readers
- Empty labels are avoided by using `label_visibility="collapsed"` when needed
- Text areas and input fields have descriptive labels that are hidden but accessible 