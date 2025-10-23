import os
import pandas as pd
from openpyxl import load_workbook
from pathlib import Path
import json
import streamlit as st
from config import CONFIG
import time
import hashlib
import openpyxl
import pandas as pd

# --- LLM Host Resolution ---
def resolve_ollama_host(llm_backend: str) -> str:
    """Return the Ollama host URL based on selected backend key.

    Recognized keys:
    - 'ollama_127' -> CONFIG['llm']['ollama_host_127'] if present, else fallback to CONFIG['llm']['ollama_host']
    - 'ollama_9'   -> CONFIG['llm']['ollama_host_9'] if present, else fallback to CONFIG['llm']['ollama_host']
    - any other    -> CONFIG['llm']['ollama_host']
    """
    try:
        if llm_backend == "ollama_127":
            return CONFIG["llm"].get("ollama_host_127") or CONFIG["llm"]["ollama_host"]
        if llm_backend == "ollama_9":
            return CONFIG["llm"].get("ollama_host_9") or CONFIG["llm"]["ollama_host"]
    except Exception:
        pass
    return CONFIG["llm"]["ollama_host"]

# --- Login Management ---
def render_login_widget():
    """Render the login widget and return the authenticated username."""

    if "authenticated_username" not in st.session_state:
        saved_username = load_username()
        if saved_username:
            st.session_state.authenticated_username = saved_username

    if "login_form_username" not in st.session_state:
        st.session_state.login_form_username = st.session_state.get(
            "authenticated_username", ""
        )

    with st.sidebar:
        st.header("🔐 用户登录")
        username_input = st.text_input(
            "请输入用户名",
            value=st.session_state.get("login_form_username", ""),
            key="login_username_input",
        )
        login_clicked = st.button("登录", type="primary")
        logout_clicked = st.button("退出登录")

    if logout_clicked:
        st.session_state.pop("authenticated_username", None)
        st.session_state.login_form_username = ""
        return None

    username_input = username_input.strip()

    if login_clicked and username_input:
        st.session_state.authenticated_username = username_input
        st.session_state.login_form_username = username_input
        save_username(username_input)

    return st.session_state.get("authenticated_username")


def get_username_file():
    """Get the path to the username storage file."""
    return os.path.join(os.path.expanduser("~"), ".streamlit_username")

def save_username(username):
    """Save username to file for persistence."""
    try:
        with open(get_username_file(), 'w') as f:
            f.write(username)
        return True
    except Exception:
        return False

def load_username():
    """Load username from file if it exists."""
    try:
        if os.path.exists(get_username_file()):
            with open(get_username_file(), 'r') as f:
                return f.read().strip()
    except Exception:
        pass
    return None

def get_user_session_id(username):
    """Generate session ID based on username for persistence."""
    # Always derive from the provided username to avoid cross-user leakage
    return username

# New helper to determine if a user is an admin
def is_admin(username: str) -> bool:
    admins_env = os.getenv("ADMIN_USERS", "")
    admins = {u.strip() for u in admins_env.split(",") if u.strip()}
    if not admins:
        admins = {"admin"}
    return username in admins

# --- Session Directory Utilities ---
def ensure_session_dirs(base_dirs, session_id):
    """Ensure session directories exist."""
    import os

    # Create session-specific directories
    session_dirs = {}
    for dir_type, base_dir in base_dirs.items():
        session_dir = os.path.join(base_dir, str(session_id))
        os.makedirs(session_dir, exist_ok=True)
        session_dirs[dir_type] = session_dir

    # Ensure standard subfolders inside generated/<session_id>
    for gen_key in ("generated", "generated_files"):
        if gen_key in session_dirs:
            generated_root = session_dirs[gen_key]
            subfolders = [
                "parameters_check",
                "file_elements_check",
                "file_completeness_check",
                "history_issues_avoidance",
            ]
            for name in subfolders:
                os.makedirs(os.path.join(generated_root, name), exist_ok=True)
            # Convenience keys
            session_dirs["generated_parameters_check"] = os.path.join(generated_root, "parameters_check")
            session_dirs["generated_file_elements_check"] = os.path.join(generated_root, "file_elements_check")
            session_dirs["generated_file_completeness_check"] = os.path.join(generated_root, "file_completeness_check")
            session_dirs["generated_history_issues_avoidance"] = os.path.join(generated_root, "history_issues_avoidance")

    # Ensure enterprise standard directories exist under project root:
    # ./PQM_AI/enterprise_standard_files/standards/<username>
    # ./PQM_AI/enterprise_standard_files/examined_files/<username>
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        enterprise_root = os.path.join(project_root, "enterprise_standard_files")
        standards_dir = os.path.join(enterprise_root, "standards", str(session_id))
        examined_dir = os.path.join(enterprise_root, "examined_files", str(session_id))
        os.makedirs(standards_dir, exist_ok=True)
        os.makedirs(examined_dir, exist_ok=True)
        # Expose convenience keys regardless of input base_dirs
        session_dirs.setdefault("enterprise_standards", standards_dir)
        session_dirs.setdefault("enterprise_examined", examined_dir)
    except Exception:
        # Fail-safe: do not break callers if path operations fail
        pass

    # Ensure history issues avoidance directories exist under project root:
    # ./PQM_AI/history_issues_avoidance_files/issue_lists/<username>
    # ./PQM_AI/history_issues_avoidance_files/target_files/<username>
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        history_root = os.path.join(project_root, "history_issues_avoidance_files")
        issue_lists_dir = os.path.join(history_root, "issue_lists", str(session_id))
        target_files_dir = os.path.join(history_root, "target_files", str(session_id))
        os.makedirs(issue_lists_dir, exist_ok=True)
        os.makedirs(target_files_dir, exist_ok=True)
        # Expose convenience keys regardless of input base_dirs
        session_dirs.setdefault("history_issue_lists", issue_lists_dir)
        session_dirs.setdefault("history_target_files", target_files_dir)
    except Exception:
        # Fail-safe: do not break callers if path operations fail
        pass

    return session_dirs

# --- Structured Session State Management ---
def get_user_session(session_id, tab_name=None):
    """Get or create a structured user session with optional tab-specific state."""
    if 'user_sessions' not in st.session_state:
        st.session_state.user_sessions = {}

    if session_id not in st.session_state.user_sessions:
        st.session_state.user_sessions[session_id] = {
            'tabs': {
                'completeness': {
                    'process_started': False,
                    'analysis_completed': False,
                    'ollama_history': [],
                    'openai_history': [],
                },
                'parameters': {
                    'process_started': False,
                    'analysis_completed': False,
                    'ollama_history': [],
                    'openai_history': [],
                },
                'ai_agent': {
                    'ollama_history': [],
                    'openai_history': [],
                },
            },
            'llm_backend': 'ollama_9',
            'ollama_model': CONFIG["llm"]["ollama_model"],
            'ollama_temperature': 0.7,
            'ollama_top_p': 0.9,
            'ollama_top_k': 40,
            'ollama_repeat_penalty': 1.1,
            'ollama_num_ctx': 40001,
            'ollama_num_thread': 4,
            'openai_model': CONFIG["llm"]["openai_model"],
            'openai_temperature': 0.7,
            'openai_top_p': 1.0,
            'openai_max_tokens': 2048,
            'openai_presence_penalty': 0.0,
            'openai_frequency_penalty': 0.0,
            'openai_logit_bias': '{}',
        }

    session = st.session_state.user_sessions[session_id]

    if tab_name and tab_name in session['tabs']:
        return session['tabs'][tab_name]

    return session


def reset_user_session(session_id, tab_name=None):
    """Reset a user's session to initial state, optionally for a specific tab."""
    session = get_user_session(session_id)

    if tab_name and tab_name in session['tabs']:
        tab_session = session['tabs'][tab_name]
        tab_session['analysis_completed'] = False
        tab_session['process_started'] = False
        tab_session['ollama_history'] = []
        tab_session['openai_history'] = []
    else:
        for tab_session in session['tabs'].values():
            tab_session['analysis_completed'] = False
            tab_session['process_started'] = False
            tab_session['ollama_history'] = []
            tab_session['openai_history'] = []


def start_analysis(session_id, tab_name):
    """Start analysis for a specific tab in a user session."""
    session = get_user_session(session_id)
    if tab_name in session['tabs']:
        tab_session = session['tabs'][tab_name]
        tab_session['analysis_completed'] = False
        tab_session['process_started'] = True
        tab_session['ollama_history'] = []
        tab_session['openai_history'] = []


def complete_analysis(session_id, tab_name):
    """Mark analysis as completed for a specific tab in a user session."""
    session = get_user_session(session_id)
    if tab_name in session['tabs']:
        session['tabs'][tab_name]['analysis_completed'] = True


# --- File Upload Utility ---
def handle_file_upload(files, save_dir):
    if files:
        for file in files:
            save_path = os.path.join(save_dir, file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
        return len(files)
    return 0

# --- Prompt Generator and File Handling ---
def save_uploaded_file(uploaded_file, session_id, file_type):
    """Handle a single file upload and save to the session directory.

    Note: Renamed from handle_file_upload to avoid clashing with the
    bulk-upload helper used across tabs (files, save_dir).
    """
    if uploaded_file is not None:
        # Use the existing ensure_session_dirs function
        base_dirs = {
            'generated_files': os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_files")
        }
        session_dirs = ensure_session_dirs(base_dirs, session_id)
        session_dir = session_dirs['generated_files']
        
        file_path = os.path.join(session_dir, f"{file_type}_{uploaded_file.name}")
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    return None


# --- Utility helpers (migrated from temporary scripts) ---
def json_canonical_sha256(obj) -> str:
    """Compute a canonical SHA-256 for a JSON-serializable object (sorted keys, compact)."""
    canon = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest().upper()

def convert_extracted_json_to_dataframe(records):
    """Convert extracted JSON format [{File, Sheet, data: [[...], ...]}, ...] to a flat DataFrame.

    Columns: File, Sheet, Row, C1..Ck
    """
    rows = []
    max_cols = 0
    for rec in records:
        file_name = rec.get("File")
        sheet_name = rec.get("Sheet")
        data = rec.get("data", []) or []
        for idx, row in enumerate(data, start=1):
            max_cols = max(max_cols, len(row))
            rows.append({
                "File": file_name,
                "Sheet": sheet_name,
                "Row": idx,
                "_cells": row,
            })
    columns = [f"C{i}" for i in range(1, max_cols + 1)]
    table_rows = []
    for r in rows:
        base = {"File": r["File"], "Sheet": r["Sheet"], "Row": r["Row"]}
        cells = r["_cells"]
        for i, col_name in enumerate(columns, start=1):
            base[col_name] = cells[i - 1] if i - 1 < len(cells) else ""
        table_rows.append(base)
    return pd.DataFrame(table_rows, columns=["File", "Sheet", "Row"] + columns)


# --- Parameter Extraction (migrated from extract_parameters.py) ---
def _column_letter_to_number(column_letter):
    if not column_letter or (isinstance(column_letter, float) and pd.isna(column_letter)):
        return None
    result = 0
    for char in str(column_letter).strip():
        result = result * 26 + (ord(char.upper()) - ord('A') + 1)
    return result

def _parse_column_list(column_string):
    if not column_string or (isinstance(column_string, float) and pd.isna(column_string)):
        return []
    columns = [col.strip() for col in str(column_string).split(',')]
    return [_column_letter_to_number(col) for col in columns if col]

def _parse_row_list(row_string):
    if not row_string or (isinstance(row_string, float) and pd.isna(row_string)):
        return []
    rows = [row.strip() for row in str(row_string).split(',')]
    return [int(row) for row in rows if row.isdigit()]

def _find_sheet_in_workbook(workbook, target_sheet_name):
    target_clean = str(target_sheet_name).strip()
    if target_sheet_name in workbook.sheetnames:
        return target_sheet_name
    for sheet_name in workbook.sheetnames:
        if sheet_name.strip() == target_clean:
            return sheet_name
    return None

def _extract_sheet_data(worksheet, header_rows, process_column, parameter_columns, value_columns):
    extracted_data = []
    all_columns = []
    if process_column:
        all_columns.append(process_column)
    all_columns.extend(parameter_columns)
    all_columns.extend(value_columns)
    all_columns = sorted(list(set(all_columns)))

    remark_row = None
    for row_num in range(worksheet.max_row, max(header_rows) if header_rows else 0, -1):
        for col_num in range(1, min(worksheet.max_column + 1, 6)):
            cell_value = worksheet.cell(row=row_num, column=col_num).value
            if cell_value and str(cell_value).strip() == "备注":
                remark_row = row_num
                break
        if remark_row:
            break

    for row_num in header_rows:
        if row_num <= worksheet.max_row:
            row_data = []
            for col_num in all_columns:
                if col_num <= worksheet.max_column:
                    cell_value = worksheet.cell(row=row_num, column=col_num).value
                    row_data.append("" if cell_value == "/" else (cell_value if cell_value is not None else ""))
                else:
                    row_data.append("")
            extracted_data.append(row_data)

    if header_rows:
        max_header_row = max(header_rows)
        end_row = remark_row if remark_row else worksheet.max_row
        for row_num in range(max_header_row + 1, end_row):
            row_data = []
            for col_num in all_columns:
                if col_num <= worksheet.max_column:
                    cell_value = worksheet.cell(row=row_num, column=col_num).value
                    row_data.append("" if cell_value == "/" else (cell_value if cell_value is not None else ""))
                else:
                    row_data.append("")
            extracted_data.append(row_data)

    filtered_data = []
    for row_data in extracted_data:
        if any(cell_value and str(cell_value).strip() for cell_value in row_data):
            filtered_data.append(row_data)
    return filtered_data, all_columns

def _remove_fubiao_tables(data):
    filtered_data = []
    i = 0
    while i < len(data):
        row_data = data[i]
        is_fubiao_start = any(
            cell_value and isinstance(cell_value, str) and "附表(" in str(cell_value)
            for cell_value in row_data
        )
        if is_fubiao_start:
            while i < len(data):
                row_data = data[i]
                if (
                    row_data and len(row_data) > 0 and row_data[0] and isinstance(row_data[0], str) and row_data[0].startswith("File:")
                ):
                    break
                i += 1
            continue
        filtered_data.append(row_data)
        i += 1
    return filtered_data

def extract_parameters_to_json(cp_session_dir, output_json_path, config_csv_path=None):
    cp_dir = Path(cp_session_dir)
    if not cp_dir.exists():
        raise FileNotFoundError(f"CP session dir not found: {cp_session_dir}")
    if config_csv_path is None:
        config_csv_path = str(cp_dir / "excel_sheets.csv")
    if not os.path.exists(config_csv_path):
        raise FileNotFoundError(
            f"Configuration CSV not found: {config_csv_path}. Place 'excel_sheets.csv' under {cp_session_dir}."
        )
    df = pd.read_csv(config_csv_path)
    false_sheets = df[df['skip'].astype(str).str.upper() == 'FALSE']
    json_data = []
    total_rows = 0
    for _, config in false_sheets.iterrows():
        file_name = str(config['file'])
        sheet_name = str(config['sheet']).strip()
        file_path = cp_dir / file_name
        if not file_path.exists():
            continue
        try:
            workbook = openpyxl.load_workbook(file_path)
            actual_sheet_name = _find_sheet_in_workbook(workbook, sheet_name)
            if not actual_sheet_name:
                continue
            worksheet = workbook[actual_sheet_name]
            header_rows = _parse_row_list(config.get('header_rows'))
            process_column = _column_letter_to_number(config.get('process_number_column')) if config.get('process_number_column') else None
            parameter_columns = _parse_column_list(config.get('parameter_columns'))
            value_columns = _parse_column_list(config.get('value_columns'))
            data, _ = _extract_sheet_data(worksheet, header_rows, process_column, parameter_columns, value_columns)
            if not data:
                continue
            filtered_data = _remove_fubiao_tables(data)
            total_rows += len(filtered_data)
            sheet_data = {"File": file_name, "Sheet": sheet_name, "data": filtered_data}
            json_data.append(sheet_data)
        except Exception:
            continue
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=2)
    return {"output": str(output_path), "sheets": len(json_data), "rows": total_rows, "config": config_csv_path}
