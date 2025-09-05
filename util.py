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
                "special_symbols_check",
                "parameters_check",
                "file_elements_check",
                "file_completeness_check",
                "history_issues_avoidance",
            ]
            for name in subfolders:
                os.makedirs(os.path.join(generated_root, name), exist_ok=True)
            # Convenience keys
            session_dirs["generated_special_symbols_check"] = os.path.join(generated_root, "special_symbols_check")
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

    return session_dirs

# --- File Upload Utility ---
def handle_file_upload(files, save_dir):
    if files:
        for file in files:
            save_path = os.path.join(save_dir, file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
        return len(files)
    return 0

# --- Prompt Generation Logic (from prompt_generation_v2.py) ---
class PromptGenerator:
    """
    Takes in control plan (CP) files and target file(s) to be checked against the CP files, 
    and generates user prompts to be used in chats with large language model. 
    The prompts will be saved in a .txt file.
    
    The class processes Excel files to extract special characteristic symbols (★, ☆, /) 
    and generates prompts for LLM-based consistency checking between control plans and target files.
    """
    
    def __init__(self, target_symbols=None, filter_terms=None):
        self.target_symbols = target_symbols or ['★', '☆', '/']
        self.filter_terms = filter_terms or ["过程", "CP", "过程检验", "修订日期", "A0", "版本", "编制日期", "涂布", "合浆", "盖板", "外观", "面密度", "绝缘", "装配", "辊压", "壳体", "铝箔", "铜箔"]
        self.filled_sheets = {}
        
    def clean_cell_content(self, value):
        """Clean cell content by removing newlines and extra whitespace."""
        if isinstance(value, str):
            cleaned = value.replace('\n', ' ').replace('\r', ' ').replace('\r\n', ' ')
            cleaned = ' '.join(cleaned.split())
            return cleaned
        return value

    def fill_merged_cells(self, file_path):
        """Fill merged cells in Excel sheets and return processed dataframes."""
        wb = load_workbook(file_path)
        filled_sheets = {}
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            for row_idx in range(len(df)):
                for col_idx in range(len(df.columns)):
                    df.iloc[row_idx, col_idx] = self.clean_cell_content(df.iloc[row_idx, col_idx])
            merged_ranges = ws.merged_cells.ranges
            for merged_range in merged_ranges:
                top_left_cell = ws[merged_range.coord.split(':')[0]]
                value = self.clean_cell_content(top_left_cell.value)
                if value is not None and value != '':
                    min_row = merged_range.min_row - 1
                    max_row = merged_range.max_row - 1
                    min_col = merged_range.min_col - 1
                    max_col = merged_range.max_col - 1
                    for row in range(min_row, max_row + 1):
                        for col in range(min_col, max_col + 1):
                            if row < len(df) and col < len(df.columns):
                                df.iloc[row, col] = value
            filled_sheets[sheet_name] = df
        return filled_sheets

    def is_single_number(self, text):
        """Check if text is a single digit."""
        return text.isdigit()

    def is_single_symbol(self, text):
        """Check if text is a single non-alphanumeric symbol."""
        return len(text) == 1 and not text.isalnum()

    def should_filter_term(self, term):
        """Determine if a term should be filtered out."""
        if not term or not term.strip(): 
            return True
        cleaned = term.strip()
        if cleaned.lower() == 'nan': 
            return True
        if self.is_single_number(cleaned) or self.is_single_symbol(cleaned): 
            return True
        if cleaned in self.filter_terms: 
            return True
        return False

    def get_sheet_rows_and_terms(self, file_path):
        """Extract rows with target symbols and relevant terms from Excel sheets."""
        file_name = os.path.basename(file_path)
        file_name_formatted = f"文件名：《{file_name}》"
        self.filled_sheets = self.fill_merged_cells(file_path)
        sheet_rows = {}
        sheet_terms = {}
        for sheet_name, df in self.filled_sheets.items():
            if "特殊特性符号对照表" in sheet_name or "标准选项" in sheet_name:
                continue
            if len(df.columns) >= 2:
                df = df.iloc[:, :-2]
            rows = []
            terms_set = set()
            for row_idx in range(len(df)):
                row = df.iloc[row_idx].tolist()
                if not any(str(cell).strip() in self.target_symbols for cell in row):
                    continue
                row_with_meta = [file_name_formatted, f"sheet: {sheet_name}"] + [str(cell) if cell is not None else '' for cell in row]
                terms = [t.strip() for t in row_with_meta[2:] if not self.should_filter_term(t)]
                if terms:
                    terms_set.update(terms)
                rows.append(row_with_meta)
            if rows and terms_set:
                sheet_rows[sheet_name] = rows
                sheet_terms[sheet_name] = list(terms_set)
        return sheet_rows, sheet_terms, file_name

    def search_terms_batch(self, directory, terms):
        """Search for terms across all Excel files in a directory."""
        dir_path = Path(directory)
        excel_files = [f for f in dir_path.glob("*.xlsx") if not f.name.startswith('~$')]
        results_by_term = {term: [] for term in terms}
        import re
        pattern = '|'.join(re.escape(term) for term in terms)
        for excel_file in excel_files:
            try:
                xl = pd.ExcelFile(excel_file)
                for sheet in xl.sheet_names:
                    try:
                        df = pd.read_excel(excel_file, sheet_name=sheet).astype(str)
                        for col in df.columns:
                            matches = df[df[col].str.contains(pattern, na=False, regex=True)]
                            if not matches.empty:
                                for idx, row in matches.iterrows():
                                    for term in terms:
                                        if term in str(row[col]):
                                            row_data = {}
                                            for column in df.columns:
                                                val = str(row[column])
                                                if (not str(column).startswith('Unnamed:') or val != 'nan') and len(val) <= 20:
                                                    row_data[column] = row[column]
                                            results_by_term[term].append({
                                                'file': excel_file.name,
                                                'sheet': sheet,
                                                'row': idx + 2,
                                                'row_data': row_data,
                                                'search_term': term
                                            })
                    except Exception:
                        continue
            except Exception:
                continue
        return results_by_term

    def target_sheet_content_json(self, file_name):
        """Convert target sheet content to JSON format."""
        content = []
        for sheet_name, df in self.filled_sheets.items():
            if "特殊特性符号对照表" in sheet_name or "标准选项" in sheet_name:
                continue
            if len(df.columns) >= 2:
                df = df.iloc[:, :-2]
            rows = []
            for row_idx in range(len(df)):
                row = df.iloc[row_idx].tolist()
                if not any(str(cell).strip() in self.target_symbols for cell in row):
                    continue
                row_clean = [str(cell) for cell in row if str(cell).lower() != 'nan']
                if row_clean:
                    rows.append(row_clean)
            if rows:
                content.append({"sheet": sheet_name, "rows": rows})
        return {"target file name": file_name, "target_sheet_content": content}

    def write_sheet_results_to_file(self, sheet_name, sheet_rows, sheet_results, output_file, file_name=None):
        """Write sheet analysis results to the output file."""
        if not sheet_rows or not sheet_results:
            return
        with open(output_file, 'a', encoding='utf-8') as file:
            file.write("我需要你根据控制计划验证一个目标文件中的特殊特性符号（/、 ★和☆）：\n\n")
            file.write("任务：\n")
            file.write("1.检查目标文件内容中每一行带有特殊特性符号的项目\n")
            file.write("2. 对于每个参数，搜索控制计划文件中相似/相同的参数\n")
            file.write("3. 比较符号分类并识别任何不一致之处\n")
            file.write("4. 仅在找到可比较示例时提供基于证据的结论\n\n")
            file.write("关键要求：\n")
            file.write("- 展示你的证据：引用确切的文件名、sheet名、参数名称及其分类\n")
            file.write("- 如果在控制计划中找不到相似参数，明确说明\"未找到可比较的证据\"\n")
            file.write("- 不要基于一般工程知识做出建议\n")
            file.write("- 在Excel文件中，'/'有时表示一般特性，有时表示该格无内容，你需要灵活辨别。\n\n")
            file.write("交付成果：\n")
            file.write("- 发现的具体不一致之处\n")
            file.write("- 来自控制计划的证据（如可获得）\n\n")
            if file_name is not None:
                file.write("以下是目标文件一个sheet的内容，我已对其进行了数据预处理(JSON)：\n")
                target_json = self.target_sheet_content_json(file_name)
                file.write(json.dumps(target_json, ensure_ascii=False, indent=2))
                file.write("\n\n")
            file.write("以下是控制计划中可能相关的内容，我同样对其进行了数据预处理(JSON)：\n")
            grouped = {}
            seen = set()
            for term, results in sheet_results.items():
                for result in results:
                    key = (result['file'], result['sheet'], result['row'])
                    if key in seen:
                        continue
                    seen.add(key)
                    file_name2 = result['file']
                    sname = result['sheet']
                    row_number = result['row']
                    row_values = [str(v).replace('\n', ' ').replace('\r', ' ') for v in result['row_data'].values() if str(v).lower() != 'nan']
                    if file_name2 not in grouped:
                        grouped[file_name2] = {}
                    if sname not in grouped[file_name2]:
                        grouped[file_name2][sname] = []
                    grouped[file_name2][sname].append({
                        'row': row_number,
                        'content': row_values
                    })
            json_results = []
            for file_name2, sheets in grouped.items():
                file_entry = {'file': file_name2, 'sheets': []}
                for sname, rows in sheets.items():
                    file_entry['sheets'].append({'sheet': sname, 'rows': rows})
                json_results.append(file_entry)
            file.write(json.dumps({'results': json_results}, ensure_ascii=False, indent=2))
            file.write("\n" + "="*80 + "\n\n")

    def generate_prompt(self, control_plan_dir, target_file, output_file):
        """
        Generate prompts for LLM-based consistency checking.
        
        Args:
            control_plan_dir: Directory containing control plan files
            target_file: Path to the target file to be checked
            output_file: Path where the generated prompts will be saved (.txt file)
        
        Returns:
            Path to the generated output file
        """
        sheet_rows, sheet_terms, file_name = self.get_sheet_rows_and_terms(target_file)
        with open(output_file, 'w', encoding='utf-8') as f: 
            pass
        for sheet_name in sheet_rows:
            sheet_results = self.search_terms_batch(control_plan_dir, sheet_terms[sheet_name])
            self.write_sheet_results_to_file(sheet_name, sheet_rows[sheet_name], sheet_results, output_file, file_name=file_name)
        return output_file 

# --- Multi-User Support (Simple Username-Based) ---

def get_user_sessions_dir():
    """Get the directory for storing user session files."""
    sessions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_sessions")
    os.makedirs(sessions_dir, exist_ok=True)
    return sessions_dir

def get_user_session_file(username):
    """Get the path to a specific user's session file."""
    sessions_dir = get_user_sessions_dir()
    return os.path.join(sessions_dir, f"{username}_session.json")

def load_user_session(username):
    """Load user session data from JSON file if it exists."""
    try:
        session_file = get_user_session_file(username)
        if os.path.exists(session_file):
            with open(session_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return None

def create_user_session(username):
    """Create a new user session with simple username-based approach."""
    session_data = {
        "username": username,
        "login_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_activity": time.strftime("%Y-%m-%d %H:%M:%S"),
        "active": True
    }
    
    # Save the session
    if save_user_session(username, session_data):
        return session_data
    return None

def save_user_session(username, session_data):
    """Save user session data to a JSON file."""
    try:
        session_file = get_user_session_file(username)
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"保存用户会话失败: {e}")
        return False

def update_user_activity(username):
    """Update the last activity timestamp for a user session."""
    session_data = load_user_session(username)
    if session_data:
        session_data["last_activity"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_user_session(username, session_data)

def deactivate_user_session(username):
    """Mark a user session as inactive (for logout)."""
    session_data = load_user_session(username)
    if session_data:
        session_data["active"] = False
        session_data["logout_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_user_session(username, session_data)

def get_active_users():
    """Get list of currently active users."""
    sessions_dir = get_user_sessions_dir()
    active_users = []
    
    if not os.path.exists(sessions_dir):
        return active_users
    
    for filename in os.listdir(sessions_dir):
        if filename.endswith('_session.json'):
            username = filename.replace('_session.json', '')
            session_data = load_user_session(username)
            if session_data and session_data.get('active', False):
                active_users.append(username)
    
    return active_users

def render_login_widget():
    """Render login widget using URL params for per-browser username and auth flag."""
    # Read browser-specific params
    current_user = st.query_params.get("user", None)
    auth = st.query_params.get("auth", None)

    # If auth flag is present, auto-login this browser
    if current_user and auth == "1":
        session_data = load_user_session(current_user)
        if not session_data or not session_data.get('active', False):
            create_user_session(current_user)
        return current_user

    # Show login form (no global server-side prefill)
    st.markdown("### 用户登录")
    st.markdown("使用OA用户名和密码，不用加@calb-tech.com后缀")

    # Prefill from this browser's query param only
    prefill_username = current_user or ""

    with st.form("login_form"):
        username = st.text_input("用户名", value=prefill_username, placeholder="请输入您的用户名")
        password = st.text_input("密码", type="password", placeholder="请输入密码")

        col1, col2 = st.columns([1, 1])
        with col1:
            login_button = st.form_submit_button("登录")
        with col2:
            # Allow clearing just this browser's remembered username
            if current_user:
                clear_button = st.form_submit_button("清除本机用户名")
            else:
                clear_button = st.form_submit_button("清除", disabled=True)

        if login_button:
            if username.strip():
                # Accept any password (as requested)
                st.session_state['logged_in'] = True
                st.session_state['username'] = username.strip()

                # Create user session for multi-user support
                create_user_session(username.strip())

                # Pin username and auth to this browser via query params
                st.query_params["user"] = username.strip()
                st.query_params["auth"] = "1"

                st.success(f"欢迎，{username.strip()}！")
                st.rerun()
            else:
                st.error("请输入用户名")

        if clear_button:
            try:
                if "user" in st.query_params:
                    del st.query_params["user"]
                if "auth" in st.query_params:
                    del st.query_params["auth"]
                st.success("已清除本机用户名")
                st.rerun()
            except Exception:
                st.error("清除失败")

    return None

# --- Structured Session State Management ---
def get_user_session(session_id, tab_name=None):
    """Get or create a structured user session with optional tab-specific state."""
    if 'user_sessions' not in st.session_state:
        st.session_state.user_sessions = {}
    
    if session_id not in st.session_state.user_sessions:
        st.session_state.user_sessions[session_id] = {
            # Tab-specific workflow state
            'tabs': {
                'special_symbols': {
                    'process_started': False,
                    'analysis_completed': False,
                    'ollama_history': [],
                    'openai_history': [],
                },
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
                        'openai_history': []
                    }
            },
            
            # Shared settings (global)
            'llm_backend': 'ollama_9',
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
            'openai_logit_bias': '{}'
        }
    
    session = st.session_state.user_sessions[session_id]
    
    # If tab_name is provided, return tab-specific state
    if tab_name and tab_name in session['tabs']:
        return session['tabs'][tab_name]
    
    # Otherwise return the full session (for backward compatibility)
    return session

def reset_user_session(session_id, tab_name=None):
    """Reset a user's session to initial state, optionally for a specific tab."""
    session = st.session_state.user_sessions[session_id]
    
    if tab_name and tab_name in session['tabs']:
        # Reset specific tab
        tab_session = session['tabs'][tab_name]
        tab_session['analysis_completed'] = False
        tab_session['process_started'] = False
        tab_session['ollama_history'] = []
        tab_session['openai_history'] = []
    else:
        # Reset all tabs (backward compatibility)
        for tab_session in session['tabs'].values():
            tab_session['analysis_completed'] = False
            tab_session['process_started'] = False
            tab_session['ollama_history'] = []
            tab_session['openai_history'] = []

def start_analysis(session_id, tab_name):
    """Start analysis for a specific tab in a user session."""
    session = st.session_state.user_sessions[session_id]
    if tab_name in session['tabs']:
        tab_session = session['tabs'][tab_name]
        tab_session['analysis_completed'] = False
        tab_session['process_started'] = True
        tab_session['ollama_history'] = []
        tab_session['openai_history'] = []

def complete_analysis(session_id, tab_name):
    """Mark analysis as completed for a specific tab in a user session."""
    session = st.session_state.user_sessions[session_id]
    if tab_name in session['tabs']:
        session['tabs'][tab_name]['analysis_completed'] = True

# --- Prompt Generator and File Handling ---
class SimplePromptGenerator:
    """Generate prompts for different analysis types.

    Note: Renamed from PromptGenerator to avoid colliding with the
    earlier PromptGenerator class used by the special symbols workflow.
    """
    
    def __init__(self, session_id):
        self.session_id = session_id
    
    def generate_special_symbols_prompt(self, control_plan_content, file_content):
        """Generate prompt for special symbols analysis."""
        return f"""请分析以下控制计划文件和待检查文件中的特殊符号使用情况：

控制计划文件内容：
{control_plan_content}

待检查文件内容：
{file_content}

请检查：
1. 特殊符号的使用是否一致
2. 是否有遗漏的特殊符号
3. 特殊符号的格式是否正确
4. 是否符合行业标准

请提供详细的分析报告。"""

    def generate_prompt(self, control_plan_dir, target_file, output_file):
        """Compatibility wrapper used by tabs expecting generate_prompt()."""
        delegate = PromptGenerator()
        return delegate.generate_prompt(control_plan_dir, target_file, output_file)

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
