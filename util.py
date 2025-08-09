import os
import pandas as pd
from openpyxl import load_workbook
from pathlib import Path
import json
import streamlit as st
from config import CONFIG

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
    if 'user_session_id' not in st.session_state:
        # Create session ID that uses only username for folder names
        # This ensures users can access their files from any computer
        session_id = username
        st.session_state['user_session_id'] = session_id
    return st.session_state['user_session_id']

# --- Session Directory Utilities ---
def ensure_session_dirs(base_dirs, session_id):
    """Ensure session directories exist. Handle None session_id gracefully."""
    if session_id is None:
        # Return empty dict if session_id is None (user not logged in)
        return {}
    
    session_dirs = {}
    for key, base_dir in base_dirs.items():
        session_dir = os.path.join(base_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        session_dirs[key] = session_dir
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

def render_login_widget():
    """Render the login widget and return username if logged in."""
    # Initialize login state
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    
    # Auto-login if user has saved username (for page refresh convenience)
    if not st.session_state['logged_in'] and not st.session_state['username']:
        saved_username = load_username()
        if saved_username:
            # Auto-login with saved username
            st.session_state['logged_in'] = True
            st.session_state['username'] = saved_username
            st.rerun()
    
    # If already logged in, show logout option
    if st.session_state['logged_in'] and st.session_state['username']:
        return st.session_state['username']
    
    # Show login form
    st.markdown("### 用户登录")
    st.markdown("使用OA用户名和密码，不用加@calb-tech.com后缀")
    
    # Load saved username
    saved_username = load_username()
    
    # Login form
    with st.form("login_form"):
        username = st.text_input("用户名", value=saved_username or "", placeholder="请输入您的用户名")
        password = st.text_input("密码", type="password", placeholder="请输入密码")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            login_button = st.form_submit_button("登录")
        with col2:
            if saved_username:
                clear_button = st.form_submit_button("清除保存的用户名")
            else:
                clear_button = st.form_submit_button("清除", disabled=True)
        
        if login_button:
            if username.strip():
                # Accept any password (as requested)
                st.session_state['logged_in'] = True
                st.session_state['username'] = username.strip()
                save_username(username.strip())
                st.success(f"欢迎，{username.strip()}！")
                st.rerun()
            else:
                st.error("请输入用户名")
        
        if clear_button:
            try:
                os.remove(get_username_file())
                st.success("已清除保存的用户名")
                st.rerun()
            except:
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
                }
            },
            
            # Shared settings (global)
            'llm_backend': 'ollama',
            'ollama_model': CONFIG["llm"]["ollama_model"],
            'ollama_temperature': 0.7,
            'ollama_top_p': 0.9,
            'ollama_top_k': 40,
            'ollama_repeat_penalty': 1.1,
            'ollama_num_ctx': 40000,
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
