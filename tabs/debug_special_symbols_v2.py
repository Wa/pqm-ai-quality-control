"""Minimal Streamlit entry point to replay the final special-symbols comparison (v2 - deterministic)."""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG
from util import ensure_session_dirs

from tabs.special_symbols import estimate_tokens, log_llm_metrics, report_exception
from tabs.special_symbols.summaries import aggregate_outputs, persist_compare_outputs, summarize_with_ollama
from tabs.special_symbols.workflow import SPECIAL_SYMBOLS_WORKFLOW_SURFACE
from tabs.shared.modelscope_client import ModelScopeClient


SESSION_ID = "Jack Zhou"
NUM_CTX = 40001
NUM_RUNS = 5
COMPARISON_PREFIX = "examined_txt_filtered_final"
PROMPT_SOURCE_PATH = (
    Path(CONFIG["directories"]["generated_files"])
    / SESSION_ID
    / "special_symbols_check"
    / "initial_results"
    / "prompt_examined_txt_filtered_final_20251111_094117_comparison.txt"
)
MODELSCOPE_MODEL = "deepseek-ai/DeepSeek-V3.1"


def _prepare_paths():
    base_dirs = {"generated": str(CONFIG["directories"]["generated_files"])}
    session_dirs = ensure_session_dirs(base_dirs, SESSION_ID)
    paths = SPECIAL_SYMBOLS_WORKFLOW_SURFACE.prepare_paths(session_dirs)
    os.makedirs(paths.initial_results_dir, exist_ok=True)
    os.makedirs(paths.final_results_dir, exist_ok=True)
    return paths


def _load_prompt(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _init_modelscope_client() -> ModelScopeClient:
    """Initialize ModelScope client with temperature=0 for deterministic outputs."""
    modelscope_api_key = os.getenv("MODELSCOPE_API_KEY") or CONFIG["llm"].get("modelscope_api_key")
    if not modelscope_api_key:
        raise RuntimeError("ModelScope API Key not configured. Set MODELSCOPE_API_KEY or config.llm.modelscope_api_key")
    
    modelscope_base_url = CONFIG["llm"].get("modelscope_base_url") or "https://api-inference.modelscope.cn/v1"
    
    return ModelScopeClient(
        api_key=modelscope_api_key,
        model=MODELSCOPE_MODEL,
        base_url=modelscope_base_url,
        temperature=0.0,  # Deterministic output
    )


def _call_comparison_llm(
    prompt_text: str,
    client: ModelScopeClient,
    run_number: int,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    """Call ModelScope DeepSeek-V3.1 with temperature=0."""
    start_ts = time.time()
    log(f"运行 {run_number}/{NUM_RUNS}: 调用 ModelScope DeepSeek-V3.1…")
    
    try:
        resp = client.chat(
            model=MODELSCOPE_MODEL,
            messages=[{"role": "user", "content": prompt_text}],
            stream=False,
            options={"num_ctx": NUM_CTX},
        )
        
        response_text = (
            (resp.get("message", {}) or {}).get("content")
            or resp.get("response")
            or ""
        )
        
        if not response_text.strip():
            raise RuntimeError("Empty response from model")
        
        duration_ms = int((time.time() - start_ts) * 1000)
        log(f"运行 {run_number}/{NUM_RUNS}: 调用成功 (耗时: {duration_ms}ms)")
        
        return {
            "text": response_text,
            "model": resp.get("model") or MODELSCOPE_MODEL,
            "engine": "modelscope",
            "stats": resp.get("stats") or {},
            "duration_ms": duration_ms,
        }
    except Exception as error:
        report_exception(f"运行 {run_number}/{NUM_RUNS}: ModelScope 调用失败", error, level="error")
        raise


def run_debug_flow(log: Callable[[str], None]) -> Dict[str, Any]:
    """Run the comparison flow NUM_RUNS times."""
    run_started_at = time.time()
    paths = _prepare_paths()
    initial_dir = Path(paths.initial_results_dir)
    final_dir = Path(paths.final_results_dir)
    output_root = Path(paths.output_root)

    log(f"读取提示词文件：{PROMPT_SOURCE_PATH}")
    prompt_text = _load_prompt(PROMPT_SOURCE_PATH)
    
    log(f"初始化 ModelScope 客户端 (temperature=0)...")
    client = _init_modelscope_client()
    
    all_results = []
    
    # Run NUM_RUNS times
    for run_num in range(1, NUM_RUNS + 1):
        log(f"\n--- 开始运行 {run_num}/{NUM_RUNS} ---")
        
        try:
            response_data = _call_comparison_llm(prompt_text, client, run_num, log)
            
            raw_response = response_data.get("text", "")
            response_clean = raw_response.strip() or "无相关发现"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_name = f"{COMPARISON_PREFIX}_{timestamp}_run{run_num}_comparison"
            
            log(f"运行 {run_num}: 生成比对基名：{comparison_name}")

            log_llm_metrics(
                str(output_root),
                SESSION_ID,
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "engine": response_data.get("engine", ""),
                    "model": response_data.get("model", ""),
                    "session_id": SESSION_ID,
                    "file": comparison_name,
                    "part": 1,
                    "phase": "compare",
                    "prompt_chars": len(prompt_text),
                    "prompt_tokens": estimate_tokens(prompt_text, response_data.get("model", "")),
                    "output_chars": len(response_clean),
                    "output_tokens": estimate_tokens(response_clean, response_data.get("model", "")),
                    "duration_ms": response_data.get("duration_ms", 0),
                    "success": 1 if response_clean else 0,
                    "stats": response_data.get("stats") or {},
                    "error": "",
                },
            )

            persist_compare_outputs(str(initial_dir), comparison_name, [prompt_text], response_clean)
            log(f"运行 {run_num}: 已保存 prompt 与 response 文件。")

            summarize_with_ollama(str(initial_dir), str(output_root), SESSION_ID, comparison_name, response_clean)
            log(f"运行 {run_num}: 已生成结构化提示并请求 JSON 结果。")

            aggregate_outputs(str(initial_dir), str(output_root), SESSION_ID)
            log(f"运行 {run_num}: 已汇总生成 CSV/XLSX/Word 结果。")

            # Collect JSON files for this run
            json_files = sorted(
                str(initial_dir / name)
                for name in os.listdir(initial_dir)
                if name.startswith(f"json_{comparison_name}_pt") and name.endswith(".txt")
            )
            
            all_results.append({
                "run_number": run_num,
                "comparison_name": comparison_name,
                "response_length": len(response_clean),
                "json_files": json_files,
                "json_count": len(json_files),
            })
            
        except Exception as error:
            report_exception(f"运行 {run_num} 失败", error, level="error")
            log(f"运行 {run_num} 失败: {error}")
            all_results.append({
                "run_number": run_num,
                "error": str(error),
            })

    # Collect final files
    final_files = sorted(
        str(path)
        for path in final_dir.glob("*")
        if path.is_file() and os.path.getmtime(path) >= run_started_at
    )
    
    if final_files:
        log(f"\n本次运行生成的最终文件 ({len(final_files)} 个):")
        for fpath in final_files:
            log(f"  {fpath}")

    return {
        "ok": True,
        "num_runs": NUM_RUNS,
        "results": all_results,
        "final_files": final_files,
    }


def _render_result(container: Any, result: Dict[str, Any]) -> None:
    """Simplified result rendering."""
    if not result.get("ok"):
        container.error(result.get("error") or "运行失败")
        return

    container.success(f"已完成 {result['num_runs']} 次运行")
    
    # Summary table
    results = result.get("results", [])
    if results:
        container.markdown("### 运行结果摘要")
        summary_data = []
        for r in results:
            if "error" in r:
                summary_data.append({
                    "运行": r["run_number"],
                    "状态": "❌ 失败",
                    "错误": r["error"][:100],
                    "JSON文件数": 0,
                })
            else:
                summary_data.append({
                    "运行": r["run_number"],
                    "状态": "✅ 成功",
                    "响应长度": f"{r['response_length']:,} 字符",
                    "JSON文件数": r["json_count"],
                })
        
        container.table(summary_data)
    
    # Final files
    final_files = result.get("final_files", [])
    if final_files:
        container.markdown(f"### 最终结果文件 ({len(final_files)} 个)")
        for fpath in final_files:
            container.write(f"- `{fpath}`")


def main() -> None:
    st.set_page_config(page_title="Debug Special Symbols v2", layout="wide")
    st.title("特殊特性符号调试工具 v2")
    st.caption(f"使用 ModelScope DeepSeek-V3.1 (temperature=0) 运行 {NUM_RUNS} 次")
    st.write(f"提示词来源：`{PROMPT_SOURCE_PATH}`")

    log_container = st.container()
    result_container = st.container()

    if st.button("运行", type="primary"):
        log_container.empty()
        result_container.empty()
        log_lines: List[str] = []

        def log(message: str) -> None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            entry = f"[{timestamp}] {message}"
            log_lines.append(entry)
            log_container.write(entry)

        with st.spinner(f"执行 {NUM_RUNS} 次特殊特性符号对比流程…"):
            try:
                result = run_debug_flow(log)
            except Exception as error:
                report_exception("debug_special_symbols_v2 运行失败", error, level="error")
                log(str(error))
                result_container.error(f"运行失败：{error}")
            else:
                _render_result(result_container, result)


if __name__ == "__main__":
    main()


