"""Summary helpers for the parameters workflow."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import List

try:  # pragma: no cover - backend execution may not ship Streamlit
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None  # type: ignore

from config import CONFIG
from ollama import Client as OllamaClient

from util import resolve_ollama_host

from tabs.shared import report_exception, stream_text

from tabs.enterprise_standard.summaries import persist_compare_outputs


def _st_available() -> bool:
    """Return True when Streamlit runtime is importable and active."""

    return st is not None


def summarize_with_ollama(
    initial_dir: str,
    parameters_out: str,
    session_id: str,
    name_no_ext: str,
    full_out_text: str,
) -> None:
    """Convert chunk-level Bisheng answers into structured JSON via Ollama."""

    try:
        original_name = name_no_ext
        prompt_lines = [
            "你是一名严谨的参数差异整理助手。以下内容来自一个基于检索增强的大语言模型 (RAG-LLM) 系统，",
            "用于分析待检文件/图纸与基准文件之间的参数一致性。该部分原始技术文件名称为：",
            f"{original_name}。",
            "\n请将后续全文内容转换为 JSON 数组（list of objects），每个对象应包含以下键：",
            "- parameter：参数或特性名称；",
            "- target_value：待检文件或图纸中的取值/范围（未知时留空字符串）；",
            "- reference_value：基准文件中的取值/范围（未知时留空字符串）；",
            "- location：用于追溯的定位信息，例如目标文件/图纸文件名、Sheet 名称、段落；",
            "- issue：一句话描述发现的不一致或风险；",
            "- suggestion：针对该问题的修订或沟通建议。",
            "\n要求：",
            "1) 仅输出严格的 JSON（UTF-8，无注释、无额外文本），键名使用上面给出的英文；",
            "2) 若内容包含多处问题，请拆分为多个对象；",
            "3) 若某项信息缺失，请以空字符串 \"\" 占位，不要编造；",
            "4) 在 location 中保留可追溯的原文线索（如文件名、页码、Sheet 名称等）。",
            "\n下面是需要转换的原始比对输出：\n\n",
        ]
        instruction = "".join(prompt_lines)
        text = full_out_text or ""

        max_thinks_per_part = 20
        max_chars_per_part = 8000
        think_indices: list[int] = []
        needle = "<think>"
        start = 0
        while True:
            pos = text.find(needle, start)
            if pos == -1:
                break
            think_indices.append(pos)
            start = pos + len(needle)

        boundaries: list[int] = []
        prev = 0
        count_since_prev = 0
        for pos in think_indices:
            count_since_prev += 1
            if count_since_prev >= max_thinks_per_part or (pos - prev) >= max_chars_per_part:
                boundaries.append(pos)
                prev = pos
                count_since_prev = 0

        parts: list[str] = []
        prev = 0
        for boundary in boundaries:
            parts.append(text[prev:boundary])
            prev = boundary
        parts.append(text[prev:])

        try:
            for fname in os.listdir(initial_dir):
                if fname.startswith(f"prompted_response_{name_no_ext}_pt") and fname.endswith(".txt"):
                    os.remove(os.path.join(initial_dir, fname))
        except Exception:
            pass

        for index, part in enumerate(parts, start=1):
            cleaned_part = re.sub(r"<think>[\s\S]*?</think>", "", part)
            prompted_path = os.path.join(
                initial_dir, f"prompted_response_{name_no_ext}_pt{index}.txt"
            )
            with open(prompted_path, "w", encoding="utf-8") as handle:
                handle.write(instruction)
                handle.write(cleaned_part)
    except Exception as error:
        report_exception("生成参数汇总提示失败", error, level="warning")

    try:
        part_files: list[str] = []
        try:
            for fname in os.listdir(initial_dir):
                if fname.startswith(f"prompted_response_{name_no_ext}_pt") and fname.endswith(".txt"):
                    part_files.append(fname)
        except Exception as error:
            report_exception("读取参数汇总提示列表失败", error, level="warning")
        part_files.sort(key=lambda value: value.lower())
        total_parts = len(part_files)
        host = resolve_ollama_host("ollama_9")
        ollama_client = OllamaClient(host=host)
        model = CONFIG["llm"].get("ollama_model")
        temperature = 0.7
        top_p = 0.9
        top_k = 40
        repeat_penalty = 1.1
        num_ctx = 40001
        num_thread = 4
        show_ui = _st_available()
        for part_idx, pfname in enumerate(part_files, start=1):
            p_path = os.path.join(initial_dir, pfname)
            try:
                with open(p_path, "r", encoding="utf-8") as handle:
                    prompt_text_all = handle.read()
            except Exception as error:
                report_exception(f"读取汇总提示失败({pfname})", error, level="warning")
                prompt_text_all = ""

            base = pfname[len("prompted_response_") :]
            json_name = f"json_{base}"
            json_path = os.path.join(initial_dir, json_name)
            existing_json_text = ""
            if os.path.isfile(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as handle:
                        existing_json_text = handle.read()
                except Exception as error:
                    report_exception(f"读取已存在的JSON汇总失败({json_name})", error, level="warning")
                    existing_json_text = ""

            placeholder = None
            if show_ui:
                col_prompt2, col_resp2 = st.columns([1, 1])
                with col_prompt2:
                    st.markdown(f"生成参数汇总提示词（第{part_idx}部分，共{total_parts}部分）")
                    prompt_container2 = st.container(height=400)
                    with prompt_container2:
                        with st.chat_message("user"):
                            placeholder = st.empty()
                            stream_text(placeholder, prompt_text_all, render_method="text")
                    st.chat_input(
                        placeholder="",
                        disabled=True,
                        key=f"parameters_prompted_prompt_{session_id}_{pfname}",
                    )
                resp_column = col_resp2
            else:
                resp_column = None

            response_placeholder = None
            if show_ui and resp_column is not None:
                with resp_column:
                    st.markdown(f"生成参数汇总结果（第{part_idx}部分，共{total_parts}部分）")
                    resp_container2 = st.container(height=400)
                    with resp_container2:
                        with st.chat_message("assistant"):
                            response_placeholder = st.empty()
                            if existing_json_text:
                                response_placeholder.write(existing_json_text)
            try:
                result_text = ""
                for chunk in ollama_client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt_text_all}],
                    stream=True,
                    options={
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "repeat_penalty": repeat_penalty,
                        "num_ctx": num_ctx,
                        "num_thread": num_thread,
                    },
                ):
                    delta = chunk["message"]["content"]
                    if not isinstance(delta, str):
                        continue
                    result_text += delta
                    if response_placeholder is not None:
                        response_placeholder.write(result_text)
                if show_ui and response_placeholder is None and resp_column is not None:
                    with resp_column:
                        with st.chat_message("assistant"):
                            response_placeholder = st.empty()
                            response_placeholder.write(result_text)
            except Exception as error:
                report_exception("调用 Ollama 生成参数JSON失败", error)
                result_text = existing_json_text or ""

            try:
                with open(json_path, "w", encoding="utf-8") as handle:
                    handle.write(result_text)
            except Exception as error:
                report_exception(f"写入参数JSON失败({json_name})", error, level="warning")
    except Exception as error:
        report_exception("参数比对结果JSON生成失败", error, level="warning")


def aggregate_outputs(initial_dir: str, parameters_out: str, session_id: str) -> None:
    """Combine chunk-level JSON outputs into tab-level CSV/XLSX reports."""

    final_dir = os.path.join(parameters_out, "final_results")
    os.makedirs(final_dir, exist_ok=True)
    try:
        json_files = [
            os.path.join(initial_dir, fname)
            for fname in os.listdir(initial_dir)
            if fname.startswith("json_") and fname.lower().endswith(".txt")
        ]
    except Exception as error:
        report_exception("读取参数初始结果目录失败", error, level="warning")
        json_files = []

    columns = ["parameter", "target_value", "reference_value", "location", "issue", "suggestion"]
    rows = []

    try:
        import pandas as pd  # type: ignore
    except ImportError as error:
        pd = None  # type: ignore[assignment]
        if _st_available():
            st.info(f"未安装 pandas，Excel 导出将被跳过：{error}")
        else:
            report_exception("未安装 pandas，Excel 导出将被跳过", error, level="warning")

    def _strip_code_fences(value: str) -> str:
        value = (value or "").strip()
        if value.startswith("```") and value.endswith("```"):
            value = value[3:-3].strip()
            if value.lower().startswith("json"):
                value = value[4:].strip()
        return value

    def _extract_json_text(text: str) -> str:
        s = (text or "").strip().lstrip("\ufeff")
        s = _strip_code_fences(s)
        try:
            json.loads(s)
            return s
        except json.JSONDecodeError:
            pass
        try:
            start = s.find("[")
            end = s.rfind("]")
            if start != -1 and end != -1 and end > start:
                candidate = s[start : end + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    merged = re.sub(r"\]\s*,?\s*\[", ",", candidate)
                    json.loads(merged)
                    return merged
        except json.JSONDecodeError:
            pass
        try:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = s[start : end + 1]
                json.loads(candidate)
                return candidate
        except json.JSONDecodeError:
            pass
        return s

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as handle:
                raw = handle.read()
        except Exception as error:
            report_exception(f"读取参数JSON失败({os.path.basename(jf)})", error, level="warning")
            continue

        parsed_text = _extract_json_text(raw)
        try:
            data = json.loads(parsed_text)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            continue

        for row in data:
            if not isinstance(row, dict):
                continue
            rows.append(
                [
                    str(row.get("parameter", "")),
                    str(row.get("target_value", "")),
                    str(row.get("reference_value", "")),
                    str(row.get("location", "")),
                    str(row.get("issue", "")),
                    str(row.get("suggestion", "")),
                ]
            )

    csv_path: str | None = None
    xlsx_path: str | None = None

    if rows:
        csv_path = os.path.join(final_dir, f"设计制程参数检查结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        try:
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
                import csv

                writer = csv.writer(handle)
                writer.writerow(["参数名称", "目标取值", "基准取值", "定位信息", "问题描述", "整改建议"])
                writer.writerows(rows)
        except Exception as error:
            report_exception("写入参数检查CSV失败", error)
            csv_path = None

    if rows and "pd" in locals() and pd is not None:
        try:
            df = pd.DataFrame(rows, columns=["参数名称", "目标取值", "基准取值", "定位信息", "问题描述", "整改建议"])
            xlsx_path = os.path.join(final_dir, f"设计制程参数检查结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
            df.to_excel(xlsx_path, index=False)
        except Exception as error:
            report_exception("写入参数检查Excel失败", error, level="warning")
            xlsx_path = None

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "row_count": len(rows),
        "csv_path": csv_path,
        "xlsx_path": xlsx_path,
    }
    try:
        with open(os.path.join(final_dir, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
    except Exception as error:
        report_exception("写入参数结果manifest失败", error, level="warning")


__all__ = ["aggregate_outputs", "persist_compare_outputs", "summarize_with_ollama"]
