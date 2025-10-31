"""Background job implementation for the file completeness workflow."""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from ollama import Client as OllamaClient

from config import CONFIG
from util import resolve_ollama_host

from .stages import STAGE_ORDER, STAGE_REQUIREMENTS, STAGE_SLUG_MAP


def _clear_directory(directory: str) -> int:
    if not os.path.isdir(directory):
        return 0
    removed = 0
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
                removed += 1
            elif os.path.isdir(path):
                import shutil

                shutil.rmtree(path)
                removed += 1
        except Exception:
            continue
    return removed


def parse_llm_table_response(response_text: str) -> List[Dict[str, str]]:
    """Parse free-form LLM output into structured rows."""

    if not response_text:
        return []
    table_data: List[Dict[str, str]] = []
    lines = response_text.split("\n")
    in_table = False
    for line in lines:
        line = line.strip()
        if "应包含的交付物文件清单" in line and (
            "存在" in line or "是" in line or "否" in line
        ):
            in_table = True
            continue
        if not line or not in_table:
            continue
        if "|" in line:
            parts = [part.strip() for part in line.split("|")]
            if len(parts) >= 2:
                filename = parts[0].strip()
                status = parts[1].strip()
                if filename and status in {"是", "否"}:
                    table_data.append({"filename": filename, "status": status})
        else:
            match = re.search(r"[：:]\s*(是|否)", line)
            if match:
                status = match.group(1)
                filename = line[: match.start()].strip()
                if filename:
                    table_data.append({"filename": filename, "status": status})
    return table_data


def generate_stage_prompt(stage_name: str, stage_folder: str, requirements: Iterable[str]) -> str:
    if not os.path.exists(stage_folder):
        return f"{stage_name}文件夹不存在"
    actual_files: List[str] = []
    try:
        actual_files = [
            f
            for f in os.listdir(stage_folder)
            if os.path.isfile(os.path.join(stage_folder, f))
        ]
    except Exception:
        actual_files = []
    actual_listing = "\n".join(actual_files) if actual_files else "（无文件）"
    requirements_listing = "\n".join(str(item) for item in requirements)
    prompt = f"""{stage_name}应包含的文件包括
{requirements_listing}

{stage_name}文件夹中已有的文件清单包括
{actual_listing}

对比{stage_name}应包含的文件清单和{stage_name}文件夹中已有的文件清单，做匹配判断（允许合理的名称近似，例如“历史问题规避清单”≈“副本 LL-lesson learn-历史问题规避-V9.4.xlsx”）。

请只输出一个JSON对象，严格符合以下结构，不要输出任何额外文本（不要有解释、markdown或其他字符）：
{{
  "stage": "{stage_name}",
  "items": [
    {{
      "name": "<应包含的交付物文件名>",
      "exists": true|false,
      "matched_file": "<若exists=true，请填写在该阶段文件夹中匹配到的实际文件名；若不存在则填空字符串>",
      "note": "<可选：关于该行的说明/备注；若无则填空字符串>"
    }}
    // 针对应包含清单中的每一项都输出一条
  ]
}}

要求：
- items必须覆盖“应包含的交付物文件清单”中的每一项，且只出现一次；
- exists为布尔类型；
- 仅输出上述JSON对象本身。"""
    return prompt


def _ensure_dirs(session_id: str) -> Tuple[Dict[str, str], str, str]:
    uploads_root = str(CONFIG["directories"]["uploads"])
    generated_root = str(CONFIG["directories"]["generated_files"])
    stage_base = os.path.join(uploads_root, session_id, "file_completeness")
    os.makedirs(stage_base, exist_ok=True)
    stage_dirs: Dict[str, str] = {}
    for stage_name in STAGE_ORDER:
        slug = STAGE_SLUG_MAP.get(stage_name, stage_name)
        stage_path = os.path.join(stage_base, slug)
        os.makedirs(stage_path, exist_ok=True)
        stage_dirs[stage_name] = stage_path
    output_root = os.path.join(generated_root, session_id, "file_completeness_check")
    initial_dir = os.path.join(output_root, "initial_results")
    final_dir = os.path.join(output_root, "final_results")
    os.makedirs(initial_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    return stage_dirs, initial_dir, final_dir


def _save_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _load_custom_requirements(root: str) -> Dict[str, Tuple[str, ...]]:
    overrides_path = os.path.join(root, "custom_requirements.json")
    overrides: Dict[str, Tuple[str, ...]] = {}
    if not os.path.isfile(overrides_path):
        return overrides
    try:
        with open(overrides_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return overrides
    stages = data.get("stages") if isinstance(data, dict) else None
    if not isinstance(stages, dict):
        return overrides
    for stage_name, items in stages.items():
        if not isinstance(stage_name, str):
            continue
        if isinstance(items, list):
            cleaned_list = [str(item).strip() for item in items if str(item).strip()]
        elif isinstance(items, str):
            cleaned_list = [part.strip() for part in items.splitlines() if part.strip()]
        else:
            cleaned_list = []
        overrides[stage_name] = tuple(cleaned_list)
    return overrides


def _build_excel_rows(
    stage_responses: Dict[str, str],
    requirements_overrides: Optional[Dict[str, Tuple[str, ...]]] = None,
) -> Dict[str, List[Dict[str, str]]]:
    requirements_overrides = requirements_overrides or {}
    all_stage_data: Dict[str, List[Dict[str, str]]] = {}
    for stage_name in STAGE_ORDER:
        response_text = stage_responses.get(stage_name)
        if response_text:
            parsed_ok = False
            try:
                data = json.loads(response_text)
                if isinstance(data, dict) and isinstance(data.get("items"), list):
                    table_data: List[Dict[str, str]] = []
                    for item in data["items"]:
                        name = str(item.get("name", "")).strip()
                        if not name:
                            continue
                        exists_raw = item.get("exists")
                        if isinstance(exists_raw, bool):
                            exists = exists_raw
                        else:
                            s = str(exists_raw).strip().lower()
                            exists = s in {"true", "1", "yes", "y", "是", "存在"}
                        matched_file = str(item.get("matched_file", "") or "").strip()
                        note = str(item.get("note", "") or "").strip()
                        if not exists:
                            matched_file = ""
                        table_data.append(
                            {
                                "filename": name,
                                "status": "是" if exists else "否",
                                "matched_file": matched_file,
                                "note": note,
                            }
                        )
                    all_stage_data[stage_name] = table_data
                    parsed_ok = True
            except Exception:
                parsed_ok = False
            if not parsed_ok:
                fallback_rows = parse_llm_table_response(response_text)
                for row in fallback_rows:
                    row.setdefault("matched_file", "")
                    row.setdefault("note", "")
                all_stage_data[stage_name] = fallback_rows
        else:
            if stage_name in requirements_overrides:
                requirements = tuple(requirements_overrides.get(stage_name, ()))
            else:
                requirements = STAGE_REQUIREMENTS.get(stage_name, ())
            all_stage_data[stage_name] = [
                {
                    "filename": req,
                    "status": "否",
                    "matched_file": "",
                    "note": "",
                }
                for req in requirements
            ]
    return all_stage_data


def export_completeness_results(
    session_id: str,
    stage_responses: Dict[str, str],
    final_results_dir: str,
    requirements_overrides: Optional[Dict[str, Tuple[str, ...]]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    data = _build_excel_rows(stage_responses, requirements_overrides=requirements_overrides)
    rows: List[Dict[str, str]] = []
    for stage_name in STAGE_ORDER:
        for item in data.get(stage_name, []):
            rows.append(
                {
                    "Stage": stage_name,
                    "Deliverable": item.get("filename", ""),
                    "Exists": item.get("status", ""),
                    "FileName": item.get("matched_file", ""),
                    "Notes": item.get("note", ""),
                }
            )
    try:
        df = pd.DataFrame(rows, columns=["Stage", "Deliverable", "Exists", "FileName", "Notes"])
    except Exception:
        return None, None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"file_completeness_results_{session_id}_{timestamp}.xlsx"
    os.makedirs(final_results_dir, exist_ok=True)
    filepath = os.path.join(final_results_dir, filename)
    try:
        df.to_excel(filepath, index=False)
        return filepath, filename
    except Exception:
        return None, None


def run_file_completeness_job(
    session_id: str,
    publish: Callable[[Dict[str, object]], None],
    check_control: Optional[Callable[[], Dict[str, bool]]] = None,
) -> Dict[str, List[str]]:
    publish({"status": "running", "stage": "initializing", "message": "准备文件齐套性检查目录"})
    stage_dirs, initial_dir, final_dir = _ensure_dirs(session_id)
    overrides_root = os.path.dirname(initial_dir)
    requirements_overrides = _load_custom_requirements(overrides_root)
    removed = _clear_directory(initial_dir)
    if removed:
        publish(
            {
                "log": {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "level": "info",
                    "message": f"已清空历史提示词与响应 {removed} 个文件",
                }
            }
        )
    publish({"progress": 5.0})
    total_stages = len(STAGE_ORDER)
    publish({"total_chunks": total_stages, "processed_chunks": 0})

    control_state = "running"

    def ensure_running(stage: str, detail: str) -> bool:
        nonlocal control_state
        if not check_control:
            publish({"status": "running", "stage": stage, "message": detail})
            return True
        stage_value = stage or "running"
        while True:
            status = check_control() or {"paused": False, "stopped": False}
            if status.get("stopped"):
                publish(
                    {
                        "status": "failed",
                        "stage": "stopped",
                        "message": "任务已被用户停止",
                    }
                )
                return False
            if status.get("paused"):
                if control_state != "paused":
                    control_state = "paused"
                publish(
                    {
                        "status": "paused",
                        "stage": "paused",
                        "message": f"暂停中：等待恢复（{detail}）",
                    }
                )
                time.sleep(1)
                continue
            if control_state != "running":
                control_state = "running"
            publish({"status": "running", "stage": stage_value, "message": detail})
            return True

    try:
        host = resolve_ollama_host("ollama_9")
        client = OllamaClient(host=host)
        model_name = CONFIG["llm"].get("ollama_model", "gpt-oss:latest")
    except Exception as error:
        publish(
            {
                "status": "failed",
                "stage": "initializing",
                "message": f"初始化Ollama客户端失败: {error}",
                "error": str(error),
            }
        )
        return {"final_results": []}

    stage_responses: Dict[str, str] = {}
    processed = 0
    progress = 5.0
    progress_step = 80.0 / total_stages if total_stages else 0.0

    for stage_name in STAGE_ORDER:
        if not ensure_running(stage_name, f"准备分析{stage_name}"):
            return {"final_results": []}
        stage_dir = stage_dirs.get(stage_name, "")
        stage_files = []
        try:
            stage_files = [
                f for f in os.listdir(stage_dir) if os.path.isfile(os.path.join(stage_dir, f))
            ]
        except Exception:
            stage_files = []
        if stage_name in requirements_overrides:
            stage_requirements = requirements_overrides.get(stage_name, ())
        else:
            stage_requirements = STAGE_REQUIREMENTS.get(stage_name, ())
        if not stage_files:
            stage_responses[stage_name] = ""
            processed += 1
            publish({"processed_chunks": processed})
            progress = min(95.0, progress + progress_step)
            publish({"progress": progress})
            continue
        prompt = generate_stage_prompt(stage_name, stage_dir, stage_requirements)
        prompt_path = os.path.join(initial_dir, f"prompt_{stage_name}.txt")
        _save_text(prompt_path, prompt)
        publish(
            {
                "stream": {
                    "kind": "prompt",
                    "file": stage_name,
                    "part": 1,
                    "total_parts": 1,
                    "engine": model_name,
                    "text": prompt,
                }
            }
        )
        response_text = ""
        try:
            for chunk in client.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options={"temperature": 0.7, "format": "json"},
            ):
                piece = chunk.get("message", {}).get("content") or chunk.get("response") or ""
                if piece:
                    response_text += piece
                    publish(
                        {
                            "stream": {
                                "kind": "response",
                                "file": stage_name,
                                "part": 1,
                                "total_parts": 1,
                                "engine": model_name,
                                "text": response_text,
                            }
                        }
                    )
                if not ensure_running(stage_name, f"{stage_name} 分析进行中"):
                    return {"final_results": []}
        except Exception as error:
            publish(
                {
                    "log": {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "level": "error",
                        "message": f"调用模型失败({stage_name}): {error}",
                    },
                    "error": str(error),
                }
            )
            response_text = ""
        response_path = os.path.join(initial_dir, f"response_{stage_name}.txt")
        _save_text(response_path, response_text)
        stage_responses[stage_name] = response_text
        processed += 1
        publish({"processed_chunks": processed})
        progress = min(95.0, progress + progress_step)
        publish({"progress": progress})

    if not ensure_running("汇总", "生成Excel汇总结果"):
        return {"final_results": []}
    filepath, filename = export_completeness_results(
        session_id,
        stage_responses,
        final_dir,
        requirements_overrides=requirements_overrides,
    )
    if not filepath:
        publish(
            {
                "status": "failed",
                "stage": "汇总",
                "message": "生成Excel失败",
            }
        )
        return {"final_results": []}
    publish({"result_files": [filepath]})
    publish(
        {
            "status": "succeeded",
            "stage": "completed",
            "message": "文件齐套性检查完成",
            "progress": 100.0,
        }
    )
    return {"final_results": [filepath]}


__all__ = [
    "run_file_completeness_job",
    "export_completeness_results",
    "generate_stage_prompt",
    "parse_llm_table_response",
]
