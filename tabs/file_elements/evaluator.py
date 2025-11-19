"""Evaluation orchestrator for文件要素检查."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ollama import Client as OllamaClient

from config import CONFIG

from .config import (
    DeliverableProfile,
    ElementRequirement,
    SEVERITY_LABELS,
    SEVERITY_ORDER,
)
from ..shared.common import report_exception
from ..shared.file_conversion import (
    process_excel_folder,
    process_pdf_folder,
    process_textlike_folder,
    process_word_ppt_folder,
)
from util import resolve_ollama_host

TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".tsv"}
EXCEL_EXTENSIONS = {".xls", ".xlsx", ".xlsm"}
PREFERRED_EXPORT_NAME = "file_elements_evaluation.json"
GPT_OSS_MODEL = CONFIG["llm"]["ollama_model"]
GPT_OSS_OPTIONS = {"num_ctx": 40001, "temperature": 0.3}
LLM_SYSTEM_PROMPT = (
    "你是汽车行业APQP质量工程师，需要根据APQP阶段要素要求对交付物内容进行逐项评估。"
    "请仅输出JSON，status字段限定为pass、partial或missing，其含义分别为“满足”、“部分满足/需完善”、“缺失”。"
    "message字段给出判断理由，evidence字段引用对应的原文或定位信息。"
)


class _SilentProgress:
    """Collect warnings from conversion helpers without emitting UI output."""

    def __init__(self) -> None:
        self.messages: List[str] = []

    def info(self, *_args, **_kwargs) -> None:  # pragma: no cover - compatibility noop
        return

    def write(self, *_args, **_kwargs) -> None:  # pragma: no cover - compatibility noop
        return

    def success(self, *_args, **_kwargs) -> None:  # pragma: no cover - compatibility noop
        return

    def warning(self, message: str, *_args, **_kwargs) -> None:
        self.messages.append(str(message))

    def error(self, message: str, *_args, **_kwargs) -> None:
        self.messages.append(str(message))


def auto_convert_sources(
    source_dir: str | None,
    parsed_dir: str | None,
    *,
    progress_area=None,
    annotate_sources: bool = True,
) -> Tuple[List[str], List[str]]:
    """Convert supported uploads into `.txt` artifacts located in ``parsed_dir``."""

    created: List[str] = []
    warnings: List[str] = []
    if not source_dir or not parsed_dir or not os.path.isdir(source_dir):
        return created, warnings

    os.makedirs(parsed_dir, exist_ok=True)
    emitter = progress_area or _SilentProgress()
    silent = isinstance(emitter, _SilentProgress)

    created.extend(
        process_pdf_folder(source_dir, parsed_dir, emitter, annotate_sources=annotate_sources)
    )
    created.extend(
        process_word_ppt_folder(source_dir, parsed_dir, emitter, annotate_sources=annotate_sources)
    )
    created.extend(
        process_excel_folder(source_dir, parsed_dir, emitter, annotate_sources=annotate_sources)
    )
    created.extend(process_textlike_folder(source_dir, parsed_dir, emitter))

    if silent:
        warnings.extend(emitter.messages)

    return created, warnings


@dataclass
class ElementEvaluation:
    """Result of evaluating a single element requirement."""

    requirement: ElementRequirement
    status: str
    severity: str
    message: str
    snippet: str | None = None
    keyword: str | None = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "key": self.requirement.key,
            "name": self.requirement.name,
            "severity": self.severity,
            "status": self.status,
            "message": self.message,
            "snippet": self.snippet,
            "keyword": self.keyword,
            "guidance": self.requirement.guidance,
            "description": self.requirement.description,
        }


@dataclass
class EvaluationResult:
    """Aggregate evaluation output for a deliverable."""

    profile: DeliverableProfile
    generated_at: datetime
    source_file: str | None
    text_length: int
    evaluations: List[ElementEvaluation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def summary_counts(self) -> Dict[str, int]:
        totals = {"pass": 0, "missing": 0}
        for item in self.evaluations:
            if item.status == "pass":
                totals["pass"] += 1
            else:
                totals["missing"] += 1
        totals["total"] = len(self.evaluations)
        return totals

    @property
    def severity_breakdown(self) -> Dict[str, int]:
        counts: Dict[str, int] = {level: 0 for level in SEVERITY_ORDER}
        for item in self.evaluations:
            counts[item.severity] = counts.get(item.severity, 0) + (1 if item.status != "pass" else 0)
        return counts

    def to_dict(self) -> Dict[str, object]:
        return {
            "stage": self.profile.stage,
            "deliverable": self.profile.name,
            "deliverable_id": self.profile.id,
            "generated_at": self.generated_at.isoformat(),
            "source_file": self.source_file,
            "text_length": self.text_length,
            "summary": self.summary_counts,
            "severity": self.severity_breakdown,
            "evaluations": [item.as_dict() for item in self.evaluations],
            "warnings": list(self.warnings),
        }

    def export_json(self, target_folder: str) -> str | None:
        os.makedirs(target_folder, exist_ok=True)
        target_path = os.path.join(target_folder, PREFERRED_EXPORT_NAME)
        with open(target_path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
        return target_path


class EvaluationOrchestrator:
    """Encapsulate requirement evaluation for a given deliverable."""

    def __init__(self, profile: DeliverableProfile):
        self.profile = profile
        self._ollama_client: Optional[OllamaClient] = None
        self._ollama_init_error: Optional[Exception] = None

    def evaluate(
        self,
        text: str,
        *,
        source_file: str | None = None,
        warnings: Sequence[str] | None = None,
    ) -> EvaluationResult:
        base_warnings = list(warnings or ())
        text = text or ""
        if not text.strip():
            evaluations = [
                ElementEvaluation(
                    requirement=req,
                    status="missing",
                    severity=req.severity,
                    message="未提供可解析的文档内容。",
                    snippet=None,
                    keyword=None,
                )
                for req in self.profile.requirements
            ]
            return EvaluationResult(
                profile=self.profile,
                generated_at=datetime.utcnow(),
                source_file=source_file,
                text_length=0,
                evaluations=evaluations,
                warnings=base_warnings,
            )

        llm_items, llm_warnings = self._analyze_with_llm(text)
        base_warnings.extend(llm_warnings)
        evaluations = self._merge_llm_results(llm_items)
        return EvaluationResult(
            profile=self.profile,
            generated_at=datetime.utcnow(),
            source_file=source_file,
            text_length=len(text),
            evaluations=evaluations,
            warnings=base_warnings,
        )

    def _merge_llm_results(self, items: List[Dict[str, str]]) -> List[ElementEvaluation]:
        lookup: Dict[str, Dict[str, str]] = {}
        for item in items:
            normalized = _normalize_name(item.get("name"))
            if normalized:
                lookup[normalized] = item

        evaluations: List[ElementEvaluation] = []
        for requirement in self.profile.requirements:
            normalized = _normalize_name(requirement.name)
            payload = lookup.get(normalized, {})
            status = _normalize_status(payload.get("status"))
            message = payload.get("message") or payload.get("analysis") or payload.get("结论")
            evidence = payload.get("evidence") or payload.get("snippet") or payload.get("引用")
            if not payload:
                message = "LLM未返回该要素的判断，请手动核查。"
            elif not message:
                message = "LLM未给出明确说明，请结合原文核对。"
            evaluations.append(
                ElementEvaluation(
                    requirement=requirement,
                    status=status,
                    severity=requirement.severity,
                    message=message,
                    snippet=evidence or None,
                    keyword=None,
                )
            )
        return evaluations

    def _analyze_with_llm(self, document_text: str) -> Tuple[List[Dict[str, str]], List[str]]:
        warnings: List[str] = []
        client = self._get_ollama_client()
        if client is None:
            message = "未能连接本地 gpt-oss，请联系系统管理员。"
            if self._ollama_init_error:
                message += f" ({self._ollama_init_error})"
            warnings.append(message)
            return [], warnings

        requirements_payload = [
            {
                "name": req.name,
                "severity": req.severity,
                "severity_label": SEVERITY_LABELS.get(req.severity, req.severity),
                "description": req.description,
                "guidance": req.guidance,
            }
            for req in self.profile.requirements
        ]
        user_prompt = (
            f"APQP阶段：{self.profile.stage}\n"
            f"交付物：{self.profile.name}\n"
            f"要素要求：\n{json.dumps(requirements_payload, ensure_ascii=False, indent=2)}\n"
            "请结合上述要素要求，逐条审阅交付物文本，判断是否满足。"
            "如需引用，请在evidence字段中保留原文片段或位置。"
            "\n\n交付物全文如下：\n<<<DOCUMENT>>>\n"
            f"{document_text}\n<<<END>>>\n"
            "请严格按照JSON格式输出，不要添加额外说明。"
            "示例输出：{\"items\": [{\"name\": \"要素\", \"status\": \"pass\", \"message\": \"说明\", \"evidence\": \"引用\"}]}"
        )

        response_text = ""
        try:
            for chunk in client.chat(
                model=GPT_OSS_MODEL,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
                options=GPT_OSS_OPTIONS,
            ):
                piece = (
                    chunk.get("message", {}).get("content")
                    or chunk.get("response")
                    or ""
                )
                if piece:
                    response_text += piece
        except Exception as error:  # pragma: no cover - network/runtime failures
            report_exception("调用 gpt-oss 失败", error, level="warning")
            warnings.append(f"调用 gpt-oss 失败：{error}")
            return [], warnings

        if not response_text.strip():
            warnings.append("gpt-oss 未返回内容，请稍后重试。")
            return [], warnings

        parsed_items = _parse_llm_items(response_text)
        if not parsed_items:
            warnings.append("gpt-oss 返回内容无法解析为JSON，请检查交付物内容或稍后重试。")
        return parsed_items, warnings

    def _get_ollama_client(self) -> Optional[OllamaClient]:
        if self._ollama_client is not None:
            return self._ollama_client
        if self._ollama_init_error is not None:
            return None
        try:
            host = resolve_ollama_host("ollama_9")
            self._ollama_client = OllamaClient(host=host)
        except Exception as error:  # pragma: no cover - initialization failure
            self._ollama_init_error = error
            report_exception("初始化本地 gpt-oss 客户端失败", error, level="warning")
            return None
        return self._ollama_client


def _strip_code_fences(value: str) -> str:
    s = (value or "").strip()
    if s.startswith("```") and s.endswith("```"):
        inner = s.split("```", 2)
        if len(inner) >= 3:
            return inner[1].strip()
        return s.strip("`")
    return s


def _extract_json_text(text: str) -> str:
    s = _strip_code_fences(text)
    s = s.lstrip("\ufeff")
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
            json.loads(candidate)
            return candidate
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


def _parse_llm_items(response_text: str) -> List[Dict[str, str]]:
    payload = _extract_json_text(response_text)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        items = data.get("items") or data.get("results") or data.get("elements")
        if isinstance(items, list):
            entries = items
        else:
            entries = [data]
    elif isinstance(data, list):
        entries = data
    else:
        return []

    normalized: List[Dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(
            entry.get("name")
            or entry.get("element")
            or entry.get("要素")
            or entry.get("title")
            or ""
        ).strip()
        if not name:
            continue
        normalized.append(
            {
                "name": name,
                "status": str(entry.get("status") or entry.get("result") or entry.get("状态") or "").strip(),
                "message": str(
                    entry.get("message")
                    or entry.get("analysis")
                    or entry.get("summary")
                    or entry.get("说明")
                    or ""
                ).strip(),
                "evidence": str(
                    entry.get("evidence")
                    or entry.get("snippet")
                    or entry.get("quote")
                    or entry.get("引用")
                    or ""
                ).strip(),
            }
        )
    return normalized


def _normalize_name(value: str | None) -> str:
    value = value or ""
    return re.sub(r"\s+", "", value).lower()


def _normalize_status(value: str | None) -> str:
    token = (value or "").strip().lower()
    if not token:
        return "missing"
    pass_tokens = {"pass", "ok", "符合", "满足", "合格", "是"}
    if token in pass_tokens:
        return "pass"
    return "missing"


def _collect_parsed_texts(original_path: str, parsed_dir: str) -> List[str]:
    """Return matching parsed text files for ``original_path`` within ``parsed_dir``."""

    if not parsed_dir or not os.path.isdir(parsed_dir):
        return []

    raw_name = os.path.basename(original_path)
    base_name = os.path.splitext(raw_name)[0]
    ext = os.path.splitext(raw_name)[1].lower()

    def _add(path: str, bucket: List[str], seen: set[str]) -> None:
        normalized = os.path.normpath(path)
        if normalized in seen:
            return
        if os.path.isfile(normalized):
            seen.add(normalized)
            bucket.append(normalized)

    results: List[str] = []
    seen: set[str] = set()

    _add(os.path.join(parsed_dir, f"{raw_name}.txt"), results, seen)
    _add(os.path.join(parsed_dir, f"{base_name}.txt"), results, seen)

    if ext in EXCEL_EXTENSIONS:
        prefix = f"{raw_name}_sheet_".lower()
        try:
            for name in os.listdir(parsed_dir):
                if not name.lower().endswith(".txt"):
                    continue
                if not name.lower().startswith(prefix):
                    continue
                _add(os.path.join(parsed_dir, name), results, seen)
        except OSError:
            pass

    try:
        results.sort(key=lambda item: os.path.getmtime(item), reverse=True)
    except OSError:
        results.sort()
    return results


def parse_deliverable_stub(
    profile: DeliverableProfile,
    source_dir: str,
    parsed_dir: str | None = None,
    *,
    source_paths: Sequence[str] | None = None,
) -> Tuple[str, str | None, List[str]]:
    """Attempt to read deliverable text using lightweight heuristics.

    This helper looks for the most recent text-like file in ``source_dir``.
    If a non-text document被上传, it searches ``parsed_dir`` for a pre-generated
    ``.txt`` with matching前缀. 失败时返回空字符串并提供警告。"""

    warnings: List[str] = []
    source_file: str | None = None
    text_content = ""

    candidates: List[str] = []
    seen: set[str] = set()

    converted: List[str] = []
    conversion_warnings: List[str] = []
    conversion_message = ""
    conversion_noted = False

    if parsed_dir:
        new_converted, conversion_warnings = auto_convert_sources(
            source_dir, parsed_dir, annotate_sources=True
        )
        if new_converted:
            converted.extend(new_converted)
            sample = ", ".join(os.path.basename(item) for item in converted[:4])
            if len(converted) > 4:
                sample += " 等"
            conversion_message = (
                f"已自动生成 {len(converted)} 个文本文件用于评估（例如：{sample}）。"
            )
        if conversion_warnings:
            warnings.extend(conversion_warnings)


    def _add_candidate(path: str | None) -> None:
        if not path:
            return
        normalized = os.path.normpath(path)
        if normalized in seen:
            return
        if os.path.isfile(normalized):
            seen.add(normalized)
            candidates.append(normalized)

    if source_paths:
        for path in source_paths:
            _add_candidate(path)

    if not candidates:
        if not source_dir or not os.path.isdir(source_dir):
            warnings.append("未找到交付物上传目录，请先上传文件。")
            return text_content, source_file, warnings

        for name in os.listdir(source_dir):
            _add_candidate(os.path.join(source_dir, name))
    if not candidates:
        warnings.append("上传目录为空，请上传交付物后再评估。")
        return text_content, source_file, warnings

    candidates.sort(key=lambda item: os.path.getmtime(item), reverse=True)

    for path in candidates:
        _, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext in TEXT_EXTENSIONS:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    text_content = handle.read()
                source_file = os.path.basename(path)
                return text_content, source_file, warnings
            except OSError as error:
                warnings.append(f"读取文件失败：{os.path.basename(path)} → {error}")
                continue

        if parsed_dir:
            parsed_candidates = _collect_parsed_texts(path, parsed_dir)
            if parsed_candidates:
                try:
                    if len(parsed_candidates) == 1:
                        with open(parsed_candidates[0], "r", encoding="utf-8") as handle:
                            text_content = handle.read()
                        source_file = os.path.basename(parsed_candidates[0])
                    else:
                        parts: List[str] = []
                        for sheet_path in parsed_candidates:
                            with open(sheet_path, "r", encoding="utf-8") as handle:
                                sheet_text = handle.read()
                            parts.append(
                                f"### 来源：{os.path.basename(sheet_path)}\n{sheet_text.strip()}"
                                if sheet_text.strip()
                                else f"### 来源：{os.path.basename(sheet_path)}"
                            )
                        text_content = "\n\n".join(parts)
                        source_file = f"{os.path.basename(parsed_candidates[0])} 等{len(parsed_candidates)}个工作表"
                    if conversion_message and not conversion_noted:
                        warnings.append(conversion_message)
                        conversion_noted = True
                    warnings.append(
                        f"已使用预解析文本 {os.path.basename(parsed_candidates[0])}，来源：{os.path.basename(path)}。"
                    )
                    return text_content, source_file, warnings
                except OSError as error:
                    warnings.append(
                        f"读取预解析文本失败：{os.path.basename(parsed_candidates[0])} → {error}"
                    )

    if conversion_message and not conversion_noted:
        warnings.append(conversion_message)
    source_file = os.path.basename(candidates[0]) if candidates else None
    if source_file:
        warnings.append(
            f"暂不支持自动解析 {os.path.splitext(source_file)[1] or '该格式'}，请上传文本版或放置同名.txt于解析目录。"
        )
    return text_content, source_file, warnings


def save_result_payload(result: EvaluationResult, target_folder: str) -> str | None:
    return result.export_json(target_folder)


__all__ = [
    "ElementEvaluation",
    "EvaluationResult",
    "EvaluationOrchestrator",
    "auto_convert_sources",
    "parse_deliverable_stub",
    "save_result_payload",
]
