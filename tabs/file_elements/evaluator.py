"""Evaluation orchestrator for文件要素检查."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple

from .config import (
    DeliverableProfile,
    ElementRequirement,
    SEVERITY_ORDER,
)

TEXT_EXTENSIONS = {".txt", ".md"}
PREFERRED_EXPORT_NAME = "file_elements_evaluation.json"


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

    def evaluate(self, text: str, *, source_file: str | None = None, warnings: Sequence[str] | None = None) -> EvaluationResult:
        evaluations = [self._evaluate_requirement(req, text) for req in self.profile.requirements]
        return EvaluationResult(
            profile=self.profile,
            generated_at=datetime.utcnow(),
            source_file=source_file,
            text_length=len(text or ""),
            evaluations=evaluations,
            warnings=list(warnings or ()),
        )

    def _evaluate_requirement(self, requirement: ElementRequirement, text: str) -> ElementEvaluation:
        text = text or ""
        if not text:
            return ElementEvaluation(
                requirement=requirement,
                status="missing",
                severity=requirement.severity,
                message="未提供可解析的文档内容。",
                snippet=None,
                keyword=None,
            )

        for keyword in requirement.keywords:
            if not keyword:
                continue
            pattern = re.compile(re.escape(keyword), flags=re.IGNORECASE)
            match = pattern.search(text)
            if match:
                snippet = _extract_snippet(text, match.start(), match.end())
                return ElementEvaluation(
                    requirement=requirement,
                    status="pass",
                    severity=requirement.severity,
                    message=f"检测到关键字“{keyword}”，初步符合要求。",
                    snippet=snippet,
                    keyword=keyword,
                )

        return ElementEvaluation(
            requirement=requirement,
            status="missing",
            severity=requirement.severity,
            message="未检测到对应关键内容，请补充。",
            snippet=None,
            keyword=None,
        )


def _extract_snippet(text: str, start: int, end: int, *, window: int = 80) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right].strip()
    snippet = re.sub(r"\s+", " ", snippet)
    return snippet


def parse_deliverable_stub(
    profile: DeliverableProfile,
    source_dir: str,
    parsed_dir: str | None = None,
    *,
    source_paths: Sequence[str] | None = None,
    preferred_source: str | None = None,
) -> Tuple[str, str | None, List[str]]:
    """Attempt to read deliverable text using lightweight heuristics.

    This helper prioritises ``preferred_source`` when provided. Otherwise it
    looks for the most recent text-like file in ``source_dir``. If a non-text
    document被上传, it searches ``parsed_dir`` for a预解析 ``.txt`` with matching
    前缀。失败时返回空字符串并提供警告。"""

    warnings: List[str] = []
    source_file: str | None = None
    text_content = ""

    candidates: List[str] = []
    seen: set[str] = set()

    def _add_candidate(path: str | None, *, front: bool = False) -> None:
        if not path:
            return
        normalized = os.path.normpath(path)
        if normalized in seen:
            return
        if os.path.isfile(normalized):
            seen.add(normalized)
            if front:
                candidates.insert(0, normalized)
            else:
                candidates.append(normalized)

    if preferred_source:
        _add_candidate(preferred_source, front=True)

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
    if preferred_source:
        preferred_norm = os.path.normpath(preferred_source)
        if preferred_norm in candidates:
            candidates.remove(preferred_norm)
            candidates.insert(0, preferred_norm)

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
            base_name = os.path.splitext(os.path.basename(path))[0]
            parsed_candidate = os.path.join(parsed_dir, f"{base_name}.txt")
            parsed_candidate = os.path.normpath(parsed_candidate)
            if os.path.isfile(parsed_candidate):
                try:
                    with open(parsed_candidate, "r", encoding="utf-8") as handle:
                        text_content = handle.read()
                    source_file = os.path.basename(parsed_candidate)
                    warnings.append(
                        f"已使用预解析文本 {os.path.basename(parsed_candidate)}，来源：{os.path.basename(path)}。"
                    )
                    return text_content, source_file, warnings
                except OSError as error:
                    warnings.append(f"读取预解析文本失败：{os.path.basename(parsed_candidate)} → {error}")

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
    "parse_deliverable_stub",
    "save_result_payload",
]
