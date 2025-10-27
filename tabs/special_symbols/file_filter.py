"""Filtering utilities for special symbols examined text files.

This module filters Excel-converted .txt files produced under the special symbols
workflow to reduce irrelevant content before sending to LLMs.

Outputs filtered copies into a sibling folder (examined_txt_filtered).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import yaml


@dataclass
class FilterConfig:
    sheet_exclude_patterns: List[str]
    sheet_soft_keep_patterns: List[str]
    sheet_hard_keep_patterns: List[str]
    line_keep_symbols: List[str]
    line_keep_keywords: List[str]
    line_drop_keywords: List[str]

    @classmethod
    def default(cls) -> "FilterConfig":
        return cls(
            sheet_exclude_patterns=[
                r"修订(履历|记录)?",
                r"历次修订",
                r"附表(一|二|三|四|五)?",
                r"附图",
                r"附件",
                r"附录",
                r"防错(清单)?",
                r"目录",
                r"封面",
                r"页码",
                r"版本(信息|记录)",
                r"审批",
                r"签(字|署)",
                r"填写说明",
            ],
            sheet_soft_keep_patterns=[
                # Not used currently; reserved for future soft filtering per sheet
            ],
            sheet_hard_keep_patterns=[
                r"控制计划",
            ],
            line_keep_symbols=["★", "☆", "/", "≤", "≥", "%", "μm", "mm"],
            line_keep_keywords=[
                # process/feature and measurement cues
                "特性",
                "产品特性",
                "过程特性",
                "关键",
                "重要",
                "检",
                "测量",
                "记录",
                "仪",
                "量具",
                "公差",
                "限度",
                "频率",
                "每卷",
                "每批",
                "每班",
                "1次",
                "2次",
            ],
            line_drop_keywords=[
                # role/sign-off and boilerplate explanations
                "编制",
                "文审",
                "审核",
                "会签",
                "批准",
                "文控",
                "签名",
                "签字",
                "日期",
                # common bilingual explanation cues
                "备注",
                "Operation",
                "Process Description",
                "Control plan",
                "PFMEA",
                "show production and safety",
            ],
        )


def load_config(config_path: Optional[str]) -> FilterConfig:
    if not config_path:
        return FilterConfig.default()
    path = Path(config_path)
    if not path.is_file():
        return FilterConfig.default()
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return FilterConfig(
            sheet_exclude_patterns=list(data.get("sheet_exclude_patterns", [])) or FilterConfig.default().sheet_exclude_patterns,
            sheet_soft_keep_patterns=list(data.get("sheet_soft_keep_patterns", [])) or [],
            sheet_hard_keep_patterns=list(data.get("sheet_hard_keep_patterns", [])) or [r"控制计划"],
            line_keep_symbols=list(data.get("line_keep_symbols", [])) or FilterConfig.default().line_keep_symbols,
            line_keep_keywords=list(data.get("line_keep_keywords", [])) or FilterConfig.default().line_keep_keywords,
            line_drop_keywords=list(data.get("line_drop_keywords", [])) or FilterConfig.default().line_drop_keywords,
        )
    except Exception:
        return FilterConfig.default()


def _compile_patterns(patterns: Iterable[str]) -> List[re.Pattern[str]]:
    compiled: List[re.Pattern[str]] = []
    for pat in patterns:
        try:
            compiled.append(re.compile(pat, re.IGNORECASE))
        except re.error:
            continue
    return compiled


def extract_sheet_name(file_name: str) -> str:
    base = os.path.basename(file_name)
    if "_SHEET_" not in base:
        return ""
    try:
        sheet_part = base.split("_SHEET_", 1)[1]
        if sheet_part.lower().endswith(".txt"):
            sheet_part = sheet_part[: -len(".txt")]
        return sheet_part
    except Exception:
        return ""


def should_keep_txt(file_name: str, config: FilterConfig) -> Tuple[bool, str, str]:
    """Return (keep, sheet_name, reason)."""

    sheet = extract_sheet_name(file_name)
    if sheet:
        for pat in _compile_patterns(config.sheet_exclude_patterns):
            if pat.search(sheet):
                return False, sheet, f"exclude:{pat.pattern}"
        for pat in _compile_patterns(config.sheet_hard_keep_patterns):
            if pat.search(sheet):
                return True, sheet, f"hard_keep:{pat.pattern}"
    # Default: keep
    return True, sheet, "default_keep"


_PUNCT_ONLY_RE = re.compile(r"^[\s,，。；;、|:：\-—]+$")
_NAME_LIKE_SEP_RE = re.compile(r"[、，,]{2,}")
_HAS_DIGIT_RE = re.compile(r"\d")


def _line_has_keep_signals(line: str, config: FilterConfig) -> bool:
    for sym in config.line_keep_symbols:
        if sym and sym in line:
            return True
    for kw in config.line_keep_keywords:
        if kw and kw in line:
            return True
    # generic: times like 3次/5% etc.
    if re.search(r"\d+\s*(次|%|μm|mm)", line):
        return True
    # simple presence of slash used as criteria separator
    if "/" in line:
        return True
    return False


def _line_is_boilerplate(line: str, config: FilterConfig) -> bool:
    text = line.strip()
    if not text:
        return True
    if _PUNCT_ONLY_RE.match(text):
        return True
    # role/sign-off and boilerplate keys
    for bad in config.line_drop_keywords:
        if bad and bad in text:
            # Keep lines that contain explicit special symbols when drop-keyword triggers via "备注" explaining symbols
            if bad == "备注":
                if any(s in text for s in ("★", "☆")) and any(k in text for k in ("表示", "show")):
                    return True
            else:
                return True
    # heuristics: many separators but no digits or symbols
    if _NAME_LIKE_SEP_RE.search(text) and not (_HAS_DIGIT_RE.search(text) or any(ch in text for ch in "★☆/≤≥%")):
        return True
    return False


def filter_text_lines(lines: Iterable[str], sheet_name: str, config: FilterConfig) -> List[str]:
    # For control plan sheets, keep all lines (as requested)
    for pat in _compile_patterns(config.sheet_hard_keep_patterns):
        if sheet_name and pat.search(sheet_name):
            return list(lines)

    kept: List[str] = []
    for raw in lines:
        line = raw.rstrip("\n\r")
        if _line_is_boilerplate(line, config):
            continue
        if _line_has_keep_signals(line, config):
            kept.append(raw)
            continue
        # fallback: keep short non-empty rows that look like CSV headers with potential relevance
        if "," in line and len(line) <= 200 and any(k in line for k in ("特性", "参数", "规格", "要求")):
            kept.append(raw)
            continue
    return kept


def run_filtering(source_dir: str, target_dir: str, config_path: Optional[str] = None) -> dict:
    """Filter .txt files from ``source_dir`` into ``target_dir`` based on heuristics.

    Returns a summary dict with keys: kept, dropped, empty_after_filter, log_path
    """

    os.makedirs(target_dir, exist_ok=True)
    # Phase 1 redesign: retain only the symbol presence requirement while keeping
    # the existing output handling & audit flow. Config loading is preserved for
    # compatibility, though the gating heuristics defined there are no longer
    # applied during this phase.
    load_config(config_path)
    kept = 0
    dropped = 0
    emptied = 0
    excluded: List[str] = []
    candidates: List[Tuple[str, str]] = []

    for name in sorted(os.listdir(source_dir)):
        src_path = os.path.join(source_dir, name)
        if not os.path.isfile(src_path) or not name.lower().endswith(".txt"):
            continue
        try:
            with open(src_path, "r", encoding="utf-8", errors="ignore") as handle:
                content = handle.read()
        except Exception:
            dropped += 1
            excluded.append(f"{name}\tread_error")
            continue
        # Global rule: if neither ★ nor ☆ appears anywhere in the file, drop the file
        if ("★" not in content) and ("☆" not in content):
            dropped += 1
            excluded.append(f"{name}\tno_special_stars")
            continue
        candidates.append((name, content))

    if candidates:
        try:
            for existing in os.listdir(target_dir):
                existing_path = os.path.join(target_dir, existing)
                if os.path.isfile(existing_path):
                    try:
                        os.remove(existing_path)
                    except Exception:
                        continue
        except Exception:
            pass

    for name, content in candidates:
        dst_path = os.path.join(target_dir, name)
        try:
            with open(dst_path, "w", encoding="utf-8") as writer:
                writer.write(content)
            kept += 1
        except Exception:
            dropped += 1
            excluded.append(f"{name}\twrite_error")

    log_path = os.path.join(target_dir, "excluded_files.log")
    try:
        if excluded:
            with open(log_path, "w", encoding="utf-8") as log:
                log.write("\n".join(excluded))
    except Exception:
        log_path = ""

    return {"kept": kept, "dropped": dropped, "empty_after_filter": emptied, "log_path": log_path}


__all__ = [
    "FilterConfig",
    "extract_sheet_name",
    "filter_text_lines",
    "load_config",
    "run_filtering",
    "should_keep_txt",
]
