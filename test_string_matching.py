"""Prototype script to compare special-symbol listings via fuzzy string matching."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Hard-coded sources pulled from the latest workflow outputs.
EXAM_DIR = Path("generated_files/Jack Zhou/special_symbols_check/examined_txt_filtered_further")
STANDARD_DIR = Path("generated_files/Jack Zhou/special_symbols_check/standards_txt_filtered_further")
DEFAULT_EXAM_FILE = EXAM_DIR / "examined_txt_filtered_final_20251111_233220.txt"
DEFAULT_STANDARD_FILE = STANDARD_DIR / "standards_txt_filtered_final_20251111_232827.txt"

# Minimal similarity needed to consider two descriptors the "same" feature.
MATCH_THRESHOLD = 0.32
MAX_MISMATCH_REPORTS = 12

SYMBOLS = {"★", "☆", "/"}


@dataclass
class Entry:
    source_file: str
    sheet_name: str
    raw_text: str
    description: str
    symbol: str
    normalized: str
    tokens: frozenset[str]


def load_entries(path: Path) -> List[Entry]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries: List[Entry] = []
    for record in data:
        file_name = str(record.get("文件名") or "")
        sheet_name = str(record.get("工作表名") or "")
        for item in record.get("特殊特性符号", []) or []:
            if not isinstance(item, str):
                continue
            description, symbol = split_description_symbol(item)
            if not symbol:
                continue
            normalized = normalize(description)
            tokens = extract_tokens(description)
            entries.append(
                Entry(
                    source_file=file_name,
                    sheet_name=sheet_name,
                    raw_text=item.strip(),
                    description=description,
                    symbol=symbol,
                    normalized=normalized,
                    tokens=frozenset(tokens),
                )
            )
    return entries


def split_description_symbol(text: str) -> Tuple[str, str]:
    for idx in range(len(text) - 1, -1, -1):
        if text[idx] in SYMBOLS:
            desc = text[:idx].strip()
            desc = desc.rstrip(",，:：;/ ")
            return desc, text[idx]
    return text.strip(), ""


def normalize(text: str) -> str:
    cleaned = text.lower()
    cleaned = re.sub(r"[★☆/]", " ", cleaned)
    cleaned = re.sub(r"\bop\s*\d+\b", " ", cleaned)
    cleaned = re.sub(r"[0-9\-\.]+", " ", cleaned)
    cleaned = re.sub(r"[\s,:：;/，。()（）\[\]]+", " ", cleaned)
    return cleaned.strip()


def extract_tokens(text: str) -> List[str]:
    cleaned = text.lower()
    cleaned = re.sub(r"[★☆/]", " ", cleaned)
    parts = re.split(r"[,\s，、:：;()（）/\-]+", cleaned)
    tokens: List[str] = []
    for part in parts:
        token = part.strip()
        if not token:
            continue
        if token.isdigit():
            continue
        if token.startswith("op") and token[2:].isdigit():
            continue
        tokens.append(token)
        for sub in re.findall(r"[a-z]+|\d+|[\u4e00-\u9fa5]+", token):
            sub_token = sub.strip()
            if not sub_token:
                continue
            if sub_token.isdigit():
                continue
            if len(sub_token) <= 1 and sub_token not in {"k"}:
                continue
            tokens.append(sub_token)

    normalized_tokens: List[str] = []
    for token in tokens:
        normalized = apply_token_replacements(token)
        if normalized:
            normalized_tokens.append(normalized)
    return normalized_tokens


def apply_token_replacements(token: str) -> str:
    replacements = {
        "电芯": "电池",
        "soc": "",
        "≤": "",
        "≥": "",
    }
    for needle, repl in replacements.items():
        token = token.replace(needle, repl)
    return token.strip()
    return tokens


def token_similarity(a: Entry, b: Entry) -> float:
    if not a.tokens or not b.tokens:
        return 0.0
    inter = len(a.tokens & b.tokens)
    if inter == 0:
        return 0.0
    union = len(a.tokens | b.tokens)
    return inter / union


def best_match(entry: Entry, candidates: Iterable[Entry]) -> Tuple[Optional[Entry], float]:
    best: Optional[Entry] = None
    best_ratio = 0.0
    for candidate in candidates:
        if not candidate.normalized:
            continue
        ratio_char = SequenceMatcher(None, entry.normalized, candidate.normalized).ratio()
        ratio_tokens = token_similarity(entry, candidate)
        ratio = 0.6 * ratio_char + 0.4 * ratio_tokens
        if ratio_tokens == 0.0 and ratio_char < 0.35:
            continue
        if ratio > best_ratio:
            best_ratio = ratio
            best = candidate
    return best, best_ratio


def resolve_latest(directory: Path, pattern: str) -> Path:
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在：{directory}")
    candidates = sorted(
        directory.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"目录 {directory} 中找不到匹配 {pattern} 的文件")
    return candidates[0]


def main() -> None:
    exam_path = DEFAULT_EXAM_FILE if DEFAULT_EXAM_FILE.exists() else resolve_latest(
        EXAM_DIR, "examined_txt_filtered_final_*.txt"
    )
    standard_path = (
        DEFAULT_STANDARD_FILE
        if DEFAULT_STANDARD_FILE.exists()
        else resolve_latest(STANDARD_DIR, "standards_txt_filtered_final_*.txt")
    )

    exam_entries = load_entries(exam_path)
    standard_entries = load_entries(standard_path)

    unmatched_exam: List[Entry] = []
    mismatches: List[Tuple[Entry, Entry, float]] = []
    matched_standards: set[int] = set()
    seen_pairs: set[Tuple[str, str]] = set()

    for exam_entry in exam_entries:
        candidate, score = best_match(exam_entry, standard_entries)
        if candidate is None or score < MATCH_THRESHOLD:
            unmatched_exam.append(exam_entry)
            continue
        idx = standard_entries.index(candidate)
        matched_standards.add(idx)
        if exam_entry.symbol != candidate.symbol:
            pair_key = (candidate.raw_text, exam_entry.raw_text)
            if pair_key not in seen_pairs:
                mismatches.append((exam_entry, candidate, score))
                seen_pairs.add(pair_key)

    unmatched_standards = [
        item for idx, item in enumerate(standard_entries) if idx not in matched_standards
    ]

    print("=== 特殊特性符号比对（字符串匹配原型） ===")
    print(f"使用待检源: {exam_path.name}")
    print(f"使用标准源: {standard_path.name}")
    print(f"标准条目总数: {len(standard_entries)}")
    print(f"待检条目总数: {len(exam_entries)}")
    print(f"匹配到的潜在不一致: {len(mismatches)}")
    print()

    if mismatches:
        mismatches.sort(key=lambda item: item[2], reverse=True)
        print(">> 可能的不一致条目（符号不同）：")
        for exam_entry, std_entry, score in mismatches[:MAX_MISMATCH_REPORTS]:
            print(f"  - 匹配置信度: {score:.2f}")
            print(f"    标准: {std_entry.raw_text} [{std_entry.symbol}]")
            print(f"    待检: {exam_entry.raw_text} [{exam_entry.symbol}]")
            print(f"    标准文件: {std_entry.source_file} / {std_entry.sheet_name}")
            print(f"    待检文件: {exam_entry.source_file} / {exam_entry.sheet_name}")
            print()
        if len(mismatches) > MAX_MISMATCH_REPORTS:
            print(f"  ... 其余 {len(mismatches) - MAX_MISMATCH_REPORTS} 条省略")
            print()
    else:
        print(">> 未发现符号不一致。")
        print()

    if unmatched_exam:
        print(f">> 待检条目未匹配到标准（{len(unmatched_exam)} 条）：")
        for entry in unmatched_exam[:20]:
            print(f"    {entry.raw_text} [{entry.symbol}] -- {entry.source_file} / {entry.sheet_name}")
        if len(unmatched_exam) > 20:
            print(f"    ... 其余 {len(unmatched_exam) - 20} 条省略")
        print()

    if unmatched_standards:
        print(f">> 标准条目未在待检中找到对照（{len(unmatched_standards)} 条）：")
        for entry in unmatched_standards[:20]:
            print(f"    {entry.raw_text} [{entry.symbol}] -- {entry.source_file} / {entry.sheet_name}")
        if len(unmatched_standards) > 20:
            print(f"    ... 其余 {len(unmatched_standards) - 20} 条省略")
        print()


if __name__ == "__main__":
    main()

