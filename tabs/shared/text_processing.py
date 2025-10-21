"""Text preprocessing utilities reusable by document workflows."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple

_TXT_SEPARATOR_LINE_RE = re.compile(r"^[,;|\s]+$")
_TXT_UNNAMED_COL_RE = re.compile(r"\bUnnamed:\s*\d+\b", re.IGNORECASE)
_TXT_MULTISPACE_RE = re.compile(r"\s+")
_TXT_DOT_LEADERS_RE = re.compile(r"\.{2,}")
_TXT_BRACKETS_RE = re.compile(r"^[\[\(\{\s]+|[\]\)\}\s]+$")
_TXT_HIGH_FREQ_THRESHOLD = 3


def read_text(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return file_path.read_text(encoding="gbk", errors="ignore")


def write_text(file_path: Path, content: str) -> None:
    file_path.write_text(content, encoding="utf-8")


def strip_html_and_artifacts(line: str) -> str:
    line_no_unnamed = _TXT_UNNAMED_COL_RE.sub("", line)
    line_no_dots = _TXT_DOT_LEADERS_RE.sub(" ", line_no_unnamed)
    line_no_brackets = _TXT_BRACKETS_RE.sub("", line_no_dots)
    return _TXT_MULTISPACE_RE.sub(" ", line_no_brackets).strip()


def is_separator_line(line: str) -> bool:
    return bool(_TXT_SEPARATOR_LINE_RE.match(line))


def normalize_for_dedup(line: str) -> str:
    line = line.strip()
    line = _TXT_DOT_LEADERS_RE.sub(" ", line)
    line = _TXT_MULTISPACE_RE.sub(" ", line)
    return line


def is_source_stamp(line: str) -> bool:
    return line.strip().startswith("【来源文件:")


def drop_high_frequency_boilerplate(lines: List[str]) -> List[str]:
    freq: dict[str, int] = {}
    norm_cache: dict[int, str] = {}
    for idx, line in enumerate(lines):
        if is_source_stamp(line):
            continue
        norm = norm_cache.setdefault(idx, normalize_for_dedup(strip_html_and_artifacts(line)))
        if not norm:
            continue
        freq[norm] = freq.get(norm, 0) + 1

    result: List[str] = []
    for idx, line in enumerate(lines):
        if is_source_stamp(line):
            result.append(line)
            continue
        norm = norm_cache.get(idx)
        if norm is None:
            norm = normalize_for_dedup(strip_html_and_artifacts(line))
            norm_cache[idx] = norm
        has_units = bool(re.search(r"(℃|°C|kPa|V|A|W|Wh|W·h|Ω|Ω/V|mm|cm|m|s|min|h|%|kW|MW|MWh)", line))
        has_digits = any(ch.isdigit() for ch in line)
        if not norm:
            continue
        if freq.get(norm, 0) >= _TXT_HIGH_FREQ_THRESHOLD and not has_units and not has_digits:
            continue
        result.append(line)
    return result


def deduplicate_conservatively(lines: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for line in lines:
        if is_source_stamp(line):
            out.append(line)
            continue
        norm = normalize_for_dedup(line)
        if not norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(line)
    return out


def bilingual_pair_prune(lines: List[str]) -> List[str]:
    def tokenize_numbers_units(text: str) -> Tuple[str, ...]:
        squeezed = _TXT_MULTISPACE_RE.sub(" ", text)
        return tuple(re.findall(r"\d+|[A-Za-z%°℃Ω/·\-]+", squeezed))

    def has_cjk(text: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in text)

    def has_latin(text: str) -> bool:
        return any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in text)

    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        if is_source_stamp(line):
            i += 1
            continue
        j = i + 1
        if j < len(lines) and not lines[j].strip():
            out.append(lines[j])
            j += 1
        if j < len(lines):
            l1 = normalize_for_dedup(line)
            l2 = normalize_for_dedup(lines[j])
            if l1 and l2 and l1 != l2:
                if (has_cjk(l1) and has_latin(l2)) or (has_latin(l1) and has_cjk(l2)):
                    tokens_l1 = tokenize_numbers_units(l1)
                    tokens_l2 = tokenize_numbers_units(l2)
                    if tokens_l1 == tokens_l2 and tokens_l1:
                        i = j
                        i += 1
                        continue
        i += 1
    return out


def remove_long_latex_like(content: str, min_length: int = 300) -> str:
    latex_triggers = re.compile(
        r"(?is)(\\begin\s*\{(?:array|align|aligned|eqnarray|pmatrix|bmatrix)\}|\\\[|\$\$|\\math[a-zA-Z]*|\\frac|\\sum|\\int|\\left|\\right|\\mathrm|\\mathcal|\\mathbb|(?:^|\W)gin\{array)"
    )
    env_tokens = ("array", "align", "aligned", "eqnarray", "pmatrix", "bmatrix")

    def stats(segment: str) -> Tuple[int, int, int, int, float]:
        backslashes = segment.count("\\")
        dollars = segment.count("$")
        braces = segment.count("{") + segment.count("}")
        cjk = sum(1 for ch in segment if "\u4e00" <= ch <= "\u9fff")
        letters = sum(1 for ch in segment if ch.isalpha())
        length = len(segment)
        cjk_ratio = cjk / max(1, length)
        return backslashes, dollars, braces, letters, cjk_ratio

    def looks_like_latex_block(segment: str) -> bool:
        backslashes, dollars, braces, _letters, cjk_ratio = stats(segment)
        contains_env = any(token in segment for token in env_tokens)
        if contains_env or segment.strip().startswith("$$"):
            return True
        if backslashes >= 10 and (dollars >= 4 or braces >= 30) and cjk_ratio <= 0.10:
            return True
        return False

    to_remove: List[Tuple[int, int]] = []
    for match in latex_triggers.finditer(content):
        start = max(0, match.start() - 20)
        end = match.end()
        while end < len(content) and (end - start) < min_length:
            end = min(len(content), end + 200)
        candidate = content[start:end]
        if not looks_like_latex_block(candidate):
            continue
        while end < len(content):
            next_end = min(len(content), end + 200)
            extended = content[start:next_end]
            if not looks_like_latex_block(extended):
                break
            end = next_end
        if (end - start) >= min_length and looks_like_latex_block(content[start:end]):
            to_remove.append((start, end))

    if not to_remove:
        return content

    to_remove.sort()
    merged: List[Tuple[int, int]] = []
    cur_start, cur_end = to_remove[0]
    for seg_start, seg_end in to_remove[1:]:
        if seg_start <= cur_end:
            cur_end = max(cur_end, seg_end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = seg_start, seg_end
    merged.append((cur_start, cur_end))

    out_parts: List[str] = []
    prev = 0
    for seg_start, seg_end in merged:
        out_parts.append(content[prev:seg_start])
        prev = seg_end
    out_parts.append(content[prev:])
    return "".join(out_parts)


def process_text(content: str) -> str:
    content = remove_long_latex_like(content, min_length=300)
    raw_lines = content.splitlines()
    stage1: List[str] = []
    for line in raw_lines:
        clean = strip_html_and_artifacts(line)
        if not clean:
            continue
        if is_separator_line(clean):
            continue
        stage1.append(clean)
    stage2 = drop_high_frequency_boilerplate(stage1)
    stage3 = deduplicate_conservatively(stage2)
    stage4 = bilingual_pair_prune(stage3)
    return "\n".join(stage4).rstrip() + "\n"


def process_file(path: Path) -> bool:
    original = read_text(path)
    processed = process_text(original)
    if processed != original:
        write_text(path, processed)
        return True
    return False


def collect_txt_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (path for path in sorted(root.glob("*.txt")) if path.is_file())


def preprocess_txt_directories(*folders: str | os.PathLike[str]) -> List[Path]:
    updated: List[Path] = []
    for folder in folders:
        if not folder:
            continue
        root = Path(folder)
        if not root.exists():
            continue
        for txt_path in collect_txt_files(root):
            if process_file(txt_path):
                updated.append(txt_path)
    return updated


def insert_source_markers(text: str, source_label: str, line_interval: int = 80) -> str:
    """Insert unobtrusive source markers so small retrieved fragments retain provenance."""

    marker = f"【来源文件: {source_label}】"
    if source_label and marker in text:
        return text
    lines = text.splitlines()
    has_md_heading = any(re.match(r"^\s{0,3}#{1,2}\s+\S", ln) for ln in lines[:500])

    annotated_lines = [marker, ""]

    if has_md_heading:
        for ln in lines:
            annotated_lines.append(ln)
            if re.match(r"^\s{0,3}#{1,2}\s+\S", ln):
                annotated_lines.append(marker)
    else:
        non_empty_count = 0
        for ln in lines:
            annotated_lines.append(ln)
            if ln.strip():
                non_empty_count += 1
                if non_empty_count % max(10, int(line_interval)) == 0:
                    annotated_lines.append(marker)

    return "\n".join(annotated_lines)


def annotate_txt_file_inplace(file_path: str, source_label: str, line_interval: int = 80) -> bool:
    """Open a .txt file and inject source markers in-place. Returns True if updated."""

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
            original = handle.read()
        annotated = insert_source_markers(original, source_label, line_interval=line_interval)
        if annotated == original:
            return False
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write(annotated)
        return True
    except Exception:
        return False


__all__ = [
    "annotate_txt_file_inplace",
    "bilingual_pair_prune",
    "collect_txt_files",
    "deduplicate_conservatively",
    "drop_high_frequency_boilerplate",
    "insert_source_markers",
    "normalize_for_dedup",
    "preprocess_txt_directories",
    "process_file",
    "process_text",
    "read_text",
    "remove_long_latex_like",
    "strip_html_and_artifacts",
    "write_text",
]
