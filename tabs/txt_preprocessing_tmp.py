#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple

BASE_DIR = Path(__file__).resolve().parent
EXAMINED_DIR = BASE_DIR / "examined_txt"
STANDARDS_DIR = BASE_DIR / "standards_txt"

# Reliable, structure-agnostic cleanup configuration
HTML_TAG_RE = re.compile(r"<[^>]+>")
# Lines composed only of commas/semicolons/pipes or whitespace (CSV artifacts)
SEPARATOR_LINE_RE = re.compile(r"^[,;|\s]+$")
# CSV artifact columns like Unnamed: 1
UNNAMED_COL_RE = re.compile(r"\bUnnamed:\s*\d+\b", re.IGNORECASE)
# Normalization helpers
MULTISPACE_RE = re.compile(r"\s+")
DOT_LEADERS_RE = re.compile(r"\.{2,}")
BRACKETS_RE = re.compile(r"^[\[\(\{\s]+|[\]\)\}\s]+$")
# HTML table parsing (lightweight)
TABLE_RE = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)
TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
TD_TH_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)

# High-frequency boilerplate threshold within a single file
HIGH_FREQ_THRESHOLD = 3


def read_text(file_path: Path) -> str:
	try:
		return file_path.read_text(encoding="utf-8")
	except UnicodeDecodeError:
		# Fallback to gbk commonly seen in CN docs
		return file_path.read_text(encoding="gbk", errors="ignore")


def write_text(file_path: Path, content: str) -> None:
	file_path.write_text(content, encoding="utf-8")


def strip_html_and_artifacts(line: str) -> str:
	# Preserve HTML; only clean CSV artifacts and trivial formatting
	line_no_unnamed = UNNAMED_COL_RE.sub("", line)
	# Collapse dotted leaders like "....."
	line_no_dots = DOT_LEADERS_RE.sub(" ", line_no_unnamed)
	# Trim superfluous surrounding brackets
	line_no_brackets = BRACKETS_RE.sub("", line_no_dots)
	# Normalize spaces
	line_norm = MULTISPACE_RE.sub(" ", line_no_brackets).strip()
	return line_norm


def is_separator_line(line: str) -> bool:
	return bool(SEPARATOR_LINE_RE.match(line))


def normalize_for_dedup(line: str) -> str:
	# Keep case-sensitive for CJK safety but trim spaces and collapse leaders
	line = line.strip()
	line = DOT_LEADERS_RE.sub(" ", line)
	line = MULTISPACE_RE.sub(" ", line)
	return line


def is_source_stamp(line: str) -> bool:
	# Explicitly preserve all stamps per user request
	return line.strip().startswith("【来源文件:")


def drop_high_frequency_boilerplate(lines: List[str]) -> List[str]:
	"""Drop lines that recur many times and carry no units/numbers content.
	Preserve source stamps."""
	freq = {}
	norm_cache = {}
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
		# Basic safeguard: keep lines with engineering units/numbers
		has_units = bool(re.search(r"(℃|°C|kPa|V|A|W|Wh|W·h|Ω|Ω/V|mm|cm|m|s|min|h|%|kW|MW|MWh)", line))
		has_digits = any(ch.isdigit() for ch in line)
		if not norm:
			continue
		if freq.get(norm, 0) >= HIGH_FREQ_THRESHOLD and not has_units and not has_digits:
			# Drop very frequent boilerplate without numeric content
			continue
		result.append(line)
	return result


def deduplicate_conservatively(lines: List[str]) -> List[str]:
	"""Remove exact or normalized duplicates while preserving stamps.
	We dedup by normalized string but keep first occurrence order."""
	seen = set()
	out: List[str] = []
	for line in lines:
		if is_source_stamp(line):
			out.append(line)
			continue
		norm = normalize_for_dedup(line)
		if not norm:
			continue
		key = norm
		if key in seen:
			continue
		seen.add(key)
		out.append(line)
	return out


def bilingual_pair_prune(lines: List[str]) -> List[str]:
	"""Prune adjacent bilingual duplicates cautiously.
	We keep the first line and drop the second only if:
	- The pair are adjacent (or separated by an empty line), and
	- They share identical numeric tokens and units sequence after stripping script-specific chars.
	"""
	def tokenize_numbers_units(s: str) -> Tuple[str, ...]:
		# Extract numbers/units/punctuation tokens for comparison
		squeezed = MULTISPACE_RE.sub(" ", s)
		tokens = re.findall(r"\d+|[A-Za-z%°℃Ω/·\-]+", squeezed)
		return tuple(tokens)

	def has_cjk(s: str) -> bool:
		return any('\u4e00' <= ch <= '\u9fff' for ch in s)

	def has_latin(s: str) -> bool:
		return any('A' <= ch <= 'Z' or 'a' <= ch <= 'z' for ch in s)

	out: List[str] = []
	i = 0
	while i < len(lines):
		line = lines[i]
		out.append(line)
		# Skip pruning for source stamps
		if is_source_stamp(line):
			i += 1
			continue
		# Lookahead 1 line (or skip empty)
		j = i + 1
		if j < len(lines) and not lines[j].strip():
			out.append(lines[j])
			j += 1
		if j < len(lines):
			l1 = normalize_for_dedup(line)
			l2 = normalize_for_dedup(lines[j])
			if l1 and l2 and l1 != l2:
				# Check bilingual nature
				if (has_cjk(l1) and has_latin(l2)) or (has_latin(l1) and has_cjk(l2)):
					# Compare numeric/unit tokens
					t1 = tokenize_numbers_units(l1)
					t2 = tokenize_numbers_units(l2)
					if t1 == t2 and t1:
						# Drop the lookahead line as a bilingual duplicate
						i = j
						# Do not append lines[j]
						i += 1
						continue
		i += 1
	return out


def remove_long_latex_like(content: str, min_length: int = 300) -> str:
	r"""Remove long LaTeX-like blobs (>min_length chars) anywhere in the text.
	Heuristics:
	- Only remove if the candidate span is LaTeX-dense: many backslashes and/or dollars, or explicit array/align environments, and low CJK text ratio.
	- Tolerate truncated starts like 'gin{array}'.
	"""
	env_tokens = ("array", "align", "aligned", "eqnarray", "pmatrix", "bmatrix")
	latex_triggers = re.compile(
		 r"(?is)(\\begin\s*\{(?:array|align|aligned|eqnarray|pmatrix|bmatrix)\}|\\\[|\$\$|\\math[a-zA-Z]*|\\frac|\\sum|\\int|\\left|\\right|\\mathrm|\\mathcal|\\mathbb|(?:^|\W)gin\{array)"
	)

	def stats(segment: str) -> Tuple[int, int, int, int, float]:
		bs = segment.count('\\')
		dollars = segment.count('$')
		braces = segment.count('{') + segment.count('}')
		# crude CJK count
		cjk = sum(1 for ch in segment if '\u4e00' <= ch <= '\u9fff')
		letters = sum(1 for ch in segment if ch.isalpha())
		length = len(segment)
		cjk_ratio = cjk / max(1, length)
		return bs, dollars, braces, letters, cjk_ratio

	def looks_like_latex_block(segment: str) -> bool:
		bs, dollars, braces, letters, cjk_ratio = stats(segment)
		contains_env = any(tok in segment for tok in env_tokens)
		# Strong signals: explicit env or $$ ... $$
		if contains_env or segment.strip().startswith('$$'):
			return True
		# Otherwise require high latex density: many backslashes and braces and dollars
		if bs >= 10 and (dollars >= 4 or braces >= 30) and cjk_ratio <= 0.10:
			return True
		return False

	to_remove: List[Tuple[int, int]] = []
	for m in latex_triggers.finditer(content):
		start = max(0, m.start() - 20)
		end = m.end()
		while end < len(content) and (end - start) < min_length:
			end = min(len(content), end + 200)
		candidate = content[start:end]
		if not looks_like_latex_block(candidate):
			continue
		# Grow forward while it still looks latex-dense
		while end < len(content):
			next_end = min(len(content), end + 200)
			ext = content[start:next_end]
			if not looks_like_latex_block(ext):
				break
			end = next_end
		# Only remove if final span is long enough and latex-like
		if (end - start) >= min_length and looks_like_latex_block(content[start:end]):
			to_remove.append((start, end))

	if not to_remove:
		return content
	to_remove.sort()
	merged: List[Tuple[int, int]] = []
	cs, ce = to_remove[0]
	for s, e in to_remove[1:]:
		if s <= ce:
			ce = max(ce, e)
		else:
			merged.append((cs, ce))
			cs, ce = s, e
	merged.append((cs, ce))

	out_parts: List[str] = []
	prev = 0
	for s, e in merged:
		out_parts.append(content[prev:s])
		prev = e
	out_parts.append(content[prev:])
	return "".join(out_parts)


def convert_html_tables_to_delimited(text: str) -> str:
	"""Convert <table> blocks to pipe-delimited text while approximating rowspan carry-forward.
	- Uses first row as header if present.
	- For subsequent rows with fewer cells, left-fill from previous data row to mimic rowspans.
	"""
	def clean_cell_html(s: str) -> str:
		# Remove nested tags and normalize whitespace
		no_tags = HTML_TAG_RE.sub("", s)
		no_dots = DOT_LEADERS_RE.sub(" ", no_tags)
		return MULTISPACE_RE.sub(" ", no_dots).strip()

	result_parts: List[str] = []
	pos = 0
	for m in TABLE_RE.finditer(text):
		# Append text before table
		result_parts.append(text[pos:m.start()])
		table_html = m.group(0)
		rows_html = TR_RE.findall(table_html)
		rows: List[List[str]] = []
		for rh in rows_html:
			cells = [clean_cell_html(c) for c in TD_TH_RE.findall(rh)]
			# Drop empty trailing cells
			while cells and not cells[-1].strip():
				cells.pop()
			rows.append(cells)
		# Build delimited lines
		lines: List[str] = []
		# Determine header length
		header_len = 0
		for r in rows:
			if r:
				header_len = max(header_len, len(r))
		if not rows or header_len == 0:
			# Fallback: remove all tags
			plain = HTML_TAG_RE.sub("", table_html)
			result_parts.append(plain)
			pos = m.end()
			continue
		# Use first row as header if it seems header-like (all short tokens or contains header cues), otherwise treat as data
		header = rows[0]
		use_header = True if header else False
		if use_header:
			lines.append(" | ".join(cell.strip() for cell in header))
			carry_prefix: List[str] = []
			# Find first data row to initialize carry_prefix when possible
			for i, row in enumerate(rows[1:], start=1):
				if not any(c.strip() for c in row):
					continue
				if len(row) == header_len:
					carry_prefix = row[:]
					lines.append(" | ".join(c.strip() for c in row))
					start_idx = i + 1
					break
				else:
					# If first data row already short, we cannot infer; just pad right
					needed = header_len - len(row)
					padded = row + [""] * max(0, needed)
					carry_prefix = padded[:]
					lines.append(" | ".join(c.strip() for c in padded))
					start_idx = i + 1
					break
			else:
				start_idx = 1
			# Process remaining rows with carry-forward for left columns
			for row in rows[start_idx:]:
				if not any(c.strip() for c in row):
					continue
				if carry_prefix and len(row) < header_len:
					prefix_needed = header_len - len(row)
					filled = carry_prefix[:prefix_needed] + row
				else:
					filled = row
				if len(filled) < header_len:
					filled = filled + [""] * (header_len - len(filled))
				lines.append(" | ".join(c.strip() for c in filled))
				carry_prefix = filled[:]
		else:
			# No header; just join rows plainly
			for row in rows:
				if not row:
					continue
				lines.append(" | ".join(c.strip() for c in row))
		result_parts.append("\n".join(lines) + "\n")
		pos = m.end()
	# Append tail after last table
	result_parts.append(text[pos:])
	return "".join(result_parts)


def process_text(content: str) -> str:
	# First, remove long LaTeX-like blocks conservatively
	content = remove_long_latex_like(content, min_length=300)
	# Keep HTML as-is (no table conversion)
	# Split preserving line boundaries
	raw_lines = content.splitlines()
	# 1) Strip artifacts, drop separator-only lines
	stage1: List[str] = []
	for line in raw_lines:
		clean = strip_html_and_artifacts(line)
		if not clean:
			continue
		if is_separator_line(clean):
			continue
		stage1.append(clean)
	# 2) Drop high-frequency boilerplate (except source stamps)
	stage2 = drop_high_frequency_boilerplate(stage1)
	# 3) Deduplicate conservatively (exact/normalized)
	stage3 = deduplicate_conservatively(stage2)
	# 4) Bilingual pair prune (adjacent/nearby only, numeric/units match)
	stage4 = bilingual_pair_prune(stage3)
	# Join with newline
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
	for p in sorted(root.glob("*.txt")):
		if p.is_file():
			yield p


def main() -> None:
	changed_files: List[Path] = []
	for folder in (EXAMINED_DIR, STANDARDS_DIR):
		for txt in collect_txt_files(folder):
			if process_file(txt):
				changed_files.append(txt)
	print(f"Processed {len(list(collect_txt_files(EXAMINED_DIR)))} examined files, {len(list(collect_txt_files(STANDARDS_DIR)))} standards files.")
	print(f"Updated {len(changed_files)} files:")
	for p in changed_files:
		print(f" - {p}")


if __name__ == "__main__":
	main()
