"""Structured PDF-to-TXT conversion utilities for engineering drawings.

This module extracts vector PDFs into a grid-aware text format so that panel- or
zone-based drawings retain their spatial grouping when processed by downstream
LLM pipelines. It also exposes a lightweight vector-text detector to decide
when the structured pipeline is appropriate versus falling back to OCR-based
conversion.

Key features
============
* Automatically infers a page's logical grid (rows × cols) by clustering word
  positions when the layout is not specified.
* Emits section headers for each page and cell, preserving a deterministic
  reading order within cells.
* Provides :func:`has_vector_text` to quickly determine whether a PDF contains
  extractable text (typical of vector drawings) or is likely a scanned image.

Usage
=====
```
python pdf_to_structured_txt.py input.pdf output.txt
```

Optional flags:
* ``--rows`` / ``--cols``: force a fixed grid per page (otherwise inferred).
* ``--max-rows-auto`` / ``--max-cols-auto``: cap the inference search space.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import fitz  # PyMuPDF

UNIT_PATTERN = r"(?:mm|cm|μm|MPa|N|V|Ω|m|A|s|kg|g|Pa|kPa)"
PLUS_SUFFIX_RE = re.compile(
    rf"^(?P<head>.*?)(?P<plus>\+\d+(?:\.\d+)?)(?P<unit>\s*{UNIT_PATTERN})?$"
)
NEXT_VALUE_RE = re.compile(
    rf"^(?P<sign>[+-]?)\s*(?P<value>\d+(?:\.\d+)?)(?:\s*(?P<unit>{UNIT_PATTERN}))?$"
)
TOLERANCE_SEPARATOR_TOKENS = {"/", "／"}


def has_vector_text(pdf_path: str, *, min_words: int = 15, sample_pages: int = 3) -> bool:
    """Heuristically detect whether a PDF likely contains vector text.

    A PDF is considered "vector" if at least ``min_words`` textual tokens are
    found across up to ``sample_pages`` pages. Scanned PDFs typically produce no
    words via PyMuPDF's extractor.
    """

    doc = fitz.open(pdf_path)
    try:
        words_found = 0
        for page_index in range(min(len(doc), max(sample_pages, 1))):
            page_words = doc[page_index].get_text("words")
            words_found += sum(1 for word in page_words if str(word[4]).strip())
            if words_found >= min_words:
                return True
        return False
    finally:
        doc.close()


def infer_axis_clusters(
    coords: List[float],
    axis_length: float,
    max_clusters: int = 4,
    tolerance: float = 0.15,
) -> Tuple[int, List[float]]:
    """
    Infer how many clusters (1..max_clusters) best fit the 1D coordinate
    distribution along an axis, assuming clusters are equally spaced.

    Returns:
        (best_k, centers) where centers is a list of cluster centers.
    """

    if not coords:
        return 1, [axis_length * 0.5]

    n = len(coords)
    best_costs = {}
    centers_by_k = {}

    for k in range(1, max_clusters + 1):
        centers = [(i + 0.5) * axis_length / k for i in range(k)]
        centers_by_k[k] = centers

        total_cost = 0.0
        for v in coords:
            nearest = min(centers, key=lambda c: abs(v - c))
            d = v - nearest
            total_cost += d * d

        best_costs[k] = total_cost / n

    min_k = min(best_costs, key=best_costs.get)
    min_cost = best_costs[min_k]

    chosen_k = min(
        (k for k in best_costs if best_costs[k] <= (1.0 + tolerance) * min_cost),
        key=lambda k: k,
    )

    return chosen_k, centers_by_k[chosen_k]


def assign_to_center(value: float, centers: List[float]) -> int:
    """Assign a scalar value to the index of its nearest center."""

    return min(range(len(centers)), key=lambda i: abs(value - centers[i]))


def detect_vertical_frames(
    coords: Sequence[float],
    axis_length: float,
    *,
    gap_ratio: float = 0.18,
    min_width_ratio: float = 0.15,
    max_frames: int = 3,
) -> List[Tuple[float, float]]:
    """Split a page into vertical frames when large empty gaps are detected."""

    if not coords:
        return [(0.0, axis_length)]

    sorted_coords = sorted(coords)
    threshold = axis_length * gap_ratio
    boundaries = [0.0]

    prev_val = sorted_coords[0]
    for value in sorted_coords[1:]:
        if value - prev_val > threshold and len(boundaries) < max_frames:
            boundaries.append(0.5 * (prev_val + value))
        prev_val = value

    boundaries.append(axis_length)
    frames: List[Tuple[float, float]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start >= axis_length * min_width_ratio:
            frames.append((max(0.0, start), min(axis_length, end)))

    if frames:
        return frames

    # Histogram-based fallback to catch narrow-but-empty gutters.
    bin_count = max(20, int(axis_length / 50))
    bin_width = axis_length / bin_count
    counts = [0] * bin_count
    for value in sorted_coords:
        idx = min(bin_count - 1, max(0, int(value / bin_width)))
        counts[idx] += 1

    quiet_threshold = max(1, int(len(coords) * 0.01))
    run_start = None
    candidate_boundaries: List[float] = []

    for idx, count in enumerate(counts):
        if count <= quiet_threshold:
            if run_start is None:
                run_start = idx
        else:
            if run_start is not None:
                run_end = idx
                width = (run_end - run_start) * bin_width
                if width >= axis_length * gap_ratio * 0.5:
                    center = (run_start + run_end) * 0.5 * bin_width
                    if 0.1 * axis_length < center < 0.9 * axis_length:
                        candidate_boundaries.append(center)
                run_start = None
    if run_start is not None:
        run_end = bin_count
        width = (run_end - run_start) * bin_width
        if width >= axis_length * gap_ratio * 0.5:
            center = (run_start + run_end) * 0.5 * bin_width
            if 0.1 * axis_length < center < 0.9 * axis_length:
                candidate_boundaries.append(center)

    if candidate_boundaries:
        candidate_boundaries = sorted(candidate_boundaries)[: max_frames - 1]
        frames = []
        last = 0.0
        for boundary in candidate_boundaries:
            frames.append((last, boundary))
            last = boundary
        frames.append((last, axis_length))
        return frames

    return [(0.0, axis_length)]


def normalize_measurements(text: str) -> str:
    """Normalize tolerant expressions and tighten spacing for numeric tokens."""

    def _subst(pattern: str, repl: str, target: str) -> str:
        return re.sub(pattern, repl, target)

    normalized = text
    normalized = _subst(
        r"(\d(?:[\d\s]*\d)?(?:\.\d+)?)\s*±\s*(\d+(?:\.\d+)?)", r"\1±\2", normalized
    )
    normalized = _subst(
        r"(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)\s*/\s*-\s*(\d+(?:\.\d+)?)",
        r"\1 +\2/-\3",
        normalized,
    )
    normalized = _subst(
        r"(?<=\d)\s*(mm|cm|μm|MPa|N|V)\b", r" \1", normalized
    )
    return normalized


def balance_parentheses(text: str) -> str:
    """Ensure opening parentheses have matching closing counterparts."""

    diff = text.count("(") - text.count(")")
    if diff > 0:
        return text + ")" * diff
    return text


def post_process_lines(lines: List[str]) -> List[str]:
    """Post-process serialized lines to join fragmented tolerances and clean text."""

    processed: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            processed.append(line)
            i += 1
            continue

        match = PLUS_SUFFIX_RE.match(stripped)
        if match:
            lookahead = i + 1
            next_stripped = ""
            while lookahead < len(lines):
                candidate = lines[lookahead].strip()
                if not candidate:
                    lookahead += 1
                    continue
                if candidate in TOLERANCE_SEPARATOR_TOKENS:
                    lookahead += 1
                    continue
                next_stripped = candidate
                break
            if next_stripped:
                next_match = NEXT_VALUE_RE.match(next_stripped)
            if next_match:
                head = match.group("head").rstrip()
                plus = match.group("plus")
                unit = (match.group("unit") or next_match.group("unit") or "").strip()
                lower_sign = next_match.group("sign") or "-"
                if lower_sign == "+":
                    lower_sign = "-"
                lower_value = next_match.group("value")
                unit_text = f" {unit}" if unit else ""
                combined = f"{head} {plus}{unit_text}/{lower_sign}{lower_value}{unit_text}".strip()
                processed.append(balance_parentheses(combined))
                i = lookahead + 1
                continue

        processed.append(balance_parentheses(stripped))
        i += 1

    return processed


def serialize_words(words: Iterable[Tuple[float, float, float, float, str, int, int, int]]) -> List[str]:
    """Serialize words sorted by (block,line,word) into contiguous text lines."""

    sorted_words = sorted(words, key=lambda w: (w[5], w[6], w[7], w[1], w[0]))
    lines: List[str] = []
    current_block = None
    current_line = None
    line_buf: List[str] = []

    def flush_line() -> None:
        if not line_buf:
            return
        joined = " ".join(line_buf)
        lines.append(normalize_measurements(joined))
        line_buf.clear()

    for (_, _, _, _, text, block_no, line_no, _word_no) in sorted_words:
        if not text.strip():
            continue

        if current_block is None:
            current_block = block_no
            current_line = line_no

        if block_no != current_block:
            flush_line()
            lines.append("")
            current_block = block_no
            current_line = line_no
        elif line_no != current_line:
            flush_line()
            current_line = line_no

        line_buf.append(text)

    flush_line()
    processed = post_process_lines(lines)
    if processed and processed[-1] != "":
        processed.append("")
    return processed


def extract_pdf_to_txt(
    pdf_path: str,
    txt_path: str,
    *,
    rows: int | None = None,
    cols: int | None = None,
    max_rows_auto: int = 4,
    max_cols_auto: int = 4,
    use_grid: bool = False,
    sheet_gap_ratio: float = 0.18,
    emit_metadata: bool = True,
    metadata_path: str | None = None,
) -> None:
    """Convert a vector PDF to a structured TXT with grid-aware grouping."""

    doc = fitz.open(pdf_path)
    out_lines: List[str] = []
    metadata: dict = {
        "source_pdf": str(pdf_path),
        "pages": [],
    }

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            width, height = page.rect.width, page.rect.height

            words = page.get_text("words")
            if not words:
                continue

            page_meta: dict = {
                "index": page_index + 1,
                "use_grid": use_grid,
                "sheets": [],
            }

            cx_list = [
                0.5 * (x0 + x1)
                for (x0, _y0, x1, _y1, text, *_rest) in words
                if text.strip()
            ]
            body_cx_list = [
                0.5 * (x0 + x1)
                for (x0, y0, x1, y1, text, *_rest) in words
                if text.strip() and 0.5 * (y0 + y1) >= height * 0.2
            ]
            frame_coords = body_cx_list if len(body_cx_list) >= 3 else cx_list
            frames = detect_vertical_frames(
                frame_coords, width, gap_ratio=sheet_gap_ratio
            )

            for sheet_idx, (frame_start, frame_end) in enumerate(frames, start=1):
                sheet_words = [
                    (x0, y0, x1, y1, text, block_no, line_no, word_no)
                    for (x0, y0, x1, y1, text, block_no, line_no, word_no) in words
                    if text.strip()
                    and frame_start <= 0.5 * (x0 + x1) <= frame_end
                ]
                if not sheet_words:
                    continue

                out_lines.append(
                    f"### Page {page_index + 1} | Sheet {sheet_idx}"
                )

                sheet_meta = {
                    "sheet_index": sheet_idx,
                    "word_count": len(sheet_words),
                }

                if use_grid:
                    local_width = frame_end - frame_start
                    cy_list = [
                        0.5 * (y0 + y1)
                        for (x0, y0, x1, y1, _text, *_rest) in sheet_words
                    ]
                    cx_local = [
                        (0.5 * (x0 + x1)) - frame_start for (x0, _y0, x1, _y1, *_rest) in sheet_words
                    ]

                    if rows is None:
                        inferred_rows, row_centers = infer_axis_clusters(
                            cy_list, height, max_clusters=max_rows_auto
                        )
                    else:
                        inferred_rows = rows
                        row_centers = [
                            (i + 0.5) * height / inferred_rows for i in range(inferred_rows)
                        ]

                    if cols is None:
                        inferred_cols, col_centers = infer_axis_clusters(
                            cx_local, local_width, max_clusters=max_cols_auto
                        )
                    else:
                        inferred_cols = cols
                        col_centers = [
                            (i + 0.5) * local_width / inferred_cols
                            for i in range(inferred_cols)
                        ]

                    buckets = {
                        (r, c): []
                        for r in range(inferred_rows)
                        for c in range(inferred_cols)
                    }

                    for (x0, y0, x1, y1, text, block_no, line_no, word_no) in sheet_words:
                        cx = 0.5 * (x0 + x1) - frame_start
                        cy = 0.5 * (y0 + y1)
                        r = assign_to_center(cy, row_centers)
                        c = assign_to_center(cx, col_centers)
                        buckets[(r, c)].append(
                            (x0, y0, x1, y1, text, block_no, line_no, word_no)
                        )

                    out_lines.append(
                        f"######## Grid (rows={inferred_rows}, cols={inferred_cols}) ########"
                    )

                    for r in range(inferred_rows):
                        for c in range(inferred_cols):
                            cell_words = buckets[(r, c)]
                            if not cell_words:
                                continue

                            out_lines.append(
                                f"=== Sheet {sheet_idx} | Cell row {r + 1}, col {c + 1} ==="
                            )
                            out_lines.extend(serialize_words(cell_words))

                    out_lines.append("")
                    sheet_meta["grid_rows"] = inferred_rows
                    sheet_meta["grid_cols"] = inferred_cols
                else:
                    out_lines.extend(serialize_words(sheet_words))

                page_meta["sheets"].append(sheet_meta)

            out_lines.append("")
            metadata["pages"].append(page_meta)
    finally:
        doc.close()

    Path(txt_path).write_text("\n".join(out_lines), encoding="utf-8")

    if emit_metadata:
        meta_path = metadata_path or f"{txt_path}.meta.json"
        Path(meta_path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert CAD-style PDF to structured TXT for LLM input, with "
            "automatic grid (rows x cols) inference."
        )
    )
    parser.add_argument("pdf", help="Input PDF path")
    parser.add_argument("txt", help="Output TXT path")
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Force number of row panels per page (otherwise inferred).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Force number of column panels per page (otherwise inferred).",
    )
    parser.add_argument(
        "--max-rows-auto",
        type=int,
        default=4,
        help="Maximum rows to consider when inferring (default: 4).",
    )
    parser.add_argument(
        "--max-cols-auto",
        type=int,
        default=4,
        help="Maximum cols to consider when inferring (default: 4).",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Enable legacy grid output (default: off).",
    )
    parser.add_argument(
        "--sheet-gap-ratio",
        type=float,
        default=0.22,
        help="Gap ratio used to split wide pages into multiple sheets.",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Disable metadata JSON emission.",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="Optional explicit metadata JSON output path.",
    )
    args = parser.parse_args()

    extract_pdf_to_txt(
        args.pdf,
        args.txt,
        rows=args.rows,
        cols=args.cols,
        max_rows_auto=args.max_rows_auto,
        max_cols_auto=args.max_cols_auto,
        use_grid=args.grid,
        sheet_gap_ratio=args.sheet_gap_ratio,
        emit_metadata=not args.no_metadata,
        metadata_path=args.metadata_path,
    )


if __name__ == "__main__":
    main()
