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
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF


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


def extract_pdf_to_txt(
    pdf_path: str,
    txt_path: str,
    *,
    rows: int | None = None,
    cols: int | None = None,
    max_rows_auto: int = 4,
    max_cols_auto: int = 4,
) -> None:
    """Convert a vector PDF to a structured TXT with grid-aware grouping."""

    doc = fitz.open(pdf_path)
    out_lines: List[str] = []

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            width, height = page.rect.width, page.rect.height

            words = page.get_text("words")
            if not words:
                continue

            cx_list = []
            cy_list = []
            for (x0, y0, x1, y1, text, block_no, line_no, word_no) in words:
                if not text.strip():
                    continue
                cx_list.append(0.5 * (x0 + x1))
                cy_list.append(0.5 * (y0 + y1))

            if rows is None:
                inferred_rows, row_centers = infer_axis_clusters(
                    cy_list, height, max_clusters=max_rows_auto
                )
            else:
                inferred_rows = rows
                row_centers = [(i + 0.5) * height / inferred_rows for i in range(inferred_rows)]

            if cols is None:
                inferred_cols, col_centers = infer_axis_clusters(
                    cx_list, width, max_clusters=max_cols_auto
                )
            else:
                inferred_cols = cols
                col_centers = [(i + 0.5) * width / inferred_cols for i in range(inferred_cols)]

            buckets = {
                (r, c): []
                for r in range(inferred_rows)
                for c in range(inferred_cols)
            }

            for (x0, y0, x1, y1, text, block_no, line_no, word_no) in words:
                if not text.strip():
                    continue
                cx = 0.5 * (x0 + x1)
                cy = 0.5 * (y0 + y1)
                r = assign_to_center(cy, row_centers)
                c = assign_to_center(cx, col_centers)
                buckets[(r, c)].append(
                    (x0, y0, x1, y1, text, block_no, line_no, word_no)
                )

            out_lines.append(
                f"######## Page {page_index + 1} "
                f"(rows={inferred_rows}, cols={inferred_cols}) ########"
            )
            out_lines.append("")

            for r in range(inferred_rows):
                for c in range(inferred_cols):
                    cell_words = buckets[(r, c)]
                    if not cell_words:
                        continue

                    out_lines.append(
                        f"=== Page {page_index + 1} | Cell row {r + 1}, col {c + 1} ==="
                    )

                    cell_words.sort(key=lambda w: (w[5], w[6], w[7]))

                    current_block = None
                    current_line = None
                    line_buf: List[str] = []

                    def flush_line() -> None:
                        if line_buf:
                            out_lines.append(" ".join(line_buf))
                            line_buf.clear()

                    for (_, _, _, _, text, block_no, line_no, _word_no) in cell_words:
                        if current_block is None:
                            current_block = block_no
                            current_line = line_no

                        if block_no != current_block:
                            flush_line()
                            out_lines.append("")
                            current_block = block_no
                            current_line = line_no

                        elif line_no != current_line:
                            flush_line()
                            current_line = line_no

                        line_buf.append(text)

                    flush_line()
                    out_lines.append("")

            out_lines.append("")
    finally:
        doc.close()

    Path(txt_path).write_text("\n".join(out_lines), encoding="utf-8")


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
    args = parser.parse_args()

    extract_pdf_to_txt(
        args.pdf,
        args.txt,
        rows=args.rows,
        cols=args.cols,
        max_rows_auto=args.max_rows_auto,
        max_cols_auto=args.max_cols_auto,
    )


if __name__ == "__main__":
    main()
