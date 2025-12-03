#!/usr/bin/env python3
"""
Convert a CAD-style PDF page into a structured .txt file suitable for LLM input.

New feature:
- Automatically infers the grid layout (rows x cols) per page
  from the distribution of text positions, unless --rows / --cols
  are explicitly specified.

Strategy summary:
- Use PyMuPDF to extract words with positions.
- For each page:
  - Get all word centers (cx, cy).
  - Infer number of rows (1..MAX_ROWS) by choosing the equispaced
    row grid that minimizes squared distance to centers, with a small
    penalty for extra rows.
  - Do the same for columns (1..MAX_COLS).
  - Assign each word to its closest (row, col) grid cell.
- Within each cell, sort by (block_no, line_no, word_no) to approximate
  reading order and dump as text.

Requires:  pip install pymupdf
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF


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
        # Fallback: 1 cluster in middle
        return 1, [axis_length * 0.5]

    n = len(coords)
    best_costs = {}
    centers_by_k = {}

    for k in range(1, max_clusters + 1):
        centers = [(i + 0.5) * axis_length / k for i in range(k)]
        centers_by_k[k] = centers

        total_cost = 0.0
        for v in coords:
            # distance to nearest center
            nearest = min(centers, key=lambda c: abs(v - c))
            d = v - nearest
            total_cost += d * d

        # Use average squared distance as cost
        best_costs[k] = total_cost / n

    # Find minimal cost
    min_k = min(best_costs, key=best_costs.get)
    min_cost = best_costs[min_k]

    # Choose the *smallest* k whose cost is within (1 + tolerance) of min_cost
    chosen_k = min(
        (k for k in best_costs if best_costs[k] <= (1.0 + tolerance) * min_cost),
        key=lambda k: k,
    )

    return chosen_k, centers_by_k[chosen_k]


def assign_to_center(value: float, centers: List[float]) -> int:
    """
    Assign a scalar value to the index of its nearest center.
    """
    return min(range(len(centers)), key=lambda i: abs(value - centers[i]))


def extract_pdf_to_txt(
    pdf_path: str,
    txt_path: str,
    rows: int = None,
    cols: int = None,
    max_rows_auto: int = 4,
    max_cols_auto: int = 4,
):
    doc = fitz.open(pdf_path)
    out_lines = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        width, height = page.rect.width, page.rect.height

        # Get all words on this page
        # words: (x0, y0, x1, y1, text, block_no, line_no, word_no)
        words = page.get_text("words")

        if not words:
            continue

        # Collect centers
        cx_list = []
        cy_list = []
        for (x0, y0, x1, y1, text, block_no, line_no, word_no) in words:
            if not text.strip():
                continue
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            cx_list.append(cx)
            cy_list.append(cy)

        # Infer grid if not specified
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

        # Prepare buckets for each cell
        buckets = {
            (r, c): []
            for r in range(inferred_rows)
            for c in range(inferred_cols)
        }

        # Assign words to nearest row+col center
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

        # Emit text per cell
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

                # Sort words within cell to approximate reading order
                cell_words.sort(key=lambda w: (w[5], w[6], w[7]))

                current_block = None
                current_line = None
                line_buf: List[str] = []

                def flush_line():
                    if line_buf:
                        out_lines.append(" ".join(line_buf))
                        line_buf.clear()

                for (_, _, _, _, text, block_no, line_no, wno) in cell_words:
                    if current_block is None:
                        current_block = block_no
                        current_line = line_no

                    if block_no != current_block:
                        flush_line()
                        out_lines.append("")  # blank line between blocks
                        current_block = block_no
                        current_line = line_no

                    elif line_no != current_line:
                        flush_line()
                        current_line = line_no

                    line_buf.append(text)

                flush_line()
                out_lines.append("")  # blank line after each cell

        out_lines.append("")  # blank line after each page

    doc.close()
    Path(txt_path).write_text("\n".join(out_lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CAD-style PDF to structured TXT for LLM input, "
                    "with automatic grid (rows x cols) inference."
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
