import json
import sys
from pathlib import Path
from typing import List

import pandas as pd


def to_table(records: List[dict]) -> pd.DataFrame:
    rows = []
    max_cols = 0
    for rec in records:
        file_name = rec.get("File")
        sheet_name = rec.get("Sheet")
        data = rec.get("data", []) or []
        for idx, row in enumerate(data, start=1):
            max_cols = max(max_cols, len(row))
            rows.append({
                "File": file_name,
                "Sheet": sheet_name,
                "Row": idx,
                "_cells": row,
            })

    # Build columns C1..Ck
    columns = [f"C{i}" for i in range(1, max_cols + 1)]
    table_rows = []
    for r in rows:
        base = {"File": r["File"], "Sheet": r["Sheet"], "Row": r["Row"]}
        cells = r["_cells"]
        for i, col_name in enumerate(columns, start=1):
            base[col_name] = cells[i - 1] if i - 1 < len(cells) else ""
        table_rows.append(base)

    df = pd.DataFrame(table_rows, columns=["File", "Sheet", "Row"] + columns)
    return df


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_extracted_to_table.py <input_json> [output_base_no_ext]")
        sys.exit(2)

    input_json = Path(sys.argv[1])
    output_base = Path(sys.argv[2]) if len(sys.argv) >= 3 else input_json.with_suffix("")

    with open(input_json, encoding="utf-8") as f:
        data = json.load(f)

    df = to_table(data)

    csv_path = Path(str(output_base) + "_table.csv")
    xlsx_path = Path(str(output_base) + "_table.xlsx")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_excel(xlsx_path, index=False, engine="openpyxl")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {xlsx_path}")


if __name__ == "__main__":
    main()



