import json
import sys
from typing import List, Tuple


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def first_diffs(a, b, limit: int = 10) -> List[Tuple]:
    diffs = []
    for idx, (sa, sb) in enumerate(zip(a, b)):
        key = (sa.get("File"), sa.get("Sheet"))
        da = sa.get("data", [])
        db = sb.get("data", [])
        if len(da) != len(db):
            diffs.append((key, "rowcount", len(da), len(db)))
            if len(diffs) >= limit:
                break
            continue
        rows = len(da)
        for r in range(rows):
            ra = da[r]
            rb = db[r]
            if len(ra) != len(rb):
                diffs.append((key, "colcount", r, len(ra), len(rb)))
                if len(diffs) >= limit:
                    break
                continue
            cols = len(ra)
            for c in range(cols):
                va = ra[c]
                vb = rb[c]
                if (va or "") != (vb or ""):
                    diffs.append((key, "cell", r, c, va, vb))
                    if len(diffs) >= limit:
                        break
            if len(diffs) >= limit:
                break
        if len(diffs) >= limit:
            break
    return diffs


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_json.py <fileA> <fileB>")
        sys.exit(2)
    p1, p2 = sys.argv[1], sys.argv[2]
    a = load_json(p1)
    b = load_json(p2)

    print(f"sheets(A): {len(a)} sheets(B): {len(b)}")
    ka = [(x.get("File"), x.get("Sheet")) for x in a]
    kb = [(x.get("File"), x.get("Sheet")) for x in b]
    sa, sb = set(ka), set(kb)
    print("missing in A:", list(sb - sa)[:5])
    print("missing in B:", list(sa - sb)[:5])

    difs = first_diffs(a, b, limit=10)
    print("diff_count:", len(difs))
    for d in difs:
        print(d)


if __name__ == "__main__":
    main()



