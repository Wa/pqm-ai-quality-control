import json
import hashlib
import sys


def canonicalize(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def compute_hash(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    canon = canonicalize(obj)
    return hash_text(canon)


def main():
    if len(sys.argv) < 2:
        print("Usage: python normalize_json.py <file1> [file2]")
        sys.exit(2)
    h1 = compute_hash(sys.argv[1])
    print(f"HASH1: {h1}")
    if len(sys.argv) >= 3:
        h2 = compute_hash(sys.argv[2])
        print(f"HASH2: {h2}")
        print("EQUAL:", h1 == h2)


if __name__ == "__main__":
    main()



