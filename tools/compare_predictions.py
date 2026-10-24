"""Compare complete prediction file bytes, excluding timing/configuration logs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

SUFFIXES = ("_model.cif", "_confidences.json", "_confidences_aggregated.json")


def prediction_files(root):
    root = Path(root)
    summary = (root / "summary.txt").read_text()
    failed = re.search(r"Failed Queries:\s*(\d+)", summary)
    successful = re.search(r"Successful Queries:\s*(\d+)", summary)
    total = re.search(r"Total Queries Processed:\s*(\d+)", summary)
    if (not failed or not successful or not total or int(failed[1]) != 0
            or int(successful[1]) < 1 or int(total[1]) != int(successful[1])):
        raise ValueError(f"Prediction summary is incomplete or contains failed queries: {root}")
    files = {p.relative_to(root).as_posix(): p for p in root.rglob("*")
             if p.is_file() and p.name.endswith(SUFFIXES)}
    models = [name for name in files if name.endswith("_model.cif")]
    if not models:
        raise ValueError(f"No prediction models found: {root}")
    expected = {name[:-len('_model.cif')] + suffix for name in models for suffix in SUFFIXES}
    if set(files) != expected:
        raise ValueError(f"Incomplete model/confidence file triplets: {root}")
    return files


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def compare(left, right):
    a, b = prediction_files(left), prediction_files(right)
    left_only = sorted(a.keys() - b.keys())
    right_only = sorted(b.keys() - a.keys())
    hashes = {name: {"left": sha256(a[name]), "right": sha256(b[name])}
              for name in sorted(a.keys() & b.keys())}
    different = [name for name, pair in hashes.items() if pair["left"] != pair["right"]]
    return {"identical": not (left_only or right_only or different),
            "left_file_count": len(a), "right_file_count": len(b),
            "left_only": left_only, "right_only": right_only,
            "different": different, "sha256": hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.left, args.right)
    except (OSError, ValueError) as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if result["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
