#!/usr/bin/env python3
"""Show the structure of JSON / JSONL result files without dumping them.

usage: inspect_json.py PATH [PATH ...] [--depth 3] [--files 2]
  PATH: a file, a directory (searched recursively for *.json / *.jsonl / *.json.gz / *.jsonl.gz), or a glob.
prints: file count per extension, then for the first --files files: the nested structure
        (dict keys, list lengths, scalar type + one short sample), to --depth levels.
stdlib only.
"""
import argparse, glob, gzip, json, os
from collections import Counter

EXT = (".json", ".jsonl", ".json.gz", ".jsonl.gz")


def files_of(paths):
    out = []
    for p in paths:
        for q in (glob.glob(p) or [p]):
            if os.path.isdir(q):
                for root, _, fs in os.walk(q):
                    out += [os.path.join(root, f) for f in sorted(fs) if f.endswith(EXT)]
            elif q.endswith(EXT) and os.path.exists(q):
                out.append(q)
    return out


def load(f):
    op = gzip.open if f.endswith(".gz") else open
    with op(f, "rt") as h:
        if ".jsonl" in f:
            rows = [json.loads(l) for _, l in zip(range(200), h) if l.strip()]
            return {"<jsonl first rows>": len(rows), "row0": rows[0] if rows else None}
        return json.load(h)


def show(x, depth, ind=""):
    if isinstance(x, dict):
        keys = list(x)
        print(f"{ind}dict[{len(keys)}]")
        if depth == 0:
            print(f"{ind}  keys: {keys[:15]}{' …' if len(keys) > 15 else ''}")
            return
        # many similar keys (e.g. one per question id): show only a few
        for k in keys[:12]:
            print(f"{ind}  {str(k)[:40]}:", end=" ")
            show(x[k], depth - 1, ind + "    ") if isinstance(x[k], (dict, list)) else print(_scalar(x[k]))
        if len(keys) > 12:
            print(f"{ind}  … {len(keys) - 12} more keys")
    elif isinstance(x, list):
        print(f"{ind}list[{len(x)}]")
        if x and depth > 0:
            print(f"{ind}  [0]:", end=" ")
            show(x[0], depth - 1, ind + "    ") if isinstance(x[0], (dict, list)) else print(_scalar(x[0]))
    else:
        print(_scalar(x))


def _scalar(v):
    s = repr(v)
    return f"{type(v).__name__} {s[:60]}{'…' if len(s) > 60 else ''}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--files", type=int, default=2)
    a = ap.parse_args()
    fs = files_of(a.paths)
    c = Counter(next(e for e in EXT[::-1] if f.endswith(e)) for f in fs)
    print(f"{len(fs)} files: " + ", ".join(f"{n} {e}" for e, n in c.most_common()))
    names = Counter(os.path.basename(f) for f in fs)
    print("most common names:", ", ".join(f"{k}×{v}" for k, v in names.most_common(5)))
    for f in fs[: a.files]:
        print(f"\n== {f}")
        try:
            show(load(f), a.depth)
        except Exception as e:  # noqa: BLE001
            print(f"  unreadable: {e}")


if __name__ == "__main__":
    main()
