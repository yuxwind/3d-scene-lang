#!/usr/bin/env python3
"""Which question types get a scene (boxes) and which are video only, in eval rows or a training mix.

usage: scene_usage.py FILE [...]      FILE: .jsonl or .jsonl.gz
Per row: question type = question_type | type | task | qtype field (else "?"); scene = the use_scene field when the row has it,
else whether a "<scene>" block or a non-empty scene_language appears anywhere in the row (prompt text included).
prints per file and question type: rows, with scene, video only, and the field the decision came from. stdlib only.
"""
import argparse, collections, gzip, json, re


def qtype(r):
    for k in ("question_type", "type", "task", "qtype"):
        if isinstance(r.get(k), str):
            return r[k]
    m = re.search(r'"(?:question_type|qtype)":\s*"([^"]+)"', json.dumps(r))
    return m.group(1) if m else "?"


def has_scene(r):
    if "use_scene" in r:
        return bool(r["use_scene"]), "use_scene"
    s = json.dumps(r)
    if "<scene>" in s:
        return True, "<scene> in text"
    sl = r.get("scene_language")
    if sl not in (None, "", "[]"):
        return True, "scene_language"
    return False, "no scene found"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); a = ap.parse_args()
    for f in a.files:
        op = gzip.open if f.endswith(".gz") else open
        c = collections.defaultdict(lambda: [0, 0]); how = collections.Counter(); keys = None
        with op(f, "rt") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                keys = keys or sorted(r)[:12]
                s, h = has_scene(r)
                c[qtype(r)][s] += 1; how[h] += 1
        print(f"== {f}: {sum(sum(v) for v in c.values())} rows; decided by {dict(how)}; row keys {keys}")
        print(f"   {'question type':42} {'rows':>7} {'scene':>7} {'video only':>10}")
        for t, (vo, sc) in sorted(c.items(), key=lambda x: -sum(x[1])):
            print(f"   {t:42} {vo + sc:7} {sc:7} {vo:10}")


if __name__ == "__main__":
    main()
