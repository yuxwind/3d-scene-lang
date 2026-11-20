#!/usr/bin/env python3
"""VSI-Bench score table from scores*.json files ({method: {n, overall, <task>_accuracy|_mra: float}}).

usage: vsibench_scores.py RUN_DIR_OR_FILE [...] [--method REGEX] [--all-methods] [--csv OUT]
  RUN_DIR: every scores*.json directly inside it is read (scores.json, scores_all.json, scores_3runs.json, ...).
  --method: keep methods whose name matches (default 'claude', case-insensitive); --all-methods keeps every method.
prints: one row per (run, file, method): n, overall, then the 8 VSI-Bench tasks (short names). Numbers as stored.
--csv: also write the full table to OUT (on the host) and print its path.
stdlib only.
"""
import argparse, csv, glob, json, os, re

TASKS = [("obj_appearance_order_accuracy", "app_ord"), ("object_abs_distance_mra", "abs_dist"),
         ("object_counting_mra", "count"), ("object_rel_distance_accuracy", "rel_dist"),
         ("object_size_estimation_mra", "obj_size"), ("room_size_estimation_mra", "room_sz"),
         ("route_planning_accuracy", "route"), ("object_rel_direction_accuracy", "rel_dir")]


def score_files(p):
    if os.path.isfile(p):
        return [p]
    return sorted(glob.glob(os.path.join(p, "scores*.json")))


def rows(paths, pat, every):
    for p in paths:
        fs = score_files(p)
        if not fs:
            yield {"run": os.path.basename(p.rstrip("/")), "file": "(no scores*.json)", "method": ""}
        for f in fs:
            d = json.load(open(f))
            if not isinstance(d, dict):
                continue
            for m, s in d.items():
                if not isinstance(s, dict) or (not every and not re.search(pat, m, re.I)):
                    continue
                r = {"run": os.path.basename(os.path.dirname(os.path.abspath(f))), "file": os.path.basename(f),
                     "method": m, "n": s.get("n"), "overall": s.get("overall")}
                r.update({short: s.get(k) for k, short in TASKS})
                extra = [k for k in s if k not in ("n", "overall") and k not in dict(TASKS)]
                if extra:
                    r["other"] = ",".join(extra)
                yield r


def fmt(v):
    return "-" if v is None else (f"{v:.1f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--method", default="claude")
    ap.add_argument("--all-methods", action="store_true")
    ap.add_argument("--csv")
    a = ap.parse_args()
    rs = list(rows(a.paths, a.method, a.all_methods))
    cols = ["n", "overall"] + [s for _, s in TASKS]
    print(f"{'run':34} {'file':18} {'method':26} " + " ".join(f"{c:>8}" for c in cols))
    for r in rs:
        print(f"{r['run'][:34]:34} {r['file'][:18]:18} {r.get('method','')[:26]:26} "
              + " ".join(f"{fmt(r.get(c)):>8}" for c in cols) + (f"  +{r['other']}" if r.get("other") else ""))
    if a.csv:
        keys = ["run", "file", "method"] + cols + ["other"]
        with open(a.csv, "w", newline="") as h:
            w = csv.DictWriter(h, keys, extrasaction="ignore"); w.writeheader(); w.writerows(rs)
        print(f"csv: {a.csv} ({len(rs)} rows)")


if __name__ == "__main__":
    main()
