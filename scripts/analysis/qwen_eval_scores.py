#!/usr/bin/env python3
"""VSI-Bench table for sharded Qwen evals (eval_scenelang.py / sr_vggt.eval outputs): predictions_shard*.jsonl.

usage: qwen_eval_scores.py EVAL_DIR [...] --rows ROWS.jsonl [--csv OUT]
  EVAL_DIR: a folder with predictions_shard*.jsonl (rows {index, question_type, ground_truth, prediction}).
            Shards are merged by index (later files win). A folder name may be given as NAME=DIR to label the row.
  --rows:   any VSI-Bench rows file with {index, debiased} (e.g. data/scenelang_vsibench_slam/
            rows_s3q_roomdoor_radiator_all5130_full.jsonl); it defines the debiased split.
Scoring: sr_vggt.eval.score (the VSI-Bench scorer used for every reported number). For numeric tasks the last
number in the prediction is taken, as in data/scenelang_vsibench_det_objq/cmp_eval.py.
prints: one row per folder: n, full overall, debiased overall (n), then the 8 tasks on the full set (%).
Needs the training env (imports sr_vggt.eval: torch, transformers); remote.py run uses $SR_PYTHON_BIN.
"""
import argparse, csv, glob, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from sr_vggt.eval import score  # noqa: E402

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
NUM = ("object_abs_distance", "object_counting", "object_size_estimation", "room_size_estimation")
TASKS = [("obj_appearance_order_accuracy", "app_ord"), ("object_abs_distance_mra", "abs_dist"),
         ("object_counting_mra", "count"), ("object_rel_distance_accuracy", "rel_dist"),
         ("object_size_estimation_mra", "obj_size"), ("room_size_estimation_mra", "room_sz"),
         ("route_planning_accuracy", "route"), ("object_rel_direction_accuracy", "rel_dir")]


def load(d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "predictions_shard*.jsonl"))):
        for line in open(f):
            try:
                x = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial last line of a killed job
            m = NUMBER.findall(str(x["prediction"]).replace(",", ""))
            if m and x["question_type"] in NUM:
                x["prediction"] = m[-1]
            rows[x["index"]] = x
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--csv")
    a = ap.parse_args()
    deb = set()
    for line in open(os.path.expandvars(a.rows)):
        r = json.loads(line)
        if r.get("debiased"):
            deb.add(r["index"])
    print(f"debiased split: {len(deb)} questions ({os.path.basename(a.rows)})")
    hdr = ["run", "n", "full", "debiased", "n_deb"] + [s for _, s in TASKS]
    print("  ".join(f"{h:>8}" if i else f"{h:44}" for i, h in enumerate(hdr)))
    out = []
    for spec in a.dirs:
        name, _, d = spec.rpartition("=") if "=" in spec else (os.path.basename(spec.rstrip("/")), "", spec)
        d = os.path.expandvars(d)
        rows = load(d)
        if not rows:
            print(f"{name:44}  no predictions_shard*.jsonl")
            continue
        f = score(list(rows.values()))
        b_rows = [v for k, v in rows.items() if k in deb]
        b = score(b_rows) if b_rows else {}
        vals = [name, len(rows), 100 * f["vsibench/overall"], 100 * b.get("vsibench/overall", float("nan")), len(b_rows)] + \
               [100 * f.get("vsibench/" + k, float("nan")) for k, _ in TASKS]
        out.append(vals)
        print("  ".join(f"{v:44}" if i == 0 else (f"{v:>8}" if isinstance(v, int) else f"{v:8.1f}") for i, v in enumerate(vals)))
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            csv.writer(fh).writerows([hdr] + out)
        print(f"csv: {a.csv} ({os.path.getsize(a.csv)} B)")


if __name__ == "__main__":
    main()
