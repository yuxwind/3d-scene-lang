#!/usr/bin/env python3
"""VSI-Bench: the code solver and model runs scored only on the questions the code solver answers (its answer is not
"none": every named object has a box; appearance order and route planning are never answered).

usage: vsibench_code_answered.py --code LIST=oracle_code_<list>.json [...] --model LIST=EVAL_DIR [...]
  --code   answers of data/scenelang_vsibench_slam/oracle_code_lists.py (a JSON list of rows with index, question_type,
           ground_truth, prediction; "none" = no answer)
  --model  eval_scenelang runs (predictions_shard*.jsonl, loaded as qwen_eval_scores.py), paired with the code list of
           the same LIST name (e.g. the model run on the same box list). Several --model per LIST are fine (NAME@LIST=DIR).
Scoring: sr_vggt.eval.score (the VSI-Bench scorer of every reported number), per task and per rel. direction subtype
(easy / medium / hard); "mean" = mean over the tasks present. Needs the training env ($SR_PYTHON_BIN via remote.py run).
"""
import argparse, collections, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, HERE)
from sr_vggt.eval import score  # noqa: E402
from qwen_eval_scores import load  # noqa: E402

TASKS = [("object_counting", "object_counting_mra", "Count"), ("object_abs_distance", "object_abs_distance_mra", "Abs."),
         ("object_size_estimation", "object_size_estimation_mra", "Size"), ("room_size_estimation", "room_size_estimation_mra", "Room"),
         ("object_rel_distance", "object_rel_distance_accuracy", "RelDist"), ("object_rel_direction", "object_rel_direction_accuracy", "RelDir")]
SUB = ["object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard"]


def table(rows):
    """-> {column: (score %, n)} for the 6 tasks, the rel. direction subtypes and the mean of the tasks."""
    out = {}
    for t, key, col in TASKS:
        r = [x for x in rows if x["question_type"].startswith(t)]
        if r: out[col] = (100 * score(r)["vsibench/" + key], len(r))
    for s in SUB:
        r = [x for x in rows if x["question_type"] == s]
        if r: out["dir " + s.rsplit("_", 1)[1]] = (100 * score(r)["vsibench/object_rel_direction_accuracy"], len(r))
    out["mean"] = (sum(out[c][0] for _, _, c in TASKS if c in out) / max(1, sum(c in out for _, _, c in TASKS)), len(rows))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--code", action="append", required=True)
    ap.add_argument("--model", action="append", default=[]); a = ap.parse_args()
    cols = [c for _, _, c in TASKS] + ["dir easy", "dir medium", "dir hard", "mean"]
    print(f"{'list':10} {'answered by':24} {'n':>5} " + " ".join(f"{c:>10}" for c in cols))
    for spec in a.code:
        L, f = spec.split("=", 1)
        code = [r for r in json.load(open(f)) if str(r["prediction"]).strip().lower() != "none"]
        ids = {r["index"] for r in code}
        runs = [("code", code)]
        for m in a.model:
            name, d = m.split("=", 1)
            nm, _, lst = name.rpartition("@") if "@" in name else ("model", "", name)
            if lst != L: continue
            P = load(d); runs.append((nm, [P[i] for i in ids if i in P]))
        for nm, rows in runs:
            t = table(rows)
            print(f"{L:10} {nm:24} {len(rows):5} " + " ".join(f"{t[c][0]:6.1f} n{t[c][1]:<3}" if c in t else f"{'--':>10}" for c in cols))
        n_t = collections.Counter(r["question_type"] for r in code)
        print(f"{'':10} answered per type: " + ", ".join(f"{k.replace('object_', '')} {v}" for k, v in sorted(n_t.items())))


if __name__ == "__main__":
    main()
