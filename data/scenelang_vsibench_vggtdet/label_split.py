"""Split the sr-q35vs + VGGT-Det predictions by whether the detector predicted the objects each question names.

The question's objects are the ground-truth labels it names (same longest-first, non-overlapping match as
eval_scenelang.rvs_scene, on ../scenelang_vsibench_scannetpp/vsibench_<ds>_scenelang.jsonl). A question is
  all      every named object has at least one VGGT-Det box with that label
  partial  some do, some do not
  none     no named object has a box
  no_obj   it names no object (room size, a few route-planning rows)
For counting, 'all' is further split by whether the number of predicted boxes equals the ground-truth count.

usage: python label_split.py [--split debiased|full] [--dataset scannet|arkitscenes|all]
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D.parents[1]))
from sr_vggt.eval import MCA_QUESTION_TYPES, score  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="debiased", choices=["debiased", "full"])
ap.add_argument("--dataset", default="scannet", choices=["scannet", "arkitscenes", "all"])
ap.add_argument("--out", default=None, help="write per-question rows (jsonl)")
args = ap.parse_args()


def named(question, labels):
    q, spans, keep = question.lower(), [], []
    for lab in sorted(labels, key=len, reverse=True):
        for m in re.finditer(r"\b" + re.escape(lab.lower()) + r"(?:s|es)?\b", q):
            if not any(a <= m.start() < b for a, b in spans):
                spans.append((m.start(), m.end()))
                if lab not in keep:
                    keep.append(lab)
    return keep


def parse(qt, p):
    m = (re.findall(r"\b([A-D])\b(?![a-z'])", p) if qt in MCA_QUESTION_TYPES
         else re.findall(r"-?\d+(?:\.\d+)?", p.replace(",", "")))
    return m[-1] if m else p


gt = {}
for ds in ("scannet", "arkitscenes"):
    for line in open(D.parent / f"scenelang_vsibench_scannetpp/vsibench_{ds}_scenelang.jsonl"):
        r = json.loads(line)
        gt[r["index"]] = {o["label"] for o in json.loads(r["scene_language"])}

rows = {}
for line in open(D / "rows_first_camera.jsonl"):
    r = json.loads(line)
    if (args.split == "full" or r["debiased"]) and args.dataset in ("all", r["dataset"]):
        rows[r["index"]] = r

for variant in ("rvs_question", "rvs_all"):
    preds = {}
    for f in sorted((D / f"sr_q35vs_569_{variant}").glob("predictions_shard*.jsonl")):
        for line in open(f):
            p = json.loads(line)
            preds[p["index"]] = p
    groups, per_q = defaultdict(list), []
    for i, r in rows.items():
        if i not in preds:
            continue
        boxes = [re.sub(r"\s+\d+$", "", o["label"]) for o in json.loads(r["scene_language"])]
        q_objs = named(r["question"], gt[i])
        hit = [o for o in q_objs if o in boxes]
        cat = ("no_obj" if not q_objs or r["question_type"].startswith("room_size")
               else "all" if len(hit) == len(q_objs) else "partial" if hit else "none")
        if cat == "all" and r["question_type"] == "object_counting":
            cat = "all, count right" if boxes.count(q_objs[0]) == int(r["ground_truth"]) else "all, count wrong"
        row = {"question_type": r["question_type"], "ground_truth": r["ground_truth"],
               "prediction": parse(r["question_type"], preds[i]["prediction"])}
        groups[cat].append(row)
        groups["overall"].append(row)
        per_q.append({"index": i, "dataset": r["dataset"], "question_type": r["question_type"], "cat": cat,
                      "question_objects": q_objs, "missing": [o for o in q_objs if o not in boxes],
                      "ground_truth": r["ground_truth"], "prediction": row["prediction"]})
    print(f"\n=== {variant}  ({args.split}, {args.dataset}) ===")
    order = ["overall", "all", "all, count right", "all, count wrong", "partial", "none", "no_obj"]
    for cat in [c for c in order if c in groups]:
        s = score(groups[cat])
        print(f"{cat:18s} n={len(groups[cat]):5d}  overall={s['vsibench/overall']:.4f}")
    # per task x category
    tasks = sorted({q["question_type"].replace("_easy", "").replace("_medium", "").replace("_hard", "")
                    for q in per_q})
    print(f"{'task':28s}" + "".join(f"{c:>22s}" for c in ["all", "partial", "none", "no_obj"]))
    for t in tasks:
        cells = []
        for c in ["all", "partial", "none", "no_obj"]:
            sel = [g for k, v in groups.items() if k == c or (c == "all" and k.startswith("all,"))
                   for g in v if g["question_type"].startswith(t)]
            if sel:
                s = score(sel)
                cells.append(f"{len(sel):5d} / {s['vsibench/overall']:.3f}".rjust(22))
            else:
                cells.append("-".rjust(22))
        print(f"{t:28s}" + "".join(cells))
    if args.out:
        with open(f"{args.out}.{variant}.jsonl", "w") as fh:
            for q in per_q:
                fh.write(json.dumps(q) + "\n")
