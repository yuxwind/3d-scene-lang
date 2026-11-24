#!/usr/bin/env python3
"""Score Claude no-code ReVSI answers (RUN_DIR/out/*.json) against RUN_DIR/rows.jsonl, ReVSI protocol.

Same metric as playground/zeroshot_revsi/eval_revsi.py (lmms-eval revsi): first word of the answer; multiple choice
exact match on the letter; numbers MRA over thresholds .50-.95 (rel. error <= 1 - thr); 13 subtypes averaged into
7 tasks (rel. direction 4, rel. distance 2, counting 2, room size 2); overall = mean of the 7.
usage: revsi_score_answers.py RUN_DIR [--gt ROWS] [--by-dataset] [--json OUT]
  --gt: GT rows, default RUN_DIR/rows.jsonl (jz runs). A .parquet (e.g. /scr/xinyu/data/ReVSI/64_frame/
        test-00000-of-00001.parquet, for gandalf claude_nocode_s3/revsi_gt) is read with pandas (training env).
prints: answered/total, overall, the 7 tasks (n). stdlib only, except --gt *.parquet.
"""
import argparse, collections, glob, json, os

MCQ = ["object_rel_direction_forward_easy", "object_rel_direction_backward_easy", "object_rel_direction_forward_hard",
       "object_rel_direction_backward_hard", "object_rel_distance_closest", "object_rel_distance_farthest", "route_planning"]
COMPOSITE = {"object_rel_direction": MCQ[:4], "object_rel_distance": MCQ[4:6],
             "object_counting": ["object_counting_single", "object_counting_multiple"],
             "room_size_estimation": ["room_size_estimation_single", "room_size_estimation_multiple"]}
mean = lambda v: sum(v) / len(v)


def acc(qt, pred, gt):
    p = str(pred).strip().split(" ")[0].rstrip(".").strip()
    if qt in MCQ:
        return float(p.lower() == gt.lower())
    try:
        p, t = float(p), float(gt)
        return mean([float(abs(p - t) / t <= 1 - (0.5 + 0.05 * i)) for i in range(10)])
    except Exception:
        return 0.0


def score(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["question_type"]].append(r["acc"])
    o = {t: (mean(v), len(v)) for t, v in by.items()}
    for name, ts in COMPOSITE.items():
        ks = [t for t in ts if t in o]
        if ks:
            o[name] = (mean([o.pop(k)[0] for k in ks]), sum(len(by[k]) for k in ks))
    return 100 * mean([v for v, _ in o.values()]) if o else 0.0, o


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("--by-dataset", action="store_true")
    ap.add_argument("--json"); ap.add_argument("--gt"); a = ap.parse_args()
    A = {}
    for f in glob.glob(os.path.join(a.run_dir, "out", "*.json")):
        if f.endswith(".meta.json"):
            continue
        try:
            A.update({str(k): v for k, v in json.load(open(f))["answers"].items()})
        except Exception:
            pass
    gt = a.gt or os.path.join(a.run_dir, "rows.jsonl")
    if gt.endswith(".parquet"):
        import pandas as pd
        rows = [{k: r[k] for k in ("id", "question_type", "ground_truth", "dataset")} for r in pd.read_parquet(gt).to_dict("records")]
    else:
        rows = [json.loads(l) for l in open(gt)]
    got = [dict(r, acc=acc(r["question_type"], A[str(r["id"])], r["ground_truth"])) for r in rows if str(r["id"]) in A]
    cost = sum(json.load(open(m)).get("total_cost_usd", 0) or 0 for m in glob.glob(os.path.join(a.run_dir, "out", "*.meta.json")))
    ov, o = score(got)
    print(f"answered {len(got)}/{len(rows)}  overall {ov:.1f}  (cost so far ${cost:.2f})")
    print("   " + " | ".join(f"{k} {100 * v:.1f} (n{n})" for k, (v, n) in sorted(o.items())))
    res = {"all": {"n": len(got), "overall": round(ov, 1), **{k: round(100 * v, 1) for k, (v, _) in o.items()}}}
    if a.by_dataset:
        for ds in sorted({r["dataset"] for r in got}):
            ovd, od = score([r for r in got if r["dataset"] == ds])
            print(f"   {ds:12} n{sum(1 for r in got if r['dataset'] == ds):5} overall {ovd:.1f}")
            res[ds] = {"overall": round(ovd, 1), **{k: round(100 * v, 1) for k, (v, _) in od.items()}}
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1); print(f"json: {a.json}")


if __name__ == "__main__":
    main()
