#!/usr/bin/env python3
"""ReVSI relative distance / direction per question type: accuracy, chance, answer balance, and how often a run gives the
answer of the FLIPPED question (farthest -> closest; facing in the opposite direction of B -> facing B), which is what a model
that ignores "farthest" / "opposite" would answer. The flipped answers (and a check of the true ones) are computed from the
ground-truth scene with the code solver's rules (data/revsi_gt_scene/code_solver.py), on the Claude GT batches.

usage: revsi_relq_check.py --batches DIR --rows ROWS.jsonl [--preds NAME=RUN_DIR ...] [--answers NAME=RUN_DIR ...]
  --batches  claude_nocode_s3/revsi_gt/batches (scene + question text per id)
  --rows     ReVSI rows with id, revsi_type (or question_type), options, ground_truth (e.g. rows_s3q_roomdoor_full.jsonl)
  --preds    eval_scenelang runs (predictions_shard*.jsonl; answer = last option letter, as score_revsi_preds.py)
  --answers  Claude / code-solver runs (out/*.json; answer = first word, as revsi_score_answers.py)
prints per type: n, options, chance, GT answer spread, then per run: accuracy, % flipped answer. numpy only.
"""
import argparse, collections, glob, importlib.util, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("cs", os.path.join(HERE, "../../data/revsi_gt_scene/code_solver.py"))
cs = importlib.util.module_from_spec(spec); spec.loader.exec_module(cs)
LETTER = re.compile(r"\b([A-D])\b(?![a-z'])")
TYPES = ["object_rel_distance_closest", "object_rel_distance_farthest", "object_rel_direction_forward_easy",
         "object_rel_direction_backward_easy", "object_rel_direction_forward_hard", "object_rel_direction_backward_hard"]


def flip(q):
    if "is the farthest from the" in q:
        return q.replace("is the farthest from the", "is the closest to the")
    if "facing in the opposite direction of the" in q:
        return q.replace("facing in the opposite direction of the", "facing the")
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--batches", required=True); ap.add_argument("--rows", required=True)
    ap.add_argument("--preds", action="append", default=[]); ap.add_argument("--answers", action="append", default=[])
    a = ap.parse_args()
    scene = {}
    for f in glob.glob(os.path.join(a.batches, "*.json")):
        for it in json.load(open(f))["questions"]:
            scene[int(it["id"])] = it["text"]
    rows = {}
    for l in open(a.rows):
        r = json.loads(l); t = r.get("revsi_type") or r["question_type"]
        if t in TYPES: rows[int(r["id"])] = dict(r, t=t)
    runs = {}
    for spec_ in a.preds:
        n, d = spec_.split("=", 1); P = {}
        for f in glob.glob(os.path.join(d, "predictions_shard*.jsonl")):
            for l in open(f):
                x = json.loads(l); m = LETTER.findall(str(x["prediction"]).split("</think>")[-1]); P[int(x["index"])] = m[-1] if m else "?"
        runs[n] = P
    for spec_ in a.answers:
        n, d = spec_.split("=", 1); P = {}
        for f in glob.glob(os.path.join(d, "out", "*.json")):
            if f.endswith(".meta.json"): continue
            for k, v in json.load(open(f))["answers"].items():
                P[int(k)] = str(v).strip().split(" ")[0].rstrip(".").strip().upper()[:1]
        runs[n] = P
    # flipped (and true) answers from the GT scene
    flipped, check = {}, collections.Counter()
    for i, r in rows.items():
        if i not in scene: continue
        bx, q, opts = cs.parse(scene[i])
        try:
            _, ans = cs.answer(bx, q, opts); check["gt_rule_ok"] += ans == r["ground_truth"]; check["gt_rule_n"] += 1
            fq = flip(q)
            if fq: flipped[i] = cs.answer(bx, fq, opts)[1]
        except (KeyError, ValueError, AttributeError, IndexError):
            check["rule_failed"] += 1
    print(f"GT-scene rule reproduces the benchmark answer on {check['gt_rule_ok']}/{check['gt_rule_n']} questions; "
          f"flipped answers for {len(flipped)} (rule failed {check['rule_failed']})")
    for t in TYPES:
        ids = [i for i, r in rows.items() if r["t"] == t]
        if not ids: continue
        nopt = collections.Counter(len(rows[i]["options"]) for i in ids).most_common(1)[0][0]
        txt = lambda i, L: next((o.split(". ", 1)[1].strip().lower() for o in rows[i]["options"] if o[0] == L), L)
        spread = collections.Counter(txt(i, rows[i]["ground_truth"]) for i in ids) if "direction" in t else None
        fl = [i for i in ids if i in flipped]; diff = [i for i in fl if flipped[i] != rows[i]["ground_truth"]]
        print(f"\n{t}: n {len(ids)}, {nopt} options (chance {100 / nopt:.0f}%)"
              + (f", GT answers {dict(spread.most_common())}" if spread else "")
              + (f"; flipped question has a different answer on {len(diff)}/{len(fl)}" if fl else ""))
        for n, P in runs.items():
            got = [i for i in ids if i in P]
            if not got: continue
            acc = 100 * sum(P[i] == rows[i]["ground_truth"] for i in got) / len(got)
            s = f"   {n:24} acc {acc:5.1f} (n{len(got)})"
            d2 = [i for i in diff if i in P]
            if d2:
                s += f" | on the {len(d2)} where the flipped answer differs: true {100 * sum(P[i] == rows[i]['ground_truth'] for i in d2) / len(d2):5.1f}%, flipped {100 * sum(P[i] == flipped[i] for i in d2) / len(d2):5.1f}%"
            if "direction" in t:
                pc = collections.Counter(txt(i, P[i]) for i in got)
                s += f" | answers {dict(pc.most_common(4))}"
            print(s)


if __name__ == "__main__":
    main()
