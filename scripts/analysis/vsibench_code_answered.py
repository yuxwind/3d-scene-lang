#!/usr/bin/env python3
"""VSI-Bench: the code solver and model runs scored only on the questions the code solver answers (its answer is not
"none": every named object has a box; appearance order and route planning are never answered).

usage: vsibench_code_answered.py --code LIST=oracle_code_<list>.json [...] --model LIST=EVAL_DIR [...]
  --code   answers of data/scenelang_vsibench_slam/oracle_code_lists.py (a JSON list of rows with index, question_type,
           ground_truth, prediction; "none" = no answer)
  --model  eval_scenelang runs (predictions_shard*.jsonl, loaded as qwen_eval_scores.py), paired with the code list of
           the same LIST name (e.g. the model run on the same box list). Several --model per LIST are fine (NAME@LIST=DIR).
           A model may also be a Claude answer dir (claude: prefix, e.g. claude:<run>/oraclecnt: <dir>/out/*.json {"answers":
           {index: answer}}; first A-D letter / first number, as vsibench_score_answers.py), or one predictions .jsonl file (e.g. eval_checkpoint-743_full/predictions_final_firstline.jsonl).
  --mixrow TASK,TASK  also print, per model, a third row on the same questions: the code's answer for these tasks, the
           model's for the others.
  --with-never  also add every appearance order and route planning question (the code never answers them: scored 0 for
           the code; the model's answer for the model and the mix rows); prints App., Route, "mean" (6 tasks) and "mean8".
  --split  instead: per list three rows that partition the benchmark: code on the questions it answers, the model on all
           the others, and the mix (code, else model) on all questions; 8 tasks (n per task), mean over the tasks present.
           --split-tasks TASK,TASK: the code takes only these tasks (prefix match); everything else goes to the model.
  --mix TASK,TASK  instead: on ALL questions, the code's answer for these tasks where it has one, the model's otherwise
           (prefix match, e.g. object_counting,object_rel_distance,object_rel_direction); prints model alone vs mix:
           full, debiased (from the code rows' "debiased"), 8 tasks, and how many answers came from the code.
Scoring: sr_vggt.eval.score (the VSI-Bench scorer of every reported number), per task and per rel. direction subtype
(easy / medium / hard); "mean" = mean over the tasks present. Needs the training env ($SR_PYTHON_BIN via remote.py run).
"""
import argparse, collections, glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, HERE)
from sr_vggt.eval import score  # noqa: E402
from qwen_eval_scores import load as _load_dir  # noqa: E402


def load(d, base=None):
    """eval dir (predictions_shard*.jsonl), a single predictions .jsonl, or claude:<dir> (needs base rows) -> {index: row}."""
    if d.startswith("claude:"):
        A = {}
        for f in glob.glob(os.path.join(d[7:], "out", "*.json")):
            if not f.endswith(".meta.json"):
                A.update({int(k): v for k, v in json.load(open(f))["answers"].items()})
        out = {}
        for i, r in base.items():
            a = str(A.get(i, ""))
            m = re.findall(r"[A-D]", a.upper()) if r.get("options") else re.findall(r"-?\d+\.?\d*", a.replace(",", ""))
            out[i] = dict(r, prediction=m[0] if m else "none")
        return out
    if not d.endswith(".jsonl"):
        return _load_dir(d)
    out = {}
    for line in open(d):
        x = json.loads(line)
        m = re.findall(r"-?\d+(?:\.\d+)?", str(x["prediction"]).replace(",", ""))
        if m and x["question_type"] in ("object_abs_distance", "object_counting", "object_size_estimation", "room_size_estimation"):
            x["prediction"] = m[-1]
        out[x["index"]] = x
    return out

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
    for t, key, col in NEVER:
        r = [x for x in rows if x["question_type"] == t]
        if r: out[col] = (100 * score(r)["vsibench/" + key], len(r))
    six = [out[c][0] for _, _, c in TASKS if c in out]; two = [out[c][0] for _, _, c in NEVER if c in out]
    if two: out["mean8"] = ((sum(six) + sum(two)) / (len(six) + len(two)), len(rows))
    return out


NEVER = [("obj_appearance_order", "obj_appearance_order_accuracy", "App."), ("route_planning", "route_planning_accuracy", "Route")]


T8 = [("obj_appearance_order_accuracy", "App."), ("object_abs_distance_mra", "Abs."), ("object_counting_mra", "Count"),
      ("object_rel_direction_accuracy", "RelDir"), ("object_rel_distance_accuracy", "RelDist"),
      ("object_size_estimation_mra", "Size"), ("room_size_estimation_mra", "Room"), ("route_planning_accuracy", "Route")]


def mix(a):
    tasks = tuple(a.mix.split(","))
    print(f"mix: code for {', '.join(tasks)} where it answers, model otherwise")
    print(f"{'list':10} {'answered by':18} {'n':>5} {'full':>6} {'deb.':>6} " + " ".join(f"{c:>7}" for _, c in T8) + "  from code")
    for spec in a.code:
        L, f = spec.split("=", 1)
        code = {r["index"]: r for r in json.load(open(f))}
        deb = {i for i, r in code.items() if r.get("debiased")}
        for m in a.model:
            name, d = m.split("=", 1)
            nm, _, lst = name.rpartition("@") if "@" in name else ("model", "", name)
            if lst != L: continue
            P = load(d, code); M, used = [], 0
            for i, r in P.items():
                c = code.get(i)
                if c and r["question_type"].startswith(tasks) and str(c["prediction"]).strip().lower() != "none":
                    M.append(dict(r, prediction=c["prediction"])); used += 1
                else:
                    M.append(r)
            for lab, rows in ((nm, list(P.values())), ("code, else " + nm, M)):
                s, sd = score(rows), score([r for r in rows if r["index"] in deb])
                print(f"{L:10} {lab:18} {len(rows):5} {100 * s['vsibench/overall']:6.1f} {100 * sd['vsibench/overall']:6.1f} "
                      + " ".join(f"{100 * s.get('vsibench/' + k, float('nan')):7.1f}" for k, _ in T8)
                      + (f"  {used}" if rows is M else ""))


def split(a):
    print(f"{'list':10} {'rows':10} {'n':>5} {'mean':>6} " + " ".join(f"{c:>12}" for _, c in T8) + "   (mean over tasks present)")
    for spec in a.code:
        L, f = spec.split("=", 1)
        code = {r["index"]: r for r in json.load(open(f))}
        ok = {i for i, r in code.items() if str(r["prediction"]).strip().lower() != "none"
              and (not a.split_tasks or r["question_type"].startswith(tuple(a.split_tasks.split(","))))}
        for m in a.model:
            name, d = m.split("=", 1)
            nm, _, lst = name.rpartition("@") if "@" in name else ("model", "", name)
            if lst != L: continue
            P = load(d, code)
            parts = [("code", [code[i] for i in P if i in ok]), (nm + " (rest)", [P[i] for i in P if i not in ok]),
                     ("mix", [dict(P[i], prediction=code[i]["prediction"]) if i in ok else P[i] for i in P])]
            for lab, rows in parts:
                s = score(rows); n = collections.Counter(r["question_type"].split("_easy")[0].split("_medium")[0].split("_hard")[0] for r in rows)
                cells, vals = [], []
                for k, c in T8:
                    t = k.rsplit("_", 1)[0]
                    if n.get(t):
                        v = 100 * s["vsibench/" + k]; vals.append(v); cells.append(f"{v:5.1f} n{n[t]:<5}")
                    else:
                        cells.append(f"{'--':>12}")
                print(f"{L:10} {lab:10} {len(rows):5} {sum(vals) / max(1, len(vals)):6.1f} " + " ".join(cells))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--code", action="append", required=True)
    ap.add_argument("--model", action="append", default=[]); ap.add_argument("--mix"); ap.add_argument("--mixrow"); ap.add_argument("--with-never", action="store_true"); ap.add_argument("--split", action="store_true"); ap.add_argument("--split-tasks"); a = ap.parse_args()
    if a.mix:
        return mix(a)
    if a.split:
        return split(a)
    cols = [c for _, _, c in TASKS] + ["dir easy", "dir medium", "dir hard", "mean"] + (["App.", "Route", "mean8"] if a.with_never else [])
    print(f"{'list':10} {'answered by':24} {'n':>5} " + " ".join(f"{c:>10}" for c in cols))
    for spec in a.code:
        L, f = spec.split("=", 1)
        allc = json.load(open(f))
        code = [r for r in allc if str(r["prediction"]).strip().lower() != "none"]
        if a.with_never:   # never answered by the code: included, scored 0 for the code (prediction "none")
            code += [r for r in allc if r["question_type"] in ("obj_appearance_order", "route_planning")]
        ids = {r["index"] for r in code}
        runs = [("code", code)]
        for m in a.model:
            name, d = m.split("=", 1)
            nm, _, lst = name.rpartition("@") if "@" in name else ("model", "", name)
            if lst != L: continue
            P = load(d); runs.append((nm, [P[i] for i in ids if i in P]))
            if a.mixrow:
                C = {r["index"]: r for r in code}; mt = tuple(a.mixrow.split(","))
                runs.append(("code+" + nm, [dict(P[i], prediction=C[i]["prediction"]) if P[i]["question_type"].startswith(mt) else P[i]
                                            for i in ids if i in P]))
        for nm, rows in runs:
            t = table(rows)
            print(f"{L:10} {nm:24} {len(rows):5} " + " ".join(f"{t[c][0]:6.1f} n{t[c][1]:<3}" if c in t else f"{'--':>10}" for c in cols))
        n_t = collections.Counter(r["question_type"] for r in code if str(r["prediction"]).strip().lower() != "none")
        print(f"{'':10} answered per type: " + ", ".join(f"{k.replace('object_', '')} {v}" for k, v in sorted(n_t.items())))


if __name__ == "__main__":
    main()
