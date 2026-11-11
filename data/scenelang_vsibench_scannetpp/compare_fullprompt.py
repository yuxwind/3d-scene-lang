"""Full prompt vs basic prompt (final answers) on the debiased rows the full prompt ran on."""
import json, re, statistics
from collections import Counter
from pathlib import Path
from sr_vggt.eval import MCA_QUESTION_TYPES, fuzzy_matching, mean_relative_accuracy, read_rows, score, to_float
D = Path(__file__).resolve().parent
rows = {r["index"]: r for r in map(json.loads, open(D / "vsibench_scannetpp_scenelang.jsonl"))}
LETTER = re.compile(r"\b([A-D])\b(?![a-z'])"); NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
def fa(r):
    p = r["prediction"]; m = re.findall(r"Answer:\s*\**\s*([^\n]+)", p); has = bool(m)
    if m: p = m[-1]
    m = (LETTER if r["question_type"] in MCA_QUESTION_TYPES else NUMBER).findall(p.replace(",", ""))
    return {**r, "prediction": m[-1] if m else "", "answered": bool(m), "answer_line": has}
def load(*ds):
    out = {}
    for d in ds:
        for f in sorted((D / d).glob("predictions*.jsonl")): out.update(read_rows(f))
    return out
def credit(r):
    if r["question_type"] in MCA_QUESTION_TYPES: return float(r["prediction"].lower() == r["ground_truth"].lower())
    p, g = to_float(r["prediction"]), to_float(r["ground_truth"])
    return 0.0 if p is None or not g else float(mean_relative_accuracy(p, g, .5, .95, .05))
basic = {i: fa(r) for i, r in load("qwen35_4b_scene_long", "qwen35_4b_scene_long_coatrack", "qwen35_4b_scene_4096", "qwen35_4b_scene_16384").items()}
rawF = load("qwen35_4b_scene_fullprompt", "qwen35_4b_scene_fullprompt_rest"); full = {i: fa(r) for i, r in rawF.items()}
ids = sorted(full)
toks = [r["gen_tokens"] for r in rawF.values()]
print(f"full prompt: {len(ids)} rows, answered {sum(r['answered'] for r in full.values())}, with an 'Answer:' line {sum(r['answer_line'] for r in full.values())}, "
      f"hit 16,384 cap {sum(t >= 16384 for t in toks)}, tokens median {statistics.median(toks):.0f} p90 {sorted(toks)[int(.9*len(toks))]} max {max(toks)}")
for name, P in (("basic (final)", basic), ("full prompt", full)):
    s = score([P[i] for i in ids])
    print(f"  {name:14s}", " ".join(f"{k.split('/')[-1].replace('_accuracy','').replace('_mra','').replace('object_','')}={100*v:.1f}"
                                   for k, v in s.items()))
by = {}
for i in ids:
    t = rows[i]["question_type"]; by.setdefault(t, [0, 0, 0, 0])
    b, f = credit(basic[i]), credit(full[i]); by[t][0] += 1; by[t][1] += b; by[t][2] += f; by[t][3] += (f > b) - (f < b)
for t, (n, b, f, d) in sorted(by.items()):
    print(f"   {t:30s} n={n:3d} basic {100*b/n:5.1f}  full {100*f/n:5.1f}  (net better rows {d:+d})")
# room size: predictions vs truth
print("\nroom size (truth / basic / full):")
for i in ids:
    if rows[i]["question_type"] == "room_size_estimation":
        print(f"  #{i} {rows[i]['ground_truth']:>6s} {basic[i]['prediction'] or '-':>6s} {full[i]['prediction'] or '-':>6s}", end="")
print()
