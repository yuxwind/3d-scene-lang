"""Debiased scene-text scores after the reruns: coat-rack fix (14 rows) and the 85 unanswered rows at 4,096 tokens,
thinking off / on. Answers are scored as in eval_scenelang.final_answer (text after </think>, last letter/number)."""
import json, os, re, statistics
from collections import Counter
from pathlib import Path
from sr_vggt.eval import MCA_QUESTION_TYPES, read_rows, score

D = Path(__file__).resolve().parent
S = Path(os.environ["SCRATCH"]) / "3d_scene_lang"
rows = {r["index"]: r for r in map(json.loads, open(D / "vsibench_scannetpp_scenelang.jsonl"))}
deb = [i for i, r in rows.items() if r["debiased"]]
LETTER = re.compile(r"\b([A-D])\b(?![a-z'])"); NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def fa(r):
    p = r["prediction"]
    if "</think>" in p: p = p.split("</think>")[-1]
    elif r.get("thinking"): p = ""
    m = (LETTER if r["question_type"] in MCA_QUESTION_TYPES else NUMBER).findall(p.replace(",", ""))
    return {**r, "prediction": m[-1] if m else "", "answered": bool(m)}


def load(name):
    out = {}
    for f in sorted((D / name).glob("predictions*.jsonl")): out.update(read_rows(f))
    return out


def fmt(s):
    return " ".join(f"{k.split('/')[-1].replace('_accuracy','').replace('_mra','').replace('object_','').replace('obj_','')}={100*v:.1f}" for k, v in s.items())


def ok(r):
    return score([r])["vsibench/overall"] if False else None


base = {i: fa(r) for i, r in load("qwen35_4b_scene_long").items()}
vs = {i: fa(r) for i, r in load("qwen35_4b_video_scene").items()}
cr_s = {i: fa(r) for i, r in load("qwen35_4b_scene_long_coatrack").items()}
cr_v = {i: fa(r) for i, r in load("qwen35_4b_video_scene_coatrack").items()}
A = {i: fa(r) for i, r in load("qwen35_4b_scene_4096").items()}
B = {i: fa(r) for i, r in load("qwen35_4b_scene_4096_thinking").items()}
raw = {"A": load("qwen35_4b_scene_4096"), "B": load("qwen35_4b_scene_4096_thinking")}


def correct(r):  # per-row credit, same rules as the benchmark
    from sr_vggt.eval import fuzzy_matching, mean_relative_accuracy, to_float
    if r["question_type"] in MCA_QUESTION_TYPES:
        return float(r["prediction"].lower() == r["ground_truth"].lower())
    p, g = to_float(r["prediction"]), to_float(r["ground_truth"])
    return 0.0 if p is None or not g else float(mean_relative_accuracy(p, g, .5, .95, .05))


print("== coat-rack rows (14): old text -> fixed text")
for name, old, new in (("scene, 1,024 tok", base, cr_s), ("video+scene, 16 tok", vs, cr_v)):
    o = sum(correct(old[i]) for i in new); n = sum(correct(new[i]) for i in new)
    print(f"  {name:20s} credit {o:.2f} -> {n:.2f} of {len(new)}")
    for i in new:
        if old[i]["prediction"] != new[i]["prediction"]:
            print(f"     #{i} {rows[i]['question_type']:28s} gt={rows[i]['ground_truth']:6s} {old[i]['prediction'] or '-':6s} -> {new[i]['prediction'] or '-'}")

print("\n== the 85 unanswered rows")
for k, P in (("A  4,096 tok, thinking off", A), ("B  4,096 tok, thinking on ", B)):
    R = raw[k[0]]
    toks = [r["gen_tokens"] for r in R.values()]
    capped = sum(t >= 4096 for t in toks)
    ans = sum(r["answered"] for r in P.values())
    cred = sum(correct(r) for r in P.values())
    print(f"  {k}: answered {ans}/85, credit {cred:.1f}/85, hit 4,096 cap {capped}, tokens median {statistics.median(toks):.0f} max {max(toks)}")
    by = Counter(); byc = Counter()
    for r in P.values():
        by[r["question_type"]] += 1; byc[r["question_type"]] += correct(r)
    print("     per type (credit/n): " + ", ".join(f"{t.replace('object_','')} {byc[t]:.0f}/{by[t]}" for t in sorted(by)))

print("\n== debiased 752, scene text only")
def merged(*over):
    m = dict(base)
    for o in over: m.update(o)
    return [m[i] for i in deb]
for name, sel in (("1,024 tok, old text (reported before)", merged()),
                  ("+ coat-rack fix", merged(cr_s)),
                  ("+ coat-rack fix + 85 rerun A (4,096, thinking off)", merged(cr_s, A)),
                  ("+ coat-rack fix + 85 rerun B (4,096, thinking on)", merged(cr_s, B))):
    print(f"  {name:52s} {fmt(score(sel))}")
print("\n== debiased 752, video + scene")
print(f"  {'16 tok, old text':52s} {fmt(score([vs[i] for i in deb]))}")
print(f"  {'+ coat-rack fix':52s} {fmt(score([{**vs, **cr_v}[i] for i in deb]))}")

# ---- 16,384-token rerun of the 137 rows still cut off (113 at 1,024 + 24 at 4,096) ----
C = {i: fa(r) for i, r in load("qwen35_4b_scene_16384").items()}
rawC = load("qwen35_4b_scene_16384")
toks = [r["gen_tokens"] for r in rawC.values()]
print(f"\n== the 137 cut-off rows at 16,384 tokens: answered {sum(r['answered'] for r in C.values())}/{len(C)}, "
      f"credit {sum(correct(r) for r in C.values()):.1f}, hit the cap {sum(t >= 16384 for t in toks)}, "
      f"tokens median {statistics.median(toks):.0f} max {max(toks)}")
before = {**base, **cr_s, **A}
ch = Counter()
for i, r in C.items():
    o, n = correct(before[i]), correct(r)
    ch["better" if n > o else "worse" if n < o else "same"] += 1
print("   vs the answer used before:", dict(ch))
by = Counter(); byc = Counter()
for r in C.values():
    by[r["question_type"]] += 1; byc[r["question_type"]] += correct(r)
print("   per type (credit/n): " + ", ".join(f"{t.replace('object_','')} {byc[t]:.0f}/{by[t]}" for t in sorted(by)))
final = merged(cr_s, A, C)
print(f"\n  {'FINAL: coat-rack fix + 85 at 4,096 + 137 at 16,384':52s} {fmt(score(final))}")
print(f"  rows with no answer left: {sum(not r['answered'] for r in final)}")
json.dump({str(r["index"]): r["prediction"] for r in final}, open(D / "scene_final_predictions.json", "w"))
