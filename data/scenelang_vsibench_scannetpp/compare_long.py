"""Debiased ScanNet++: video only (baseline) vs scene / video+scene at 16 and 1024 new tokens, last-answer scoring."""
import importlib.util, json, os, re, statistics
from collections import Counter
from pathlib import Path
from sr_vggt.eval import read_rows, score
S = Path(os.environ["SCRATCH"]) / "3d_scene_lang"; D = S / "scenelang_vsibench_scannetpp"
rows = [json.loads(l) for l in open(D / "vsibench_scannetpp_scenelang.jsonl")]
deb = [r["index"] for r in rows if r["debiased"]]
MCA = {"obj_appearance_order", "object_rel_distance", "route_planning", "object_rel_direction_easy",
       "object_rel_direction_medium", "object_rel_direction_hard"}
LETTER = re.compile(r"\b([A-D])\b(?![a-z'])"); NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
def fa(r):
    m = (LETTER if r["question_type"] in MCA else NUMBER).findall(r["prediction"].replace(",", ""))
    return {**r, "prediction": m[-1] if m else r["prediction"]}
def load(d):
    out = {}
    for f in sorted(d.glob("predictions*.jsonl")): out.update(read_rows(f))
    return out
base = {}
for f in sorted((S / "baseline_qwen35_4b").glob("predictions_shard*.jsonl")): base.update(read_rows(f))
runs = {"video": base, "scene@16": load(D / "qwen35_4b_scene"), "video_scene@16": load(D / "qwen35_4b_video_scene"),
        "scene@1024": load(D / "qwen35_4b_scene_long"), "video_scene@1024": load(D / "qwen35_4b_video_scene_long")}
res = {}
for m, p in runs.items():
    sel = [fa(p[i]) for i in deb if i in p]; s = score(sel); res[m] = {"n": len(sel), **s}
    print(f"{m:17s} n={len(sel)} " + " ".join(f"{k.split('/')[-1].replace('_accuracy','').replace('_mra','')}={100*v:.1f}" for k, v in s.items()))
(D / "compare_debiased.json").write_text(json.dumps(res, indent=2))
for m in ("scene@1024", "video_scene@1024"):
    p = runs[m]; sel = [p[i] for i in deb]
    toks = [len(r["prediction"].split()) for r in sel]
    nomatch = sum(not (LETTER if r["question_type"] in MCA else NUMBER).findall(r["prediction"]) for r in sel)
    long = sum(len(r["prediction"]) > 40 for r in sel)
    print(m, "words median/max", statistics.median(toks), max(toks), "| reasoned (>40 chars)", long, "| no answer found", nomatch)
