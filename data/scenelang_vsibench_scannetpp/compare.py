"""Table-2-style comparison: video only (baseline_qwen35_4b) vs scene language vs video + scene language."""
import json, os, sys
from pathlib import Path
from sr_vggt.eval import read_rows, score
S = Path(os.environ["SCRATCH"]) / "3d_scene_lang"; D = S / "scenelang_vsibench_scannetpp"
rows = [json.loads(l) for l in open(D / "vsibench_scannetpp_scenelang.jsonl")]
base = {}
for f in sorted((S / "baseline_qwen35_4b").glob("predictions_shard*.jsonl")):
    base.update(read_rows(f))
res = {"video": base, **{m: read_rows(D / f"qwen35_4b_{m}" / "predictions.jsonl") for m in ("scene", "video_scene")}}
out = {}
for split, ids in (("debiased", [r["index"] for r in rows if r["debiased"]]), ("full", [r["index"] for r in rows])):
    print("==", split)
    for m, p in res.items():
        sel = [p[i] for i in ids if i in p]
        s = score(sel); out[f"{split}/{m}"] = {"n": len(sel), **s}
        print(f"{m:12s} n={len(sel)} " + " ".join(f"{k.split('/')[-1]}={100 * v:.1f}" for k, v in s.items()))
(D / "compare.json").write_text(json.dumps(out, indent=2))
sc = list(res["scene"].values())
print("scene prompt tokens max", max(r["prompt_tokens"] for r in sc), "empty", sum(not r["prediction"] for r in sc))
print([r["prediction"] for r in sc[:10]])
