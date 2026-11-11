"""Zero-shot of an UNMODIFIED Qwen VLM on the ScanNet++ rows of data/simplesr_train_traces.jsonl.gz.

Same frames, token budget and decoding as eval_base.py (1 fps, <=64 frames, 64-128 tokens/frame,
greedy, 16 new tokens, no system prompt, thinking off). The prompt is the trace's human turn with
its "<image>\n" tag removed; it already carries the lmms-eval post-prompt. The target is
meta.answer, the original short 190k label (not the trace's own final answer).
Scoring: option letter -> exact match (lmms fuzzy_matching); number -> MRA .5:.95:.05.

    python eval_traces.py --out_dir D --model M [--num_shards N --shard_index i]
    python eval_traces.py --out_dir D --merge
"""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

from sr_vggt.data import LLM_MAX_PIXELS, LLM_MIN_PIXELS, decode_video_frames
from sr_vggt.eval import exact_match, fuzzy_matching, mean_relative_accuracy, read_rows, to_float

MCA = {"relative_direction_object", "relative_distance_object", "relative_size_object",
       "appearance_order", "route_planning"}

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=False)
ap.add_argument("--traces", required=False)
ap.add_argument("--video_root", required=False)
ap.add_argument("--subset", default="scannetppv2")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--num_shards", type=int, default=1)
ap.add_argument("--shard_index", type=int, default=0)
ap.add_argument("--max_new_tokens", type=int, default=16)
ap.add_argument("--merge", action="store_true")
args = ap.parse_args()
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)


def metric(r):
    pred, gt = fuzzy_matching(r["prediction"]), r["ground_truth"]
    if r["question_type"] in MCA:
        return exact_match(pred, gt)
    p, t = to_float(pred), to_float(gt)
    if p is None or not t:  # unparsable prediction, or a zero target (MRA divides by it)
        return 0.0
    return float(mean_relative_accuracy(p, t, .5, .95, .05))


def score(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["question_type"]].append(metric(r))
    s = {k: {"n": len(v), "score": sum(v) / len(v)} for k, v in sorted(by.items())}
    s["overall_mean_of_types"] = sum(x["score"] for x in s.values()) / len(by)
    s["overall_mean_of_rows"] = sum(sum(v) for v in by.values()) / len(rows)
    return s


if args.merge:
    rows = {}
    for f in sorted(out.glob("predictions_shard*.jsonl")):
        rows.update(read_rows(f))
    s = score(list(rows.values()))
    (out / "results_merged.json").write_text(json.dumps({"num_samples": len(rows), "scores": s}, indent=2))
    print(len(rows), "rows"); print(json.dumps(s, indent=2))
    raise SystemExit

rows = []
with gzip.open(args.traces, "rt") as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        if r["video"].startswith(args.subset + "/"):
            rows.append({"index": i, "video": r["video"], "question_type": r["question_type"],
                         "prompt": r["conversations"][0]["value"].replace("<image>\n", ""),
                         "ground_truth": str(r["meta"]["answer"])})
rows.sort(key=lambda r: (r["video"], r["index"]))  # one decode per video within a shard
n = len(rows)
mine = rows[n * args.shard_index // args.num_shards: n * (args.shard_index + 1) // args.num_shards]
preds_path = out / f"predictions_shard{args.shard_index}of{args.num_shards}.jsonl"
done = read_rows(preds_path)
print(f"{n} rows, shard {len(mine)}, done {len(done)}", flush=True)

processor = AutoProcessor.from_pretrained(args.model)
model = AutoModelForImageTextToText.from_pretrained(
    args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()

memo = {}
with preds_path.open("a") as fout:
    for k, ex in enumerate(mine):
        if ex["index"] in done:
            continue
        if memo.get("key") != ex["video"]:
            memo = {"key": ex["video"],
                    "frames": decode_video_frames(str(Path(args.video_root) / ex["video"]), 1.0, 64)}
        msgs = [{"role": "user", "content": [
            {"type": "video", "video": memo["frames"], "min_pixels": LLM_MIN_PIXELS, "max_pixels": LLM_MAX_PIXELS},
            {"type": "text", "text": ex["prompt"]}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        _, vision, vkw = process_vision_info(msgs, return_video_kwargs=True, image_patch_size=16,
                                             return_video_metadata=True)
        videos, vmeta = map(list, zip(*vision))
        inputs = processor(text=text, videos=videos, video_metadata=vmeta, return_tensors="pt", **vkw).to("cuda")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 temperature=None, top_p=None, top_k=None)
        pred = processor.tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        row = {"index": ex["index"], "video": ex["video"], "question_type": ex["question_type"],
               "ground_truth": ex["ground_truth"], "prediction": pred}
        fout.write(json.dumps(row) + "\n"); fout.flush()
        done[ex["index"]] = row
        if k < 3 or k % 500 == 0:
            print(k, row, flush=True)

res = [done[ex["index"]] for ex in mine if ex["index"] in done]
(out / f"results_shard{args.shard_index}of{args.num_shards}.json").write_text(
    json.dumps({"num_samples": len(res), "scores": score(res)}, indent=2))
print(json.dumps(score(res), indent=2))
