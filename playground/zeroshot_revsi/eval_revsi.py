"""Zero-shot ReVSI (64_frame subset, 6,616 questions) of an UNMODIFIED Qwen VLM, in our VSI-Bench setting.

Inference as eval_base.py on VSI-Bench: frames from the subset's 64-frame video by our rule (1 fps, <= 64 frames:
decode_video_frames), or with --all_frames every frame of it (ReVSI's protocol: its answers assume exactly those
frames), 64-128 tokens/frame, greedy, 16 new tokens, no system prompt, thinking off.
Prompts and scoring follow the official lmms-eval task (lmms_eval/tasks/revsi): pre-prompt "These are frames of
a video.", MCQ post-prompt as VSI-Bench, number post-prompt "Answer the question using a single integer or decimal
number."; first word of the output, exact match / MRA .5:.95:.05; subtypes averaged into 7 tasks; overall = mean.

    python eval_revsi.py --out_dir D --model M --revsi_dir R [--num_shards N --shard_index i]
    python eval_revsi.py --out_dir D --revsi_dir R --merge
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

MCQ = ["object_rel_direction_forward_easy", "object_rel_direction_backward_easy", "object_rel_direction_forward_hard",
       "object_rel_direction_backward_hard", "object_rel_distance_closest", "object_rel_distance_farthest", "route_planning"]
NQ = ["object_counting_single", "object_counting_multiple", "object_abs_distance", "object_size_estimation",
      "room_size_estimation_single", "room_size_estimation_multiple"]
COMPOSITE = {"object_rel_direction": MCQ[:4], "object_rel_distance": MCQ[4:6],
             "object_counting": NQ[:2], "room_size_estimation": NQ[4:6]}
PRE, MCQ_POST, NQ_POST = ("These are frames of a video.", "Answer with the option's letter from the given choices directly.",
                          "Answer the question using a single integer or decimal number.")

ap = argparse.ArgumentParser()
ap.add_argument("--model")
ap.add_argument("--revsi_dir", required=True)
ap.add_argument("--subset", default="64_frame")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--num_shards", type=int, default=1)
ap.add_argument("--shard_index", type=int, default=0)
ap.add_argument("--max_new_tokens", type=int, default=16)
ap.add_argument("--merge", action="store_true")
ap.add_argument("--all_frames", action="store_true",
                help="ReVSI's exact protocol: feed every frame of the subset video (64 for 64_frame) instead of our 1 fps rule")
args = ap.parse_args()
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(Path(args.revsi_dir) / args.subset / "test-00000-of-00001.parquet")


def prompt(r):
    if r["question_type"] in NQ:
        return "\n".join([PRE, r["question"], NQ_POST]).strip()
    return "\n".join([PRE, r["question"], "Options:\n" + "\n".join(r["options"]), MCQ_POST]).strip()


def acc(r):  # revsi_process_results
    pred = str(r["prediction"]).strip().split(" ")[0].rstrip(".").strip()
    if r["question_type"] in MCQ:
        return float(pred.lower() == r["ground_truth"].lower())
    try:
        p, t = float(pred), float(r["ground_truth"])
        return float(((abs(p - t) / t) <= (1 - np.linspace(.5, .95, 10))).mean())
    except Exception:
        return 0.0


def score(rows):  # _compute_all_subscores
    by = defaultdict(list)
    for r in rows:
        by[r["question_type"]].append(acc(r))
    o = {f"{t}_acc": float(np.mean(v)) for t, v in by.items()}
    for name, ts in COMPOSITE.items():
        ks = [f"{t}_acc" for t in ts if f"{t}_acc" in o]
        if ks:
            o[f"{name}_acc"] = float(np.mean([o.pop(k) for k in ks]))
    o["overall_acc"] = sum(o.values()) / len(o) if o else 0.0
    return o


def read(path):
    d = {}
    if path.exists():
        for line in path.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            d[r["id"]] = r
    return d


def report(rows, name):
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    res = {"num_samples": len(rows), "scores": score(rows),
           "by_dataset": {k: {"n": len(v), "overall_acc": score(v)["overall_acc"]} for k, v in sorted(by_ds.items())}}
    (out / name).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if args.merge:
    done = {}
    for f in sorted(out.glob("predictions_shard*.jsonl")):
        done.update(read(f))
    report(list(done.values()), "results_merged.json")
    raise SystemExit

import torch  # noqa: E402  (not needed for --merge)
from qwen_vl_utils import process_vision_info  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

from sr_vggt.data import LLM_MAX_PIXELS, LLM_MIN_PIXELS, decode_video_frames  # noqa: E402

rows = df.sort_values(["scene_id", "id"]).to_dict("records")  # one decode per video within a shard
n = len(rows)
mine = rows[n * args.shard_index // args.num_shards: n * (args.shard_index + 1) // args.num_shards]
preds_path = out / f"predictions_shard{args.shard_index}of{args.num_shards}.jsonl"
done = read(preds_path)
print(f"{n} rows, shard {len(mine)}, done {len(done)}\n--- example prompt ---\n{prompt(mine[0])}\n---", flush=True)

processor = AutoProcessor.from_pretrained(args.model)
model = AutoModelForImageTextToText.from_pretrained(
    args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
memo = {}
with preds_path.open("a") as fout:
    for k, r in enumerate(mine):
        if int(r["id"]) in done:
            continue
        if memo.get("key") != r["scene_id"]:
            memo = {"key": r["scene_id"], "frames": decode_video_frames(
                str(Path(args.revsi_dir) / "video" / args.subset / f"{r['scene_id']}.mp4"),
                1e9 if args.all_frames else 1.0, 64)}
        msgs = [{"role": "user", "content": [
            {"type": "video", "video": memo["frames"], "min_pixels": LLM_MIN_PIXELS, "max_pixels": LLM_MAX_PIXELS},
            {"type": "text", "text": prompt(r)}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        _, vision, vkw = process_vision_info(msgs, return_video_kwargs=True, image_patch_size=16,
                                             return_video_metadata=True)
        videos, vmeta = map(list, zip(*vision))
        inputs = processor(text=text, videos=videos, video_metadata=vmeta, return_tensors="pt", **vkw).to("cuda")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 temperature=None, top_p=None, top_k=None)
        row = {"id": int(r["id"]), "dataset": r["dataset"], "scene_id": r["scene_id"], "question_type": r["question_type"],
               "ground_truth": str(r["ground_truth"]), "n_frames": len(memo["frames"]),
               "prediction": processor.tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()}
        fout.write(json.dumps(row) + "\n"); fout.flush()
        done[row["id"]] = row
        if k < 3 or k % 500 == 0:
            print(k, row, flush=True)
report([done[int(r["id"])] for r in mine if int(r["id"]) in done], f"results_shard{args.shard_index}of{args.num_shards}.json")
