# Reproduced experiments

Runs made on Jean Zay (H100 80GB, bf16) while going through the README. Scores use the repo's
VSI-Bench protocol (`sr_vggt/eval.py`): full split, 5,130 questions, 288 videos, lmms-eval prompts
and scoring.

## Zero-shot baselines (no fine-tuning)

Unmodified base VLMs, with no VGGT, no adapter and no system prompt. Everything else matches the repo eval:

- **Frames:** `sr_vggt.data.decode_video_frames`, 1 fps, capped at 64, spread uniformly over the whole video.
- **Token budget:** 64–128 tokens per frame (`LLM_MIN/MAX_PIXELS`), through `qwen_vl_utils.process_vision_info`.
- **Prompts:** the lmms-eval question prompts.
- **Decoding:** greedy, `max_new_tokens 16`, batch 1, sdpa attention.

Script: `$SCRATCH/3d_scene_lang/baseline_qwen3vl2b/eval_base.py`. It sits outside the repo and imports
`sr_vggt`. Forward hooks on `model.model.visual` (ViT) and `model.model.language_model` (LLM) record
CUDA time and peak activation memory per call. The first LLM call of each question is the prefill;
the remaining calls are decode steps.

```bash
sbatch $SCRATCH/3d_scene_lang/baseline_qwen3vl2b/job.slurm        # 2 shards x 1 H100, qos_gpu_h100-dev
python eval_base.py --out_dir <dir> --merge && python summarize.py <dir>   # score + stats -> summary.json
```

### Qwen3-VL-2B-Instruct (job 1263053, 2 × 13–14 min)

| Task | Score |
|---|---|
| **Overall** | **0.5088** |
| Object size (MRA) | 0.7101 |
| Object counting (MRA) | 0.6326 |
| Appearance order (acc) | 0.6149 |
| Room size (MRA) | 0.5010 |
| Relative distance (acc) | 0.4761 |
| Absolute distance (MRA) | 0.4240 |
| Relative direction (acc) | 0.4232 |
| Route planning (acc) | 0.2887 |

For reference, the trained 2B runs in EXPERIMENTS.md score `main` 0.6000 and `text256` 0.6060. Those
numbers come from the original cached-feature runs, so the comparison is approximate. Our rerun of
`main` scores 0.5992 (see "Fine-tuned: Perceiver + VGGT-Omega" below).

#### Input and preprocessing
- **Videos:** all 640×480 (231) or 480×640 (57).
- **Frames per video:** 22–64, mean 58. 195 of the 288 videos hit the 64-frame cap.
- **Resize:** frames reach the ViT at **384×256** (portrait 256×384), normalized with mean/std 0.5.
- **Timestamps:** frame lists get placeholder metadata from `qwen_vl_utils` that assumes 2 fps, so
  the model's timestamps don't match the real sampling rate.

> **Resize quirk (qwen_vl_utils 0.0.14).** For a list of frames, `fetch_video` passes the merge factor
> (32) to `fetch_image` as `image_patch_size`. Each frame is therefore snapped to multiples of 64,
> not 32. 640×480 becomes 384×256 (96 tokens per 2-frame step) instead of 416×288 (117 tokens),
> and the 4:3 frames are stretched to 3:2. Training and eval in this repo use the same code path,
> so all experiments are affected the same way.

#### Tokens and dimensions (one 64-frame clip)
| Stage | Shape |
|---|---|
| ViT input | 12,288 patches × 1,536 (3 ch × 2 frames × 16 × 16); grid t,h,w = 32×16×24 |
| ViT blocks (24 layers, width 1024, 16 heads) | 12,288 × 1,024 |
| 2×2 merge + projection | **3,072 tokens × 2,048** (96 per 2-frame step) |
| DeepStack (ViT layers 5, 11, 17) | 3 × [3,072 × 2,048], added into the first LLM layers |
| LLM input | ~3,400 tokens: 3,072 video + ~330 text (125–507) |
| LLM (28 layers, width 2048, 16 query / 8 KV heads, head dim 128) | 1 × 3,401 × 2,048 |
| Answer | median 4 tokens (2–14) |

#### Memory and speed (H100, bf16, batch 1, warm-up rows excluded)
| | ViT | LLM |
|---|---|---|
| Parameters | 407 M | 1.72 B (embedding tied with the 311 M output layer) |
| Activation memory above weights | 0.48 GB | 0.54 GB (prefill) |
| Time per 64-frame clip | 52 ms | 38 ms prefill (~3,400 tokens); 16 ms per generated token |

- **Peak GPU memory:** 4.8 GB, of which 4.06 GB is weights.
- **GPU time per question:** about 0.15 s.
- **Video decoding (CPU, decord):** 1.1 s median per video (p90 2.2 s), done once per video and
  reused for all of its questions. Without that reuse it would dominate.

### Qwen3.5-4B (job 1263940, 2 × 20–22 min)

- **Model:** `Qwen/Qwen3.5-4B` at revision `851bf6e8`, loaded as `Qwen3_5ForConditionalGeneration`
  through `AutoModelForImageTextToText`.
- **Architecture:** the ViT has the same shape as Qwen3-VL's (24 layers, width 1024, patch 16, temporal
  patch 2), but no DeepStack. The 2560-wide LLM is a 32-layer hybrid: 24 Gated-DeltaNet
  linear-attention layers and 8 full-attention layers.
- **Thinking off:** the chat template is called with `enable_thinking=False`. Otherwise the model
  starts with a `<think>` block and uses up the 16-token answer limit.
- **Kernels:** `flash-linear-attention` 0.5.2 (plus `fla-core` and `einops`) is installed only for this
  job, in `$SCRATCH/3d_scene_lang/pyextra/site` (added to `PYTHONPATH`). The repo venv is unchanged.
  `causal-conv1d` is not installed, so its short convolution runs in plain PyTorch, which may make
  the speed numbers slightly pessimistic.

- **Offline loading:** the job loads the model from its snapshot folder. The model was downloaded at a
  pinned revision, so the cache has no `refs/main` and offline loading by repo name fails.

| Task | Qwen3.5-4B | Qwen3-VL-2B |
|---|---|---|
| **Overall** | **0.5510** | 0.5088 |
| Appearance order (acc) | 0.6877 | 0.6149 |
| Object size (MRA) | 0.6749 | 0.7101 |
| Relative direction (acc) | 0.6141 | 0.4232 |
| Relative distance (acc) | 0.5789 | 0.4761 |
| Room size (MRA) | 0.5726 | 0.5010 |
| Object counting (MRA) | 0.5227 | 0.6326 |
| Absolute distance (MRA) | 0.3916 | 0.4240 |
| Route planning (acc) | 0.3660 | 0.2887 |

All answers are short (4–12 tokens, none hit the 16-token limit, no `<think>` text).

#### Tokens and dimensions (one 64-frame clip)
Preprocessing is the same as for Qwen3-VL-2B: 384×256 frames, 12,288 ViT patches × 1,536, 3,072 video
tokens. The differences:

| Stage | Shape |
|---|---|
| ViT (24 layers, width 1024) | 12,288 × 1,024, then 2×2 merge → **3,072 × 2,560**; no DeepStack |
| LLM input | ~3,400 tokens: 3,072 video + ~340 text |
| LLM (32 layers, width 2560) | 24 Gated-DeltaNet layers (16 key / 32 value heads, head dim 128) + 8 gated full-attention layers (16 query / 4 KV heads, head dim 256); output 1 × 3,392 × 2,560 |
| Vocabulary | 248,320 (vs 151,936) |

#### Memory and speed (H100, bf16, batch 1, warm-up rows excluded)
| | ViT | LLM |
|---|---|---|
| Parameters | 334 M | 4.21 B (embedding tied with the 636 M output layer) |
| Activation memory above weights | 0.44 GB | 0.57 GB (prefill) |
| Time per 64-frame clip | 51 ms | 87 ms prefill (~3,400 tokens); 27 ms per generated token |

- **Peak GPU memory:** 9.4 GB, of which 8.66 GB is weights.
- **First question per shard:** 40 s, while Triton compiles the `fla` kernels.

## Subset scores (re-scored from the saved predictions, no new inference)

`load_dataset(<VSI-Bench>, "full")` reads `test*.parquet`, i.e. `test_debiased.parquet` followed by
`test_pruned.parquet` (2,362 + 2,768 = 5,130). The eval's row index therefore maps to that
concatenation, not to `test.jsonl`. Script: `$SCRATCH/3d_scene_lang/rescore.py`; output:
`baseline_subset_scores.json`.

| Subset | Questions | Qwen3-VL-2B | Qwen3.5-4B |
|---|---|---|---|
| Full | 5,130 | 0.5088 | 0.5510 |
| Full · ARKitScenes | 1,601 | 0.5215 | 0.5612 |
| Full · ScanNet | 2,071 | 0.5021 | 0.5422 |
| Full · ScanNet++ | 1,458 | 0.4499 | 0.5037 |
| Debiased | 2,362 | 0.4267 | 0.4590 |
| Debiased · ARKitScenes | 675 | 0.4255 | 0.4373 |
| Debiased · ScanNet | 935 | 0.4383 | 0.4604 |
| **Debiased · ScanNet++** | 752 | **0.3646** | **0.4288** |

ScanNet++ per task (zero-shot):

| Task | Full · 2B | Full · 4B | Debiased · 2B | Debiased · 4B |
|---|---|---|---|---|
| Questions | 1,458 | 1,458 | 752 | 752 |
| **Overall** | **0.4499** | **0.5037** | **0.3646** | **0.4288** |
| Appearance order (acc) | 0.6316 | 0.7149 | 0.5797 | 0.6884 |
| Object size (MRA) | 0.6721 | 0.6742 | 0.5406 | 0.5691 |
| Relative distance (acc) | 0.4754 | 0.5704 | 0.4272 | 0.5049 |
| Relative direction (acc) | 0.3879 | 0.5222 | 0.3126 | 0.3587 |
| Room size (MRA) | 0.3320 | 0.4680 | 0.2973 | 0.4622 |
| Object counting (MRA) | 0.6017 | 0.4472 | 0.4612 | 0.3336 |
| Absolute distance (MRA) | 0.3833 | 0.3252 | 0.2983 | 0.2407 |
| Route planning (acc) | 0.1154 | 0.3077 | 0.0000 | 0.2727 |

- **Where 4B loses:** counting and absolute distance, on both subsets.
- **Route planning, debiased:** only 22 questions. The 2B model gives a valid letter for all 22
  and gets all of them wrong.

## Zero-shot with the scene language alone (Debiased · ScanNet++, 752 questions)

The video is replaced by the ground-truth scene language: one line per object,
`{"label": ..., "bbox_3d": [x, y, z, dx, dy, dz, yaw, roll, pitch]}` (Table 2 setup). Script:
`data/scenelang_vsibench_scannetpp/eval_scenelang.py`, rows in `vsibench_scannetpp_scenelang.jsonl` next to
it. Qwen3.5-4B (revision `851bf6e8`) is unmodified: no system prompt, thinking off, greedy decoding.

Prompts (`--prompt`):

- **basic:** a two-sentence scene intro (box centre and size in meters, rotation in radians, z up), then the question and the lmms-eval post-prompt. Up to 1,024 new
  tokens (`qwen35_4b_scene_long`, `_long_coatrack`); the 137 answers cut off there were rerun with
  up to 4,096 or 16,384 tokens (`qwen35_4b_scene_4096`, `_16384`).
- **full:** the complete `bbox_3d` description (floor height, box axes, yaw/roll/pitch), a per-task
  "how to compute" line (e.g. closest-point box distance, a cross/dot rule for left/right and
  front/back), and a final line "Compute the answer step by step ..., then end with 'Answer: ...'".
  Up to 16,384 tokens (`qwen35_4b_scene_fullprompt`, `_fullprompt_rest`).
- **simple1:** a one-paragraph format description, no per-task line, the same final line
  (`qwen35_4b_scene_simpleprompt`).
- **simple2:** the full prompt without the per-task line (`qwen35_4b_scene_simple2prompt`; 744 of 752
  answered, the last 8 hit the 2 h limit).

For full, simple1 and simple2 the score keeps the last option letter or number after the last `</think>`.

Two references, also run on the same 752 questions:

- **Oracle (code):** `oracle.py` answers from the same text by computation (label count, longest box
  side, closest-point distance between oriented boxes, centre cross/dot products, and for room size the
  smallest rotated rectangle covering every box point in the xy plane, see below). Appearance order and
  route planning are not computable from boxes and score 0.
- **Claude (no code):** Claude Opus 5.5, as 55 + 8 sub-agents allowed only to read a question file
  and write an answer JSON (transcripts checked: no Bash or Python, no file with answers read). Input:
  the simple2 description, the scene, the question and options, then "Answer with the option letter"
  or "Answer with a number only". Answers and scores:
  `$SCRATCH/3d_scene_lang/claude_nocode_scenelang/`.

| Task | Qwen3.5-4B basic | full | simple1 | simple2 | Oracle (code) | Claude (no code) |
|---|---|---|---|---|---|---|
| Questions | 752 | 752 | 752 | 744 | 752 | 752 |
| **Overall** | **0.5668** | **0.6901** | **0.6839** | **0.7051** | **0.7161** | **0.8987** |
| Appearance order (acc) | 0.3551 | 0.3841 | 0.3551 | 0.4130 | – | 0.8188 |
| Object size (MRA) | 0.8894 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Relative distance (acc) | 0.6796 | 0.7961 | 0.6699 | 0.6289 | 1.0000 | 1.0000 |
| Relative direction (acc) | 0.5861 | 0.9524 | 0.8185 | 0.8438 | 1.0000 | 1.0000 |
| Room size (MRA) | 0.3135 | 0.4135 | 0.5514 | 0.6189 | 0.7919 | 0.7973 |
| Object counting (MRA) | 0.9560 | 0.9362 | 0.9586 | 0.9664 | 0.9888 | 0.9888 |
| Absolute distance (MRA) | 0.3907 | 0.8110 | 0.8000 | 0.8060 | 0.9483 | 0.9483 |
| Route planning (acc) | 0.3636 | 0.2273 | 0.3182 | 0.3636 | – | 0.6364 |

- **Text vs video:** with the scene language, basic already beats the video-only zero-shot (0.4288);
  asking the model to compute and end with "Answer:" adds about 12 points (basic → simple1/simple2).
- **Per-task lines:** worth +11 to +17 on relative distance and direction (full vs simple1/simple2),
  but the room-size line hurts (0.41 vs 0.55–0.62). Overall, full and simple2 are within 1.5 points.
- **The text holds the answer:** the oracle and Claude match on every geometry task (counting, size,
  both distances, direction); on room size both are close (0.79 vs 0.80). The 4B
  model's gap is the arithmetic, not missing information.
- **Caveat, Claude on appearance order and route planning:** neither is computable from the boxes.
  Scene-only heuristics reach 28–64 of 138 appearance questions (list order, option consensus,
  shortest path); Claude gets 113. It may use a camera-walk prior or may have seen VSI-Bench in
  training; this run cannot tell them apart. Without these two tasks Claude averages 0.956 over the
  other six.
- **Later finding, the 0.8188 is inflated by the batching:** each sub-agent's file held several questions
  from the same scene. All appearance-order questions of a scene follow one true viewing order, so an
  agent can pick the single order that fits an option in every question without using the boxes. On
  the other 4,378 VSI-Bench rows, with at most one question per scene per batch, appearance order drops
  to 0.452 (see "Claude (no code) on all of VSI-Bench" below). These 752 rows have not been rerun that
  way.

### Oracle room size: the smallest rotated rectangle

The scene text has no walls, so the oracle estimates the floor area from the objects: every box is
filled with an 11 × 11 × 11 grid of points (rotation applied), the points are projected to the xy
plane, and the area of a shape covering them is the answer. The ScanNet++ frame has z up, but its x/y
axes are not aligned with the walls: over the 50 scenes the dominant wall direction (from the yaw of
doors, whiteboards, heaters, shelves, ...) is a median 6.9° off the nearest axis, within 2° in only 9
scenes, and up to 43°.

- **Axis-aligned rectangle:** `(max x − min x) × (max y − min y)`. Too big when the room is rotated
  off the axes: up to (|cos θ| + |sin θ|)² × the true area (+37% at 15°).
- **Convex hull:** the tightest convex polygon around the points (monotone chain + shoelace area).
  Too small, as furniture rarely fills the room's corners.
- **Smallest rotated rectangle (used):** the least-area rectangle at any angle that covers all the
  points, so it lines up with the walls. One of its sides always lies along a convex-hull edge, so
  `min_rect_area()` rotates the hull into each hull edge's frame, takes the width × height of the
  axis-aligned box there, and keeps the smallest (rotating calipers).

| Room-size estimate (37 questions) | MRA | Median predicted / true area |
|---|---|---|
| Axis-aligned rectangle | 0.6459 | 1.12 |
| Convex hull (earlier oracle) | 0.6865 | 0.87 |
| **Smallest rotated rectangle** | **0.7919** | **1.01** |
| Claude (no code) | 0.7973 | – |

The rectangle still falls short in large or irregular rooms (60.0 vs 67.1 m², 39.2 vs 44.0 m²), where
the furniture does not reach every wall.

### Video + scene language (Qwen3.5-4B, zero-shot)

The same 752 questions with the video (1 fps, at most 64 frames) before the text; the text starts with
"These are frames of a video." and is otherwise the scene-only prompt. Thinking off, greedy decoding.

- **basic:** 16 new tokens (`qwen35_4b_video_scene` + `_coatrack`). A rerun with up to 1,024 tokens
  (`qwen35_4b_video_scene_long`) still answers in at most 17 characters and scores 0.5077.
- **simple2:** up to 16,384 new tokens (`qwen35_4b_video_scene_simple2prompt`, job 1449517, 10 shards,
  21–75 min each). Median answer 321 tokens; 19 answers hit the cap.

| Task | Video only | Video + scene, basic | Video + scene, simple2 | Scene only, basic | Scene only, simple2 |
|---|---|---|---|---|---|
| Questions | 752 | 752 | 752 | 752 | 744 |
| **Overall** | **0.4288** | **0.5100** | **0.5483** | **0.5668** | **0.7051** |
| Appearance order (acc) | 0.6884 | 0.6232 | 0.6522 | 0.3551 | 0.4130 |
| Object size (MRA) | 0.5691 | 0.7455 | 0.9528 | 0.8894 | 1.0000 |
| Relative distance (acc) | 0.5049 | 0.4660 | 0.5534 | 0.6796 | 0.6289 |
| Relative direction (acc) | 0.3587 | 0.3646 | 0.4203 | 0.5861 | 0.8438 |
| Room size (MRA) | 0.4622 | 0.4081 | 0.2703 | 0.3135 | 0.6189 |
| Object counting (MRA) | 0.3336 | 0.9129 | 0.9560 | 0.9560 | 0.9664 |
| Absolute distance (MRA) | 0.2407 | 0.3322 | 0.4907 | 0.3907 | 0.8060 |
| Route planning (acc) | 0.2727 | 0.2273 | 0.0909 | 0.3636 | 0.3636 |

- **Adding the video hurts:** with simple2, video + scene is 15.7 points below scene only. With frames
  in the prompt the model reasons from what it sees instead of computing from the coordinates:
  relative direction 0.84 → 0.42, room size 0.62 → 0.27, absolute distance 0.81 → 0.49, route
  planning 0.36 → 0.09.
- **Only appearance order gains** (0.41 → 0.65): the frame order is the one thing the boxes lack.
- **simple2 vs basic, with video:** +3.8 points. As in Table 2 of the paper, video on top of the scene
  language does not help the 4B model.

## Fine-tuned: Perceiver + VGGT-Omega (`main`, 743 steps)

`configs/main.env` retrained from scratch with sr_vggt: Qwen3-VL-2B, VGGT-Omega 512 checkpoint (frozen,
run live in bf16), frame Perceiver (512 latents) + camera Perceiver (32 latents), "clean" prompt,
LoRA r64/a128 on the LLM, lr 1e-4, 190k mix, 743 steps at global batch 256. Frames are decoded live
at 1 fps (at most 64); the original run used det_es frames and cached VGGT features.

- **Training:** jz-chain `sr-main` (`scripts/jobs/chain_dev.sbatch`), 14 links on 2 × H100
  (qos_gpu_h100-dev), 25.6 h wall-clock, 51.2 GPU-h, 116.5 s/step, peak 29.3 GiB per GPU,
  373 M trainable parameters (10.1% of 3.71 B).
- **Output:** `$SR_OUTPUT_ROOT/sr_vggt_main/checkpoint-743`.
- **Eval:** `scripts/eval.sh main $SR_OUTPUT_ROOT/sr_vggt_main/checkpoint-743` (8 shards), merged in
  `eval_checkpoint-743_full/results_full_merged.json`. Subsets re-scored from those predictions as in
  the section above.

The no-VGGT baseline (`novggt`: the same recipe with `--arch none` and the plain prompt, 743 steps)
is shown for comparison.

| Task | `main` (ours) | `novggt` (ours) | `main`, EXPERIMENTS.md |
|---|---|---|---|
| **Overall, full (5,130)** | **0.5992** | **0.6023** | **0.6000** |
| Appearance order (acc) | 0.7799 | 0.7605 | |
| Object size (MRA) | 0.7395 | 0.7400 | |
| Object counting (MRA) | 0.7092 | 0.7135 | |
| Room size (MRA) | 0.6851 | 0.6903 | |
| Relative distance (acc) | 0.6028 | 0.6113 | |
| Relative direction (acc) | 0.4864 | 0.4766 | |
| Absolute distance (MRA) | 0.4508 | 0.4393 | |
| Route planning (acc) | 0.3402 | 0.3866 | |
| Debiased (2,362) | 0.5364 | 0.5392 | |
| **Debiased · ScanNet++ (752)** | **0.5242** | **0.5234** | |

- **Reproduced:** 0.5992 against 0.6000 for the original run.
- **VGGT branch:** no gain over `novggt` in our runs (−0.3 points on the full split, +0.1 on
  debiased ScanNet++). It helps appearance order (+1.9), relative direction (+1.0) and absolute
  distance (+1.2), and loses on route planning (−4.6, 194 questions).

## Fine-tuned: Qwen3.5-4B on video (`novggt_qwen35_4b`, 743 steps)

`configs/novggt_qwen35_4b.env`: `novggt`'s recipe on Qwen3.5-4B (revision `851bf6e8`). No VGGT branch
(`--arch none`), plain prompt, thinking off at train and eval (the empty `<think></think>` sits in the
prompt), LoRA r64/a128 on the LLM including the Gated-DeltaNet `in_proj_qkv` / `in_proj_z` / `out_proj`,
lr 1e-4, 190k mix, 1 fps (at most 64 frames), 743 steps at global batch 256. Input: video + question;
target: the bare answer.

- **Training:** jz-chain `sr-q35`, 12 links on 2 × H100 (qos_gpu_h100-dev), 28.5 h wall-clock
  (2026-10-21 23:16 to 10-23 03:47), 44.9 GPU-h, 109 s/step, peak 31.9 GiB per GPU, 122 M trainable
  parameters (2.6% of 4.66 B).
- **Output:** `$SR_OUTPUT_ROOT/sr_vggt_novggt_qwen35_4b/checkpoint-743`.
- **Eval:** `scripts/eval.sh novggt_qwen35_4b $SR_OUTPUT_ROOT/sr_vggt_novggt_qwen35_4b/checkpoint-743`
  (8 shards, 16 new tokens). Subsets in `eval_checkpoint-743_full/results_full_firstline_subsets.json`.
- **Stop-token fix:** Qwen3.5-4B ships no `generation_config.json`, and its `config.json` stops only on
  `<|endoftext|>`. The fine-tuned model ends its answer with `<|im_end|>` (as trained) and then runs on
  to the 16-token limit (`"3\nuser\nuser\nassistant\n<think>"`), which scores 0.0538 as written. The
  scores below take the first line of each prediction; none is empty. `sr_vggt/eval.py` now stops on
  both `<|im_end|>` and `<|endoftext|>`.

| Task | `novggt_qwen35_4b` | Qwen3.5-4B zero-shot | `novggt` (2B) |
|---|---|---|---|
| **Overall, full (5,130)** | **0.6936** | 0.5510 | 0.6023 |
| Appearance order (acc) | 0.8107 | 0.6877 | 0.7605 |
| Object size (MRA) | 0.7621 | 0.6749 | 0.7400 |
| Object counting (MRA) | 0.7365 | 0.5227 | 0.7135 |
| Room size (MRA) | 0.7337 | 0.5726 | 0.6903 |
| Relative distance (acc) | 0.6901 | 0.5789 | 0.6113 |
| Relative direction (acc) | 0.7757 | 0.6141 | 0.4766 |
| Absolute distance (MRA) | 0.5506 | 0.3916 | 0.4393 |
| Route planning (acc) | 0.4897 | 0.3660 | 0.3866 |
| Debiased (2,362) | 0.6370 | 0.4590 | 0.5392 |
| **Debiased · ScanNet++ (752)** | **0.6230** | **0.4288** | **0.5234** |

Debiased · ScanNet++ per task:

| Task | `novggt_qwen35_4b` | Qwen3.5-4B zero-shot | `novggt` (2B) |
|---|---|---|---|
| **Overall** | **0.6230** | 0.4288 | 0.5234 |
| Appearance order (acc) | 0.7609 | 0.6884 | 0.7246 |
| Object size (MRA) | 0.6455 | 0.5691 | 0.5667 |
| Object counting (MRA) | 0.6284 | 0.3336 | 0.5509 |
| Room size (MRA) | 0.7000 | 0.4622 | 0.6162 |
| Relative distance (acc) | 0.6893 | 0.5049 | 0.5243 |
| Relative direction (acc) | 0.5693 | 0.3587 | 0.3422 |
| Absolute distance (MRA) | 0.4907 | 0.2407 | 0.3627 |
| Route planning (acc) | 0.5000 | 0.2727 | 0.5000 |

- **Gain from fine-tuning:** +14.3 points on the full split (0.5510 → 0.6936) and +19.4 on debiased
  ScanNet++ (0.4288 → 0.6230). Every task improves; counting (+21.4 full) and absolute distance (+15.9)
  gain the most, the two tasks where zero-shot 4B was weakest.
- **4B against 2B, same recipe:** +9.1 points on the full split and +10.0 on debiased ScanNet++. The
  largest gap is relative direction (+29.9 full, +22.7 debiased ScanNet++), where 4B already led
  zero-shot (+19.1). Route planning ties on debiased ScanNet++ (22 questions).
- **Against the scene language:** on debiased ScanNet++, video-only fine-tuned 4B (0.6230) is below
  zero-shot 4B reading the ground-truth scene with the simple2 prompt (0.7051) but above
  zero-shot video + scene (0.5483).

## Fine-tuned: video + scene language → answer (`novggt_reason190k_vs`, `novggt_qwen35_4b_reason190k_vs`)

Both runs train on reason-190k (the 145,647 SimpleSR traces) rebuilt as video + scene → answer by
`data/reason190k_videoscene/build.py`:

- **Input:** "This is the 3D scene context." + video + the human turn. Inside the human turn, after "These
  are frames of a video.", come a one-line description of the format and a `<scene>` list of the objects
  the question needs, one `{"label", "bbox_3d"}` per line. Each box is upright and axis-aligned,
  `[x, y, z, w, d, h]` in metres, origin at the first frame's camera, z up. The list is shuffled (seeded
  by row), because the traces list objects in order of first appearance, which would give away
  appearance order.
- **Target:** the bare answer of the source 190k row. No CoT, no scene copy.
- **Recipe:** `novggt`'s (`--arch none`, plain prompt, LoRA r64/a128, lr 1e-4, 1 fps with at most 64
  frames), 1 epoch at global batch 256 = 569 steps. Qwen3.5-4B runs with thinking off.

| | sr-rvs | sr-q35vs |
|---|---|---|
| Base model | Qwen3-VL-2B | Qwen3.5-4B |
| Config | `configs/novggt_reason190k_vs.env` | `configs/novggt_qwen35_4b_reason190k_vs.env` |
| Output | `$SR_OUTPUT_ROOT/sr_vggt_novggt_reason190k_vs/checkpoint-569` | `$SR_OUTPUT_ROOT/sr_vggt_novggt_qwen35_4b_reason190k_vs/checkpoint-569` |
| Chain links (2 × H100) | 7 | 11 completed + 2 failed |
| Compute | 12.3 h (24.6 GPU-h) | 19.3 h (38.6 GPU-h), failed links included |
| Trainable parameters | 69.7 M (3.2% of 2.20 B) | 121.9 M (2.6% of 4.66 B) |
| Peak memory per GPU | 27.0 GiB | 51.1 GiB (last link) |

- **Out-of-memory at step 132 (sr-q35vs):** a few rows have long scene lists. Input text has a median of
  291 tokens, but the maximum is 10,312, and 8 rows are over 8k.
  - One such row sits in step 132's batch. Without gradient checkpointing it needed more than 79 GiB, so
    two links in a row crashed at that step and the chain stopped.
  - **Fix:** `--checkpoint_above 8192` (`sr_vggt/train.py`, with `GRAD_CKPT=True` in the config). This
    turns gradient checkpointing on only for micro-batches longer than 8,192 tokens. The rest run at full
    speed, and the gradients are the same.
  - The run resumed from checkpoint-125.

### Eval: debiased ScanNet++ (752), video + ground-truth scene

`data/scenelang_vsibench_scannetpp/eval_scenelang.py --input video_scene --system short --max_new_tokens 16`
(jobs: `job_rvs.slurm`, `job_q35vs.slurm`). The input is in the training format, built from the VSI-Bench
metadata boxes (`scannetpp_meta_info_val.json`). Each 9-DoF box becomes an upright, axis-aligned box
enclosing it, `[x, y, z, w, d, h]` rounded to 0.01. The coordinates stay in the ScanNet++ world frame: the
metadata gives no first-frame camera to move the origin to. Repeated labels are numbered ("chair 1", "chair
2"), and the list is shuffled with a fixed seed. There are two variants of which objects go into the scene:

- **Question objects (`--prompt rvs_question`):** only the boxes whose label appears in the question.
  - Labels are matched on word boundaries, longest label first, with an optional plural "s"/"es". Every
    instance of a matched label is kept.
  - Room size keeps every object, since it needs the whole room.
  - This matches the training format, where each trace lists only the objects the question needs.
- **All objects (`--prompt rvs_all`):** every annotated object in the scene. The model has to find the
  relevant objects itself.

| Task | sr-q35vs, question objects | sr-q35vs, all objects | sr-rvs, question objects | sr-rvs, all objects |
|---|---|---|---|---|
| **Overall** | **0.7924** | **0.7586** | 0.6662 | 0.5231 |
| Appearance order (acc) | 0.7246 | 0.7319 | 0.7464 | 0.7391 |
| Absolute distance (MRA) | 0.8720 | 0.6144 | 0.8051 | 0.3246 |
| Object counting (MRA) | 0.9888 | 0.9853 | 0.9888 | 0.7224 |
| Relative distance (acc) | 0.8350 | 0.8544 | 0.5825 | 0.5922 |
| Object size (MRA) | 0.9276 | 0.8407 | 0.9163 | 0.5577 |
| Room size (MRA) | 0.6730 | 0.6730 | 0.6081 | 0.6081 |
| Route planning (acc, 22) | 0.3182 | 0.4091 | 0.3182 | 0.2727 |
| Relative direction (acc) | 1.0000 | 0.9603 | 0.3641 | 0.3679 |

- **4B against 2B, same data:** +12.6 points with question objects and +23.6 with all objects.
  - Relative direction goes from 0.36 to 1.00, so 4B learned to work out left, right and behind from the
    box coordinates.
  - Relative distance gains about 25 points.
- **More objects in the scene:** going from question objects to all objects costs 4B 3.4 points and 2B
  14.3. Most of 4B's loss is absolute distance (0.87 → 0.61), where it has to pick the right instance
  among many objects.
- **How to read these numbers:** the boxes are ground truth, so they are close to an upper bound for a
  model that reads a scene. They don't measure how well it perceives the scene from video.
  - With question objects, counting is nearly free: the scene lists every instance of the label asked
    about, so the answer is the number of lines.
  - Choosing objects from the question text leaks no answer, since the model sees the question anyway,
    but it does the "find the relevant objects" step for the model.
- **Against the other runs on this subset:**

  | Run | Score |
  |---|---|
  | Video-only fine-tuned 4B (`novggt_qwen35_4b`) | 0.6230 |
  | Zero-shot 4B, scene only (simple2) | 0.7051 |
  | Zero-shot 4B, video + scene (simple2) | 0.5483 |
  | Oracle code | 0.7161 |
  | Claude, no code | 0.8987 |

  The remaining gaps to Claude are room size (0.67 against 0.80) and route planning (0.32–0.41 against
  0.64).
- **Stop token:** `eval_scenelang.py` now also stops on `<|im_end|>` (as `sr_vggt/eval.py`). Every
  sr-q35vs answer is one line of at most 6 tokens.

### Eval: ScanNet and ARKitScenes, ground-truth vs VGGT-Det scene (sr-q35vs)

The same eval (`--input video_scene --system short --max_new_tokens 16`, both variants) on the other two
VSI-Bench datasets, with two sources for the scene.

- **Ground-truth scene:** `build.py <meta_info> <VSI-Bench> <out> <dataset>` builds the scene language
  for each dataset: `vsibench_scannet_scenelang.jsonl` has 2,071 rows (935 debiased, 88 scenes), and
  `vsibench_arkitscenes_scenelang.jsonl` has 1,601 (675 debiased, 150 scenes).
  - The boxes stay in each dataset's metadata frame: for ScanNet the raw mesh minus its mean, for
    ARKitScenes the ARKit mesh frame. Both have z up.
  - Only 20 route-planning rows name no metadata label; they get every object.
  - Job: `job_q35vs_scannet_arkit.slurm` (8 tasks, about 20 min each).
- **VGGT-Det scene:** `data/scenelang_vsibench_vggtdet/rows_first_camera.jsonl` (its `build.py`), VGGT-Det
  seed 0.
  - Score thresholds are 0.3 for ScanNet and 0.2 for ARKitScenes. Boxes are upright and axis-aligned
    `[x, y, z, w, d, h]`, with labels mapped to the VSI-Bench names.
  - The origin is the first video frame's camera, the training convention.
  - It covers 2,969 rows: all of ScanNet, plus the 82 of 150 ARKitScenes scenes the detector ran on.
  - `rvs_scene` strips the detector's numbering before matching labels and renumbering.
  - Job: `data/scenelang_vsibench_vggtdet/job_q35vs.slurm` (4 tasks on H100, about 14 min each).

On the 2,969 rows that have VGGT-Det boxes:

| Task | Ground truth, question objects | Ground truth, all objects | VGGT-Det, question objects | VGGT-Det, all objects | Video only (`novggt_qwen35_4b`) | CoT (`reason190k_think`) |
|---|---|---|---|---|---|---|
| **Overall, full (2,969)** | **0.8174** | **0.7625** | **0.5947** | **0.5934** | **0.7011** | **0.5612** |
| Appearance order (acc) | 0.8077 | 0.8205 | 0.7974 | 0.8179 | 0.8154 | 0.7513 |
| Object size (MRA) | 0.9281 | 0.8844 | 0.6091 | 0.5961 | 0.7558 | 0.7373 |
| Object counting (MRA) | 0.9857 | 0.9757 | 0.5399 | 0.5744 | 0.7525 | 0.6761 |
| Room size (MRA) | 0.6700 | 0.6700 | 0.6735 | 0.6741 | 0.7382 | 0.4824 |
| Relative distance (acc) | 0.9123 | 0.8743 | 0.5263 | 0.5673 | 0.6930 | 0.4503 |
| Relative direction (acc) | 0.9878 | 0.9820 | 0.7148 | 0.7006 | 0.8097 | 0.6391 |
| Absolute distance (MRA) | 0.8527 | 0.4980 | 0.4688 | 0.4057 | 0.5765 | 0.3986 |
| Route planning (acc) | 0.3952 | 0.3952 | 0.4274 | 0.4113 | 0.4677 | 0.3548 |
| Full · ScanNet (2,071) | 0.8076 | 0.7475 | 0.5919 | 0.5933 | 0.6843 | 0.5769 |
| Full · ARKitScenes (898) | 0.8334 | 0.7806 | 0.5621 | 0.5582 | 0.7160 | 0.4996 |
| **Debiased (1,312)** | **0.7973** | **0.7448** | **0.5430** | **0.5521** | **0.6505** | **0.4821** |
| Debiased · ScanNet (935) | 0.7985 | 0.7435 | 0.5611 | 0.5700 | 0.6418 | 0.5094 |
| Debiased · ARKitScenes (377) | 0.8004 | 0.7520 | 0.4498 | 0.4605 | 0.6588 | 0.4096 |

With the ground-truth scene on all ScanNet + ARKitScenes rows (3,672: 0.8184 / 0.7656 full, 0.7965 /
0.7446 debiased), the results agree with debiased ScanNet++ (0.7924 / 0.7586).

- **VGGT-Det boxes cost 22 points against ground truth** (0.8174 → 0.5947) and land 10.6 below video only
  (0.7011). sr-q35vs was trained on ground-truth boxes, so it trusts the scene over the video, and
  detector errors go straight into the answer.
- **Where it loses:**
  - Counting (0.99 → 0.54): the model counts the boxes, and the detector misses and adds instances.
  - Relative direction (0.99 → 0.71) and relative distance (0.91 → 0.53).
  - Size (0.93 → 0.61).
- **What holds:**
  - Appearance order (about 0.80, set by the video).
  - Room size (0.67 in both): with no object named, `rvs_question` gives every box in both cases.
  - Route planning, which is low everywhere.
- **The two variants are close with VGGT-Det** (0.5947 / 0.5934): the detector predicts about 18
  classes, so most of a question's objects either have the same few boxes or none.
  - With ground truth, all objects costs 5.5 points, mostly in absolute distance (0.85 → 0.50).
- **VGGT-Det scene vs the model's own CoT scene:** the detector's boxes are 3.4 points better (0.5947 vs
  0.5612), but neither beats answering from video alone. A predicted scene helps this model only if it
  is close to ground truth.

## Fine-tuned: video → scene + CoT → answer (`novggt_qwen35_4b_reason190k_think`, 569 steps)

`configs/novggt_qwen35_4b_reason190k_think.env`: the same rows, model setup and steps as sr-q35vs, but the
scene is in the target instead of the input. Built by `data/reason190k_think/build.py` from the 145,647
SimpleSR traces (all parsed).

- **Input:** video + the human turn, with `PLAIN_SYSTEM_PROMPT` (`sr_vggt/reasoning.py`). The system prompt
  describes the `<scene>` format: the objects the question needs as `{"label", "bbox_3d": [x, y, z, w, d,
  h]}` in metres, origin at the first frame's camera, z up.
- **Target:** `<think>\n{opener}\n<scene>…</scene>\n{reasoning}\n</think>\n\nAnswer: X`.
  - Qwen3.5's chat template keeps only the text between `<think>` and `</think>` and drops anything
    before `<think>`. So the trace's own layout (opener and scene before `<think>`) would lose the scene
    from the supervision. The build moves everything into Qwen3.5's native thinking block.
  - Thinking is on at train and eval (`--prompt reason_plain`). `sr_vggt/data.py` asserts that the prompt
    is a prefix of the full text.
- **Recipe:** LoRA r64/a128, lr 1e-4, 1 fps with at most 64 frames, 569 steps at global batch 256,
  `--max_seq_len 16384`. CoT targets have a median of 608 tokens, but the maximum is 12,127 (10 over 8k).
- **Training:** jz-chain `sr-q35think` on 2 × H100. It hit the same out-of-memory at step 132 as sr-q35vs,
  fixed the same way (`--checkpoint_above 8192`) and resumed from checkpoint-125. 9 links after the fix,
  15.5 h, peak 58.9 GiB per GPU. Finished 2026-10-25 22:27.
- **Output:** `$SR_OUTPUT_ROOT/sr_vggt_novggt_qwen35_4b_reason190k_think/checkpoint-569`.
- **Eval:** `sr_vggt/eval.py` with up to 8,192 new tokens. The score uses the text after the last
  "Answer:".
  - It ran as 8 shards; shards still unfinished after a 2 h round were split into 4 parts each
    (`EVAL_PARTS=4`).
  - The A100 t3 queue never started the jobs (A100 fairshare ≈ 0.001). The first round ran on H100 dev
    (`GPU=h100 scripts/eval.sh …`, now an option of `scripts/eval.sh` and `scripts/jobs/eval.slurm`).
  - The remaining parts ran on H100 dev, falling back to A100 dev when H100 was full. They were
    submitted by a driver loop (`$SR_LOG_DIR/cot_driver.sh`) that kept every part in flight.
  - Merged: `eval_checkpoint-569_full/results_full_merged.json`.
- **Output format:** every checked prediction follows the trained layout: opener, `<scene>` list of 1–4
  boxes, step-by-step arithmetic over them, `</think>`, `Answer: …`. The median length is 571 tokens.
  276 of 5,130 have no parseable `<scene>` block. 340 (6.6%) run to the 8,192-token limit without an
  answer and score 0: 197 relative distance (28% of that task), 61 counting, 54 route planning and 21
  appearance order.

| Task | CoT (`reason190k_think`) | Video → answer (`novggt_qwen35_4b`) |
|---|---|---|
| **Overall, full (5,130)** | **0.5307** | **0.6936** |
| Appearance order (acc) | 0.7233 | 0.8107 |
| Object size (MRA) | 0.7404 | 0.7621 |
| Object counting (MRA) | 0.6255 | 0.7365 |
| Room size (MRA) | 0.4431 | 0.7337 |
| Relative distance (acc) | 0.3620 | 0.6901 |
| Relative direction (acc) | 0.6103 | 0.7757 |
| Absolute distance (MRA) | 0.3802 | 0.5506 |
| Route planning (acc) | 0.3608 | 0.4897 |
| Full · ScanNet / ARKitScenes / ScanNet++ | 0.5769 / 0.4799 / 0.5043 | 0.6843 / 0.7023 / 0.6747 |
| Debiased (2,362) | 0.4552 | 0.6370 |
| Debiased · ScanNet / ARKitScenes | 0.5094 / 0.3922 | 0.6418 / 0.6230 |
| **Debiased · ScanNet++ (752)** | **0.4441** | **0.6230** |

- **CoT is below direct answering everywhere:** 16.3 points on the full split and 18.2 on debiased, and on
  every task and dataset. The largest losses are relative distance (−32.8) and room size (−29.1). Object
  size, which needs only the box's own extents, nearly holds (−2.2). Part of the relative-distance loss
  is runaway generation: 28% of those questions never reach an answer.
- **The cause is the boxes, not the reasoning:** the arithmetic follows the traces, but it starts from boxes
  the model estimates from a moving camera with no depth or poses. Every error in a box goes straight into
  the answer, while the direct-answer model never commits to coordinates.

### Quality of the predicted scene: mAP@0.25 / mAP@0.5

`data/reason190k_think/eval_scene_map.py`, run in the VGGT-Det mmdet3d env (CPU), uses the metric of
`../VGGT-Det-CVPR2026` (mmdet3d `indoor_eval`, IoU 0.25 and 0.5).

- **Ground truth:** VSI-Bench's `meta_info` boxes, as upright boxes with yaw. Each question is one image: the
  ground truth is every instance of the labels the question names (the `rvs_question` rule). For room
  size, which names no object, it is every instance of the predicted labels.
- **Predictions:** the CoT's `<scene>` boxes, with the numbering stripped from labels and a score of 1 each
  (the model gives no confidence). AP is averaged over classes with ground truth. Plain precision and
  recall are given too, since with constant scores AP follows the list order.
- **Frame:** the CoT frame is the first video frame's camera, whose pose is unknown. One rigid transform
  per scene (yaw about z + 3D translation) is fitted by RANSAC over same-label box centres, pooled over all
  of the scene's questions, then refined by least squares. This fixes the origin only: the boxes'
  positions relative to each other, their sizes and their shapes are scored as written.
- **Ceiling (`--selftest`):** the ground-truth boxes, written the CoT way (upright and axis-aligned in a
  room-aligned frame, random rigid offset per scene), as predictions. It loses to axis-aligning oriented
  boxes and to the 12 of 288 scenes with too few distinct objects to fit a frame.

| All 5,130 questions | mAP@0.25 | mAP@0.5 | P / R @0.25 | P / R @0.5 |
|---|---|---|---|---|
| **CoT boxes** | **0.0731** | **0.0126** | 0.147 / 0.104 | 0.033 / 0.023 |
| CoT · ScanNet | 0.1325 | 0.0264 | 0.150 / 0.120 | 0.023 / 0.018 |
| CoT · ARKitScenes | 0.1378 | 0.0220 | 0.232 / 0.228 | 0.070 / 0.069 |
| CoT · ScanNet++ | 0.0161 | 0.0031 | 0.068 / 0.034 | 0.015 / 0.008 |
| Ceiling (ground-truth boxes) | 0.8965 | 0.6001 | 0.941 / 0.940 | 0.773 / 0.772 |

- **Few boxes are right:** 15% of predicted boxes overlap a ground-truth box at IoU 0.25, and 3% at 0.5.
  After alignment only half of a scene's boxes lie within 0.5 m of a same-label box (median), and those
  are off by a median of 0.23 m.
- **ScanNet++ is the worst:** its rooms are larger with many small objects, and the model lists only some
  of the instances (4,869 predicted against 9,768 ground-truth boxes).
- **Recall is lowest on the many-object questions:** appearance order (R@0.25 0.06), relative distance
  (0.05) and counting (0.09).

## VSI-Bench ground-truth scenes in the training frame (first-frame camera)

Every VSI-Bench scene has ground-truth 3D boxes in `meta_info/{scannet,scannetpp,arkitscenes}_meta_info_val.json`
(288 scenes, 5,130 questions). These are the boxes VSI-Bench generated its QA from.
`data/scenelang_vsibench_scannetpp/build.py` writes them in each dataset's own world frame, and all the
ground-truth-scene evals above use those files.

`data/scenelang_vsibench_gt_firstcam/build.py` moves the same boxes into the frame of the reason-190k
traces: z up, horizontal axes along the room, origin at the camera of the first video frame.

- **Format unchanged:** boxes stay 9-DoF `[x, y, z, dx, dy, dz, yaw, roll, pitch]`, and the rows have the
  same fields and order as before. The zero-shot prompts and `eval_scenelang.py` read them unchanged.
- **ScanNet and ARKitScenes:** the frames match `data/scenelang_vsibench_vggtdet` (`rows_first_camera`).
  For the ARKitScenes room rotation, see the caveat below.

```bash
# VGGT-Det mmdet3d env (CPU): module pytorch-gpu/py3/2.1.1, PYTHONPATH=$VGGTDET/pylib:$VGGTDET/code_main
python data/scenelang_vsibench_gt_firstcam/build.py $SR_VSIBENCH data/scenelang_vsibench_gt_firstcam
python3 data/scenelang_vsibench_gt_firstcam/make_viewer.py     # -> docs/scene_viewer.html
```

The builder writes `vsibench_{scannet,scannetpp,arkitscenes}_scenelang.jsonl` and `frames.json`. The
latter records each scene's room rotation, origin in the VSI-Bench frame, and whether the origin is the
first-frame camera or the world.

| Dataset | Source frame (VSI-Bench) | Axes | Origin (first-frame camera) |
|---|---|---|---|
| ScanNet | raw mesh minus its mean | the scan's `axis_align_matrix` frame (VGGT-Det's frame) | posed image 00000, which is the video's first frame. From VGGT-Det's `scannet_infos_val_vsi.pkl`. |
| ARKitScenes | ARKit mesh frame | turned about z by the room yaw | the first `lowres_wide.traj` pose of the raw release. It matches VGGT-Det's infos to 1e-16 m on 147 scenes. |
| ScanNet++ | ScanNet++ mesh frame | turned about z by the room yaw | the ARKit `aligned_pose` of iPhone pose frame 0, which is mp4 frame 0. From `~/scratch/data/scannetpp_cam0` (`prebuilt/det_es/<scene>/meta.json`, `cam0_cam2global`). |

- **Room yaw:** the main direction of the upright ground-truth boxes modulo 90°, weighted by footprint
  (the circular mean of 4 × yaw).
- **ScanNet++ poses:** they share the VSI-Bench frame. The bundle's boxes map back onto the VSI-Bench
  centroids to within 2 cm in all 49 scenes.

**Two ScanNet++ scenes stay in the world frame (80 of 1,458 questions).** Their axes are still turned to
the room and z is up, but the origin is the ScanNet++ world origin, not the first-frame camera.

- **e7af285f7d** (3 questions) has no pose in the bundle.
- **d755b3d9d8** (77 questions) has a broken `aligned_pose` track: frame 0 sits 55.8 m above the floor,
  and later frames drift to hundreds of metres.
- **Plausibility check:** the builder rejects a first-frame camera that is not 0.3–3 m above the lowest
  box bottom, or that lies more than 2 m outside the scene's extent. These two scenes are the only ones
  it rejects.
- **Effect:** distances, sizes, counts and directions don't depend on the origin. Only the prompt's
  "origin at the first frame's camera" is untrue for these rows.

| | ScanNet | ScanNet++ | ARKitScenes |
|---|---|---|---|
| Scenes / questions | 88 / 2,071 | 50 / 1,458 | 150 / 1,601 |
| Origin at the first-frame camera | 88 | 48 (world: e7af285f7d, d755b3d9d8) | 150 |
| Max change in object–object distance | 0.018 m | 0.021 m | 0.020 m |
| Large boxes (footprint > 0.3 m²) within 5° of the axes, world → new | 16% → 61% | 24% → 63% | 51% → 85% |
| Median \|yaw mod 90°\| of large boxes, world → new | 16.6° → 3.5° | 10.4° → 2.3° | 4.6° → 1.1° |
| Upright-AABB footprint / true footprint (what sr-q35vs sees), world → new | 1.78 → 1.27 | 1.61 → 1.31 | 1.42 → 1.09 |
| First camera above the lowest box bottom, median (range) | 1.43 m (0.92–1.64) | 1.43 m | 1.06 m (0.27–4.15) |

The distance changes come only from the 2-decimal rounding.

**Caveats**
- **The match with the traces is unverified.** The trace README gives only "origin at the first sampled
  camera, axes follow the room, z up". The room alignment here (ScanNet `axis_align_matrix`, the
  dominant box yaw elsewhere) is a reconstruction. Two checks remain:
  - draw the boxes onto frame 0 of the video with that camera's pose and intrinsics;
  - rebuild a few training scenes with this recipe and compare them with their trace boxes.
- **ARKitScenes origin error:** the VSI-Bench video (vga_wide) starts 0.6–1.4 s before the first
  lowres_wide pose. Over that time the camera moves a median 0.11 m (90th percentile 0.51 m), which is
  the error on the origin. One scene, 42899461, puts the camera 4.15 m above its lowest box, possibly
  because the scene has several levels.
- **VGGT-Det ARKitScenes scenes are in a different frame.** `scenelang_vsibench_vggtdet/build.py` takes
  the room yaw from its own detections (mAP@0.25 0.30 on ARKitScenes). On 40 of the 82 scenes that yaw
  is more than 10° from the ground-truth room yaw, and the median gap is 8.6°. The raw ARKit frame is
  already within 4.6° of the walls (median), so this rotation makes those scenes worse. The VGGT-Det
  ARKitScenes rows therefore don't share the ground-truth frame. ScanNet does, since both use
  `axis_align_matrix`.
- **Earlier evals used the world frame.** The fine-tuned 81.7 / 79.2 and the zero-shot 54.8 above were
  measured on the world-frame scenes. The evals in the next section use these files.

**Viewer:** `docs/scene_viewer.html` shows all 288 scenes in either frame, with the 9-DoF boxes or the
upright AABBs that sr-q35vs sees. It marks the origin and the first-frame camera, flags the two
world-origin scenes, and loads three.js 0.128 from jsDelivr.

## Evals on the training-frame scenes (exps 1–7, all of VSI-Bench)

All runs below use Qwen3.5-4B (revision `851bf6e8`), greedy decoding, thinking off, and up to 16,384 new
tokens. They stop on `<|im_end|>` or `<|endoftext|>`. "Fine-tuned" means sr-q35vs
(`novggt_qwen35_4b_reason190k_vs`, checkpoint-569, trained on video + scene → answer) with its training
system prompt. Scenes are the training-frame ground truth from the section above unless stated.

- **Runner:** `data/scenelang_vsibench_scannetpp/eval_scenelang.py`, one shard per 2 h dev job.
  `data/scenelang_vsibench_gt_firstcam/zs_simple2_driver.sh` keeps the shards queued on h100-dev and
  a100-dev and resubmits until each is complete (`job_zs_simple2.slurm`, logs in `logs_zs_simple2/`).
  Outputs are in the same directory.
- **Scoring:** the last option letter or number after the last `Answer:` line, as in `final_answer()`.
  Scored with `sr_vggt.eval.score`.

| Exp | Model | Input | Prompt | Rows file | Output dir |
|---|---|---|---|---|---|
| 1 | zero-shot | GT scene | simple2 | `rows_zs_simple2_rest.jsonl` (4,378) + the earlier world-frame debiased ScanNet++ run (752) | `qwen35_4b_scene_simple2prompt` |
| 1, check | zero-shot | GT scene | simple2 | `rows_zs_simple2_debiased_scannetpp.jsonl` (752) | `qwen35_4b_scene_simple2prompt_debiased_scannetpp` |
| 2 | zero-shot | video + GT scene | simple2 | `rows_zs_simple2_rest.jsonl` (4,378) + the earlier debiased ScanNet++ run (752) | `qwen35_4b_video_scene_simple2prompt` |
| 3 | zero-shot | video + VGGT-Det boxes | simple2 | `rows_zs_vggtdet_simple2.jsonl` (2,969) | `qwen35_4b_video_vggtdet_simple2prompt` |
| 4 | fine-tuned | video + GT scene | simple2 | `rows_all5130.jsonl` | `sr_q35vs_569_video_scene_simple2prompt` |
| 4, rerun | fine-tuned | video + GT scene | `rvs_question` | `rows_all5130.jsonl` | `sr_q35vs_569_video_scene_rvs_question` |
| 4, world | fine-tuned | video + GT scene, VSI-Bench world frame | `rvs_question` | `rows_all5130_world.jsonl` | `sr_q35vs_569_video_scene_world_rvs_question` |
| 6 | zero-shot | video + MAP-Det3D boxes, oracle labels | simple2 | `rows_zs_mapdet3d_simple2.jsonl` (5,094) | `qwen35_4b_video_mapdet3d_simple2prompt` |
| 7 | fine-tuned | video + MAP-Det3D boxes, oracle labels | `rvs_question` / `rvs_all` | `rows_zs_mapdet3d_simple2.jsonl` (5,094) | `sr_q35vs_569_mapdet3d_rvs_question`, `_rvs_all` |

- **Prompts:**
  - **simple2:** the full format description (`FULL_INTRO`) with every object as a 9-DoF box. It ends
    with "Compute the answer step by step from the numbers above, then end with a final line
    'Answer: …'".
  - **`rvs_question`:** the sr-q35vs training format. Only the objects named in the question are listed
    (all objects for room size), as upright axis-aligned `[x, y, z, w, d, h]` boxes. Repeated labels are
    numbered and the list is shuffled. It ends with "Answer with the option's letter from the given
    choices directly." or "Please answer the question using a single word or phrase."
  - **`rvs_all`:** the same as `rvs_question`, with every object listed.
- **VGGT-Det rows (exp 3):** `scenelang_vsibench_vggtdet/build.py --keep-yaw`, seed 0, first-camera
  frame. Label numbers are stripped and roll = pitch = 0. These cover all of ScanNet plus 82 of 150
  ARKitScenes scenes.
- **MAP-Det3D rows (exps 6, 7):**
  - Source: `data/scenelang_vsibench_mapdet3d/vsibench_meta_info_val_mapdet3d_oracle0.1.json`, written in
    the training frame with `build.py $SR_VSIBENCH <dir> <that file> mapdet3d_oracle0.1`.
  - Coverage: 284 scenes. Four have no output (e7af285f7d, 42897538, 47115525, 47115543), which drops
    36 questions.
  - Content: a median of 5 boxes per scene, and 6 scenes are empty. A box centre is a median 0.16 m from
    the ground-truth centre with the same label.
- **World-frame rows (exp 4, world):** `rows_all5130_world.jsonl` is `rows_all5130.jsonl` with
  `scene_language` taken from `data/scenelang_vsibench_scannetpp/vsibench_*_scenelang.jsonl`. All 5,130
  questions and answers match.

### Results

| Run | Full | Debiased | Debiased ScanNet++ | 2,969 VGGT-Det rows | ScanNet | ScanNet++ | ARKitScenes | Answer tokens, median |
|---|---|---|---|---|---|---|---|---|
| Questions | 5,130 | 2,362 | 752 | 2,969 | 2,071 | 1,458 | 1,601 | |
| Fine-tuned video only (`novggt_qwen35_4b`, for reference) | 0.694 | 0.637 | 0.623 | 0.701 | 0.684 | 0.675 | 0.702 | – |
| 1: zero-shot, GT scene, simple2 (752 in the world frame) | 0.690 | 0.675 | 0.707 | 0.683 | 0.704 | 0.711 | 0.727 | 1,794 |
| 1: the same, with the 752 rerun in the training frame | 0.696 | 0.685 | 0.750 | 0.683 | 0.704 | 0.742 | 0.727 | 1,807 |
| 2: zero-shot, video + GT scene, simple2 | 0.591 | 0.560 | 0.548 | 0.586 | 0.585 | 0.591 | 0.571 | 319 |
| 3: zero-shot, video + VGGT-Det, simple2 | – | – | – | 0.430 (debiased 0.379) | 0.423 | – | 0.399 (898) | 296 |
| 4: fine-tuned, video + GT scene, simple2 | 0.771 | 0.744 | 0.740 | 0.776 | 0.775 | 0.767 | 0.771 | 2 |
| **4 rerun: fine-tuned, video + GT scene, `rvs_question`** | **0.825** | **0.809** | **0.804** | **0.829** | 0.831 | 0.817 | 0.831 | 2 |
| 4 world: the same in the world frame | running | | | | | | | |
| 6: zero-shot, video + MAP-Det3D, simple2 (5,094) | 0.448 | 0.391 | 0.340 | 0.461 | 0.440 | 0.393 | 0.470 | 275 |
| 7: fine-tuned, video + MAP-Det3D, `rvs_question` (5,094) | 0.628 | 0.575 | 0.566 | 0.648 | 0.643 | 0.595 | 0.629 | 2 |
| 7: fine-tuned, video + MAP-Det3D, `rvs_all` (5,094) | 0.620 | 0.574 | 0.564 | 0.640 | 0.640 | 0.589 | 0.605 | 2 |

Scores are rounded to 3 decimals. The video-only row is `sr_vggt_novggt_qwen35_4b/eval_checkpoint-743_full`,
scored from the first line of each prediction (see the stop-token note below).

**Per task, all of VSI-Bench (5,130; exp 7 on 5,094):**

| Task | 4: fine-tuned, GT, simple2 | 4 rerun: fine-tuned, GT, `rvs_question` | 7: fine-tuned, MAP-Det3D, `rvs_question` | Fine-tuned video only | Code oracle, GT |
|---|---|---|---|---|---|
| **Overall** | **0.771** | **0.825** | **0.628** | **0.694** | **0.678** (six-task mean 0.904) |
| Appearance order (acc) | 0.804 | 0.785 | 0.803 | 0.811 | – |
| Absolute distance (MRA) | 0.582 | 0.918 | 0.521 | 0.551 | 0.974 |
| Counting (MRA) | 0.898 | 0.990 | 0.752 | 0.736 | 0.990 |
| Relative distance (acc) | 0.846 | 0.893 | 0.650 | 0.690 | 1.000 |
| Size (MRA) | 0.961 | 0.970 | 0.622 | 0.762 | 1.000 |
| Room size (MRA) | 0.666 | 0.642 | 0.577 | 0.734 | 0.458 |
| Route planning (acc) | 0.438 | 0.412 | 0.417 | 0.490 | – |
| Relative direction (acc) | 0.974 | 0.992 | 0.686 | 0.776 | 1.000 |

**Debiased ScanNet++ (752), exp 1 world frame vs training frame:**

| Task | World frame (earlier run) | Training frame (rerun) |
|---|---|---|
| **Overall** | **0.707** | **0.750** |
| Appearance order (acc) | 0.413 | 0.420 |
| Absolute distance (MRA) | 0.803 | 0.840 |
| Counting (MRA) | 0.966 | 0.946 |
| Relative distance (acc) | 0.641 | 0.767 |
| Size (MRA) | 1.000 | 1.000 |
| Room size (MRA) | 0.638 | 0.676 |
| Route planning (acc) | 0.364 | 0.545 |
| Relative direction (acc) | 0.834 | 0.808 |

**Takeaways**
- **The training prompt matters for the fine-tuned model:** `rvs_question` gives +5.4 points over
  simple2 with the same boxes (0.771 → 0.825). Almost all of it is absolute distance (0.58 → 0.92),
  with counting and relative distance adding the rest. The fine-tuned model ignores simple2's
  step-by-step instruction and answers in 2–5 tokens either way.
- **The training frame helps the fine-tuned model a little:** with `rvs_question`, 0.817 → 0.829 on the
  2,969 VGGT-Det rows and 0.792 → 0.804 on debiased ScanNet++, against the world-frame runs above. The
  full world-frame rerun (exp 4 world) is still running.
- **Zero-shot is not stable across frames:**
  - The 752 rerun scores 0.750 against 0.707 in the world frame, and only 498 of 752 final answers are
    identical.
  - Decoding is greedy, but the reasoning runs to about 1,800 tokens, so changed numbers in the prompt
    change the path.
  - The gain is mostly relative distance and route planning (22 questions). A likely cause, not tested,
    is that axis-aligned boxes make the closest-point arithmetic easier.
- **The video hurts zero-shot:** video + GT scene (0.591) is 10 points below the scene alone (0.690).
  Its answers are shorter (median 319 against about 1,800 tokens), and room size and route planning
  collapse.
- **Detected boxes:**
  - With the same fine-tuned model and prompt, MAP-Det3D (oracle labels) costs 20 points against ground
    truth (0.825 → 0.628). The biggest drops are size, absolute distance and relative direction.
  - Exp 7 is still below the fine-tuned video-only model (0.648 against 0.701 on the 2,969 rows).
  - Zero-shot, neither VGGT-Det (0.430) nor MAP-Det3D (0.448) helps.
- **Weak spots with ground-truth boxes:** room size (0.642) and route planning (0.412) stay below the
  video-only model. The boxes have no walls and no path, and the model does not reason.

### Code oracle on all of VSI-Bench

`oracle.py`'s computation, run on all 5,130 rows in both frames. It counts labels, takes the longest
box side, the closest-point distance between oriented boxes and centre cross/dot products. For room
size it uses the smallest rotated rectangle covering every box point in the xy plane. Appearance order
and route planning are not computable from boxes.

| Task | All (5,130) | Debiased ScanNet++ (752) | ScanNet | ScanNet++ | ARKitScenes |
|---|---|---|---|---|---|
| **Six-task mean** | **0.904** | **0.955** | 0.933 | 0.943 | 0.873 |
| Overall, the two other tasks as 0 | 0.678 | 0.716 | | | |
| Size (MRA) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Relative distance (acc) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Relative direction (acc) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Counting (MRA) | 0.990 | 0.989 | 0.985 | 0.993 | 0.994 |
| Absolute distance (MRA) | 0.974 | 0.949 | 0.982 | 0.963 | 0.971 |
| Room size (MRA) | 0.458 | 0.789 | 0.630 | 0.704 | **0.275** |

- **Frame check:** the world-frame boxes give the same scores to within 0.4 points per task, and every
  question is answered in both frames. Every quantity above is invariant to a rigid motion, so this
  confirms the conversion moved each box rigidly. It does not check the room alignment.
- **ARKitScenes room size:** these scenes have a median of 4 labelled boxes, against 12.5 for ScanNet and
  17.5 for ScanNet++. The rectangle covers a median 0.55 of the true floor area (10th percentile 0.22),
  against 1.10 and 0.98. The 0.458 overall is therefore a sparse-box artefact, not a ceiling. The
  fine-tuned model (0.642) beats it, likely helped by the video.

### Claude (no code) on all of VSI-Bench

Claude Opus 5.5 answered the 4,378 rows outside debiased ScanNet++ from the scene text alone. These are
the rows of `rows_zs_simple2_rest.jsonl`: training-frame ground truth with the simple2 description,
scene, question and options, ending in "Answer with the option letter." or "Answer with a number only,
in the unit the question asks for." This is the same per-question text as the earlier 752-row run,
except for the frame.

- **Batches:** 254 batch files, one question type each, with at most one question per scene per batch.
  Caps are 25 questions per batch (12 for relative distance, 10 for room size, 15 for route planning)
  and 90k characters. Built by `build.py` and listed in `manifest.json`.
- **Agents:** one general-purpose sub-agent per batch, at most 20 at a time. Each one reads a shared
  `RULES.md`: only the Read and Write tools, no code or calculators, answer every question from its own
  scene, write one answer JSON. Each final message lists the tools it called.
  - Every one of the 254 reported Read and Write only.
  - A Read/Write-only agent type (`.claude/agents/nocode-answerer.md`) could not be loaded mid-session.
  - The subagent transcripts are incomplete, so the tool use is self-reported, not enforced.
- **Discarded first attempt:** batched like the 752-row run, by scene. One of its agents used Bash to
  read the file, and solved appearance order by cross-checking same-scene questions. That attempt was
  stopped and rebuilt as above.
- **Files:** `scratchpad/manual3/` of this session (`batches/`, `out/`, `predictions.jsonl`,
  `scores.json`, `score.py`). The full row set combines these answers with the earlier 752
  (`$SCRATCH/3d_scene_lang/claude_nocode_scenelang/predictions.jsonl`, world frame, batched by scene).

| Split | Questions | Overall | Appearance order | Absolute distance | Counting | Relative distance | Size | Room size | Route planning | Relative direction |
|---|---|---|---|---|---|---|---|---|---|---|
| **Full (4,378 new + 752 earlier)** | 5,130 | **0.843** | 0.534 | 0.968 | 0.992 | 1.000 | 1.000 | 0.645 | 0.603 | 1.000 |
| New rows only (one question per scene per batch) | 4,378 | 0.830 | 0.452 | 0.972 | 0.993 | 1.000 | 1.000 | 0.623 | 0.599 | 1.000 |
| Debiased | 2,362 | 0.841 | 0.557 | 0.962 | 0.984 | 1.000 | 1.000 | 0.673 | 0.553 | 1.000 |
| Debiased ScanNet++ (earlier run, batched by scene) | 752 | 0.899 | 0.819 | 0.948 | 0.989 | 1.000 | 1.000 | 0.797 | 0.636 | 1.000 |
| 2,969 VGGT-Det rows | 2,969 | 0.829 | 0.433 | 0.972 | 0.989 | 1.000 | 1.000 | 0.655 | 0.581 | 1.000 |
| ScanNet | 2,071 | 0.846 | 0.433 | 0.974 | 0.990 | 1.000 | 1.000 | 0.731 | 0.641 | 1.000 |
| ScanNet++ | 1,458 | 0.889 | 0.706 | 0.961 | 0.993 | 1.000 | 1.000 | 0.762 | 0.692 | 1.000 |
| ARKitScenes (no appearance order) | 1,601 | 0.868 | – | 0.968 | 0.994 | 1.000 | 0.999 | 0.556 | 0.558 | 1.000 |

- **Appearance order depends on the batching:** 0.452 with one question per scene against 0.819 when a
  batch shares scenes. The full-split 0.534 still mixes in the earlier, inflated 752. Rerunning those
  752 with the new batching would give a clean full number.
- **Geometry:** relative distance, relative direction and size are perfect, counting is 0.992 and
  absolute distance 0.968. The six-task mean is 0.934, against 0.904 for the code oracle.
- **Room size:** 0.645 against the oracle's 0.458, and 0.556 against 0.275 on sparse ARKitScenes. Claude
  uses priors on room size instead of covering the few boxes.
- **Route planning:** 0.603 on 194 questions (chance is 0.25–0.33), although only 6 of them have boxes for
  every object named. As with appearance order, this may come from priors or from having seen VSI-Bench.
- **Against the fine-tuned model:** on all of VSI-Bench Claude scores 0.843 and sr-q35vs with
  `rvs_question` 0.825. The model is ahead on appearance order (0.785 against 0.534), which it reads from
  the video. Claude is ahead on room size and route planning.

**Stop-token note (video-only 0.6936):** that eval wrote predictions such as
`"3\nuser\nuser\nassistant\n<think>"`. Qwen3.5-4B has no `generation_config.json`, and its `config.json`
lists only `<|endoftext|>` as end-of-sequence, so `generate()` ran past the model's `<|im_end|>`.
Scored as written it gives 0.054. The 0.6936 keeps the first line of each prediction
(`results_full_firstline_subsets.json`). `sr_vggt/eval.py` and `eval_scenelang.py` now stop on both
tokens.

## Zero-shot on the training traces: ScanNet++ rows (Qwen3.5-4B, job 1272402, 8 × 52–54 min)

Unmodified Qwen3.5-4B (revision `851bf6e8`) on every `scannetppv2/` row of
`data/simplesr_train_traces.jsonl.gz`: 66,183 questions over 839 videos. These are training rows,
so the scores measure the model before fine-tuning on that data, not held-out performance. The
traces share no scene with VSI-Bench.

- **Script:** `$SCRATCH/3d_scene_lang/zeroshot_traces_scannetpp/eval_traces.py` (with `job.slurm`
  next to it). It sits outside the repo. Output: `qwen35_4b/results_merged.json` and per-row
  predictions in that folder.
- **Inference:** the same as the VSI-Bench baseline: 1 fps, at most 64 frames, 64–128 tokens per
  frame, no system prompt, thinking off, greedy decoding with 16 new tokens.
- **Prompt:** the row's human turn with its `<image>\n` tag removed. It already includes the
  lmms-eval post-prompt.
- **Target:** `meta.answer`, the original 190k label, not the trace's own final answer.
- **Scoring:** option letters by exact match after lmms-eval's `fuzzy_matching`; numbers by MRA
  .5:.95:.05. `absolute_direction_object` (degrees) gets the same MRA, though VSI-Bench has no
  such type. Every prediction parsed as a valid letter or number.

| Question type | Questions | Qwen3.5-4B | VSI-Bench full · ScanNet++, closest task |
|---|---|---|---|
| relative_size_object (acc) | 920 | 0.9196 | – |
| appearance_order (acc) | 5,953 | 0.7442 | 0.7149 |
| absolute_size_object (MRA) | 8,227 | 0.6254 | 0.6742 |
| relative_distance_object (acc) | 14,031 | 0.5860 | 0.5704 |
| relative_direction_object (acc) | 12,184 | 0.5703 | 0.5222 |
| absolute_size_room (MRA) | 963 | 0.5361 | 0.4680 |
| route_planning (acc) | 144 | 0.4444 | 0.3077 |
| absolute_count (MRA) | 7,931 | 0.4335 | 0.4472 |
| absolute_distance_object (MRA) | 14,982 | 0.3738 | 0.3252 |
| absolute_direction_object (MRA, degrees) | 848 | 0.3221 | – |
| **Mean of the 10 types** | 66,183 | **0.5555** | 0.5037 (8 tasks) |
| Mean over rows | 66,183 | 0.5361 | – |

The two overall means are not comparable: the traces add two task types VSI-Bench lacks, and
`relative_size_object` is close to a two-way choice. Compare per task instead.
