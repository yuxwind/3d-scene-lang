---
type: experiment
project: 3d-scene-lang
config: novggt_qwen35_4b_reason190k_vs_esl31
host: wukong
branch: qwen35-scene-sft
commit: 605eccd plus uncommitted edits on the host
status: paused
started: 2026-11-13
---
# novggt_qwen35_4b_reason190k_vs_esl31

**Question:** what is the upper bound for a 31-class detector? Train Qwen3.5-4B on Reason-190K with the video plus a ground-truth scene list restricted to the detector's 31 classes (ESL31).

**Baseline:** [[novggt_qwen35_4b_reason190k_vs]] (same recipe, full ground-truth scene).

## Setup

- Config: `configs/novggt_qwen35_4b_reason190k_vs_esl31.env`. Same as the baseline except the data mix (`MIX=reason190k_vs_esl31`), gradient checkpointing, and `EVAL_ARGS=(--max_new_tokens 16)`.
- Data: `data/reason190k_videoscene_esl31/build.py` → `data/reason190k_videoscene_esl31_answer.jsonl.gz`. Keeps objects whose label is one of the 58 labels (31 class names + 27 synonyms) in `docs/vsibench_esl31_label_mapping.txt`, section 5.
  - 145,647 rows: 74,960 keep the full scene, 57,364 a partial one, 13,323 an empty list (kept, not dropped).
  - 430,448 of 584,807 objects kept (73.6%), matching the mapping file.
- Launcher: `scripts/wukong/launch_q35vs_esl31_wukong.sh`. Two `wukong`-only changes: gradient checkpointing on every micro-batch (`--checkpoint_above 0`, needed on 46 GB cards) and transparent huge pages disabled (see [[Compute access#Log of what went wrong]]).
- Note: GitHub has a later commit, `9b8dba1`, that sets the config to checkpoint only micro-batches over 4,200 tokens. The launch below did not use it.

## Budget

4 GPUs × about 19–23 h (569 steps at 120–150 s per step). Not formally approved under the vault's rules; the run predates the vault.

## Where

- Launch: `bash scripts/wukong/launch_q35vs_esl31_wukong.sh` in `wukong:/home/student/xinyu/projects/3d-scene-lang`. Rerunning it resumes from the newest checkpoint.
- Stop: `bash /scr/xinyu/stop_train.sh` on `wukong`.
- Log: `wukong:/scr/xinyu/runs/3d-scene-lang/logs/train_q35vs_esl31.log`
- Output dir: `wukong:/scr/xinyu/runs/3d-scene-lang/sr_vggt_novggt_qwen35_4b_reason190k_vs_esl31`

## Log

- 2026-11-13 — Smoke and two launches stalled after 2–3 steps (memory compaction storm). Fixed in the launcher.
- 2026-11-13 10:21 CST — Third launch. Step 1 took 147 s.
- 2026-11-13 10:25–10:31 CST — The progress bar restarted from step 1 after a warning line; cause not examined.
- 2026-11-13 — The owner said "stop working on wukong for now". The process was left running and has not been looked at since. **Its state is unknown.**

## Result

None yet.
