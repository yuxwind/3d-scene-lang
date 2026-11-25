---
type: experiment
project: 3d-scene-lang
config: novggt_qwen35_4b_reason190k_vs
host: gandalf
branch: qwen35-scene-sft
commit: unknown
status: done
---
# novggt_qwen35_4b_reason190k_vs

**Question:** how well does Qwen3.5-4B answer Reason-190K questions when given the video plus the full ground-truth scene list?

> Written on 2026-11-13 from another session's notes, not from the run's own logs. Check the numbers against `gandalf` before quoting them.

## Setup

- Qwen3.5-4B, no VGGT branch (`--arch none`), plain prompt, LoRA r64/a128, lr 1e-4, 1 fps, at most 64 frames.
- Data: `data/reason190k_videoscene/build.py` → video + `<scene>` list of all ground-truth objects → bare answer. 145,647 rows.
- 569 steps at global batch 256.

## Where

- Host: `gandalf`, output root `/scr/xinyu/projects/3d-scene-lang/runs/`.

## Result

- Training: about 65 s per step, about 11 h, peak GPU memory 57–59 GB.
- Score: not recorded here.
