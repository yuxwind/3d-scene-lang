#!/bin/bash
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread
#SBATCH --time=02:00:00
# Zero-shot Qwen3.5-4B, GT scene text only (training frame, build.py here), --prompt simple2, thinking off, up to
# 16,384 new tokens, on the 4,378 VSI-Bench rows not yet run (all but debiased ScanNet++, which
# scenelang_vsibench_scannetpp/qwen35_4b_scene_simple2prompt covers, world frame). 40 shards; one shard per job
# (SHARD=<k>, GPU=a100|h100 via --export); rows already written are skipped, so a timed-out shard is resubmitted.
# Submitted by zs_simple2_driver.sh, which passes -A/-C/--qos/--output. ROWS / NSH / OUT / EXTRA override the
# rows file, shard count, output dir and extra flags (the driver's other tasks: the debiased ScanNet++ check in this
# frame, and the last 8 rows of the world-frame run). INPUT=video_scene adds the video (exps 2-4); ADAPTER=<LoRA dir>
# runs a fine-tuned model with its training system prompt (--system short, exps 4, 7; SYSTEM=scene for sr-q35sc); PROMPT / MAXTOK override
# --prompt simple2 / --max_new_tokens 16384 (exp 7: the training format rvs_question / rvs_all).
set -euo pipefail
source /lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/scripts/lib.sh
activate_env "${GPU}"
offline
export PYTHONPATH=$PYTHONPATH:/lustre/fsn1/projects/rech/bef/uxt97ti/3d_scene_lang/pyextra/site
G=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_gt_firstcam
python $G/../scenelang_vsibench_scannetpp/eval_scenelang.py --rows ${ROWS:-$G/rows_zs_simple2_rest.jsonl} --input ${INPUT:-scene} --prompt ${PROMPT:-simple2} --vsibench_dir "$SR_VSIBENCH" \
  --max_new_tokens ${MAXTOK:-16384} --num_shards ${NSH:-40} --shard_index "${SHARD}" --out_dir ${OUT:-$G/qwen35_4b_scene_simple2prompt} ${EXTRA:-} ${ADAPTER:+--adapter $ADAPTER --system ${SYSTEM:-short}} \
  --model $HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
