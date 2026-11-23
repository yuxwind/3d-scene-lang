#!/bin/bash
# gandalf: Qwen3.5-4B video + mixed GT / VGG-T-Det scene with a score per box -> answer SFT (novggt_qwen35_4b_reason190k_vs_mixdet_score), 4 x RTX PRO 6000, global batch 256.
# Rerun the same command to resume from the newest checkpoint.
source "$(dirname "$(readlink -f "$0")")/../exp_paths.sh"
cd /scr/xinyu/projects/3d-scene-lang
export SR_ROOT=$PWD SLURM_JOB_ID=gandalf SLURM_GPUS_ON_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3
OUT=${OUT:-runs/sr_vggt_novggt_qwen35_4b_reason190k_vs_mixdet_score}
exp_paths INPUT=configs/novggt_qwen35_4b_reason190k_vs_mixdet_score.env OUTPUT="$OUT" RESULT="$OUT/checkpoint-*" GPUS="$CUDA_VISIBLE_DEVICES"
bash scripts/jobs/train.slurm novggt_qwen35_4b_reason190k_vs_mixdet_score "$OUT" "$@"
