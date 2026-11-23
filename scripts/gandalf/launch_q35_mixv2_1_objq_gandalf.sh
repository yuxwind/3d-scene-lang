#!/bin/bash
# gandalf: Exp1 sft-qwen-test2-hybrid (configs/novggt_qwen35_4b_reason190k_mixv2_1_objq.env): Qwen3.5-4B video + mix-v2-1 with
# IndoorDet-65-31k boxes, GPUs 2, 3, 6, 7 (RTX PRO 6000, 96 GB), global batch 256. Rerun the same command to resume.
source "$(dirname "$(readlink -f "$0")")/../exp_paths.sh"
cd /scr/xinyu/projects/3d-scene-lang
export SR_ROOT=$PWD SLURM_JOB_ID=gandalf SLURM_GPUS_ON_NODE=4 CUDA_VISIBLE_DEVICES=${GPUS:-2,3,6,7}
OUT=${OUT:-runs/sr_vggt_novggt_qwen35_4b_reason190k_mixv2_1_objq}
exp_paths INPUT=configs/novggt_qwen35_4b_reason190k_mixv2_1_objq.env OUTPUT="$OUT" RESULT="$OUT/checkpoint-*" GPUS="$CUDA_VISIBLE_DEVICES"
bash scripts/jobs/train.slurm novggt_qwen35_4b_reason190k_mixv2_1_objq "$OUT" "$@"
