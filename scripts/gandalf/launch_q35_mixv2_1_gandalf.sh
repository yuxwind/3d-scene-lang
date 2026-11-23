#!/bin/bash
# gandalf: Qwen3.5-4B video + mix-v2-1 (see configs) -> answer SFT (novggt_qwen35_4b_reason190k_mixv2_1), 4 x RTX PRO 6000, global batch 256.
# Rerun the same command to resume from the newest checkpoint.
source "$(dirname "$(readlink -f "$0")")/../exp_paths.sh"
cd "$(dirname "$(readlink -f "$0")")/../.."   # repo root of the clone this launcher is in (deployed by exp.py deploy)
source site.env; [ -e runs ] || ln -s "$SR_OUTPUT_ROOT" runs   # runs/ -> SR_OUTPUT_ROOT (outputs stay in the old tree's runs/)
export SR_ROOT=$PWD SLURM_JOB_ID=gandalf SLURM_GPUS_ON_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3
OUT=${OUT:-runs/sr_vggt_novggt_qwen35_4b_reason190k_mixv2_1}
exp_paths INPUT=configs/novggt_qwen35_4b_reason190k_mixv2_1.env OUTPUT="$OUT" RESULT="$OUT/checkpoint-*" GPUS="$CUDA_VISIBLE_DEVICES"
bash scripts/jobs/train.slurm novggt_qwen35_4b_reason190k_mixv2_1 "$OUT" "$@"
