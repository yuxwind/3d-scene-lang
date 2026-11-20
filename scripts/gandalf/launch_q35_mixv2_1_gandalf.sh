#!/bin/bash
# gandalf: Qwen3.5-4B video + mix-v2-1 (see configs) -> answer SFT (novggt_qwen35_4b_reason190k_mixv2_1), 4 x RTX PRO 6000, global batch 256.
# Rerun the same command to resume from the newest checkpoint.
cd /scr/xinyu/projects/3d-scene-lang
export SR_ROOT=$PWD SLURM_JOB_ID=gandalf SLURM_GPUS_ON_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/jobs/train.slurm novggt_qwen35_4b_reason190k_mixv2_1 runs/sr_vggt_novggt_qwen35_4b_reason190k_mixv2_1 "$@"
