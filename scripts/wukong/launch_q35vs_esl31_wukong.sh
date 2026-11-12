#!/bin/bash
# wukong: Qwen3.5-4B video + partial GT scene (31 detector classes) -> answer SFT (novggt_qwen35_4b_reason190k_vs_esl31),
# 4 x RTX 6000 Ada (46 GB), global batch 256. Rerun the same command to resume from the newest checkpoint.
# --checkpoint_above 4500: gradient checkpointing only for micro-batches over 4,500 tokens (same gradients). Measured
# on one card, per-device batch 1: no checkpointing peaks at 10.5 GiB + 5.9 GiB per 1k tokens (32 GiB at a typical
# 3.7k-token row, 37 GiB at 4.5k) and takes 1.22 s per typical row; checkpointing every layer peaks at 12 GiB and
# takes 1.65 s. 99.2% of the rows are under 4.5k tokens; the longest (about 9k) stay checkpointed.
# --dataloader_num_workers 16: with 8 workers per GPU the video decoding, not the GPU, set the pace.
cd /home/student/xinyu/projects/3d-scene-lang
export SR_ROOT=$PWD SLURM_JOB_ID=wukong SLURM_GPUS_ON_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3
source site.env
# No transparent huge pages for this process tree (PR_SET_THP_DISABLE = prctl 41, inherited; numpy asks for them for
# the frame arrays). wukong's free memory is fragmented: after a few minutes every loader worker spun in the kernel
# compacting memory (1.1 M failed page migrations per second, memory pressure 90%), steps went from 150 s to 565 s.
export NUMPY_MADVISE_HUGEPAGE=0
exec python3 -c 'import ctypes, os, sys; ctypes.CDLL(None).prctl(41, 1, 0, 0, 0); os.execvp(sys.argv[1], sys.argv[1:])' \
  bash scripts/jobs/train.slurm novggt_qwen35_4b_reason190k_vs_esl31 "${OUT:-${SR_OUTPUT_ROOT}/sr_vggt_novggt_qwen35_4b_reason190k_vs_esl31}" --checkpoint_above 4500 --dataloader_num_workers 16 "$@"
