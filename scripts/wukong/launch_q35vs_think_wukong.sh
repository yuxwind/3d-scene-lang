#!/bin/bash
# wukong: Qwen3.5-4B video + GT scene -> CoT + answer SFT (novggt_qwen35_4b_reason190k_vs_think),
# 4 x RTX 6000 Ada (46 GB), global batch 256. Rerun the same command to resume from the newest checkpoint.
# --checkpoint_tiers 4700:all,8500:16,14500:8: the config's partial gradient checkpointing, widened for 46 GB (about
# 37-38 GiB peak at the top of each tier; measurements in configs/novggt_qwen35_4b_reason190k_vs_think.env).
# --dataloader_num_workers 16: with 8 workers per GPU the video decoding, not the GPU, set the pace.
cd /home/student/xinyu/projects/3d-scene-lang
export SR_ROOT=$PWD SLURM_JOB_ID=wukong SLURM_GPUS_ON_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3
source site.env
# No transparent huge pages for this process tree (PR_SET_THP_DISABLE = prctl 41, inherited; numpy asks for them for
# the frame arrays). wukong's free memory is fragmented: after a few minutes every loader worker spun in the kernel
# compacting memory (1.1 M failed page migrations per second, memory pressure 90%), steps went from 150 s to 565 s.
export NUMPY_MADVISE_HUGEPAGE=0
exec python3 -c 'import ctypes, os, sys; ctypes.CDLL(None).prctl(41, 1, 0, 0, 0); os.execvp(sys.argv[1], sys.argv[1:])' \
  bash scripts/jobs/train.slurm novggt_qwen35_4b_reason190k_vs_think "${OUT:-${SR_OUTPUT_ROOT}/sr_vggt_novggt_qwen35_4b_reason190k_vs_think}" --checkpoint_tiers 4700:all,8500:16,14500:8 --dataloader_num_workers 16 "$@"
