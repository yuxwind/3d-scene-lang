#!/bin/bash
# wukong: Exp1 sft-qwen-test2-hybrid (configs/novggt_qwen35_4b_reason190k_mixv2_1_objq.env),
# 4 x RTX 6000 Ada (46 GB), global batch 256. Rerun the same command to resume from the newest checkpoint.
# --checkpoint_above 4500: gradient checkpointing only for micro-batches over 4,500 tokens (same gradients). Measured
# on one card, per-device batch 1: no checkpointing peaks at 10.5 GiB + 5.9 GiB per 1k tokens (32 GiB at a typical
# 3.7k-token row, 37 GiB at 4.5k) and takes 1.22 s per typical row; checkpointing every layer peaks at 12 GiB and
# takes 1.65 s. 99.2% of the rows are under 4.5k tokens; the longest (about 9k) stay checkpointed.
# --dataloader_num_workers 16: with 8 workers per GPU the video decoding, not the GPU, set the pace.
cd /home/student/xinyu/projects/3d-scene-lang
# Wait until wukong GPUs 0-3 are free of other jobs (each under 6 GiB used, 10 checks in a row 60 s apart): another user's RL job
# moves between 13 and 44 GiB per GPU, so starting beside it would make one of the two jobs run out of memory.
ok=0; while [ $ok -lt 10 ]; do
  M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3 | sort -n | tail -1)
  if [ "$M" -lt 6144 ]; then ok=$((ok+1)); else ok=0; echo "$(date "+%m-%d %H:%M") waiting: a GPU holds ${M} MiB"; fi
  sleep 60
done
echo "$(date "+%m-%d %H:%M") GPUs free; starting Exp1"; CKPT_ABOVE=4500
export SR_ROOT=$PWD SLURM_JOB_ID=wukong SLURM_GPUS_ON_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3
source site.env
# No transparent huge pages for this process tree (PR_SET_THP_DISABLE = prctl 41, inherited; numpy asks for them for
# the frame arrays). wukong's free memory is fragmented: after a few minutes every loader worker spun in the kernel
# compacting memory (1.1 M failed page migrations per second, memory pressure 90%), steps went from 150 s to 565 s.
export NUMPY_MADVISE_HUGEPAGE=0
exec python3 -c 'import ctypes, os, sys; ctypes.CDLL(None).prctl(41, 1, 0, 0, 0); os.execvp(sys.argv[1], sys.argv[1:])' \
  bash scripts/jobs/train.slurm novggt_qwen35_4b_reason190k_mixv2_1_objq "${OUT:-${SR_OUTPUT_ROOT}/sr_vggt_novggt_qwen35_4b_reason190k_mixv2_1_objq}" --checkpoint_above ${CKPT_ABOVE:-4500} --dataloader_num_workers 16 "$@"
