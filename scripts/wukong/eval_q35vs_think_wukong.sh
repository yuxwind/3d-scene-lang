#!/bin/bash
# wukong: VSI-Bench full-split eval (5,130 rows) of novggt_qwen35_4b_reason190k_vs_think, 12 shards over the 4 GPUs
# (RTX 6000 Ada), bf16, thinking on, up to 8,192 new tokens. VIDEO ONLY: sr_vggt.eval feeds the frames and the
# question, without the <scene> block the model was trained with (the VSI-Bench ground-truth scene rows,
# data/scenelang_vsibench_gt_firstcam/rows_all5130.jsonl, are not on wukong). Rerun to resume; merges when all
# shards are done.
#   scripts/wukong/eval_q35vs_think_wukong.sh [<checkpoint dir>]      default: <run>/checkpoint-569
cd /home/student/xinyu/projects/3d-scene-lang
export SR_ROOT=$PWD
source site.env
source scripts/lib.sh
load_config novggt_qwen35_4b_reason190k_vs_think
export PATH=/scr/xinyu/envs/sr_vggt/bin:$PATH HF_HUB_OFFLINE=1 PYTHONUTF8=1
CKPT=${1:-${SR_OUTPUT_ROOT}/sr_vggt_novggt_qwen35_4b_reason190k_vs_think/checkpoint-569}
EVAL_DIR=${EVAL_DIR:-$(dirname "${CKPT}")/eval_$(basename "${CKPT}")_full_noscene}
mkdir -p "${EVAL_DIR}"
NUM_SHARDS=${NUM_SHARDS:-12}   # 3 per GPU: one process uses about 11 GB and half of a GPU
for ((i = 0; i < NUM_SHARDS; i++)); do
  CUDA_VISIBLE_DEVICES=$((i % 4)) python -m sr_vggt.eval "${MODEL_ARGS[@]}" "${VGGT_ARGS[@]}" "${EVAL_ARGS[@]}" \
    --adapter_path "${CKPT}" --output_dir "${EVAL_DIR}" --vsibench_dir "${SR_VSIBENCH}" \
    --num_shards "${NUM_SHARDS}" --shard_index $i --dtype bf16 > "${EVAL_DIR}/shard$i.log" 2>&1 &
done
wait
python -m sr_vggt.eval merge "${EVAL_DIR}" 2>&1 | tee "${EVAL_DIR}/merge.log"
