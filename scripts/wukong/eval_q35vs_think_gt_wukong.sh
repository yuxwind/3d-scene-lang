#!/bin/bash
# wukong: VSI-Bench (all 5,130 rows) eval of novggt_qwen35_4b_reason190k_vs_think with its training input: video +
# the question's ground-truth objects (training-frame rows, data/scenelang_vsibench_gt_firstcam/rows_all5130.jsonl,
# prompt rvs_question) + the CoT system prompt, thinking on, up to 8,192 new tokens. 12 shards over the 4 GPUs
# (one process uses about 11 GB and half of a GPU). Rerun to resume; merges and scores when all shards are done.
#   scripts/wukong/eval_q35vs_think_gt_wukong.sh [<checkpoint dir>]      default: <run>/checkpoint-569
source "$(dirname "$(readlink -f "$0")")/../exp_paths.sh"
cd "$(dirname "$(readlink -f "$0")")/../.."   # repo root of the clone this launcher is in (deployed by exp.py deploy)
export SR_ROOT=$PWD
source site.env
source scripts/lib.sh
export PATH=/scr/xinyu/envs/sr_vggt/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CKPT=${1:-${SR_OUTPUT_ROOT}/sr_vggt_novggt_qwen35_4b_reason190k_vs_think/checkpoint-569}
OUT=${OUT:-$(dirname "${CKPT}")/eval_$(basename "${CKPT}")_gt_firstcam_${P:=rvs_question}}
D=data/scenelang_vsibench_scannetpp
NUM_SHARDS=${NUM_SHARDS:-12}
exp_paths INPUT="${ROWS:-data/scenelang_vsibench_gt_firstcam/rows_all5130.jsonl}" INPUT="$CKPT" OUTPUT="$OUT" RESULT="$OUT/results*" GPUS=0,1,2,3
mkdir -p "${OUT}"
run() {  # extra args after the common ones
  python $D/eval_scenelang.py --rows "${ROWS:-data/scenelang_vsibench_gt_firstcam/rows_all5130.jsonl}" \
    --input video_scene --prompt "$P" --system reason_scene --thinking --max_new_tokens "${MAX_NEW_TOKENS:-8192}" \
    --vsibench_dir "${SR_VSIBENCH}" --out_dir "${OUT}" --num_shards "${NUM_SHARDS}" "$@"
}
for ((i = 0; i < NUM_SHARDS; i++)); do
  CUDA_VISIBLE_DEVICES=$((i % 4)) run --shard_index $i --adapter "${CKPT}" \
    --model "${SR_HF_HOME}/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a" \
    >> "${OUT}/log_shard$i.out" 2>&1 &
done
wait
run --merge 2>&1 | tee "${OUT}/merge.log"
