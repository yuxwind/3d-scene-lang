#!/bin/bash
# VSI-Bench (all 5,130 rows) eval of the video + GT-scene adapter, training-format prompt (rvs_question), 4 shards on GPUs 0-3.
source "$(dirname "$(readlink -f "$0")")/../exp_paths.sh"
cd "$(dirname "$(readlink -f "$0")")/../.."   # repo root of the clone this launcher is in (deployed by exp.py deploy)
source site.env; [ -e runs ] || ln -s "$SR_OUTPUT_ROOT" runs   # runs/ -> SR_OUTPUT_ROOT (outputs stay in the old tree's runs/)
export SR_ROOT=$PWD
source scripts/lib.sh
set +u; export PATH=/scr/xinyu/envs/sr_vggt/bin:$PATH HF_HOME=/scr/xinyu/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
D=data/scenelang_vsibench_scannetpp; P=${P:-rvs_question}
OUT=${OUT:-runs/eval_q35vs_gandalf/${NAME:-gt_firstcam}_${P}}
exp_paths INPUT="${ROWS:-data/scenelang_vsibench_gt_firstcam/rows_all5130.jsonl}" INPUT="${ADAPTER:-runs/sr_vggt_novggt_qwen35_4b_reason190k_vs/checkpoint-569}" OUTPUT="$OUT" RESULT="$OUT/results*.json" GPUS=0,1,2,3
mkdir -p $OUT
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i setsid nohup python $D/eval_scenelang.py --rows ${ROWS:-data/scenelang_vsibench_gt_firstcam/rows_all5130.jsonl} \
    --input video_scene --prompt $P ${SCORE:+--scene_score} ${V2:+--prompt_v2} --system short --max_new_tokens 16 --vsibench_dir /scr/xinyu/data/VSI-Bench \
    --num_shards 4 --shard_index $i --out_dir $OUT \
    --adapter ${ADAPTER:-runs/sr_vggt_novggt_qwen35_4b_reason190k_vs/checkpoint-569} \
    --model /scr/xinyu/hf_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
    >> $OUT/log_shard$i.out 2>&1 &
done
