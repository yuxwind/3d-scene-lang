#!/bin/bash
#SBATCH --job-name=scenelang_unans
#SBATCH --account=bef@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_scannetpp/log_%A_%a.out
# The 85 debiased questions the scene-text run (1,024 tokens, thinking off) left without an answer, re-run on the
# fixed scene text with up to 4,096 new tokens. Tasks 0-3: thinking off, shards 0-3; tasks 4-7: thinking on, shards 0-3.
set -euo pipefail
source /lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/scripts/lib.sh
activate_env a100
offline
export PYTHONPATH=$PYTHONPATH:/lustre/fsn1/projects/rech/bef/uxt97ti/3d_scene_lang/pyextra/site
D=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_scannetpp
if (( SLURM_ARRAY_TASK_ID < 4 )); then OUT=qwen35_4b_scene_4096; TH=(); else OUT=qwen35_4b_scene_4096_thinking; TH=(--thinking); fi
python $D/eval_scenelang.py --rows $D/vsibench_scannetpp_scenelang.jsonl --indices $D/unanswered85_indices.txt \
  --input scene --max_new_tokens 4096 "${TH[@]}" --num_shards 4 --shard_index $((SLURM_ARRAY_TASK_ID % 4)) \
  --out_dir $D/$OUT --model $HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
