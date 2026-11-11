#!/bin/bash
#SBATCH --job-name=scenelang_cornpts
#SBATCH --account=bef@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_scannetpp/log_%A_%a.out
# Scene text only, simple2 prompt with the boxes rewritten, Qwen3.5-4B zero-shot, thinking off, up to 16,384 new
# tokens, the 752 debiased rows:
#   tasks 0-4 --prompt corners: {label, center, corners}: the 8 vertices of each 9-DoF box (5 shards)
#   tasks 5-9 --prompt points:  {label, center, points}: 8 random points on each box's surface (5 shards)
# Rows already in an output file are skipped, so a timed-out task is finished by resubmitting.
set -euo pipefail
source /lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/scripts/lib.sh
activate_env a100
offline
export PYTHONPATH=$PYTHONPATH:/lustre/fsn1/projects/rech/bef/uxt97ti/3d_scene_lang/pyextra/site
D=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_scannetpp
I=$SLURM_ARRAY_TASK_ID
P=$([ $I -lt 5 ] && echo corners || echo points)
python $D/eval_scenelang.py --rows $D/vsibench_scannetpp_scenelang.jsonl --input scene --prompt $P --debiased_only \
  --max_new_tokens 16384 \
  --num_shards 5 --shard_index $((I % 5)) --out_dir $D/qwen35_4b_scene_${P}prompt \
  --model $HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
