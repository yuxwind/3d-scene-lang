#!/bin/bash
#SBATCH --job-name=scenelang_4b_zs
#SBATCH --account=bef@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_scannetpp/log_%A_%a.out
# Zero-shot Qwen3-VL-4B-Instruct (no adapter, no system prompt), scene text only, the 752 debiased rows,
# up to 16,384 new tokens (as the Qwen3.5-4B runs):
#   task 0    basic prompt   (1 shard)
#   tasks 1-3 full prompt    (3 shards)
#   tasks 4-6 simple1 prompt (3 shards)
#   tasks 7-9 simple2 prompt (3 shards)
# Rows already in an output file are skipped, so a timed-out task is finished by resubmitting.
set -euo pipefail
source /lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/scripts/lib.sh
activate_env a100
offline
export PYTHONPATH=$PYTHONPATH:/lustre/fsn1/projects/rech/bef/uxt97ti/3d_scene_lang/pyextra/site
D=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_scannetpp
I=$SLURM_ARRAY_TASK_ID
if   [ $I -eq 0 ]; then P=basic;   S=(--num_shards 1 --shard_index 0)
elif [ $I -le 3 ]; then P=full;    S=(--num_shards 3 --shard_index $((I - 1)))
elif [ $I -le 6 ]; then P=simple1; S=(--num_shards 3 --shard_index $((I - 4)))
else                    P=simple2; S=(--num_shards 3 --shard_index $((I - 7)))
fi
python $D/eval_scenelang.py --rows $D/vsibench_scannetpp_scenelang.jsonl --input scene --prompt $P --debiased_only \
  --max_new_tokens 16384 "${S[@]}" --out_dir $D/qwen3vl_4b_scene_$P \
  --model $HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
