#!/bin/bash
#SBATCH --job-name=scenelang_cut16k
#SBATCH --account=bef@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_scannetpp/log_%A_%a.out
# The 137 debiased scene-text rows still cut off: 113 that stopped at 1,024 tokens (cap_check.py) and 24 that stopped
# at 4,096 (qwen35_4b_scene_4096). Fixed scene text, thinking off, up to 16,384 new tokens, 8 shards.
set -euo pipefail
source /lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/scripts/lib.sh
activate_env a100
offline
export PYTHONPATH=$PYTHONPATH:/lustre/fsn1/projects/rech/bef/uxt97ti/3d_scene_lang/pyextra/site
D=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_scannetpp
python $D/eval_scenelang.py --rows $D/vsibench_scannetpp_scenelang.jsonl --indices $D/cutoff137_indices.txt \
  --input scene --max_new_tokens 16384 --num_shards 8 --shard_index $SLURM_ARRAY_TASK_ID \
  --out_dir $D/qwen35_4b_scene_16384 --model $HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
