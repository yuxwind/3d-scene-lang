#!/bin/bash
#SBATCH --job-name=scenelang_fullrest
#SBATCH --account=bef@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_scannetpp/log_%A_%a.out
# Scene text only with --prompt full on the remaining debiased tasks (counting, size, relative distance, route,
# appearance order: 502 rows), thinking off, up to 16,384 new tokens, 10 shards.
set -euo pipefail
source /lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/scripts/lib.sh
activate_env a100
offline
export PYTHONPATH=$PYTHONPATH:/lustre/fsn1/projects/rech/bef/uxt97ti/3d_scene_lang/pyextra/site
D=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_scannetpp
python $D/eval_scenelang.py --rows $D/vsibench_scannetpp_scenelang.jsonl --input scene --prompt full --debiased_only \
  --types object_counting,object_size_estimation,object_rel_distance,route_planning,obj_appearance_order --max_new_tokens 16384 \
  --num_shards 10 --shard_index $SLURM_ARRAY_TASK_ID --out_dir $D/qwen35_4b_scene_fullprompt_rest \
  --model $HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
