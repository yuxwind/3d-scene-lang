#!/bin/bash
#SBATCH --job-name=scenelang_2b_zs
#SBATCH --account=bef@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --array=0-6
#SBATCH --output=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_scannetpp/log_%A_%a.out
# Zero-shot Qwen3-VL-2B-Instruct (no adapter, no system prompt) on the 752 debiased rows, as the Qwen3.5-4B runs:
#   task 0    scene, basic prompt (1 shard)          up to 4,096 tokens
#   tasks 1-4 scene, full prompt (4 shards)          up to 4,096 tokens (2B outputs past 4,096 were all loops)
#   tasks 5-6 video + scene, basic prompt (2 shards) 16 tokens, as qwen35_4b_video_scene
set -euo pipefail
source /lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/scripts/lib.sh
activate_env a100
offline
export PYTHONPATH=$PYTHONPATH:/lustre/fsn1/projects/rech/bef/uxt97ti/3d_scene_lang/pyextra/site
D=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_scannetpp
I=$SLURM_ARRAY_TASK_ID
if   [ $I -eq 0 ]; then A=(--input scene --prompt basic --max_new_tokens 4096 --num_shards 1 --shard_index 0 --out_dir $D/qwen3vl_2b_scene_basic)
elif [ $I -le 4 ]; then A=(--input scene --prompt full --max_new_tokens 4096 --num_shards 4 --shard_index $((I - 1)) --out_dir $D/qwen3vl_2b_scene_full)
else A=(--input video_scene --prompt basic --max_new_tokens 16 --num_shards 2 --shard_index $((I - 5)) --out_dir $D/qwen3vl_2b_video_scene --vsibench_dir "$SR_VSIBENCH")
fi
python $D/eval_scenelang.py --rows $D/vsibench_scannetpp_scenelang.jsonl --debiased_only "${A[@]}" \
  --model $HF_HOME/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
