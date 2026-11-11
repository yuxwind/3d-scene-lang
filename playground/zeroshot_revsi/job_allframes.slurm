#!/bin/bash
#SBATCH --job-name=zs_revsi64_all
#SBATCH --account=bef@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=24 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/playground/zeroshot_revsi/log_%A_%a.out
# Zero-shot ReVSI 64_frame, ReVSI protocol (all 64 frames of each subset video): tasks 0-1 Qwen3-VL-2B-Instruct (2 shards), tasks 2-3 Qwen3.5-4B (2 shards).
set -euo pipefail
source /lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/scripts/lib.sh
activate_env h100
offline
export PYTHONPATH=$PYTHONPATH:/lustre/fsn1/projects/rech/bef/uxt97ti/3d_scene_lang/pyextra/site
P=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/playground/zeroshot_revsi
if (( SLURM_ARRAY_TASK_ID < 2 )); then
  NAME=qwen3vl_2b_allframes; M=$HF_HOME/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
else
  NAME=qwen35_4b_allframes; M=$HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
fi
nvidia-smi --query-gpu=name,memory.total --format=csv
python $P/eval_revsi.py --out_dir $P/$NAME --model $M --revsi_dir /lustre/fsn1/projects/rech/bef/uxt97ti/data/ReVSI \
  --all_frames --num_shards 2 --shard_index $((SLURM_ARRAY_TASK_ID % 2))
