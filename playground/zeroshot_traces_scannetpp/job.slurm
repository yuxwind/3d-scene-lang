#!/bin/bash
#SBATCH --job-name=zs_traces_spp_q35_4b
#SBATCH --account=bef@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=24 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=/lustre/fsn1/projects/rech/bef/uxt97ti/3d_scene_lang/zeroshot_traces_scannetpp/log_%A_%a.out
set -euo pipefail
source /lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/scripts/lib.sh
activate_env h100
offline
export PYTHONPATH=$PYTHONPATH:/lustre/fsn1/projects/rech/bef/uxt97ti/3d_scene_lang/pyextra/site
D=/lustre/fsn1/projects/rech/bef/uxt97ti/3d_scene_lang/zeroshot_traces_scannetpp
python $D/eval_traces.py --out_dir $D/qwen35_4b \
  --model $HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --traces "$SR_ROOT/data/simplesr_train_traces.jsonl.gz" --video_root "$SR_VSI590K" \
  --num_shards 8 --shard_index $SLURM_ARRAY_TASK_ID
