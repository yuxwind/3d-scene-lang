#!/bin/bash
#SBATCH --job-name=scenelang_spp_q35_4b
#SBATCH --account=bef@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/lustre/fsn1/projects/rech/bef/uxt97ti/3d_scene_lang/scenelang_vsibench_scannetpp/log_%A_%a.out
# task 0: scene language only; task 1: video + scene language. All 1,458 ScanNet++ rows (debiased 752 inside).
set -euo pipefail
source /lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/scripts/lib.sh
activate_env a100
offline
export PYTHONPATH=$PYTHONPATH:/lustre/fsn1/projects/rech/bef/uxt97ti/3d_scene_lang/pyextra/site
D=/lustre/fsn1/projects/rech/bef/uxt97ti/3d_scene_lang/scenelang_vsibench_scannetpp
IN=(scene video_scene); IN=${IN[$SLURM_ARRAY_TASK_ID]}
nvidia-smi --query-gpu=name,memory.total --format=csv
python $D/eval_scenelang.py --rows $D/vsibench_scannetpp_scenelang.jsonl --input $IN --out_dir $D/qwen35_4b_$IN \
  --model $HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --vsibench_dir "$SR_VSIBENCH"
