#!/bin/bash
#SBATCH --job-name=mapdet3d_vsi
#SBATCH --account=bef@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_mapdet3d/log_%A_%a.out
# Class-agnostic Map-Det3D on the VSI-Bench scenes of one dataset, every 5th posed frame, one shard per array task.
# DATASET=scannet (88 scenes), arkitscenes (147) or scannetpp (49), e.g. sbatch --export=ALL,DATASET=arkitscenes --array=0-5 job.slurm
# Scenes already in preds/ are skipped, so a timed-out shard is finished by resubmitting.
# Extra args go to run.py, e.g. sbatch --array=0 job.slurm --scenes scene0025_01 (smoke test).
DATASET=${DATASET:-scannet}
set -euo pipefail
source /lustre/fswork/projects/rech/bef/uxt97ti/miniforge3/etc/profile.d/conda.sh
conda activate mapdet3d
export HF_HOME=/lustre/fswork/projects/rech/bef/uxt97ti/hf_cache
export TORCH_HOME=$HF_HOME/torch
export HF_HUB_OFFLINE=1
D=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_mapdet3d
nvidia-smi -L
python $D/run.py $D --dataset $DATASET --stride 5 --shard $SLURM_ARRAY_TASK_ID --nshards ${SLURM_ARRAY_TASK_COUNT} "$@"
