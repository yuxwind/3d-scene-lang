#!/bin/bash
#SBATCH --job-name=scenelang_q35vs_sa
#SBATCH --account=bef@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_scannetpp/log_%A_%a.out
# sr-q35vs (as job_q35vs.slurm) on the ScanNet and ARKitScenes rows of VSI-Bench, all rows (the merged results
# report the debiased and full splits), scene language from vsibench_<dataset>_scenelang.jsonl (build.py ... <dataset>):
#   tasks 0-3 scannet, 4-7 arkitscenes; within each: 2 shards of rvs_question, then 2 of rvs_all
# Rows already in an output file are skipped, so a timed-out task is finished by resubmitting.
set -euo pipefail
source /lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/scripts/lib.sh
activate_env a100
offline
export PYTHONPATH=$PYTHONPATH:/lustre/fsn1/projects/rech/bef/uxt97ti/3d_scene_lang/pyextra/site
D=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_scannetpp
I=$SLURM_ARRAY_TASK_ID
DS=$([ $I -lt 4 ] && echo scannet || echo arkitscenes)
P=$([ $((I % 4)) -lt 2 ] && echo rvs_question || echo rvs_all)
python $D/eval_scenelang.py --rows $D/vsibench_${DS}_scenelang.jsonl --input video_scene --prompt $P \
  --system short --max_new_tokens 16 --vsibench_dir "$SR_VSIBENCH" \
  --num_shards 2 --shard_index $((I % 2)) --out_dir $D/sr_q35vs_569_${P}_${DS} \
  --adapter $SR_OUTPUT_ROOT/sr_vggt_novggt_qwen35_4b_reason190k_vs/checkpoint-569 \
  --model $HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
