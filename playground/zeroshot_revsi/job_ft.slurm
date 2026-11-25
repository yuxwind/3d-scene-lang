#!/bin/bash
#SBATCH --job-name=ft_revsi64
#SBATCH --account=bef@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=24 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/playground/zeroshot_revsi/log_%A_%a.out
# Fine-tuned Qwen3.5-4B on ReVSI 64_frame (all 64 frames of each subset video), each in its training format, 2 shards:
#   tasks 0-1 video -> answer:                    novggt_qwen35_4b checkpoint-743               -> ft_q35_video_743
#   tasks 2-3 video + ground-truth scene -> answer: novggt_qwen35_4b_reason190k_vs checkpoint-569 -> ft_q35vs_video_gtscene_569
# Resubmit to resume (rows on disk are skipped); then `eval_revsi.py --out_dir <dir> --revsi_dir R --merge`.
set -euo pipefail
source /lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/scripts/lib.sh
activate_env h100
offline
export PYTHONPATH=$PYTHONPATH:/lustre/fsn1/projects/rech/bef/uxt97ti/3d_scene_lang/pyextra/site
P=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/playground/zeroshot_revsi
if (( SLURM_ARRAY_TASK_ID < 2 )); then
  NAME=ft_q35_video_743; MODE=ft_video; A=$SR_OUTPUT_ROOT/sr_vggt_novggt_qwen35_4b/checkpoint-743
else
  NAME=ft_q35vs_video_gtscene_569; MODE=ft_video_scene; A=$SR_OUTPUT_ROOT/sr_vggt_novggt_qwen35_4b_reason190k_vs/checkpoint-569
fi
nvidia-smi --query-gpu=name,memory.total --format=csv
python $P/eval_revsi.py --out_dir $P/$NAME --mode $MODE --adapter $A \
  --model $HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --revsi_dir /lustre/fswork/projects/rech/bef/uxt97ti/data/ReVSI --all_frames --num_shards 2 --shard_index $((SLURM_ARRAY_TASK_ID % 2))
