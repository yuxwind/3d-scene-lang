#!/bin/bash
#SBATCH --job-name=scenelang_rvs
#SBATCH --account=bef@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_scannetpp/log_%A_%a.out
# sr-rvs (Qwen3-VL-2B + LoRA, trained on reason-190k as video + scene -> answer, checkpoint-569) on the 752
# debiased rows, video (1 fps, <= 64 frames) + scene text in its training format (sr_vggt system prompt,
# "This is the 3D scene context." + video + human turn), 16 new tokens (the target is the bare answer):
#   tasks 0-1 rvs_question: the objects named in the question (all objects for room size)
#   tasks 2-3 rvs_all:      every object in the scene
set -euo pipefail
source /lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/scripts/lib.sh
activate_env a100
offline
export PYTHONPATH=$PYTHONPATH:/lustre/fsn1/projects/rech/bef/uxt97ti/3d_scene_lang/pyextra/site
D=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_scannetpp
I=$SLURM_ARRAY_TASK_ID
P=(rvs_question rvs_question rvs_all rvs_all); P=${P[$I]}
python $D/eval_scenelang.py --rows $D/vsibench_scannetpp_scenelang.jsonl --input video_scene --prompt $P --debiased_only \
  --system short --max_new_tokens 16 --vsibench_dir "$SR_VSIBENCH" \
  --num_shards 2 --shard_index $((I % 2)) --out_dir $D/sr_rvs_569_$P \
  --adapter $SR_OUTPUT_ROOT/sr_vggt_novggt_reason190k_vs/checkpoint-569 \
  --model $HF_HOME/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
