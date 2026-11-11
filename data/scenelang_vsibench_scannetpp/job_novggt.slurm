#!/bin/bash
#SBATCH --job-name=scenelang_novggt
#SBATCH --account=bef@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_scannetpp/log_%A_%a.out
# sr_vggt_novggt (Qwen3-VL-2B + its checkpoint-743 LoRA) on the 752 debiased rows, scene text only, with the
# training system prompt. Tasks 0-3: basic prompt; 4-7: full prompt. 4 shards each, thinking off, 16,384 tokens (MAXTOK=4096 for the full-prompt resume: outputs past 4,096 were all loops).
set -euo pipefail
source /lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/scripts/lib.sh
activate_env a100
offline
export PYTHONPATH=$PYTHONPATH:/lustre/fsn1/projects/rech/bef/uxt97ti/3d_scene_lang/pyextra/site
D=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_scannetpp
P=(basic full); I=$SLURM_ARRAY_TASK_ID; PR=${P[$((I / 4))]}
python $D/eval_scenelang.py --rows $D/vsibench_scannetpp_scenelang.jsonl --input scene --prompt $PR --debiased_only \
  --system short --max_new_tokens ${MAXTOK:-16384} --num_shards 4 --shard_index $((I % 4)) --out_dir $D/novggt_2b_scene_$PR \
  --model $HF_HOME/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203 \
  --adapter /lustre/fsn1/projects/rech/bef/uxt97ti/3d_scene_lang/results/sr_vggt_novggt/checkpoint-743
