#!/bin/bash
#SBATCH --job-name=scenelang_coatrack
#SBATCH --account=bef@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread
#SBATCH --time=00:40:00
#SBATCH --output=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_scannetpp/log_%j.out
# Re-run the 13 debiased coat-rack questions on the fixed scene text (coat hanger -> coat rack), same settings as
# the original runs: scene text up to 1,024 new tokens (as qwen35_4b_scene_long), video + scene 16 tokens (as qwen35_4b_video_scene).
set -euo pipefail
source /lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/scripts/lib.sh
activate_env a100
offline
export PYTHONPATH=$PYTHONPATH:/lustre/fsn1/projects/rech/bef/uxt97ti/3d_scene_lang/pyextra/site
D=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_scannetpp
M=$HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
python $D/eval_scenelang.py --rows $D/vsibench_scannetpp_scenelang.jsonl --indices $D/coat_rack_debiased_indices.txt \
  --input scene --max_new_tokens 1024 --out_dir $D/qwen35_4b_scene_long_coatrack --model $M --vsibench_dir "$SR_VSIBENCH"
python $D/eval_scenelang.py --rows $D/vsibench_scannetpp_scenelang.jsonl --indices $D/coat_rack_debiased_indices.txt \
  --input video_scene --out_dir $D/qwen35_4b_video_scene_coatrack --model $M --vsibench_dir "$SR_VSIBENCH"
