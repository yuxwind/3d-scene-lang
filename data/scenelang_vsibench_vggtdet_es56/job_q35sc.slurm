#!/bin/bash
#SBATCH --job-name=q35sc_det
#SBATCH --account=bef@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_vggtdet_es56/log_q35sc_%A_%a.out
# sr-q35sc (Qwen3.5-4B + LoRA trained on reason-190k with the GT scene alone, checkpoint-569), scene input only (no video),
# its training system prompt (--system scene), rvs_question, 16 new tokens, on the 2,071 VSI-Bench ScanNet rows:
#   task 0: the original 18-class VGG-T-Det scene (rows_det18_scannet.jsonl = ScanNet rows of ../scenelang_vsibench_vggtdet)
#   task 1: the ES-56 (cls56) VGG-T-Det scene (rows_first_camera.jsonl)
# GT-scene counterpart: ../scenelang_vsibench_gt_firstcam/sr_q35sc_569_scene_rvs_question
set -euo pipefail
source /lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/scripts/lib.sh
activate_env h100
offline
export PYTHONPATH=$PYTHONPATH:/lustre/fsn1/projects/rech/bef/uxt97ti/3d_scene_lang/pyextra/site
D=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_vggtdet_es56
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then ROWS=$D/rows_det18_scannet.jsonl; OUT=$D/sr_q35sc_569_det18_rvs_question
else ROWS=$D/rows_first_camera.jsonl; OUT=$D/sr_q35sc_569_es56_rvs_question; fi
python $D/../scenelang_vsibench_scannetpp/eval_scenelang.py --rows $ROWS --input scene --prompt rvs_question \
  --system scene --max_new_tokens 16 --out_dir $OUT \
  --adapter $SR_OUTPUT_ROOT/sr_vggt_novggt_qwen35_4b_reason190k_scene/checkpoint-569 \
  --model $HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
