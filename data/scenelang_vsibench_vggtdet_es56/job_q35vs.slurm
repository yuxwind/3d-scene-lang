#!/bin/bash
#SBATCH --job-name=es56_q35vs
#SBATCH --account=bef@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_vggtdet_es56/log_%A_%a.out
# sr-q35vs (Qwen3.5-4B + LoRA trained on reason-190k video + GT scene, checkpoint-569) with the ES-56 VGG-T-Det boxes
# (build.py here) as the scene: 2,071 VSI-Bench ScanNet rows, rvs_question (boxes of the labels named in the question),
# video 1 fps <= 64 frames, 16 new tokens; 2 shards. Same settings as ../scenelang_vsibench_vggtdet/job_q35vs.slurm.
set -euo pipefail
source /lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/scripts/lib.sh
activate_env h100
offline
export PYTHONPATH=$PYTHONPATH:/lustre/fsn1/projects/rech/bef/uxt97ti/3d_scene_lang/pyextra/site
D=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_vggtdet_es56
I=$SLURM_ARRAY_TASK_ID
P=rvs_question
python $D/../scenelang_vsibench_scannetpp/eval_scenelang.py --rows $D/rows_first_camera.jsonl --input video_scene --prompt $P \
  --system short --max_new_tokens 16 --vsibench_dir "$SR_VSIBENCH" \
  --num_shards 2 --shard_index $((I % 2)) --out_dir $D/sr_q35vs_569_$P \
  --adapter $SR_OUTPUT_ROOT/sr_vggt_novggt_qwen35_4b_reason190k_vs/checkpoint-569 \
  --model $HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
