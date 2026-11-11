#!/bin/bash
# Keeps three sets of job_zs_simple2.slurm shards running on the dev QoS (2 h jobs, <= 10 per user per QoS) until
# each shard has all its rows. Every 5 min, in priority order, each unfinished shard with no job queued gets one on
# whichever of h100-dev / a100-dev has a free slot. Rows already written are skipped, so a timed-out shard resumes.
#   zw_7   : world-frame debiased ScanNet++ run (scenelang_vsibench_scannetpp/qwen35_4b_scene_simple2prompt),
#            shard 7 of 10, its last 8 rows (646, 757, 785-790)
#   zv_<k> : debiased ScanNet++ (752) in the training frame, 8 shards -> qwen35_4b_scene_simple2prompt_debiased_scannetpp
#   zvd_<k>: exp 3, video + VGGT-Det boxes (seed 0, scenelang_vsibench_vggtdet/build.py --keep-yaw, first_camera,
#            label numbers stripped, roll = pitch = 0) on its 2,969 rows, 16 shards -> qwen35_4b_video_vggtdet_simple2prompt
#   zft_<k>: exp 4, fine-tuned sr-q35vs (novggt_qwen35_4b_reason190k_vs checkpoint-569) with video + GT scene (training
#            frame) and the simple2 prompt instead of its training prompt, all 5,130 rows, 20 shards
#            -> sr_q35vs_569_video_scene_simple2prompt
#   zmd_<k>: exp 6, zero-shot video + MAP-Det3D boxes with oracle GT labels (scenelang_vsibench_mapdet3d/
#            vsibench_meta_info_val_mapdet3d_oracle0.1.json, build.py <bench> <dir> <that file> mapdet3d_oracle0.1),
#            5,094 rows (scenes without MAP-Det3D output skipped), 20 shards -> qwen35_4b_video_mapdet3d_simple2prompt
#   zrq_<k>, zra_<k>: exp 7, fine-tuned sr-q35vs with video + the exp-6 MAP-Det3D scenes in its training format
#            (--prompt rvs_question / rvs_all, up to 16,384 new tokens), 6 shards each
#            -> sr_q35vs_569_mapdet3d_rvs_question, sr_q35vs_569_mapdet3d_rvs_all
#   zgq_<k>: exp 4 again in the training format: fine-tuned sr-q35vs, video + GT scene (training frame), --prompt
#            rvs_question (up to 16,384 new tokens), all 5,130 rows, 6 shards -> sr_q35vs_569_video_scene_rvs_question
#   zgw_<k>: zgq with the scenes in VSI-Bench's world frame (scenelang_vsibench_scannetpp/vsibench_*_scenelang.jsonl ->
#            rows_all5130_world.jsonl), 6 shards -> sr_q35vs_569_video_scene_world_rvs_question
#   zs2_<k>: the other 4,378 rows in the training frame, 40 shards -> qwen35_4b_scene_simple2prompt
#   zvs_<k>: exp 2, the same 4,378 rows with the video (--input video_scene), 20 shards -> qwen35_4b_video_scene_simple2prompt
# Log: zs_simple2_driver.log.
G=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_gt_firstcam
P=$G/../scenelang_vsibench_scannetpp
mapfile -t S_REST < $G/zs_simple2_shard_sizes.txt
mapfile -t S_VER < $G/zv_shard_sizes.txt
mapfile -t S_VS < $G/zvs_shard_sizes.txt
mapfile -t S_VD < $G/zvd_shard_sizes.txt
mapfile -t S_FT < $G/zft_shard_sizes.txt
mapfile -t S_MD < $G/zmd_shard_sizes.txt
mapfile -t S_ZR < $G/zr_shard_sizes.txt
mapfile -t S_GQ < $G/zgq_shard_sizes.txt
AD=/lustre/fsn1/projects/rech/bef/uxt97ti/3d_scene_lang/results/sr_vggt_novggt_qwen35_4b_reason190k_vs/checkpoint-569
TASKS=("zw 7 10 75 scene $P/vsibench_scannetpp_scenelang.jsonl $P/qwen35_4b_scene_simple2prompt --debiased_only")
for k in $(seq 0 7); do TASKS+=("zv $k 8 ${S_VER[$k]} scene $G/rows_zs_simple2_debiased_scannetpp.jsonl $G/qwen35_4b_scene_simple2prompt_debiased_scannetpp"); done
for k in $(seq 0 19); do TASKS+=("zvs $k 20 ${S_VS[$k]} video_scene $G/rows_zs_simple2_rest.jsonl $G/qwen35_4b_video_scene_simple2prompt"); done
for k in $(seq 0 15); do TASKS+=("zvd $k 16 ${S_VD[$k]} video_scene $G/rows_zs_vggtdet_simple2.jsonl $G/qwen35_4b_video_vggtdet_simple2prompt"); done
for k in $(seq 0 19); do TASKS+=("zft $k 20 ${S_FT[$k]} video_scene $G/rows_all5130.jsonl $G/sr_q35vs_569_video_scene_simple2prompt"); done
for k in $(seq 0 19); do TASKS+=("zmd $k 20 ${S_MD[$k]} video_scene $G/rows_zs_mapdet3d_simple2.jsonl $G/qwen35_4b_video_mapdet3d_simple2prompt"); done
for k in $(seq 0 5); do TASKS+=("zrq $k 6 ${S_ZR[$k]} video_scene $G/rows_zs_mapdet3d_simple2.jsonl $G/sr_q35vs_569_mapdet3d_rvs_question"); done
for k in $(seq 0 5); do TASKS+=("zra $k 6 ${S_ZR[$k]} video_scene $G/rows_zs_mapdet3d_simple2.jsonl $G/sr_q35vs_569_mapdet3d_rvs_all"); done
for k in $(seq 0 5); do TASKS+=("zgq $k 6 ${S_GQ[$k]} video_scene $G/rows_all5130.jsonl $G/sr_q35vs_569_video_scene_rvs_question"); done
for k in $(seq 0 5); do TASKS+=("zgw $k 6 ${S_GQ[$k]} video_scene $G/rows_all5130_world.jsonl $G/sr_q35vs_569_video_scene_world_rvs_question"); done
for k in $(seq 0 39); do TASKS+=("zs2 $k 40 ${S_REST[$k]} scene $G/rows_zs_simple2_rest.jsonl $G/qwen35_4b_scene_simple2prompt"); done
mkdir -p $G/logs_zs_simple2 $G/qwen35_4b_scene_simple2prompt_debiased_scannetpp $G/qwen35_4b_video_scene_simple2prompt $G/qwen35_4b_video_vggtdet_simple2prompt $G/sr_q35vs_569_video_scene_simple2prompt $G/qwen35_4b_video_mapdet3d_simple2prompt $G/sr_q35vs_569_mapdet3d_rvs_question $G/sr_q35vs_569_mapdet3d_rvs_all $G/sr_q35vs_569_video_scene_rvs_question $G/sr_q35vs_569_video_scene_world_rvs_question
while true; do
  q=$(squeue -u $USER -h -o "%j %q")
  left=0
  for t in "${TASKS[@]}"; do
    read -r name k nsh size input rows out extra <<< "$t"
    f=$out/predictions_shard${k}of${nsh}.jsonl
    n=$(cat $f 2>/dev/null | grep -c .)
    [ "$n" -ge "$size" ] && continue
    left=$((left+1))
    echo "$q" | grep -q "^${name}_${k} " && continue
    case $name in zft|zrq|zra|zgq|zgw) ad=$AD ;; *) ad= ;; esac
    case $name in zrq|zgq|zgw) pr=rvs_question; mt=16384 ;; zra) pr=rvs_all; mt=16384 ;; *) pr=simple2; mt=16384 ;; esac
    for gpu in h100 a100; do
      used=$(echo "$q" | grep -c "qos_gpu_${gpu}-dev")
      [ "$used" -ge 10 ] && continue
      id=$(sbatch --parsable -J ${name}_${k} -A bef@${gpu} -C ${gpu} --qos=qos_gpu_${gpu}-dev \
           --output=$G/logs_zs_simple2/%j_${name}_${k}.out \
           --export=ALL,SHARD=${k},GPU=${gpu},ADAPTER=${ad},PROMPT=${pr},MAXTOK=${mt},INPUT=${input},NSH=${nsh},ROWS=${rows},OUT=${out},EXTRA=${extra} $G/job_zs_simple2.slurm) \
        && echo "$(date +%F_%T) ${name}_${k} ($n/$size) -> $gpu job $id"
      q="$q"$'\n'"${name}_${k} qos_gpu_${gpu}-dev"
      break
    done
  done
  [ $left -eq 0 ] && { echo "$(date +%F_%T) all shards done"; break; }
  sleep 300
done
