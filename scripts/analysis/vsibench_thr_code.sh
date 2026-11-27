#!/bin/bash
# VSI-Bench box-score threshold sweep (SAM 3 + VGGT-SLAM, question names + "room door", estimated scale): the code solver on
# each threshold list (data/scenelang_vsibench_slam/oracle_code_lists.py -> oracle_code_thr<t>.json next to the rows, skipped
# if present), then vsibench_code_answered.py: code vs the mix-v2-1 model run on the same list, on the questions the code
# answers. No GPU. Paths: gandalf old tree (rows and runs live there).
#   bash scripts/analysis/vsibench_thr_code.sh [THR ...]     default 0.1 0.3 0.5 0.7 0.9 (0.7 = rows_s3q_roomdoor_all5130_full)
R=${R:-/scr/xinyu/projects/3d-scene-lang}; D=$R/data/scenelang_vsibench_slam; E=$R/runs/eval_q35vs_gandalf
PYS=${PYS:-/scr/xinyu/envs/vggtdet/bin/python}; PYT=${SR_PYTHON_BIN:+$SR_PYTHON_BIN/python}; PYT=${PYT:-/scr/xinyu/envs/sr_vggt/bin/python}
cd "$(dirname "$0")/../.." || exit 1
args=()
for t in ${*:-0.1 0.3 0.5 0.7 0.9}; do
  if [ "$t" = 0.7 ]; then rows=$D/rows_s3q_roomdoor_all5130_full.jsonl; run=$E/mixv2_1_s3q_roomdoor_rvs_question
  else rows=$D/rows_s3q_roomdoor_thr${t}_all5130_full.jsonl; run=$E/mixv2_1_s3q_roomdoor_thr${t}_rvs_question; fi
  [ -f "$D/oracle_code_thr$t.json" ] || $PYS data/scenelang_vsibench_slam/oracle_code_lists.py "thr$t=$rows" | tail -1
  args+=(--code "thr$t=$D/oracle_code_thr$t.json" --model "mixv2_1@thr$t=$run")
done
$PYT scripts/analysis/vsibench_code_answered.py "${args[@]}"
