#!/bin/bash
#SBATCH --account=bef@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
# One 2 h dev link of a training run in the jz-chain scheme (~/work/tools/jz-chain/chain-job-instruction.md).
#   args: <experiment> <output dir> [extra sr_vggt.train flags]  (from the chain's .args; chain_submit.sh
#   sets --job-name/--output and CHAIN_SHORT)
# train.slurm resumes from the newest checkpoint-N; sr_vggt.train saves 5 min before the time limit and
# exits (SaveBeforeTimeout). Then: a new checkpoint-N touches <out>/chain_progress (the conf's
# PROGRESS_FILE), the final adapter (<out>/adapter_config.json) writes <out>/DONE, and chain_next.sh
# submits the next link.
set -uo pipefail
export SR_ROOT=/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang
OUT=$2
latest() { ls -d "${OUT}"/checkpoint-* 2>/dev/null | sed 's#.*-##' | sort -n | tail -1; }
before=$(latest)
ck_before=$(stat -c %Y "${OUT}/chain_progress" 2>/dev/null || echo 0)
rc=0
bash "${SR_ROOT}/scripts/jobs/train.slurm" "$@" || rc=$?
after=$(latest)
echo "LINK FINISHED $(date +%T) rc=${rc} checkpoint ${before:-none} -> ${after:-none}"
[[ -n "${after}" && "${after}" != "${before}" ]] && touch "${OUT}/chain_progress"
[[ -f "${OUT}/adapter_config.json" ]] && touch "${OUT}/DONE"
[[ -n "${CHAIN_SHORT:-}" ]] && bash /lustre/fswork/projects/rech/bef/uxt97ti/tools/jz-chain/chain_next.sh \
  "${CHAIN_SHORT}" "${ck_before}" "${rc}"
exit "${rc}"
