#!/bin/bash
# Sourced by the host launchers (scripts/<host>/*.sh) for the vault's `exp.py launch`.
# Call it once every path variable is set and before anything is created or started:
#   exp_paths INPUT=<path> [INPUT=<path> ...] OUTPUT=<dir> RESULT=<glob> GPUS=<ids>
# Without EXP_DRYRUN it does nothing. With EXP_DRYRUN=1 it prints EXP_<KEY>=<value> lines (relative paths made
# absolute against the launcher's working directory) and exits, so exp.py knows the paths before the run.
exp_paths() {
  [ -n "${EXP_DRYRUN:-}" ] || return 0
  local kv k v
  for kv in "$@"; do
    k=${kv%%=*}; v=${kv#*=}
    case $k in INPUT|OUTPUT|RESULT) case $v in /*) ;; *) v=$PWD/$v ;; esac ;; esac
    echo "EXP_$k=$v"
  done
  exit 0
}
