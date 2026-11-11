"""The jz-chain conf's TRAINER for sr_vggt chains: chain_submit.sh runs check_args.py, which extracts
parse_args() below (plus the upper-case constants) and checks the chain's .args before any sbatch.
The args are chain_dev.sbatch's: <experiment> <output dir> [extra sr_vggt.train flags]."""
import argparse
import os

CONFIG_DIR = "/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/configs"


def parse_args():
    exps = sorted(f[:-4] for f in os.listdir(CONFIG_DIR) if f.endswith(".env"))
    ap = argparse.ArgumentParser(description="sr_vggt chain link")
    ap.add_argument("experiment", choices=exps)
    ap.add_argument("output_dir")
    ap.add_argument("extra", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if not os.path.isabs(args.output_dir):
        ap.error(f"output_dir must be absolute, got {args.output_dir!r}")
    return args
