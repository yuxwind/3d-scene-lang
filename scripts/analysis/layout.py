#!/usr/bin/env python3
"""Markdown layout of this project on the current host: code, data, models, runs/outputs, logs, envs.

Run from the deployed clone (remote.py run does that); paths come from site.env ($SR_*), plus --extra dirs.
usage: layout.py HOSTNAME [--extra LABEL=DIR ...] [--sizes]
  --extra: more places to describe, e.g. old_tree=/scr/xinyu/projects/3d-scene-lang  nocode=$SCRATCH/3d_scene_lang
  --sizes: du -sh every listed dir (slow on Lustre; off by default -> only run dirs get sizes)
prints markdown: a table of locations, then every run dir under SR_OUTPUT_ROOT (and extras that look like run roots)
with: status (DONE / step x of y), final weights (yes/no + size), checkpoints count. stdlib only.
"""
import argparse, glob, json, os, re, subprocess, datetime as dt


def du(p):
    r = subprocess.run(["du", "-sh", p], capture_output=True, text=True, timeout=1800)
    return r.stdout.split("\t")[0] if r.returncode == 0 else "?"


def exists(p):
    return p and os.path.exists(p)


def run_info(d):
    cks = sorted(int(m.group(1)) for c in glob.glob(os.path.join(d, "checkpoint-*")) if (m := re.search(r"-(\d+)$", c)))
    w = [f for f in glob.glob(os.path.join(d, "*.safetensors"))]
    st = None
    for s in ([os.path.join(d, f"checkpoint-{c}", "trainer_state.json") for c in cks[::-1]] + [os.path.join(d, "trainer_state.json")]):
        if os.path.exists(s):
            st = json.load(open(s)); break
    prog = f"step {st.get('global_step')}/{st.get('max_steps')}" if st else "-"
    done = "DONE" if os.path.exists(os.path.join(d, "DONE")) else ""
    wsz = sum(os.path.getsize(f) for f in w)
    return (f"{'yes ' + str(round(wsz / 2**20)) + 'M' if w else 'no'}", len(cks), prog, done)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("host"); ap.add_argument("--extra", action="append", default=[]); ap.add_argument("--sizes", action="store_true")
    a = ap.parse_args()
    E = os.environ.get
    code = os.getcwd()
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    rows = [("code (deployed clone, git)", code, f"git {head}"),
            ("generated data (gitignored, in clone)", os.path.join(code, "data"), "mixes *.jsonl.gz, det/rows files"),
            ("VSI-590K videos ($SR_VSI590K)", E("SR_VSI590K"), "training videos"),
            ("VSI-Bench ($SR_VSIBENCH)", E("SR_VSIBENCH"), "eval videos + parquet"),
            ("base model weights ($SR_HF_HOME)", E("SR_HF_HOME"), ""),
            ("python envs ($SR_ENV_ROOT)", E("SR_ENV_ROOT"), f"python bin: {E('SR_PYTHON_BIN') or '(module/venv per GPU kind)'}"),
            ("runs / outputs ($SR_OUTPUT_ROOT)", E("SR_OUTPUT_ROOT"), "sr_vggt_<exp>/ (checkpoint-N, final adapter at root), _smoke_*"),
            ("logs ($SR_LOG_DIR)", E("SR_LOG_DIR"), "")]
    for x in a.extra:
        lab, _, p = x.partition("=")
        rows.append((lab, os.path.expandvars(p), "extra"))
    print(f"# {a.host}: 3d-scene-lang layout\n\nGenerated {dt.date.today()} by `scripts/analysis/layout.py` (repo {head}) from `site.env`.\n")
    print("| What | Path | Exists | Notes |\n|---|---|---|---|")
    for lab, p, note in rows:
        ex = ("yes" + (f", {du(p)}" if a.sizes else "")) if exists(p) else ("**missing**" if p else "not set")
        print(f"| {lab} | `{p or '-'}` | {ex} | {note} |")
    hf = E("SR_HF_HOME")
    if exists(hf):
        ms = sorted(os.path.basename(m).replace("models--", "").replace("--", "/") for m in glob.glob(os.path.join(hf, "hub", "models--*")))
        print(f"\n## Base models in HF cache\n" + ("\n".join(f"- `{m}`" for m in ms) or "- none"))
    roots = [E("SR_OUTPUT_ROOT")] + [os.path.expandvars(x.partition("=")[2]) for x in a.extra]
    for root in [r for r in roots if exists(r)]:
        runs = sorted(d for d in glob.glob(os.path.join(root, "*")) + glob.glob(os.path.join(root, "*", "sr_vggt_*"))
                      if os.path.isdir(d) and (glob.glob(os.path.join(d, "checkpoint-*")) or glob.glob(os.path.join(d, "*.safetensors"))
                                               or os.path.basename(d).startswith(("sr_vggt", "_smoke"))))
        if not runs:
            continue
        print(f"\n## Runs under `{root}`\n\n| Run | Final weights | Checkpoints | Progress | Done | Size |\n|---|---|---|---|---|---|")
        for d in runs:
            w, n, prog, done = run_info(d)
            print(f"| `{os.path.relpath(d, root)}` | {w} | {n} | {prog} | {done} | {du(d)} |")


if __name__ == "__main__":
    main()
