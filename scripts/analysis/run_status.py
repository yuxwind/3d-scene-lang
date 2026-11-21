#!/usr/bin/env python3
"""Status of training run dirs: checkpoints, final weights, progress, and a weights fingerprint for cross-host comparison.

usage: run_status.py RUN_DIR [RUN_DIR ...] [--fp-mb 64]
per run prints:
  checkpoints: count, first..last step;  DONE marker;  total size (du)
  final weights at the run root (adapter/model *.safetensors|*.bin): size + fingerprint
  progress from the newest trainer_state.json: global_step / max_steps, epoch, last logged loss
fingerprint = sha256 of (file size + first --fp-mb MB): cheap; identical files on two hosts give the same value.
stdlib only.
"""
import argparse, glob, hashlib, json, os, re, subprocess


def fp(path, mb):
    h = hashlib.sha256(str(os.path.getsize(path)).encode())
    with open(path, "rb") as f:
        h.update(f.read(mb << 20))
    return h.hexdigest()[:12]


def human(n):
    for u in ("B", "K", "M", "G", "T"):
        if n < 1024:
            return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.1f}P"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--fp-mb", type=int, default=64)
    a = ap.parse_args()
    for r in a.runs:
        print(f"== {r}")
        if not os.path.isdir(r):
            print("   MISSING"); continue
        cks = sorted((int(m.group(1)), d) for d in glob.glob(os.path.join(r, "checkpoint-*"))
                     if (m := re.search(r"checkpoint-(\d+)$", d)))
        du = subprocess.run(["du", "-sh", r], capture_output=True, text=True).stdout.split("\t")[0]
        print(f"   size {du}  checkpoints {len(cks)}" + (f" ({cks[0][0]}..{cks[-1][0]})" if cks else "")
              + f"  DONE {'yes' if os.path.exists(os.path.join(r, 'DONE')) else 'no'}")
        ws = [f for f in sorted(glob.glob(os.path.join(r, "*.safetensors")) + glob.glob(os.path.join(r, "*.bin")))
              if not f.endswith("training_args.bin")]
        for f in ws:
            print(f"   final {os.path.basename(f)}  {human(os.path.getsize(f))}  fp {fp(f, a.fp_mb)}  "
                  f"mtime {os.path.getmtime(f):.0f}")
        if not ws:
            print("   final weights: none at run root")
            if cks:
                last = [f for f in glob.glob(os.path.join(cks[-1][1], "*.safetensors"))]
                for f in last:
                    print(f"   last-ckpt {os.path.basename(f)}  {human(os.path.getsize(f))}  fp {fp(f, a.fp_mb)}")
        states = [os.path.join(d, "trainer_state.json") for _, d in cks[::-1]] + [os.path.join(r, "trainer_state.json")]
        st = next((s for s in states if os.path.exists(s)), None)
        if st:
            s = json.load(open(st))
            loss = next((h.get("loss") for h in reversed(s.get("log_history", [])) if "loss" in h), None)
            print(f"   progress step {s.get('global_step')}/{s.get('max_steps')}  epoch {s.get('epoch', 0):.2f}  "
                  f"last loss {loss}  ({os.path.relpath(st, r)})")


if __name__ == "__main__":
    main()
