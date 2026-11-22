#!/usr/bin/env python3
"""Driver for Claude no-code runs: one fresh headless `claude -p` per batch file, agent nocode-answerer (Read + Write only).

usage: run_batches.py RUN_DIR --model MODEL [--parallel 4] [--limit N] [--only NAME ...] [--timeout 3600] [--dry-run]
  RUN_DIR/batches/<name>.txt  -> RUN_DIR/out/<name>.json  {"answers": {"<id>": "<answer>"}, "tools_used": [...]}
  RUN_DIR/manifest.json gives the ids each batch must answer. Batches whose out file already has every id are skipped
  (resumable). Per batch the claude JSON result (cost, usage, duration) goes to RUN_DIR/out/<name>.meta.json.
  --limit: run at most N batches (smoke test). Prints one line per batch and a running cost total.
Agent definition: scripts/nocode/nocode_answerer.json (passed with --agents, so nothing in .claude/ is needed).
"""
import argparse, concurrent.futures as cf, json, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def done(out, ids):
    try:
        a = json.load(open(out))["answers"]
        return all(str(i) in a for i in ids)
    except Exception:
        return False


def one(run, name, ids, model, timeout):
    b, o = run / "batches" / f"{name}.txt", run / "out" / f"{name}.json"
    msg = (f"Use the nocode-answerer agent's rules. Read the batch file {b}. It has {len(ids)} questions (=== QUESTION id=... ===). "
           f"Answer every one. Then write {o} as JSON: {{\"answers\": {{\"<id>\": \"<answer>\", ...}}, \"tools_used\": [\"Read\", \"Write\"]}} "
           f"with exactly these ids: {', '.join(map(str, ids))}. Multiple choice: the option letter only. Numbers: the number only.")
    cmd = ["claude", "-p", msg, "--model", model, "--agents", str(HERE / "nocode_answerer.json"), "--agent", "nocode-answerer",
           "--allowedTools", "Read,Write", "--permission-mode", "acceptEdits", "--output-format", "json"]
    t = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=run)
        meta = json.loads(p.stdout) if p.stdout.strip().startswith("{") else {"raw": p.stdout[-2000:], "stderr": p.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        meta = {"error": "timeout"}
    meta["wall_s"] = round(time.time() - t, 1)
    json.dump(meta, open(run / "out" / f"{name}.meta.json", "w"), indent=1)
    return name, done(o, ids), meta.get("total_cost_usd", 0.0) or 0.0, meta["wall_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("--model", required=True); ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--limit", type=int); ap.add_argument("--only", nargs="*"); ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run = Path(a.run_dir).resolve(); (run / "out").mkdir(exist_ok=True)
    man = json.load(open(run / "manifest.json"))
    todo = [n for n in man if (not a.only or n in a.only) and not done(run / "out" / f"{n}.json", man[n])]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(man)} batches, {len(todo)} to run, model {a.model}, parallel {a.parallel}", flush=True)
    if a.dry_run:
        return
    total = 0.0
    with cf.ThreadPoolExecutor(a.parallel) as ex:
        for name, ok, cost, wall in ex.map(lambda n: one(run, n, man[n], a.model, a.timeout), todo):
            total += cost
            print(f"{time.strftime('%H:%M:%S')} {'OK  ' if ok else 'MISS'} {name}  ${cost:.2f}  {wall:.0f}s  total ${total:.2f}", flush=True)


if __name__ == "__main__":
    main()
