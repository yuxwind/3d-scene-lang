#!/usr/bin/env python3
"""Run records of this repo (Experiments/runs.json + runs.md), managed from any clone: on a server or on the Mac.

On a workstation (gandalf, wukong), from the clone root:
    python3 scripts/runs.py launch scripts/gandalf/launch_x.sh NAME=a ROWS=b --name N --desc "what/why" --hours H [--smoke]
        plan: the launcher's paths (EXP_DRYRUN), input sizes, output empty, its GPUs idle, GPU-hours; starts nothing
    python3 scripts/runs.py launch ... --yes        start it in tmux (exp_<name>), add the record, commit + push the records
    python3 scripts/runs.py status [--all]          refresh this host's running records (tmux / processes / result files)
    python3 scripts/runs.py record --name N --desc D --input P [--input P] --output DIR --result PATH_OR_GLOB [--info S]
        a run that already finished; Time = newest result file
    python3 scripts/runs.py list [-n 20]

Same records and table as the vault's `exp.py launch/record/runs` (the vault's exp.py calls save() below), so a run can
be started from the Mac or from a server. Rows added to runs.md by hand (Exp not in runs.json) are kept at the end.
Before reading, the records are pulled (git pull --ff-only); after writing, only Experiments/runs.{json,md} are committed
and pushed. A failed pull/push is reported, never forced: push later from any clone.
Rules kept from exp.py: idle GPUs only (shared workstations), a finished --smoke launch of the launcher before a full one
(or --skip-smoke-reason), and the clone must be clean with HEAD pushed (the record keeps the commit).
"""
import argparse, datetime as dt, glob, json, os, re, shlex, socket, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "Experiments"
COLS = ["Time", "Exp", "Remote", "Description", "Input", "Output", "Result", "Status"]
LIVE = {"launched", "running"}


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def git(*a, root=ROOT):
    return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True)


def die(m):
    sys.exit(f"ERROR {m}")


def load(root=ROOT, pull=True):
    if pull:
        p = git("pull", "-q", "--ff-only", root=root)
        if p.returncode:
            print(f"WARN  git pull failed ({(p.stderr.strip().splitlines() or ['?'])[-1][:150]}); using the local records")
    j = Path(root) / "Experiments" / "runs.json"
    return json.loads(j.read_text()) if j.exists() else []


def hand_rows(md: Path, names: set) -> list:
    """Table rows of runs.md whose Exp is not in runs.json (added by hand): kept verbatim."""
    rows = []
    if md.exists():
        for l in md.read_text().splitlines():
            c = [x.strip() for x in l.strip().strip("|").split("|")] if l.startswith("| ") else []
            if len(c) >= 3 and c[0] != "Time":
                n = c[1].strip("*` ")
                if n and n not in names:
                    rows.append(l)
    return rows


def save(recs, root=ROOT, project=None, commit=True):
    """Write runs.json + runs.md (hand rows kept); commit + push only those two files."""
    root = Path(root); d = root / "Experiments"; d.mkdir(exist_ok=True)
    j, md = d / "runs.json", d / "runs.md"
    pname = project or (root.resolve().parent.name if root.resolve().name == "repo" else root.resolve().name)
    hand = hand_rows(md, {r["name"] for r in recs})
    j.write_text(json.dumps(recs, indent=2))
    code = lambda x: f"`{x}`" if x else ""
    lines = [f"# {pname}: runs", "",
             "Written by `scripts/runs.py` (on a server) or the vault's `exp.py launch` / `record` / `runs` (on the Mac) from "
             "`runs.json`. Rows added by hand (on any clone) are kept at the end of the table; `record` turns one into a full "
             "record.", "",
             "Input, Output and Result are paths on the Remote host. Time is when the run started (local time of the machine "
             "that launched it); for runs recorded after they finished (marked *recorded*) it is the time of the newest result file.", "",
             "| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    for r in sorted(recs, key=lambda r: r["time"], reverse=True):
        st = r["status"] + (f" ({r['status_info']})" if r.get("status_info") else "")
        res = code(r["result"]) + (f" ({r['result_n']} files)" if r.get("result_n") else "")
        cells = [r["time"] + (" *recorded*" if r.get("recorded") else ""), f"**{r['name']}**", r["host"],
                 r.get("description", "").replace("|", "/"), "<br>".join(code(i) for i in r["inputs"]), code(r["output"]), res, st]
        lines.append("| " + " | ".join(cells) + " |")
    lines += hand
    if hand:
        print(f"note: kept {len(hand)} hand-added row(s) of runs.md that are not in runs.json")
    lines += ["", "Launch details (launcher, variables, commit, GPUs, tmux session) are in `runs.json`."]
    md.write_text("\n".join(lines) + "\n")
    if not commit:
        return
    rel = ["Experiments/runs.json", "Experiments/runs.md"]
    if git("status", "--porcelain", "--", *rel, root=root).stdout.strip():
        git("add", "--", *rel, root=root)
        c = git("commit", "-q", "-m", f"Experiments: run records ({pname})", "--", *rel, root=root)
        p = git("push", "-q", root=root) if not c.returncode else c
        print("runs: committed Experiments/runs.{json,md}" + ("" if not p.returncode else
              f"; WARN {'push' if not c.returncode else 'commit'} failed: {(p.stderr.strip().splitlines() or ['?'])[-1][:150]}"
              " (push later: git push)"))


def host():
    return socket.gethostname().split(".")[0]


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash", **kw)


def cmd_launch(a):
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", a.name)
    recs = load()
    h = host()
    if any(r["name"] == name and r["host"] == h and r["status"] in LIVE for r in recs):
        die(f"{name} on {h} is still running/launched")
    if not (ROOT / a.launcher).exists():
        die(f"no {a.launcher} in {ROOT}")
    if any("=" not in v for v in a.vars):
        die("variables must be KEY=VALUE")
    env = dict(v.split("=", 1) for v in a.vars)
    if a.output:
        env["OUT"] = a.output
    envs = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    problems = []
    # code: clean tracked tree, HEAD pushed (the record keeps the commit)
    git("fetch", "-q", "origin")
    head = git("rev-parse", "HEAD").stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no").stdout.strip():
        problems.append("the clone has uncommitted changes to tracked files: commit + push first")
    if git("branch", "-r", "--contains", head).stdout.strip() == "":
        problems.append(f"HEAD {head[:7]} is not pushed: git push first")
    # dry run: the launcher prints its resolved paths and exits
    p = sh(f'cd {shlex.quote(str(ROOT))} && env EXP_DRYRUN=1 {envs} bash {shlex.quote(a.launcher)}')
    got = {}
    for line in p.stdout.splitlines():
        if line.startswith("EXP_") and "=" in line:
            k, v = line[4:].split("=", 1); got.setdefault(k, []).append(v)
    if "OUTPUT" not in got:
        die(f"{a.launcher} printed no paths under EXP_DRYRUN (rc={p.returncode}); add an exp_paths call.\n"
            f"{(p.stderr or p.stdout).strip()[-400:]}")
    inputs = a.input or got.get("INPUT", [])
    output = got["OUTPUT"][0]
    result = a.result or (got.get("RESULT") or [""])[0]
    gpus = (got.get("GPUS") or [""])[0]
    ngpu = len([g for g in gpus.split(",") if g.strip()])
    print(f"plan  {name} on {h}  ({a.launcher}{' ' + envs if envs else ''})  commit {head[:7]}")
    for i in inputs:
        ok = os.path.exists(i)
        size = sh(f"du -sh {shlex.quote(i)} 2>/dev/null | cut -f1").stdout.strip() if ok else "MISSING"
        print(f"  input   {i}  ({size})")
        if not ok:
            problems.append(f"input missing: {i}")
    n_out = len(os.listdir(output)) if os.path.isdir(output) else 0
    print(f"  output  {output}  ({'empty/new' if not n_out else f'{n_out} entries'})")
    if n_out and not a.resume:
        problems.append(f"output {output} is not empty (another --name/--output, or --resume)")
    if gpus:
        q = sh(f"nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i {shlex.quote(gpus)}")
        for line in q.stdout.strip().splitlines():
            idx, mem, util = [x.strip() for x in line.split(",")]
            busy = int(mem) > 2000 or int(util) > 10
            print(f"  gpu {idx}   {mem} MiB used, {util}% util{'  BUSY' if busy else ''}")
            if busy:
                problems.append(f"GPU {idx} is in use (shared workstation: idle GPUs only)")
    print(f"  result  {result or '(none declared)'}")
    if a.hours is None:
        problems.append("give --hours (expected wall clock)")
    else:
        print(f"  cost    1 run x {ngpu} GPU(s) x {a.hours:g} h = {ngpu * a.hours:g} GPU-h")
    smoke_ok = any(r.get("launcher") == a.launcher and r.get("smoke") and r["status"] == "done" for r in recs)
    if not (a.smoke or smoke_ok or a.skip_smoke_reason):
        problems.append("no finished smoke launch of this launcher: run one with --smoke (<=10 min, 1 GPU), or --skip-smoke-reason")
    if problems:
        print("blocked:\n  - " + "\n  - ".join(problems)); sys.exit(1)
    if not a.yes:
        print("plan only. Rerun with --yes to start it."); return
    sess = f"exp_{name}"[:60]
    os.makedirs(output, exist_ok=True)
    wrap = Path(output) / "exp_launch.sh"
    wrap.write_text(f"#!/bin/bash\ncd {shlex.quote(str(ROOT))}\n" + "".join(f"export {k}={shlex.quote(v)}\n" for k, v in env.items())
                    + f"bash {shlex.quote(a.launcher)}\necho \"EXP_EXIT=$?\"\n")
    sh(f'tmux new-session -d -s {sess} "bash {shlex.quote(str(wrap))} > {shlex.quote(output)}/exp_launch.log 2>&1"')
    time.sleep(3)
    alive = sh(f"tmux has-session -t {sess}").returncode == 0
    print(f"tmux {sess}: {'alive' if alive else 'NOT alive'}; " + sh(f"tail -3 {shlex.quote(output)}/exp_launch.log").stdout.strip()[-300:])
    recs = load()  # re-read: someone may have pushed records meanwhile
    recs.append({"time": now(), "name": name, "host": h, "description": a.desc or "", "inputs": inputs, "output": output,
                 "result": result, "status": "launched", "status_info": "", "launcher": a.launcher, "vars": env,
                 "commit": head[:7], "gpus": gpus, "hours": a.hours, "tmux": sess, "smoke": bool(a.smoke),
                 "skip_smoke_reason": a.skip_smoke_reason or "", "launched_from": "runs.py"})
    save(recs)
    print(f"launched {name} on {h}; record in Experiments/runs.md")


def refresh(r):
    alive = r.get("tmux") and sh(f"tmux has-session -t {r['tmux']}").returncode == 0
    pat = "/".join(r["output"].rstrip("/").split("/")[-2:])
    procs = len([l for l in sh(f"pgrep -u $USER -f -- {shlex.quote(pat)}").stdout.split() if l.strip() and int(l) != os.getpid()])
    n = len(glob.glob(r["result"])) if r["result"] else 0
    if alive or procs:
        r["status"], r["status_info"] = "running", f"{procs} procs" if procs else ""
    elif n:
        r["status"], r["status_info"] = "done", ""
    else:
        log = Path(r["output"]) / "exp_launch.log"
        last = log.read_text(errors="ignore").strip().splitlines()[-1:] if log.exists() else []
        r["status"], r["status_info"] = "failed", ((last or ["no result, no process"])[0].replace("|", "/"))[:60]
    r["result_n"], r["checked"] = n, now()


def cmd_status(a):
    recs = load(); h = host()
    for r in recs:
        if r["host"] == h and (r["status"] in LIVE or a.all):
            refresh(r)
    save(recs)
    cmd_list(a, recs)


def cmd_record(a):
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", a.name)
    recs = load(); h = host()
    if any(r["name"] == name and r["host"] == h for r in recs) and not a.replace:
        die(f"{name} on {h} is already recorded (--replace to overwrite)")
    missing = [p for p in a.input + [a.output] + [a.result] if not glob.glob(os.path.expanduser(p))]
    if missing:
        die(f"not found: {', '.join(missing)}")
    files = [f for x in glob.glob(a.result) for f in ([x] if os.path.isfile(x) else
             [os.path.join(dp, fn) for dp, _, fns in os.walk(x) for fn in fns])]
    if not files:
        die(f"no files under {a.result}")
    t = max(os.path.getmtime(f) for f in files)
    rec = {"time": dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M"), "name": name, "host": h, "description": a.desc,
           "inputs": a.input, "output": a.output, "result": a.result, "status": a.status, "status_info": a.info or "",
           "result_n": len(files), "recorded": now(), "commit": a.commit or ""}
    save([r for r in recs if not (r["name"] == name and r["host"] == h)] + [rec])
    print(f"recorded {name} on {h}: time {rec['time']} (newest of {len(files)} result files)")


def cmd_list(a, recs=None):
    recs = recs if recs is not None else load()
    for r in sorted(recs, key=lambda r: r["time"], reverse=True)[:a.n]:
        print(f'{r["time"]}  {r["name"]:40.40} {r["host"]:8} {r["status"]:9} {r.get("status_info", "")}')
    print(f"{len(recs)} records: Experiments/runs.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("launch"); p.add_argument("launcher"); p.add_argument("vars", nargs="*")
    p.add_argument("--name", required=True); p.add_argument("--desc", default=""); p.add_argument("--hours", type=float)
    p.add_argument("--output"); p.add_argument("--input", action="append"); p.add_argument("--result")
    p.add_argument("--smoke", action="store_true"); p.add_argument("--skip-smoke-reason"); p.add_argument("--resume", action="store_true")
    p.add_argument("--yes", action="store_true"); p.set_defaults(f=cmd_launch)
    p = sp.add_parser("status"); p.add_argument("--all", action="store_true"); p.add_argument("-n", type=int, default=20)
    p.set_defaults(f=cmd_status)
    p = sp.add_parser("record"); p.add_argument("--name", required=True); p.add_argument("--desc", default="")
    p.add_argument("--input", action="append", default=[]); p.add_argument("--output", required=True)
    p.add_argument("--result", required=True); p.add_argument("--status", default="done"); p.add_argument("--info")
    p.add_argument("--commit"); p.add_argument("--replace", action="store_true"); p.set_defaults(f=cmd_record)
    p = sp.add_parser("list"); p.add_argument("-n", type=int, default=20); p.set_defaults(f=cmd_list)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
