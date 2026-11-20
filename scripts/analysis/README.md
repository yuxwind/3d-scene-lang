# scripts/analysis: result and visualization scripts

Every script that reads results, scores, logs or makes plots lives here. The scripts are **reused and extended, never rewritten**.

## Rules (for people and agents)
- **Write here, on the Mac** (`exp-vault/60 Code/3d-scene-lang/repo`), commit, push, then `exp.py deploy 3d-scene-lang all`. Never edit them on a server.
- **Run them where the data is**, from the Mac:
  `python3 scripts/remote.py run [-n LINES] <host> 3d-scene-lang <script> [args...]` (in the exp-vault).
  The runner sources `site.env` and uses the env's python (`$SR_PYTHON_BIN`) if there is one, else `python3`. Paths like `$SCRATCH/...` expand on the server.
- **Before writing a new script, check the table below.** If one is close, add a flag or subcommand to it instead of making a new file.
- **Compact output:** print a summary table (a few dozen lines at most), never raw predictions or file lists. Put large outputs (plots, CSV) in a file on the server and print its path and size. Use `remote.py collect` if the Mac needs it.
- Prefer the standard library (it runs everywhere, including jz login nodes). If a script needs the training env, say so in its docstring.
- Every script starts with a docstring: what it reads, its arguments, what it prints. Add a row to the table below.

## Scripts
| Script | What it does |
|---|---|
| `inspect_json.py` | Structure of JSON/JSONL result files (keys, types, list lengths, one sample value), plus a file count per pattern. Use it first on an unfamiliar result folder. |
| `peek.py` | First lines (or `--grep` matches) of text files: docstrings, markdown summaries, logs. Capped. |
| `vsibench_scores.py` | VSI-Bench table (n, overall, 8 tasks) from `scores*.json` (`{method: {...}}`). Claude rows by default; `--all-methods`, `--csv`. |
