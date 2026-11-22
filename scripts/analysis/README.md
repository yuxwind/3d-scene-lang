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
| `vsibench_scores.py` | VSI-Bench table (n, overall, 8 tasks) from `scores*.json` (`{method: {...}}` or `{'subset / full|debiased / scene': {...}}`). Drops baselines by default (`--exclude`), `--subset all`, `--csv`. |
| `vsibench_score_answers.py` | Score raw answer files (`<variant>/out/*.json`) against GT rows (`--gt`): full + debiased overall and per task. For runs without `scores*.json` (e.g. gandalf `claude_nocode_s3`). |
| `run_status.py` | Training run dirs: checkpoints (count, first..last), DONE, final weights (size + fingerprint for cross-host comparison), step/max_steps, last loss. |
| `layout.py` | Markdown layout of the project on a host (code, data, base models, runs with final-weights/progress, logs, envs) from `site.env` + `--extra`. Feeds the vault's `Layouts/<host>.md`. |
| `table_info.py` | parquet/csv/jsonl summary: rows, columns + dtypes, value counts of low-cardinality columns, a sample row. Needs pandas. |
| `revsi_score_answers.py` | Score Claude no-code ReVSI answers (`out/*.json`) vs `rows.jsonl`: ReVSI protocol (7 tasks from 13 subtypes), `--by-dataset`; also sums the run's cost from `*.meta.json`. |
