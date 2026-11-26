# scripts/analysis: result and visualization scripts

Every script that reads results, scores, logs or makes plots lives here. The scripts are **reused and extended, never rewritten**.

## Rules (for people and agents)
- **Write in any clone** (the Mac `exp-vault/60_Code/3d-scene-lang/repo` or a server clone; `git pull` first), commit, push, then `exp.py deploy 3d-scene-lang all` so every host has them.
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
| `qwen_eval_scores.py` | Sharded Qwen evals (`predictions_shard*.jsonl`, e.g. `runs/eval_q35vs_gandalf/*`): merge shards, score full + debiased (`--rows` gives the split) and 8 tasks with `sr_vggt.eval.score`. Needs the training env. Replaces `data/scenelang_vsibench_det_objq/cmp_eval.py`. |
| `revsi_score_answers.py` | Score Claude no-code ReVSI answers (`out/*.json`) vs `rows.jsonl` or `--gt` (jsonl, or the ReVSI parquet): ReVSI protocol (7 tasks from 13 subtypes), `--by-dataset`; also sums the run's cost from `*.meta.json`. |
| `bench_compare.py` | VSI-Bench vs ReVSI question sets: per question type n (debiased n), datasets, scenes, question templates, examples, options, answer letters / numeric GT quantiles; `--json` for plots. Needs pandas. |
| `route_check.py` | Route planning vs scene geometry for Claude no-code runs: resolves the named objects in the scene list, walks the route on the boxes, compares the geometric turns with the benchmark answer (and left/right swapped) and with Claude, split by name status and turns. |
| `vsibench_code_answered.py` | VSI-Bench code solver (`oracle_code_<list>.json`) and model runs on the same box list, scored only on the questions the code answers: 6 tasks, rel. direction easy / medium / hard, mean (`sr_vggt.eval.score`). |
| `revsi_relq_check.py` | ReVSI relative distance / direction per question type (closest / farthest, forward / backward, easy / hard): accuracy, chance, GT answer spread, and how often a run gives the answer of the flipped question (closest for farthest, forward for backward), computed from the GT scene with the code solver. Takes eval_scenelang predictions and Claude / code-solver answers. |
| `scene_usage.py` | Per question type: rows with a scene vs video only, in eval rows (`use_scene`) or a training mix (`<scene>` in the prompt); `.jsonl(.gz)`. |
