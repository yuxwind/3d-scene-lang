"""Which 1,024-token scene-text answers stopped at the cap? Re-tokenize each raw output (greedy decoding: the first
1,024 tokens of a longer run are identical, so only capped rows can change with a larger limit)."""
import json, re
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from sr_vggt.eval import MCA_QUESTION_TYPES, read_rows
D = Path(__file__).resolve().parent
tok = AutoTokenizer.from_pretrained("/lustre/fswork/projects/rech/bef/uxt97ti/hf_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
P = {}
for f in sorted((D / "qwen35_4b_scene_long").glob("predictions*.jsonl")): P.update(read_rows(f))
deb = {r["index"] for r in map(json.loads, open(D / "vsibench_scannetpp_scenelang.jsonl")) if r["debiased"]}
un = {int(x) for x in open(D / "unanswered85_indices.txt").read().split()}
L = re.compile(r"\b([A-D])\b(?![a-z'])"); N = re.compile(r"-?\d+(?:\.\d+)?")
n = {i: len(tok(P[i]["prediction"])["input_ids"]) for i in deb}
capped = [i for i in deb - un if n[i] >= 1015]
print("answered rows:", len(deb - un), "| stopped at the 1,024 cap:", len(capped), "| short answers (<=5 tokens):", sum(n[i] <= 5 for i in deb - un))
print("capped by type:", dict(Counter(P[i]["question_type"] for i in capped)))
for i in capped[:4]:
    print("---", i, P[i]["question_type"], "gt", P[i]["ground_truth"], "| tail:", P[i]["prediction"][-160:].replace("\n", " "))
B = {}
for f in sorted((D / "qwen35_4b_scene_4096_thinking").glob("predictions*.jsonl")): B.update(read_rows(f))
r = next(iter(B.values())); print("\n=== thinking sample tail:", r["prediction"][-400:].replace("\n", " "))
json.dump(capped, open(D / "capped_answered_indices.json", "w"))
