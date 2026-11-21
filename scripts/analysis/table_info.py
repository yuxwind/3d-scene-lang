#!/usr/bin/env python3
"""Summary of tabular files (parquet / csv / jsonl): rows, columns + dtypes, value counts of low-cardinality columns, one sample row.

usage: table_info.py FILE [FILE ...] [--counts COL ...] [--max-card 30] [--sample 1]
  --counts: always print value counts for these columns (top 30).
needs pandas (+ pyarrow for parquet): run with the training env (on jz the runner uses python3; see README).
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--counts", nargs="*", default=[])
ap.add_argument("--max-card", type=int, default=30)
ap.add_argument("--sample", type=int, default=1)
a = ap.parse_args()
for f in a.files:
    df = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_json(f, lines=True) if f.endswith(".jsonl") else pd.read_csv(f)
    print(f"== {f}: {len(df)} rows x {df.shape[1]} cols")
    for c in df.columns:
        try:
            n = df[c].nunique()
        except TypeError:
            n = -1
        print(f"   {c:28} {str(df[c].dtype):10} unique={n}")
    for c in df.columns:
        try:
            n = df[c].nunique()
        except TypeError:
            continue
        if c in a.counts or 1 < n <= a.max_card:
            vc = df[c].value_counts().head(30)
            print(f"-- {c}: " + ", ".join(f"{k}={v}" for k, v in vc.items()))
    for i in range(min(a.sample, len(df))):
        print("-- sample:", {k: (str(v)[:80]) for k, v in df.iloc[i].to_dict().items()})
