#!/usr/bin/env python3
"""Print the first lines of text files (docstrings, markdown summaries, logs) with a hard cap.

usage: peek.py FILE [FILE ...] [-n 30] [--grep REGEX]
  --grep: print only matching lines (with line numbers) instead of the head.
stdlib only.
"""
import argparse, re

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("-n", type=int, default=30)
ap.add_argument("--grep")
a = ap.parse_args()
for f in a.files:
    print(f"== {f}")
    try:
        with open(f, errors="replace") as h:
            shown = 0
            for i, line in enumerate(h, 1):
                if a.grep and not re.search(a.grep, line):
                    continue
                print(f"{i:5}: {line.rstrip()[:200]}" if a.grep else line.rstrip()[:200])
                shown += 1
                if shown >= a.n:
                    print("  …"); break
    except OSError as e:
        print(f"  {e}")
