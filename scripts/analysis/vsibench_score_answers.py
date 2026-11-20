#!/usr/bin/env python3
"""Score answer files against VSI-Bench GT rows (for runs that saved answers but no scores*.json).

Ported from vggtdet_es56/runs/claude_nocode_s3/score.py (2026-11-20). Same metric: accuracy for multiple choice
(first A-D letter), MRA over thresholds .50-.95 for numbers. overall = mean of the 8 task scores (rel. direction
easy/medium/hard pooled), on all rows ("full") and on rows with debiased=true ("debiased").

usage: vsibench_score_answers.py RUN_DIR --gt ROWS.jsonl [--variants v1 v2 ...] [--json OUT]
  RUN_DIR/<variant>/out/*.json   answer files: {"answers": {index: answer}, "tools_used": [...]}
  RUN_DIR/<variant>/batches/*.json  (only counted, for "x/y batches done")
  --variants: default = every subfolder of RUN_DIR that has out/.
  --gt: JSONL rows with index, question_type, options, ground_truth, debiased.
prints per variant: batches done, answers, full and debiased overall, per-task score (n). stdlib only.
"""
import argparse, collections, glob, json, os, re

T = ['obj_appearance_order', 'object_abs_distance', 'object_counting', 'object_rel_direction',
     'object_rel_distance', 'object_size_estimation', 'room_size_estimation', 'route_planning']
task = lambda t: re.sub('_(easy|medium|hard)$', '', t)
mean = lambda xs: sum(xs) / len(xs)


def score(r, a):
    if r['options']:
        m = re.findall(r'[A-D]', str(a).upper())
        return float(bool(m) and m[0] == r['ground_truth'])
    m = re.findall(r'-?\d+\.?\d*', str(a).replace(',', ''))
    if not m:
        return 0.0
    p, g = float(m[0]), float(r['ground_truth'])
    e = abs(p - g) / max(abs(g), 1e-9)
    return mean([float(e < 1 - (0.5 + 0.05 * i)) for i in range(10)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir'); ap.add_argument('--gt', required=True)
    ap.add_argument('--variants', nargs='*'); ap.add_argument('--json')
    a = ap.parse_args()
    R = {}
    with open(a.gt) as h:
        for l in h:
            if l.strip():
                r = json.loads(l); R[r['index']] = r
    vs = a.variants or sorted(d for d in os.listdir(a.run_dir) if os.path.isdir(os.path.join(a.run_dir, d, 'out')))
    out = {}
    for v in vs:
        D = os.path.join(a.run_dir, v)
        A, tools = {}, collections.Counter()
        outs = glob.glob(f'{D}/out/*.json'); nb = len(glob.glob(f'{D}/batches/*.json'))
        for f in outs:
            o = json.load(open(f)); tools[','.join(sorted(o.get('tools_used', []))) or 'none'] += 1
            A.update({int(k): x for k, x in o['answers'].items()})
        s, sd, miss = collections.defaultdict(list), collections.defaultdict(list), 0
        for i, x in A.items():
            r = R.get(i)
            if r is None:
                miss += 1; continue
            y = score(r, x); s[task(r['question_type'])].append(y)
            if r.get('debiased'):
                sd[task(r['question_type'])].append(y)
        full = 100 * mean([mean(s[t]) for t in T if s[t]]) if s else 0
        deb = 100 * mean([mean(sd[t]) for t in T if sd[t]]) if sd else 0
        print(f"{v}: {len(outs)}/{nb} batches, {len(A)}/{len(R)} answers{f', {miss} not in GT' if miss else ''} | "
              f"full {full:.1f} debiased {deb:.1f} | tools {dict(tools)}")
        print('   ' + ' | '.join(f"{t[:14]} {100 * mean(s[t]):.1f} (n{len(s[t])})" for t in T if s[t]))
        out[v] = {'n': len(A), 'overall': round(full, 1), 'overall_debiased': round(deb, 1),
                  **{t: round(100 * mean(s[t]), 1) for t in T if s[t]}}
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1); print(f"json: {a.json}")


if __name__ == '__main__':
    main()
