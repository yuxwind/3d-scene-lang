"""Gap between the SAM 3 + VGGT-SLAM list (real scale) and the GT scene on counting and object size (the two tasks that carry
most of it). Lists: rows_s3q_roomdoor_radiator_gtscale_mixv2 (S3 real) vs scenelang_vsibench_gt_firstcam/rows_all5130 (GT).
  counting  boxes of the named label in the S3 list vs the GT count (exact / too few / too many / none); model accuracy (MRA) per case
  size      longest side of the named object's box (first = best-scoring box) vs the GT answer: what reading the list gives; per case
Models: Exp1 on S3 real, mix-v2-1 on S3 real, mix-v2-1 on the GT list (gtfull_full).    python3 gap_analysis.py"""
import json, re, glob, collections
import numpy as np
D = '/scr/xinyu/projects/3d-scene-lang/'; NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
SYN = {'trash can': 'trash bin', 'heater': 'radiator', 'whiteboard': 'blackboard', 'table lamp': 'lamp', 'exhaust fan': 'fan', 'washing machine': 'washer'}
c = lambda l: SYN.get(l, l)
def mra(p, g):
    if p is None: return 0.0
    e = abs(p - g) / g; return float(np.mean([e < 1 - t for t in np.arange(0.5, 0.96, 0.05)]))
def preds(run):
    P = {}
    for f in glob.glob(f'{D}runs/eval_q35vs_gandalf/{run}_rvs_question/predictions_shard*.jsonl'):
        for l in open(f): x = json.loads(l); m = NUMBER.findall(str(x['prediction']).replace(',', '')); P[int(x['index'])] = float(m[-1]) if m else None
    return P
S = {json.loads(l)['index']: json.loads(l) for l in open(D + 'data/scenelang_vsibench_slam/rows_s3q_roomdoor_radiator_gtscale_mixv2_all5130_full.jsonl')}
G = {json.loads(l)['index']: json.loads(l) for l in open(D + 'data/scenelang_vsibench_gt_firstcam/rows_all5130.jsonl')}
M = {'Exp1 / S3 real': preds('exp1_s3q_rr_gt_mixv2'), 'mix-v2-1 / S3 real': preds('mixv2_1_s3q_roomdoor_radiator_gtscale'), 'mix-v2-1 / GT list': preds('mixv2_1_gtfull_full')}
def table(title, cases, order):
    n = sum(len(v) for v in cases.values()); print(f'\n== {title}: {n} questions')
    print(f"  {'case':38s} {'n':>5s} {'share':>6s} " + ' '.join(f'{m:>20s}' for m in M) + f" {'list read':>10s}")
    for k in order:
        v = cases.get(k, [])
        if not v: continue
        print(f"  {k:38s} {len(v):5d} {100 * len(v) / n:5.0f}% " + ' '.join(f"{100 * np.mean([mra(M[m].get(i), g) for i, g, _ in v]):20.1f}" for m in M) + f" {100 * np.mean([x for _, _, x in v]):10.1f}")
    print(f"  {'all':38s} {n:5d} {'':6s} " + ' '.join(f"{100 * np.mean([mra(M[m].get(i), g) for v in cases.values() for i, g, _ in v]):20.1f}" for m in M) + f" {100 * np.mean([x for v in cases.values() for _, _, x in v]):10.1f}")
# counting
cases = collections.defaultdict(list)
for i, r in S.items():
    if r['question_type'] != 'object_counting': continue
    lab = re.search(r'How many (.+?)\(s\)', r['question']).group(1).strip(); g = int(float(r['ground_truth']))
    n = sum(c(o['label']) == c(lab) for o in json.loads(r['scene_language']))
    k = 'no box' if n == 0 else 'exact' if n == g else f'too few' if n < g else 'too many'
    cases[k].append((i, g, mra(n, g)))
table('Counting: S3 boxes of the named label vs GT count', cases, ['exact', 'too few', 'too many', 'no box'])
miss = [g - sum(1 for _ in []) for _ in []]
# size
cases = collections.defaultdict(list); err = []
for i, r in S.items():
    if r['question_type'] != 'object_size_estimation': continue
    lab = re.search(r'height\) of the (.+?), measured', r['question']).group(1).strip(); g = float(r['ground_truth'])
    b = [o for o in json.loads(r['scene_language']) if c(o['label']) == c(lab)]
    if not b: cases['no box'].append((i, g, 0.0)); continue
    L = 100 * max(b[0]['bbox_3d'][3:6]); x = mra(L, g); err.append(L / g)
    cases['box, longest side within 25% of GT' if abs(L / g - 1) < 0.25 else 'box, too small (<0.75x)' if L < g else 'box, too large (>1.25x)'].append((i, g, x))
table('Object size: longest side of the best S3 box vs GT', cases, ['box, longest side within 25% of GT', 'box, too small (<0.75x)', 'box, too large (>1.25x)', 'no box'])
print('  box longest side / GT, percentiles 10/25/50/75/90:', ' '.join(f'{v:.2f}' for v in np.percentile(err, [10, 25, 50, 75, 90])))
