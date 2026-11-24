"""Why is absolute distance only ~61 with predicted boxes? For each VSI-Bench abs-distance question (834) and each list:
  coverage   both named objects have a box / one / none (labels as the question words them)
  list dist  closest-point distance between the two named boxes (axis-aligned [x,y,z,w,d,h]; several boxes of a label: the pair
             with the highest scores = the first in the list; also 'min over instances'); its MRA against the GT answer is what a
             model that reads the list perfectly would score
  model      MRA of the model's answer; how close the answer is to the list distance (does the model follow the boxes?)
GT boxes (9-DoF, first-camera frame) are converted the same way (axis-aligned extent after yaw) as a reference.
    python3 absdist_analysis.py"""
import json, re, glob, collections, sys
import numpy as np
D = '/scr/xinyu/projects/3d-scene-lang/'
LISTS = {'S3 real (exp1_s3q_rr_gt_mixv2)': ('data/scenelang_vsibench_slam/rows_s3q_roomdoor_radiator_gtscale_mixv2_all5130_full.jsonl', 'exp1_s3q_rr_gt_mixv2'),
         'S3+IndoorDet refine (exp1_c2_refine_mixv2)': ('data/scenelang_vsibench_slam/rows_c2_refine_mixv2_all5130_full.jsonl', 'exp1_c2_refine_mixv2'),
         'S3 estimated (exp1_s3q_rr_mixv2)': ('data/scenelang_vsibench_slam/rows_s3q_roomdoor_radiator_mixv2_all5130_full.jsonl', 'exp1_s3q_rr_mixv2'),
         'GT boxes (reference, no model run)': ('data/scenelang_vsibench_gt_firstcam/rows_all5130.jsonl', None)}
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
def mra(p, g):
    if p is None: return 0.0
    e = abs(p - g) / g; return float(np.mean([e < 1 - t for t in np.arange(0.5, 0.96, 0.05)]))
def names(q):
    m = re.search(r'distance between the (.+?) and the (.+?) \(in meters\)', q); return m.groups() if m else None
def aabb(b):
    if len(b) == 9:                                            # GT: centre, size, yaw -> axis-aligned extent
        c, s, y = np.array(b[:3]), np.array(b[3:6]), b[6]; w = abs(s[0] * np.cos(y)) + abs(s[1] * np.sin(y)); d = abs(s[0] * np.sin(y)) + abs(s[1] * np.cos(y)); s = np.array([w, d, s[2]])
    else: c, s = np.array(b[:3]), np.array(b[3:6])
    return c - s / 2, c + s / 2
def gap(a, b):
    (l1, h1), (l2, h2) = aabb(a), aabb(b); return float(np.linalg.norm(np.maximum(0, np.maximum(l1 - h2, l2 - h1))))
SYN = {'trash can': 'trash bin', 'heater': 'radiator', 'whiteboard': 'blackboard', 'table lamp': 'lamp', 'exhaust fan': 'fan', 'washing machine': 'washer'}
def find(objs, n):
    hit = [o for o in objs if o['label'] == n] or [o for o in objs if SYN.get(o['label'], o['label']) == SYN.get(n, n)]
    return hit
def preds(run):
    P = {}
    for f in glob.glob(f'{D}runs/eval_q35vs_gandalf/{run}_rvs_question/predictions_shard*.jsonl'):
        for l in open(f):
            x = json.loads(l); m = NUMBER.findall(str(x['prediction']).replace(',', '')); P[int(x['index'])] = float(m[-1]) if m else None
    return P
VID = {}
for l in open(D + 'runs/sr_vggt_novggt_qwen35_4b/eval_checkpoint-743_full/predictions_final_firstline.jsonl'):
    x = json.loads(l); m = NUMBER.findall(str(x['prediction']).replace(',', '')); VID[int(x['index'])] = float(m[-1]) if m else None
for name, (rows_f, run) in LISTS.items():
    R = [json.loads(l) for l in open(D + rows_f)]; R = [r for r in R if r['question_type'] == 'object_abs_distance']
    P = preds(run) if run else {}
    st = collections.defaultdict(list); follow = []; ratio = []
    for r in R:
        g = float(r['ground_truth']); nm = names(r['question']); objs = json.loads(r['scene_language'])
        if not nm: st['unparsed'].append(1); continue
        A, B = find(objs, nm[0]), find(objs, nm[1]); cov = 'both' if A and B else 'one' if A or B else 'none'
        ld = gap(A[0]['bbox_3d'], B[0]['bbox_3d']) if A and B else None
        lmin = min(gap(a['bbox_3d'], b['bbox_3d']) for a in A for b in B) if A and B else None
        p = P.get(int(r['index'])); v = VID.get(int(r['index']))
        st['cov_' + cov].append(1)
        st['list_mra_' + cov].append(mra(ld, g) if ld is not None else 0.0)
        if ld is not None: st['list_mra_both_min'].append(mra(lmin, g)); ratio.append(np.log(max(ld, 0.01) / g)); st['multi_inst'].append(int(len(A) > 1 or len(B) > 1))
        if run:
            st['model_' + cov].append(mra(p, g)); st['video_' + cov].append(mra(v, g))
            if ld is not None and p is not None: follow.append(abs(p - ld) / max(ld, 0.05) < 0.15); st['model_vs_list_mra'].append(mra(p, max(ld, 0.01)))
    n = len(R); f = lambda k: 100 * np.mean(st[k]) if st[k] else float('nan')
    print(f'\n== {name}: {n} questions')
    print(f"  coverage: both named objects {len(st['cov_both'])} ({100 * len(st['cov_both']) / n:.0f}%), one {len(st['cov_one'])}, none {len(st['cov_none'])}; >1 box of a named label (both-covered rows) {f('multi_inst'):.0f}%")
    print(f"  distance read from the list (first box of each label), MRA vs GT: both-covered {f('list_mra_both'):.1f} | min over instances {f('list_mra_both_min'):.1f} | all rows (0 if not covered) {100 * np.mean(st['list_mra_both'] + st['list_mra_one'] + st['list_mra_none']):.1f}")
    if ratio: q = np.exp(np.percentile(ratio, [10, 25, 50, 75, 90])); print(f"  list distance / GT, percentiles 10/25/50/75/90: " + ' '.join(f'{x:.2f}' for x in q) + f"; within 25%: {100 * np.mean(np.abs(np.exp(ratio) - 1) < 0.25):.0f}%")
    if run:
        print(f"  model MRA: both-covered {f('model_both'):.1f} (video only {f('video_both'):.1f}) | one {f('model_one'):.1f} (video {f('video_one'):.1f}) | none {f('model_none'):.1f} (video {f('video_none'):.1f}) | all {100 * np.mean(st['model_both'] + st['model_one'] + st['model_none']):.1f}")
        print(f"  model answer within 15% of the list distance: {100 * np.mean(follow):.0f}% of both-covered rows; MRA of the answer against the list distance {f('model_vs_list_mra'):.1f}")
