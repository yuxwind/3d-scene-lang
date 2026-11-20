"""Exp1 check: coverage of the detector boxes on Reason-190K, cls-31 VGG-T-Det (det_<ds>_seed0.json, per-class thresholds, as in
mix-v2-1) against IndoorDet-65-31k (det_<ds>_objq.json, refine selection). Per question (every question with named objects, room
size excluded): named labels the detector's class set covers, and named labels that have at least one box of a matching class
in the scene ('found'). Per scene: boxes, and the share of the scene's named objects (over all its questions) that are found.
    python coverage_compare.py <simplesr_train_traces.jsonl.gz>"""
import gzip, json, re, sys, collections, numpy as np
sys.argv, src = [sys.argv[0]], sys.argv[1]
import build_mix as B
SC = re.compile(r"<scene>\n(\[.*?\])\n</scene>", re.S); S65 = B.syn65()
DET = {(t, ds): B.load_det(ds, 'thr_arkitscenes.json' if (t == 'seed0' and ds == 'arkitscenes') else None, t) for t in ('seed0', 'objq') for ds in ('scannet', 'arkitscenes', 'scannetpp')}
SYN = {'seed0': B.SYN, 'objq': S65}; Q = collections.defaultdict(collections.Counter); SCN = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
for l in gzip.open(src, 'rt'):
    r = json.loads(l); m = r['meta']; ds = m['dataset']
    if r['question_type'] == 'absolute_size_room': continue
    labs = sorted({re.sub(r'\s+\d+$', '', o['label']) for o in json.loads(SC.search(r['conversations'][1]['value']).group(1))})
    if not labs: continue
    for t in ('seed0', 'objq'):
        syn = SYN[t]; boxes = DET[(t, ds)].get(m['scene']); cls = {c for c, _, _ in boxes} if boxes else set()
        cov = [l for l in labs if l in syn]; fnd = [l for l in cov if cls & set(syn[l])]
        q = Q[(t, ds)]; q['questions'] += 1; q['labels'] += len(labs); q['covered'] += len(cov); q['found'] += len(fnd)
        q['q all covered'] += len(cov) == len(labs); q['q all found'] += len(fnd) == len(labs); q['q none found'] += not fnd; q['q scene has boxes'] += boxes is not None
        s = SCN[(t, ds)][m['scene']]; s[0] += len(labs); s[1] += len(fnd)
print(f"{'detector':9s} {'dataset':12s} {'questions':>9s} {'scene has boxes':>15s} {'labels covered':>14s} {'labels found':>12s} {'q: all found':>12s} {'q: none found':>13s} {'boxes/scene':>11s} {'scene found share (median)':>26s}")
for ds in ('scannet', 'arkitscenes', 'scannetpp'):
    for t in ('seed0', 'objq'):
        q = Q[(t, ds)]; n = q['questions']; b = DET[(t, ds)]; sh = [f / a for a, f in SCN[(t, ds)].values() if a]
        print(f"{'cls-31' if t == 'seed0' else 'IndoorDet':9s} {ds:12s} {n:9d} {100 * q['q scene has boxes'] / n:14.1f}% {100 * q['covered'] / q['labels']:13.1f}% {100 * q['found'] / q['labels']:11.1f}% {100 * q['q all found'] / n:11.1f}% {100 * q['q none found'] / n:12.1f}% {np.mean([len(v) for v in b.values()]):11.1f} {100 * np.median(sh):25.1f}%")
