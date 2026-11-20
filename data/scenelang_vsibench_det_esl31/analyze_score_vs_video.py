import json, glob, re, sys, collections, numpy as np
sys.path.insert(0, '.')
from sr_vggt.eval import score, MCA_QUESTION_TYPES
ESL=set('cabinet bed chair sofa table door window bookshelf picture counter desk curtain refrigerator toilet sink bathtub pillow monitor backpack printer lamp mirror tv radiator nightstand plant microwave fan blackboard stool'.split())|{'trash bin','trash can','heater','whiteboard','table lamp','exhaust fan'}
LETTER = re.compile(r"\b([A-D])\b(?![a-z'])"); NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
def norm(x):
    p = x['prediction']; m = LETTER.findall(p) if x['question_type'] in MCA_QUESTION_TYPES else NUMBER.findall(p.replace(',', ''))
    x['prediction'] = m[-1] if m else p; return x
def load(ns):
    d = {}
    for n in ns:
        for f in glob.glob(f'runs/eval_q35vs_gandalf/{n}_rvs_question/predictions_shard*.jsonl'):
            for l in open(f): x = norm(json.loads(l)); d[x['index']] = x
    return d
P = {'video': {}, 'score_det': load(['score_det_all']), 'mixed_det': load(['mixdet_det_scannet_arkit', 'mixdet_det_scannetpp', 'mixdet_det_scannetpp_extra80']), 'score_gt': load(['score_gt_firstcam'])}
for l in open('runs/sr_vggt_novggt_qwen35_4b/eval_checkpoint-743_full/predictions_final_firstline.jsonl'):
    x = norm(json.loads(l)); P['video'][x['index']] = x
Q = {k: {i: float(score([x])['vsibench/overall']) for i, x in p.items()} for k, p in P.items()}
det = {json.loads(l)['index']: json.loads(l) for l in open('data/scenelang_vsibench_det_esl31/rows_det_all5130.jsonl')}
F = {}
for l in open('data/scenelang_vsibench_gt_firstcam/rows_all5130.jsonl'):
    r = json.loads(l); q = r['question'].lower(); G = json.loads(r['scene_language']); named = []
    for lab in sorted({o['label'] for o in G}, key=len, reverse=True):
        q2 = re.sub(r'\b' + re.escape(lab) + r'(s|es)?\b', ' # ', q)
        if q2 != q: q = q2; named.append(lab)
    if not named or any(n not in ESL for n in named): continue
    t = r['question_type']
    if any(k in t for k in ('appearance_order', 'route_planning', 'room_size')): continue
    D = json.loads(det[r['index']]['scene_language']); best, cerr, cnt_ok, serr = [], [], [], []
    for n in named:
        db = [o for o in D if o['label'] == n or (n == 'table' and o['label'] == 'desk')]; gb = [o for o in G if o['label'] == n]
        best.append(max([o['score'] for o in db], default=0.0)); cnt_ok.append(len(db) == len(gb))
        if len(gb) == 1 and db:
            g = np.array(gb[0]['bbox_3d'][:6]); j = int(np.argmin([np.linalg.norm(np.array(o['bbox_3d'][:3]) - g[:3]) for o in db]))
            cerr.append(float(np.linalg.norm(np.array(db[j]['bbox_3d'][:3]) - g[:3])))
            serr.append(float(abs(max(db[j]['bbox_3d'][3:6]) - max(g[3:6])) / max(g[3:6])))
        elif len(gb) == 1: cerr.append(9.0)
    F[r['index']] = dict(task=re.sub(r'_(easy|medium|hard)$', '', t), ds=r['dataset'], minscore=min(best), all_found=min(best) > 0, cnt_ok=all(cnt_ok),
                         cerr=max(cerr) if cerr else None, serr=max(serr) if serr else None)
def table(title, groups):
    print('\n== ' + title); print(f"{'group':34s} {'n':>5s} {'video':>6s} {'scoreDet':>8s} {'mixDet':>7s} {'scoreGT':>8s} {'det-video':>9s}")
    for name, idx in groups:
        if len(idx) < 20: continue
        m = {k: 100 * np.mean([Q[k][i] for i in idx]) for k in Q}
        print(f"{name:34s} {len(idx):5d} {m['video']:6.1f} {m['score_det']:8.1f} {m['mixed_det']:7.1f} {m['score_gt']:8.1f} {m['score_det'] - m['video']:+9.1f}")
I = list(F)
bins = [('named object not detected', lambda f: f['minscore'] == 0), ('min score < 0.5', lambda f: 0 < f['minscore'] < .5), ('0.5 - 0.7', lambda f: .5 <= f['minscore'] < .7),
        ('0.7 - 0.9', lambda f: .7 <= f['minscore'] < .9), ('>= 0.9', lambda f: f['minscore'] >= .9)]
table('fully covered, scene tasks: by lowest best-box score among named objects', [('all', I)] + [(n, [i for i in I if fn(F[i])]) for n, fn in bins])
for t in ('object_counting', 'object_abs_distance', 'object_size_estimation', 'object_rel_distance', 'object_rel_direction'):
    table(t, [('all', [i for i in I if F[i]['task'] == t])] + [(n, [i for i in I if F[i]['task'] == t and fn(F[i])]) for n, fn in bins])
for d in ('scannet', 'arkitscenes', 'scannetpp'):
    table(d, [('all', [i for i in I if F[i]['ds'] == d])] + [(n, [i for i in I if F[i]['ds'] == d and fn(F[i])]) for n, fn in bins])
table('oracle: by detector box quality (single-instance named objects)', [
    ('centre error < 0.15 m', [i for i in I if F[i]['cerr'] is not None and F[i]['cerr'] < .15]), ('0.15 - 0.3 m', [i for i in I if F[i]['cerr'] is not None and .15 <= F[i]['cerr'] < .3]),
    ('0.3 - 1 m', [i for i in I if F[i]['cerr'] is not None and .3 <= F[i]['cerr'] < 1]), ('> 1 m or missing', [i for i in I if F[i]['cerr'] is not None and F[i]['cerr'] >= 1])])
c = [i for i in I if F[i]['task'] == 'object_counting']
table('counting: detector box count == GT count?', [('count right', [i for i in c if F[i]['cnt_ok']]), ('count wrong', [i for i in c if not F[i]['cnt_ok']])])
s = [i for i in I if F[i]['task'] == 'object_size_estimation' and F[i]['serr'] is not None]
table('object size: detector longest-side error', [('< 10%', [i for i in s if F[i]['serr'] < .1]), ('10 - 25%', [i for i in s if .1 <= F[i]['serr'] < .25]), ('> 25%', [i for i in s if F[i]['serr'] >= .25])])
# how well the score predicts box quality
v = [(F[i]['minscore'], F[i]['cerr']) for i in I if F[i]['cerr'] is not None and F[i]['minscore'] > 0]
for n, fn in bins[1:]:
    e = [c for s_, c in v if fn(dict(minscore=s_))]; print(f'score {n:12s}: n={len(e):4d} median centre error {np.median(e):.2f} m, share < 0.3 m {np.mean(np.array(e) < .3):.2f}')
