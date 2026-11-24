"""Real scale (s_gt) vs estimated scale (class size priors) for the SAM 3 + VGGT-SLAM lists, Exp1 model (objq checkpoint-569).
  1. per-task scores of the 4 runs: exp1_s3q_rr_{gt_,}{mixv2,full}  (gt_ = real scale; mixv2 = video only for route / room / no
     named object; full = list on every question)
  2. do the two scales give the same boxes? per question: same labels, same number of boxes, same scores (geometry aside);
     counting: boxes of the named label in each list, and the model answer on each
    python3 scale_compare.py"""
import json, re, glob, collections
import numpy as np
D = '/scr/xinyu/projects/3d-scene-lang/'; NUMBER = re.compile(r"-?\d+(?:\.\d+)?"); LETTER = re.compile(r"\b([A-D])\b(?![a-z'])")
SYN = {'trash can': 'trash bin', 'heater': 'radiator', 'whiteboard': 'blackboard', 'table lamp': 'lamp', 'exhaust fan': 'fan', 'washing machine': 'washer'}
c = lambda l: SYN.get(l, l)
def mra(p, g):
    if p is None: return 0.0
    e = abs(p - g) / abs(g); return float(np.mean([e < 1 - t for t in np.arange(0.5, 0.96, 0.05)]))
def raw(run):
    P = {}
    for f in glob.glob(f'{D}runs/eval_q35vs_gandalf/{run}_rvs_question/predictions_shard*.jsonl'):
        for l in open(f): x = json.loads(l); P[int(x['index'])] = str(x['prediction']).split('</think>')[-1]
    return P
def score(r, p):
    if p is None: return 0.0
    if r['options']: m = LETTER.findall(p); return float(bool(m) and m[-1] == r['ground_truth'])
    m = NUMBER.findall(p.replace(',', '')); return mra(float(m[-1]) if m else None, float(r['ground_truth']))
load = lambda f: {json.loads(l)['index']: json.loads(l) for l in open(D + 'data/scenelang_vsibench_slam/' + f)}
EST, REAL = load('rows_s3q_roomdoor_radiator_all5130_full.jsonl'), load('rows_s3q_roomdoor_radiator_gtscale_all5130_full.jsonl')
EM, RM = load('rows_s3q_roomdoor_radiator_mixv2_all5130_full.jsonl'), load('rows_s3q_roomdoor_radiator_gtscale_mixv2_all5130_full.jsonl')
RUNS = {'real mixv2': 'exp1_s3q_rr_gt_mixv2', 'est mixv2': 'exp1_s3q_rr_mixv2', 'real full': 'exp1_s3q_rr_gt_full', 'est full': 'exp1_s3q_rr_full'}
P = {k: raw(v) for k, v in RUNS.items()}
T = sorted({r['question_type'] for r in EST.values()})
print('== 1. per task (MRA / accuracy x100); n answers per run:', {k: len(v) for k, v in P.items()})
print(f"  {'task':28s} {'n':>5s} " + ' '.join(f'{k:>11s}' for k in RUNS))
S = {k: {t: [score(EST[i], P[k].get(i)) for i in EST if EST[i]['question_type'] == t] for t in T} for k in RUNS}
for t in T: print(f"  {t:28s} {len(S['est full'][t]):5d} " + ' '.join(f"{100 * np.mean(S[k][t]):11.1f}" for k in RUNS))
print(f"  {'overall (mean of tasks)':28s} {'':5s} " + ' '.join(f"{100 * np.mean([np.mean(S[k][t]) for t in T]):11.1f}" for k in RUNS))
# 2. boxes
print('\n== 2. lists, real vs estimated scale (full rows, same questions)')
lab = lambda r: collections.Counter(c(o['label']) for o in json.loads(r['scene_language'])) if r['scene_language'] else collections.Counter()
sig = lambda r: sorted((c(o['label']), o.get('score')) for o in json.loads(r['scene_language'])) if r['scene_language'] else []
same_lab = sum(lab(EST[i]) == lab(REAL[i]) for i in EST); same_sig = sum(sig(EST[i]) == sig(REAL[i]) for i in EST)
nb = [(sum(lab(EST[i]).values()), sum(lab(REAL[i]).values())) for i in EST]
print(f"  questions {len(EST)}; same label multiset {same_lab}; same (label, score) multiset {same_sig}; boxes per list est {np.mean([a for a, _ in nb]):.2f} / real {np.mean([b for _, b in nb]):.2f}")
print(f"  use_scene identical between est and real mixv2 rows: {sum(EM[i]['use_scene'] == RM[i]['use_scene'] for i in EM)}/{len(EM)}")
diff = [i for i in EST if lab(EST[i]) != lab(REAL[i])]
if diff:
    scn = collections.Counter(EST[i]['scene_name'] for i in diff); print(f"  differing questions span {len(scn)} scenes; by type:", dict(collections.Counter(EST[i]['question_type'] for i in diff)))
    for i in diff[:3]:
        a, b = lab(EST[i]), lab(REAL[i]); print(f"    {EST[i]['scene_name']} q{i}: est-only {dict(a - b)} real-only {dict(b - a)}")
# counting
print('\n  counting questions:')
cnt = collections.Counter(); ex = []
for i, r in EST.items():
    if r['question_type'] != 'object_counting': continue
    L = re.search(r'How many (.+?)\(s\)', r['question']).group(1).strip(); g = int(float(r['ground_truth']))
    ne, nr = (sum(c(o['label']) == c(L) for o in json.loads(x['scene_language'])) for x in (r, REAL[i]))
    cnt['same box count' if ne == nr else 'est more' if ne > nr else 'real more'] += 1
    for k in ('mixv2', 'full'):
        pe, pr = P['est ' + k].get(i), P['real ' + k].get(i)
        if pe != pr: cnt[f'answer differs ({k})'] += 1; ex.append((k, i, ne, nr, g, pe, pr)) if ne == nr else None
print('   ', dict(cnt))
for k, i, ne, nr, g, pe, pr in ex[:8]: print(f"    {k:5s} q{i} boxes est {ne} real {nr} GT {g}: answer est {pe.strip()!r} real {pr.strip()!r}")
