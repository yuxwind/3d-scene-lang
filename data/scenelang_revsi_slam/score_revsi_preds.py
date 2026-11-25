"""Score eval_scenelang.py predictions on ReVSI rows (revsi_rows.py) the ReVSI way: answer extraction as eval_scenelang.py
(last option letter for multiple choice, last number otherwise), accuracy / MRA (.50-.95), the 13 question types pooled into
7 tasks, overall = mean of the 7 tasks; also the mean over the 13 types, per source dataset, and video-only vs list rows.
    python3 score_revsi_preds.py <rows.jsonl> <run dir with predictions_shard*.jsonl> [...more run dirs]
    env IDS=<file of question ids, one per line>: score only those questions (e.g. revsi_score_answers.py --write-ids)"""
import json, glob, os, re, sys, collections
import numpy as np
LETTER = re.compile(r"\b([A-D])\b(?![a-z'])"); NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
T = ['object_abs_distance', 'object_counting', 'object_rel_direction', 'object_rel_distance', 'object_size_estimation', 'room_size_estimation', 'route_planning']
task = lambda t: re.sub(r'_(forward|backward)_(easy|hard)$|_(closest|farthest|single|multiple)$', '', t)
rows = {int(json.loads(l)['index']): json.loads(l) for l in open(sys.argv[1])}
def sc(r, p):
    p = p.split('</think>')[-1]
    if r['options'] is not None: m = LETTER.findall(p); return float(bool(m) and m[-1] == r['ground_truth'])
    m = NUMBER.findall(p.replace(',', ''))
    if not m: return 0.0
    e = abs(float(m[-1]) - float(r['ground_truth'])) / max(abs(float(r['ground_truth'])), 1e-9)
    return float(np.mean([e < 1 - t for t in np.arange(0.5, 0.96, 0.05)]))
mean7 = lambda s: 100 * np.mean([np.mean(s[t]) for t in T if s[t]])
for d in sys.argv[2:]:
    P = {}
    for f in glob.glob(f'{d}/predictions_shard*.jsonl'):
        for l in open(f): x = json.loads(l); P[int(x['index'])] = x['prediction']
    if os.environ.get('IDS'): keep = {int(l) for l in open(os.environ['IDS']) if l.strip()}; P = {i: p for i, p in P.items() if i in keep}
    s, sq, sd, su = (collections.defaultdict(list) for _ in range(4)); sdd = collections.defaultdict(lambda: collections.defaultdict(list))
    for i, p in P.items():
        r = rows[i]; x = sc(r, str(p)); s[task(r['revsi_type'])].append(x); sq[r['revsi_type']].append(x); sdd[r['dataset']][task(r['revsi_type'])].append(x)
        su['list' if r.get('use_scene', True) else 'video only'].append(x)
    print(f"{d.rstrip('/').split('/')[-1]}: {len(P)}/{len(rows)} answers | overall (7 tasks) {mean7(s):.1f} | mean of {len(sq)} types {100 * np.mean([np.mean(v) for v in sq.values()]):.1f}")
    print('   ' + ' | '.join(f"{t} {100 * np.mean(s[t]):.1f}" for t in T if s[t]))
    print('   ' + ' | '.join(f"{k} {mean7(v):.1f} (n{sum(map(len, v.values()))})" for k, v in sorted(sdd.items())))
    print('   ' + ' | '.join(f"{k}: n{len(v)} mean {100 * np.mean(v):.1f}" for k, v in su.items()))
