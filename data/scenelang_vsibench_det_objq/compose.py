"""Per-task VSI-Bench scores of several runs and a composite that takes each task from a chosen run.
    python compose.py <run>[,<run>...] <task>=<run> ... [default=<run>]
A run is a name under runs/eval_q35vs_gandalf/<name>_rvs_question, or 'video' = the video-only model
(runs/sr_vggt_novggt_qwen35_4b/eval_checkpoint-743_full/predictions_final_firstline.jsonl).
Tasks: appearance, abs_distance, counting, rel_direction, rel_distance, size, room_size, route."""
import json, glob, re, sys
sys.path.insert(0, '.')
from sr_vggt.eval import score
NUMBER = re.compile(r"-?\d+(?:\.\d+)?"); NUM = ('object_abs_distance', 'object_counting', 'object_size_estimation', 'room_size_estimation')
T = {'appearance': 'obj_appearance_order', 'abs_distance': 'object_abs_distance', 'counting': 'object_counting', 'rel_direction': 'object_rel_direction', 'rel_distance': 'object_rel_distance',
     'size': 'object_size_estimation', 'room_size': 'room_size_estimation', 'route': 'route_planning'}
def load(n):
    fs = ['runs/sr_vggt_novggt_qwen35_4b/eval_checkpoint-743_full/predictions_final_firstline.jsonl'] if n == 'video' else glob.glob(f'runs/eval_q35vs_gandalf/{n}_rvs_question/predictions_shard*.jsonl'); d = {}
    for f in fs:
        for l in open(f):
            x = json.loads(l); m = NUMBER.findall(x['prediction'].replace(',', ''))
            if m and x['question_type'] in NUM: x['prediction'] = m[-1]
            d[x['index']] = x
    return d
deb = {json.loads(l)['index'] for l in open('data/scenelang_vsibench_slam/rows_sam3_all5130_full.jsonl') if json.loads(l).get('debiased')}
runs = sys.argv[1].split(','); choice = dict(a.split('=') for a in sys.argv[2:]); R = {n: load(n) for n in set(runs) | set(choice.values())}
def per_task(d, sub=None):
    out = {}
    for t, p in T.items():
        rows = [v for k, v in d.items() if v['question_type'].startswith(p) and (sub is None or k in sub)]; s = score(rows); out[t] = 100 * next(v for k, v in s.items() if p in k) if rows else float('nan')
    return out
P = {n: (per_task(d), per_task(d, deb)) for n, d in R.items()}
print(f"{'run':34s} {'full':>5s} {'deb.':>5s}  " + ' '.join(f'{t[:9]:>9s}' for t in T))
for n in runs:
    f, b = P[n]; print(f"{n.replace('mixv2_1_', ''):34s} {sum(f.values()) / 8:5.1f} {sum(b.values()) / 8:5.1f}  " + ' '.join(f'{f[t]:9.1f}' for t in T))
if choice:
    src = {t: choice.get(t, choice.get('default')) for t in T}; f = {t: P[src[t]][0][t] for t in T}; b = {t: P[src[t]][1][t] for t in T}
    print(f"{'composite':34s} {sum(f.values()) / 8:5.1f} {sum(b.values()) / 8:5.1f}  " + ' '.join(f'{f[t]:9.1f}' for t in T)); print('  sources:', {t: src[t].replace('mixv2_1_', '') for t in T})
