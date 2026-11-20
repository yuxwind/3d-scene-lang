import json, glob, re, sys
sys.path.insert(0, '.')
from sr_vggt.eval import score
NUMBER = re.compile(r"-?\d+(?:\.\d+)?"); NUM = ('object_abs_distance', 'object_counting', 'object_size_estimation', 'room_size_estimation')
def load(n):
    d = {}
    for f in glob.glob(f'runs/eval_q35vs_gandalf/{n}_rvs_question/predictions_shard*.jsonl'):
        for l in open(f):
            x = json.loads(l); m = NUMBER.findall(x['prediction'].replace(',', ''))
            if m and x['question_type'] in NUM: x['prediction'] = m[-1]
            d[x['index']] = x
    return d
deb = {json.loads(l)['index'] for l in open('data/scenelang_vsibench_slam/rows_sam3_all5130_full.jsonl') if json.loads(l).get('debiased')}
R = {n: load(n) for n in sys.argv[1:]}
keys = None
for n, d in R.items():
    f = score(list(d.values())); b = score([v for k, v in d.items() if k in deb]); keys = keys or sorted(k for k in f if k != 'vsibench/overall')
    print(f"{n:28s} n={len(d)} full {100*f['vsibench/overall']:.1f} debiased {100*b['vsibench/overall']:.1f} | " + ' '.join(f"{k.split('/')[1].replace('_accuracy','').replace('_mra','')[:14]} {100*f[k]:.1f}" for k in keys))
