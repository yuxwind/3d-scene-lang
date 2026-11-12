"""VSI-Bench GT scenes restricted to the objects a 31-class VGGT-Det (ESL31) could output.

Reads ../scenelang_vsibench_gt_firstcam/rows_all5130.jsonl and keeps, per row, only the GT boxes whose label is
one of the detector's 31 classes or a synonym of one (SYN). Labels keep their VSI-Bench spelling, which is what
the questions use. Writes rows_all5130_esl31.jsonl and prints what was kept and dropped.
"""
import json
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parent
ESL31 = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
         'curtain', 'refrigerator', 'toilet', 'sink', 'bathtub', 'trash bin', 'pillow', 'monitor', 'backpack',
         'printer', 'lamp', 'mirror', 'tv', 'radiator', 'nightstand', 'plant', 'microwave', 'fan', 'blackboard',
         'stool']
SYN = {'trash can': 'trash bin', 'heater': 'radiator', 'whiteboard': 'blackboard', 'table lamp': 'lamp',
       'exhaust fan': 'fan'}

kept, dropped, seen, nq = Counter(), Counter(), set(), Counter()
with open(D / 'rows_all5130_esl31.jsonl', 'w') as f:
    for line in open(D.parent / 'scenelang_vsibench_gt_firstcam/rows_all5130.jsonl'):
        r = json.loads(line)
        objs = json.loads(r['scene_language'])
        keep = [o for o in objs if SYN.get(o['label'], o['label']) in ESL31]
        if (r['dataset'], r['scene_name']) not in seen:
            seen.add((r['dataset'], r['scene_name']))
            for o in objs:
                (kept if o in keep else dropped)[o['label']] += 1
        nq['empty scene' if not keep else 'ok'] += 1
        r['scene_language'] = '[\n' + ',\n'.join(json.dumps(o) for o in keep) + '\n]'
        f.write(json.dumps(r) + '\n')
print('scenes', len(seen), 'rows', dict(nq))
print('kept', sum(kept.values()), sorted(kept.items(), key=lambda x: -x[1]))
print('dropped', sum(dropped.values()), sorted(dropped.items(), key=lambda x: -x[1]))
