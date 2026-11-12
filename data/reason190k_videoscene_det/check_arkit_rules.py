import json, sys, numpy as np, csv, decord
from arkit_common import *
root = '/scr/xinyu/data/arkitscenes_reason190k/Training'
F = json.load(open('frames_arkit.json')); sky = {r['video_id']: r['sky_direction'] for r in csv.DictReader(open(root + '/../metadata.csv'))}
d, n, turn = [], 0, {}
import collections; tab = collections.Counter()
for i, (s, v) in enumerate(sorted(F.items())):
    ts, c2w = traj(f'{root}/{s}/lowres_wide.traj'); C, L, A, lab = boxes(f'{root}/{s}/{s}_3dod_annotation.json')
    q, R, up = upright_turn(c2w); tab[(sky.get(s), q)] += 1
    if v['fitted'] and v['inliers'] >= 5 and v['inliers'] / v['trace_boxes'] >= 0.8:
        th = room_theta(A, L, c2w[0, :2, 2]); e = np.degrees((th - v['theta'] + np.pi) % (2 * np.pi) - np.pi); d.append(e)
d = np.array(d); print('rule vs fitted theta on', len(d), 'reliable scenes: |diff| deg p50 p90 p95 p99:', np.percentile(np.abs(d), [50, 90, 95, 99]).round(2),
      '| within 2 deg: %.3f, within 5: %.3f, off by ~90k: %.3f' % ((np.abs(d) < 2).mean(), (np.abs(d) < 5).mean(), (np.abs(d) > 45).mean()))
print('sky_direction vs quarter turns:', sorted(tab.items(), key=lambda x: -x[1]))
for s in ['40777060', '43896321', '42897490', '47332398']:
    ts, c2w = traj(f'{root}/{s}/lowres_wide.traj'); print(s, sky[s], upright_turn(c2w)[0], decord.VideoReader(f'/scr/xinyu/data/VSI-590K/arkitscenes/{s}.mp4')[0].shape)
