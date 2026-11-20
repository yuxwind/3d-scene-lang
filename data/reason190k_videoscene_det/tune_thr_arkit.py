"""Merge the ARKitScenes detector exports into det_arkitscenes_seed0.json and pick a score threshold per class:
best F1 at IoU 0.25 against the 3dod boxes (yaw dropped, trace frame) stored in the detector infos. Classes without
ARKitScenes ground truth keep 0.5. Also prints, per class, coverage / precision at the chosen threshold and the
centre error and size ratio of matched boxes.   python tune_thr_arkit.py  (-> thr_arkitscenes.json)
"""
import json, pickle, numpy as np
from collections import defaultdict
from build_mix import ESL31
A = '/scr/xinyu/projects/vggtdet_es56/data/arkit_r190k/'


def iou(a, b):
    if len(a) == 0 or len(b) == 0: return np.zeros((len(a), len(b)))
    amin, amax, bmin, bmax = a[:, :3] - a[:, 3:] / 2, a[:, :3] + a[:, 3:] / 2, b[:, :3] - b[:, 3:] / 2, b[:, :3] + b[:, 3:] / 2
    inter = np.clip(np.minimum(amax[:, None], bmax[None]) - np.maximum(amin[:, None], bmin[None]), 0, None).prod(-1)
    return inter / (a[:, 3:].prod(-1)[:, None] + b[:, 3:].prod(-1)[None] - inter + 1e-9)


det = {**json.load(open('det_arkit_main.json')), **json.load(open('det_arkit_extra45.json'))}
json.dump(det, open('det_arkitscenes_seed0.json', 'w'))
infos = {}
for f in ('arkit_infos_reason190k_esl31.pkl', 'arkit_infos_arkit_extra45.pkl'):
    for i in pickle.load(open(A + f, 'rb'))['data_list']: infos[i['lidar_points']['lidar_path'][:-4]] = i
print(len(det), 'scenes with detections,', len(infos), 'infos')
rec = defaultdict(list); npos = defaultdict(int); err = defaultdict(list); szr = defaultdict(list)
for s, v in det.items():
    G = np.array([i['bbox_3d'] for i in infos[s]['instances']]).reshape(-1, 6); gl = np.array([ESL31[i['bbox_label_3d']] for i in infos[s]['instances']])
    B = np.array(v['boxes']).reshape(-1, 6); sc = np.array(v['scores']); lab = np.array(v['labels'])
    for c in set(gl) | set(lab):
        g, m = G[gl == c], np.nonzero(lab == c)[0]; npos[c] += len(g)
        I = iou(B[m], g); used = set()
        for j in np.argsort(-sc[m]):                       # greedy, best score first
            k = int(I[j].argmax()) if len(g) else -1
            hit = k >= 0 and I[j, k] >= 0.25 and k not in used
            if hit:
                used.add(k); err[c].append(np.linalg.norm(B[m[j], :3] - g[k, :3])); szr[c].append(B[m[j], 3:] / g[k, 3:])
            rec[c].append((sc[m[j]], hit))
thr = {}
print(f"{'class':14s}{'GT':>7s}{'thr':>6s}{'cover':>7s}{'prec':>7s}{'F1':>6s}{'ctr err m':>10s}  size ratio w d h (matched, IoU>=.25)")
for c in ESL31:
    if npos[c] == 0: continue
    r = np.array(rec[c]) if rec[c] else np.zeros((0, 2)); best = (0, 0.5, 0, 0)
    for t in np.arange(0.05, 0.96, 0.05):
        k = r[:, 0] >= t; tp = r[k, 1].sum(); p = tp / max(k.sum(), 1); rc = tp / npos[c]; f = 2 * p * rc / max(p + rc, 1e-9)
        if f > best[0]: best = (f, float(round(t, 2)), rc, p)
    thr[c] = best[1]
    print(f"{c:14s}{npos[c]:7d}{best[1]:6.2f}{best[2]:7.3f}{best[3]:7.3f}{best[0]:6.3f}{np.median(err[c]) if err[c] else float('nan'):10.2f}  {np.round(np.median(np.array(szr[c]), 0), 2) if szr[c] else ''}")
json.dump(thr, open('thr_arkitscenes.json', 'w')); print(thr)
for t in (0.3, 0.5):
    print(f'boxes/scene at score >= {t}:', np.mean([sum(x >= t for x in v['scores']) for v in det.values()]).round(1), end=' | ')
print('GT (12 classes)/scene:', np.mean([len(i['instances']) for i in infos.values()]).round(1))
