"""Verify the box frames of Reason-190K traces and the VSI-Bench first-camera GT against the detector's GT
(ES-56 boxes in the ScanNet axis-aligned frame), on ScanNet scenes.
Hypotheses for a trace box centre c_t given an aligned-world centre c_w and posed image 00000 (cam2raw pose P0,
axis_align A):
  aligned+cam0 : c_t = c_w - (A P0)[:3,3]                 (axes of the aligned scan, origin at camera 0)
  raw+cam0     : c_t = A^-1-rotated axes, origin camera 0  (axes of the raw scan)
  camera0      : c_t = inv(A P0) c_w                       (full camera-0 frame)
  aligned world: c_t = c_w
"""
import gzip, json, pickle, re, sys
from collections import defaultdict
import numpy as np
D = '/scr/xinyu/projects/vggtdet_es56/data/scannet/'
def load(split):
    d = pickle.load(open(D + f'scannet_infos_{split}_es56.pkl', 'rb'))
    names = {v: k for k, v in d['metainfo']['categories'].items()}
    out = {}
    for x in d['data_list']:
        sid = x['lidar_points']['lidar_path'][:-4]
        b = np.array([i['bbox_3d'] for i in x['instances']], float).reshape(-1, len(x['instances'][0]['bbox_3d']) if x['instances'] else 6)
        lab = [names[i['bbox_label_3d']] for i in x['instances']]
        out[sid] = dict(b=b, lab=lab, A=np.array(x['axis_align_matrix'], float), P0=np.array(x['lidar2cam'][0], float),
                        img0=x['img_paths'][0])
    return out
def hyps(g):
    A, P0 = g['A'], g['P0']; c = g['b'][:, :3]; T = A @ P0
    ch = np.c_[c, np.ones(len(c))]
    return {'aligned world': c, 'aligned+cam0': c - T[:3, 3],
            'raw+cam0': (np.linalg.inv(A) @ ch.T).T[:, :3] - P0[:3, 3],
            'camera0': (np.linalg.inv(T) @ ch.T).T[:, :3]}
base = lambda s: re.sub(r'\s+\d+$', '', s.strip().lower())
def score(tb, g, H):
    """tb: list of (label, centre, size). nearest same-label GT centre distance per hypothesis."""
    res = defaultdict(list); szs = []
    for lab, c, s in tb:
        idx = [i for i, l in enumerate(g['lab']) if l == lab]
        if not idx: continue
        for h, C in H.items():
            d = np.linalg.norm(C[idx] - c, axis=1); res[h].append(d.min())
            if h == 'aligned+cam0':
                j = idx[int(d.argmin())]
                if d.min() < 0.15: szs.append(np.r_[s, g['b'][j, 3:6]])
    return res, szs
def fit_yaw(tb, g):
    """residual yaw + xy shift of trace vs aligned+cam0 from matched pairs (Procrustes in xy)."""
    C = hyps(g)['aligned+cam0']; P, Q = [], []
    for lab, c, s in tb:
        idx = [i for i, l in enumerate(g['lab']) if l == lab]
        if len(idx) == 1 and sum(1 for t in tb if t[0] == lab) == 1: P.append(C[idx[0]]); Q.append(c)   # unique labels only
    if len(P) < 3: return None
    P, Q = np.array(P)[:, :2], np.array(Q)[:, :2]; p0, q0 = P.mean(0), Q.mean(0)
    H = (P - p0).T @ (Q - q0); th = np.degrees(np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1]))
    return th, np.linalg.norm(q0 - p0), len(P)
def report(name, scenes_boxes, G):
    agg = defaultdict(list); yaw = []; S = []
    for sid, tb in scenes_boxes.items():
        if sid not in G: continue
        r, szs = score(tb, G[sid], hyps(G[sid])); S += szs
        for h, v in r.items(): agg[h] += v
        f = fit_yaw(tb, G[sid])
        if f: yaw.append(f)
    print(f'\n== {name}: {sum(1 for s in scenes_boxes if s in G)} scenes, {len(next(iter(agg.values())))} boxes with a same-label GT box')
    for h, v in agg.items():
        v = np.array(v); print(f'  {h:14s} median centre distance {np.median(v):.3f} m | within 0.10 m {np.mean(v<0.1)*100:5.1f}% | within 0.30 m {np.mean(v<0.3)*100:5.1f}%')
    y = np.array(yaw)
    if len(y): print(f'  free fit vs aligned+cam0 ({len(y)} scenes): yaw median {np.median(y[:,0]):+.2f} deg, |yaw|<2deg in {np.mean(np.abs(y[:,0])<2)*100:.0f}% ; xy shift median {np.median(y[:,1]):.3f} m, 90th pct {np.percentile(y[:,1],90):.3f} m')
    S = np.array(S)
    if len(S): print(f'  size (w,d,h) trace/GT median ratio on matched boxes: {np.median(S[:,:3]/S[:,3:],0).round(3)}  (n={len(S)})')
G = {**load('train'), **load('val')}
print('detector GT box dims', next(iter(G.values()))['b'].shape)
# Reason-190K
tr = defaultdict(dict)
with gzip.open('../../data/simplesr_train_traces.jsonl.gz', 'rt') as f:
    for l in f:
        r = json.loads(l)
        if r['meta']['dataset'] != 'scannet': continue
        m = re.search(r'<scene>\n(.*?)\n</scene>', r['conversations'][1]['value'], re.S)
        for o in json.loads(m.group(1)):
            tr[r['meta']['scene']][(base(o['label']), tuple(o['bbox_3d']))] = 1
rb = {s: [(l, np.array(b[:3]), np.array(b[3:6])) for (l, b) in v] for s, v in tr.items()}
print('reason-190k scannet scenes', len(rb), 'in detector infos', sum(s in G for s in rb))
report('Reason-190K traces (ScanNet)', rb, G)
# VSI-Bench first-camera GT
vb = defaultdict(dict)
for l in open('../../data/scenelang_vsibench_gt_firstcam/rows_all5130.jsonl'):
    r = json.loads(l)
    if r['dataset'] != 'scannet': continue
    for o in json.loads(r['scene_language']): vb[r['scene_name']][(base(o['label']), tuple(o['bbox_3d']))] = 1
vbx = {s: [(l, np.array(b[:3]), np.array(b[3:6])) for (l, b) in v] for s, v in vb.items()}
print('vsi-bench scannet scenes', len(vbx), 'in detector infos', sum(s in G for s in vbx))
report('VSI-Bench GT, first-camera file (ScanNet)', vbx, G)
