"""Free 2D rigid fit (yaw about z + xy shift + z shift) of trace boxes onto detector-frame GT boxes, per scene.
Reference frame for the fit = 'aligned+cam0' (ScanNet axis-aligned axes, origin at the camera of posed image 00000).
A scene's fit uses labels that occur exactly once in both sets; reported: yaw, residual origin shift, residual error."""
import gzip, json, pickle, re, sys
from collections import defaultdict, Counter
import numpy as np
D = '/scr/xinyu/projects/vggtdet_es56/data/scannet/'
REN = {'garbagebin': 'trash bin', 'showercurtrain': 'shower curtain'}
def load(f):
    d = pickle.load(open(D + f, 'rb')); names = {v: k for k, v in d['metainfo']['categories'].items()}; out = {}
    for x in d['data_list']:
        sid = x['lidar_points']['lidar_path'][:-4]; ins = x.get('instances', [])
        out[sid] = dict(b=np.array([i['bbox_3d'][:6] for i in ins], float).reshape(-1, 6),
                        lab=[REN.get(names[i['bbox_label_3d']], names[i['bbox_label_3d']]) for i in ins],
                        A=np.array(x['axis_align_matrix'], float), P=np.array(x['lidar2cam'], float))
    return out
base = lambda s: re.sub(r'\s+\d+$', '', s.strip().lower())
def fit(tb, g, k=0):
    T = g['A'] @ g['P'][k]; C = g['b'][:, :3] - T[:3, 3]
    ct, cg = Counter(l for l, _, _ in tb), Counter(g['lab'])
    P, Q = [], []
    for lab, c, s in tb:
        if ct[lab] == 1 and cg.get(lab) == 1: P.append(C[g['lab'].index(lab)]); Q.append(c)
    if len(P) < 4: return None
    P, Q = np.array(P), np.array(Q); p0, q0 = P.mean(0), Q.mean(0)
    H = (P[:, :2] - p0[:2]).T @ (Q[:, :2] - q0[:2]); th = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    t = q0[:2] - R @ p0[:2]                       # Q = R P + t  -> t = where the cam0 origin lands in the trace frame
    res = np.linalg.norm((P[:, :2] @ R.T + t) - Q[:, :2], axis=1)
    camyaw = np.degrees(np.arctan2(T[1, 2], T[0, 2]))   # heading of camera-0 optical axis in the aligned frame
    return np.degrees(th), np.linalg.norm(t), q0[2] - p0[2], np.median(res), len(P), camyaw
def report(name, SB, G):
    F = {s: fit(tb, G[s]) for s, tb in SB.items() if s in G}; F = {s: f for s, f in F.items() if f}
    a = np.array(list(F.values()))
    print(f'\n== {name}: {len(F)} scenes fitted (>=4 unique-label pairs)')
    print(f'  residual after fit: median {np.median(a[:,3]):.3f} m, 90th pct {np.percentile(a[:,3],90):.3f} m')
    ok = a[a[:, 3] < 0.10]; print(f'  scenes with a clean fit (residual < 0.10 m): {len(ok)}')
    y = ok[:, 0]; q = np.round(y / 90) * 90
    print(f'  yaw vs ScanNet axis-aligned axes: nearest multiple of 90 deg ->', dict(sorted(Counter(q.astype(int) % 360).items())))
    print(f'    deviation from that multiple: median |d| {np.median(np.abs(y-q)):.2f} deg, 95th pct {np.percentile(np.abs(y-q),95):.2f} deg')
    print(f'  origin vs camera of posed image 00000: xy offset median {np.median(ok[:,1]):.3f} m, 90th pct {np.percentile(ok[:,1],90):.3f} m, max {ok[:,1].max():.2f} m; z offset median {np.median(ok[:,2]):+.3f} m, 90th pct |dz| {np.percentile(np.abs(ok[:,2]),90):.3f} m')
    d = (ok[:, 0] - ok[:, 5] + 180) % 360 - 180
    print(f'  yaw minus camera-0 heading (if the frame followed the camera, this would be constant): std {d.std():.0f} deg, spread of nearest-90 bins', dict(sorted(Counter((np.round(d/90)*90).astype(int) % 360).items())))
    return F
G = {**load('scannet_infos_train_pts.pkl'), **load('scannet_infos_val_pts.pkl')}
for f in ('scannet_infos_train_es56.pkl', 'scannet_infos_val_es56.pkl'):
    for s, g in load(f).items(): G[s] = g           # ES-56 boxes (more labels) where available
print('scenes with detector GT:', len(G))
tr = defaultdict(dict); src = {}
with gzip.open('../../data/simplesr_train_traces.jsonl.gz', 'rt') as f:
    for l in f:
        r = json.loads(l)
        if r['meta']['dataset'] != 'scannet': continue
        m = re.search(r'<scene>\n(.*?)\n</scene>', r['conversations'][1]['value'], re.S)
        for o in json.loads(m.group(1)): tr[r['meta']['scene']][(base(o['label']), tuple(o['bbox_3d']))] = 1
rb = {s: [(l, np.array(b[:3]), np.array(b[3:6])) for (l, b) in v] for s, v in tr.items()}
F1 = report('Reason-190K traces (ScanNet train scenes)', rb, G)
vb = defaultdict(dict)
for l in open('../../data/scenelang_vsibench_gt_firstcam/rows_all5130.jsonl'):
    r = json.loads(l)
    if r['dataset'] != 'scannet': continue
    for o in json.loads(r['scene_language']): vb[r['scene_name']][(base(o['label']), tuple(o['bbox_3d']))] = 1
vbx = {s: [(l, np.array(b[:3]), np.array(b[3:6])) for (l, b) in v] for s, v in vb.items()}
print('\nvsi-bench scannet scenes', len(vbx), 'with detector GT', sum(s in G for s in vbx))
F2 = report('VSI-Bench GT, first-camera file rows_all5130.jsonl (ScanNet val scenes)', vbx, G)
json.dump({'reason190k': F1, 'vsibench': F2}, open('fit_scannet.json', 'w'))
