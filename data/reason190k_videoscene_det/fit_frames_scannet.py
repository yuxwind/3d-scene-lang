"""Per ScanNet scene of Reason-190K: the rigid map from the detector frame (the scan's axis-aligned frame) to the
trace frame (origin at the camera of mp4 frame 0, axes along the room, z up):  c_trace = Rz(k * 90 deg) @ c_det + t.
Found by voting: every (trace box, scan GT box) pair whose sizes agree after the turn votes for a translation; the
(k, t) with most votes wins and t is refined as the median over its inliers. Scan GT = ES-56 boxes where the scan
has them, else the ScanNet 18-class boxes (labels are not used, only centres and sizes).
    python fit_frames_scannet.py <simplesr_train_traces.jsonl.gz> <out.json>
"""
import gzip, json, pickle, re, sys
from collections import defaultdict, Counter
import numpy as np
D = '/scr/xinyu/projects/vggtdet_es56/data/scannet/'


def load(f):
    out = {}
    for x in pickle.load(open(D + f, 'rb'))['data_list']:
        out[x['lidar_points']['lidar_path'][:-4]] = np.array([i['bbox_3d'][:6] for i in x.get('instances', [])], float).reshape(-1, 6)
    return out


def fit(T, G, tol=0.03):
    best = None
    for k in range(4):
        a = k * np.pi / 2; R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]).round()
        gs = G[:, 3:6][:, [1, 0, 2]] if k % 2 else G[:, 3:6]
        ok = (np.abs(T[:, None, 3:6] - gs[None]) < tol).all(-1)                  # (nT, nG) size agreement
        ti, gi = np.nonzero(ok)
        if not len(ti): continue
        t = T[ti, :3] - G[gi, :3] @ R.T
        key = np.round(t / 0.05).astype(int); cnt = Counter(map(tuple, key)); (kb, v) = cnt.most_common(1)[0]
        inl = np.abs(t - np.array(kb) * 0.05).max(1) < 0.08
        tt = np.median(t[inl], 0); res = np.abs(t[inl] - tt).max(1)
        n_in = len(set(ti[inl]))
        if best is None or n_in > best[0]: best = (n_in, k, tt, float(np.median(res)))
    return best


def main(src, dst):
    G = {**load('scannet_infos_train_pts.pkl'), **load('scannet_infos_val_pts.pkl')}
    E = {**load('scannet_infos_train_es56.pkl'), **load('scannet_infos_val_es56.pkl')}
    tr = defaultdict(set)
    with gzip.open(src, 'rt') as f:
        for l in f:
            r = json.loads(l)
            if r['meta']['dataset'] != 'scannet': continue
            for o in json.loads(re.search(r'<scene>\n(.*?)\n</scene>', r['conversations'][1]['value'], re.S).group(1)):
                tr[r['meta']['scene']].add(tuple(o['bbox_3d']))
    out = {}
    for s, boxes in tr.items():
        T = np.array(sorted(boxes)); cands = [fit(T, g) for g in (E.get(s), G.get(s)) if g is not None and len(g)]
        cands = [c for c in cands if c]
        if not cands: continue
        n, k, t, res = max(cands, key=lambda c: c[0])
        out[s] = dict(k=int(k), t=[float(v) for v in t], inliers=int(n), trace_boxes=len(T), residual=res)
    json.dump(out, open(dst, 'w'))
    a = np.array([[v['inliers'], v['trace_boxes'], v['residual']] for v in out.values()])
    print(f'{len(out)} of {len(tr)} scenes fitted | inlier share of trace boxes: median {np.median(a[:,0]/a[:,1]):.2f}, 5th pct {np.percentile(a[:,0]/a[:,1],5):.2f} | '
          f'scenes with < 5 inliers: {(a[:,0] < 5).sum()}, with inlier share < 0.5: {(a[:,0]/a[:,1] < 0.5).sum()} | residual median {np.median(a[:,2])*100:.1f} cm')
    print('turn k (x90 deg):', dict(sorted(Counter(v['k'] for v in out.values()).items())))


if __name__ == '__main__':
    main(*sys.argv[1:])
