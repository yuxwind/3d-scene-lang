"""Per ARKitScenes scene of Reason-190K: the rigid map from the ARKit world frame (3dod annotation `obbAligned`,
lowres_wide.traj) to the trace frame:  c_trace = Rz(theta) @ c_arkit + t, theta any angle. Found by RANSAC on box
centres (pairs must agree in height), refined by least squares. Also reports where the traj cameras land in the
trace frame (the trace origin should be the camera of mp4 frame 0), and how trace sizes relate to the ARKit boxes.
    python fit_frames_arkit.py <simplesr_train_traces.jsonl.gz> <arkit root> <out.json>
"""
import gzip, json, re, sys, itertools
from collections import defaultdict
import numpy as np


def rodrigues(r):
    th = np.linalg.norm(r)
    if th < 1e-12: return np.eye(3)
    k = r / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def traj(path):  # -> timestamps, cam2world (N,4,4)
    a = np.loadtxt(path).reshape(-1, 7); out = []
    for l in a:
        E = np.eye(4); E[:3, :3] = rodrigues(l[1:4]); E[:3, 3] = l[4:7]; out.append(np.linalg.inv(E))
    return a[:, 0], np.array(out)


def rz(a): return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def solve(P, Q):  # 2D rigid Q ~ R P + t
    pc, qc = P.mean(0), Q.mean(0); H = (P - pc).T @ (Q - qc)
    a = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1]); return a, qc - rz(a) @ pc


def fit(T, C, L, tol=0.05):
    ok = np.abs(T[:, None, 5] - L[None, :, 2]) < 0.03
    pairs = list(zip(*np.nonzero(ok)))
    best = (0, None)
    for (i, j), (i2, j2) in itertools.combinations(pairs, 2):
        if i == i2 or j == j2: continue
        dT, dC = T[i2, :2] - T[i, :2], C[j2, :2] - C[j, :2]
        if abs(np.linalg.norm(dT) - np.linalg.norm(dC)) > tol or np.linalg.norm(dT) < 0.3: continue
        a = np.arctan2(dT[1], dT[0]) - np.arctan2(dC[1], dC[0]); t = T[i, :2] - rz(a) @ C[j, :2]
        d = np.linalg.norm(T[:, None, :2] - (C[:, :2] @ rz(a).T + t)[None], axis=2); d[~ok] = 9
        n = int((d.min(1) < tol).sum())
        if n > best[0]: best = (n, (a, t))
    if best[0] < 3: return None
    for _ in range(2):
        a, t = best[1]
        d = np.linalg.norm(T[:, None, :2] - (C[:, :2] @ rz(a).T + t)[None], axis=2); d[~ok] = 9
        ti = np.nonzero(d.min(1) < tol)[0]; gi = d.argmin(1)[ti]
        best = (len(ti), solve(C[gi, :2], T[ti, :2]))
    a, t = best[1]
    tz = float(np.median(T[ti, 2] - C[gi, 2]))
    res = np.linalg.norm(T[ti, :2] - (C[gi, :2] @ rz(a).T + t), axis=1)
    return a, np.array([t[0], t[1], tz]), ti, gi, float(np.median(res))


def main(src, root, dst):
    tr = defaultdict(set)
    with gzip.open(src, 'rt') as f:
        for l in f:
            r = json.loads(l)
            if r['meta']['dataset'] != 'arkitscenes': continue
            for o in json.loads(re.search(r'<scene>\n(.*?)\n</scene>', r['conversations'][1]['value'], re.S).group(1)):
                tr[r['meta']['scene']].add(tuple(o['bbox_3d']))
    out = {}
    for s, boxes in sorted(tr.items()):
        T = np.array(sorted(boxes))
        ann = json.load(open(f'{root}/{s}/{s}_3dod_annotation.json'))['data']
        ob = [d['segments']['obbAligned'] for d in ann]
        if len(ob) < 2 or len(T) < 3: out[s] = dict(fitted=False, trace_boxes=len(T), gt_boxes=len(ob)); continue
        C = np.array([o['centroid'] for o in ob]); L = np.array([o['axesLengths'] for o in ob])
        A = np.array([o['normalizedAxes'] for o in ob]).reshape(-1, 3, 3)
        f = fit(T, C, L)
        if f is None: out[s] = dict(fitted=False, trace_boxes=len(T), gt_boxes=len(ob)); continue
        a, t, ti, gi, res = f
        ts, c2w = traj(f'{root}/{s}/lowres_wide.traj')
        cam = c2w[:, :3, 3]; q = cam[:, :2] @ rz(a).T + t[:2]; dz = cam[:, 2] + t[2]
        dist = np.sqrt((q ** 2).sum(1) + dz ** 2); j = int(dist.argmin())
        fwd = rz(a) @ c2w[j, :2, 2]                                   # camera z (forward) in the trace frame
        yaw = np.arctan2(A[gi, 0, 1], A[gi, 0, 0]) + a                # box axis 0 direction in the trace frame (rows = axes)
        off = np.abs((np.degrees(yaw) + 45) % 90 - 45)
        # sizes: trace (w, d) vs box lengths as-is / swapped
        same = np.abs(T[ti, 3:5] - L[gi, :2]).max(1); swap = np.abs(T[ti, 3:5] - L[gi, 1::-1]).max(1)
        out[s] = dict(fitted=True, theta=float(a), t=t.tolist(), inliers=len(ti), trace_boxes=len(T), gt_boxes=len(ob), residual=res,
                      cam_idx=j, n_cam=len(ts), cam_dist=float(dist[j]), cam0_dist=float(dist[0]), cam_ts=float(ts[j]), ts0=float(ts[0]),
                      fwd_deg=float(np.degrees(np.arctan2(fwd[1], fwd[0]))), box_axis_off_deg=float(np.median(off)),
                      size_same=float(np.median(same)), size_swap=float(np.median(swap)), size_min=float(np.median(np.minimum(same, swap))))
    json.dump(out, open(dst, 'w'))
    F = [v for v in out.values() if v['fitted']]
    g = lambda k: np.array([v[k] for v in F]); p = lambda a, qs=(5, 25, 50, 75, 95): ' '.join(f'{x:.3f}' for x in np.percentile(a, qs))
    good = (g('inliers') >= 5) & (g('inliers') / g('trace_boxes') >= 0.8)
    print(f'{len(F)} of {len(out)} scenes fitted; reliable (>=5 inliers, share >=0.8): {good.sum()}')
    print('inlier share p5..p95:', p(g('inliers') / g('trace_boxes')), '| residual m:', p(g('residual')))
    for name, m in (('all fitted', np.ones(len(F), bool)), ('reliable', good)):
        print(f'[{name}] nearest traj camera to trace origin, distance m p5..p95:', p(g('cam_dist')[m]), '| first traj camera distance:', p(g('cam0_dist')[m]))
        print(f'[{name}] nearest camera index p5..p95:', p(g('cam_idx')[m]), '| share idx==0: %.3f, idx<=5: %.3f' % ((g('cam_idx')[m] == 0).mean(), (g('cam_idx')[m] <= 5).mean()))
        print(f'[{name}] camera forward azimuth in trace frame (deg; +y = 90) p5..p95:', p(g('fwd_deg')[m]), '| within 45 deg of +y: %.3f' % (np.abs(g('fwd_deg')[m] - 90) <= 45).mean())
        print(f'[{name}] box axes off the trace axes (deg, median per scene) p5..p95:', p(g('box_axis_off_deg')[m]))
        print(f'[{name}] size diff m (median per scene): as-is', p(g('size_same')[m]), '| swapped', p(g('size_swap')[m]), '| better of the two', p(g('size_min')[m]))
    print('theta deg spread:', p(np.degrees(g('theta')) % 360))


if __name__ == '__main__':
    main(*sys.argv[1:])
