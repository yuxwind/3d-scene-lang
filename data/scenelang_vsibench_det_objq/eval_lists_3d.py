"""3D box evaluation of VSI-Bench object lists against the VSI-Bench ground truth, the same way for every source.
Ground truth per scene: the union of the GT objects of all its questions (rows_all5130.jsonl, rows frame: origin at the first
camera, z up, oriented boxes). Predictions per scene: the full list of a rows file (one row of the scene; boxes before the
question filter). Detector lists are already in the rows frame. SAM 3 + VGGT-SLAM lists are in their own reconstructed frame;
they are moved into the rows frame with a rigid fit of the camera centres (scale as estimated) or a similarity fit (true scale),
as s3_report.py does. Labels are compared after merging the VSI-Bench synonyms (trash can / trash bin, heater / radiator, ...,
desk -> table). Matching per scene and label, best score first, oriented IoU (bird's-eye polygon x height).
    python eval_lists_3d.py name=rows:<rows file> ... | name=s3:<list prefix>:<est|true> ...
Columns: AP and recall at IoU 0.25 / 0.5, precision of the whole list at IoU 0.25, recall with the centre within 0.5 m."""
import json, glob, os, re, sys, collections, numpy as np
from shapely.geometry import Polygon
SL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..') + '/'  # repo root (was the gandalf old tree)
S3 = '/scr/xinyu/projects/VGGT-SLAM/exps/vsibench_objects/all/'
SYN = {'trash can': 'trash bin', 'heater': 'radiator', 'whiteboard': 'blackboard', 'table lamp': 'lamp', 'exhaust fan': 'fan', 'washing machine': 'washer', 'desk': 'table'}
can = lambda l: SYN.get(re.sub(r'\s+\d+$', '', l).lower(), re.sub(r'\s+\d+$', '', l).lower())
GT, DS = collections.defaultdict(dict), {}
for l in open(SL + 'data/scenelang_vsibench_gt_firstcam/rows_all5130.jsonl'):
    r = json.loads(l); DS[r['scene_name']] = r['dataset']
    for o in json.loads(r['scene_language']): GT[r['scene_name']][(o['label'], tuple(np.round(o['bbox_3d'][:7], 3)))] = (can(o['label']), np.array((o['bbox_3d'] + [0])[:7], float))
GT = {s: list(v.values()) for s, v in GT.items()}
def poly(b):
    c, s = np.cos(b[6]), np.sin(b[6]); R = np.array([[c, -s], [s, c]]); P = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * b[3:5] / 2
    return Polygon(P @ R.T + b[:2])
def iou(a, b):
    i2 = poly(a).intersection(poly(b)).area; z = max(0, min(a[2] + a[5] / 2, b[2] + b[5] / 2) - max(a[2] - a[5] / 2, b[2] - b[5] / 2)); i = i2 * z
    return i / (np.prod(a[3:6]) + np.prod(b[3:6]) - i + 1e-12)
def rigid(X, Y, scale):
    mx, my = X.mean(0), Y.mean(0); U, D, Vt = np.linalg.svd((Y - my).T @ (X - mx)); d = np.sign(np.linalg.det(U @ Vt)); R = U @ np.diag([1, 1, d]) @ Vt
    k = (D * [1, 1, d]).sum() / ((X - mx) ** 2).sum() if scale else 1.0; return k, R, my - k * R @ mx
def preds_rows(f):
    out = {}
    for l in open(f):
        r = json.loads(l)
        if r['scene_name'] in out: continue
        out[r['scene_name']] = [(can(o['label']), np.array((o['bbox_3d'] + [0])[:7], float), o.get('score', 1.0)) for o in json.loads(r['scene_language'])]
    return out
def preds_s3(pre, mode):
    out = {}
    for f in glob.glob(f'{S3}{pre}_*.json'):
        s = os.path.basename(f)[len(pre) + 1:-5]
        if '+' in s or 'alt-' in s: continue
        D = json.load(open(f)); v = np.array(D['valid']); k, R, t = rigid(np.array(D['cams'])[v], np.array(D['gt_cams'])[v], mode == 'true'); yaw = np.arctan2(R[1, 0], R[0, 0])
        out[s] = [(can(o['label']), np.r_[k * R @ np.array(o['manh'][:3]) + t, k * np.array(o['manh'][3:6]), yaw], o['score']) for o in D['pred'] if o['score'] >= 0.7]
    return out
def evaluate(P, sel=lambda s: True, lab=lambda l: True):
    res = {}
    for thr in (0.25, 0.5):
        rec = []; n_gt = 0; per = collections.defaultdict(list); npos = collections.Counter(); hit_list = 0; n_pred = 0; cen = 0
        for s, G in GT.items():
            if not sel(s): continue
            G = [g for g in G if lab(g[0])]; n_gt += len(G)
            for g in G: npos[g[0]] += 1
            Q = [p for p in P.get(s, []) if lab(p[0])]; n_pred += len(Q)
            for c in {p[0] for p in Q} | {g[0] for g in G}:
                gs = [g for g in G if g[0] == c]; used = set()
                for p in sorted([p for p in Q if p[0] == c], key=lambda p: -p[2]):
                    best, bi = 0, -1
                    for j, g in enumerate(gs):
                        if j in used: continue
                        v = iou(p[1], g[1])
                        if v > best: best, bi = v, j
                    tp = best >= thr
                    if tp: used.add(bi)
                    per[c].append((p[2], tp))
                rec.append(len(used)); hit_list += len(used)
                if thr == 0.25:
                    cu = set()
                    for p in [p for p in Q if p[0] == c]:
                        d = [(np.linalg.norm(p[1][:3] - g[1][:3]), j) for j, g in enumerate(gs) if j not in cu]
                        if d and min(d)[0] < 0.5: cu.add(min(d)[1])
                    cen += len(cu)
        aps = []
        for c, n in npos.items():
            L = sorted(per.get(c, []), key=lambda x: -x[0]); tp = np.cumsum([x[1] for x in L]); fp = np.cumsum([not x[1] for x in L])
            if not len(L): aps.append(0.0); continue
            r = tp / n; p = tp / np.maximum(tp + fp, 1); ap = 0
            for t in np.linspace(0, 1, 11): ap += (p[r >= t].max() if (r >= t).any() else 0) / 11
            aps.append(ap)
        res[thr] = dict(ap=100 * np.mean(aps), rec=100 * sum(rec) / max(n_gt, 1), prec=100 * hit_list / max(n_pred, 1), cen=100 * cen / max(n_gt, 1), n_pred=n_pred, n_gt=n_gt)
    return res
ESL = set(map(can, 'cabinet bed chair sofa table door window bookshelf picture counter desk curtain refrigerator toilet sink bathtub trash bin pillow monitor backpack printer lamp mirror tv radiator nightstand plant microwave fan blackboard stool'.replace('trash bin', 'trash_bin').split()))
ESL = {x.replace('trash_bin', 'trash bin') for x in ESL}
print(f"{'list':30s} {'scope':22s} {'GT':>5s} {'boxes':>6s} {'AP25':>6s} {'AP50':>6s} {'rec25':>6s} {'rec50':>6s} {'prec25':>6s} {'cen0.5':>6s}")
for a in sys.argv[1:]:
    name, spec = a.split('=', 1); kind, rest = spec.split(':', 1)
    P = preds_rows(SL + 'data/scenelang_vsibench_slam/' + rest if not rest.startswith('/') else rest) if kind == 'rows' else preds_s3(*rest.split(':'))
    for scope, sel, lab in (('all labels', lambda s: True, lambda l: True), ('31 classes', lambda s: True, lambda l: l in ESL),
                            ('ScanNet', lambda s: DS[s] == 'scannet', lambda l: True), ('ARKitScenes', lambda s: DS[s] == 'arkitscenes', lambda l: True), ('ScanNet++', lambda s: DS[s] == 'scannetpp', lambda l: True)):
        R = evaluate(P, sel, lab); a2, a5 = R[0.25], R[0.5]
        print(f"{name:30s} {scope:22s} {a2['n_gt']:5d} {a2['n_pred']:6d} {a2['ap']:6.1f} {a5['ap']:6.1f} {a2['rec']:6.1f} {a5['rec']:6.1f} {a2['prec']:6.1f} {a2['cen']:6.1f}")
