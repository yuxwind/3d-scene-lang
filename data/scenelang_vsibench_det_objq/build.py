"""VSI-Bench rows with the scene taken from the task2-det detector (VGG-T-Det test2-hybrid: SAM 3 + VGGT-SLAM proposals refined
and filtered by the detector, plus its own boxes for objects without a proposal).
In : the three dumps of vggtdet_es56/runs/vsibench_det_objq/launch.sh (ScanNet, ScanNet++, ARKitScenes).
     The model uses the first-camera axis convention: boxes and the dumped axis_align_matrix are in a frame turned about z by a
     multiple of 90 deg against the info file's frame. The turn is undone here (R = dumped A @ inv(info A)); then, as in
     ../scenelang_vsibench_det_esl31/build.py, ScanNet boxes (scan's aligned frame) are shifted by the camera of posed image 0,
     ARKitScenes / ScanNet++ boxes are already in the rows frame.
Out: ../scenelang_vsibench_slam/rows_<tag>_all5130_full.jsonl (all 5,130 rows; boxes with score >= thr, {"label", "bbox_3d", "score"}).
     Labels: the 65 class names, in the question's wording where it differs (trash can, heater, whiteboard, table lamp, exhaust
     fan, washing machine; desk -> table unless the question says desk).
    python build.py <tag> <thr> <scannet preds.pkl> <scannetpp preds.pkl> <arkit preds.pkl> [<thr for free-query boxes>]      (detector env)"""
import io, json, pickle, re, sys
from collections import Counter
from pathlib import Path
import numpy as np, torch
D = Path(__file__).resolve().parent; V = '/scr/xinyu/projects/vggtdet_es56/'
CL = json.load(open(V + 'code/tools/dataset_converters/vsi65_label_map.json'))['target_classes']
SYN = {'trash can': 'trash bin', 'heater': 'radiator', 'whiteboard': 'blackboard', 'table lamp': 'lamp', 'exhaust fan': 'fan', 'washing machine': 'washer'}
torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location='cpu')
ST = Counter()
INFO = {'scannet': V + 'data/scannet/scannet_infos_vsibench88_esl31.pkl', 'scannetpp': V + 'data/scannetpp_vsibench/scannetpp_infos_vsibench.pkl', 'arkitscenes': V + 'data/arkit_vsibench/arkit_infos_vsibench150.pkl'}


def n_props(sid, cap=512):
    """number of leading boxes of a scene's output that come from SAM 3 proposals (the detector keeps whole proposals, best first, up to `cap` queries)"""
    f = V + f'data/objq_vsibench/{sid}.npz'
    if not Path(f).exists(): return 0
    Z = np.load(f); n = len(Z['label'])
    while n > 0 and int((Z['obj'] < n).sum()) > cap: n -= 1
    return n


def load(ds, pkl, thr, thr_free=None, thr_sam=None):
    info = {i['img_paths'][0].split('/')[-2]: i for i in pickle.load(open(INFO[ds], 'rb'))['data_list']}; out = {}
    for r in pickle.load(open(pkl, 'rb')):
        sid = r['img_path'][0].split('/')[-2]; p = r['pred_instances_3d']; b = p['bboxes_3d']
        b = torch.cat([b.gravity_center, b.tensor[:, 3:6]], 1).cpu().numpy().astype(float) if hasattr(b, 'gravity_center') else np.asarray(b.cpu(), float)[:, :6]
        A0 = np.array(info[sid]['axis_align_matrix'], float); R = (np.array(r['axis_align_matrix'], float) @ np.linalg.inv(A0))[:3, :3]   # turned <- info frame
        assert np.allclose(np.abs(R).round(), np.abs(R), atol=1e-3) and abs(R[2, 2] - 1) < 1e-3, (sid, R)
        b[:, :3] = b[:, :3] @ R                                              # = R^T applied to each centre: back to the info frame
        if abs(R[0, 1]) > 0.5: b[:, [3, 4]] = b[:, [4, 3]]
        if ds == 'scannet': b[:, :3] -= (A0 @ np.array(info[sid]['lidar2cam'][0], float))[:3, 3]
        sc = np.asarray(p['scores_3d'].cpu(), float); lab = [CL[int(i)] for i in p['labels_3d'].cpu()]
        npr = n_props(sid); tf = thr if thr_free is None else thr_free
        if thr_sam is not None:            # select the proposals as the SAM 3 list does (SAM 3 box score >= thr_sam) and show that score; the detector supplies the box (and may veto with thr)
            zs = np.load(V + f'data/objq_vsibench/{sid}.npz')['score'][:npr] if npr else np.zeros(0); keep_p = sc[:npr].copy()
            sc = sc.copy(); sc[:npr] = np.where((zs >= thr_sam) & (keep_p >= thr), np.maximum(zs, thr), -1.0)          # boxes 0..npr-1: proposals (score = keep probability); the rest: free queries
        out[sid] = sorted(((l, [round(float(x), 2) for x in bb], round(float(s), 3)) for i, (l, bb, s) in enumerate(zip(lab, b, sc)) if s >= (thr if i < npr else tf)), key=lambda x: -x[2])
        ST['proposal boxes kept'] += sum(1 for i, s in enumerate(sc) if i < npr and s >= thr); ST['free boxes kept'] += sum(1 for i, s in enumerate(sc) if i >= npr and s >= tf); ST['scenes'] += 1
    return out


if __name__ == '__main__':
    tag, thr = sys.argv[1], float(sys.argv[2]); tf = float(sys.argv[6]) if len(sys.argv) > 6 else None   # optional 6th argument: threshold for free-query boxes (2 = none of them)
    ts = float(sys.argv[7]) if len(sys.argv) > 7 else None   # optional 7th argument: SAM 3 box-score threshold (selection by the SAM 3 score, boxes from the detector)
    det = {ds: load(ds, f, thr, tf, ts) for ds, f in zip(('scannet', 'scannetpp', 'arkitscenes'), sys.argv[3:6])}; print('per scene:', {k: round(v / ST['scenes'], 1) for k, v in ST.items() if k != 'scenes'})
    st, nb = Counter(), {k: [] for k in det}
    with open(D.parent / f'scenelang_vsibench_slam/rows_{tag}_all5130_full.jsonl', 'w') as f:
        for line in open(D.parent / 'scenelang_vsibench_det_esl31/rows_det_all5130_mixv2.jsonl'):
            r = json.loads(line); ds = r['dataset']; q = r['question'].lower(); ren = {c: v for v, c in SYN.items() if re.search(r'\b' + re.escape(v), q)}
            if not re.search(r'\bdesks?\b', q): ren['desk'] = 'table'
            objs = [{'label': ren.get(c, c), 'bbox_3d': b, 'score': s} for c, b, s in det[ds].get(r['scene_name'], [])]
            nb[ds].append(len(objs)); st[ds] += 1; st[ds + ' empty'] += not objs; st[ds + ' no scene'] += r['scene_name'] not in det[ds]
            r['scene_language'] = '[\n' + ',\n'.join(json.dumps(o) for o in objs) + '\n]'; r['use_scene'] = True; f.write(json.dumps(r) + '\n')
    print(f'rows_{tag}_all5130_full.jsonl thr {thr}:', dict(st), {ds: round(float(np.mean(v)), 1) for ds, v in nb.items()}, 'boxes per row')
