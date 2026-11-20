"""VSI-Bench rows with the scene taken from VGG-T-Det (ESL31, ScanNet-trained) instead of ground truth.

ScanNet (88 scenes) and ARKitScenes (150 scenes); ScanNet++ has no detector inputs on this machine, so its rows are
left out. Frames are those of ../scenelang_vsibench_gt_firstcam/rows_all5130.jsonl, so only the boxes differ:
  * ScanNet: detector boxes are in the scan's axis-aligned frame -> minus the position of the camera of posed image 0.
  * ARKitScenes: the detector was run with axis_align_matrix = ARKit world -> that frame
    (../reason190k_videoscene_det/build_arkit_inputs.py with frames_arkit_vsibench.json), boxes used as dumped.
Boxes over the per-class score threshold of the training data (../reason190k_videoscene_det/build_mix.py; ARKitScenes
with thr_arkitscenes.json) are kept as {"label", "bbox_3d": [x, y, z, w, d, h], "score"}. Labels are the detector
class names, except that VSI-Bench wording is used when the question uses it (trash can, heater, whiteboard,
table lamp, exhaust fan) and desk -> table unless the question says desk (VSI-Bench has no desk label).
    python build.py <scannet preds.pkl> <arkit preds.pkl>        (detector env; PYTHONPATH=<vggtdet code>)
writes rows_det_scannet_arkit.jsonl
    python build.py scannetpp <scannetpp preds.pkl>
writes rows_det_scannetpp.jsonl: ScanNet++ scenes with detector output (48 of 50; d755b3d9d8 has broken iPhone poses,
e7af285f7d has no mesh / poses here). The detector was run with axis_align_matrix = mesh -> rows frame
(../reason190k_videoscene_det/build_scannetpp_inputs.py), boxes used as dumped; ScanNet thresholds (none tuned yet).
"""
import io, json, pickle, re, sys
from collections import Counter
from pathlib import Path
import numpy as np, torch

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D.parent / 'reason190k_videoscene_det'))
from build_mix import ESL31, THR
SYN = {'trash can': 'trash bin', 'heater': 'radiator', 'whiteboard': 'blackboard', 'table lamp': 'lamp', 'exhaust fan': 'fan'}
torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location='cpu')


def load(pkl, thr, shift_cam0):
    out = {}
    for r in pickle.load(open(pkl, 'rb')):
        sid = r['lidar_path'].split('/')[-1][:-4] if 'lidar_path' in r else r['img_path'][0].split('/')[-2]
        p = r['pred_instances_3d']; b = p['bboxes_3d']
        b = torch.cat([b.gravity_center, b.tensor[:, 3:6]], 1).cpu().numpy().astype(float) if hasattr(b, 'gravity_center') else np.asarray(b.cpu(), float)[:, :6]
        if shift_cam0:
            b[:, :3] -= (np.array(r['axis_align_matrix'], float) @ np.array(r['lidar2cam'][0], float))[:3, 3]
        sc = np.asarray(p['scores_3d'].cpu(), float); lab = [ESL31[int(i)] for i in p['labels_3d']]
        keep = sorted(((l, [round(float(x), 2) for x in bb], round(float(s), 3)) for l, bb, s in zip(lab, b, sc) if s >= thr.get(l, 0.5)), key=lambda x: -x[2])
        out[sid] = keep
    return out


def main():
    thr_a = {**THR['scannet'], **json.load(open(D.parent / 'reason190k_videoscene_det/thr_arkitscenes.json'))}
    if sys.argv[1] == 'scannetpp':
        det, name = {'scannetpp': load(sys.argv[2], THR['scannet'], False)}, 'rows_det_scannetpp.jsonl'
    else:
        det, name = {'scannet': load(sys.argv[1], THR['scannet'], True), 'arkitscenes': load(sys.argv[2], thr_a, False)}, 'rows_det_scannet_arkit.jsonl'
    st, nb = Counter(), {k: [] for k in det}
    with open(D / name, 'w') as f:
        for line in open(D.parent / 'scenelang_vsibench_gt_firstcam/rows_all5130.jsonl'):
            r = json.loads(line); ds = r['dataset']
            if ds not in det or r['scene_name'] not in det[ds]: continue
            q = r['question'].lower(); ren = {c: v for v, c in SYN.items() if re.search(r'\b' + re.escape(v), q)}
            if not re.search(r'\bdesks?\b', q): ren['desk'] = 'table'
            objs = [{'label': ren.get(c, c), 'bbox_3d': b, 'score': s} for c, b, s in det[ds][r['scene_name']]]
            nb[ds].append((len(json.loads(r['scene_language'])), len(objs))); st[ds] += 1; st[ds + ' empty'] += not objs
            r['scene_language'] = '[\n' + ',\n'.join(json.dumps(o) for o in objs) + '\n]'
            f.write(json.dumps(r) + '\n')
    print(dict(st))
    for ds, v in nb.items():
        v = np.array(v); print(ds, 'boxes per row: GT median', np.median(v[:, 0]), '-> detector median', np.median(v[:, 1]), 'mean', v[:, 1].mean().round(1))


if __name__ == '__main__':
    main()
