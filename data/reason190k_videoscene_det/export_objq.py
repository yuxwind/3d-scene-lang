"""test2-hybrid detector dump (VGG-T-Det with SAM 3 + VGGT-SLAM proposals) -> det_<ds>_objq.json for build_mix.py --det_tag objq.
Same json as export_preds.py: {scene: {"boxes": [[x, y, z, w, d, h]], "scores": [], "labels": []}}, boxes in the frame of the
info file's axis_align_matrix (ScanNet: the scan's aligned frame; ARKitScenes / ScanNet++: the trace frame).
"refine" selection, as the best VSI-Bench list: one box per SAM 3 proposal with SAM 3 box score >= 0.7, the detector's refined
centre and size, the proposal's label (65 classes), score = the SAM 3 box score. The detector's keep probability and its
free-query boxes are not used.
The model uses the first-camera axis convention, so boxes come out in a frame turned about z by a multiple of 90 deg against the
info file's frame; the turn is undone here (R = dumped axis_align @ inv(info axis_align)).
Run in the detector environment:  python export_objq.py <preds.pkl> <infos.pkl> <objq dir> <out.json> [thr_sam = 0.7]"""
import io, json, pickle, sys
from pathlib import Path
import numpy as np, torch
V = '/scr/xinyu/projects/vggtdet_es56/'
CL = json.load(open(V + 'code/tools/dataset_converters/vsi65_label_map.json'))['target_classes']
torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location='cpu')
pkl, infos, qdir, dst = sys.argv[1:5]; THR = float(sys.argv[5]) if len(sys.argv) > 5 else 0.7
info = {i['img_paths'][0].split('/')[-2]: i for i in pickle.load(open(infos, 'rb'))['data_list']}; out = {}; n_all = n_keep = 0
for r in pickle.load(open(pkl, 'rb')):
    sid = r['img_path'][0].split('/')[-2]; p = r['pred_instances_3d']; b = p['bboxes_3d']
    b = torch.cat([b.gravity_center, b.tensor[:, 3:6]], 1).cpu().numpy().astype(float) if hasattr(b, 'gravity_center') else np.asarray(b.cpu(), float)[:, :6]
    R = (np.array(r['axis_align_matrix'], float) @ np.linalg.inv(np.array(info[sid]['axis_align_matrix'], float)))[:3, :3]
    assert np.allclose(np.abs(R).round(), np.abs(R), atol=1e-3) and abs(R[2, 2] - 1) < 1e-3, (sid, R)
    b[:, :3] = b[:, :3] @ R
    if abs(R[0, 1]) > 0.5: b[:, [3, 4]] = b[:, [4, 3]]
    f = Path(qdir) / f'{sid}.npz'
    if not f.exists(): out[sid] = dict(boxes=[], scores=[], labels=[]); continue
    Z = np.load(f); n = len(Z['label'])
    while n > 0 and int((Z['obj'] < n).sum()) > 512: n -= 1            # the detector keeps whole proposals, best first, up to 512 queries
    keep = [i for i in range(min(n, len(b))) if Z['score'][i] >= THR]; n_all += n; n_keep += len(keep)
    out[sid] = dict(boxes=np.round(b[keep], 4).tolist(), scores=[round(float(Z['score'][i]), 4) for i in keep], labels=[CL[int(Z['label'][i])] for i in keep])
json.dump(out, open(dst, 'w')); print(len(out), 'scenes;', n_keep, 'of', n_all, f'proposals kept (SAM 3 score >= {THR}) ->', dst)
