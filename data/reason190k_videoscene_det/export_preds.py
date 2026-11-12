"""Detector dump (mmdet3d DumpResults pkl) -> plain json: {scene: {"boxes": [[x, y, z, w, d, h]], "scores": [], "labels": []}},
boxes as the detector outputs them (upright, axis-aligned, in the frame its axis_align_matrix maps to).
Run in the detector environment:  python export_preds.py <preds.pkl> <out.json>
"""
import io, json, pickle, sys, numpy as np, torch
ESL31 = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
         'curtain', 'refrigerator', 'toilet', 'sink', 'bathtub', 'trash bin', 'pillow', 'monitor', 'backpack',
         'printer', 'lamp', 'mirror', 'tv', 'radiator', 'nightstand', 'plant', 'microwave', 'fan', 'blackboard', 'stool']
torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location='cpu')
out = {}
for r in pickle.load(open(sys.argv[1], 'rb')):
    sid = r['lidar_path'].split('/')[-1][:-4] if 'lidar_path' in r else r['img_path'][0].split('/')[-2]
    p = r['pred_instances_3d']; b = p['bboxes_3d']
    b = torch.cat([b.gravity_center, b.tensor[:, 3:6]], 1).cpu().numpy() if hasattr(b, 'gravity_center') else np.asarray(b.cpu())[:, :6]
    out[sid] = dict(boxes=np.round(b.astype(float), 4).tolist(), scores=np.round(np.asarray(p['scores_3d'].cpu(), float), 4).tolist(),
                    labels=[ESL31[int(i)] for i in p['labels_3d']])
json.dump(out, open(sys.argv[2], 'w')); print(len(out), 'scenes,', sum(len(v['scores']) for v in out.values()), 'boxes ->', sys.argv[2])
