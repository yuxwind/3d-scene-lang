"""Map-Det3D predictions (run.py) as VSI-Bench meta_info files, plus class-agnostic recall against VSI-Bench's boxes.

meta_info: the format of ../scenelang_vsibench_vggtdet/build_meta.py, in VSI-Bench's own frame: ScanNet = raw mesh minus
its mean, ARKitScenes = the ARKit mesh frame (the frame of the poses, so no change). Map-Det3D has no classes, so every box is labelled "object":
  object_bbox["object"] = [{"centroid", "axesLengths", "normalizedAxes" (rows = box axes), "min", "max", "score"}, ...]
                          sorted by descending score; object_counts = {"object": N}; room_size, room_center = null.
Boxes are oriented (Map-Det3D predicts a full rotation; it is kept, not made upright).

Recall: a VSI-Bench GT box is found if some prediction overlaps it with oriented 3D IoU >= thr, ignoring labels. VSI-Bench
only annotates its own categories, so precision is not defined; the mean number of boxes per scene is reported next
to recall instead. VGGT-Det (../scenelang_vsibench_vggtdet/<dataset>_meta_info_val_vggtdet_seed<k>.json, its labels
dropped) is scored the same way, on the scenes both have, for reference.

    python build_meta.py <dir with preds/> [--thr 0.25] [--vggtdet-seed 0]
for every <dir>/preds/<dataset>/, writes <dir>/<dataset>_meta_info_val_mapdet3d.json; and <dir>/recall.json. Env: conda env mapdet3d, CPU is enough.
"""
import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np
import torch

from mapdet3d.data.const import AxisMode
from mapdet3d.op.box3d import box3d_overlap, boxes3d_to_corners, check_coplanar

VGGTDET = "/lustre/fsn1/projects/rech/bef/uxt97ti/vggtdet"
META = "/lustre/fsn1/projects/rech/bef/uxt97ti/data/VSI-Bench/meta_info/{}_meta_info_val.json"
VGGTDET_META = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "../scenelang_vsibench_vggtdet/{}_meta_info_val_vggtdet_seed{}.json")
# box3d_overlap's corner order: the unit cube (0,0,0) (1,0,0) (1,1,0) (0,1,0) (0,0,1) (1,0,1) (1,1,1) (0,1,1), centred
UNIT = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]) - 0.5
IOU_THRS = (0.25, 0.5)


def raw_means():
    """Scene -> mean of the raw scan points, i.e. raw - mean = VSI-Bench frame (as to_vsi in vggtdet/build_meta.py)."""
    d = pickle.load(open(f"{VGGTDET}/data/scannet/scannet_infos_val_vsi.pkl", "rb"))
    out = {}
    for info in d["data_list"]:
        sid = info["lidar_points"]["lidar_path"][:-4]
        A = np.array(info["axis_align_matrix"], np.float64)
        pts = np.fromfile(f"{VGGTDET}/data/scannet/points/{sid}.bin", dtype=np.float32).reshape(-1, 6)[:, :3]
        aligned_mean = (pts.astype(np.float64) @ A[:3, :3].T + A[:3, 3]).mean(0)
        out[sid] = A[:3, :3].T @ (aligned_mean - A[:3, 3])
    return out


def corners_from_vsi(b):
    """VSI-Bench box dict -> (8, 3) corners in box3d_overlap's order (axes made right-handed)."""
    axes = np.array(b["normalizedAxes"], np.float64).reshape(3, 3)
    if np.linalg.det(axes) < 0:
        axes[2] *= -1
    return np.asarray(b["centroid"]) + (UNIT * np.asarray(b["axesLengths"])) @ axes


def vsi_from_corners(c, score):
    """(8, 3) corners in box3d_overlap's order -> VSI-Bench box dict."""
    edges = np.stack([c[1] - c[0], c[3] - c[0], c[4] - c[0]])
    lengths = np.linalg.norm(edges, axis=1)
    return {"centroid": c.mean(0).tolist(), "axesLengths": lengths.tolist(),
            "normalizedAxes": (edges / lengths[:, None]).reshape(-1).tolist(),
            "min": c.min(0).tolist(), "max": c.max(0).tolist(), "score": round(float(score), 4)}


def iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    return box3d_overlap(torch.as_tensor(np.asarray(a), dtype=torch.float32),
                         torch.as_tensor(np.asarray(b), dtype=torch.float32)).numpy()


def recall(pred, gt):
    """pred: scene -> (P, 8, 3); gt: scene -> list of (label, (8, 3)). Class-agnostic recall per IoU threshold."""
    hit = {t: defaultdict(lambda: [0, 0]) for t in IOU_THRS}
    for s, g in gt.items():
        m = iou(np.stack([c for _, c in g]), pred.get(s, np.zeros((0, 8, 3)))).max(1, initial=0)
        for t in IOU_THRS:
            for (label, _), v in zip(g, m):
                for k in (label, "all"):
                    hit[t][k][0] += int(v >= t)
                    hit[t][k][1] += 1
    n_pred = [len(pred.get(s, [])) for s in gt]
    return {"scenes": len(gt), "gt_boxes": hit[IOU_THRS[0]]["all"][1], "pred_boxes_per_scene": float(np.mean(n_pred)),
            **{f"recall@{t}": {k: round(h / n, 4) for k, (h, n) in sorted(hit[t].items())} for t in IOU_THRS}}


def build(a, ds, offsets):
    gt_meta = json.load(open(META.format(ds)))
    pred, out = {}, {}
    for f in sorted(os.listdir(f"{a.dir}/preds/{ds}")):
        s = f[:-4]
        p = np.load(f"{a.dir}/preds/{ds}/{f}")
        keep = p["scores"] >= a.thr
        boxes, scores = torch.from_numpy(p["boxes"][keep]), p["scores"][keep]
        c = (boxes3d_to_corners(boxes, AxisMode.ROS).numpy().astype(np.float64) - offsets.get(s, 0)
             if len(boxes) else np.zeros((0, 8, 3)))
        order = np.argsort(-scores)
        c, scores = c[order], scores[order]
        objs = [vsi_from_corners(ci, si) for ci, si in zip(c, scores)]
        if len(c):  # the corner order must round-trip, or IoUs below are meaningless (self-IoU is not a usable test:
            # iou_box3d is unstable on two identical boxes, e.g. 1.37 for one box of d755b3d9d8)
            back = np.stack([corners_from_vsi(o) for o in objs])
            assert np.allclose(back, c, atol=1e-4), s
            assert check_coplanar(torch.from_numpy(c).float(), eps=1e-4).all(), s
        pred[s] = c
        out[s] = {"video_path": gt_meta[s]["video_path"], "dataset": ds, "room_size": None, "room_center": None,
                  "object_counts": {"object": len(objs)}, "object_bbox": {"object": objs}}
    json.dump(out, open(f"{a.dir}/{ds}_meta_info_val_mapdet3d.json", "w"))

    gt = {s: [(label, corners_from_vsi(b)) for label, bs in gt_meta[s]["object_bbox"].items() for b in bs]
          for s in pred}
    res = {"mapdet3d": {"score_thr": a.thr, **recall(pred, gt)}}
    vf = VGGTDET_META.format(ds, a.vggtdet_seed)
    if os.path.exists(vf):
        v = json.load(open(vf))
        both = [s for s in gt if s in v]
        vpred = {s: np.stack([corners_from_vsi(b) for bs in v[s]["object_bbox"].values() for b in bs])
                 if v[s]["object_counts"] else np.zeros((0, 8, 3)) for s in both}
        res[f"vggtdet_seed{a.vggtdet_seed}"] = recall(vpred, {s: gt[s] for s in both})
        res["mapdet3d_on_vggtdet_scenes"] = recall({s: pred[s] for s in both}, {s: gt[s] for s in both})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--thr", type=float, default=0.25, help="score threshold (run.py already drops < 0.25)")
    ap.add_argument("--vggtdet-seed", type=int, default=0)
    a = ap.parse_args()

    res = {}
    for ds in sorted(os.listdir(f"{a.dir}/preds")):
        res[ds] = build(a, ds, raw_means() if ds == "scannet" else {})
    json.dump(res, open(f"{a.dir}/recall.json", "w"), indent=1)
    for ds, r_ds in res.items():
        for k, r in r_ds.items():
            print(f"{ds:12s} {k:28s} scenes {r['scenes']:3d}  gt {r['gt_boxes']:5d}  boxes/scene {r['pred_boxes_per_scene']:5.1f}  "
                  + "  ".join(f"R@{t} {r[f'recall@{t}']['all']:.3f}" for t in IOU_THRS))


if __name__ == "__main__":
    main()
