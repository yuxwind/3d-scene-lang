"""VGGT-Det predictions on VSI-Bench as VSI-Bench meta_info files (the format of data/meta_info/*_meta_info_val.json
from github.com/vision-x-nyu/thinking-in-space), so they can stand in for the ground-truth scene annotations.

Same predictions, score thresholds and VSI-Bench label names as build.py. Per scene:
  object_bbox[label] = [{"centroid", "axesLengths", "normalizedAxes" (rows = box axes), "min", "max", "score"}, ...]
                       boxes sorted by descending score; "score" is the only key VSI-Bench's files do not have
  object_counts      = number of kept boxes per label
  video_path, dataset copied from VSI-Bench's file; room_size, room_center = null (a detector does not predict
  them, and copying them would leak ground truth)
in VSI-Bench's own frame: ScanNet = raw mesh minus its mean (inverse of vsi_eval/vsi_gt.py's frame_transform),
ARKitScenes = the ARKit mesh frame (identical to the detector's). Boxes are upright: ScanNet yaw 0, ARKit yaw kept.
Classes are whatever the detector predicts, including ones VSI-Bench does not annotate (ScanNet: cabinet, picture,
desk, curtain, shower curtain, toilet, sink, bathtub; ARKitScenes: cabinet, shelf, sink, oven).

    python build_meta.py <out dir> [--seed 0] [--thr-scannet 0.3] [--thr-arkit 0.2]
writes <out dir>/{scannet,arkitscenes}_meta_info_val_vggtdet_seed<k>.json. Same env as build.py.
"""
import argparse
import json
import math
import pickle
from collections import defaultdict

import numpy as np

from build import DATA, RENAME, VGGTDET, load_preds, scene_id

META = "/lustre/fsn1/projects/rech/bef/uxt97ti/data/VSI-Bench/meta_info/{}_meta_info_val.json"
SIGNS = np.array([[i, j, k] for i in (-.5, .5) for j in (-.5, .5) for k in (-.5, .5)])


def to_vsi(dataset, info):
    """(R, t) with vsi = R^T (detector - t): inverse of vsi_gt.frame_transform."""
    if dataset == "arkitscenes":
        return np.eye(3), np.zeros(3)
    A = np.array(info["axis_align_matrix"], np.float64)
    pts = np.fromfile(f"{VGGTDET}/data/scannet/points/{scene_id(dataset, info)}.bin",
                      dtype=np.float32).reshape(-1, 6)[:, :3].astype(np.float64)
    return A[:3, :3], (pts @ A[:3, :3].T + A[:3, 3]).mean(0)


def vsi_box(b, R, t, score):
    """mmdet3d bottom-centre [x, y, z, dx, dy, dz, yaw] in the detector frame -> VSI-Bench box dict."""
    c, s = math.cos(b[6]), math.sin(b[6])
    axes = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]]) @ R      # rows = box axes, VSI frame
    centre = R.T @ (np.array([b[0], b[1], b[2] + b[5] / 2]) - t)
    L = np.array(b[3:6], np.float64)
    corners = centre + (SIGNS * L) @ axes
    return {"centroid": centre.tolist(), "axesLengths": L.tolist(), "normalizedAxes": axes.reshape(-1).tolist(),
            "min": corners.min(0).tolist(), "max": corners.max(0).tolist(), "score": round(float(score), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thr-scannet", type=float, default=0.3)
    ap.add_argument("--thr-arkit", type=float, default=0.2)
    a = ap.parse_args()
    thr = {"scannet": a.thr_scannet, "arkitscenes": a.thr_arkit}
    for ds, (pred_f, info_f) in DATA.items():
        gt = json.load(open(META.format(ds)))
        d = pickle.load(open(f"{VGGTDET}/{info_f}", "rb"))
        infos, names = d["data_list"], {v: k for k, v in d["metainfo"]["categories"].items()}
        preds = load_preds(f"{VGGTDET}/preds/{pred_f.format(a.seed)}")
        out, n = {}, 0
        for info, (lp, b, s, l) in zip(infos, preds):
            sid = scene_id(ds, info)
            assert lp is None or sid in lp, (sid, lp)
            R, t = to_vsi(ds, info)
            objs = defaultdict(list)
            for i in np.argsort(-s):
                if s[i] > thr[ds]:
                    objs[RENAME.get(names[int(l[i])], names[int(l[i])])].append(vsi_box(b[i], R, t, s[i]))
            n += sum(len(v) for v in objs.values())
            out[sid] = {"video_path": gt[sid]["video_path"], "dataset": ds, "room_size": None, "room_center": None,
                        "object_counts": {k: len(v) for k, v in sorted(objs.items())},
                        "object_bbox": dict(sorted(objs.items()))}
        path = f"{a.out}/{ds}_meta_info_val_vggtdet_seed{a.seed}.json"
        json.dump(out, open(path, "w"))
        print(f"{ds}: {len(out)} of {len(gt)} VSI-Bench scenes, {n} boxes (score > {thr[ds]}) -> {path}")


if __name__ == "__main__":
    main()
