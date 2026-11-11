"""Predicted structured scene language for the ScanNet and ARKitScenes questions of VSI-Bench, from VGGT-Det.

Source: VGGT-Det test dumps on the VSI-Bench scenes ($VGGTDET/preds/{scannet_scannet,arkit_arkit}_infos_val_vsi_seed<k>.pkl,
88 ScanNet and 82 of the 150 ARKitScenes scenes; ScanNet++ has no detector) and the infos files they were run on
(scene order, class names, camera poses). The dumps are already NMS'd (no same-class pair above IoU 0.25); boxes
are kept above a score threshold per dataset (the two models' scores live on different scales).

Every kept box becomes one line {"label": ..., "bbox_3d": [x, y, z, w, d, h], "score": s}: centre and full size in
metres, upright and axis-aligned, two decimals, the format of the reason-190k traces. Labels use the VSI-Bench names
(garbagebin -> trash bin, tv_monitor -> tv, showercurtrain -> shower curtain); repeated labels are numbered by
descending score ("chair 1", "chair 2"); lines are sorted by label, then score.

Frames (both z up, metres; written as two files):
  * room:  ScanNet: the scan's axis-aligned frame (axis_align_matrix), which is the detector's frame.
           ARKitScenes: the ARKit mesh frame turned about z so the walls lie on x / y (room yaw = score- and
           footprint-weighted circular mean of 4 * yaw of the kept boxes, i.e. the dominant box direction mod 90 deg).
  * first_camera: the room frame shifted so the origin is the camera of the first video frame (the first frame
           sr_vggt.data.decode_video_frames samples), the traces' convention. ScanNet: posed image 00000 is the
           video's first frame. ARKitScenes: the VSI-Bench video is vga_wide and starts 0.6-1.4 s before the first
           lowres_wide pose, so the first pose is used; over a span that long the camera moves a median 0.11 m
           (90th pct 0.51 m), which is the error on this origin. The room frame has no such error.
ARKitScenes boxes carry a yaw; after the room rotation it is enclosed in an axis-aligned box (--keep-yaw instead
appends the residual yaw in radians as a 7th number). ScanNet boxes have yaw 0.

    python build.py <VSI-Bench dir> <out dir> [--seed 0] [--thr-scannet 0.3] [--thr-arkit 0.2] [--keep-yaw]
writes <out dir>/scenes_<frame>.jsonl (one line per scene) and <out dir>/rows_<frame>.jsonl (one line per
question, the scenelang_vsibench_scannetpp row format). Run in the VGGT-Det mmdet3d env (module pytorch-gpu/py3/2.1.1,
PYTHONPATH=$VGGTDET/pylib:$VGGTDET/code_main), CPU only.
"""
import argparse
import io
import json
import math
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

VGGTDET = "/lustre/fsn1/projects/rech/bef/uxt97ti/vggtdet"
DATA = {
    "scannet": ("scannet_scannet_infos_val_vsi_seed{}.pkl", "data/scannet/scannet_infos_val_vsi.pkl"),
    "arkitscenes": ("arkit_arkit_infos_val_vsi_seed{}.pkl", "data/arkit/arkit_infos_val_vsi.pkl"),
}
RENAME = {"garbagebin": "trash bin", "tv_monitor": "tv", "showercurtrain": "shower curtain"}


def scene_id(dataset, info):
    p = info["lidar_points"]["lidar_path"]
    return p[:-4] if dataset == "scannet" else p.split("_")[0]


def load_preds(path):
    # the dumps pickle CUDA tensors; map them to CPU
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu")
    out = []
    for r in pickle.load(open(path, "rb")):
        p = r["pred_instances_3d"]
        b = p["bboxes_3d"]
        t = (b.tensor if hasattr(b, "tensor") else torch.as_tensor(b)).float().numpy()
        if t.shape[1] == 6:
            t = np.concatenate([t, np.zeros((len(t), 1), np.float32)], 1)
        out.append((r.get("lidar_path"), t[:, :7], np.asarray(p["scores_3d"], np.float32), np.asarray(p["labels_3d"])))
    return out


def room_yaw(b, s):
    """Dominant box direction mod 90 deg, weighted by score x footprint."""
    if not len(b):
        return 0.0
    w = s * b[:, 3] * b[:, 4]
    return math.atan2((w * np.sin(4 * b[:, 6])).sum(), (w * np.cos(4 * b[:, 6])).sum()) / 4


def first_camera(dataset, info, rot):
    """Camera centre of the first video frame in the room frame (before rotation by -rot, then rotated)."""
    c2w = np.array(info["lidar2cam"][0], np.float64)  # camera-to-world (ScanNet: raw scan frame)
    c = c2w[:3, 3]
    if dataset == "scannet":
        A = np.array(info["axis_align_matrix"], np.float64)
        c = A[:3, :3] @ c + A[:3, 3]
    return rotz(-rot) @ c


def rotz(a):
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def convert(b, rot, origin, keep_yaw):
    """mmdet3d bottom-centre [x, y, z, dx, dy, dz, yaw] -> [x, y, z, w, d, h(, yaw)] centre, room frame."""
    c = rotz(-rot) @ np.array([b[0], b[1], b[2] + b[5] / 2]) - origin
    yaw = b[6] - rot
    if keep_yaw:
        yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
        return [*c, b[3], b[4], b[5], yaw]
    cy, sy = abs(math.cos(yaw)), abs(math.sin(yaw))
    return [*c, b[3] * cy + b[4] * sy, b[3] * sy + b[4] * cy, b[5]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thr-scannet", type=float, default=0.3)
    ap.add_argument("--thr-arkit", type=float, default=0.2)
    ap.add_argument("--keep-yaw", action="store_true")
    a = ap.parse_args()
    thr = {"scannet": a.thr_scannet, "arkitscenes": a.thr_arkit}

    scenes = {f: {} for f in ("room", "first_camera")}
    for ds, (pred_f, info_f) in DATA.items():
        d = pickle.load(open(f"{VGGTDET}/{info_f}", "rb"))
        infos, names = d["data_list"], {v: k for k, v in d["metainfo"]["categories"].items()}
        preds = load_preds(f"{VGGTDET}/preds/{pred_f.format(a.seed)}")
        assert len(preds) == len(infos)
        for info, (lp, b, s, l) in zip(infos, preds):
            sid = scene_id(ds, info)
            assert lp is None or sid in lp, (sid, lp)
            k = s > thr[ds]
            b, s, l = b[k], s[k], l[k]
            rot = room_yaw(b, s) if ds == "arkitscenes" else 0.0
            cam = first_camera(ds, info, rot)
            by_label = defaultdict(list)
            for bi, si, li in zip(b, s, l):
                by_label[RENAME.get(names[int(li)], names[int(li)])].append((float(si), bi))
            for frame, origin in (("room", np.zeros(3)), ("first_camera", cam)):
                objs = []
                for lab in sorted(by_label):
                    items = sorted(by_label[lab], key=lambda x: -x[0])
                    for n, (si, bi) in enumerate(items, 1):
                        objs.append({"label": f"{lab} {n}" if len(items) > 1 else lab,
                                     "bbox_3d": [round(float(t), 2) + 0.0 for t in convert(bi, rot, origin, a.keep_yaw)],
                                     "score": round(si, 3)})
                scenes[frame][(ds, sid)] = {"dataset": ds, "scene_name": sid, "frame": frame, "seed": a.seed,
                                            "score_thr": thr[ds], "room_yaw_deg": round(math.degrees(rot), 2),
                                            "first_camera_in_room": [round(float(t), 3) for t in cam],
                                            "objects": objs}

    deb = pd.read_parquet(f"{a.bench}/test_debiased.parquet")
    full = pd.concat([deb, pd.read_parquet(f"{a.bench}/test_pruned.parquet")], ignore_index=True)  # eval row order
    deb_ids = set(deb.id)
    for frame, sc in scenes.items():
        with open(f"{a.out}/scenes_{frame}.jsonl", "w") as f:
            for v in sc.values():
                f.write(json.dumps(v) + "\n")
        n = miss = 0
        with open(f"{a.out}/rows_{frame}.jsonl", "w") as f:
            for i, r in full.iterrows():
                if r.dataset not in DATA:
                    continue
                v = sc.get((r.dataset, r.scene_name))
                if v is None:
                    miss += 1
                    continue
                lines = [json.dumps({"label": o["label"], "bbox_3d": o["bbox_3d"]}) for o in v["objects"]]
                f.write(json.dumps({
                    "index": int(i), "id": int(r.id), "debiased": int(r.id) in deb_ids, "dataset": r.dataset,
                    "scene_name": r.scene_name, "question_type": r.question_type, "question": r.question,
                    "options": None if r.options is None else list(r.options), "ground_truth": str(r.ground_truth),
                    "scene_language": "[\n" + ",\n".join(lines) + "\n]"}) + "\n")
                n += 1
        print(f"{frame}: {len(sc)} scenes, {sum(len(v['objects']) for v in sc.values())} boxes; "
              f"{n} question rows, {miss} skipped (scene without predictions) -> {a.out}/rows_{frame}.jsonl")


if __name__ == "__main__":
    main()
