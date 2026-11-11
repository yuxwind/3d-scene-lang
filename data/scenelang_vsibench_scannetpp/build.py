"""Ground-truth structured scene language for the ScanNet++ questions of VSI-Bench (paper Table 2 setup).

Source: VSI-Bench's own per-scene ground truth, data/meta_info/scannetpp_meta_info_val.json from
github.com/vision-x-nyu/thinking-in-space (the file its QA was generated from; 50 scenes). Every object
becomes one line {"label": ..., "bbox_3d": [x, y, z, dx, dy, dz, yaw, roll, pitch]}: centre and extents in
metres, angles in radians, intrinsic ZXY Euler (R = Rz(yaw) Rx(roll) Ry(pitch)), ScanNet++ world frame
(z up), two decimals, objects in the meta file's order (grouped by label, not sorted). room_size and
room_center are left out: room_size is the answer to the room-size questions. The meta file's "coat hanger"
is called "coat rack" in the questions (its counts and sizes are the answers), so it is renamed.

    python build.py <meta_info json> <VSI-Bench dir> <out jsonl> [<dataset: scannetpp (default) | scannet | arkitscenes>]
The ScanNet and ARKitScenes meta files are read the same way (their own world frames, z up).
"""
import json
import math
import sys

import numpy as np
import pandas as pd

meta_path, B, out_path = sys.argv[1:4]
DATASET = sys.argv[4] if len(sys.argv) > 4 else "scannetpp"
meta = json.load(open(meta_path))
RENAME = {"coat hanger": "coat rack"} if DATASET == "scannetpp" else {}  # meta label -> the name the questions use


def nine_dof(o):
    axes = np.array(o["normalizedAxes"], float).reshape(3, 3)  # rows are the box axes
    ext = np.array(o["axesLengths"], float)
    v = int(np.argmax(np.abs(axes[:, 2])))  # most vertical axis -> box z
    h = [i for i in range(3) if i != v]
    z = axes[v] * np.sign(axes[v, 2] or 1.0)
    x = axes[h[0]] - z * (axes[h[0]] @ z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    dx, dy, dz = ext[h[0]], ext[h[1]], ext[v]
    R = np.stack([x, y, z], 1)
    yaw = math.atan2(-R[0, 1], R[1, 1])
    if not -math.pi / 2 <= yaw < math.pi / 2:  # the same box with its x and y axes negated: yaw in [-pi/2, pi/2)
        R = R * np.array([-1.0, -1.0, 1.0])
        yaw = math.atan2(-R[0, 1], R[1, 1])
    roll = math.asin(max(-1.0, min(1.0, R[2, 1])))
    pitch = math.atan2(-R[2, 0], R[2, 2])
    # check: the decoded rotation spans the same axes (up to sign) as the source box
    cz, sz, cx, sx, cy, sy = math.cos(yaw), math.sin(yaw), math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    Rd = (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
          @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]))
    err = np.abs(np.abs(Rd.T @ R) - np.eye(3)).max()
    return [*o["centroid"], dx, dy, dz, yaw, roll, pitch], err


scenes, worst = {}, 0.0
for sid, s in meta.items():
    lines = []
    for label, boxes in s["object_bbox"].items():
        for o in boxes:
            b, err = nine_dof(o)
            worst = max(worst, err)
            lines.append(json.dumps({"label": RENAME.get(label, label), "bbox_3d": [round(float(t), 2) + 0.0 for t in b]}))
    scenes[sid] = "[\n" + ",\n".join(lines) + "\n]"
print("scenes", len(scenes), "max rotation error", worst)

deb = pd.read_parquet(f"{B}/test_debiased.parquet")
full = pd.concat([deb, pd.read_parquet(f"{B}/test_pruned.parquet")], ignore_index=True)  # eval row order
deb_ids = set(deb.id)
n = 0
with open(out_path, "w") as f:
    for i, r in full.iterrows():
        if r.dataset != DATASET:
            continue
        f.write(json.dumps({
            "index": int(i), "id": int(r.id), "dataset": r.dataset, "debiased": int(r.id) in deb_ids, "scene_name": r.scene_name,
            "question_type": r.question_type, "question": r.question,
            "options": None if r.options is None else list(r.options), "ground_truth": str(r.ground_truth),
            "scene_language": scenes[r.scene_name]}) + "\n")
        n += 1
print("rows", n, "->", out_path)
