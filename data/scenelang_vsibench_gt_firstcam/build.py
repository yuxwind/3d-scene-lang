"""Ground-truth scene language for all 5,130 VSI-Bench questions, in the frame of the reason-190k traces.

Source: VSI-Bench's meta_info boxes (the ones its QA was generated from), the same input as
data/scenelang_vsibench_scannetpp/build.py, which writes them in each dataset's own world frame. Here they are moved
to the traces' convention (simplesr_traces README): z up, horizontal axes along the room, origin at the camera of
the first video frame. Frames match data/scenelang_vsibench_vggtdet/build.py (rows_first_camera), so GT and VGGT-Det
scenes differ only in the boxes:
  * ScanNet: VSI frame = raw mesh minus its mean (vsi_eval/vsi_gt.py) -> axis-aligned scan frame (axis_align_matrix),
    origin = posed image 00000 (the video's first frame), from VGGT-Det's scannet_infos_val_vsi.pkl.
  * ARKitScenes: VSI frame = ARKit mesh frame, turned about z by the room yaw so the walls lie on x / y; origin = the
    first lowres_wide.traj pose of the raw release (as VGGT-Det's infos; the video starts 0.6-1.4 s earlier).
  * ScanNet++: VSI frame = ScanNet++ mesh frame; axes turned by the room yaw as for ARKitScenes; origin = the
    ARKit `aligned_pose` of iPhone pose frame 0 (= mp4 frame 0), from ~/scratch/data/scannetpp_cam0 (README there;
    prebuilt/det_es/<scene>/meta.json cam0_cam2global, whose det.json boxes map back onto the VSI centroids to < 2 cm).
    Two scenes keep the world origin: e7af285f7d (no pose in the bundle, 3 rows) and d755b3d9d8 (its aligned_pose
    track is broken: frame 0 sits 55.8 m above the floor and later frames drift to hundreds of metres). Distances,
    sizes, counts and directions do not depend on the origin; only the prompt's "origin at the first frame's camera"
    is untrue for those rows.
Room yaw (ARKitScenes, ScanNet++) = footprint-weighted circular mean of 4 x yaw of the upright GT boxes, i.e. the
dominant box direction mod 90 deg (build.py's room_yaw, on GT boxes instead of detections).

Boxes keep the 9-DoF format of scenelang_vsibench_scannetpp/build.py ([x, y, z, dx, dy, dz, yaw, roll, pitch],
intrinsic ZXY), so the zero-shot prompts and eval_scenelang.py's rvs_scene (which encloses the yaw-rotated
footprint in an axis-aligned box) read them unchanged. Row fields and order are those of the scannetpp builder.

    python build.py <VSI-Bench dir> <out dir> [<boxes meta_info json> <out prefix>]      (VGGT-Det mmdet3d env, CPU)
With the two optional arguments the frames are still taken from the GT scenes (room yaw, first camera), but the boxes
written are those of another meta_info file in the VSI-Bench frame (e.g. ../scenelang_vsibench_mapdet3d/
vsibench_meta_info_val_mapdet3d_oracle0.1.json, keyed by scene with a "dataset" field): <out dir>/<prefix>_<dataset>_
scenelang.jsonl, rows of scenes missing from that file skipped; frames.json is not rewritten.
writes <out dir>/vsibench_{scannet,scannetpp,arkitscenes}_scenelang.jsonl and <out dir>/frames.json (per scene:
room yaw, origin in the VSI frame).
"""
import json
import math
import pickle
import sys

import numpy as np
import pandas as pd

META = "/lustre/fsn1/projects/rech/bef/uxt97ti/data/VSI-Bench/meta_info/{}_meta_info_val.json"
VGGTDET = "/lustre/fsn1/projects/rech/bef/uxt97ti/vggtdet"
ARKIT_RAW = "/lustre/fsmisc/dataset/ARKitScenes/raw/Validation"
SPP_CAM0 = "/linkhome/rech/genclw01/uxt97ti/scratch/data/scannetpp_cam0/prebuilt/det_es/{}/meta.json"
RENAME = {"scannetpp": {"coat hanger": "coat rack"}}


def nine_dof(o):  # = scenelang_vsibench_scannetpp/build.py (that file runs at import)
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


def rotz(a):
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def room_yaw(boxes):
    """boxes: VSI box dicts in a z-up frame. Dominant upright-box direction mod 90 deg, weighted by footprint."""
    s = c = 0.0
    for o in boxes:
        axes = np.array(o["normalizedAxes"], float).reshape(3, 3)
        L = np.array(o["axesLengths"], float)
        up = int(np.argmax(np.abs(axes[:, 2])))
        i, j = [k for k in range(3) if k != up]
        w = L[i] * L[j]
        a = math.atan2(axes[i, 1], axes[i, 0])
        s, c = s + w * math.sin(4 * a), c + w * math.cos(4 * a)
    return math.atan2(s, c) / 4


def scannet_frames():
    """scene -> (M, t): room = M @ vsi + t, origin at the first video frame's camera."""
    out = {}
    for info in pickle.load(open(f"{VGGTDET}/data/scannet/scannet_infos_val_vsi.pkl", "rb"))["data_list"]:
        sid = info["lidar_points"]["lidar_path"][:-4]
        A = np.array(info["axis_align_matrix"], np.float64)
        pts = np.fromfile(f"{VGGTDET}/data/scannet/points/{sid}.bin", np.float32).reshape(-1, 6)[:, :3].astype(np.float64)
        mean_aligned = (pts @ A[:3, :3].T + A[:3, 3]).mean(0)       # aligned = A_R vsi + mean_aligned (vsi_gt.py)
        cam = A[:3, :3] @ np.array(info["lidar2cam"][0], np.float64)[:3, 3] + A[:3, 3]  # c2w, raw scan frame
        out[sid] = (A[:3, :3], mean_aligned - cam, 0.0, cam)
    return out


def arkit_first_camera(sid):
    """Camera centre of the first lowres_wide.traj pose (ts, axis-angle, t: world-to-camera), ARKit mesh frame."""
    tok = [float(v) for v in open(f"{ARKIT_RAW}/{sid}/lowres_wide.traj").readline().split()]
    aa, t = np.array(tok[1:4]), np.array(tok[4:7])
    th = np.linalg.norm(aa)
    k = aa / th if th > 0 else np.zeros(3)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
    return -R.T @ t


def scannetpp_first_camera(sid, boxes):
    """Camera centre of mp4 frame 0 in the mesh frame, or None if missing / implausible (outside the scene's box
    extent grown by 2 m, or not 0.3-3 m above the lowest box bottom)."""
    try:
        c = np.array(json.load(open(SPP_CAM0.format(sid)))["cam0_cam2global"], float)[:3, 3]
    except FileNotFoundError:
        return None
    lo = np.min([o["min"] for o in boxes], 0)
    hi = np.max([o["max"] for o in boxes], 0)
    ok = np.all(c[:2] > lo[:2] - 2) and np.all(c[:2] < hi[:2] + 2) and 0.3 < c[2] - lo[2] < 3
    return c if ok else None


def main(bench, out_dir, boxes_meta=None, prefix="vsibench"):
    deb = pd.read_parquet(f"{bench}/test_debiased.parquet")
    full = pd.concat([deb, pd.read_parquet(f"{bench}/test_pruned.parquet")], ignore_index=True)  # eval row order
    deb_ids = set(deb.id)
    sc = scannet_frames()
    # check the ARKit first-camera reading against VGGT-Det's infos (c2w lidar2cam[0]) where they exist
    chk = [abs(arkit_first_camera(i["lidar_points"]["lidar_path"].split("_")[0]) - np.array(i["lidar2cam"][0])[:3, 3]).max()
           for i in pickle.load(open(f"{VGGTDET}/data/arkit/arkit_infos_val_vsi150.pkl", "rb"))["data_list"]
           if i["lidar2cam"]]  # 47115525 / 47115543 were rebuilt without frames
    print(f"ARKit first camera vs infos: {len(chk)} scenes, max |diff| {max(chk):.2e} m")
    frames = {}
    other = json.load(open(boxes_meta)) if boxes_meta else None
    for ds in ("scannet", "scannetpp", "arkitscenes"):
        meta = json.load(open(META.format(ds)))
        scenes, worst = {}, 0.0
        for sid, m in meta.items():
            if other is not None and sid not in other:
                continue
            boxes = [o for bl in m["object_bbox"].values() for o in bl]
            if ds == "scannet":
                M, t, rot, cam = sc[sid]
                origin = "first_camera"
            else:
                rot = room_yaw(boxes)
                cam = arkit_first_camera(sid) if ds == "arkitscenes" else scannetpp_first_camera(sid, boxes)
                origin = "first_camera" if cam is not None else "world"
                cam = np.zeros(3) if cam is None else cam
                M, t = rotz(-rot), -rotz(-rot) @ cam
            frames[f"{ds}/{sid}"] = {"room_yaw_deg": round(math.degrees(rot), 2),
                                     "origin_vsi": [round(float(v), 3) for v in np.linalg.solve(M, -t)],
                                     "origin": origin}
            lines = []
            src = other[sid] if other is not None else m
            for label, bl in src["object_bbox"].items():
                for o in bl:
                    axes = np.array(o["normalizedAxes"], float).reshape(3, 3) @ M.T   # rows = box axes, new frame
                    b, err = nine_dof({"centroid": (M @ np.array(o["centroid"], float) + t).tolist(),
                                       "normalizedAxes": axes.reshape(-1).tolist(), "axesLengths": o["axesLengths"]})
                    worst = max(worst, err)
                    lines.append(json.dumps({"label": RENAME.get(ds, {}).get(label, label),
                                             "bbox_3d": [round(float(v), 2) + 0.0 for v in b]}))
            scenes[sid] = "[\n" + ",\n".join(lines) + "\n]"
        n = skip = 0
        with open(f"{out_dir}/{prefix}_{ds}_scenelang.jsonl", "w") as f:
            for i, r in full.iterrows():
                if r.dataset != ds:
                    continue
                if r.scene_name not in scenes:
                    skip += 1
                    continue
                f.write(json.dumps({
                    "index": int(i), "id": int(r.id), "dataset": r.dataset, "debiased": int(r.id) in deb_ids,
                    "scene_name": r.scene_name, "question_type": r.question_type, "question": r.question,
                    "options": None if r.options is None else list(r.options), "ground_truth": str(r.ground_truth),
                    "scene_language": scenes[r.scene_name]}) + "\n")
                n += 1
        world = [k for k, v in frames.items() if k.startswith(ds + "/") and v["origin"] == "world"]
        print(f"{ds}: {len(scenes)} scenes, {n} rows ({skip} skipped: scene not in the boxes file), max rotation error {worst:.1e}, world origin kept: {world}")
    if other is None:
        json.dump(frames, open(f"{out_dir}/frames.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:5])
