"""Class-agnostic Map-Det3D on the VSI-Bench scenes (GPU part).

Inputs: the posed frames VGGT-Det was run on, <= 600 frames evenly spread over each scan.
  * scannet: $VGGTDET/data/scannet/posed_images/<scene>/: 00000.jpg + 00000.txt (camera-to-world, raw scan frame),
    intrinsic.txt; 1296x968. 88 / 88 VSI-Bench scenes.
  * arkitscenes: $VGGTDET/data/arkit/arkit_infos_val_vsi150.pkl: per frame img_paths (lowres_wide 256x192 png),
    cam2img, lidar2cam (camera-to-world in the ARKit mesh frame, checked by back-projecting depth onto the scan:
    ~2 cm vs up to 2 m the other way). 147 / 150 VSI-Bench scenes: 42897538 is not in the infos, 47115525 and
    47115543 have no frames.
  * scannetpp: frames decoded from VSI-Bench's mp4 (640x480, 30 fps), 600 evenly spaced; poses = ARKit aligned_pose
    (camera-to-world, mesh frame = VSI-Bench's frame) of iPhone frame 2i - 3, intrinsics scaled 1920x1440 -> 640x480,
    from $SCANNETPP (scannetpp_cam0 bundle, see its README). 49 / 50 scenes: e7af285f7d has no pose file.
Every --stride-th frame with a finite pose is streamed through Map-Det3D with scene-level tracking, in the settings of
the repo's mapdet3d_track_scannet config (NMS, score > 0.25, IoU 0.5, window 5). The model is class-agnostic
(num_classes = 1), so the output is boxes and scores only.

Per scene, writes <out>/preds/<dataset>/<scene>.npz: boxes (N, 10) = the final scene tracks in the frame of the
poses (Map-Det3D's world parametrisation: xyz centre, wlh, wxyz quaternion, ROS axes; use
mapdet3d.op.box3d.boxes3d_to_corners(.., AxisMode.ROS)), scores (N,), track_ids (N,), frames (indices into the
scene's frame list).

    python run.py <out dir> --dataset scannet|arkitscenes [--stride 5] [--scenes ...] [--shard 0 --nshards 1]
Env: conda env mapdet3d, HF_HOME / TORCH_HOME with the cached weights, HF_HUB_OFFLINE=1 on compute nodes.
"""
import argparse
import json
import os
import pickle
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from mapdet3d.model.mapdet3d import MapDet3D
from mapdet3d.op.mapdet3d.head import RoI2Det

VGGTDET = "/lustre/fsn1/projects/rech/bef/uxt97ti/vggtdet/data"
SCANNETPP = "/lustre/fsn1/projects/rech/bef/uxt97ti/data/scannetpp_cam0"
VSIBENCH = "/lustre/fsn1/projects/rech/bef/uxt97ti/data/VSI-Bench"


def scannet_scenes():
    """Scene -> [(image path, K 3x3, camera-to-world 4x4)]."""
    root = f"{VGGTDET}/scannet/posed_images"
    out = {}
    for s in sorted(os.listdir(root)):
        d = f"{root}/{s}"
        K = np.loadtxt(f"{d}/intrinsic.txt").astype(np.float32)[:3, :3]
        ids = sorted(int(f[:-4]) for f in os.listdir(d) if f.endswith(".jpg"))
        out[s] = [(f"{d}/{i:05d}.jpg", K, np.loadtxt(f"{d}/{i:05d}.txt").astype(np.float32)) for i in ids]
    return out


def arkit_scenes():
    out = {}
    for info in pickle.load(open(f"{VGGTDET}/arkit/arkit_infos_val_vsi150.pkl", "rb"))["data_list"]:
        if info["img_paths"]:
            out[info["lidar_points"]["lidar_path"].split("_")[0]] = [
                (f"{VGGTDET}/arkit/{p}", np.asarray(K, np.float32)[:3, :3], np.asarray(c2w, np.float32))
                for p, K, c2w in zip(info["img_paths"], info["cam2img"], info["lidar2cam"])]
    return out


def scannetpp_scenes(n=600, names=None):
    """Frames decoded from VSI-Bench's own mp4 (640x480, 30 fps): n evenly spaced ones, as ("mp4 path", index)."""
    out = {}
    for s in names or sorted(os.listdir(f"{SCANNETPP}/scannetpp/data")):
        poses = json.load(open(f"{SCANNETPP}/scannetpp/data/{s}/iphone/pose_intrinsic_imu.json"))
        keys = sorted(poses)
        mp4 = f"{VSIBENCH}/scannetpp/{s}.mp4"
        cap = cv2.VideoCapture(mp4)
        n_mp4 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        frames = []
        for i in np.unique(np.linspace(0, n_mp4 - 1, n).round().astype(int)):
            p = poses[keys[min(max(2 * i - 3, 0), len(keys) - 1)]]  # mp4 frame i == pose frame 2i - 3 (bundle README)
            K = np.asarray(p["intrinsic"], np.float32) * np.array([[1 / 3], [1 / 3], [1]], np.float32)  # 1920x1440 -> 640x480
            frames.append(((mp4, int(i)), K, np.asarray(p["aligned_pose"], np.float32)))
        out[s] = frames
    return out


SCENES = {"scannet": scannet_scenes, "arkitscenes": arkit_scenes, "scannetpp": scannetpp_scenes}


class Images:
    """Loads a frame given as an image path or as (mp4 path, index); mp4 indices must be requested in increasing order."""

    def __init__(self):
        self.cap, self.mp4, self.pos = None, None, 0

    def __call__(self, src):
        if isinstance(src, str):
            return np.array(ImageOps.exif_transpose(Image.open(src)).convert("RGB"))
        mp4, i = src
        if mp4 != self.mp4:
            self.cap, self.mp4, self.pos = cv2.VideoCapture(mp4), mp4, 0
        while self.pos < i:
            self.cap.grab()
            self.pos += 1
        ok, bgr = self.cap.read()
        assert ok, src
        self.pos += 1
        return bgr[:, :, ::-1]


def run_scene(model, frames, stride, device):
    used = [i for i, (_, _, pose) in enumerate(frames) if np.all(np.isfinite(pose))][::stride]
    load = Images()
    pred = None
    for fid, i in enumerate(used):  # frame id 0 resets the window and the tracks
        path, K, pose = frames[i]
        img = load(path).astype(np.float32)
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)[None])).to(device)
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            pred = model(images=[img], intrinsics=[torch.from_numpy(K).to(device)],
                         extrinsics=[torch.from_numpy(pose).to(device)], frame_ids=[fid])
    return (pred.boxes3d[0].float().cpu().numpy(), pred.scores[0].float().cpu().numpy(),
            pred.track_ids[0].cpu().numpy(), np.array(used))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dataset", choices=SCENES, required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")
    device = "cuda"

    model = MapDet3D.from_pretrained("RoyYang0714/Map-Det3D").to(device)
    model.track_whole_scene = True
    model.roi2det = RoI2Det(nms=True, score_threshold=0.25, iou_threshold=0.5)
    model.eval()

    scenes = SCENES[a.dataset]()
    names = a.scenes or sorted(scenes)
    out = f"{a.out}/preds/{a.dataset}"
    os.makedirs(out, exist_ok=True)
    for s in names[a.shard::a.nshards]:
        path = f"{out}/{s}.npz"
        if os.path.exists(path):  # resubmitting finishes a timed-out shard
            continue
        t = time.time()
        with torch.no_grad():
            boxes, scores, tids, used = run_scene(model, scenes[s], a.stride, device)
        np.savez(path, boxes=boxes, scores=scores, track_ids=tids, frames=used)
        print(f"{s}: {len(used)} frames, {len(boxes)} boxes, {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
