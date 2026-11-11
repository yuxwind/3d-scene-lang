"""Self-contained HTML viewers of VSI-Bench GT, VGGT-Det (seed 0) and Map-Det3D boxes, in 3D and projected on the frames.

Per scene, one HTML file (three.js from unpkg, everything else inlined):
  * 3D: the scan's points (ScanNet, ARKitScenes; ScanNet++ has no geometry in the bundle, so only boxes and the
    camera path), the boxes of the three sources, the camera trajectory and the current frame's frustum.
  * 2D: --frames frames evenly spread over the scene, playable as a video, with every box projected (edges clipped at
    the near plane, so boxes behind the camera are not drawn).
Toggles per source, a score threshold per detector, labels on/off.

All geometry is in VSI-Bench's frame (the meta_info files'): ScanNet poses and points are moved by the raw-mesh mean
(build_meta.raw_means); ARKitScenes and ScanNet++ poses are already in it.

    python vis.py <dir with *_meta_info_val_mapdet3d.json> [--datasets ...] [--scenes ...] [--frames 60]
writes <dir>/vis/<dataset>/<scene>.html and <dir>/vis/index.html. Env: conda env mapdet3d, CPU.
"""
import argparse
import base64
import json
import os

import cv2
import numpy as np

from build_meta import META, VGGTDET, VGGTDET_META, corners_from_vsi, raw_means
from run import SCENES, Images, scannetpp_scenes

POINTS = {"scannet": f"{VGGTDET}/data/scannet/points/{{}}.bin", "arkitscenes": f"{VGGTDET}/data/arkit/points/{{}}_points.bin"}
MAX_W = 512
N_POINTS = 40000


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def boxes(meta, s, source):
    if meta is None or s not in meta:
        return []
    return [{"src": source, "label": label, "score": b.get("score", 1.0),
             "c": np.round(corners_from_vsi(b), 4).tolist()}
            for label, bs in meta[s]["object_bbox"].items() for b in bs]


def frames_data(frames, offset, n):
    ok = [i for i, (_, _, pose) in enumerate(frames) if np.all(np.isfinite(pose))]
    pick = [ok[int(round(j))] for j in np.linspace(0, len(ok) - 1, min(n, len(ok)))]
    load, out = Images(), []
    for i in sorted(set(pick)):
        src, K, c2w = frames[i]
        img = np.ascontiguousarray(load(src)[:, :, ::-1])  # BGR for cv2
        h, w = img.shape[:2]
        f = min(1.0, MAX_W / w)
        if f < 1:
            img = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
        c2w = np.array(c2w, np.float64)
        c2w[:3, 3] -= offset
        out.append({"img": base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).decode(),
                    "K": (np.array(K, np.float64)[:3, :3] * [[f], [f], [1]]).round(3).tolist(),
                    "w2c": np.round(np.linalg.inv(c2w), 5).tolist(), "c2w": np.round(c2w, 5).tolist(),
                    "w": img.shape[1], "h": img.shape[0], "idx": i})
    return out


def points_data(ds, s, offset):
    if ds not in POINTS or not os.path.exists(POINTS[ds].format(s)):
        return None
    p = np.fromfile(POINTS[ds].format(s), dtype=np.float32).reshape(-1, 6)
    if len(p) > N_POINTS:
        p = p[np.random.default_rng(0).choice(len(p), N_POINTS, replace=False)]
    xyz = (p[:, :3].astype(np.float64) - offset).astype(np.float32)
    return {"xyz": b64(xyz), "rgb": b64(np.clip(p[:, 3:], 0, 255).astype(np.uint8))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--datasets", nargs="*", default=["scannet", "arkitscenes", "scannetpp"])
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--frames", type=int, default=60)
    a = ap.parse_args()
    template = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vis_template.html")).read()

    index = []
    for ds in a.datasets:
        gt = json.load(open(META.format(ds)))
        md = json.load(open(f"{a.dir}/{ds}_meta_info_val_mapdet3d.json"))
        vf = VGGTDET_META.format(ds, 0)
        vg = json.load(open(vf)) if os.path.exists(vf) else None
        offsets = raw_means() if ds == "scannet" else {}
        scenes = SCENES[ds]() if ds != "scannetpp" else None  # 49 x 20 MB pose json: load per scene
        os.makedirs(f"{a.dir}/vis/{ds}", exist_ok=True)
        for s in sorted(md):
            if a.scenes and s not in a.scenes:
                continue
            off = offsets.get(s, np.zeros(3))
            data = {"scene": s, "dataset": ds,
                    "boxes": boxes(gt, s, "gt") + boxes(vg, s, "vggtdet") + boxes(md, s, "mapdet3d"),
                    "has": {"vggtdet": vg is not None and s in vg},
                    "points": points_data(ds, s, off),
                    "frames": frames_data(scenes[s] if scenes else scannetpp_scenes(names=[s])[s], off, a.frames)}
            html = template.replace("__TITLE__", f"{ds} {s}").replace("__DATA__", json.dumps(data, separators=(",", ":")))
            with open(f"{a.dir}/vis/{ds}/{s}.html", "w") as f:
                f.write(html)
            n = {k: sum(b["src"] == k for b in data["boxes"]) for k in ("gt", "vggtdet", "mapdet3d")}
            index.append((ds, s, n, data["has"]["vggtdet"]))
            print(ds, s, n, f"{len(html) / 1e6:.1f} MB", flush=True)

    rows = "\n".join(f"<tr><td>{ds}</td><td><a href='{ds}/{s}.html'>{s}</a></td><td>{n['gt']}</td>"
                     f"<td>{n['vggtdet'] if hv else '–'}</td><td>{n['mapdet3d']}</td></tr>" for ds, s, n, hv in index)
    with open(f"{a.dir}/vis/index.html", "w") as f:
        f.write(f"""<!doctype html><html><head><meta charset="utf-8"><title>VSI-Bench boxes</title>
<style>body{{font:14px system-ui,sans-serif;margin:16px;background:#fff;color:#111}}table{{border-collapse:collapse}}
td,th{{padding:3px 10px;border-bottom:1px solid #ddd;text-align:left}}a{{color:#0645ad}}</style></head><body>
<h2>VSI-Bench boxes: GT, VGGT-Det (seed 0), Map-Det3D</h2>
<table><tr><th>dataset</th><th>scene</th><th>GT</th><th>VGGT-Det</th><th>Map-Det3D</th></tr>
{rows}</table></body></html>""")


if __name__ == "__main__":
    main()
