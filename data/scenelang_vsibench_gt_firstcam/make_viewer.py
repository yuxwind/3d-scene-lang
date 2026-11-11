"""Write docs/scene_viewer.html: a self-contained 3D viewer of the VSI-Bench GT scenes, in the training frame built by
build.py (origin = first-frame camera, axes along the room, z up) and, for comparison, in VSI-Bench's world frame
(data/scenelang_vsibench_scannetpp). Boxes are the 9-DoF [x, y, z, dx, dy, dz, yaw, roll, pitch] of the scene files
(R = Rz(yaw) Rx(roll) Ry(pitch)); the page can also draw the upright axis-aligned envelope that eval_scenelang.py's
rvs_scene gives the fine-tuned model.

    python make_viewer.py            (plain python3, no packages)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OLD = os.path.join(ROOT, "data/scenelang_vsibench_scannetpp/vsibench_{}_scenelang.jsonl")
NEW = os.path.join(HERE, "vsibench_{}_scenelang.jsonl")


def scenes(path):
    out, nq = {}, {}
    for line in open(path):
        r = json.loads(line)
        s = r["scene_name"]
        nq[s] = nq.get(s, 0) + 1
        if s not in out:
            out[s] = [[o["label"]] + o["bbox_3d"] for o in json.loads(r["scene_language"])]
    return out, nq


def main():
    frames = json.load(open(os.path.join(HERE, "frames.json")))
    data = []
    for ds in ("scannet", "scannetpp", "arkitscenes"):
        new, nq = scenes(NEW.format(ds))
        old, _ = scenes(OLD.format(ds))
        for s in sorted(new):
            f = frames[f"{ds}/{s}"]
            data.append({"ds": ds, "s": s, "nq": nq[s], "yaw": f["room_yaw_deg"], "origin": f["origin"],
                         "o_vsi": f["origin_vsi"], "new": new[s], "old": old[s]})
    html = open(os.path.join(HERE, "viewer_template.html")).read().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    out = os.path.join(ROOT, "docs/scene_viewer.html")
    open(out, "w").write(html)
    print(f"{len(data)} scenes -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
