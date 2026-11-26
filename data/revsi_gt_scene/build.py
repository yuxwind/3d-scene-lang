"""ReVSI (64_frame subset, 6,616 questions) with ground-truth scene language, as text batches for the Claude no-code runs.

Scene source: ReVSI's own metadata/3d_annotation.json (381 scenes; ARKitScenes, ScanNet v2, ScanNet++ v2, MultiScan,
3RScan): every object -> {"label": name, "bbox_3d": [x, y, z, dx, dy, dz, yaw, roll, pitch]} (centre + extents in m,
intrinsic ZXY Euler as data/scenelang_vsibench_scannetpp/build.py), in the scan's world frame (z up), two decimals,
all annotated objects (not only those visible in the 64 frames), in the file's order. The floor polygon / room area
is left out (it is the answer to the room-size questions). obb.rotation is read as Open3D does: its COLUMNS are the
box axes. A per-dataset check prints how many boxes have an axis within 25 deg of vertical (should be ~100%).

Batches use the prompt of the VSI-Bench full-GT no-code runs (claude_nocode_scenelang_fullgt_all): the same scene
description, then the question, "Options:" + one option per line + "Answer with the option letter." for multiple
choice, "Answer with a number only, in the unit the question asks for." otherwise. Batches are per question type,
<= --max_q questions and <= --max_chars characters, and hold AT MOST ONE QUESTION PER SCENE (owner's rule, 2026-11-22:
questions on the same scene must not see each other's options); each question goes to the first batch of its type
that has no question on its scene yet and still has room.

    python build.py --revsi_dir R --out_dir D [--max_q 40] [--max_chars 150000]
writes D/batches/<question_type>_<NNN>.txt, D/rows.jsonl (id, dataset, scene_id, question_type, ground_truth, batch),
D/manifest.json ({batch: [ids]}), D/scene_check.json. Needs numpy + pandas (jz: remote.py run --env h100).
"""
import argparse, collections, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd

INTRO = """This is a 3D scene, given as a list of objects, one per line: {"label": <category>, "bbox_3d": [x, y, z, dx, dy, dz, yaw, roll, pitch]}.
- x, y, z: the centre of the object's 3D bounding box, in meters. x and y are horizontal coordinates on the floor plane; z is the vertical axis (height, pointing up). z = 0 is not the floor: the floor is wherever the lowest objects sit.
- dx, dy, dz: the full size of the box in meters along its own axes. dx and dy are horizontal sizes; dz is the vertical height of the object.
- yaw: rotation of the box around the vertical z axis, in radians (yaw = 0 means dx runs along x and dy along y). roll and pitch are tilts around the box's own x and y axes, in radians; they are 0 for upright objects.
- Lines with the same label are different instances of that category. All lengths are in meters.
"""
MCQ_POST, NUM_POST = "Answer with the option letter.", "Answer with a number only, in the unit the question asks for."


def nine_dof(center, axes_rows, ext):  # = data/scenelang_vsibench_gt_firstcam/build.py nine_dof
    axes = np.asarray(axes_rows, float)
    v = int(np.argmax(np.abs(axes[:, 2])))
    h = [i for i in range(3) if i != v]
    z = axes[v] * np.sign(axes[v, 2] or 1.0)
    x = axes[h[0]] - z * (axes[h[0]] @ z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    dx, dy, dz = ext[h[0]], ext[h[1]], ext[v]
    R = np.stack([x, y, z], 1)
    yaw = math.atan2(-R[0, 1], R[1, 1])
    if not -math.pi / 2 <= yaw < math.pi / 2:
        R = R * np.array([-1.0, -1.0, 1.0]); yaw = math.atan2(-R[0, 1], R[1, 1])
    roll = math.asin(max(-1.0, min(1.0, R[2, 1])))
    pitch = math.atan2(-R[2, 0], R[2, 2])
    return [*center, dx, dy, dz, yaw, roll, pitch], abs(axes[v, 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revsi_dir", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--subset", default="64_frame"); ap.add_argument("--max_q", type=int, default=40)
    ap.add_argument("--max_chars", type=int, default=150000)
    a = ap.parse_args()
    R, D = Path(a.revsi_dir), Path(a.out_dir)
    (D / "batches").mkdir(parents=True, exist_ok=True)
    scenes, check = {}, collections.defaultdict(lambda: [0, 0])
    for s in json.load(open(R / "metadata" / "3d_annotation.json")):
        lines = []
        for o in s["objects"]:
            b = o["obb"]
            box, vert = nine_dof(b["center"], np.asarray(b["rotation"], float).T, b["extent"])
            check[s["dataset"]][0] += 1; check[s["dataset"]][1] += vert > math.cos(math.radians(25))
            lines.append(json.dumps({"label": o["name"], "bbox_3d": [round(float(v), 2) + 0.0 for v in box]}))
        scenes[(s["dataset"], str(s["scene_id"]))] = "[\n" + ",\n".join(lines) + "\n]"
    df = pd.read_parquet(next((R / a.subset).glob("*.parquet")))
    groups = collections.defaultdict(list)
    missing = 0
    for r in df.itertuples():
        key = (r.dataset, str(r.scene_id))
        if key not in scenes:
            missing += 1; continue
        opts = r.options
        has_opts = opts is not None and not (isinstance(opts, str) and opts in ("None", "")) and len(opts) > 0
        if has_opts:
            ol = list(opts) if not isinstance(opts, str) else eval(opts)
            tail = "Options:\n" + "\n".join(map(str, ol)) + f"\n{MCQ_POST}\n"
        else:
            tail = NUM_POST + "\n"
        block = f"=== QUESTION id={r.id} ===\n{INTRO}<scene>\n{scenes[key]}\n</scene>\n{r.question}\n{tail}\n"
        groups[r.question_type].append((int(r.id), block, r))
    man, rows, sizes = {}, [], []
    for qt, items in sorted(groups.items()):
        bins = []  # each: {"scenes": set, "items": [...], "chars": int}
        for it in items:
            sc = (it[2].dataset, str(it[2].scene_id))
            b = next((b for b in bins if sc not in b["scenes"] and len(b["items"]) < a.max_q
                      and b["chars"] + len(it[1]) <= a.max_chars), None)
            if b is None:
                b = {"scenes": set(), "items": [], "chars": 0}; bins.append(b)
            b["scenes"].add(sc); b["items"].append(it); b["chars"] += len(it[1])
        for k, b in enumerate(bins):
            name = f"{qt}_{k:03d}"
            (D / "batches" / f"{name}.txt").write_text("".join(x[1] for x in b["items"]))
            man[name] = [x[0] for x in b["items"]]; sizes.append(b["chars"])
            for i, _, r in b["items"]:
                rows.append({"id": i, "dataset": r.dataset, "scene_id": str(r.scene_id), "question_type": r.question_type,
                             "ground_truth": str(r.ground_truth), "batch": name})
    # the rule, checked
    per = collections.Counter((r["batch"], r["dataset"], r["scene_id"]) for r in rows)
    assert max(per.values()) == 1, "a batch has two questions on one scene"
    json.dump(man, open(D / "manifest.json", "w"), indent=0)
    with open(D / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    chk = {ds: {"boxes": t, "vertical_axis_pct": round(100 * v / max(t, 1), 1)} for ds, (t, v) in check.items()}
    json.dump(chk, open(D / "scene_check.json", "w"), indent=1)
    print(f"{len(rows)} questions in {len(man)} batches ({missing} without a scene); chars/batch: "
          f"mean {sum(sizes) / len(sizes):.0f}, max {max(sizes)}, total {sum(sizes)} (~{sum(sizes) / 4 / 1e6:.1f}M tokens of input)")
    for ds, c in sorted(chk.items()):
        print(f"  {ds:12} boxes {c['boxes']:6}  vertical axis within 25 deg: {c['vertical_axis_pct']}%")


if __name__ == "__main__":
    main()
