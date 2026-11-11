"""Predicted scene language for the VSI-Bench ScanNet questions from the ES-56 fine-tuned VGG-T-Det
(runs/scannet_es56_ft_1gpu/epoch_80.pth; dump preds/scannet_es56_infos_val_vsi_seed<k>.pkl on the 88 VSI scenes).

Same conversion as ../scenelang_vsibench_vggtdet/build.py (reused: load_preds, first_camera, convert; rows in the eval
order of test_debiased + test_pruned), with two differences:
  * labels: the 56 ES-56 class names (already the VSI-Bench names: trash bin, tv, computer mouse, ...);
    --drop-structure removes wall / floor / ceiling boxes (default on: VSI-Bench has no such objects)
  * ScanNet only (the ES-56 model is ScanNet-only).
    python build.py <VSI-Bench dir> <out dir> [--seed 0] [--thr 0.3] [--keep-structure]
Same env as the original build.py (VGG-T-Det mmdet3d env, CPU).
"""
import argparse, json, pickle, sys
from collections import defaultdict
import numpy as np
import pandas as pd
sys.path.insert(0, "/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_vggtdet")
from build import VGGTDET, load_preds, scene_id, first_camera, convert  # noqa: E402

ES56 = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter',
        'desk', 'curtain', 'refrigerator', 'toilet', 'sink', 'bathtub', 'trash bin', 'pillow',
        'monitor', 'backpack', 'printer', 'lamp', 'mirror', 'towel', 'tv', 'radiator', 'nightstand',
        'plant', 'telephone', 'keyboard', 'microwave', 'clock', 'closet', 'fan', 'washing machine',
        'oven', 'mattress', 'computer mouse', 'guitar', 'piano', 'wall', 'floor', 'ceiling', 'book',
        'paper', 'box', 'socket', 'bottle', 'doorframe', 'clothes', 'cup', 'switch', 'blackboard',
        'bag', 'stool', 'shoe')
STRUCT = {'wall', 'floor', 'ceiling'}

ap = argparse.ArgumentParser()
ap.add_argument("bench"); ap.add_argument("out")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--thr", type=float, default=0.3)
ap.add_argument("--keep-structure", action="store_true")
a = ap.parse_args()
infos = pickle.load(open(f"{VGGTDET}/data/scannet/scannet_infos_val_vsi.pkl", "rb"))["data_list"]
preds = load_preds(f"{VGGTDET}/preds/scannet_es56_infos_val_vsi_seed{a.seed}.pkl")
assert len(preds) == len(infos)
scenes = {}
for info, (lp, b, s, l) in zip(infos, preds):
    sid = scene_id("scannet", info)
    assert lp is None or sid in lp, (sid, lp)
    cam = first_camera("scannet", info, 0.0)
    by_label = defaultdict(list)
    for bi, si, li in zip(b, s, l):
        name = ES56[int(li)]
        if si > a.thr and (a.keep_structure or name not in STRUCT):
            by_label[name].append((float(si), bi))
    objs = []
    for lab in sorted(by_label):
        items = sorted(by_label[lab], key=lambda x: -x[0])
        for n, (si, bi) in enumerate(items, 1):
            objs.append({"label": f"{lab} {n}" if len(items) > 1 else lab,
                         "bbox_3d": [round(float(t), 2) + 0.0 for t in convert(bi, 0.0, cam, False)], "score": round(si, 3)})
    scenes[sid] = {"dataset": "scannet", "scene_name": sid, "frame": "first_camera", "seed": a.seed, "score_thr": a.thr,
                   "first_camera_in_room": [round(float(t), 3) for t in cam], "objects": objs}
deb = pd.read_parquet(f"{a.bench}/test_debiased.parquet")
full = pd.concat([deb, pd.read_parquet(f"{a.bench}/test_pruned.parquet")], ignore_index=True)
deb_ids = set(deb.id)
with open(f"{a.out}/scenes_first_camera.jsonl", "w") as f:
    for v in scenes.values():
        f.write(json.dumps(v) + "\n")
n = 0
with open(f"{a.out}/rows_first_camera.jsonl", "w") as f:
    for i, r in full.iterrows():
        if r.dataset != "scannet":
            continue
        v = scenes[r.scene_name]
        lines = [json.dumps({"label": o["label"], "bbox_3d": o["bbox_3d"]}) for o in v["objects"]]
        f.write(json.dumps({
            "index": int(i), "id": int(r.id), "debiased": int(r.id) in deb_ids, "dataset": r.dataset,
            "scene_name": r.scene_name, "question_type": r.question_type, "question": r.question,
            "options": None if r.options is None else list(r.options), "ground_truth": str(r.ground_truth),
            "scene_language": "[\n" + ",\n".join(lines) + "\n]"}) + "\n")
        n += 1
print(f"{len(scenes)} scenes, {sum(len(v['objects']) for v in scenes.values())} boxes (thr {a.thr}); {n} rows -> {a.out}")
