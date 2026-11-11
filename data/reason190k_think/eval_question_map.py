"""Per-question mAP@0.25 / mAP@0.5, precision and recall of three box sources on VSI-Bench, all on eval_scene_map.py's
protocol, so that the detectors are scored the way the CoT model is:
  * qwen:      the <scene> block of each CoT answer (novggt_qwen35_4b_reason190k_think), score 1, one rigid frame per
               scene fitted by RANSAC exactly as eval_scene_map.py does (same seed and order, so the same frames);
  * mapdet3d:  Map-Det3D's boxes with GT labels at IoU >= 0.1 (../scenelang_vsibench_mapdet3d/*_oracle0.1.json),
               their own scores, oriented boxes in the meta_info frame (no fit needed);
  * vggtdet:   VGGT-Det seed 0 with its own labels (../scenelang_vsibench_vggtdet/*_seed0.json), scores, meta frame.
Each question is one "image": GT = the meta_info boxes of the labels the question names (room-size questions name
none: the labels the method predicts that exist in the scene). Every method keeps only its boxes of those labels
(--keep-all-qwen keeps the CoT's other boxes too, which is eval_scene_map.py's own scoring). A question whose scene a
method has no output for is scored as no boxes. Precision / recall: one-to-one, same label, IoU > thr, boxes taken in
score order.

    python eval_question_map.py <qwen predictions dir> <vsi_full_rows.json> <VSI-Bench meta_info dir> [--json out]
(VGGT-Det mmdet3d env, CPU)
"""
import argparse
import glob
import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_scene_map import (RENAME, boxes3d, fit_frame, gt_boxes, parse_scene, pr_at,  # noqa: E402
                            question_labels, rigid)
from mmdet3d.evaluation.functional.indoor_eval import indoor_eval  # noqa: E402
from mmdet3d.structures import Box3DMode  # noqa: E402

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DET = {"mapdet3d": f"{D}/scenelang_vsibench_mapdet3d/{{}}_meta_info_val_mapdet3d_oracle0.1.json",
       "vggtdet": f"{D}/scenelang_vsibench_vggtdet/{{}}_meta_info_val_vggtdet_seed0.json"}


def det_boxes(meta):
    """{scene: [(label, [x, y, z, dx, dy, dz, yaw], score)]}, sorted by descending score."""
    lab = gt_boxes({s: {"object_bbox": {k: v for k, v in m["object_bbox"].items()}} for s, m in meta.items()})
    out = {}
    for s, m in meta.items():
        sc = [b.get("score", 1.0) for _, bl in m["object_bbox"].items() for b in bl]
        out[s] = sorted([(l, b, c) for (l, b), c in zip(lab[s], sc)], key=lambda x: -x[2])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir"); ap.add_argument("rows"); ap.add_argument("meta_dir")
    ap.add_argument("--json", default="")
    ap.add_argument("--keep-all-qwen", action="store_true")
    a = ap.parse_args()
    rows = json.load(open(a.rows))
    meta, det = {}, {k: {} for k in DET}
    for d in ("scannet", "scannetpp", "arkitscenes"):
        for s, m in json.load(open(f"{a.meta_dir}/{d}_meta_info_val.json")).items():
            meta[(d, s)] = m
        for k, f in DET.items():
            if os.path.exists(f.format(d)):
                det[k].update({(d, s): v for s, v in det_boxes(json.load(open(f.format(d)))).items()})
    GT = gt_boxes(meta)
    vggt_scenes = set(det["vggtdet"])

    preds = {}
    for f in sorted(glob.glob(f"{a.pred_dir}/predictions_full_shard*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            preds[r["index"]] = r

    # the CoT boxes and frames exactly as eval_scene_map.py builds them
    qs = []
    for idx, r in sorted(preds.items()):
        row = rows[idx]
        key = (row["dataset"], row["scene_name"])
        labels = {lab for lab, _ in GT[key]}
        named = question_labels(row["question"], labels) if not row["question_type"].startswith("room_size") else set()
        qs.append({"key": key, "dataset": row["dataset"], "type": row["question_type"], "named": named,
                   "labels": labels, "qwen": parse_scene(r.get("raw", "")) or []})
    rng = np.random.default_rng(0)
    by_scene = defaultdict(list)
    for q in qs:
        by_scene[q["key"]].extend((lab, b[:3]) for lab, b in q["qwen"])
    frames = {k: fit_frame(p, [(lab, b[:3]) for lab, b in GT[k]], rng) for k, p in by_scene.items()}

    def method_boxes(q, m):
        """(want, [(label, 7-vector, score)]) of method m for question q."""
        if m == "qwen":
            f = frames[q["key"]]
            bx = []
            if f is not None:
                R, t = rigid(f[0], f[1])
                bx = [(lab, [*(R @ np.array(b[:3]) + t), *b[3:6], f[0]], 1.0) for lab, b in q["qwen"]]
        else:
            bx = [(RENAME.get(l, l), b, s) for l, b, s in det[m].get(q["key"], [])]
        want = q["named"] or ({lab for lab, _, _ in bx} & q["labels"])
        if not (m == "qwen" and a.keep_all_qwen):
            bx = [x for x in bx if x[0] in want]
        return want, bx

    cats = sorted({lab for q in qs for lab in q["labels"]} | {lab for q in qs for lab, _ in q["qwen"]})
    cid = {c: i for i, c in enumerate(cats)}
    logging.getLogger("mmengine").setLevel(logging.ERROR)

    def score(sel, m):
        dt, gt = [], []
        for q in sel:
            want, bx = method_boxes(q, m)
            gb = [b for lab, b in GT[q["key"]] if lab in want]
            dt.append({"labels_3d": torch.tensor([cid[l] for l, _, _ in bx], dtype=torch.long),
                       "bboxes_3d": boxes3d([b for _, b, _ in bx]),
                       "scores_3d": torch.tensor([s for _, _, s in bx], dtype=torch.float32)})
            gt.append({"gt_labels_3d": np.array([cid[lab] for lab, _ in GT[q["key"]] if lab in want], np.int64),
                       "gt_bboxes_3d": boxes3d(gb)})
        has_gt = {int(l) for g in gt for l in g["gt_labels_3d"]}
        dt_map = []
        for d in dt:
            k = torch.tensor([int(l) in has_gt for l in d["labels_3d"]], dtype=torch.bool)
            dt_map.append({"labels_3d": d["labels_3d"][k], "bboxes_3d": d["bboxes_3d"][k] if len(k) else d["bboxes_3d"],
                           "scores_3d": d["scores_3d"][k]})
        r = indoor_eval(gt, dt_map, [0.25, 0.5], {i: c for c, i in cid.items()}, logger="silent",
                        box_mode_3d=Box3DMode.DEPTH)
        out = {"questions": len(sel), "pred_boxes": sum(len(d["labels_3d"]) for d in dt),
               "gt_boxes": sum(len(g["gt_labels_3d"]) for g in gt)}
        for thr in (0.25, 0.5):
            p, rc = pr_at(dt, gt, thr)
            out[f"mAP@{thr}"] = round(float(r[f"mAP_{thr:.2f}"]), 4)
            out[f"P@{thr}"], out[f"R@{thr}"] = round(p, 4), round(rc, 4)
        return out

    groups = [("scannet", [q for q in qs if q["dataset"] == "scannet"]),
              ("arkitscenes (vggtdet's 82 scenes)", [q for q in qs if q["key"] in vggt_scenes and q["dataset"] == "arkitscenes"]),
              ("arkitscenes (all)", [q for q in qs if q["dataset"] == "arkitscenes"]),
              ("scannetpp", [q for q in qs if q["dataset"] == "scannetpp"]),
              ("all", qs)]
    res = {}
    for name, sel in groups:
        for m in ("mapdet3d", "vggtdet", "qwen"):
            if m == "vggtdet" and not all(q["key"] in vggt_scenes for q in sel):
                continue
            r = res.setdefault(name, {})[m] = score(sel, m)
            print(f"{name:34s} {m:9s} q={r['questions']:5d} pred={r['pred_boxes']:6d} gt={r['gt_boxes']:6d}  "
                  f"mAP25={r['mAP@0.25']:.4f} mAP50={r['mAP@0.5']:.4f}  P25={r['P@0.25']:.3f} R25={r['R@0.25']:.3f} "
                  f"P50={r['P@0.5']:.3f} R50={r['R@0.5']:.3f}", flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
