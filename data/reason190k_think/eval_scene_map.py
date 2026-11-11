"""mAP@0.25 / mAP@0.5 of the <scene> boxes that the CoT model (novggt_qwen35_4b_reason190k_think) writes on VSI-Bench.

Metric: mmdet3d indoor_eval as used by VGGT-Det (../VGGT-Det-CVPR2026, IndoorMetric iou_thr [0.25, 0.5]): per-class AP
(area under the PR curve, greedy matching, IoU > thr) on DepthInstance3DBoxes, mAP = mean over classes. Each question
is one "image":
  * predictions: the boxes of the CoT's <scene> block ([x, y, z, w, d, h], upright, axis-aligned in its own frame),
    label numbering ("chair 2") stripped, all with score 1 (the model gives no confidence, so AP follows list order;
    precision / recall at each IoU are printed as well);
  * ground truth: VSI-Bench meta_info boxes (the file its QA was generated from) of every label the question names
    (the eval_scenelang.py rvs_question rule); room-size questions name none, so their GT is every instance of the
    labels predicted.
Frame: the CoT frame is the first sampled camera (z up, axes along the room), whose pose is unknown, so one rigid
transform per scene (yaw about z + 3D translation, 4 DoF) is fitted by RANSAC over same-label box centres pooled over
all the scene's questions, then refined by least squares on the inliers. This is an oracle alignment of the origin
only: positions relative to each other, sizes and shapes are scored as written.

--aabb-gt: the GT boxes are instead the scenes of data/scenelang_vsibench_gt_firstcam (the training frame: origin at
the first-frame camera, axes along the room, z up), each turned into the upright axis-aligned box around its
yaw-rotated footprint (eval_scenelang.py's rvs_scene), i.e. the box format the CoT writes. Oriented objects then no
longer cap the IoU, so the ceiling is ~1. --no-fit skips the per-scene frame fit (identity), which scores the CoT
boxes in the training frame as written and so also tests that this frame is the traces' one.

usage (VGGT-Det mmdet3d env, CPU):
  python eval_scene_map.py <predictions_full_shard*.jsonl dir> <rows.json: index -> dataset, scene_name, question>
         <VSI-Bench meta_info dir> [--json out.json]
"""
import argparse
import glob
import json
import math
import re
from collections import defaultdict

import numpy as np
import torch
from mmdet3d.evaluation.functional.indoor_eval import indoor_eval
from mmdet3d.structures import Box3DMode, DepthInstance3DBoxes

RENAME = {"coat hanger": "coat rack"}  # ScanNet++ meta label -> the name the questions use
SCENE = re.compile(r"<scene>\s*(\[.*?\])\s*</scene>", re.S)
INLIER = 0.5  # m, centre distance for a RANSAC inlier


def norm_label(s):
    return re.sub(r"\s+\d+$", "", s.strip().lower())


def gt_boxes(meta):
    """{scene: [(label, [x, y, z, dx, dy, dz, yaw])]}: oriented boxes -> upright yaw boxes (vsi_gt.obb_to_yawbox)."""
    out = {}
    for s, m in meta.items():
        boxes = []
        for lab, bl in m["object_bbox"].items():
            for b in bl:
                axes = np.array(b["normalizedAxes"], float).reshape(3, 3)
                lens = np.array(b["axesLengths"], float)
                up = int(np.argmax(np.abs(axes[:, 2])))
                i, j = [k for k in range(3) if k != up]
                boxes.append((RENAME.get(lab, lab).lower(),
                              [*b["centroid"], lens[i], lens[j], lens[up], math.atan2(axes[i, 1], axes[i, 0])]))
        out[s] = boxes
    return out


def aabb_gt(gt_dir):
    """{(dataset, scene): [(label, [x, y, z, w, d, h, 0])]}: training-frame GT scenes as upright axis-aligned boxes."""
    out = {}
    for d in ("scannet", "scannetpp", "arkitscenes"):
        for line in open(f"{gt_dir}/vsibench_{d}_scenelang.jsonl"):
            r = json.loads(line)
            if (d, r["scene_name"]) in out:
                continue
            boxes = []
            for o in json.loads(r["scene_language"]):
                x, y, z, dx, dy, dz, yaw = o["bbox_3d"][:7]
                c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
                boxes.append((o["label"].lower(), [x, y, z, dx * c + dy * s, dx * s + dy * c, dz, 0.0]))
            out[(d, r["scene_name"])] = boxes
    return out


def parse_scene(raw):
    m = SCENE.search(raw)
    if not m:
        return None
    try:
        objs = json.loads(m.group(1))
        return [(norm_label(o["label"]), [float(v) for v in o["bbox_3d"]][:6]) for o in objs
                if len(o.get("bbox_3d", [])) >= 6]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def question_labels(q, labels):
    q, spans, keep = q.lower(), [], set()
    for lab in sorted(labels, key=len, reverse=True):
        for m in re.finditer(r"\b" + re.escape(lab) + r"(?:s|es)?\b", q):
            if not any(a <= m.start() < b for a, b in spans):
                spans.append((m.start(), m.end())); keep.add(lab)
    return keep


def rigid(yaw, t):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]), np.asarray(t, float)


def fit_frame(pred, gt, rng, iters=3000):
    """pred, gt: [(label, centre)]. Returns (yaw, t, inlier fraction, mean inlier error) or None."""
    by_lab = defaultdict(list)
    for j, (lab, c) in enumerate(gt):
        by_lab[lab].append(np.asarray(c, float))
    P = [(lab, np.asarray(c, float)) for lab, c in pred if lab in by_lab]
    if len(P) < 2:
        return None

    def residuals(R, t):
        return np.array([min(np.linalg.norm(R @ p + t - g) for g in by_lab[lab]) for lab, p in P])

    best = (-1, None)
    for _ in range(iters):
        a, b = rng.choice(len(P), 2, replace=False)
        (la, pa), (lb, pb) = P[a], P[b]
        ga = by_lab[la][rng.integers(len(by_lab[la]))]
        gb = by_lab[lb][rng.integers(len(by_lab[lb]))]
        dp, dg = (pb - pa)[:2], (gb - ga)[:2]
        if np.linalg.norm(dp) < 0.3 or abs(np.linalg.norm(dp) - np.linalg.norm(dg)) > INLIER:
            continue
        yaw = math.atan2(dg[1], dg[0]) - math.atan2(dp[1], dp[0])
        R, _ = rigid(yaw, 0)
        t = (ga + gb) / 2 - R @ (pa + pb) / 2
        n = int((residuals(R, t) < INLIER).sum())
        if n > best[0]:
            best = (n, (yaw, t))
    if best[1] is None:
        return None
    yaw, t = best[1]
    for _ in range(5):  # least squares on the inliers (2D Procrustes for yaw, mean for t)
        R, _ = rigid(yaw, 0)
        pairs = []
        for lab, p in P:
            g = min(by_lab[lab], key=lambda g: np.linalg.norm(R @ p + t - g))
            if np.linalg.norm(R @ p + t - g) < INLIER:
                pairs.append((p, g))
        if len(pairs) < 2:
            break
        A, B = np.array([p for p, _ in pairs]), np.array([g for _, g in pairs])
        a0, b0 = A - A.mean(0), B - B.mean(0)
        yaw = math.atan2((a0[:, 0] * b0[:, 1] - a0[:, 1] * b0[:, 0]).sum(), (a0[:, :2] * b0[:, :2]).sum())
        R, _ = rigid(yaw, 0)
        t = B.mean(0) - R @ A.mean(0)
    r = residuals(*rigid(yaw, t))
    inl = r < INLIER
    return yaw, t, float(inl.mean()), float(r[inl].mean()) if inl.any() else float("nan")


def boxes3d(arr):
    arr = np.asarray(arr, np.float32).reshape(-1, 7)
    return DepthInstance3DBoxes(torch.from_numpy(arr), origin=(0.5, 0.5, 0.5))


def pr_at(dt, gt, thr):
    """Precision / recall over all questions with one-to-one same-label matching at IoU > thr (score-free)."""
    tp = n_pred = n_gt = 0
    for d, g in zip(dt, gt):
        n_pred += len(d["labels_3d"]); n_gt += len(g["gt_labels_3d"])
        if not len(d["labels_3d"]) or not len(g["gt_labels_3d"]):
            continue
        iou = DepthInstance3DBoxes.overlaps(d["bboxes_3d"], g["gt_bboxes_3d"]).numpy()
        iou[d["labels_3d"].numpy()[:, None] != np.asarray(g["gt_labels_3d"])[None, :]] = 0
        used = set()
        for i in range(iou.shape[0]):
            for j in np.argsort(-iou[i]):
                if iou[i, j] <= thr:
                    break
                if j not in used:
                    used.add(j); tp += 1
                    break
    return tp / max(n_pred, 1), tp / max(n_gt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir"); ap.add_argument("rows"); ap.add_argument("meta_dir")
    ap.add_argument("--json", default="")
    ap.add_argument("--selftest", action="store_true",
                    help="replace each <scene> by the question's GT boxes written the CoT way (upright and axis-aligned in "
                         "a room-aligned frame, random rigid offset per scene): the ceiling of the protocol")
    ap.add_argument("--aabb-gt", default="", metavar="GT_DIR",
                    help="score against the axis-aligned training-frame GT of GT_DIR (data/scenelang_vsibench_gt_firstcam)")
    ap.add_argument("--no-fit", action="store_true", help="identity frame instead of the per-scene RANSAC fit")
    args = ap.parse_args()
    rows = json.load(open(args.rows))
    meta = {}
    for d in ("scannet", "scannetpp", "arkitscenes"):
        for s, m in json.load(open(f"{args.meta_dir}/{d}_meta_info_val.json")).items():
            meta[(d, s)] = m
    GT = aabb_gt(args.aabb_gt) if args.aabb_gt else gt_boxes(meta)  # keyed (dataset, scene)

    preds = {}
    for f in sorted(glob.glob(f"{args.pred_dir}/predictions_full_shard*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            preds[r["index"]] = r
    print(f"{len(preds)} predictions")

    # per question: predicted boxes and GT labels
    qs, no_scene = [], 0
    for idx, r in sorted(preds.items()):
        row = rows[idx]
        key = (row["dataset"], row["scene_name"])
        sc = parse_scene(r.get("raw", ""))
        if sc is None:
            no_scene += 1
            sc = []
        labels = {lab for lab, _ in GT[key]}
        want = (question_labels(row["question"], labels) if not row["question_type"].startswith("room_size")
                else set())
        if not want:
            want = {lab for lab, _ in sc} & labels
        qs.append({"idx": idx, "key": key, "dataset": row["dataset"], "type": row["question_type"],
                   "pred": sc, "want": want})
    print(f"{no_scene} predictions without a parseable <scene> block (scored as no boxes)")
    if args.selftest:
        rng0 = np.random.default_rng(1)
        offset = {}
        for key in {q["key"] for q in qs}:
            yaws = np.array([b[6] for _, b in GT[key]])
            room = math.atan2(np.sin(4 * yaws).sum(), np.cos(4 * yaws).sum()) / 4  # dominant direction mod 90 deg
            offset[key] = (int(rng0.integers(4)), rng0.uniform(-3, 3, 3), room)  # k * 90 deg keeps boxes axis-aligned
        for q in qs:
            k, t, room = offset[q["key"]]
            R, _ = rigid(k * math.pi / 2 - room, 0)  # world -> room frame -> k quarter turns
            want = q["want"] or {lab for lab, _ in GT[q["key"]]}
            q["pred"] = []
            for lab, (x, y, z, dx, dy, dz, yaw) in GT[q["key"]]:
                if lab in want:
                    a = yaw - room  # box yaw in the room frame -> axis-aligned extents
                    w, d = abs(dx * math.cos(a)) + abs(dy * math.sin(a)), abs(dx * math.sin(a)) + abs(dy * math.cos(a))
                    w, d = (d, w) if k % 2 else (w, d)
                    q["pred"].append((lab, [*(R @ np.array([x, y, z]) + t), w, d, dz]))
            if not q["want"]:
                q["want"] = want

    # one frame per scene
    rng = np.random.default_rng(0)
    frames, fit_stats = {}, []
    by_scene = defaultdict(list)
    for q in qs:
        by_scene[q["key"]].extend((lab, b[:3]) for lab, b in q["pred"])
    for key, pred in by_scene.items():
        f = (0.0, np.zeros(3), float("nan"), float("nan")) if args.no_fit else \
            fit_frame(pred, [(lab, b[:3]) for lab, b in GT[key]], rng)
        frames[key] = f
        if f:
            fit_stats.append((f[2], f[3]))
    print(f"frames fitted for {sum(f is not None for f in frames.values())}/{len(frames)} scenes; "
          f"median inlier fraction {np.median([a for a, _ in fit_stats]):.2f}, "
          f"median inlier centre error {np.nanmedian([b for _, b in fit_stats]):.2f} m")

    cats = sorted({lab for q in qs for lab in q["want"]} | {lab for q in qs for lab, _ in q["pred"]})
    cid = {c: i for i, c in enumerate(cats)}

    def build(sel):
        dt, gt = [], []
        for q in sel:
            f = frames[q["key"]]
            pb = []
            if f is not None:
                R, t = rigid(f[0], f[1])
                for lab, (x, y, z, w, d, h) in q["pred"]:
                    c = R @ np.array([x, y, z]) + t
                    pb.append([*c, w, d, h, f[0]])
            gb = [(lab, b) for lab, b in GT[q["key"]] if lab in q["want"]]
            dt.append({"labels_3d": torch.tensor([cid[lab] for lab, _ in q["pred"]] if pb else [], dtype=torch.long),
                       "bboxes_3d": boxes3d(pb), "scores_3d": torch.ones(len(pb))})
            gt.append({"gt_labels_3d": np.array([cid[lab] for lab, _ in gb], np.int64), "gt_bboxes_3d": boxes3d([b for _, b in gb])})
        return dt, gt

    import logging
    logging.getLogger("mmengine").setLevel(logging.ERROR)
    res = {}
    groups = [("all", qs)] + [(d, [q for q in qs if q["dataset"] == d]) for d in ("scannet", "scannetpp", "arkitscenes")]
    types = sorted({q["type"] for q in qs})
    groups += [(t, [q for q in qs if q["type"] == t]) for t in types]
    for name, sel in groups:
        if not sel:
            continue
        dt, gt = build(sel)
        # AP is defined for classes with GT: predictions of labels with no GT box in this group are left out of the
        # mAP (they still count as false positives in P@thr)
        has_gt = {int(l) for g in gt for l in g["gt_labels_3d"]}
        dt_map = []
        for d in dt:
            k = torch.tensor([int(l) in has_gt for l in d["labels_3d"]], dtype=torch.bool)
            dt_map.append({"labels_3d": d["labels_3d"][k], "bboxes_3d": d["bboxes_3d"][k] if len(k) else d["bboxes_3d"],
                           "scores_3d": d["scores_3d"][k]})
        r = indoor_eval(gt, dt_map, [0.25, 0.5], {i: c for c, i in cid.items()}, logger="silent",
                        box_mode_3d=Box3DMode.DEPTH) if any(len(g["gt_labels_3d"]) for g in gt) else {}
        out = {"questions": len(sel), "pred_boxes": sum(len(d["labels_3d"]) for d in dt),
               "gt_boxes": sum(len(g["gt_labels_3d"]) for g in gt)}
        for thr in (0.25, 0.5):
            p, rc = pr_at(dt, gt, thr)
            out[f"mAP@{thr}"] = r.get(f"mAP_{thr:.2f}", float("nan"))
            out[f"P@{thr}"], out[f"R@{thr}"] = p, rc
        res[name] = out
        print(f"{name:32s} q={out['questions']:5d} pred={out['pred_boxes']:5d} gt={out['gt_boxes']:5d}  "
              f"mAP25={out['mAP@0.25']:.4f} mAP50={out['mAP@0.5']:.4f}  "
              f"P25={out['P@0.25']:.3f} R25={out['R@0.25']:.3f} P50={out['P@0.5']:.3f} R50={out['R@0.5']:.3f}",
              flush=True)
    if args.json:
        json.dump(res, open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()
