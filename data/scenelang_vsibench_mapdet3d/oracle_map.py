"""Class-aware mAP of Map-Det3D with labels borrowed from the GT (an oracle labeller, i.e. an upper bound).

GT boxes and detections are matched one-to-one: pairs with 3D IoU >= the labelling threshold are taken in decreasing
IoU, each GT box and each detection at most once, and the detection gets that GT box's label. Unmatched detections get
no label and are dropped. mAP@thr is then the mean over GT categories of the per-class
AP (mmdet3d indoor_eval style: greedy by score, a detection is a TP if its highest-IoU GT box of that class is
unmatched and IoU >= thr; VOC all-point area). The labelling threshold is the evaluation threshold unless
--label-thr is given.

The same oracle is applied to VGGT-Det (seed 0, its own labels discarded) for reference, next to VGGT-Det's own mAP.

    python oracle_map.py <dir with *_meta_info_val_mapdet3d.json> [--label-thr 0.1]
        -> <dir>/oracle_map.json, or <dir>/oracle_map_label<thr>.json with --label-thr
    python oracle_map.py <dir> --write-meta 0.25    -> <dir>/<dataset>_meta_info_val_mapdet3d_oracle0.25.json
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

from build_meta import META, VGGTDET_META, corners_from_vsi, iou

THRS = (0.25, 0.5)


def load(meta, scenes):
    """scene -> (labels, scores, (N, 8, 3) corners)."""
    out = {}
    for s in scenes:
        bs = [(label, b.get("score", 1.0), corners_from_vsi(b)) for label, v in meta[s]["object_bbox"].items() for b in v]
        out[s] = ([b[0] for b in bs], np.array([b[1] for b in bs]), np.stack([b[2] for b in bs]) if bs else np.zeros((0, 8, 3)))
    return out


def oracle_pairs(gc, pc, thr):
    """One-to-one: (GT, detection) pairs with IoU >= thr are taken in decreasing IoU, each GT box and each detection
    at most once. Returns [(detection index, GT index, IoU)] sorted by detection index."""
    I = iou(gc, pc)
    pairs, used_g, used_p = [], set(), set()
    for g, p in zip(*np.unravel_index(np.argsort(-I, axis=None), I.shape)):
        if I[g, p] < thr:
            break
        if g not in used_g and p not in used_p:
            used_g.add(g); used_p.add(p)
            pairs.append((int(p), int(g), float(I[g, p])))
    return sorted(pairs)


def oracle_label(pred, gt, thr):
    """Each matched detection (oracle_pairs) gets its GT box's label; unmatched ones are dropped."""
    out = {}
    for s, (gl, _, gc) in gt.items():
        _, ps, pc = pred[s]
        pairs = [(p, gl[g]) for p, g, _ in oracle_pairs(gc, pc, thr)]
        out[s] = ([l for _, l in pairs], ps[[p for p, _ in pairs]] if pairs else np.zeros(0),
                  pc[[p for p, _ in pairs]] if pairs else np.zeros((0, 8, 3)))
    return out


def ap_class(pred, gt, cls, thr):
    dets, npos, used = [], 0, {}
    for s, (gl, _, gc) in gt.items():
        g = gc[[i for i, l in enumerate(gl) if l == cls]]
        pl, ps, pc = pred[s]
        k = [i for i, l in enumerate(pl) if l == cls]
        npos += len(g)
        used[s] = np.zeros(len(g), bool)
        I = iou(pc[k], g) if k and len(g) else np.zeros((len(k), len(g)))
        dets += [(ps[i], s, I[j]) for j, i in enumerate(k)]
    if not dets:
        return 0.0
    dets.sort(key=lambda d: -d[0])
    tp = np.zeros(len(dets))
    for n, (_, s, row) in enumerate(dets):
        if len(row) and row.max() >= thr and not used[s][row.argmax()]:
            used[s][row.argmax()] = True
            tp[n] = 1
    rec = np.cumsum(tp) / npos
    prec = np.cumsum(tp) / np.arange(1, len(dets) + 1)
    mrec, mpre = np.r_[0, rec, 1], np.r_[0, prec, 0]
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[i + 1] - mrec[i]) * mpre[i + 1]).sum())


def mean_ap(pred, gt, thr):
    cats = sorted({l for gl, _, _ in gt.values() for l in gl})
    per = {c: round(ap_class(pred, gt, c, thr), 4) for c in cats}
    return float(np.mean(list(per.values()))), per


def write_meta(d, ds, thr):
    """<d>/<ds>_meta_info_val_mapdet3d_oracle<thr>.json: the meta_info format of build_meta.py, the matched Map-Det3D
    boxes under their GT labels (each box dict as in the mapdet3d file, plus "gt_iou"), sorted by score per label;
    object_counts = boxes per label. Unmatched boxes are dropped."""
    md = json.load(open(f"{d}/{ds}_meta_info_val_mapdet3d.json"))
    gtm = json.load(open(META.format(ds)))
    gt, pred = load(gtm, sorted(md)), load(md, sorted(md))
    out = {}
    for s in sorted(md):
        dets = md[s]["object_bbox"]["object"] if md[s]["object_counts"] else []  # same order as load()
        bbox = defaultdict(list)
        for p, g, v in oracle_pairs(gt[s][2], pred[s][2], thr):
            bbox[gt[s][0][g]].append({**dets[p], "gt_iou": round(v, 4)})
        bbox = {k: sorted(v, key=lambda b: -b["score"]) for k, v in sorted(bbox.items())}
        out[s] = {**md[s], "object_counts": {k: len(v) for k, v in bbox.items()}, "object_bbox": bbox}
    path = f"{d}/{ds}_meta_info_val_mapdet3d_oracle{thr}.json"
    json.dump(out, open(path, "w"))
    n = sum(len(v) for o in out.values() for v in o["object_bbox"].values())
    print(f"{path}: {len(out)} scenes, {n} labelled boxes", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--label-thr", type=float)
    ap.add_argument("--write-meta", type=float, metavar="THR",
                    help="only write the oracle-labelled Map-Det3D meta_info files at this labelling threshold")
    a = ap.parse_args()
    if a.write_meta is not None:
        for ds in ["scannet", "arkitscenes", "scannetpp"]:
            write_meta(a.dir, ds, a.write_meta)
        return
    res = defaultdict(dict)
    for ds in ["scannet", "arkitscenes", "scannetpp"]:
        md = json.load(open(f"{a.dir}/{ds}_meta_info_val_mapdet3d.json"))
        gtm = json.load(open(META.format(ds)))
        runs = {"mapdet3d": (md, sorted(md))}
        vf = VGGTDET_META.format(ds, 0)
        if os.path.exists(vf):
            vg = json.load(open(vf))
            both = [s for s in md if s in vg]
            runs["mapdet3d (vggtdet scenes)"] = (md, both)
            runs["vggtdet"] = (vg, both)
        for name, (meta, scenes) in runs.items():
            gt, pred = load(gtm, scenes), load(meta, scenes)
            r = {"scenes": len(scenes)}
            for t in THRS:
                m, per = mean_ap(oracle_label(pred, gt, t if a.label_thr is None else a.label_thr), gt, t)
                r[f"oracle mAP@{t}"], r[f"oracle per-class AP@{t}"] = round(m, 4), per
                if name == "vggtdet":  # its own labels
                    r[f"own-label mAP@{t}"] = round(mean_ap(pred, gt, t)[0], 4)
            res[ds][name] = r
            print(f"{ds:12s} {name:26s} scenes {len(scenes):3d}  "
                  + "  ".join(f"{k} {v:.3f}" for k, v in r.items() if "mAP" in k), flush=True)
    json.dump(res, open(f"{a.dir}/oracle_map.json" if a.label_thr is None else f"{a.dir}/oracle_map_label{a.label_thr}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
