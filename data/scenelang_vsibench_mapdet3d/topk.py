"""Box-count stats and top-K-by-score truncation of Map-Det3D vs VGGT-Det on VSI-Bench scenes.

Per (method, K): boxes/scene, class-agnostic AP (false positives counted), 1:1 recall, and oracle-label mAP
(label threshold 0.1, and label threshold = eval threshold). K = "vggt" keeps, in each scene, as many Map-Det3D
boxes as VGGT-Det predicted there.
"""
import json
import sys

import numpy as np

D = "/lustre/fswork/projects/rech/bef/uxt97ti/projects/3d-scene-lang/data/scenelang_vsibench_mapdet3d"
sys.path.insert(0, D)
from build_meta import META, VGGTDET_META  # noqa: E402
from oracle_map import ap_class, load, mean_ap, oracle_label, oracle_pairs  # noqa: E402

KS = ["vggt", 5, 10, 20, 30, 50, 100, "all"]


def topk(pred, k):
    return {s: (l[:k[s]], sc[:k[s]], c[:k[s]]) for s, (l, sc, c) in pred.items()}


def metrics(pred, gt):
    n = np.array([len(p[0]) for p in pred.values()])
    r = {"boxes/scene": round(float(n.mean()), 1), "median": float(np.median(n))}
    agn_p = {s: (["obj"] * len(v[0]), v[1], v[2]) for s, v in pred.items()}
    agn_g = {s: (["obj"] * len(v[0]), v[1], v[2]) for s, v in gt.items()}
    ngt = sum(len(v[0]) for v in gt.values())
    for t in (0.25, 0.5):
        r[f"agnAP@{t}"] = round(ap_class(agn_p, agn_g, "obj", t), 4)
        r[f"R@{t}"] = round(sum(len(oracle_pairs(gt[s][2], pred[s][2], t)) for s in gt) / ngt, 4)
        r[f"oracle0.1 mAP@{t}"] = round(mean_ap(oracle_label(pred, gt, 0.1), gt, t)[0], 4)
        r[f"oracle mAP@{t}"] = round(mean_ap(oracle_label(pred, gt, t), gt, t)[0], 4)
    return r


res = {}
for ds in ["scannet", "arkitscenes", "scannetpp"]:
    md = json.load(open(f"{D}/{ds}_meta_info_val_mapdet3d.json"))
    gtm = json.load(open(META.format(ds)))
    try:
        vg = json.load(open(VGGTDET_META.format(ds, 0)))
        scenes = [s for s in sorted(md) if s in vg]
    except FileNotFoundError:
        vg, scenes = None, sorted(md)
    gt, pm = load(gtm, scenes), load(md, scenes)
    rows = {}
    if vg:
        pv = load(vg, scenes)
        rows["vggtdet (all)"] = metrics(pv, gt)
        rows["vggtdet (all)"]["own-label mAP@0.25"] = round(mean_ap(pv, gt, 0.25)[0], 4)
        rows["vggtdet (all)"]["own-label mAP@0.5"] = round(mean_ap(pv, gt, 0.5)[0], 4)
    for k in KS:
        if k == "vggt" and not vg:
            continue
        kk = {s: len(pv[s][0]) if k == "vggt" else (len(pm[s][0]) if k == "all" else k) for s in scenes}
        rows[f"mapdet3d top-{k}"] = metrics(topk(pm, kk), gt)
    res[ds] = {"scenes": len(scenes), "gt": sum(len(v[0]) for v in gt.values()), "rows": rows}
    print(f"== {ds}: {len(scenes)} scenes, {res[ds]['gt']} GT", flush=True)
    for name, r in rows.items():
        print(f"{name:22s} " + "  ".join(f"{a} {b}" for a, b in r.items()), flush=True)
json.dump(res, open(f"{D}/topk.json", "w"), indent=1)
