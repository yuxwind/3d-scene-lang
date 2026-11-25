#!/usr/bin/env python3
"""Oracle box lists for ReVSI from the SAM 3 + VGGT-SLAM candidates, as VGGT-SLAM/exps/vsibench_objects/s3_oracle_rows.py does
for VSI-Bench: instead of a score threshold, keep the candidates that match a GT object.

  oracle     all candidates (any score) matched one-to-one to the scene's GT boxes of their label (Hungarian on centre
             distance, in the GT frame); a pair is kept if the 3D IoU is >= 0.1 or the centres are within
             max(0.5 m, 0.5 x the GT box's longest side)
  oraclecnt  as oracle without the position check: each GT object keeps the candidate it is assigned, if its label has one
The kept boxes are written unchanged (the list's estimated scale, SLAM frame, wall-aligned), as rows_s3q_roomdoor_full.jsonl.

Difference from VSI-Bench: ReVSI has no camera poses, so the SLAM frame cannot be fitted to GT cameras. Each scene's candidates
are registered to its GT boxes directly: a similarity (scale s, yaw, xy shift, z shift; z up in both) by RANSAC over
same-label pairs of box centres (candidates with score >= 0.5; 2 pairs per sample), refined by least squares on the inliers
(centre within max(0.5 m, 0.5 x GT longest side)). Scenes with < 3 inliers are 'unregistered': there, oracle keeps nothing and
oraclecnt keeps, per GT label, its highest-scoring candidates (as many as GT objects).
GT: ReVSI metadata/3d_annotation.json, objects visible in the 64 frames (obj_visibility.json visibility_64), upright box
(centre, horizontal sizes, height, yaw; obb.rotation columns are the box axes). Labels: candidate labels are ReVSI names (the
SAM 3 prompts were the questions' names); 'door' (the 'room door' prompt) matches any GT name containing 'door'.
numpy only.
    python3 oracle_lists.py --boxes DIR --prefix s3qbox_alt-door --revsi_dir R --rows rows_s3q_roomdoor_full.jsonl [--seed 0]
writes next to --rows: rows_s3q_rr_oracle_full.jsonl, rows_s3q_rr_oraclecnt_full.jsonl, oracle_lists_stats.json
"""
import argparse, collections, json, math, os
import numpy as np

KEY = lambda l: "door" if ("door" in l.lower() and not any(x in l.lower() for x in ("car door", "elevator door", "outdoor"))) else l.lower()


def gt_boxes(ann, vis):
    out = []
    for o in ann["objects"]:
        if o["id"] not in vis:
            continue
        R, e, c = np.array(o["obb"]["rotation"], float).reshape(3, 3), np.array(o["obb"]["extent"], float), np.array(o["obb"]["center"], float)
        up = int(np.argmax(np.abs(R[2]))); i, j = [k for k in range(3) if k != up]
        out.append(dict(label=o["name"], c=c, s=np.array([e[i], e[j], e[up]]), yaw=math.atan2(R[1, i], R[0, i])))
    return out


def corners2(c, s, yaw):
    cs, sn = math.cos(yaw), math.sin(yaw)
    return [(c[0] + cs * a * s[0] / 2 - sn * b * s[1] / 2, c[1] + sn * a * s[0] / 2 + cs * b * s[1] / 2) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def clip(P, Q):  # Sutherland-Hodgman: convex polygon P clipped by convex polygon Q (both counter-clockwise)
    def inside(p, a, b): return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0
    def inter(p, q, a, b):
        x1, y1, x2, y2, x3, y3, x4, y4 = *p, *q, *a, *b; d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(d) < 1e-12: return q
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d; return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    out = P
    for k in range(len(Q)):
        a, b = Q[k], Q[(k + 1) % len(Q)]; inp, out = out, []
        for m in range(len(inp)):
            p, q = inp[m - 1], inp[m]
            if inside(q, a, b):
                if not inside(p, a, b): out.append(inter(p, q, a, b))
                out.append(q)
            elif inside(p, a, b): out.append(inter(p, q, a, b))
        if not out: return []
    return out


def area(P): return 0.5 * abs(sum(P[k][0] * P[(k + 1) % len(P)][1] - P[(k + 1) % len(P)][0] * P[k][1] for k in range(len(P)))) if len(P) >= 3 else 0.0


def iou3d(c1, s1, y1, c2, s2, y2):
    s1, s2 = np.maximum(s1, 1e-3), np.maximum(s2, 1e-3)
    a = area(clip(corners2(c1, s1, y1), corners2(c2, s2, y2)))
    h = max(0.0, min(c1[2] + s1[2] / 2, c2[2] + s2[2] / 2) - max(c1[2] - s1[2] / 2, c2[2] - s2[2] / 2))
    inter = a * h; return float(inter / (np.prod(s1) + np.prod(s2) - inter + 1e-9))


def hungarian(C):
    """min-cost assignment for a (n x m) matrix; returns [(row, col)] for min(n, m) pairs (O(n^2 m), small n)."""
    C = np.asarray(C, float); T = C.shape[0] > C.shape[1]
    if T: C = C.T
    n, m = C.shape; u, v, p, way = np.zeros(n + 1), np.zeros(m + 1), np.zeros(m + 1, int), np.zeros(m + 1, int)
    for i in range(1, n + 1):
        p[0] = i; j0 = 0; minv = np.full(m + 1, np.inf); used = np.zeros(m + 1, bool)
        while True:
            used[j0] = True; i0 = p[j0]; d = np.inf; j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]: minv[j], way[j] = cur, j0
                    if minv[j] < d: d, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]: u[p[j]] += d; v[j] -= d
                else: minv[j] -= d
            j0 = j1
            if p[j0] == 0: break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0: break
    pairs = [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j]]
    return [(b, a) for a, b in pairs] if T else pairs


def apply(T, c):
    s, th, t = T; R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return np.r_[s * R @ c[:2] + t[:2], s * c[2] + t[2]]


def fit(P, G):
    """similarity in xy (scale, yaw, shift) + z = s*z + tz, least squares on paired centres."""
    p, g = P[:, :2] - P[:, :2].mean(0), G[:, :2] - G[:, :2].mean(0)
    th = math.atan2((p[:, 0] * g[:, 1] - p[:, 1] * g[:, 0]).sum(), (p * g).sum())
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]); s = float((g * (p @ R.T)).sum() / max((p ** 2).sum(), 1e-9))
    t = np.r_[G[:, :2].mean(0) - s * R @ P[:, :2].mean(0), np.median(G[:, 2] - s * P[:, 2])]
    return s, th, t


def register(pred, gt, rng, iters=1500):
    cand = [(a, b) for a, pr in enumerate(pred) if pr["score"] >= 0.5 for b, g in enumerate(gt) if KEY(pr["label"]) == KEY(g["label"])]
    if len(cand) < 2:
        return None, 0
    use = [a for a, pr in enumerate(pred) if pr["score"] >= 0.5]
    Pc = np.array([pred[a]["c"] for a in use]); Gc = np.array([g["c"] for g in gt])
    L = np.array([[KEY(pred[a]["label"]) == KEY(g["label"]) for g in gt] for a in use])
    tol = np.array([max(0.5, 0.5 * max(g["s"])) for g in gt])
    def inliers(T):
        s, th, t = T; R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        Q = np.c_[s * Pc[:, :2] @ R.T + t[:2], s * Pc[:, 2] + t[2]]
        D = np.linalg.norm(Q[:, None, :] - Gc[None], axis=-1); D[~L] = np.inf; D[D > tol[None]] = np.inf
        best = D.min(0); arg = D.argmin(0); ok = np.isfinite(best)
        return [(use[arg[b]], b) for b in np.flatnonzero(ok)], float(best[ok].sum())
    bestT, bestI, bestE = None, [], math.inf
    for _ in range(iters):
        (a1, b1), (a2, b2) = (cand[k] for k in rng.choice(len(cand), 2, replace=False))
        if a1 == a2 or b1 == b2: continue
        dp, dg = pred[a2]["c"][:2] - pred[a1]["c"][:2], gt[b2]["c"][:2] - gt[b1]["c"][:2]
        if np.linalg.norm(dg) < 0.5 or np.linalg.norm(dp) < 1e-3: continue
        s = np.linalg.norm(dg) / np.linalg.norm(dp)
        if not 0.33 <= s <= 3.0: continue
        th = math.atan2(dg[1], dg[0]) - math.atan2(dp[1], dp[0]); R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        t = np.r_[gt[b1]["c"][:2] - s * R @ pred[a1]["c"][:2], np.median([gt[b]["c"][2] - s * pred[a]["c"][2] for a, b in ((a1, b1), (a2, b2))])]
        I, E = inliers((s, th, t))
        if len(I) > len(bestI) or (len(I) == len(bestI) and E < bestE): bestT, bestI, bestE = (s, th, t), I, E
    if len(bestI) < 3:
        return None, len(bestI)
    T = bestT
    for _ in range(3):
        I, _ = inliers(T)
        if len(I) < 3: break
        T = fit(np.array([pred[a]["c"] for a, b in I]), np.array([gt[b]["c"] for a, b in I]))
    return T, len(inliers(T)[0])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--boxes", required=True); ap.add_argument("--prefix", default="s3qbox_alt-door")
    ap.add_argument("--revsi_dir", required=True); ap.add_argument("--rows", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    ann = {x["scene_id"]: x for x in json.load(open(os.path.join(a.revsi_dir, "metadata/3d_annotation.json")))}
    vis = {s: {o["object_id"] for o in v if o["visibility_64"]} for s, v in json.load(open(os.path.join(a.revsi_dir, "metadata/obj_visibility.json"))).items()}
    rows = [json.loads(l) for l in open(a.rows)]
    keep = {"oracle": {}, "oraclecnt": {}}; stats = {}; st = collections.Counter()
    for s in sorted({r["scene_name"] for r in rows}):
        f = os.path.join(a.boxes, f"{a.prefix}_{s}.json")
        if not os.path.exists(f):
            st["scenes_no_box_file"] += 1; continue
        J = json.load(open(f)); P = J["pred"]
        pred = [dict(label=o["label"], c=np.array(o["manh"][:3], float), s=np.array(o["manh"][3:6], float), score=o["score"]) for o in P]
        gt = gt_boxes(ann[s], vis.get(s, set()))
        T, nin = register(pred, gt, rng)
        stats[s] = dict(gt=len(gt), cand=len(pred), registered=T is not None, inliers=nin,
                        scale=None if T is None else round(T[0], 3), s_cls=J.get("s_cls"))
        st["scenes"] += 1; st["registered"] += T is not None; st["gt"] += len(gt); st["cand"] += len(pred)
        ko, kc = [], []
        for lab in sorted({KEY(g["label"]) for g in gt}):
            gi = [b for b, g in enumerate(gt) if KEY(g["label"]) == lab]; pi = [k for k, pr in enumerate(pred) if KEY(pr["label"]) == lab]
            if not pi: st["gt_no_candidate_label"] += len(gi); continue
            if T is None:
                kc += sorted(pi, key=lambda k: -pred[k]["score"])[:len(gi)]; continue
            Q = [apply(T, pred[k]["c"]) for k in pi]
            C = [[float(np.linalg.norm(q - gt[b]["c"])) for b in gi] for q in Q]
            for x, y in hungarian(C):
                g = gt[gi[y]]; k = pi[x]; kc.append(k)
                ok = iou3d(Q[x], pred[k]["s"] * T[0], T[1], g["c"], g["s"], g["yaw"]) >= 0.1 or C[x][y] <= max(0.5, 0.5 * max(g["s"]))
                if ok: ko.append(k)
        st["kept_oracle"] += len(ko); st["kept_oraclecnt"] += len(kc); st["gt_matched_oracle"] += len(ko)
        for name, K in (("oracle", ko), ("oraclecnt", kc)):
            keep[name][s] = sorted([{"label": P[k]["label"], "bbox_3d": [round(float(v), 2) for v in P[k]["manh"][:6]], "score": round(P[k]["score"], 3)} for k in K],
                                   key=lambda o: -o["score"])
    d = os.path.dirname(os.path.abspath(a.rows))
    for name, K in keep.items():
        n = collections.Counter()
        with open(os.path.join(d, f"rows_s3q_rr_{name}_full.jsonl"), "w") as fo:
            for r in rows:
                objs = K.get(r["scene_name"], [])
                r2 = {**r, "scene_language": "[\n" + ",\n".join(json.dumps(o) for o in objs) + "\n]", "use_scene": True}
                fo.write(json.dumps(r2) + "\n"); n["boxes"] += len(objs); n["empty"] += not objs
        print(f"wrote rows_s3q_rr_{name}_full.jsonl: {len(rows)} rows, boxes per row {n['boxes'] / len(rows):.1f}, empty lists {n['empty']}")
    sc = [v["scale"] for v in stats.values() if v["scale"]]
    print(dict(st), f"| GT kept by oracle {st['kept_oracle'] / max(st['gt'], 1):.3f}, oraclecnt {st['kept_oraclecnt'] / max(st['gt'], 1):.3f}"
          f" | registration scale median {np.median(sc):.2f} (p10 {np.percentile(sc, 10):.2f}, p90 {np.percentile(sc, 90):.2f})")
    json.dump({"summary": dict(st), "scenes": stats}, open(os.path.join(d, "oracle_lists_stats.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
