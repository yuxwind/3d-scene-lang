"""Answer the debiased ScanNet++ questions directly from the scene language (no model), scored like predictions.

Uses the text the model sees (2-decimal 9-DoF boxes). Per task:
  counting: number of lines with the label; size: longest box extent;
  abs / rel distance: closest point between boxes (11^3 grid of points inside each box, min pairwise distance);
  rel direction: standing at A's centre facing B's centre (xy plane), direction of C's centre seen from A
    (for the easy questions also seen from B);
  room size: area of the smallest (rotated) rectangle covering every box point in the xy plane (a proxy: the
    text has no walls; found over the convex-hull edge directions, rotating calipers);
  appearance order / route planning: not computable from boxes (no time, no path) -> left out.

    python oracle.py
"""
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from sr_vggt.eval import score

D = os.path.dirname(os.path.abspath(__file__))
rows = [r for r in map(json.loads, open(f"{D}/vsibench_scannetpp_scenelang.jsonl")) if r["debiased"]]
G = np.stack(np.meshgrid(*[np.linspace(-.5, .5, 11)] * 3, indexing="ij"), -1).reshape(-1, 3)


def boxes(text):
    out = defaultdict(list)
    for o in json.loads(text):
        x, y, z, dx, dy, dz, yaw, roll, pitch = o["bbox_3d"]
        cz, sz, cx, sx, cy, sy = math.cos(yaw), math.sin(yaw), math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
        R = (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
             @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]))
        out[o["label"]].append((np.array([x, y, z]), np.array([dx, dy, dz]), R))
    return dict(out)  # a missing label raises KeyError


def pts(b):
    c, e, R = b
    return c + (G * e) @ R.T


def dist(insts_a, insts_b):
    tree = cKDTree(np.concatenate([pts(b) for b in insts_b]))
    return float(min(tree.query(pts(a))[0].min() for a in insts_a))


def min_rect_area(P):
    """Smallest-area rectangle covering the 2D points P: one of its sides lies along a convex-hull edge."""
    H = P[ConvexHull(P).vertices]
    best = math.inf
    for i in range(len(H)):
        e = H[(i + 1) % len(H)] - H[i]
        c, s = e / np.linalg.norm(e)
        Q = H @ np.array([[c, -s], [s, c]])  # hull in the frame of this edge
        w, h = Q.max(0) - Q.min(0)
        best = min(best, w * h)
    return best


def letter(r, choice):
    return next((o[0] for o in r["options"] if o.split(". ", 1)[1].strip().lower() == choice), "?")


def side(A, B, C):
    """Standing at A facing B: (cross > 0 means C is on the left, dot > 0 means in front, angle in degrees)."""
    f, v = (B - A)[:2], (C - A)[:2]
    cos = (f @ v) / (np.linalg.norm(f) * np.linalg.norm(v) + 1e-9)
    return f[0] * v[1] - f[1] * v[0], f @ v, math.degrees(math.acos(max(-1.0, min(1.0, cos))))


preds, from_b, miss = [], [], defaultdict(int)
for r in rows:
    bx, q, t = boxes(r["scene_language"]), r["question"], r["question_type"]
    if t in ("obj_appearance_order", "route_planning"):
        continue
    try:
        if t == "object_counting":
            p = str(len(bx.get(re.search(r"How many (.+?)\(s\)", q).group(1), [])))
        elif t == "object_size_estimation":
            p = str(round(100 * max(bx[re.search(r"\) of the (.+?), measured", q).group(1)][0][1])))
        elif t == "object_abs_distance":
            a, b = re.search(r"between the (.+?) and the (.+?) \(in meters\)", q).groups()
            p = f"{dist(bx[a], bx[b]):.2f}"
        elif t == "object_rel_distance":
            m = re.search(r"objects \((.+?)\) is the closest to the (.+?)\?", q)
            d = {n.strip(): dist(bx[n.strip()], bx[m.group(2)]) for n in m.group(1).split(",")}
            p = letter(r, min(d, key=d.get))
        elif t.startswith("object_rel_direction"):
            a, b, c = re.search(r"standing by the (.+?) and facing the (.+?), is the (.+?) (?:to|on) ", q).groups()
            A, B, C = bx[a][0][0], bx[b][0][0], bx[c][0][0]
            cross, dot, ang = side(A, B, C)
            if t.endswith("easy"):
                p = letter(r, "left" if cross > 0 else "right")
                cb = side(B, 2 * B - A, C)[0]
                from_b.append({**r, "prediction": letter(r, "left" if cb > 0 else "right")})
            elif t.endswith("medium"):
                p = letter(r, "back" if ang >= 135 else ("left" if cross > 0 else "right"))
            else:
                p = letter(r, ("front" if dot > 0 else "back") + "-" + ("left" if cross > 0 else "right"))
        elif t == "room_size_estimation":
            p = f"{min_rect_area(np.concatenate([pts(b)[:, :2] for v in bx.values() for b in v])):.1f}"
    except (AttributeError, KeyError):
        miss[t] += 1
        p = "none"
    preds.append({**r, "prediction": p})

fmt = lambda s: " ".join(f"{k.split('/')[-1].replace('_accuracy', '').replace('_mra', '')}={100 * v:.1f}"
                         for k, v in s.items())
print("could not compute:", dict(miss))
print("ORACLE", fmt(score(preds)))
per = defaultdict(list)
for r in preds + [{**x, "question_type": "easy, C seen from B"} for x in from_b]:
    if "direction" in r["question_type"] or "seen from" in r["question_type"]:
        per[r["question_type"]].append(r["prediction"] == r["ground_truth"])
print("rel-dir:", {k: f"{sum(v)}/{len(v)}" for k, v in per.items()})
json.dump(preds, open(f"{D}/oracle_predictions.json", "w"))
