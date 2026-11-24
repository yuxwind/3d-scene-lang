"""The code solver of data/scenelang_vsibench_scannetpp/oracle.py (written by Claude) on all 5,130 VSI-Bench questions, for
any box list: answers are computed from the list text, no model, no video.
  counting: boxes with the label; size: longest side of the first (best-scoring) box of the label; abs / rel distance:
  closest point between boxes (11^3 grid inside each box, min over all instances); rel direction: centres of the first
  box of each label (standing at A facing B, C left/right/back...); room size: smallest rotated rectangle covering every
  box point in the xy plane; appearance order / route planning: not computable -> scored 0 (as oracle.py).
A named object with no box -> no answer (scored 0). Boxes: 9 values (centre, size, yaw, roll, pitch; GT) or 6 (centre,
size, wall-aligned; SAM 3 + VGGT-SLAM).
    /scr/xinyu/envs/vggtdet/bin/python oracle_code_lists.py name=rows.jsonl [...]        (scipy) -> oracle_code_<name>.json next to rows
    /scr/xinyu/envs/sr_vggt/bin/python oracle_code_lists.py --score name=rows.jsonl [...] (scores with sr_vggt.eval.score)"""
import json, math, re, sys, os
from collections import defaultdict
import numpy as np
SCORE = sys.argv[1:2] == ['--score']
if SCORE: sys.path.insert(0, '/scr/xinyu/projects/3d-scene-lang'); from sr_vggt.eval import score
else: from scipy.spatial import ConvexHull, cKDTree
G = np.stack(np.meshgrid(*[np.linspace(-.5, .5, 11)] * 3, indexing="ij"), -1).reshape(-1, 3)
def boxes(text):
    out = defaultdict(list)
    for o in json.loads(text or '[]'):
        b = list(o["bbox_3d"]) + [0.0] * (9 - len(o["bbox_3d"])); x, y, z, dx, dy, dz, yaw, roll, pitch = b
        cz, sz, cx, sx, cy, sy = math.cos(yaw), math.sin(yaw), math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
        R = (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]))
        out[o["label"]].append((np.array([x, y, z]), np.array([dx, dy, dz]), R))
    return dict(out)
pts = lambda b: b[0] + (G * b[1]) @ b[2].T
def dist(A, B):
    tree = cKDTree(np.concatenate([pts(b) for b in B])); return float(min(tree.query(pts(a))[0].min() for a in A))
def min_rect_area(P):
    H = P[ConvexHull(P).vertices]; best = math.inf
    for i in range(len(H)):
        e = H[(i + 1) % len(H)] - H[i]; c, s = e / np.linalg.norm(e); Q = H @ np.array([[c, -s], [s, c]]); w, h = Q.max(0) - Q.min(0); best = min(best, w * h)
    return best
letter = lambda r, ch: next((o[0] for o in r["options"] if o.split(". ", 1)[1].strip().lower() == ch), "?")
def side(A, B, C):
    f, v = (B - A)[:2], (C - A)[:2]; cos = (f @ v) / (np.linalg.norm(f) * np.linalg.norm(v) + 1e-9)
    return f[0] * v[1] - f[1] * v[0], f @ v, math.degrees(math.acos(max(-1.0, min(1.0, cos))))
def answer(r):
    bx, q, t = boxes(r["scene_language"]), r["question"], r["question_type"]
    if t == "object_counting": return str(len(bx.get(re.search(r"How many (.+?)\(s\)", q).group(1), [])))
    if t == "object_size_estimation": return str(round(100 * max(bx[re.search(r"\) of the (.+?), measured", q).group(1)][0][1])))
    if t == "object_abs_distance": a, b = re.search(r"between the (.+?) and the (.+?) \(in meters\)", q).groups(); return f"{dist(bx[a], bx[b]):.2f}"
    if t == "object_rel_distance":
        m = re.search(r"objects \((.+?)\) is the closest to the (.+?)\?", q); d = {n.strip(): dist(bx[n.strip()], bx[m.group(2)]) for n in m.group(1).split(",") if n.strip() in bx}
        if not d: raise KeyError
        return letter(r, min(d, key=d.get))
    if t.startswith("object_rel_direction"):
        a, b, c = re.search(r"standing by the (.+?) and facing the (.+?), is the (.+?) (?:to|on) ", q).groups(); A, B, C = bx[a][0][0], bx[b][0][0], bx[c][0][0]
        cross, dot, ang = side(A, B, C)
        if t.endswith("easy"): return letter(r, "left" if cross > 0 else "right")
        if t.endswith("medium"): return letter(r, "back" if ang >= 135 else ("left" if cross > 0 else "right"))
        return letter(r, ("front" if dot > 0 else "back") + "-" + ("left" if cross > 0 else "right"))
    if t == "room_size_estimation": return f"{min_rect_area(np.concatenate([pts(b)[:, :2] for v in bx.values() for b in v])):.1f}"
    raise KeyError                                                    # appearance order, route planning
T = ['obj_appearance_order', 'object_abs_distance', 'object_counting', 'object_rel_direction', 'object_rel_distance', 'object_size_estimation', 'room_size_estimation', 'route_planning']
if SCORE: print(f"{'list':12s} {'full':>5s} {'deb.':>5s} {'6 tasks':>7s} " + ' '.join(f'{t.replace("object_", "").replace("obj_", "")[:9]:>9s}' for t in T) + '  no answer (named object without a box)')
for arg in sys.argv[1 + SCORE:]:
    name, f = arg.split('=', 1); preds = []; miss = defaultdict(int); out = os.path.join(os.path.dirname(os.path.abspath(f)), f'oracle_code_{name}.json')
    if not SCORE:
        for r in map(json.loads, open(f)):
            try: p = answer(r)
            except (AttributeError, KeyError, ValueError, IndexError): p = "none"
            preds.append({**r, "prediction": p})
        json.dump(preds, open(out, 'w')); print('wrote', out); continue
    preds = json.load(open(out))
    for p in preds:
        if p['prediction'] == 'none' and p['question_type'] not in ('obj_appearance_order', 'route_planning'): miss[p['question_type'].replace('object_', '')] += 1
    s, sd = score(preds), score([p for p in preds if p['debiased']])
    g = lambda S, t: next(v for k, v in S.items() if k.split('/')[-1].startswith(t))
    six = np.mean([g(s, t) for t in T if t not in ('obj_appearance_order', 'route_planning')])
    print(f"{name:12s} {100 * g(s, 'overall'):5.1f} {100 * g(sd, 'overall'):5.1f} {100 * six:7.1f} " + ' '.join(f'{100 * g(s, t):9.1f}' for t in T) + '  ' + str(dict(miss)))
