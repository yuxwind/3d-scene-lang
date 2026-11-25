#!/usr/bin/env python3
"""Code solver on ReVSI with the ground-truth scene: answers computed from the scene text by fixed rules, no model, no video.

Port of the Claude-written VSI-Bench solver (data/scenelang_vsibench_slam/oracle_code_lists.py, from
data/scenelang_vsibench_scannetpp/oracle.py) to ReVSI's question wording and subtypes. Input: the batches of the Claude
no-code ReVSI GT run (claude_nocode_s3/revsi_gt/batches/*.json: {"batch", "questions": [{"id", "text"}]}, each text holding
the <scene> list and the question), so both see exactly the same boxes. Output: RUN_DIR/out/<batch>.json
{"answers": {id: answer}} as the Claude run, scored by scripts/analysis/revsi_score_answers.py RUN_DIR --gt <parquet>.
Rules (as oracle_code_lists.py):
  counting (single / multiple): boxes with the label(s); size: longest side of the first box of the label, in cm;
  abs distance: closest points between the boxes (11^3 grid inside each box, min over all instances); rel distance
  (closest / farthest): the same distance per option; rel direction (forward / backward = facing B / facing away from B;
  easy = left / right / back with back at >= 135 deg, hard = front|back - left|right): centres of the first box of each
  label; room size (single / multiple): smallest rotated rectangle covering every box point in the xy plane;
  route planning: not computable -> no answer (scored 0, as the VSI-Bench solver).
A named object with no box, or a question the rules do not parse -> no answer (scored 0).
Boxes: [x, y, z, dx, dy, dz, yaw, roll, pitch] (rotation Rz(yaw) Rx(roll) Ry(pitch), as oracle_code_lists.py) or 6 values.
numpy only.
    python3 code_solver.py BATCH_DIR RUN_DIR        prints answered / no answer per question kind
    python3 code_solver.py --rows ROWS.jsonl RUN_DIR  any ReVSI box list as eval rows (scene_language, question, options; e.g.
                                                     data/scenelang_revsi_slam/rows_s3q_rr_{oracle,oraclecnt}_full.jsonl)
"""
import glob, json, math, os, re, sys
from collections import Counter, defaultdict
import numpy as np

G = np.stack(np.meshgrid(*[np.linspace(-.5, .5, 11)] * 3, indexing="ij"), -1).reshape(-1, 3)


def boxes(scene_text):
    out = defaultdict(list)
    for line in scene_text.strip().splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        o = json.loads(line)
        x, y, z, dx, dy, dz, yaw, roll, pitch = list(o["bbox_3d"]) + [0.0] * (9 - len(o["bbox_3d"]))
        cz, sz, cx, sx, cy, sy = math.cos(yaw), math.sin(yaw), math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
        R = (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
             @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]))
        out[o["label"]].append((np.array([x, y, z]), np.array([dx, dy, dz]), R))
    return dict(out)


pts = lambda b: b[0] + (G * b[1]) @ b[2].T


def dist(A, B):
    P = np.concatenate([pts(b) for b in B])
    return float(min(np.sqrt(((pts(a)[:, None, :] - P[None]) ** 2).sum(-1)).min() for a in A))


def hull(P):  # monotone chain, 2D
    P = np.unique(np.round(P, 4), axis=0)
    if len(P) < 3:
        return P
    P = P[np.lexsort((P[:, 1], P[:, 0]))]
    cr = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in P[::-1]:
        while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])


def min_rect_area(P):
    H = hull(P); best = math.inf
    for i in range(len(H)):
        e = H[(i + 1) % len(H)] - H[i]
        if not np.linalg.norm(e):
            continue
        c, s = e / np.linalg.norm(e); Q = H @ np.array([[c, -s], [s, c]]); w, h = Q.max(0) - Q.min(0); best = min(best, w * h)
    return best


def side(A, B, C):
    f, v = (B - A)[:2], (C - A)[:2]
    cos = (f @ v) / (np.linalg.norm(f) * np.linalg.norm(v) + 1e-9)
    return f[0] * v[1] - f[1] * v[0], f @ v, math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def parse(text):
    scene = text.split("<scene>", 1)[1].split("</scene>", 1)[0]
    rest = text.split("</scene>", 1)[1].strip()
    q = rest.split("\nOptions:", 1)[0].split("\nAnswer with", 1)[0].strip()
    opts = {}
    if "\nOptions:" in rest:
        for l in rest.split("\nOptions:", 1)[1].splitlines():
            m = re.match(r"([A-D])\. (.+)$", l.strip())
            if m: opts[m.group(2).strip().lower()] = m.group(1)
    return boxes(scene), q, opts


def answer(bx, q, opts):
    """-> (kind, answer string); KeyError / AttributeError / ValueError when not computable."""
    m = re.search(r"How many (.+?)\s?\(s\)(?: and (.+?)\s?\(s\))? are in the scene", q)
    if m:
        return "counting", str(sum(len(bx.get(n, [])) for n in m.groups() if n))
    m = re.search(r"\) of the (.+?), measured in centimeters", q)
    if m:
        return "size", str(round(100 * max(bx[m.group(1)][0][1])))
    m = re.search(r"distance between the (.+?) and the (.+?) \(in meters\)", q)
    if m:
        return "abs_distance", f"{dist(bx[m.group(1)], bx[m.group(2)]):.2f}"
    m = re.search(r"which of these objects \((.+?)\) is the (closest to|farthest from) the (.+?)\?", q)
    if m:
        d = {n.strip(): dist(bx[n.strip()], bx[m.group(3)]) for n in m.group(1).split(",") if n.strip() in bx}
        if not d: raise KeyError
        pick = (min if m.group(2) == "closest to" else max)(d, key=d.get)
        return "rel_distance", opts[pick.lower()]
    m = re.search(r"standing by the (.+?) and facing (in the opposite direction of )?the (.+?), is the (.+?) to my ", q)
    if m:
        A, B, C = (bx[n][0][0] for n in (m.group(1), m.group(3), m.group(4)))
        if m.group(2): B = 2 * A - B                     # facing away from B
        cross, dot, ang = side(A, B, C)
        if "front-left" in q or "front - left" in q:
            return "rel_direction", opts[("front" if dot > 0 else "back") + "-" + ("left" if cross > 0 else "right")]
        return "rel_direction", opts["back" if ang >= 135 else ("left" if cross > 0 else "right")]
    if "size of the main room" in q or "size of the scene" in q:
        return "room_size", f"{min_rect_area(np.concatenate([pts(b)[:, :2] for v in bx.values() for b in v])):.1f}"
    if q.startswith("You are a robot"):
        raise ValueError("route")
    raise ValueError("unparsed")


def items():
    """-> [(out name, [(id, boxes, question, options)])] from Claude batches or from rows."""
    if sys.argv[1] == "--rows":
        qs = []
        for l in open(sys.argv[2]):
            r = json.loads(l)
            opts = {o.split(". ", 1)[1].strip().lower(): o[0] for o in (r.get("options") or [])}
            qs.append((r["id"], boxes(r["scene_language"]), r["question"], opts))
        return [("answers.json", qs)]
    out = []
    for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
        out.append((os.path.basename(f), [(it["id"], *parse(it["text"])) for it in json.load(open(f))["questions"]]))
    return out


def main():
    run = sys.argv[-1]
    os.makedirs(os.path.join(run, "out"), exist_ok=True)
    stat = Counter()
    for fname, qs in items():
        ans = {}
        for qid, bx, q, opts in qs:
            item = {"id": qid}
            try:
                kind, a = answer(bx, q, opts); ans[str(item["id"])] = a; stat[kind, "answered"] += 1
            except ValueError as e:
                ans[str(item["id"])] = "none"; stat[str(e), "no answer"] += 1
            except (KeyError, AttributeError, IndexError):
                ans[str(item["id"])] = "none"; stat["object not in scene / option", "no answer"] += 1
        json.dump({"batch": fname[:-5], "answers": ans, "solver": "data/revsi_gt_scene/code_solver.py", "input": sys.argv[1:-1]},
                  open(os.path.join(run, "out", fname), "w"))
    for (k, s), n in sorted(stat.items()):
        print(f"  {k:32} {s:10} {n}")
    print(f"wrote {run}/out ({sum(stat.values())} questions)")


if __name__ == "__main__":
    main()
