#!/usr/bin/env python3
"""Route planning vs the scene geometry: do the benchmark answers follow the boxes, and do the named objects exist?

For each route-planning question of a Claude no-code run (RUN_DIR/batches/*.json hold the full prompt: scene list
{"label", "bbox_3d": [x, y, z, ...]}, question, options; RUN_DIR/out/*.json hold the answers), it
  1. resolves every object the route names (start, facing, each "go forward until the X", destination) against the
     scene labels: exact (one box), ambiguous (several boxes of that label; the nearest one is used), fuzzy (only after
     dropping modifiers such as "back of the" / "by the door", or a label contained in the phrase), missing;
  2. walks the route on the floor plane (x, y): heading = start -> facing object; each blank is the turn from the current
     heading to the next waypoint (|angle| > 135 deg back, else left if counter-clockwise, else right); moving sets the
     heading. A blank with no later waypoint is "trailing" (not computable);
  3. compares the geometric answer with the benchmark answer (also with left/right swapped, which shows a mirrored frame)
     and with Claude's answer.
usage: route_check.py NAME=RUN_DIR,GT_FILE [...] [--json OUT]
  GT_FILE: VSI-Bench rows jsonl with {index|id, ground_truth} or the ReVSI parquet (pandas). ground_truth may be a letter.
prints per run: questions, name resolution, geometric answer vs benchmark answer, Claude accuracy split by both.
"""
import argparse, glob, json, math, os, re, collections

TURN = re.compile(r"turn (left|right|back)")
EX = 0


def load_gt(path):
    if path.endswith(".parquet"):
        import pandas as pd
        rows = pd.read_parquet(path).to_dict("records")
    else:
        rows = [json.loads(l) for l in open(path)]
    return {int(r["index"] if "index" in r else r["id"]): str(r["ground_truth"]) for r in rows}


def parse_prompt(t):
    scene = [json.loads(l) for l in t.split("<scene>\n", 1)[1].split("\n</scene>", 1)[0].splitlines() if l.strip()]
    rest = t.split("</scene>\n", 1)[1]
    q, _, o = rest.partition("\nOptions:\n")
    opts = [l.strip() for l in o.split("\nAnswer with")[0].splitlines() if l.strip()]
    return scene, q.strip(), opts


def opt_text(x):
    return " ".join(TURN.findall(x.lower())) if TURN.search(x.lower()) else x.lower().strip()  # "left right"


def to_text(ans, opts):
    a = str(ans).strip()
    m = re.match(r"^\(?([A-D])\)?(?:[.:)\s]|$)", a)
    if m and opts:
        i = "ABCD".index(m.group(1))
        if i < len(opts):
            return opt_text(re.sub(r"^[A-D]\.\s*", "", opts[i]))
    return opt_text(a)


def clean(name):
    n = name.lower().strip().rstrip(".").strip()
    n = re.sub(r"^(the )", "", n)
    return n


def resolve(phrase, labels):
    """-> (status, label or None)"""
    p = clean(phrase)
    if p in labels:
        return ("exact" if labels[p] == 1 else "ambiguous"), p
    q = re.sub(r"^(back|front|side|left side|right side|end|corner) of (the )?", "", p)
    q = re.split(r" (by|near|next to|on|in|under|above|beside|close to|at|with|between|facing) ", q)[0].strip()
    for c in (q, q.rstrip("s"), p.rstrip("s")):
        if c in labels:
            return "fuzzy", c
    inside = [l for l in labels if re.search(r"\b" + re.escape(l) + r"\b", q)]
    if inside:
        return "fuzzy", max(inside, key=len)
    outside = [l for l in labels if re.search(r"\b" + re.escape(q) + r"\b", l)]
    if outside:
        return "fuzzy", min(outside, key=len)
    return "missing", None


def parse_route(q):
    ql = re.sub(r"\s+", " ", q.lower())
    m = re.search(r"beginning at the (.+?)(?:,)?(?: and)? facing the (.+?)\. you want to navigate to the (.+?)\.", ql)
    if not m:
        return None
    body = ql.split("):", 1)[1] if "):" in ql else ql
    body = body.split("you have reached")[0]
    steps = []
    for it in re.split(r"\b\d+\.\s", body):
        it = it.strip().rstrip(".").strip()
        if not it:
            continue
        if "please fill in" in it:
            steps.append(("turn", None))
        else:
            g = re.search(r"go forward until (?:you reach )?(?:the )?(.+)", it)
            if g:
                steps.append(("go", g.group(1)))
    return m.group(1), m.group(2), m.group(3), steps


def walk(start, facing, steps, boxes):
    """boxes: label -> list of (x, y). Returns list of turns or None (trailing blank)."""
    def near(label, ref):
        return min(boxes[label], key=lambda c: (c[0] - ref[0]) ** 2 + (c[1] - ref[1]) ** 2)
    f0 = boxes[facing]
    s = min(boxes[start], key=lambda c: min((c[0] - f[0]) ** 2 + (c[1] - f[1]) ** 2 for f in f0))
    f = near(facing, s)
    pos, h = s, (f[0] - s[0], f[1] - s[1])
    turns = []
    for i, (kind, lab) in enumerate(steps):
        if kind == "go":
            t = near(lab, pos)
            h = (t[0] - pos[0], t[1] - pos[1]) if (t[0], t[1]) != (pos[0], pos[1]) else h
            pos = t
        else:
            nxt = next((l for k, l in steps[i + 1:] if k == "go"), None)
            if nxt is None:
                return None
            t = near(nxt, pos)
            d = (t[0] - pos[0], t[1] - pos[1])
            ang = math.degrees(math.atan2(h[0] * d[1] - h[1] * d[0], h[0] * d[0] + h[1] * d[1]))
            turns.append("back" if abs(ang) > 135 else ("left" if ang > 0 else "right"))
            h = d
    return turns


def check(name, run, gt_file):
    gt = load_gt(gt_file)
    ans = {}
    for f in glob.glob(os.path.join(run, "out", "*route*.json")):
        try:
            ans.update({int(k): v for k, v in json.load(open(f))["answers"].items()})
        except Exception:
            pass
    recs = []
    for f in sorted(glob.glob(os.path.join(run, "batches", "*route*.json"))):
        for qq in json.load(open(f))["questions"]:
            scene, q, opts = parse_prompt(qq["text"])
            labels = collections.Counter(o["label"].lower().strip() for o in scene)
            boxes = collections.defaultdict(list)
            for o in scene:
                boxes[o["label"].lower().strip()].append(tuple(o["bbox_3d"][:2]))
            r = {"id": qq["id"], "opts": [opt_text(re.sub(r"^[A-D]\.\s*", "", o)) for o in opts]}
            r["gt"] = to_text(gt.get(qq["id"], ""), opts)
            r["claude"] = to_text(ans[qq["id"]], opts) if qq["id"] in ans else None
            p = parse_route(q)
            if not p:
                r["names"], r["q"] = "unparsed", q; recs.append(r); continue
            start, facing, dest, steps = p
            st, lab = {}, {}
            for key, ph in [("start", start), ("facing", facing), ("dest", dest)] + [(f"go{i}", s[1]) for i, s in enumerate(steps) if s[0] == "go"]:
                st[key], lab[key] = resolve(ph, labels)
            order = ["missing", "fuzzy", "ambiguous", "exact"]
            r["names"] = min(st.values(), key=order.index)
            r["missing_names"] = [p for k, p in [("start", start), ("facing", facing), ("dest", dest)] + [(f"go{i}", s[1]) for i, s in enumerate(steps) if s[0] == "go"] if st[k] == "missing"]
            r["n_turns"] = sum(1 for s in steps if s[0] == "turn")
            r["q"], r["labels"] = q, sorted(labels)
            if r["names"] != "missing":
                gl =[lab[k] for k in sorted((k for k in lab if k.startswith("go")), key=lambda k: int(k[2:]))]
                gsteps, it = [], iter(gl)
                for s in steps:
                    gsteps.append(("go", next(it)) if s[0] == "go" else s)
                t = walk(lab["start"], lab["facing"], gsteps, boxes)
                if t is None:
                    r["geo"] = "trailing"
                else:
                    r["geo"] = " ".join(t)
                    sw = {"left": "right", "right": "left", "back": "back"}
                    r["geo_swapped"] = " ".join(sw[x] for x in t)
            recs.append(r)
    return recs


def pct(a, b):
    return f"{100 * a / b:5.1f}% ({a}/{b})" if b else "-"


def report(name, recs):
    n = len(recs)
    acc = lambda rs: pct(sum(r["claude"] == r["gt"] for r in rs), len(rs))
    print(f"== {name}: {n} route questions, Claude answered {sum(r['claude'] is not None for r in recs)}; Claude accuracy {acc(recs)}")
    by = collections.defaultdict(list)
    for r in recs:
        by[r["names"]].append(r)
    print("   names: " + " | ".join(f"{k} {len(v)} (Claude {acc(v)})" for k, v in sorted(by.items())))
    miss = collections.Counter(m for r in recs for m in r.get("missing_names", []))
    if miss:
        print("   most often missing: " + ", ".join(f"{k} x{v}" for k, v in miss.most_common(8)))
    for r in [r for r in recs if r["names"] == "missing"][:EX]:
        print(f"   e.g. missing {r['missing_names']} | labels: {', '.join(r['labels'])[:260]}")
    for r in [r for r in recs if r["names"] == "unparsed"][:2]:
        print(f"   e.g. unparsed: {r.get('q', '')[:200]}")
    geo = [r for r in recs if r.get("geo") and r["geo"] != "trailing"]
    tr = sum(1 for r in recs if r.get("geo") == "trailing")
    print(f"   geometric answer computed for {len(geo)} (trailing blank {tr}, names missing {len(by['missing'])})")
    if geo:
        agree = [r for r in geo if r["geo"] == r["gt"]]
        print(f"   geometry = benchmark answer {pct(len(agree), len(geo))}; with left/right swapped "
              f"{pct(sum(r['geo_swapped'] == r['gt'] for r in geo), len(geo))}; geometry among the options "
              f"{pct(sum(r['geo'] in r['opts'] for r in geo), len(geo))}")
        dis = [r for r in geo if r["geo"] != r["gt"]]
        print(f"   Claude accuracy where geometry = benchmark {acc(agree)}; where they differ {acc(dis)}; "
              f"Claude = geometry where they differ {pct(sum(r['claude'] == r['geo'] for r in dis), len(dis))}")
        for k in sorted({r['n_turns'] for r in geo}):
            g = [r for r in geo if r["n_turns"] == k]
            print(f"   {k} turn(s): n {len(g):3}  geometry = benchmark {pct(sum(r['geo'] == r['gt'] for r in g), len(g))}  Claude {acc(g)}")
    return {"n": n, "names": {k: len(v) for k, v in by.items()}, "geo_n": len(geo),
            "geo_agree": sum(r["geo"] == r["gt"] for r in geo), "geo_swapped_agree": sum(r["geo_swapped"] == r["gt"] for r in geo),
            "claude_correct": sum(r["claude"] == r["gt"] for r in recs),
            "claude_correct_where_geo_agrees": sum(r["claude"] == r["gt"] for r in geo if r["geo"] == r["gt"]),
            "claude_correct_where_geo_differs": sum(r["claude"] == r["gt"] for r in geo if r["geo"] != r["gt"]),
            "claude_eq_geo_where_differs": sum(r["claude"] == r["geo"] for r in geo if r["geo"] != r["gt"]),
            "by_turns": {k: {"n": sum(1 for r in geo if r["n_turns"] == k),
                             "geo_agree": sum(1 for r in geo if r["n_turns"] == k and r["geo"] == r["gt"]),
                             "claude": sum(1 for r in geo if r["n_turns"] == k and r["claude"] == r["gt"])}
                         for k in sorted({r["n_turns"] for r in geo})},
            "missing": dict(miss.most_common(20))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=RUN_DIR,GT_FILE")
    ap.add_argument("--json"); ap.add_argument("--examples", type=int, default=0, help="print N questions with missing names")
    a = ap.parse_args()
    global EX
    EX = a.examples
    out = {}
    for spec in a.runs:
        name, _, rest = spec.partition("=")
        run, gt = rest.split(",", 1)
        out[name] = report(name, check(name, os.path.expandvars(run), os.path.expandvars(gt)))
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        json.dump(out, open(a.json, "w"), indent=1)
        print(f"json: {a.json}")


if __name__ == "__main__":
    main()
