#!/usr/bin/env python3
"""Compare VSI-Bench and ReVSI question sets: counts, question templates, options, answers, per question type.

usage: bench_compare.py [--overlap] [--vsi JSONL] [--vsi-debiased PARQUET] [--revsi PARQUET] [--json OUT]
  --overlap: instead, match the two sets: shared scenes per dataset, then per task the VSI-Bench questions that ReVSI
             asks again in the same scene (same wording after normalizing the template changes, or same object names),
             and whether their options (set, order) and answers agree; numeric answers: |relative difference|.
  defaults: /scr/xinyu/data/VSI-Bench/test.jsonl, .../test_debiased.parquet, /scr/xinyu/data/ReVSI/64_frame/test-00000-of-00001.parquet
For each benchmark and question type: n (and debiased n for VSI-Bench), n per source dataset, scenes, question templates
(words in >=40 % of that type's questions kept, the rest shown as "…"; top 3 with counts), two example questions, the
number of options and an example option list, the answer-letter distribution (multiple choice) or numeric ground-truth
quantiles (min/p25/median/p75/max), words per question, and for route planning the number of turns to fill in.
prints: a compact table per benchmark; --json writes everything (for plots). Needs pandas (training env).
"""
import argparse, collections, json, os, re
import pandas as pd

TOK = re.compile(r"[A-Za-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[^\sA-Za-z\d]")


def opts(o):
    if o is None or (isinstance(o, float) and pd.isna(o)) or str(o) in ("None", "nan", ""):
        return None
    if isinstance(o, str):
        try:
            o = json.loads(o)
        except Exception:
            return [o]
    return [str(x) for x in list(o)]


def templates(qs, k=3):
    toks = [TOK.findall(q) for q in qs]
    df = collections.Counter(w.lower() for t in toks for w in set(t))
    keep = {w for w, c in df.items() if c >= 0.4 * len(qs)}
    out = collections.Counter()
    for t in toks:
        s, prev = [], False
        for w in t:
            if w.lower() in keep:
                s.append(w); prev = False
            elif not prev:
                s.append("…"); prev = True
        out[re.sub(r"\s+([,.?!:;)])", r"\1", " ".join(s)).replace("( ", "(")] += 1
    return [{"template": t, "n": n} for t, n in out.most_common(k)], len(out)


def num(x):
    try:
        return float(x)
    except Exception:
        return None


def summarize(df, name, scene_col, deb_ids=None):
    res = {"name": name, "n": len(df), "scenes": int(df[scene_col].nunique()),
           "datasets": df["dataset"].value_counts().to_dict(), "types": {}}
    for qt, g in sorted(df.groupby("question_type"), key=lambda x: -len(x[1])):
        os_ = [opts(o) for o in g["options"]]
        mc = [o for o in os_ if o]
        t, nt = templates(list(g["question"]))
        e = {"n": len(g), "datasets": g["dataset"].value_counts().to_dict(), "scenes": int(g[scene_col].nunique()),
             "templates": t, "n_templates": nt, "examples": list(g["question"].iloc[[0, len(g) // 2]]),
             "words": round(g["question"].str.split().str.len().mean(), 1), "mc": bool(mc)}
        if deb_ids is not None:
            e["n_debiased"] = int(g["id"].isin(deb_ids).sum())
        if mc:
            e["n_options"] = dict(collections.Counter(len(o) for o in mc))
            e["option_example"] = mc[0]
            gt = [str(x).strip() for x in g["ground_truth"]]
            letters = collections.Counter(x[:1].upper() if re.match(r"^[A-Da-d]\b", x) else
                                          ("ABCD"[o.index(x)] if o and x in o else "?") for x, o in zip(gt, os_))
            e["answer_letters"] = dict(sorted(letters.items()))
            def val(x, o):
                if re.match(r"^[A-Da-d]\b", x) and o and "ABCD".index(x[:1].upper()) < len(o):
                    x = o["ABCD".index(x[:1].upper())]
                return re.sub(r"^[A-D]\.\s*", "", x).strip().lower()
            vals = collections.Counter(val(x, o) for x, o in zip(gt, os_))
            e["answer_values"] = dict(vals.most_common(12))
            e["answer_values_distinct"] = len(vals)
        blanks = g["question"].str.count(r"\[please fill in\]")
        if blanks.sum():
            e["blanks"] = {int(k): int(v) for k, v in blanks.value_counts().sort_index().items()}
        else:
            v = pd.Series([num(x) for x in g["ground_truth"]]).dropna()
            e["gt"] = {q: round(float(v.quantile(p)), 2) for q, p in
                       (("min", 0), ("p25", .25), ("median", .5), ("p75", .75), ("max", 1))} if len(v) else {}
            e["gt_hist"] = [round(float(x), 3) for x in v.sample(min(len(v), 400), random_state=0)] if len(v) else []
        res["types"][qt] = e
    return res


GROUPS = [("abs_dist", "object_abs_distance", "object_abs_distance"),
          ("obj_size", "object_size_estimation", "object_size_estimation"),
          ("count", "object_counting", "object_counting_single"),
          ("rel_dist", "object_rel_distance", "object_rel_distance_closest"),
          ("rel_dir_3way", "object_rel_direction_medium", "object_rel_direction_forward_easy"),
          ("rel_dir_quad", "object_rel_direction_hard", "object_rel_direction_forward_hard"),
          ("room_size", "room_size_estimation", "room_size_estimation_multiple"),
          ("route", "route_planning", "route_planning")]
DSMAP = {"arkitscenes": "ARKitScenes", "scannet": "ScanNetv2", "scannetpp": "ScanNetPPv2"}
OBJ = {"abs_dist": r"distance between the (.+?) and the (.+?) \(in meters\)",
       "obj_size": r"height\) of the (.+?), measured",
       "count": r"how many (.+?)\(s\) are in",
       "rel_dist": r"objects \((.+?)\) is the closest to the (.+?)\?",
       "rel_dir_3way": r"standing by the (.+?) and facing the (.+?), is the (.+?) to my",
       "rel_dir_quad": r"standing by the (.+?) and facing the (.+?), is the (.+?) to my"}


def norm_q(q):
    q = re.sub(r"\s+", " ", str(q).lower()).strip()
    q = q.replace("based on visual evidence from the video, w", "w").replace("direct distance", "distance")
    q = q.replace("in the scene", "in this room").replace("size of the scene", "size of this room")
    q = re.sub(r" the directions refer to the quadrants.*$", "", q)
    q = q.replace("[please fill in].", "[please fill in]")
    return re.sub(r"\.(\s+\d+\.)", r"\1", q).strip()


def objs(g, q):
    m = re.search(OBJ[g], q) if g in OBJ else None
    if not m:
        return None
    parts = [p.strip() for x in m.groups() for p in x.split(",")]
    return tuple(parts) if g.startswith("rel_dir") else tuple(sorted(parts))


def opt_texts(o):
    o = opts(o)
    return [re.sub(r"^[a-d]\.\s*", "", x.strip().lower()) for x in o] if o else None


def answer(gt, o):
    x = str(gt).strip()
    ot = opt_texts(o)
    if ot and re.match(r"^[A-Da-d]$", x):
        return ot["ABCD".index(x.upper())] if "ABCD".index(x.upper()) < len(ot) else x.lower()
    return re.sub(r"^[a-d]\.\s*", "", x.lower())


def overlap(vsi, rv, json_out=None, show=None):
    """Same scenes, then same question (normalized wording) or same objects, then same options / answer."""
    vsi = vsi.assign(scene=vsi["dataset"].map(DSMAP) + "/" + vsi["scene_name"].astype(str))
    rv = rv.assign(scene=rv["dataset"] + "/" + rv["scene_id"].astype(str))
    if show:
        g = {x[0]: x for x in GROUPS}[show[1]]
        for name, df, qt in (("VSI-Bench", vsi, g[1]), ("ReVSI", rv, g[2])):
            sub = df[(df.scene == show[0]) & (df.question_type == qt)]
            print(f"== {name} {qt} in {show[0]}: {len(sub)}")
            for r in sub.head(12).itertuples():
                print(f"  {norm_q(r.question)[:170]} | opts {opt_texts(r.options)} | gt {answer(r.ground_truth, r.options)}")
        return
    sv, sr = set(vsi["scene"]), set(rv["scene"])
    both = sv & sr
    print(f"scenes: VSI-Bench {len(sv)}, ReVSI {len(sr)}, shared {len(both)}")
    for d in sorted(set(DSMAP.values()) | set(rv["dataset"])):
        a = {s for s in sv if s.startswith(d + "/")}; b = {s for s in sr if s.startswith(d + "/")}
        print(f"   {d:12} VSI {len(a):4}  ReVSI {len(b):4}  shared {len(a & b):4}"
              f"   e.g. VSI {sorted(a)[:1]} ReVSI {sorted(b)[:1]}")
    res = {"scenes": {"vsi": len(sv), "revsi": len(sr), "shared": len(both)}, "groups": {}}
    hdr = f"{'task':13} {'VSI n':>6} {'in shared':>9} {'ReVSI n':>7} {'in shared':>9} | {'same q':>6} {'same objs':>9} | " \
          f"{'same ans':>8} {'same opt set':>12} {'same opt order':>14} {'num |rel diff| med':>18}"
    print(hdr)
    examples = []
    for g, qv, qr in GROUPS:
        A = vsi[vsi.question_type == qv]; B = rv[rv.question_type == qr]
        A2, B2 = A[A.scene.isin(both)], B[B.scene.isin(both)]
        bq = {(r.scene, norm_q(r.question)): r for r in B2.itertuples()}
        bo = collections.defaultdict(list)
        for r in B2.itertuples():
            k = objs(g, norm_q(r.question))
            if k:
                bo[(r.scene, k)].append(r)
        same_q = same_o = same_ans = same_set = same_order = 0
        rel = []
        for r in A2.itertuples():
            nq = norm_q(r.question)
            m = bq.get((r.scene, nq))
            if m is not None:
                same_q += 1
            else:
                k = objs(g, nq)
                m = bo[(r.scene, k)][0] if k and bo.get((r.scene, k)) else None
            if m is None:
                continue
            same_o += 1
            oa, ob = opt_texts(r.options), opt_texts(m.options)
            if oa and ob:
                same_set += set(oa) == set(ob); same_order += oa == ob
                ok = answer(r.ground_truth, r.options) == answer(m.ground_truth, m.options)
                same_ans += ok
                if not ok and len(examples) < 40:
                    examples.append({"task": g, "scene": r.scene, "vsi_q": r.question, "revsi_q": m.question,
                                     "vsi_opts": oa, "revsi_opts": ob, "vsi_gt": answer(r.ground_truth, r.options),
                                     "revsi_gt": answer(m.ground_truth, m.options)})
            else:
                x, y = num(r.ground_truth), num(m.ground_truth)
                if x is not None and y is not None:
                    same_ans += abs(x - y) <= 0.01 * max(abs(x), 1e-9)
                    rel.append(abs(x - y) / max(abs(x), 1e-9))
                    if abs(x - y) > 0.1 * max(abs(x), 1e-9) and len(examples) < 40:
                        examples.append({"task": g, "scene": r.scene, "vsi_q": r.question, "revsi_q": m.question,
                                         "vsi_gt": x, "revsi_gt": y})
        med = f"{100 * pd.Series(rel).median():.1f}%" if rel else "-"
        print(f"{g:13} {len(A):6} {len(A2):9} {len(B):7} {len(B2):9} | {same_q:6} {same_o:9} | "
              f"{same_ans:8} {same_set if not rel else '-':>12} {same_order if not rel else '-':>14} {med:>18}")
        res["groups"][g] = {"vsi_n": len(A), "vsi_shared_scenes": len(A2), "revsi_n": len(B), "revsi_shared_scenes": len(B2),
                            "same_question": same_q, "same_objects": same_o, "same_answer": same_ans,
                            "same_option_set": None if rel else same_set, "same_option_order": None if rel else same_order,
                            "numeric_rel_diff_median": None if not rel else float(pd.Series(rel).median()),
                            "numeric_rel_diff_p90": None if not rel else float(pd.Series(rel).quantile(.9))}
    print("same q: identical after normalizing the template wording; same objs: same scene + same object names "
          "(includes same q); same ans: MC same answer text, numeric within 1 %")
    for e in examples[:6]:
        print("  diff:", json.dumps(e)[:400])
    if json_out:
        os.makedirs(os.path.dirname(os.path.abspath(json_out)), exist_ok=True)
        json.dump({**res, "examples": examples}, open(json_out, "w"), indent=1, default=str)
        print(f"json: {json_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", action="store_true", help="match questions across the two benchmarks instead")
    ap.add_argument("--show", nargs=2, metavar=("SCENE", "TASK"), help="with --overlap: print both sets' questions + answers "
                    "for one scene (e.g. ScanNetPPv2/09c1414f1b) and task group (abs_dist, rel_dist, rel_dir_3way, ...)")
    ap.add_argument("--vsi", default="/scr/xinyu/data/VSI-Bench/test.jsonl")
    ap.add_argument("--vsi-debiased", default="/scr/xinyu/data/VSI-Bench/test_debiased.parquet")
    ap.add_argument("--revsi", default="/scr/xinyu/data/ReVSI/64_frame/test-00000-of-00001.parquet")
    ap.add_argument("--json")
    a = ap.parse_args()
    vsi = pd.read_json(a.vsi, lines=True)
    deb = set(pd.read_parquet(a.vsi_debiased)["id"])
    rv = pd.read_parquet(a.revsi)
    if a.overlap:
        return overlap(vsi, rv, a.json, a.show)
    out = {"vsibench": summarize(vsi, "VSI-Bench", "scene_name", deb), "revsi": summarize(rv, "ReVSI (64 frames)", "scene_id")}
    for b in out.values():
        print(f"== {b['name']}: {b['n']} questions, {b['scenes']} scenes, datasets {b['datasets']}")
        for qt, e in b["types"].items():
            extra = (f"opts {e['n_options']} letters {e['answer_letters']}" if e["mc"] else f"gt {e['gt']}")
            print(f"  {qt:38} n{e['n']:5}{' deb' + str(e['n_debiased']) if 'n_debiased' in e else ''} "
                  f"words {e['words']:5} tmpl {e['n_templates']:4} | {extra}")
            print(f"      {e['templates'][0]['template'][:230]}")
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        json.dump(out, open(a.json, "w"), indent=1, default=str)
        print(f"json: {a.json}")


if __name__ == "__main__":
    main()
