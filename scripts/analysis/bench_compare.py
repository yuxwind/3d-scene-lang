#!/usr/bin/env python3
"""Compare VSI-Bench and ReVSI question sets: counts, question templates, options, answers, per question type.

usage: bench_compare.py [--vsi JSONL] [--vsi-debiased PARQUET] [--revsi PARQUET] [--json OUT]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vsi", default="/scr/xinyu/data/VSI-Bench/test.jsonl")
    ap.add_argument("--vsi-debiased", default="/scr/xinyu/data/VSI-Bench/test_debiased.parquet")
    ap.add_argument("--revsi", default="/scr/xinyu/data/ReVSI/64_frame/test-00000-of-00001.parquet")
    ap.add_argument("--json")
    a = ap.parse_args()
    vsi = pd.read_json(a.vsi, lines=True)
    deb = set(pd.read_parquet(a.vsi_debiased)["id"])
    rv = pd.read_parquet(a.revsi)
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
