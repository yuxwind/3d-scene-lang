"""Build traces_atlas.html: three example videos of data/simplesr_train_traces.jsonl.gz that together cover every
question type, with each question's scene block, reasoning trace and answer, and a 3D view of the boxes.

    python make_atlas.py    (stdlib only; run from anywhere)

Writes, next to this file:
  traces_atlas.html       standalone page (open locally; videos load from media/)
  traces_atlas.body.html  the same page without the document skeleton (what the artifact is published from)
Videos: media/<dataset>_<video>.mp4, small copies (320 px, 6 fps) made with ffmpeg beforehand.
"""
import gzip
import json
import re
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parent
TRACES = D.parent / "simplesr_train_traces.jsonl.gz"
VIDEOS = ["scannetppv2/108ec0b806.mp4", "scannet/scene0477_01.mp4", "arkitscenes/42444751.mp4"]  # greedy type cover
SCENE = re.compile(r"<scene>\s*(\[.*?\])\s*</scene>", re.S)
THINK = re.compile(r"<think>\s*(.*?)\s*</think>", re.S)

all_types, per = Counter(), {v: [] for v in VIDEOS}
with gzip.open(TRACES, "rt") as f:
    for line in f:
        r = json.loads(line)
        all_types[r["question_type"]] += 1
        if r["video"] in per:
            per[r["video"]].append(r)

videos = []
for v in VIDEOS:
    objs, key = [], {}
    qs = []
    for r in per[v]:
        g = r["conversations"][1]["value"]
        m = SCENE.search(g)
        scene = json.loads(m.group(1)) if m else []
        ids = []
        for o in scene:
            k = tuple(o["bbox_3d"])
            if k not in key:
                key[k] = len(objs)
                objs.append({"label": re.sub(r"\s\d+$", "", o["label"]), "b": o["bbox_3d"]})
            ids.append(key[k])
        t = THINK.search(g)
        qs.append({
            "type": r["question_type"], "source": r["source"], "i": r["meta"].get("i"),
            "prompt": r["conversations"][0]["value"].replace("<image>\n", ""),
            "answer": str(r["meta"]["answer"]), "unit": r["meta"].get("unit"),
            "intro": g[:m.start()].strip() if m else "", "scene": m.group(1) if m else "", "obj": ids,
            "think": t.group(1) if t else "", "final": g[t.end():].strip() if t else g[-200:],
        })
    qs.sort(key=lambda q: (q["type"], q["i"] or 0))
    videos.append({"video": v, "dataset": v.split("/")[0], "media": "media/" + v.replace("/", "_"),
                   "objects": objs, "questions": qs})
    print(v, len(qs), "questions,", len(objs), "distinct boxes,", len(set(q["type"] for q in qs)), "types")

data = {"types": dict(all_types.most_common()), "rows": sum(all_types.values()), "videos": videos}
body = (D / "template.html").read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
(D / "traces_atlas.body.html").write_text(body)
SKELETON = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<style>:root{color-scheme:light}body{margin:0;font-size:14px}img{max-width:100%}'
            '[hidden]{display:none!important}</style>\n</head>\n<body>\n')
(D / "traces_atlas.html").write_text(SKELETON + body + "\n</body>\n</html>\n")
print("wrote", D / "traces_atlas.html", round(len(body) / 1e6, 2), "MB")
