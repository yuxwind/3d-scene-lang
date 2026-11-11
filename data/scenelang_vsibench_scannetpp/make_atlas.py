"""Build scenelang_atlas.html: the 50 VSI-Bench ScanNet++ scenes, their scene language, and their questions
with every input's answer (video only, scene language, video + scene language, computed from the boxes).

    python make_atlas.py   (needs the GPU env for decord: source scripts/lib.sh; activate_env a100)
Reads atlas_template.html next to it and writes scenelang_atlas.html here (the page body, published as the artifact)
and a standalone copy at data/scenelang_atlas.html. Frames are embedded as JPEG data URIs; the videos are separate
files in data/scenelang_atlas_media/ (make_atlas_videos.sh), loaded by relative path.
"""
import ast
import base64
import io
import json
import os
import re
from pathlib import Path

from sr_vggt.data import decode_video_frames
from sr_vggt.eval import MCA_QUESTION_TYPES, fuzzy_matching, mean_relative_accuracy, read_rows, to_float

D = Path(__file__).resolve().parent
VSI = "/lustre/fswork/projects/rech/bef/uxt97ti/data/VSI-Bench"
BASE = Path(os.environ["SCRATCH"]) / "3d_scene_lang" / "baseline_qwen35_4b"
N_FRAMES = 6

rows = [json.loads(l) for l in open(D / "vsibench_scannetpp_scenelang.jsonl")]


def load(d):
    out = {}
    for f in sorted(Path(d).glob("predictions*.jsonl")):
        out.update(read_rows(f))
    return out


LETTER = re.compile(r"\b([A-D])\b(?![a-z'])")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def last_answer(t, p):
    """As eval_scenelang.final_answer: the final 'Answer:' line if any, else the last option letter / number."""
    m = re.findall(r"Answer:\s*\**\s*([^\n]+)", p)
    if m:
        p = m[-1]
    m = (LETTER if t in MCA_QUESTION_TYPES else NUMBER).findall(p.replace(",", ""))
    return m[-1] if m else ""


def load_chain(*dirs):  # later directories override earlier ones (reruns of some rows)
    out = {}
    for d in dirs:
        out.update(load(D / d))
    return out


# the prompt texts, read from eval_scenelang.py without running it
_tree = ast.parse((D / "eval_scenelang.py").read_text())
PROMPTS = {t.id: ast.literal_eval(n.value) for n in _tree.body if isinstance(n, ast.Assign)
           for t in n.targets if isinstance(t, ast.Name) and t.id in ("SCENE_INTRO", "FULL_INTRO", "HOW")}


def credit(t, pred, gt):
    if t in MCA_QUESTION_TYPES:
        return float(fuzzy_matching(pred).lower() == gt.lower())
    p, g = to_float(fuzzy_matching(pred)), to_float(gt)
    return 0.0 if p is None or not g else float(mean_relative_accuracy(p, g, .5, .95, .05))


runs = {"video": load(BASE),
        # 1,024 tokens; the 14 coat-rack rows on the fixed text; the 85 unanswered at 4,096; the 137 still cut off at 16,384
        "scene": load_chain("qwen35_4b_scene_long", "qwen35_4b_scene_long_coatrack", "qwen35_4b_scene_4096",
                            "qwen35_4b_scene_16384"),
        "scene_full": load_chain("qwen35_4b_scene_fullprompt", "qwen35_4b_scene_fullprompt_rest"),
        "video_scene": load_chain("qwen35_4b_video_scene", "qwen35_4b_video_scene_coatrack")}
LONG = {"scene", "scene_full"}  # runs whose outputs may reason before answering
oracle = {r["index"]: r["prediction"] for r in json.load(open(D / "oracle_predictions.json"))}

scenes, questions = {}, []
for r in rows:
    s = r["scene_name"]
    if s not in scenes:
        scenes[s] = {"objects": [json.loads(l.rstrip(",")) for l in r["scene_language"].split("\n")[1:-1]],
                     "text": r["scene_language"], "n_q": 0, "n_deb": 0}
    scenes[s]["n_q"] += 1
    scenes[s]["n_deb"] += r["debiased"]
    t = r["question_type"]
    ans = {}
    for name, p in runs.items():
        if r["index"] in p:
            raw = p[r["index"]]["prediction"]
            final = last_answer(t, raw) if name in LONG else raw
            ans[name] = {"final": fuzzy_matching(final) if final else "", "credit": round(credit(t, final, r["ground_truth"]), 3),
                         **({"raw": raw} if name in LONG and len(raw) > 30 else {})}
    if r["index"] in oracle:
        ans["boxes"] = {"final": oracle[r["index"]], "credit": round(credit(t, oracle[r["index"]], r["ground_truth"]), 3)}
    questions.append({"i": r["index"], "scene": s, "type": t, "deb": r["debiased"], "q": r["question"],
                      "options": r["options"], "gt": r["ground_truth"], "ans": ans})

for k, s in enumerate(sorted(scenes)):
    frames = decode_video_frames(f"{VSI}/scannetpp/{s}.mp4", 1.0, 64)
    pick = [frames[round(j * (len(frames) - 1) / (N_FRAMES - 1))] for j in range(N_FRAMES)]
    uris = []
    for im in pick:
        im = im.copy()
        im.thumbnail((360, 360))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=68)
        uris.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
    scenes[s]["frames"] = uris
    scenes[s]["n_video_frames"] = len(frames)
    print(k, s, len(frames), flush=True)

def example(style, r):
    scene = "\n".join(r["scene_language"].split("\n")[:3] + ["  ... (one line per object) ...", "]"])
    q = r["question"] + ("\nOptions:\n" + "\n".join(r["options"]) if r["options"] else "")
    if style == "basic":
        post = ("Answer with the option's letter from the given choices directly." if r["options"]
                else "Please answer the question using a single word or phrase.")
        return f"{PROMPTS['SCENE_INTRO']}\n<scene>\n{scene}\n</scene>\n{q}\n{post}"
    how = next((h for k, h in PROMPTS["HOW"].items() if r["question_type"].startswith(k)), "")
    end = ("Compute the answer step by step from the numbers above, then end with a final line "
           + ("'Answer: <option letter>'." if r["options"] else "'Answer: <number>' (only the number, in the unit the question asks for)."))
    return f"{PROMPTS['FULL_INTRO']}\n<scene>\n{scene}\n</scene>\n{q}\n{how}\n{end}"


ex_rows = {t: next(r for r in rows if r["debiased"] and r["question_type"].startswith(t))
           for t in ("object_rel_direction", "object_abs_distance", "room_size_estimation")}
prompts = {"examples": {t: {"basic": example("basic", r), "full": example("full", r), "index": r["index"]}
                        for t, r in ex_rows.items()},
           "how": PROMPTS["HOW"]}
data = {"scenes": scenes, "questions": questions, "prompts": prompts}
html = (D / "atlas_template.html").read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
(D / "scenelang_atlas.html").write_text(html)  # the page body: what the artifact is published from
SKELETON = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<style>:root{color-scheme:light}body{margin:0;font-size:14px}img{max-width:100%}'
            '[hidden]{display:none!important}</style>\n</head>\n<body>\n')
(D.parent / "scenelang_atlas.html").write_text(SKELETON + html + "\n</body>\n</html>\n")  # standalone, next to the media
print("wrote", D / "scenelang_atlas.html", "and", D.parent / "scenelang_atlas.html", round(len(html) / 1e6, 2), "MB;",
      "videos: data/scenelang_atlas_media/ (make_atlas_videos.sh)")
