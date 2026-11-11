"""Zero-shot of an UNMODIFIED Qwen VLM on VSI-Bench ScanNet++ with ground-truth scene language (Table 2 setup).

--input scene        : scene language + question, no frames (Table 2 "Structured scene language input")
--input video_scene  : video + scene language + question (Table 2 "Video + Structured scene language")
Video-only is the existing baseline (baseline_qwen35_4b, same row index). Same frames, token budget,
decoding and scoring as eval_base.py (1 fps, <=64 frames, 64-128 tokens/frame, greedy, 16 new tokens,
no system prompt, thinking off, lmms-eval post-prompts, sr_vggt.eval.score).

    python eval_scenelang.py --rows vsibench_scannetpp_scenelang.jsonl --input scene --out_dir D --model M [--vsibench_dir V]
    python eval_scenelang.py --rows ... --input scene --out_dir D --merge
Scoring takes the LAST option letter / number of the output (final_answer), so a model that reasons
before answering is not cut off; with the default 16 tokens this is the same as the first one.
"""
import argparse
import json
import math
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from sr_vggt.eval import LMMS_DEFAULT_VSIBENCH_KW as KW, MCA_QUESTION_TYPES, read_rows, score

SCENE_INTRO = ("This is a 3D scene, given as a list of objects. Each object has a label and a 3D bounding box "
               "bbox_3d = [x, y, z, dx, dy, dz, yaw, roll, pitch]: the box center and size in meters, and its "
               "rotation in radians. z points up.")

FULL_INTRO = """This is a 3D scene, given as a list of objects, one per line: {"label": <category>, "bbox_3d": [x, y, z, dx, dy, dz, yaw, roll, pitch]}.
- x, y, z: the centre of the object's 3D bounding box, in meters. x and y are horizontal coordinates on the floor plane; z is the vertical axis (height, pointing up). z = 0 is not the floor: the floor is at about the lowest box bottom (z - dz/2) in the scene.
- dx, dy, dz: the full size of the box in meters along its own axes. dx and dy are horizontal sizes; dz is the vertical height of the object.
- yaw: rotation of the box around the vertical z axis, in radians (yaw = 0 means dx runs along x and dy along y). roll and pitch are tilts around the box's own x and y axes, in radians; they are 0 for upright objects.
- Lines with the same label are different instances of that category. All lengths are in meters."""

SIMPLE_INTRO = ("This is a 3D scene, given as a list of objects, one per line: {\"label\": <category>, \"bbox_3d\": "
                "[x_center, y_center, z_center, x_size, y_size, z_size, yaw, roll, pitch]}. x and y are horizontal "
                "coordinates on the floor plane; z is vertical (height, pointing up). Lines with the same label are "
                "different instances of that category. All lengths are in meters.")

CORNERS_INTRO = """This is a 3D scene, given as a list of objects, one per line: {"label": <category>, "center": [x, y, z], "corners": [[x, y, z], ... 8 points]}.
- center: the centre of the object's 3D bounding box, in meters. x and y are horizontal coordinates on the floor plane; z is the vertical axis (height, pointing up). z = 0 is not the floor: the floor is at about the lowest corner z in the scene.
- corners: the 8 vertices of the object's 3D bounding box, in meters, in the same coordinates. The box may be rotated, so its edges need not be parallel to the x and y axes; the object's size and extent follow from its corners.
- Lines with the same label are different instances of that category. All lengths are in meters."""


POINTS_INTRO = """This is a 3D scene, given as a list of objects, one per line: {"label": <category>, "center": [x, y, z], "points": [[x, y, z], ... 8 points]}.
- center: the centre of the object's 3D bounding box, in meters. x and y are horizontal coordinates on the floor plane; z is the vertical axis (height, pointing up). z = 0 is not the floor: the floor is at about the lowest point z in the scene.
- points: 8 points sampled at random on the surface of the object's 3D bounding box, in meters, in the same coordinates. The object's extent and size can be estimated from them.
- Lines with the same label are different instances of that category. All lengths are in meters."""


def corners_scene(text, points_seed=None):
    """The scene's 9-DoF boxes as {label, center, corners}: the 8 vertices (+-1/2 size along the box axes), with
    oracle.py's rotation R = Rz(yaw) Rx(roll) Ry(pitch); 2 decimals, same object order.
    points_seed given: {label, center, points} instead, 8 points uniform on the box surface (a face pair picked
    with probability proportional to its area, the other two coordinates uniform), seeded by points_seed + index."""
    lines = []
    for idx, o in enumerate(json.loads(text)):
        x, y, z, dx, dy, dz, yaw, roll, pitch = o["bbox_3d"]
        cz, sz, cx, sx, cy, sy = math.cos(yaw), math.sin(yaw), math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
        Rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]
        Rx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]]
        Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
        mul = lambda A, B: [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        R = mul(mul(Rz, Rx), Ry)
        if points_seed is None:
            local = [(a * dx, b * dy, c * dz) for a in (-.5, .5) for b in (-.5, .5) for c in (-.5, .5)]
        else:
            rng, size = random.Random(f"{points_seed}-{idx}"), (dx, dy, dz)
            area = (dy * dz, dx * dz, dx * dy)
            local = []
            for _ in range(8):
                axis = rng.choices(range(3), weights=area)[0] if sum(area) > 0 else rng.randrange(3)
                l = [rng.uniform(-.5, .5) * e for e in size]
                l[axis] = rng.choice((-.5, .5)) * size[axis]
                local.append(l)
        world = [[round(ctr + sum(R[i][k] * l[k] for k in range(3)), 2) for i, ctr in enumerate((x, y, z))] for l in local]
        lines.append(json.dumps({"label": o["label"], "center": [round(x, 2), round(y, 2), round(z, 2)],
                                 ("corners" if points_seed is None else "points"): world}))
    return "[\n" + ",\n".join(lines) + "\n]"


HOW = {
    "object_counting": "To count, count the lines whose label is exactly the category asked about.",
    "object_size_estimation": "The longest dimension of an object is max(dx, dy, dz) of its box, in meters; multiply by 100 for centimeters.",
    "object_abs_distance": ("Measure the distance between the closest points of the two boxes. Treat each box as axis-aligned "
                            "with half-sizes hx = (|dx*cos(yaw)| + |dy*sin(yaw)|)/2 and hy = (|dx*sin(yaw)| + |dy*cos(yaw)|)/2 "
                            "along x and y, and hz = dz/2 along z. For each axis the gap is max(0, |difference of centres| - "
                            "(half-size of one + half-size of the other)); the distance is sqrt(gap_x^2 + gap_y^2 + gap_z^2). "
                            "If a category has several instances, use the closest pair."),
    "object_rel_distance": ("Compute, for every candidate, the distance between the closest points of its box and the target's box: "
                            "treat each box as axis-aligned with half-sizes hx = (|dx*cos(yaw)| + |dy*sin(yaw)|)/2, "
                            "hy = (|dx*sin(yaw)| + |dy*cos(yaw)|)/2, hz = dz/2; per axis gap = max(0, |difference of centres| - "
                            "sum of half-sizes); distance = sqrt(gap_x^2 + gap_y^2 + gap_z^2). For a category with several "
                            "instances use its closest instance. Then compare the distances."),
    "room_size_estimation": ("The room size is its floor area in square meters, measured in the x-y plane (z is height and is "
                             "not used). Estimate the walls from the horizontal extent of the objects: the room spans at least from "
                             "the smallest to the largest x and y reached by the box edges (centre -/+ half the horizontal size), "
                             "and furniture usually stands along the walls."),
    "object_rel_direction": ("Work in the x-y floor plane with the box centres. Standing at the centre A of the object you stand by "
                             "and facing the centre B of the object you face, let f = B - A and, for the queried object C, v = C - A. "
                             "cross = f_x*v_y - f_y*v_x: positive means C is on your left, negative on your right. "
                             "dot = f_x*v_x + f_y*v_y: positive means C is in front of you, negative behind. The angle between f and v "
                             "is acos(dot / (|f| |v|)); C is behind you ('back') when it is at least 135 degrees."),
}


def how(t):
    return next((h for k, h in HOW.items() if t.startswith(k)), "")


ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True)
ap.add_argument("--input", choices=["scene", "video_scene"], required=True)
ap.add_argument("--model")
ap.add_argument("--vsibench_dir")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--max_new_tokens", type=int, default=16)
ap.add_argument("--merge", action="store_true")
ap.add_argument("--debiased_only", action="store_true")
ap.add_argument("--indices", help="file with one row index per line: run only those rows")
ap.add_argument("--thinking", action="store_true", help="Qwen3.5 thinking mode (enable_thinking=True)")
ap.add_argument("--prompt", choices=["basic", "full", "simple1", "simple2", "simple", "corners", "points", "rvs_question", "rvs_all"], default="basic",
                help="full: complete format description, a per-task instruction to compute, and a final 'Answer:' line; "
                     "simple1 (alias simple): short format description, no per-task instruction, the same final line; "
                     "simple2: the full prompt without the per-task instruction; "
                     "corners: simple2 with each box given as its center + 8 corner vertices; "
                     "points: the same with 8 random points on the box surface instead of the corners"
                     "; rvs_question / rvs_all: the sr-rvs training turn (data/reason190k_videoscene/build.py) with the "
                     "objects named in the question / all objects, as upright axis-aligned [x, y, z, w, d, h] boxes")
ap.add_argument("--adapter", help="LoRA adapter directory (e.g. an sr_vggt --arch none checkpoint) applied on top of --model")
ap.add_argument("--system", choices=["none", "short", "scene"], default="none",
                help="short: sr_vggt's training system prompt (data.py _SHORT_SYSTEM), for --adapter runs; "
                     "scene: its scene-only one (data.py _SCENE_SYSTEM, data/reason190k_scene)")
ap.add_argument("--types", help="comma-separated question-type prefixes: run only those rows")
ap.add_argument("--num_shards", type=int, default=1)
ap.add_argument("--shard_index", type=int, default=0)
args = ap.parse_args()
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in open(args.rows)]
if args.debiased_only:
    rows = [r for r in rows if r["debiased"]]
if args.types:
    rows = [r for r in rows if r["question_type"].startswith(tuple(args.types.split(",")))]
if args.indices:
    keep = {int(x) for x in open(args.indices).read().split()}
    rows = [r for r in rows if r["index"] in keep]
LETTER = re.compile(r"\b([A-D])\b(?![a-z'])")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def final_answer(r):
    """With a long --max_new_tokens the model may reason first: keep its last option letter or number."""
    p = r["prediction"]
    m = re.findall(r"Answer:\s*\**\s*([^\n]+)", p)  # --prompt full asks for a final 'Answer: ...' line
    if m:
        p = m[-1]
    if "</think>" in p:  # thinking mode: only the text after the thinking counts
        p = p.split("</think>")[-1]
    elif r.get("thinking"):  # cut off while still thinking: no answer
        p = ""
    if r["question_type"] in MCA_QUESTION_TYPES:
        m = LETTER.findall(p)
    else:
        m = NUMBER.findall(p.replace(",", ""))
    return {**r, "raw_prediction": p, "prediction": m[-1] if m else p}


def report(done):
    res = {}
    for name, keep in (("debiased", lambda r: r["debiased"]), ("full", lambda r: True)):
        sel = [final_answer(done[r["index"]]) for r in rows if keep(r) and r["index"] in done]
        res[name] = {"num_samples": len(sel), "scores": score(sel)}
    (out / f"results{'' if args.merge or args.num_shards == 1 else args.shard_index}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


RVS_DESC = ("This is the 3D scene: the objects the question is about, one {\"label\", \"bbox_3d\"} per line. "
            "bbox_3d = [x, y, z, w, d, h]: the centre and full size of an upright, axis-aligned box, in metres; "
            "origin at the first frame's camera, z points up.")  # = build.py SCENE_DESC
RVS_MCA = "Answer with the option's letter from the given choices directly."
RVS_NA = "Please answer the question using a single word or phrase."


def rvs_scene(r):
    """The sr-rvs scene block: 9-DoF boxes -> axis-aligned [x, y, z, w, d, h] (the box's xy extent after yaw),
    2 decimals, in the VSI-Bench scene frame (the training frame's origin is the first camera; distances, sizes
    and directions do not depend on it). rvs_question keeps the labels named in the question (all objects for room
    size, which names none); repeated labels are numbered ("chair 1", "chair 2") as in the traces; the list is
    shuffled with the row index as seed, as build.py does."""
    objs = json.loads(r["scene_language"])
    for o in objs:  # predicted scenes (data/scenelang_vsibench_vggtdet) come numbered and as [x, y, z, w, d, h]
        o["label"] = re.sub(r"\s+\d+$", "", o["label"])
        o["bbox_3d"] = (o["bbox_3d"] + [0.0])[:7] if len(o["bbox_3d"]) == 6 else o["bbox_3d"]
    if args.prompt == "rvs_question" and not r["question_type"].startswith("room_size"):
        q, spans, keep = r["question"].lower(), [], set()
        for lab in sorted({o["label"] for o in objs}, key=len, reverse=True):
            for m in re.finditer(r"\b" + re.escape(lab.lower()) + r"(?:s|es)?\b", q):
                if not any(a <= m.start() < b for a, b in spans):
                    spans.append((m.start(), m.end())); keep.add(lab)
        objs = [o for o in objs if o["label"] in keep] or objs
    n, seen, out = {}, {}, []
    for o in objs:
        n[o["label"]] = n.get(o["label"], 0) + 1
    for o in objs:
        x, y, z, dx, dy, dz, yaw = o["bbox_3d"][:7]
        c, si = abs(math.cos(yaw)), abs(math.sin(yaw))
        lab = o["label"]
        if n[lab] > 1:
            seen[lab] = seen.get(lab, 0) + 1
            lab = f"{lab} {seen[lab]}"
        out.append({"label": lab, "bbox_3d": [round(v, 2) for v in (x, y, z, dx * c + dy * si, dx * si + dy * c, dz)]})
    random.Random(r["index"]).shuffle(out)
    return "<scene>\n[\n" + ",\n".join(json.dumps(o) for o in out) + "\n]\n</scene>\n"


def prompt(r):
    if args.prompt.startswith("rvs_"):
        # --input scene: the sr-q35sc training turn (data/reason190k_scene/build.py drops the video line)
        body = ("These are frames of a video.\n" if args.input == "video_scene" else "") + RVS_DESC + "\n" + rvs_scene(r) + r["question"]
        if r["question_type"] in MCA_QUESTION_TYPES:
            return body + "\nOptions:\n" + "\n".join(r["options"]) + "\n" + RVS_MCA
        return body + "\n" + RVS_NA
    if args.prompt in ("full", "simple1", "simple2", "simple", "corners", "points"):
        intro, scene = ((CORNERS_INTRO, corners_scene(r["scene_language"])) if args.prompt == "corners" else
                        (POINTS_INTRO, corners_scene(r["scene_language"], points_seed=r["scene_name"])) if args.prompt == "points" else
                        (SIMPLE_INTRO if args.prompt in ("simple1", "simple") else FULL_INTRO, r["scene_language"]))
        head = ("These are frames of a video.\n" if args.input == "video_scene" else "") + \
            intro + "\n<scene>\n" + scene + "\n</scene>"
        h = [how(r["question_type"])] if args.prompt == "full" else []
        if r["question_type"] in MCA_QUESTION_TYPES:
            return "\n".join([head, r["question"], "Options:\n" + "\n".join(r["options"]), *h,
                              "Compute the answer step by step from the numbers above, then end with a final line "
                              "'Answer: <option letter>'."])
        return "\n".join([head, r["question"], *h,
                          "Compute the answer step by step from the numbers above, then end with a final line "
                          "'Answer: <number>' (only the number, in the unit the question asks for)."])
    head = ("These are frames of a video.\n" if args.input == "video_scene" else "") + \
        SCENE_INTRO + "\n<scene>\n" + r["scene_language"] + "\n</scene>"
    if r["question_type"] in MCA_QUESTION_TYPES:
        return "\n".join([head, r["question"], "Options:\n" + "\n".join(r["options"]), KW["mca_post_prompt"]])
    return "\n".join([head, r["question"], KW["na_post_prompt"]])


preds_path = out / f"predictions_shard{args.shard_index}of{args.num_shards}.jsonl"
done = read_rows(preds_path)
if args.merge:
    done = {}
    for f in sorted(out.glob("predictions*.jsonl")):
        done.update(read_rows(f))
    report(done)
    raise SystemExit
print(f"{len(rows)} rows, done {len(done)}\n--- example prompt ---\n{prompt(rows[0])}\n---", flush=True)

processor = AutoProcessor.from_pretrained(args.model)
model = AutoModelForImageTextToText.from_pretrained(
    args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter).to("cuda").eval()
if args.input == "video_scene":
    from qwen_vl_utils import process_vision_info
    from sr_vggt.data import LLM_MAX_PIXELS, LLM_MIN_PIXELS, decode_video_frames

memo = {}
with preds_path.open("a") as fout:
    order = sorted(rows, key=lambda r: (r["scene_name"], r["index"]))  # one decode per video
    n = len(order)
    order = order[n * args.shard_index // args.num_shards: n * (args.shard_index + 1) // args.num_shards]
    for k, r in enumerate(order):
        if r["index"] in done:
            continue
        content = [{"type": "text", "text": prompt(r)}]
        if args.prompt.startswith("rvs_"):  # sr_vggt.data.task_messages, --prompt plain: head text, video, human turn
            content.insert(0, {"type": "text", "text": "This is the 3D scene context.\n"})
        if args.input == "video_scene":
            if memo.get("key") != r["scene_name"]:
                memo = {"key": r["scene_name"], "frames": decode_video_frames(
                    f"{args.vsibench_dir}/{r.get('dataset', 'scannetpp')}/{r['scene_name']}.mp4", 1.0, 64)}
            content.insert(1 if args.prompt.startswith("rvs_") else 0, {"type": "video", "video": memo["frames"],
                               "min_pixels": LLM_MIN_PIXELS, "max_pixels": LLM_MAX_PIXELS})
        msgs = [{"role": "user", "content": content}]
        if args.system != "none":
            from sr_vggt.data import _SCENE_SYSTEM, _SHORT_SYSTEM
            sys_text = _SHORT_SYSTEM if args.system == "short" else _SCENE_SYSTEM
            msgs.insert(0, {"role": "system", "content": [{"type": "text", "text": sys_text}]})
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=args.thinking)
        if args.input == "video_scene":
            _, vision, vkw = process_vision_info(msgs, return_video_kwargs=True, image_patch_size=16,
                                                 return_video_metadata=True)
            videos, vmeta = map(list, zip(*vision))
            inputs = processor(text=text, videos=videos, video_metadata=vmeta, return_tensors="pt", **vkw)
        else:
            inputs = processor(text=text, return_tensors="pt")
        inputs = inputs.to("cuda")
        with torch.no_grad():
            # Stop at the end of the turn too: Qwen3.5's config.json stops on <|endoftext|> only, so a
            # fine-tuned model runs on past its <|im_end|> (sr_vggt/eval.py has the same fix).
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 temperature=None, top_p=None, top_k=None,
                                 eos_token_id=[processor.tokenizer.convert_tokens_to_ids(t)
                                               for t in ("<|im_end|>", "<|endoftext|>")])
        new = gen[0, inputs["input_ids"].shape[1]:]
        row = {"index": r["index"], "question_type": r["question_type"], "ground_truth": r["ground_truth"],
               "prediction": processor.tokenizer.decode(new, skip_special_tokens=True).strip(),
               "prompt_tokens": int(inputs["input_ids"].shape[1]), "gen_tokens": int(new.shape[0]),
               "thinking": args.thinking}
        fout.write(json.dumps(row) + "\n"); fout.flush()
        done[r["index"]] = row
        if k < 3 or k % 200 == 0:
            print(k, row, flush=True)
report(done)
