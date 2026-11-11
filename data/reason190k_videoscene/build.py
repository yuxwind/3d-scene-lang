"""Reason-190k as a video + scene -> answer mix.

Input: the trace's <scene> list (only the objects the question needs, [x, y, z, w, d, h], origin at the first
sampled frame's camera, z up) inserted into the human turn right after the "These are frames of a video." line.
The objects are shuffled (seeded by row): the traces list them in order of first appearance in the video,
which would give away every appearance-order answer.
Target: the bare answer of the source 190k row (meta.answer), no opener, no scene copy, no <think>.
usage: python build.py <simplesr_train_traces.jsonl.gz> <out.jsonl.gz>
"""
import gzip, json, random, re, sys

VIDEO_LINE = "These are frames of a video.\n"
SCENE_DESC = ("This is the 3D scene: the objects the question is about, one {\"label\", \"bbox_3d\"} per line. "
              "bbox_3d = [x, y, z, w, d, h]: the centre and full size of an upright, axis-aligned box, in metres; "
              "origin at the first frame's camera, z points up.")
SCENE = re.compile(r"<scene>\n(\[.*?\])\n</scene>", re.S)


def scene_block(objs_json: str, seed: int) -> str:
    objs = json.loads(objs_json)
    random.Random(seed).shuffle(objs)
    return "<scene>\n[\n" + ",\n".join(json.dumps(o) for o in objs) + "\n]\n</scene>\n"


def main(src, dst):
    n = 0
    with gzip.open(src, "rt") as f, gzip.open(dst, "wt") as out:
        for line in f:
            r = json.loads(line)
            human, gpt = r["conversations"][0]["value"], r["conversations"][1]["value"]
            assert human.count(VIDEO_LINE) == 1  # 92,688 rows start with "<image>\n" (kept, as in the 190k mix)
            scene = scene_block(SCENE.search(gpt).group(1), r["meta"]["i"])
            r["conversations"] = [
                {"from": "human", "value": human.replace(VIDEO_LINE, VIDEO_LINE + SCENE_DESC + "\n" + scene)},
                {"from": "gpt", "value": r["meta"]["answer"]}]
            out.write(json.dumps(r) + "\n")
            n += 1
    print(f"wrote {n} rows to {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:])
