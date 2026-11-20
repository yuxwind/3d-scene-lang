"""Reason-190k as a video + GT scene -> CoT + answer mix in Qwen3.5's native thinking layout.

Input: ../reason190k_videoscene/build.py's human turn: the trace's <scene> list (only the objects the question
needs, [x, y, z, w, d, h], origin at the first sampled frame's camera, z up, shuffled by row) inserted right after
the "These are frames of a video." line.
Target: the trace's reasoning inside one <think> block, then the answer:
    <think>\n{reasoning}\n</think>\n\nAnswer: X
The trace's opener ("Let me look at the 3D scene ...") and its <scene> list are left out of the target: the scene
is given. Trained and evaluated with thinking on (--prompt reason_scene): the generation prompt ends in "<think>\n".
usage: python build.py <simplesr_train_traces.jsonl.gz> <out.jsonl.gz>
"""
import gzip, json, random, re, sys

VIDEO_LINE = "These are frames of a video.\n"
SCENE_DESC = ("This is the 3D scene: the objects the question is about, one {\"label\", \"bbox_3d\"} per line. "
              "bbox_3d = [x, y, z, w, d, h]: the centre and full size of an upright, axis-aligned box, in metres; "
              "origin at the first frame's camera, z points up.")
TRACE = re.compile(r"(.*?)\n<scene>\n(\[.*?\])\n</scene>\n<think>\n(.*?)\n</think>\n(Answer: .*)", re.S)


def scene_block(objs_json: str, seed: int) -> str:
    objs = json.loads(objs_json)
    random.Random(seed).shuffle(objs)  # the traces list objects in order of first appearance
    return "<scene>\n[\n" + ",\n".join(json.dumps(o) for o in objs) + "\n]\n</scene>\n"


def main(src, dst):
    n = 0
    with gzip.open(src, "rt") as f, gzip.open(dst, "wt") as out:
        for line in f:
            r = json.loads(line)
            human = r["conversations"][0]["value"]
            assert human.count(VIDEO_LINE) == 1
            m = TRACE.fullmatch(r["conversations"][1]["value"].strip())
            assert m, r["meta"]["i"]
            _opener, objs, reasoning, answer = (g.strip() for g in m.groups())
            scene = scene_block(objs, r["meta"]["i"])
            r["conversations"] = [
                {"from": "human", "value": human.replace(VIDEO_LINE, VIDEO_LINE + SCENE_DESC + "\n" + scene)},
                {"from": "gpt", "value": f"<think>\n{reasoning}\n</think>\n\n{answer}"}]
            out.write(json.dumps(r) + "\n")
            n += 1
    print(f"wrote {n} rows to {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:])
