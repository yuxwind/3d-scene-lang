"""Reason-190k as a video -> scene + CoT + answer mix in Qwen3.5's native thinking layout.

Input: the trace's human turn as-is (video + question; no scene).
Target: the trace's opener, <scene> list and reasoning moved inside one <think> block, then the answer:
    <think>\n{opener}\n<scene>\n[...]\n</scene>\n{reasoning}\n</think>\n\nAnswer: X
Qwen3.5's chat template keeps only the text between <think> and </think> as the reasoning and drops anything
before <think>, so the traces' "opener <scene> <think>reasoning</think> Answer" layout would lose the scene.
Trained and evaluated with thinking on (--prompt reason_plain): the generation prompt ends in "<think>\n".
usage: python build.py <simplesr_train_traces.jsonl.gz> <out.jsonl.gz>
"""
import gzip, json, re, sys

TRACE = re.compile(r"(.*?)\n(<scene>\n.*?\n</scene>)\n<think>\n(.*?)\n</think>\n(Answer: .*)", re.S)


def main(src, dst):
    n = 0
    with gzip.open(src, "rt") as f, gzip.open(dst, "wt") as out:
        for line in f:
            r = json.loads(line)
            m = TRACE.fullmatch(r["conversations"][1]["value"].strip())
            assert m, r["meta"]["i"]
            opener, scene, reasoning, answer = (g.strip() for g in m.groups())
            r["conversations"][1]["value"] = f"<think>\n{opener}\n{scene}\n{reasoning}\n</think>\n\n{answer}"
            out.write(json.dumps(r) + "\n")
            n += 1
    print(f"wrote {n} rows to {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:])
