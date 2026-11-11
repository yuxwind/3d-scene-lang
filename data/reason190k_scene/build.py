"""Reason-190k as a scene -> answer mix (no video): data/reason190k_videoscene's rows with the video removed.

Same rows, scene blocks (question objects, upright [x, y, z, w, d, h], origin at the first sampled frame's camera,
z up, shuffled) and bare-answer targets as reason190k_videoscene_answer.jsonl.gz. Dropped: the row's `video` key
(sr_vggt.data then builds a text-only turn), the "These are frames of a video." line, and the "<image>\n" prefix
that 92,688 rows carry from the 190k mix.
usage: python build.py <reason190k_videoscene_answer.jsonl.gz> <out.jsonl.gz>
"""
import gzip, json, sys

VIDEO_LINE = "These are frames of a video.\n"


def main(src, dst):
    n = 0
    with gzip.open(src, "rt") as f, gzip.open(dst, "wt") as out:
        for line in f:
            r = json.loads(line)
            h = r["conversations"][0]["value"]
            assert h.count(VIDEO_LINE) == 1
            h = h.replace(VIDEO_LINE, "")
            h = h[len("<image>\n"):] if h.startswith("<image>\n") else h
            assert "<image>" not in h and "<video>" not in h
            r["conversations"][0]["value"] = h
            r.pop("video")
            out.write(json.dumps(r) + "\n")
            n += 1
    print(f"wrote {n} rows to {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:])
