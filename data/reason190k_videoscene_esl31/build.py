"""Reason-190k as video + partial GT scene -> answer: the scene a perfect 31-class detector (ESL31) would give.

../reason190k_videoscene/build.py with one change: each row's <scene> list keeps only the trace objects whose
label (number stripped) is one of the detector's 31 classes or a synonym of one (KEEP: the 58 labels of
docs/vsibench_esl31_label_mapping.txt, section 5). Kept objects keep the trace's label, number, box and the
shuffled order they have in the full-scene mix; a row whose objects are all dropped keeps an empty list, so every
one of the 145,647 rows stays. Target: the bare answer.
usage: python build.py <simplesr_train_traces.jsonl.gz> <out.jsonl.gz>
"""
import gzip, json, random, re, sys
from collections import Counter

VIDEO_LINE = "These are frames of a video.\n"
SCENE_DESC = ("This is the 3D scene: the objects the question is about, one {\"label\", \"bbox_3d\"} per line. "
              "bbox_3d = [x, y, z, w, d, h]: the centre and full size of an upright, axis-aligned box, in metres; "
              "origin at the first frame's camera, z points up.")
SCENE = re.compile(r"<scene>\n(\[.*?\])\n</scene>", re.S)
ESL31 = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
         'curtain', 'refrigerator', 'toilet', 'sink', 'bathtub', 'trash bin', 'pillow', 'monitor', 'backpack',
         'printer', 'lamp', 'mirror', 'tv', 'radiator', 'nightstand', 'plant', 'microwave', 'fan', 'blackboard', 'stool']
SYN = {'heater': 'radiator', 'trash can': 'trash bin', 'board': 'blackboard', 'shelf': 'bookshelf',
       'whiteboard': 'blackboard', 'table lamp': 'lamp', 'poster': 'picture', 'office chair': 'chair',
       'storage cabinet': 'cabinet', 'exhaust fan': 'fan', 'painting': 'picture', 'couch': 'sofa', 'bin': 'trash bin',
       'kitchen cabinet': 'cabinet', 'tv_monitor': 'tv', 'build_in_cabinet': 'cabinet', 'countertop': 'counter',
       'armchair': 'chair', 'bathroom cabinet': 'cabinet', 'shower curtain': 'curtain', 'basin': 'sink',
       'end table': 'table', 'coffee table': 'table', 'washbasin': 'sink', 'kitchen counter': 'counter',
       'dining table': 'table', 'file cabinet': 'cabinet'}
KEEP = set(ESL31) | set(SYN)


def main(src, dst):
    kept, dropped, rows = Counter(), Counter(), Counter()
    with gzip.open(src, "rt") as f, gzip.open(dst, "wt") as out:
        for line in f:
            r = json.loads(line)
            human, gpt = r["conversations"][0]["value"], r["conversations"][1]["value"]
            assert human.count(VIDEO_LINE) == 1
            objs = json.loads(SCENE.search(gpt).group(1))
            random.Random(r["meta"]["i"]).shuffle(objs)  # the full-scene mix's order
            keep = []
            for o in objs:
                lab = re.sub(r"\s+\d+$", "", o["label"])
                (kept if lab in KEEP else dropped)[lab] += 1
                if lab in KEEP: keep.append(o)
            rows["all kept" if len(keep) == len(objs) else "empty" if not keep else "partial"] += 1
            r["meta"]["scene_objs"], r["meta"]["scene_objs_kept"] = len(objs), len(keep)
            scene = "<scene>\n[\n" + ",\n".join(json.dumps(o) for o in keep) + "\n]\n</scene>\n"
            r["conversations"] = [
                {"from": "human", "value": human.replace(VIDEO_LINE, VIDEO_LINE + SCENE_DESC + "\n" + scene)},
                {"from": "gpt", "value": r["meta"]["answer"]}]
            out.write(json.dumps(r) + "\n")
    nk, nd = sum(kept.values()), sum(dropped.values())
    print(f"wrote {sum(rows.values())} rows to {dst}: {dict(rows)}")
    print(f"objects kept {nk} in {len(kept)} labels ({100 * nk / (nk + nd):.1f}%), dropped {nd} in {len(dropped)} labels")


if __name__ == "__main__":
    main(*sys.argv[1:])
