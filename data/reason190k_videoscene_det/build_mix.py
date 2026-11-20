"""Reason-190K as video + scene -> answer, with the scene taken from ground truth or from VGG-T-Det (ESL31).

Same row format as ../reason190k_videoscene/build.py. For every row whose scene has detector boxes, the scene is
the detector's with probability P_DET (seeded by row), else the trace's ground truth. Rows of scenes without
detector boxes keep the ground truth.

Detector scene of a row:
  - the labels the trace lists (numbers stripped) are the objects the question needs; each is mapped to detector
    classes (SYN) and gets every detector box of those classes with score >= the class threshold, under the
    trace's label, numbered when there are several ("chair 1", "chair 2");
  - room-size rows, and rows where no needed label has a box, get every detector box over threshold (detector
    class names), as eval_scenelang.py --prompt rvs_question does at test time;
  - boxes are [x, y, z, w, d, h] in the trace frame (origin at the first frame's camera, room axes, z up):
    ScanNet boxes are moved from the scan's aligned frame with (k, t) from the box fit when it is reliable
    (>= 8 inliers, share >= 0.8), else from the image-matching fit; ARKitScenes boxes are already in that frame.
usage: python build_mix.py <simplesr_train_traces.jsonl.gz> <out.jsonl.gz> [--p_det 0.5] [--datasets scannet,arkitscenes]

--cover: no ground truth at all. A row whose needed labels are fully or partially in the detector's classes (SYN) and
whose scene has detector boxes gets the detector scene (only the covered labels can get boxes); every other row
(no needed label covered, room size, or no detector output for the scene) is video only: the human turn is left
without scene text. meta.scene_src = det | video, meta.cover = full | partial | zero | none.
--video_only_types a,b,c: with --cover, rows of these question types are video only whatever their coverage
(mix-v2: appearance_order,route_planning,absolute_size_room).
ScanNet++ (det_scannetpp_seed0.json, boxes already in the trace frame) uses the ScanNet thresholds.
--p_gt P: with --cover, a row that would get the detector scene gets instead, with probability P (seeded by row), the
ground-truth boxes of its covered labels only (score 1.0) - what a perfect 31-class detector would list.
--prompt_v2: scene rows end the scene description with SCORE_DESC_V2 (trust each box according to its score),
video-only rows say NO_SCENE (answer from the video alone). Implies a score per box.
mix-v2 = --cover --prompt_v2 --p_gt 0.5 --video_only_types appearance_order,route_planning,absolute_size_room
         --datasets scannet,arkitscenes,scannetpp
"""
import argparse, gzip, json, random, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent
VIDEO_LINE = "These are frames of a video.\n"
SCENE_DESC = ("This is the 3D scene: the objects the question is about, one {\"label\", \"bbox_3d\"} per line. "
              "bbox_3d = [x, y, z, w, d, h]: the centre and full size of an upright, axis-aligned box, in metres; "
              "origin at the first frame's camera, z points up.")
SCORE_DESC = " score: the detector's confidence in the box, 1.0 for a ground-truth box."
SCORE_DESC_V2 = (" score: the detector's confidence in the box, 1.0 for a ground-truth box. Trust each box according to its score.")
NO_SCENE = "No 3D scene is given: answer from the video alone.\n"
SCENE = re.compile(r"<scene>\n(\[.*?\])\n</scene>", re.S)
ESL31 = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
         'curtain', 'refrigerator', 'toilet', 'sink', 'bathtub', 'trash bin', 'pillow', 'monitor', 'backpack',
         'printer', 'lamp', 'mirror', 'tv', 'radiator', 'nightstand', 'plant', 'microwave', 'fan', 'blackboard', 'stool']
SYN = {**{c: [c] for c in ESL31}, 'table': ['table', 'desk'], 'desk': ['desk', 'table'],
       'trash can': ['trash bin'], 'bin': ['trash bin'], 'heater': ['radiator'], 'couch': ['sofa'], 'shelf': ['bookshelf'],
       'whiteboard': ['blackboard'], 'board': ['blackboard'], 'table lamp': ['lamp'], 'exhaust fan': ['fan'],
       'office chair': ['chair'], 'tv_monitor': ['tv'], 'storage cabinet': ['cabinet'], 'kitchen cabinet': ['cabinet'],
       'painting': ['picture'], 'poster': ['picture'], 'armchair': ['chair'], 'shower curtain': ['curtain'],
       'coffee table': ['table'], 'end table': ['table'], 'dining table': ['table'], 'bathroom cabinet': ['cabinet'],
       'file cabinet': ['cabinet'], 'build_in_cabinet': ['cabinet'], 'kitchen counter': ['counter'], 'countertop': ['counter'],
       'basin': ['sink'], 'washbasin': ['sink']}  # full table: docs/vsibench_esl31_label_mapping.txt
# score thresholds (best F1 at IoU 0.25); ScanNet: tuned on the 88 VSI-Bench ScanNet scenes; classes without GT there: 0.5
THR = {'scannet': {'cabinet': .50, 'bed': .20, 'chair': .60, 'sofa': .40, 'table': .55, 'door': .40, 'window': .30, 'bookshelf': .65,
                   'picture': .35, 'counter': .40, 'desk': .20, 'curtain': .35, 'refrigerator': .45, 'toilet': .50, 'sink': .55,
                   'bathtub': .50, 'trash bin': .85, 'pillow': .90, 'monitor': .95, 'backpack': .95, 'printer': .95, 'lamp': .75,
                   'mirror': .80, 'tv': .25, 'radiator': .75, 'nightstand': .20, 'plant': .95, 'microwave': .95, 'fan': .95}}


def load_det(ds, thr_file=None, tag='seed0'):
    """-> {scene: [(class, [x, y, z, w, d, h] in the trace frame, score)]} over threshold, best score first"""
    f = D / f'det_{ds}_{tag}.json'
    if not f.exists(): return {}
    # a threshold file (ARKitScenes: tune_thr_arkit.py, its 12 annotated classes) overrides the ScanNet values
    thr = {**THR['scannet'], **json.load(open(thr_file))} if thr_file else THR.get(ds, THR['scannet'])
    if tag != 'seed0': thr = {}                    # objq: the boxes are already selected (SAM 3 box score >= 0.7); no per-class thresholds
    det, out = json.load(open(f)), {}
    if ds == 'scannet':
        fit, f0 = json.load(open(D / 'frames_scannet.json')), json.load(open(D / 'frame0_scannet.json'))
    for s, v in det.items():
        b = np.array(v['boxes'], float).reshape(-1, 6)
        if ds == 'scannet':
            a = fit.get(s)
            a = a if a and a['inliers'] >= 8 and a['inliers'] / a['trace_boxes'] >= 0.8 else f0.get(s)
            if a is None: continue
            k = a['k']; R = np.round([[np.cos(k * np.pi / 2), -np.sin(k * np.pi / 2), 0], [np.sin(k * np.pi / 2), np.cos(k * np.pi / 2), 0], [0, 0, 1]])
            b = np.hstack([b[:, :3] @ R.T + np.array(a['t']), b[:, [4, 3, 5]] if k % 2 else b[:, 3:]])
        keep = [(l, [round(float(x), 2) for x in bb], sc) for l, bb, sc in zip(v['labels'], b, v['scores']) if sc >= thr.get(l, 0.5 if thr else 0.0)]
        out[s] = sorted(keep, key=lambda x: -x[2])
    return out


def syn65():
    """trace label -> detector classes for the 65-class detector: the class names, the VSI-Bench wordings, the dataset names the
    label map folds into a class (ScanNet++, EmbodiedScan, ARKitScenes 3DOD), and the 31-class synonyms above; table and desk
    stay interchangeable as in SYN."""
    lm = json.load(open('/scr/xinyu/projects/vggtdet_es56/code/tools/dataset_converters/vsi65_label_map.json')); CL = set(lm['target_classes'])
    out = {c: [c] for c in CL}
    for k in ('vsibench_to_target', 'scannetpp_to_target', 'es_name_to_target', 'arkit_to_target'):
        for a, c in lm.get(k, {}).items():
            if c in CL: out.setdefault(a, [c])
    for a, cs in SYN.items(): out[a] = [c for c in cs if c in CL] or out.get(a, [])
    return {a: cs for a, cs in out.items() if cs}


def numbered(items):  # [(label, box, score)] -> [{"label", "bbox_3d", "score"}], labels numbered when repeated
    n, seen, out = Counter(l for l, _, _ in items), Counter(), []
    for l, b, sc in items:
        if n[l] > 1: seen[l] += 1; l = f'{l} {seen[l]}'
        out.append({'label': l, 'bbox_3d': b, 'score': round(sc, 2)})
    return out


def det_scene(gt_objs, boxes, room, syn=None):
    syn = syn or SYN
    items, how = [], 'all'
    if not room:
        how = 'named'
        for lab in dict.fromkeys(re.sub(r'\s+\d+$', '', o['label']) for o in gt_objs):
            items += [(lab, b, sc) for c, b, sc in boxes if c in syn.get(lab, ())]
    if not items:
        items, how = [(c, b, sc) for c, b, sc in boxes], ('all' if room else 'fallback_all')
    return numbered(items), how


def block(objs, seed):
    random.Random(seed).shuffle(objs)
    return "<scene>\n[\n" + ",\n".join(json.dumps(o) for o in objs) + "\n]\n</scene>\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('--p_det', type=float, default=0.5)
    ap.add_argument('--datasets', default='scannet,arkitscenes'); ap.add_argument('--thr_arkitscenes')
    ap.add_argument('--score', action='store_true', help='write a "score" per box: detector confidence, 1.0 for ground truth'); ap.add_argument('--cover', action='store_true', help='detector scene on fully / partially covered rows, video only elsewhere; no GT')
    ap.add_argument('--video_only_types', default='', help='with --cover: question types that are always video only')
    ap.add_argument('--p_gt', type=float, default=0.0, help='with --cover: share of scene rows that get the GT boxes of the covered labels (score 1.0)')
    ap.add_argument('--prompt_v2', action='store_true')
    ap.add_argument('--det_tag', default='seed0', help='det_<ds>_<tag>.json; objq = test2-hybrid (export_objq.py)')
    ap.add_argument('--classes65', action='store_true', help='coverage and label matching over the 65 classes (vsi65_label_map.json) instead of the 31')
    a = ap.parse_args(); a.score = a.score or a.prompt_v2; vo = set(filter(None, a.video_only_types.split(',')))
    det = {ds: load_det(ds, a.thr_arkitscenes if ds == 'arkitscenes' else None, a.det_tag) for ds in a.datasets.split(',')}
    syn = syn65() if a.classes65 else SYN
    st, nb = Counter(), defaultdict(list)
    with gzip.open(a.src, 'rt') as f, gzip.open(a.dst, 'wt') as out:
        for line in f:
            r = json.loads(line); m = r['meta']; ds = m['dataset']
            human, gpt = r['conversations'][0]['value'], r['conversations'][1]['value']
            assert human.count(VIDEO_LINE) == 1
            objs = json.loads(SCENE.search(gpt).group(1)); src = 'gt'
            boxes = det.get(ds, {}).get(m['scene'])
            if a.cover:
                labs = {re.sub(r'\s+\d+$', '', o['label']) for o in objs}; ncov = sum(l in syn for l in labs)
                room = r['question_type'] == 'absolute_size_room'
                m['cover'] = 'none' if room or not labs else 'full' if ncov == len(labs) else 'partial' if ncov else 'zero'
                if m['cover'] in ('full', 'partial') and boxes is not None and r['question_type'] not in vo:
                    n_gt = len(objs)
                    if random.Random(2_000_003 * m['i'] + 29).random() < a.p_gt:
                        objs = [{**o, 'score': 1.0} for o in objs if re.sub(r'\s+\d+$', '', o['label']) in syn]; m['scene_src'] = 'gt'
                    else:
                        objs, how = det_scene(objs, boxes, False, syn); m['scene_src'] = 'det'
                        st[(ds, 'det:' + how)] += 1; nb[(ds, how)].append((n_gt, len(objs)))
                    objs = [{**o, 'score': o['score']} if a.score else {k: v for k, v in o.items() if k != 'score'} for o in objs]
                    desc = SCENE_DESC + (SCORE_DESC_V2 if a.prompt_v2 else SCORE_DESC if a.score else '')
                    human = human.replace(VIDEO_LINE, VIDEO_LINE + desc + '\n' + block(objs, m['i']))
                else:
                    m['scene_src'] = 'video'
                    if a.prompt_v2: human = human.replace(VIDEO_LINE, VIDEO_LINE + NO_SCENE)
                st[(ds, m['scene_src'] + '/' + m['cover'])] += 1
                r['conversations'] = [{'from': 'human', 'value': human}, {'from': 'gpt', 'value': m['answer']}]
                out.write(json.dumps(r) + '\n'); continue
            if boxes is not None and random.Random(1_000_003 * m['i'] + 17).random() < a.p_det:
                n_gt = len(objs)
                objs, how = det_scene(objs, boxes, r['question_type'] == 'absolute_size_room'); src = 'det'
                st[(ds, 'det:' + how)] += 1; nb[(ds, how)].append((n_gt, len(objs)))
            st[(ds, src)] += 1
            objs = [{**o, 'score': o.get('score', 1.0)} for o in objs] if a.score else [{k: v for k, v in o.items() if k != 'score'} for o in objs]
            m['scene_src'] = src
            r['conversations'] = [{'from': 'human', 'value': human.replace(VIDEO_LINE, VIDEO_LINE + SCENE_DESC + (SCORE_DESC if a.score else '') + '\n' + block(objs, m['i']))},
                                  {'from': 'gpt', 'value': m['answer']}]
            out.write(json.dumps(r) + '\n')
    print('rows:', {f'{k[0]}/{k[1]}': v for k, v in sorted(st.items())})
    for k, v in sorted(nb.items()):
        v = np.array(v); print(f'{k[0]} det rows ({k[1]}): {len(v)}; boxes per row GT median {np.median(v[:,0]):.0f} -> det median {np.median(v[:,1]):.0f} (mean {v[:,1].mean():.1f}, max {v[:,1].max()}); empty scenes {(v[:,1] == 0).sum()}')
    print({ds: f'{len(v)} scenes, {np.mean([len(x) for x in v.values()]):.1f} boxes/scene over threshold' for ds, v in det.items() if v})


if __name__ == '__main__':
    main()
