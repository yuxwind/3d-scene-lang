#!/usr/bin/env python3
"""Batches for Claude (no code) on VSI-Bench with any box list, as vggtdet_es56/runs/claude_nocode_s3/build_est.py (variant
v2/e2: score-aware prompt). Rows: a VSI-Bench rows file (all 5,130; scene_language = the boxes of the objects the question
names, 6-number upright boxes with "score"), e.g. rows_s3q_rr_oraclecnt_all5130_full.jsonl.
Batches: one question type each, at most one question per scene, at most 25 questions (12 rel. distance, 10 room size,
15 route planning) and 90k characters. No ground truth in the batch files. Answer with RULES.md (copied into OUT).
    python3 claude_batches.py ROWS.jsonl OUT_DIR [--rules RULES.md]   -> OUT_DIR/batches/*.json, OUT_DIR/out/, manifest.json
Score: scripts/analysis/vsibench_score_answers.py <parent of OUT_DIR> --variants <name of OUT_DIR> --gt ROWS.jsonl"""
import argparse, collections, json, os, shutil
MCA = {'object_rel_direction_easy', 'object_rel_direction_medium', 'object_rel_direction_hard', 'object_rel_distance', 'route_planning',
       'obj_appearance_order'}
BASE = """This is a 3D scene, given as a list of objects, one per line: {"label": <category>, "bbox_3d": [x, y, z, dx, dy, dz], "score": <number>}.
- x, y, z: the centre of the object's 3D bounding box, in meters. x and y are horizontal coordinates on the floor plane; z is the vertical axis (height, pointing up). z = 0 is not the floor: the floor is at about the lowest box bottom (z - dz/2) in the scene.
- dx, dy, dz: the full size of the box in meters; the box is upright and axis-aligned: dx runs along x, dy along y, and dz is the vertical height of the object.
- Lines with the same label are different instances of that category. All lengths are in meters."""
V2 = """
- These boxes were not measured: a 3D detector (VGGT-SLAM reconstruction + SAM 3 segmentation) predicted them from the video. Some objects may be missing, duplicated, mislabelled or misplaced, and sizes can be off.
- score (0 to 1) is the detector's confidence in the box. Trust boxes with a high score; treat low-score boxes with caution, and use your knowledge of typical indoor scenes where the boxes are missing or unreliable."""
CAP = {'object_rel_distance': 12, 'room_size_estimation': 10, 'route_planning': 15}


def text(r):
    t = BASE + V2 + '\n<scene>\n' + '\n'.join(json.dumps(o) for o in json.loads(r['scene_language'] or '[]')) + '\n</scene>\n' + r['question']
    if r['question_type'] in MCA:
        return t + '\nOptions:\n' + '\n'.join(r['options']) + '\nAnswer with the option letter.'
    return t + '\nAnswer with a number only, in the unit the question asks for.'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('rows'); ap.add_argument('out')
    ap.add_argument('--rules', default='/scr/xinyu/projects/vggtdet_es56/runs/claude_nocode_s3/RULES.md'); a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.rows)]
    by = collections.defaultdict(list)
    for r in rows:
        by[r['question_type'].replace('_easy', '').replace('_medium', '').replace('_hard', '')].append(r)
    batches = []
    for t, R in sorted(by.items()):
        open_b = []
        for r in sorted(R, key=lambda r: r['index']):
            n = len(text(r))
            for b in open_b:
                if r['scene_name'] not in b['scenes'] and len(b['rows']) < CAP.get(t, 25) and b['chars'] + n < 90000:
                    b['rows'].append(r); b['scenes'].add(r['scene_name']); b['chars'] += n; break
            else:
                open_b.append(dict(type=t, rows=[r], scenes={r['scene_name']}, chars=n))
        batches += open_b
    os.makedirs(f'{a.out}/batches', exist_ok=True); os.makedirs(f'{a.out}/out', exist_ok=True)
    man = []
    for i, b in enumerate(batches):
        name = f'b{i:03d}_{b["type"]}'
        json.dump({'batch': name, 'questions': [{'id': r['index'], 'text': text(r)} for r in b['rows']]},
                  open(f'{a.out}/batches/{name}.json', 'w'), indent=1)
        man.append(dict(batch=name, type=b['type'], n=len(b['rows']), chars=b['chars']))
    json.dump(dict(rows=os.path.abspath(a.rows), batches=man), open(f'{a.out}/manifest.json', 'w'), indent=0)
    shutil.copy(a.rules, f'{a.out}/RULES.md')
    print(len(batches), 'batches;', dict(collections.Counter(m['type'] for m in man)), '; questions', sum(m['n'] for m in man),
          '; max chars', max(m['chars'] for m in man))


if __name__ == '__main__':
    main()
