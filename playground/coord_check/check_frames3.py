import json, pickle, numpy as np, sys
from collections import Counter
sys.argv = ['x']; exec(open('check_frames2.py').read().split("G = {**load")[0])
G = {**load('scannet_infos_train_pts.pkl')}
for s, g in load('scannet_infos_train_es56.pkl').items(): G[s] = g
import gzip, re
from collections import defaultdict
tr = defaultdict(dict); nsrc = defaultdict(Counter)
with gzip.open('../../data/simplesr_train_traces.jsonl.gz', 'rt') as f:
    for l in f:
        r = json.loads(l)
        if r['meta']['dataset'] != 'scannet': continue
        nsrc[r['meta']['scene']][r['source']] += 1
        m = re.search(r'<scene>\n(.*?)\n</scene>', r['conversations'][1]['value'], re.S)
        for o in json.loads(m.group(1)): tr[r['meta']['scene']][(base(o['label']), tuple(o['bbox_3d']))] = 1
rb = {s: [(l, np.array(b[:3]), np.array(b[3:6])) for (l, b) in v] for s, v in tr.items()}
rows = []
for s, tb in rb.items():
    g = G[s]; f0 = fit(tb, g, 0)
    if not f0 or f0[3] > 0.1: continue
    th = np.radians(f0[0]); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    T0 = g['A'] @ g['P'][0]
    fwd = R @ T0[:2, 2]; right = R @ T0[:2, 0]                    # camera-0 optical axis / right axis, in trace xy
    # which posed image's camera is the trace origin? origin in aligned+cam0 coords = -R^T t ; search all poses
    best = min(range(len(g['P'])), key=lambda k: fit(tb, g, k)[1]) if f0[1] > 0.05 else 0
    fb = fit(tb, g, best)
    rows.append((s, f0[0], np.degrees(np.arctan2(fwd[1], fwd[0])), np.degrees(np.arctan2(right[1], right[0])), f0[1], best, fb[1], fb[2], len(g['P'])))
a = np.array([r[1:] for r in rows], float)
print('scenes', len(a))
h = lambda v: dict(sorted(Counter((np.round(((v + 180) % 360 - 180) / 45) * 45).astype(int)).items()))
print('camera-0 forward direction in the trace frame (deg from +x, 45-deg bins):', h(a[:, 1]))
print('camera-0 right   direction in the trace frame (deg from +x, 45-deg bins):', h(a[:, 2]))
fw = (a[:, 1] + 180) % 360 - 180
print('forward within +-45 deg of +y: %.1f%%' % (np.mean(np.abs(fw - 90) <= 45) * 100), '| within +-50: %.1f%%' % (np.mean(np.abs(fw - 90) <= 50) * 100))
off = a[:, 3]
print('origin = camera of posed image 00000 within 5 cm: %.1f%% of scenes; within 20 cm: %.1f%%' % (np.mean(off < 0.05) * 100, np.mean(off < 0.2) * 100))
bad = a[off >= 0.05]
print('other scenes (%d): best-matching posed image index (of the every-2nd/-Nth frames kept):' % len(bad), Counter(bad[:, 4].astype(int)).most_common(8))
print('   their offset to that best image: median %.3f m, 90th pct %.3f m; z median %+.3f' % (np.median(bad[:, 5]), np.percentile(bad[:, 5], 90), np.median(bad[:, 6])))
for r in [r for r in rows if r[4] >= 0.05][:8]: print('   ', r[0], 'offset to image0 %.2f m, best image %d/%d offset %.3f m' % (r[4], r[5], r[8], r[6]), dict(nsrc[r[0]]))
