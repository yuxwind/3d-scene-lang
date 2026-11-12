import sys, json, numpy as np
from collections import Counter
src = open('check_frames3.py').read().split("rows = []")[0]; exec(src)
rows = []
for s, tb in rb.items():
    g = G[s]; f0 = fit(tb, g, 0)
    if not f0 or f0[3] > 0.1: continue
    k = min(range(len(g['P'])), key=lambda k: fit(tb, g, k)[1]) if f0[1] > 0.05 else 0
    fk = fit(tb, g, k); th = np.radians(f0[0]); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Tk = g['A'] @ g['P'][k]; fwd = R @ Tk[:2, 2]; rgt = R @ Tk[:2, 0]
    rows.append((s, k, len(g['P']), fk[1], np.degrees(np.arctan2(fwd[1], fwd[0])), np.degrees(np.arctan2(rgt[1], rgt[0])), f0[0]))
a = np.array([r[1:] for r in rows], float); ok = a[:, 2] < 0.05
print('scenes', len(a), '| origin matches some posed image within 5 cm:', ok.sum())
fw = a[ok, 3]; rg = a[ok, 4]
print('forward of the ORIGIN camera in trace frame: within +-45 deg of +y: %.1f%%, +x: %.1f%%, -y: %.1f%%, -x: %.1f%%' % tuple(np.mean(np.abs((fw - c + 180) % 360 - 180) <= 45) * 100 for c in (90, 0, -90, 180)))
print('right   of the ORIGIN camera in trace frame: within +-45 deg of +x: %.1f%%' % (np.mean(np.abs((rg + 180) % 360 - 180) <= 45) * 100))
print('origin image index / number of posed images, histogram of the fraction:', np.histogram(a[ok, 0] / a[ok, 1], bins=10, range=(0, 1))[0])
json.dump([dict(scene=r[0], k=int(r[1]), n=int(r[2]), off=float(r[3]), yaw=float(r[6])) for r in rows], open('reason190k_scannet_origin.json', 'w'))
