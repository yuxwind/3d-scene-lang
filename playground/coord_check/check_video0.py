"""Which posed image is frame 0 of the mp4? (image similarity of mp4 frame 0 vs every posed image of the scan)"""
import json, pickle, sys, numpy as np
from PIL import Image
import decord
D = '/scr/xinyu/projects/vggtdet_es56/data/scannet/'
infos = {}
for f in ('scannet_infos_train_pts.pkl', 'scannet_infos_val_pts.pkl'):
    for x in pickle.load(open(D + f, 'rb'))['data_list']: infos[x['lidar_points']['lidar_path'][:-4]] = x
def small(im): 
    a = np.asarray(im.convert('L').resize((64, 48)), np.float32); return (a - a.mean()) / (a.std() + 1e-6)
def best(frame, paths):
    q = small(Image.fromarray(frame)); sc = [float((q * small(Image.open(D + p))).mean()) for p in paths]
    k = int(np.argmax(sc)); return k, sc[k], sc[0]
root, scenes = sys.argv[1], sys.argv[2:]
for s in scenes:
    vr = decord.VideoReader(f'{root}/scannet/{s}.mp4'); n, fps = len(vr), vr.get_avg_fps()
    paths = infos[s]['img_paths']; k, sk, s0 = best(vr[0].asnumpy(), paths)
    kl, skl, _ = best(vr[n - 1].asnumpy(), paths)
    print(f'{s}: mp4 {n} frames @ {fps:.1f} fps ({n/fps:.0f} s), {len(paths)} posed images | mp4 frame 0 ~ posed image {k} (corr {sk:.2f}; corr with image 0: {s0:.2f}) | mp4 last frame ~ posed image {kl} (corr {skl:.2f})')
