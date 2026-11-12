"""Pose of mp4 frame 0 for ScanNet videos: match the first mp4 frame to the scan's posed images (grey 64x48,
normalised correlation) and take that image's pose. Then the trace-frame rule: origin = that camera, axes = the
scan's aligned axes turned by k * 90 deg so that the camera's forward is closest to +y.
    c_trace = Rz(k * 90 deg) @ c_det + t
    python match_frame0.py <video root with scannet/*.mp4> <out.json> [scene ...]   (default: every mp4 found)
"""
import json, os, pickle, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image
D = '/scr/xinyu/projects/vggtdet_es56/data/scannet/'
INFOS = {}


def small(im):
    a = np.asarray(im.convert('L').resize((64, 48)), np.float32); return (a - a.mean()) / (a.std() + 1e-6)


def one(args):
    root, s = args
    import decord
    x = INFOS[s]; q = small(Image.fromarray(decord.VideoReader(f'{root}/scannet/{s}.mp4')[0].asnumpy()))
    sc = [float((q * small(Image.open(D + p))).mean()) for p in x['img_paths']]
    i = int(np.argmax(sc)); T = np.array(x['axis_align_matrix'], float) @ np.array(x['lidar2cam'][i], float)
    if not np.isfinite(T).all(): return s, None
    k = int(np.round((np.pi / 2 - np.arctan2(T[1, 2], T[0, 2])) / (np.pi / 2))) % 4      # turn forward onto +y
    a = k * np.pi / 2; R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]).round()
    fwd = R[:2, :2] @ T[:2, 2]; dev = float(np.degrees(np.arctan2(fwd[0], fwd[1])))   # camera heading off +y, in (-45, 45]
    return s, dict(k=k, t=[float(v) for v in -R @ T[:3, 3]], image=i, n_images=len(sc), corr=sc[i], corr0=sc[0], heading_dev=dev)


if __name__ == '__main__':
    root, dst, scenes = sys.argv[1], sys.argv[2], sys.argv[3:]
    for f in ('scannet_infos_train_pts.pkl', 'scannet_infos_val_pts.pkl'):
        for x in pickle.load(open(D + f, 'rb'))['data_list']: INFOS[x['lidar_points']['lidar_path'][:-4]] = x
    scenes = scenes or sorted(f[:-4] for f in os.listdir(f'{root}/scannet') if f[:-4] in INFOS)
    with Pool(48) as p: res = dict(p.map(one, [(root, s) for s in scenes], chunksize=4))
    json.dump({s: v for s, v in res.items() if v}, open(dst, 'w'))
    print(len(res), 'scenes,', sum(v is None for v in res.values()), 'failed')
