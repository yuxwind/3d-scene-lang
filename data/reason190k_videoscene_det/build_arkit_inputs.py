"""Detector inputs for the 2,830 ARKitScenes scenes of Reason-190K, in the ScanNet layout VGG-T-Det reads:
  <out>/posed_images/<scene>/NNNNN.jpg + .txt   up to 120 mp4 frames evenly spread over the trajectory, with the
                                                camera-to-world pose of the upright image (ARKit world frame)
  <out>/points/<scene>.bin                      mesh vertices (<= 200k), float32 x y z r g b
  <out>/arkit_infos_reason190k_esl31.pkl        infos; axis_align_matrix = ARKit world -> Reason-190K trace frame
                                                (box fit when reliable, else the room-yaw rule), so the detector's
                                                boxes come out directly in the trace frame. GT instances = the 3dod
                                                boxes of ESL31 classes, yaw dropped, in the trace frame.
mp4 frame i is the lowres_wide image at intrinsics timestamp 2i (60 Hz -> 30 fps); each trajectory pose (10 Hz)
takes the mp4 frame nearest in time. The first frame is the first trajectory pose = the trace origin.
    python build_arkit_inputs.py <scene list or 'all'> <n procs>
"""
import json, os, pickle, sys, numpy as np, decord
from PIL import Image
from multiprocessing import Pool
from arkit_common import traj, boxes, room_theta

ROOT = '/scr/xinyu/data/arkitscenes_reason190k/Training'; MP4 = '/scr/xinyu/data/VSI-590K/arkitscenes'
OUT = '/scr/xinyu/projects/vggtdet_es56/data/arkit_r190k'; N = 120
ESL31 = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
         'curtain', 'refrigerator', 'toilet', 'sink', 'bathtub', 'trash bin', 'pillow', 'monitor', 'backpack',
         'printer', 'lamp', 'mirror', 'tv', 'radiator', 'nightstand', 'plant', 'microwave', 'fan', 'blackboard', 'stool']
SYN = {'tv_monitor': 'tv', 'shelf': 'bookshelf'}
FIT = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frames_arkit.json')))


def ply_vertices(path):
    with open(path, 'rb') as f:
        n = None
        while True:
            l = f.readline().decode().strip()
            if l.startswith('element vertex'): n = int(l.split()[-1])
            if l == 'end_header': break
        v = np.frombuffer(f.read(n * 16), dtype=np.dtype([('p', '<f4', 3), ('c', 'u1', 4)]))
    return np.hstack([v['p'], v['c'][:, :3].astype(np.float32)])


def one(s):
    try:
        ts, c2w = traj(f'{ROOT}/{s}/lowres_wide.traj')
        idir = next(f'{ROOT}/{s}/{n}' for n in ('lowres_wide_intrinsics', 'vga_wide_intrinsics') if os.path.isdir(f'{ROOT}/{s}/{n}'))
        pin = np.array(sorted(float(f.split('_')[-1][:-7]) for f in os.listdir(idir)))
        K = np.loadtxt(f'{idir}/' + sorted(os.listdir(idir))[0])
        vr = decord.VideoReader(f'{MP4}/{s}.mp4'); H, W = vr[0].shape[:2]
        step = max(1, int(round(len(pin) / len(vr))))                  # intrinsics files per mp4 frame (2 for lowres_wide)
        # upright turn: quarter turns consistent with the mp4 shape, the one whose image-up is nearest world +z
        best = None
        for q in ((1, 3) if H > W else (0, 2)):
            a = q * np.pi / 2; R = np.eye(4); R[:2, :2] = np.round([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            up = -(c2w[:, :3, :3] @ R[:3, :3])[:, 2, 1].mean()
            if best is None or up > best[0]: best = (up, q, R)
        up, q, R = best; c2u = c2w @ R
        sel = np.unique(np.linspace(0, len(ts) - 1, min(N, len(ts))).round().astype(int))
        ft = pin[::step][:len(vr)]                                        # time of each mp4 frame
        fi = np.abs(ft[None] - ts[sel][:, None]).argmin(1)
        dt = np.abs(ft[fi] - ts[sel])
        d = f'{OUT}/posed_images/{s}'; os.makedirs(d, exist_ok=True)
        fr = vr.get_batch(list(fi)).asnumpy()
        for j in range(len(sel)):
            Image.fromarray(fr[j]).save(f"{d}/{j:05d}.jpg", quality=92)
            np.savetxt(f'{d}/{j:05d}.txt', c2u[sel[j]], fmt='%.6f')
        P = ply_vertices(f'{ROOT}/{s}/{s}_3dod_mesh.ply')
        if len(P) > 200000: P = P[np.random.RandomState(0).choice(len(P), 200000, replace=False)]
        os.makedirs(f'{OUT}/points', exist_ok=True); P.astype(np.float32).tofile(f'{OUT}/points/{s}.bin')
        C, L, A, lab = boxes(f'{ROOT}/{s}/{s}_3dod_annotation.json')
        v = FIT.get(s, {'fitted': False})
        if v['fitted'] and v['inliers'] >= 5 and v['inliers'] / v['trace_boxes'] >= 0.8:
            th, src = v['theta'], 'box_fit'
        else:
            th, src = float(room_theta(A, L, c2w[0, :2, 2])), 'rule'
        M = np.eye(4); M[:2, :2] = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]; M[:3, 3] = -M[:3, :3] @ c2w[0, :3, 3]
        inst = []
        for c, l, a, n in zip(C, L, A, lab):
            n = SYN.get(n, n)
            if n not in ESL31: continue
            yaw = np.arctan2(a[0, 1], a[0, 0]) + th; sw = abs(np.sin(yaw)) > abs(np.cos(yaw))
            ct = M[:3, :3] @ c + M[:3, 3]
            inst.append(dict(bbox_3d=[float(x) for x in (*ct, *(l[[1, 0, 2]] if sw else l))], bbox_label_3d=ESL31.index(n)))
        K4 = np.eye(4); K4[0, 0], K4[1, 1], K4[0, 2], K4[1, 2] = K[2] * W / K[0], K[3] * W / K[0], W / 2, H / 2
        info = dict(lidar_points=dict(num_pts_feats=6, lidar_path=f'{s}.bin'), instances=inst, cam2img=K4,
                    lidar2cam=[c2u[i] for i in sel], img_paths=[f'posed_images/{s}/{j:05d}.jpg' for j in range(len(sel))],
                    axis_align_matrix=M.tolist(), pts_semantic_mask_path=f'{s}.bin', pts_instance_mask_path=f'{s}.bin')
        stat = dict(scene=s, ok=True, n_img=len(sel), q=int(q), up=float(up), HW=[int(H), int(W)], dt_max=float(dt.max()), theta_src=src,
                    n_inst=len(inst), n_mp4=len(vr), n_pin=len(pin) / step * 2, step=step, avg_dist=float(np.linalg.norm(P[:, :3] - c2w[0, :3, 3], axis=1).mean()))
        return info, stat
    except Exception as e:
        return None, dict(scene=s, ok=False, err=repr(e))


if __name__ == '__main__':
    scenes = sorted(FIT) if sys.argv[1] == 'all' else [l.strip() for l in open(sys.argv[1]) if l.strip()]
    name = 'arkit_infos_reason190k_esl31' if sys.argv[1] == 'all' else 'arkit_infos_' + os.path.basename(sys.argv[1]).split('.')[0]
    with Pool(int(sys.argv[2])) as p: res = p.map(one, scenes, chunksize=4)
    infos = [i for i, _ in res if i]; stats = [s for _, s in res]
    meta = pickle.load(open('/scr/xinyu/projects/vggtdet_es56/data/scannet/scannet_infos_val_esl31.pkl', 'rb'))['metainfo']
    pickle.dump(dict(metainfo=meta, data_list=infos), open(f'{OUT}/{name}.pkl', 'wb'))
    json.dump(stats, open(f'{OUT}/{name}_stats.json', 'w'))
    ok = [s for s in stats if s['ok']]; g = lambda k: np.array([s[k] for s in ok])
    print(len(ok), 'ok;', len(stats) - len(ok), 'failed', [s for s in stats if not s['ok']][:5])
    print('images/scene min/med', g('n_img').min(), np.median(g('n_img')), '| upright score p1/p50', np.percentile(g('up'), [1, 50]).round(2),
          '| quarter turns', dict(zip(*np.unique(g('q'), return_counts=True))), '| max |dt| s p50/p99/max', np.percentile(g('dt_max'), [50, 99, 100]).round(3))
    print('theta source', dict(zip(*np.unique([s['theta_src'] for s in ok], return_counts=True))), '| scenes without ESL31 GT', int((g('n_inst') == 0).sum()),
          '| scale m p5/p50/p95', np.percentile(g('avg_dist'), [5, 50, 95]).round(2), '| mp4 frames vs pincam/2 max diff', int(np.abs(g('n_mp4') - g('n_pin') / 2).max()))
