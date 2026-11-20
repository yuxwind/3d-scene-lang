"""Detector inputs for ScanNet++ scenes, in the ScanNet layout VGG-T-Det reads (as build_arkit_inputs.py):
  <out>/posed_images/<scene>/NNNNN.jpg + .txt   up to 120 mp4 frames evenly spread over the video, camera-to-world pose
  <out>/points/<scene>.bin                      mesh vertices (<= 200k), float32 x y z r g b
  <out>/scannetpp_infos_<name>.pkl              infos; axis_align_matrix = mesh frame -> target frame
VSI-Bench mode (the only one so far): poses = `aligned_pose` of iphone/pose_intrinsic_imu.json (mesh frame, OpenCV axes),
mp4 frame i == pose frame 2i - 3 (README of the scannetpp_cam0 bundle); target frame = that of
../scenelang_vsibench_gt_firstcam/rows_all5130.jsonl (room yaw about z, origin at the camera of pose frame 0), found by
fitting the segments_anno.json box centres of the row labels onto the row boxes. GT instances = the row boxes of ESL31
classes (yaw enclosed), for the detector's own mAP readout.
    python build_scannetpp_inputs.py vsibench <n procs>
Reason-190K mode: the ScanNet++ scenes of the training traces (../reason190k_videoscene_answer.jsonl.gz) that have poses
under /scr/xinyu/data/scannetpp_r190k/data (unpacked scannetpp_cam_pose_intrinsic.tar / scannetpp_segments_anno.tar).
Videos = VSI-590K scannetppv2/<scene>.mp4 (60 fps, mp4 frame i == pose frame i). Target frame = the trace frame, fitted
as above on the union of the trace boxes of the scene. Scenes whose `aligned_pose` cameras leave the mesh (ARKit drift,
as d755b3d9d8) use the COLMAP poses (iphone/colmap/images.txt, stride 10, mesh frame) instead. GT instances = the trace
boxes of ESL31 classes (build_mix.SYN). Output: <vggtdet>/data/scannetpp_r190k/scannetpp_infos_reason190k.pkl
    python build_scannetpp_inputs.py reason190k <n procs>
"""
import json, os, pickle, sys, math, numpy as np, decord
from PIL import Image
from multiprocessing import Pool

B = '/scr/xinyu/data/scannetpp_cam0_bundle/scannetpp_cam0'; PLY = '/home/student/xinyu/data/sync/scannetpp_ply'
MP4 = '/scr/xinyu/data/VSI-Bench/scannetpp'; OUT = '/scr/xinyu/projects/vggtdet_es56/data/scannetpp_vsibench'; N = 120
ROWS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../scenelang_vsibench_gt_firstcam/rows_all5130.jsonl')
ESL31 = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
         'curtain', 'refrigerator', 'toilet', 'sink', 'bathtub', 'trash bin', 'pillow', 'monitor', 'backpack',
         'printer', 'lamp', 'mirror', 'tv', 'radiator', 'nightstand', 'plant', 'microwave', 'fan', 'blackboard', 'stool']
SYN = {'trash can': 'trash bin', 'heater': 'radiator', 'whiteboard': 'blackboard', 'table lamp': 'lamp', 'exhaust fan': 'fan'}
DATA = f'{B}/scannetpp/data'; FMAP = lambda i: 2 * i - 3; NAME = 'scannetpp_infos_vsibench'; CLS = lambda l: SYN.get(l, l)
GT = {}
if len(sys.argv) > 1 and sys.argv[1] == 'reason190k':
    import gzip, re
    from build_mix import SYN as _S, SCENE as _SC
    DATA = '/scr/xinyu/data/scannetpp_r190k/data'; MP4 = '/scr/xinyu/data/VSI-590K/scannetppv2'; FMAP = lambda i: i
    OUT = '/scr/xinyu/projects/vggtdet_es56/data/scannetpp_r190k'; NAME = 'scannetpp_infos_reason190k'
    CLS = lambda l: _S.get(l, [l])[0]
    _seen = {}
    for l in gzip.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../reason190k_videoscene_answer.jsonl.gz'), 'rt'):
        r = json.loads(l)
        if r['meta']['dataset'] != 'scannetpp': continue
        for o in json.loads(_SC.search(r['conversations'][0]['value']).group(1)):
            _seen.setdefault(r['meta']['scene'], {})[(re.sub(r'\s+\d+$', '', o['label']), tuple(o['bbox_3d']))] = 1
    GT = {s: [dict(label=k[0], bbox_3d=list(k[1]) + [0.0]) for k in v] for s, v in _seen.items()
          if os.path.exists(f'{DATA}/{s}/iphone/pose_intrinsic_imu.json')}
else:
    for l in open(ROWS):
        r = json.loads(l)
        if r['dataset'] == 'scannetpp' and r['scene_name'] not in GT: GT[r['scene_name']] = json.loads(r['scene_language'])


def qmat(w, x, y, z):
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def colmap_poses(s):
    """{pose frame index: camera-to-mesh 4x4} from iphone/colmap/images.txt"""
    col = {}
    for l in open(f'{DATA}/{s}/iphone/colmap/images.txt'):
        p = l.split()
        if len(p) == 10 and p[9].endswith('.jpg'):
            R = qmat(*np.array(p[1:5], float)); M = np.eye(4); M[:3, :3] = R.T; M[:3, 3] = -R.T @ np.array(p[5:8], float); col[int(p[9][6:12])] = M
    return col


def ply_vertices(path):
    with open(path, 'rb') as f:
        n, props = None, []
        while True:
            l = f.readline().decode().strip()
            if l.startswith('element vertex'): n = int(l.split()[-1])
            if l.startswith('element face'): n_done = True
            if l.startswith('property') and 'list' not in l and len(props) < 6: props.append(l.split()[1:])
            if l == 'end_header': break
        assert [p[1] for p in props] == ['x', 'y', 'z', 'red', 'green', 'blue'], props
        v = np.frombuffer(f.read(n * 15), dtype=np.dtype([('p', '<f4', 3), ('c', 'u1', 3)]))
    return np.hstack([v['p'], v['c'].astype(np.float32)])


def rz(th): return np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])


def fit(s, cam0):
    """mesh frame -> row frame: p_row = Rz(th) p + t. Tries t from the first camera, then a free translation."""
    a = json.load(open(f'{DATA}/{s}/scans/segments_anno.json'))['segGroups']
    G = np.array([o['bbox_3d'][:3] for o in GT[s]]); C = np.array([g['obb']['centroid'] for g in a])
    best = None
    for th in np.deg2rad(np.arange(0, 360, 0.25)):
        P = C @ rz(th).T
        for t in (-rz(th) @ cam0, None):
            if t is None:  # free: translation = median offset of nearest neighbours after centring
                Q = P - np.median(P, 0) + np.median(G, 0)
                t = np.median(G - Q[np.linalg.norm(G[:, None] - Q[None], axis=2).argmin(1)], 0) - np.median(P, 0) + np.median(G, 0); free = True
            else: free = False
            d = np.linalg.norm(G[:, None] - (P + t)[None], axis=2).min(1); m = np.median(d)
            if best is None or m < best[0] - (1e-3 if free else 0): best = (m, th, t, free, float((d < 0.1).mean()))
    return best


def one(s):
    try:
        d = json.load(open(f'{DATA}/{s}/iphone/pose_intrinsic_imu.json')); keys = sorted(d)
        vr = decord.VideoReader(f'{MP4}/{s}.mp4'); H, W = vr[0].shape[:2]
        sel = np.unique(np.linspace(0, len(vr) - 1, min(N, len(vr))).round().astype(int))
        pk = [keys[min(max(FMAP(i), 0), len(keys) - 1)] for i in sel]
        c2w = np.array([d[k]['aligned_pose'] for k in pk], float); K = np.array(d[pk[0]]['intrinsic'], float)
        P = ply_vertices(f'{PLY}/{s}.ply'); lo, hi = P[:, :3].min(0) - 1.0, P[:, :3].max(0) + 1.0; src = 'arkit'
        cam0 = np.array(d[keys[0]]['aligned_pose'], float)[:3, 3]
        if ((c2w[:, :3, 3] < lo) | (c2w[:, :3, 3] > hi)).any(1).mean() > 0.05:   # ARKit drift -> COLMAP poses
            col = colmap_poses(s); inv = {FMAP(i): i for i in range(len(vr))}; fs = [f for f in sorted(col) if f in inv]
            fs = [fs[i] for i in np.unique(np.linspace(0, len(fs) - 1, min(N, len(fs))).round().astype(int))]
            sel = np.array([inv[f] for f in fs]); c2w = np.array([col[f] for f in fs]); src = 'colmap'; cam0 = c2w[0, :3, 3]
        res, th, t, free, share = fit(s, cam0)
        M = np.eye(4); M[:3, :3] = rz(th); M[:3, 3] = t
        dd = f'{OUT}/posed_images/{s}'; os.makedirs(dd, exist_ok=True)
        fr = vr.get_batch(list(sel)).asnumpy()
        for j in range(len(sel)):
            Image.fromarray(fr[j]).save(f'{dd}/{j:05d}.jpg', quality=92); np.savetxt(f'{dd}/{j:05d}.txt', c2w[j], fmt='%.6f')
        if len(P) > 200000: P = P[np.random.RandomState(0).choice(len(P), 200000, replace=False)]
        os.makedirs(f'{OUT}/points', exist_ok=True); P.astype(np.float32).tofile(f'{OUT}/points/{s}.bin')
        inst = []
        for o in GT[s]:
            n = CLS(o['label'])
            if n not in ESL31: continue
            x, y, z, dx, dy, dz, yaw = o['bbox_3d'][:7]; c, si = abs(math.cos(yaw)), abs(math.sin(yaw))
            inst.append(dict(bbox_3d=[x, y, z, dx * c + dy * si, dx * si + dy * c, dz], bbox_label_3d=ESL31.index(n)))
        K4 = np.eye(4); K4[0, 0], K4[1, 1], K4[0, 2], K4[1, 2] = K[0, 0] * W / 1920, K[1, 1] * H / 1440, K[0, 2] * W / 1920, K[1, 2] * H / 1440
        info = dict(lidar_points=dict(num_pts_feats=6, lidar_path=f'{s}.bin'), instances=inst, cam2img=K4,
                    lidar2cam=[c for c in c2w], img_paths=[f'posed_images/{s}/{j:05d}.jpg' for j in range(len(sel))],
                    axis_align_matrix=M.tolist(), pts_semantic_mask_path=f'{s}.bin', pts_instance_mask_path=f'{s}.bin')
        up = float(-c2w[:, 2, 1].mean()); cz = c2w[:, 2, 3]
        stat = dict(scene=s, ok=True, pose_src=src, n_img=len(sel), up=up, fit_res=float(res), fit_share=share, fit_free=bool(free), n_inst=len(inst),
                    cam_z_range=[float(cz.min()), float(cz.max())], pts_z=[float(P[:, 2].min()), float(P[:, 2].max())],
                    avg_dist=float(np.linalg.norm(P[:, :3] - c2w[0, :3, 3], axis=1).mean()), n_pose=len(keys), n_mp4=len(vr))
        return info, stat
    except Exception as e:
        return None, dict(scene=s, ok=False, err=repr(e))


if __name__ == '__main__':
    scenes = sorted(GT); name = NAME
    if os.environ.get('NEW_ONLY') == '1':            # only scenes not in the existing info file (e.g. after downloading more poses) -> <NAME>_new.pkl
        old = {i['lidar_points']['lidar_path'][:-4] for i in pickle.load(open(f'{OUT}/{NAME}.pkl', 'rb'))['data_list']}
        scenes = [s for s in scenes if s not in old]; name = NAME + '_new'; print(len(scenes), 'new scenes', flush=True)
    with Pool(int(sys.argv[2])) as p: res = p.map(one, scenes, chunksize=1)
    infos = [i for i, _ in res if i]; stats = [s for _, s in res]
    meta = pickle.load(open('/scr/xinyu/projects/vggtdet_es56/data/scannet/scannet_infos_val_esl31.pkl', 'rb'))['metainfo']
    os.makedirs(OUT, exist_ok=True); pickle.dump(dict(metainfo=meta, data_list=infos), open(f'{OUT}/{name}.pkl', 'wb'))
    json.dump(stats, open(f'{OUT}/{name}_stats.json', 'w'))
    ok = [s for s in stats if s['ok']]; g = lambda k: np.array([s[k] for s in ok])
    print(len(ok), 'ok;', len(stats) - len(ok), 'failed', [(s['scene'], s['err'][:80]) for s in stats if not s['ok']][:60])
    if ok:
        print('upright score p0/p50', np.percentile(g('up'), [0, 50]).round(2), '| fit residual p50/max', np.percentile(g('fit_res'), [50, 100]).round(3),
              '| share<0.1m min', g('fit_share').min().round(2), '| free-translation fits', [s['scene'] for s in ok if s['fit_free']],
              '| colmap fallback', [s['scene'] for s in ok if s['pose_src'] == 'colmap'][:40], '| scale m p5/p50/p95', np.percentile(g('avg_dist'), [5, 50, 95]).round(2), '| pose/mp4 frames ratio', np.percentile(g('n_pose') / g('n_mp4'), [0, 50, 100]).round(3))
        print('scenes with avg_dist > 8 m or camera outside mesh z:', [(s['scene'], round(s['avg_dist'], 1), s['cam_z_range'], s['pts_z']) for s in ok if s['avg_dist'] > 8 or s['cam_z_range'][1] > s['pts_z'][1] + 0.5])
