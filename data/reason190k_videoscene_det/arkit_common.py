"""Shared ARKitScenes helpers: trajectory, room yaw rule, upright-camera rotation."""
import json, numpy as np


def rodrigues(r):
    th = np.linalg.norm(r)
    if th < 1e-12: return np.eye(3)
    k = r / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def traj(path):  # -> timestamps, cam2world (N,4,4), sensor (landscape) camera, OpenCV axes
    a = np.loadtxt(path).reshape(-1, 7); out = []
    for l in a:
        E = np.eye(4); E[:3, :3] = rodrigues(l[1:4]); E[:3, 3] = l[4:7]; out.append(np.linalg.inv(E))
    return a[:, 0], np.array(out)


def boxes(path):  # obbAligned: centres (n,3), lengths (n,3), axes (n,3,3 rows), labels
    ob = json.load(open(path))['data']
    o = [d['segments']['obbAligned'] for d in ob]
    return (np.array([x['centroid'] for x in o]).reshape(-1, 3), np.array([x['axesLengths'] for x in o]).reshape(-1, 3),
            np.array([x['normalizedAxes'] for x in o]).reshape(-1, 3, 3), [d['label'] for d in ob])


def room_theta(A, L, fwd0):
    """World -> trace yaw: room axes = dominant box direction (mod 90 deg, weighted by footprint), then the multiple
    of 90 deg that puts the first camera's forward nearest +y. No boxes: heading only (forward = +y)."""
    h = np.arctan2(fwd0[1], fwd0[0])
    if len(A) == 0: return np.pi / 2 - h
    yaw = np.arctan2(A[:, 0, 1], A[:, 0, 0]); w = L[:, 0] * L[:, 1]
    dom = np.arctan2((w * np.sin(4 * yaw)).sum(), (w * np.cos(4 * yaw)).sum()) / 4      # room axis direction in world, mod 90
    th = -dom                                                                          # turn that axis onto +x
    k = np.round((np.pi / 2 - (h + th)) / (np.pi / 2))
    return th + k * np.pi / 2


def upright_turn(c2w):
    """Quarter turns q (about the optical axis) so that the image shown upright has its up direction closest to
    world +z: c2w_upright = c2w @ Rz(q * 90 deg). Returns q and the 4x4 Rz."""
    best = None
    for q in range(4):
        a = q * np.pi / 2; R = np.eye(4); R[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
        up = -(c2w[:, :3, :3] @ R[:3, :3])[:, 2, 1].mean()        # world-z component of image-up (-y)
        if best is None or up > best[0]: best = (up, q, R.round())
    return best[1], best[2], best[0]
