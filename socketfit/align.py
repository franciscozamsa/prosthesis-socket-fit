import itertools

import numpy as np
from scipy.spatial import cKDTree

from .mesh import Mesh


def kabsch(src: np.ndarray, dst: np.ndarray):
    """Rigid transform (R, t) minimising ||R src + t - dst||."""
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cd - R @ cs


def icp(src: np.ndarray, dst: np.ndarray, R0=None, t0=None, iters=60, tol=1e-7):
    """Point-to-point ICP aligning src onto dst. Returns (R, t, mean closest-point distance)."""
    tree = cKDTree(dst)
    R = np.eye(3) if R0 is None else R0
    t = np.zeros(3) if t0 is None else t0
    prev = np.inf
    for _ in range(iters):
        moved = src @ R.T + t
        dist, idx = tree.query(moved)
        dR, dt = kabsch(moved, dst[idx])
        R, t = dR @ R, dR @ t + dt
        err = dist.mean()
        if abs(prev - err) < tol:
            break
        prev = err
    dist, _ = tree.query(src @ R.T + t)
    return R, t, float(dist.mean())


def _principal_axes(points):
    centered = points - points.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov(centered.T))
    return vecs[:, ::-1]


def align(limb: Mesh, target: np.ndarray):
    """Align the limb to the target points from an unknown pose.

    Starts from the principal-axes frames (the axis signs are ambiguous, so all four proper
    combinations are tried) and keeps the ICP result with the lowest residual.
    """
    Vs, Vl = _principal_axes(target), _principal_axes(limb.vertices)
    best = None
    for signs in itertools.product([1, -1], repeat=3):
        R0 = Vs @ np.diag(signs) @ Vl.T
        if np.linalg.det(R0) < 0:
            continue
        t0 = target.mean(axis=0) - R0 @ limb.vertices.mean(axis=0)
        R, t, err = icp(limb.vertices, target, R0, t0)
        if best is None or err < best[2]:
            best = (R, t, err)
    R, t, err = best
    return Mesh(limb.vertices @ R.T + t, limb.faces.copy()), R, t, err
