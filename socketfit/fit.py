from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .mesh import Mesh, adjacency, surface_samples, vertex_normals


def clearance(points: np.ndarray, reference: Mesh, reference_outward: bool) -> np.ndarray:
    """Signed distance in mm from points to a reference surface, positive on the side the normals point to."""
    ref_pts, ref_normals = surface_samples(reference, reference_outward)
    _, idx = cKDTree(ref_pts).query(points)
    return np.sum((points - ref_pts[idx]) * ref_normals[idx], axis=1)


def limb_clearance(limb: Mesh, socket: Mesh) -> np.ndarray:
    """Gap between each limb vertex and the socket wall; negative means the limb presses into the wall."""
    return clearance(limb.vertices, socket, reference_outward=False)


@dataclass
class FitReport:
    min_mm: float
    mean_mm: float
    max_mm: float
    interference_pct: float
    within_band_pct: float

    def __str__(self):
        return (f"clearance min {self.min_mm:.2f} / mean {self.mean_mm:.2f} / max {self.max_mm:.2f} mm, "
                f"{self.interference_pct:.1f}% of surface in interference, {self.within_band_pct:.1f}% within the target band")


def report(c: np.ndarray, target: float, band: float = 1.0) -> FitReport:
    return FitReport(float(c.min()), float(c.mean()), float(c.max()), float(100 * np.mean(c < 0)),
                     float(100 * np.mean(np.abs(c - target) <= band)))


def adapt_socket(socket: Mesh, limb: Mesh, target: float = 2.0, iters: int = 3, smooth: int = 8) -> Mesh:
    """Move the socket wall so its gap to the limb approaches `target` mm everywhere.

    Each pass measures the gap at every socket vertex, smooths the required displacement over the
    mesh so the wall stays fair, and moves vertices along their normals. A final one-sided pass
    pushes out any region still tighter than the target.
    """
    out = socket.copy()
    adj = adjacency(out)
    degree = np.asarray(adj.sum(axis=1)).ravel()

    def gap():
        return clearance(out.vertices, limb, reference_outward=True)

    for _ in range(iters):
        delta = target - gap()
        for _ in range(smooth):
            delta = 0.5 * delta + 0.5 * (adj @ delta) / degree
        out.vertices -= vertex_normals(out, outward=False) * delta[:, None]

    short = np.maximum(target - gap(), 0)
    out.vertices -= vertex_normals(out, outward=False) * short[:, None]
    return out
