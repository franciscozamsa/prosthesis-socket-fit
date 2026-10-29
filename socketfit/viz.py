import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .mesh import Mesh

NORM = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=6)
CMAP = "RdYlGn"


def _draw(ax, mesh: Mesh, values: np.ndarray, azim: float):
    tri = mesh.vertices[mesh.faces]
    colors = plt.get_cmap(CMAP)(NORM(values[mesh.faces].mean(axis=1)))
    ax.add_collection3d(Poly3DCollection(tri, facecolors=colors, linewidths=0))
    lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    center, half = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.view_init(elev=15, azim=azim)
    ax.set_axis_off()


def before_after(limb: Mesh, before: np.ndarray, after: np.ndarray, path: str, azims=(35, 215)):
    fig = plt.figure(figsize=(8, 5.6))
    fig.subplots_adjust(hspace=-0.3, wspace=-0.25)
    for row, azim in enumerate(azims):
        for col, (title, values) in enumerate((("Before adaptation", before), ("After adaptation", after))):
            ax = fig.add_subplot(2, 2, row * 2 + col + 1, projection="3d")
            _draw(ax, limb, values, azim)
            if row == 0:
                ax.set_title(title)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), ax=fig.axes, shrink=0.6, pad=0.02)
    cbar.set_label("clearance to socket wall (mm)\nred = limb presses into the wall")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
