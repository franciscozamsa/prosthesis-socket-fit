from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class Mesh:
    vertices: np.ndarray  # (n, 3) millimetres
    faces: np.ndarray     # (m, 3) vertex indices

    def copy(self) -> "Mesh":
        return Mesh(self.vertices.copy(), self.faces.copy())


def face_normals(mesh: Mesh) -> np.ndarray:
    v = mesh.vertices[mesh.faces]
    return np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])


def vertex_normals(mesh: Mesh, outward: bool) -> np.ndarray:
    """Area-weighted vertex normals, flipped as a whole so they point away from (outward) or toward the mesh centroid."""
    fn = face_normals(mesh)
    normals = np.zeros_like(mesh.vertices)
    for k in range(3):
        np.add.at(normals, mesh.faces[:, k], fn)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    pointing_out = np.sum(normals * (mesh.vertices - mesh.vertices.mean(axis=0))) > 0
    return normals if pointing_out == outward else -normals


def surface_samples(mesh: Mesh, outward: bool):
    """Vertices plus triangle centroids with their normals, so distance queries are not limited by vertex spacing."""
    vn = vertex_normals(mesh, outward)
    fn = face_normals(mesh)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    face_sign = np.sign(np.sum(fn * vn[mesh.faces[:, 0]]))
    return np.vstack([mesh.vertices, centroids]), np.vstack([vn, fn * face_sign])


def adjacency(mesh: Mesh) -> sp.csr_matrix:
    n = len(mesh.vertices)
    f = mesh.faces
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    a = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    a.data[:] = 1
    return a


def write_stl(mesh: Mesh, path: str, name: str = "mesh") -> None:
    tri = mesh.vertices[mesh.faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    with open(path, "w") as f:
        f.write(f"solid {name}\n")
        for normal, t in zip(n, tri):
            f.write(f"facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n outer loop\n")
            for p in t:
                f.write(f"  vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}\n")
            f.write(" endloop\nendfacet\n")
        f.write(f"endsolid {name}\n")


def read_stl(path: str) -> Mesh:
    """Read an ASCII STL, merging duplicate vertices."""
    pts = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] == "vertex":
                pts.append([float(x) for x in parts[1:4]])
    pts = np.array(pts)
    if len(pts) == 0 or len(pts) % 3:
        raise ValueError("no ASCII STL triangles found (binary STL is not supported)")
    unique, inverse = np.unique(np.round(pts, 6), axis=0, return_inverse=True)
    return Mesh(unique, inverse.reshape(-1, 3))
