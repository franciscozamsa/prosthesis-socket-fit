import numpy as np

from .mesh import Mesh

LENGTH = 150.0
DISTAL_ROUNDING = 25.0


def _nominal_radius(z):
    taper = 32.0 + 13.0 * (z / LENGTH)
    rounding = np.sqrt(np.clip(1 - ((DISTAL_ROUNDING - np.minimum(z, DISTAL_ROUNDING)) / DISTAL_ROUNDING) ** 2, 0, 1))
    return taper * rounding


def _bump(z, theta, z0, theta0, height, width_z, width_theta):
    dtheta = np.angle(np.exp(1j * (theta - theta0)))
    return height * np.exp(-0.5 * ((z - z0) / width_z) ** 2 - 0.5 * (dtheta / width_theta) ** 2)


def _revolve(radius_fn, n_z: int, n_t: int) -> Mesh:
    """Closed distal end at z=0 (apex vertex), open proximal end at z=LENGTH; slightly elliptical cross-section."""
    z = np.linspace(0, LENGTH, n_z + 1)[1:]
    theta = np.linspace(0, 2 * np.pi, n_t, endpoint=False)
    Z, T = np.meshgrid(z, theta, indexing="ij")
    R = radius_fn(Z, T)
    ring = np.stack([1.1 * R * np.cos(T), 0.9 * R * np.sin(T), Z], axis=-1).reshape(-1, 3)
    vertices = np.vstack([[0.0, 0.0, 0.0], ring])

    idx = lambda i, j: 1 + i * n_t + (j % n_t)
    faces = [[0, idx(0, j + 1), idx(0, j)] for j in range(n_t)]
    for i in range(n_z - 1):
        for j in range(n_t):
            faces.append([idx(i, j), idx(i, j + 1), idx(i + 1, j)])
            faces.append([idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)])
    return Mesh(vertices, np.array(faces))


def make_limb(n_z=50, n_t=64, noise=0.3, seed=0) -> Mesh:
    """Synthetic residual limb 'scan': a tapered, rounded cylinder with a bony prominence and scanner noise."""
    rng = np.random.default_rng(seed)

    def radius(z, theta):
        return _nominal_radius(z) + _bump(z, theta, 100, 0.8, 4.0, 14, 0.35)

    mesh = _revolve(radius, n_z, n_t)
    mesh.vertices[1:] += rng.normal(0, noise, size=mesh.vertices[1:].shape)
    return mesh


def make_socket(clearance=2.0, n_z=60, n_t=72) -> Mesh:
    """Synthetic socket inner surface, designed from the nominal limb without its prominence and with a loose region."""

    def radius(z, theta):
        return _nominal_radius(z) + clearance + _bump(z, theta, 55, 3.8, 3.0, 18, 0.5)

    return _revolve(radius, n_z, n_t)


def rotation_matrix(rx_deg, ry_deg, rz_deg) -> np.ndarray:
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def transform(mesh: Mesh, R: np.ndarray, t: np.ndarray) -> Mesh:
    return Mesh(mesh.vertices @ R.T + t, mesh.faces.copy())
