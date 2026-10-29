import numpy as np
import pytest

from socketfit.align import align, kabsch
from socketfit.fit import adapt_socket, limb_clearance
from socketfit.mesh import read_stl, write_stl
from socketfit.synthetic import make_limb, make_socket, rotation_matrix, transform


@pytest.fixture(scope="module")
def socket():
    return make_socket()


def test_kabsch_recovers_a_known_transform():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(100, 3))
    R, t = rotation_matrix(30, 20, -50), np.array([5.0, -2.0, 9.0])
    R_est, t_est = kabsch(pts, pts @ R.T + t)
    assert np.allclose(R_est, R) and np.allclose(t_est, t)


@pytest.mark.parametrize("angles", [(25, -15, 140), (-40, 10, 200), (5, 60, -90)])
def test_alignment_recovers_an_unknown_pose(socket, angles):
    R, t = rotation_matrix(*angles), np.array([60.0, -35.0, 80.0])
    scan = transform(make_limb(), R, t)
    limb, R_est, _, _ = align(scan, socket.vertices)
    angle = np.degrees(np.arccos(np.clip((np.trace(R_est @ R) - 1) / 2, -1, 1)))
    assert angle < 1.0
    offset = limb.vertices.mean(axis=0) - make_limb().vertices.mean(axis=0)
    assert np.abs(offset[:2]).max() < 1.0
    assert abs(offset[2]) < 5.0


def test_unmodified_socket_has_interference_at_the_prominence(socket):
    assert limb_clearance(make_limb(), socket).min() < 0


def test_adaptation_removes_interference_and_reaches_the_target(socket):
    limb = make_limb()
    adapted = adapt_socket(socket, limb, target=2.0)
    c = limb_clearance(limb, adapted)
    assert c.min() > 1.0
    assert abs(c.mean() - 2.0) < 0.3


def test_stl_roundtrip(tmp_path):
    mesh = make_limb(n_z=10, n_t=16)
    path = tmp_path / "limb.stl"
    write_stl(mesh, str(path))
    back = read_stl(str(path))
    assert back.faces.shape == mesh.faces.shape
    assert np.allclose(np.sort(back.vertices, axis=0), np.sort(mesh.vertices, axis=0), atol=1e-4)
