"""The treams adapter: a pure reindexing of treams' parity-basis coefficients (no phase, no scale).

Needs treams (its own Python 3.13 venv); skipped elsewhere.  Checks the map
(``pol = 0`` is ``M`` by mask, not by array position), the refusals (helicity
basis, several expansion centres, an unsolved cluster), and end-to-end physics
against the independent oracles: treams' plane waves and T-matrices equal the
oracle's, the force is Mie radiation pressure and parallel to ``khat``, a
sphere in a standing wave built by treams' translations feels ``A sin(2kz)``,
and the per-particle cluster forces add up to the global force.
"""

from __future__ import annotations

import numpy as np
import pytest

treams = pytest.importorskip("treams", reason="treams needs its own Python 3.13 venv")

import oracles as O  # noqa: E402
import swforce as sf  # noqa: E402
from swforce.adapters import treams as swt  # noqa: E402

pytestmark = pytest.mark.treams

K0 = 2 * np.pi                  # vacuum wavelength 1


def khat_of(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def sphere(lmax, radius, index, poltype="parity"):
    return treams.TMatrix.sphere(lmax, K0, radius, [treams.Material(complex(index) ** 2), treams.Material()],
                                 poltype=poltype)


def illuminate(tm, khat, pol, basis=None):
    beam = treams.plane_wave((K0 * np.asarray(khat)).tolist(), list(pol), k0=K0, material=treams.Material(),
                             poltype="parity")
    a = beam.expand(tm.basis if basis is None else basis)
    return a, tm @ a


# --------------------------------------------------------------------------
# The map
# --------------------------------------------------------------------------


def test_split_is_a_pure_permutation(rng):
    basis = treams.SphericalWaveBasis.default(3)
    c = rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis))
    cM, cN = swt.split(c, basis)
    assert cM.size == cN.size == sf.basis_size(3)
    for value, l, m, pol in zip(c, basis.l, basis.m, basis.pol):
        target = cM if pol == 0 else cN
        assert target[sf.basis.ci_index(l, m)] == value
    assert np.count_nonzero(cM) + np.count_nonzero(cN) == np.count_nonzero(c)


def test_pol_zero_is_the_magnetic_multipole_by_mask():
    """treams stores the electric multipole *first* but labels it ``pol = 1``: select by label."""
    basis = treams.SphericalWaveBasis.default(2)
    assert list(basis.pol[:2]) == [1, 0]                           # the trap: array order N, M
    for i, (l, m, pol) in enumerate(zip(basis.l, basis.m, basis.pol)):
        unit = np.zeros(len(basis), complex)
        unit[i] = 1.0
        cM, cN = swt.split(unit, basis)
        hit, other = (cM, cN) if pol == 0 else (cN, cM)
        assert hit[sf.basis.ci_index(l, m)] == 1 and np.count_nonzero(hit) == 1 and not other.any()


def test_split_zero_fills_modes_missing_from_the_basis():
    basis = treams.SphericalWaveBasis.default(3)
    keep = np.asarray(basis.l) != 2
    sub = treams.SphericalWaveBasis([(l, m, p) for l, m, p, k in zip(basis.l, basis.m, basis.pol, keep) if k])
    cM, cN = swt.split(np.ones(len(sub)), sub)
    l, _ = sf.mode_indices(3)
    assert np.all(cM[l == 2] == 0) and np.all(cM[l != 2] == 1) and np.all(cN[l != 2] == 1)


def test_treams_vswf_are_the_oracle_vswf():
    """No rephasing is justified only if treams' functions are the Condon-Shortley ones of the sums."""
    from treams import special as ts
    kr, th, ph = np.array([0.9, 3.3]), np.array([0.4, 2.2]), np.array([1.0, -2.0])
    pts = (kr / K0)[:, None] * O.rhat(th, ph)
    frame = np.stack([O.rhat(th, ph), O.theta_hat(th, ph), O.phi_hat(th, ph)], 1)
    for l, m in O.modes(3):
        M, N = O.vswf(l, m, pts, K0, "outgoing")
        tM = np.array([ts.vsw_M(l, m, x, t, p) for x, t, p in zip(kr, th, ph)])
        tN = np.array([ts.vsw_N(l, m, x, t, p) for x, t, p in zip(kr, th, ph)])
        assert np.abs(np.einsum("pij,pi->pj", frame, tM) - M).max() < 1e-12
        assert np.abs(np.einsum("pij,pi->pj", frame, tN) - N).max() < 1e-12


@pytest.mark.parametrize("theta,phi", [(0.0, 0.0), (0.7, 1.9), (2.6, -0.4)])
def test_treams_plane_wave_equals_the_oracle_expansion(theta, phi):
    k = khat_of(theta, phi)
    pol = (O.theta_hat(theta, phi) + 0.4j * O.phi_hat(theta, phi)) / np.sqrt(1.16)
    basis = treams.SphericalWaveBasis.default(6)
    a, _ = illuminate(sphere(6, 0.1, 1.5), k, pol, basis)
    aM, aN = swt.split(a, basis)
    oM, oN = O.plane_wave_coefficients(k, pol, 6)
    assert max(np.abs(aM - oM).max(), np.abs(aN - oN).max()) < 1e-10 * np.abs(oM).max()


def test_treams_sphere_tmatrix_is_minus_b_minus_a():
    ka, index, lmax = 2.2, 4.0 + 0.05j, 8
    tm = sphere(lmax, ka / K0, index)
    t = np.asarray(tm)
    assert np.abs(t - np.diag(np.diag(t))).max() < 1e-14
    a, b = O.mie(index, ka, lmax)
    for i, (l, pol) in enumerate(zip(tm.basis.l, tm.basis.pol)):
        assert t[i, i] == pytest.approx(-b[l - 1] if pol == 0 else -a[l - 1], rel=1e-10)


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_helicity_basis_is_refused():
    basis = treams.SphericalWaveBasis.default(2)
    with pytest.raises(ValueError, match="parity"):
        swt.split(np.zeros(len(basis)), basis, poltype="helicity")
    tm = sphere(2, 0.1, 1.5, poltype="helicity")
    beam = treams.plane_wave([0, 0, K0], [1, 0, 0], k0=K0, material=treams.Material(), poltype="helicity")
    a = beam.expand(tm.basis)
    with pytest.raises(ValueError, match="parity"):
        swt.force_torque(a, tm @ a, poltype=str(tm.poltype))


def test_several_expansion_centres_are_refused():
    basis = treams.SphericalWaveBasis.default(2, 2, [[0, 0, 0], [0.5, 0, 0]])
    with pytest.raises(ValueError, match="single origin"):
        swt.split(np.zeros(len(basis)), basis)


def test_wrong_length_and_missing_basis_are_refused():
    basis = treams.SphericalWaveBasis.default(2)
    with pytest.raises(ValueError, match="entries"):
        swt.split(np.zeros(len(basis) + 1), basis)
    with pytest.raises(ValueError, match="basis"):
        swt.force_torque(np.zeros(len(basis)), np.zeros(len(basis)))


def test_basis_is_inferred_from_the_arrays():
    tm = sphere(5, 0.2, 2.0)
    a, p = illuminate(tm, [0, 0, 1.0], [1, 0, 0])
    inferred = swt.force_torque(a, p)
    explicit = swt.force_torque(np.asarray(a), np.asarray(p), basis=tm.basis)
    for x, y in zip(inferred, explicit):
        assert np.array_equal(x, y)
    from_scattered = swt.force_torque(np.asarray(a), p)                 # basis read off the second array
    assert np.array_equal(from_scattered.force, explicit.force)


# --------------------------------------------------------------------------
# Physics through treams
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ka", [0.5, 2.2, 5.0])
@pytest.mark.parametrize("index", [1.5, 4.0 + 0.05j])
def test_mie_radiation_pressure_through_treams(ka, index):
    lmax = sf.ka2nmax(ka) + 3
    tm = sphere(lmax, ka / K0, index)
    ft = swt.force_torque(*illuminate(tm, [0, 0, 1.0], [1, 0, 0]))
    xs = O.mie_cross_sections(index, ka, K0, lmax)
    assert ft.force[2] / (4 * K0**2) == pytest.approx(xs["pr"], rel=1e-9)


@pytest.mark.parametrize("theta,phi", [(0.6, 1.1), (1.2, -0.4), (2.5, 2.0)])
def test_force_is_parallel_to_khat_through_treams(theta, phi):
    k = khat_of(theta, phi)
    tm = sphere(10, 2.2 / K0, 4.0)
    F = swt.force_torque(*illuminate(tm, k, O.theta_hat(theta, phi))).force
    assert np.linalg.norm(F - (F @ k) * k) < 1e-10 * np.linalg.norm(F)


@pytest.mark.parametrize("sigma", [+1, -1])
def test_absorbing_sphere_torque_through_treams(sigma):
    ka, index = 3.5, 4.0 + 0.05j
    lmax = sf.ka2nmax(ka) + 3
    tm = sphere(lmax, ka / K0, index)
    ft = swt.force_torque(*illuminate(tm, [0, 0, 1.0], np.array([1, 1j * sigma, 0]) / np.sqrt(2)))
    xs = O.mie_cross_sections(index, ka, K0, lmax)
    assert ft.torque[2] / (4 * K0**2) == pytest.approx(sigma * xs["abs"], rel=1e-8)


def test_standing_wave_through_treams_translations():
    """Sphere placed at ``z`` with ``TMatrix.cluster([tm], [[0, 0, z]])``; the beams are translated by
    treams.  ``F_z(z) = A sin(2kz)`` (antinode of ``E`` at ``z = 0``)."""
    tm = sphere(6, 1.0 / K0, 1.5)
    vac = treams.Material()
    fwd = treams.plane_wave([0, 0, K0], [1, 0, 0], k0=K0, material=vac, poltype="parity")
    bwd = treams.plane_wave([0, 0, -K0], [1, 0, 0], k0=K0, material=vac, poltype="parity")
    zs = np.linspace(0.02, 0.48, 9)
    fz = []
    for z in zs:
        placed = treams.TMatrix.cluster([tm], [[0.0, 0.0, z]])
        a = fwd.expand(placed.basis) + bwd.expand(placed.basis)
        ft = swt.force_torque(a, placed @ a, basis=placed.basis)
        assert np.abs(ft.force[:2]).max() < 1e-9 * abs(ft.force[2]) + 1e-12
        fz.append(ft.force[2])
    fz = np.array(fz)
    A = np.sum(fz * np.sin(2 * K0 * zs)) / np.sum(np.sin(2 * K0 * zs) ** 2)
    assert np.abs(fz - A * np.sin(2 * K0 * zs)).max() < 1e-8 * abs(A)
    assert A < 0


# --------------------------------------------------------------------------
# Clusters
# --------------------------------------------------------------------------


POSITIONS = [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.35, 0.6, 0.0]]


@pytest.fixture(scope="module")
def trimer():
    lmax = 6
    tm = sphere(lmax, 0.15, 4.0)
    cluster = treams.TMatrix.cluster([tm] * 3, POSITIONS)
    solved = cluster.interaction.solve()
    beam = treams.plane_wave([0, 0, K0], [1, 0, 0], k0=K0, material=treams.Material(), poltype="parity")
    return cluster, solved, beam, lmax


def test_unsolved_cluster_is_refused(trimer):
    cluster, _, beam, _ = trimer
    with pytest.raises(ValueError, match="block diagonal"):
        swt.cluster_force_torque(cluster, beam)


def test_cluster_forces_respect_the_mirror_symmetry(trimer):
    """Particles 0 and 1 are mirror images in the plane ``x = 0.35``; particle 2 sits on it."""
    _, solved, beam, _ = trimer
    res = swt.cluster_force_torque(solved, beam)
    assert len(res) == 3 and res.force.shape == (3, 3)
    F = res.force
    scale = np.abs(F).max()
    assert np.abs(F[0] - np.array([-1, 1, 1]) * F[1]).max() < 1e-8 * scale
    assert abs(F[2][0]) < 1e-9 * scale
    assert res.k == pytest.approx(K0)


def test_cluster_forces_sum_to_the_global_force(trimer):
    """Per-particle forces (each from its own exciting field) add up to the force on the whole cluster
    from one global expansion -- different origins, surfaces and truncations."""
    _, solved, beam, lmax = trimer
    res = swt.cluster_force_torque(solved, beam)
    basis = treams.SphericalWaveBasis.default(lmax + 12)
    glob = solved.expand(basis)
    a = beam.expand(basis)
    total = swt.force_torque(a, glob @ a, basis=basis).force
    assert np.abs(res.total_force() - total).max() < 1e-7 * np.linalg.norm(total)


def test_total_torque_carries_the_transport_term(trimer):
    _, solved, beam, _ = trimer
    res = swt.cluster_force_torque(solved, beam)
    about = np.array([0.1, -0.2, 0.3])
    expected = res.torque.sum(0) + res.k * np.cross(res.positions - about, res.force).sum(0)
    assert np.allclose(res.total_torque(about), expected)
    shift = res.total_torque(about) - res.total_torque()
    assert np.allclose(shift, -res.k * np.cross(about, res.total_force()))
