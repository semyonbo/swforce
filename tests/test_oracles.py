"""The yardsticks themselves: every oracle in ``oracles.py`` against something more basic.

Nothing here trusts swforce, except the last test, which checks that the
60-digit ``flux_mp`` reproduces ``swforce.flux`` where double precision is
adequate (so that it can be trusted where double precision is *not*).
Checks used: explicit low-order formulas, finite differences, orthonormality by
quadrature, Maxwell's equations, closed-form stress integrals of dipoles and of
a fully absorbed truncated plane wave, and the classic limits of Mie theory.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import spherical_jn, spherical_yn

import oracles as O

K = 1.3                       # wavenumber in arbitrary inverse length units


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.abs(a - b).max() / np.abs(b).max()


def random_points(rng, n, rmin, rmax):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1)[:, None] * rng.uniform(rmin, rmax, (n, 1))


# --------------------------------------------------------------------------
# Harmonics
# --------------------------------------------------------------------------


def test_Y_is_condon_shortley_orthonormal_harmonic():
    th, ph = np.array([0.3, 1.1, 2.4]), np.array([-0.7, 0.4, 2.9])
    assert np.allclose(O.Y(0, 0, th, ph), 1 / np.sqrt(4 * np.pi))
    assert np.allclose(O.Y(1, 0, th, ph), np.sqrt(3 / (4 * np.pi)) * np.cos(th))
    # the Condon-Shortley (-1)^m: Y_11 carries a minus sign, Y_1-1 does not
    assert np.allclose(O.Y(1, 1, th, ph), -np.sqrt(3 / (8 * np.pi)) * np.sin(th) * np.exp(1j * ph))
    assert np.allclose(O.Y(1, -1, th, ph), np.sqrt(3 / (8 * np.pi)) * np.sin(th) * np.exp(-1j * ph))
    assert np.allclose(O.Y(2, 2, th, ph), np.sqrt(15 / (32 * np.pi)) * np.sin(th) ** 2 * np.exp(2j * ph))
    assert np.all(O.Y(2, 3, th, ph) == 0) and np.all(O.Y(-1, 0, th, ph) == 0)


def test_Y_orthonormal_on_the_sphere():
    th, ph, w = O.sphere_quadrature(12, 24)
    ms = O.modes(5)
    G = np.array([[np.sum(w * np.conj(O.Y(l, m, th, ph)) * O.Y(l2, m2, th, ph)) for l2, m2 in ms] for l, m in ms])
    assert np.abs(G - np.eye(len(ms))).max() < 1e-13


def test_dY_dtheta_recurrence_matches_finite_differences(rng):
    th, ph = rng.uniform(0.2, 2.9, 30), rng.uniform(-3, 3, 30)
    h = 1e-6
    for l, m in O.modes(7):
        fd = (O.Y(l, m, th + h, ph) - O.Y(l, m, th - h, ph)) / (2 * h)
        assert np.abs(O.dY_dtheta(l, m, th, ph) - fd).max() < 1e-8 * max(1.0, np.abs(fd).max())


def test_X_cartesian_ladder_form_equals_theta_phi_form(rng):
    """``L_+-`` form of ``X`` (used everywhere) vs ``-(m/sin) Y e_th - i dY/dth e_ph``."""
    th, ph = rng.uniform(0.2, 2.9, 30), rng.uniform(-3, 3, 30)
    for l, m in O.modes(7):
        x = O.X(l, m, th, ph)
        assert np.abs(x - O.X_spherical(l, m, th, ph)).max() < 1e-13
        assert np.abs(x - O.X_spherical(l, m, th, ph, dtheta=1e-6)).max() < 1e-8


def test_X_vanishes_outside_the_allowed_orders():
    th, ph = np.array([0.4, 2.0]), np.array([0.1, -1.0])
    for l, m in ((1, 2), (2, -3), (0, 0), (3, 4)):
        assert np.array_equal(O.X(l, m, th, ph), np.zeros((2, 3), complex))


def test_X_is_tangential_and_regular_at_the_poles():
    th = np.array([0.0, np.pi, 0.7])
    ph = np.array([0.3, -1.2, 0.5])
    for l, m in O.modes(5):
        x = O.X(l, m, th, ph)
        assert np.all(np.isfinite(x))
        assert np.abs(np.sum(x * O.rhat(th, ph), -1)).max() < 1e-14
        if abs(m) != 1:                                 # only m = +-1 survive on the axis
            assert np.abs(x[:2]).max() < 1e-14


def test_vector_harmonics_orthonormal_on_the_sphere():
    """``X_lm``, ``rhat x X_lm`` and ``Y_lm rhat`` are three mutually orthogonal orthonormal families."""
    th, ph, w = O.sphere_quadrature(12, 24)
    rh = O.rhat(th, ph)
    table = O._angular_table(5, th, ph)
    fams = []
    for (l, m), (y, x, rx) in table.items():
        fams += [x, rx, y[:, None] * rh]
    G = np.array([[np.sum(w * np.sum(np.conj(a) * b, -1)) for b in fams] for a in fams])
    assert np.abs(G - np.eye(len(fams))).max() < 1e-13


# --------------------------------------------------------------------------
# VSWFs and fields
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["regular", "outgoing", "incoming"])
def test_vswf_curl_relations(kind, rng):
    """``curl M = k N`` and ``curl N = k M`` by central differences."""
    pts = random_points(rng, 6, 1.0, 4.0)
    for l, m in O.modes(4):
        M = lambda x: O.vswf(l, m, x, K, kind)[0]          # noqa: E731
        N = lambda x: O.vswf(l, m, x, K, kind)[1]          # noqa: E731
        m0, n0 = O.vswf(l, m, pts, K, kind)
        scale = max(np.abs(m0).max(), np.abs(n0).max())
        assert np.abs(O.curl(M, pts) - K * n0).max() < 1e-8 * scale
        assert np.abs(O.curl(N, pts) - K * m0).max() < 1e-8 * scale


@pytest.mark.parametrize("kind", ["regular", "outgoing"])
def test_vswf_solves_the_vector_helmholtz_equation(kind, rng):
    pts = random_points(rng, 5, 1.5, 4.0)
    for l, m in ((1, 0), (2, -1), (3, 2), (4, -4)):
        for which in (0, 1):
            f = lambda x: O.vswf(l, m, x, K, kind)[which]      # noqa: E731
            val = f(pts)
            assert np.abs(O.laplacian(f, pts, h=3e-4) + K**2 * val).max() < 1e-5 * np.abs(val).max()


def test_vswf_matches_the_textbook_low_order_forms(rng):
    """``M_10 ~ j_1 sin(theta) phi_hat`` and the radial part of ``N_10`` from ``i sqrt(2) (j_1/x) Y_10``."""
    pts = random_points(rng, 5, 0.5, 3.0)
    r, th, ph = O.to_spherical(pts)
    x = K * r
    M, N = O.vswf(1, 0, pts, K)
    expect_M = (spherical_jn(1, x) * np.sqrt(3 / (8 * np.pi)) * 1j * np.sin(th))[:, None] * O.phi_hat(th, ph)
    assert rel(M, expect_M) < 1e-14
    radial_N = np.sum(N * O.rhat(th, ph), -1)
    assert rel(radial_N, 1j * np.sqrt(2) * spherical_jn(1, x) / x * np.sqrt(3 / (4 * np.pi)) * np.cos(th)) < 1e-14


@pytest.mark.parametrize("n", [1.0, 1.7])
def test_fields_satisfy_faraday_and_ampere(n, rng):
    """``curl E = i omega mu H``, ``curl H = -i omega eps E`` in units eps0 = mu0 = 1 (``omega = k/n``)."""
    a = O.random_coefficients(rng, 3)
    p = O.random_coefficients(rng, 3)
    pts = random_points(rng, 5, 1.5, 4.0)
    E = lambda x: O.fields(a, p, x, K, n)[0]                  # noqa: E731
    H = lambda x: O.fields(a, p, x, K, n)[1]                  # noqa: E731
    e0, h0 = O.fields(a, p, pts, K, n)
    omega = K / n
    assert rel(O.curl(E, pts), 1j * omega * h0) < 1e-8
    assert rel(O.curl(H, pts), -1j * omega * n**2 * e0) < 1e-8


def test_efield_rejects_bad_lengths():
    with pytest.raises(ValueError):
        O.efield(np.ones(3), np.ones(4), [[1.0, 0, 0]], K)
    with pytest.raises(ValueError):
        O.radial(1, 1.0, "standing")


def test_pattern_is_the_far_field(rng):
    """``k r e^{-ikr} E_out -> -i A_c`` and ``k r e^{ikr} E_in -> i B_c``, error ``~ l^2 / (kr)``."""
    cM, cN = O.random_coefficients(rng, 4)
    th, ph = rng.uniform(0.2, 2.9, 6), rng.uniform(-3, 3, 6)
    kr = 1e7
    pts = (kr / K) * O.rhat(th, ph)
    E_out = O.efield(cM, cN, pts, K, "outgoing")
    E_in = O.efield(cM, cN, pts, K, "incoming")
    assert rel(kr * np.exp(-1j * kr) * E_out, -1j * O.pattern(cM, cN, th, ph)) < 1e-5
    assert rel(kr * np.exp(1j * kr) * E_in, 1j * O.pattern(cM, cN, th, ph, incoming=True)) < 1e-5


def test_oracle_vswf_equals_treams():
    """``oracles.vswf`` is treams' ``vsw_rM/vsw_rN`` (regular) and ``vsw_M/vsw_N`` (outgoing)."""
    ts = pytest.importorskip("treams.special")
    kr = np.array([0.83, 2.4, 5.1, 7.7])
    th = np.array([0.31, 1.2, 2.6, 0.9])
    ph = np.array([0.2, 1.7, -0.6, 2.2])
    pts = (kr / K)[:, None] * O.rhat(th, ph)
    basis = np.stack([O.rhat(th, ph), O.theta_hat(th, ph), O.phi_hat(th, ph)], 1)    # rows r, theta, phi
    for kind, fM, fN in (("regular", ts.vsw_rM, ts.vsw_rN), ("outgoing", ts.vsw_M, ts.vsw_N)):
        for l, m in O.modes(4):
            M, N = O.vswf(l, m, pts, K, kind)
            tM = np.array([fM(l, m, x, t, p) for x, t, p in zip(kr, th, ph)])
            tN = np.array([fN(l, m, x, t, p) for x, t, p in zip(kr, th, ph)])
            assert np.abs(np.einsum("pij,pi->pj", basis, tM) - M).max() < 1e-12
            assert np.abs(np.einsum("pij,pi->pj", basis, tN) - N).max() < 1e-12
    th1, ph1 = np.array([0.4, 2.0]), np.array([0.3, -1.1])
    for l, m in O.modes(3):
        tX = np.array([ts.vsh_X(l, m, t, p) for t, p in zip(th1, ph1)])
        b = np.stack([O.rhat(th1, ph1), O.theta_hat(th1, ph1), O.phi_hat(th1, ph1)], 1)
        assert np.abs(np.einsum("pij,pi->pj", b, tX) - O.X(l, m, th1, ph1)).max() < 1e-12


# --------------------------------------------------------------------------
# Analytic fields satisfy Maxwell's equations
# --------------------------------------------------------------------------


def test_plane_and_standing_waves_satisfy_maxwell(rng):
    pts = random_points(rng, 5, 0.1, 3.0)
    khat = np.array([0.3, -0.5, 0.8])
    khat /= np.linalg.norm(khat)
    pol = np.cross(khat, [0, 0, 1.0])
    pol = (pol + 0.4j * np.cross(khat, pol)) / np.linalg.norm(pol)
    for n in (1.0, 1.4):
        Z = 1 / n
        for E, H in ((lambda x: O.plane_wave_fields(x, K, khat, pol, Z=Z)[0], lambda x: O.plane_wave_fields(x, K, khat, pol, Z=Z)[1]),
                     (lambda x: O.standing_wave_fields(x, K, z0=0.37, Z=Z)[0], lambda x: O.standing_wave_fields(x, K, z0=0.37, Z=Z)[1])):
            omega = K / n
            assert rel(O.curl(E, pts), 1j * omega * H(pts)) < 1e-8
            assert rel(O.curl(H, pts), -1j * omega * n**2 * E(pts)) < 1e-8
    assert np.allclose(O.standing_wave([[0, 0, 0.0]], K, z0=0.0), [[2, 0, 0]])


def test_dipole_fields_satisfy_maxwell_in_si(rng):
    k = 2 * np.pi / 1e-6
    p = (rng.normal(size=3) + 1j * rng.normal(size=3)) * 1e-30
    m = (rng.normal(size=3) + 1j * rng.normal(size=3)) * 1e-30 * O.C0
    pts = random_points(rng, 5, 0.3 / k, 3.0 / k)
    omega = k * O.C0
    E = lambda x: O.dipole_fields(x, k, p, m)[0]          # noqa: E731
    H = lambda x: O.dipole_fields(x, k, p, m)[1]          # noqa: E731
    h = 1e-5 / k
    assert rel(O.curl(E, pts, h), 1j * omega * O.MU0 * H(pts)) < 1e-7
    assert rel(O.curl(H, pts, h), -1j * omega * O.EPS0 * E(pts)) < 1e-7


# --------------------------------------------------------------------------
# Stress tensor
# --------------------------------------------------------------------------


def test_stress_of_a_source_free_field_vanishes(rng):
    """A plane wave (or standing wave) has no sources inside any sphere: ``F = tau = 0``."""
    khat = np.array([0.0, 0.6, 0.8])
    pol = np.array([1.0, 0.3j * 0.8, -0.3j * 0.6])
    for r in (0.5, 3.0, 20.0):
        # the integrands are O(r^2 |E|^2) and O(r^3 |E|^2): rounding of that size is all that may remain
        for field in (lambda x: O.plane_wave_fields(x, K, khat, pol), lambda x: O.standing_wave_fields(x, K, z0=0.3)):
            F, T = O.stress_force_torque(field, r, n_theta=60, n_phi=120)
            assert np.abs(F).max() < 1e-13 * (1 + r**2)
            assert np.abs(T).max() < 1e-13 * (1 + r**3)


@pytest.mark.parametrize("N", [1, 2, 5])
@pytest.mark.parametrize("sigma", [+1, -1])
def test_stress_of_a_fully_absorbed_truncated_plane_wave(N, sigma):
    """"Absorbing disc": the incoming half ``a h^(2) / 2`` of a truncated circular plane wave.

    A perfect absorber keeps only the waves converging on it, so the stress
    integral is the momentum and angular momentum they bring in.  Summing the
    ``l = 1..N`` partial waves of ``(x + i sigma y)/sqrt(2) e^{ikz}`` by hand:

        F_z = pi N^2 (N+2) / (2 k^2 (N+1)),     tau_z = sigma pi N (N+2) / (2 k^3)

    (``eps0 = mu0 = 1``).  For large ``N`` this is ``I A / c`` for a disc of
    radius ``N/k`` -- the localisation principle -- and ``k F_z / tau_z = sigma N/(N+1)``.
    Independent of the force sums; also exercises ``plane_wave_coefficients``.
    """
    aM, aN = O.plane_wave_coefficients([0, 0, 1.0], np.array([1, 1j * sigma, 0]) / np.sqrt(2), N)

    def field(x):
        return (O.efield(aM / 2, aN / 2, x, K, "incoming"), -1j * O.efield(aN / 2, aM / 2, x, K, "incoming"))

    for kr in (N / 2 + 1, 5.0, 40.0):                  # closer in, h^(2)_N near fields cost digits
        F, T = O.stress_force_torque(field, kr / K, n_theta=N + 3, n_phi=2 * N + 6)
        Fz = np.pi * N**2 * (N + 2) / (2 * K**2 * (N + 1))
        Tz = sigma * np.pi * N * (N + 2) / (2 * K**3)
        assert F == pytest.approx([0, 0, Fz], abs=1e-12 * Fz)
        assert T == pytest.approx([0, 0, Tz], abs=1e-12 * abs(Tz))


def test_stress_of_dipoles_is_the_recoil(rng):
    """ED ``p`` + MD ``m`` (SI): ``F = -(k^4 / 12 pi eps0 c) Re(p x m*)``,
    ``tau = -(k^3 / 12 pi eps0) Im(p* x p) - (mu0 k^3 / 12 pi) Im(m* x m)`` on every sphere."""
    k = 2 * np.pi / 1e-6
    for _ in range(3):
        p = (rng.normal(size=3) + 1j * rng.normal(size=3)) * 1e-30
        m = (rng.normal(size=3) + 1j * rng.normal(size=3)) * 1e-30 * O.C0
        F_ref = -k**4 / (12 * np.pi * O.EPS0 * O.C0) * np.real(np.cross(p, np.conj(m)))
        T_ref = (-k**3 / (12 * np.pi * O.EPS0) * np.imag(np.cross(np.conj(p), p))
                 - O.MU0 * k**3 / (12 * np.pi) * np.imag(np.cross(np.conj(m), m)))
        for kr in (0.2, 1.0, 30.0):
            F, T = O.stress_force_torque(lambda x: O.dipole_fields(x, k, p, m), kr / k, eps=O.EPS0, mu=O.MU0,
                                         n_theta=10, n_phi=20)
            assert rel(F, F_ref) < 1e-11
            assert rel(T, T_ref) < 1e-11


# --------------------------------------------------------------------------
# Projector
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["regular", "outgoing", "incoming"])
@pytest.mark.parametrize("lmax", [1, 4, 7])
def test_project_inverts_efield(kind, lmax, rng):
    cM, cN = O.random_coefficients(rng, lmax)
    gM, gN = O.project(lambda x: O.efield(cM, cN, x, K, kind), K, lmax, kind)
    assert max(np.abs(gM - cM).max(), np.abs(gN - cN).max()) < 1e-12 * np.abs(np.r_[cM, cN]).max()


def test_project_about_a_shifted_centre(rng):
    cM, cN = O.random_coefficients(rng, 3)
    c = np.array([0.2, -0.4, 0.9])
    gM, gN = O.project(lambda x: O.efield(cM, cN, x - c, K), K, 3, center=c)
    assert max(np.abs(gM - cM).max(), np.abs(gN - cN).max()) < 1e-12


@pytest.mark.parametrize("theta,phi", [(0.0, 0.0), (0.7, 0.4), (np.pi, 1.0), (2.2, -2.5)])
def test_plane_wave_coefficients_equal_projection(theta, phi):
    khat = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    e1 = O.theta_hat(theta, phi)
    e2 = O.phi_hat(theta, phi)
    pol = (e1 + 0.6j * e2) / np.sqrt(1.36)
    aM, aN = O.plane_wave_coefficients(khat, pol, 8)
    gM, gN = O.project(lambda x: O.plane_wave(x, K, khat, pol), K, 8)
    assert max(np.abs(gM - aM).max(), np.abs(gN - aN).max()) < 1e-11 * np.abs(aM).max()


def test_plane_wave_coefficients_reproduce_jackson_10_55():
    """``e^{ikz} (x +- i y) = sum_l i^l sqrt(4 pi (2l+1)) [j_l X_{l,+-1} +- (1/k) curl(j_l X_{l,+-1})]``."""
    for s in (+1, -1):
        aM, aN = O.plane_wave_coefficients([0, 0, 1.0], [1, 1j * s, 0], 6)
        for l, m in O.modes(6):
            expect = (1j) ** l * np.sqrt(4 * np.pi * (2 * l + 1)) if m == s else 0.0
            assert aM[O.slot(l, m)] == pytest.approx(expect, abs=1e-12)
            assert aN[O.slot(l, m)] == pytest.approx(s * expect, abs=1e-12)


def test_projected_dipole_is_pure_l1(rng):
    k = 2 * np.pi / 1e-6
    p = (rng.normal(size=3) + 1j * rng.normal(size=3)) * 1e-30
    m = (rng.normal(size=3) + 1j * rng.normal(size=3)) * 1e-30 * O.C0
    pM, pN = O.project(lambda x: O.dipole_fields(x, k, p, m)[0], k, 3, "outgoing")
    assert np.abs(pM[3:]).max() < 1e-13 * np.abs(pM[:3]).max()
    assert np.abs(pN[3:]).max() < 1e-13 * np.abs(pN[:3]).max()
    pM_e, pN_e = O.project(lambda x: O.dipole_fields(x, k, p, 0 * m)[0], k, 2, "outgoing")
    assert np.abs(pM_e).max() < 1e-13 * np.abs(pN_e).max()          # an ED is purely electric (N)


# --------------------------------------------------------------------------
# Mie
# --------------------------------------------------------------------------


@pytest.mark.parametrize("m,x", [(1.5, 0.7), (4.0, 3.5), (1.33, 12.0), (3.614, 0.9)])
def test_mie_lossless_sphere_is_unitary(m, x):
    """``|1 + 2T| = 1`` with ``T = -a_n`` or ``-b_n`` -- the sign convention of the T-matrix."""
    a, b = O.mie(m, x, 20)
    assert np.allclose(np.abs(1 - 2 * a), 1, atol=1e-12)
    assert np.allclose(np.abs(1 - 2 * b), 1, atol=1e-12)
    xs = O.mie_cross_sections(m, x, 1.0, 20)
    assert xs["abs"] == pytest.approx(0.0, abs=1e-12 * xs["ext"])


def test_mie_rayleigh_limit():
    """``a_1 ~ -(2i/3) x^3 (m^2 - 1)/(m^2 + 2)``, ``b_1 = O(x^5)``."""
    for m in (1.5, 4.0 + 0.3j):
        x = 1e-3
        a, b = O.mie(m, x, 3)
        assert a[0] == pytest.approx(-2j / 3 * x**3 * (m**2 - 1) / (m**2 + 2), rel=1e-5)
        assert abs(b[0]) < 1e-12 and abs(a[1]) < 1e-12


@pytest.mark.parametrize("m,x", [(1.5, 0.7), (4.0 + 0.05j, 3.5), (1.33 + 0.01j, 8.0)])
def test_mie_matches_direct_riccati_bessel_formula(m, x):
    """BHMIE recurrence vs the textbook ``a_n = (m psi_n(mx) psi_n'(x) - psi_n(x) psi_n'(mx)) / (...)``."""
    nmax = 12
    n = np.arange(1, nmax + 1)

    def psi(z):
        return z * spherical_jn(n, z), spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)

    px, dpx = psi(x)
    pm, dpm = psi(m * x)
    h = spherical_jn(n, x) + 1j * spherical_yn(n, x)
    dh = spherical_jn(n, x, derivative=True) + 1j * spherical_yn(n, x, derivative=True)
    xi, dxi = x * h, h + x * dh
    a_ref = (m * pm * dpx - px * dpm) / (m * pm * dxi - xi * dpm)
    b_ref = (pm * dpx - m * px * dpm) / (pm * dxi - m * xi * dpm)
    a, b = O.mie(m, x, nmax)
    assert np.abs(a - a_ref).max() < 1e-12 and np.abs(b - b_ref).max() < 1e-12


def test_mie_absorbing_sphere_is_passive():
    xs = O.mie_cross_sections(4.0 + 0.05j, 3.5, 2.0, 15)
    assert xs["ext"] > xs["sca"] > 0 and xs["abs"] > 0
    assert xs["pr"] == pytest.approx(xs["ext"] - xs["gsca"])


def test_mie_scatter_applies_minus_b_minus_a(rng):
    aM, aN = O.random_coefficients(rng, 3)
    pM, pN = O.mie_scatter(aM, aN, 1.5, 1.2)
    a, b = O.mie(1.5, 1.2, 3)
    for l, m in O.modes(3):
        assert pM[O.slot(l, m)] == pytest.approx(-b[l - 1] * aM[O.slot(l, m)])
        assert pN[O.slot(l, m)] == pytest.approx(-a[l - 1] * aN[O.slot(l, m)])


# --------------------------------------------------------------------------
# flux_mp
# --------------------------------------------------------------------------


def test_flux_mp_matches_swforce_flux_at_moderate_values(rng):
    """The independent 60-digit loops reproduce the library's vectorised sums where doubles suffice."""
    pytest.importorskip("mpmath")
    from swforce.force import _flux_sums      # the packed sums (internal)

    for lmax in (1, 2, 5, 8):
        cM, cN = O.random_coefficients(rng, lmax)
        mine = [complex(v) for v in O.flux_mp(cM, cN)]
        lib = [complex(v) for v in _flux_sums(cM, cN)]
        scale = np.sum(np.abs(cM) ** 2 + np.abs(cN) ** 2)
        assert np.abs(np.array(mine) - np.array(lib)).max() < 1e-13 * scale
