"""Analytic physics: every number here comes from an independent formula, not from another code.

* radiation pressure of a Mie sphere = Bohren-Huffman ``C_pr`` (calibrates ``f = 4 k^2 C``);
* a sphere in a plane wave is pushed exactly along ``khat`` (transverse sums, Condon-Shortley);
* the recoil of an electric + magnetic dipole pair, coefficients *projected from the exact fields*;
* a perfect absorber ``T = -1/2``: ``F_z / |T_z| = N/(N+1)`` (truncation law, signs of force and torque);
* lossless sphere: no torque; absorbing sphere: torque = ``sigma C_abs``; spin flips with helicity;
* a sphere displaced in a standing wave: ``F_z = A sin(2 k z0)`` (coefficients by projection about the
  displaced centre);
* convergence in ``nmax`` from below.

Units: ``k = 1`` unless stated, so cross sections are in ``1/k^2``.  Plane-wave
coefficients from ``oracles.plane_wave_coefficients`` (itself checked against a
projection of the analytic field), T-matrix from the Bohren-Huffman oracle.
"""

from __future__ import annotations

import numpy as np
import pytest

import oracles as O
import swforce as sf
from swforce.force import _flux_sums  # the packed sums (internal)


def khat_of(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def circular(theta, phi, sigma):
    """Circular polarisation of helicity ``sigma`` about ``khat(theta, phi)``: ``(e_theta + i sigma e_phi)/sqrt 2``."""
    return (O.theta_hat(theta, phi) + 1j * sigma * O.phi_hat(theta, phi)) / np.sqrt(2)


def mie_force(ka, m, pol, theta=0.0, phi=0.0, extra=4, form="quadratic", nmax=None):
    nmax = nmax or int(ka + 4 * ka ** (1 / 3) + extra)
    aM, aN = O.plane_wave_coefficients(khat_of(theta, phi), pol, nmax)
    pM, pN = O.mie_scatter(aM, aN, m, ka)
    return sf.force_torque(aM, aN, pM, pN, form=form), O.mie_cross_sections(m, ka, 1.0, nmax)


# --------------------------------------------------------------------------
# Radiation pressure
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ka", [0.5, 1.0, 1.6, 2.2, 3.5, 5.0, 8.0])
@pytest.mark.parametrize("m", [1.5, 4.0, 4.0 + 0.05j])
@pytest.mark.parametrize("form", ["quadratic", "bilinear"])
def test_axial_force_is_mie_radiation_pressure(ka, m, form):
    """``f_z / (4 k^2) = C_pr``: the force sums and the Bohren-Huffman series are independent algebra."""
    ft, xs = mie_force(ka, m, (1.0, 0.0, 0.0), form=form)
    assert ft.force[2] / 4 == pytest.approx(xs["pr"], rel=1e-10)
    assert np.abs(ft.force[:2]).max() < 1e-12 * ft.force[2]


@pytest.mark.parametrize("theta,phi", [(0.6, 1.1), (1.2, -0.4), (np.pi / 2, 0.8), (2.5, 2.0), (3.0, -2.7), (np.pi, 0.3)])
@pytest.mark.parametrize("m", [4.0, 1.5 + 0.1j])
def test_force_is_parallel_to_the_wavevector_at_oblique_incidence(theta, phi, m):
    """Rotational symmetry about ``khat`` allows no transverse force; the magnitude stays ``4 C_pr``.
    A sign error in the transverse sums (e.g. a missing Condon-Shortley phase) breaks the first."""
    k = khat_of(theta, phi)
    pol = (O.theta_hat(theta, phi) + 0.3j * O.phi_hat(theta, phi)) / np.sqrt(1.09)
    ft, xs = mie_force(2.2, m, pol, theta, phi)
    F = ft.force
    assert np.linalg.norm(F - (F @ k) * k) < 1e-12 * np.linalg.norm(F)
    assert F @ k / 4 == pytest.approx(xs["pr"], rel=1e-10)


# --------------------------------------------------------------------------
# Dipole recoil
# --------------------------------------------------------------------------


@pytest.mark.parametrize("form", ["quadratic", "bilinear"])
def test_recoil_of_an_electric_and_magnetic_dipole(form, rng):
    """Coefficients projected from the exact ED + MD fields (Jackson 9.18, 9.36), SI:

        F = -(eps0 / 2 k^2) Q[p] = -(k^4 / 12 pi c eps0) Re(p x m*),
        T = eps0 t / (8 k^3)     = -(k^3 / 12 pi eps0) Im(p* x p) - (mu0 k^3 / 12 pi) Im(m* x m),

    and a lone ED or MD feels no recoil force.  Pins the normalisation of ``Q``.
    """
    k = 2 * np.pi / 800e-9
    zero = np.zeros(3)
    for _ in range(2):
        p = (rng.normal(size=3) + 1j * rng.normal(size=3)) * 1e-31
        m = (rng.normal(size=3) + 1j * rng.normal(size=3)) * 1e-31 * O.C0
        pM, pN = O.project(lambda x: O.dipole_fields(x, k, p, m)[0], k, 2, "outgoing")
        a0 = np.zeros_like(pM)
        ft = sf.force_torque(a0, a0, pM, pN, form=form)
        F_ref = -k**4 / (12 * np.pi * O.C0 * O.EPS0) * np.real(np.cross(p, np.conj(m)))
        T_ref = (-k**3 / (12 * np.pi * O.EPS0) * np.imag(np.cross(np.conj(p), p))
                 - O.MU0 * k**3 / (12 * np.pi) * np.imag(np.cross(np.conj(m), m)))
        assert np.abs(O.EPS0 * ft.force / (8 * k**2) - F_ref).max() < 1e-11 * np.linalg.norm(F_ref)
        assert np.abs(O.EPS0 * ft.torque / (8 * k**3) - T_ref).max() < 1e-11 * np.linalg.norm(T_ref)
        fz, fxy = _flux_sums(pM, pN)[:2]
        Q = np.array([fxy.real, fxy.imag, fz])
        assert np.abs(-O.EPS0 / (2 * k**2) * Q - F_ref).max() < 1e-11 * np.linalg.norm(F_ref)
        for pp, mm in ((p, zero), (zero, m)):
            qM, qN = O.project(lambda x: O.dipole_fields(x, k, pp, mm)[0], k, 2, "outgoing")
            alone = sf.force_torque(a0, a0, qM, qN, form=form).force
            assert np.linalg.norm(alone) < 1e-13 * np.sum(np.abs(np.r_[qM, qN]) ** 2)


# --------------------------------------------------------------------------
# Perfect absorber, torque, spin
# --------------------------------------------------------------------------


@pytest.mark.parametrize("N", [1, 2, 3, 6, 10])
@pytest.mark.parametrize("sigma", [+1, -1])
@pytest.mark.parametrize("theta,phi", [(0.0, 0.0), (0.9, -0.6)])
def test_perfect_absorber_truncation_law(N, sigma, theta, phi):
    """``T = -1/2 I`` swallows every incoming wave (``a + 2p = 0``).  For a circular plane wave
    truncated at ``N``: ``F.khat / |T| = N/(N+1)`` (dimensionless sums), ``F`` along ``khat``,
    ``T`` along ``sigma khat`` -- the signs of force and torque and the ``l -> l+1`` coupling."""
    k = khat_of(theta, phi)
    aM, aN = O.plane_wave_coefficients(k, circular(theta, phi, sigma), N)
    for form in ("quadratic", "bilinear"):
        ft = sf.force_torque(aM, aN, -aM / 2, -aN / 2, form=form)
        Fk, Tk = ft.force @ k, ft.torque @ k
        assert Fk > 0 and np.sign(Tk) == sigma
        assert Fk / abs(Tk) == pytest.approx(N / (N + 1), rel=1e-12)
        assert np.linalg.norm(ft.force - Fk * k) < 1e-12 * Fk
        assert np.linalg.norm(ft.torque - Tk * k) < 1e-12 * abs(Tk)
        assert abs(Tk) == pytest.approx(4 * np.pi * N * (N + 2), rel=1e-12)


@pytest.mark.parametrize("theta,phi", [(0.0, 0.0), (0.6, 1.1), (2.5, 2.0)])
@pytest.mark.parametrize("pol", ["linear", "circular", "elliptic"])
def test_lossless_sphere_feels_no_torque(theta, phi, pol):
    """A lossless sphere returns every ``m`` channel's angular momentum (``|1 + 2t| = 1``, diagonal T)."""
    e = {"linear": O.theta_hat(theta, phi), "circular": circular(theta, phi, 1),
         "elliptic": (O.theta_hat(theta, phi) + 0.5j * O.phi_hat(theta, phi)) / np.sqrt(1.25)}[pol]
    ft, _ = mie_force(3.5, 4.0, e, theta, phi)
    assert np.abs(ft.torque).max() < 1e-12 * np.linalg.norm(ft.force)


@pytest.mark.parametrize("theta,phi", [(0.0, 0.0), (0.7, 1.1), (1.3, -0.5)])
@pytest.mark.parametrize("sigma", [+1, -1])
def test_absorbing_sphere_torque_is_absorbed_angular_momentum(theta, phi, sigma):
    """``t / (4 k^2) = sigma C_abs khat``: each absorbed photon deposits ``sigma hbar``."""
    k = khat_of(theta, phi)
    ft, xs = mie_force(3.5, 4.0 + 0.05j, circular(theta, phi, sigma), theta, phi)
    assert np.abs(ft.torque / 4 - sigma * xs["abs"] * k).max() < 1e-10 * xs["abs"]


@pytest.mark.parametrize("theta,phi", [(0.0, 0.0), (0.7, 1.1), (2.2, 2.0)])
@pytest.mark.parametrize("m", [4.0, 4.0 + 0.05j])
def test_spin_is_along_khat_and_flips_with_helicity(theta, phi, m):
    k = khat_of(theta, phi)
    plus, _ = mie_force(3.5, m, circular(theta, phi, +1), theta, phi)
    minus, _ = mie_force(3.5, m, circular(theta, phi, -1), theta, phi)
    assert np.abs(plus.spin + minus.spin).max() < 1e-10 * np.linalg.norm(plus.spin)
    assert plus.spin @ k / np.linalg.norm(plus.spin) == pytest.approx(1.0, abs=1e-10)
    # linear polarisation carries no spin
    lin, _ = mie_force(3.5, m, O.theta_hat(theta, phi), theta, phi)
    assert np.abs(lin.spin).max() < 1e-10 * np.linalg.norm(plus.spin)


def test_spin_converges_to_torque_for_a_rayleigh_absorber():
    """A plane wave carries angular momentum only as spin, so an absorber that scatters nothing takes it
    all through the spin channel: ``spin_z / torque_z -> 1`` -- this fixes the axial sign of the spin."""
    ratios = []
    for ka in (0.2, 0.05, 0.01):
        ft, xs = mie_force(ka, 1.5 + 0.3j, circular(0.0, 0.0, 1), nmax=4)
        assert xs["sca"] < 1e-2 * xs["abs"]
        ratios.append(ft.spin[2] / ft.torque[2])
    assert ratios[-1] == pytest.approx(1.0, abs=1e-5)
    assert abs(ratios[-1] - 1) < abs(ratios[0] - 1)


def test_lossless_sphere_converts_spin_to_orbital_angular_momentum():
    """Zero torque but non-zero spin transfer: the spin sums are not the torque sums in disguise."""
    ft, _ = mie_force(3.5, 4.0, circular(0.0, 0.0, 1))
    assert abs(ft.torque[2]) < 1e-12 * np.linalg.norm(ft.force)
    assert abs(ft.spin[2]) > 1e-3 * np.linalg.norm(ft.force)


# --------------------------------------------------------------------------
# Standing wave: a displaced centre
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ka,m", [(0.5, 1.5), (1.0, 4.0)])
def test_standing_wave_force_follows_sin_2kz(ka, m):
    """Sphere at ``z0`` in ``E = x (e^{ikz} + e^{-ikz})``, coefficients projected about ``z0``:
    ``F_z(z0) = A sin(2 k z0)`` exactly (the only ``z0`` dependence is the relative phase
    ``e^{2ikz0}``), no transverse force, and a small high-index sphere is pulled to the antinode."""
    lmax = 6
    z0s = np.linspace(0.0, np.pi, 7)[1:-1] / 2 + 0.013              # k = 1, avoid the exact nodes
    fz = []
    for z0 in z0s:
        aM, aN = O.project(lambda x: O.standing_wave(x, 1.0, z0=z0), 1.0, lmax)
        pM, pN = O.mie_scatter(aM, aN, m, ka)
        ft = sf.force_torque(aM, aN, pM, pN, form="bilinear")
        assert np.abs(ft.force[:2]).max() < 1e-10 * np.abs(ft.force[2])
        fz.append(ft.force[2])
    fz = np.array(fz)
    A = np.sum(fz * np.sin(2 * z0s)) / np.sum(np.sin(2 * z0s) ** 2)
    assert np.abs(fz - A * np.sin(2 * z0s)).max() < 1e-10 * abs(A)
    if ka < 1:
        assert A < 0                                             # restoring force towards z0 = 0 (|E| maximum)
    # at the antinode and at the node the force vanishes
    for z0 in (0.0, np.pi / 2):
        aM, aN = O.project(lambda x: O.standing_wave(x, 1.0, z0=z0), 1.0, lmax)
        pM, pN = O.mie_scatter(aM, aN, m, ka)
        assert abs(sf.force_torque(aM, aN, pM, pN).force[2]) < 1e-10 * abs(A)


# --------------------------------------------------------------------------
# Convergence
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ka,m", [(3.0, 4.0), (2.2, 1.5), (5.0, 4.0 + 0.05j), (8.0, 1.5)])
def test_mie_force_converges_from_below_in_nmax(ka, m):
    """Truncation drops the ``(nmax, nmax+1)`` terms; the force approaches ``4 C_pr`` from below,
    monotonically, and is converged to rounding a few orders past Wiscombe's rule."""
    converged = O.mie_cross_sections(m, ka, 1.0, 60)["pr"]
    errors = []
    for nmax in range(1, sf.ka2nmax(ka) + 8):
        ft, _ = mie_force(ka, m, (1.0, 0.0, 0.0), nmax=nmax)
        errors.append(ft.force[2] / 4 - converged)
    errors = np.array(errors) / converged
    big = np.abs(errors) > 1e-12
    assert np.all(errors[big] < 0)                                # from below
    assert np.all(np.diff(np.abs(errors[big])) < 0)               # monotonically
    assert abs(errors[-1]) < 1e-12
    assert abs(errors[sf.ka2nmax(ka) - 1]) < 1e-5                # Wiscombe's rule is already good to 1e-5
