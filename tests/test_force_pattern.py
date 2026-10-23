"""The flux sums as moments of the far-field pattern, and the ladder integrals behind them.

``flux(c)`` claims to be ``Q[c] = oint |A_c|^2 rhat dOmega`` with the far-field
pattern ``A_c(rhat) = sum_lm (-i)^l [cM X_lm + i cN rhat x X_lm]`` of the waves
``c`` (``oracles.pattern``).  Here that is checked by quadrature, independently
of the ladder formulas:

* momentum:          ``(Re fxy, Im fxy, fz) = oint |A_c|^2 rhat dOmega``;
* incoming waves:    ``B_c(rhat) = A_c(-rhat)`` and ``-oint |B_c|^2 rhat dOmega = +Q[c]``;
* interference:      ``B(u,v) + B(v,u) = 2 Re oint A_u* . A_v rhat dOmega``;
* spin:              ``(Re(i sxy), Im(i sxy), -sz) = oint -i A_c* x A_c dOmega``  (``A* . S A`` with the spin-1 matrices);
* angular momentum:  ``(Re txy, Im txy, tz) = oint A_c* . J A_c dOmega``, ``J = L + S``
  (``J_z`` spectrally in ``phi``, ``J_x, J_y`` from infinitesimal rotations of the pattern);
* the force itself:  ``force_torque`` = pattern moments of ``a`` minus those of ``a + 2p``.

Gauss-Legendre x uniform-phi quadrature is exact for these band-limited
integrands, so the tolerances are rounding level.  Finally the angular integrals
``K_lm``, ``J_z``, ``I_1``, ``I_2``, ``I_3`` behind the ladder sums and their
parity zeros, one by one.
"""

from __future__ import annotations

import numpy as np
import pytest

import oracles as O
import swforce as sf
from swforce.force import _bilinear_sums, _flux_sums  # the packed sums (internal)


def grid(lmax):
    return O.sphere_quadrature(lmax + 4, 2 * lmax + 8)


def momentum(A, th, ph, w):
    return np.sum(w[:, None] * np.sum(np.abs(A) ** 2, -1)[:, None] * O.rhat(th, ph), 0)


def spin(A, w):
    return np.real(np.sum(w[:, None] * (-1j) * np.cross(np.conj(A), A), 0))


def force_vec(sums):
    fz, fxy = sums[0], sums[1]
    return np.array([np.real(fxy), np.imag(fxy), np.real(fz)])


def spin_vec(sums):
    sz, sxy = sums[4], sums[5]
    return np.array([np.real(1j * sxy), np.imag(1j * sxy), -np.real(sz)])


def torque_vec(sums):
    tz, txy = sums[2], sums[3]
    return np.array([np.real(txy), np.imag(txy), np.real(tz)])


@pytest.mark.parametrize("lmax", [1, 2, 3, 4, 5, 6])
def test_flux_is_the_momentum_moment_of_the_pattern(lmax, rng):
    th, ph, w = grid(lmax)
    for _ in range(3):
        cM, cN = O.random_coefficients(rng, lmax)
        q = momentum(O.pattern(cM, cN, th, ph), th, ph, w)
        assert np.abs(force_vec(_flux_sums(cM, cN)) - q).max() < 1e-13 * np.sum(np.abs(cM) ** 2 + np.abs(cN) ** 2)


@pytest.mark.parametrize("lmax", [1, 3, 6])
def test_incoming_pattern_is_the_parity_image_and_brings_in_plus_Q(lmax, rng):
    """``B_c(rhat) = A_c(-rhat)``: the waves ``c h^(2)`` arrive from where ``c h^(1)`` would leave,
    so the momentum they bring in, ``-oint |B_c|^2 rhat``, is ``+Q[c]``."""
    th, ph, w = grid(lmax)
    cM, cN = O.random_coefficients(rng, lmax)
    Bc = O.pattern(cM, cN, th, ph, incoming=True)
    assert np.abs(Bc - O.pattern(cM, cN, np.pi - th, ph + np.pi)).max() < 1e-13 * np.abs(Bc).max()
    q_in = -momentum(Bc, th, ph, w)
    assert np.abs(q_in - force_vec(_flux_sums(cM, cN))).max() < 1e-13 * np.sum(np.abs(cM) ** 2 + np.abs(cN) ** 2)


@pytest.mark.parametrize("lmax", [1, 4, 6])
def test_interference_is_the_cross_moment_of_two_patterns(lmax, rng):
    th, ph, w = grid(lmax)
    (uM, uN), (vM, vN) = O.random_coefficients(rng, lmax), O.random_coefficients(rng, lmax)
    Au, Av = O.pattern(uM, uN, th, ph), O.pattern(vM, vN, th, ph)
    cross = 2 * np.sum(w[:, None] * np.real(np.sum(np.conj(Au) * Av, -1))[:, None] * O.rhat(th, ph), 0)
    pair = np.array(_bilinear_sums(uM, uN, vM, vN)) + np.array(_bilinear_sums(vM, vN, uM, uN))
    scale = np.sum(np.abs(np.r_[uM, uN]) ** 2) + np.sum(np.abs(np.r_[vM, vN]) ** 2)
    assert np.abs(force_vec(pair) - cross).max() < 1e-13 * scale
    assert np.abs(sf.interference(uM, uN, vM, vN).force - cross).max() < 1e-13 * scale
    spin_cross = np.real(np.sum(w[:, None] * (-1j) * (np.cross(np.conj(Au), Av) + np.cross(np.conj(Av), Au)), 0))
    assert np.abs(sf.interference(uM, uN, vM, vN).spin - spin_cross).max() < 1e-13 * scale


@pytest.mark.parametrize("lmax", [1, 2, 4, 6])
def test_spin_sums_are_the_spin_moment_of_the_pattern(lmax, rng):
    """``A* . S_k A = -i (A* x A)_k``: an independent check of the spin sums *and* of their packing."""
    th, ph, w = grid(lmax)
    for _ in range(3):
        cM, cN = O.random_coefficients(rng, lmax)
        s = spin(O.pattern(cM, cN, th, ph), w)
        assert np.abs(spin_vec(_flux_sums(cM, cN)) - s).max() < 1e-13 * np.sum(np.abs(cM) ** 2 + np.abs(cN) ** 2)


def _Jz(A, n_theta, n_phi):
    """``J_z A = -i dA/dphi + i zhat x A`` on the quadrature grid (``phi`` derivative by FFT, exact)."""
    Ag = A.reshape(n_theta, n_phi, 3)
    freq = np.fft.fftfreq(n_phi, d=1.0 / n_phi)
    dA = np.fft.ifft(1j * freq[None, :, None] * np.fft.fft(Ag, axis=1), axis=1).reshape(A.shape)
    return -1j * dA + 1j * np.cross([0.0, 0.0, 1.0], A)


def _rotation(axis, alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _J_expectation(cM, cN, axis, th, ph, w, h=1e-5):
    """``<A, J A> = i d/dalpha <A, U(alpha) A>`` at 0, ``(U A)(r) = R A(R^T r)`` -- central difference."""
    rh = O.rhat(th, ph)
    A = O.pattern(cM, cN, th, ph)
    vals = []
    for alpha in (h, -h):
        R = _rotation(axis, alpha)
        _, t2, p2 = O.to_spherical(rh @ R)                          # rows: R^T r
        UA = O.pattern(cM, cN, t2, p2) @ R.T
        vals.append(np.sum(w * np.sum(np.conj(A) * UA, -1)))
    return np.real(1j * (vals[0] - vals[1]) / (2 * h))


@pytest.mark.parametrize("lmax", [1, 3, 5])
def test_torque_sums_are_the_angular_momentum_moment_of_the_pattern(lmax, rng):
    n_theta, n_phi = lmax + 4, 2 * lmax + 8
    th, ph, w = O.sphere_quadrature(n_theta, n_phi)
    cM, cN = O.random_coefficients(rng, lmax)
    scale = np.sum(np.abs(cM) ** 2 + np.abs(cN) ** 2)
    A = O.pattern(cM, cN, th, ph)
    tz = np.real(np.sum(w * np.sum(np.conj(A) * _Jz(A, n_theta, n_phi), -1)))
    t = torque_vec(_flux_sums(cM, cN))
    assert abs(t[2] - tz) < 1e-13 * scale
    tx = _J_expectation(cM, cN, 0, th, ph, w)
    ty = _J_expectation(cM, cN, 1, th, ph, w)
    assert abs(t[0] - tx) < 1e-8 * scale and abs(t[1] - ty) < 1e-8 * scale


@pytest.mark.parametrize("form", ["quadratic", "bilinear"])
def test_force_torque_is_momentum_in_minus_momentum_out(form, rng):
    """``force_torque(a, p)`` = moments of the incoming waves ``a`` minus those of the outgoing ``a + 2p``."""
    lmax = 4
    th, ph, w = grid(lmax)
    a = O.random_coefficients(rng, lmax)
    p = O.random_coefficients(rng, lmax)
    out = tuple(x + 2 * y for x, y in zip(a, p))
    Aa, Ao = O.pattern(*a, th, ph), O.pattern(*out, th, ph)
    ft = sf.force_torque(*a, *p, form=form)
    scale = sum(np.sum(np.abs(c) ** 2) for c in (*a, *out))
    assert np.abs(ft.force - (momentum(Aa, th, ph, w) - momentum(Ao, th, ph, w))).max() < 1e-13 * scale
    assert np.abs(ft.spin - (spin(Aa, w) - spin(Ao, w))).max() < 1e-13 * scale


# --------------------------------------------------------------------------
# Ladder integrals
# --------------------------------------------------------------------------


def _integral(f, g, weight, w):
    return np.sum(w * weight * np.sum(np.conj(f) * g, -1))


def _same(x, y):
    return all(np.array_equal(u, v) for u, v in zip(x, y))


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
@pytest.mark.parametrize("lmax", [1, 3, 8])
def test_public_flux_is_the_packed_sums_as_vectors(lmax, scale, rng):
    """``sf.flux`` / ``sf.flux_bilinear`` are the internal sums unpacked, bit for bit; the quadratic force is
    ``flux(a) - flux(a + 2p)`` bit for bit.  To rounding: ``B(c, c) = flux(c)`` and the polarisation identity
    ``flux(u + v) = flux(u) + flux(v) + B(u, v) + B(v, u)``.  No hypothesis needed, so this runs everywhere."""
    from swforce.force import ForceTorque, _pack
    aM, aN, pM, pN = O.random_coefficients(rng, lmax, count=4, scale=scale)
    q, b = sf.flux(aM, aN), sf.flux_bilinear(aM, aN, pM, pN)
    assert isinstance(q, ForceTorque) and isinstance(b, ForceTorque)
    assert _same(q, _pack(*_flux_sums(aM, aN)))
    assert _same(b, _pack(*_bilinear_sums(aM, aN, pM, pN)))
    assert _same(sf.force_torque(aM, aN, pM, pN, form="quadratic"), sf.flux(aM, aN) - sf.flux(aM + 2 * pM, aN + 2 * pN))
    size = sum(np.sum(np.abs(c) ** 2) for c in (aM, aN, pM, pN))         # natural scale of a quadratic quantity
    for x, y in zip(sf.flux_bilinear(aM, aN, aM, aN), sf.flux(aM, aN)):
        assert np.abs(x - y).max() <= 1e-13 * size
    lhs = sf.flux(aM + pM, aN + pN)
    rhs = sf.flux(aM, aN) + sf.flux(pM, pN) + sf.flux_bilinear(aM, aN, pM, pN) + sf.flux_bilinear(pM, pN, aM, aN)
    for x, y in zip(lhs, rhs):
        assert np.abs(x - y).max() <= 1e-13 * size


@pytest.mark.parametrize("l", [1, 2, 3, 4, 5])
def test_ladder_angular_integrals(l):
    """The angular integrals the ladder coefficients stand for, all ``m``:

    * ``K_lm = oint X*_{l+1,m} . X_lm cos(theta) = sqrt(l(l+2)(l-m+1)(l+m+1)/((2l+1)(2l+3))) / (l+1)``
    * ``J_z  = oint X*_lm . (rhat x X_lm) cos(theta) = -i m / (l(l+1))``
    * ``I_1  = oint X*_{l+1,m+1} . X_lm sin(theta) e^{i phi} = -L_l sqrt((l+m+1)(l+m+2))``
    * ``I_2  = oint X*_lm . X_{l+1,m-1} sin(theta) e^{i phi} = +L_l sqrt((l-m+1)(l-m+2))``
    * ``I_3  = oint X*_{l,m+1} . (rhat x X_lm) sin(theta) e^{i phi} = -i sqrt((l-m)(l+m+1)) / (l(l+1))``
    * parity zeros ``oint X*_lm . X_lm cos(theta) = 0``, ``oint X*_{l+1,m} . (rhat x X_lm) cos(theta) = 0``

    with ``L_l = sqrt(l(l+2)/((2l+1)(2l+3))) / (l+1)``.
    """
    th, ph, w = O.sphere_quadrature(l + 4, 2 * l + 10)
    rh = O.rhat(th, ph)
    ct = np.cos(th)
    spl = np.sin(th) * np.exp(1j * ph)
    L_l = np.sqrt(l * (l + 2) / ((2 * l + 1) * (2 * l + 3))) / (l + 1)
    for m in range(-l, l + 1):
        x = O.X(l, m, th, ph)
        rx = np.cross(rh, x)
        checks = {
            "K": (_integral(O.X(l + 1, m, th, ph), x, ct, w),
                  np.sqrt(l * (l + 2) * (l - m + 1) * (l + m + 1) / ((2 * l + 1) * (2 * l + 3))) / (l + 1)),
            "J_z": (_integral(x, rx, ct, w), -1j * m / (l * (l + 1))),
            "I_1": (_integral(O.X(l + 1, m + 1, th, ph), x, spl, w), -L_l * np.sqrt((l + m + 1) * (l + m + 2))),
            "I_2": (_integral(x, O.X(l + 1, m - 1, th, ph), spl, w), L_l * np.sqrt((l - m + 1) * (l - m + 2))),
            "I_3": (_integral(O.X(l, m + 1, th, ph), rx, spl, w), -1j * np.sqrt((l - m) * (l + m + 1)) / (l * (l + 1))),
            "zero_XX": (_integral(x, x, ct, w), 0.0),
            "zero_X_rX": (_integral(O.X(l + 1, m, th, ph), rx, ct, w), 0.0),
        }
        for name, (num, ref) in checks.items():
            assert abs(num - ref) < 1e-14, (name, l, m, num, ref)
