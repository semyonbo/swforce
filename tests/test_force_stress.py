"""The sums against a direct numerical Maxwell stress integral -- physics without the ladder formulas.

For arbitrary coefficients the truncated field ``E = sum a (regular) + p (outgoing)``
is an exact solution of Maxwell's equations away from the origin, so
``oint T . n dA`` and ``oint r x (T . n) dA`` do not depend on the radius and
must equal ``eps0 n^2 f / (8 k^2)`` and ``eps0 n^2 t / (8 k^3)``.  Units:
``eps0 = mu0 = 1`` (``Z0 = 1``), medium ``eps = n^2``, ``mu = 1``.  The angular
quadrature is exact for the band-limited integrand, so the only errors are
rounding -- larger close to the origin, where the ``h_l`` near fields cancel.
"""

from __future__ import annotations

import numpy as np
import pytest

import oracles as O
import swforce as sf

K = 2 * np.pi                  # wavelength 1


def stress(a, p, r, n):
    lmax = int(round(np.sqrt(a[0].size + 1) - 1))
    return O.stress_force_torque(lambda x: O.fields(a, p, x, K, n), r, eps=n**2, mu=1.0,
                                 n_theta=lmax + 4, n_phi=2 * lmax + 6)


def tolerance(r, lmax):
    """Rounding budget: close in the near field ``|h_l(kr)|^2`` swamps the result; far out the
    torque integrand ``r (T . n)`` exceeds the torque by ``~ kr``."""
    from scipy.special import spherical_yn
    kr = K * r
    return 1e-13 * max(1.0, abs(spherical_yn(lmax, kr)) ** 2 * kr**2 / 10, kr / 50)


@pytest.mark.parametrize("lmax", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("form", ["quadratic", "bilinear"])
def test_random_coefficients_force_and_torque(lmax, form, rng):
    """Every ladder formula and the ``x + i y`` packing at once, for arbitrary coefficients."""
    for _ in range(2):
        a = O.random_coefficients(rng, lmax)
        p = O.random_coefficients(rng, lmax, scale=0.7)
        ft = sf.force_torque(*a, *p, form=form)
        F, T = stress(a, p, 1.3, 1.0)
        assert np.abs(ft.force / (8 * K**2) - F).max() < 1e-12 * np.linalg.norm(F)
        assert np.abs(ft.torque / (8 * K**3) - T).max() < 1e-12 * np.linalg.norm(T)


@pytest.mark.parametrize("r", [0.3, 1.0, 10.0, 100.0])
@pytest.mark.parametrize("lmax", [2, 6])
def test_stress_integral_is_independent_of_the_radius(r, lmax, rng):
    """r = 0.3, 1, 10, 100 wavelengths: the truncated series is exact off the origin."""
    a = O.random_coefficients(rng, lmax)
    p = O.random_coefficients(rng, lmax, scale=0.5)
    ft = sf.force_torque(*a, *p, form="bilinear")
    F, T = stress(a, p, r, 1.0)
    tol = tolerance(r, lmax)
    assert np.abs(ft.force / (8 * K**2) - F).max() < tol * np.linalg.norm(F)
    assert np.abs(ft.torque / (8 * K**3) - T).max() < tol * np.linalg.norm(T)


@pytest.mark.parametrize("n", [1.33, 2.5])
@pytest.mark.parametrize("form", ["quadratic", "bilinear"])
def test_medium_scales_as_n_squared(n, form, rng):
    """In a medium ``F = eps0 n^2 f / (8 k^2)`` with ``k`` the wavenumber in the medium."""
    k = K * n
    a = O.random_coefficients(rng, 4)
    p = O.random_coefficients(rng, 4)
    ft = sf.force_torque(*a, *p, form=form)
    F, T = O.stress_force_torque(lambda x: O.fields(a, p, x, k, n), 1.1 / n, eps=n**2, n_theta=8, n_phi=14)
    assert np.abs(n**2 * ft.force / (8 * k**2) - F).max() < 1e-12 * np.linalg.norm(F)
    assert np.abs(n**2 * ft.torque / (8 * k**3) - T).max() < 1e-12 * np.linalg.norm(T)


def test_total_outgoing_input_matches_the_stress_tensor(rng):
    """``scattered=False``: the outgoing coefficients are the total ``a + 2p``."""
    a = O.random_coefficients(rng, 3)
    p = O.random_coefficients(rng, 3)
    total = tuple(x + 2 * y for x, y in zip(a, p))
    F, T = stress(a, p, 0.8, 1.0)
    for form in ("quadratic", "bilinear"):
        ft = sf.force_torque(*a, *total, scattered=False, form=form)
        assert np.abs(ft.force / (8 * K**2) - F).max() < 1e-12 * np.linalg.norm(F)
        assert np.abs(ft.torque / (8 * K**3) - T).max() < 1e-12 * np.linalg.norm(T)


def test_single_mode_pairs_one_at_a_time():
    """Each ``(M/N, l, m)`` x ``(M/N, l', m')`` pair on its own: isolates any single wrong ladder entry."""
    lmax = 3
    n = lmax * (lmax + 2)
    worst = 0.0
    rng = np.random.default_rng(11)
    for i in range(2 * n):
        for j in range(2 * n):
            a = [np.zeros(n, complex), np.zeros(n, complex)]
            p = [np.zeros(n, complex), np.zeros(n, complex)]
            a[i // n][i % n] = 1.0
            p[j // n][j % n] = np.exp(1j * rng.uniform(0, 2 * np.pi))
            ft = sf.force_torque(*a, *p, form="bilinear")
            F, T = stress(tuple(a), tuple(p), 1.0, 1.0)
            worst = max(worst, np.abs(ft.force / (8 * K**2) - F).max(), np.abs(ft.torque / (8 * K**3) - T).max() * K)
    assert worst < 1e-13
