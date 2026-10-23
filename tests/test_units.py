"""Dimensionless sums -> SI and cross sections, against independent physics.

``F = eps0 n^2 |E|^2 f / (8 k^2)``, ``T, S = eps0 n^2 |E|^2 (t, s) / (8 k^3)``
(``k`` in the host).  Checked against: the Bohren-Huffman Mie series
(``F_z = (n/c) I C_pr``; ``T_z = sigma I C_abs / omega`` in circular
polarisation), the SI Maxwell stress tensor with ``eps = n^2 eps0``, and the
constants of ``scipy.constants``.  The oracles use the same CODATA 2022
``eps0`` as the library, so the tolerances are rounding and quadrature level.
"""

from __future__ import annotations

import numpy as np
import pytest

import oracles as O
import swforce as sf
from swforce import units


def test_constants_are_si_2019_and_codata_2022():
    assert units.C0 == 299792458.0
    assert units.EPS0 == 8.8541878188e-12
    assert units.MU0 * units.EPS0 * units.C0**2 == pytest.approx(1.0, rel=1e-15)
    assert units.MU0 == pytest.approx(1.25663706127e-6, rel=1e-11)       # CODATA 2022, rounded to its 12 digits
    constants = pytest.importorskip("scipy.constants")
    if constants.epsilon_0 != 8.8541878188e-12:
        pytest.skip("scipy older than 1.15 carries CODATA 2018")
    assert units.EPS0 == constants.epsilon_0 and units.C0 == constants.c


def test_cross_section_is_f_over_4k2():
    wavelength, n = 633.0, 1.33
    k = 2 * np.pi * n / wavelength
    f = np.array([1.5, -2.0, 3.25])
    assert np.array_equal(sf.cross_section(f, wavelength, n), f / (4 * k**2))
    assert sf.cross_section(2.0, 2 * np.pi / 0.5) == pytest.approx(2.0)
    assert np.ndim(sf.cross_section(1.0, 900.0)) == 0


def mie_case(wavelength, radius, n_particle, n_medium, pol, khat=(0.0, 0.0, 1.0)):
    """``(ForceTorque, cross sections, k)`` for a sphere in a unit-amplitude plane wave (lengths in nm)."""
    k = 2 * np.pi * n_medium / wavelength
    x = k * radius
    m = n_particle / n_medium
    lmax = int(x + 4 * x ** (1 / 3) + 3)
    aM, aN = O.plane_wave_coefficients(khat, pol, lmax)
    pM, pN = O.mie_scatter(aM, aN, m, x)
    return sf.force_torque(aM, aN, pM, pN), O.mie_cross_sections(m, x, k, lmax), k


@pytest.mark.parametrize("n_medium", [1.0, 1.33])
def test_force_is_radiation_pressure(n_medium):
    """``f_z / (4 k^2) = C_pr`` and ``F_z = (n/c) I C_pr`` with ``C_pr`` from the Bohren-Huffman series."""
    wavelength, intensity = 700.0, 3.7e9
    ft, xs, k = mie_case(wavelength, 150.0, 1.9 + 0.01j, n_medium, (1.0, 0.0, 0.0))
    assert sf.cross_section(ft.force[2], wavelength, n_medium) == pytest.approx(xs["pr"], rel=1e-10)
    got = sf.to_si(ft, wavelength, intensity=intensity, index_medium=n_medium).force
    assert got[2] == pytest.approx(n_medium / O.C0 * intensity * xs["pr"] * 1e-18, rel=1e-10)
    assert np.abs(got[:2]).max() < 1e-12 * abs(got[2])


@pytest.mark.parametrize("n_medium", [1.0, 1.33])
def test_amplitude_and_intensity_give_the_same_newtons(n_medium):
    """``F = eps0 n^2 |E|^2 f / (8 k^2)`` (k in 1/m) and equals the intensity form at ``I = n eps0 c |E|^2 / 2``."""
    wavelength, amplitude = 532.0, 2.5e6
    ft, xs, k = mie_case(wavelength, 90.0, 1.5, n_medium, (0.0, 1.0, 0.0))
    got = sf.to_si(ft, wavelength, amplitude=amplitude, index_medium=n_medium)
    k_m = k * 1e9
    assert got.force[2] == pytest.approx(O.EPS0 * n_medium**2 * amplitude**2 * ft.force[2] / (8 * k_m**2), rel=1e-14)
    intensity = 0.5 * O.EPS0 * O.C0 * n_medium * amplitude**2
    assert got.force[2] == pytest.approx(n_medium / O.C0 * intensity * xs["pr"] * 1e-18, rel=1e-10)
    by_i = sf.to_si(ft, wavelength, intensity=intensity, index_medium=n_medium)
    for x, y in zip(got, by_i):
        assert np.abs(x - y).max() <= 1e-14 * np.abs(x).max()
    assert sf.to_si(ft, wavelength, amplitude=-amplitude * 1j, index_medium=n_medium).force[2] == got.force[2]


@pytest.mark.parametrize("sigma", [+1, -1])
@pytest.mark.parametrize("n_medium", [1.0, 1.33])
def test_torque_is_absorbed_angular_momentum(sigma, n_medium):
    """Absorbing sphere, circular polarisation of helicity ``sigma``: ``T_z = sigma I C_abs / omega``."""
    wavelength, intensity = 800.0, 1.0e10
    pol = np.array([1.0, 1j * sigma, 0.0]) / np.sqrt(2)
    ft, xs, k = mie_case(wavelength, 200.0, 2.0 + 0.3j, n_medium, pol)
    assert sf.cross_section(ft.torque[2], wavelength, n_medium) == pytest.approx(sigma * xs["abs"], rel=1e-10)
    omega = 2 * np.pi * O.C0 / (wavelength * 1e-9)
    got = sf.to_si(ft, wavelength, intensity=intensity, index_medium=n_medium).torque
    assert got[2] == pytest.approx(sigma * intensity * xs["abs"] * 1e-18 / omega, rel=1e-10)
    assert np.abs(got[:2]).max() < 1e-12 * abs(got[2])


@pytest.mark.parametrize("n_medium", [1.0, 1.33, 2.0])
def test_si_force_and_torque_in_a_medium_match_the_stress_tensor(n_medium, rng):
    """Random coefficients as SI fields (V/m), stress tensor with ``eps = n^2 eps0``, ``Z = Z0 / n``.

    Catches a forgotten ``n^2`` (or ``n``) in the conversion.
    """
    wavelength = 640.0                                        # nm, vacuum
    k_m = 2 * np.pi * n_medium / (wavelength * 1e-9)
    a = O.random_coefficients(rng, 3)
    p = O.random_coefficients(rng, 3)
    ft = sf.force_torque(*a, *p)
    Z0 = np.sqrt(O.MU0 / O.EPS0)
    F, T = O.stress_force_torque(lambda x: O.fields(a, p, x, k_m, n_medium, Z0=Z0), 2.5 / k_m,
                                 eps=n_medium**2 * O.EPS0, mu=O.MU0, n_theta=8, n_phi=16)
    got = sf.to_si(ft, wavelength, index_medium=n_medium)
    assert np.abs(got.force - F).max() < 1e-12 * np.linalg.norm(F)
    assert np.abs(got.torque - T).max() < 1e-12 * np.linalg.norm(T)


def test_spin_carries_the_torque_factor_and_scales_with_the_field(rng):
    """``S`` and ``T`` share ``eps0 n^2 |E|^2 / (8 k^3)``; everything scales as ``|E|^2``."""
    f, t, s = (rng.normal(size=3) for _ in range(3))
    ft = sf.ForceTorque(f, t, s)
    one = sf.to_si(ft, 900.0, index_medium=1.5)
    assert np.array_equal(sf.to_si(sf.ForceTorque(f, s, t), 900.0, index_medium=1.5).spin, one.torque)
    assert np.array_equal(sf.to_si(ft, 900.0, amplitude=1.0, index_medium=1.5).force, one.force)
    three = sf.to_si(ft, 900.0, amplitude=3.0, index_medium=1.5)
    for x, y in zip(three, one):
        assert np.abs(x - 9 * y).max() <= 1e-15 * np.abs(9 * y).max()


@pytest.mark.parametrize("n", [1.33 + 0.01j, 1.33 + 0j, 0.0, -1.0])
def test_lossy_or_nonpositive_host_and_bad_wavelength_are_rejected(n):
    """The stress-tensor conversion holds for a lossless host only: no silent use of a complex index."""
    ft = sf.ForceTorque(np.ones(3), np.ones(3), np.ones(3))
    with pytest.raises(ValueError, match="index_medium"):
        sf.to_si(ft, 900.0, index_medium=n)
    with pytest.raises(ValueError, match="index_medium"):
        sf.cross_section(1.0, 900.0, n)
    with pytest.raises(ValueError, match="wavelength"):
        units.si_scale(-900.0)


def test_amplitude_and_intensity_together_are_rejected():
    ft = sf.ForceTorque(np.ones(3), np.ones(3), np.ones(3))
    with pytest.raises(ValueError, match="not both"):
        sf.to_si(ft, 900.0, amplitude=1.0, intensity=1.0)


def test_to_si_is_si_scale_times_the_components(rng):
    """``si_scale`` (for raw vectors: cluster totals, torques with a transport term) is what ``to_si`` applies."""
    ft = sf.ForceTorque(*(rng.normal(size=3) for _ in range(3)))
    for kw in (dict(), dict(amplitude=2.0, index_medium=1.33), dict(intensity=1e10, index_medium=1.5)):
        f, t = units.si_scale(700.0, **kw)
        si = sf.to_si(ft, 700.0, **kw)
        assert np.array_equal(si.force, f * ft.force)
        assert np.array_equal(si.torque, t * ft.torque) and np.array_equal(si.spin, t * ft.spin)
        assert t == f / (2 * np.pi * kw.get("index_medium", 1.0) / (700.0 * units.NM))
    with pytest.raises(ValueError, match="not both"):
        units.si_scale(700.0, amplitude=1.0, intensity=1.0)
