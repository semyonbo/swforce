"""The OTT adapter: OTT-Fixed coefficients go in unchanged, zero-padded to one ``Nmax``.

OTT-Fixed (https://github.com/semyonbo/ott, FIXES.md FIX 1) carries the Condon-Shortley phase,
so its ordering, normalisation and phase all match the sums.

``tests/data/ott_fixed_validation.json`` holds ten OTT-Fixed cases -- sphere,
spheroid, cylinder and cube in plane waves and focused beams, with OTT-Fixed's
own ``ott.forcetorque`` result -- dumped from MATLAB.  Unpatched (upstream) OTT
data need a ``(-1)^m`` rephasing first and are not tested here.
Pure numpy, runs everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import oracles as O
import swforce as sf
from swforce.adapters import ott

CASES = json.loads((Path(__file__).resolve().parent / "data" / "ott_fixed_validation.json").read_text())["cases"]


def ott_fixed(case):
    """``(a, b, p, q)`` as OTT-Fixed's ``getCoefficients`` returned them, and its own force, torque and spin."""
    vec = lambda k: np.asarray(case[k + "_re"]) + 1j * np.asarray(case[k + "_im"])   # noqa: E731
    return tuple(vec(k) for k in "abpq"), np.asarray(case["f"]), np.asarray(case["t"]), np.asarray(case["s"])


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_ott_fixed_force_torque_and_spin_equal_its_own_forcetorque(case):
    """The coefficients of OTT-Fixed, unchanged, give OTT-Fixed's own ``ott.forcetorque`` (an independent
    implementation of the same Farsund sums in MATLAB): force, torque and spin to ~1e-14 of ``|F|``.
    The spin agrees since OTT-Fixed FIX 4 (``fb543ab``); before it OTT returned ``(-S_x, S_y, -S_z)``."""
    (a, b, p, q), f_ott, t_ott, s_ott = ott_fixed(case)
    ft = ott.force_torque(a, b, p, q)
    scale = np.linalg.norm(f_ott)
    assert np.abs(ft.force - f_ott).max() < 1e-13 * scale
    assert np.abs(ft.torque - t_ott).max() < 1e-13 * scale
    assert np.abs(ft.spin - s_ott).max() < 1e-13 * scale


def test_some_ott_fixed_cases_have_a_transverse_force():
    """The cases are not all symmetric: seven of ten have a transverse force of 1.6-86 % of ``|F|``, so a
    wrong phase (which flips ``F_x, F_y``) could not pass the test above."""
    transverse = [np.hypot(*ott_fixed(c)[1][:2]) / np.linalg.norm(ott_fixed(c)[1]) for c in CASES]
    assert sum(t > 0.01 for t in transverse) == 7


@pytest.mark.parametrize("scattered", [True, False])
def test_force_is_the_core_force_of_the_coefficients_as_they_are(scattered, rng):
    a, b, p, q = O.random_coefficients(rng, 5, count=4)
    got = ott.force_torque(a, b, p, q, scattered=scattered)
    ref = sf.force_torque(a, b, p, q, scattered=scattered)
    for x, y in zip(got, ref):
        assert np.array_equal(x, y)


def test_sphere_in_an_oblique_plane_wave_is_pushed_along_the_wavevector():
    """End to end in the Condon-Shortley convention (OTT-Fixed's): ``F || khat``."""
    theta, phi = 0.8, 0.5
    khat = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    nmax = 9
    aM, aN = O.plane_wave_coefficients(khat, O.theta_hat(theta, phi), nmax)
    pM, pN = O.mie_scatter(aM, aN, 1.6, 2.0)
    F = ott.force_torque(aM, aN, pM, pN).force
    assert np.linalg.norm(F - (F @ khat) * khat) < 1e-12 * np.linalg.norm(F)


def test_scattered_beam_with_a_lower_nmax_is_zero_padded(rng):
    """OTT truncates the scattered beam separately; padding with zeros changes nothing."""
    a, b = O.random_coefficients(rng, 6, count=2)
    p, q = O.random_coefficients(rng, 4, count=2)
    pad = lambda v: np.concatenate([v, np.zeros(a.size - v.size, complex)])         # noqa: E731
    got = ott.force_torque(a, b, p, q)
    ref = ott.force_torque(a, b, pad(p), pad(q))
    for x, y in zip(got, ref):
        assert np.array_equal(x, y)


def test_length_errors():
    with pytest.raises(ValueError, match="not a valid VSWF vector length"):
        ott.force_torque(np.ones(5), np.ones(5), np.ones(3), np.ones(3))


def test_public_api_is_one_function():
    assert ott.__all__ == ["force_torque"]
