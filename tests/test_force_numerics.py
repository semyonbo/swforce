"""Floating-point behaviour: where the quadratic form fails and the bilinear form does not.

Near a substrate the regular coefficients grow like ``10^l`` and the scattered
ones shrink like ``10^-l``.  ``Q[a] - Q[a + 2p]`` then subtracts two numbers of
size ``|a|^2`` to get one of size ``|a p|``; the bilinear form never forms
``|a|^2``.  The reference is ``oracles.force_torque_mp`` -- the same
difference in 60-digit arithmetic, from independent loops.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("mpmath")

import oracles as O  # noqa: E402
import swforce as sf  # noqa: E402
from swforce.force import _flux_sums  # noqa: E402  # the packed sums (internal)


def graded(rng, lmax, decade):
    """Random-phase coefficients with ``|c_l| ~ 10^(decade * l)``."""
    l = np.array([l for l, _ in O.modes(lmax)])
    size = 10.0 ** (decade * l) * rng.uniform(0.5, 2.0, l.size)
    return tuple(size * np.exp(2j * np.pi * rng.uniform(size=l.size)) for _ in range(2))


def rel_err(got, ref):
    return np.abs(np.asarray(got) - ref).max() / np.linalg.norm(ref)


def test_bilinear_form_survives_a_cancellation_of_45_digits(rng):
    """``|a_l| ~ 10^(3l)``, ``|p_l| ~ 10^(-3l)``, ``nmax = 8``: ``Q[a] ~ 1e48`` against a force ``~ 1e3``."""
    lmax = 8
    aM, aN = graded(rng, lmax, +3)
    pM, pN = graded(rng, lmax, -3)
    ref = O.force_torque_mp(aM, aN, pM, pN)
    bil = sf.force_torque(aM, aN, pM, pN, form="bilinear")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quad = sf.force_torque(aM, aN, pM, pN, form="quadratic")
    for name, got in zip(("force", "torque", "spin"), bil):
        assert rel_err(got, ref[name]) < 1e-10, name
    # ...and the quadratic form demonstrably does not: a + 2p rounds to a, Q[a] - Q[a + 2p] to 0
    assert rel_err(quad.force, ref["force"]) > 0.5
    assert rel_err(quad.spin, ref["spin"]) > 0.5


@pytest.mark.parametrize("decade", [1, 2])
def test_bilinear_form_under_milder_grading(decade, rng):
    """``10^(+-l)`` and ``10^(+-2l)`` -- the regime of a sphere a few nm above gold at l ~ 10."""
    lmax = 10
    aM, aN = graded(rng, lmax, +decade)
    pM, pN = graded(rng, lmax, -decade)
    ref = O.force_torque_mp(aM, aN, pM, pN)
    bil = sf.force_torque(aM, aN, pM, pN, form="bilinear")
    for name, got in zip(("force", "torque", "spin"), bil):
        assert rel_err(got, ref[name]) < 1e-10, name


@pytest.mark.parametrize("lmax", [1, 3, 6])
def test_both_forms_match_60_digits_at_order_one(lmax, rng):
    """Where doubles suffice both forms agree with the 60-digit loops: an independent check of the sums."""
    aM, aN = O.random_coefficients(rng, lmax)
    pM, pN = O.random_coefficients(rng, lmax)
    ref = O.force_torque_mp(aM, aN, pM, pN)
    for form in ("quadratic", "bilinear"):
        for name, got in zip(("force", "torque", "spin"), sf.force_torque(aM, aN, pM, pN, form=form)):
            assert rel_err(got, ref[name]) < 1e-13, (form, name)


@pytest.mark.parametrize("lmax", [1, 2, 5, 9])
def test_flux_mp_implementation_matches_flux(lmax, rng):
    """``flux_mp`` (dictionary lookups, loops) vs ``flux`` (shifted padded arrays), all six sums, O(1)."""
    cM, cN = O.random_coefficients(rng, lmax)
    mine = np.array([complex(v) for v in O.flux_mp(cM, cN)])
    lib = np.array(_flux_sums(cM, cN), dtype=complex)
    assert np.abs(mine - lib).max() < 1e-13 * np.sum(np.abs(cM) ** 2 + np.abs(cN) ** 2)


def test_extreme_but_uniform_amplitudes_are_harmless(rng):
    """Uniform scale 1e-150 or 1e+150 (no grading) only rescales: both forms stay exact."""
    aM, aN = O.random_coefficients(rng, 4)
    pM, pN = O.random_coefficients(rng, 4)
    base = sf.force_torque(aM, aN, pM, pN, form="bilinear")
    for s in (1e-150, 1e150):
        for form in ("quadratic", "bilinear"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")               # see test_cancellation_check_itself_does_not_overflow
                got = sf.force_torque(s * aM, s * aN, s * pM, s * pN, form=form)
            assert rel_err(got.force / s**2, base.force) < 1e-13


# --------------------------------------------------------------------------
# The cancellation warning of form="quadratic"
# --------------------------------------------------------------------------
# Criterion: the terms of Q[a] are products of neighbouring a's, those of the force products of
# neighbouring a's and p's; about log10 of their size ratio digits are lost; warn above 1e10.


@pytest.mark.parametrize("decade,lmax", [(2, 10), (3, 8)])
def test_quadratic_form_warns_when_all_digits_cancel(decade, lmax, rng):
    """Graded coefficients (a particle at a substrate): ``a + 2p`` rounds to ``a`` in the high orders
    and the quadratic form returns a force with no correct digit -- precisely when the advice
    ``form='bilinear'`` is needed."""
    aM, aN = graded(rng, lmax, +decade)
    pM, pN = graded(rng, lmax, -decade)
    ref = O.force_torque_mp(aM, aN, pM, pN)
    with pytest.warns(RuntimeWarning, match="bilinear"):
        quad = sf.force_torque(aM, aN, pM, pN, form="quadratic")
    assert rel_err(quad.force, ref["force"]) > 0.5              # the premise: nothing is left


def test_quadratic_form_warns_for_graded_coefficients_below_the_norm_threshold(rng):
    """The documented promise is a warning when fewer than ~6 digits are left.  With graded coefficients
    (``|a_l| ~ 10^l``, ``|p_l| ~ 10^-l``, ``lmax = 8``) the force is not of order ``|a||p|``: the large
    ``a_l`` meet the tiny ``p_l``, so ``|a|/|p| ~ 2e9`` already leaves only ~2 correct digits."""
    aM, aN = graded(rng, 8, +1)
    pM, pN = graded(rng, 8, -1)
    ref = O.force_torque_mp(aM, aN, pM, pN)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quad = sf.force_torque(aM, aN, pM, pN, form="quadratic")
    assert rel_err(quad.force, ref["force"]) > 1e-6                 # the premise: fewer than 6 digits survive
    assert any(issubclass(w.category, RuntimeWarning) and "bilinear" in str(w.message) for w in caught)


@pytest.mark.parametrize("amp_a,amp_p", [(1e6, 1e-6), (1e8, 1e-4)])
def test_quadratic_form_warns_on_partial_cancellation(amp_a, amp_p, rng):
    """``|a|/|p| = 1e12``: about 12 digits cancel, ~4 survive -- warned, and the loss is real."""
    aM, aN = (amp_a * c for c in O.random_coefficients(rng, 4))
    pM, pN = (amp_p * c for c in O.random_coefficients(rng, 4))
    ref = O.force_torque_mp(aM, aN, pM, pN)
    with pytest.warns(RuntimeWarning, match="digits cancel"):
        quad = sf.force_torque(aM, aN, pM, pN, form="quadratic")
    assert 1e-10 < rel_err(quad.force, ref["force"]) < 1e-2
    with pytest.warns(RuntimeWarning, match="digits cancel"):          # same for the total outgoing input
        sf.force_torque(aM, aN, aM + 2 * pM, aN + 2 * pN, scattered=False, form="quadratic")


def test_no_warning_when_the_digits_suffice(rng):
    aM, aN = O.random_coefficients(rng, 6)
    pM, pN = O.random_coefficients(rng, 6)
    zero = np.zeros_like(pM)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sf.force_torque(aM, aN, pM, pN, form="quadratic")
        sf.force_torque(aM, aN, zero, zero, form="quadratic")       # no scattering: F = 0 exactly, nothing lost
        sf.force_torque(aM, aN, aM, aN, scattered=False, form="quadratic")
        sf.force_torque(1e3 * aM, 1e3 * aN, 1e-3 * pM, 1e-3 * pN, form="quadratic")   # |a|/|p| ~ 1e6
        sf.force_torque(1e8 * aM, 1e8 * aN, 1e-4 * pM, 1e-4 * pN, form="bilinear")     # never for bilinear


def test_no_warning_when_the_force_vanishes_by_symmetry():
    """A sphere at the antinode of a standing wave: ``F = 0`` by symmetry, yet nothing cancels numerically
    (``|a|/|p|`` is moderate) -- a warning keyed on ``|Q[a]|/|F|`` would fire here, this one must not."""
    aM, aN = O.project(lambda x: O.standing_wave(x, 1.0, z0=0.0), 1.0, 6)
    pM, pN = O.mie_scatter(aM, aN, 1.5, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ft = sf.force_torque(aM, aN, pM, pN, form="quadratic")
    assert np.abs(ft.force).max() < 1e-10 * np.sum(np.abs(np.r_[aM, aN]) ** 2)


def test_cancellation_check_itself_does_not_overflow(rng):
    """Uniform amplitudes near the top of the double range (``|Q| ~ 1e300``, still finite) must not
    trigger a numpy overflow warning from the cancellation test."""
    aM, aN = O.random_coefficients(rng, 4)
    pM, pN = O.random_coefficients(rng, 4)
    s = 1e150
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = sf.force_torque(s * aM, s * aN, s * pM, s * pN, form="quadratic")
    assert np.all(np.isfinite(got.force))
