"""Algebraic identities of the quadratic form ``Q`` (:func:`flux`) and its bilinear form ``B``.

Property-based (hypothesis, deterministic profile from ``conftest.py``): random
``nmax`` and coefficients.  Each identity catches a specific class of error --
a lost ``Re``/``Im``, a linear term in a quadratic form, a wrong line in
``flux_bilinear``, a wrong factor in the split -- that a single physical check
could miss.  Plus the error paths and the ``ForceTorque`` container.
"""

from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st  # noqa: E402
from hypothesis.extra import numpy as hnp  # noqa: E402

import swforce as sf  # noqa: E402
from swforce.force import ForceTorque, _bilinear_sums, _flux_sums  # noqa: E402  (the packed sums are internal)

REL = 1e-12


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------

finite = dict(allow_nan=False, allow_infinity=False)


@st.composite
def coefficient_sets(draw, count=2, max_nmax=8, max_magnitude=1e3):
    """``count`` pairs ``(cM, cN)`` sharing one random ``nmax``."""
    nmax = draw(st.integers(1, max_nmax))
    n = nmax * (nmax + 2)
    elem = st.complex_numbers(max_magnitude=max_magnitude, **finite)
    arrays = [draw(hnp.arrays(np.complex128, n, elements=elem)) for _ in range(2 * count)]
    return nmax, [(arrays[2 * i], arrays[2 * i + 1]) for i in range(count)]


def sums(x):
    return np.array(x, dtype=complex)


def scale_of(*pairs):
    """Natural size of a quadratic quantity built from these coefficient pairs."""
    return sum(np.sum(np.abs(c) ** 2) for pair in pairs for c in pair) + 1e-300


def close(a, b, scale, rel=REL):
    return np.abs(np.asarray(a) - np.asarray(b)).max() <= rel * scale


def ft_close(x, y, scale, rel=REL):
    return all(close(u, v, scale, rel) for u, v in zip(x, y))


# --------------------------------------------------------------------------
# Q
# --------------------------------------------------------------------------


@given(coefficient_sets(count=1))
def test_flux_axial_sums_are_real(data):
    """``fz, tz, sz`` are real; the transverse sums are packed ``x + i y`` and stay complex."""
    _, [(cM, cN)] = data
    fz, fxy, tz, txy, sz, sxy = _flux_sums(cM, cN)
    for v in (fz, tz, sz):
        assert np.isrealobj(v)
    for v in (fxy, txy, sxy):
        assert np.iscomplexobj(v)


@given(coefficient_sets(count=1), st.complex_numbers(min_magnitude=1e-3, max_magnitude=1e3, **finite))
def test_flux_is_homogeneous_of_degree_two_and_phase_blind(data, lam):
    """``Q[lam c] = |lam|^2 Q[c]`` -- no linear term; the global phase drops out."""
    _, [(cM, cN)] = data
    q = sums(_flux_sums(cM, cN))
    assert close(sums(_flux_sums(lam * cM, lam * cN)), abs(lam) ** 2 * q, abs(lam) ** 2 * scale_of((cM, cN)))
    phase = lam / abs(lam)
    assert close(sums(_flux_sums(phase * cM, phase * cN)), q, scale_of((cM, cN)))


@given(coefficient_sets(count=2))
def test_polarization_identity_all_six_sums(data):
    """``Q[u + v] = Q[u] + Q[v] + B(u, v) + B(v, u)`` for every one of the six outputs."""
    _, [(uM, uN), (vM, vN)] = data
    lhs = sums(_flux_sums(uM + vM, uN + vN))
    rhs = (sums(_flux_sums(uM, uN)) + sums(_flux_sums(vM, vN))
           + sums(_bilinear_sums(uM, uN, vM, vN)) + sums(_bilinear_sums(vM, vN, uM, uN)))
    assert close(lhs, rhs, scale_of((uM, uN), (vM, vN)))


@given(coefficient_sets(count=1))
def test_bilinear_on_the_diagonal_is_the_quadratic_form(data):
    """``B(c, c) = Q[c]`` for all six -- catches ``u`` and ``v`` swapped in a line of ``flux_bilinear``."""
    _, [(cM, cN)] = data
    assert close(sums(_bilinear_sums(cM, cN, cM, cN)), sums(_flux_sums(cM, cN)), scale_of((cM, cN)))


@given(coefficient_sets(count=3), st.floats(-1e3, 1e3, **finite))
def test_bilinear_is_additive_and_real_homogeneous_in_each_argument(data, t):
    _, [(uM, uN), (vM, vN), (wM, wN)] = data
    s = scale_of((uM, uN), (vM, vN), (wM, wN))
    B = lambda a, b: sums(_bilinear_sums(*a, *b))           # noqa: E731
    u, v, w = (uM, uN), (vM, vN), (wM, wN)
    uw = (uM + wM, uN + wN)
    vw = (vM + wM, vN + wN)
    assert close(B(uw, v), B(u, v) + B(w, v), s)
    assert close(B(u, vw), B(u, v) + B(u, w), s)
    assert close(B((t * uM, t * uN), v), t * B(u, v), max(1.0, abs(t)) * s)
    assert close(B(u, (t * vM, t * vN)), t * B(u, v), max(1.0, abs(t)) * s)


@given(coefficient_sets(count=2))
def test_axial_torque_bilinear_form_is_hermitian(data):
    """``tz`` is ``sum m (conj(u) v)``, so ``B_tz(v, u) = conj(B_tz(u, v))`` -- conjugated factor from ``u``."""
    _, [(uM, uN), (vM, vN)] = data
    b_uv = sums(_bilinear_sums(uM, uN, vM, vN))
    b_vu = sums(_bilinear_sums(vM, vN, uM, uN))
    assert close(b_vu[2], np.conj(b_uv[2]), scale_of((uM, uN), (vM, vN)))
    lm = np.concatenate([np.arange(-l, l + 1) for l in range(1, int(np.sqrt(uM.size + 1)))])
    assert close(b_uv[2], np.sum(lm * (np.conj(uM) * vM + np.conj(uN) * vN)), scale_of((uM, uN), (vM, vN)))


# --------------------------------------------------------------------------
# force_torque: the two forms
# --------------------------------------------------------------------------


# same as below: extreme draws make the quadratic form warn, which is its intended behaviour
@pytest.mark.filterwarnings("ignore:force_torque\\(form='quadratic'\\)")
@given(coefficient_sets(count=2, max_magnitude=10.0))
def test_bilinear_form_equals_quadratic_form(data):
    """``Q[a] - Q[a + 2p] = -2B(a,p) - 2B(p,a) - 4B(p,p)`` at moderate size (catches a wrong -2/-2/-4)."""
    _, [(aM, aN), (pM, pN)] = data
    s = scale_of((aM, aN), (pM, pN))
    for scattered in (True, False):
        q = sf.force_torque(aM, aN, pM, pN, scattered=scattered, form="quadratic")
        b = sf.force_torque(aM, aN, pM, pN, scattered=scattered, form="bilinear")
        assert ft_close(q, b, s)


@given(coefficient_sets(count=1))
def test_no_scattering_no_force(data):
    """``p = 0``: nothing is scattered, so force, torque and spin vanish -- exactly, in both forms."""
    _, [(aM, aN)] = data
    zero = np.zeros_like(aM)
    for form in ("quadratic", "bilinear"):
        ft = sf.force_torque(aM, aN, zero, zero, form=form)
        for v in ft:
            assert np.array_equal(v, np.zeros(3))
    # scattered=False with the total outgoing field equal to the regular one is the same statement
    for form in ("quadratic", "bilinear"):
        for v in sf.force_torque(aM, aN, aM, aN, scattered=False, form=form):
            assert np.abs(v).max() <= 1e-12 * scale_of((aM, aN))


@given(coefficient_sets(count=2), st.complex_numbers(min_magnitude=1e-2, max_magnitude=1e2, **finite))
def test_force_is_quadratic_in_the_fields(data, lam):
    _, [(aM, aN), (pM, pN)] = data
    s = scale_of((aM, aN), (pM, pN))
    base = sf.force_torque(aM, aN, pM, pN, form="bilinear")
    got = sf.force_torque(lam * aM, lam * aN, lam * pM, lam * pN, form="bilinear")
    assert ft_close(got, base.scaled(abs(lam) ** 2), abs(lam) ** 2 * s)


# --------------------------------------------------------------------------
# interference
# --------------------------------------------------------------------------


@given(coefficient_sets(count=3))
def test_interference_is_symmetric_and_additive(data):
    _, [(uM, uN), (vM, vN), (wM, wN)] = data
    s = scale_of((uM, uN), (vM, vN), (wM, wN))
    i_uv = sf.interference(uM, uN, vM, vN)
    i_vu = sf.interference(vM, vN, uM, uN)
    for x, y in zip(i_uv, i_vu):
        assert np.array_equal(x, y)                       # B(u,v)+B(v,u) and B(v,u)+B(u,v): same floats
    total = sf.interference(uM + wM, uN + wN, vM, vN)
    parts = sf.interference(uM, uN, vM, vN) + sf.interference(wM, wN, vM, vN)
    assert ft_close(total, parts, s)


@given(coefficient_sets(count=2))
def test_interference_is_the_cross_term_of_the_quadratic_form(data):
    """``interference(u, v)`` packs ``Q[u+v] - Q[u] - Q[v]`` the way ``force_torque`` packs its sums."""
    _, [(uM, uN), (vM, vN)] = data
    zero = np.zeros_like(uM)
    # Q[c] alone, packed: force_torque with p = -c/2 gives Q[c] - Q[0] = Q[c]
    packed_q = lambda cM, cN: sf.force_torque(cM, cN, -cM / 2, -cN / 2, form="bilinear")   # noqa: E731
    cross = packed_q(uM + vM, uN + vN) - packed_q(uM, uN) - packed_q(vM, vN)
    assert ft_close(sf.interference(uM, uN, vM, vN), cross, scale_of((uM, uN), (vM, vN)))
    assert all(np.array_equal(x, np.zeros(3)) for x in sf.interference(uM, uN, zero, zero))


# --------------------------------------------------------------------------
# force_torque_split
# --------------------------------------------------------------------------


@st.composite
def split_case(draw):
    nmax, [(aM, aN), (pM, pN)] = draw(coefficient_sets(count=2, max_magnitude=10.0))
    n_parts = draw(st.integers(1, 5))
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    weights = [rng.normal(size=aM.size) + 1j * rng.normal(size=aM.size) for _ in range(n_parts - 1)]
    partsM, partsN = [], []
    restM, restN = aM.copy(), aN.copy()
    for w in weights:                                      # random pieces, the last one is the remainder
        partsM.append(w * aM / n_parts)
        partsN.append(np.conj(w) * aN / n_parts)
        restM, restN = restM - partsM[-1], restN - partsN[-1]
    partsM.append(restM)
    partsN.append(restN)
    parts = {f"part{i}": (m, n) for i, (m, n) in enumerate(zip(partsM, partsN))}
    return (aM, aN), (pM, pN), parts


# hypothesis draws |p| down to ~1e-300 next to |a| ~ 10: the quadratic form rightly warns there,
# and the tolerance scale_of(...) is built for exactly that case.
@pytest.mark.filterwarnings("ignore:force_torque\\(form='quadratic'\\)")
@given(split_case())
def test_split_parts_and_recoil_sum_to_the_total_force(case):
    """The parts carry ``-2 interference(a_i, p)`` and add up to the force -- the test that caught a -1 for -2."""
    (aM, aN), (pM, pN), parts = case
    s = scale_of((aM, aN), (pM, pN), *parts.values())
    out = sf.force_torque_split(parts, pM, pN)
    assert set(out) == set(parts) | {"recoil", "total"}
    summed = out["recoil"]
    for name in parts:
        summed = summed + out[name]
    for form in ("bilinear", "quadratic"):
        whole = sf.force_torque(aM, aN, pM, pN, form=form)
        assert ft_close(summed, whole, s)
        assert ft_close(out["total"], whole, s)
    for name, (xM, xN) in parts.items():
        assert ft_close(out[name], sf.interference(xM, xN, pM, pN).scaled(-2.0), s)
    zero = np.zeros_like(pM)
    assert ft_close(out["recoil"], sf.force_torque(zero, zero, pM, pN, form="bilinear"), s)


@given(split_case())
def test_split_does_not_depend_on_the_order_of_the_parts(case):
    _, (pM, pN), parts = case
    forward = sf.force_torque_split(parts, pM, pN)
    backward = sf.force_torque_split(dict(reversed(list(parts.items()))), pM, pN)
    for name in list(parts) + ["recoil"]:
        for x, y in zip(forward[name], backward[name]):
            assert np.array_equal(x, y)
    s = scale_of((pM, pN), *parts.values())
    assert ft_close(forward["total"], backward["total"], s)


@given(coefficient_sets(count=2, max_magnitude=10.0))
def test_split_with_one_part_is_total_minus_recoil(data):
    _, [(aM, aN), (pM, pN)] = data
    out = sf.force_torque_split({"everything": (aM, aN)}, pM, pN)
    whole = sf.force_torque(aM, aN, pM, pN, form="bilinear")
    assert ft_close(out["everything"], whole - out["recoil"], scale_of((aM, aN), (pM, pN)))


def test_split_input_errors():
    a = (np.ones(3, complex), np.ones(3, complex))
    p = (np.ones(3, complex), np.ones(3, complex))
    with pytest.raises(ValueError, match="at least one part"):
        sf.force_torque_split({}, *p)
    for name in ("recoil", "total"):
        with pytest.raises(ValueError, match="reserved"):
            sf.force_torque_split({name: a}, *p)
    with pytest.raises(ValueError, match="different Nmax"):
        sf.force_torque_split({"x": (np.ones(8), np.ones(8))}, *p)
    with pytest.raises(ValueError):
        sf.force_torque_split({"x": (np.ones(3), np.ones(8))}, *p)


# --------------------------------------------------------------------------
# Error paths and the container
# --------------------------------------------------------------------------


def test_input_validation():
    three, eight = np.ones(3, complex), np.ones(8, complex)
    with pytest.raises(ValueError, match="same length"):
        _flux_sums(three, eight)
    with pytest.raises(ValueError, match="not a valid VSWF vector length"):
        _flux_sums(np.ones(4), np.ones(4))
    with pytest.raises(ValueError, match="same Nmax"):
        _bilinear_sums(three, three, eight, eight)
    with pytest.raises(ValueError, match="same length"):
        _bilinear_sums(three, eight, three, three)
    with pytest.raises(ValueError, match="all four"):
        sf.force_torque(three, three, three, eight)
    with pytest.raises(ValueError, match="form="):
        sf.force_torque(three, three, three, three, form="cubic")


def test_inputs_are_raveled_and_cast():
    rng = np.random.default_rng(3)
    aM, aN, pM, pN = (rng.normal(size=8) + 1j * rng.normal(size=8) for _ in range(4))
    flat = sf.force_torque(aM, aN, pM, pN)
    shaped = sf.force_torque(aM.reshape(2, 4), aN.reshape(4, 2), list(pM), tuple(pN))
    for x, y in zip(flat, shaped):
        assert np.array_equal(x, y)
    real = _flux_sums(np.arange(8.0), np.arange(8.0)[::-1])
    assert np.all(np.isfinite(np.array(real, dtype=complex)))


def test_force_torque_container():
    a = ForceTorque(np.array([1.0, 2, 3]), np.array([4.0, 5, 6]), np.array([7.0, 8, 9]))
    b = ForceTorque(np.array([0.5, 0, 1]), np.array([1.0, 1, 1]), np.array([2.0, 0, 0]))
    force, torque, spin = a
    assert np.array_equal(force, [1, 2, 3]) and np.array_equal(torque, [4, 5, 6]) and np.array_equal(spin, [7, 8, 9])
    s = a + b
    assert np.array_equal(s.force, [1.5, 2, 4]) and np.array_equal(s.torque, [5, 6, 7]) and np.array_equal(s.spin, [9, 8, 9])
    d = a - b
    assert np.array_equal(d.force, [0.5, 2, 2]) and np.array_equal(d.torque, [3, 4, 5]) and np.array_equal(d.spin, [5, 8, 9])
    h = a.scaled(2.0)
    assert np.array_equal(h.force, [2, 4, 6]) and np.array_equal(h.torque, [8, 10, 12]) and np.array_equal(h.spin, [14, 16, 18])
    assert isinstance(s, ForceTorque) and isinstance(d, ForceTorque) and isinstance(h, ForceTorque)
