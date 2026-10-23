"""Symmetries: rotating, inverting or mirroring the whole field must transform force, torque and spin
as a polar, an axial and an axial vector.

Each coefficient transformation is first checked against the fields themselves
(``oracles.efield`` at transformed points), so the test of the library cannot
rest on a wrong transformation.  With ``E'(r) = R E(R^{-1} r)``:

* rotation by ``alpha`` about ``z``:  ``c_lm -> c_lm e^{-i m alpha}``;
* inversion ``E'(r) = -E(-r)``:        ``c^M -> (-1)^(l+1) c^M``, ``c^N -> (-1)^l c^N``;
* mirror ``y -> -y``, ``E'(r) = s E(s r)``, ``s = diag(1, -1, 1)``:
  ``c^M_lm -> -(-1)^m c^M_{l,-m}``, ``c^N_lm -> (-1)^m c^N_{l,-m}``;
* a general rotation, by projecting the rotated field (no Wigner matrices needed).
"""

from __future__ import annotations

import numpy as np
import pytest

import oracles as O
import swforce as sf

K = 1.0


def lm(lmax):
    l = np.array([l for l, _ in O.modes(lmax)])
    m = np.array([m for _, m in O.modes(lmax)])
    return l, m


def rotate_z(c, alpha, lmax):
    _, m = lm(lmax)
    return c * np.exp(-1j * m * alpha)


def invert(cM, cN, lmax):
    l, _ = lm(lmax)
    return (-1.0) ** (l + 1) * cM, (-1.0) ** l * cN


def mirror_y(cM, cN, lmax):
    l, m = lm(lmax)
    flip = np.array([O.slot(ll, -mm) for ll, mm in zip(l, m)])
    return -((-1.0) ** m) * cM[flip], ((-1.0) ** m) * cN[flip]


def Rz(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def Ry(beta):
    c, s = np.cos(beta), np.sin(beta)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


SIGMA = np.diag([1.0, -1.0, 1.0])


def points(rng):
    v = rng.normal(size=(8, 3))
    return v / np.linalg.norm(v, axis=1)[:, None] * rng.uniform(0.5, 3.0, (8, 1))


# --------------------------------------------------------------------------
# The coefficient transformations are right
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["regular", "outgoing"])
def test_coefficient_transformations_match_the_fields(kind, rng):
    lmax = 4
    cM, cN = O.random_coefficients(rng, lmax)
    pts = points(rng)
    E = lambda x: O.efield(cM, cN, x, K, kind)                 # noqa: E731
    alpha = 0.83
    R = Rz(alpha)
    rot = O.efield(rotate_z(cM, alpha, lmax), rotate_z(cN, alpha, lmax), pts, K, kind)
    assert np.abs(rot - E(pts @ R) @ R.T).max() < 1e-12 * np.abs(rot).max()      # rows: R^-1 r = R^T r
    inv = O.efield(*invert(cM, cN, lmax), pts, K, kind)
    assert np.abs(inv + E(-pts)).max() < 1e-12 * np.abs(inv).max()
    mir = O.efield(*mirror_y(cM, cN, lmax), pts, K, kind)
    assert np.abs(mir - E(pts @ SIGMA) @ SIGMA).max() < 1e-12 * np.abs(mir).max()


# --------------------------------------------------------------------------
# Force, torque, spin transform as they must
# --------------------------------------------------------------------------


@pytest.mark.parametrize("form", ["quadratic", "bilinear"])
@pytest.mark.parametrize("alpha", [0.4, np.pi / 2, 2.9, -1.7])
def test_rotation_about_z(form, alpha, rng):
    """``F_z`` unchanged, ``F_x + i F_y -> e^{i alpha} (F_x + i F_y)`` (catches swapped ``m +- 1`` neighbours)."""
    lmax = 5
    aM, aN = O.random_coefficients(rng, lmax)
    pM, pN = O.random_coefficients(rng, lmax)
    base = sf.force_torque(aM, aN, pM, pN, form=form)
    turned = sf.force_torque(*(rotate_z(c, alpha, lmax) for c in (aM, aN, pM, pN)), form=form)
    for b, t in zip(base, turned):
        scale = np.linalg.norm(b)
        assert t[2] == pytest.approx(b[2], abs=1e-12 * scale)
        assert (t[0] + 1j * t[1]) == pytest.approx(np.exp(1j * alpha) * (b[0] + 1j * b[1]), abs=1e-12 * scale)


@pytest.mark.parametrize("form", ["quadratic", "bilinear"])
def test_inversion(form, rng):
    """Parity: ``F -> -F`` (polar), torque and spin unchanged (axial)."""
    lmax = 5
    aM, aN = O.random_coefficients(rng, lmax)
    pM, pN = O.random_coefficients(rng, lmax)
    base = sf.force_torque(aM, aN, pM, pN, form=form)
    inv = sf.force_torque(*invert(aM, aN, lmax), *invert(pM, pN, lmax), form=form)
    s = np.linalg.norm(base.force) + np.linalg.norm(base.torque)
    assert np.abs(inv.force + base.force).max() < 1e-12 * s
    assert np.abs(inv.torque - base.torque).max() < 1e-12 * s
    assert np.abs(inv.spin - base.spin).max() < 1e-12 * s


@pytest.mark.parametrize("form", ["quadratic", "bilinear"])
def test_mirror_y(form, rng):
    """Mirror ``y -> -y``: ``F -> s F`` (only ``F_y`` flips), torque and spin ``-> -s T``."""
    lmax = 5
    aM, aN = O.random_coefficients(rng, lmax)
    pM, pN = O.random_coefficients(rng, lmax)
    base = sf.force_torque(aM, aN, pM, pN, form=form)
    mir = sf.force_torque(*mirror_y(aM, aN, lmax), *mirror_y(pM, pN, lmax), form=form)
    s = np.linalg.norm(base.force) + np.linalg.norm(base.torque)
    assert np.abs(mir.force - SIGMA @ base.force).max() < 1e-12 * s
    assert np.abs(mir.torque + SIGMA @ base.torque).max() < 1e-12 * s
    assert np.abs(mir.spin + SIGMA @ base.spin).max() < 1e-12 * s


def test_mirror_symmetric_field_has_no_y_force(rng):
    """A field symmetric under ``y -> -y`` (sum of a field and its mirror image) is pushed in the ``xz`` plane."""
    lmax = 4
    aM, aN = O.random_coefficients(rng, lmax)
    pM, pN = O.random_coefficients(rng, lmax)
    (bM, bN), (qM, qN) = mirror_y(aM, aN, lmax), mirror_y(pM, pN, lmax)
    ft = sf.force_torque(aM + bM, aN + bN, pM + qM, pN + qN, form="bilinear")
    s = np.linalg.norm(ft.force) + np.linalg.norm(ft.torque)
    assert abs(ft.force[1]) < 1e-12 * s
    assert abs(ft.torque[0]) < 1e-12 * s and abs(ft.torque[2]) < 1e-12 * s


def test_general_rotation_by_projection(rng):
    """Rotate the *fields* about ``y`` and re-project: ``F -> R F``, ``T -> R T``, ``S -> R S``."""
    lmax = 3
    a = O.random_coefficients(rng, lmax)
    p = O.random_coefficients(rng, lmax, scale=0.5)
    R = Ry(0.7) @ Rz(0.3)
    a_rot = O.project(lambda x: O.efield(*a, x @ R, K, "regular") @ R.T, K, lmax + 1, "regular")
    p_rot = O.project(lambda x: O.efield(*p, x @ R, K, "outgoing") @ R.T, K, lmax + 1, "outgoing")
    pad = lambda c: np.concatenate([c, np.zeros(O.size(lmax + 1) - c.size, complex)])   # noqa: E731
    base = sf.force_torque(*(pad(c) for c in (*a, *p)), form="bilinear")
    turned = sf.force_torque(*a_rot, *p_rot, form="bilinear")
    for b, t in zip(base, turned):
        assert np.abs(t - R @ b).max() < 1e-10 * np.linalg.norm(b)
