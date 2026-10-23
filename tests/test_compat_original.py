"""Bit-for-bit compatibility with the original implementations (the main regression layer).

``tests/data/original_*.npz`` hold outputs of the original code on fixed inputs,
written by the (unpublished) validation tools: ``swforce`` from BIC-Force (``flux``,
``force_torque``, ``units``, the adapters) and the SPP-validated
``smuthi_swforce.py`` (``flux_bilinear``, ``force_dimensionless``).  These tests
need only the ``.npz`` files and never skip.  ``np.array_equal`` everywhere:
any change of formula *or of the order of floating-point operations* fails.

The golden files were written with one numpy build; a different build could in
principle round differently.  :func:`test_live_original_bit_for_bit` therefore
also runs the original, when its checkout is present, in *this* interpreter.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest

import swforce as sf
from swforce.force import _bilinear_sums, _flux_sums  # the packed sums (internal)

DATA = Path(__file__).resolve().parent / "data"
ROOT = Path(__file__).resolve().parents[1]
BIC_FORCE, BIC_COMMIT = Path("/Users/semyon/git/BIC-Force"), "eea25eb"   # last commit with the in-repo swforce


def _original_checkout() -> Path:
    """A directory holding the original ``swforce/`` package: extracted once from BIC-Force@eea25eb
    (the copy was removed from the working tree afterwards).  Missing repository -> a path that does not
    exist, and the live tests skip; the golden ``.npz`` files keep the bit-for-bit checks running."""
    target = Path(tempfile.gettempdir()) / f"swforce_original_{BIC_COMMIT}"
    if (target / "swforce" / "force.py").exists():
        return target
    try:
        archive = subprocess.run(["git", "-C", str(BIC_FORCE), "archive", BIC_COMMIT, "swforce"],
                                 check=True, capture_output=True).stdout
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-x", "-C", str(target)], input=archive, check=True)
    except (OSError, subprocess.CalledProcessError):
        return target
    return target


ORIGINAL = _original_checkout()

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")    # the quadratic-form cancellation warning is not under test here

CORE = dict(np.load(DATA / "original_core.npz"))
BILINEAR = dict(np.load(DATA / "original_bilinear.npz"))


def cases():
    for i, nmax in enumerate(CORE["nmax"]):
        s = slice(CORE["offset"][i], CORE["offset"][i + 1])
        yield i, int(nmax), CORE["aM"][s], CORE["aN"][s], CORE["pM"][s], CORE["pN"][s]


def ft_array(ft):
    return np.array([ft.force, ft.torque, ft.spin])


def test_golden_inputs_cover_the_plan():
    """50 cases, every ``nmax = 1..12``, amplitudes from 1e-8 to 1e40."""
    assert len(CORE["nmax"]) == 50
    assert set(CORE["nmax"].tolist()) == set(range(1, 13))
    assert CORE["amp_a"].min() == 1e-8 and CORE["amp_a"].max() == 1e40
    for _, nmax, aM, *_ in cases():
        assert aM.size == nmax * (nmax + 2)


def test_flux_bit_for_bit():
    """All six outputs of ``flux``, on ``a`` and on ``p``."""
    for i, _, aM, aN, pM, pN in cases():
        assert np.array_equal(np.array(_flux_sums(aM, aN), dtype=complex), CORE["flux_a"][i]), i
        assert np.array_equal(np.array(_flux_sums(pM, pN), dtype=complex), CORE["flux_p"][i]), i


def test_flux_output_types_unchanged():
    """``fz, tz, sz`` are real numpy floats, the transverse sums complex -- as in the original."""
    _, _, aM, aN, _, _ = next(cases())
    fz, fxy, tz, txy, sz, sxy = _flux_sums(aM, aN)
    assert all(isinstance(v, np.floating) for v in (fz, tz, sz))
    assert all(isinstance(v, np.complexfloating) for v in (fxy, txy, sxy))


@pytest.mark.parametrize("scattered,key", [(True, "ft_scattered"), (False, "ft_total")])
def test_force_torque_quadratic_bit_for_bit(scattered, key):
    for i, _, aM, aN, pM, pN in cases():
        got = sf.force_torque(aM, aN, pM, pN, scattered=scattered, form="quadratic")
        assert np.array_equal(ft_array(got), CORE[key][i]), i


def test_default_is_the_original_quadratic_scattered_form():
    for i, _, aM, aN, pM, pN in cases():
        assert np.array_equal(ft_array(sf.force_torque(aM, aN, pM, pN)), CORE["ft_scattered"][i]), i


def test_flux_bilinear_bit_for_bit_with_smuthi_swforce():
    """``(fz, fxy)`` of ``B(a, p)`` and ``B(p, a)`` equal ``smuthi_swforce.flux_bilinear``."""
    for i, _, aM, aN, pM, pN in cases():
        assert np.array_equal(np.array(_bilinear_sums(aM, aN, pM, pN)[:2], dtype=complex), BILINEAR["bilinear_ap"][i]), i
        assert np.array_equal(np.array(_bilinear_sums(pM, pN, aM, aN)[:2], dtype=complex), BILINEAR["bilinear_pa"][i]), i


def test_force_bilinear_bit_for_bit_with_force_dimensionless():
    for i, _, aM, aN, pM, pN in cases():
        got = sf.force_torque(aM, aN, pM, pN, form="bilinear").force
        assert np.array_equal(got, BILINEAR["force_dimensionless"][i]), i


def test_cross_section_bit_for_bit():
    """``cross_section(f, lambda0, n)`` is the original ``cross_section_from_force(f, 2 pi n / lambda0)``."""
    f, lam, idx = CORE["units_f"], CORE["units_wavelength"], CORE["units_index"]
    for j in range(len(lam)):
        assert np.array_equal(sf.cross_section(f[j], lam[j], idx[j]), CORE["cross_section"][j])


def test_newtons_by_intensity_equal_the_original_to_rounding():
    """By intensity ``eps0`` cancels, so ``to_si`` (through ``eps0 n^2 |E|^2 / (8 k^2)``) reproduces the original
    ``force_newtons`` (``(n/c) I f / (4 k^2)``) and its ``torque_newton_metres`` with the ``area_mso2`` typo fixed
    (the original raised ``NameError``) up to operation order."""
    assert str(CORE["torque_original_error"]) == "NameError: name 'area_mso2' is not defined"
    f, lam, idx, inten = CORE["units_f"], CORE["units_wavelength"], CORE["units_index"], CORE["units_intensity"]
    for j in range(len(lam)):
        fj = np.asarray(f[j], dtype=float)
        got = sf.to_si(sf.ForceTorque(fj, fj, fj), lam[j], intensity=inten[j], index_medium=idx[j])
        for new, old in ((got.force, CORE["force_newtons"][j]), (got.torque, CORE["torque_fixed"][j])):
            assert np.abs(new - old).max() <= 2e-15 * np.abs(old).max(), j          # measured: 2.7 ulp


LIVE = r'''
import sys
import numpy as np
sys.path.insert(0, ORIGINAL)
import swforce as sw
from swforce.adapters import ott as sw_ott
assert sw.__file__.startswith(ORIGINAL)
d = dict(np.load(GOLDEN))
out = []
for i in range(len(d["nmax"])):
    s = slice(d["offset"][i], d["offset"][i + 1])
    aM, aN, pM, pN = d["aM"][s], d["aN"][s], d["pM"][s], d["pN"][s]
    out.append(np.concatenate([np.array(sw.flux(aM, aN), complex), np.array(sw.flux(pM, pN), complex),
                               np.array(list(sw.force_torque(aM, aN, pM, pN))).ravel(),
                               np.array(list(sw.force_torque(aM, aN, pM, pN, scattered=False))).ravel(),
                               np.array(list(sw_ott.force_torque(aM, aN, pM, pN))).ravel(),
                               np.array(list(sw_ott.force_torque(aM, aN, pM, pN, scattered=False))).ravel()]))
np.save(OUTPUT, np.array(out))
'''


@pytest.mark.skipif(not (ORIGINAL / "swforce" / "force.py").exists(), reason="original swforce (BIC-Force@eea25eb) not available")
def test_live_original_bit_for_bit(tmp_path):
    """Run the original ``swforce`` in this very interpreter (subprocess) and compare bit for bit -- the core,
    and the original OTT adapter against ``upstream.force_torque(...)``."""
    upstream = pytest.importorskip("validation.reference.upstream")    # the private validation code
    out = tmp_path / "live.npy"
    header = f"ORIGINAL = {str(ORIGINAL)!r}\nGOLDEN = {str(DATA / 'original_core.npz')!r}\nOUTPUT = {str(out)!r}\n"
    subprocess.run([sys.executable, "-c", header + textwrap.dedent(LIVE)], check=True, cwd=tmp_path)
    live = np.load(out)
    for i, _, aM, aN, pM, pN in cases():
        mine = np.concatenate([np.array(_flux_sums(aM, aN), complex), np.array(_flux_sums(pM, pN), complex),
                               ft_array(sf.force_torque(aM, aN, pM, pN)).ravel(),
                               ft_array(sf.force_torque(aM, aN, pM, pN, scattered=False)).ravel(),
                               ft_array(upstream.force_torque(aM, aN, pM, pN)).ravel(),
                               ft_array(upstream.force_torque(aM, aN, pM, pN, scattered=False)).ravel()])
        assert np.array_equal(mine, live[i]), i
