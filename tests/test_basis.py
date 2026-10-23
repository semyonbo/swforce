"""Combined-index bookkeeping: ``ci = l(l+1) + m``, vectors of length ``lmax(lmax+2)``.

An off-by-one here sends every coefficient to the wrong mode without any error,
so the order is pinned explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest

import swforce as sf
from swforce import basis


def test_ci_index_order():
    """Slot of ``(l, m)`` is ``l(l+1) + m - 1``: (1,-1) first, then (1,0), (1,1), (2,-2), ..."""
    expected = [((1, -1), 0), ((1, 0), 1), ((1, 1), 2), ((2, -2), 3), ((2, 2), 7), ((3, -3), 8), ((3, 3), 14)]
    for (l, m), slot in expected:
        assert basis.ci_index(l, m) == slot


def test_ci_index_is_vectorised():
    l = np.array([1, 2, 3])
    m = np.array([0, -1, 3])
    assert np.array_equal(basis.ci_index(l, m), l * (l + 1) + m - 1)


@pytest.mark.parametrize("lmax", [1, 2, 3, 7, 12])
def test_mode_indices_enumerate_every_mode_in_order(lmax):
    l, m = sf.mode_indices(lmax)
    expected = [(ll, mm) for ll in range(1, lmax + 1) for mm in range(-ll, ll + 1)]
    assert list(zip(l.tolist(), m.tolist())) == expected
    assert len(l) == sf.basis_size(lmax) == lmax * (lmax + 2)
    # the slot of every mode is its position
    assert np.array_equal(basis.ci_index(l, m), np.arange(len(l)))
    assert l.dtype.kind == "i" and m.dtype.kind == "i"


def test_mode_indices_first_entries():
    l, m = sf.mode_indices(2)
    assert l.tolist() == [1, 1, 1, 2, 2, 2, 2, 2]
    assert m.tolist() == [-1, 0, 1, -2, -1, 0, 1, 2]


@pytest.mark.parametrize("lmax", range(0, 40))
def test_nmax_from_size_inverts_basis_size(lmax):
    assert sf.nmax_from_size(sf.basis_size(lmax)) == lmax


@pytest.mark.parametrize("bad", [1, 2, 4, 5, 6, 7, 9, 14, 16, 23, 25, 1000])
def test_nmax_from_size_rejects_invalid_lengths(bad):
    """A vector of the wrong length must raise, never be silently truncated or padded."""
    with pytest.raises(ValueError, match="not a valid VSWF vector length"):
        sf.nmax_from_size(bad)


@pytest.mark.parametrize("ka,expected", [(0.0, 0), (1.0, 4), (-1.0, 4), (2.2, 7), (8.0, 14), (10.0, 17),
                                         (27.0, 36), (0.001, 1), (100.0, 114)])
def test_ka2nmax_wiscombe_rule(ka, expected):
    """``Nmax = ceil(|ka| + 3 |ka|^(1/3))`` at hand-computed values."""
    assert sf.ka2nmax(ka) == expected
    assert isinstance(sf.ka2nmax(ka), int)
