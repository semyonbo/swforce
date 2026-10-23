"""Combined-index bookkeeping for the vector spherical wave function (VSWF) basis.

Every coefficient vector in this package is indexed by the *combined index*

    ci = n*(n+1) + m          (1-based)

so ``ci`` runs 1, 2, ... over ``(n,m) = (1,-1), (1,0), (1,1), (2,-2), ...``.  A
truncation at ``Nmax`` gives vectors of length ``Nmax*(Nmax+2)``.  Python arrays
are 0-based, so the array slot for ``(n,m)`` is ``ci - 1``; :func:`ci_index`
returns that slot directly and is what the rest of the package uses.
"""

from __future__ import annotations

import numpy as np


def ci_index(n, m):
    """0-based array slot of mode ``(n, m)``: the combined index ``ci = n(n+1) + m`` minus one."""
    n = np.asarray(n)
    m = np.asarray(m)
    return n * (n + 1) + m - 1


def mode_indices(nmax: int):
    """The ``(n, m)`` of every mode up to ``Nmax``, in combined-index order.

    Returns two int arrays of length :func:`basis_size`.
    """
    n = np.concatenate([np.full(2 * k + 1, k) for k in range(1, nmax + 1)])
    m = np.concatenate([np.arange(-k, k + 1) for k in range(1, nmax + 1)])
    return n, m


def basis_size(nmax: int) -> int:
    """Number of ``(n,m)`` modes up to ``Nmax``: ``Nmax*(Nmax+2)``."""
    return nmax * (nmax + 2)


def nmax_from_size(size: int) -> int:
    """Invert :func:`basis_size` -- the ``Nmax`` of a coefficient vector."""
    nmax = int(round(np.sqrt(size + 1) - 1))
    if basis_size(nmax) != size:
        raise ValueError(f"{size} is not a valid VSWF vector length (Nmax*(Nmax+2))")
    return nmax


def ka2nmax(ka: float) -> int:
    """Wiscombe's truncation rule ``Nmax = ka + 3*(ka)^(1/3)``.

    ``ka`` is the size parameter of the sphere enclosing the region of interest.
    """
    return int(np.ceil(np.abs(ka) + 3.0 * np.abs(ka) ** (1.0 / 3.0)))
