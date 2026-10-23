"""MATLAB OTT (OTT-Fixed) coefficients -> the force sums.

OTT-Fixed (https://github.com/semyonbo/ott, branch
``fix/condon-shortley-and-smarties-tmatrix``, FIXES.md FIX 1) restored the Condon-Shortley phase
in ``ott.utils.spharm``, so its coefficients are already in the convention of the
sums: the combined index ``ci = n(n+1)+m``, ``a``/``p`` the magnetic (TE) and
``b``/``q`` the electric (TM) multipoles, and its ``Nn = 1/sqrt(n(n+1))``
normalisation all match.  What is left of OTT's conventions is the naming and
that OTT may truncate the scattered beam at a lower ``Nmax`` than the incident
one -- this module zero-pads, as ``ott.forcetorque`` does.

Checked against OTT-Fixed's own ``ott.forcetorque`` on ten cases (sphere,
spheroid, cylinder, cube; plane waves and focused beams) to ~6e-15,
``tests/test_adapter_ott.py``.

Unpatched upstream OTT omits the phase (a ``(-1)^m`` per mode, which flips the sign
of ``F_x, F_y, T_x, T_y``); multiply its coefficients by ``(-1)^m`` before passing them here.

    from swforce.adapters import ott
    ft = ott.force_torque(a, b, p, q)      # what beam.getCoefficients() hands you
"""

from __future__ import annotations

import numpy as np

from ..basis import nmax_from_size
from ..force import ForceTorque
from ..force import force_torque as _force_torque

__all__ = ["force_torque"]


def _same_nmax(*vectors):
    """Zero-pad to a common ``Nmax``.  OTT lets the scattered beam stop at a lower order than the
    incident one; the missing modes carry nothing, which is also how ``ott.forcetorque`` treats them."""
    vectors = [np.asarray(v, dtype=complex).ravel() for v in vectors]
    for v in vectors:
        nmax_from_size(v.size)                       # raises on a length that is not n(n+2)
    n = max(v.size for v in vectors)
    return [np.concatenate([v, np.zeros(n - v.size, dtype=complex)]) for v in vectors]


def force_torque(a, b, p, q, scattered: bool = True) -> ForceTorque:
    """Force, torque and spin from OTT-Fixed's ``(a, b, p, q)``.

    ``(a, b)`` incident (regular), ``(p, q)`` scattered (outgoing) -- or, with
    ``scattered=False``, the total outgoing field ``2p + a``.  Vectors of different
    ``Nmax`` are zero-padded to the largest.  Force, torque and spin equal OTT-Fixed's
    ``ott.forcetorque`` (to ~1e-14, ``tests/test_adapter_ott.py``); for the spin this
    needs OTT-Fixed with FIX 4 (``fb543ab``) -- upstream OTT and earlier OTT-Fixed return
    ``(-S_x, S_y, -S_z)``.
    """
    return _force_torque(*_same_nmax(a, b, p, q), scattered=scattered)
