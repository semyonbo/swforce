"""swforce -- exact optical force and torque from spherical-wave (T-matrix) coefficients.

The optical force, torque and spin on a nano-object from the Maxwell stress
tensor, with the field expanded in vector spherical waves about the object::

    import swforce as sf

    ft = sf.force_torque(aM, aN, pM, pN)
    ft.force        # array([Fx, Fy, Fz]), dimensionless
    ft.torque       # array([Tx, Ty, Tz])
    sf.to_si(ft, wavelength=900.0).force    # newtons, for a field of amplitude 1 V/m

``(aM, aN)`` are the **regular** magnetic and electric multipole coefficients
of everything exciting the particle, ``(pM, pN)`` the **outgoing** ones of its
scattered field, indexed by ``ci = l(l+1)+m`` (Condon-Shortley convention).
The result is the Maxwell stress integral evaluated analytically (Farsund &
Felderhof 1996): exact for the given expansion, the only approximation is the
truncation at ``lmax``.

The force is linear in the exciting field at fixed ``p``, so it splits exactly
by the parts of that field::

    parts = sf.force_torque_split({"incident": (a0M, a0N), "reflected": (arM, arN)}, pM, pN)
    parts["incident"].force + parts["reflected"].force + parts["recoil"].force  # == total

Adapters (not imported here, so ``import swforce`` needs only numpy/scipy):
:mod:`swforce.adapters.smuthi` (SMUTHI conventions, finished simulations),
:mod:`swforce.adapters.treams` (clusters, arbitrary T-matrices),
:mod:`swforce.adapters.ott` (MATLAB OTT-Fixed coefficients).  Documentation: ``README.md``.
"""

from __future__ import annotations

from . import basis
from .basis import basis_size, ka2nmax, mode_indices, nmax_from_size
from .force import ForceTorque, flux, flux_bilinear, force_torque, force_torque_split, interference
from . import units
from .units import cross_section, to_si

__version__ = "0.2.0"

__all__ = [
    "force_torque", "force_torque_split", "interference", "flux", "flux_bilinear", "ForceTorque",
    "to_si", "cross_section", "units",
    "basis_size", "mode_indices", "nmax_from_size", "ka2nmax", "basis",
    "__version__",
]
