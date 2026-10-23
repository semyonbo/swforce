"""Dimensionless sums -> SI.

The sums of :mod:`swforce.force` are dimensionless; the coefficients carry the
field amplitude ``|E|`` [V/m] of the wave they were built from (``treams.plane_wave``
with a unit polarisation vector, SMUTHI's ``PlaneWave(amplitude=1)``: ``|E| = 1``).
The Maxwell stress tensor in a lossless non-magnetic host of index ``n``
(``eps = eps0 n^2``, ``Z = Z0 / n``) gives

    F = eps0 n^2 |E|^2 f / (8 k^2),    T = eps0 n^2 |E|^2 t / (8 k^3),    S = eps0 n^2 |E|^2 s / (8 k^3)

with ``k = n k0 = 2 pi n / lambda0`` the wavenumber **in the host**.  ``T`` is the
torque about the expansion centre (the particle centre), ``S`` its spin part --
the spin angular momentum the field hands over per unit time (N m, like ``T``).

For a plane wave of intensity ``I = n eps0 c |E|^2 / 2`` the force is
``F = (n/c) I C`` with ``C = f / (4 k^2)`` (:func:`cross_section`): ``f_z`` on the
beam axis gives the radiation-pressure cross section ``C_pr`` of Mie theory.  For
circular polarisation of helicity ``sigma = +-1`` (``(e1 + i sigma e2)/sqrt(2)``,
``e1 x e2 = khat``, ``exp(-i omega t)``) and a particle that conserves ``m`` about
the beam axis (a sphere, a body of revolution about ``khat``),
``t_z / (4 k^2) = sigma C_abs``, i.e. ``T_z = sigma I C_abs / omega``.  Both are
pinned against the Bohren-Huffman series and the stress tensor in ``tests/test_units.py``.

Constants: ``c`` exact (SI 2019), ``eps0`` CODATA 2022 (as ``scipy.constants`` >= 1.15),
``mu0 = 1 / (eps0 c^2)``.  Wavelengths are vacuum wavelengths in nm; cross sections in nm^2.
"""

from __future__ import annotations

import numpy as np

from .force import ForceTorque

C0 = 299792458.0                  # speed of light [m/s], exact
EPS0 = 8.8541878188e-12           # vacuum permittivity [F/m], CODATA 2022
MU0 = 1.0 / (EPS0 * C0**2)        # vacuum permeability [H/m]
NM = 1e-9                         # 1 nm in metres


def _check(wavelength, index_medium):
    """The conversion holds for a lossless host (real ``n > 0``) and a positive vacuum wavelength."""
    if np.iscomplexobj(index_medium) or not np.isreal(index_medium) or not index_medium > 0:
        raise ValueError(f"index_medium must be real and positive (lossless host), got {index_medium!r}")
    if not wavelength > 0:
        raise ValueError(f"wavelength must be positive (vacuum wavelength in nm), got {wavelength!r}")


def si_scale(wavelength: float, *, amplitude: float | None = None, intensity: float | None = None,
             index_medium: float = 1.0) -> tuple[float, float]:
    """``(eps0 n^2 |E|^2 / (8 k^2), eps0 n^2 |E|^2 / (8 k^3))``: newtons per unit ``f`` and N m per unit ``t``, ``s``.

    ``wavelength`` is the vacuum wavelength [nm].  The field strength is the
    amplitude ``|E|`` [V/m] the coefficients were built with (default 1) or, for a
    plane wave, its intensity ``I`` [W/m^2] (``|E|^2 = 2 I / (n eps0 c)``); not both.
    For vectors that are not a :class:`~swforce.force.ForceTorque` (a cluster total, a
    torque with its transport term); otherwise use :func:`to_si`.
    """
    if amplitude is not None and intensity is not None:
        raise ValueError("give amplitude or intensity, not both")
    _check(wavelength, index_medium)
    n = index_medium
    if intensity is not None:
        e2 = 2.0 * intensity / (n * EPS0 * C0)
    else:
        e2 = abs(1.0 if amplitude is None else amplitude) ** 2
    k = 2.0 * np.pi * n / (wavelength * NM)              # 1/m, in the host
    force = EPS0 * n**2 * e2 / (8.0 * k**2)               # N per unit f
    return force, force / k


def to_si(ft: ForceTorque, wavelength: float, *, amplitude: float | None = None,
          intensity: float | None = None, index_medium: float = 1.0) -> ForceTorque:
    """Force [N], torque [N m] and spin [N m] from a dimensionless :class:`~swforce.force.ForceTorque`.

    Arguments as :func:`si_scale`: vacuum ``wavelength`` [nm]; ``amplitude`` ``|E|`` [V/m]
    (default 1) or plane-wave ``intensity`` [W/m^2]; ``index_medium`` of the host.
    """
    f, t = si_scale(wavelength, amplitude=amplitude, intensity=intensity, index_medium=index_medium)
    return ForceTorque(force=f * ft.force, torque=t * ft.torque, spin=t * ft.spin)


def cross_section(f, wavelength: float, index_medium: float = 1.0):
    """``C = f / (4 k^2)`` [nm^2], ``k = 2 pi n / lambda0`` [1/nm]; ``f`` a component or a vector.

    For a unit-amplitude plane wave ``F = (n/c) I C``: ``f_z`` on axis gives ``C_pr``,
    ``t_z`` in circular polarisation gives ``sigma C_abs`` (see the module docstring
    for when).
    """
    _check(wavelength, index_medium)
    k = 2.0 * np.pi * index_medium / wavelength
    return np.asarray(f) / (4.0 * k**2)
