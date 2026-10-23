"""Feed ``treams`` coefficients into the exact multipole force sums.

``treams`` supplies the physics inputs -- plane waves, T-matrices, clusters --
and this module reindexes its spherical-wave coefficients into the layout
:mod:`swforce.force` expects.  Nothing here recomputes physics, and after
:func:`split` nothing here changes a coefficient's value either.

**Only the reindexing is real work.**  ``treams`` uses ``SphericalWaveBasis``
ordered by ``l``, then ``m = -l..l``, with the polarisation index fastest and
*descending* (``1, 0``), so in the ``parity`` poltype the **electric multipole
comes first** in the array while the **index value** ``pol = 0`` is the
*magnetic* one (``M``) and ``pol = 1`` the electric (``N``).  Those two facts are easy to conflate, and :func:`split` is safe from it
because it selects by the mask ``pol == 0`` rather than by array position: an
implementation that indexed by position would silently swap ``M`` and ``N``, and
because both are populated in most problems it would not crash, only give wrong
answers.  The force sums want two separate vectors indexed by ``ci = l(l+1)+m``.

**No rephasing.**  ``treams``' harmonics carry the Condon-Shortley phase, and the
Farsund sums of :mod:`swforce.force` were derived with it, so treams coefficients
are already in the right convention.

Everything here requires ``poltype="parity"``.  ``treams``' default is
``helicity``, whose ``pol`` indices are the two circular polarisations and do not
map onto ``M``/``N`` mode-by-mode; the converters refuse it rather than silently
producing nonsense -- call ``.changepoltype()`` first, or build with
``poltype="parity"`` from the start (as the examples in ``README.md`` do).

Moving and rotating the particle -- treams does all of it
---------------------------------------------------------
This package implements **no** translation or rotation machinery, and does not
need to.  The force sums only require that the expansion be centred on the
particle; getting the coefficients into that frame is ``treams``' job::

    placed = treams.TMatrix.cluster([tm], [[0, 0, z]])     # particle at z
    a_inc  = beam.expand(placed.basis)                     # beam about that centre
    ft     = force_torque(a_inc, placed @ a_inc)

    turned = tm.rotate(alpha, beta, gamma)                 # particle orientation

``expand`` applies the vector translation-addition theorem; ``rotate`` applies the
Wigner rotation.  Both are treams', both are exercised by
``tests/test_adapter_treams.py``; a sphere scanned through a standing wave
reproduces the exact analytic result ``F_z ∝ sin(2kz)``.

For a *single* plane wave the translation is only a global phase, so the force is
position-independent -- as it must be.  Structured fields (standing waves,
Gaussian beams, lattice near fields) are where the position dependence lives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import treams

from ..basis import basis_size, ci_index
from ..force import ForceTorque
from ..force import force_torque as _force_torque


def _basis_arrays(basis):
    """``(l, m, pol)`` int arrays from a treams ``SphericalWaveBasis``."""
    l = np.asarray(basis.l, dtype=int)
    m = np.asarray(basis.m, dtype=int)
    pol = np.asarray(basis.pol, dtype=int)
    return l, m, pol


def _check_single_origin(basis) -> None:
    pidx = np.asarray(getattr(basis, "pidx", np.zeros(len(basis.l))), dtype=int)
    if pidx.max() != 0:
        raise ValueError(
            "coefficients must be expanded about a single origin -- the force sums "
            "assume one expansion centre.  For a cluster, re-expand into a global "
            "basis (treams' TMatrix.cluster(...).expand(global_basis)) first."
        )


def _check_parity(poltype: str) -> None:
    if poltype != "parity":
        raise ValueError(
            f"poltype={poltype!r}: swforce needs the parity basis, where pol=0 is the "
            "magnetic (M/TE) multipole and pol=1 the electric (N/TM) one.  Convert "
            "with treams' .changepoltype() or build the T-matrix and beam with "
            "poltype='parity'."
        )


def split(coefficients, basis, poltype: str = "parity"):
    """Reindex a treams coefficient vector into ``(cM, cN)`` -- values unchanged.

    ``coefficients`` is any array-like of length ``len(basis)`` (a
    ``treams.PhysicsArray`` works directly).  Returns the magnetic (``pol=0``)
    and electric (``pol=1``) multipole coefficients as two complex vectors of
    length ``Nmax(Nmax+2)`` indexed by ``ci = l(l+1)+m``, zero-filled for any
    mode the treams basis happens to omit.

    This is a pure permutation: no phase, no scale.  treams and the Farsund sums
    share the Condon-Shortley convention.
    """
    _check_parity(poltype)
    _check_single_origin(basis)

    c = np.asarray(coefficients, dtype=complex).ravel()
    l, m, pol = _basis_arrays(basis)
    if c.size != l.size:
        raise ValueError(f"coefficient vector has {c.size} entries, basis has {l.size}")

    nmax = int(l.max())
    cM = np.zeros(basis_size(nmax), dtype=complex)
    cN = np.zeros(basis_size(nmax), dtype=complex)

    idx = ci_index(l, m)
    is_m = pol == 0                     # by mask, never by array position
    cM[idx[is_m]] = c[is_m]
    cN[idx[~is_m]] = c[~is_m]
    return cM, cN


def force_torque(incident, scattered, basis=None,
                 poltype: str = "parity") -> ForceTorque:
    """Force, torque and spin straight from treams coefficient vectors.

    ``incident`` are the illumination coefficients expanded about the particle
    (``treams.plane_wave(...).expand(tm.basis)``) and ``scattered`` the T-matrix
    response (``tm @ a_inc``).  ``basis`` defaults to the one carried by the
    ``incident`` array, so the usual call is simply::

        a_inc = treams.plane_wave(kvec, epol, k0=k0, material=vac,
                                  poltype="parity").expand(tm.basis)
        a_sca = tm @ a_inc
        ft = force_torque(a_inc, a_sca)

    This reindexes and sums -- no phase, no convention knob, because ``treams``
    and the sums already share the Condon-Shortley convention.

    Returns force/torque/spin in the dimensionless units of
    :mod:`swforce.force`; :func:`swforce.to_si` converts them to SI, and for a
    unit-amplitude plane wave ``force / (4 k^2)`` is directly a cross section (:func:`swforce.cross_section`).
    """
    if basis is None:
        basis = _infer_basis(incident, scattered)

    aM, aN = split(incident, basis, poltype=poltype)
    pM, pN = split(scattered, basis, poltype=poltype)
    return _force_torque(aM, aN, pM, pN, scattered=True)


def _infer_basis(incident, scattered):
    for arr in (incident, scattered):
        basis = getattr(arr, "basis", None)
        if basis is not None:
            return basis
    raise ValueError(
        "could not read a basis off the coefficient arrays -- pass basis=... "
        "explicitly (e.g. the T-matrix's tm.basis)"
    )


# --------------------------------------------------------------------------
# Arrays
# --------------------------------------------------------------------------


@dataclass
class ClusterForceTorque:
    """Per-particle force, torque and spin for a cluster, plus the correct totals.

    ``force[j]`` is the force on particle ``j``; ``torque[j]`` is its torque
    **about its own centre**, which is what a rigid particle actually spins
    about.  Referring those to a common origin is not a sum -- it needs a
    transport term -- so :meth:`total_torque` does it for you.
    """

    positions: np.ndarray
    force: np.ndarray
    torque: np.ndarray
    spin: np.ndarray
    k: float

    def __len__(self) -> int:
        return len(self.positions)

    def total_force(self) -> np.ndarray:
        """Net force on the whole cluster."""
        return self.force.sum(axis=0)

    def total_torque(self, about=(0.0, 0.0, 0.0)) -> np.ndarray:
        """Net torque about ``about``, including the ``k (r x F)`` transport term.

        The ``k`` is not decoration.  These sums carry torque with one more power
        of ``k`` than force (:mod:`swforce.units`), so ``k (r x F)`` is the term
        with the same units as ``T``.  Adding the per-particle torques *without*
        it is a common and silent error -- for a spread-out cluster the transport
        term usually dominates, and the answer comes out both wrong and
        plausible.
        """
        arm = self.positions - np.asarray(about, dtype=float)
        return self.torque.sum(axis=0) + self.k * np.cross(arm, self.force).sum(axis=0)


def _check_multiple_scattering(tmatrix, pidx) -> None:
    """Refuse a cluster whose interaction has not been solved.

    ``treams.TMatrix.cluster(...)`` alone is block diagonal: each particle
    responds only to the external beam.  Feeding that in computes forces with the
    interparticle coupling switched off -- which is usually the entire physics of
    interest (optical binding, lattice resonances, BIC-induced forces) and looks
    perfectly reasonable in the output.  ``.interaction.solve()`` is the fix.
    """
    t = np.asarray(tmatrix)
    if len(np.unique(pidx)) < 2:
        return
    same = pidx[:, None] == pidx[None, :]
    coupling = np.linalg.norm(t[~same])
    if coupling <= 1e-12 * np.linalg.norm(t):
        raise ValueError(
            "this cluster T-matrix is block diagonal, so the particles do not see "
            "each other -- call .interaction.solve() on it first.  Using it as-is "
            "would silently discard all interparticle coupling."
        )


def cluster_force_torque(solved, illumination) -> ClusterForceTorque:
    """Force and torque on **every particle** of a cluster, in one call.

    ``solved`` is a treams cluster T-matrix with the multiple scattering already
    solved::

        cluster = treams.TMatrix.cluster(tmatrices, positions)
        solved  = cluster.interaction.solve()
        beam    = treams.plane_wave(..., poltype="parity")

        result = swforce.adapters.treams.cluster_force_torque(solved, beam)
        result.force            # (N, 3), per particle
        result.total_torque()   # about the origin, transport term included

    What it does per particle, which is the part that is easy to get wrong: the
    coefficients fed to the sums must be expanded about **that particle's**
    centre, and the regular part is *not* the external beam.  It is the beam
    **plus every other particle's scattered field re-expanded about ``j``** as a
    regular series (the translation-addition theorem).  Using the bare beam
    throws away the interparticle coupling entirely.

    Validity is geometric, not a distance in wavelengths: each particle's
    circumscribing sphere must exclude every other particle's.  Two long
    particles side by side can violate this while their surfaces are still far
    apart.
    """
    basis = solved.basis
    _check_parity(str(solved.poltype))

    positions = np.asarray(basis.positions, dtype=float)
    pidx = np.asarray(basis.pidx, dtype=int)
    if positions.ndim != 2 or len(positions) < 1:
        raise ValueError("expected a cluster basis carrying one position per particle")
    _check_multiple_scattering(solved, pidx)

    k0, material, poltype = solved.k0, solved.material, solved.poltype
    lmax = int(np.asarray(basis.l).max())
    n_particles = len(positions)

    incident = np.asarray(illumination.expand(basis))
    scattered = np.asarray(solved @ incident)

    force = np.empty((n_particles, 3))
    torque = np.empty((n_particles, 3))
    spin = np.empty((n_particles, 3))

    for j in range(n_particles):
        here = treams.SphericalWaveBasis.default(lmax, 1, [list(positions[j])])
        others = [l for l in range(n_particles) if l != j]

        exciting = np.asarray(illumination.expand(here))
        if others:
            there = treams.SphericalWaveBasis.default(
                lmax, len(others), [list(positions[l]) for l in others]
            )
            # singular about the neighbours -> regular about j
            translate = np.asarray(
                treams.expand((here, there), ("regular", "singular"),
                              k0=k0, material=material, poltype=poltype)
            )
            exciting = exciting + translate @ np.concatenate(
                [scattered[pidx == l] for l in others]
            )

        got = force_torque(exciting, scattered[pidx == j], basis=here,
                           poltype=str(poltype))
        force[j], torque[j], spin[j] = got.force, got.torque, got.spin

    return ClusterForceTorque(
        positions=positions, force=force, torque=torque, spin=spin,
        # wavenumber in the embedding medium -- what the units and the torque
        # transport term are expressed in
        k=float(np.real(k0 * material.n)),
    )
