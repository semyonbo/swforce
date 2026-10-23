"""SMUTHI -> swforce: basis convention and the force on a particle of a finished SMUTHI simulation.

SMUTHI (Egel et al., JQSRT 273, 107846, 2021) solves light scattering by
particles near planar interfaces with T-matrices and Sommerfeld integrals.  The
force sums of :mod:`swforce.force` need, about the particle centre, the
**regular** coefficients ``a`` of everything that excites the particle and the
**outgoing** coefficients ``p`` of its own scattered field, in the
Condon-Shortley basis.  This module only translates: it reads ``a`` and ``p``
from a finished ``smuthi.simulation.Simulation`` and converts the basis.

Setting up and converging the SMUTHI calculation (Sommerfeld contour, ``l_max``,
solver accuracy) belongs to the project that uses it.  For a particle a few nm above a
metal the Sommerfeld contour must reach ``n_eff ~ l_max`` (SMUTHI's default stops near 2),
and the force must be converged in ``l_max``.

    sim = smuthi.simulation.Simulation(...); sim.run()
    ft = force_torque_from_simulation(sim, index=0)       # dimensionless, as swforce.force_torque
    aM, aN, pM, pN = coefficients(sim, index=0)            # the same coefficients, e.g. to superpose runs

The exciting field of particle ``j``
------------------------------------
``a_j = a0_j + (W b)_j``: ``a0_j`` is SMUTHI's initial-field expansion (the
incident field together with its layer response), ``W`` SMUTHI's coupling
operator (direct + layer-mediated; for a single particle only its own field
reflected by the interfaces), ``b`` the scattered-field coefficients of all
particles.  ``p_j = b_j``.

Convention (measured against SMUTHI's own fields in ``tests/test_adapter_smuthi.py``)
------------------------------------------------------------------------------------
SMUTHI's spherical vector wave functions differ from the Condon-Shortley
(treams) basis of the force sums by a per-mode factor::

    Psi_SMUTHI(tau, l, m) = -i sqrt(pi) s(m) * (M or N)_swforce,   s(m) = (-1)^m (m > 0), 1 (m <= 0)

so ``c_swforce = -i sqrt(pi) s(m) c_SMUTHI`` for regular and outgoing
coefficients alike; ``tau = 0`` is ``M`` (TE), ``tau = 1`` is ``N`` (TM).
"""

from __future__ import annotations

import contextlib
import io
import warnings
from functools import lru_cache

import numpy as np

if not hasattr(np, "trapz"):            # SMUTHI 2.2 calls np.trapz, removed in numpy 2
    np.trapz = np.trapezoid

with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import smuthi.fields as sfields

from ..basis import basis_size, ci_index
from ..force import ForceTorque, force_torque

__all__ = ["to_swforce", "from_swforce", "coefficients", "force_torque_from_simulation"]


# --------------------------------------------------------------------------
# Coefficient convention
# --------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _index_map(l_max: int, m_max: int):
    """SMUTHI slots, swforce slots, factor and ``(l, m)`` of every mode, per ``tau``."""
    out = []
    for tau in (0, 1):
        smuthi_idx, sw_idx, factor, ls, ms = [], [], [], [], []
        for l in range(1, l_max + 1):
            for m in range(-min(l, m_max), min(l, m_max) + 1):
                smuthi_idx.append(sfields.multi_to_single_index(tau, l, m, l_max, m_max))
                sw_idx.append(int(ci_index(l, m)))
                factor.append(-1j * np.sqrt(np.pi) * ((-1.0) ** m if m > 0 else 1.0))
                ls.append(l)
                ms.append(m)
        out.append(tuple(np.array(x) for x in (smuthi_idx, sw_idx, factor, ls, ms)))
    return tuple(out)


def to_swforce(coefficients, l_max: int, m_max: int | None = None):
    """SMUTHI coefficient vector -> ``(cM, cN)`` in the swforce convention.

    ``c_swforce = -i sqrt(pi) s(m) c_SMUTHI``; modes with ``|m| > m_max`` are
    zero.  Works for regular and outgoing coefficients alike.
    """
    m_max = l_max if m_max is None else m_max
    c = np.asarray(coefficients, dtype=complex).ravel()
    if c.size != sfields.blocksize(l_max, m_max):
        raise ValueError(f"{c.size} coefficients do not match SMUTHI's l_max={l_max}, m_max={m_max}")
    pair = []
    for smuthi_idx, sw_idx, factor, _, _ in _index_map(l_max, m_max):
        v = np.zeros(basis_size(l_max), dtype=complex)
        v[sw_idx] = factor * c[smuthi_idx]
        pair.append(v)
    return pair[0], pair[1]


def from_swforce(cM, cN, m_max: int | None = None):
    """Inverse of :func:`to_swforce`: ``(cM, cN)`` -> SMUTHI coefficient vector."""
    cM = np.asarray(cM, dtype=complex).ravel()
    cN = np.asarray(cN, dtype=complex).ravel()
    l_max = int(round(np.sqrt(cM.size + 1) - 1))
    if basis_size(l_max) != cM.size or cN.size != cM.size:
        raise ValueError("cM and cN must be swforce vectors of equal length l_max(l_max+2)")
    m_max = l_max if m_max is None else m_max
    c = np.zeros(sfields.blocksize(l_max, m_max), dtype=complex)
    for v, (smuthi_idx, sw_idx, factor, _, _) in zip((cM, cN), _index_map(l_max, m_max)):
        c[smuthi_idx] = v[sw_idx] / factor
    return c


# --------------------------------------------------------------------------
# Finished SMUTHI simulations
# --------------------------------------------------------------------------


def _check_particle(particle, layer_system):
    """The force formula needs a lossless host and a source-free shell ``R < r < d`` around the particle."""
    i = layer_system.layer_number(particle.position[2])
    n_host = layer_system.refractive_indices[i]
    if np.imag(n_host) != 0 or np.real(n_host) <= 0:
        raise ValueError("the force formula needs a lossless host medium around the particle")
    z = particle.position[2]
    d = min(z - layer_system.lower_zlimit(i), layer_system.upper_zlimit(i) - z)
    if d <= particle.circumscribing_sphere_radius():
        raise ValueError("the particle's circumscribing sphere crosses an interface: the expansion about its "
                         "centre (and the force formula) needs a source-free shell R < r < d in the host")


def coefficients(simulation, index: int = 0):
    """``(aM, aN, pM, pN)`` of particle ``index`` of a finished SMUTHI simulation, swforce convention.

    ``a = a0 + (W b)_index`` with SMUTHI's own coupling operator applied to the
    scattered-field coefficients ``b`` of all particles (``particle_list``
    order); ``p = b_index``.  Both are expanded to the particle's ``l_max``,
    ``m_max``.  The fields are linear in the initial field, so the coefficients
    of several runs (e.g. TE and TM plane waves) can be superposed before the
    force is taken.
    """
    particles = simulation.particle_list
    particle = particles[index]
    _check_particle(particle, simulation.layer_system)
    b = np.concatenate([np.asarray(pt.scattered_field.coefficients, dtype=complex) for pt in particles])
    Wb = simulation.linear_system.coupling_matrix.linear_operator.matvec(b)
    sizes = [sfields.blocksize(pt.l_max, pt.m_max) for pt in particles]
    start = sum(sizes[:index])
    a = np.asarray(particle.initial_field.coefficients, dtype=complex) + Wb[start:start + sizes[index]]
    p = np.asarray(particle.scattered_field.coefficients, dtype=complex)
    return (*to_swforce(a, particle.l_max, particle.m_max), *to_swforce(p, particle.l_max, particle.m_max))


def force_torque_from_simulation(simulation, index: int = 0) -> ForceTorque:
    """Force, torque and spin on particle ``index`` of a finished SMUTHI simulation.

    Dimensionless, as :func:`swforce.force_torque` (bilinear form); for a field
    amplitude ``|E|`` [V/m] in a host of index ``n`` the force is
    ``eps0 n^2 |E|^2 f / (8 k^2)`` (:func:`swforce.to_si`).

    What the result depends on (the SMUTHI calculation, not this function):

    * the field is expanded to the particle's own ``l_max``, so the
      ``(l_max, l_max + 1)`` terms of the sums are missing -- the same order of
      error as the truncation itself; check convergence in ``l_max``;
    * near an interface the Sommerfeld contour must reach far enough in
      ``k_parallel`` for the near field of the gap (``neff_max``);
    * ``(1 - T W) b = T a0`` is solved by SMUTHI; at high ``l_max`` next to a
      metal this system is badly scaled and the solution loses digits;
    * with SMUTHI's lookup-table coupling the direct coupling is stored in
      complex64 and interpolated, so ``a`` is good to ~1e-7 only.
    """
    return force_torque(*coefficients(simulation, index), form="bilinear")
