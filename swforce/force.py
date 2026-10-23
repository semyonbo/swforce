"""The force, torque and spin sums: a quadratic form ``Q`` and its bilinear form ``B``.

The Maxwell stress integral of Farsund & Felderhof (1996), evaluated
analytically, so the result is **exact** for the given multipole expansion --
no near-field integration, no discretisation, the only approximation is the
truncation at ``Nmax``.  Written in the Condon-Shortley convention that
``treams``, ``scipy`` and the source papers share.

**Momentum in minus momentum out.**  About the particle the field is a regular
part ``a`` (sources outside) plus an outgoing part ``p`` (the particle).  Since
``j_l = (h_l^(1) + h_l^(2)) / 2``, the incoming waves carry ``a/2`` and the
outgoing ones ``(a + 2p)/2``, and the stress integral on a sphere at infinity is

.. math::  \\mathbf F \\propto \\mathcal Q[a] - \\mathcal Q[a + 2p],
           \\qquad \\mathcal Q[c] = \\oint |\\mathbf A_c(\\hat r)|^2\\,\\hat r\\,d\\Omega

with ``A_c`` the far-field pattern of the waves ``c``.  :func:`flux` is ``Q``
written out in coefficients (the "ladder" sums), returned as force, torque and spin
vectors; internally six packed sums (axial and ``x + i y``, see :func:`_pack`).  Derivation:
Farsund & Felderhof, Physica A 227, 108 (1996).

**Bilinear form.**  ``Q`` is a sum of terms ``<kappa conj(c_i) c_j>`` (``<>`` =
Re, Im or nothing).  :func:`flux_bilinear` ``B(u, v)`` is the same sum with the
conjugated factor taken from ``u`` and the plain one from ``v``, so

.. math::  \\mathcal Q[u + v] = \\mathcal Q[u] + \\mathcal Q[v] + B(u, v) + B(v, u),
           \\qquad B(c, c) = \\mathcal Q[c].

Expanding ``Q[a + 2p]`` with it cancels ``Q[a]`` *algebraically*:

.. math::  \\mathcal Q[a] - \\mathcal Q[a + 2p] = -2B(a,p) - 2B(p,a) - 4B(p,p).

:func:`force_torque` with ``form="bilinear"`` uses it.  It is equal to the quadratic form
in exact arithmetic and much better in floating point: near a substrate the
regular coefficients grow like ``10^l`` and ``Q[a]`` can exceed the force by
``10^100``, which the quadratic form subtracts away digit by digit.  It is also
**additive in** ``a`` at fixed ``p`` -- which is what makes
:func:`force_torque_split` exact.

Inputs are ordered by the combined index ``ci = l(l+1)+m`` of
:mod:`swforce.basis`, split into the magnetic/TE multipole (``M``) and the
electric/TM one (``N``).  Coefficients from a package whose harmonics are *not*
Condon-Shortley go through that package's adapter, never through a flag here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .basis import ci_index, mode_indices, nmax_from_size

#: ``force_torque(form='quadratic')`` warns when the cancellation ratio exceeds this (fewer than ~6 digits left).
_CANCELLATION_WARN = 1e10


@dataclass
class ForceTorque:
    """Force, torque and spin, each a length-3 vector in Cartesian components."""

    force: np.ndarray
    torque: np.ndarray
    spin: np.ndarray

    def __iter__(self):
        return iter((self.force, self.torque, self.spin))

    def __add__(self, other: "ForceTorque") -> "ForceTorque":
        return ForceTorque(self.force + other.force, self.torque + other.torque, self.spin + other.spin)

    def __sub__(self, other: "ForceTorque") -> "ForceTorque":
        return ForceTorque(self.force - other.force, self.torque - other.torque, self.spin - other.spin)

    def scaled(self, factor: float) -> "ForceTorque":
        """All three vectors multiplied by a real ``factor`` (e.g. a unit conversion)."""
        return ForceTorque(factor * self.force, factor * self.torque, factor * self.spin)


def _as_pair(cM, cN):
    cM = np.asarray(cM, dtype=complex).ravel()
    cN = np.asarray(cN, dtype=complex).ravel()
    if cM.size != cN.size:
        raise ValueError("the M and N coefficient vectors must have the same length")
    return cM, cN


def _shifted(cM, cN, nmax):
    """The four neighbour lookups the selection rules need.

    Returns ``(l, m)`` and, for each mode, its partners at ``(l, m+1)`` and at
    ``(l+1, m)``, ``(l+1, m+1)``, ``(l+1, m-1)``.  Reaching past the truncation
    or past ``|m| <= l`` yields zero: the truncated sums simply drop the
    ``(Nmax, Nmax + 1)`` pairs.  For plane-wave illumination the force is then
    observed to converge from below in ``Nmax`` (not a theorem).
    """
    l, m = mode_indices(nmax)
    slot = ci_index(l, m)

    pad = 2 * nmax + 3
    M = np.concatenate([cM, np.zeros(pad, dtype=complex)])
    N = np.concatenate([cN, np.zeros(pad, dtype=complex)])

    def take(shift):
        idx = slot + shift
        return M[idx].copy(), N[idx].copy()

    up = 2 * l + 2                      # ci(l+1, m) - ci(l, m)
    Mp, Np = take(1)                    # (l, m+1)
    top = m >= l                        # ...which does not exist at m = l
    Mp[top] = 0.0
    Np[top] = 0.0
    return (l, m, Mp, Np) + take(up) + take(up + 1) + take(up - 1)



def _cancellation_ratio(aM, aN, pM, pN) -> float:
    """Size of the terms of ``Q[a]`` over the size of the terms that build the force.

    Every term of the sums pairs a mode with itself or with a neighbour (``l+1`` and/or
    ``m+1``, M with N).  ``Q[a]`` is built from ``|a| |a|`` pairs, the force from ``|a| |p|``
    pairs, so ``sum |a||a_nb| / sum |a||p_nb|`` estimates ``|Q[a]| / |F|`` without being
    fooled by a force that vanishes by symmetry.  ``0`` when nothing is scattered.
    """
    A = np.abs(aM) + np.abs(aN)
    P = np.abs(pM) + np.abs(pN)
    top = max(A.max(initial=0.0), P.max(initial=0.0))
    if top == 0 or P.max(initial=0.0) == 0:
        return 0.0
    A, P = A / top, P / top                                    # keeps the products far from overflow
    nmax = nmax_from_size(A.size)
    _, _, Ap, _, A1, _, A1p, _, A1m, _ = _shifted(A.astype(complex), A.astype(complex), nmax)
    _, _, Pp, _, P1, _, P1p, _, P1m, _ = _shifted(P.astype(complex), P.astype(complex), nmax)
    nb_a = np.real(A + Ap + A1 + A1p + A1m)
    nb_p = np.real(P + Pp + P1 + P1p + P1m)
    noise = np.sum(A * nb_a)
    signal = np.sum(P * nb_a) + np.sum(A * nb_p)
    return float(noise / signal) if signal > 0 else 0.0

def _ladder(l, m):
    """Ladder coefficients ``K_lm`` (l -> l+1, same m), ``L_l`` (l -> l+1, m -> m+-1),
    ``Lambda_lm = sqrt((l-m)(l+m+1))`` (the L_+ operator), ``(l+m+1)(l+m+2)``, ``(l-m+1)(l-m+2)``
    under square roots, and ``1/(l(l+1))``: the coefficients of the sums of
    Farsund & Felderhof, Physica A 227, 108 (1996).
    """
    K = np.sqrt(l * (l + 2) * (l - m + 1) * (l + m + 1) / ((2 * l + 1) * (2 * l + 3))) / (l + 1)
    L = np.sqrt(l * (l + 2) / ((2 * l + 1) * (2 * l + 3))) / (l + 1)
    lad = np.sqrt((l - m) * (l + m + 1))
    up_p = np.sqrt((l + m + 1) * (l + m + 2))
    up_m = np.sqrt((l - m + 1) * (l - m + 2))
    inv = 1.0 / (l * (l + 1))
    return K, L, lad, up_p, up_m, inv


def _flux_sums(cM, cN):
    """The six packed sums of ``Q[c]`` for one coefficient pair (internal; see :func:`flux`).

    ``cM``, ``cN`` are the magnetic and electric multipole coefficients of *one*
    set of purely incoming or purely outgoing waves, indexed by ``ci = l(l+1)+m``.
    Returns ``(fz, fxy, tz, txy, sz, sxy)``.  ``(fz, fxy)``: ``oint |A_c|^2 r dOmega``
    (the momentum the waves carry, up to ``eps0 n^2 / (2 k^2)``), ``fxy = x + i y``;
    ``(tz, txy)``: the angular momentum, ``txy = x + i y``; ``(sz, sxy)``: its spin
    part, packed differently -- ``S_z = -sz`` and ``S_x + i S_y = i sxy``
    (:func:`_pack` does this unpacking).

    On its own this is not a force -- a regular field carries no net flux; the
    force is a difference (:func:`force_torque`).
    """
    cM, cN = _as_pair(cM, cN)
    nmax = nmax_from_size(cM.size)
    l, m, Mp, Np, M1, N1, M1p, N1p, M1m, N1m = _shifted(cM, cN, nmax)
    K, L, lad, up_p, up_m, inv = _ladder(l, m)

    # --- linear momentum ---
    fz = 2 * np.sum(
        m * inv * np.real(cM * np.conj(cN))
        + K * np.imag(M1 * np.conj(cM) + N1 * np.conj(cN))
    )
    fxy = np.sum(
        inv * lad * (np.conj(Mp) * cN + np.conj(Np) * cM)
        - 1j * L * (
            up_p * (cM * np.conj(M1p) + cN * np.conj(N1p))
            + up_m * (M1m * np.conj(cM) + N1m * np.conj(cN))
        )
    )

    # --- angular momentum (diagonal in l: no ladder, no cross terms) ---
    tz = np.sum(m * (np.abs(cM) ** 2 + np.abs(cN) ** 2))
    txy = np.sum(lad * (cM * np.conj(Mp) + cN * np.conj(Np)))

    # --- its spin part; same shape with M and N swapped in every mixed term ---
    sz = np.sum(
        -m * inv * (np.abs(cM) ** 2 + np.abs(cN) ** 2)
        - 2 * K * np.imag(M1 * np.conj(cN) + N1 * np.conj(cM))
    )
    sxy = np.sum(
        -1j * inv * lad * (np.conj(Mp) * cM + np.conj(Np) * cN)
        - L * (
            up_p * (cM * np.conj(N1p) + cN * np.conj(M1p))
            + up_m * (M1m * np.conj(cN) + N1m * np.conj(cM))
        )
    )
    return fz, fxy, tz, txy, sz, sxy


def _bilinear_sums(uM, uN, vM, vN):
    """The six packed sums of the bilinear form ``B(u, v)`` (internal; see :func:`flux_bilinear`).

    Line by line the same sums as :func:`_flux_sums`, with the **conjugated** factor
    of every product taken from ``u`` and the plain one from ``v``.  Hence, for
    each of the six outputs,

        Q(u + v) = Q(u) + Q(v) + B(u, v) + B(v, u),     B(c, c) = Q(c).

    ``B(u, v)`` alone is not symmetric and has no physical meaning; the pair
    ``B(u, v) + B(v, u) = 2 Re oint A_u* . A_v r dOmega`` is the momentum carried
    by the *interference* of the two sets of waves -- see :func:`interference`.
    """
    uM, uN = _as_pair(uM, uN)
    vM, vN = _as_pair(vM, vN)
    if uM.size != vM.size:
        raise ValueError("u and v must be truncated at the same Nmax")
    nmax = nmax_from_size(uM.size)
    l, m, uMp, uNp, uM1, uN1, uM1p, uN1p, uM1m, uN1m = _shifted(uM, uN, nmax)
    _, _, vMp, vNp, vM1, vN1, vM1p, vN1p, vM1m, vN1m = _shifted(vM, vN, nmax)
    K, L, lad, up_p, up_m, inv = _ladder(l, m)

    fz = 2 * np.sum(
        m * inv * np.real(vM * np.conj(uN))
        + K * np.imag(vM1 * np.conj(uM) + vN1 * np.conj(uN))
    )
    fxy = np.sum(
        inv * lad * (np.conj(uMp) * vN + np.conj(uNp) * vM)
        - 1j * L * (
            up_p * (vM * np.conj(uM1p) + vN * np.conj(uN1p))
            + up_m * (vM1m * np.conj(uM) + vN1m * np.conj(uN))
        )
    )
    tz = np.sum(m * (np.conj(uM) * vM + np.conj(uN) * vN))
    txy = np.sum(lad * (vM * np.conj(uMp) + vN * np.conj(uNp)))
    sz = np.sum(
        -m * inv * (np.conj(uM) * vM + np.conj(uN) * vN)
        - 2 * K * np.imag(vM1 * np.conj(uN) + vN1 * np.conj(uM))
    )
    sxy = np.sum(
        -1j * inv * lad * (np.conj(uMp) * vM + np.conj(uNp) * vN)
        - L * (
            up_p * (vM * np.conj(uN1p) + vN * np.conj(uM1p))
            + up_m * (vM1m * np.conj(uN) + vN1m * np.conj(uM))
        )
    )
    return fz, fxy, tz, txy, sz, sxy


def _pack(fz, fxy, tz, txy, sz, sxy) -> ForceTorque:
    """Six sums -> Cartesian vectors.  Real-linear, so it commutes with sums of forms.

    The spin packing and axial sign are the corrections pinned by the physics tests
    (spin flips with helicity, converges to the torque for a Rayleigh absorber).
    """
    return ForceTorque(
        force=np.real(np.array([np.real(fxy), np.imag(fxy), np.real(fz)])),
        torque=np.real(np.array([np.real(txy), np.imag(txy), np.real(tz)])),
        spin=np.real(np.array([np.real(1j * sxy), np.imag(1j * sxy), -np.real(sz)])),
    )


def _combine(*weighted):
    """Sum ``w * sums`` over ``(w, six_sums)`` pairs, component by component."""
    out = [0.0] * 6
    for w, sums in weighted:
        out = [o + w * s for o, s in zip(out, sums)]
    return out


def flux(cM, cN) -> ForceTorque:
    """``Q[c]``: momentum, angular momentum and spin carried by one set of purely outgoing
    (or purely incoming) waves with coefficients ``(cM, cN)``, as Cartesian vectors.

    For the force part ``Q[c] = closed-integral |A_c(rhat)|^2 rhat dOmega``, ``A_c`` the
    far-field pattern.  Same units as :func:`force_torque`, which in the quadratic form is
    ``flux(a) - flux(a + 2p)`` (incoming minus outgoing momentum).
    """
    return _pack(*_flux_sums(cM, cN))


def flux_bilinear(uM, uN, vM, vN) -> ForceTorque:
    """``B(u, v)``: the sums of :func:`flux` with the conjugated factor of every product taken
    from ``u`` and the plain one from ``v``, as Cartesian vectors.

    ``flux(u + v) = flux(u) + flux(v) + B(u, v) + B(v, u)`` and ``B(c, c) = flux(c)``.  Alone
    ``B(u, v)`` is not symmetric and has no physical meaning; the pair is :func:`interference`.
    """
    return _pack(*_bilinear_sums(uM, uN, vM, vN))


def interference(uM, uN, vM, vN) -> ForceTorque:
    """``B(u, v) + B(v, u)``: momentum, angular momentum and spin carried by the
    interference of two wave sets -- ``2 Re oint A_u* . A_v r dOmega`` for the force.

    Symmetric in ``u`` and ``v`` and additive in each, so
    ``interference(u1 + u2, v) = interference(u1, v) + interference(u2, v)``.
    Dimensionless, like :func:`force_torque`.
    """
    return _pack(*_combine((1.0, _bilinear_sums(uM, uN, vM, vN)), (1.0, _bilinear_sums(vM, vN, uM, uN))))


def force_torque(aM, aN, pM, pN, scattered: bool = True, form: str = "quadratic") -> ForceTorque:
    """Force, torque and spin from Condon-Shortley multipole coefficients.

    ``(aM, aN)`` are the **regular** (``j_l``) coefficients of the field exciting
    the particle -- the incident beam plus anything that comes back to it, e.g.
    its own field reflected by a substrate -- expanded about the particle
    centre; ``(pM, pN)`` are the **outgoing** (``h_l^(1)``) coefficients of its
    scattered field.  All indexed by ``ci = l(l+1)+m``.  With ``scattered=True``
    (the default) the total outgoing field ``2p + a`` is formed here; pass
    ``scattered=False`` if you already hold the total.

    ``form="quadratic"`` (default) is the literal ``Q[a] - Q[a + 2p]`` -- the
    original swforce computation, reproduced bit for bit.  ``form="bilinear"``
    evaluates ``-2B(a,p) - 2B(p,a) - 4B(p,p)``: equal in exact arithmetic, but
    free of the ``|a|^2`` cancellation -- use it whenever the regular
    coefficients are large (a particle near a substrate, a dense cluster at
    high ``lmax``).  The SMUTHI adapter always uses it.

    Dimensionless: for a field of amplitude ``|E|`` in a medium of index ``n``
    the force in newtons is ``eps0 n^2 |E|^2 F / (8 k^2)`` (:func:`swforce.to_si`).
    """
    aM = np.asarray(aM, dtype=complex).ravel()
    aN = np.asarray(aN, dtype=complex).ravel()
    pM = np.asarray(pM, dtype=complex).ravel()
    pN = np.asarray(pN, dtype=complex).ravel()
    if not (aM.size == aN.size == pM.size == pN.size):
        raise ValueError("all four coefficient vectors must have the same length")

    if form == "quadratic":
        # the original swforce operations, in the original order
        if scattered:
            pM, pN = 2 * pM + aM, 2 * pN + aN

        fz_i, fxy_i, tz_i, txy_i, sz_i, sxy_i = _flux_sums(aM, aN)
        fz_o, fxy_o, tz_o, txy_o, sz_o, sxy_o = _flux_sums(pM, pN)

        fz, fxy = fz_i - fz_o, fxy_i - fxy_o
        tz, txy = tz_i - tz_o, txy_i - txy_o
        sz, sxy = sz_i - sz_o, sxy_i - sxy_o

        # Q[a] and Q[a + 2p] are subtracted: their terms are products of neighbouring a's, the force
        # is built from products of neighbouring a's and p's; the ratio of the two sizes is the loss.
        lost = _cancellation_ratio(aM, aN, (pM - aM) / 2, (pN - aN) / 2)
        if lost > _CANCELLATION_WARN:
            warnings.warn(
                f"force_torque(form='quadratic'): the terms of Q[a] exceed those of the force by {lost:.1e}, "
                f"so about {np.log10(lost):.0f} of 16 digits cancel in Q[a] - Q[a+2p].  Use form='bilinear'.",
                RuntimeWarning, stacklevel=2)

        return ForceTorque(
            force=np.real(np.array([np.real(fxy), np.imag(fxy), fz])),
            torque=np.real(np.array([np.real(txy), np.imag(txy), tz])),
            spin=np.real(np.array([np.real(1j * sxy), np.imag(1j * sxy), -sz])),
        )
    if form == "bilinear":
        if not scattered:
            pM, pN = (pM - aM) / 2, (pN - aN) / 2
        return _pack(*_combine(
            (-2.0, _bilinear_sums(aM, aN, pM, pN)),
            (-2.0, _bilinear_sums(pM, pN, aM, aN)),
            (-4.0, _bilinear_sums(pM, pN, pM, pN)),
        ))
    raise ValueError(f"form={form!r}: expected 'quadratic' or 'bilinear'")


def force_torque_split(parts: Mapping[str, tuple], pM, pN) -> dict:
    """Split the force exactly by the parts of the exciting field.

    ``parts`` maps a name to the regular coefficients ``(aM, aN)`` of one part of
    the exciting field; the parts must add up to the whole exciting field ``a``.
    Returns a dict with one :class:`ForceTorque` per part,

        F_i = -2 [B(a_i, p) + B(p, a_i)] = -2 interference(a_i, p),

    plus ``"recoil"`` ``= -4 B(p, p)`` (the momentum the scattered field would
    carry away in free space, with a minus sign) and ``"total"``.  Because the
    force is additive in ``a`` at fixed ``p`` there are no cross terms between
    parts: the parts and the recoil sum to :func:`force_torque` of the whole
    ``a`` to rounding (``tests/test_force_algebra.py``).  The names ``"recoil"`` and
    ``"total"`` are reserved.
    """
    pM, pN = _as_pair(pM, pN)
    if not parts:
        raise ValueError("give at least one part of the exciting field")
    reserved = {"recoil", "total"} & set(parts)
    if reserved:
        raise ValueError(f"part names {sorted(reserved)} are reserved")
    out = {}
    for name, (aM, aN) in parts.items():
        aM, aN = _as_pair(aM, aN)
        if aM.size != pM.size:
            raise ValueError(f"part {name!r} is truncated at a different Nmax than p")
        out[name] = interference(aM, aN, pM, pN).scaled(-2.0)
    out["recoil"] = _pack(*_bilinear_sums(pM, pN, pM, pN)).scaled(-4.0)
    total = out["recoil"]
    for name in parts:
        total = total + out[name]
    out["total"] = total
    return out
