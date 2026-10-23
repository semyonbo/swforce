"""Independent reference implementations for the swforce test suite.

Everything the library is compared against lives here, written from the
textbook definitions with numpy, scipy and mpmath only.  **Nothing in this
module imports swforce** -- not even the index helpers -- so a bug in the
library cannot hide behind the same bug in its yardstick.  Every function is
itself validated in ``tests/test_oracles.py`` (finite differences, Maxwell's
equations, orthonormality, analytic limits).

Conventions (those of treams, scipy and the Farsund-Felderhof sums):

* time dependence ``exp(-i omega t)``;
* spherical harmonics with the Condon-Shortley phase, ``scipy.special.sph_harm_y``;
* vector spherical harmonic ``X_lm = L Y_lm / sqrt(l(l+1))``, ``L = -i r x grad``;
* ``M_lm = z_l(kr) X_lm``,
  ``N_lm = ([x z_l]'/x) rhat x X_lm + i sqrt(l(l+1)) (z_l/x) Y_lm rhat``,  ``x = kr``,
  with ``z_l = j_l`` (regular), ``h_l^(1)`` (outgoing) or ``h_l^(2)`` (incoming);
* ``E = sum c^M M + c^N N``,  ``H = -(i/Z) sum (c^M N + c^N M)``,  ``Z = Z0 / n``;
* coefficient vectors indexed by ``ci = l(l+1) + m`` (slot ``ci - 1``), length
  ``lmax(lmax+2)``, magnetic (``M``, TE) and electric (``N``, TM) kept apart.

Lengths and field units are whatever the caller uses; the SI constants below are
needed only by :func:`dipole_fields` and the unit tests.
"""

from __future__ import annotations

import numpy as np
from scipy.special import sph_harm_y, spherical_jn, spherical_yn

C0 = 299792458.0                      # speed of light [m/s]
EPS0 = 8.8541878188e-12               # vacuum permittivity [F/m], CODATA 2022 (written out, not imported)
MU0 = 1.0 / (EPS0 * C0**2)            # vacuum permeability [H/m], consistent with EPS0


# --------------------------------------------------------------------------
# Index bookkeeping (deliberately re-derived, not imported from swforce.basis)
# --------------------------------------------------------------------------


def modes(lmax):
    """``[(l, m), ...]`` in combined-index order: (1,-1), (1,0), (1,1), (2,-2), ..."""
    return [(l, m) for l in range(1, lmax + 1) for m in range(-l, l + 1)]


def slot(l, m):
    """0-based array slot of mode ``(l, m)``: ``l(l+1) + m - 1``."""
    return l * (l + 1) + m - 1


def size(lmax):
    """Length of a coefficient vector truncated at ``lmax``."""
    return len(modes(lmax))


def random_coefficients(rng, lmax, count=2, scale=1.0):
    """``count`` complex Gaussian vectors of length ``size(lmax)``."""
    n = size(lmax)
    return tuple(scale * (rng.normal(size=n) + 1j * rng.normal(size=n)) for _ in range(count))


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def to_spherical(xyz):
    """``(r, theta, phi)`` of Cartesian points ``(..., 3)``."""
    xyz = np.asarray(xyz, dtype=float)
    r = np.linalg.norm(xyz, axis=-1)
    theta = np.arccos(np.clip(xyz[..., 2] / r, -1.0, 1.0))
    phi = np.arctan2(xyz[..., 1], xyz[..., 0])
    return r, theta, phi


def rhat(theta, phi):
    """Radial unit vector, Cartesian components ``(..., 3)``."""
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)


def theta_hat(theta, phi):
    return np.stack([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)], -1)


def phi_hat(theta, phi):
    return np.stack([-np.sin(phi), np.cos(phi), 0.0 * np.asarray(theta, dtype=float)], -1)


def sphere_quadrature(n_theta, n_phi):
    """Nodes and weights on the unit sphere: Gauss-Legendre in ``cos(theta)``, uniform in ``phi``.

    Exact for spherical harmonics up to degree ``2 n_theta - 1`` and azimuthal
    order ``< n_phi``.  Returns flat arrays ``theta, phi, w`` with ``sum(w) = 4 pi``.
    """
    x, wx = np.polynomial.legendre.leggauss(n_theta)
    theta = np.arccos(x)
    phi = np.arange(n_phi) * (2 * np.pi / n_phi)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    W = wx[:, None] * np.full(n_phi, 2 * np.pi / n_phi)[None, :]
    return TH.ravel(), PH.ravel(), W.ravel()


# --------------------------------------------------------------------------
# Scalar and vector spherical harmonics
# --------------------------------------------------------------------------


def Y(l, m, theta, phi):
    """Orthonormal Condon-Shortley ``Y_lm`` (``scipy.special.sph_harm_y``); zero for ``|m| > l``."""
    theta, phi = np.broadcast_arrays(np.asarray(theta, dtype=float), np.asarray(phi, dtype=float))
    if l < 0 or abs(m) > l:
        return np.zeros(theta.shape, dtype=complex)
    return sph_harm_y(l, m, theta, phi)


def dY_dtheta(l, m, theta, phi):
    """``d Y_lm / d theta`` from the recurrence ``m cot(theta) Y_lm + sqrt((l-m)(l+m+1)) e^{-i phi} Y_{l,m+1}``."""
    theta = np.asarray(theta, dtype=float)
    return (m / np.tan(theta) * Y(l, m, theta, phi)
            + np.sqrt((l - m) * (l + m + 1)) * np.exp(-1j * np.asarray(phi)) * Y(l, m + 1, theta, phi))


def X(l, m, theta, phi):
    """Vector spherical harmonic ``X_lm = L Y_lm / sqrt(l(l+1))``, Cartesian components ``(..., 3)``.

    Written with the Cartesian angular-momentum operators, ``L_z Y = m Y`` and
    ``L_+- Y_lm = sqrt((l -+ m)(l +- m + 1)) Y_{l,m+-1}``, so
    ``L_x = (L_+ + L_-)/2``, ``L_y = (L_+ - L_-)/(2i)``: regular at the poles,
    no derivatives.  :func:`X_spherical` is the same object from its
    ``(theta, phi)`` components; ``test_oracles`` checks they agree and that
    both agree with finite differences of ``Y``.
    """
    if l < 1 or abs(m) > l:
        shape = np.broadcast(np.asarray(theta), np.asarray(phi)).shape
        return np.zeros(shape + (3,), dtype=complex)
    cp = np.sqrt((l - m) * (l + m + 1))
    cm = np.sqrt((l + m) * (l - m + 1))
    yp, ym, y = Y(l, m + 1, theta, phi), Y(l, m - 1, theta, phi), Y(l, m, theta, phi)
    Lx = 0.5 * (cp * yp + cm * ym)
    Ly = (cp * yp - cm * ym) / 2j
    Lz = m * y
    return np.stack([Lx, Ly, Lz], -1) / np.sqrt(l * (l + 1))


def X_spherical(l, m, theta, phi, dtheta=None):
    """``X_lm`` from ``L Y = -(m / sin theta) Y theta_hat - i (dY/dtheta) phi_hat``, in Cartesian components.

    ``dtheta=None`` uses :func:`dY_dtheta` (recurrence); a float uses a central
    finite difference with that step.  Singular at the poles -- test use only.
    """
    theta = np.asarray(theta, dtype=float)
    if dtheta is None:
        dy = dY_dtheta(l, m, theta, phi)
    else:
        dy = (Y(l, m, theta + dtheta, phi) - Y(l, m, theta - dtheta, phi)) / (2 * dtheta)
    vth = -m * Y(l, m, theta, phi) / np.sin(theta)
    vph = -1j * dy
    return (vth[..., None] * theta_hat(theta, phi) + vph[..., None] * phi_hat(theta, phi)) / np.sqrt(l * (l + 1))


def _angular_table(lmax, theta, phi):
    """``{(l, m): (Y, X, rhat x X)}`` for every mode up to ``lmax`` on one set of angles."""
    rh = rhat(theta, phi)
    table = {}
    for l, m in modes(lmax):
        x = X(l, m, theta, phi)
        table[l, m] = (Y(l, m, theta, phi), x, np.cross(rh, x))
    return table


# --------------------------------------------------------------------------
# Vector spherical wave functions and fields
# --------------------------------------------------------------------------


def radial(l, x, kind):
    """``(z_l(x), z_l'(x))``: ``j_l`` (regular), ``h_l^(1)`` (outgoing), ``h_l^(2)`` (incoming)."""
    j, jp = spherical_jn(l, x), spherical_jn(l, x, derivative=True)
    if kind == "regular":
        return j + 0j, jp + 0j
    y, yp = spherical_yn(l, x), spherical_yn(l, x, derivative=True)
    if kind == "outgoing":
        return j + 1j * y, jp + 1j * yp
    if kind == "incoming":
        return j - 1j * y, jp - 1j * yp
    raise ValueError(f"kind={kind!r}")


def vswf(l, m, xyz, k, kind="regular"):
    """``(M_lm, N_lm)`` at Cartesian points ``xyz`` (``(npts, 3)``), each ``(npts, 3)``.

    ``M = z_l(kr) X_lm``,  ``N = ([x z_l]'/x) rhat x X_lm + i sqrt(l(l+1)) (z_l/x) Y_lm rhat``.
    """
    xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
    r, th, ph = to_spherical(xyz)
    x = k * r
    z, zp = radial(l, x, kind)
    rh = rhat(th, ph)
    xl = X(l, m, th, ph)
    M = z[:, None] * xl
    N = ((z / x + zp)[:, None] * np.cross(rh, xl)
         + (1j * np.sqrt(l * (l + 1)) * z / x * Y(l, m, th, ph))[:, None] * rh)
    return M, N


def efield(cM, cN, xyz, k, kind="regular"):
    """``E = sum_lm cM_lm M_lm + cN_lm N_lm`` of one kind of wave, at points ``(npts, 3)``."""
    cM = np.asarray(cM, dtype=complex).ravel()
    cN = np.asarray(cN, dtype=complex).ravel()
    lmax = int(round(np.sqrt(cM.size + 1) - 1))
    if size(lmax) != cM.size or cN.size != cM.size:
        raise ValueError("coefficient vectors must both have length lmax(lmax+2)")
    xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
    r, th, ph = to_spherical(xyz)
    x = k * r
    rh = rhat(th, ph)
    table = _angular_table(lmax, th, ph)
    E = np.zeros(xyz.shape, dtype=complex)
    for l in range(1, lmax + 1):
        z, zp = radial(l, x, kind)
        a_rx = (z / x + zp)[:, None]
        a_r = (1j * np.sqrt(l * (l + 1)) * z / x)[:, None]
        for m in range(-l, l + 1):
            y, xl, rxl = table[l, m]
            i = slot(l, m)
            if cM[i] != 0:
                E += cM[i] * z[:, None] * xl
            if cN[i] != 0:
                E += cN[i] * (a_rx * rxl + a_r * y[:, None] * rh)
    return E


def fields(a, p, xyz, k, n=1.0, Z0=1.0):
    """``(E, H)`` of a regular part ``a = (aM, aN)`` plus an outgoing part ``p = (pM, pN)``.

    ``H = -(i/Z) sum (c^M N + c^N M)`` with ``Z = Z0 / n`` -- Faraday's law
    ``curl E = i omega mu H`` for ``curl M = k N``, ``curl N = k M``.
    ``p=None`` means no outgoing part.  In the natural units ``eps0 = mu0 = 1``
    of the stress tests, ``Z0 = 1`` and the medium has ``eps = n^2``, ``mu = 1``.
    """
    Z = Z0 / n
    aM, aN = a
    E = efield(aM, aN, xyz, k, "regular")
    H = efield(aN, aM, xyz, k, "regular")
    if p is not None:
        pM, pN = p
        E = E + efield(pM, pN, xyz, k, "outgoing")
        H = H + efield(pN, pM, xyz, k, "outgoing")
    return E, -1j / Z * H


def pattern(cM, cN, theta, phi, incoming=False):
    """Far-field pattern of a set of waves, ``(..., 3)``.

    Outgoing (``h^(1)``): ``k r e^{-ikr} E -> -i A_c``,  ``A_c = sum (-i)^l [cM X + i cN rhat x X]``.
    Incoming (``h^(2)``): ``k r e^{+ikr} E -> i B_c``,   ``B_c = sum i^l [cM X - i cN rhat x X]``.
    (The constant phases ``-i``, ``i`` are left out of the patterns: the flux
    sums only see ``|A|^2``.)
    """
    cM = np.asarray(cM, dtype=complex).ravel()
    cN = np.asarray(cN, dtype=complex).ravel()
    lmax = int(round(np.sqrt(cM.size + 1) - 1))
    table = _angular_table(lmax, np.asarray(theta, dtype=float), np.asarray(phi, dtype=float))
    out = 0j
    for (l, m), (_, xl, rxl) in table.items():
        i = slot(l, m)
        if incoming:
            out = out + (1j) ** l * (cM[i] * xl - 1j * cN[i] * rxl)
        else:
            out = out + (-1j) ** l * (cM[i] * xl + 1j * cN[i] * rxl)
    return out


# --------------------------------------------------------------------------
# Maxwell stress tensor
# --------------------------------------------------------------------------


def stress_force_torque(field, r, eps=1.0, mu=1.0, n_theta=40, n_phi=80, center=(0.0, 0.0, 0.0)):
    """Time-averaged force and torque from the Maxwell stress tensor on a sphere.

    ``field(xyz) -> (E, H)`` at points ``(npts, 3)``.  With
    ``T = (1/2) Re[eps E E* + mu H H*] - (1/4)(eps |E|^2 + mu |H|^2) I``,

        F = oint T . n dA,       tau = oint (r n) x (T . n) dA

    over the sphere of radius ``r`` about ``center`` (torque about ``center``),
    by Gauss-Legendre x uniform-phi quadrature.  Exact for fields of bounded
    angular degree when ``n_theta, n_phi`` exceed it.
    """
    center = np.asarray(center, dtype=float)
    th, ph, w = sphere_quadrature(n_theta, n_phi)
    n = rhat(th, ph)
    E, H = field(center + r * n)
    En = np.sum(np.conj(E) * n, -1)
    Hn = np.sum(np.conj(H) * n, -1)
    energy = eps * np.sum(np.abs(E) ** 2, -1) + mu * np.sum(np.abs(H) ** 2, -1)
    Tn = 0.5 * np.real(eps * E * En[:, None] + mu * H * Hn[:, None]) - 0.25 * energy[:, None] * n
    F = r**2 * np.sum(w[:, None] * Tn, 0)
    tau = r**3 * np.sum(w[:, None] * np.cross(n, Tn), 0)
    return F, tau


# --------------------------------------------------------------------------
# Field -> coefficients
# --------------------------------------------------------------------------


def project(field, k, lmax, kind="regular", radii=None, n_theta=None, n_phi=None, center=(0.0, 0.0, 0.0)):
    """Coefficients ``(cM, cN)`` of a field of one kind about ``center``, by projection on spheres.

    On a sphere of radius ``r`` the three families ``X_lm``, ``rhat x X_lm`` and
    ``Y_lm rhat`` are orthonormal, so for ``E = sum cM M + cN N``

        <X_lm, E>          = cM z_l(x),
        <rhat x X_lm, E>   = cN [x z_l]'/x,
        <Y_lm rhat, E>     = cN i sqrt(l(l+1)) z_l / x.

    These are collected on several radii (``radii`` in units of ``1/k``) and
    solved by least squares per mode, so no single zero of ``j_l`` spoils a
    coefficient.  Defaults: ``kr = 1 ... lmax + 3`` for regular waves (``j_l`` is
    tiny below ``kr ~ l``); ``kr = lmax/2 + 1 ... lmax + 4`` for outgoing and
    incoming ones (closer in, the ``h_l`` near fields of high ``l`` swamp the
    low orders).  ``field(xyz) -> E`` at ``(npts, 3)``.
    """
    center = np.asarray(center, dtype=float)
    if radii is None and kind == "regular":
        radii = np.linspace(1.0, lmax + 3.0, max(4, lmax // 2 + 3))
    elif radii is None:
        radii = np.linspace(lmax / 2 + 1.0, lmax + 4.0, 4)
    radii = np.asarray(radii, dtype=float)
    if n_theta is None:
        n_theta = int(np.max(radii)) + lmax + 30
    if n_phi is None:
        n_phi = 2 * n_theta
    th, ph, w = sphere_quadrature(n_theta, n_phi)
    rh = rhat(th, ph)
    table = _angular_table(lmax, th, ph)
    fields_on = [np.asarray(field(center + (x / k) * rh), dtype=complex) for x in radii]

    cM = np.zeros(size(lmax), dtype=complex)
    cN = np.zeros(size(lmax), dtype=complex)
    for l in range(1, lmax + 1):
        zs = [radial(l, x, kind) for x in radii]
        rowM = np.array([z for z, _ in zs])
        rowN = np.concatenate([np.array([z / x + zp for (z, zp), x in zip(zs, radii)]),
                               np.array([1j * np.sqrt(l * (l + 1)) * z / x for (z, _), x in zip(zs, radii)])])
        for m in range(-l, l + 1):
            y, xl, rxl = table[l, m]
            projM = np.array([np.sum(w * np.sum(np.conj(xl) * E, -1)) for E in fields_on])
            projRX = np.array([np.sum(w * np.sum(np.conj(rxl) * E, -1)) for E in fields_on])
            projR = np.array([np.sum(w * np.conj(y) * np.sum(rh * E, -1)) for E in fields_on])
            i = slot(l, m)
            cM[i] = np.vdot(rowM, projM) / np.vdot(rowM, rowM)
            projN = np.concatenate([projRX, projR])
            cN[i] = np.vdot(rowN, projN) / np.vdot(rowN, rowN)
    return cM, cN


# --------------------------------------------------------------------------
# Analytic fields
# --------------------------------------------------------------------------


def plane_wave(xyz, k, khat, pol, amplitude=1.0):
    """``E = amplitude * pol * exp(i k khat . r)`` at points ``(npts, 3)``."""
    xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
    khat = np.asarray(khat, dtype=float)
    pol = np.asarray(pol, dtype=complex)
    return amplitude * pol[None, :] * np.exp(1j * k * xyz @ khat)[:, None]


def plane_wave_fields(xyz, k, khat, pol, amplitude=1.0, Z=1.0):
    """``(E, H)`` of the plane wave, ``H = khat x E / Z``."""
    E = plane_wave(xyz, k, khat, pol, amplitude)
    return E, np.cross(np.asarray(khat, dtype=float)[None, :], E) / Z


def plane_wave_coefficients(khat, pol, lmax):
    """Regular coefficients of ``pol exp(i k khat . r)``:

        aM_lm = 4 pi i^l  X*_lm(khat) . pol,     aN_lm = 4 pi i^(l+1) X*_lm(khat) . (khat x pol).

    The first is the projection of ``e^{ik.r} = 4 pi sum i^l j_l Y*_lm(khat) Y_lm``
    on ``X_lm``; the second follows from ``curl E = i k khat x E`` and
    ``curl M = k N``.  (For ``khat = z``, ``pol = x + i y`` this is Jackson 10.55.)
    Checked against :func:`project` in ``test_oracles``.
    """
    khat = np.asarray(khat, dtype=float)
    khat = khat / np.linalg.norm(khat)
    pol = np.asarray(pol, dtype=complex)
    _, th, ph = to_spherical(khat)
    h = np.cross(khat, pol)
    aM = np.zeros(size(lmax), dtype=complex)
    aN = np.zeros(size(lmax), dtype=complex)
    for l, m in modes(lmax):
        xs = np.conj(X(l, m, th, ph))
        aM[slot(l, m)] = 4 * np.pi * (1j) ** l * np.sum(xs * pol)
        aN[slot(l, m)] = 4 * np.pi * (1j) ** (l + 1) * np.sum(xs * h)
    return aM, aN


def standing_wave(xyz, k, pol=(1.0, 0.0, 0.0), z0=0.0):
    """``E = pol [e^{ik(z+z0)} + e^{-ik(z+z0)}]``: two counter-propagating plane waves along ``z``.

    The coordinates ``xyz`` are relative to a particle at height ``z0`` above the
    antinode of ``E`` at ``z = 0``.
    """
    xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
    pol = np.asarray(pol, dtype=complex)
    zz = xyz[:, 2] + z0
    return pol[None, :] * (np.exp(1j * k * zz) + np.exp(-1j * k * zz))[:, None]


def standing_wave_fields(xyz, k, pol=(1.0, 0.0, 0.0), z0=0.0, Z=1.0):
    """``(E, H)`` of :func:`standing_wave`; ``H = [zhat x pol] (e^{ikz} - e^{-ikz}) / Z``."""
    xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
    pol = np.asarray(pol, dtype=complex)
    zz = xyz[:, 2] + z0
    E = standing_wave(xyz, k, pol, z0)
    h = np.cross([0.0, 0.0, 1.0], pol)
    return E, h[None, :] * (np.exp(1j * k * zz) - np.exp(-1j * k * zz))[:, None] / Z


def dipole_fields(xyz, k, p=(0, 0, 0), m=(0, 0, 0), eps=EPS0, mu=MU0):
    """Exact ``(E, H)`` of a point electric dipole ``p`` and magnetic dipole ``m`` at the origin.

    Jackson (9.18) and (9.36), SI, in a medium ``(eps, mu)``: with ``n = rhat``,
    ``g = e^{ikr}/r``, ``omega = k / sqrt(eps mu)``,

        E_p = [k^2 (n x p) x n g + (3n(n.p) - p)(1/r^3 - ik/r^2) e^{ikr}] / (4 pi eps)
        H_p = (omega k / 4 pi) (n x p) g (1 - 1/(ikr))
        H_m = [k^2 (n x m) x n g + (3n(n.m) - m)(1/r^3 - ik/r^2) e^{ikr}] / (4 pi)
        E_m = -(mu omega k / 4 pi) (n x m) g (1 - 1/(ikr))

    (``omega k = c k^2`` and ``mu omega k = Z0 k^2`` in vacuum, as Jackson writes them).
    """
    xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
    p = np.asarray(p, dtype=complex)
    m = np.asarray(m, dtype=complex)
    omega = k / np.sqrt(eps * mu)
    r = np.linalg.norm(xyz, axis=-1)[:, None]
    n = xyz / r
    e = np.exp(1j * k * r)
    near = (1 / r**3 - 1j * k / r**2) * e
    tail = (e / r) * (1 - 1 / (1j * k * r))

    def dip(d):
        nd = np.sum(n * d[None, :], -1)[:, None]
        far = k**2 * (d[None, :] - n * nd) * e / r                  # (n x d) x n = d - n (n.d)
        return far + (3 * n * nd - d[None, :]) * near

    nxp = np.cross(n, p[None, :])
    nxm = np.cross(n, m[None, :])
    E = dip(p) / (4 * np.pi * eps) - mu * omega * k / (4 * np.pi) * nxm * tail
    H = omega * k / (4 * np.pi) * nxp * tail + dip(m) / (4 * np.pi)
    return E, H


def curl(field, x0, h=1e-5):
    """Curl of a vector field at points ``x0`` ``(npts, 3)`` by central differences, ``(npts, 3)``."""
    x0 = np.atleast_2d(np.asarray(x0, dtype=float))
    J = np.zeros(x0.shape + (3,), dtype=complex)      # J[p, i, j] = d F_i / d x_j
    for j in range(3):
        d = np.zeros(3)
        d[j] = h
        J[:, :, j] = (field(x0 + d) - field(x0 - d)) / (2 * h)
    return np.stack([J[:, 2, 1] - J[:, 1, 2], J[:, 0, 2] - J[:, 2, 0], J[:, 1, 0] - J[:, 0, 1]], -1)


def laplacian(field, x0, h=1e-3):
    """Laplacian of a vector field at points ``x0`` by the 7-point stencil."""
    x0 = np.atleast_2d(np.asarray(x0, dtype=float))
    out = -6 * field(x0)
    for j in range(3):
        d = np.zeros(3)
        d[j] = h
        out = out + field(x0 + d) + field(x0 - d)
    return out / h**2


# --------------------------------------------------------------------------
# Mie theory (Bohren & Huffman)
# --------------------------------------------------------------------------


def mie(m, x, nmax):
    """Bohren-Huffman ``a_n, b_n`` (``n = 1..nmax``) of a sphere, relative index ``m``, size ``x = k R``.

    BHMIE's algorithm: the logarithmic derivative ``D_n(mx)`` by downward
    recurrence ``D_{n-1} = n/(mx) - 1/(D_n + n/(mx))``, the Riccati-Bessel
    ``psi_n = x j_n(x)`` and ``xi_n = x h_n^(1)(x)``, and

        a_n = [(D_n/m + n/x) psi_n - psi_{n-1}] / [(D_n/m + n/x) xi_n - xi_{n-1}],
        b_n = [(m D_n + n/x) psi_n - psi_{n-1}] / [(m D_n + n/x) xi_n - xi_{n-1}].

    In the Condon-Shortley VSWF basis the T-matrix is ``p^M = -b_n a^M``, ``p^N = -a_n a^N``.
    """
    m = complex(m)
    x = float(x)
    mx = m * x
    nstart = int(max(nmax, abs(mx)) + 16)
    D = np.zeros(nstart + 1, dtype=complex)
    for n in range(nstart, 0, -1):
        D[n - 1] = n / mx - 1.0 / (D[n] + n / mx)
    orders = np.arange(0, nmax + 1)
    psi = x * spherical_jn(orders, x)
    xi = x * (spherical_jn(orders, x) + 1j * spherical_yn(orders, x))
    n = np.arange(1, nmax + 1)
    Dn = D[1:nmax + 1]
    ga = Dn / m + n / x
    gb = m * Dn + n / x
    a = (ga * psi[1:] - psi[:-1]) / (ga * xi[1:] - xi[:-1])
    b = (gb * psi[1:] - psi[:-1]) / (gb * xi[1:] - xi[:-1])
    return a, b


def mie_cross_sections(m, x, k, nmax):
    """``C_ext, C_sca, C_abs, C_pr`` (and ``g C_sca``) of a sphere, in units of ``1/k^2``'s length squared.

    Bohren & Huffman (4.61), (4.62), (4.74):
    ``C_ext = (2 pi/k^2) sum (2n+1) Re(a_n + b_n)``,
    ``C_sca = (2 pi/k^2) sum (2n+1)(|a_n|^2 + |b_n|^2)``,
    ``g C_sca = (4 pi/k^2) [sum n(n+2)/(n+1) Re(a_n a*_{n+1} + b_n b*_{n+1}) + sum (2n+1)/(n(n+1)) Re(a_n b*_n)]``,
    ``C_pr = C_ext - g C_sca``, ``C_abs = C_ext - C_sca``.  Truncated at ``nmax``
    (the ``n, n+1`` sum stops at ``n + 1 = nmax``).
    """
    a, b = mie(m, x, nmax)
    n = np.arange(1, nmax + 1)
    ext = 2 * np.pi / k**2 * np.sum((2 * n + 1) * np.real(a + b))
    sca = 2 * np.pi / k**2 * np.sum((2 * n + 1) * (np.abs(a) ** 2 + np.abs(b) ** 2))
    n1 = n[:-1]
    gsca = 4 * np.pi / k**2 * (np.sum(n1 * (n1 + 2) / (n1 + 1) * np.real(a[:-1] * np.conj(a[1:]) + b[:-1] * np.conj(b[1:])))
                               + np.sum((2 * n + 1) / (n * (n + 1)) * np.real(a * np.conj(b))))
    return dict(ext=ext, sca=sca, abs=ext - sca, pr=ext - gsca, gsca=gsca)


def mie_scatter(aM, aN, m, x):
    """Scattered coefficients ``pM = -b_l aM``, ``pN = -a_l aN`` of a sphere."""
    aM = np.asarray(aM, dtype=complex)
    aN = np.asarray(aN, dtype=complex)
    lmax = int(round(np.sqrt(aM.size + 1) - 1))
    a, b = mie(m, x, lmax)
    l = np.array([l for l, _ in modes(lmax)])
    return -b[l - 1] * aM, -a[l - 1] * aN


# --------------------------------------------------------------------------
# The ladder sums in 60-digit arithmetic
# --------------------------------------------------------------------------


def flux_mp(cM, cN, dps=60):
    """The six sums ``(fz, fxy, tz, txy, sz, sxy)`` of ``Q[c]`` as explicit loops over ``(l, m)`` in mpmath.

    An independent transcription of the Farsund-Felderhof ladder sums: a
    dictionary lookup ``c[l, m]`` (zero outside ``1 <= l <= lmax``, ``|m| <= l``)
    instead of shifted, padded arrays, and every ladder coefficient in ``dps``
    digits::

        K_lm   = sqrt(l(l+2)(l-m+1)(l+m+1) / ((2l+1)(2l+3))) / (l+1)
        L_l    = sqrt(l(l+2) / ((2l+1)(2l+3))) / (l+1)
        Lam_lm = sqrt((l-m)(l+m+1))

        fz  = 2 sum [ m/(l(l+1)) Re(M_lm N*_lm) + K_lm Im(M_{l+1,m} M*_lm + N_{l+1,m} N*_lm) ]
        fxy = sum [ Lam/(l(l+1)) (M*_{l,m+1} N_lm + N*_{l,m+1} M_lm)
                    - i L_l ( sqrt((l+m+1)(l+m+2)) (M_lm M*_{l+1,m+1} + N_lm N*_{l+1,m+1})
                            + sqrt((l-m+1)(l-m+2)) (M_{l+1,m-1} M*_lm + N_{l+1,m-1} N*_lm) ) ]
        tz  = sum m (|M_lm|^2 + |N_lm|^2)
        txy = sum Lam (M_lm M*_{l,m+1} + N_lm N*_{l,m+1})
        sz  = sum [ -m/(l(l+1)) (|M|^2 + |N|^2) - 2 K_lm Im(M_{l+1,m} N*_lm + N_{l+1,m} M*_lm) ]
        sxy = sum [ -i Lam/(l(l+1)) (M*_{l,m+1} M_lm + N*_{l,m+1} N_lm)
                    - L_l ( sqrt((l+m+1)(l+m+2)) (M_lm N*_{l+1,m+1} + N_lm M*_{l+1,m+1})
                          + sqrt((l-m+1)(l-m+2)) (M_{l+1,m-1} N*_lm + N_{l+1,m-1} M*_lm) ) ]

    Returns mpmath numbers (``mpf`` for the axial sums, ``mpc`` for ``x + i y``).
    """
    import mpmath as mp

    with mp.workdps(dps):
        cM = [mp.mpc(complex(v)) for v in np.asarray(cM, dtype=complex).ravel()]
        cN = [mp.mpc(complex(v)) for v in np.asarray(cN, dtype=complex).ravel()]
        return _flux_mp_sums(cM, cN)


def _flux_mp_sums(cM, cN):
    import mpmath as mp

    lmax = int(round(np.sqrt(len(cM) + 1) - 1))
    zero = mp.mpc(0)

    def M(l, m):
        return cM[slot(l, m)] if 1 <= l <= lmax and abs(m) <= l else zero

    def N(l, m):
        return cN[slot(l, m)] if 1 <= l <= lmax and abs(m) <= l else zero

    fz = fxy = tz = txy = sz = sxy = mp.mpf(0)
    for l, m in modes(lmax):
        K = mp.sqrt(mp.mpf(l * (l + 2) * (l - m + 1) * (l + m + 1)) / ((2 * l + 1) * (2 * l + 3))) / (l + 1)
        L = mp.sqrt(mp.mpf(l * (l + 2)) / ((2 * l + 1) * (2 * l + 3))) / (l + 1)
        lam = mp.sqrt(mp.mpf((l - m) * (l + m + 1)))
        up_p = mp.sqrt(mp.mpf((l + m + 1) * (l + m + 2)))
        up_m = mp.sqrt(mp.mpf((l - m + 1) * (l - m + 2)))
        inv = mp.mpf(1) / (l * (l + 1))
        cj = mp.conj
        Mlm, Nlm = M(l, m), N(l, m)
        Mp, Np = M(l, m + 1), N(l, m + 1)
        M1, N1 = M(l + 1, m), N(l + 1, m)
        M1p, N1p = M(l + 1, m + 1), N(l + 1, m + 1)
        M1m, N1m = M(l + 1, m - 1), N(l + 1, m - 1)
        norm2 = abs(Mlm) ** 2 + abs(Nlm) ** 2

        fz += 2 * (m * inv * mp.re(Mlm * cj(Nlm)) + K * mp.im(M1 * cj(Mlm) + N1 * cj(Nlm)))
        fxy += (inv * lam * (cj(Mp) * Nlm + cj(Np) * Mlm)
                - 1j * L * (up_p * (Mlm * cj(M1p) + Nlm * cj(N1p)) + up_m * (M1m * cj(Mlm) + N1m * cj(Nlm))))
        tz += m * norm2
        txy += lam * (Mlm * cj(Mp) + Nlm * cj(Np))
        sz += -m * inv * norm2 - 2 * K * mp.im(M1 * cj(Nlm) + N1 * cj(Mlm))
        sxy += (-1j * inv * lam * (cj(Mp) * Mlm + cj(Np) * Nlm)
                - L * (up_p * (Mlm * cj(N1p) + Nlm * cj(M1p)) + up_m * (M1m * cj(Nlm) + N1m * cj(Mlm))))
    return fz, fxy, tz, txy, sz, sxy


def force_torque_mp(aM, aN, pM, pN, dps=60):
    """``Q[a] - Q[a + 2p]`` in ``dps``-digit arithmetic, packed as force, torque and spin (float arrays).

    The input doubles are converted exactly and ``a + 2p`` is formed in mpmath,
    so the only rounding is at ``dps`` digits.  Packing: force ``(Re fxy, Im fxy, fz)``,
    torque ``(Re txy, Im txy, tz)``, spin ``(Re(i sxy), Im(i sxy), -sz)``.
    """
    import mpmath as mp

    with mp.workdps(dps):
        a = [[mp.mpc(complex(v)) for v in np.asarray(c, dtype=complex).ravel()] for c in (aM, aN)]
        p = [[mp.mpc(complex(v)) for v in np.asarray(c, dtype=complex).ravel()] for c in (pM, pN)]
        out = [[x + 2 * y for x, y in zip(ca, cp)] for ca, cp in zip(a, p)]
        qi = _flux_mp_sums(*a)
        qo = _flux_mp_sums(*out)
        fz, fxy, tz, txy, sz, sxy = (x - y for x, y in zip(qi, qo))
        sxy_i = 1j * sxy
        return dict(
            force=np.array([float(mp.re(fxy)), float(mp.im(fxy)), float(mp.re(fz))]),
            torque=np.array([float(mp.re(txy)), float(mp.im(txy)), float(mp.re(tz))]),
            spin=np.array([float(mp.re(sxy_i)), float(mp.im(sxy_i)), float(-mp.re(sz))]),
        )
