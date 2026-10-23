"""The SMUTHI adapter: basis convention and the force on a particle of a finished SMUTHI simulation.

Skipped without smuthi.  The adapter only translates, so every test runs a
plain ``smuthi.simulation.Simulation`` (SMUTHI's own solver) and checks what
the adapter reads from it.  The article system throughout: Si sphere 2 nm above
gold, lambda = 900 nm, plane waves at 25 deg travelling to ``+x, -z``.

What is checked against what:

* basis bridge -- SMUTHI's own vector wave functions vs ``oracles.vswf``; a SMUTHI
  expansion evaluated by SMUTHI vs its conversion evaluated by the oracle;
* ``a0`` -- SMUTHI's initial-field expansion reproduces SMUTHI's plane-wave field;
* ``coefficients`` -- ``a = a0 + W b`` with SMUTHI's explicit coupling block;
  ``a - a0`` reproduces SMUTHI's layer-mediated field of the scattered wave;
  several particles: the block of the right particle (mirror symmetry);
* the force -- free space against Mie; above gold and inside a film against the
  Maxwell stress tensor of the truncated field;
* rotated particles, ``m_max < l_max``, refusals, and regression values.
"""

from __future__ import annotations

import contextlib
import io
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("smuthi")

import oracles as O  # noqa: E402
import swforce as sf  # noqa: E402
from swforce.adapters import smuthi as sws  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import smuthi.fields as sfields
    import smuthi.fields.expansions as fldex
    import smuthi.fields.transformations as trf
    import smuthi.fields.vector_wave_functions as svwf
    import smuthi.initial_field as sinit
    import smuthi.layers as slayers
    import smuthi.linearsystem.particlecoupling.layer_mediated_coupling as lmc
    import smuthi.particles as sparticles
    import smuthi.simulation as ssim

pytestmark = [pytest.mark.smuthi, pytest.mark.filterwarnings("ignore::RuntimeWarning")]

LAMBDA = 900.0
GAP = 2.0
THETA = np.deg2rad(25.0)
R = 130.0
N_SI = 3.614 + 0.0021701j                           # the article's nk tables at 900 nm
N_AU = 0.17434782608695654 + 5.722739130434783j
W_TE, W_TM = 1j / np.sqrt(2), -1 / np.sqrt(2)       # circular polarisation of the article
K = 2 * np.pi / LAMBDA
KHAT = np.array([np.sin(THETA), 0.0, -np.cos(THETA)])


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def layers(substrate=N_AU):
    return slayers.LayerSystem(thicknesses=[0, 0], refractive_indices=[substrate, 1.0])


def plane_wave(pol, polar_angle=np.pi - THETA):
    return sinit.PlaneWave(vacuum_wavelength=LAMBDA, polar_angle=polar_angle, azimuthal_angle=0.0,
                           polarization=pol, amplitude=1.0, reference_point=[0, 0, 0])


def sphere(l_max, radius=R, z=None, x=0.0, index=N_SI):
    z = radius + GAP if z is None else z
    return sparticles.Sphere(position=[x, 0, z], refractive_index=index, radius=radius, l_max=l_max, m_max=l_max)


def simulate(particles, field, layer_system=None, **contour):
    """A plain SMUTHI simulation; the contour reaches the near field of the gap unless given."""
    contour = contour or dict(neff_imag=5e-2, neff_resolution=1e-2, neff_max=20.0)
    sim = ssim.Simulation(layer_system=layers() if layer_system is None else layer_system, particle_list=particles,
                          initial_field=field, solver_type="LU", log_to_terminal=False, **contour)
    quiet(sim.run)
    return sim


def circular(make_particle, layer_system=None, polar_angle=np.pi - THETA):
    """Coefficients ``(aM, aN, pM, pN)`` for the article's circular polarisation: TE and TM runs, superposed."""
    runs = [np.array(sws.coefficients(simulate([make_particle()], plane_wave(pol, polar_angle), layer_system)))
            for pol in (0, 1)]
    return tuple(W_TE * runs[0] + W_TM * runs[1])


def s_of(m):
    return (-1.0) ** m if m > 0 else 1.0


def near_points(rng, n, rmin, rmax):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1)[:, None] * rng.uniform(rmin, rmax, (n, 1))


@pytest.fixture(scope="module")
def gold4():
    """The article case at ``l_max = 4``: ``(aM, aN, pM, pN)`` in the swforce convention."""
    return circular(lambda: sphere(4))


# --------------------------------------------------------------------------
# Coefficient convention
# --------------------------------------------------------------------------


def test_smuthi_basis_is_minus_i_sqrt_pi_s_m_times_the_treams_basis(rng):
    """``Psi_SMUTHI(tau, l, m) = -i sqrt(pi) s(m) (M or N)``, ``tau = 0 -> M``, regular (nu=1) and outgoing (nu=3)."""
    pts = near_points(rng, 7, 40.0, 400.0)
    for nu, kind in ((1, "regular"), (3, "outgoing")):
        for l, m in O.modes(6):
            M, N = O.vswf(l, m, pts, K, kind)
            for tau, ref in ((0, M), (1, N)):
                got = np.stack(svwf.spherical_vector_wave_function(*pts.T, K, nu, tau, l, m), 1)
                expected = -1j * np.sqrt(np.pi) * s_of(m) * ref
                assert np.abs(got - expected).max() < 1e-12 * np.abs(expected).max()


@pytest.mark.parametrize("l_max,m_max", [(1, 1), (4, 4), (5, 2), (6, 0)])
def test_to_and_from_swforce_are_inverse(l_max, m_max, rng):
    n = sfields.blocksize(l_max, m_max)
    c = rng.normal(size=n) + 1j * rng.normal(size=n)
    cM, cN = sws.to_swforce(c, l_max, m_max)
    assert np.allclose(sws.from_swforce(cM, cN, m_max), c, rtol=1e-15, atol=0)
    l, m = sf.mode_indices(l_max)
    assert np.all(cM[np.abs(m) > m_max] == 0) and np.all(cN[np.abs(m) > m_max] == 0)
    for tau, out in ((0, cM), (1, cN)):
        for ll, mm in zip(l, m):
            if abs(mm) <= m_max:
                i = sfields.multi_to_single_index(tau, ll, mm, l_max, m_max)
                assert out[sf.basis.ci_index(ll, mm)] == -1j * np.sqrt(np.pi) * s_of(mm) * c[i]


def test_converted_coefficients_reproduce_smuthi_fields(rng):
    """The decisive bridge test: a SMUTHI expansion evaluated by SMUTHI = its conversion evaluated by the oracle."""
    l_max = 5
    n = sfields.blocksize(l_max, l_max)
    pts = near_points(rng, 6, 60.0, 300.0)
    for kind in ("regular", "outgoing"):
        swe = fldex.SphericalWaveExpansion(k=K, l_max=l_max, m_max=l_max, kind=kind, reference_point=[0, 0, 0])
        swe.coefficients = rng.normal(size=n) + 1j * rng.normal(size=n)
        E_smuthi = np.array(swe.electric_field(*pts.T)).T
        E_oracle = O.efield(*sws.to_swforce(swe.coefficients, l_max), pts, K, kind)
        assert np.abs(E_smuthi - E_oracle).max() < 1e-12 * np.abs(E_oracle).max()


def test_conversion_errors():
    with pytest.raises(ValueError, match="do not match"):
        sws.to_swforce(np.zeros(7), 2)
    with pytest.raises(ValueError, match="equal length"):
        sws.from_swforce(np.zeros(8), np.zeros(3))
    with pytest.raises(ValueError, match="equal length"):
        sws.from_swforce(np.zeros(5), np.zeros(5))


# --------------------------------------------------------------------------
# What the adapter reads from a simulation: a0, a = a0 + W b, p = b
# --------------------------------------------------------------------------


def test_initial_field_coefficients_reproduce_smuthi_plane_wave(rng):
    """``a0`` (incident + reflected by the gold) through ``oracles.vswf`` = SMUTHI's plane-wave field,
    evaluated in double precision from its own plane-wave expansion, at points around the particle."""
    fp = sphere(12)
    pts = near_points(rng, 8, 5.0, 120.0)
    xyz = pts + np.array(fp.position)
    E_smuthi, a0 = 0, 0
    for w, pol in ((W_TE, 0), (W_TM, 1)):
        up, down = plane_wave(pol).plane_wave_expansion(layers(), 1)
        E_smuthi = E_smuthi + w * (np.array(up.electric_field(*xyz.T, cpu_precision="double precision")).T
                                   + np.array(down.electric_field(*xyz.T, cpu_precision="double precision")).T)
        a0 = a0 + w * quiet(lambda: plane_wave(pol).spherical_wave_expansion(fp, layers()).coefficients)
    E_oracle = O.efield(*sws.to_swforce(a0, 12), pts, K)
    assert np.abs(E_oracle - E_smuthi).max() < 1e-12 * np.abs(E_smuthi).max()


def test_coefficients_are_a0_plus_W_b_and_b():
    """One particle: ``a = a0 + W b`` with SMUTHI's explicit layer-mediated block on the simulation's contour,
    ``p = b``; ``force_torque_from_simulation`` is the bilinear force of these coefficients."""
    sim = simulate([sphere(3)], plane_wave(0))
    part = sim.particle_list[0]
    W = quiet(lmc.layer_mediated_coupling_block, LAMBDA, part, part, layers())       # SMUTHI's default contour
    b = part.scattered_field.coefficients
    aM, aN, pM, pN = sws.coefficients(sim)
    ref_a = sws.to_swforce(part.initial_field.coefficients + W @ b, 3)
    assert np.abs(np.r_[aM, aN] - np.r_[ref_a]).max() < 1e-13 * np.abs(np.r_[ref_a]).max()
    assert np.array_equal(np.r_[pM, pN], np.r_[sws.to_swforce(b, 3)])
    ft = sws.force_torque_from_simulation(sim)
    ref = sf.force_torque(aM, aN, pM, pN, form="bilinear")
    for x, y in zip(ft, ref):
        assert np.array_equal(x, y)


def test_reflected_field_is_the_regular_expansion_a_minus_a0(rng):
    """The meaning of ``a_r = a - a0 = W b``: in the source-free shell ``R < |r - r_S| < d`` between the particle and
    the gold, ``sum a_r,n Psi^(1)_n`` is the particle's own field sent back by the gold -- SMUTHI's layer-mediated
    field of the scattered wave (``swe_to_pwe``, ``layer_system_mediated=True``, same contour, closed azimuthal grid).
    A 50 nm sphere 100 nm above the gold leaves a shell to test in; order 16, since the image field re-expanded about
    the particle converges like ``(r / 2 z0)^l`` (``r <= 80`` nm, ``2 z0 = 300`` nm)."""
    radius, z0, L = 50.0, 150.0, 16
    kp = sfields.reasonable_Sommerfeld_kpar_contour(vacuum_wavelength=LAMBDA, layer_refractive_indices=[N_AU, 1.0],
                                                    neff_imag=5e-2, neff_max=20.0, neff_resolution=1e-2)
    sim = simulate([sphere(L, radius=radius, z=z0)], plane_wave(1), k_parallel=kp)
    part = sim.particle_list[0]
    swe = fldex.SphericalWaveExpansion(k=K, l_max=L, m_max=L, kind="outgoing", reference_point=part.position,
                                       lower_z=0, upper_z=np.inf)
    swe.coefficients = part.scattered_field.coefficients
    alpha = np.linspace(0, 2 * np.pi, 65)                       # SMUTHI integrates alpha on the closed grid
    up, down = quiet(trf.swe_to_pwe_conversion, swe=swe, k_parallel=kp, azimuthal_angles=alpha,
                     layer_system=layers(), layer_number=1, layer_system_mediated=True)
    assert np.abs(down.coefficients).max() == 0                  # nothing comes back downwards in the top layer
    up.lower_z = -np.inf                                          # SMUTHI marks it valid only above the centre
    pts = near_points(rng, 8, radius + 5.0, 80.0)                 # R < r < d (d = z0 = 150 nm)
    E_smuthi = np.array(up.electric_field(*(pts + part.position).T, cpu_precision="double precision")).T
    aM, aN, _, _ = sws.coefficients(sim)
    a0M, a0N = sws.to_swforce(part.initial_field.coefficients, L)
    E_oracle = O.efield(aM - a0M, aN - a0N, pts, K)
    assert np.abs(E_smuthi - E_oracle).max() < 1e-7 * np.abs(E_oracle).max()


def test_two_mirror_image_particles_feel_mirror_image_forces():
    """Two spheres at ``x = -+150`` nm above gold, normal incidence, TE (``E || y``): symmetric under
    ``x -> -x``, so ``F_1 = s F_2`` and ``T_1 = -s T_2``, ``s = diag(-1, 1, 1)`` -- the block indexing
    of ``coefficients``."""
    parts = [sphere(3, radius=100.0, z=110.0, x=s * 150.0) for s in (-1, 1)]
    sim = simulate(parts, plane_wave(0, polar_angle=np.pi), neff_max=10, neff_resolution=1e-2, neff_imag=5e-2)
    f0, f1 = (sws.force_torque_from_simulation(sim, j) for j in (0, 1))
    s = np.array([-1.0, 1.0, 1.0])
    scale = np.linalg.norm(f0.force) + K * np.linalg.norm(f0.torque)
    assert np.abs(f0.force - s * f1.force).max() < 1e-10 * scale
    assert np.abs(f0.torque + s * f1.torque).max() * K < 1e-10 * scale
    assert abs(f0.force[0]) > 1e-3 * scale                    # the premise: they do push each other sideways


# --------------------------------------------------------------------------
# Force
# --------------------------------------------------------------------------


@pytest.mark.parametrize("radius", [100.0, 130.0, 160.0])
def test_free_space_force_is_mie_radiation_pressure(radius):
    """All refractive indices equal, circular polarisation: ``F.khat / (4k^2) = C_pr``, ``F || khat``, and the
    torque along ``khat`` is ``C_abs``.  The field stops at the particle's order 10, so the ``(10, 11)`` pair of
    the sums is missing -- of the size of ``p_10``, below 1e-12 here."""
    L = 10
    coeffs = circular(lambda: sphere(L, radius=radius), layers(substrate=1.0))
    ft = sf.force_torque(*coeffs, form="bilinear")
    xs = O.mie_cross_sections(N_SI, K * radius, K, L)
    assert ft.force @ KHAT / (4 * K**2) == pytest.approx(xs["pr"], rel=1e-10)
    assert np.linalg.norm(ft.force - (ft.force @ KHAT) * KHAT) < 1e-10 * np.linalg.norm(ft.force)
    tz = ft.torque @ KHAT / (4 * K**2)
    assert tz == pytest.approx(xs["abs"] * np.sign(tz), rel=1e-8)


@pytest.mark.parametrize("r", [20.0, 60.0, 131.0, 1000.0, 1e4])
def test_force_above_gold_matches_the_maxwell_stress_tensor(gold4, r):
    """The formula and the basis bridge against a direct stress integral of the truncated field (``a``, ``p``
    converted to the swforce convention, evaluated by the oracle) on a sphere of radius ``r`` (nm) about the
    particle.  An identity for *any* truncated ``(a, p)``: it checks the formula and the conversion, not
    SMUTHI's ``a`` and ``p`` -- their meaning is pinned by the ``a0`` and reflected-field tests."""
    aM, aN, pM, pN = gold4
    ft = sf.force_torque(aM, aN, pM, pN, form="bilinear")
    F, T = O.stress_force_torque(lambda x: O.fields((aM, aN), (pM, pN), x, K), r, n_theta=10, n_phi=20)
    kr = K * r
    tol = 1e-12 * max(1.0, kr * 100, 1e3 / kr**4)              # far out: r x T.n; close in: h_4(kr) near fields
    scale = np.linalg.norm(F) + K * np.linalg.norm(T)
    assert np.abs(ft.force / (8 * K**2) - F).max() < tol * scale
    assert np.abs(ft.torque / (8 * K**3) - T).max() * K < tol * scale


def test_particle_in_a_finite_layer():
    """Sphere inside a 400 nm film (n = 1.5) on gold under air: the force of the coefficients read from the
    simulation passes the stress-tensor check in the medium (``eps = n^2``, ``k = n k0``)."""
    ls = slayers.LayerSystem(thicknesses=[0, 400, 0], refractive_indices=[N_AU, 1.5, 1.0])
    sim = simulate([sphere(3, radius=100.0, z=200.0)], plane_wave(1), ls)
    n, k = 1.5, 1.5 * K
    aM, aN, pM, pN = sws.coefficients(sim)
    ft = sws.force_torque_from_simulation(sim)
    for r in (50.0, 101.0, 150.0):
        F, T = O.stress_force_torque(lambda x: O.fields((aM, aN), (pM, pN), x, k, n), r, eps=n**2,
                                     n_theta=10, n_phi=20)
        scale = np.linalg.norm(F) + k * np.linalg.norm(T)
        assert np.abs(n**2 * ft.force / (8 * k**2) - F).max() < 1e-11 * scale
        assert np.abs(n**2 * ft.torque / (8 * k**3) - T).max() * k < 1e-11 * scale


def test_rotated_sphere_gives_the_same_force():
    """A sphere is rotation invariant: SMUTHI rotates its T-matrix by the Euler angles (a full, non-diagonal
    matrix up to rounding) and the force read through the adapter does not change."""
    plain = sws.force_torque_from_simulation(simulate([sphere(4)], plane_wave(1)))
    turned_particle = sphere(4)
    turned_particle.euler_angles = [0.3, 0.7, 1.1]
    turned = sws.force_torque_from_simulation(simulate([turned_particle], plane_wave(1)))
    assert np.abs(turned.force - plain.force).max() < 1e-10 * np.linalg.norm(plain.force)
    assert np.abs(turned.torque - plain.torque).max() < 1e-10 * np.linalg.norm(plain.torque)


def test_m_max_below_l_max():
    """``m_max < l_max``: the missing modes are zero in the swforce vectors.  At normal incidence only
    ``|m| = 1`` is excited, so truncating ``m`` at 1 changes nothing physical."""
    part = sparticles.Sphere(position=[0, 0, R + GAP], refractive_index=N_SI, radius=R, l_max=3, m_max=1)
    sim = simulate([part], plane_wave(1, polar_angle=np.pi))
    aM, aN, pM, pN = sws.coefficients(sim)
    _, m = sf.mode_indices(3)
    for v in (aM, aN, pM, pN):
        assert np.all(v[np.abs(m) > 1] == 0)
    full = sws.force_torque_from_simulation(simulate([sphere(3)], plane_wave(1, polar_angle=np.pi)))
    f = sws.force_torque_from_simulation(sim).force
    assert np.abs(f - full.force).max() < 1e-10 * np.linalg.norm(full.force)


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_lossy_host_is_refused():
    lossy = slayers.LayerSystem(thicknesses=[0, 0], refractive_indices=[N_AU, 1.0 + 0.01j])
    fake = SimpleNamespace(particle_list=[sphere(2)], layer_system=lossy)
    for fn in (sws.coefficients, sws.force_torque_from_simulation):
        with pytest.raises(ValueError, match="lossless host"):
            fn(fake)


def test_particle_crossing_an_interface_is_refused():
    """The expansion about the centre needs a source-free shell ``R < r < d``: touching or crossing is refused."""
    for z in (R, R - 10.0):
        with pytest.raises(ValueError, match="circumscribing sphere"):
            sws.coefficients(SimpleNamespace(particle_list=[sphere(2, z=z)], layer_system=layers()))


# --------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------


def test_regression_article_system_l_max_4(gold4):
    """Dimensionless force at ``l_max = 4`` on the contour of ``simulate`` (``neff_max = 20``).  A change here
    means the adapter (or SMUTHI) changed what it returns."""
    ft = sf.force_torque(*gold4, form="bilinear")
    assert ft.force == pytest.approx(REGRESSION_L4, rel=1e-9)


REGRESSION_L4 = [26.904650981717605, 5.668082836570284, 162.90829754071325]
