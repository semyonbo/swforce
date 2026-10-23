# swforce

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![tests](https://github.com/semyonbo/swforce/actions/workflows/tests.yml/badge.svg)](https://github.com/semyonbo/swforce/actions/workflows/tests.yml)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue)](LICENSE.md)

**Exact optical force, torque and spin from vector spherical wave (T-matrix) coefficients.**

swforce is a Python package that calculates the time-averaged optical (electromagnetic) force, torque and spin transfer acting on a nano-object from the Maxwell stress tensor, with the fields expanded in vector spherical wave functions (VSWFs). Its input is the set of multipole coefficients of the field around the object: the regular field that excites it and the outgoing field it scatters. These coefficients can be computed by hand or taken from a T-matrix code such as [SMUTHI](https://gitlab.com/AmosEgel/smuthi), [treams](https://github.com/tfp-photonics/treams) or MATLAB [OTT](https://github.com/ilent2/ott).

The stress tensor is integrated over a sphere enclosing the object analytically, in the multipole coefficients (Farsund & Felderhof, 1996). No mesh or numerical surface integration is involved: the result is exact for the given expansion, and the only approximation is truncating the series at order `lmax`.

## Features

- **Force, torque and spin** in two algebraically equal forms: the quadratic form $Q[a] - Q[a+2p]$ and the bilinear form $-2B(a,p) - 2B(p,a) - 4B(p,p)$. The bilinear form avoids subtracting large terms, so it stays accurate when the exciting-field coefficients are huge, e.g. for a particle a few nanometres above a metal.
- **Exact split of the force** by parts of the exciting field (incident wave, reflected wave, field of neighbouring particles, ...), with no cross terms between the parts.
- **Adapters** that convert other codes' conventions into the basis of the formula: SMUTHI (reads a finished `Simulation`), treams (single T-matrices and clusters, force on every particle) and MATLAB OTT-Fixed. Setting up and converging the scattering calculation is left to the code that computes the field.
- **SI units**: newtons and N·m for a given field amplitude or plane-wave intensity, and cross sections in nm².

The core depends only on numpy and scipy; each adapter needs its own package.

## Installation

```bash
pip install "swforce @ git+https://github.com/semyonbo/swforce"             # core
pip install "swforce[treams] @ git+https://github.com/semyonbo/swforce"     # + treams
pip install "swforce[smuthi] @ git+https://github.com/semyonbo/swforce"     # + SMUTHI
```

or from a clone, for development and tests:

```bash
git clone https://github.com/semyonbo/swforce.git
cd swforce
pip install -e ".[test]"
```

Python ≥ 3.10. SMUTHI builds most easily on Python 3.10.

## Quick start

### Your own coefficients

```python
import swforce as sf

# aM, aN: regular coefficients of the field that excites the particle (magnetic M / TE and electric N / TM);
# pM, pN: outgoing coefficients of the particle's scattered field.
# Index ci = l(l+1) + m, vectors of length lmax(lmax+2), expansions about the particle centre.
ft = sf.force_torque(aM, aN, pM, pN)                   # quadratic form
ft = sf.force_torque(aM, aN, pM, pN, form="bilinear")  # same value, no cancellation at large |a|
ft.force, ft.torque, ft.spin                           # dimensionless
si = sf.to_si(ft, wavelength=900.0, amplitude=1.0)     # N and N·m for |E0| = 1 V/m, vacuum wavelength in nm
```

### A particle above a substrate (SMUTHI)

A silicon sphere with $R = 130$ nm sits 2 nm above gold, at $\lambda = 900$ nm, lit at 25° with circular polarisation. SMUTHI solves the problem as usual; the adapter reads the exciting field $a = a_0 + Wb$ and the scattered field $p = b$ from the finished `Simulation` and converts the basis:

```python
import numpy as np
import smuthi.initial_field, smuthi.layers, smuthi.particles, smuthi.simulation
import swforce as sf
from swforce.adapters import smuthi as sws

gold = smuthi.layers.LayerSystem(thicknesses=[0, 0], refractive_indices=[0.174348 + 5.722739j, 1.0])
l_max, z0, k0 = 16, 132.0, 2 * np.pi / 900.0

def run(pol): # pol = 0: TE, 1: TM; the wave travels along +x, −z at 25°, 1 V/m at the origin
    sphere = smuthi.particles.Sphere(position=[0, 0, z0], refractive_index=3.614 + 0.00217j, radius=130.0,
                                     l_max=l_max, m_max=l_max)
    wave = smuthi.initial_field.PlaneWave(vacuum_wavelength=900.0, polar_angle=np.pi - np.deg2rad(25),
                                          azimuthal_angle=0.0, polarization=pol, amplitude=1.0,
                                          reference_point=[0, 0, 0])
    sim = smuthi.simulation.Simulation(layer_system=gold, particle_list=[sphere], initial_field=wave,
                                       log_to_terminal=False, neff_imag=5e-2, neff_resolution=1e-2,
                                       neff_max=(2 * l_max + 26) / (2 * k0 * z0))   # near field of the 2 nm gap
    sim.run()
    return np.array(sws.coefficients(sim)) # aM, aN, pM, pN in the swforce convention

aM, aN, pM, pN = 1j / np.sqrt(2) * run(0) - 1 / np.sqrt(2) * run(1)
force = sf.to_si(sf.force_torque(aM, aN, pM, pN, form="bilinear"), 900.0).force
print(force) # [Fx, Fy, Fz] in newtons
```

What the adapter does **not** do, and the person computing the field must check:
- the extent of the Sommerfeld contour: for a gap of a few nm it must reach $n_{\rm eff}\sim l_{\max}$, while SMUTHI's default stops near $n_{\rm eff}\approx 2$;
- convergence in `l_max`;
- the accuracy of the SMUTHI linear solve.

### treams: a single particle and a cluster

```python
import numpy as np, treams
import swforce as sf
from swforce.adapters import treams as swt

k0 = 2 * np.pi / 700 # 1/nm
vacuum = treams.Material()
beam = treams.plane_wave([0, 0, k0], [1, 0, 0], k0=k0, material=vacuum, poltype="parity")

tm = treams.TMatrix.sphere(12, k0, 150.0, [treams.Material(16), vacuum], poltype="parity")
a = beam.expand(tm.basis)
ft = swt.force_torque(a, tm @ a) # single sphere; C_pr = f_z / (4 k^2)
print(sf.cross_section(ft.force[2], 700.0), "nm^2")

small = treams.TMatrix.sphere(6, k0, 60.0, [treams.Material(16), vacuum], poltype="parity")
solved = treams.TMatrix.cluster([small] * 3, [[0, 0, 0], [250, 0, 0], [125, 200, 0]]).interaction.solve()
res = swt.cluster_force_torque(solved, beam) # res.force[j]: force on particle j
```

### MATLAB OTT

```python
from swforce.adapters import ott
ft = ott.force_torque(a, b, p, q)     # OTT-Fixed beam coefficients as they are
```

[OTT-Fixed](https://github.com/semyonbo/ott/tree/fix/condon-shortley-and-smarties-tmatrix) restores the Condon–Shortley phase in OTT's spherical harmonics, so its coefficients are already in the swforce convention. Upstream OTT lacks the $(-1)^m$ phase: multiply its coefficients by $(-1)^m$ first, otherwise $F_x, F_y, T_x, T_y$ change sign.

## Splitting the force

At a fixed scattered field the force is linear in the exciting field, so a split of the exciting field $a = \sum_i a_i$ splits the force exactly:

```python
parts = sf.force_torque_split({"incident": (a0M, a0N), "reflected": (arM, arN)}, pM, pN)
parts["incident"].force + parts["reflected"].force + parts["recoil"].force   # == parts["total"].force
```

`recoil` is the recoil of the scattered field, as if it left into empty space.

## Units

The core is dimensionless; the coefficients carry the field amplitude $|E|$ [V/m]. In a lossless non-magnetic host of refractive index $n$, with $k = nk_0$ the wavenumber in the host,

$$F\ [\mathrm{N}] = \frac{\varepsilon_0 n^2|E|^2}{8k^2} f,\qquad T,\ S\ [\mathrm{N\cdot m}] = \frac{\varepsilon_0 n^2|E|^2}{8k^3} (t,\ s),\qquad C = \frac{f}{4k^2}.$$

For a plane wave of intensity $I = n\varepsilon_0c|E|^2/2$ this is $F = (n/c)\, I\, C$: on the beam axis $f_z$ gives the radiation-pressure cross section $C_{\rm pr}$, and for circular polarisation $t_z$ gives $\sigma C_{\rm abs}$.

- `sf.to_si(ft, wavelength, amplitude=... or intensity=..., index_medium=1)` returns a `ForceTorque` in N, N·m, N·m;
- `sf.cross_section(f, wavelength, index_medium=1)` returns nm²;
- `sf.units.si_scale(...)` gives the factors for raw vectors (e.g. cluster forces).

Wavelengths are vacuum wavelengths in nm. Constants: $c$ exact (SI 2019), $\varepsilon_0$ from CODATA 2022.

## Conventions

A wrong convention gives a plausible but wrong answer, so they are fixed here:

- time dependence $e^{-i\omega t}$; outgoing waves are $h_l^{(1)}$;
- spherical harmonics **with** the Condon–Shortley phase (as in treams and scipy);
- combined index $c_i = l(l+1) + m$; vectors have length $l_{\max}(l_{\max}+2)$;
- `M` is the magnetic (TE) multipole, `N` the electric (TM) one;
- the **regular** part `a` is everything that arrives at the particle (incident beam plus whatever comes back from a substrate or neighbours); the **outgoing** part `p` is the particle's own scattered field; both are expanded about the particle centre.

Other codes' conventions are converted in the adapters, never by flags in the core: SMUTHI by a per-mode factor $-i\sqrt\pi\, s(m)$, upstream OTT by $(-1)^m$, OTT-Fixed by nothing.



## Structure

```
swforce/
  force.py          quadratic and bilinear forms, force_torque, interference, force_torque_split
  basis.py          combined index c_i = l(l+1)+m
  units.py          dimensionless -> SI, cross sections
  adapters/
    smuthi.py       SMUTHI conventions, reading a finished Simulation
    treams.py       treams T-matrices and clusters
    ott.py          MATLAB OTT-Fixed coefficients
tests/              tests and independent references (tests/oracles.py)
```

## Citing

If you use swforce in published work, please cite the software and the article it was developed for:

- **Software:** Semyon Borodulin, *swforce: exact optical force and torque from vector spherical wave coefficients*, Zenodo (2026). DOI: [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)
- **Article:** Semyon Borodulin, Natalia Kostina, Mihail Petrov, *Directional Scattering-Induced Optical Forces on a Mie Particle near a Metal Interface*, arXiv:2604.19428 (2026). DOI: [10.48550/arXiv.2604.19428](https://doi.org/10.48550/arXiv.2604.19428)

```bibtex
@software{borodulin_swforce,
  author    = {Borodulin, Semyon},
  title     = {swforce: exact optical force and torque from vector spherical wave coefficients},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/semyonbo/swforce}
}

@article{borodulin2026directional,
  author        = {Borodulin, Semyon and Kostina, Natalia and Petrov, Mihail},
  title         = {Directional Scattering-Induced Optical Forces on a {Mie} Particle near a Metal Interface},
  year          = {2026},
  eprint        = {2604.19428},
  archivePrefix = {arXiv},
  primaryClass  = {physics.optics},
  doi           = {10.48550/arXiv.2604.19428}
}
```

GitHub's "Cite this repository" button reads [CITATION.cff](CITATION.cff).

## References

The method and the codes it connects to:

1. Ø. Farsund, B. U. Felderhof, *Force, torque, and absorbed energy for a body of arbitrary shape and constitution in an electromagnetic radiation field*, Physica A **227**, 108 (1996). [doi:10.1016/0378-4371(96)00009-X](https://doi.org/10.1016/0378-4371(96)00009-X) — the force and torque sums.
2. C. F. Bohren, D. R. Huffman, *Absorption and Scattering of Light by Small Particles* (Wiley, 1983). [doi:10.1002/9783527618156](https://doi.org/10.1002/9783527618156) — Mie theory, used in the tests.
3. A. Egel et al., *SMUTHI: A python package for the simulation of light scattering by multiple particles near or between planar interfaces*, J. Quant. Spectrosc. Radiat. Transf. **273**, 107846 (2021). [doi:10.1016/j.jqsrt.2021.107846](https://doi.org/10.1016/j.jqsrt.2021.107846)
4. D. Beutel, I. Fernandez-Corbaton, C. Rockstuhl, *treams – a T-matrix-based scattering code for nanophotonics*, Comput. Phys. Commun. **297**, 109076 (2024). [doi:10.1016/j.cpc.2023.109076](https://doi.org/10.1016/j.cpc.2023.109076)
5. T. A. Nieminen et al., *Optical tweezers computational toolbox*, J. Opt. A: Pure Appl. Opt. **9**, S196 (2007). [doi:10.1088/1464-4258/9/8/S12](https://doi.org/10.1088/1464-4258/9/8/S12)

## License

[PolyForm Noncommercial License 1.0.0](LICENSE.md): free for research, teaching, personal and other noncommercial use, including by universities and public research organisations. Commercial use requires a separate licence; contact the author, Semyon Borodulin (semyon.borodulin@metalab.ifmo.ru).
