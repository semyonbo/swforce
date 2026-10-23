"""Adapters: another package's spherical-wave coefficients, in the layout the sums want.

:mod:`swforce.force` takes four arrays and returns a force.  Everything that
knows about a particular scattering package lives here, one module per package,
so adding a backend never touches the physics.  None is imported by
``import swforce``; import the one you use::

    from swforce.adapters import smuthi as sws   # needs smuthi
    from swforce.adapters import treams as swt   # needs treams
    from swforce.adapters import ott             # pure numpy

An adapter owes the core three things and nothing else:

1. **Reindexing** into ``ci = l(l+1)+m`` with magnetic (``M``) and electric
   (``N``) multipoles split -- :mod:`swforce.basis`;
2. **Rephasing** to the Condon-Shortley convention, measured against the
   package's own basis functions (``tests/test_adapter_*.py``);
3. **Refusing** input it cannot map (helicity basis, several expansion centres,
   lossy host) instead of returning a plausible wrong number.
"""

__all__ = ["smuthi", "treams", "ott"]
