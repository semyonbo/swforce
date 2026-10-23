"""Shared test configuration.

* Puts the repository root first on ``sys.path`` so the ``swforce`` under test is
  this checkout (BIC-Force ships a package of the same name), and ``tests/`` so
  ``import oracles`` works.
* Skips tests marked ``smuthi`` / ``treams`` when that package is not importable
  (the adapter modules also ``importorskip`` at import time).
* Registers a deterministic hypothesis profile when hypothesis is installed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
DATA = TESTS / "data"
for path in (str(TESTS), str(ROOT)):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

import swforce  # noqa: E402

assert Path(swforce.__file__).resolve().parent == ROOT / "swforce", (
    f"testing the wrong swforce: {swforce.__file__}")


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


HAVE = {name: _importable(name) for name in ("smuthi", "treams", "hypothesis", "mpmath", "matplotlib")}

try:
    from hypothesis import HealthCheck, settings

    settings.register_profile(
        "swforce", derandomize=True, deadline=None, max_examples=60, print_blob=True,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )
    settings.load_profile("swforce")
except ImportError:                                    # pragma: no cover - env without hypothesis
    pass


def pytest_collection_modifyitems(config, items):
    for item in items:
        for name in ("smuthi", "treams"):
            if item.get_closest_marker(name) is not None and not HAVE[name]:
                item.add_marker(pytest.mark.skip(reason=f"{name} is not importable in this environment"))


@pytest.fixture
def rng():
    """A fresh generator with a fixed seed for every test."""
    return np.random.default_rng(20260930)
