"""End to end: every script and notebook in ``examples/`` and every Python block of ``README.md`` runs.

Each runs in a fresh interpreter with this checkout first on ``PYTHONPATH``.  Scripts
check themselves (``assert`` + a final ``OK``); a notebook runs as its code cells
joined in order, and its own ``assert`` lines are the check.  Whatever needs smuthi
or treams is skipped where that package is not importable.  The README blocks use
placeholder coefficients where they name user data (``aM``, ``p``, ...); the SMUTHI
block is additionally compared with a separate l_max = 16 SMUTHI force tutorial.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import HAVE

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "examples").glob("*.py"))
NOTEBOOKS = sorted((ROOT / "examples").glob("*.ipynb"))


def run(args, timeout=600):
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]))
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=timeout, env=env, cwd=ROOT)


def requirements(code):
    return [name for name in ("smuthi", "treams", "matplotlib") if re.search(rf"\b{name}\b", code.replace("swforce.adapters", ""))]


def skip_unless(requirements):
    missing = [r for r in requirements if not HAVE.get(r, False)]
    if missing:
        pytest.skip(f"needs {', '.join(missing)}")


def notebook_code(path):
    """The code cells of a notebook, in order, without IPython magics."""
    cells = json.loads(path.read_text())["cells"]
    lines = [line for c in cells if c["cell_type"] == "code"
             for line in "".join(c["source"]).splitlines() if not line.lstrip().startswith(("%", "!"))]
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_example_runs(path):
    skip_unless(requirements(path.read_text()))
    out = run([str(path)])
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.strip().splitlines()[-1] == "OK", out.stdout[-2000:]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs(path):
    code = notebook_code(path)
    skip_unless(requirements(code))
    out = run(["-c", code])
    assert out.returncode == 0, out.stderr[-3000:]


# --------------------------------------------------------------------------
# README
# --------------------------------------------------------------------------

README = (ROOT / "README.md").read_text() if (ROOT / "README.md").exists() else ""
BLOCKS = re.findall(r"```python\n(.*?)```", README, flags=re.S)

PRELUDE = """
import numpy as np
import swforce as sf                                                  # as in the README's first block
_rng = np.random.default_rng(0)
_c = lambda: _rng.normal(size=15) + 1j * _rng.normal(size=15)          # lmax = 3 placeholders
aM, aN, pM, pN, a, b, p, q, a0M, a0N, arM, arN = (_c() for _ in range(12))
"""

SMUTHI_CHECK = """
_ref = np.array([5.9599e-25, 1.2054e-25, 3.2790e-24])   # the tutorial's l_max = 16 force (5 digits, exact n)
assert np.allclose(force, _ref, rtol=1e-3), (force, _ref)
"""


def test_readme_has_python_examples():
    if not README:
        pytest.skip("no README.md")
    assert len(BLOCKS) >= 4


@pytest.mark.parametrize("index", range(len(BLOCKS)))
def test_readme_block_runs(index):
    block = BLOCKS[index]
    skip_unless(requirements(block))
    code = PRELUDE + block
    if "sws.coefficients" in block:
        code += SMUTHI_CHECK
    out = run(["-c", code])
    assert out.returncode == 0, out.stderr[-3000:]
