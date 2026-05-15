"""Sibling-path discovery for the Grokking-Monitor package.

When Kairos is in development, the Grokking-Monitor repo lives as a
sibling directory. This shim adds it (and transitively Cassandra) to
sys.path so ``from grokking_monitor import GrokkingMonitor`` works
without ``pip install -e``.
"""

from __future__ import annotations

import pathlib
import sys


def _try_sibling() -> bool:
    here = pathlib.Path(__file__).resolve()
    for cand in [
        here.parents[2] / "Grokking-Monitor",
        here.parents[2] / "grokking-monitor",
    ]:
        if (cand / "grokking_monitor" / "__init__.py").exists():
            sp = str(cand)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            return True
    return False


try:
    import grokking_monitor  # noqa: F401
except ImportError:
    if not _try_sibling():
        raise
