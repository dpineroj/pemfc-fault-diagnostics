"""
Read-only access to the frozen reference generator.

scripts/spatial/fault_generator.py is imported, never modified. hfc reuses
its radial clamping template, fault modulation constants, trigger finders and
episode builders so that the new generator cannot silently drift from the
reference the project's results were obtained on.
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO, "scripts", "common"),
           os.path.join(REPO, "scripts", "spatial")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fault_generator as F  # noqa: E402,F401
