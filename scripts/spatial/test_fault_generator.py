"""
Regression tests for healthy spatial-structure model
(generate_healthy_grid_v2, fault_generator.py): asserts CV(J) stays within
its documented range and remains load-invariant.

Guards against two ways this could silently break in a future refactor:
1. A change to RADIAL_EDGE_RATIO, RADIAL_EDGE_EXPONENT, or
   TEMPORAL_SUBORDINATION_FACTOR that isn't reflected in the documented
   range in docs/synthetic_generator_notes.md (the range below IS that
   documented range, not a separately-chosen tolerance).
2. A change to generate_healthy_grid_v2 that breaks the S/mean(S)
   renormalization and reintroduces load-dependence in CV(J) -- this is
   the load-invariance property the whole model is built around; it must
   never be assumed again without a test catching a regression.

No pytest dependency (not installed in hfc_env, see test_spatial/cdm_parser.py
for the same note) -- plain functions + asserts.
"""

from __future__ import annotations
import numpy as np
# --- path bootstrap: siblings moved into scripts/{common,spatial} by the
# November 2026 restructure. Keeps `python scripts/<dir>/<file>.py` working
# without turning scripts/ into a package, which would change every
# documented invocation.
import os as _os
import sys as _sys
_S = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(_S, "common"), _os.path.join(_S, "spatial")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path bootstrap ---

from fault_generator import RADIAL_EDGE_RATIO, compute_cv, generate_healthy_grid_v2

# Documented range for CV(J) at the default edge_ratio=2.0 (see
# docs/synthetic_generator_notes.md's Phase 3 sensitivity table): measured
# mean 0.2024 (n=2000 seeds) with per-seed range [0.1920, 0.2148]. A wide
# band around that is used here -- this test catches a broken model
# (wrong exponent, dropped renormalization, wildly different noise scale),
# not minor seed-to-seed noise.
CV_RANGE = (0.15, 0.25)

# CV(J) must be stable across load_level to within tight numerical
# tolerance -- it's exact by construction (S is renormalized to mean 1
# before load_level scales the grid), so any drift here means the
# renormalization broke, not sampling noise.
LOAD_INVARIANCE_TOLERANCE = 1e-9


def test_healthy_grid_cv_in_documented_range():
    cvs = np.array([compute_cv(generate_healthy_grid_v2(load_level=0.697, seed=s)) for s in range(500)])
    mean_cv = float(cvs.mean())
    assert CV_RANGE[0] <= mean_cv <= CV_RANGE[1], (
        f"generate_healthy_grid_v2 mean CV(J)={mean_cv:.4f} at edge_ratio="
        f"{RADIAL_EDGE_RATIO} fell outside the documented range {CV_RANGE} -- "
        f"check RADIAL_EDGE_RATIO/RADIAL_EDGE_EXPONENT/TEMPORAL_SUBORDINATION_FACTOR "
        f"against docs/synthetic_generator_notes.md."
    )
    return mean_cv


def test_healthy_grid_cv_load_invariant():
    load_levels = (0.15, 0.35, 0.697, 1.0, 1.5)
    means = []
    for load in load_levels:
        cvs = np.array([compute_cv(generate_healthy_grid_v2(load_level=load, seed=s)) for s in range(200)])
        means.append(float(cvs.mean()))
    spread = max(means) - min(means)
    assert spread <= LOAD_INVARIANCE_TOLERANCE, (
        f"CV(J) varied by {spread:.2e} across load_level={load_levels} -- "
        f"expected exact invariance (S/mean(S) renormalization broken?). Per-level means: {means}"
    )
    return means


if __name__ == "__main__":
    mean_cv = test_healthy_grid_cv_in_documented_range()
    print(f"PASS  CV(J) in documented range  mean={mean_cv:.4f}  range={CV_RANGE}")
    means = test_healthy_grid_cv_load_invariant()
    print(f"PASS  CV(J) load-invariant  means={[f'{m:.6f}' for m in means]}")
