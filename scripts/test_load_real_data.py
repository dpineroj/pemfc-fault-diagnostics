"""
Regression test for load_real_data.py's CDM_C units conversion (Task 3
review finding): asserts the loader's current_grid, integrated over all
segments and multiplied by segment area, tracks the bulk file's independent
INTENSIDAD (total load current) channel to within a small tolerance.

This is the cross-check that didn't exist when the units bug was
introduced -- without it, a future refactor of parse_cdm_blocks() or
load_run()'s conversion step could silently reintroduce a scale error (e.g.
dropping the SEGMENT_AREA_CM2 division, or applying it twice) with nothing
catching it, since current_grid's own internal consistency (sign, shape,
grid-block structure) is already checked elsewhere and wouldn't change.

No pytest dependency (not installed in hfc_env) -- plain functions +
asserts, runnable directly or picked up by pytest later if it's ever added.
"""

from __future__ import annotations

import numpy as np

from load_real_data import TOTAL_ACTIVE_AREA_CM2, load_run

# Tolerance on the mean ratio between grid-integrated current and bulk
# INTENSIDAD. 5% comfortably covers the ~0.5% residual actually observed
# (see load_real_data.py's module docstring) while still catching a real
# scale error (e.g. a dropped or doubled area factor, which would be off by
# ~6.48x or ~2x respectively -- nowhere near this tolerance).
RATIO_TOLERANCE = 0.05


def check_grid_matches_bulk_current(config: str, run_type: str = "FC-DLC") -> dict:
    """
    Loads one run and checks that current_grid.sum(axis=(1,2)) * segment_area
    tracks bulk INTENSIDAD, timestamp-aligned by load_run's own merge_asof.
    Returns the ratio array's summary stats; raises AssertionError if the
    mean ratio falls outside RATIO_TOLERANCE of 1.0.
    """
    run = load_run(config, run_type=run_type)
    current_grid = np.abs(run["current_grid"])  # magnitude only -- sign convention, see docs/real_data_notes.md
    n_segments = current_grid.shape[1] * current_grid.shape[2]
    segment_area_cm2 = TOTAL_ACTIVE_AREA_CM2 / n_segments

    grid_integrated_current = current_grid.reshape(len(current_grid), -1).sum(axis=1) * segment_area_cm2
    intensidad = run["bulk"]["INTENSIDAD"].to_numpy()

    valid = ~np.isnan(intensidad)
    ratio = grid_integrated_current[valid] / intensidad[valid]

    mean_ratio = float(ratio.mean())
    assert abs(mean_ratio - 1.0) <= RATIO_TOLERANCE, (
        f"{config}/{run_type}: grid-integrated current vs. bulk INTENSIDAD mean "
        f"ratio={mean_ratio:.4f}, outside +/-{RATIO_TOLERANCE:.0%} of 1.0 -- "
        f"current_grid's units conversion (SEGMENT_AREA_CM2) may be wrong or missing."
    )

    return {
        "config": config,
        "run_type": run_type,
        "n_valid": int(valid.sum()),
        "mean_ratio": mean_ratio,
        "std_ratio": float(ratio.std()),
        "min_ratio": float(ratio.min()),
        "max_ratio": float(ratio.max()),
    }


def test_normal_flow_grid_matches_bulk_current():
    check_grid_matches_bulk_current("Normal_Flow")


def test_inverse_hydrogen_flow_grid_matches_bulk_current():
    check_grid_matches_bulk_current("Inverse_Hydrogen_Flow")


if __name__ == "__main__":
    for config in ("Normal_Flow", "Inverse_Hydrogen_Flow"):
        stats = check_grid_matches_bulk_current(config)
        print(
            f"PASS  {stats['config']:22s}  n={stats['n_valid']:6d}  "
            f"mean_ratio={stats['mean_ratio']:.4f}  std={stats['std_ratio']:.4f}  "
            f"range=[{stats['min_ratio']:.4f}, {stats['max_ratio']:.4f}]"
        )
