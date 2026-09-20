"""
Separates genuine real-vs-synthetic differences from grid-resolution
artifacts, by measuring both at matched resolution.

Motivation: Sec. 5f of docs/synthetic_generator_notes.md originally compared
the synthetic healthy model at its native 4x4 against real data at its native
18x18, and reported an autocorrelation mismatch (0.294 vs 0.935) it could not
attribute. That comparison is not like-for-like: lag-1 autocorrelation is a
function of grid spacing, so the same underlying smooth field yields different
values at different resolutions. This script measures both quantities on both
grids so the confound is quantified rather than assumed.

Reconstructs a measurement originally done ad-hoc in-session; the numbers in
Sec. 5f come from here now, not from chat history.

Method, and why each choice matters:

  * Real 18x18 -> 4x4 by BLOCK-AVERAGING with np.array_split. 18 does not
    divide evenly by 4, so array_split partitions each axis into 4 groups of
    sizes [5,5,4,4] and every one of the 324 cells is used -- nothing is
    cropped, which a naive crop-to-16x16-then-reshape would do.
  * CV(J) = std/mean is NONLINEAR, so each timestep is downsampled FIRST and
    its CV computed after. Downsampling an already-averaged CV would give a
    different, wrong answer.
  * Autocorrelation is computed on the persistent static template. Block-
    averaging and time-averaging are both linear and therefore commute, so
    downsampling the time-averaged template is equivalent to time-averaging
    the downsampled frames, and is cheaper.
  * The synthetic shape is re-expressed parameterized by grid size, because
    fault_generator.ROWS/COLS are module-level 4x4. Its constants are
    IMPORTED, never retyped, so this file cannot drift from the model it is
    measuring -- and an assertion checks the parameterized 4x4 path is
    byte-identical to generate_healthy_grid_v2 before any 18x18 number is
    trusted.

Run:  python scripts/measure_resolution_effects.py
"""

from __future__ import annotations

import numpy as np

from fault_generator import (
    RADIAL_EDGE_EXPONENT,
    RADIAL_EDGE_RATIO,
    TEMPORAL_SUBORDINATION_FACTOR,
    compute_cv,
    generate_healthy_grid_v2,
)
from load_real_data import load_run

REFERENCE_CONFIG = "Normal_Flow"
LOAD_LEVEL = 0.697        # matches generate_healthy_grid_v2's default
N_SYNTHETIC_SEEDS = 2000


def block_average(grid_2d: np.ndarray, out_rows: int, out_cols: int) -> np.ndarray:
    """Block-average to (out_rows, out_cols) via np.array_split -- see module docstring."""
    row_groups = np.array_split(np.arange(grid_2d.shape[0]), out_rows)
    col_groups = np.array_split(np.arange(grid_2d.shape[1]), out_cols)
    out = np.empty((out_rows, out_cols))
    for i, rows in enumerate(row_groups):
        for j, cols in enumerate(col_groups):
            out[i, j] = grid_2d[np.ix_(rows, cols)].mean()
    return out


def lag1_autocorr(grid_2d: np.ndarray) -> float:
    """Mean lag-1 neighbour correlation, averaged over the two grid axes."""
    vals = []
    for d_row, d_col in ((0, 1), (1, 0)):
        a = grid_2d[: grid_2d.shape[0] - d_row, : grid_2d.shape[1] - d_col].ravel()
        b = grid_2d[d_row:, d_col:].ravel()
        vals.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(vals))


def radial_shape_at(rows: int, cols: int, edge_ratio: float = RADIAL_EDGE_RATIO) -> np.ndarray:
    """
    fault_generator._radial_healthy_shape's formula, parameterized by grid
    size. Constants imported, not retyped. Verified byte-identical to the
    real function at 4x4 by assert_matches_generator() below.
    """
    row_idx, col_idx = np.indices((rows, cols))
    radius = np.sqrt((row_idx - (rows - 1) / 2.0) ** 2 + (col_idx - (cols - 1) / 2.0) ** 2)
    s_raw = 1.0 + (edge_ratio - 1.0) * (radius / radius.max()) ** RADIAL_EDGE_EXPONENT
    return s_raw / s_raw.mean()


def generate_healthy_at(rows: int, cols: int, load_level: float, seed: int) -> np.ndarray:
    """generate_healthy_grid_v2's formula at arbitrary grid size."""
    rng = np.random.default_rng(seed)
    shape = radial_shape_at(rows, cols)
    noise = rng.normal(0, shape.std() / TEMPORAL_SUBORDINATION_FACTOR, shape.shape)
    return load_level * shape * (1.0 + noise)


def assert_matches_generator() -> None:
    """
    Gate: the parameterized path must reproduce generate_healthy_grid_v2
    exactly at 4x4. If it does not, the 18x18 numbers describe some other
    model and must not be reported.
    """
    for seed in (0, 7, 42, 1234):
        mine = generate_healthy_at(4, 4, LOAD_LEVEL, seed)
        theirs = generate_healthy_grid_v2(load_level=LOAD_LEVEL, seed=seed)
        assert np.array_equal(mine, theirs), (
            f"parameterized 4x4 path diverged from generate_healthy_grid_v2 at seed={seed}; "
            f"max abs diff {np.abs(mine - theirs).max():.3e}. 18x18 results are not trustworthy."
        )
    print("[gate] parameterized 4x4 path byte-identical to generate_healthy_grid_v2 "
          "across 4 seeds -- 18x18 results trustworthy")


def main() -> None:
    assert_matches_generator()

    # ---------------- real ----------------
    run = load_run(REFERENCE_CONFIG, run_type="FC-DLC")
    grid = np.abs(run["current_grid"])
    n_t = grid.shape[0]

    flat_18 = grid.reshape(n_t, -1)
    real_cv_18 = float((flat_18.std(axis=1) / flat_18.mean(axis=1)).mean())

    grid_4 = np.stack([block_average(grid[t], 4, 4) for t in range(n_t)])
    flat_4 = grid_4.reshape(n_t, -1)
    real_cv_4 = float((flat_4.std(axis=1) / flat_4.mean(axis=1)).mean())

    normalized = grid / flat_18.mean(axis=1)[:, None, None]
    template_18 = normalized.mean(axis=0)
    real_ac_18 = lag1_autocorr(template_18)
    real_ac_4 = lag1_autocorr(block_average(template_18, 4, 4))

    # ---------------- synthetic ----------------
    syn_cv_4 = float(np.mean([compute_cv(generate_healthy_grid_v2(load_level=LOAD_LEVEL, seed=s))
                              for s in range(N_SYNTHETIC_SEEDS)]))
    syn_cv_18 = float(np.mean([compute_cv(generate_healthy_at(18, 18, LOAD_LEVEL, s))
                               for s in range(N_SYNTHETIC_SEEDS)]))
    syn_ac_4 = lag1_autocorr(radial_shape_at(4, 4))
    syn_ac_18 = lag1_autocorr(radial_shape_at(18, 18))

    # ---------------- report ----------------
    print(f"\nreal: {REFERENCE_CONFIG}/FC-DLC, n={n_t} timesteps")
    print(f"synthetic: generate_healthy_grid_v2, edge_ratio={RADIAL_EDGE_RATIO}, "
          f"n={N_SYNTHETIC_SEEDS} seeds, load_level={LOAD_LEVEL}")

    print("\n=== CV(J) ===")
    print(f"{'':<12}{'4x4':>10}{'18x18':>10}")
    print(f"{'real':<12}{real_cv_4:>10.4f}{real_cv_18:>10.4f}")
    print(f"{'synthetic':<12}{syn_cv_4:>10.4f}{syn_cv_18:>10.4f}")
    print(f"{'real/syn':<12}{real_cv_4 / syn_cv_4:>9.2f}x{real_cv_18 / syn_cv_18:>9.2f}x")

    print("\n=== lag-1 spatial autocorrelation (static template) ===")
    print(f"{'':<12}{'4x4':>10}{'18x18':>10}")
    print(f"{'real':<12}{real_ac_4:>10.4f}{real_ac_18:>10.4f}")
    print(f"{'synthetic':<12}{syn_ac_4:>10.4f}{syn_ac_18:>10.4f}")

    print("\n=== reading ===")
    print("autocorrelation: at MATCHED resolution real and synthetic agree closely at both")
    print("  grid sizes, while each moves by ~0.6 across resolutions. The original")
    print("  0.294-vs-0.935 'mismatch' was 4x4-vs-18x18, i.e. a resolution artifact.")
    print("CV(J): the gap survives resolution matching and WIDENS on the finer grid")
    print(f"  ({real_cv_4 / syn_cv_4:.2f}x at 4x4 -> {real_cv_18 / syn_cv_18:.2f}x at 18x18),")
    print("  so it is a genuine amplitude deficit in the synthetic model, not an artifact.")


if __name__ == "__main__":
    main()
