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

Covers TWO synthetic forms:

  v2 (superseded)  S_raw = 1 + A*(rho)^2                 -- Sec. 5
  logistic (adopted, Sec. 6c)
                   g     = [sig((rho-r0)/w) - sig(-r0/w)]
                           / [sig((1-r0)/w) - sig(-r0/w)]
                   S_raw = c + (1-c)*g
  both then normalized S = S_raw / mean(S_raw).

The logistic form is NOT yet implemented in fault_generator.py -- it is
defined here because this is a measurement of it, not an adoption of it into
the generator. When Phase 4 needs the family, it gets implemented there.

Method, and why each choice matters:

  * Real 18x18 -> 4x4 by BLOCK-AVERAGING with np.array_split. 18 does not
    divide evenly by 4, so array_split partitions each axis into 4 groups of
    sizes [5,5,4,4] and every one of the 324 cells is used -- nothing is
    cropped, which a naive crop-to-16x16-then-reshape would do. Block-
    averaging is the right operator because it is what a physically coarser
    sensor would do: integrate current over a larger segment area.
  * CV(J) = std/mean is NONLINEAR, so each timestep is downsampled FIRST and
    its CV computed after. Downsampling an already-averaged CV would give a
    different, wrong answer.
  * Autocorrelation is computed on the persistent static template. Block-
    averaging and time-averaging are both linear and therefore commute, so
    downsampling the time-averaged template is equivalent to time-averaging
    the downsampled frames, and is cheaper.
  * For v2, an assertion checks the grid-size-parameterized path is
    byte-identical to generate_healthy_grid_v2 at 4x4 before any 18x18 number
    is trusted.
  * For the logistic family, the SAME (c, r0, w) is evaluated at both
    resolutions (a paired comparison), which isolates the pure resolution
    effect from parameter variation.

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

CONFIGS = ("Normal_Flow", "Inverse_Flow", "Inverse_Air_Flow", "Inverse_Hydrogen_Flow")
REFERENCE_CONFIG = "Normal_Flow"
LOAD_LEVEL = 0.697        # matches generate_healthy_grid_v2's default
N_SYNTHETIC_SEEDS = 2000

# Adopted family bounds (docs/synthetic_generator_notes.md Sec. 6g).
C_RANGE = (0.05, 0.60)
R0_RANGE = (0.25, 0.70)
W_RANGE = (0.04, 0.35)


def rho_grid(n: int) -> np.ndarray:
    """Normalized radius r/r_max on an n x n grid."""
    row, col = np.indices((n, n))
    r = np.sqrt((row - (n - 1) / 2.0) ** 2 + (col - (n - 1) / 2.0) ** 2)
    return r / r.max()


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


# ---------------------------------------------------------------- v2 (superseded)

def radial_shape_at(rows: int, cols: int, edge_ratio: float = RADIAL_EDGE_RATIO) -> np.ndarray:
    """fault_generator._radial_healthy_shape's formula, parameterized by grid size."""
    rho = rho_grid(rows) if rows == cols else None
    if rho is None:
        raise ValueError("square grids only")
    s_raw = 1.0 + (edge_ratio - 1.0) * rho ** RADIAL_EDGE_EXPONENT
    return s_raw / s_raw.mean()


def generate_healthy_at(rows: int, cols: int, load_level: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = radial_shape_at(rows, cols)
    noise = rng.normal(0, shape.std() / TEMPORAL_SUBORDINATION_FACTOR, shape.shape)
    return load_level * shape * (1.0 + noise)


def assert_matches_generator() -> None:
    """Gate: parameterized path must reproduce generate_healthy_grid_v2 at 4x4."""
    for seed in (0, 7, 42, 1234):
        mine = generate_healthy_at(4, 4, LOAD_LEVEL, seed)
        theirs = generate_healthy_grid_v2(load_level=LOAD_LEVEL, seed=seed)
        assert np.array_equal(mine, theirs), (
            f"parameterized 4x4 path diverged from generate_healthy_grid_v2 at seed={seed}; "
            f"max abs diff {np.abs(mine - theirs).max():.3e}. 18x18 results are not trustworthy."
        )
    print("[gate] v2 parameterized 4x4 path byte-identical to generate_healthy_grid_v2 "
          "across 4 seeds -- 18x18 results trustworthy")


# ---------------------------------------------------------------- logistic (adopted)

def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def logistic_shape(n: int, c: float, r0: float, w: float) -> np.ndarray:
    """Adopted inflected form (Sec. 6c), normalized to mean 1."""
    rho = rho_grid(n)
    w = max(w, 1e-4)
    lo, hi = _sig(-r0 / w), _sig((1.0 - r0) / w)
    g = (_sig((rho - r0) / w) - lo) / (hi - lo)
    raw = c + (1.0 - c) * g
    return raw / raw.mean()


def main() -> None:
    assert_matches_generator()

    # ============================== real ==============================
    print("\n" + "=" * 72)
    print("REAL -- all four configurations, native 18x18 vs block-averaged 4x4")
    print("=" * 72)
    real_rows = []
    for cfg in CONFIGS:
        run = load_run(cfg, run_type="FC-DLC")
        grid = np.abs(run["current_grid"])
        n_t = grid.shape[0]

        flat18 = grid.reshape(n_t, -1)
        cv18 = float((flat18.std(axis=1) / flat18.mean(axis=1)).mean())

        g4 = np.stack([block_average(grid[t], 4, 4) for t in range(n_t)])
        flat4 = g4.reshape(n_t, -1)
        cv4 = float((flat4.std(axis=1) / flat4.mean(axis=1)).mean())

        tmpl18 = (grid / flat18.mean(axis=1)[:, None, None]).mean(axis=0)
        ac18 = lag1_autocorr(tmpl18)
        ac4 = lag1_autocorr(block_average(tmpl18, 4, 4))
        real_rows.append((cfg, cv4, cv18, ac4, ac18))

    print(f"{'config':<23}{'CV 4x4':>9}{'CV 18x18':>10}{'ratio':>8}{'AC 4x4':>9}{'AC 18x18':>10}")
    for cfg, cv4, cv18, ac4, ac18 in real_rows:
        print(f"{cfg:<23}{cv4:>9.4f}{cv18:>10.4f}{cv18/cv4:>8.2f}{ac4:>9.4f}{ac18:>10.4f}")
    rcv4 = np.mean([r[1] for r in real_rows]); rcv18 = np.mean([r[2] for r in real_rows])
    rac4 = np.mean([r[3] for r in real_rows]); rac18 = np.mean([r[4] for r in real_rows])
    print(f"{'MEAN':<23}{rcv4:>9.4f}{rcv18:>10.4f}{rcv18/rcv4:>8.2f}{rac4:>9.4f}{rac18:>10.4f}")

    # ====================== synthetic: v2 (superseded) ======================
    print("\n" + "=" * 72)
    print("SYNTHETIC v2 (superseded form, edge_ratio=2.0)")
    print("=" * 72)
    v2_cv4 = float(np.mean([compute_cv(generate_healthy_grid_v2(load_level=LOAD_LEVEL, seed=s))
                            for s in range(N_SYNTHETIC_SEEDS)]))
    v2_cv18 = float(np.mean([compute_cv(generate_healthy_at(18, 18, LOAD_LEVEL, s))
                             for s in range(N_SYNTHETIC_SEEDS)]))
    v2_ac4 = lag1_autocorr(radial_shape_at(4, 4))
    v2_ac18 = lag1_autocorr(radial_shape_at(18, 18))
    print(f"  CV(J)          4x4={v2_cv4:.4f}   18x18={v2_cv18:.4f}   ratio={v2_cv18/v2_cv4:.2f}")
    print(f"  autocorrelation 4x4={v2_ac4:.4f}   18x18={v2_ac18:.4f}")

    # ====================== synthetic: adopted logistic family ======================
    print("\n" + "=" * 72)
    print("SYNTHETIC logistic family (adopted, Sec. 6c/6g)")
    print("=" * 72)
    print("PAIRED test -- the SAME (c,r0,w) evaluated at both resolutions, which")
    print("isolates the pure resolution effect from parameter variation.")

    rng = np.random.default_rng(20260810)
    n_draw = 4000
    cs = rng.uniform(*C_RANGE, n_draw)
    r0s = rng.uniform(*R0_RANGE, n_draw)
    ws = rng.uniform(*W_RANGE, n_draw)

    cv4_arr = np.empty(n_draw); cv18_arr = np.empty(n_draw)
    for i in range(n_draw):
        cv4_arr[i] = logistic_shape(4, cs[i], r0s[i], ws[i]).std()
        cv18_arr[i] = logistic_shape(18, cs[i], r0s[i], ws[i]).std()
    ratio = cv18_arr / cv4_arr

    print(f"\n  family CV(J) at 4x4  : mean={cv4_arr.mean():.4f}  "
          f"p5={np.percentile(cv4_arr,5):.4f}  p95={np.percentile(cv4_arr,95):.4f}")
    print(f"  family CV(J) at 18x18: mean={cv18_arr.mean():.4f}  "
          f"p5={np.percentile(cv18_arr,5):.4f}  p95={np.percentile(cv18_arr,95):.4f}")
    print(f"\n  paired 18x18/4x4 ratio: mean={ratio.mean():.4f}  "
          f"p5={np.percentile(ratio,5):.4f}  p95={np.percentile(ratio,95):.4f}  "
          f"max|dev from 1|={np.abs(ratio-1).max():.4f}")

    # at the real sessions' own fitted parameters (measurement, not fitting)
    print("\n  at the real sessions' fitted (c,r0,w) -- measurement only, no parameter set here:")
    fitted = {"Normal_Flow": (0.1390, 0.4478, 0.1123),
              "Inverse_Flow": (0.2000, 0.4273, 0.1146),
              "Inverse_Air_Flow": (0.1403, 0.4482, 0.1131),
              "Inverse_Hydrogen_Flow": (0.2055, 0.4273, 0.1142)}
    print(f"    {'config':<23}{'CV 4x4':>9}{'CV 18x18':>10}{'ratio':>8}{'AC 4x4':>9}{'AC 18x18':>10}")
    for cfg, (c, r0, w) in fitted.items():
        s4, s18 = logistic_shape(4, c, r0, w), logistic_shape(18, c, r0, w)
        print(f"    {cfg:<23}{s4.std():>9.4f}{s18.std():>10.4f}"
              f"{s18.std()/s4.std():>8.2f}{lag1_autocorr(s4):>9.4f}{lag1_autocorr(s18):>10.4f}")

    # ============================== 2x2 ==============================
    print("\n" + "=" * 72)
    print("2x2 SUMMARY -- {real, synthetic} x {4x4, 18x18}")
    print("=" * 72)
    fam4, fam18 = cv4_arr.mean(), cv18_arr.mean()
    nf = fitted["Normal_Flow"]
    nf4, nf18 = logistic_shape(4, *nf), logistic_shape(18, *nf)
    print("\nCV(J)")
    print(f"{'':<34}{'4x4':>10}{'18x18':>10}")
    print(f"{'real (4-config mean)':<34}{rcv4:>10.4f}{rcv18:>10.4f}")
    print(f"{'synthetic v2 (superseded)':<34}{v2_cv4:>10.4f}{v2_cv18:>10.4f}")
    print(f"{'synthetic logistic family (mean)':<34}{fam4:>10.4f}{fam18:>10.4f}")
    print(f"{'  real/v2':<34}{rcv4/v2_cv4:>9.2f}x{rcv18/v2_cv18:>9.2f}x")
    print(f"{'  real/family':<34}{rcv4/fam4:>9.2f}x{rcv18/fam18:>9.2f}x")
    print("\nlag-1 autocorrelation (static template)")
    print(f"{'':<34}{'4x4':>10}{'18x18':>10}")
    print(f"{'real (4-config mean)':<34}{rac4:>10.4f}{rac18:>10.4f}")
    print(f"{'synthetic v2 (superseded)':<34}{v2_ac4:>10.4f}{v2_ac18:>10.4f}")
    print(f"{'synthetic logistic @ NF params':<34}"
          f"{lag1_autocorr(nf4):>10.4f}{lag1_autocorr(nf18):>10.4f}")


if __name__ == "__main__":
    main()
