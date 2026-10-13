"""
Counterfactual band for Sec. 11b Test 3's threshold.

WHY (Sec. 11i item 5): Test 3's 0.95 has weaker provenance than Test 1's and
Test 2's thresholds. Those were pre-registered in Sec. 11b before any fault
code existed, with a measured null behind 0.560. The 0.95 was chosen while
first writing the acceptance script -- before any Test 3 measurement existed,
but never recorded in the design document in advance. "Chosen before
measuring" is not the same standard as "pre-registered", and only the latter
is tamper-evident.

So rather than asserting the threshold does not matter, this script BOUNDS the
dependence: across what range of thresholds does the Phase 3b conclusion
survive unchanged? The conclusion under test is

    flooding-starvation FAILS on the no_axis arm, while every other
    fault pair on either arm PASSES

which survives for any threshold strictly above the worst-case failing pair
and at or below the worst-case passing pair. Reported across several seeds, so
the band reflects sampling variability rather than one draw.

Run:  python scripts/measure_test3_band.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fault_generator as F  # noqa: E402
import measure_battery_null as B  # noqa: E402

N_SEEDS = 10
N_PER_FAULT = 1000
PAIRS = (("drying", "flooding"), ("drying", "starvation"),
         ("flooding", "starvation"))
FAILING = ("no_axis", "flooding", "starvation")


def pairwise_once(arm, seed, rows, cols, rho, fields):
    rng = np.random.default_rng(seed)
    feats = {}
    for ft in F.PHASE3B_FAULTS:
        c, r0, w, tr = B.sample_params_vec(N_PER_FAULT, rng)
        axes = rng.uniform(0.0, 2.0 * np.pi, N_PER_FAULT)
        tpl = B.templates_vec(c, r0, w, rho)
        grids = B.grids_vec(tpl, tr, rng)
        fg = np.empty_like(grids)
        for i in range(N_PER_FAULT):
            p = {"c": c[i], "r0": r0[i], "w": w[i], "temporal_ratio": tr[i],
                 "starvation_axis_rad": axes[i]}
            fg[i] = grids[i] * F.fault_modulation(ft, 1.0, rows, cols, p, arm)
        feats[ft] = B.battery_vec(fg, tpl, fields)
    return {(a, b): float(B.best_threshold_accuracy_cols(feats[a], feats[b]).max())
            for a, b in PAIRS}


def main():
    rows, cols = F.DEFAULT_GRID_SHAPE
    rho = B._rho(rows, cols)
    fields = B.coord_fields(rows, cols)

    print("Test 3 pairwise separability across seeds")
    print(f"grid={rows}x{cols}  N={N_PER_FAULT}/fault  seeds={N_SEEDS}")
    print()
    res = {}
    for seed in range(N_SEEDS):
        for arm in F.STARVATION_ARMS:
            for pair, v in pairwise_once(arm, 1000 + seed, rows, cols,
                                         rho, fields).items():
                res.setdefault((arm,) + pair, []).append(v)

    print(f"{'arm':>12} {'pair':>24} {'min':>7} {'max':>7}")
    print("-" * 54)
    worst_fail, worst_pass = 0.0, 1.0
    for key, vals in res.items():
        arm, a, b = key
        v = np.array(vals)
        print(f"{arm:>12} {a + '-' + b:>24} {v.min():>7.3f} {v.max():>7.3f}")
        if key == FAILING:
            worst_fail = max(worst_fail, float(v.max()))
        else:
            worst_pass = min(worst_pass, float(v.min()))

    print()
    print(f"  worst case of the failing pair {FAILING[1]}-{FAILING[2]}"
          f"[{FAILING[0]}] : {worst_fail:.3f}")
    print(f"  worst case across every other pair                    "
          f"   : {worst_pass:.3f}")
    print()
    print(f"  CONCLUSION HOLDS for any threshold T in "
          f"({worst_fail:.3f}, {worst_pass:.3f}]   width {worst_pass - worst_fail:.3f}")
    print(f"  adopted T3_MIN = 0.95 -> margin {0.95 - worst_fail:.3f} below, "
          f"{worst_pass - 0.95:.3f} above")
    print()
    print("  The band is wide but the adopted value is NOT centred in it: at")
    print("  T > the upper edge, drying-flooding fails too and the conclusion")
    print("  changes from 'one pair collapsed' to 'Test 3 failing broadly'.")


if __name__ == "__main__":
    main()
