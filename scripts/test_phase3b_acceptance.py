"""
Sec. 11b acceptance tests for the Phase 3b fault rework -- drying and flooding.

Every threshold here was fixed in docs/synthetic_generator_notes.md Sec. 11b
BEFORE this code existed, specifically so it could not drift to accommodate
whatever the rework produced. Nothing in this file may relax one. If a test
fails, the generator is wrong, not the threshold.

The battery is imported from measure_battery_null rather than redefined --
Sec. 11b names that module's FEATURE_NAMES as the authority for the battery's
seven features (across six numbered items; item 4 contributes two diagonals).

Run:  python scripts/test_phase3b_acceptance.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fault_generator as F  # noqa: E402
import measure_battery_null as B  # noqa: E402

GRID = F.DEFAULT_GRID_SHAPE
FAULTS = F.PHASE3B_FAULTS
ARMS = F.STARVATION_ARMS
PAIRS = (("drying", "flooding"), ("drying", "starvation"),
         ("flooding", "starvation"))

# Sec. 11h item 3, fixed BEFORE measuring: starvation's pass/fail quantity is
# mean(|J|) depression, robust to whether a flow-path gradient exists. CV(J)
# is deliberately NOT a criterion for starvation -- the two arms disagree on
# its direction by construction -- and is reported as a diagnostic only.
# WEAKER PROVENANCE than T1/T2's thresholds -- see Sec. 11i item 5. This was
# chosen while first writing this script, before any Test 3 measurement
# existed, but it was never pre-registered in Sec. 11 the way T2_THRESHOLD was.
# It is bounded instead: the conclusion (flooding-starvation fails on the
# no_axis arm, every other pair passes) holds for ANY threshold in
# (0.667, 0.970] across 10 seeds -- scripts/measure_test3_band.py.
T3_MIN = 0.95

# ---- Sec. 11b Test 1, as specified -----------------------------------------
T1_EPS = (0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00)
T1_CELLS = 200
T1_CONVERGENCE = 0.02      # d(0.01) <= this
T1_JITTER = 0.002          # allowed local decrease
T1_TREND = ((0.01, 0.10), (0.10, 0.50), (0.50, 1.00))

# ---- Sec. 11b Test 2, as specified -----------------------------------------
T2_N = 1000
T2_SEVERITIES = (0.01, 0.02, 0.05)
T2_THRESHOLD = 0.560

# ---- Sec. 11b Test 4 -------------------------------------------------------
T4_EXPECTED_EPISODES = {"starvation": 34, "flooding": 24, "drying": 9}
T4_LOADS = (0.2, 0.4, 0.6, 0.8, 1.0)

_results = []


def mod(ft, sev, params, arm, rows, cols):
    """fault_modulation, supplying params/arm only where starvation needs them."""
    return F.fault_modulation(ft, sev, rows, cols, params, arm)


def _cases():
    """(fault, arm) pairs. Starvation appears TWICE, once per arm; Sec. 11d/11h
    fixes that neither arm is the default, so neither is skipped."""
    out = []
    for ft in FAULTS:
        if ft == "starvation":
            out.extend((ft, a) for a in ARMS)
        else:
            out.append((ft, None))
    return out


def record(name, passed, detail=""):
    _results.append((name, passed, detail))
    print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))


# ===========================================================================
def test1_continuity():
    print("=" * 74)
    print("TEST 1 -- Continuity (PRIMARY). Deterministic templates, no noise.")
    print(f"  grid {T1_EPS}")
    print(f"  {T1_CELLS} cells; every criterion required of EVERY cell")
    print("=" * 74)
    rng = np.random.default_rng(31337)
    rows, cols = GRID

    for ft, arm in _cases():
        label = ft if arm is None else f"{ft}[{arm}]"
        d_all = np.empty((T1_CELLS, len(T1_EPS)))
        for i in range(T1_CELLS):
            p = F.sample_healthy_params(rng)
            healthy = F.fault_template_v3(ft, 0.0, p, rows=rows, cols=cols,
                                          starvation_arm=arm)
            denom = np.sqrt((healthy ** 2).sum())
            for k, eps in enumerate(T1_EPS):
                tpl = F.fault_template_v3(ft, eps, p, rows=rows, cols=cols,
                                          starvation_arm=arm)
                d_all[i, k] = np.sqrt(((tpl - healthy) ** 2).sum()) / denom

        conv = d_all[:, 0]
        record(f"T1(i) convergence  {label}",
               bool(conv.max() <= T1_CONVERGENCE),
               f"max d(0.01)={conv.max():.5f} <= {T1_CONVERGENCE}")

        drops = d_all[:, :-1] - d_all[:, 1:]          # positive = a decrease
        record(f"T1(ii) local monotonicity  {label}",
               bool(drops.max() <= T1_JITTER),
               f"worst decrease={max(drops.max(), 0.0):.2e} <= {T1_JITTER}")

        ok_trend, worst = True, None
        for a, b in T1_TREND:
            ia, ib = T1_EPS.index(a), T1_EPS.index(b)
            margin = (d_all[:, ib] - d_all[:, ia]).min()
            if margin <= 0:
                ok_trend = False
            if worst is None or margin < worst:
                worst = margin
        record(f"T1(iii) trend  {label}", ok_trend,
               f"min margin across checkpoints={worst:.5f} > 0")

        print(f"    {label}: mean d(eps) = "
              + ", ".join(f"{e}:{d_all[:, k].mean():.4f}"
                          for k, e in enumerate(T1_EPS)))
    print()


# ===========================================================================
def test2_separability():
    print("=" * 74)
    print(f"TEST 2 -- Severity-resolved separability. N={T2_N}/class,")
    print(f"  threshold {T2_THRESHOLD} at severity <= 0.05 (chance 0.500)")
    print("=" * 74)
    rows, cols = GRID
    rho = B._rho(rows, cols)
    fields = B.coord_fields(rows, cols)
    rng = np.random.default_rng(20261012)

    for ft, arm in _cases():
        label = ft if arm is None else f"{ft}[{arm}]"
        for sev in T2_SEVERITIES:
            c, r0, w, tr = B.sample_params_vec(2 * T2_N, rng)
            axes = rng.uniform(0.0, 2.0 * np.pi, 2 * T2_N)
            tpl = B.templates_vec(c, r0, w, rho)
            grids = B.grids_vec(tpl, tr, rng)
            # healthy half untouched; fault half modulated, cell by cell so
            # starvation's per-cell axis / per-cell template coupling is real
            fg = np.empty_like(grids[T2_N:])
            for i in range(T2_N):
                p = {"c": c[T2_N + i], "r0": r0[T2_N + i], "w": w[T2_N + i],
                     "temporal_ratio": tr[T2_N + i],
                     "starvation_axis_rad": axes[T2_N + i]}
                fg[i] = grids[T2_N + i] * mod(ft, sev, p, arm, rows, cols)
            h_feats = B.battery_vec(grids[:T2_N], tpl[:T2_N], fields)
            f_feats = B.battery_vec(fg, tpl[T2_N:], fields)
            acc = B.best_threshold_accuracy_cols(h_feats, f_feats)
            worst_k = int(np.argmax(acc))
            record(f"T2  {label} @ sev={sev}",
                   bool(acc.max() <= T2_THRESHOLD),
                   f"max={acc.max():.3f} ({B.FEATURE_NAMES[worst_k]}) "
                   f"<= {T2_THRESHOLD}")
    print()


# ===========================================================================
def test3_cross_fault():
    """
    Sec. 11h: reported PAIRWISE across all three fault pairs, per arm. An
    aggregate would hide a starvation-flooding collapse behind
    drying-flooding's separation, which passes the letter and fails the intent.
    """
    print("=" * 74)
    print("TEST 3 -- Cross-fault distinguishability at full severity, PAIRWISE")
    print("=" * 74)
    rows, cols = GRID
    rho = B._rho(rows, cols)
    fields = B.coord_fields(rows, cols)
    n = T2_N

    for arm in ARMS:
        print(f"  --- starvation arm: {arm} ---")
        rng = np.random.default_rng(555)
        feats, radial = {}, {}
        for ft in FAULTS:
            c, r0, w, tr = B.sample_params_vec(n, rng)
            axes = rng.uniform(0.0, 2.0 * np.pi, n)
            tpl = B.templates_vec(c, r0, w, rho)
            grids = B.grids_vec(tpl, tr, rng)
            fg = np.empty_like(grids)
            for i in range(n):
                p = {"c": c[i], "r0": r0[i], "w": w[i], "temporal_ratio": tr[i],
                     "starvation_axis_rad": axes[i]}
                fg[i] = grids[i] * mod(ft, 1.0, p, arm, rows, cols)
            feats[ft] = B.battery_vec(fg, tpl, fields)
            radial[ft] = feats[ft][:, 0].mean()
        print("    radial-corr means: "
              + ", ".join(f"{ft} {radial[ft]:+.3f}" for ft in FAULTS))
        for a, b in PAIRS:
            acc = B.best_threshold_accuracy_cols(feats[a], feats[b])
            k = int(np.argmax(acc))
            record(f"T3[{arm}]  {a} vs {b}", bool(acc.max() >= T3_MIN),
                   f"best={acc.max():.3f} ({B.FEATURE_NAMES[k]}) >= {T3_MIN}")
        print()


def test4_invariants():
    print("=" * 74)
    print("TEST 4 -- Preserved invariants")
    print("=" * 74)
    rows, cols = GRID
    rng = np.random.default_rng(99)
    p = F.sample_healthy_params(rng)

    for ft, arm in _cases():
        label = ft if arm is None else f"{ft}[{arm}]"
        cvs = []
        for load in T4_LOADS:
            g = F.generate_fault_grid_v3(ft, 0.6, p, seed=7, load_level=load,
                                         rows=rows, cols=cols,
                                         starvation_arm=arm)
            cvs.append(F.compute_cv(g))
        spread = max(cvs) - min(cvs)
        record(f"T4  CV(J) load-invariant  {label}", bool(spread < 1e-9),
               f"spread={spread:.2e}")

    # episode counts must be identical on BOTH arms: the trigger stream keys
    # off the load trace, which no spatial choice may touch
    runs = {}
    for arm in ARMS:
        run = F.generate_labeled_run(starvation_arm=arm)
        runs[arm] = run
        counts = {}
        for ep in run["episodes"]:
            counts[ep["fault_type"]] = counts.get(ep["fault_type"], 0) + 1
        record(f"T4  episode counts unchanged [{arm}]",
               counts == T4_EXPECTED_EPISODES, f"{counts}")
    a, b = ARMS
    ta = np.array([e["severity_target"] for e in runs[a]["episodes"]])
    tb = np.array([e["severity_target"] for e in runs[b]["episodes"]])
    record("T4  severity_targets bit-identical across arms",
           bool(np.array_equal(ta, tb)),
           f"n={len(ta)}  sum={ta.sum():.6f}")
    return runs


def test_sec11f_item3():
    """
    Sec. 11f item 3 / Sec. 11h item 3, all criteria fixed before measuring:
      drying      -- must RAISE CV(J)
      flooding    -- must DEPRESS mean(|J|)   (CV(J) is vacuous per Sec. 2)
      starvation  -- must DEPRESS mean(|J|), both arms. CV(J) is NOT a
                     criterion and is reported as a diagnostic only.
    """
    print("=" * 74)
    print("Sec. 11f item 3 -- magnitude/dispersion direction")
    print("=" * 74)
    rows, cols = GRID
    rng = np.random.default_rng(4242)
    N = 300
    acc = {k: [] for k in ("cv_h", "mj_h")}
    per = {}
    for ft, arm in _cases():
        per[(ft, arm)] = {"cv": [], "mj": []}
    for _ in range(N):
        p = F.sample_healthy_params(rng)
        sd = int(rng.integers(1 << 62))
        h = F.generate_healthy_grid_v3(p, sd, rows=rows, cols=cols)
        acc["cv_h"].append(F.compute_cv(h))
        acc["mj_h"].append(np.abs(h).mean())
        for ft, arm in _cases():
            g = F.generate_fault_grid_v3(ft, 0.9, p, sd, rows=rows, cols=cols,
                                         starvation_arm=arm)
            per[(ft, arm)]["cv"].append(F.compute_cv(g))
            per[(ft, arm)]["mj"].append(np.abs(g).mean())
    cv_h = np.array(acc["cv_h"])
    mj_h = np.array(acc["mj_h"])

    for (ft, arm), d in per.items():
        label = ft if arm is None else f"{ft}[{arm}]"
        cv = np.array(d["cv"])
        mj = np.array(d["mj"])
        if ft == "drying":
            record(f"11f3  {label} raises CV(J)",
                   bool((cv > cv_h).mean() > 0.99),
                   f"CV {cv_h.mean():.4f} -> {cv.mean():.4f} "
                   f"({100*(cv > cv_h).mean():.1f}%)")
        else:
            record(f"11f3  {label} depresses mean(|J|)",
                   bool((mj < mj_h).mean() > 0.99),
                   f"mean|J| {mj_h.mean():.4f} -> {mj.mean():.4f} "
                   f"({100*(mj < mj_h).mean():.1f}%)")
        if ft == "starvation":
            direction = "RISES" if cv.mean() > cv_h.mean() else "FALLS"
            print(f"    diagnostic only, not a criterion: {label} CV(J) "
                  f"{cv_h.mean():.4f} -> {cv.mean():.4f}  ({direction})")
    print()


def main():
    test1_continuity()
    test2_separability()
    test3_cross_fault()
    test4_invariants()
    test_sec11f_item3()

    print("=" * 74)
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"SUMMARY: {len(_results) - n_fail}/{len(_results)} passed")
    for name, ok, detail in _results:
        if not ok:
            print(f"  FAILED: {name}  {detail}")
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
