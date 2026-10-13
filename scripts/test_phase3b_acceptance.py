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

    for ft in FAULTS:
        d_all = np.empty((T1_CELLS, len(T1_EPS)))
        for i in range(T1_CELLS):
            p = F.sample_healthy_params(rng)
            healthy = F.fault_template_v3(ft, 0.0, p, rows=rows, cols=cols)
            denom = np.sqrt((healthy ** 2).sum())
            for k, eps in enumerate(T1_EPS):
                tpl = F.fault_template_v3(ft, eps, p, rows=rows, cols=cols)
                d_all[i, k] = np.sqrt(((tpl - healthy) ** 2).sum()) / denom

        conv = d_all[:, 0]
        record(f"T1(i) convergence  {ft}",
               bool(conv.max() <= T1_CONVERGENCE),
               f"max d(0.01)={conv.max():.5f} <= {T1_CONVERGENCE}")

        drops = d_all[:, :-1] - d_all[:, 1:]          # positive = a decrease
        record(f"T1(ii) local monotonicity  {ft}",
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
        record(f"T1(iii) trend  {ft}", ok_trend,
               f"min margin across checkpoints={worst:.5f} > 0")

        print(f"    {ft}: mean d(eps) = "
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

    for ft in FAULTS:
        for sev in T2_SEVERITIES:
            c, r0, w, tr = B.sample_params_vec(2 * T2_N, rng)
            tpl = B.templates_vec(c, r0, w, rho)
            grids = B.grids_vec(tpl, tr, rng)
            # healthy half untouched; fault half modulated by the same cells
            mod = F.fault_modulation(ft, sev, rows, cols)
            h_feats = B.battery_vec(grids[:T2_N], tpl[:T2_N], fields)
            f_feats = B.battery_vec(grids[T2_N:] * mod[None], tpl[T2_N:], fields)
            acc = B.best_threshold_accuracy_cols(h_feats, f_feats)
            worst_k = int(np.argmax(acc))
            record(f"T2  {ft} @ sev={sev}",
                   bool(acc.max() <= T2_THRESHOLD),
                   f"max={acc.max():.3f} ({B.FEATURE_NAMES[worst_k]}) "
                   f"<= {T2_THRESHOLD}")
    print()


# ===========================================================================
def test3_cross_fault():
    print("=" * 74)
    print("TEST 3 -- Cross-fault distinguishability at full severity")
    print("=" * 74)
    rows, cols = GRID
    rho = B._rho(rows, cols)
    fields = B.coord_fields(rows, cols)
    rng = np.random.default_rng(555)
    n = T2_N

    c, r0, w, tr = B.sample_params_vec(2 * n, rng)
    tpl = B.templates_vec(c, r0, w, rho)
    grids = B.grids_vec(tpl, tr, rng)
    md = F.fault_modulation("drying", 1.0, rows, cols)
    mf = F.fault_modulation("flooding", 1.0, rows, cols)
    fd = B.battery_vec(grids[:n] * md[None], tpl[:n], fields)
    ff = B.battery_vec(grids[n:] * mf[None], tpl[n:], fields)
    acc = B.best_threshold_accuracy_cols(fd, ff)
    best_k = int(np.argmax(acc))
    print(f"  drying vs flooding, per battery feature:")
    for k, nm in enumerate(B.FEATURE_NAMES):
        print(f"    {nm:>16}  {acc[k]:.3f}")
    # the Sec. 11c expectation is specifically about RADIAL sign
    radial_d = fd[:, 0].mean()
    radial_f = ff[:, 0].mean()
    print(f"  radial-corr means: drying {radial_d:+.3f}, "
          f"flooding {radial_f:+.3f}  (Sec. 11c expects OPPOSITE movement)")
    record("T3  drying separable from flooding at full severity",
           bool(acc.max() >= 0.95),
           f"best={acc.max():.3f} ({B.FEATURE_NAMES[best_k]})")
    return acc.max(), radial_d, radial_f


# ===========================================================================
def test4_invariants():
    print()
    print("=" * 74)
    print("TEST 4 -- Preserved invariants")
    print("=" * 74)
    rows, cols = GRID
    rng = np.random.default_rng(99)
    p = F.sample_healthy_params(rng)

    # CV(J) load-invariance must survive the modulation
    for ft in FAULTS:
        cvs = []
        for load in T4_LOADS:
            g = F.generate_fault_grid_v3(ft, 0.6, p, seed=7,
                                         load_level=load, rows=rows, cols=cols)
            cvs.append(F.compute_cv(g))
        spread = max(cvs) - min(cvs)
        record(f"T4  CV(J) load-invariant  {ft}", bool(spread < 1e-9),
               f"spread={spread:.2e} over loads {T4_LOADS}")

    # episode counts and severity_targets must be bit-identical
    run = F.generate_labeled_run()
    counts = {}
    for ep in run["episodes"]:
        counts[ep["fault_type"]] = counts.get(ep["fault_type"], 0) + 1
    record("T4  episode counts unchanged", counts == T4_EXPECTED_EPISODES,
           f"{counts} == {T4_EXPECTED_EPISODES}")
    tgts = np.array([ep["severity_target"] for ep in run["episodes"]])
    record("T4  severity_targets finite and unchanged in count",
           bool(len(tgts) == sum(T4_EXPECTED_EPISODES.values())
                and np.all(np.isfinite(tgts))),
           f"n={len(tgts)}  sum={tgts.sum():.6f}")
    return run


# ===========================================================================
def test_sec11f_item3(run):
    """Sec. 11f item 3: CV(J) for drying, mean(|J|) for flooding (vacuous
    for flooding on CV(J), per Sec. 2)."""
    print()
    print("=" * 74)
    print("Sec. 11f item 3 -- coherence of the magnitude/dispersion direction")
    print("=" * 74)
    rows, cols = GRID
    rng = np.random.default_rng(4242)
    cv_h, cv_d, mj_h, mj_f, cv_f = [], [], [], [], []
    for _ in range(300):
        p = F.sample_healthy_params(rng)
        sd = int(rng.integers(1 << 62))
        h = F.generate_healthy_grid_v3(p, sd, rows=rows, cols=cols)
        d = F.generate_fault_grid_v3("drying", 0.9, p, sd, rows=rows, cols=cols)
        f = F.generate_fault_grid_v3("flooding", 0.9, p, sd, rows=rows, cols=cols)
        cv_h.append(F.compute_cv(h)); cv_d.append(F.compute_cv(d))
        cv_f.append(F.compute_cv(f))
        mj_h.append(np.abs(h).mean()); mj_f.append(np.abs(f).mean())
    cv_h, cv_d, cv_f = map(np.array, (cv_h, cv_d, cv_f))
    mj_h, mj_f = map(np.array, (mj_h, mj_f))
    record("11f3  drying raises CV(J) vs its own healthy baseline",
           bool((cv_d > cv_h).mean() > 0.99),
           f"CV(J) {cv_h.mean():.4f} -> {cv_d.mean():.4f} "
           f"({100*(cv_d > cv_h).mean():.1f}% of cells)")
    record("11f3  flooding depresses mean(|J|) vs its own healthy baseline",
           bool((mj_f < mj_h).mean() > 0.99),
           f"mean|J| {mj_h.mean():.4f} -> {mj_f.mean():.4f} "
           f"({100*(mj_f < mj_h).mean():.1f}% of cells)")
    print(f"  (for the record, flooding on CV(J): {cv_h.mean():.4f} -> "
          f"{cv_f.mean():.4f} -- reported because Sec. 2 calls this check")
    print("   vacuous for flooding, which is why mean(|J|) substitutes)")


def main():
    test1_continuity()
    test2_separability()
    t3 = test3_cross_fault()
    run = test4_invariants()
    test_sec11f_item3(run)

    print()
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
