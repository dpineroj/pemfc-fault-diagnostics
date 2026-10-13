"""
Healthy-vs-healthy NULL for the Sec. 11b acceptance battery.

Purpose (Sec. 11b): the severity-resolved separability test says "no feature
in the battery may exceed THRESHOLD at severity <= 0.05". Chance is 0.500,
but best-single-threshold accuracy on FINITE samples is upward-biased even
when the two classes are drawn from an identical distribution. Setting the
threshold from intuition therefore risks either permitting real separation
(too high) or failing a correct generator (too low).

This script measures the empirical floor: draw grids from the healthy family
ALONE -- no fault, no modulation -- split them into two arbitrary groups, and
run the same battery. Whatever the battery scores there is attributable to
finite samples and nothing else.

ANALYTIC NOTE, which makes the null principled rather than merely observed.
For two equal-size samples A, B and a threshold rule, accuracy at threshold t
is [F_B(t) + 1 - F_A(t)] / 2, so maximizing over t and over both orientations
gives

    best-threshold accuracy  ==  0.5 + D / 2

where D is the two-sample Kolmogorov-Smirnov statistic. The null distribution
of the PER-FEATURE score is therefore the KS null, which is distribution-free:
it depends ONLY on the sample size N -- not on the feature, and not on the
parameter spread, because under the null both groups share one distribution.
The large-N p95 is D ~= 1.358 * sqrt(2/N), i.e.

    N=50 -> 0.636    N=100 -> 0.596    N=200 -> 0.568    N=500 -> 0.543

The script checks the measurement against this, and reports the battery-MAX
null (the quantity Test 2 actually constrains), which sits ABOVE the
per-feature null because it maximizes over 7 correlated features.

CONSEQUENCE: the threshold cannot be fixed without also fixing N. Both are
reported so both can be recorded.

PERFORMANCE: the template and noise construction are vectorized over samples
rather than calling the generator once per grid, which the null's sample
counts make prohibitive. `_assert_matches_generator` checks the vectorized
path against `fault_generator` exactly -- identical templates, identical
battery values, and identical grids for a per-seed draw -- so the fast path
cannot silently diverge from the generator it is meant to characterize.

Run:  python scripts/measure_battery_null.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fault_generator import (  # noqa: E402
    DEFAULT_GRID_SHAPE,
    HEALTHY_C_RANGE,
    HEALTHY_R0_RANGE,
    HEALTHY_W_RANGE,
    HEALTHY_TEMPORAL_RATIO_RANGE,
    logistic_healthy_shape,
    generate_healthy_grid_v3,
)

N_TRIALS = 4000
N_PER_CLASS = (25, 50, 100, 200, 500, 1000)
GRID = DEFAULT_GRID_SHAPE
LOAD_LEVEL = 0.697
SEED = 20261012
EXCEEDANCE_SEED = 777  # the stream the Sec. 11b false-failure table came from


# ------------------------------------------------------- vectorized generator
def _rho(rows, cols):
    row_idx, col_idx = np.indices((rows, cols))
    r = np.sqrt((row_idx - (rows - 1) / 2.0) ** 2
                + (col_idx - (cols - 1) / 2.0) ** 2)
    return r / r.max()


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def templates_vec(c, r0, w, rho):
    """
    n healthy templates at once. Mirrors logistic_healthy_shape exactly; see
    _assert_matches_generator.
    """
    c = np.asarray(c, float).reshape(-1, 1, 1)
    r0 = np.asarray(r0, float).reshape(-1, 1, 1)
    w = np.maximum(np.asarray(w, float).reshape(-1, 1, 1), 1e-4)
    lo = _logistic(-r0 / w)
    hi = _logistic((1.0 - r0) / w)
    g = (_logistic((rho[None] - r0) / w) - lo) / (hi - lo)
    raw = c + (1.0 - c) * g
    return raw / raw.mean(axis=(1, 2), keepdims=True)


def grids_vec(templates, temporal_ratio, rng, load_level=LOAD_LEVEL):
    """n healthy timesteps, one per template. Mirrors generate_healthy_grid_v3."""
    sigma = (templates.std(axis=(1, 2), keepdims=True)
             / np.asarray(temporal_ratio, float).reshape(-1, 1, 1))
    noise = rng.normal(0.0, 1.0, templates.shape) * sigma
    return load_level * templates * (1.0 + noise)


def sample_params_vec(n, rng):
    return (rng.uniform(*HEALTHY_C_RANGE, n),
            rng.uniform(*HEALTHY_R0_RANGE, n),
            rng.uniform(*HEALTHY_W_RANGE, n),
            rng.uniform(*HEALTHY_TEMPORAL_RATIO_RANGE, n))


# ---------------------------------------------------------------- the battery
FEATURE_NAMES = ("radial", "row", "col", "diag_sum", "diag_diff",
                 "lag1_autocorr", "template_match")


def coord_fields(rows, cols):
    """The fixed geometric regressors the battery correlates grids against."""
    row_idx, col_idx = np.indices((rows, cols))
    r = np.sqrt((row_idx - (rows - 1) / 2.0) ** 2
                + (col_idx - (cols - 1) / 2.0) ** 2)
    return np.stack([r,
                     row_idx.astype(float),
                     col_idx.astype(float),
                     (row_idx + col_idx).astype(float),
                     (row_idx - col_idx).astype(float)]).reshape(5, -1)


def _corr_rows(A, B):
    """Row-wise Pearson correlation of (n,M) against (n,M) or (M,)."""
    A = A - A.mean(axis=-1, keepdims=True)
    B = B - B.mean(axis=-1, keepdims=True)
    num = (A * B).sum(axis=-1)
    den = (np.sqrt((A * A).sum(axis=-1)) * np.sqrt((B * B).sum(axis=-1)))
    return np.where(den < 1e-12, 0.0, num / np.maximum(den, 1e-300))


def battery_vec(grids, templates, fields):
    """
    The 7 scalars of the Sec. 11b battery, for grids shaped (n, rows, cols).

    Deliberately excludes mean(J) and CV(J): genuine fault physics should move
    those, so constraining them would wrongly demand faults be undetectable.
    """
    n = grids.shape[0]
    flat = grids.reshape(n, -1)
    out = np.empty((n, len(FEATURE_NAMES)))
    for k in range(fields.shape[0]):
        out[:, k] = _corr_rows(flat, fields[k][None, :])
    h = _corr_rows(grids[:, :, :-1].reshape(n, -1),
                   grids[:, :, 1:].reshape(n, -1))
    v = _corr_rows(grids[:, :-1, :].reshape(n, -1),
                   grids[:, 1:, :].reshape(n, -1))
    out[:, 5] = 0.5 * (h + v)
    out[:, 6] = _corr_rows(flat, templates.reshape(n, -1))
    return out


# ------------------------------------------------------------- separability
def best_threshold_accuracy_cols(A, B):
    """
    Best single-threshold accuracy separating A from B, column-wise.
    A, B are (N, K). Returns (K,). Equals 0.5 + D/2 for the two-sample KS
    statistic D (see module docstring).
    """
    na, nb = A.shape[0], B.shape[0]
    xs = np.concatenate([A, B], axis=0)
    is_b = np.concatenate([np.zeros(na), np.ones(nb)])
    order = np.argsort(xs, axis=0, kind="mergesort")
    lab = is_b[order]
    cb = np.cumsum(lab, axis=0)
    ca = np.cumsum(1.0 - lab, axis=0)
    acc1 = (ca / na + (nb - cb) / nb) / 2.0
    acc2 = (cb / nb + (na - ca) / na) / 2.0
    return np.maximum(np.maximum(acc1.max(axis=0), acc2.max(axis=0)), 0.5)


def ks_two_sample(a, b):
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    xs = np.concatenate([a, b])
    fa = np.searchsorted(a, xs, side="right") / len(a)
    fb = np.searchsorted(b, xs, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


# -------------------------------------------------------------- null sampling
def run_null(design, n_per_class, n_trials, rng, fields, rho):
    """
    design "within": one cell per trial, 2N timesteps of it -- the structure a
      real run has, since generate_labeled_run draws healthy params once per
      run, so healthy and fault timesteps within a run share one template.
    design "across": a fresh cell (c,r0,w) for every sample.
    Returns per_feat (n_trials, n_feat).
    """
    n2 = 2 * n_per_class
    per_feat = np.empty((n_trials, len(FEATURE_NAMES)))
    for t in range(n_trials):
        if design == "across":
            c, r0, w, tr = sample_params_vec(n2, rng)
        else:
            c1, r01, w1, tr1 = sample_params_vec(1, rng)
            c, r0, w, tr = (np.repeat(c1, n2), np.repeat(r01, n2),
                            np.repeat(w1, n2), np.repeat(tr1, n2))
        tpl = templates_vec(c, r0, w, rho)
        grids = grids_vec(tpl, tr, rng)
        feats = battery_vec(grids, tpl, fields)
        per_feat[t] = best_threshold_accuracy_cols(feats[:n_per_class],
                                                   feats[n_per_class:])
    return per_feat


# ------------------------------------------------------------------ integrity
def _assert_matches_generator(rho, fields, rows, cols):
    """The vectorized path must not diverge from fault_generator."""
    rng = np.random.default_rng(7)
    c, r0, w, tr = sample_params_vec(12, rng)

    tv = templates_vec(c, r0, w, rho)
    for i in range(12):
        ref = logistic_healthy_shape(float(c[i]), float(r0[i]), float(w[i]),
                                     rows, cols)
        assert np.allclose(tv[i], ref, rtol=0, atol=1e-12), f"template {i}"

    # grids: exact match when the noise is drawn per-seed the way the
    # generator draws it, which pins the noise CONSTRUCTION, not just the
    # template.
    for i in range(12):
        p = {"c": float(c[i]), "r0": float(r0[i]), "w": float(w[i]),
             "temporal_ratio": float(tr[i])}
        seed = 1000 + i
        ref = generate_healthy_grid_v3(p, seed, rows=rows, cols=cols)
        S = tv[i]
        sigma = S.std() / p["temporal_ratio"]
        mine = LOAD_LEVEL * S * (
            1.0 + np.random.default_rng(seed).normal(0, sigma, S.shape))
        assert np.allclose(mine, ref, rtol=0, atol=1e-12), f"grid {i}"

    # battery: vectorized == scalar reference, on generator-produced grids
    gs = np.stack([generate_healthy_grid_v3(
        {"c": float(c[i]), "r0": float(r0[i]), "w": float(w[i]),
         "temporal_ratio": float(tr[i])}, 2000 + i, rows=rows, cols=cols)
        for i in range(12)])
    bv = battery_vec(gs, tv, fields)
    for i in range(12):
        g = gs[i]
        ref = [float(np.corrcoef(g.ravel(), fields[k])[0, 1]) for k in range(5)]
        ref.append(0.5 * (
            float(np.corrcoef(g[:, :-1].ravel(), g[:, 1:].ravel())[0, 1])
            + float(np.corrcoef(g[:-1, :].ravel(), g[1:, :].ravel())[0, 1])))
        ref.append(float(np.corrcoef(g.ravel(), tv[i].ravel())[0, 1]))
        assert np.allclose(bv[i], ref, rtol=0, atol=1e-10), f"battery {i}"

    # the KS identity the thresholds are reasoned from
    a, b = rng.normal(size=300), rng.normal(size=300)
    lhs = float(best_threshold_accuracy_cols(a.reshape(-1, 1), b.reshape(-1, 1))[0])
    rhs = 0.5 + ks_two_sample(a, b) / 2.0
    assert abs(lhs - rhs) < 1e-12, f"KS identity {lhs} vs {rhs}"
    print("integrity: templates, grids, battery and the KS identity all match"
          " the generator / analytic reference")


def main():
    rows, cols = GRID
    rho = _rho(rows, cols)
    fields = coord_fields(rows, cols)

    print("Healthy-vs-healthy NULL for the Sec. 11b battery")
    print(f"grid={rows}x{cols}  trials={N_TRIALS}  seed={SEED}")
    print(f"features ({len(FEATURE_NAMES)}): {', '.join(FEATURE_NAMES)}")
    print()
    _assert_matches_generator(rho, fields, rows, cols)
    print()

    rng = np.random.default_rng(SEED)
    designs = (
        ("within", "WITHIN-CELL (shared c,r0,w -- the structure a real run has)"),
        ("across", "ACROSS-CELL (fresh c,r0,w per sample)"),
    )
    store = {}
    for design, label in designs:
        print("=" * 76)
        print(label)
        print("=" * 76)
        hdr = (f"{'N/class':>8} {'median':>8} {'p95':>8} {'p99':>8} {'max':>8}"
               f" | {'per-feat p95':>12} {'KS p95 pred':>12}")
        print(hdr)
        print("-" * len(hdr))
        for n in N_PER_CLASS:
            trials = N_TRIALS if n <= 200 else max(600, N_TRIALS // 4)
            per_feat = run_null(design, n, trials, rng, fields, rho)
            bm = per_feat.max(axis=1)
            ks_pred = 0.5 + (1.358 * np.sqrt(2.0 / n)) / 2.0
            store[(design, n)] = per_feat
            print(f"{n:>8} {np.median(bm):>8.3f} {np.percentile(bm, 95):>8.3f}"
                  f" {np.percentile(bm, 99):>8.3f} {bm.max():>8.3f}"
                  f" | {np.percentile(per_feat, 95):>12.3f} {ks_pred:>12.3f}")
        print()
        pf = store[(design, 100)]
        am = pf.argmax(axis=1)
        print("  at N=100 -- per-feature p95, and share of the battery max:")
        for k, nm in enumerate(FEATURE_NAMES):
            print(f"    {nm:>16}  p95={np.percentile(pf[:, k], 95):.3f}"
                  f"   drives max {100.0 * (am == k).mean():5.1f}%")
        print()

    # Exceedance rates: the false-failure rate a candidate threshold would
    # carry if the generator were perfectly correct. This is the quantity the
    # adopted (N, threshold) pair is justified by, so it is reported here
    # rather than computed ad-hoc.
    print("=" * 76)
    print("FALSE-FAILURE RATE -- P(battery-max > t) under the null")
    print("=" * 76)
    cands = (0.54, 0.56, 0.58, 0.60)
    # Own seeded stream, so this table reproduces independently of how many
    # draws the percentile loops above consumed. The figures quoted in
    # Sec. 11b come from exactly this block.
    exc_rng = np.random.default_rng(EXCEEDANCE_SEED)
    hdr = f"{'N/class':>8}" + "".join(f"{t:>10.2f}" for t in cands)
    print(hdr)
    print("-" * len(hdr))
    for n in (100, 200, 500, 1000):
        bm = run_null("across", n, 3000, exc_rng, fields, rho).max(axis=1)
        print(f"{n:>8}" + "".join(f"{(bm > t).mean():>10.4f}" for t in cands))
    print()
    print("  ADOPTED: N=1000, threshold 0.560 -- see Sec. 11b.")


if __name__ == "__main__":
    main()
