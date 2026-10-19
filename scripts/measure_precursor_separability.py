"""
Is there a learnable SPATIAL precursor, or would a predictive model be reading
the load trace?

THE QUESTION (Phase 4 scoping, measurement only -- nothing is changed here).
Phase 4's predictive arm is meant to recognise a fault before severity crosses
CLASSIFICATION_THRESHOLD, from spatial grids inside the precursor window. That
only works if those grids actually differ from healthy. If severity is
effectively zero until onset and then rises sharply, there is no spatial
precursor to learn, and a model scoring well would be predicting from the LOAD
TRACE (ramp rate, dwell duration) rather than from spatial state. The load
trace is bulk-observable, so the spatial arm's apparent advantage would be
spurious in exactly the class Sec. 10 warns about: an artifact that decides the
comparison before it is measured.

TWO DEFINITIONS OF "LEAD TIME", AND WHY BOTH ARE REPORTED

  (a) negative onset_field_s -- "seconds until the next episode begins".
      Measured for completeness, but the answer is ANALYTIC rather than
      empirical: _install_episode writes severity only inside
      [onset_start_s, episode_end_s], so every timestep with
      onset_field_s < 0 has severity EXACTLY 0 and active_fault_type None.
      Verified: 767 such timesteps in a run, severity min = max = 0. Those
      grids ARE healthy grids, drawn from the same distribution, so
      separability there is the null by construction -- it says nothing about
      severity trajectories.

  (b) seconds before the THRESHOLD CROSSING -- the quantity Phase 4 actually
      needs, and the one that matches Kim et al.'s 30 s horizon. Kim predicts
      30 s before a fault is *confirmed*, not 30 s before a precursor starts.
      For each episode, T_cross is the first time its own severity_fn reaches
      CLASSIFICATION_THRESHOLD; lead time L samples at T_cross - L. Where
      T_cross - L falls before onset_start_s, severity is 0 -- which is a
      finding about that fault's onset duration, not an artifact of the
      measurement.

Definition (b) is the headline. Definition (a) is reported so the distinction
is on the record.

METHOD. Reuses the Sec. 11b battery from measure_battery_null (same seven
features, same best-threshold-accuracy metric, N = 1000 per class, 18x18) and
compares against the SAME measured null -- threshold 0.560, null p99 0.543,
null max 0.553 at N = 1000. Using the already-measured null is the point:
"above null" has a calibrated meaning here without needing a fresh reference.

Lead-time selection comes from the generator's own signed onset_field_s and
each episode's own severity_fn. Cell-parameter variation comes from
independent draws to reach N = 1000, matching the null's across-cell design --
a run contains only 67 episodes, far fewer than 1000, so the severity
DISTRIBUTION is taken from the run and the cell draws are independent.
near_seam_buffer timesteps are excluded.

Run:  python scripts/measure_precursor_separability.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fault_generator as F  # noqa: E402
import measure_battery_null as B  # noqa: E402

LEAD_TIMES_S = (1, 2, 5, 10, 20, 30, 45, 60)
N_PER_CLASS = 1000
GRID = F.DEFAULT_GRID_SHAPE
FAULTS = ("starvation", "flooding", "drying")

# The already-measured null at N=1000, 18x18 (Sec. 11b). NOT recomputed here.
NULL_THRESHOLD = 0.560
NULL_P99 = 0.543
NULL_MAX = 0.553


def _cases():
    for ft in FAULTS:
        if ft == "starvation":
            for arm in F.STARVATION_ARMS:
                yield ft, arm
        else:
            yield ft, None


def severity_at_lead(run, lead_s):
    """
    Definition (b): severity at T_cross - lead_s, per episode, grouped by
    fault type. Returns {fault: [severities]} plus bookkeeping.
    """
    t = run["t_seconds"]
    nsb = run["near_seam_buffer"]
    out = {ft: [] for ft in FAULTS}
    skipped_seam = 0
    before_onset = {ft: 0 for ft in FAULTS}
    for ep in run["episodes"]:
        pw = ep["precursor_window_s"]
        if pw is None:                      # never crosses the threshold
            continue
        t_cross = ep["onset_start_s"] + pw
        t_sample = t_cross - lead_s
        i = int(np.clip(np.searchsorted(t, t_sample), 0, len(t) - 1))
        if nsb[i]:
            skipped_seam += 1
            continue
        if t_sample < ep["onset_start_s"]:
            # before the episode exists at all -> severity is exactly 0
            out[ep["fault_type"]].append(0.0)
            before_onset[ep["fault_type"]] += 1
        else:
            s = float(ep["severity_fn"](np.array([t_sample]))[0])
            out[ep["fault_type"]].append(s)
    return out, skipped_seam, before_onset


def separability(fault, arm, severities, rng, rho, fields, rows, cols):
    """Healthy vs fault at this lead time's severity distribution, N per class."""
    if not severities:
        return None
    sev = np.asarray(severities, dtype=float)
    draw = sev[rng.integers(0, len(sev), N_PER_CLASS)]

    c, r0, w, tr = B.sample_params_vec(2 * N_PER_CLASS, rng)
    axes = rng.uniform(0.0, 2.0 * np.pi, 2 * N_PER_CLASS)
    tpl = B.templates_vec(c, r0, w, rho)
    grids = B.grids_vec(tpl, tr, rng)

    fg = np.empty_like(grids[N_PER_CLASS:])
    for i in range(N_PER_CLASS):
        j = N_PER_CLASS + i
        p = {"c": c[j], "r0": r0[j], "w": w[j], "temporal_ratio": tr[j],
             "starvation_axis_rad": axes[j]}
        fg[i] = grids[j] * F.fault_modulation(fault, draw[i], rows, cols, p, arm)

    h = B.battery_vec(grids[:N_PER_CLASS], tpl[:N_PER_CLASS], fields)
    f = B.battery_vec(fg, tpl[N_PER_CLASS:], fields)
    acc = B.best_threshold_accuracy_cols(h, f)
    k = int(np.argmax(acc))
    return float(acc.max()), B.FEATURE_NAMES[k], float(sev.mean()), float(sev.max())


def main():
    rows, cols = GRID
    rho = B._rho(rows, cols)
    fields = B.coord_fields(rows, cols)

    print("Precursor separability vs lead time")
    print(f"grid={rows}x{cols}  N={N_PER_CLASS}/class  "
          f"battery={len(B.FEATURE_NAMES)} features")
    print(f"measured null at N=1000 (Sec. 11b): threshold {NULL_THRESHOLD}, "
          f"p99 {NULL_P99}, max {NULL_MAX}")
    print()

    runs = {arm: F.generate_labeled_run(starvation_arm=arm)
            for arm in F.STARVATION_ARMS}
    run0 = runs[F.STARVATION_ARMS[0]]

    # ---- definition (a), the analytic check -------------------------------
    of = run0["onset_field_s"]
    sv = run0["severity"]
    neg = np.isfinite(of) & (of < 0)
    print("=" * 76)
    print("(a) negative onset_field_s -- 'seconds until the episode begins'")
    print("=" * 76)
    print(f"  {neg.sum()} timesteps; severity min {sv[neg].min():.6g} "
          f"max {sv[neg].max():.6g}")
    if sv[neg].max() == 0.0:
        print("  severity is EXACTLY 0 throughout, and active_fault_type is None.")
        print("  These grids are healthy grids from the same distribution, so")
        print("  separability at every negative lead time is the null BY")
        print("  CONSTRUCTION (~0.527 median, <= 0.553 observed max at N=1000).")
        print("  Not measured per lead time: it would restate the null, and it")
        print("  says nothing about severity trajectories.")
    print()

    # ---- precursor windows, which bound what (b) can show ----------------
    print("=" * 76)
    print("Precursor windows (onset_start -> severity crosses 0.5)")
    print("=" * 76)
    for ft in FAULTS:
        pw = [e["precursor_window_s"] for e in run0["episodes"]
              if e["fault_type"] == ft]
        nn = [p for p in pw if p is not None]
        s = (f"min {min(nn):5.1f}s  median {np.median(nn):5.1f}s  "
             f"max {max(nn):5.1f}s" if nn else "")
        print(f"  {ft:>10}: n={len(pw):3d}  crossed: {len(nn):3d}  "
              f"never: {len(pw)-len(nn):2d}   {s}")
    print()
    print("  Any lead time exceeding a fault's precursor window necessarily")
    print("  samples BEFORE its episode starts, where severity is exactly 0.")
    print()

    # ---- definition (b), the headline ------------------------------------
    print("=" * 76)
    print("(b) seconds before the THRESHOLD CROSSING  -- Kim et al. compares")
    print("    at 30 s (horizon) and 45 s (lookback), both in the sweep")
    print("=" * 76)
    rng = np.random.default_rng(20261013)
    for ft, arm in _cases():
        label = ft if arm is None else f"{ft}[{arm}]"
        print(f"  --- {label} ---")
        hdr = (f"    {'lead':>5} {'mean sev':>9} {'max sev':>8} "
               f"{'separability':>13} {'driving feature':>16}  verdict")
        print(hdr)
        run = runs[arm] if arm else run0
        for L in LEAD_TIMES_S:
            sev_map, seam, before = severity_at_lead(run, L)
            res = separability(ft, arm, sev_map[ft], rng, rho, fields, rows, cols)
            if res is None:
                print(f"    {L:>5} {'(no episodes)':>9}")
                continue
            acc, feat, msev, xsev = res
            if acc <= NULL_MAX:
                verdict = "AT NULL"
            elif acc <= NULL_THRESHOLD:
                verdict = "within null band"
            else:
                verdict = "ABOVE NULL"
            note = ""
            if before[ft]:
                note = f"  [{before[ft]} of {len(sev_map[ft])} eps pre-onset]"
            print(f"    {L:>5} {msev:>9.4f} {xsev:>8.4f} {acc:>13.3f} "
                  f"{feat:>16}  {verdict}{note}")
        print()


if __name__ == "__main__":
    main()
