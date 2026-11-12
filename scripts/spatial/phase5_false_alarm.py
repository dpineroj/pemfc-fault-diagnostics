"""
Phase 5 -- the no-false-alarm test. The only point in the project where a
trained model meets measured data.

Criterion, transform and the consequences of both were pre-registered in
docs/spatial/synthetic_generator_notes.md Sec. 14 (commit fc3fb28), which
predates this file:

    FAR <= 5%            PASS  (quiet)
    5% < FAR <= 50%      FAIL  (not quiet)
    FAR > 50%            FAIL, simulator-learned
    an arm passes only if it passes on ALL FOUR real configurations

The real grids contain NO faults, so this does not test detection. It tests
whether models trained on synthetic faults stay quiet on a genuinely healthy
cell.

INPUT TRANSFORM (Sec. 14b): the generator expresses grids in load-normalised
units (healthy mean 0.697 at every load), so each real grid is divided by the
bulk-measured current density at that instant (INTENSIDAD / active area) and
multiplied by 0.697. load_ratio is INTENSIDAD min-max normalised per run, as
in generation.

INDEPENDENCE (Sec. 14c) is checked, not asserted in prose: every final model
is trained BEFORE the real-data loader is imported, and the script asserts
`cdm_parser` is absent from sys.modules at that point. What remains true and
is restated with the result: INTENSIDAD drove generation, and the healthy
family's FORM was chosen with the real template in view (Sec. 6f).

Run:  python scripts/spatial/phase5_false_alarm.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fault_generator as F  # noqa: E402
import model_b_data as D  # noqa: E402
import phase4_model_a as MA  # noqa: E402
import phase4_model_b as MB  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT_DIR = os.path.join(D.REPO, "run_logs", "phase5")
SUMMARY = os.path.join(D.REPO, "run_logs", "phase5_summary.json")
REAL_CONFIGS = ("Normal_Flow", "Inverse_Flow", "Inverse_Air_Flow",
                "Inverse_Hydrogen_Flow")
PASS_FAR = 0.05          # pre-registered
SIM_LEARNED_FAR = 0.50   # pre-registered
MAIN_CFGS = ("starvation", "flooding", "drying")
NONCONVERGED = []


def _fit(model, X, y, tag, max_iter):
    sc = StandardScaler().fit(X)
    m = model.fit(sc.transform(X), y)
    ni = int(np.max(np.atleast_1d(m.n_iter_)))
    if ni >= max_iter:
        NONCONVERGED.append((tag, X.shape[1], ni))
    print(f"  fit {tag:<44} n={len(y):>6,} p={X.shape[1]:>3} iters={ni}",
          flush=True)
    return {"mean": sc.mean_, "scale": sc.scale_, "coef": m.coef_,
            "intercept": m.intercept_, "classes": np.array(m.classes_, "U12"),
            "n_iter": ni}


def _predict(mod, X):
    Z = (X - mod["mean"]) / mod["scale"]
    s = Z @ mod["coef"].T + mod["intercept"]
    if s.shape[1] == 1:                       # binary
        return np.where(s[:, 0] > 0, mod["classes"][1], mod["classes"][0])
    return mod["classes"][np.argmax(s, axis=1)]


def _cached(name, builder):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, name + ".npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=False)
        print(f"  [resume] {name}", flush=True)
        return {k: z[k] for k in z.files}
    mod = builder()
    tmp = p.replace(".npz", ".tmp.npz")
    np.savez(tmp, **mod)
    os.replace(tmp, p)
    return mod


def train_final_models():
    """Every final model, trained on ALL synthetic groups. No real data."""
    rows, cols = F.DEFAULT_GRID_SHAPE
    models = {}
    label_cache = {}
    for arm in F.STARVATION_ARMS:
        def build_a(which, arm=arm):
            Xs, Xb, ys = [], [], []
            for tk in MA.TRACES:
                for ci in range(MA.CELLS_PER_TRACE):
                    sp, bu, lab, _s, _g = MA.build_group(
                        tk, ci, arm, 1.0, rows, cols, label_cache)
                    Xs.append(sp); Xb.append(bu); ys.append(lab)
            X = np.vstack(Xs if which == "spatial" else Xb)
            return _fit(MA.make_model("logreg"), X, np.concatenate(ys),
                        f"ModelA/{arm}/{which}", 5000)
        for which in ("spatial", "bulk"):
            models[("A", "snapshot", arm, which)] = _cached(
                f"modelA__{arm}__{which}", lambda w=which: build_a(w))
        for cfg in MAIN_CFGS:
            def build_b(which, arm=arm, cfg=cfg):
                ds = MB.assemble(cfg, arm)
                X = ds["Xs"] if which == "spatial" else ds["Xb"]
                return _fit(MB.make_model(), X, ds["y"],
                            f"ModelB/{cfg}/{arm}/{which}", MB.MAX_ITER)
            for which in ("spatial", "bulk"):
                models[("B", cfg, arm, which)] = _cached(
                    f"modelB__{cfg}__{arm}__{which}",
                    lambda w=which: build_b(w))
    return models


def sustained(alarm, k=3):
    """Fraction of samples lying inside a run of >= k consecutive alarms."""
    a = alarm.astype(int)
    if a.size == 0:
        return float("nan")
    out = np.zeros_like(a)
    run = 0
    for i, v in enumerate(a):
        run = run + 1 if v else 0
        if run >= k:
            out[i - k + 1:i + 1] = 1
    return float(out.mean())


def main():
    print("Phase 5 -- no-false-alarm test on real healthy data")
    print(f"pre-registered: PASS iff FAR <= {PASS_FAR:.0%} on all four real "
          f"configurations; > {SIM_LEARNED_FAR:.0%} = simulator-learned")
    print()
    D.assert_generator_agreement()

    print("training final models on ALL synthetic groups ...", flush=True)
    models = train_final_models()

    # ---- Sec. 14c independence: checked, not asserted in prose -------------
    assert "cdm_parser" not in sys.modules, (
        "the real-data loader was imported before training finished")
    print()
    print("INDEPENDENCE CHECK (Sec. 14c): every final model above was trained")
    print("  with the real-data loader NOT imported (cdm_parser absent from")
    print("  sys.modules). No real spatial grid entered training.")
    print("  Still true, and restated at the result: INTENSIDAD drove")
    print("  generation, and the healthy FORM was chosen with the real")
    print("  template in view (Sec. 6f).")
    print()
    # convergence read from what is STORED with each model, so a resumed run
    # reports it for models fit by an earlier process too
    NONCONVERGED[:] = [
        (k, int(m["n_iter"])) for k, m in models.items()
        if int(m["n_iter"]) >= (5000 if k[0] == "A" else MB.MAX_ITER)]
    print("CONVERGENCE of final models (reported before any number): "
          + (f"all {len(models)} converged" if not NONCONVERGED
             else f"NON-CONVERGED, results below NOT reportable: {NONCONVERGED}"))
    print()

    from cdm_parser import TOTAL_ACTIVE_AREA_CM2, load_run  # noqa: E402
    import extract_load_trace as E  # noqa: E402

    results = {}
    for cfg in REAL_CONFIGS:
        run = load_run(cfg)
        J = np.abs(run["current_grid"]).astype(np.float64)
        I = run["bulk"]["INTENSIDAD"].to_numpy().astype(np.float64)
        ts = run["timestamp"]
        t = np.asarray((ts - ts[0]) / np.timedelta64(1, "s"), dtype=float)
        n_raw = len(I)
        good = np.isfinite(I) & (I > 0) & np.isfinite(J).all(axis=(1, 2))
        i_min, i_max = np.nanmin(I), np.nanmax(I)
        load_ratio = (I - i_min) / (i_max - i_min)
        dens = I / TOTAL_ACTIVE_AREA_CM2
        with np.errstate(divide="ignore", invalid="ignore"):
            Jn = MA.LOAD_LEVEL * J / dens[:, None, None]
        flat = Jn.reshape(n_raw, -1)
        mj = flat.mean(axis=1)
        bulk_ch = np.c_[mj, load_ratio]
        sp_ch = np.c_[flat, bulk_ch]

        holds = E.detect_holds(t[good], load_ratio[good])
        in_hold_good = np.zeros(int(good.sum()), dtype=bool)
        for hseg in holds:
            in_hold_good[hseg["start_idx"]:hseg["end_idx"] + 1] = True
        in_hold = np.zeros(n_raw, dtype=bool)
        in_hold[np.flatnonzero(good)] = in_hold_good

        print("=" * 78)
        print(f"REAL CONFIGURATION: {cfg}   T={n_raw:,}  usable={int(good.sum()):,}"
              f"  dropped (non-finite/zero INTENSIDAD)={int((~good).sum())}")
        print(f"  transformed mean(J): {np.nanmean(mj[good]):.4f} +- "
              f"{np.nanstd(mj[good]):.4f}   (synthetic healthy = 0.697)")
        print(f"  median dt = {np.median(np.diff(t)):.2f} s;  hold share "
              f"{in_hold[good].mean():.1%}")
        print("=" * 78)
        print(f"  {'model':<22}{'starv-arm':<13}{'input':<9}{'n':>7}{'FAR':>8}"
              f"{'hold':>8}{'dynamic':>9}{'sustain3':>9}  verdict / alarms name")

        for key, mod in models.items():
            kind, mcfg, arm, which = key
            if kind == "A":
                X = (sp_ch if which == "spatial" else bulk_ch)[good]
                hold_mask = in_hold[good]
            else:
                L = D.CONFIGS[mcfg]["L"]
                ch = sp_ch if which == "spatial" else bulk_ch
                chf = np.where(good[:, None], ch, np.nan)
                wf = D.window_features(chf.astype(np.float32), L)
                idx = np.arange(n_raw)
                lo = np.clip(idx - L + 1, 0, n_raw - 1)
                span_ok = (idx >= L - 1) & ((t - t[lo]) <= (L - 1) + D.MAX_GAP_S)
                ok = span_ok & np.isfinite(wf).all(axis=1)
                X = wf[ok].astype(np.float64)
                hold_mask = in_hold[ok]
            pred = _predict(mod, X)
            quiet = "healthy" if kind == "A" else "none"
            alarm = pred != quiet
            far = float(alarm.mean())
            names = {c: float((pred == c).mean())
                     for c in np.unique(pred) if c != quiet}
            v = ("PASS" if far <= PASS_FAR else
                 "FAIL simulator-learned" if far > SIM_LEARNED_FAR else "FAIL")
            far_h = float(alarm[hold_mask].mean()) if hold_mask.any() else float("nan")
            far_d = float(alarm[~hold_mask].mean()) if (~hold_mask).any() else float("nan")
            label = "A snapshot" if kind == "A" else f"B {mcfg}"
            print(f"  {label:<22}{arm:<13}{which:<9}{len(pred):>7,}{far:>8.3f}"
                  f"{far_h:>8.3f}{far_d:>9.3f}{sustained(alarm):>9.3f}  {v}"
                  + ("  " + ", ".join(f"{k} {p:.3f}" for k, p in names.items())
                     if names else ""))
            results[f"{cfg}|{kind}|{mcfg}|{arm}|{which}"] = {
                "n": int(len(pred)), "far": far, "far_hold": far_h,
                "far_dynamic": far_d, "sustained3": sustained(alarm),
                "alarm_classes": names, "verdict": v}
        print()

    print("=" * 78)
    print("PHASE 5 VERDICT PER MODEL AND INPUT ARM (must pass on ALL FOUR)")
    print("=" * 78)
    overall = {}
    for key in models:
        kind, mcfg, arm, which = key
        fars = [results[f"{c}|{kind}|{mcfg}|{arm}|{which}"]["far"]
                for c in REAL_CONFIGS]
        passed = all(f <= PASS_FAR for f in fars)
        label = "A snapshot" if kind == "A" else f"B {mcfg}"
        print(f"  {label:<22}{arm:<13}{which:<9} FAR per config "
              + " ".join(f"{f:.3f}" for f in fars)
              + f"   -> {'PASS' if passed else 'FAIL'}")
        overall[f"{kind}|{mcfg}|{arm}|{which}"] = {"far": fars, "pass": passed}
    with open(SUMMARY, "w", encoding="utf-8") as fh:
        json.dump({"nonconverged": NONCONVERGED, "per_config": results,
                   "overall": overall}, fh, indent=1)
    print(f"\nsummary written to {SUMMARY}")
    print("CLAIM AVAILABLE (Sec. 6f / 14c): quietness on unseen SPATIAL")
    print("realisations of operating conditions the models have seen -- not on")
    print("unseen operating conditions.")


if __name__ == "__main__":
    main()
