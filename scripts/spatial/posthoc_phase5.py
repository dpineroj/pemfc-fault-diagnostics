"""
POST-HOC (Sec. 17e). Phase 5 follow-up on the real healthy recording.

  --mode split   3a: the 16 STORED Phase 5 models (run_logs/phase5/, read
                 only), false-alarm rate on conditioning holds and on dynamic
                 segments separately. No refitting. Descriptive; Sec. 16b's
                 verdicts stand.
  --mode v2      3c: the 16 final models RETRAINED on v2 (v1 + fault-free
                 companions), same estimators and settings, then the same
                 transform and the same registered criterion.

EVERY NUMBER FROM --mode v2 IS **NOT AN INDEPENDENT TEST**: the companion
change was designed after this recording's failures were inspected (Sec. 16d).

The real-data evaluation below is the Phase 5 loop (phase5_false_alarm.py),
with per-segment-type rates added to the output. The transform is unchanged.

Run:  python -u -W ignore scripts/spatial/posthoc_phase5.py --mode split
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fault_generator as F  # noqa: E402
import model_b_data as D  # noqa: E402
import phase4_model_a as MA  # noqa: E402
import phase4_model_b as MB  # noqa: E402
import phase5_false_alarm as P5  # noqa: E402
import posthoc_lib as PL  # noqa: E402


def stored_models():
    models = {}
    for arm in F.STARVATION_ARMS:
        for which in ("spatial", "bulk"):
            z = np.load(os.path.join(P5.OUT_DIR, f"modelA__{arm}__{which}.npz"))
            models[("A", "snapshot", arm, which)] = {k: z[k] for k in z.files}
        for cfg in P5.MAIN_CFGS:
            for which in ("spatial", "bulk"):
                z = np.load(os.path.join(
                    P5.OUT_DIR, f"modelB__{cfg}__{arm}__{which}.npz"))
                models[("B", cfg, arm, which)] = {k: z[k] for k in z.files}
    return models


def train_v2():
    out = os.path.join(PL.OUT, "phase5_v2_models")
    os.makedirs(out, exist_ok=True)
    models = {}

    def cached(name, builder):
        p = os.path.join(out, name + ".npz")
        if os.path.exists(p):
            z = np.load(p)
            print(f"  [on disk] {name}", flush=True)
            return {k: z[k] for k in z.files}
        mod = builder()
        PL.save_npz(p, **mod)
        return mod

    for arm in F.STARVATION_ARMS:
        def build_a(which, arm=arm):
            ds = PL.assemble_a(arm, 1.0, cap=None, companions=True)
            X = ds["Xs"] if which == "spatial" else ds["Xb"]
            return P5._fit(MA.make_model("logreg"), X, ds["y"],
                           f"v2 ModelA/{arm}/{which}", 5000)
        for which in ("spatial", "bulk"):
            models[("A", "snapshot", arm, which)] = cached(
                f"modelA__{arm}__{which}", lambda w=which: build_a(w))
        for cfg in P5.MAIN_CFGS:
            def build_b(which, arm=arm, cfg=cfg):
                ds = PL.assemble_b(cfg, arm, companions=True)
                X = ds["Xs"] if which == "spatial" else ds["Xb"]
                return P5._fit(MB.make_model(), X, ds["y"],
                               f"v2 ModelB/{cfg}/{arm}/{which}", MB.MAX_ITER)
            for which in ("spatial", "bulk"):
                models[("B", cfg, arm, which)] = cached(
                    f"modelB__{cfg}__{arm}__{which}",
                    lambda w=which: build_b(w))
    return models


def evaluate_real(models):
    from cdm_parser import TOTAL_ACTIVE_AREA_CM2, load_run
    import extract_load_trace as E
    res = {}
    for cfg in P5.REAL_CONFIGS:
        run = load_run(cfg)
        J = np.abs(run["current_grid"]).astype(np.float64)
        I = run["bulk"]["INTENSIDAD"].to_numpy().astype(np.float64)
        ts = run["timestamp"]
        t = np.asarray((ts - ts[0]) / np.timedelta64(1, "s"), dtype=float)
        n_raw = len(I)
        good = np.isfinite(I) & (I > 0) & np.isfinite(J).all(axis=(1, 2))
        load_ratio = (I - np.nanmin(I)) / (np.nanmax(I) - np.nanmin(I))
        with np.errstate(divide="ignore", invalid="ignore"):
            Jn = MA.LOAD_LEVEL * J / (I / TOTAL_ACTIVE_AREA_CM2)[:, None, None]
        flat = Jn.reshape(n_raw, -1)
        bulk_ch = np.c_[flat.mean(axis=1), load_ratio]
        sp_ch = np.c_[flat, bulk_ch]
        in_hold_good = np.zeros(int(good.sum()), dtype=bool)
        for h in E.detect_holds(t[good], load_ratio[good]):
            in_hold_good[h["start_idx"]:h["end_idx"] + 1] = True
        in_hold = np.zeros(n_raw, dtype=bool)
        in_hold[np.flatnonzero(good)] = in_hold_good
        print(f"  {cfg}: usable {int(good.sum()):,}; hold share "
              f"{in_hold[good].mean():.1%}", flush=True)
        for key, mod in models.items():
            kind, mcfg, arm, which = key
            if kind == "A":
                X = (sp_ch if which == "spatial" else bulk_ch)[good]
                hm = in_hold[good]
            else:
                L = D.CONFIGS[mcfg]["L"]
                ch = sp_ch if which == "spatial" else bulk_ch
                wf = D.window_features(
                    np.where(good[:, None], ch, np.nan).astype(np.float32), L)
                idx = np.arange(n_raw)
                lo = np.clip(idx - L + 1, 0, n_raw - 1)
                ok = ((idx >= L - 1) & ((t - t[lo]) <= (L - 1) + D.MAX_GAP_S)
                      & np.isfinite(wf).all(axis=1))
                X = wf[ok].astype(np.float64)
                hm = in_hold[ok]
            pred = P5._predict(mod, X)
            alarm = pred != ("healthy" if kind == "A" else "none")
            res[(cfg,) + key] = {
                "n": int(len(alarm)), "far": float(alarm.mean()),
                "n_hold": int(hm.sum()), "far_hold": float(alarm[hm].mean()),
                "n_dyn": int((~hm).sum()),
                "far_dyn": float(alarm[~hm].mean())}
    return res


def table(models, res, title, verdicts=True):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    print(f"  {'model':<14}{'starv-arm':<13}{'input':<9}"
          f"{'ALL timesteps (4 configs)':<30}{'HOLDS only':<28}"
          f"{'DYNAMIC only':<28}")
    out = {}
    for key in models:
        kind, mcfg, arm, which = key
        r = [res[(c,) + key] for c in P5.REAL_CONFIGS]
        fa = [x["far"] for x in r]
        fh = [x["far_hold"] for x in r]
        fd = [x["far_dyn"] for x in r]
        v = ("PASS" if max(fa) <= P5.PASS_FAR else
             "FAIL simulator-learned" if min(fa) > P5.SIM_LEARNED_FAR
             else "FAIL")
        label = "A snapshot" if kind == "A" else f"B {mcfg}"
        fmt = lambda xs: " ".join(f"{x:.3f}" for x in xs)  # noqa: E731
        print(f"  {label:<14}{arm:<13}{which:<9}{fmt(fa):<30}{fmt(fh):<28}"
              f"{fmt(fd):<28}" + (v if verdicts else "")
              + ("" if max(fd) > P5.PASS_FAR else "  [dynamic <= 5% on all 4]"))
        out["|".join(key)] = {"far": fa, "far_hold": fh, "far_dynamic": fd,
                              "verdict_all_timesteps": v,
                              "dynamic_le_5pct_all4": bool(max(fd) <= P5.PASS_FAR)}
    print("  configs in order: " + ", ".join(P5.REAL_CONFIGS))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("split", "v2"))
    a = ap.parse_args()
    if a.mode == "split":
        print("POST-HOC Sec. 17e-3a -- stored Phase 5 models, holds vs dynamic."
              " No refitting. Descriptive: Sec. 16b's verdicts stand.")
        models = stored_models()
        res = evaluate_real(models)
        out = table(models, res,
                    "STORED v1 MODELS on the real healthy recording "
                    "(verdict column = Sec. 16b's criterion, unchanged)")
        print("  The 5% line against the dynamic-only rate is shown for "
              "reference and is NOT a re-adjudication.")
        PL.dump_json(os.path.join(PL.OUT, "phase5_split_summary.json"), out)
        return

    print("POST-HOC Sec. 17e-3c -- v2 models on the real healthy recording.")
    print("**NOT AN INDEPENDENT TEST.** One change, tested once.")
    D.assert_generator_agreement()
    models = train_v2()
    assert "cdm_parser" not in sys.modules, \
        "the real-data loader was imported before training finished"
    print("independence check: real-data loader not imported during training")
    bad = [(k, int(m["n_iter"])) for k, m in models.items()
           if int(m["n_iter"]) >= (5000 if k[0] == "A" else MB.MAX_ITER)]
    print("CONVERGENCE of v2 final models (first): "
          + (f"all {len(models)} converged" if not bad
             else f"NON-CONVERGED, not reportable: {bad}"))
    res = evaluate_real(models)
    out = table(models, res,
                "v2 MODELS on the real healthy recording -- NOT AN "
                "INDEPENDENT TEST (criterion: <= 5% on all four = PASS; "
                "> 50% on all four = simulator-learned)")
    PL.dump_json(os.path.join(PL.OUT, "phase5_v2_summary.json"),
                 {"nonconverged": bad, "not_an_independent_test": True,
                  "results": out})


if __name__ == "__main__":
    main()
