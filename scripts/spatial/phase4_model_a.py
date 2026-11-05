"""
Phase 4 Model A -- snapshot fault classifier, spatial arm vs bulk arm.

Design fixed in docs/synthetic_generator_notes.md Sec. 12 BEFORE this code was
written. Nothing here may relax a decision recorded there. In particular:
Sec. 12e's amplitude sweep is a REQUIRED part of the experiment, not a
sensitivity footnote, and Sec. 12f's falsification criteria are binding.

  - one timestep in, class out: healthy / drying / flooding / starvation
  - SPATIAL arm = 324 raw 18x18 current values + the two bulk scalars
                  (a spatial sensor also reports the mean, so spatial is a
                  strict SUPERSET of bulk and the delta is the added value of
                  resolution)
  - BULK arm    = mean(J) and load_ratio only
  - temperature EXCLUDED from both arms (Sec. 12c; the 18x18-vs-9x9 resolution
    mismatch with the real sensor is the structurally decisive reason)
  - all four load traces; groups are (trace, cell); leave-(trace,cell)-out
  - near_seam_buffer excluded; class_weight='balanced'; no resampling
  - both starvation arms run end to end, neither promoted

AMPLITUDE SWEEP (Sec. 12e). The per-fault UNIFORM loss constants make mean(J)
nearly diagnostic by itself, which puts the bulk arm near ceiling. The sweep
scales those uniform components only, holding the SPATIAL components fixed, so
spatial information content stays constant while bulk headroom varies. Per
Sec. 11 the committed amplitudes are NOT changed -- the module constants are
restored after each setting and an equality check confirms scale=1.0 reproduces
the generator exactly.

Run:  python scripts/phase4_model_a.py
"""

import contextlib
import gc
import itertools
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- path bootstrap: siblings moved into scripts/{common,spatial} by the
# November 2026 restructure. Keeps `python scripts/<dir>/<file>.py` working
# without turning scripts/ into a package, which would change every
# documented invocation.
import os as _os
import sys as _sys
_S = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(_S, "common"), _os.path.join(_S, "spatial")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path bootstrap ---

import fault_generator as F  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, recall_score  # noqa: E402
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

TRACES = {
    "normal": "normal_flow_fc-dlc_load_trace_dynamic_only.npz",
    "inv_flow": "inverse_flow_fc-dlc_load_trace_dynamic_only.npz",
    "inv_air": "inverse_air_flow_fc-dlc_load_trace_dynamic_only.npz",
    "inv_h2": "inverse_hydrogen_flow_fc-dlc_load_trace_dynamic_only.npz",
}
CACHE_DIR = os.environ.get("MODEL_A_CACHE") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".model_a_cache")
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "run_logs", "model_a_results")
CELLS_PER_TRACE = 5
MAX_PER_GROUP = None            # NO CAP. See Sec. 12h: a 2500 cap here silently
                                # equalized all four traces and erased the
                                # Inverse_Flow asymmetry the trace split exists
                                # to expose. Set to an int only to bind on
                                # nothing, never to make folds comparable.
N_FOLDS = 4
AMPLITUDE_SCALES = (0.0, 0.25, 0.50, 1.00)
SEVERITY_BINS = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.01))
CLASSES = ("healthy", "drying", "flooding", "starvation")
LOAD_LEVEL = 0.697
BOOTSTRAP = 400
CEILING = 0.95                  # Sec. 12f criterion 3

NONCONVERGED = []   # any logreg fit that hit max_iter; reported at the end

UNIFORM_CONSTS = ("DRYING_UNIFORM_LOSS", "FLOODING_UNIFORM_LOSS",
                  "STARVATION_UNIFORM_LOSS")


@contextlib.contextmanager
def uniform_amplitude_scale(scale):
    """Temporarily scale ONLY the uniform-loss constants; always restore."""
    saved = {k: getattr(F, k) for k in UNIFORM_CONSTS}
    try:
        for k, v in saved.items():
            setattr(F, k, v * scale)
        yield
    finally:
        for k, v in saved.items():
            setattr(F, k, v)


# --------------------------------------------------------------- data building
def _weight_fields(params, rows, cols, arm):
    """
    Per-fault 18x18 loss-weight array W, so that M = 1 - sev**p * W.
    Mirrors fault_modulation exactly; _assert_matches_generator checks it.
    """
    w_hot = F._thermal_hot_weight(rows, cols)
    W = {
        "drying": F.DRYING_UNIFORM_LOSS + F.DRYING_CENTRE_LOSS * w_hot,
        "flooding": F.FLOODING_UNIFORM_LOSS
        + F.FLOODING_EDGE_LOSS * (1.0 - w_hot),
    }
    if arm == "directional":
        wv = F._axis_weight(params["starvation_axis_rad"], rows, cols)
        amp = F.STARVATION_AXIS_LOSS
    else:
        wv = F._current_coupled_weight(params, rows, cols)
        amp = F.STARVATION_COUPLED_LOSS
    W["starvation"] = F.STARVATION_UNIFORM_LOSS + amp * wv
    return W


def _assert_matches_generator(params, arm, rows, cols):
    """The vectorized grid path must not diverge from fault_generator."""
    W = _weight_fields(params, rows, cols, arm)
    S = F.logistic_healthy_shape(params["c"], params["r0"], params["w"],
                                 rows, cols)
    for ft in ("drying", "flooding", "starvation"):
        for sev in (0.0, 0.07, 0.5, 1.0):
            mine = 1.0 - (sev ** F.SEVERITY_CONVEXITY) * W[ft]
            ref = F.fault_modulation(ft, sev, rows, cols, params, arm)
            assert np.allclose(mine, ref, rtol=0, atol=1e-12), (ft, sev)
    # and the healthy template path
    ref_h = F.generate_healthy_grid_v3(params, 12345, rows=rows, cols=cols)
    rng = np.random.default_rng(12345)
    sigma = S.std() / params["temporal_ratio"]
    mine_h = LOAD_LEVEL * S * (1.0 + rng.normal(0, sigma, S.shape))
    assert np.allclose(mine_h, ref_h, rtol=0, atol=1e-12)


def build_group(trace_key, cell_idx, arm, scale, rows, cols, label_cache):
    """One (trace, cell) group's features and labels at this amplitude scale."""
    key = (trace_key, cell_idx)
    if key not in label_cache:
        # Label arrays are independent of arm AND of amplitude scale, so this
        # expensive step runs once per (trace, cell) and is cached to disk --
        # generate_labeled_run builds a grid for every timestep, which the
        # experiment then rebuilds vectorized anyway.
        os.makedirs(CACHE_DIR, exist_ok=True)
        cpath = os.path.join(CACHE_DIR, f"{trace_key}_{cell_idx}.npz")
        if os.path.exists(cpath):
            z = np.load(cpath, allow_pickle=True)
            label_cache[key] = {
                "severity": z["severity"], "label": z["label"].astype(object),
                "load": z["load"], "segment": z["segment"],
                "params": z["params"].item(),
            }
            return _features(label_cache[key], trace_key, cell_idx, arm,
                             rows, cols)
        path = os.path.join(F.PROCESSED_DIR, TRACES[trace_key])
        run = F.generate_labeled_run(trace_path=path,
                                     base_seed=F.BASE_SEED + 1000 * cell_idx,
                                     starvation_arm="directional")
        ok = ~run["near_seam_buffer"]
        ft = np.asarray(run["active_fault_type"], dtype=object)[ok]
        lab = np.where(ft == None, "healthy", ft)  # noqa: E711
        label_cache[key] = {
            "severity": run["severity"][ok],
            "label": lab.astype(object),
            "load": run["load_ratio"][ok],
            "segment": run["segment_id"][ok],
            "params": run["healthy_params"],
        }
        # subsample cap, natural proportions preserved
        n = len(lab)
        if MAX_PER_GROUP is not None and n > MAX_PER_GROUP:
            sel = np.random.default_rng(777 + cell_idx).choice(
                n, MAX_PER_GROUP, replace=False)
            sel.sort()
            for f in ("severity", "label", "load", "segment"):
                label_cache[key][f] = label_cache[key][f][sel]
        d0 = label_cache[key]
        np.savez(cpath, severity=d0["severity"],
                 label=np.asarray(d0["label"], dtype=object),
                 load=d0["load"], segment=d0["segment"],
                 params=np.array(d0["params"], dtype=object))
    return _features(label_cache[key], trace_key, cell_idx, arm, rows, cols)


def _features(d, trace_key, cell_idx, arm, rows, cols):
    """Vectorized grid build + feature matrices for one cached group."""
    p = d["params"]
    sev, lab = d["severity"], np.asarray(d["label"], dtype=object)
    n = len(sev)

    S = F.logistic_healthy_shape(p["c"], p["r0"], p["w"], rows, cols)
    # DETERMINISTIC seed: Python's hash() on a str is randomized per process
    # (PYTHONHASHSEED), which would make this script's output irreproducible --
    # unacceptable given every docs figure must come from a re-runnable script.
    noise_seed = F.BASE_SEED + 7919 * (list(TRACES).index(trace_key) + 1)         + 104729 * (cell_idx + 1)
    rng = np.random.default_rng(noise_seed)
    sigma = S.std() / p["temporal_ratio"]
    grids = LOAD_LEVEL * S[None] * (1.0 + rng.normal(0, sigma, (n, rows, cols)))

    W = _weight_fields(p, rows, cols, arm)
    sp = sev ** F.SEVERITY_CONVEXITY
    for ft in ("drying", "flooding", "starvation"):
        m = lab == ft
        if m.any():
            grids[m] *= (1.0 - sp[m][:, None, None] * W[ft][None])

    flat = grids.reshape(n, -1).astype(np.float32)
    mj = flat.mean(axis=1)
    bulk = np.c_[mj, d["load"]].astype(np.float32)
    spatial = np.c_[flat, bulk].astype(np.float32)
    return spatial, bulk, lab, sev, d["segment"]


# ------------------------------------------------------------------- modelling
def make_model(kind):
    if kind == "logreg":
        # max_iter is high deliberately. At 300 the spatial arm (326 features)
        # did NOT converge while the bulk arm (2 features) did -- which would
        # penalise spatial more than bulk and bias the delta toward zero,
        # manufacturing a false negative of exactly the kind Sec. 10 warns
        # about. Convergence is asserted per fit in evaluate().
        # solver: newton-cholesky, adopted for speed (3x) after VERIFYING it is
        # neutral between the arms -- Sec. 10's question applies to a solver as
        # much as to a generator choice. Measured on a held-out fold: switching
        # from lbfgs moved spatial macro-F1 by +0.0004, bulk by +0.0001 and the
        # DELTA by +0.0003, with both arms converging (399 and 9 iterations
        # against a 400 cap). max_iter stays generous and is still asserted.
        return LogisticRegression(max_iter=5000, C=1.0,
                                  class_weight="balanced",
                                  solver="newton-cholesky")
    return RandomForestClassifier(n_estimators=60, class_weight="balanced",
                                  random_state=0, n_jobs=-1)


def evaluate(X, y, groups, kind):
    """
    Grouped CV. `groups` chooses the configuration; the splitter follows from
    it. Returns (out-of-fold predictions, fold assignment).

    TWO CONFIGURATIONS ARE RUN, because they answer different questions and
    only one rehearses Phase 5:

      split="cell"  -- GroupKFold on (trace, cell). Generalization to an unseen
          CELL, with every trace's characteristics present in training. Folds
          balanced by sample count.
      split="trace" -- LeaveOneGroupOut on trace. Generalization to an unseen
          LOAD PROFILE. Four folds, deliberately UNEQUAL (Inverse_Flow is ~40%
          of all samples). Closer to what Phase 5 demands -- a real cell under
          conditions absent from training -- though per the Sec. 6f narrowing
          Phase 5's operating-condition claim has already been surrendered, so
          this configuration rehearses a STRONGER test than Phase 5 can make.

    Nothing is equalized, reweighted or subsampled in either configuration.
    If the two disagree materially, that disagreement is a finding about what
    the model generalizes over, reported as such rather than resolved by
    picking one.
    """
    pred = np.empty(len(y), dtype=object)
    fold = np.full(len(y), -1, dtype=int)
    n_groups = len(np.unique(groups))
    splitter = (GroupKFold(n_splits=N_FOLDS).split(X, y, groups)
                if n_groups > N_FOLDS
                else LeaveOneGroupOut().split(X, y, groups))
    for fi, (tr, te) in enumerate(splitter):
        sc = StandardScaler().fit(X[tr])
        m = make_model(kind).fit(sc.transform(X[tr]), y[tr])
        if kind == "logreg":
            ni = int(np.max(np.atleast_1d(m.n_iter_)))
            if ni >= m.max_iter:
                NONCONVERGED.append((kind, X.shape[1], fi, ni))
        pred[te] = m.predict(sc.transform(X[te]))
        fold[te] = fi
    return pred, fold


def macro_f1(y, p):
    return float(f1_score(y, p, average="macro", labels=list(CLASSES),
                          zero_division=0))


def group_bootstrap_delta(y, p_sp, p_bu, groups, rng):
    """CI on (spatial - bulk) macro-F1, resampling GROUPS not timesteps."""
    uniq = np.unique(groups)
    out = []
    for _ in range(BOOTSTRAP):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in pick])
        out.append(macro_f1(y[idx], p_sp[idx]) - macro_f1(y[idx], p_bu[idx]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(np.mean(out)), float(lo), float(hi)


def fs_confusion(y, p):
    """Flooding<->starvation confusion rate -- Sec. 12d names it separately."""
    cm = confusion_matrix(y, p, labels=list(CLASSES))
    i, j = CLASSES.index("flooding"), CLASSES.index("starvation")
    tot = cm[i].sum() + cm[j].sum()
    return float((cm[i, j] + cm[j, i]) / tot) if tot else float("nan")


def _trace_of(g):
    return str(g).split("|")[0]


def _res_path(arm, scale):
    return os.path.join(RESULTS_DIR, f"{arm}_scale{scale:.2f}.npz")


def _persist(arm, scale, per_split, nfeat):
    """
    Write one (arm, scale) cell's results to RESULTS_DIR immediately.

    Written as each cell completes rather than at the end of the run. The
    previous uncapped attempt died with exit code 4 after hours of fitting and
    lost EVERYTHING, including the traceback, because nothing was materialised
    until the reporting section. Incremental writes mean a mid-run death costs
    only the cell in progress, and a relaunch resumes from here.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {"nfeat": np.array(nfeat)}
    for split, d in per_split.items():
        out[f"{split}__y"] = d["y"].astype("U12")
        # float64, not float32: severity decides the Sec. 12d bin boundaries,
        # and a downcast round-trip is not worth saving ~300 KB per cell.
        out[f"{split}__sev"] = d["sev"].astype(np.float64)
        out[f"{split}__groups"] = d["groups"].astype("U24")
        out[f"{split}__psp"] = d["psp"].astype("U12")
        out[f"{split}__pbu"] = d["pbu"].astype("U12")
        out[f"{split}__fold"] = d["fold"].astype(np.int8)
    # NOTE the ordering: np.savez_compressed APPENDS ".npz" when the path does
    # not already end in it, so a "....npz.tmp" temp name silently becomes
    # "....npz.tmp.npz" and the rename then fails. Keep .npz last.
    final = _res_path(arm, scale)
    tmp = final.replace(".npz", ".tmp.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, final)   # atomic: a reader never sees a half-written file


def _load_persisted(arm, scale):
    p = _res_path(arm, scale)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=False)
    nfeat = int(z["nfeat"])
    per_split = {}
    for split in ("cell", "trace"):
        if f"{split}__y" not in z:
            return None
        per_split[split] = dict(
            y=z[f"{split}__y"].astype(object),
            sev=z[f"{split}__sev"].astype(float),
            groups=z[f"{split}__groups"].astype(object),
            psp=z[f"{split}__psp"].astype(object),
            pbu=z[f"{split}__pbu"].astype(object),
            fold=z[f"{split}__fold"].astype(int),
            nfeat=nfeat,
        )
    return per_split


def fit_cell(arm, scale, rows, cols, label_cache):
    """Fit one (arm, scale) across both split configurations, then free."""
    with uniform_amplitude_scale(scale):
        Xs, Xb, ys, ss, gc_, gt = [], [], [], [], [], []
        for tk in TRACES:
            for ci in range(CELLS_PER_TRACE):
                sp, bu, lab, sev, _ = build_group(
                    tk, ci, arm, scale, rows, cols, label_cache)
                Xs.append(sp)
                Xb.append(bu)
                ys.append(lab)
                ss.append(sev)
                gc_.append(np.full(len(lab), f"{tk}|{ci}", dtype=object))
                gt.append(np.full(len(lab), tk, dtype=object))
        X_sp = np.vstack(Xs)
        X_bu = np.vstack(Xb)
        del Xs, Xb
        gc.collect()
        y = np.concatenate(ys)
        sev = np.concatenate(ss)
        g_cell = np.concatenate(gc_)
        g_trace = np.concatenate(gt)
        nfeat = X_sp.shape[1]

        per_split = {}
        for split, groups in (("cell", g_cell), ("trace", g_trace)):
            psp, fold = evaluate(X_sp, y, groups, "logreg")
            pbu, _ = evaluate(X_bu, y, groups, "logreg")
            per_split[split] = dict(y=y, sev=sev, groups=groups, psp=psp,
                                    pbu=pbu, fold=fold, nfeat=nfeat)
            print(f"[done] arm={arm:<12} scale={scale:<5} split={split:<6} "
                  f"n={len(y):,}  spatial={macro_f1(y, psp):.3f} "
                  f"bulk={macro_f1(y, pbu):.3f}", flush=True)
        # free the large design matrices before the next cell builds its own
        del X_sp, X_bu
        gc.collect()
    _persist(arm, scale, per_split, nfeat)
    return per_split


def main():
    rows, cols = F.DEFAULT_GRID_SHAPE
    print("Phase 4 Model A -- spatial vs bulk, snapshot classifier")
    print(f"grid {rows}x{cols}  traces {list(TRACES)}  "
          f"cells/trace {CELLS_PER_TRACE}  MAX_PER_GROUP={MAX_PER_GROUP}")
    print(f"amplitude scales on the UNIFORM loss constants: {AMPLITUDE_SCALES}")
    print("splits: GroupKFold on (trace,cell)  AND  LeaveOneGroupOut on trace")
    print(f"results written incrementally to {RESULTS_DIR}")
    print()
    rngp = np.random.default_rng(1)
    _assert_matches_generator(F.sample_healthy_params(rngp), "directional",
                              rows, cols)
    _assert_matches_generator(F.sample_healthy_params(rngp), "no_axis",
                              rows, cols)
    print("integrity: vectorized grid path matches fault_generator exactly",
          flush=True)
    print()

    label_cache = {}
    R = {}
    for arm in F.STARVATION_ARMS:
        for scale in AMPLITUDE_SCALES:
            done = _load_persisted(arm, scale)
            if done is not None:
                for split, d in done.items():
                    R[(arm, scale, split)] = d
                print(f"[resume] arm={arm:<12} scale={scale:<5} "
                      f"loaded from disk, skipping fits", flush=True)
                continue
            per_split = fit_cell(arm, scale, rows, cols, label_cache)
            for split, d in per_split.items():
                R[(arm, scale, split)] = d
    print()

    rng = np.random.default_rng(42)
    for arm in F.STARVATION_ARMS:
        print("=" * 78)
        print(f"STARVATION ARM: {arm}")
        print("=" * 78)
        d0 = R[(arm, 1.0, "cell")]
        print(f"  n={len(d0['y']):,}  spatial features={d0['nfeat']}  bulk=2")
        print(f"  class counts: {dict(Counter(d0['y'].tolist()))}")
        print("  per-trace sample counts -- UNEQUAL by design; Inverse_Flow")
        print("  carries ~2x any other trace. NOT reweighted or subsampled:")
        for tk in TRACES:
            n = sum(1 for g in d0["groups"] if _trace_of(g) == tk)
            print(f"    {tk:>10} {n:>8,}  ({100 * n / len(d0['y']):4.1f}%)")
        print()
        for split in ("cell", "trace"):
            tag = ("GroupKFold on (trace,cell) -- unseen CELL"
                   if split == "cell" else
                   "LeaveOneGroupOut on trace -- unseen LOAD PROFILE")
            print(f"  ### split: {tag}")
            print(f"  {'scale':>6} {'spatial':>9} {'bulk':>9} {'delta':>8} "
                  f"{'95% CI':>18} {'bulk>0.95':>10}")
            for scale in AMPLITUDE_SCALES:
                d = R[(arm, scale, split)]
                fsp = macro_f1(d["y"], d["psp"])
                fbu = macro_f1(d["y"], d["pbu"])
                mu, lo, hi = group_bootstrap_delta(
                    d["y"], d["psp"], d["pbu"], d["groups"], rng)
                flag = "YES" if fbu > CEILING else "no"
                print(f"  {scale:>6.2f} {fsp:>9.3f} {fbu:>9.3f} "
                      f"{fsp - fbu:>+8.3f} [{lo:>+.3f},{hi:>+.3f}]{'':>3} "
                      f"{flag:>10}")
            d = R[(arm, 1.0, split)]
            print("  per-fold at scale 1.00:")
            print(f"    {'fold':>4} {'n_test':>8} {'held out':>42} "
                  f"{'spatial':>8} {'bulk':>8} {'delta':>8}")
            for fi in sorted(set(d["fold"].tolist())):
                m = d["fold"] == fi
                held = sorted({_trace_of(g) for g in d["groups"][m]})
                a = macro_f1(d["y"][m], d["psp"][m])
                b = macro_f1(d["y"][m], d["pbu"][m])
                print(f"    {fi:>4} {m.sum():>8,} {','.join(held):>42} "
                      f"{a:>8.3f} {b:>8.3f} {a - b:>+8.3f}")
            print()

        d = R[(arm, 1.0, "cell")]
        y, sv, psp, pbu = d["y"], d["sev"], d["psp"], d["pbu"]
        print("  --- severity bins, scale 1.00, split=cell ---")
        print(f"  {'bin':>12} {'n':>8} {'spatial':>9} {'bulk':>9} {'delta':>8}")
        for lo_b, hi_b in SEVERITY_BINS:
            m = (sv >= lo_b) & (sv < hi_b)
            if m.sum() < 50:
                continue
            a, b = macro_f1(y[m], psp[m]), macro_f1(y[m], pbu[m])
            print(f"  [{lo_b:.1f},{hi_b:.1f}){'':>3} {m.sum():>8,} "
                  f"{a:>9.3f} {b:>9.3f} {a - b:>+8.3f}")
        print()
        rsp = recall_score(y, psp, labels=list(CLASSES), average=None,
                           zero_division=0)
        rbu = recall_score(y, pbu, labels=list(CLASSES), average=None,
                           zero_division=0)
        print(f"  per-class recall  {'class':>11} {'spatial':>9} {'bulk':>9}")
        for k, a, b in zip(CLASSES, rsp, rbu):
            print(f"  {'':>18}{k:>11} {a:>9.3f} {b:>9.3f}")
        print()
        print(f"  flooding<->starvation confusion: "
              f"spatial {fs_confusion(y, psp):.3f}  "
              f"bulk {fs_confusion(y, pbu):.3f}")
        print()
        for nm, p in (("SPATIAL", psp), ("BULK", pbu)):
            print(f"  {nm} confusion (rows=true, cols=pred) {CLASSES}:")
            cm = confusion_matrix(y, p, labels=list(CLASSES))
            for k, row in zip(CLASSES, cm):
                print(f"    {k:>11} " + " ".join(f"{v:>7,}" for v in row))
        print()
        print("  --- random forest robustness, scale 1.00, split=cell ---")
        with uniform_amplitude_scale(1.0):
            Xs, Xb, ys, gcl = [], [], [], []
            for tk in TRACES:
                for ci in range(CELLS_PER_TRACE):
                    sp, bu, lab, _s, _ = build_group(tk, ci, arm, 1.0, rows,
                                                     cols, label_cache)
                    Xs.append(sp)
                    Xb.append(bu)
                    ys.append(lab)
                    gcl.append(np.full(len(lab), f"{tk}|{ci}", dtype=object))
            Xs2 = np.vstack(Xs)
            Xb2 = np.vstack(Xb)
            del Xs, Xb
            gc.collect()
            yy = np.concatenate(ys)
            gg = np.concatenate(gcl)
            a, _ = evaluate(Xs2, yy, gg, "rf")
            b, _ = evaluate(Xb2, yy, gg, "rf")
            del Xs2, Xb2
            gc.collect()
        print(f"    spatial {macro_f1(yy, a):.3f}  bulk {macro_f1(yy, b):.3f}"
              f"  delta {macro_f1(yy, a) - macro_f1(yy, b):+.3f}")
        print()

    print("=" * 78)
    if NONCONVERGED:
        print(f"WARNING: {len(NONCONVERGED)} logreg fits hit max_iter "
              f"(n_features, fold, n_iter): {NONCONVERGED[:8]}")
        print("  A non-converged arm understates its own score. Treat any")
        print("  delta involving those fits as unreliable.")
    else:
        print("All logistic-regression fits converged before max_iter.")
    print("=" * 78)


if __name__ == "__main__":
    main()
