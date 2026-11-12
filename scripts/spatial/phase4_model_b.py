"""
Phase 4 Model B -- early-warning prediction, spatial arm vs bulk arm.

Design and every criterion evaluated here were fixed in
docs/spatial/synthetic_generator_notes.md Sec. 13 (commit fc3fb28) and amended
ONLY by Sec. 13l (commit df434c5) -- both of which predate this file. Nothing
here may relax them. The report prints which Sec. 13i criterion fires, as
written, before any descriptive number.

  task      window t-L+1..t -> state at t+h, on windows with no confirmed fault
  arms      spatial = 326 channels -> 652 [level|slope] features
            bulk    =   2 channels ->   4 features
  model     the SAME multinomial logistic regression in both arms
            (lbfgs, C=1.0, standardised, class_weight='balanced')
  configs   per-fault (h, L): starvation (1,3), flooding (2,5), drying (20,48)
            + 30 s negative controls for starvation and flooding
  splits    GroupKFold on (trace, cell)  AND  LeaveOneGroupOut on trace
  arms      both starvation arms, neither promoted
  also      binary healthy-vs-starvation on the starvation configuration

Convergence is ASSERTED per fit (n_iter_ < max_iter), collected into
NONCONVERGED and printed first. A library warning is no signal: this runs
with -W ignore.

Results persist per (configuration, starvation arm, task) to
run_logs/model_b_results/ via an atomic write, and a relaunch resumes.

Run:  python scripts/spatial/phase4_model_b.py
"""

from __future__ import annotations

import gc
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fault_generator as F  # noqa: E402
import model_b_data as D  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

RESULTS_DIR = os.path.join(D.REPO, "run_logs", "model_b_results")
SUMMARY_PATH = os.path.join(D.REPO, "run_logs", "model_b_summary.json")
N_FOLDS = 4
MAX_ITER = 20000
BOOTSTRAP = 1000
CEILING = 0.95          # Sec. 13i criterion 1
MIN_POSITIVES = 30      # Sec. 13i criterion 1
CFG_ORDER = ("starvation", "flooding", "drying",
             "starvation_ctrl30", "flooding_ctrl30")
NONCONVERGED = []       # (task, n_features, fold, n_iter)
FIT_LOG = []            # every fit: (task, n_features, fold, n_iter)


def make_model():
    return LogisticRegression(solver="lbfgs", C=1.0, class_weight="balanced",
                              max_iter=MAX_ITER)


def evaluate(X, y, groups, tag):
    """Grouped CV. GroupKFold when there are more groups than folds, else
    LeaveOneGroupOut. Returns out-of-fold predictions."""
    pred = np.empty(len(y), dtype=object)
    n_groups = len(np.unique(groups))
    splitter = (GroupKFold(n_splits=N_FOLDS).split(X, y, groups)
                if n_groups > N_FOLDS
                else LeaveOneGroupOut().split(X, y, groups))
    for fi, (tr, te) in enumerate(splitter):
        sc = StandardScaler().fit(X[tr])
        m = make_model().fit(sc.transform(X[tr]), y[tr])
        ni = int(np.max(np.atleast_1d(m.n_iter_)))
        FIT_LOG.append((tag, X.shape[1], fi, ni))
        if ni >= MAX_ITER:
            NONCONVERGED.append((tag, X.shape[1], fi, ni))
        pred[te] = m.predict(sc.transform(X[te]))
    return pred


def assemble(cfg, arm):
    parts = {k: [] for k in ("Xs", "Xb", "y", "visible", "event", "gc", "gt")}
    for tk in D.TRACES:
        for ci in range(D.CELLS_PER_TRACE):
            g = D.group_dataset(tk, ci, arm, cfg)
            n = len(g["y"])
            for k in ("Xs", "Xb", "y", "visible", "event"):
                parts[k].append(g[k])
            parts["gc"].append(np.full(n, f"{tk}|{ci}", dtype=object))
            parts["gt"].append(np.full(n, tk, dtype=object))
    return {k: (np.vstack(v) if k in ("Xs", "Xb") else np.concatenate(v))
            for k, v in parts.items()}


def _path(cfg, arm, task):
    return os.path.join(RESULTS_DIR, f"{cfg}__{arm}__{task}.npz")


def run_cell(cfg, arm, task):
    """Fit one (configuration, starvation arm, task) under both splits."""
    final = _path(cfg, arm, task)
    if os.path.exists(final):
        print(f"[resume] {cfg:<18} {arm:<12} {task:<6} loaded from disk",
              flush=True)
        return
    ds = assemble(cfg, arm)
    if task == "binary":
        m = np.isin(ds["y"], ("none", "starvation"))
        ds = {k: v[m] for k, v in ds.items()}
    n_before = len(FIT_LOG)
    out = {"y": ds["y"].astype("U12"), "visible": ds["visible"].astype(bool),
           "event": ds["event"].astype("U40"), "gc": ds["gc"].astype("U24"),
           "gt": ds["gt"].astype("U24")}
    for split, groups in (("cell", ds["gc"]), ("trace", ds["gt"])):
        tag = f"{cfg}/{arm}/{task}/{split}"
        out[f"{split}__psp"] = evaluate(ds["Xs"], ds["y"], groups,
                                        tag + "/spatial").astype("U12")
        out[f"{split}__pbu"] = evaluate(ds["Xb"], ds["y"], groups,
                                        tag + "/bulk").astype("U12")
    # the fit log travels WITH the result, so convergence stays reportable
    # across a resume instead of living in one process's memory
    out["fits"] = np.array(json.dumps(FIT_LOG[n_before:]))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp = final.replace(".npz", ".tmp.npz")   # keep .npz last (savez appends it)
    np.savez_compressed(tmp, **out)
    os.replace(tmp, final)
    print(f"[done]   {cfg:<18} {arm:<12} {task:<6} n={len(ds['y']):,}",
          flush=True)
    del ds
    gc.collect()


# ------------------------------------------------------------------ metrics
def _counts(y, p, vis, f):
    pos = y == f
    hit = p == f
    return np.array([
        (pos & hit).sum(),            # 0 TP
        (~pos & hit).sum(),           # 1 FP
        (pos & ~hit).sum(),           # 2 FN
        (~pos).sum(),                 # 3 negatives
        (pos & hit & vis).sum(),      # 4 TP among precursor-visible
        (pos & vis).sum(),            # 5 n precursor-visible positives
        (pos & hit & ~vis).sum(),     # 6 TP among blind
        (pos & ~vis).sum(),           # 7 n blind positives
    ], dtype=float)


def _metrics(c):
    tp, fp, fn, neg = c[0], c[1], c[2], c[3]
    rec = tp / (tp + fn) if tp + fn else np.nan
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / neg if neg else np.nan
    return {"recall": rec, "precision": prec, "f1": f1, "fpr": fpr,
            "J": rec - fpr,
            "recall_visible": c[4] / c[5] if c[5] else np.nan,
            "recall_blind": c[6] / c[7] if c[7] else np.nan}


def analyse(z, split, f, rng):
    """Point metrics + group-bootstrap CIs (resampling (trace, cell) groups)."""
    y, vis, gc_ = z["y"], z["visible"], z["gc"]
    groups = np.unique(gc_)
    per = {}
    for arm_key in ("psp", "pbu"):
        p = z[f"{split}__{arm_key}"]
        per[arm_key] = np.stack([
            _counts(y[gc_ == g], p[gc_ == g], vis[gc_ == g], f)
            for g in groups])
    tot = {k: v.sum(axis=0) for k, v in per.items()}
    m_sp, m_bu = _metrics(tot["psp"]), _metrics(tot["pbu"])
    boot = {k: [] for k in ("dF1", "dJ", "J_sp", "J_bu", "dBlind")}
    ng = len(groups)
    for _ in range(BOOTSTRAP):
        pick = rng.integers(0, ng, ng)
        a = _metrics(per["psp"][pick].sum(axis=0))
        b = _metrics(per["pbu"][pick].sum(axis=0))
        boot["dF1"].append(a["f1"] - b["f1"])
        boot["dJ"].append(a["J"] - b["J"])
        boot["J_sp"].append(a["J"])
        boot["J_bu"].append(b["J"])
        boot["dBlind"].append(a["recall_blind"] - b["recall_blind"])
    ci = {}
    for k, v in boot.items():
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        ci[k] = ((float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
                 if len(v) > 50 else (np.nan, np.nan))
    n_pos = int(tot["psp"][0] + tot["psp"][2])
    n_blind = int(tot["psp"][7])
    n_events = len({e for e, yy in zip(z["event"], y) if yy == f and e})
    return {"spatial": m_sp, "bulk": m_bu, "ci": ci, "n_pos": n_pos,
            "n_blind": n_blind, "n_neg": int(tot["psp"][3]),
            "n_events": n_events, "n": int(len(y))}


def verdict(a, is_control):
    """Apply Sec. 13i (as amended by Sec. 13l) in order. Returns the list of
    criteria that fire, as written."""
    fired = []
    ci = a["ci"]
    if a["bulk"]["f1"] > CEILING or a["n_pos"] < MIN_POSITIVES:
        fired.append("1 UNINFORMATIVE (bulk F1 > 0.95 or < 30 positives)")
    neither = ci["J_sp"][0] <= 0 and ci["J_bu"][0] <= 0
    if neither:
        fired.append("2 NEITHER ARM PREDICTS (J CI includes/below 0 in both)")
    if ci["dF1"][0] <= 0:
        fired.append("3 SPATIAL DOES NOT HELP (delta-F1 CI includes/below 0)")
    if a["n_blind"] > 0 and np.isfinite(ci["dBlind"][0]) and (
            ci["dBlind"][0] > 0 or ci["dBlind"][1] < 0):
        fired.append("4 ARTIFACT ALARM (blind-stratum recall delta CI "
                     "excludes 0)")
    if not fired and not is_control:
        fired.append("5 SPATIAL HELPS (delta-F1 CI above 0; none of 1-4)")
    if not fired and is_control:
        fired.append("control: delta-F1 CI above 0 with no blind-stratum "
                     "alarm (descriptive; see Sec. 13l)")
    return fired


def fmt_ci(c):
    return f"[{c[0]:+.3f},{c[1]:+.3f}]"


def report():
    rng = np.random.default_rng(20261112)
    summary = {}
    print()
    print("=" * 78)
    print("CONVERGENCE (reported before any number, Sec. 13h)")
    print("=" * 78)
    all_fits = []
    for fn in sorted(os.listdir(RESULTS_DIR)):
        if fn.endswith(".npz") and ".tmp" not in fn:
            zz = np.load(os.path.join(RESULTS_DIR, fn), allow_pickle=False)
            all_fits += json.loads(str(zz["fits"]))
    bad = [f for f in all_fits if f[3] >= MAX_ITER]
    NONCONVERGED[:] = bad
    it = np.array([f[3] for f in all_fits])
    sp = np.array([f[1] > 10 for f in all_fits])
    print(f"  fits recorded in the result files: {len(all_fits)}   "
          f"non-converged: {len(bad)}")
    print(f"  max iterations used: spatial {it[sp].max()}, bulk {it[~sp].max()}"
          f"   (cap {MAX_ITER})")
    if bad:
        print("  NON-CONVERGED FITS -- the experiments below that contain them")
        print("  are NOT REPORTABLE:")
        for r in bad:
            print("   ", r)
    print()

    A = {}
    for cfg in CFG_ORDER:
        f = D.OWN_FAULT[cfg]
        for arm in F.STARVATION_ARMS:
            z = dict(np.load(_path(cfg, arm, "multi"), allow_pickle=False))
            for split in ("cell", "trace"):
                A[(cfg, arm, split, "multi")] = analyse(z, split, f, rng)
    for arm in F.STARVATION_ARMS:
        z = dict(np.load(_path("starvation", arm, "binary"), allow_pickle=False))
        for split in ("cell", "trace"):
            A[("starvation", arm, split, "binary")] = analyse(
                z, split, "starvation", rng)

    print("=" * 78)
    print("WHICH PRE-REGISTERED CRITERION FIRES (Sec. 13i as amended by 13l)")
    print("=" * 78)
    for cfg in CFG_ORDER:
        hl = D.CONFIGS[cfg]
        print(f"  {cfg}  (own fault {D.OWN_FAULT[cfg]}, h={hl['h']} s, "
              f"L={hl['L']} s)")
        for arm in F.STARVATION_ARMS:
            for split in ("cell", "trace"):
                a = A[(cfg, arm, split, "multi")]
                v = verdict(a, cfg.endswith("ctrl30"))
                ach = (f"horizon achieved: spatial "
                       f"{'YES' if a['ci']['J_sp'][0] > 0 else 'no'}, bulk "
                       f"{'YES' if a['ci']['J_bu'][0] > 0 else 'no'}")
                print(f"    starv-arm={arm:<12} split={split:<6} -> "
                      + "; ".join(v))
                print(f"{'':>45}{ach}")
                summary[f"{cfg}|{arm}|{split}|multi"] = {
                    "fired": v, **{k: a[k] for k in
                                   ("spatial", "bulk", "n_pos", "n_blind",
                                    "n_neg", "n_events", "n")},
                    "ci": {k: list(c) for k, c in a["ci"].items()}}
        print()

    print("=" * 78)
    print("PER-FAULT RESULTS AT THEIR OWN HORIZONS -- never aggregated")
    print("=" * 78)
    hdr = (f"  {'starv-arm':<12}{'split':<7}{'arm':<8}{'prec':>6}{'rec':>7}"
           f"{'F1':>7}{'FPR':>7}{'J':>7}{'rec-vis':>9}{'rec-blind':>10}")
    for cfg in CFG_ORDER:
        hl = D.CONFIGS[cfg]
        a0 = A[(cfg, F.STARVATION_ARMS[0], "cell", "multi")]
        print(f"  --- {cfg}: {D.OWN_FAULT[cfg]} at h={hl['h']} s, "
              f"L={hl['L']} s; {a0['n_pos']} positives "
              f"({a0['n_blind']} blind) from {a0['n_events']} events; "
              f"{a0['n_neg']} non-{D.OWN_FAULT[cfg]} targets ---")
        print(hdr)
        for arm in F.STARVATION_ARMS:
            for split in ("cell", "trace"):
                a = A[(cfg, arm, split, "multi")]
                for nm in ("spatial", "bulk"):
                    m = a[nm]
                    print(f"  {arm:<12}{split:<7}{nm:<8}{m['precision']:>6.3f}"
                          f"{m['recall']:>7.3f}{m['f1']:>7.3f}{m['fpr']:>7.3f}"
                          f"{m['J']:>7.3f}{m['recall_visible']:>9.3f}"
                          f"{m['recall_blind']:>10.3f}")
                c = a["ci"]
                print(f"  {'':<19}delta-F1 "
                      f"{a['spatial']['f1'] - a['bulk']['f1']:+.3f} "
                      f"{fmt_ci(c['dF1'])}   delta-J {fmt_ci(c['dJ'])}   "
                      f"J_sp {fmt_ci(c['J_sp'])}  J_bu {fmt_ci(c['J_bu'])}"
                      + (f"  dBlind {fmt_ci(c['dBlind'])}"
                         if a["n_blind"] else ""))
        print()

    print("=" * 78)
    print("BINARY healthy-vs-starvation (Sec. 11i / 13g), h=1 s, L=3 s")
    print("=" * 78)
    print(hdr)
    for arm in F.STARVATION_ARMS:
        for split in ("cell", "trace"):
            a = A[("starvation", arm, split, "binary")]
            for nm in ("spatial", "bulk"):
                m = a[nm]
                print(f"  {arm:<12}{split:<7}{nm:<8}{m['precision']:>6.3f}"
                      f"{m['recall']:>7.3f}{m['f1']:>7.3f}{m['fpr']:>7.3f}"
                      f"{m['J']:>7.3f}{m['recall_visible']:>9.3f}"
                      f"{m['recall_blind']:>10.3f}")
            c = a["ci"]
            print(f"  {'':<19}delta-F1 "
                  f"{a['spatial']['f1'] - a['bulk']['f1']:+.3f} "
                  f"{fmt_ci(c['dF1'])}   J_sp {fmt_ci(c['J_sp'])}  "
                  f"J_bu {fmt_ci(c['J_bu'])}")
            summary[f"starvation|{arm}|{split}|binary"] = {
                "fired": verdict(a, False),
                **{k: a[k] for k in ("spatial", "bulk", "n_pos", "n_blind",
                                     "n_neg", "n_events", "n")},
                "ci": {k: list(c) for k, c in a["ci"].items()}}
    print()

    print("=" * 78)
    print("CONFUSION MATRICES, uncollapsed (rows = true target, cols = "
          "predicted), split=cell")
    print("=" * 78)
    for cfg in CFG_ORDER:
        for arm in F.STARVATION_ARMS:
            z = np.load(_path(cfg, arm, "multi"), allow_pickle=False)
            for nm, key in (("SPATIAL", "cell__psp"), ("BULK", "cell__pbu")):
                cm = confusion_matrix(z["y"], z[key], labels=list(D.CLASSES))
                print(f"  {cfg} / starv-arm={arm} / {nm}   {D.CLASSES}")
                for k, row in zip(D.CLASSES, cm):
                    print(f"    {k:>11} " + " ".join(f"{v:>7,}" for v in row))
        print()

    with open(SUMMARY_PATH, "w", encoding="utf-8") as fh:
        json.dump({"nonconverged": NONCONVERGED, "n_fits": len(all_fits),
                   "results": summary}, fh, indent=1, default=float)
    print(f"summary written to {SUMMARY_PATH}")
    print("=" * 78)
    print("All logistic-regression fits converged before max_iter."
          if not NONCONVERGED else
          f"{len(NONCONVERGED)} FITS DID NOT CONVERGE -- see top of report.")
    print("=" * 78)


def main():
    print("Phase 4 Model B -- early-warning prediction, spatial vs bulk")
    print(f"configs {dict(D.CONFIGS)}")
    print(f"results -> {RESULTS_DIR}", flush=True)
    D.assert_generator_agreement()
    print("integrity: vectorized grid path matches fault_generator exactly",
          flush=True)
    for cfg in CFG_ORDER:
        for arm in F.STARVATION_ARMS:
            run_cell(cfg, arm, "multi")
    for arm in F.STARVATION_ARMS:
        run_cell("starvation", arm, "binary")
    report()


if __name__ == "__main__":
    main()
