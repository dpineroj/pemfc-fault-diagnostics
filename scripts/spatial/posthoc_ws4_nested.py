"""
POST-HOC, EXPLORATORY (Sec. 17f). Matched tuning for both arms.

Does "this model gains nothing from the grid" (Sec. 15f) survive when both
arms get the SAME tuning budget? Nested cross-validation on the Sec. 15 data
(v1), three main configurations, both starvation arms, 4-class task.

  outer   S3 crossed (20 folds): test (trace T, cell c); train trace != T and
          cell != c
  inner   leave-one-cell-out among the 4 training cells (4 folds)
  grid    C in {0.001, 0.01, 0.1, 1, 10}, identical for both arms
  select  own-fault F1 pooled over inner validation folds; ties -> smaller C
  budget  5 candidates x 4 inner fits + 1 refit per outer fold, per arm

A candidate with a non-converged inner fit is ineligible in that fold for
that arm. A non-converged outer refit makes the configuration unreportable.

This does NOT override the pre-registered Sec. 15 null, whatever it shows.

Run:  python -u -W ignore scripts/spatial/posthoc_ws4_nested.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fault_generator as F  # noqa: E402
import model_b_data as D  # noqa: E402
import phase4_model_b as MB  # noqa: E402
import posthoc_lib as PL  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
CFGS = ("starvation", "flooding", "drying")
OUT = os.path.join(PL.OUT, "ws4_nested")


def _f1(y, p, f):
    tp = ((y == f) & (p == f)).sum()
    fp = ((y != f) & (p == f)).sum()
    fn = ((y == f) & (p != f)).sum()
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def outer_fold(X, y, cell, tr, te, f, tag):
    """Select C on inner leave-one-cell-out, refit, predict the test fold."""
    recs, score, eligible = [], {}, {}
    tr_cells = np.unique(cell[tr])
    for C in GRID:
        pv = np.empty(len(tr), dtype=object)
        ok = True
        for c in tr_cells:
            va = np.flatnonzero(cell[tr] == c)
            fi = np.flatnonzero(cell[tr] != c)
            p, rec = PL.fit_one(X, y, tr[fi], tr[va], "B", C)
            pv[va] = p
            ok &= bool(rec["converged"])
            recs.append({"tag": tag, "stage": "inner", "C": C,
                         "val_cell": int(c), **rec})
        eligible[C] = ok
        score[C] = _f1(y[tr], pv, f)
    cands = [C for C in GRID if eligible[C]]
    if not cands:
        return None, None, recs, score
    best = max(cands, key=lambda C: (round(score[C], 6), -C))
    p, rec = PL.fit_one(X, y, tr, te, "B", best)
    recs.append({"tag": tag, "stage": "outer", "C": best, **rec})
    return p, best, recs, score


def run(cfg, arm):
    path = os.path.join(OUT, f"{cfg}__{arm}.npz")
    if os.path.exists(path):
        print(f"[on disk] {cfg} {arm}", flush=True)
        return
    ds = PL.assemble_b(cfg, arm)
    f = D.OWN_FAULT[cfg]
    folds = list(PL.fold_iter("S3", ds["gt"], ds["cell"]))
    out = {"y": ds["y"].astype("U12"), "visible": ds["visible"],
           "event": ds["event"].astype("U40"), "gt": ds["gt"].astype("U12"),
           "cell": ds["cell"]}
    fits = []
    for which, key in (("psp", "Xs"), ("pbu", "Xb")):
        X = ds[key]
        res = Parallel(n_jobs=PL.N_JOBS)(
            delayed(outer_fold)(X, ds["y"], ds["cell"], tr, te, f,
                                f"{cfg}/{arm}/{which}/fold{i}")
            for i, (tr, te) in enumerate(folds))
        pred = np.full(len(ds["y"]), "UNREPORTABLE", dtype=object)
        chosen, scores = [], []
        for (tr, te), (p, best, recs, score) in zip(folds, res):
            fits += recs
            chosen.append(-1.0 if best is None else best)
            scores.append([score[C] for C in GRID])
            if p is not None:
                pred[te] = p
        out[f"S3__{which}"] = pred.astype("U12")
        out[f"C__{which}"] = np.array(chosen)
        out[f"inner_f1__{which}"] = np.array(scores)
        print(f"[done] {cfg:<11} {arm:<12} {which}  selected C per fold: "
              + " ".join(f"{c:g}" for c in chosen), flush=True)
    out["fits"] = np.array(json.dumps(fits))
    PL.save_npz(path, **out)


def report():
    rng = np.random.default_rng(PL.BOOT_SEED)
    fits = []
    for fn in sorted(os.listdir(OUT)):
        if fn.endswith(".npz") and ".tmp" not in fn:
            fits += json.loads(str(np.load(os.path.join(OUT, fn))["fits"]))
    inner_bad = [f for f in fits if f["stage"] == "inner" and not f["converged"]]
    outer_bad = [f for f in fits if f["stage"] == "outer" and not f["converged"]]
    print()
    print("=" * 78)
    print("POST-HOC, EXPLORATORY -- matched tuning (Sec. 17f). CONVERGENCE first")
    print("=" * 78)
    print(f"  fits recorded: {len(fits)}  (inner "
          f"{sum(f['stage'] == 'inner' for f in fits)}, outer "
          f"{sum(f['stage'] == 'outer' for f in fits)})")
    print(f"  non-converged inner fits (candidate made ineligible): "
          f"{len(inner_bad)}")
    by = {}
    for f in inner_bad:
        k = (f["tag"].rsplit("/", 1)[0], f["C"])
        by[k] = by.get(k, 0) + 1
    for k, v in sorted(by.items()):
        print(f"     {k[0]}  C={k[1]:g}: {v}")
    print(f"  non-converged OUTER refits (configuration unreportable): "
          f"{len(outer_bad)}")
    for f in outer_bad:
        print("    ", f)
    print()
    summary = {}
    for cfg in CFGS:
        f = D.OWN_FAULT[cfg]
        for arm in F.STARVATION_ARMS:
            z = dict(np.load(os.path.join(OUT, f"{cfg}__{arm}.npz")))
            z["gc"] = PL.gkey_of(z["gt"], z["cell"])
            if (z["S3__psp"] == "UNREPORTABLE").any() or \
                    (z["S3__pbu"] == "UNREPORTABLE").any():
                print(f"  {cfg} / {arm}: UNREPORTABLE (a fold had no eligible "
                      f"candidate)")
                continue
            a = MB.analyse(z, "S3", f, rng)
            u = dict(np.load(os.path.join(PL.OUT, "model_b_v1",
                                          f"{cfg}__{arm}__multi.npz")))
            u["gc"] = PL.gkey_of(u["gt"], u["cell"])
            assert (u["y"] == z["y"]).all()
            b = MB.analyse(u, "S3", f, rng)
            cs = {w: {f"{c:g}": int((z[f"C__{w}"] == c).sum()) for c in GRID}
                  for w in ("psp", "pbu")}
            print(f"  {cfg} (h={D.CONFIGS[cfg]['h']} s)  starv-arm={arm}")
            print(f"    selected C, folds per value: spatial {cs['psp']}  "
                  f"bulk {cs['pbu']}")
            print(f"    untuned (C=1, Sec. 17c S3): spatial F1 "
                  f"{b['spatial']['f1']:.3f} bulk {b['bulk']['f1']:.3f}  "
                  f"delta {b['spatial']['f1'] - b['bulk']['f1']:+.3f} "
                  f"{MB.fmt_ci(b['ci']['dF1'])}")
            print(f"    TUNED  (nested, S3):        spatial F1 "
                  f"{a['spatial']['f1']:.3f} bulk {a['bulk']['f1']:.3f}  "
                  f"delta {a['spatial']['f1'] - a['bulk']['f1']:+.3f} "
                  f"{MB.fmt_ci(a['ci']['dF1'])}")
            lo, hi = a["ci"]["dF1"]
            read = ("null SURVIVES matched tuning (CI includes or lies below 0)"
                    if lo <= 0 else
                    "EXPLORATORY INDICATION that tuning changes the answer "
                    "(CI lower bound > 0)")
            print(f"    -> {read}")
            summary[f"{cfg}|{arm}"] = {
                "selected_C": cs, "tuned": {"spatial": a["spatial"],
                                            "bulk": a["bulk"],
                                            "ci_dF1": list(a["ci"]["dF1"])},
                "untuned": {"spatial": b["spatial"], "bulk": b["bulk"],
                            "ci_dF1": list(b["ci"]["dF1"])}, "reading": read}
    PL.dump_json(os.path.join(PL.OUT, "ws4_nested_summary.json"),
                 {"n_fits": len(fits), "inner_nonconverged": len(inner_bad),
                  "outer_nonconverged": outer_bad, "results": summary})
    print()
    print("EXPLORATORY. Neither outcome overrides the pre-registered Sec. 15 "
          "null.")


def main():
    print("POST-HOC EXPLORATORY Sec. 17f -- nested CV, matched budget",
          flush=True)
    D.assert_generator_agreement()
    for cfg in CFGS:
        for arm in F.STARVATION_ARMS:
            run(cfg, arm)
    report()


if __name__ == "__main__":
    main()
