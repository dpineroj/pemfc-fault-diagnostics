"""
Train, evaluate and persist the detector's models.

  state       4-class multinomial logistic regression (healthy / starvation /
              flooding / drying) on samples that are healthy or CONFIRMED
              faulty; 'developing' samples are not trained on.
  ahead/<f>   one binary logistic regression per fault: "will fault f be
              confirmed h seconds from now?", on windows with no confirmed
              fault. Horizons 1 s / 2 s / 20 s (generator notes Sec. 13, 15).
  state_level level-only state classifier for data sampled slower than 1 Hz.

Two variants of each: 'bulk' and 'spatial' (bulk + spatial features).

Each synthetic cell is calibrated on its own fault-free calibration run --
the same procedure a user runs on their own healthy data.

TRAIN on the dataset's 'train' cells; thresholds from a held-back fifth of
them; EVALUATE on 'test' cells, which share no cell and no load profile with
training. CIs: bootstrap over test cells. Every fit writes a convergence
record. All faults are synthetic: none of this is real-fault accuracy.
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import GENERATOR_VERSION, UNVERIFIED, __version__, features, io, leaks
from ._ref import F

THRESH = F.CLASSIFICATION_THRESHOLD
FAULTS = ("starvation", "flooding", "drying")
STATES = ("healthy",) + FAULTS
AHEAD = {"starvation": {"h": 1, "L": 3}, "flooding": {"h": 2, "L": 5},
         "drying": {"h": 20, "L": 48}}
STATE_L = 5
MODES = ("bulk", "spatial")
MAX_ITER = 5000
TARGET_FPR = 0.005
N_BOOT = 1000
BOOT_SEED = 20261202


def task_list():
    return ["state", "state_level"] + [f"ahead/{f}" for f in FAULTS]


def task_L(task):
    return {"state": STATE_L, "state_level": None}.get(
        task, AHEAD.get(task.split("/")[-1], {}).get("L"))


# ---------------------------------------------------------------- extraction
def _rows_state(rec, reason):
    st = rec["labels"]["state"]
    keep = (reason == "") & (st != "developing")
    return keep, st


def _rows_ahead(rec, reason, fault):
    """Eligible windows and targets, as generator notes Sec. 13c-e."""
    h, L = AHEAD[fault]["h"], AHEAD[fault]["L"]
    sev = rec["labels"]["severity"].astype(np.float64)
    st = rec["labels"]["state"]
    n = len(sev)
    idx = np.arange(n)
    wmax = np.zeros(n)
    for k in range(L):
        wmax = np.maximum(wmax, sev[np.clip(idx - k, 0, n - 1)])
    hi = np.clip(idx + h, 0, n - 1)
    tsev = sev[hi]
    keep = ((reason == "") & (idx + h < n) & (wmax < THRESH)
            & ~((tsev > 0) & (tsev < THRESH)))
    y = (st[hi] == fault) & (tsev >= THRESH)
    tso = rec["labels"]["time_since_onset_s"][hi] if \
        "time_since_onset_s" in rec["labels"] else np.zeros(n)
    onset = np.round(rec["t"][hi] - np.nan_to_num(tso), 0)
    return keep, y, onset


def extract_cell(dataset_dir, cell_id, roles, modes=MODES):
    """All model tables for one cell. Returns {(task, mode): dict}."""
    runs = os.path.join(dataset_dir, "runs")
    cal_rec = io.load_canonical_run(os.path.join(runs, f"{cell_id}_cal.npz"))
    out = {}
    for mode in modes:
        cal = features.calibrate(
            cal_rec if mode == "spatial" else dict(cal_rec, grid=None))
        for role in roles:
            rec = io.load_canonical_run(
                os.path.join(runs, f"{cell_id}_{role}.npz"))
            if mode == "bulk":
                rec = dict(rec, grid=None)
            Ls = sorted({task_L(t) for t in task_list()},
                        key=lambda v: -1 if v is None else v)
            fm = features.compute_multi(rec, cal, mode, Ls)
            n = len(rec["t"])
            base = {"cell": np.full(n, cell_id), "role": np.full(n, role),
                    "family": np.full(n, rec["meta"]["profile_family"]),
                    "seg": rec["labels"]["segment_type"]}
            for task in task_list():
                X, names, reason = fm[task_L(task)]
                if task.startswith("state"):
                    keep, y = _rows_state(rec, reason)
                    ev = np.full(n, "", dtype="U32")
                else:
                    f = task.split("/")[1]
                    keep, yb, onset = _rows_ahead(rec, reason, f)
                    y = np.where(yb, f, "none")
                    ev = np.where(yb, np.char.add(
                        f"{cell_id}_{role}|", onset.astype(int).astype(str)), "")
                d = out.setdefault((task, mode), {k: [] for k in
                                                  ("X", "y", "ev", "cell",
                                                   "role", "family", "seg",
                                                   "n_abstain", "n_total")})
                d["X"].append(X[keep].astype(np.float32))
                d["y"].append(np.asarray(y)[keep])
                d["ev"].append(ev[keep])
                for k in ("cell", "role", "family", "seg"):
                    d[k].append(base[k][keep])
                d["n_abstain"].append(int((reason != "").sum()))
                d["n_total"].append(n)
                d["names"] = names
    return out


def build_tables(dataset_dir, cells, n_jobs=4, modes=MODES):
    from joblib import Parallel, delayed
    rows = leaks.read_index(dataset_dir)
    roles = {}
    for r in rows:
        if r["role"] != "cal":
            roles.setdefault(r["cell_id"], []).append(r["role"])
    res = Parallel(n_jobs=n_jobs)(
        delayed(extract_cell)(dataset_dir, c, roles[c], modes) for c in cells)
    tabs = {}
    for part in res:
        for key, d in part.items():
            t = tabs.setdefault(key, {k: [] for k in d if k != "names"})
            for k in t:
                t[k] += d[k]
            t["names"] = d["names"]
    for key, t in tabs.items():
        for k in ("y", "ev", "cell", "role", "family", "seg"):
            t[k] = np.concatenate(t[k])
        t["X"] = np.vstack(t["X"])
        t["n_abstain"], t["n_total"] = sum(t["n_abstain"]), sum(t["n_total"])
    return tabs


# ------------------------------------------------------------------- fitting
def fit(X, y, tag):
    sc = StandardScaler().fit(X)
    m = LogisticRegression(solver="lbfgs", C=1.0, class_weight="balanced",
                           max_iter=MAX_ITER)
    m.fit(sc.transform(X), y)
    ni = int(np.max(np.atleast_1d(m.n_iter_)))
    Z = sc.transform(X)
    lo, hi = np.quantile(Z, 0.001, axis=0), np.quantile(Z, 0.999, axis=0)
    pad = np.maximum(0.5 * (hi - lo), 3.0)
    coef, inter = m.coef_, m.intercept_
    if coef.shape[0] == 1:            # binary -> two explicit rows
        coef = np.vstack([-coef[0] / 2, coef[0] / 2])
        inter = np.array([-inter[0] / 2, inter[0] / 2])
    return {"tag": tag, "classes": [str(c) for c in m.classes_],
            "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
            "coef": coef.tolist(), "intercept": inter.tolist(),
            "envelope_lo": (lo - pad).tolist(),
            "envelope_hi": (hi + pad).tolist(),
            "convergence": {"n_iter": ni, "max_iter": MAX_ITER,
                            "converged": bool(ni < MAX_ITER),
                            "n_rows": int(len(y)), "n_features": int(X.shape[1])}}


def predict_proba(model, X):
    """(probabilities (n, k), out-of-envelope mask (n,))."""
    Z = (np.asarray(X, dtype=np.float64) - np.array(model["mean"])) \
        / np.array(model["scale"])
    s = Z @ np.array(model["coef"]).T + np.array(model["intercept"])
    s -= s.max(axis=1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=1, keepdims=True)
    ood = ((Z < np.array(model["envelope_lo"]))
           | (Z > np.array(model["envelope_hi"]))).any(axis=1)
    return p, ood


def decide(model, X):
    """Labels under the model's threshold; 'abstain' outside the envelope."""
    p, ood = predict_proba(model, X)
    cls = np.array(model["classes"])
    quiet = model["quiet_class"]
    qi = list(cls).index(quiet)
    score = 1.0 - p[:, qi]
    pf = p.copy()
    pf[:, qi] = -1.0
    lab = np.where(score >= model["threshold"], cls[pf.argmax(axis=1)], quiet)
    lab = np.where(ood, "abstain", lab).astype("U12")
    return lab, p, score, ood


def _threshold(model, Xv, yv, quiet):
    p, ood = predict_proba(model, Xv)
    score = 1.0 - p[:, model["classes"].index(quiet)]
    neg = score[(yv == quiet) & ~ood]
    if neg.size < 50:
        return 0.5
    return float(max(0.5, np.quantile(neg, 1.0 - TARGET_FPR)))


def train(dataset_dir, out_dir, n_jobs=4, log=print):
    """Fit every model on the dataset's train cells; write models.json."""
    ok, rep = leaks.check(dataset_dir)
    log(leaks.format_report(rep))
    if not ok:
        raise RuntimeError("refusing to train: the dataset's split leaks")
    with open(os.path.join(dataset_dir, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    rows = leaks.read_index(dataset_dir)
    cells = sorted({r["cell_id"] for r in rows if r["split"] == "train"})
    val = set(cells[4::5])
    fitc = [c for c in cells if c not in val]
    log(f"train cells {len(cells)}: fit on {len(fitc)}, thresholds on "
        f"{len(val)} held back")
    tabs = build_tables(dataset_dir, cells, n_jobs)
    bundle = {"SYNTHETIC_TRAINING_DATA": True, "statement": UNVERIFIED,
              "hfc_version": __version__,
              "trained_on": {"dataset": man["name"], "seed": man["seed"],
                             "generator_version": man["generator_version"],
                             "n_train_cells": len(cells),
                             "grid": man["config"]["grid"]},
              "ahead": AHEAD, "state_L": STATE_L, "models": {}}
    for (task, mode), t in sorted(tabs.items()):
        quiet = "healthy" if task.startswith("state") else "none"
        is_val = np.isin(t["cell"], list(val))
        m = fit(t["X"][~is_val], t["y"][~is_val], f"{task}/{mode}")
        m.update(task=task, mode=mode, features=t["names"], L=task_L(task),
                 h=AHEAD.get(task.split("/")[-1], {}).get("h"),
                 quiet_class=quiet)
        m["threshold"] = _threshold(m, t["X"][is_val], t["y"][is_val], quiet)
        c = m["convergence"]
        log(f"  fit {task:<17} {mode:<8} rows {c['n_rows']:>8,} p "
            f"{c['n_features']:>2} iters {c['n_iter']:>4} "
            f"{'converged' if c['converged'] else 'NOT CONVERGED'}  "
            f"threshold {m['threshold']:.3f}")
        bundle["models"][f"{task}|{mode}"] = m
    bad = [k for k, m in bundle["models"].items()
           if not m["convergence"]["converged"]]
    bundle["nonconverged"] = bad
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "models.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(bundle, fh, sort_keys=True)
    log("CONVERGENCE: " + ("all fits converged" if not bad
                           else f"NOT CONVERGED: {bad}"))
    return bundle


def load_bundle(models_dir):
    with open(os.path.join(models_dir, "models.json"), encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------- evaluation
def _counts_state(y, lab):
    cm = np.zeros((4, 5))
    for a, s in enumerate(STATES):
        for b, q in enumerate(STATES + ("abstain",)):
            cm[a, b] = ((y == s) & (lab == q)).sum()
    return cm


def _macro_f1(cm):
    c = cm[:, :4]
    tp = np.diag(c)
    den = c.sum(0) + cm.sum(1)          # abstained rows count as misses
    return float(np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0).mean())


def _prf(tp, fp, fn, neg):
    r = tp / (tp + fn) if tp + fn else np.nan
    p = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    return {"precision": p, "recall": r, "f1": f1,
            "fpr": fp / neg if neg else np.nan}


def _ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            if len(v) > 20 else [float("nan")] * 2)


def evaluate(dataset_dir, bundle, n_jobs=4, log=print):
    """Held-out cells and load profiles. Returns a metrics dict with CIs."""
    rows = leaks.read_index(dataset_dir)
    cells = sorted({r["cell_id"] for r in rows if r["split"] == "test"})
    tabs = build_tables(dataset_dir, cells, n_jobs)
    rng = np.random.default_rng(BOOT_SEED)
    picks = rng.integers(0, len(cells), (N_BOOT, len(cells)))
    cidx = {c: k for k, c in enumerate(cells)}
    out = {"n_test_cells": len(cells), "tasks": {},
           "families": sorted({r["profile_family"] for r in rows
                               if r["split"] == "test" and r["role"] != "cal"})}
    per = {}
    for (task, mode), t in sorted(tabs.items()):
        m = bundle["models"][f"{task}|{mode}"]
        lab, p, score, ood = decide(m, t["X"])
        g = np.array([cidx[c] for c in t["cell"]])
        res = {"n_rows": int(len(lab)), "abstain_envelope": float(ood.mean()),
               "abstain_inputs": t["n_abstain"] / max(t["n_total"], 1)}
        if task.startswith("state"):
            cms = np.stack([_counts_state(t["y"][g == k], lab[g == k])
                            for k in range(len(cells))])
            comp = (t["role"] == "companion")
            hl = t["y"] == "healthy"
            al = (lab != "healthy") & (lab != "abstain")
            far_c = np.array([[al[(g == k) & comp].sum(), (comp & (g == k)).sum()]
                              for k in range(len(cells))], float)
            far_h = np.array([[al[(g == k) & hl].sum(), (hl & (g == k)).sum()]
                              for k in range(len(cells))], float)
            cm = cms.sum(0)
            res.update(
                macro_f1=_macro_f1(cm),
                macro_f1_ci=_ci([_macro_f1(cms[pk].sum(0)) for pk in picks]),
                recall={s: float(cm[a, a] / cm[a].sum()) if cm[a].sum() else None
                        for a, s in enumerate(STATES)},
                precision={s: float(cm[a, a] / cm[:, a].sum())
                           if cm[:, a].sum() else None
                           for a, s in enumerate(STATES)},
                confusion={"rows_true": list(STATES),
                           "cols_pred": list(STATES) + ["abstain"],
                           "matrix": cm.astype(int).tolist()},
                false_alarm_companion=float(far_c[:, 0].sum() / far_c[:, 1].sum()),
                false_alarm_companion_ci=_ci(
                    [far_c[pk, 0].sum() / max(far_c[pk, 1].sum(), 1)
                     for pk in picks]),
                false_alarm_all_healthy=float(far_h[:, 0].sum() / far_h[:, 1].sum()),
                by_family={})
            for fam in out["families"]:
                fm = t["family"] == fam
                if fm.any():
                    res["by_family"][fam] = {
                        "macro_f1": _macro_f1(_counts_state(t["y"][fm], lab[fm])),
                        "false_alarm_companion": float(
                            al[fm & comp].mean()) if (fm & comp).any() else None}
            per[(task, mode)] = cms
        else:
            f = task.split("/")[1]
            pos, hit = t["y"] == f, lab == f
            c = np.array([[(pos & hit & (g == k)).sum(),
                           (~pos & hit & (g == k)).sum(),
                           (pos & ~hit & (g == k)).sum(),
                           (~pos & (g == k)).sum()]
                          for k in range(len(cells))], float)
            tot = c.sum(0)
            ev = {}
            for e, h_ in zip(t["ev"][pos], hit[pos]):
                ev[e] = ev.get(e, False) or bool(h_)
            res.update(_prf(*tot), n_pos=int(tot[0] + tot[2]),
                       n_neg=int(tot[3]), n_events=len(ev),
                       event_recall=(float(np.mean(list(ev.values())))
                                     if ev else None),
                       horizon_s=AHEAD[f]["h"])
            boots = [_prf(*c[pk].sum(0)) for pk in picks]
            for k in ("precision", "recall", "f1", "fpr"):
                res[k + "_ci"] = _ci([b[k] for b in boots])
            per[(task, mode)] = c
        out["tasks"][f"{task}|{mode}"] = res
    # spatial minus bulk, same cells, bootstrap over cells
    out["spatial_minus_bulk"] = {}
    for task in task_list():
        a, b = per.get((task, "spatial")), per.get((task, "bulk"))
        if a is None or b is None:
            continue
        if task.startswith("state"):
            fn = lambda x: _macro_f1(x.sum(0))                    # noqa: E731
            name = "macro_f1"
        else:
            fn = lambda x: _prf(*x.sum(0))["f1"]                  # noqa: E731
            name = "f1"
        out["spatial_minus_bulk"][task] = {
            "metric": name, "delta": fn(a) - fn(b),
            "ci": _ci([fn(a[pk]) - fn(b[pk]) for pk in picks])}
    return out


def apply_shipping_rule(models_dir, metrics):
    """
    docs/tool_design.md Sec. 2.5, fixed before any tool model was trained:
    bulk-only always ships and is the default. The spatial variant ships as
    an option only if, on held-out cells and load profiles, it is not worse
    than bulk-only (delta CI not wholly below zero) for state classification
    and for every fault's prediction.
    """
    b = load_bundle(models_dir)
    need = ["state"] + [f"ahead/{f}" for f in FAULTS]
    d = metrics["spatial_minus_bulk"]
    worse = [t for t in need if d[t]["ci"][1] < 0]
    b["shipped_modes"] = ["bulk"] + ([] if worse else ["spatial"])
    b["shipping_decision"] = {
        "rule": "spatial ships as an option iff its held-out delta CI is not "
                "wholly below zero for state and for every fault-ahead task",
        "deltas": {t: d[t] for t in need}, "worse_tasks": worse,
        "default_mode": "bulk",
        "caveat": "The spatial-versus-bulk difference on this generator is "
                  "ASSUMPTION-DRIVEN: it follows from the assumed size of "
                  "the voltage signature and from per-segment noise being "
                  "independent. It is not evidence about real cells."}
    with open(os.path.join(models_dir, "models.json"), "w", encoding="utf-8",
              newline=chr(10)) as fh:
        json.dump(b, fh, sort_keys=True)
    return b["shipped_modes"], worse
