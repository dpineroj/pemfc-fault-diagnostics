"""
POST-HOC (Sec. 17). Shared machinery for the follow-up runs.

Everything here is imported from the registered experiments wherever a
registered definition exists (windows, labels, features, model settings,
metrics, criteria) -- nothing is retyped. What is new, and defined in Sec. 17:

  splits      S0 GroupKFold(4) on (trace, cell)      -- original "cell" split
              S1 leave-one-trace-out                  -- original "trace" split
              S2 leave-one-cell(-block)-out
              S3 crossed: test (trace T, cell block b); train trace != T AND
                 block != b
  companions  a fault-free run per (trace, cell): same load trace, same cell
              parameters, severity 0, independent noise (seed offset
              32452843). Same (trace, cell) group as its faulted twin.
  records     every fit writes n_iter / max_iter / converged into the result
              file it belongs to.

fault_generator.py is not modified by anything in this file.
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
from joblib import Parallel, delayed  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = os.path.join(D.REPO, "run_logs", "posthoc")
COMP_OFFSET = 32452843
BOOT_SEED = 20261201
TRACE_KEYS = list(D.TRACES)
N_BLOCKS = 5
SPLITS = ("S0", "S1", "S2", "S3")
SPLIT_NAME = {
    "S0": "S0 (trace,cell) GroupKFold [original; leaks cell AND trace]",
    "S1": "S1 leave-one-trace-out   [original; leaks cell]",
    "S2": "S2 leave-one-cell-out    [holds out cell; trace seen]",
    "S3": "S3 crossed               [holds out cell AND trace]",
}
N_JOBS = int(os.environ.get("POSTHOC_JOBS", "8"))


# ------------------------------------------------------------------ splits
def fold_iter(split, gt, cell):
    """Yield (train_idx, test_idx). gt = trace key per row, cell = int idx."""
    gt = np.asarray(gt, dtype=object)
    cell = np.asarray(cell, dtype=int)
    blk = cell % N_BLOCKS
    n = len(gt)
    if split == "S0":
        groups = np.array([f"{t}|{c}" for t, c in zip(gt, cell)], dtype=object)
        yield from GroupKFold(n_splits=4).split(np.zeros((n, 1)), None, groups)
    elif split == "S1":
        for T in TRACE_KEYS:
            te = gt == T
            yield np.flatnonzero(~te), np.flatnonzero(te)
    elif split == "S2":
        for b in range(N_BLOCKS):
            te = blk == b
            yield np.flatnonzero(~te), np.flatnonzero(te)
    elif split == "S3":
        for T in TRACE_KEYS:
            for b in range(N_BLOCKS):
                te = (gt == T) & (blk == b)
                tr = (gt != T) & (blk != b)
                yield np.flatnonzero(tr), np.flatnonzero(te)
    else:
        raise ValueError(split)


# ------------------------------------------------------------------ fitting
def make_model(kind, C=1.0):
    if kind == "B":      # Model B's registered estimator, C exposed for WS4
        return LogisticRegression(solver="lbfgs", C=C, class_weight="balanced",
                                  max_iter=MB.MAX_ITER), MB.MAX_ITER
    if kind == "A":
        m = MA.make_model("logreg")
        return m, m.max_iter
    if kind == "rf":
        return MA.make_model("rf"), None
    raise ValueError(kind)


def fit_one(X, y, tr, te, kind, C=1.0):
    """One fit. Returns (test predictions, convergence record)."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=int(os.environ.get("POSTHOC_THREADS", "2"))):
        ytr = y[tr]
        if len(np.unique(ytr)) < 2:
            return (np.full(len(te), ytr[0], dtype=object),
                    {"n_iter": 0, "max_iter": 0, "converged": True,
                     "trivial_single_class": True})
        sc = StandardScaler().fit(X[tr])
        m, max_iter = make_model(kind, C)
        if kind == "rf":
            m.set_params(n_jobs=2)
        m.fit(sc.transform(X[tr]), ytr)
        pred = m.predict(sc.transform(X[te]))
        if max_iter is None:
            rec = {"n_iter": None, "max_iter": None, "converged": None}
        else:
            ni = int(np.max(np.atleast_1d(m.n_iter_)))
            rec = {"n_iter": ni, "max_iter": int(max_iter),
                   "converged": bool(ni < max_iter)}
    return pred, rec


def cv_predict(X, y, gt, cell, split, kind, tag, C=1.0):
    """Out-of-fold predictions under `split`, plus one record per fit."""
    folds = list(fold_iter(split, gt, cell))
    res = Parallel(n_jobs=N_JOBS)(
        delayed(fit_one)(X, y, tr, te, kind, C) for tr, te in folds)
    pred = np.empty(len(y), dtype=object)
    seen = np.zeros(len(y), dtype=int)
    fits = []
    for fi, ((tr, te), (p, rec)) in enumerate(zip(folds, res)):
        pred[te] = p
        seen[te] += 1
        fits.append({"tag": tag, "p": int(X.shape[1]), "fold": fi,
                     "n_train": int(len(tr)), **rec})
    assert (seen == 1).all(), f"{tag}: every row must be tested exactly once"
    return pred, fits


def nonconverged(fits):
    return [f for f in fits if f["converged"] is False]


def save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path.replace(".npz", ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


# ------------------------------------------------------- Model B datasets
def _b_noise_seed(trace_key, cell_idx):
    return (F.BASE_SEED + 7919 * (TRACE_KEYS.index(trace_key) + 1)
            + 104729 * (cell_idx + 1) + 15485863)


def companion_b(trace_key, cell_idx, cfg_name):
    """Fault-free companion windows for one (trace, cell), Model B features."""
    cfg = D.CONFIGS[cfg_name]
    h, L = cfg["h"], cfg["L"]
    d = D.load_labels(trace_key, cell_idx)
    n = len(d["sev"])
    d2 = dict(d, sev=np.zeros(n), ft=np.array([""] * n, dtype=object),
              episodes=[])
    wi = D.window_index(d2, h, L)
    rows, cols = F.DEFAULT_GRID_SHAPE
    p = d["params"]
    S = F.logistic_healthy_shape(p["c"], p["r0"], p["w"], rows, cols)
    rng = np.random.default_rng(_b_noise_seed(trace_key, cell_idx) + COMP_OFFSET)
    sigma = S.std() / p["temporal_ratio"]
    grids = (D.LOAD_LEVEL * S[None]
             * (1.0 + rng.normal(0, sigma, (n, rows, cols))))
    grids = grids.reshape(n, -1).astype(np.float32)
    bulk_ch = np.c_[grids.mean(axis=1), d["load"]].astype(np.float32)
    sp_ch = np.c_[grids, bulk_ch]
    keep = wi["keep"]
    y = wi["y"][keep]
    assert (y == "none").all()
    return {"Xs": D.window_features(sp_ch, L)[keep],
            "Xb": D.window_features(bulk_ch, L)[keep], "y": y,
            "visible": wi["visible"][keep],
            "event": np.full(int(keep.sum()), "", dtype=object)}


def assemble_b(cfg, arm, cells=range(5), companions=False):
    keys = ("Xs", "Xb", "y", "visible", "event", "gt", "cell", "comp")
    parts = {k: [] for k in keys}
    for tk in TRACE_KEYS:
        for ci in cells:
            todo = [(D.group_dataset(tk, ci, arm, cfg), False)]
            if companions:
                todo.append((companion_b(tk, ci, cfg), True))
            for g, is_comp in todo:
                n = len(g["y"])
                for k in ("Xs", "Xb", "y", "visible", "event"):
                    parts[k].append(g[k])
                parts["gt"].append(np.full(n, tk, dtype=object))
                parts["cell"].append(np.full(n, ci, dtype=int))
                parts["comp"].append(np.full(n, is_comp, dtype=bool))
    return {k: (np.vstack(v) if k in ("Xs", "Xb") else np.concatenate(v))
            for k, v in parts.items()}


def index_b(cfg, cells=range(5), companions=False):
    """LABEL-ONLY view of assemble_b: no grids. gt / cell / comp / y."""
    h, L = D.CONFIGS[cfg]["h"], D.CONFIGS[cfg]["L"]
    gt, cell, comp, ys = [], [], [], []
    for tk in TRACE_KEYS:
        for ci in cells:
            d = D.load_labels(tk, ci)
            wi = D.window_index(d, h, L)
            n = int(wi["keep"].sum())
            gt += [tk] * n; cell += [ci] * n; comp += [False] * n
            ys.append(wi["y"][wi["keep"]])
            if companions:
                nn = len(d["sev"])
                d2 = dict(d, sev=np.zeros(nn),
                          ft=np.array([""] * nn, dtype=object))
                n2 = int(D.window_index(d2, h, L)["keep"].sum())
                gt += [tk] * n2; cell += [ci] * n2; comp += [True] * n2
                ys.append(np.full(n2, "none", dtype=object))
    return (np.array(gt, dtype=object), np.array(cell), np.array(comp),
            np.concatenate(ys))


# ------------------------------------------------------- Model A datasets
def _a_cached(trace_key, cell_idx):
    z = np.load(os.path.join(MA.CACHE_DIR, f"{trace_key}_{cell_idx}.npz"),
                allow_pickle=True)
    return {"severity": z["severity"], "label": z["label"].astype(object),
            "load": z["load"], "segment": z["segment"],
            "params": z["params"].item()}


def group_a(trace_key, cell_idx, arm, cap=None, companion=False):
    """Model A features for one group. Call inside uniform_amplitude_scale.
    cap=2500 reproduces the quoted capped run's subsample (seed 777+cell)."""
    rows, cols = F.DEFAULT_GRID_SHAPE
    d = _a_cached(trace_key, cell_idx)
    n = len(d["label"])
    if cap is not None and n > cap:
        sel = np.random.default_rng(777 + cell_idx).choice(n, cap, replace=False)
        sel.sort()
        for f in ("severity", "label", "load", "segment"):
            d[f] = d[f][sel]
    if not companion:
        sp, bu, lab, _sev, _seg = MA._features(d, trace_key, cell_idx, arm,
                                               rows, cols)
        return sp, bu, lab
    p = d["params"]
    n = len(d["label"])
    S = F.logistic_healthy_shape(p["c"], p["r0"], p["w"], rows, cols)
    seed = (F.BASE_SEED + 7919 * (TRACE_KEYS.index(trace_key) + 1)
            + 104729 * (cell_idx + 1) + COMP_OFFSET)
    rng = np.random.default_rng(seed)
    sigma = S.std() / p["temporal_ratio"]
    grids = MA.LOAD_LEVEL * S[None] * (1.0 + rng.normal(0, sigma, (n, rows, cols)))
    flat = grids.reshape(n, -1).astype(np.float32)
    bu = np.c_[flat.mean(axis=1), d["load"]].astype(np.float32)
    return (np.c_[flat, bu].astype(np.float32), bu,
            np.full(n, "healthy", dtype=object))


def assemble_a(arm, scale, cap=None, companions=False, cells=range(5)):
    Xs, Xb, ys, gt, cell, comp = [], [], [], [], [], []
    with MA.uniform_amplitude_scale(scale):
        for tk in TRACE_KEYS:
            for ci in cells:
                for is_comp in ((False, True) if companions else (False,)):
                    sp, bu, lab = group_a(tk, ci, arm, cap, is_comp)
                    Xs.append(sp); Xb.append(bu); ys.append(lab)
                    gt.append(np.full(len(lab), tk, dtype=object))
                    cell.append(np.full(len(lab), ci, dtype=int))
                    comp.append(np.full(len(lab), is_comp, dtype=bool))
    return {"Xs": np.vstack(Xs), "Xb": np.vstack(Xb), "y": np.concatenate(ys),
            "gt": np.concatenate(gt), "cell": np.concatenate(cell),
            "comp": np.concatenate(comp)}


# ------------------------------------------------------------ Model A stats
def _cms(y, p, gkey, classes):
    idx = {c: i for i, c in enumerate(classes)}
    yi = np.array([idx[v] for v in y])
    pi = np.array([idx[v] for v in p])
    groups, gi = np.unique(gkey, return_inverse=True)
    k = len(classes)
    cm = np.zeros((len(groups), k, k))
    np.add.at(cm, (gi, yi, pi), 1)
    return cm


def _macro_f1_cm(cm):
    tp = np.diag(cm)
    den = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    return float(f1.mean())


def macro_delta(y, psp, pbu, gkey, rng, classes=MA.CLASSES, n_boot=1000):
    """macro-F1 per arm, delta, and a group-bootstrap 95% CI on the delta."""
    a, b = _cms(y, psp, gkey, classes), _cms(y, pbu, gkey, classes)
    fa, fb = _macro_f1_cm(a.sum(0)), _macro_f1_cm(b.sum(0))
    ng = len(a)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, ng, ng)
        out.append(_macro_f1_cm(a[pick].sum(0)) - _macro_f1_cm(b[pick].sum(0)))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"spatial": fa, "bulk": fb, "delta": fa - fb,
            "ci": [float(lo), float(hi)]}


def gkey_of(gt, cell):
    return np.array([f"{t}|{c}" for t, c in zip(gt, cell)], dtype=object)


def dump_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=float)
