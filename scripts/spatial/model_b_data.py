"""
Model B data layer -- windows, labels and features, exactly as pre-registered
in docs/spatial/synthetic_generator_notes.md Sec. 13c-13f (commit fc3fb28,
which predates this file).

Nothing here may relax a rule fixed there. In particular:

  eligible window   max severity over the window < CLASSIFICATION_THRESHOLD
  label             state at t+h: 'none' iff severity == 0, fault f iff
                    severity >= threshold; targets in (0, threshold) are
                    EXCLUDED as "developing, unconfirmed"
  strata            precursor-visible (max window severity > 0) vs blind
                    (severity exactly 0 throughout the window)
  seams             a window is dropped if any sample in t-L+1..t+h is
                    near_seam_buffer, crosses a segment_id change, or spans a
                    time gap > 1.5 s
  features          per channel: last value (level) + least-squares slope
                    over the lookback (trend). spatial = 326 channels -> 652,
                    bulk = 2 channels -> 4.

LABELS come from the generator's own code path (generate_labeled_run), with
its per-timestep grid and temperature builders stubbed out: those loops are
the slow part, their output is discarded here, and grids are rebuilt
vectorized below with exact agreement against fault_generator asserted
(_assert_matches_generator, shared with Model A). Stubbing changes no label:
severity, fault type, seams and episodes are all fixed before the grid loop.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

# --- path bootstrap (see restructure note in sibling scripts) ---
_S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_S, "common"), os.path.join(_S, "spatial")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# --- end path bootstrap ---

import fault_generator as F  # noqa: E402
import phase4_model_a as MA  # noqa: E402

REPO = os.path.dirname(_S)
CACHE_DIR = os.path.join(REPO, ".model_b_cache")
TRACES = MA.TRACES
CELLS_PER_TRACE = MA.CELLS_PER_TRACE
LOAD_LEVEL = MA.LOAD_LEVEL
THRESH = F.CLASSIFICATION_THRESHOLD
MAX_GAP_S = 1.5
FAULTS = ("starvation", "flooding", "drying")
CLASSES = ("none",) + FAULTS

# Sec. 13d, pre-registered. (horizon h, lookback L), in samples at 1 Hz.
CONFIGS = {
    "starvation": {"h": 1, "L": 3},
    "flooding": {"h": 2, "L": 5},
    "drying": {"h": 20, "L": 48},
    # negative controls (Sec. 13d): Kim et al.'s 30 s horizon
    "starvation_ctrl30": {"h": 30, "L": 3},
    "flooding_ctrl30": {"h": 30, "L": 5},
}
OWN_FAULT = {"starvation": "starvation", "flooding": "flooding",
             "drying": "drying", "starvation_ctrl30": "starvation",
             "flooding_ctrl30": "flooding"}


def _seed(cell_idx):
    return F.BASE_SEED + 1000 * cell_idx   # same cells as Model A


def load_labels(trace_key, cell_idx):
    """Per-timestep labels in TIME ORDER for one (trace, cell); cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cpath = os.path.join(CACHE_DIR, f"{trace_key}_{cell_idx}.npz")
    if os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=False)
        return {
            "t": z["t"], "sev": z["sev"], "ft": z["ft"].astype(object),
            "load": z["load"], "seg": z["seg"], "nsb": z["nsb"],
            "params": json.loads(str(z["params"])),
            "episodes": json.loads(str(z["episodes"])),
        }
    path = os.path.join(F.PROCESSED_DIR, TRACES[trace_key])
    real_cg, real_tg, real_cv = (F.generate_current_grid,
                                 F.generate_temperature_grid, F.compute_cv)
    try:   # stub the slow per-timestep builders; labels are unaffected
        F.generate_current_grid = lambda *a, **k: np.zeros(
            (k.get("rows", F.ROWS), k.get("cols", F.COLS)))
        F.generate_temperature_grid = lambda cg, *a, **k: cg
        F.compute_cv = lambda g: float("nan")
        run = F.generate_labeled_run(trace_path=path, base_seed=_seed(cell_idx),
                                     starvation_arm="directional")
    finally:
        (F.generate_current_grid, F.generate_temperature_grid,
         F.compute_cv) = real_cg, real_tg, real_cv
    ft = np.array(["" if v is None else v for v in run["active_fault_type"]])
    eps = [{"fault_type": e["fault_type"],
            "onset_start_s": float(e["onset_start_s"]),
            "episode_end_s": float(e["episode_end_s"]),
            "precursor_window_s": (None if e["precursor_window_s"] is None
                                   else float(e["precursor_window_s"])),
            "severity_target": float(e["severity_target"]),
            "segment_id": int(e["segment_id"])} for e in run["episodes"]]
    np.savez(cpath, t=run["t_seconds"].astype(np.float64),
             sev=run["severity"].astype(np.float64), ft=ft.astype("U12"),
             load=run["load_ratio"].astype(np.float64),
             seg=run["segment_id"].astype(np.int32),
             nsb=run["near_seam_buffer"].astype(bool),
             params=np.array(json.dumps(run["healthy_params"])),
             episodes=np.array(json.dumps(eps)))
    return load_labels(trace_key, cell_idx)


def build_grids(d, trace_key, cell_idx, arm):
    """(n, 324) float32 synthetic grids in time order for this cell and arm."""
    rows, cols = F.DEFAULT_GRID_SHAPE
    p = d["params"]
    n = len(d["sev"])
    S = F.logistic_healthy_shape(p["c"], p["r0"], p["w"], rows, cols)
    noise_seed = (F.BASE_SEED + 7919 * (list(TRACES).index(trace_key) + 1)
                  + 104729 * (cell_idx + 1) + 15485863)   # distinct from Model A
    rng = np.random.default_rng(noise_seed)
    sigma = S.std() / p["temporal_ratio"]
    grids = LOAD_LEVEL * S[None] * (1.0 + rng.normal(0, sigma, (n, rows, cols)))
    W = MA._weight_fields(p, rows, cols, arm)
    sp = d["sev"] ** F.SEVERITY_CONVEXITY
    for ft in FAULTS:
        m = d["ft"] == ft
        if m.any():
            grids[m] *= (1.0 - sp[m][:, None, None] * W[ft][None])
    return grids.reshape(n, -1).astype(np.float32)


def slope_weights(L):
    k = np.arange(L, dtype=np.float64)
    k -= k.mean()
    return k / (k ** 2).sum()


def window_features(channels, L):
    """
    channels: (n, C) in time order. Returns (n, 2C): [level | slope], where row
    t summarises samples t-L+1..t. Rows t < L-1 are invalid (left as NaN).
    """
    n, C = channels.shape
    out = np.full((n, 2 * C), np.nan, dtype=np.float32)
    out[:, :C] = channels
    w = slope_weights(L)
    acc = np.zeros((n - L + 1, C), dtype=np.float64)
    for k in range(L):
        acc += w[k] * channels[k:n - L + 1 + k]
    out[L - 1:, C:] = acc
    return out


def window_index(d, h, L):
    """
    Decide, per window end t, validity / eligibility / label / stratum.
    Returns dict of arrays over ALL t plus drop counts (Sec. 13e).
    """
    t, sev, ft, seg, nsb = d["t"], d["sev"], d["ft"], d["seg"], d["nsb"]
    n = len(sev)
    idx = np.arange(n)
    in_range = (idx >= L - 1) & (idx + h < n)
    lo = np.clip(idx - L + 1, 0, n - 1)
    hi = np.clip(idx + h, 0, n - 1)

    cs_nsb = np.concatenate([[0], np.cumsum(nsb.astype(int))])
    seam = (cs_nsb[hi + 1] - cs_nsb[lo]) > 0
    crosses = seg[lo] != seg[hi]
    gap = np.concatenate([[False], np.diff(t) > MAX_GAP_S])
    cs_gap = np.concatenate([[0], np.cumsum(gap.astype(int))])
    has_gap = (cs_gap[hi + 1] - cs_gap[lo + 1]) > 0

    wmax = np.zeros(n)
    for k in range(L):
        src = np.clip(idx - k, 0, n - 1)
        wmax = np.maximum(wmax, sev[src])
    tsev = sev[hi]
    tft = ft[hi]

    valid = in_range & ~seam & ~crosses & ~has_gap
    eligible = valid & (wmax < THRESH)
    ambiguous = eligible & (tsev > 0) & (tsev < THRESH)
    keep = eligible & ~ambiguous
    y = np.where(tsev >= THRESH, tft, "none").astype(object)
    return {
        "keep": keep, "y": y, "visible": wmax > 0, "wmax": wmax,
        "t_target": t[hi], "tsev": tsev,
        "drops": {
            "total_t": int(n), "out_of_range": int((~in_range).sum()),
            "seam_or_segment": int((in_range & (seam | crosses)).sum()),
            "time_gap": int((in_range & ~seam & ~crosses & has_gap).sum()),
            "already_confirmed_in_window": int((valid & ~eligible).sum()),
            "ambiguous_target": int(ambiguous.sum()),
            "kept": int(keep.sum()),
        },
    }


def group_dataset(trace_key, cell_idx, arm, cfg_name):
    """Kept windows for one (trace, cell): spatial X, bulk X, y, strata, event id."""
    cfg = CONFIGS[cfg_name]
    h, L = cfg["h"], cfg["L"]
    d = load_labels(trace_key, cell_idx)
    wi = window_index(d, h, L)
    grids = build_grids(d, trace_key, cell_idx, arm)
    bulk_ch = np.c_[grids.mean(axis=1), d["load"]].astype(np.float32)
    sp_ch = np.c_[grids, bulk_ch]
    Xs = window_features(sp_ch, L)[wi["keep"]]
    Xb = window_features(bulk_ch, L)[wi["keep"]]
    y = wi["y"][wi["keep"]]
    vis = wi["visible"][wi["keep"]]
    # event id of the target's episode (distinct load-trace events, Sec. 13b)
    tt = wi["t_target"][wi["keep"]]
    ev = np.full(len(y), "", dtype=object)
    for e in d["episodes"]:
        m = ((y == e["fault_type"]) & (tt >= e["onset_start_s"])
             & (tt <= e["episode_end_s"]))
        ev[m] = f"{trace_key}|{e['fault_type']}|{e['onset_start_s']:.1f}"
    assert np.isfinite(Xs).all() and np.isfinite(Xb).all()
    return {"Xs": Xs, "Xb": Xb, "y": y, "visible": vis, "event": ev,
            "drops": wi["drops"]}


def assert_generator_agreement():
    rows, cols = F.DEFAULT_GRID_SHAPE
    rng = np.random.default_rng(1)
    for arm in F.STARVATION_ARMS:
        MA._assert_matches_generator(F.sample_healthy_params(rng), arm,
                                     rows, cols)
