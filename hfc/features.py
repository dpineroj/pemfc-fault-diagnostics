"""
Healthy calibration and features.

Every feature is a RESIDUAL against a baseline learned from a healthy stretch
of the same cell, scaled by that stretch's own spread. The tool therefore
never needs the training data's idea of "healthy" to match the user's cell:
it needs the user's cell to look like ITSELF when healthy.

Bulk features need stack current and cell voltage. Spatial features need a
current-density grid of any resolution >= 3x3 and reduce it to a handful of
physically named quantities (CV(J), centre/edge contrast, residual energy,
radial correlation, dipole) -- never the raw segments.
"""

from __future__ import annotations

import numpy as np

BULK = ("b_load", "b_dv_z", "b_dv_rel", "b_didt")
SPATIAL = ("s_dcv_z", "s_dcv_rel", "s_dce_z", "s_rad_z", "s_energy_z",
           "s_dipole_z")
SLOPE_OF = {"bulk": ("b_dv_z",), "spatial": ("b_dv_z", "s_dcv_z", "s_rad_z",
                                             "s_energy_z")}
STD_OF = ("b_dv_z",)
MIN_CAL_SAMPLES = 120
LOAD_MARGIN = 0.03          # tolerated excursion beyond the calibrated range
MAX_DT_FULL = 1.25          # s: all capabilities
MAX_DT_LEVEL = 5.0          # s: level-only state classification


def feature_names(mode, L):
    lv = list(BULK) + (list(SPATIAL) if mode == "spatial" else [])
    if L is None:               # level-only (slow sampling)
        return lv
    return (lv + [f"{k}__slope{L}" for k in SLOPE_OF[mode]]
            + [f"{k}__std{L}" for k in STD_OF])


def _geometry(shape):
    rows, cols = shape
    ri, ci = np.indices((rows, cols))
    y = (ri - (rows - 1) / 2) / max((rows - 1) / 2, 1e-9)
    x = (ci - (cols - 1) / 2) / max((cols - 1) / 2, 1e-9)
    rho = np.sqrt(x ** 2 + y ** 2) / np.sqrt(2.0)
    r2 = rho ** 2
    return {"x": x, "y": y, "inner": rho < 0.40, "outer": rho > 0.75,
            "r2c": (r2 - r2.mean()) / r2.std()}


def _spatial_raw(grid):
    """cv, centre/edge contrast, normalised grid g (mean 1)."""
    n = len(grid)
    a = np.abs(grid).reshape(n, -1)
    mean = a.mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.abs(grid) / mean[:, None, None]
        cv = a.std(axis=1) / mean
    geo = _geometry(grid.shape[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        ce = g[:, geo["inner"]].mean(axis=1) / g[:, geo["outer"]].mean(axis=1)
    return cv, ce, g, geo


def _mad(x):
    x = x[np.isfinite(x)]
    if x.size < 5:
        return 0.0, 1.0
    med = float(np.median(x))
    return med, float(1.4826 * np.median(np.abs(x - med)))


class Calibration:
    """Per-cell healthy baseline. Built by `calibrate`, stored as plain data."""

    def __init__(self, d):
        self.d = d

    # ---- baselines by load
    def _interp(self, i, key):
        return np.interp(i, self.d["i_knots"], self.d[key])

    def template(self, i):
        k = np.asarray(self.d["i_knots"])
        T = np.asarray(self.d["template"])
        if len(k) == 1:
            return np.broadcast_to(T[0], (len(i),) + T[0].shape)
        j = np.clip(np.searchsorted(k, i) - 1, 0, len(k) - 2)
        w = np.clip((i - k[j]) / (k[j + 1] - k[j]), 0.0, 1.0)
        return (1 - w)[:, None, None] * T[j] + w[:, None, None] * T[j + 1]

    @property
    def has_spatial(self):
        return "template" in self.d

    def to_json(self):
        return {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.d.items()}

    @classmethod
    def from_json(cls, d):
        return cls({k: (np.asarray(v) if isinstance(v, list) else v)
                    for k, v in d.items()})


def _raw_residuals(rec, cal, mode):
    i, v = rec["i"], rec["v"]
    out = {"dv": v - cal._interp(i, "v_base")}
    out["dv_rel"] = out["dv"] / cal._interp(i, "v_base")
    if mode == "spatial":
        cv, ce, g, geo = _spatial_raw(rec["grid"])
        res = g - cal.template(i)
        out["dcv"] = cv - cal._interp(i, "cv_base")
        out["dcv_rel"] = cv / cal._interp(i, "cv_base") - 1.0
        out["dce"] = ce - cal._interp(i, "ce_base")
        out["rad"] = (res * geo["r2c"][None]).mean(axis=(1, 2))
        out["energy"] = np.sqrt((res ** 2).mean(axis=(1, 2)))
        out["dipole"] = np.hypot((res * geo["x"][None]).mean(axis=(1, 2)),
                                 (res * geo["y"][None]).mean(axis=(1, 2)))
    return out


def calibrate(rec, mask=None, n_bins=10):
    """
    Learn the healthy baseline from `rec[mask]` -- a stretch the USER asserts
    is healthy. Returns a Calibration. Raises if the stretch is unusable.
    """
    i, v = rec["i"], rec["v"]
    if v is None:
        raise ValueError("calibration needs cell voltage")
    ok = np.isfinite(i) & np.isfinite(v)
    if rec.get("grid") is not None:
        ok &= np.isfinite(rec["grid"]).all(axis=(1, 2))
    if mask is not None:
        ok &= mask
    if ok.sum() < MIN_CAL_SAMPLES:
        raise ValueError(f"calibration stretch has {int(ok.sum())} usable "
                         f"samples; need at least {MIN_CAL_SAMPLES}")
    ii = i[ok]
    edges = np.unique(np.quantile(ii, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([ii.min(), ii.min() + 1e-9])
    b = np.clip(np.searchsorted(edges, ii, side="right") - 1, 0, len(edges) - 2)
    used = [k for k in range(len(edges) - 1) if (b == k).sum() >= 5]
    d = {"i_min": float(ii.min()), "i_max": float(ii.max()),
         "n_samples": int(ok.sum()),
         "i_knots": np.array([np.median(ii[b == k]) for k in used]),
         "v_base": np.array([np.median(v[ok][b == k]) for k in used])}
    spatial = rec.get("grid") is not None
    if spatial:
        cv, ce, g, _ = _spatial_raw(rec["grid"][ok])
        d["cv_base"] = np.array([np.median(cv[b == k]) for k in used])
        d["ce_base"] = np.array([np.median(ce[b == k]) for k in used])
        d["template"] = np.stack([g[b == k].mean(axis=0) for k in used])
        d["grid_shape"] = list(rec["grid"].shape[1:])
    cal = Calibration(d)
    sub = {"i": ii, "v": v[ok], "grid": rec["grid"][ok] if spatial else None}
    raw = _raw_residuals(sub, cal, "spatial" if spatial else "bulk")
    for k, x in raw.items():
        med, s = _mad(x)
        floor = 1e-4 if k in ("dv", "dv_rel") else 1e-5
        d[f"med_{k}"], d[f"sig_{k}"] = med, max(s, floor)
    return cal


# ------------------------------------------------------------ rolling ops
def _slope(x, L):
    k = np.arange(L, dtype=np.float64)
    k -= k.mean()
    w = k / (k ** 2).sum()
    out = np.full(len(x), np.nan)
    if len(x) >= L:
        out[L - 1:] = np.convolve(x, w[::-1], mode="valid")
    return out


def _std(x, L):
    out = np.full(len(x), np.nan)
    if len(x) >= L:
        v = np.lib.stride_tricks.sliding_window_view(x, L)
        out[L - 1:] = v.std(axis=1)
    return out


def sampling(rec):
    dt = np.diff(rec["t"])
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.median(dt)) if dt.size else float("nan")


def _slog(z):
    """Signed log: keeps the sign and order of a z-score, tames its tail."""
    return np.sign(z) * np.log1p(np.abs(z))


def compute_multi(rec, cal, mode, Ls):
    """
    Feature matrices for one recording, one per lookback in `Ls`.

    mode 'bulk' | 'spatial';  a lookback is a number of samples, or None for
    level-only. Returns {L: (X (n, p), names, reason (n,) str)} where reason
    is '' for a usable row, else why the tool must not answer there.
    """
    n = len(rec["t"])
    if rec["v"] is None:
        raise ValueError("no cell voltage: nothing in the data contract can "
                         "be computed")
    if mode == "spatial":
        if rec.get("grid") is None or not cal.has_spatial:
            raise ValueError("spatial mode needs a current grid in both the "
                             "recording and the calibration")
        if list(rec["grid"].shape[1:]) != list(cal.d["grid_shape"]):
            raise ValueError("grid resolution differs from the calibration's")
    i = rec["i"]
    raw = _raw_residuals(rec, cal, mode)
    z = lambda k: _slog((raw[k] - cal.d[f"med_{k}"]) / cal.d[f"sig_{k}"])  # noqa: E731
    f = {"b_load": i / cal.d["i_max"], "b_dv_z": z("dv"),
         "b_dv_rel": raw["dv_rel"]}
    dt = np.gradient(rec["t"])
    with np.errstate(invalid="ignore", divide="ignore"):
        f["b_didt"] = np.gradient(i) / np.where(dt > 0, dt, np.nan) / cal.d["i_max"]
    if mode == "spatial":
        f.update(s_dcv_z=z("dcv"), s_dcv_rel=raw["dcv_rel"], s_dce_z=z("dce"),
                 s_rad_z=z("rad"), s_energy_z=z("energy"),
                 s_dipole_z=z("dipole"))
    missing = ~np.isfinite(i) | ~np.isfinite(rec["v"])
    if mode == "spatial":
        missing |= ~np.isfinite(rec["grid"]).all(axis=(1, 2))
    lo = cal.d["i_min"] * (1 - LOAD_MARGIN) - 1e-9
    hi = cal.d["i_max"] * (1 + LOAD_MARGIN) + 1e-9
    with np.errstate(invalid="ignore"):
        out_load = (i < lo) | (i > hi)
    out = {}
    for L in Ls:
        names = feature_names(mode, L)
        g = dict(f)
        gap = np.zeros(n, dtype=bool)
        if L is not None:
            for k in SLOPE_OF[mode]:
                g[f"{k}__slope{L}"] = _slope(f[k], L)
            for k in STD_OF:
                g[f"{k}__std{L}"] = _std(f[k], L)
            idx = np.arange(n)
            t = rec["t"]
            lo_i = np.clip(idx - L + 1, 0, n - 1)
            gap = (idx < L - 1) | ((t - t[lo_i]) > (L - 1) * 1.0 + 1.5)
        X = np.stack([g[k] for k in names], axis=1)
        reason = np.full(n, "", dtype="U32")
        reason[~np.isfinite(X).all(axis=1)] = "window_incomplete"
        reason[gap] = "gap_in_window"
        reason[out_load] = "load_outside_calibration"
        reason[missing] = "missing_input"
        out[L] = (X, names, reason)
    return out


def compute(rec, cal, mode, L):
    return compute_multi(rec, cal, mode, [L])[L]
