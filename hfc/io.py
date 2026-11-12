"""
Adapters: map a dataset into the tool's internal Recording, and refuse to
proceed on implausible units.

A Recording is a dict:
  name      str
  t         (n,) seconds
  i         (n,) mean current density, A/cm2, positive
  v         (n,) cell voltage, V, or None
  grid      (n, r, c) current density, A/cm2, positive, or None
  labels    dict of per-sample arrays (synthetic data only) or None
  meta      dict (adapter, units as declared, SYNTHETIC flag if present)

A mapping is a small JSON object; see docs/tool_design.md Sec. 2.2 and
examples/. Three adapters: "canonical" (this tool's own format), "seville"
(the existing cdm_parser.load_run, wrapped), "table" (a delimited bulk file).

UNIT CHECKS ARE ERRORS, NOT WARNINGS. This project once read per-segment
amps as A/cm2 and carried a 6.5x error for weeks, hidden because every ratio
looked fine.
"""

from __future__ import annotations

import json
import os

import numpy as np


class UnitError(ValueError):
    """Input units are implausible or contradict the mapping."""


class ContractError(ValueError):
    """Input does not carry what the data contract requires."""


# ------------------------------------------------------------- unit checks
def check_grid_units(raw_grid, stack_current_A, declared_units, area_cm2,
                     tol=0.10):
    """
    Cross-check the grid against the independent stack-current channel.
    `raw_grid` is as stored (any sign); `declared_units` is 'A/cm2' or
    'A_per_segment'. Returns the integral/stack ratio under the declaration.
    """
    if declared_units not in ("A/cm2", "A_per_segment"):
        raise UnitError(f"unknown current grid units {declared_units!r}")
    n_seg = raw_grid.shape[1] * raw_grid.shape[2]
    seg_area = area_cm2 / n_seg
    tot = np.abs(raw_grid).reshape(len(raw_grid), -1).sum(axis=1)
    ok = np.isfinite(tot) & np.isfinite(stack_current_A) & (stack_current_A > 0)
    if ok.sum() < 10:
        raise ContractError("too few samples with both grid and stack current "
                            "to cross-check units")
    s = float(np.median(tot[ok] / stack_current_A[ok]))
    ratio = {"A_per_segment": s, "A/cm2": s * seg_area}
    r = ratio[declared_units]
    if abs(r - 1.0) <= tol:
        return r
    other = "A_per_segment" if declared_units == "A/cm2" else "A/cm2"
    if abs(ratio[other] - 1.0) <= tol:
        raise UnitError(
            f"current grid is declared {declared_units} but integrates to "
            f"{r:.3f}x the stack current; read as {other} it integrates to "
            f"{ratio[other]:.3f}x. The mapping's units are wrong -- set "
            f"current grid units to {other!r}. (Segment area "
            f"{seg_area:.4f} cm2; this is the per-segment-amps error this "
            f"project has made before.)")
    raise UnitError(
        f"current grid integrates to {r:.3f}x the stack current as "
        f"{declared_units} and {ratio[other]:.3f}x as {other}; neither is "
        f"within {tol:.0%}. Check active area ({area_cm2} cm2), units, and "
        f"that the two channels are the same cell.")


def check_sign(raw_grid, declared_sign):
    v = raw_grid[np.isfinite(raw_grid)]
    neg = float((v < 0).mean()) if v.size else 0.0
    if declared_sign == 1 and neg > 0.9:
        raise UnitError(
            f"{neg:.0%} of current-grid values are negative but the mapping "
            f"declares a positive sign convention. Some mapping sensors "
            f"(the Seville CDM) report negative current: set current_sign "
            f"to -1.")
    if declared_sign == -1 and neg < 0.1:
        raise UnitError("mapping declares a negative sign convention but the "
                        "current grid is positive")
    return neg


def check_bulk(i, v):
    med_i = float(np.nanmedian(i))
    if not (1e-3 <= med_i <= 4.0):
        raise UnitError(
            f"median current density {med_i:.4g} A/cm2 is outside the "
            f"plausible PEMFC range 0.001-4. Check current units and active "
            f"area (a stack current given as A/cm2, or mA given as A, would "
            f"do this).")
    if v is not None:
        med_v = float(np.nanmedian(v))
        if 50.0 <= abs(med_v) <= 1300.0:
            raise UnitError(f"median voltage {med_v:.4g} looks like mV, not V; "
                            f"set the voltage unit to 'mV'.")
        if not (0.05 <= med_v <= 1.3):
            raise UnitError(
                f"median voltage {med_v:.4g} V is not plausible for a single "
                f"cell (0.05-1.3 V). If this is a stack voltage, give "
                f"n_cells in the mapping.")


def _nearest(t_src, x_src, t_dst, tol):
    """x_src sampled at t_src -> t_dst by nearest timestamp within tol."""
    j = np.clip(np.searchsorted(t_src, t_dst), 1, len(t_src) - 1)
    j = np.where(np.abs(t_src[j - 1] - t_dst) <= np.abs(t_src[j] - t_dst),
                 j - 1, j)
    out = np.asarray(x_src, dtype=np.float64)[j]
    out[np.abs(t_src[j] - t_dst) > tol] = np.nan
    return out


# ---------------------------------------------------------------- adapters
LABEL_KEYS = ("state", "fault_type", "starvation_subtype", "severity",
              "segment_type", "quality_truth", "load_ratio",
              "time_since_onset_s",
              "ttc_starvation_s", "ttc_flooding_s", "ttc_drying_s",
              "tto_starvation_s", "tto_flooding_s", "tto_drying_s")


def load_canonical_run(path, use_grid=True):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    area = float(meta["active_area_cm2"])
    t = z["t"].astype(np.float64)
    I = _nearest(z["t_bulk"].astype(np.float64), z["INTENSIDAD"], t, 1.5)
    v = _nearest(z["t_bulk"].astype(np.float64), z["V001"], t, 1.5)
    grid = None
    if use_grid:
        raw = z["current"].astype(np.float64)
        sign = int(meta["current_sign"])
        check_sign(raw, sign)
        check_grid_units(raw, I, meta["current_units"], area)
        grid = raw * sign
        if meta["current_units"] == "A_per_segment":
            grid = grid / (area / (raw.shape[1] * raw.shape[2]))
    rec = {"name": meta["run_id"], "t": t, "i": I / area, "v": v, "grid": grid,
           "labels": {k: z[k] for k in LABEL_KEYS if k in z.files},
           "meta": {"adapter": "canonical", "SYNTHETIC": True,
                    **{k: meta[k] for k in ("run_id", "role", "cell_id",
                                            "pair_id", "profile_id",
                                            "profile_family", "flow_config",
                                            "split", "generator_version",
                                            "seed")}}}
    check_bulk(rec["i"], rec["v"])
    return rec


def load_seville(mapping):
    from ._ref import F  # noqa: F401  (puts scripts/ on the path)
    import cdm_parser
    area = float(mapping.get("active_area_cm2", cdm_parser.TOTAL_ACTIVE_AREA_CM2))
    run = cdm_parser.load_run(
        mapping["config"], mapping.get("run_type", "FC-DLC"),
        data_root=mapping.get("data_root", cdm_parser.DATA_ROOT),
        warmup_seconds=float(mapping.get("warmup_seconds", 300.0)))
    raw = run["current_grid"].astype(np.float64)      # load_run: A/cm2, signed
    ts = run["timestamp"]
    t = np.asarray((ts - ts[0]) / np.timedelta64(1, "s"), dtype=np.float64)
    I = run["bulk"]["INTENSIDAD"].to_numpy().astype(np.float64)
    vch = mapping.get("voltage_channel", "V001")
    v = run["bulk"][vch].to_numpy().astype(np.float64)
    sign = int(mapping.get("current_sign", -1))
    check_sign(raw, sign)
    check_grid_units(raw, I, "A/cm2", area)
    import glob
    # an export of generated data carries SYNTHETIC_README.json (rule: generated
    # data must never be mistakable for real data, even in the raw layout)
    synthetic = bool(glob.glob(os.path.join(
        mapping.get("data_root", cdm_parser.DATA_ROOT), mapping["config"],
        f"*_{mapping.get('run_type', 'FC-DLC')}_*", "SYNTHETIC_README.json")))
    rec = {"name": f"seville/{mapping['config']}", "t": t, "i": I / area,
           "v": v, "grid": raw * sign,
           "temperature": run["temp_grid"], "labels": None,
           "meta": {"adapter": "seville", "config": mapping["config"],
                    "voltage_channel": vch, "SYNTHETIC": bool(synthetic),
                    "n_unmatched_bulk": int(run["n_unmatched_bulk"])}}
    check_bulk(rec["i"], rec["v"])
    return rec


def load_table(mapping):
    import pandas as pd
    df = pd.read_csv(mapping["path"], sep=mapping.get("sep", ","),
                     decimal=mapping.get("decimal", "."))
    tm = mapping["time"]
    if "format" in tm:
        cols = tm["column"] if isinstance(tm["column"], list) else [tm["column"]]
        ts = pd.to_datetime(df[cols].astype(str).agg(" ".join, axis=1),
                            format=tm["format"])
        t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
    else:
        t = df[tm["column"]].to_numpy(dtype=np.float64) * {
            "s": 1.0, "ms": 1e-3, "min": 60.0, "h": 3600.0}[tm.get("unit", "s")]
        t = t - t[0]
    cm = mapping["current"]
    cur = df[cm["column"]].to_numpy(dtype=np.float64)
    unit = cm.get("unit", "A")
    if unit == "A":
        if "active_area_cm2" not in mapping:
            raise ContractError("stack current in A needs active_area_cm2")
        i = cur / float(mapping["active_area_cm2"])
    elif unit == "mA/cm2":
        i = cur / 1000.0
    elif unit == "A/cm2":
        i = cur
    else:
        raise UnitError(f"unknown current unit {unit!r}")
    v = None
    if "voltage" in mapping:
        vm = mapping["voltage"]
        v = df[vm["column"]].to_numpy(dtype=np.float64)
        v = v * {"V": 1.0, "mV": 1e-3}[vm.get("unit", "V")]
        v = v / float(mapping.get("n_cells", 1))
    rec = {"name": os.path.basename(mapping["path"]), "t": t, "i": i, "v": v,
           "grid": None, "labels": None,
           "meta": {"adapter": "table", "SYNTHETIC": False}}
    check_bulk(i, v)
    return rec


def load(mapping):
    """Mapping (dict or path to JSON) -> list of Recordings."""
    if isinstance(mapping, str):
        with open(mapping, encoding="utf-8") as fh:
            mapping = json.load(fh)
    kind = mapping.get("adapter")
    if kind == "canonical":
        d = mapping["path"]
        ids = mapping.get("runs", "all")
        if ids == "all":
            ids = sorted(f[:-4] for f in os.listdir(os.path.join(d, "runs"))
                         if f.endswith(".npz"))
        return [load_canonical_run(os.path.join(d, "runs", r + ".npz"),
                                   mapping.get("use_grid", True)) for r in ids]
    if kind == "seville":
        rec = load_seville(mapping)
        if not mapping.get("use_grid", True):
            rec["grid"] = None
        return [rec]
    if kind == "table":
        return [load_table(mapping)]
    raise ContractError(f"unknown adapter {kind!r}; use 'canonical', "
                        f"'seville' or 'table'")
