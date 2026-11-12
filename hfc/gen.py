"""
hfcgen-4.0 -- synthetic PEM fuel cell dataset generator.

SYNTHETIC DATA. Every file written here carries a SYNTHETIC flag, the
generator version, the seed and the full config.

Signal model (docs/tool_design.md Sec. 1.1): the cell is current-controlled.
A fault cannot remove current; it REDISTRIBUTES it across the plate and
LOWERS THE VOLTAGE. Fault trajectories come from the frozen reference
generator's episode builders, called unmodified.

Nothing here is fit to, tuned on or selected with the real recording's
healthy behaviour. See hfc/provenance.py for the basis of every parameter.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import os
import zipfile

import numpy as np
from scipy.ndimage import zoom
from scipy.signal import lfilter

from . import GENERATOR_VERSION, profiles
from ._ref import F
from .provenance import PROVENANCE, table_markdown

FAULTS = ("starvation", "flooding", "drying")
FLOW_CONFIGS = ("Normal_Flow", "Inverse_Flow", "Inverse_Air_Flow",
                "Inverse_Hydrogen_Flow")
# (air axis angle, hydrogen axis angle), radians. ASSUMPTION: see provenance.
FLOW_AXES = {
    "Normal_Flow": (np.pi / 2, 0.0),
    "Inverse_Air_Flow": (-np.pi / 2, 0.0),
    "Inverse_Hydrogen_Flow": (np.pi / 2, np.pi),
    "Inverse_Flow": (-np.pi / 2, np.pi),
}
ROLES = ("cal", "fault", "companion")
THRESH = F.CLASSIFICATION_THRESHOLD
E_TN = 1.254

# Per-cell brackets; keys must all appear in provenance.PROVENANCE.
DEFAULT_PHYSICS = {
    "edge_ratio": (1.5, 2.5),
    "edge_load_slope": (-0.15, 0.15),
    "temporal_factor": (10.0, 20.0),
    "fault_amplitude_scale": (0.6, 1.2),
    "starvation_axis_mix": (0.0, 1.0),
    "rated_current_density": (0.5, 1.0),
    "idle_fraction": (0.03, 0.10),
    "v_transient_gain": (0.02, 0.10),
    "v_transient_tau": (2.0, 15.0),
    "drying_resistance_gain": (0.5, 2.0),
    "flooding_voltage_loss": (0.05, 0.20),
    "flooding_overshoot": (0.005, 0.03),
    "o2_starvation_voltage_loss": (0.4, 0.9),
    "h2_starvation_reversal": (0.1, 0.8),
    "voltage_noise": (0.001, 0.004),
    "thermal_centre_gain": (2.0, 8.0),
    "thermal_local_gain": (1.0, 4.0),
}
POL = {"pol_E_oc": (1.031, 0.97, 1.03), "pol_A": (0.03, 0.8, 1.2),
       "pol_r": (2.45e-4, 0.7, 1.5), "pol_m": (2.11e-5, 0.5, 2.0),
       "pol_n": (8e-3, 0.9, 1.1)}

DEFAULT_CONFIG = {
    "name": "custom",
    "seed": 20261112,
    "dt_s": 1.0,
    "grid": {"current": [18, 18], "temperature": [9, 9]},
    "active_area_cm2": 50.0,
    "temperature_setpoint_C": 65.0,
    "temperature_noise_C": 0.2,
    "p_trigger": {"starvation": 0.5, "flooding": 0.5, "drying": 0.5},
    "h2_starvation_fraction": 0.3,
    "companions": True,
    "cells_recur_across_profiles": False,
    "profiles": {"n_cycles": 1,
                 "holds": {"share": 0.25, "duration_s": [120.0, 480.0],
                           "level": [0.10, 0.85]},
                 "static_duration_s": [900, 1500]},
    "cohorts": [],
    "physics": {k: list(v) for k, v in DEFAULT_PHYSICS.items()},
    "messiness": {
        "startup_transient_s": 0.0,
        "dropout_per_hour": 0.0,
        "dropout_max_s": 20.0,
        "bulk_clock_offset_s": 0.0,
        "bulk_whole_second": False,
        "slow_sampling_factor": 1,
        "current_sign": 1,
        "current_units": "A/cm2",
    },
}

PRESETS = {
    "smoke": {
        "name": "smoke", "grid": {"current": [8, 8], "temperature": [4, 4]},
        "cohorts": [
            {"family": "fcdlc_random", "n_cells": 6, "split": "train"},
            {"family": "static", "n_cells": 2, "split": "train"},
            {"family": "fcdlc_random", "n_cells": 3, "split": "test"},
            {"family": "static", "n_cells": 1, "split": "test"},
        ]},
    "extensive": {
        "name": "extensive",
        "cohorts": [
            {"family": "fcdlc_random", "n_cells": 220, "split": "train"},
            {"family": "static", "n_cells": 20, "split": "train"},
            {"family": "fcdlc_random", "n_cells": 40, "split": "test"},
            {"family": "static", "n_cells": 16, "split": "test"},
            {"family": "fcdlc_real", "n_cells": 24, "split": "test",
             "optional": True},
        ]},
    "messy": {
        "name": "messy", "grid": {"current": [8, 8], "temperature": [4, 4]},
        "cohorts": [{"family": "fcdlc_random", "n_cells": 3, "split": "test"}],
        "messiness": {"startup_transient_s": 200.0, "dropout_per_hour": 20.0,
                      "dropout_max_s": 20.0, "bulk_clock_offset_s": 13.0,
                      "bulk_whole_second": True, "slow_sampling_factor": 1,
                      "current_sign": -1, "current_units": "A_per_segment"}},
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def make_config(preset=None, overrides=None):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if preset:
        cfg = _merge(cfg, PRESETS[preset])
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


def _ss(cfg, *key):
    return np.random.SeedSequence(int(cfg["seed"]),
                                  spawn_key=tuple(int(k) for k in key))


# ------------------------------------------------------------------- plan
def plan(cfg):
    """One entry per cell: IDs, split, profile, flow config. No arrays."""
    cells, g = [], 0
    real_ok = profiles.real_profiles_available()
    skipped = []
    for co in cfg["cohorts"]:
        fam = co["family"]
        if fam == "fcdlc_real" and not real_ok:
            if co.get("optional"):
                skipped.append(co)
                continue
            raise FileNotFoundError(
                "cohort needs the real FC-DLC load traces in data/processed/ "
                "(third-party derived; see the README to regenerate them)")
        for k in range(int(co["n_cells"])):
            fi = profiles.FAMILIES.index(fam)
            if fam == "fcdlc_real":
                pidx = k % profiles.n_real_profiles()
            elif cfg.get("cells_recur_across_profiles"):
                pidx = k            # same cell index under each cohort
            else:
                pidx = g
            cid = (f"c{k:05d}" if cfg.get("cells_recur_across_profiles")
                   else f"c{g:05d}")
            cseed = k if cfg.get("cells_recur_across_profiles") else g
            cells.append({"g": g, "cell_id": cid, "cell_seed": cseed,
                          "pair_id": f"{cid}@{fam}/{pidx}",
                          "family": fam, "family_idx": fi, "profile_idx": pidx,
                          "flow_config": FLOW_CONFIGS[g % 4],
                          "flow_idx": g % 4, "split": co["split"]})
            g += 1
    return cells, skipped


# ------------------------------------------------------------------ physics
def draw_cell(cfg, cell):
    rng = np.random.default_rng(_ss(cfg, 3, cell["cell_seed"]))
    p = {k: float(rng.uniform(*cfg["physics"][k])) for k in DEFAULT_PHYSICS}
    for k, (v, lo, hi) in POL.items():
        p[k] = float(v * rng.uniform(lo, hi))
    return p


def build_episodes(prof, cfg, rng):
    """Fault episodes for one faulted run. Non-overlapping by construction."""
    t, lr = prof["t"], prof["load_ratio"]
    cands = []
    trig = {f: 0 for f in FAULTS}
    if prof["induced"] is not None:
        for onset, ft, dur in prof["induced"]:
            cands.append((onset, ft, {"start_s": onset, "end_s": onset + dur,
                                      "duration_s": dur, "peak_rate": 0.0},
                          "induced"))
    else:
        for e in F.find_ramp_events(t, lr, F.RAMP_THRESHOLD):
            if e["direction"] == "up":      # air-supply lag: load INCREASES
                cands.append((e["start_s"], "starvation", e, "ramp_up"))
        for e in F.find_dwell_events(t, lr, F.LOW_LOAD_THRESH, "low",
                                     F.MIN_DWELL_S):
            cands.append((e["start_s"], "flooding", e, "low_load_dwell"))
        for e in F.find_dwell_events(t, lr, F.HIGH_LOAD_THRESH, "high",
                                     F.MIN_DWELL_S):
            cands.append((e["start_s"], "drying", e, "high_load_dwell"))
    cands.sort(key=lambda c: (c[0], c[1]))
    eps, busy, skipped = [], -1.0, 0
    for onset, ft, ev, kind in cands:
        trig[ft] += 1
        fire = rng.random() < (1.0 if kind == "induced"
                               else float(cfg["p_trigger"][ft]))
        sub = "h2" if rng.random() < float(cfg["h2_starvation_fraction"]) else "o2"
        if not fire:
            continue
        ep = F._EPISODE_BUILDERS[ft](ev, rng)
        if ep["onset_start_s"] <= busy:
            skipped += 1                   # no concurrent faults
            continue
        ep["episode_end_s"] = float(min(ep["episode_end_s"], t[-1]))
        ep["subtype"] = sub if ft == "starvation" else ""
        ep["trigger"] = kind
        busy = ep["episode_end_s"] + 2.0
        eps.append(ep)
    return eps, trig, skipped


def labels_from(prof, eps):
    t = prof["t"]
    n = len(t)
    sev = np.zeros(n)
    ft = np.full(n, "", dtype="U12")
    sub = np.full(n, "", dtype="U4")
    tso = np.full(n, np.nan)
    table = []
    for ep in eps:
        i0 = int(np.searchsorted(t, ep["onset_start_s"], "left"))
        i1 = int(np.searchsorted(t, ep["episode_end_s"], "right"))
        if i0 >= i1:
            continue
        s = np.clip(ep["severity_fn"](t[i0:i1]), 0.0, 1.0)
        sev[i0:i1] = s
        act = s > 0
        ft[i0:i1][act] = ep["fault_type"]
        sub[i0:i1][act] = ep["subtype"]
        tso[i0:i1][act] = t[i0:i1][act] - ep["onset_start_s"]
        above = np.flatnonzero(s >= THRESH)
        table.append({
            "fault_type": ep["fault_type"], "subtype": ep["subtype"],
            "trigger": ep["trigger"],
            "onset_s": float(ep["onset_start_s"]),
            "confirm_s": float(t[i0 + above[0]]) if above.size else None,
            "offset_s": float(ep["episode_end_s"]),
            "severity_target": float(ep["severity_target"]),
            "peak_severity": float(s.max())})
    state = np.where(sev >= THRESH, ft, np.where(sev > 0, "developing",
                                                 "healthy")).astype("U12")
    out = {"severity": sev.astype(np.float32), "fault_type": ft,
           "starvation_subtype": sub, "state": state,
           "time_since_onset_s": tso.astype(np.float32)}
    for f in FAULTS:
        for key, name in (("onset_s", "tto"), ("confirm_s", "ttc")):
            times = np.array(sorted(e[key] for e in table
                                    if e["fault_type"] == f
                                    and e[key] is not None))
            v = np.full(n, np.nan)
            if times.size:
                j = np.searchsorted(times, t, "left")
                ok = j < times.size
                v[ok] = times[j[ok]] - t[ok]
            out[f"{name}_{f}_s"] = v.astype(np.float32)
    return out, table


def simulate(prof, lab, p, cfg, flow_config, rng):
    """Current-density grid, temperature grid, stack current and voltage."""
    rows, cols = cfg["grid"]["current"]
    tr, tc = cfg["grid"]["temperature"]
    t, lr = prof["t"], prof["load_ratio"]
    n = len(t)
    sev = lab["severity"].astype(np.float64)
    ft, sub = lab["fault_type"], lab["starvation_subtype"]
    i = p["rated_current_density"] * (p["idle_fraction"]
                                      + (1 - p["idle_fraction"]) * lr)

    # healthy template: the radial clamping form of generate_healthy_grid_v2
    ri, ci = np.indices((rows, cols))
    r = np.sqrt((ri - (rows - 1) / 2) ** 2 + (ci - (cols - 1) / 2) ** 2)
    rr = (r / r.max()) ** F.RADIAL_EDGE_EXPONENT
    S0 = F._radial_healthy_shape(p["edge_ratio"], rows, cols)
    a = np.clip((p["edge_ratio"] - 1) * (1 + p["edge_load_slope"] * (lr - 0.5)),
                0.05, None)
    S = 1.0 + a[:, None, None] * rr[None]
    S /= S.mean(axis=(1, 2), keepdims=True)

    # fault modulation, same forms and constants as the reference generator
    sp = sev ** F.SEVERITY_CONVEXITY
    amp = p["fault_amplitude_scale"]
    w_hot = F._thermal_hot_weight(rows, cols)
    coupled = (S0 - S0.min()) / max(S0.max() - S0.min(), 1e-12)
    th_air, th_h2 = FLOW_AXES[flow_config]
    mix = p["starvation_axis_mix"]
    W = {
        ("drying", ""): F.DRYING_UNIFORM_LOSS + F.DRYING_CENTRE_LOSS * w_hot,
        ("flooding", ""): F.FLOODING_UNIFORM_LOSS
        + F.FLOODING_EDGE_LOSS * (1 - w_hot),
    }
    for s_, th in (("o2", th_air), ("h2", th_h2)):
        w = mix * F._axis_weight(th, rows, cols) + (1 - mix) * coupled
        W[("starvation", s_)] = (F.STARVATION_UNIFORM_LOSS
                                 + F.STARVATION_AXIS_LOSS * w)
    M = np.ones((n, rows, cols))
    for (f, s_), w in W.items():
        m = (ft == f) & (sub == s_)
        if m.any():
            M[m] = 1.0 - sp[m][:, None, None] * (amp * w)[None]
    P = S * M
    P /= P.mean(axis=(1, 2), keepdims=True)     # current-controlled
    sigma = S0.std() / p["temporal_factor"]
    J = i[:, None, None] * P * (1.0 + rng.normal(0, sigma, (n, rows, cols)))

    # voltage
    ima = i * 1000.0
    Vh = (p["pol_E_oc"] - p["pol_r"] * ima - p["pol_A"] * np.log(ima)
          - p["pol_m"] * np.exp(p["pol_n"] * ima))
    dt = float(cfg["dt_s"])
    al = np.exp(-dt / p["v_transient_tau"])
    step = lfilter([1.0], [1.0, -al], np.gradient(i, t) * dt)
    V = Vh - p["v_transient_gain"] * step
    dry, flo = ft == "drying", ft == "flooding"
    o2 = (ft == "starvation") & (sub == "o2")
    h2 = (ft == "starvation") & (sub == "h2")
    V = V - dry * (p["pol_r"] * ima * p["drying_resistance_gain"] * sp)
    V = V - flo * (p["flooding_voltage_loss"] * sp
                   * i / p["rated_current_density"])
    drain = np.clip(-np.gradient(sev, t), 0.0, None) * flo
    over = lfilter([1 - np.exp(-dt / 2.0)], [1.0, -np.exp(-dt / 2.0)],
                   np.clip(drain / 0.1, 0.0, 1.0))
    V = V + p["flooding_overshoot"] * over
    V = V - o2 * (Vh * p["o2_starvation_voltage_loss"] * sp)
    V = V - h2 * ((Vh + p["h2_starvation_reversal"]) * sp)
    V = V + rng.normal(0, p["voltage_noise"], n)

    # temperature from local heat flux
    q = J * (E_TN - V[:, None, None])
    qd = zoom(q, (1, tr / rows, tc / cols), order=1) if (tr, tc) != (rows, cols) else q
    qbar = q.mean(axis=(1, 2))[:, None, None]
    T = (cfg["temperature_setpoint_C"]
         + p["thermal_centre_gain"] * F._thermal_hot_weight(tr, tc)[None] * qbar
         + p["thermal_local_gain"] * (qd - qbar)
         + rng.normal(0, cfg["temperature_noise_C"], (n, tr, tc)))
    return {"current": J.astype(np.float32), "temperature": T.astype(np.float32),
            "INTENSIDAD": (i * cfg["active_area_cm2"]).astype(np.float32),
            "V001": V.astype(np.float32)}


# ---------------------------------------------------------------- messiness
def apply_messiness(run, cfg, rng):
    """Each piece is switchable (config 'messiness'). Truth labels are kept;
    quality_truth records what was done to each sample."""
    m = cfg["messiness"]
    n = len(run["t"])
    q = np.full(n, "ok", dtype="U8")
    t = run["t"].astype(np.float64).copy()
    tb = t.copy()
    I, V = run["INTENSIDAD"].astype(np.float64), run["V001"].astype(np.float64)
    if m["bulk_clock_offset_s"] or m["bulk_whole_second"]:
        off = float(m["bulk_clock_offset_s"])
        tb = np.arange(np.ceil(t[0] + off), t[-1] - 5.0, 1.0)
        if not m["bulk_whole_second"]:
            tb = tb + 0.37
        I, V = np.interp(tb, t, I), np.interp(tb, t, V)
    if m["dropout_per_hour"] > 0:
        k = rng.poisson(m["dropout_per_hour"] * (t[-1] - t[0]) / 3600.0)
        for _ in range(int(k)):
            t0 = rng.uniform(t[0], t[-1])
            d = rng.uniform(2.0, m["dropout_max_s"])
            if rng.random() < 0.5:
                g = (t >= t0) & (t < t0 + d)
                run["current"][g] = np.nan
                run["temperature"][g] = np.nan
                q[g] = "dropout"
            else:
                V[(tb >= t0) & (tb < t0 + d)] = np.nan
    if m["startup_transient_s"] > 0:
        s = t < t[0] + m["startup_transient_s"]
        jit = rng.uniform(-0.45, 0.07, int(s.sum()))
        t[s] = np.maximum.accumulate(t[s] + jit)
        alt = (np.indices(run["current"].shape[1:]).sum(axis=0) % 2)
        run["current"][0] = run["current"][0] * alt * 1e-3
        q[s] = "startup"
    k = int(m["slow_sampling_factor"])
    if k > 1:
        for key in list(run):
            if isinstance(run[key], np.ndarray) and len(run[key]) == n:
                run[key] = run[key][::k]
        t, q = t[::k], q[::k]
        tb, I, V = tb[::k], I[::k], V[::k]
    cur = run["current"] * float(m["current_sign"])
    if m["current_units"] == "A_per_segment":
        r_, c_ = cfg["grid"]["current"]
        cur = cur * (cfg["active_area_cm2"] / (r_ * c_))
    run.update(t=t, t_bulk=tb, INTENSIDAD=I.astype(np.float32),
               V001=V.astype(np.float32), current=cur.astype(np.float32),
               quality_truth=q)
    return run


# ------------------------------------------------------------------- writing
def save_npz_deterministic(path, arrays):
    """np.savez-compatible, but byte-identical for identical content (fixed
    zip timestamps; numpy's own writer stamps the current time)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(arrays[name]),
                                      allow_pickle=False)
            zi = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, buf.getvalue(), compresslevel=4)
    os.replace(tmp, path)


def make_run(cfg, cell, role):
    """Generate one run in memory. role in ROLES."""
    ri = ROLES.index(role)
    if role == "cal":
        prof = profiles.build("calibration", cell["g"],
                              _ss(cfg, 2, cell["g"]), cfg)
    else:
        prof = profiles.build(cell["family"], cell["profile_idx"],
                              _ss(cfg, 1, cell["family_idx"],
                                  cell["profile_idx"]), cfg)
    p = draw_cell(cfg, cell)
    key = (cell["cell_seed"], cell["family_idx"], cell["profile_idx"],
           cell["flow_idx"])
    if role == "fault":
        eps, trig, skipped = build_episodes(
            prof, cfg, np.random.default_rng(_ss(cfg, 4, *key)))
    else:
        eps, trig, skipped = [], {f: 0 for f in FAULTS}, 0
    lab, table = labels_from(prof, eps)
    rng = np.random.default_rng(_ss(cfg, 5, *key, ri))
    sig = simulate(prof, lab, p, cfg, cell["flow_config"], rng)
    run_id = f"{cell['cell_id']}_{role}"
    meta = {
        "SYNTHETIC": True, "generator_version": GENERATOR_VERSION,
        "seed": int(cfg["seed"]), "run_id": run_id, "role": role,
        "cell_id": cell["cell_id"], "pair_id": cell["pair_id"],
        "profile_id": prof["profile_id"], "profile_family": prof["family"],
        "flow_config": cell["flow_config"], "split": cell["split"],
        "cell_params": p, "active_area_cm2": cfg["active_area_cm2"],
        "current_units": cfg["messiness"]["current_units"],
        "current_sign": cfg["messiness"]["current_sign"],
        "trigger_events": trig, "episodes_skipped_overlap": skipped,
        "config": cfg,
    }
    run = {"t": prof["t"].copy(), "load_ratio": prof["load_ratio"].astype(np.float32),
           "segment_type": prof["segment_type"], **sig, **lab}
    run = apply_messiness(run, cfg, np.random.default_rng(_ss(cfg, 6, *key, ri)))
    run["episodes_json"] = np.array(json.dumps(table))
    run["meta_json"] = np.array(json.dumps(meta, sort_keys=True))
    return run, meta, table


def _write_cell(cfg, cell, out_dir):
    rows = []
    roles = ROLES if cfg["companions"] else ("cal", "fault")
    for role in roles:
        run, meta, table = make_run(cfg, cell, role)
        save_npz_deterministic(
            os.path.join(out_dir, "runs", meta["run_id"] + ".npz"), run)
        row = {k: meta[k] for k in ("run_id", "role", "cell_id", "pair_id",
                                    "profile_id", "profile_family",
                                    "flow_config", "split")}
        row["n_samples"] = len(run["t"])
        row["duration_s"] = round(float(run["t"][-1] - run["t"][0]), 3)
        st = run["segment_type"]
        row["hold_share"] = round(float((st == "hold").mean()), 4)
        for f in FAULTS:
            row[f"triggers_{f}"] = meta["trigger_events"][f]
            row[f"episodes_{f}"] = sum(e["fault_type"] == f for e in table)
            row[f"confirmed_{f}"] = sum(e["fault_type"] == f
                                        and e["confirm_s"] is not None
                                        for e in table)
        row["skipped_overlap"] = meta["episodes_skipped_overlap"]
        rows.append(row)
    return rows


def generate(cfg, out_dir, n_jobs=1, log=print):
    """Write a dataset to `out_dir`, one run per file (chunked)."""
    from joblib import Parallel, delayed
    cells, skipped = plan(cfg)
    if not cells:
        raise ValueError("config has no cohorts / cells")
    os.makedirs(os.path.join(out_dir, "runs"), exist_ok=True)
    log(f"SYNTHETIC dataset '{cfg['name']}'  generator {GENERATOR_VERSION}  "
        f"seed {cfg['seed']}  cells {len(cells)}")
    for co in skipped:
        log(f"  cohort skipped (real load traces not present): {co}")
    res = Parallel(n_jobs=n_jobs)(
        delayed(_write_cell)(cfg, c, out_dir) for c in cells)
    rows = [r for rr in res for r in rr]
    cols = list(rows[0])
    with open(os.path.join(out_dir, "index.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    fr = [r for r in rows if r["role"] == "fault"]
    totals = {
        "cells": len(cells), "runs": len(rows),
        "samples": int(sum(r["n_samples"] for r in rows)),
        "hours": round(sum(r["duration_s"] for r in rows) / 3600.0, 2),
        "by_split": {s: sum(c["split"] == s for c in cells)
                     for s in sorted({c["split"] for c in cells})},
        "by_family": {f: sum(c["family"] == f for c in cells)
                      for f in sorted({c["family"] for c in cells})},
        "events": {f: {k: int(sum(r[f"{k}_{f}"] for r in fr))
                       for k in ("triggers", "episodes", "confirmed")}
                   for f in FAULTS},
        "events_by_split": {s: {f: int(sum(r[f"confirmed_{f}"] for r in fr
                                           if r["split"] == s))
                                for f in FAULTS}
                            for s in sorted({c["split"] for c in cells})},
        "episodes_skipped_overlap": int(sum(r["skipped_overlap"] for r in fr)),
        "cohorts_skipped": skipped,
    }
    manifest = {"SYNTHETIC": True, "generator_version": GENERATOR_VERSION,
                "seed": int(cfg["seed"]), "name": cfg["name"], "config": cfg,
                "totals": totals,
                "provenance": {k: {"tag": v[0], "value": v[1], "unit": v[2],
                                   "basis": v[3]}
                               for k, v in PROVENANCE.items()},
                "warning": "SYNTHETIC DATA. Not a recording of a real cell."}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    with open(os.path.join(out_dir, "DATA_CARD.md"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write(data_card(cfg, totals))
    log(f"  runs {totals['runs']}, samples {totals['samples']:,}, "
        f"{totals['hours']} h simulated")
    for f in FAULTS:
        e = totals["events"][f]
        log(f"  {f:<11} trigger events {e['triggers']:>6}  episodes "
            f"{e['episodes']:>6}  confirmed {e['confirmed']:>6}")
    return manifest


POWER_BAR = 30   # Sec. 13b: contributing events needed per fault


def data_card(cfg, totals):
    ev = totals["events"]
    power = "\n".join(
        f"| {f} | {ev[f]['triggers']} | {ev[f]['episodes']} | "
        f"{ev[f]['confirmed']} | "
        + " / ".join(f"{s}: {totals['events_by_split'][s][f]}"
                     for s in totals["events_by_split"]) + " | "
        + ("meets the bar" if all(totals["events_by_split"][s][f] >= POWER_BAR
                                  for s in totals["events_by_split"])
           else "**UNDERPOWERED in at least one split**") + " |"
        for f in FAULTS)
    m = cfg["messiness"]
    return f"""# DATA CARD — `{cfg['name']}`

> **SYNTHETIC DATA.** Generated by `{GENERATOR_VERSION}`, seed `{cfg['seed']}`.
> This is not a recording of a real fuel cell. Every fault in it is generated.

## Contents

- {totals['cells']} independent cells, {totals['runs']} runs,
  {totals['samples']:,} samples ({totals['hours']} h simulated at
  {cfg['dt_s']} s).
- Grids: current density {cfg['grid']['current'][0]}×{cfg['grid']['current'][1]},
  temperature {cfg['grid']['temperature'][0]}×{cfg['grid']['temperature'][1]}.
- Cells by split: {totals['by_split']}; by load-profile family:
  {totals['by_family']}.
- Each cell has a fault-free **calibration** run, a **fault** run and (if
  enabled) a fault-free **companion** run over the same load profile.
- Bulk channels emitted: `INTENSIDAD` [A], `V001` [V]. **No other bulk
  channel is modelled, and none is filled in.**
- Group IDs in `index.csv`: `cell_id`, `pair_id`, `profile_id`,
  `profile_family`, `flow_config`, `split`. Every cell appears under exactly
  one load profile{' (config asks for recurrence)' if cfg.get('cells_recur_across_profiles') else ''}.
  Check any split with `python -m hfc check-leaks`.

## Fault events (statistical power)

The project's power bar is {POWER_BAR} contributing events per fault
(§13b of the generator notes). "Confirmed" = episodes whose severity crosses
the {THRESH} threshold.

| fault | trigger events | episodes | confirmed | confirmed by split | power |
|---|---|---|---|---|---|
{power}

Episodes dropped because they would overlap an active fault (no concurrent
faults are modelled): {totals['episodes_skipped_overlap']}.

## What is grounded and what is assumed

{table_markdown()}

## Messiness switched on in this dataset

`{json.dumps(m)}`

## Known synthetic-versus-real gaps — DIAGNOSTICS, NOT TARGETS

From `docs/real_data_notes.md`. **None of these was used to fit, tune or
select anything in the generator**; matching them would turn the
false-alarm test on the real recording into a tautology.

- The real healthy template inflects and saturates with radius; this
  generator uses the plain quadratic clamping form. A free radial function
  explains 94% of the real template's variance, so ~6% of it is not radial at
  all and is absent here.
- The real recording shows two session clusters in static spread
  (CV(J) ≈ 0.38 and ≈ 0.325). The generator's CV(J) follows from the assumed
  edge-ratio bracket and is not matched to either.
- The real residual pattern is fixture-locked (consistent with contact
  resistance), not flow-locked. The generator has no fixture-specific
  residual.
- Real voltage: the recording's `V001` (~0.5 V) is taken as cell voltage;
  the CDM file's own "voltage" scalar (~1.9 V) is unreconciled. Generated
  voltage comes from a textbook polarisation curve, not from either.
- Real sampling has a start-up transient and a separate bulk clock; these
  are available here only as switchable messiness.
- Flow configuration effects are an engineering assumption: the real data
  has one session per configuration.

## What this data must NOT be used to claim

- **Any accuracy on real faults.** No fault pattern here is validated
  against a real faulted cell; none exists in this project.
- That a spatial-versus-bulk difference measured on it holds for real cells.
  The relative size of the voltage and spatial signatures is an assumption.
- That the healthy runs represent any particular real cell.
- Concurrent faults are not modelled; hydrogen and oxygen starvation differ
  only by assumed axis and voltage behaviour.
"""
