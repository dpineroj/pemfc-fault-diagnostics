"""
Run the detector on a Recording: state, fault-ahead alarms, input-quality
flags, and a summary report.

The tool ABSTAINS rather than alarms when it should not answer: a required
channel is missing, sampling is too slow for a capability, load is outside
the calibrated range, a feature is outside the training envelope, or the
sample sits in a gap.

Output is a screening aid. Real-fault accuracy is unverified.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np

from . import UNVERIFIED, __version__, features, io
from .models import AHEAD, FAULTS, STATES, decide


def choose_mode(rec, cal, bundle, want=None):
    """
    Bulk-only is the default. The spatial variant is used only on request,
    and only if the recording and the calibration both carry a grid.
    Returns (mode, note).
    """
    shipped = bundle.get("shipped_modes", ["bulk"])
    can = rec.get("grid") is not None and cal.has_spatial
    if want == "spatial":
        if not can:
            raise io.ContractError("spatial mode requested but the recording "
                                   "or the calibration has no current grid")
        if "spatial" not in shipped:
            return "spatial", ("spatial variant was NOT shipped (it did not "
                               "earn its place on held-out data); used on "
                               "explicit request")
        return "spatial", ""
    if can and "spatial" in shipped:
        return "bulk", ("bulk-only is the default; a grid is present and the "
                        "spatial variant is available with --mode spatial")
    return "bulk", ""


def run_recording(rec, cal, bundle, mode=None):
    """Returns (per-sample dict of arrays, info dict)."""
    if rec["v"] is None:
        raise io.ContractError(
            "the data contract needs cell voltage and stack current (or "
            "current density); this recording has no voltage")
    n = len(rec["t"])
    mode, note = choose_mode(rec, cal, bundle, mode)
    if mode == "bulk":
        rec = dict(rec, grid=None)
    dt = features.sampling(rec)
    info = {"mode": mode, "mode_note": note, "median_dt_s": dt,
            "capabilities": {}, "warnings": []}
    out = {"t": rec["t"], "state": np.full(n, "abstain", dtype="U12"),
           "abstain_reason": np.full(n, "", dtype="U32")}
    for s in STATES:
        out[f"p_{s}"] = np.full(n, np.nan)
    for f in FAULTS:
        out[f"ahead_{f}_prob"] = np.full(n, np.nan)
        out[f"ahead_{f}_alarm"] = np.zeros(n, dtype=bool)
        out[f"ahead_{f}_status"] = np.full(n, "disabled", dtype="U32")

    if not np.isfinite(dt) or dt > features.MAX_DT_LEVEL:
        out["abstain_reason"][:] = "sampling_too_slow"
        info["capabilities"] = {"state": "disabled: sampling slower than "
                                         f"{features.MAX_DT_LEVEL} s",
                                "ahead": "disabled"}
        info["warnings"].append(
            f"median sampling interval {dt:.2f} s exceeds the contract "
            f"maximum; the tool abstains on every sample")
        return out, info
    slow = dt > features.MAX_DT_FULL
    state_task = "state_level" if slow else "state"
    info["capabilities"]["state"] = (
        "level-only classifier (sampling slower than 1 Hz)" if slow else "full")
    info["capabilities"]["ahead"] = (
        f"disabled: fault-ahead needs ~1 Hz, data is {dt:.2f} s" if slow
        else "full")
    if slow:
        info["warnings"].append(
            f"sampling interval {dt:.2f} s: fault-ahead prediction disabled, "
            f"state classification uses level features only")
    tasks = [state_task] + ([] if slow else [f"ahead/{f}" for f in FAULTS])
    Ls = sorted({bundle["models"][f"{t}|{mode}"]["L"] for t in tasks},
                key=lambda v: -1 if v is None else v)
    fm = features.compute_multi(rec, cal, mode, Ls)
    for task in tasks:
        m = bundle["models"][f"{task}|{mode}"]
        X, names, reason = fm[m["L"]]
        assert names == m["features"], "feature order differs from the model"
        ok = reason == ""
        lab = np.full(n, "abstain", dtype="U12")
        p = np.full((n, len(m["classes"])), np.nan)
        why = reason.copy()
        if ok.any():
            l_, p_, _s, ood = decide(m, X[ok])
            lab[ok], p[ok] = l_, p_
            w = why[ok]
            w[ood] = "outside_training_envelope"
            why[ok] = w
        if task.startswith("state"):
            out["state"] = lab
            out["abstain_reason"] = why
            for s in STATES:
                out[f"p_{s}"] = p[:, m["classes"].index(s)]
        else:
            f = task.split("/")[1]
            out[f"ahead_{f}_prob"] = p[:, m["classes"].index(f)]
            out[f"ahead_{f}_alarm"] = lab == f
            out[f"ahead_{f}_status"] = np.where(why == "", "ok", why)
    return out, info


def _events(mask, t):
    """Contiguous True runs -> list of (start_s, end_s)."""
    ev, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            ev.append((float(t[i]), float(t[j])))
            i = j + 1
        else:
            i += 1
    return ev


def summarise(rec, cal, out, info, bundle, extra=None):
    n = len(out["t"])
    st = out["state"]
    reasons, counts = np.unique(out["abstain_reason"][st == "abstain"],
                                return_counts=True)
    rep = {
        "tool": f"hfc {__version__}", "statement": UNVERIFIED,
        "recording": rec["name"], "input_is_synthetic":
            bool(rec["meta"].get("SYNTHETIC", False)),
        "n_samples": n, "duration_s": float(out["t"][-1] - out["t"][0]),
        **info,
        "calibration": {"n_samples": cal.d["n_samples"],
                        "load_range_A_cm2": [cal.d["i_min"], cal.d["i_max"]],
                        "spatial": cal.has_spatial},
        "models_trained_on": bundle["trained_on"],
        "state_share": {s: float((st == s).mean())
                        for s in STATES + ("abstain",)},
        "answered_share": float((st != "abstain").mean()),
        "abstain_reasons": {str(r): int(c) for r, c in zip(reasons, counts)},
        "state_alarm_events": {s: len(_events(st == s, out["t"]))
                               for s in FAULTS},
        "ahead": {f: {"horizon_s": AHEAD[f]["h"],
                      "alarm_share": float(out[f"ahead_{f}_alarm"].mean()),
                      "alarm_events": len(_events(out[f"ahead_{f}_alarm"],
                                                  out["t"])),
                      "answered_share": float(
                          (out[f"ahead_{f}_status"] == "ok").mean())}
                  for f in FAULTS},
    }
    if extra:
        rep.update(extra)
    return rep


def report_markdown(rep):
    L = [f"# hfc detector report — `{rep['recording']}`", "",
         f"> **{rep['statement']}**", ""]
    for k in ("label", ):
        if rep.get(k):
            L += [f"> **{rep[k]}**", ""]
    L += [f"- input is synthetic: **{rep['input_is_synthetic']}**",
          f"- samples: {rep['n_samples']:,} over {rep['duration_s']:.0f} s; "
          f"median sampling interval {rep['median_dt_s']:.2f} s",
          f"- mode: **{rep['mode']}**"
          + (f" ({rep['mode_note']})" if rep["mode_note"] else ""),
          f"- capabilities: state — {rep['capabilities'].get('state')}; "
          f"fault-ahead — {rep['capabilities'].get('ahead')}",
          f"- calibration: {rep['calibration']['n_samples']:,} healthy "
          f"samples, load {rep['calibration']['load_range_A_cm2'][0]:.3f}–"
          f"{rep['calibration']['load_range_A_cm2'][1]:.3f} A/cm², "
          f"spatial baseline: {rep['calibration']['spatial']}",
          f"- models trained on: synthetic dataset "
          f"`{rep['models_trained_on']['dataset']}` "
          f"({rep['models_trained_on']['generator_version']}, seed "
          f"{rep['models_trained_on']['seed']})", ""]
    for w in rep["warnings"]:
        L.append(f"**Warning:** {w}")
    L += ["", "## State", "",
          f"Answered on {rep['answered_share']:.1%} of samples.", "",
          "| state | share of samples | alarm events |", "|---|---|---|"]
    for s, v in rep["state_share"].items():
        L.append(f"| {s} | {v:.2%} | {rep['state_alarm_events'].get(s, '')} |")
    L += ["", "## Fault-ahead alarms", "",
          "| fault | horizon | share of samples alarmed | alarm events | "
          "answered |", "|---|---|---|---|---|"]
    for f, a in rep["ahead"].items():
        L.append(f"| {f} | {a['horizon_s']} s | {a['alarm_share']:.2%} | "
                 f"{a['alarm_events']} | {a['answered_share']:.1%} |")
    L += ["", "## Abstentions", ""]
    if rep["abstain_reasons"]:
        L += ["| reason | samples |", "|---|---|"]
        L += [f"| {k} | {v:,} |" for k, v in rep["abstain_reasons"].items()]
    else:
        L.append("None.")
    L += ["", "An abstention means the input was outside what the tool can "
          "judge. It is not a finding about the cell."]
    return "\n".join(L) + "\n"


def write_outputs(out_dir, rec, out, rep):
    os.makedirs(out_dir, exist_ok=True)
    stem = rec["name"].replace("/", "_").replace("\\", "_")
    cols = [k for k in out]
    with open(os.path.join(out_dir, f"{stem}_predictions.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for k in range(len(out["t"])):
            w.writerow([(f"{out[c][k]:.5g}" if out[c].dtype.kind == "f"
                         else out[c][k]) for c in cols])
    with open(os.path.join(out_dir, f"{stem}_report.json"), "w",
              encoding="utf-8", newline="\n") as fh:
        json.dump(rep, fh, indent=1, default=float)
    with open(os.path.join(out_dir, f"{stem}_report.md"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write(report_markdown(rep))
    return stem
