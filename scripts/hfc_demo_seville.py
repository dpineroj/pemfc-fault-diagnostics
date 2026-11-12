"""
Demo: the hfc detector on the real Seville recording (healthy only).

**NOT AN INDEPENDENT TEST.** This recording has been used as a test set
before, and its failures under the old models were inspected (generator notes
Sec. 16d) before this tool was designed. A quiet result here shows the tool
does not repeat that failure on this recording; it validates nothing.

The recording contains NO faults. The only thing measured is how often the
tool raises an alarm on a healthy cell, and how often it abstains.

PROTOCOL, FIXED IN THIS FILE BEFORE IT WAS FIRST RUN. One run; no parameter
of the tool is adjusted afterwards.

  A. Within-session: for each of the four sessions, calibrate on the first
     half of the post-warm-up recording (by time), measure on the second
     half. The calibration half is never scored.
  B. Cross-session: calibrate once on the whole Normal_Flow session, measure
     on the other three (different days, different gas configurations).

Both the bulk-only default and the spatial option are run. Models are the
shipped ones (hfc_models/), trained only on synthetic data; no threshold or
feature is changed for this data. Voltage channel V001; load from INTENSIDAD.

Rates are reported over ALL scored samples, and separately for conditioning
holds and dynamic segments (holds are 78% of the recording).

Run:  python -u -W ignore scripts/hfc_demo_seville.py
"""

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from hfc import detect, features, io, models  # noqa: E402

CONFIGS = ("Normal_Flow", "Inverse_Flow", "Inverse_Air_Flow",
           "Inverse_Hydrogen_Flow")
OUT = os.path.join(REPO, "run_logs", "hfc", "seville_demo")
LABEL = "NOT AN INDEPENDENT TEST"


def holds_mask(rec):
    import extract_load_trace as E
    i = rec["i"]
    ok = np.isfinite(i)
    lr = (i - np.nanmin(i)) / (np.nanmax(i) - np.nanmin(i))
    m = np.zeros(len(i), dtype=bool)
    mg = np.zeros(int(ok.sum()), dtype=bool)
    for h in E.detect_holds(rec["t"][ok], lr[ok]):
        mg[h["start_idx"]:h["end_idx"] + 1] = True
    m[np.flatnonzero(ok)] = mg
    return m


def score(out, sel, hold):
    st = out["state"][sel]
    ans = st != "abstain"
    alarm = ans & (st != "healthy")
    h = hold[sel]
    r = {"n": int(sel.sum()), "answered": float(ans.mean()),
         "state_alarm_all": float(alarm.mean()),
         "state_alarm_of_answered": float(alarm.sum() / max(ans.sum(), 1)),
         "state_alarm_hold": float(alarm[h].mean()) if h.any() else None,
         "state_alarm_dynamic": float(alarm[~h].mean()) if (~h).any() else None,
         "answered_hold": float(ans[h].mean()) if h.any() else None,
         "answered_dynamic": float(ans[~h].mean()) if (~h).any() else None,
         "named": {s: float((st == s).mean()) for s in models.FAULTS}}
    rs, cs = np.unique(out["abstain_reason"][sel][~ans], return_counts=True)
    r["abstain_reasons"] = {str(a): int(b) for a, b in zip(rs, cs)}
    for f in models.FAULTS:
        a = out[f"ahead_{f}_alarm"][sel]
        r[f"ahead_{f}_alarm_all"] = float(a.mean())
        r[f"ahead_{f}_alarm_dynamic"] = float(a[~h].mean()) if (~h).any() else None
    return r


def main():
    print(f"hfc on the real Seville recording -- {LABEL}")
    print("The recording is healthy-only: every alarm below is a false alarm.")
    bundle = models.load_bundle(os.path.join(REPO, "hfc_models"))
    print(f"models: trained on synthetic '{bundle['trained_on']['dataset']}', "
          f"shipped modes {bundle.get('shipped_modes')}")
    recs = {}
    for c in CONFIGS:
        recs[c] = io.load({"adapter": "seville", "config": c})[0]
        r = recs[c]
        print(f"  {c}: {len(r['t']):,} samples, i "
              f"{np.nanmin(r['i']):.3f}-{np.nanmax(r['i']):.3f} A/cm2, V001 "
              f"median {np.nanmedian(r['v']):.3f} V "
              f"(unit and sign checks passed)", flush=True)
    holds = {c: holds_mask(recs[c]) for c in CONFIGS}
    res = {}
    for mode in ("bulk", "spatial"):
        # ---- A. within-session
        for c in CONFIGS:
            rec = recs[c]
            half = rec["t"] <= rec["t"][0] + 0.5 * (rec["t"][-1] - rec["t"][0])
            src = rec if mode == "spatial" else dict(rec, grid=None)
            cal = features.calibrate(src, half)
            out, info = detect.run_recording(rec, cal, bundle, mode)
            res[f"A|{mode}|{c}"] = dict(score(out, ~half, holds[c]),
                                        cal_load=[cal.d["i_min"], cal.d["i_max"]],
                                        mode_note=info["mode_note"])
            rep = detect.summarise(rec, cal, out, info, bundle,
                                   {"label": LABEL, "protocol": "A within-session"})
            detect.write_outputs(os.path.join(OUT, f"A_{mode}"), rec, out, rep)
        # ---- B. cross-session
        src = recs["Normal_Flow"] if mode == "spatial" else dict(
            recs["Normal_Flow"], grid=None)
        cal = features.calibrate(src)
        for c in CONFIGS[1:]:
            out, info = detect.run_recording(recs[c], cal, bundle, mode)
            res[f"B|{mode}|{c}"] = dict(
                score(out, np.ones(len(recs[c]["t"]), dtype=bool), holds[c]),
                cal_load=[cal.d["i_min"], cal.d["i_max"]])
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"label": LABEL, "results": res}, fh, indent=1)

    def pct(x):
        return "   n/a" if x is None else f"{100 * x:6.2f}"
    for proto, title in (("A", "A. WITHIN-SESSION (calibrate first half, "
                               "score second half)"),
                         ("B", "B. CROSS-SESSION (calibrate on Normal_Flow, "
                               "score the other three)")):
        print()
        print("=" * 100)
        print(f"{title}  -- {LABEL}")
        print("=" * 100)
        print(f"  {'mode':<8}{'session':<23}{'answered%':>10}{'FALSE ALARM %':>15}"
              f"{'hold':>8}{'dynamic':>9}{'ans.dyn%':>10}   fault-ahead alarm % "
              f"(starv/flood/dry)")
        for mode in ("bulk", "spatial"):
            for c in CONFIGS:
                r = res.get(f"{proto}|{mode}|{c}")
                if r is None:
                    continue
                print(f"  {mode:<8}{c:<23}{pct(r['answered']):>10}"
                      f"{pct(r['state_alarm_all']):>15}"
                      f"{pct(r['state_alarm_hold']):>8}"
                      f"{pct(r['state_alarm_dynamic']):>9}"
                      f"{pct(r['answered_dynamic']):>10}   "
                      + " / ".join(pct(r[f'ahead_{f}_alarm_all']).strip()
                                   for f in models.FAULTS)
                      + "   abstain: "
                      + ", ".join(f"{k} {v}" for k, v in
                                  r["abstain_reasons"].items()))
    print()
    print("FALSE ALARM % = share of scored samples where the state output "
          "names a fault. Abstentions are not alarms.")
    print(f"{LABEL}. There are no faults in this recording, so nothing here "
          "measures detection.")


if __name__ == "__main__":
    main()
