"""
Generation volume for Model B -- measured AFTER the drying minimum was
pre-registered (Sec. 13b, commit fc3fb28: >= 30 distinct drying trigger events
across the four traces, >= 5 in every trace).

Answers, in order:
  1. Are episode triggers a deterministic function of the load trace? (Sec.
     13b-i -- verified here, not assumed.) Checked two ways: the trigger
     finders are run on each trace directly, and the episodes generated for
     five different seeds per trace are compared for identical (fault, onset).
  2. What do the four traces yield, per fault: trigger events, and
     threshold-crossing episodes per cell.
  3. Does drying meet the pre-registered minimum?
  4. Window class balance per Model B configuration (Sec. 13g), including
     precursor-visible vs blind positives and what each exclusion rule drops.

Labels only -- no grids are built and no model is fit.

Run:  python scripts/spatial/measure_generation_volume.py
"""

import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fault_generator as F  # noqa: E402
import model_b_data as D  # noqa: E402

MIN_DRYING_EVENTS = 30      # pre-registered, Sec. 13b
MIN_PER_TRACE = 5           # pre-registered, Sec. 13b


def trace_events(trace_key):
    z = np.load(os.path.join(F.PROCESSED_DIR, D.TRACES[trace_key]))
    t, lr, seg = z["t_seconds"], z["load_ratio"], z["segment_id"]
    out = {"starvation": [], "flooding": [], "drying": []}
    for s in np.unique(seg):
        m = seg == s
        ts, ls = t[m], lr[m]
        out["starvation"] += [e["start_s"] for e in
                              F.find_ramp_events(ts, ls, F.RAMP_THRESHOLD)]
        out["flooding"] += [e["start_s"] for e in F.find_dwell_events(
            ts, ls, F.LOW_LOAD_THRESH, "low", F.MIN_DWELL_S)]
        out["drying"] += [e["start_s"] for e in F.find_dwell_events(
            ts, ls, F.HIGH_LOAD_THRESH, "high", F.MIN_DWELL_S)]
    return out, len(t), int(len(np.unique(seg)))


def main():
    print("Generation volume for Model B (labels only)")
    print(f"pre-registered drying minimum: {MIN_DRYING_EVENTS} distinct events, "
          f">= {MIN_PER_TRACE} per trace")
    print()

    labels = {}
    for tk in D.TRACES:
        for ci in range(D.CELLS_PER_TRACE):
            labels[(tk, ci)] = D.load_labels(tk, ci)
    print(f"label cache: {len(labels)} (trace, cell) groups")
    print()

    # ---- 1. determinism -----------------------------------------------------
    print("=" * 76)
    print("1. Are triggers a deterministic function of the load trace?")
    print("=" * 76)
    all_det = True
    ev_by_trace = {}
    for tk in D.TRACES:
        ev, n_t, n_seg = trace_events(tk)
        ev_by_trace[tk] = ev
        ref = None
        same = True
        for ci in range(D.CELLS_PER_TRACE):
            got = sorted((e["fault_type"], round(e["onset_start_s"], 3))
                         for e in labels[(tk, ci)]["episodes"])
            if ref is None:
                ref = got
            same &= (got == ref)
        direct = sorted((f, round(s, 3)) for f, v in ev.items() for s in v)
        match_direct = (direct == ref)
        all_det &= same and match_direct
        print(f"  {tk:>9}: T={n_t:5d} segs={n_seg}  episodes identical across "
              f"{D.CELLS_PER_TRACE} seeds: {same}   == trigger finders on the "
              f"bare trace: {match_direct}")
    print(f"  -> deterministic in the trace: {all_det}")
    print("     (seeds change the drawn cell and the severity trajectory, not")
    print("      which events fire or when)")
    print()

    # ---- 2. yield per trace ---------------------------------------------------
    print("=" * 76)
    print("2. Yield per trace: trigger events, and crossing episodes per cell")
    print("=" * 76)
    print(f"  {'trace':>9} {'fault':>11} {'events':>7}   crossing 0.5, per cell")
    for tk in D.TRACES:
        for f in D.FAULTS:
            cross = [sum(1 for e in labels[(tk, ci)]["episodes"]
                         if e["fault_type"] == f
                         and e["precursor_window_s"] is not None)
                     for ci in range(D.CELLS_PER_TRACE)]
            print(f"  {tk:>9} {f:>11} {len(ev_by_trace[tk][f]):>7}   {cross}")
    print()

    # ---- 3. drying vs the pre-registered minimum -----------------------------
    print("=" * 76)
    print("3. Drying against the pre-registered minimum (Sec. 13b)")
    print("=" * 76)
    per_trace = {tk: len(ev_by_trace[tk]["drying"]) for tk in D.TRACES}
    total = sum(per_trace.values())
    print(f"  distinct drying trigger events per trace: {per_trace}")
    print(f"  total distinct events: {total}   (minimum {MIN_DRYING_EVENTS})")
    print(f"  smallest trace: {min(per_trace.values())}   "
          f"(minimum {MIN_PER_TRACE})")
    ever_cross = {}
    for tk in D.TRACES:
        s = set()
        for ci in range(D.CELLS_PER_TRACE):
            for e in labels[(tk, ci)]["episodes"]:
                if (e["fault_type"] == "drying"
                        and e["precursor_window_s"] is not None):
                    s.add(round(e["onset_start_s"], 3))
        ever_cross[tk] = len(s)
    print(f"  events that cross threshold in >= 1 cell, per trace: {ever_cross}"
          f"  total {sum(ever_cross.values())}")
    met = total >= MIN_DRYING_EVENTS and min(per_trace.values()) >= MIN_PER_TRACE
    print(f"  MINIMUM MET: {met}")
    if not met:
        print("  -> per Sec. 13b-ii the minimum is NOT lowered; drying is run on")
        print("     what exists and reported as UNDERPOWERED / indicative.")
    print()

    # ---- 4. window class balance ---------------------------------------------
    print("=" * 76)
    print("4. Window class balance per configuration (Sec. 13g)")
    print("=" * 76)
    for cfg, hl in D.CONFIGS.items():
        tot = Counter()
        vis = Counter()
        drops = Counter()
        events = set()
        for (tk, ci), d in labels.items():
            wi = D.window_index(d, hl["h"], hl["L"])
            k = wi["keep"]
            tot.update(wi["y"][k].tolist())
            vis.update(wi["y"][k & wi["visible"]].tolist())
            drops.update(wi["drops"])
            own = D.OWN_FAULT[cfg]
            pos = k & (wi["y"] == own)
            tt = wi["t_target"][pos]
            for e in d["episodes"]:
                if e["fault_type"] == own and np.any(
                        (tt >= e["onset_start_s"]) & (tt <= e["episode_end_s"])):
                    events.add((tk, round(e["onset_start_s"], 3)))
        own = D.OWN_FAULT[cfg]
        print(f"  --- {cfg}  (h={hl['h']} s, L={hl['L']} s) ---")
        print("      kept windows by target: "
              + ", ".join(f"{c}={tot.get(c, 0):,}" for c in D.CLASSES))
        print(f"      own fault '{own}': {tot.get(own, 0):,} positives, of which "
              f"precursor-visible {vis.get(own, 0):,}, "
              f"blind {tot.get(own, 0) - vis.get(own, 0):,}; "
              f"from {len(events)} distinct load-trace events")
        print(f"      drops: seam/segment {drops['seam_or_segment']:,}, "
              f"time-gap {drops['time_gap']:,}, already-confirmed "
              f"{drops['already_confirmed_in_window']:,}, ambiguous-target "
              f"{drops['ambiguous_target']:,}; kept {drops['kept']:,} of "
              f"{drops['total_t']:,}")
        print()


if __name__ == "__main__":
    main()
