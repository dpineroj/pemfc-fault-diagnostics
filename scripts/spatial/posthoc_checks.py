"""
POST-HOC (Sec. 17g volume, Sec. 17j item 4). LABEL-ONLY checks: no grids,
no fits.

  --check companions   every fault-free companion must land on the same side
                       of every fold as its faulted twin, in S0-S3, for every
                       Model B configuration. Must pass before any v2 fit.
  --check drying       drying volume at 40 cells per trace against the Sec.
                       17g target (>= 30 distinct contributing trigger events,
                       >= 5 from each trace). Reports; decides nothing else.

Run:  python -u -W ignore scripts/spatial/posthoc_checks.py --check companions
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_b_data as D  # noqa: E402
import phase4_model_b as MB  # noqa: E402
import posthoc_lib as PL  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402


def companions():
    print("Sec. 17j item 4 -- companion/twin split check (label-only)")
    fail = 0
    for cfg in MB.CFG_ORDER:
        gt, cell, comp, _y = PL.index_b(cfg, companions=True)
        g = PL.gkey_of(gt, cell)
        groups = np.unique(g)
        assert all(comp[g == k].any() and (~comp[g == k]).any()
                   for k in groups), "a group lacks a twin or a companion"
        for s in PL.SPLITS:
            nf = 0
            for tr, te in PL.fold_iter(s, gt, cell):
                nf += 1
                side = np.zeros(len(g), dtype=int)
                side[tr] = 1
                side[te] = 2
                for k in groups:
                    m = g == k
                    tw, co = set(side[m & ~comp]), set(side[m & comp])
                    if len(tw) != 1 or tw != co:
                        fail += 1
                        print(f"  VIOLATION {cfg} {s} group {k}: twin {tw} "
                              f"companion {co}")
            print(f"  {cfg:<18} {s}: {nf} folds, {len(groups)} groups, "
                  f"{int(comp.sum()):,} companion + {int((~comp).sum()):,} "
                  f"twin windows -- every pair on one side")
    print()
    print("RESULT: " + ("PASS -- no fold separates a companion from its twin."
                        if not fail else f"FAIL -- {fail} violations."))
    sys.exit(1 if fail else 0)


def _one(tk, ci, h, L):
    d = D.load_labels(tk, ci)
    wi = D.window_index(d, h, L)
    k = wi["keep"]
    y, tt = wi["y"][k], wi["t_target"][k]
    ev = set()
    inst = 0
    for e in d["episodes"]:
        if e["fault_type"] != "drying":
            continue
        m = (y == "drying") & (tt >= e["onset_start_s"]) & (tt <= e["episode_end_s"])
        if m.any():
            ev.add(f"{tk}|{e['onset_start_s']:.1f}")
            inst += 1
    trig = {f"{tk}|{e['onset_start_s']:.1f}" for e in d["episodes"]
            if e["fault_type"] == "drying"}
    return tk, ci, ev, inst, trig, {c: int((y == c).sum()) for c in D.CLASSES}


def drying():
    h, L = D.CONFIGS["drying"]["h"], D.CONFIGS["drying"]["L"]
    print("Sec. 17g -- drying volume at 40 cells per trace (label-only)")
    print("TARGET (fixed in Sec. 17g): >= 30 distinct contributing drying "
          "trigger events, >= 5 from each trace")
    res = Parallel(n_jobs=PL.N_JOBS)(
        delayed(_one)(tk, ci, h, L) for tk in PL.TRACE_KEYS for ci in range(40))
    met = True
    tot = 0
    for ncell in (5, 10, 20, 40):
        line = []
        total = 0
        for tk in PL.TRACE_KEYS:
            ev = set().union(*[r[2] for r in res if r[0] == tk and r[1] < ncell])
            line.append(f"{tk} {len(ev)}")
            total += len(ev)
        print(f"  first {ncell:>2} cells/trace: contributing events "
              + ", ".join(line) + f"  -> total {total}")
    print()
    for tk in PL.TRACE_KEYS:
        ev = set().union(*[r[2] for r in res if r[0] == tk])
        trig = set().union(*[r[4] for r in res if r[0] == tk])
        inst = sum(r[3] for r in res if r[0] == tk)
        tot += len(ev)
        met &= len(ev) >= 5
        print(f"  {tk:<9} trigger events {len(trig):>2}; contributing "
              f"{len(ev):>2}; (cell, event) instances {inst}")
    met &= tot >= 30
    cls = {c: sum(r[5][c] for r in res) for c in D.CLASSES}
    print(f"  kept windows by target, 160 groups: {cls}")
    print()
    print(f"RESULT: {tot} distinct contributing events -> TARGET "
          + ("MET." if met else "NOT MET -- the comparison is run once and "
             "reported as UNDERPOWERED; the count is not raised."))
    PL.dump_json(os.path.join(PL.OUT, "drying_volume.json"),
                 {"contributing_total": tot, "target_met": bool(met),
                  "windows": cls})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, choices=("companions", "drying"))
    a = ap.parse_args()
    companions() if a.check == "companions" else drying()
