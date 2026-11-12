"""
Label-only leak checker. Reads index.csv (group IDs only); no signal arrays.

A split leaks if two sides share a cell, a faulted/companion pair, or a load
profile. This project lost its "held-out cell" claim once to exactly this
(generator notes Sec. 17b): cells were seeded without a trace term, so the
same five cells sat on both sides of every fold and nothing checked.
"""

from __future__ import annotations

import csv
import json
import os

KEYS = ("cell_id", "pair_id", "profile_id")


def read_index(dataset_dir):
    with open(os.path.join(dataset_dir, "index.csv"), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def check(dataset_dir, split_file=None, keys=KEYS):
    """
    Returns (ok, report dict). Sides come from index.csv's `split` column, or
    from `split_file`: JSON {"side name": [run_id, ...], ...}.
    """
    rows = read_index(dataset_dir)
    by_run = {r["run_id"]: r for r in rows}
    if split_file:
        with open(split_file, encoding="utf-8") as fh:
            sides = json.load(fh)
        missing = [r for ids in sides.values() for r in ids if r not in by_run]
        if missing:
            raise KeyError(f"run ids not in the dataset: {missing[:5]}")
        dup = set()
        seen = set()
        for ids in sides.values():
            dup |= seen & set(ids)
            seen |= set(ids)
        side_rows = {s: [by_run[i] for i in ids] for s, ids in sides.items()}
    else:
        dup = set()
        side_rows = {}
        for r in rows:
            side_rows.setdefault(r["split"], []).append(r)
    names = sorted(side_rows)
    report = {"sides": {s: len(side_rows[s]) for s in names},
              "runs_on_two_sides": sorted(dup), "shared": {}}
    ok = not dup
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for k in keys:
                sa = {r[k] for r in side_rows[a]}
                sb = {r[k] for r in side_rows[b]}
                shared = sorted(sa & sb)
                report["shared"][f"{a}~{b}:{k}"] = shared
                ok &= not shared
    # a cell under more than one profile is not a split leak by itself, but it
    # is what made the old splits leak; report it.
    prof = {}
    for r in rows:
        if r["role"] != "cal":
            prof.setdefault(r["cell_id"], set()).add(r["profile_id"])
    report["cells_under_multiple_profiles"] = sorted(
        c for c, p in prof.items() if len(p) > 1)
    report["ok"] = bool(ok)
    return bool(ok), report


def format_report(report):
    out = ["LEAK CHECK (label-only): "
           + ("PASS" if report["ok"] else "FAIL -- the split leaks"),
           "  runs per side: " + ", ".join(f"{k}={v}"
                                           for k, v in report["sides"].items())]
    if report["runs_on_two_sides"]:
        out.append(f"  runs on two sides: {report['runs_on_two_sides'][:8]}")
    for k, v in report["shared"].items():
        out.append(f"  shared {k}: {len(v)}" + (f"  e.g. {v[:4]}" if v else ""))
    n = len(report["cells_under_multiple_profiles"])
    out.append(f"  cells appearing under more than one load profile: {n}"
               + ("  (a config asked for this; any split by profile alone "
                  "will leak cells)" if n else ""))
    return "\n".join(out)
