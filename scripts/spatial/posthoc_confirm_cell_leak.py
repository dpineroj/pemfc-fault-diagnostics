"""
POST-HOC (Sec. 17). Label-only: no grids, no fits.

Confirms whether simulated cells are shared between train and test folds in
the two splits used by Model A and Model B:

  "cell"  split -- GroupKFold(4) on (trace, cell)
  "trace" split -- LeaveOneGroupOut on trace

A "cell" is its healthy-parameter draw (c, r0, w, temporal_ratio,
starvation_axis_rad). The generator draws it from base_seed + 100003, and
both models call it with base_seed = BASE_SEED + 1000 * cell_idx -- no trace
term. This script checks the consequence directly rather than trusting that
reading of the code.

Run:  python -u scripts/spatial/posthoc_confirm_cell_leak.py
"""

import os
import sys

import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_b_data as D  # noqa: E402


def cell_key(p):
    return tuple(round(float(p[k]), 12) for k in sorted(p))


def main():
    params = {}
    for tk in D.TRACES:
        for ci in range(D.CELLS_PER_TRACE):
            params[(tk, ci)] = cell_key(D.load_labels(tk, ci)["params"])

    print("1. Are cell parameters identical across traces for a given cell_idx?")
    for ci in range(D.CELLS_PER_TRACE):
        keys = {params[(tk, ci)] for tk in D.TRACES}
        print(f"   cell_idx {ci}: {len(keys)} distinct parameter set(s) "
              f"across {len(D.TRACES)} traces")
    n_distinct = len(set(params.values()))
    print(f"   distinct cells overall: {n_distinct} "
          f"(groups: {len(params)})")
    print()

    for cfg in ("starvation_ctrl30", "starvation", "drying"):
        h, L = D.CONFIGS[cfg]["h"], D.CONFIGS[cfg]["L"]
        g_cell, g_trace = [], []
        for tk in D.TRACES:
            for ci in range(D.CELLS_PER_TRACE):
                n = int(D.window_index(D.load_labels(tk, ci), h, L)["keep"].sum())
                g_cell += [f"{tk}|{ci}"] * n
                g_trace += [tk] * n
        g_cell = np.array(g_cell, dtype=object)
        g_trace = np.array(g_trace, dtype=object)
        X = np.zeros((len(g_cell), 1))
        print(f"2. Fold composition, Model B config '{cfg}' "
              f"({len(g_cell):,} windows)")
        for name, splitter, groups in (
                ("cell  split (GroupKFold on trace|cell)",
                 GroupKFold(n_splits=4), g_cell),
                ("trace split (LeaveOneGroupOut on trace)",
                 LeaveOneGroupOut(), g_trace)):
            print(f"   {name}")
            for fi, (tr, te) in enumerate(splitter.split(X, None, groups)):
                te_groups = sorted(set(g_cell[te]))
                tr_groups = sorted(set(g_cell[tr]))
                te_cells = {params[(g.split("|")[0], int(g.split("|")[1]))]
                            for g in te_groups}
                tr_cells = {params[(g.split("|")[0], int(g.split("|")[1]))]
                            for g in tr_groups}
                te_tr = {g.split("|")[0] for g in te_groups}
                tr_tr = {g.split("|")[0] for g in tr_groups}
                print(f"     fold {fi}: test cells {len(te_cells)}, of which "
                      f"also in training {len(te_cells & tr_cells)};  "
                      f"test traces {len(te_tr)}, of which also in training "
                      f"{len(te_tr & tr_tr)}")
        print()


if __name__ == "__main__":
    main()
