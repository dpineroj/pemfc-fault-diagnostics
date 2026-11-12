"""
Diagnostics for the Phase 5 failure -- run AFTER the pre-registered result
(run_logs/phase5.log) and changing nothing about it. Verdicts stand as
registered; this script only answers "is the failure real, and what is it?"

Three checks:

  1. PIPELINE CHECK (verify the operation, not the report). The stored final
     models are applied, through the SAME _predict path Phase 5 used, to
     SYNTHETIC grids. If that path were broken the Phase 5 result would be an
     artifact. Expect near-zero alarm rate on synthetic healthy rows.

  2. WHAT THE BULK ARM LEARNED. The bulk Model A input is (mean J, load_ratio).
     Phase 5's transform puts real mean(J) at 0.7007 +- 0.005, essentially the
     synthetic healthy value, yet the arm alarms. Sweep load_ratio at a fixed
     healthy mean(J) and print the predicted class: if the prediction flips
     with load alone, the arm learned the generator's TRIGGER RULE (sustained
     high load => drying), not a fault signature.

  3. HOW MUCH OF TRAINING LOOKED LIKE A HOLD. Share of synthetic training rows,
     by label, sitting at sustained high load.

Run:  python scripts/spatial/phase5_diagnostics.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fault_generator as F  # noqa: E402
import model_b_data as D  # noqa: E402
import phase4_model_a as MA  # noqa: E402
import phase5_false_alarm as P5  # noqa: E402


def load_model(name):
    z = np.load(os.path.join(P5.OUT_DIR, name + ".npz"), allow_pickle=False)
    return {k: z[k] for k in z.files}


def main():
    rows, cols = F.DEFAULT_GRID_SHAPE
    cache = {}

    print("=" * 76)
    print("1. PIPELINE CHECK -- stored models on SYNTHETIC rows, same _predict")
    print("=" * 76)
    for arm in F.STARVATION_ARMS:
        Xs, Xb, ys = [], [], []
        for tk in MA.TRACES:
            sp, bu, lab, _s, _g = MA.build_group(tk, 0, arm, 1.0, rows, cols,
                                                 cache)
            Xs.append(sp); Xb.append(bu); ys.append(lab)
        Xs, Xb, y = np.vstack(Xs), np.vstack(Xb), np.concatenate(ys)
        for which, X in (("spatial", Xs), ("bulk", Xb)):
            mod = load_model(f"modelA__{arm}__{which}")
            p = P5._predict(mod, X.astype(np.float64))
            h = y == "healthy"
            print(f"  Model A {arm:<12}{which:<8} in-sample accuracy "
                  f"{(p == y).mean():.3f};  alarm rate on synthetic HEALTHY "
                  f"rows {(p[h] != 'healthy').mean():.4f}  (n={h.sum():,})")
    print("  -> the prediction path is sound if these alarm rates are small.")
    print()

    print("=" * 76)
    print("2. WHAT THE BULK ARM LEARNED -- sweep load_ratio at healthy mean(J)")
    print("=" * 76)
    loads = np.linspace(0.0, 1.0, 11)
    for mj in (0.697, 0.7007):
        X = np.c_[np.full_like(loads, mj), loads]
        for arm in F.STARVATION_ARMS:
            mod = load_model(f"modelA__{arm}__bulk")
            p = P5._predict(mod, X)
            print(f"  mean(J)={mj:.4f}  starv-arm={arm:<12} load "
                  + " ".join(f"{l:.1f}:{c[:4]}" for l, c in zip(loads, p)))
    print("  -> a class that changes with load alone, at a healthy mean(J), is")
    print("     the generator's trigger rule, not a fault signature.")
    print()

    print("=" * 76)
    print("3. Synthetic training rows by label and load level (directional)")
    print("=" * 76)
    lab, load = [], []
    for tk in MA.TRACES:
        for ci in range(MA.CELLS_PER_TRACE):
            _sp, bu, lb, _s, _g = MA.build_group(tk, ci, "directional", 1.0,
                                                 rows, cols, cache)
            lab.append(lb); load.append(bu[:, 1])
    lab, load = np.concatenate(lab), np.concatenate(load)
    print(f"  {'load_ratio bin':>16}" + "".join(f"{c:>12}" for c in MA.CLASSES)
          + f"{'healthy %':>11}")
    edges = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0001]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (load >= lo) & (load < hi)
        cnt = [int((lab[m] == c).sum()) for c in MA.CLASSES]
        tot = max(sum(cnt), 1)
        print(f"  [{lo:.1f},{hi:.1f})".rjust(18)
              + "".join(f"{v:>12,}" for v in cnt)
              + f"{100 * cnt[0] / tot:>10.1f}%")
    print()
    print("  Real conditioning holds sit at sustained constant load -- an")
    print("  operating condition the dynamic-only training traces exclude.")


if __name__ == "__main__":
    main()
