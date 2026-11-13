"""
POST-HOC (Sec. 17e-3d). Does adding fault-free companions cost detection?

Paired measure fixed in Sec. 17e: recall on own-fault positives, v2 minus v1,
under S3, on the IDENTICAL positive windows (v2 leaves every v1 faulted run
bit-identical; asserted here), 95% group-bootstrap CI over (trace, cell)
groups, per fault and arm. "Costs detection" iff the CI lies wholly below 0.

Reads result files only; fits nothing.

Run:  python -u -W ignore scripts/spatial/posthoc_ws3d_paired.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fault_generator as F  # noqa: E402
import model_b_data as D  # noqa: E402
import posthoc_lib as PL  # noqa: E402


def main():
    rng = np.random.default_rng(PL.BOOT_SEED)
    print("POST-HOC Sec. 17e-3d -- paired recall, v2 minus v1, S3, identical "
          "positive windows")
    print(f"  {'fault':<11}{'starv-arm':<13}{'input':<9}{'recall v1':>10}"
          f"{'recall v2':>10}{'diff':>8}  95% CI            FPR v1 -> v2     "
          f"verdict")
    out = {}
    for cfg in ("starvation", "flooding", "drying"):
        f = D.OWN_FAULT[cfg]
        for arm in F.STARVATION_ARMS:
            a = np.load(os.path.join(PL.OUT, "model_b_v1",
                                     f"{cfg}__{arm}__multi.npz"))
            b = np.load(os.path.join(PL.OUT, "model_b_v2",
                                     f"{cfg}__{arm}__multi.npz"))
            tw = ~b["comp"]
            assert (b["y"][tw] == a["y"]).all(), "v2 twins differ from v1"
            assert (b["gt"][tw] == a["gt"]).all()
            assert (b["cell"][tw] == a["cell"]).all()
            g = PL.gkey_of(a["gt"], a["cell"])
            groups = np.unique(g)
            pos = a["y"] == f
            for key, nm in (("psp", "spatial"), ("pbu", "bulk")):
                h1 = a[f"S3__{key}"] == f
                h2 = b[f"S3__{key}"][tw] == f
                c = np.array([[(pos & h1 & (g == k)).sum(),
                               (pos & h2 & (g == k)).sum(),
                               (pos & (g == k)).sum()] for k in groups], float)
                r1, r2 = c[:, 0].sum() / c[:, 2].sum(), c[:, 1].sum() / c[:, 2].sum()
                bs = []
                for _ in range(1000):
                    pk = rng.integers(0, len(groups), len(groups))
                    n = c[pk, 2].sum()
                    if n:
                        bs.append((c[pk, 1].sum() - c[pk, 0].sum()) / n)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                fp1 = float((~pos & h1).sum() / (~pos).sum())
                yb = b["y"]
                fp2 = float(((yb != f) & (b[f"S3__{key}"] == f)).sum()
                            / (yb != f).sum())
                v = ("COSTS DETECTION" if hi < 0 else
                     "improves" if lo > 0 else "no measurable cost")
                print(f"  {cfg:<11}{arm:<13}{nm:<9}{r1:>10.3f}{r2:>10.3f}"
                      f"{r2 - r1:>+8.3f}  [{lo:+.3f},{hi:+.3f}]  "
                      f"{fp1:.4f} -> {fp2:.4f}   {v}")
                out[f"{cfg}|{arm}|{nm}"] = {
                    "recall_v1": r1, "recall_v2": r2, "ci": [lo, hi],
                    "fpr_v1": fp1, "fpr_v2_incl_companions": fp2, "verdict": v}
    PL.dump_json(os.path.join(PL.OUT, "ws3d_paired_summary.json"), out)
    print("  FPR v2 is over v1's negatives PLUS the companion windows "
          "(unpaired).")


if __name__ == "__main__":
    main()
