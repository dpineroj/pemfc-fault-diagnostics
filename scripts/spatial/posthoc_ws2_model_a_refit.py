"""
POST-HOC (Sec. 17d). Verification only.

Re-fits the three uncapped Model A cells that have no convergence record
(directional arm, scales 0.00 / 0.25 / 0.50; resumed in Sec. 12j from
processes that died), writing a convergence record for every fit, and
compares the re-fit with the persisted predictions.

The persisted Sec. 12j files are READ, never written. The capped run remains
the quoted Model A result whatever this prints.

Run:  python -u -W ignore scripts/spatial/posthoc_ws2_model_a_refit.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phase4_model_a as MA  # noqa: E402
import posthoc_lib as PL  # noqa: E402

OUT = os.path.join(PL.OUT, "model_a_refit")
SCALES = (0.0, 0.25, 0.5)
ARM = "directional"
TOL = 0.001


def main():
    print("POST-HOC Sec. 17d -- Model A convergence-record re-fit "
          "(verification only)")
    all_fits, rows = [], []
    for scale in SCALES:
        path = os.path.join(OUT, f"{ARM}_scale{scale:.2f}.npz")
        if os.path.exists(path):
            z = dict(np.load(path, allow_pickle=False))
            print(f"[on disk] scale {scale}", flush=True)
        else:
            ds = PL.assemble_a(ARM, scale, cap=None)
            z = {"y": ds["y"].astype("U12"), "gt": ds["gt"].astype("U12"),
                 "cell": ds["cell"]}
            fits = []
            for split, name in (("S0", "cell"), ("S1", "trace")):
                for which, X in (("spatial", ds["Xs"]), ("bulk", ds["Xb"])):
                    p, f = PL.cv_predict(X, ds["y"], ds["gt"], ds["cell"],
                                         split, "A",
                                         f"A/{ARM}/{scale}/{name}/{which}")
                    z[f"{name}__{'psp' if which == 'spatial' else 'pbu'}"] = \
                        p.astype("U12")
                    fits += f
                print(f"[done] scale {scale} split {name}", flush=True)
            z["fits"] = np.array(json.dumps(fits))
            PL.save_npz(path, **z)
            del ds
        fits = json.loads(str(z["fits"]))
        all_fits += fits
        old = np.load(os.path.join(MA.RESULTS_DIR,
                                   f"{ARM}_scale{scale:.2f}.npz"),
                      allow_pickle=False)
        for name in ("cell", "trace"):
            y_old = old[f"{name}__y"]
            assert (y_old == z["y"]).all(), "label order differs from Sec. 12j"
            for key, nm in (("psp", "spatial"), ("pbu", "bulk")):
                po, pn = old[f"{name}__{key}"], z[f"{name}__{key}"]
                fo, fn = MA.macro_f1(y_old, po), MA.macro_f1(z["y"], pn)
                rows.append((scale, name, nm, fo, fn, float((po == pn).mean())))

    bad = PL.nonconverged(all_fits)
    it = [f["n_iter"] for f in all_fits]
    print()
    print("CONVERGENCE (first):")
    print(f"  fits recorded: {len(all_fits)}   non-converged: {len(bad)}   "
          f"max n_iter {max(it)} against cap {all_fits[0]['max_iter']}")
    for b in bad:
        print("   NON-CONVERGED:", b)
    print()
    print(f"  {'scale':>5} {'split':<6} {'arm':<8} {'resumed F1':>11} "
          f"{'re-fit F1':>10} {'diff':>8} {'per-sample agreement':>21}")
    ok = not bad
    for scale, name, nm, fo, fn, ag in rows:
        ok &= abs(fo - fn) <= TOL
        print(f"  {scale:>5} {name:<6} {nm:<8} {fo:>11.4f} {fn:>10.4f} "
              f"{fn - fo:>+8.4f} {ag:>21.5f}")
    print()
    print("  re-fit deltas (spatial - bulk):")
    for scale in SCALES:
        for name in ("cell", "trace"):
            r = {nm: (fo, fn) for s, n_, nm, fo, fn, _ in rows
                 if s == scale and n_ == name}
            print(f"    scale {scale:<5} {name:<6} resumed "
                  f"{r['spatial'][0] - r['bulk'][0]:+.4f}   re-fit "
                  f"{r['spatial'][1] - r['bulk'][1]:+.4f}")
    print()
    print("VERDICT (Sec. 17d): " + (
        "MATCH -- all fits converged and every macro-F1 is within "
        f"{TOL} of the resumed value." if ok else
        "NOT A MATCH -- see above; these cells are not quoted."))
    PL.dump_json(os.path.join(OUT, "summary.json"),
                 {"n_fits": len(all_fits), "nonconverged": bad,
                  "rows": rows, "match": bool(ok)})


if __name__ == "__main__":
    main()
