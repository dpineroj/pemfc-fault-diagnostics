"""
POST-HOC (Sec. 17c, and 17e-3d with --dataset v2). Model A under S0-S3.

  --dataset capped   the quoted configuration: MAX_PER_GROUP=2500 (sampling
                     seed 777+cell), four amplitude scales, both arms.
                     Stage 1 re-fits S0/S1 and applies the REPRODUCTION GATE
                     against Sec. 12's recorded ranges; the run stops with
                     exit code 3 if it fails. Stage 2 fits S2/S3 and the
                     random forest (scale 1.00, S0 and S3).
  --dataset v2       uncapped v1 + fault-free companions, scale 1.00, S3 only.

Nothing in run_logs/model_a_results/ is read or written.

Run:  python -u -W ignore scripts/spatial/posthoc_model_a.py --dataset capped
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fault_generator as F  # noqa: E402
import phase4_model_a as MA  # noqa: E402
import posthoc_lib as PL  # noqa: E402

# Sec. 12h, capped run, as recorded (ranges across arms and scales).
REC = {"S0": (0.103, 0.123), "S1": (0.109, 0.130)}
REC_SP_S0_1 = 0.945
REC_BU_S0_1 = (0.822, 0.828)
GATE_TOL = 0.005


def path(dataset, arm, scale):
    return os.path.join(PL.OUT, f"model_a_{dataset}",
                        f"{arm}_scale{scale:.2f}.npz")


def run(dataset, arm, scale, jobs):
    """jobs: list of (split, kind). Appends to the cell's result file."""
    p = path(dataset, arm, scale)
    have = dict(np.load(p, allow_pickle=False)) if os.path.exists(p) else {}
    todo = [(s, k) for s, k in jobs if f"{s}__{k}__psp" not in have]
    if not todo:
        return
    ds = PL.assemble_a(arm, scale, cap=2500 if dataset == "capped" else None,
                       companions=(dataset == "v2"))
    out = dict(have)
    out.update(y=ds["y"].astype("U12"), gt=ds["gt"].astype("U12"),
               cell=ds["cell"], comp=ds["comp"])
    fits = json.loads(str(have["fits"])) if "fits" in have else []
    for s, k in todo:
        for which, key in (("psp", "Xs"), ("pbu", "Xb")):
            pr, f = PL.cv_predict(ds[key], ds["y"], ds["gt"], ds["cell"], s, k,
                                  f"{dataset}/{arm}/{scale}/{s}/{k}/{which}")
            out[f"{s}__{k}__{which}"] = pr.astype("U12")
            fits += f
        out["fits"] = np.array(json.dumps(fits))
        PL.save_npz(p, **out)
        print(f"[done] {dataset} {arm:<12} scale {scale:<5} {s} {k} "
              f"n={len(ds['y']):,}", flush=True)


def stats(dataset, arm, scale, s, k, rng):
    z = np.load(path(dataset, arm, scale), allow_pickle=False)
    return PL.macro_delta(z["y"], z[f"{s}__{k}__psp"], z[f"{s}__{k}__pbu"],
                          PL.gkey_of(z["gt"], z["cell"]), rng)


def convergence(dataset):
    d = os.path.join(PL.OUT, f"model_a_{dataset}")
    fits = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".npz") and ".tmp" not in fn:
            fits += json.loads(str(np.load(os.path.join(d, fn))["fits"]))
    lr = [f for f in fits if f["converged"] is not None]
    bad = PL.nonconverged(fits)
    print(f"CONVERGENCE (first): {len(lr)} logistic fits recorded, "
          f"{len(bad)} non-converged; max n_iter "
          f"{max(f['n_iter'] for f in lr)} against cap {lr[0]['max_iter']}; "
          f"{len(fits) - len(lr)} random-forest fits (no iterative criterion)")
    for b in bad:
        print("  NON-CONVERGED (cell unreportable):", b)
    return fits, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("capped", "v2"))
    a = ap.parse_args()
    rng = np.random.default_rng(PL.BOOT_SEED)
    print(f"POST-HOC Model A  dataset={a.dataset}", flush=True)
    summary = {}

    if a.dataset == "v2":
        for arm in F.STARVATION_ARMS:
            run("v2", arm, 1.0, [("S3", "A")])
        print()
        fits, bad = convergence("v2")
        print()
        print("Model A on v2 (uncapped v1 + companions), scale 1.00, S3:")
        for arm in F.STARVATION_ARMS:
            r = stats("v2", arm, 1.0, "S3", "A", rng)
            z = np.load(path("v2", arm, 1.0), allow_pickle=False)
            y, c = z["y"], z["comp"]
            far = {k: float((z[f"S3__A__{k}"][y == "healthy"] != "healthy").mean())
                   for k in ("psp", "pbu")}
            print(f"  {arm:<12} spatial {r['spatial']:.3f} bulk {r['bulk']:.3f} "
                  f"delta {r['delta']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]"
                  f"   synthetic-healthy false-alarm: spatial {far['psp']:.4f} "
                  f"bulk {far['pbu']:.4f}  (n healthy {int((y == 'healthy').sum()):,},"
                  f" companions {int(c.sum()):,})")
            summary[f"{arm}|1.0|S3|A"] = {**r, "far_healthy": far}
        PL.dump_json(os.path.join(PL.OUT, "model_a_v2_summary.json"),
                     {"nonconverged": bad, "n_fits": len(fits),
                      "results": summary})
        return

    # ---------------- stage 1: originals + reproduction gate ---------------
    for arm in F.STARVATION_ARMS:
        for scale in MA.AMPLITUDE_SCALES:
            run("capped", arm, scale, [("S0", "A"), ("S1", "A")])
    print()
    print("REPRODUCTION GATE (Sec. 17c): re-fitted S0/S1 against Sec. 12h")
    R = {}
    for arm in F.STARVATION_ARMS:
        for scale in MA.AMPLITUDE_SCALES:
            for s in ("S0", "S1"):
                R[(arm, scale, s)] = stats("capped", arm, scale, s, "A", rng)
    ok = True
    for s in ("S0", "S1"):
        d = [R[(arm, sc, s)]["delta"] for arm in F.STARVATION_ARMS
             for sc in MA.AMPLITUDE_SCALES]
        good = (abs(min(d) - REC[s][0]) <= GATE_TOL
                and abs(max(d) - REC[s][1]) <= GATE_TOL)
        ok &= good
        print(f"  {s}: re-fitted delta range {min(d):+.3f} to {max(d):+.3f}; "
              f"recorded {REC[s][0]:+.3f} to {REC[s][1]:+.3f}  -> "
              f"{'ok' if good else 'MISMATCH'}")
    sp = [R[(arm, 1.0, "S0")]["spatial"] for arm in F.STARVATION_ARMS]
    bu = [R[(arm, 1.0, "S0")]["bulk"] for arm in F.STARVATION_ARMS]
    good = (all(abs(v - REC_SP_S0_1) <= GATE_TOL for v in sp)
            and min(bu) >= REC_BU_S0_1[0] - GATE_TOL
            and max(bu) <= REC_BU_S0_1[1] + GATE_TOL)
    ok &= good
    print(f"  S0, scale 1.00: spatial {sp[0]:.3f}/{sp[1]:.3f} (recorded "
          f"{REC_SP_S0_1}); bulk {bu[0]:.3f}/{bu[1]:.3f} (recorded "
          f"{REC_BU_S0_1[0]}-{REC_BU_S0_1[1]})  -> {'ok' if good else 'MISMATCH'}")
    if not ok:
        print("GATE FAILED -- stopping before S2/S3, as registered. Ask before"
              " continuing.")
        sys.exit(3)
    print("GATE PASSED -- continuing to S2, S3 and the random forest.",
          flush=True)

    # ---------------- stage 2 ----------------------------------------------
    for arm in F.STARVATION_ARMS:
        for scale in MA.AMPLITUDE_SCALES:
            run("capped", arm, scale, [("S2", "A"), ("S3", "A")])
        run("capped", arm, 1.0, [("S0", "rf"), ("S3", "rf")])
    print()
    fits, bad = convergence("capped")
    print()
    print("Model A, capped configuration, macro-F1 (spatial / bulk / delta "
          "[95% group-bootstrap CI])")
    for arm in F.STARVATION_ARMS:
        print(f"  starvation arm: {arm}")
        for s in PL.SPLITS:
            for scale in MA.AMPLITUDE_SCALES:
                r = stats("capped", arm, scale, s, "A", rng)
                print(f"    {s} scale {scale:<5} {r['spatial']:.3f} "
                      f"{r['bulk']:.3f} {r['delta']:+.3f} "
                      f"[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]")
                summary[f"{arm}|{scale}|{s}|A"] = r
        for s in ("S0", "S3"):
            r = stats("capped", arm, 1.0, s, "rf", rng)
            print(f"    {s} scale 1.0   random forest {r['spatial']:.3f} "
                  f"{r['bulk']:.3f} {r['delta']:+.3f} "
                  f"[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]")
            summary[f"{arm}|1.0|{s}|rf"] = r
    print()
    for s in PL.SPLITS:
        print("  " + PL.SPLIT_NAME[s])
    PL.dump_json(os.path.join(PL.OUT, "model_a_capped_summary.json"),
                 {"nonconverged": bad, "n_fits": len(fits), "results": summary})
    print("All logistic fits converged." if not bad
          else f"{len(bad)} FITS DID NOT CONVERGE.")


if __name__ == "__main__":
    main()
