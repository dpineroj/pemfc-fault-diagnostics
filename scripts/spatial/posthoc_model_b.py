"""
POST-HOC (Sec. 17c / 17e-3d / 17g). Model B under the corrected splits.

Model, features, windows, labels, metrics, bootstrap and the Sec. 13i
criteria are imported from the registered experiment (phase4_model_b.py /
model_b_data.py). Only the SPLIT and the DATASET vary here:

  --dataset v1     the Sec. 15 data (4 traces x 5 cells)
  --dataset v2     v1 + one fault-free companion run per (trace, cell)
  --dataset dry40  v1 generator, 40 cells per trace (drying power, Sec. 17g)

Original S0/S1 results for v1 are READ from run_logs/model_b_results/ and
never re-fitted or rewritten.

Run:  python -u -W ignore scripts/spatial/posthoc_model_b.py --dataset v1 \
          --splits S2,S3 --cfgs all --binary
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fault_generator as F  # noqa: E402
import model_b_data as D  # noqa: E402
import phase4_model_b as MB  # noqa: E402
import posthoc_lib as PL  # noqa: E402

DATASETS = {
    "v1": dict(cells=range(5), companions=False),
    "v2": dict(cells=range(5), companions=True),
    "dry40": dict(cells=range(40), companions=False),
}


def out_dir(dataset):
    return os.path.join(PL.OUT, f"model_b_{dataset}")


def res_path(dataset, cfg, arm, task):
    return os.path.join(out_dir(dataset), f"{cfg}__{arm}__{task}.npz")


def run_cell(dataset, cfg, arm, task, splits):
    path = res_path(dataset, cfg, arm, task)
    have = dict(np.load(path, allow_pickle=False)) if os.path.exists(path) else {}
    todo = [s for s in splits if f"{s}__psp" not in have]
    if not todo:
        print(f"[on disk] {dataset} {cfg:<18} {arm:<12} {task}", flush=True)
        return
    ds = PL.assemble_b(cfg, arm, **DATASETS[dataset])
    if task == "binary":
        m = np.isin(ds["y"], ("none", "starvation"))
        ds = {k: v[m] for k, v in ds.items()}
    out = dict(have)
    out.update(y=ds["y"].astype("U12"), visible=ds["visible"].astype(bool),
               event=ds["event"].astype("U40"), gt=ds["gt"].astype("U12"),
               cell=ds["cell"], comp=ds["comp"])
    fits = json.loads(str(have["fits"])) if "fits" in have else []
    for s in todo:
        for which, key in (("spatial", "Xs"), ("bulk", "Xb")):
            p, f = PL.cv_predict(ds[key], ds["y"], ds["gt"], ds["cell"], s,
                                 "B", f"{dataset}/{cfg}/{arm}/{task}/{s}/{which}")
            out[f"{s}__{'psp' if which == 'spatial' else 'pbu'}"] = p.astype("U12")
            fits += f
        out["fits"] = np.array(json.dumps(fits))
        PL.save_npz(path, **out)       # after every split: a death costs one
        print(f"[done] {dataset} {cfg:<18} {arm:<12} {task:<6} {s} "
              f"n={len(ds['y']):,}", flush=True)


def load_z(dataset, cfg, arm, task):
    """Result dict with predictions for every split available."""
    z = dict(np.load(res_path(dataset, cfg, arm, task), allow_pickle=False))
    z["gc"] = PL.gkey_of(z["gt"], z["cell"])
    if dataset == "v1":    # originals, read-only
        o = np.load(MB._path(cfg, arm, task), allow_pickle=False)
        assert (o["y"] == z["y"]).all() and (o["gc"] == z["gc"]).all(), \
            "v1 assembly differs from the Sec. 15 files"
        z["S0__psp"], z["S0__pbu"] = o["cell__psp"], o["cell__pbu"]
        z["S1__psp"], z["S1__pbu"] = o["trace__psp"], o["trace__pbu"]
    return z


def report(dataset, cfgs, binary, splits_show):
    rng = np.random.default_rng(PL.BOOT_SEED)
    all_fits = []
    d = out_dir(dataset)
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".npz") and ".tmp" not in fn:
            all_fits += json.loads(str(np.load(os.path.join(d, fn))["fits"]))
    bad = PL.nonconverged(all_fits)
    it = np.array([f["n_iter"] for f in all_fits])
    sp = np.array([f["p"] > 10 for f in all_fits])
    print()
    print("=" * 78)
    print(f"POST-HOC Model B, dataset {dataset} -- CONVERGENCE (first)")
    print("=" * 78)
    print(f"  fits recorded in the result files: {len(all_fits)}   "
          f"non-converged: {len(bad)}")
    print(f"  max iterations: spatial {it[sp].max()}, bulk {it[~sp].max()} "
          f"(cap {MB.MAX_ITER})")
    for b in bad:
        print("  NON-CONVERGED (cell unreportable):", b)
    print()
    summary = {}
    tasks = [(c, "multi") for c in cfgs] + ([("starvation", "binary")]
                                           if binary else [])
    hdr = (f"    {'split':<4}{'arm':<8}{'prec':>6}{'rec':>7}{'F1':>7}"
           f"{'FPR':>7}{'J':>7}{'rec-blind':>10}")
    for cfg, task in tasks:
        f = D.OWN_FAULT[cfg]
        hl = D.CONFIGS[cfg]
        for arm in F.STARVATION_ARMS:
            z = load_z(dataset, cfg, arm, task)
            print("-" * 78)
            print(f"  {cfg} [{task}]  own fault {f}, h={hl['h']} s, "
                  f"L={hl['L']} s   starv-arm={arm}   n={len(z['y']):,}")
            first = True
            for s in splits_show:
                if f"{s}__psp" not in z:
                    continue
                a = MB.analyse(z, s, f, rng)
                if first:
                    print(f"  {a['n_pos']} positives ({a['n_blind']} blind) "
                          f"from {a['n_events']} distinct events; "
                          f"{a['n_neg']} non-{f} targets")
                    print(hdr)
                    first = False
                v = MB.verdict(a, cfg.endswith("ctrl30"))
                for nm in ("spatial", "bulk"):
                    m = a[nm]
                    print(f"    {s:<4}{nm:<8}{m['precision']:>6.3f}"
                          f"{m['recall']:>7.3f}{m['f1']:>7.3f}{m['fpr']:>7.3f}"
                          f"{m['J']:>7.3f}{m['recall_blind']:>10.3f}")
                c = a["ci"]
                print(f"    {s:<4}delta-F1 "
                      f"{a['spatial']['f1'] - a['bulk']['f1']:+.3f} "
                      f"{MB.fmt_ci(c['dF1'])}  J_sp {MB.fmt_ci(c['J_sp'])} "
                      f"J_bu {MB.fmt_ci(c['J_bu'])}"
                      + (f"  dBlind {MB.fmt_ci(c['dBlind'])}"
                         if a["n_blind"] else ""))
                print(f"    {s:<4}-> " + "; ".join(v))
                summary[f"{cfg}|{arm}|{s}|{task}"] = {
                    "fired": v, "spatial": a["spatial"], "bulk": a["bulk"],
                    "n_pos": a["n_pos"], "n_blind": a["n_blind"],
                    "n_neg": a["n_neg"], "n_events": a["n_events"],
                    "n": a["n"], "ci": {k: list(x) for k, x in a["ci"].items()}}
    print()
    for s in splits_show:
        print("  " + PL.SPLIT_NAME[s])
    PL.dump_json(os.path.join(PL.OUT, f"model_b_{dataset}_summary.json"),
                 {"n_fits": len(all_fits), "nonconverged": bad,
                  "results": summary})
    print("=" * 78)
    print("All fits converged before max_iter." if not bad else
          f"{len(bad)} FITS DID NOT CONVERGE.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--splits", required=True)
    ap.add_argument("--cfgs", default="all")
    ap.add_argument("--binary", action="store_true")
    ap.add_argument("--show", default=None,
                    help="splits to show in the report (default: S0..S3)")
    a = ap.parse_args()
    splits = a.splits.split(",")
    cfgs = list(MB.CFG_ORDER) if a.cfgs == "all" else a.cfgs.split(",")
    print(f"POST-HOC Model B  dataset={a.dataset} splits={splits} cfgs={cfgs} "
          f"binary={a.binary}", flush=True)
    D.assert_generator_agreement()
    for cfg in cfgs:
        for arm in F.STARVATION_ARMS:
            run_cell(a.dataset, cfg, arm, "multi", splits)
    if a.binary:
        for arm in F.STARVATION_ARMS:
            run_cell(a.dataset, "starvation", arm, "binary", splits)
    report(a.dataset, cfgs, a.binary,
           (a.show.split(",") if a.show else list(PL.SPLITS)))


if __name__ == "__main__":
    main()
