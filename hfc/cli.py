"""
Command line.

  python -m hfc generate    --preset smoke|extensive|messy --out DIR
  python -m hfc generate    --config cfg.json --out DIR
  python -m hfc export-seville DATASET RUN_ID --out DIR
  python -m hfc check-leaks DATASET [--split-file split.json]
  python -m hfc train       DATASET --out MODELS
  python -m hfc evaluate    DATASET --models MODELS --out metrics.json
  python -m hfc run         --mapping map.json --models MODELS \
                            (--calibrate-mapping healthy.json |
                             --calibrate-range START_S END_S) --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import UNVERIFIED


def _generate(a):
    from . import gen
    over = None
    if a.config:
        with open(a.config, encoding="utf-8") as fh:
            over = json.load(fh)
    cfg = gen.make_config(a.preset, over)
    if a.seed is not None:
        cfg["seed"] = a.seed
    gen.generate(cfg, a.out, n_jobs=a.jobs)
    print(f"written: {a.out}   (SYNTHETIC; see DATA_CARD.md)")


def _export(a):
    from . import gen, seville_export
    z = np.load(os.path.join(a.dataset, "runs", a.run_id + ".npz"))
    meta = json.loads(str(z["meta_json"]))
    d = seville_export.export_run({k: z[k] for k in z.files}, meta, a.out)
    print(f"SYNTHETIC export in Seville raw layout: {d}")


def _leaks(a):
    from . import leaks
    ok, rep = leaks.check(a.dataset, a.split_file)
    print(leaks.format_report(rep))
    sys.exit(0 if ok else 1)


def _train(a):
    from . import models
    models.train(a.dataset, a.out, n_jobs=a.jobs)
    print(f"written: {os.path.join(a.out, 'models.json')}")
    print(UNVERIFIED)


def _evaluate(a):
    from . import models
    b = models.load_bundle(a.models)
    m = models.evaluate(a.dataset, b, n_jobs=a.jobs)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(m, fh, indent=1, default=float)
    print(f"written: {a.out}   (synthetic faults; not real-fault accuracy)")


def _run(a):
    from . import detect, features, io, models
    bundle = models.load_bundle(a.models)
    recs = io.load(a.mapping)
    cal_fixed = None
    if a.calibrate_mapping:
        crecs = io.load(a.calibrate_mapping)
        if len(crecs) != 1:
            raise SystemExit("the calibration mapping must select exactly one "
                             "recording")
        cal_fixed = crecs[0]
    elif not a.calibrate_range:
        raise SystemExit(
            "No healthy calibration given. The tool does not classify "
            "without a baseline from a known-healthy stretch of this cell: "
            "pass --calibrate-mapping or --calibrate-range START_S END_S.")
    print(UNVERIFIED)
    for rec in recs:
        if cal_fixed is not None:
            src, mask, extra = cal_fixed, None, {
                "calibration_source": cal_fixed["name"]}
        else:
            t0, t1 = a.calibrate_range
            src, mask = rec, (rec["t"] >= t0) & (rec["t"] <= t1)
            extra = {"calibration_source":
                     f"{rec['name']} t in [{t0}, {t1}] s (asserted healthy by "
                     f"the user; that stretch is included in the output)"}
        if a.mode == "bulk":
            src = dict(src, grid=None)
        cal = features.calibrate(src, mask)
        out, info = detect.run_recording(rec, cal, bundle,
                                         None if a.mode == "auto" else a.mode)
        rep = detect.summarise(rec, cal, out, info, bundle, extra)
        stem = detect.write_outputs(a.out, rec, out, rep)
        print(f"{rec['name']}: mode {rep['mode']}, answered "
              f"{rep['answered_share']:.1%}, states "
              + ", ".join(f"{k} {v:.1%}" for k, v in rep["state_share"].items())
              + f"  -> {os.path.join(a.out, stem + '_report.md')}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hfc", description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--preset", choices=("smoke", "extensive", "messy"))
    g.add_argument("--config")
    g.add_argument("--seed", type=int)
    g.add_argument("--out", required=True)
    g.add_argument("--jobs", type=int, default=4)
    g.set_defaults(fn=_generate)
    e = sub.add_parser("export-seville")
    e.add_argument("dataset"); e.add_argument("run_id")
    e.add_argument("--out", required=True)
    e.set_defaults(fn=_export)
    c = sub.add_parser("check-leaks")
    c.add_argument("dataset")
    c.add_argument("--split-file")
    c.set_defaults(fn=_leaks)
    t = sub.add_parser("train")
    t.add_argument("dataset"); t.add_argument("--out", required=True)
    t.add_argument("--jobs", type=int, default=4)
    t.set_defaults(fn=_train)
    v = sub.add_parser("evaluate")
    v.add_argument("dataset"); v.add_argument("--models", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--jobs", type=int, default=4)
    v.set_defaults(fn=_evaluate)
    r = sub.add_parser("run")
    r.add_argument("--mapping", required=True)
    r.add_argument("--models", required=True)
    r.add_argument("--calibrate-mapping")
    r.add_argument("--calibrate-range", nargs=2, type=float,
                   metavar=("START_S", "END_S"))
    r.add_argument("--mode", choices=("auto", "bulk", "spatial"),
                   default="auto")
    r.add_argument("--out", required=True)
    r.set_defaults(fn=_run)
    a = ap.parse_args(argv)
    if a.cmd == "generate" and not (a.preset or a.config):
        ap.error("generate needs --preset or --config")
    a.fn(a)


if __name__ == "__main__":
    main()
