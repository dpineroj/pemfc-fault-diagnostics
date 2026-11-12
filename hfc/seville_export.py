"""
Export a generated run in the Seville recording's RAW file layout.

  <out>/<flow_config>/SYNTHETIC_FC-DLC_<run_id>/
      CDM_C_SYNTHETIC_<run_id>.dat   26-line blocks, 18x18 grid
      CDM_T_SYNTHETIC_<run_id>.dat   14-line blocks, 9x9 grid
      FC-DLC_SYNTHETIC_<run_id>.dat  flat bulk table
      SYNTHETIC_README.json          flag, generator version, seed, config

Raw conventions are reproduced because the real files have them: per-segment
current in AMPS (not A/cm2), NEGATIVE sign, comma decimals, tab-delimited,
and a bulk table on whole seconds.

The export is still SYNTHETIC and says so in every folder name, every file
name and the README beside the files. Bulk channels the generator does not
model are left out. The CDM block's "voltage" scalar is not modelled either;
the layout requires a value there, so a constant 0 is written and documented
-- it is a placeholder, not a signal.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import numpy as np

EPOCH = datetime(2000, 1, 1, 0, 0, 0)          # obviously not a real test date
LABVIEW0 = datetime(1904, 1, 1)


def _fmt(v, width=8, dec=4):
    return f"{v:{width}.{dec}f}".replace(".", ",")


def _grid_lines(g, dec):
    return ["\t".join(_fmt(v, 8, dec) for v in row) + "\n" for row in g]


def export_run(run, meta, out_root):
    """`run` is a canonical run in clean units (A/cm2, positive)."""
    if meta.get("current_units") != "A/cm2" or meta.get("current_sign") != 1:
        raise ValueError("export expects a canonical run in A/cm2, positive")
    rows, cols = run["current"].shape[1:]
    seg_area = meta["active_area_cm2"] / (rows * cols)
    d = os.path.join(out_root, meta["flow_config"],
                     f"SYNTHETIC_FC-DLC_{meta['run_id']}")
    os.makedirs(d, exist_ok=True)
    t = run["t"]
    stamps = [EPOCH + timedelta(seconds=float(x)) for x in t]
    raw = -run["current"].astype(np.float64) * seg_area      # amps, negative
    with open(os.path.join(d, f"CDM_C_SYNTHETIC_{meta['run_id']}.dat"), "w",
              encoding="utf-8", newline="\n") as fc, \
         open(os.path.join(d, f"CDM_T_SYNTHETIC_{meta['run_id']}.dat"), "w",
              encoding="utf-8", newline="\n") as ft:
        for k, ts in enumerate(stamps):
            ms = int(round(ts.microsecond / 1000.0))
            ts = ts.replace(microsecond=0)
            if ms == 1000:
                ts, ms = ts + timedelta(seconds=1), 0
            head = [ts.strftime("%d/%m/%y %H:%M:%S") + f",{ms:03d}\n",
                    f"{(ts - LABVIEW0).total_seconds() + ms / 1000:.3f}"
                    .replace(".", ",") + "\n", "\n"]
            fc.writelines(head + ["voltage\n", _fmt(0.0) + "\n", "\n",
                                  "current\n"] + _grid_lines(raw[k], 4) + ["\n"])
            ft.writelines(head + ["temperature\n"]
                          + _grid_lines(run["temperature"][k], 4) + ["\n"])
    tb = np.arange(np.ceil(t[0]), np.floor(t[-1]) + 1.0)
    I = np.interp(tb, run["t_bulk"], run["INTENSIDAD"])
    V = np.interp(tb, run["t_bulk"], run["V001"])
    with open(os.path.join(d, f"FC-DLC_SYNTHETIC_{meta['run_id']}.dat"), "w",
              encoding="utf-8", newline="\n") as fb:
        fb.write("FECHA\tHORA\tV001\tINTENSIDAD\n")
        for x, i_, v_ in zip(tb, I, V):
            ts = EPOCH + timedelta(seconds=float(x))
            fb.write(f"{ts.strftime('%d/%m/%Y')}\t{ts.strftime('%H:%M:%S')}\t"
                     f"{v_:.4f}\t{i_:.3f}\n".replace(".", ","))
    readme = {
        "SYNTHETIC": True,
        "warning": "SYNTHETIC DATA in the Seville raw layout. Not a recording.",
        "generator_version": meta["generator_version"], "seed": meta["seed"],
        "run_id": meta["run_id"], "config": meta["config"],
        "layout_notes": {
            "CDM_C grid": "per-segment current in A, negative sign",
            "CDM_C 'voltage' scalar": "NOT MODELLED; constant 0 placeholder",
            "bulk columns": "V001 [V], INTENSIDAD [A]; no other channel is "
                            "modelled and none is emitted",
            "timestamps": "start at 2000-01-01 00:00:00 by construction"}}
    with open(os.path.join(d, "SYNTHETIC_README.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(readme, fh, indent=1, sort_keys=True)
    return d
