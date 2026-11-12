"""
Load profiles: load_ratio(t) in [0, 1] plus a per-sample segment type.

Families (docs/tool_design.md Sec. 1.4):

  fcdlc_real    one dynamic segment of a real FC-DLC load trace. Needs
                data/processed/*_dynamic_only.npz (third-party derived, not
                redistributed). One profile per dynamic segment present.
  fcdlc_random  a new cycle assembled from the FC-DLC's structure: urban
                blocks (~195 s) and one highway block (~400 s), with order,
                levels, dwell lengths and ramp rates randomised.
  static        a constant-load hold (Kim et al.'s lab protocol); faults are
                induced on a schedule, independent of load.
  calibration   a polarisation-style staircase up and down, then one
                randomised cycle. Always fault-free: the healthy stretch a
                user would calibrate on.

Conditioning holds are optional labelled 'hold' segments.

A profile is deterministic in its SeedSequence. Only the LOAD trace of the
real recording is ever read here -- never a grid, never a bulk signal.
"""

from __future__ import annotations

import os

import numpy as np

from ._ref import REPO

REAL_TRACES = {
    "normal": "normal_flow_fc-dlc_load_trace_dynamic_only.npz",
    "inv_flow": "inverse_flow_fc-dlc_load_trace_dynamic_only.npz",
    "inv_air": "inverse_air_flow_fc-dlc_load_trace_dynamic_only.npz",
    "inv_h2": "inverse_hydrogen_flow_fc-dlc_load_trace_dynamic_only.npz",
}
PROCESSED_DIR = os.path.join(REPO, "data", "processed")
FAMILIES = ("fcdlc_real", "fcdlc_random", "static", "calibration")


def real_profiles_available():
    return all(os.path.exists(os.path.join(PROCESSED_DIR, f))
               for f in REAL_TRACES.values())


def real_profile_list():
    """(trace key, segment id) for every dynamic segment that exists."""
    out = []
    for tk, fn in REAL_TRACES.items():
        z = np.load(os.path.join(PROCESSED_DIR, fn))
        out += [(tk, int(s)) for s in np.unique(z["segment_id"])]
    return out


def n_real_profiles():
    return len(real_profile_list())


class _Builder:
    """Piecewise-linear waypoints with a segment type, sampled at dt."""

    def __init__(self, level=0.0):
        self.t = [0.0]
        self.y = [float(level)]
        self.kind = []          # (t0, t1, type)

    @property
    def now(self):
        return self.t[-1]

    def dwell(self, seconds, kind="dynamic"):
        t0 = self.now
        self.t.append(t0 + float(seconds))
        self.y.append(self.y[-1])
        self.kind.append((t0, self.now, kind))

    def ramp(self, level, rate, kind="dynamic"):
        level = float(np.clip(level, 0.0, 1.0))
        d = abs(level - self.y[-1])
        if d < 1e-9:
            return
        t0 = self.now
        self.t.append(t0 + max(d / rate, 1.0))
        self.y.append(level)
        self.kind.append((t0, self.now, kind))

    def sample(self, dt):
        n = int(np.floor(self.now / dt)) + 1
        t = np.arange(n) * dt
        y = np.interp(t, self.t, self.y)
        st = np.full(n, "dynamic", dtype="U8")
        for t0, t1, k in self.kind:
            if k != "dynamic":
                st[(t >= t0) & (t <= t1)] = k
        return t, y, st


def _rate(rng, fast_share=0.35):
    """Ramp rate, load ratio per second: routine or fast cluster."""
    if rng.random() < fast_share:
        return float(rng.uniform(0.022, 0.040))
    return float(rng.uniform(0.008, 0.018))


def _urban(b, rng):
    """~195 s: idle, two or three low/mid excursions, back to idle."""
    target = b.now + float(rng.uniform(165, 225))
    b.ramp(rng.uniform(0.0, 0.05), _rate(rng))
    b.dwell(rng.uniform(10, 28))
    while b.now < target - 45:
        b.ramp(rng.uniform(0.10, 0.38), _rate(rng))
        b.dwell(rng.uniform(8, 35))
        if rng.random() < 0.5:
            b.ramp(rng.uniform(0.12, 0.45), _rate(rng))
            b.dwell(rng.uniform(8, 25))
        b.ramp(rng.uniform(0.0, 0.07), _rate(rng))
        b.dwell(rng.uniform(8, 25))


def _highway(b, rng):
    """~400 s: stepped climb to high load with long dwells, stepped descent."""
    target = b.now + float(rng.uniform(340, 460))
    b.ramp(rng.uniform(0.25, 0.45), _rate(rng))
    b.dwell(rng.uniform(15, 50))
    while b.now < target - 110:
        b.ramp(rng.uniform(0.45, 1.0), _rate(rng))
        b.dwell(rng.uniform(20, 95))
        if rng.random() < 0.4:
            b.ramp(rng.uniform(0.25, 0.60), _rate(rng))
            b.dwell(rng.uniform(10, 40))
    b.ramp(rng.uniform(0.15, 0.35), _rate(rng))
    b.dwell(rng.uniform(10, 30))
    b.ramp(rng.uniform(0.0, 0.06), _rate(rng))
    b.dwell(rng.uniform(10, 25))


def _hold(b, rng, hold_cfg):
    lo, hi = hold_cfg.get("duration_s", (120.0, 480.0))
    b.ramp(rng.uniform(*hold_cfg.get("level", (0.10, 0.85))), _rate(rng, 0.0),
           kind="hold")
    b.dwell(rng.uniform(lo, hi), kind="hold")


def _cycle(b, rng):
    blocks = ["urban"] * 4 + ["highway"]
    rng.shuffle(blocks)
    for blk in blocks:
        (_urban if blk == "urban" else _highway)(b, rng)


def build(family, index, ss, cfg):
    """
    Build one profile. `ss` is a numpy SeedSequence unique to this profile.
    Returns dict(profile_id, family, t, load_ratio, segment_type, induced).
    """
    rng = np.random.default_rng(ss)
    dt = float(cfg.get("dt_s", 1.0))
    pc = cfg.get("profiles", {})
    induced = None
    if family == "fcdlc_real":
        lst = real_profile_list()
        tk, seg = lst[index % len(lst)]
        z = np.load(os.path.join(PROCESSED_DIR, REAL_TRACES[tk]))
        m = z["segment_id"] == seg
        t = z["t_seconds"][m] - z["t_seconds"][m][0]
        tu = np.arange(int(np.floor(t[-1] / dt)) + 1) * dt
        y = np.interp(tu, t, z["load_ratio"][m])
        pid = f"fcdlc_real/{tk}/seg{seg}"
        return {"profile_id": pid, "family": family, "t": tu,
                "load_ratio": np.clip(y, 0, 1),
                "segment_type": np.full(len(tu), "dynamic", dtype="U8"),
                "induced": None}
    b = _Builder(level=float(rng.uniform(0.0, 0.05)))
    if family == "fcdlc_random":
        holds = pc.get("holds", {})
        use_holds = rng.random() < float(holds.get("share", 0.0))
        for _ in range(int(pc.get("n_cycles", 1))):
            if use_holds:
                _hold(b, rng, holds)
            _cycle(b, rng)
    elif family == "calibration":
        b.dwell(15)
        steps = np.linspace(0.0, 1.0, 11)
        for lv in list(steps[1:]) + list(steps[-2::-1]):
            b.ramp(lv, 0.012)
            b.dwell(18)
        _cycle(b, rng)
    elif family == "static":
        level = float(rng.uniform(0.15, 0.85))
        b.ramp(level, 0.012)
        b.dwell(float(rng.uniform(*pc.get("static_duration_s", (900, 1500)))),
                kind="hold")
        # induced-fault schedule: (onset_s, fault, duration_s)
        induced, t0 = [], b.t[1] + float(rng.uniform(60, 150))
        while t0 < b.now - 220:
            induced.append((t0, str(rng.choice(["starvation", "flooding",
                                                "drying"])),
                            float(rng.uniform(40, 160))))
            t0 += float(rng.uniform(220, 420))
    else:
        raise ValueError(f"unknown profile family {family!r}")
    t, y, st = b.sample(dt)
    return {"profile_id": f"{family}/{index:05d}", "family": family, "t": t,
            "load_ratio": y, "segment_type": st, "induced": induced}
