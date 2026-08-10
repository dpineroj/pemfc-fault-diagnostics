"""
Synthetic fault-onset generator driven by the real dynamic-only load trace.

Grounding / conventions (verified by reading the source before writing this,
not assumed):

- Grid shape and per-state spatial patterns are grounded in notebook 01
  (`01_spatial_viz_sandbox.ipynb`, cells `d3fb30fd`/`899e7d48`/`a1edd97b`):
  4x4 segment grid, the same row/column gradient + noise + clip formulas per
  state. Notebook 01 itself is NOT imported or modified -- its generators are
  static, single-shot, discretely switched by state name. Here they are
  reimplemented as continuous functions of a `severity` in [0, 1], built by
  linearly interpolating notebook 01's NORMAL formula (severity=0) toward its
  fully-formed fault formula (severity=1), so severity=1 reproduces notebook
  01's numbers exactly and intermediate severities are a physically smooth
  blend, not a new invented pattern.
- CV(J) is copied verbatim from notebook 02's verified formula (cells
  `7f90ce08` + `a70af676`): population std (ddof=0) of |current_grid| divided
  by mean(|current_grid|), per timestep. Not reimplemented independently.
- Ramp-rate and dwell thresholds (0.02/s, 0.10, 0.40, 10s minimum sustained
  duration) are the exact values `extract_load_trace.py`'s dynamic-only
  report already derived from this trace's own distribution (see that
  script's "DYNAMIC-ONLY TRACE ... USE THESE FOR TASK 2 TRIGGERS" section).
  `find_ramp_events` is imported and reused as-is from that module; a sibling
  `find_dwell_events` is added here (mirrors `find_dwell_fraction`'s
  contiguous-run detection, but returns each run's own boundaries -- needed
  to build onset trajectories, not just an aggregate fraction).
- The dynamic-only trace concatenates 3 real driving segments with 2
  artificial 1s seams (`segment_id` marks which). Every trigger-detection and
  episode-building step below operates strictly within one segment's own
  index range; nothing is ever computed or carried across a segment boundary.

Fault dynamics (severity(t) shape per type) are engineering choices grounded
in the literal onset-speed/shape description from the Task 2 brief -- see
each `_build_*_episode` function's comment for the specific reasoning. Every
numeric constant in those functions that is a calibration choice rather than
something derived from the real trace or notebook 01/02 is labeled inline as
such, at its point of definition (same real/literature/tuned separation
`docs/real_data_notes.md` uses for the dataset itself).

SCOPE NOTE -- starvation is a single combined fault type, not split into air-
vs. hydrogen-starvation subtypes, even though both are physically distinct
failure modes with (per the literature) different spatial signatures (air
starvation: cathode-flow-path column gradient, per Yu et al. 2024, which is
what notebook 01's `generate_starvation` and this file's
`generate_starvation_grid` model; hydrogen starvation would need a different,
anode-flow-path-oriented spatial pattern that isn't implemented here). This
is a deliberate current-scope simplification carried over from the Task 2
brief's own phrasing ("STARVATION (air or hydrogen)"), not an oversight --
splitting them into two distinct fault types with their own spatial grounding
is deferred to a later task.

KNOWN LIMITATION -- CV(J) does NOT discriminate flooding from healthy
operation on this generator's output: baseline (no-fault) CV(J) mean is
0.076 vs. flooding-at-severity>0.5 mean 0.086 -- flooding is not lower, the
opposite of the raw sigma(J) intuition from notebook 01 / the Task 1
real-data finding. Cause: CV(J) normalizes sigma(J) by mean(|J|), and
flooding drives both down together (notebook 01: J_std 0.047->0.041, J_mean
0.697->0.432), so the normalization cancels almost all of the raw-sigma
signal that made flooding look distinct. CV(J) remains valid as the
headline spatial-vs-bulk diagnostic argument (spatial nonuniformity carries
information bulk voltage doesn't), but it is NOT sufficient alone as a
fault-TYPE classifier feature -- downstream classification must use the full
spatial grid, or CV(J) jointly with mean(|J|), to actually discriminate
flooding from healthy. Full writeup: docs/synthetic_generator_notes.md.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from extract_load_trace import find_ramp_events

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

DEFAULT_TRACE_PATH = os.path.join(PROCESSED_DIR, "normal_flow_fc-dlc_load_trace_dynamic_only.npz")

ROWS, COLS = 4, 4  # notebook 01's grid convention

# Trigger thresholds -- lifted directly from extract_load_trace.py's
# dynamic-only-trace report, not re-derived here.
RAMP_THRESHOLD = 0.02           # /s, starvation trigger
LOW_LOAD_THRESH = 0.10          # flooding trigger
HIGH_LOAD_THRESH = 0.40         # drying trigger
MIN_DWELL_S = 10.0              # sustained-dwell minimum, matches extract_load_trace.py

# Placeholder "confirmed fault" cutoff -- per Task 2 brief this is
# "still-to-be-defined". Used only to derive each episode's own precursor
# window from its actual onset speed (never to gate/hide labeling elsewhere).
CLASSIFICATION_THRESHOLD = 0.5

# CALIBRATION CHOICE: window (seconds) on either side of an internal
# segment-boundary seam flagged via near_seam_buffer. 90s is set to roughly
# match the longest precursor window actually observed across all fault
# types in this run (drying, ~76s -- see the Task 2 review's onset-event
# summary) plus a margin, not the starvation precursor window specifically
# (starvation's is much shorter, ~1-3s). Any window built from a sample this
# close to a seam would include the hard reset-then-reonset artifact
# described in generate_labeled_run's docstring.
NEAR_SEAM_WINDOW_S = 90.0

BASE_SEED = 20260810  # fixed seed -> fully reproducible noise realizations

FAULT_COLORS = {"starvation": "#d7191c", "flooding": "#2166ac", "drying": "#e08214"}


# ---------------------------------------------------------------------------
# Spatial grid generators -- continuous-by-severity, grounded in notebook 01
# ---------------------------------------------------------------------------

def generate_normal_grid(seed: int) -> np.ndarray:
    """Verbatim notebook 01 `generate_normal` (cell 899e7d48), reseeded per call."""
    rng = np.random.default_rng(seed)
    row_grad = np.linspace(1.0, 0.65, ROWS)
    col_grad = np.linspace(1.0, 0.72, COLS)
    gradient = np.outer(row_grad, col_grad)
    base = 0.45 + 0.35 * gradient
    noise = rng.normal(0, 0.02, (ROWS, COLS))
    return np.clip(base + noise, 0.30, 0.90)


def generate_starvation_grid(severity: float, seed: int) -> np.ndarray:
    """
    O2 starvation, continuous. Interpolates NORMAL's row/col gradient, base
    range, and clip bounds toward notebook 01's `generate_starvation` (cell
    899e7d48) fully-formed values as severity -> 1. At severity=0 this is
    exactly generate_normal_grid; at severity=1 it's exactly notebook 01's
    starvation formula.
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    row_grad = (1 - severity) * np.linspace(1.0, 0.65, ROWS) + severity * np.linspace(0.95, 1.0, ROWS)
    col_grad = (1 - severity) * np.linspace(1.0, 0.72, COLS) + severity * np.linspace(1.0, 0.04, COLS)
    gradient = np.outer(row_grad, col_grad)
    base_lo = (1 - severity) * 0.45 + severity * 0.04
    base_scale = (1 - severity) * 0.35 + severity * 0.62
    base = base_lo + base_scale * gradient
    noise = rng.normal(0, 0.02, (ROWS, COLS))
    clip_lo = (1 - severity) * 0.30 + severity * 0.0
    clip_hi = (1 - severity) * 0.90 + severity * 0.75
    return np.clip(base + noise, clip_lo, clip_hi)


def generate_flooding_grid(severity: float, seed: int) -> np.ndarray:
    """
    Flooding, continuous. Interpolates toward notebook 01's
    `generate_flooding` (cell 899e7d48), including the bottom-right water-
    pooling suppression term, scaled by severity.
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    row_grad = (1 - severity) * np.linspace(1.0, 0.65, ROWS) + severity * np.linspace(1.0, 0.82, ROWS)
    col_grad = (1 - severity) * np.linspace(1.0, 0.72, COLS) + severity * np.linspace(1.0, 0.87, COLS)
    gradient = np.outer(row_grad, col_grad)
    base_lo = (1 - severity) * 0.45 + severity * 0.28
    base_scale = (1 - severity) * 0.35 + severity * 0.20
    base = base_lo + base_scale * gradient
    suppression = np.zeros((ROWS, COLS))
    suppression[2:, 2:] = 0.07 * severity
    noise_std = (1 - severity) * 0.02 + severity * 0.012
    noise = rng.normal(0, noise_std, (ROWS, COLS))
    clip_lo = (1 - severity) * 0.30 + severity * 0.10
    clip_hi = (1 - severity) * 0.90 + severity * 0.58
    return np.clip(base - suppression + noise, clip_lo, clip_hi)


def generate_drying_grid(severity: float, seed: int) -> np.ndarray:
    """Drying, continuous. Interpolates toward notebook 01's `generate_drying` (cell 899e7d48)."""
    severity = float(np.clip(severity, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    row_grad = (1 - severity) * np.linspace(1.0, 0.65, ROWS) + severity * np.linspace(1.0, 0.42, ROWS)
    col_grad = (1 - severity) * np.linspace(1.0, 0.72, COLS) + severity * np.linspace(1.0, 0.38, COLS)
    gradient = np.outer(row_grad, col_grad)
    base_lo = (1 - severity) * 0.45 + severity * 0.08
    base_scale = (1 - severity) * 0.35 + severity * 0.52
    base = base_lo + base_scale * gradient
    noise = rng.normal(0, 0.02, (ROWS, COLS))
    clip_lo = (1 - severity) * 0.30 + severity * 0.04
    clip_hi = (1 - severity) * 0.90 + severity * 0.68
    return np.clip(base + noise, clip_lo, clip_hi)


_GRID_GENERATORS = {
    "starvation": generate_starvation_grid,
    "flooding": generate_flooding_grid,
    "drying": generate_drying_grid,
}


def generate_current_grid(fault_type: str | None, severity: float, seed: int) -> np.ndarray:
    if fault_type is None or severity <= 0:
        return generate_normal_grid(seed)
    return _GRID_GENERATORS[fault_type](severity, seed)


def generate_temperature_grid(current_grid: np.ndarray, fault_type: str | None, severity: float, seed: int) -> np.ndarray:
    """
    Continuous version of notebook 01's `generate_temperature` (cell
    a1edd97b): same center-bias / current-correlation / gaussian-smoothing
    mechanism, but the per-state (base_t, center_scale, j_scale) triple is
    interpolated by severity instead of switched by a state string.
    """
    severity = float(np.clip(severity, 0.0, 1.0)) if fault_type else 0.0
    rng = np.random.default_rng(seed + 100)
    row_center = np.array([0.85, 1.0, 1.0, 0.85])
    col_center = np.array([0.88, 1.0, 1.0, 0.88])
    center_bias = np.outer(row_center, col_center)
    j_norm = current_grid / 0.80

    normal_params = (70.0, 5.0, 8.0)
    fault_params = {
        "starvation": (69.0, 4.0, 11.0),
        "flooding": (68.0, 3.5, 6.0),
        "drying": (73.0, 6.0, 14.0),
    }
    target = fault_params.get(fault_type, normal_params) if fault_type else normal_params

    base_t = (1 - severity) * normal_params[0] + severity * target[0]
    center_scale = (1 - severity) * normal_params[1] + severity * target[1]
    j_scale = (1 - severity) * normal_params[2] + severity * target[2]

    temp = base_t + center_scale * center_bias + j_scale * j_norm
    noise = rng.normal(0, 0.4, (ROWS, COLS))
    return gaussian_filter(temp + noise, sigma=0.8)


# ---------------------------------------------------------------------------
# CV(J) -- copied verbatim from notebook 02's verified formula
# ---------------------------------------------------------------------------

def compute_cv(current_grid: np.ndarray) -> float:
    """
    CV(J) = std(|J|) / mean(|J|), copied verbatim from notebook 02's verified
    formula (cells 7f90ce08 + a70af676): population std (ddof=0, numpy
    default) of the grid's magnitude, divided by the grid's magnitude mean.
    abs() kept for formula fidelity even though synthetic current values here
    are non-negative by construction (unlike the real dataset's unresolved
    sign convention that motivated it there).
    """
    j_abs = np.abs(current_grid).reshape(-1)
    return float(j_abs.std() / j_abs.mean())


# ---------------------------------------------------------------------------
# Trigger detection
# ---------------------------------------------------------------------------

def find_dwell_events(
    t_seconds: np.ndarray, load_ratio: np.ndarray, threshold: float, mode: str, min_dwell_s: float
) -> list[dict]:
    """
    Contiguous runs (duration >= min_dwell_s) where load_ratio stays
    continuously below (mode="low") or above (mode="high") threshold.
    Mirrors extract_load_trace.find_dwell_fraction's contiguous-run
    detection loop exactly, but returns each qualifying run's own boundaries
    (start_s/end_s/duration_s) instead of an aggregate fraction, since
    building onset trajectories needs individual event timing. Same seam
    caveat as find_ramp_events: caller must not pass samples spanning a
    segment boundary.
    """
    if mode == "low":
        mask = load_ratio <= threshold
    elif mode == "high":
        mask = load_ratio >= threshold
    else:
        raise ValueError(f"mode must be 'low' or 'high', got {mode!r}")

    events = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            end_idx = min(j, n - 1)
            duration = t_seconds[end_idx] - t_seconds[i]
            if duration >= min_dwell_s:
                events.append(
                    {
                        "start_s": float(t_seconds[i]),
                        "end_s": float(t_seconds[end_idx]),
                        "duration_s": float(duration),
                    }
                )
            i = j
        else:
            i += 1
    return events


# ---------------------------------------------------------------------------
# Per-fault-type severity(t) episode builders
# ---------------------------------------------------------------------------

def _build_starvation_episode(event: dict, rng: np.random.Generator) -> dict:
    """
    Fast onset (~2-5s ramp, per Task 2 brief), triggered by a single ramp
    event. Ramps linearly to a target severity, holds briefly, then decays --
    a short symmetric bump, since O2 starvation during a load ramp is a
    transient that resolves once air supply catches up.

    Target severity is sampled independently of the triggering ramp's
    magnitude (NOT derived from peak_rate). Earlier this was a linear map of
    peak_rate onto [0.5, 1.0], on the reasoning that harder ramps should
    produce more severe starvation -- but this trace's ramp-rate distribution
    is bimodal with a gap (routine load steps cluster ~0.0135/s, well below
    the 0.02 trigger; genuine fast transitions cluster tightly ~0.0264-
    0.0275/s, per extract_load_trace.py's report), so 33/34 triggering events
    landed within that narrow top slice and the map saturated severity_target
    at ~[0.85, 1.0] for nearly every event instead of spanning a meaningful
    range. Decoupled sampling is used instead, purely for severity diversity
    in generated training data -- see the CALIBRATION CHOICE comment below.
    """
    onset_start_s = event["start_s"]
    # CALIBRATION CHOICE, not literature-derived: [2, 5]s draw for each of the
    # ramp/plateau/decay stages, matching the brief's "~2-5s" but with no
    # source for the exact distribution shape (uniform) or for splitting the
    # ~2-5s three ways rather than some other split.
    onset_dur = float(rng.uniform(2.0, 5.0))
    plateau_dur = float(rng.uniform(2.0, 5.0))
    decay_dur = float(rng.uniform(2.0, 5.0))
    # CALIBRATION CHOICE: severity_target drawn independently and uniformly
    # from [0.3, 1.0], deliberately wider than flooding's/drying's [0.7-0.8,
    # 1.0] target ranges -- starvation is a fast transient with no real
    # magnitude signal available to scale it by (see docstring above), so a
    # broad, unconditioned range is used to give downstream training data
    # severity diversity rather than clustering near full severity.
    severity_target = float(rng.uniform(0.3, 1.0))

    def severity_fn(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        dt = t - onset_start_s
        sev = np.zeros_like(dt)

        ramp = (dt >= 0) & (dt < onset_dur)
        sev[ramp] = severity_target * dt[ramp] / onset_dur

        plateau = (dt >= onset_dur) & (dt < onset_dur + plateau_dur)
        sev[plateau] = severity_target

        decay = (dt >= onset_dur + plateau_dur) & (dt < onset_dur + plateau_dur + decay_dur)
        frac = (dt[decay] - onset_dur - plateau_dur) / decay_dur
        sev[decay] = severity_target * (1.0 - frac)

        return sev

    episode_end_s = onset_start_s + onset_dur + plateau_dur + decay_dur
    return {
        "fault_type": "starvation",
        "onset_start_s": onset_start_s,
        "episode_end_s": episode_end_s,
        "severity_fn": severity_fn,
        "severity_target": severity_target,
        "trigger_duration_s": event["duration_s"],
        "trigger_metric": event["peak_rate"],  # kept as metadata only, no longer drives severity_target
        "trigger_metric_name": "peak_ramp_rate_per_s",
    }


def _build_flooding_episode(event: dict, rng: np.random.Generator) -> dict:
    """
    Fast onset but OSCILLATORY (per Task 2 brief): an envelope ramps up fast
    (~2-5s) at dwell start, holds through the sustained low-load dwell, then
    decays fast (~2-5s) once the dwell ends (load rises, drainage improves).
    Within the envelope, severity oscillates on a tens-of-seconds cycle:
    accumulate toward a ceiling over the first 70% of each cycle, then
    breakthrough/drain sharply over the remaining 30% back down to a partial
    floor (not fully clearing) -- water doesn't fully drain between
    breakthrough events.
    """
    onset_start_s = event["start_s"]
    dwell_end_s = event["end_s"]
    dwell_dur = dwell_end_s - onset_start_s
    # CALIBRATION CHOICE, not literature-derived: same [2, 5]s envelope
    # draw as starvation's onset/decay, reused here for the fast ramp-up/
    # recovery of the oscillation's amplitude envelope; matches the brief's
    # "fast onset" qualifier but the exact distribution is tuned.
    fast_dur = float(rng.uniform(2.0, 5.0))
    decay_dur = float(rng.uniform(2.0, 5.0))
    # CALIBRATION CHOICE: oscillation period drawn from [20, 45]s -- matches
    # the brief's "tens of seconds" qualifier but the specific range and
    # uniform-draw shape are tuned, not derived from any accumulate/
    # breakthrough/drain timing reported in the literature.
    cycle_period = float(rng.uniform(20.0, 45.0))
    # CALIBRATION CHOICE: partial-drain floor -- each breakthrough cycle only
    # drains back down to 35% of severity_target, not to 0, on the modeling
    # assumption that water doesn't fully clear between breakthrough events.
    # The 0.35 value itself is an illustrative constant, not measured.
    floor = 0.35
    # CALIBRATION CHOICE: flooding's target severity once fully established,
    # drawn uniformly from [0.7, 1.0] -- no literature source for this
    # specific range, chosen only to make flooding "quite severe" per the
    # brief's general framing.
    severity_target = float(rng.uniform(0.7, 1.0))

    def severity_fn(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        dt = t - onset_start_s
        dt_clip = np.clip(dt, 0, None)

        env = np.ones_like(dt)
        ramp = dt < fast_dur
        env[ramp] = np.clip(dt[ramp] / fast_dur, 0.0, 1.0)
        after = dt >= dwell_dur
        frac = np.clip((dt[after] - dwell_dur) / decay_dur, 0.0, 1.0)
        env[after] = np.clip(1.0 - frac, 0.0, 1.0)
        env = np.clip(env, 0.0, 1.0)

        # CALIBRATION CHOICE: 70/30 accumulate/drain split of each cycle, and
        # the -3.0 decay-rate constant shaping the accumulate curve -- both
        # illustrative, not measured or literature-derived.
        phase = np.mod(dt_clip, cycle_period) / cycle_period
        osc = np.empty_like(phase)
        acc = phase < 0.7
        osc[acc] = floor + (1.0 - floor) * (1.0 - np.exp(-3.0 * phase[acc] / 0.7))
        drain = ~acc
        frac2 = (phase[drain] - 0.7) / 0.3
        osc[drain] = 1.0 - (1.0 - floor) * frac2

        sev = severity_target * env * osc
        sev[dt < 0] = 0.0
        return sev

    episode_end_s = dwell_end_s + decay_dur
    return {
        "fault_type": "flooding",
        "onset_start_s": onset_start_s,
        "episode_end_s": episode_end_s,
        "severity_fn": severity_fn,
        "severity_target": severity_target,
        "trigger_duration_s": event["duration_s"],
        "trigger_metric": dwell_dur,
        "trigger_metric_name": "dwell_duration_s",
    }


def _build_drying_episode(event: dict, rng: np.random.Generator) -> dict:
    """
    Slow, roughly monotonic onset (tens of seconds to minutes, per Task 2
    brief), triggered by a sustained high-load dwell: severity rises along a
    saturating exponential with time constant tau in [30, 90]s toward a
    target, for as long as the dwell lasts -- if the dwell ends before
    saturating, severity simply doesn't reach the target within this episode
    (explicitly allowed by the brief). After the dwell ends, severity decays
    back toward 0 (membrane rehumidifies once load drops) on a comparable
    time constant.
    """
    onset_start_s = event["start_s"]
    dwell_end_s = event["end_s"]
    dwell_dur = dwell_end_s - onset_start_s
    # CALIBRATION CHOICE: [30, 90]s time constant -- matches the brief's
    # "tens of seconds to minutes" qualifier, but the specific range, the
    # uniform-draw shape, and the choice of a saturating exponential (vs.
    # some other slow-monotonic curve) are tuned, not literature-derived.
    tau = float(rng.uniform(30.0, 90.0))
    # CALIBRATION CHOICE: recovery time constant, drawn independently from
    # [20, 60]s -- "comparable, maybe a bit faster than onset" is a modeling
    # assumption (membrane rehydration once load drops), not a measured rate.
    recovery_tau = float(rng.uniform(20.0, 60.0))
    # CALIBRATION CHOICE: target severity drawn from [0.8, 1.0] -- no
    # literature source, chosen only to make drying "severe once fully
    # developed" per the brief's general framing.
    severity_target = float(rng.uniform(0.8, 1.0))
    sev_at_dwell_end = severity_target * (1.0 - np.exp(-dwell_dur / tau))

    def severity_fn(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        dt = t - onset_start_s
        sev = np.zeros_like(dt)

        rising = (dt >= 0) & (dt < dwell_dur)
        sev[rising] = severity_target * (1.0 - np.exp(-dt[rising] / tau))

        after = dt >= dwell_dur
        sev[after] = sev_at_dwell_end * np.exp(-(dt[after] - dwell_dur) / recovery_tau)

        return sev

    episode_end_s = dwell_end_s + 3.0 * recovery_tau  # ~95% decayed by 3 time constants
    return {
        "fault_type": "drying",
        "onset_start_s": onset_start_s,
        "episode_end_s": episode_end_s,
        "severity_fn": severity_fn,
        "severity_target": severity_target,
        "trigger_duration_s": event["duration_s"],
        "trigger_metric": tau,
        "trigger_metric_name": "onset_time_constant_s",
    }


_EPISODE_BUILDERS = {
    "starvation": _build_starvation_episode,
    "flooding": _build_flooding_episode,
    "drying": _build_drying_episode,
}


def find_near_seam_buffer(
    t_seconds: np.ndarray, segment_id: np.ndarray, window_s: float = NEAR_SEAM_WINDOW_S
) -> np.ndarray:
    """
    Boolean mask, True for timesteps within `window_s` seconds of either side
    of an INTERNAL segment-boundary seam (i.e. the two concatenation points
    between the 3 real driving segments) -- not the true start of segment 0
    or the true end of the last segment, since those are real boundaries of
    the driving data, not artificial seams, and carry no reset artifact.

    Flags both the tail of the earlier segment and the head of the later
    segment around each seam, since the artifact described in
    generate_labeled_run's docstring (a fault episode hard-truncated at one
    segment's end, immediately followed by a fresh episode re-onsetting at
    the very first sample of the next) can land on either side.
    """
    near = np.zeros(len(t_seconds), dtype=bool)
    segs = np.unique(segment_id)
    for s in segs:
        idx = np.flatnonzero(segment_id == s)
        t_seg = t_seconds[idx]
        if s != segs[-1]:
            near[idx[(t_seg[-1] - t_seg) <= window_s]] = True
        if s != segs[0]:
            near[idx[(t_seg - t_seg[0]) <= window_s]] = True
    return near


def _install_episode(
    t_seg: np.ndarray,
    severity_seg: np.ndarray,
    fault_type_seg: np.ndarray,
    onset_field_seg: np.ndarray,
    episode: dict,
) -> None:
    """
    Writes one episode's severity_fn into the segment-local arrays (views
    into the global per-timestep arrays), only where this episode's severity
    exceeds whatever is currently installed -- the merge rule for the rare
    case of two episodes' spans overlapping (physically uncommon, since the
    three trigger conditions are close to mutually exclusive by
    construction, but handled defensively).
    """
    idx0 = int(np.searchsorted(t_seg, episode["onset_start_s"], side="left"))
    idx1 = int(np.searchsorted(t_seg, episode["episode_end_s"], side="right"))
    idx1 = min(idx1, len(t_seg))
    if idx0 >= idx1:
        return

    local_t = t_seg[idx0:idx1]
    sev_vals = episode["severity_fn"](local_t)

    for k in range(idx0, idx1):
        v = sev_vals[k - idx0]
        if v > severity_seg[k]:
            severity_seg[k] = v
            fault_type_seg[k] = episode["fault_type"]
            onset_field_seg[k] = max(0.0, local_t[k - idx0] - episode["onset_start_s"])


# ---------------------------------------------------------------------------
# Full labeled run
# ---------------------------------------------------------------------------

def generate_labeled_run(trace_path: str | None = None, base_seed: int = BASE_SEED) -> dict:
    """
    Loads the real dynamic-only load trace and produces a full labeled
    synthetic run: for every timestep, a (current_grid, temp_grid,
    active_fault_type_or_none, severity, onset_field_s, cv_j,
    near_seam_buffer) tuple.

    onset_field_s is signed: >= 0 and counting up from an active episode's
    own onset ("seconds since onset") while inside that episode's span;
    negative and counting down to the next known episode's onset within the
    same segment ("seconds until onset") while no fault is active; NaN if no
    episode (past or future) is relevant at that index within its segment.

    Every trigger-detection and episode-installation step is scoped to a
    single segment_id's own index range -- fault state never carries across
    the trace's 2 artificial seams, and ramp rate is never computed across
    one either (find_ramp_events/find_dwell_events only ever see one
    segment's own t_seconds/load_ratio slice). This is correct in the sense
    that no state leaks across a seam, but it produces a physically
    unrealistic artifact right at each seam: a fault episode active at one
    segment's last sample gets hard-truncated there (severity drops straight
    to 0 instead of decaying naturally), and because the 3 real driving
    segments are similarly-shaped cycles that tend to both start and end
    inside a qualifying dwell, a fresh episode often re-onsets immediately at
    the very first sample of the next segment. The returned `near_seam_buffer`
    boolean array flags timesteps close enough to either of the 2 internal
    seams to contain this artifact -- any training set built from this
    generator's output should exclude those timesteps, not because the
    no-carryover logic is wrong, but because the concatenation itself isn't
    physically continuous there.
    """
    if trace_path is None:
        trace_path = DEFAULT_TRACE_PATH
    d = np.load(trace_path)
    t_seconds = d["t_seconds"]
    load_ratio = d["load_ratio"]
    segment_id = d["segment_id"]

    n = len(t_seconds)
    severity = np.zeros(n, dtype=float)
    fault_type = np.full(n, None, dtype=object)
    onset_field = np.full(n, np.nan, dtype=float)
    all_episodes: list[dict] = []

    rng = np.random.default_rng(base_seed)

    for seg in np.unique(segment_id):
        idx = np.flatnonzero(segment_id == seg)
        idx0, idx1 = int(idx[0]), int(idx[-1]) + 1
        t_seg = t_seconds[idx0:idx1]
        lr_seg = load_ratio[idx0:idx1]

        ramp_events = find_ramp_events(t_seg, lr_seg, RAMP_THRESHOLD)
        low_events = find_dwell_events(t_seg, lr_seg, LOW_LOAD_THRESH, "low", MIN_DWELL_S)
        high_events = find_dwell_events(t_seg, lr_seg, HIGH_LOAD_THRESH, "high", MIN_DWELL_S)

        seg_episodes = (
            [_build_starvation_episode(e, rng) for e in ramp_events]
            + [_build_flooding_episode(e, rng) for e in low_events]
            + [_build_drying_episode(e, rng) for e in high_events]
        )

        sev_seg = severity[idx0:idx1]
        ft_seg = fault_type[idx0:idx1]
        onset_seg = onset_field[idx0:idx1]

        for ep in seg_episodes:
            _install_episode(t_seg, sev_seg, ft_seg, onset_seg, ep)

        # "seconds until onset" for indices with no active episode: countdown
        # to the next episode's onset within THIS segment only, never a
        # neighboring one.
        onset_times_sorted = sorted(ep["onset_start_s"] for ep in seg_episodes)
        for i in range(len(t_seg)):
            if ft_seg[i] is None and onset_times_sorted:
                j = int(np.searchsorted(onset_times_sorted, t_seg[i], side="left"))
                if j < len(onset_times_sorted):
                    onset_seg[i] = -(onset_times_sorted[j] - t_seg[i])

        # Derive each episode's own precursor window + achieved peak severity
        # by finely sampling its own severity_fn -- never a hardcoded table.
        for ep in seg_episodes:
            fine_t = np.linspace(ep["onset_start_s"], ep["episode_end_s"], 200)
            fine_sev = ep["severity_fn"](fine_t)
            ep["achieved_peak_severity"] = float(fine_sev.max())
            above = np.flatnonzero(fine_sev >= CLASSIFICATION_THRESHOLD)
            ep["precursor_window_s"] = float(fine_t[above[0]] - ep["onset_start_s"]) if above.size else None
            ep["segment_id"] = int(seg)
            ep["episode_duration_s"] = ep["episode_end_s"] - ep["onset_start_s"]
            all_episodes.append(ep)

    current_grids = np.empty((n, ROWS, COLS), dtype=np.float64)
    temp_grids = np.empty((n, ROWS, COLS), dtype=np.float64)
    cv_j = np.empty(n, dtype=float)
    for i in range(n):
        ft, sev = fault_type[i], severity[i]
        cg = generate_current_grid(ft, sev, seed=base_seed + i)
        tg = generate_temperature_grid(cg, ft, sev, seed=base_seed + i)
        current_grids[i] = cg
        temp_grids[i] = tg
        cv_j[i] = compute_cv(cg)

    near_seam_buffer = find_near_seam_buffer(t_seconds, segment_id, NEAR_SEAM_WINDOW_S)

    return {
        "t_seconds": t_seconds,
        "load_ratio": load_ratio,
        "segment_id": segment_id,
        "current_grid": current_grids,
        "temp_grid": temp_grids,
        "active_fault_type": fault_type,
        "severity": severity,
        "onset_field_s": onset_field,
        "cv_j": cv_j,
        "near_seam_buffer": near_seam_buffer,
        "episodes": all_episodes,
    }


if __name__ == "__main__":
    run = generate_labeled_run()
    t_seconds = run["t_seconds"]
    load_ratio = run["load_ratio"]
    segment_id = run["segment_id"]
    severity = run["severity"]
    cv_j = run["cv_j"]
    near_seam_buffer = run["near_seam_buffer"]
    episodes = run["episodes"]

    os.makedirs(FIGURES_DIR, exist_ok=True)
    t_min = t_seconds / 60.0

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(
        "Synthetic Fault-Onset Generator -- Real Dynamic-Only Load Trace (Normal_Flow FC-DLC)",
        fontsize=12, fontweight="bold",
    )

    ax1.plot(t_min, load_ratio, color="#333333", linewidth=0.8)
    ax1.set_ylabel("load_ratio")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_min, cv_j, color="#762a83", linewidth=0.8)
    ax2.set_ylabel("CV(J)")
    ax2.grid(True, alpha=0.3)

    ax3.plot(t_min, severity, color="#333333", linewidth=0.6)
    ax3.set_ylabel("severity")
    ax3.set_xlabel("Elapsed time (min)")
    ax3.grid(True, alpha=0.3)

    for s in np.unique(segment_id):
        mask = segment_id == s
        for ax in (ax1, ax2, ax3):
            ax.axvspan(t_min[mask][0], t_min[mask][-1], alpha=0.04, color="black" if s % 2 else "gray", zorder=0)

    # Hatch the near-seam buffer windows -- visual flag for the hard
    # reset-then-reonset artifact described in generate_labeled_run's
    # docstring; these windows should be excluded from any training set.
    nsb = near_seam_buffer.astype(int)
    starts = np.flatnonzero(np.diff(np.concatenate(([0], nsb))) == 1)
    ends = np.flatnonzero(np.diff(np.concatenate((nsb, [0]))) == -1)
    for s0, s1 in zip(starts, ends):
        for ax in (ax1, ax2, ax3):
            ax.axvspan(t_min[s0], t_min[s1], color="black", alpha=0.08, hatch="//", zorder=0.5)

    for ep in episodes:
        color = FAULT_COLORS[ep["fault_type"]]
        span = (ep["onset_start_s"] / 60.0, ep["episode_end_s"] / 60.0)
        for ax in (ax1, ax2, ax3):
            ax.axvspan(*span, color=color, alpha=0.25, zorder=1)

    legend_handles = [
        plt.Line2D([0], [0], color=c, lw=6, alpha=0.5, label=k) for k, c in FAULT_COLORS.items()
    ]
    ax3.legend(handles=legend_handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "fault_generator_run_normal_flow.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {save_path}")

    print()
    print("=== Onset-event summary (dynamic-only trace, Normal_Flow FC-DLC) ===")
    for label in ("starvation", "flooding", "drying"):
        evs = [e for e in episodes if e["fault_type"] == label]
        print(f"\n{label.upper()} -- {len(evs)} onset events")
        if not evs:
            continue
        durs = [e["episode_duration_s"] for e in evs]
        peaks = [e["achieved_peak_severity"] for e in evs]
        pws = [e["precursor_window_s"] for e in evs if e["precursor_window_s"] is not None]
        n_no_cross = sum(1 for e in evs if e["precursor_window_s"] is None)
        print(f"  episode duration:  min={min(durs):.1f}s  max={max(durs):.1f}s  mean={np.mean(durs):.1f}s")
        print(f"  achieved severity: min={min(peaks):.3f}  max={max(peaks):.3f}  mean={np.mean(peaks):.3f}")
        if pws:
            print(
                f"  precursor window:  min={min(pws):.1f}s  max={max(pws):.1f}s  mean={np.mean(pws):.1f}s  "
                f"(n={len(pws)}/{len(evs)} crossed the {CLASSIFICATION_THRESHOLD} classification threshold)"
            )
        if n_no_cross:
            print(f"  {n_no_cross}/{len(evs)} event(s) never crossed the {CLASSIFICATION_THRESHOLD} threshold within their episode")

    starv_targets = np.array([e["severity_target"] for e in episodes if e["fault_type"] == "starvation"])
    print()
    print("=== STARVATION severity_target distribution (decoupled from ramp magnitude) ===")
    print(f"  n={len(starv_targets)}  min={starv_targets.min():.3f}  max={starv_targets.max():.3f}  "
          f"mean={starv_targets.mean():.3f}  median={np.median(starv_targets):.3f}")
    bin_edges = np.linspace(0.3, 1.0, 8)
    hist, edges = np.histogram(starv_targets, bins=bin_edges)
    for count, lo, hi in zip(hist, edges[:-1], edges[1:]):
        print(f"    [{lo:.2f}, {hi:.2f}): {'#' * count} ({count})")

    n_near_seam = int(near_seam_buffer.sum())
    print()
    print(f"=== near_seam_buffer (window={NEAR_SEAM_WINDOW_S:.0f}s each side of the 2 internal seams) ===")
    print(f"  {n_near_seam}/{len(near_seam_buffer)} timesteps flagged ({100 * n_near_seam / len(near_seam_buffer):.1f}% of the full run)")
