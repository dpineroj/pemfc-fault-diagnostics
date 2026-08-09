"""
Extracted the real FC-DLC load trace (Normal_Flow) and package it as a
reusable driving signal for the synthetic fault generator.

Uses scripts.load_real_data.load_run() -- does not re-parse the raw .dat
files directly. Output is cached to data/processed/ so downstream code
(scripts/fault_generator.py, notebook 03) doesn't need to re-run the ~2s
CDM block parse + merge_asof every time.

Load-ratio normalization: INTENSIDAD is min-max normalized to a 0-1
fraction using THIS RUN's own observed min/max (1.95-39.06 A for
Normal_Flow FC-DLC), not an external nameplate/rated-current value (none is
documented for this cell -- see docs/real_data_notes.md). This means
load_ratio=1.0 means "the highest INTENSIDAD seen in this specific run," not
"the cell's absolute maximum capacity." Fine for driving a generator off
this one run; would need re-deriving (not just re-scaling) if ever combined
with another run's trace.

fix -- steady-state conditioning holds vs. dynamic cycling:
per Suarez et al. (2023), each FC-DLC run brackets three ~1180s dynamic
cycles (4x195s urban segments + 400s highway segment) with ~1-hour
steady-state conditioning holds. The full 294.7-minute Normal_Flow trace
turns out to be dominated by these holds (~231 min of the 294.7), not real
driving -- confirmed empirically below (rolling-std of load_ratio is
essentially exactly flat for ~78% of samples, at a mean load_ratio
identical across all four detected holds: 0.8340). Two traces are cached:
the full trace (as before) and a dynamic-cycling-only trace with the holds
excluded, since fault-onset triggers should be derived from real driving
behavior, not conditioning holds.

Threshold choices (ramp-rate and dwell, for both the full and dynamic-only
traces) are derived from each trace's own actual distributions -- see the
printed report and inline comments for the reasoning, not blind guesses.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from load_real_data import load_run

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def extract_load_trace(
    config: str = "Normal_Flow",
    run_type: str = "FC-DLC",
    warmup_seconds: float = 300.0,
) -> dict:
    """
    Load one FC-DLC run and extract a real-timed, 0-1 normalized load trace.

    Returns
    -------
    dict with:
        't_seconds' : np.ndarray[float64], shape (T,) -- elapsed seconds
            since the first post-warm-up sample (real CDM-clock timing,
            not resampled to an assumed-uniform grid)
        'load_ratio' : np.ndarray[float64], shape (T,) -- INTENSIDAD
            min-max normalized to [0, 1] using this run's own observed
            range; NaN rows from unmatched CDM<->bulk merges (Task 2) are
            linearly interpolated first, since they're sparse, isolated
            single-timestamp gaps, not real missing spans
        'intensidad_min', 'intensidad_max' : float -- the raw A values the
            0/1 endpoints correspond to, kept for traceability
        'n_interpolated' : int -- how many NaN rows were filled
    """
    run = load_run(config, run_type=run_type, warmup_seconds=warmup_seconds)

    ts = run["timestamp"]
    intensidad = run["bulk"]["INTENSIDAD"].to_numpy().astype(np.float64)

    n_nan = int(np.isnan(intensidad).sum())
    if n_nan:
        intensidad = pd.Series(intensidad).interpolate(limit_direction="both").to_numpy()

    t_seconds = (ts - ts[0]) / np.timedelta64(1, "s")

    i_min, i_max = float(intensidad.min()), float(intensidad.max())
    load_ratio = (intensidad - i_min) / (i_max - i_min)

    return {
        "config": config,
        "run_type": run_type,
        "t_seconds": t_seconds.astype(np.float64),
        "load_ratio": load_ratio,
        "intensidad_min": i_min,
        "intensidad_max": i_max,
        "n_interpolated": n_nan,
    }


def find_ramp_events(t_seconds: np.ndarray, load_ratio: np.ndarray, ramp_threshold: float) -> list[dict]:
    """
    Contiguous runs where |d(load_ratio)/dt| exceeds ramp_threshold, each
    counted as one distinct ramp event. Operates on whatever t_seconds/
    load_ratio slice it's given -- callers are responsible for not passing
    in samples that span a physical discontinuity (e.g. across an excluded
    hold), since a diff across such a seam would be spurious.
    """
    dt = np.diff(t_seconds)
    d_ratio = np.diff(load_ratio)
    ramp_rate = d_ratio / dt
    above = np.abs(ramp_rate) > ramp_threshold

    events = []
    i = 0
    n = len(above)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            events.append(
                {
                    "start_s": float(t_seconds[i]),
                    "end_s": float(t_seconds[j]),
                    "duration_s": float(t_seconds[j] - t_seconds[i]),
                    "peak_rate": float(np.max(np.abs(ramp_rate[i:j]))),
                    "direction": "up" if ramp_rate[i:j].mean() > 0 else "down",
                }
            )
            i = j
        else:
            i += 1
    return events


def find_dwell_fraction(
    t_seconds: np.ndarray, load_ratio: np.ndarray, threshold: float, mode: str, min_dwell_s: float
) -> tuple[float, int]:
    """
    Fraction of total time spent in contiguous dwell segments (duration >=
    min_dwell_s) where load_ratio stays continuously below (mode="low") or
    above (mode="high") threshold. Filters out brief threshold crossings
    that happen mid-ramp, which aren't real "sustained dwell." Same seam
    caveat as find_ramp_events applies to the input slice.
    """
    if mode == "low":
        mask = load_ratio <= threshold
    elif mode == "high":
        mask = load_ratio >= threshold
    else:
        raise ValueError(f"mode must be 'low' or 'high', got {mode!r}")

    total_duration = t_seconds[-1] - t_seconds[0]
    dwell_time = 0.0
    n_segments = 0
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            seg_end_idx = min(j, n - 1)
            duration = t_seconds[seg_end_idx] - t_seconds[i]
            if duration >= min_dwell_s:
                dwell_time += duration
                n_segments += 1
            i = j
        else:
            i += 1
    return dwell_time / total_duration, n_segments


def detect_holds(
    t_seconds: np.ndarray,
    load_ratio: np.ndarray,
    window_s: float = 60.0,
    std_threshold: float = 0.001,
    min_hold_s: float = 1800.0,
) -> list[dict]:
    """
    Detect steady-state conditioning holds: contiguous windows where the
    rolling std of load_ratio stays below std_threshold for at least
    min_hold_s. window_s sets the rolling window in samples (this trace
    samples at ~1 Hz, so window_s ~= window in samples).

    std_threshold=0.001 comes from this run's own rolling-std distribution:
    it has a sharp elbow between the ~78th percentile (~0.00014, still
    essentially flat) and the ~80th percentile (~0.021, genuinely varying).
    0.001 sits in that gap.
    """
    window = max(3, int(round(window_s)))
    roll_std = pd.Series(load_ratio).rolling(window, center=True, min_periods=window).std().to_numpy()
    hold_mask = np.nan_to_num(roll_std, nan=np.inf) < std_threshold

    holds = []
    i, n = 0, len(hold_mask)
    while i < n:
        if hold_mask[i]:
            j = i
            while j < n and hold_mask[j]:
                j += 1
            end_idx = min(j, n - 1)
            duration = t_seconds[end_idx] - t_seconds[i]
            if duration >= min_hold_s:
                holds.append(
                    {
                        "start_idx": i,
                        "end_idx": end_idx,
                        "start_s": float(t_seconds[i]),
                        "end_s": float(t_seconds[end_idx]),
                        "duration_s": float(duration),
                        "mean_load_ratio": float(load_ratio[i:j].mean()),
                    }
                )
            i = j
        else:
            i += 1
    return holds


def extract_dynamic_segments(
    t_seconds: np.ndarray,
    holds: list[dict],
    min_dynamic_s: float = 900.0,
) -> list[dict]:
    """
    Everything not inside a detected hold, filtered to duration >=
    min_dynamic_s to exclude tiny edge fragments (e.g. the trace starting
    a few seconds before a hold boundary -- an artifact of the warm-up-
    exclusion window and the rolling-std window's edge smearing, not a
    real dynamic cycle). min_dynamic_s is set well below the protocol's
    stated ~1180s cycle length but well above the observed ~30s edge
    fragments, so it separates the two without needing to match 1180s
    exactly.
    """
    n = len(t_seconds)
    in_hold = np.zeros(n, dtype=bool)
    for h in holds:
        in_hold[h["start_idx"] : h["end_idx"] + 1] = True

    segments = []
    i = 0
    while i < n:
        if not in_hold[i]:
            j = i
            while j < n and not in_hold[j]:
                j += 1
            end_idx = min(j, n - 1)
            duration = t_seconds[end_idx] - t_seconds[i]
            if duration >= min_dynamic_s:
                segments.append(
                    {
                        "start_idx": i,
                        "end_idx": end_idx,
                        "start_s": float(t_seconds[i]),
                        "end_s": float(t_seconds[end_idx]),
                        "duration_s": float(duration),
                    }
                )
            i = j
        else:
            i += 1
    return segments


def concatenate_segments(
    t_seconds: np.ndarray, load_ratio: np.ndarray, segments: list[dict]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenate the given segments into one continuous, rebased trace for
    caching/downstream use. Returns a segment_id array alongside so
    consumers can avoid computing a spurious ramp rate across the seam
    between two segments -- consecutive segments were not physically
    adjacent in the real run (there's a ~1 hour conditioning hold between
    them), so a naive diff() across that boundary would be meaningless.
    """
    t_parts, lr_parts, seg_id_parts = [], [], []
    cursor = 0.0
    for seg_id, seg in enumerate(segments):
        sl = slice(seg["start_idx"], seg["end_idx"] + 1)
        seg_t = t_seconds[sl]
        t_local = seg_t - seg_t[0] + cursor
        t_parts.append(t_local)
        lr_parts.append(load_ratio[sl])
        seg_id_parts.append(np.full(sl.stop - sl.start, seg_id, dtype=np.int32))
        cursor = t_local[-1] + 1.0  # +1s seam gap keeps t_seconds strictly increasing
    return np.concatenate(t_parts), np.concatenate(lr_parts), np.concatenate(seg_id_parts)


def pooled_ramp_rates(t_seconds: np.ndarray, load_ratio: np.ndarray, segments: list[dict]) -> np.ndarray:
    """abs(d(load_ratio)/dt) computed within each segment separately, then pooled."""
    parts = []
    for seg in segments:
        sl = slice(seg["start_idx"], seg["end_idx"] + 1)
        dt = np.diff(t_seconds[sl])
        d_ratio = np.diff(load_ratio[sl])
        parts.append(np.abs(d_ratio / dt))
    return np.concatenate(parts)


def pooled_load_ratio(load_ratio: np.ndarray, segments: list[dict]) -> np.ndarray:
    return np.concatenate([load_ratio[seg["start_idx"] : seg["end_idx"] + 1] for seg in segments])


def aggregate_ramp_events(
    t_seconds: np.ndarray, load_ratio: np.ndarray, segments: list[dict], ramp_threshold: float
) -> list[dict]:
    events = []
    for seg in segments:
        sl = slice(seg["start_idx"], seg["end_idx"] + 1)
        events.extend(find_ramp_events(t_seconds[sl], load_ratio[sl], ramp_threshold))
    return events


def aggregate_dwell_fraction(
    t_seconds: np.ndarray, load_ratio: np.ndarray, segments: list[dict], threshold: float, mode: str, min_dwell_s: float
) -> tuple[float, int]:
    total_dwell, total_duration, total_segments = 0.0, 0.0, 0
    for seg in segments:
        sl = slice(seg["start_idx"], seg["end_idx"] + 1)
        frac, n_seg = find_dwell_fraction(t_seconds[sl], load_ratio[sl], threshold, mode, min_dwell_s)
        seg_duration = t_seconds[sl][-1] - t_seconds[sl][0]
        total_dwell += frac * seg_duration
        total_duration += seg_duration
        total_segments += n_seg
    return total_dwell / total_duration, total_segments


def _print_dwell_ramp_report(label, t_seconds, load_ratio, ramp_threshold, ramp_events, low_thresh, high_thresh, min_dwell_s, low_frac, n_low, high_frac, n_high):
    dt = np.diff(t_seconds)
    abs_rate = np.abs(np.diff(load_ratio) / dt)
    print(f"--- {label} ---")
    print(f"ramp-rate threshold: {ramp_threshold:.5f}/s   max observed: {abs_rate.max():.5f}/s")
    print(f"distinct ramp events: {len(ramp_events)}")
    if ramp_events:
        durations = [e["duration_s"] for e in ramp_events]
        peaks = [e["peak_rate"] for e in ramp_events]
        n_up = sum(1 for e in ramp_events if e["direction"] == "up")
        print(f"  event duration: min={min(durations):.1f}s max={max(durations):.1f}s mean={np.mean(durations):.1f}s")
        print(f"  event peak rate: min={min(peaks):.5f}/s max={max(peaks):.5f}/s mean={np.mean(peaks):.5f}/s")
        print(f"  direction: {n_up} up, {len(ramp_events) - n_up} down")
    print(f"low-load dwell (load_ratio <= {low_thresh}, sustained >= {min_dwell_s:.0f}s): {low_frac * 100:.1f}% of time, {n_low} segments")
    print(f"high-load dwell (load_ratio >= {high_thresh}, sustained >= {min_dwell_s:.0f}s): {high_frac * 100:.1f}% of time, {n_high} segments")


if __name__ == "__main__":
    trace = extract_load_trace("Normal_Flow", "FC-DLC", warmup_seconds=300.0)
    t_seconds = trace["t_seconds"]
    load_ratio = trace["load_ratio"]
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ============================================================
    # Full trace (unchanged extraction, renamed for clarity)
    # ============================================================
    full_path = os.path.join(PROCESSED_DIR, "normal_flow_fc-dlc_load_trace_full.npz")
    np.savez(
        full_path,
        t_seconds=t_seconds,
        load_ratio=load_ratio,
        intensidad_min=trace["intensidad_min"],
        intensidad_max=trace["intensidad_max"],
    )
    duration_s = t_seconds[-1] - t_seconds[0]
    dt = np.diff(t_seconds)

    full_ramp_threshold = float(np.percentile(np.abs(np.diff(load_ratio) / dt), 95))
    full_ramp_events = find_ramp_events(t_seconds, load_ratio, full_ramp_threshold)
    full_low_thresh, full_high_thresh, min_dwell_s = 0.25, 0.70, 10.0
    full_low_frac, full_n_low = find_dwell_fraction(t_seconds, load_ratio, full_low_thresh, "low", min_dwell_s)
    full_high_frac, full_n_high = find_dwell_fraction(t_seconds, load_ratio, full_high_thresh, "high", min_dwell_s)

    print(f"Saved -> {full_path}")
    print(f"n_interpolated NaN rows: {trace['n_interpolated']}")
    print(f"total duration: {duration_s:.1f}s ({duration_s / 60:.1f} min), samples: {len(t_seconds)}")
    print(f"INTENSIDAD range: {trace['intensidad_min']:.2f} - {trace['intensidad_max']:.2f} A")
    print()
    _print_dwell_ramp_report(
        "FULL TRACE (holds + dynamic cycling mixed together)",
        t_seconds, load_ratio, full_ramp_threshold, full_ramp_events,
        full_low_thresh, full_high_thresh, min_dwell_s, full_low_frac, full_n_low, full_high_frac, full_n_high,
    )

    # ============================================================
    # Task 1b: isolate genuine dynamic-cycling segments
    # ============================================================
    print()
    print("=== SEGMENTATION: steady-state holds vs. dynamic cycling ===")
    holds = detect_holds(t_seconds, load_ratio, window_s=60.0, std_threshold=0.001, min_hold_s=1800.0)
    print(f"steady-state holds detected: {len(holds)}")
    for h in holds:
        print(
            f"  {h['start_s']:8.1f}s -> {h['end_s']:8.1f}s  dur={h['duration_s']:7.1f}s "
            f"({h['duration_s'] / 60:5.1f} min)  mean_load_ratio={h['mean_load_ratio']:.4f}"
        )
    total_hold_s = sum(h["duration_s"] for h in holds)
    print(f"total hold time: {total_hold_s:.1f}s ({total_hold_s / 60:.1f} min)")

    dynamic_segments = extract_dynamic_segments(t_seconds, holds, min_dynamic_s=900.0)
    print(f"\ngenuine dynamic-cycling segments detected: {len(dynamic_segments)}")
    for s in dynamic_segments:
        print(f"  {s['start_s']:8.1f}s -> {s['end_s']:8.1f}s  dur={s['duration_s']:7.1f}s ({s['duration_s'] / 60:5.1f} min)")
    total_dynamic_s = sum(s["duration_s"] for s in dynamic_segments)
    print(f"total dynamic time: {total_dynamic_s:.1f}s ({total_dynamic_s / 60:.1f} min)")
    print(
        f"(hold + dynamic = {total_hold_s + total_dynamic_s:.1f}s vs. full trace {duration_s:.1f}s; "
        f"the ~{duration_s - total_hold_s - total_dynamic_s:.0f}s difference is edge fragments excluded "
        f"by the min_hold_s/min_dynamic_s duration filters and rolling-window edge smearing)"
    )

    # ---- Save the dynamic-only trace ----
    dyn_t, dyn_lr, dyn_seg_id = concatenate_segments(t_seconds, load_ratio, dynamic_segments)
    dynamic_path = os.path.join(PROCESSED_DIR, "normal_flow_fc-dlc_load_trace_dynamic_only.npz")
    np.savez(
        dynamic_path,
        t_seconds=dyn_t,
        load_ratio=dyn_lr,
        segment_id=dyn_seg_id,
        intensidad_min=trace["intensidad_min"],
        intensidad_max=trace["intensidad_max"],
    )
    print(f"\nSaved -> {dynamic_path}")
    print(
        "(concatenated from the dynamic segments above, time rebased continuous with a 1s seam gap; "
        "segment_id marks which original segment each sample came from -- do not compute ramp rate "
        "across a segment_id boundary, consecutive segments were not physically adjacent in the real run.)"
    )

    # ============================================================
    # Recompute ramp-rate / dwell thresholds on the DYNAMIC-ONLY data
    # ============================================================
    dyn_abs_rate = pooled_ramp_rates(t_seconds, load_ratio, dynamic_segments)
    dyn_load_pool = pooled_load_ratio(load_ratio, dynamic_segments)

    # Ramp threshold: with holds removed, "flat/noise" no longer dominates,
    # so the elbow moves out much further than the full trace's 95th
    # percentile. Here routine small load-steps cluster tightly at
    # ~0.0135-0.0140/s all the way through the 99th percentile; only above
    # the 99.3rd percentile do genuinely fast edge transitions appear
    # (~0.027/s). 0.02/s is used as the threshold, sitting in that gap.
    dyn_ramp_threshold = 0.02
    dyn_ramp_events = aggregate_ramp_events(t_seconds, load_ratio, dynamic_segments, dyn_ramp_threshold)

    # Dwell thresholds: the dynamic-only load_ratio distribution is much
    # more continuously graded than the full trace's sharply bimodal one
    # (makes sense -- the flat holds that created that bimodality are gone).
    # There's no single sharp elbow here; percentiles rise fairly smoothly
    # (25th~0.04, 50th~0.23, 70th~0.39, 90th~0.82, with the top ~5-10%
    # approaching the hold plateau level 0.834 as segments ramp back up
    # toward the next hold). 0.10 and 0.40 are chosen as reasonable
    # dividing points bracketing roughly the bottom third (low/ripple-like)
    # and top 30% (high/highway-like) of samples, not a precise gap.
    dyn_low_thresh, dyn_high_thresh = 0.10, 0.40
    dyn_low_frac, dyn_n_low = aggregate_dwell_fraction(t_seconds, load_ratio, dynamic_segments, dyn_low_thresh, "low", min_dwell_s)
    dyn_high_frac, dyn_n_high = aggregate_dwell_fraction(t_seconds, load_ratio, dynamic_segments, dyn_high_thresh, "high", min_dwell_s)

    print()
    print("=== BEFORE / AFTER: full trace vs. dynamic-only ===")
    print(f"full trace:      low-load dwell {full_low_frac * 100:5.1f}%  ({full_n_low} segments, thresh<={full_low_thresh})   "
          f"high-load dwell {full_high_frac * 100:5.1f}%  ({full_n_high} segments, thresh>={full_high_thresh})")
    print(f"dynamic-only:    low-load dwell {dyn_low_frac * 100:5.1f}%  ({dyn_n_low} segments, thresh<={dyn_low_thresh})   "
          f"high-load dwell {dyn_high_frac * 100:5.1f}%  ({dyn_n_high} segments, thresh>={dyn_high_thresh})")
    print(f"full trace:      {len(full_ramp_events)} ramp events above {full_ramp_threshold:.5f}/s")
    print(f"dynamic-only:    {len(dyn_ramp_events)} ramp events above {dyn_ramp_threshold:.5f}/s")
    print()
    _print_dwell_ramp_report(
        "DYNAMIC-ONLY TRACE (holds excluded) -- USE THESE FOR TASK 2 TRIGGERS",
        dyn_t, dyn_lr, dyn_ramp_threshold, dyn_ramp_events,
        dyn_low_thresh, dyn_high_thresh, min_dwell_s, dyn_low_frac, dyn_n_low, dyn_high_frac, dyn_n_high,
    )
