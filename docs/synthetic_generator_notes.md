# Synthetic Fault Generator Notes — `scripts/fault_generator.py` (Task 2)

Reference notes for the synthetic fault-onset generator built in Task 2.
Companion to `scripts/fault_generator.py` (implementation), grounded in
`01_spatial_viz_sandbox.ipynb` (frozen, spatial pattern source) and
`02_real_data_exploration.ipynb` (CV(J) formula source), driven by
`data/processed/normal_flow_fc-dlc_load_trace_dynamic_only.npz`.

## What's real, literature-grounded, vs. tuned

Same three-way separation `docs/real_data_notes.md` uses for the dataset:

- **Real** (measured from the actual trace/data): the trigger thresholds
  (0.02/s ramp rate, 0.10/0.40 load-ratio dwell, 10s minimum dwell duration)
  and the observed ramp-rate distribution shape that motivated decoupling
  starvation's severity (see below) — all come directly from
  `extract_load_trace.py`'s report on this specific trace.
- **Literature-grounded**: the spatial pattern formulas themselves (row/col
  gradients, which corner/side is affected per fault) come from notebook
  01's citations (Wang et al. 2024, González et al. 2025, Yang et al. 2024,
  Yu et al. 2024) — carried over unchanged, just made continuous by severity.
- **Tuned / calibration choices**: every onset-timescale constant (the [2,5]s
  ramp/plateau/decay draws, flooding's cycle period and partial-drain floor,
  drying's time constants, all three fault types' severity_target sampling
  ranges) is an engineering choice matching only the qualitative onset-speed
  description in the Task 2 brief ("fast", "tens of seconds", "slow,
  monotonic") — not derived from a specific literature source or measurement.
  Each is labeled `# CALIBRATION CHOICE` inline in `fault_generator.py` at
  its point of definition.

## Known limitations (found during Task 2 review, before commit)

### 1. Starvation severity was originally magnitude-scaled, then decoupled

First implementation mapped the triggering ramp event's peak rate linearly
onto a `[0.5, 1.0]` severity_target range, on the reasoning that a harder
ramp should produce a more severe starvation transient. Reviewing the actual
34 triggering events showed this doesn't work on this trace: ramp rates are
bimodal with a gap (routine load steps cluster ~0.0135/s, well under the
0.02 trigger; genuine fast transitions cluster tightly in ~0.0264-0.0275/s),
so 33/34 events landed in that narrow top slice and severity_target
saturated at ~[0.85, 1.0] (mean 0.965) instead of spanning the intended
range.

Fixed by decoupling: severity_target is now drawn independently and
uniformly from `[0.3, 1.0]` (a CALIBRATION CHOICE, not derived from
anything), purely for severity diversity in generated training data. Result
after the fix, same 34 events: min=0.344, max=0.991, mean=0.648, spread
roughly evenly across the `[0.3, 1.0]` range (see `fault_generator.py`'s
`__main__` output for the exact per-run histogram). `trigger_metric` /
`peak_ramp_rate_per_s` is still recorded per episode as metadata, it just no
longer drives severity.

### 2. CV(J) does not discriminate flooding from healthy operation

On this generator's output: baseline (no-fault) CV(J) mean = **0.076**;
flooding at severity > 0.5, mean = **0.086**. Flooding is *not* lower than
baseline — mildly the opposite of the raw σ(J) intuition from notebook 01 /
the Task 1 real-data finding ("flooding lowers NUI, more uniform").

**Why**: CV(J) = σ(J) / mean(|J|) is deliberately load-invariant (that's the
whole point of adopting it over raw σ(J) in Task 1 — see
`docs/real_data_notes.md`'s CV(J) recommendation). But flooding's spatial
pattern (`generate_flooding_grid`) lowers *both* σ(J) and mean(J) together —
notebook 01's own numbers: J_std 0.047→0.041, J_mean 0.697→0.432. The
normalization that makes CV(J) load-invariant also cancels almost all of the
raw-σ signal that made flooding look spatially distinct in the first place.

**Consequence for downstream work**: CV(J) remains valid as the headline
spatial-vs-bulk diagnostic argument (a spatially resolved signal carries
information a single bulk voltage sensor can't) — that argument doesn't
depend on discriminating between fault *types*. But CV(J) alone is **not
sufficient as a fault-type classifier feature**: it cannot tell flooding
apart from healthy operation on this generator's output. Any downstream
classifier (Task 3+) needs the full spatial grid, or CV(J) evaluated jointly
with mean(|J|) (which flooding does depress, per the numbers above), not
CV(J) as a single scalar feature.

Scope: this is a same-generator, single-trigger-condition observation, not
re-verified against real fault data (none exists — see
`docs/real_data_notes.md`'s limitations). It should hold for any generator
run since it follows directly from the fixed formulas, not from anything
run-specific.

### 3. Segment-boundary seams produce a hard reset-then-reonset artifact

The dynamic-only trace concatenates 3 real driving segments with 2
artificial seams (`segment_id`). Fault state correctly never carries across
a seam (a deliberate, correct design choice) — but because the 3 segments
are similarly-shaped ~1246s driving cycles that each tend to both start and
end inside a qualifying high-load dwell, this produces a real, visible
artifact at both seams: a drying episode active at one segment's last sample
gets hard-truncated there (severity drops straight to 0 instead of decaying
naturally), and a fresh drying episode's onset frequently lands at the very
first sample of the next segment. Confirmed directly on this trace: severity
drops 0.89→0.00 and 0.99→0.00 at the two seams respectively, instead of a
gradual decay.

`generate_labeled_run()` now returns a `near_seam_buffer` boolean field,
`True` within `NEAR_SEAM_WINDOW_S` (default 90s, chosen to roughly cover the
longest precursor window observed across all fault types on this run —
drying's ~76s — plus margin) of either side of the 2 internal seams. On the
current run: **362/3739 timesteps (9.7%) flagged.** Any training set built
from this generator's output should exclude these timesteps — the artifact
is a concatenation-boundary effect, not a real physical event, and would
teach a downstream model a fault signature that doesn't exist in reality.
