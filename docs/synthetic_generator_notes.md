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

### 4. Synthetic-vs-real CV(J) gap (~5.7x) — and a units bug entangled with finding it

**Headline finding**: synthetic healthy (NORMAL, severity=0) CV(J) ≈
**0.067** (notebook 01's own generate_normal() output; this generator's
full healthy population, n=795 non-`near_seam_buffer` timesteps, averages
0.0762) vs. real healthy Normal_Flow CV(J) ≈ **0.381** (notebook 02) — a
**~5.7x gap** (5.0x against the generator's own population mean). This is
the first time CV(J) has been checked synthetic-vs-real at all; CV(J) had
only ever been computed on real data before this notebook. The gap is
**scale-invariant and unaffected by the units correction below** — CV(J)'s
value doesn't change no matter what units the underlying grid is in, since
it's a ratio of two quantities in the same units.

**Corrected attribution — an earlier version of this finding was wrong, and
here is why**: an initial pass at this comparison decomposed the CV(J) gap
into mean(J) and σ(J) components and found synthetic mean(J) (0.697 A/cm²)
was ~7.84x higher than real mean(\|J\|) (0.0889 A/cm²), and attributed the
whole CV(J) gap to notebook 01 assuming too high an operating current. **That
attribution was wrong, and the reason it was wrong is itself the finding
below.** The real `current_grid` values, as parsed by
`scripts/load_real_data.py` at the time, were per-segment current in Amps,
not A/cm² density — a units bug, not a physical operating-point difference.
Corrected (see units-bug evidence below), real mean(\|J\|) is **0.576
A/cm²**, and synthetic/real operating points actually agree within **~21%**
(0.697 vs. 0.576) — not 7.84x. The CV(J) gap doesn't come from mean(J) at
all. It comes from **σ(J)**: real σ(J), corrected, is **0.219 A/cm²**,
against synthetic's 0.047 A/cm² — **real spatial spread is ~4.67x larger
than synthetic's.** The generator's healthy spatial pattern
(`generate_normal_grid`, ported unchanged from notebook 01) is **too
spatially uniform relative to real healthy operation**, not built around too
high a current level. The earlier "operating point too high" framing is
retracted here explicitly, not silently replaced.

**Why this was invisible until now**: the original Task 1 real-vs-synthetic
comparison (`02_real_data_exploration.ipynb`) only ever checked raw σ(J)
directly in absolute terms (real 0.0338 vs. synthetic 0.047 A/cm², at the
time — "same order of magnitude"). It never checked mean(J) as an
independent quantity, and CV(J) (which divides by mean(J), and would have
surfaced the units bug immediately, since a ~6.5x error in mean(J) doesn't
hide inside a ratio the way it hides inside two separately-eyeballed
absolute numbers) didn't exist as a metric until later in that same
notebook — and even then, only ever evaluated on real data. No independent
cross-check of the real grid's absolute scale existed anywhere in the
project until this review.

**The units bug, as its own recorded finding** (full detail:
`docs/real_data_notes.md`'s "Units fix" note; fix applied in
`scripts/load_real_data.py`; regression test in
`scripts/test_load_real_data.py`): the real CDM_C grid's raw values are
per-segment current in Amps, not A/cm² density, confirmed by integrating the
grid per timestep and comparing against the bulk file's independent
`INTENSIDAD` (total load current) channel, timestamp-aligned via the
existing `merge_asof` path:

| hypothesis | computation | ratio (grid / `INTENSIDAD`) | stability |
|---|---|---|---|
| A: raw values already A/cm² (the old assumption) | `sum(\|J\|) × segment_area` | mean 0.1551 | CoV 0.75% |
| B: raw values are per-segment Amps | `sum(\|J\|)`, no area factor | **mean 1.0053** | CoV 0.75%, tighter at high load (std 0.0004) |

Reproduced on a second flow configuration (`Inverse_Hydrogen_Flow`: ratio
1.0055) to confirm this is a property of the parser, not one run.
`TOTAL_ACTIVE_AREA_CM2 = 50.0` was left as documented in the source paper
(not back-fit); the 0.53% residual is recorded as measured agreement,
implying an effective active area of ~49.7 cm² against the paper's ~50 cm².
`load_real_data.load_run()` now divides by `SEGMENT_AREA_CM2` before
returning `current_grid`, so it returns true A/cm².

**A process point, not a shared mechanism**: this units bug and the
pre-existing internal inconsistency already documented in frozen notebook 01
(cell 8's actual NORMAL J_std=0.047 vs. cell 12's hardcoded illustrative
`nui_target=0.10`, a ~2x mismatch — see `docs/real_data_notes.md`'s known
limitations) are **not the same underlying cause** — one is a real-data
unit-conversion omission in a parser, the other is a same-notebook mismatch
between a computed value and an unrelated hand-picked constant. But both
share the same *shape* of failure: a number was carried forward through the
project and used in downstream comparisons without ever being checked
against a second, independent source, until this review went looking for
one. Worth keeping in mind as a process lesson going forward — any new
absolute (non-ratio) number introduced into this project should get an
independent cross-check before it's relied on, the way `INTENSIDAD` now
cross-checks `current_grid`.

**Downstream implication — this is the headline finding, not the
arithmetic**: a synthetic healthy baseline that is ~5x more spatially
uniform than real healthy data is not a cosmetic mismatch. If any CV(J)
-based threshold or classifier is trained on this generator's healthy
output, real healthy dynamically-loaded data — data with no fault at all —
will look anomalous to it almost continuously, simply because real healthy
operation is naturally far noisier/more spatially structured than this
generator currently produces. **The planned no-false-alarm validation
(comparing a trained model against real healthy data) would fail under
these conditions, and it would fail for a reason that has nothing to do
with fault physics.** This has to be resolved — most plausibly by
recalibrating the synthetic NORMAL/fault grid generators' spatial-noise
characteristics against the real, corrected σ(J) distribution — before any
classifier or threshold work (Phase 3) is meaningful. Not addressed here;
`fault_generator.py` was not retuned as part of this review, per
instruction.
