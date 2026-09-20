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
classifier or threshold work (Phase 3) is meaningful. Addressed below
(Phase 3, Sec. 5): `fault_generator.py` was not retuned at the time this
finding was written, but a first physically-grounded model has since been
built (additive, `generate_normal_grid` itself still untouched).

## 5. Phase 3: healthy spatial-structure model (`generate_healthy_grid_v2`)

Built to close the gap in Sec. 4. Ground rule enforced throughout: the real
healthy data is Phase 7's validation set, so nothing here was fit to it —
parameters come from physical reasoning and (where found) literature,
checked against real data only after the fact, never as a fitting target.

### 5a. Variance decomposition — static-dominant

Decomposed real Normal_Flow's spatial pattern by normalizing each timestep's
grid by its own spatial mean (removes load level, isolates shape), then
splitting the persistent time-averaged shape from the per-timestep residual:

| component | value | share of variance |
|---|---|---|
| static (persistent shape) | std=0.3802 | **99.6%** |
| temporal (per-timestep residual) | std=0.0253 | 0.4% |

Static structure dominates by ~15×. This is a static-template problem, not
a temporal-noise problem — it determines everything about the model below.

### 5b. The static pattern's shape — supersedes notebook 01's assumption, doesn't correct it

**Finding, recorded here as a distinct result, not a footnote**: notebook
01's `generate_normal` assumes a monotonic inlet-to-outlet flow-path
gradient (linear `row_grad`/`col_grad`, H₂-inlet-to-outlet framing). The
real cell's persistent shape does not show this: correlation with row index
= -0.065, column index = -0.026 — both negligible. Instead it correlates
with **distance from grid center at r=0.936**, and a radial quadratic
explains **R²=0.906** of the shape by itself. The real pattern is a
radially-symmetric "bathtub": low in the center, high toward the edges —
consistent with a **compression/clamping mechanism** (contact pressure
falling off from a bolt-clamped periphery toward the cell center), not the
reactant-depletion flow-path mechanism notebook 01 modeled.

**Notebook 01 is not modified and remains frozen.** This finding
*supersedes* its flow-path-gradient assumption for downstream generator
work (this section, and any future fault-state work built on it) — it does
not go back and correct notebook 01 in place. Notebook 01's figures and
numbers stand as originally computed; readers of this doc should treat its
NORMAL-state spatial assumption as superseded here, not as still current.

### 5c. Literature search for the amplitude — bounded effort, no citable ratio found

Six targeted queries across two review passes (segmented-cell edge/clamping
current-density distribution; clamping-pressure/current-density
correlation; commercial-size PEMFC non-uniform current density; bolt-frame
periphery current-density ratios; max/min current density under bolt
clamping; "times higher"/"% higher" edge-vs-center phrasing). Multiple
independent papers confirm the **mechanism** —

- [Analysis of local current density, temperature, and mechanical pressure distributions in an operating PEMFC under variable compression](https://www.sciencedirect.com/science/article/pii/S0306261925009171)
- [Advanced parametric model for analysis of the influence of channel cross section dimensions and clamping pressure on current density distribution in PEMFC](https://www.sciencedirect.com/science/article/abs/pii/S0306261921014094)
- [Investigation of the non-uniform distribution of current density in commercial-size proton exchange membrane fuel cells](https://www.sciencedirect.com/science/article/abs/pii/S0378775320301397)
- [Current density and temperature distribution measurement and homogeneity analysis for a large-area PEM fuel cell](https://www.sciencedirect.com/science/article/abs/pii/S0360544221021708)

— uneven clamping pressure drives uneven contact resistance, which drives
uneven local current density, typically higher at the bolted periphery of
large-area cells. **None gave a stated, citable edge-to-center or max/min
current-density ratio precise enough to use directly.** Recorded here as a
stated limitation, not papered over with an invented-sounding number.

### 5d. Model

```
S_raw(i,j) = 1 + A_edge * (r(i,j) / r_max)^p_edge
S(i,j)     = S_raw / mean(S_raw)                      # renormalize -> shape only, mean=1
J(i,j)     = load_level * S(i,j) * (1 + noise(i,j))    # noise ~ N(0, sigma_temporal), i.i.d. per cell
```
`r(i,j)` = distance from grid center; `r_max` = distance to the farthest
grid cell.

- **`p_edge = 2` — physics-derived, not chosen.** A quadratic falloff is
  the natural leading-order term for a smooth, radially-symmetric pressure
  field peaked at the clamped frame and dipping at the center (the first
  non-trivial even term in a radial expansion around the center). The real
  radial fit (R²=0.906, Sec. 5b) was checked against this assumed form, not
  used to derive it.
- **`A_edge` (via `RADIAL_EDGE_RATIO = 2.0`) — CALIBRATION CHOICE, per Sec.
  5c's stated limitation.** No citable literature ratio found, so set to
  produce an idealized center(r=0)-to-corner(r=r_max) ratio of 2.0×,
  bracketed by 1.5×/2.5× in the sensitivity table below (Sec. 5e) so the
  result's dependence on this specific choice is visible rather than hidden
  behind one precise-looking number. **Never adjusted to make CV(J) land
  near 0.381** — chosen before any CV(J) was computed with it.
- **`sigma_temporal` (via `TEMPORAL_SUBORDINATION_FACTOR = 15.0`) —
  CALIBRATION CHOICE.** Set to 1/15 of the static shape's own spatial std
  (within the instructed 1/10–1/20 "clearly subordinate" range), NOT fit to
  the real ~1/15 static:temporal ratio found in Sec. 5a — that real ratio
  is used only as the post-hoc comparison in Sec. 5f, never as a fitting
  target. (Note: any choice in the 1/10–1/20 range mathematically forces
  the static-variance share above 99% regardless of the exact value, since
  squaring a ratio ≥10 dominates — so Sec. 5f's 99.6%-vs-99.6% match below
  is a near-automatic consequence of picking a value in the instructed
  range, not a meaningful independent alignment with the real figure.)
- **`load_level` scales the whole grid; `S` is renormalized to mean 1**, so
  it carries no load information. This makes CV(J) load-invariant **by
  construction** — verified explicitly (Sec. 5e), not assumed.
- **Additive**: `generate_healthy_grid_v2` is a new function.
  `generate_normal_grid` (notebook 01's exact formula) is untouched, and
  remains what fault-state severity interpolation in this file is built
  from — this section does not wire into that interpolation yet.

### 5e. Sensitivity analysis (the point, not a single number)

CV(J), n=2000 seeds per ratio, `load_level=0.697`:

| `edge_ratio` | CV(J) mean | CV(J) range (per-seed) |
|---|---|---|
| 1.5 | 0.1232 | 0.1167 – 0.1306 |
| **2.0 (chosen)** | **0.2024** | 0.1920 – 0.2148 |
| 2.5 | 0.2577 | 0.2446 – 0.2734 |

All three land well below real Normal_Flow's 0.381 — the chosen value
(0.2024) recovers roughly half the gap from the original synthetic figure
(0.067 → 0.2024 is a 3× improvement toward 0.381), not the full gap. **This
was not tuned toward 0.381 and none of these three numbers were adjusted
after computing them.** Reported as the result to discuss, per instruction,
not as a target to keep pushing until it lands closer.

Load-invariance, verified explicitly (not assumed) at `edge_ratio=2.0`
across `load_level ∈ {0.15, 0.35, 0.697, 1.0, 1.5}`: CV(J) mean identical to
6 decimal places (0.202001) at every load level — exact, as the
renormalization design predicts.

### 5f. Honest comparison, synthetic v2 vs. real

| metric | synthetic v2 (`edge_ratio=2.0`) | real Normal_Flow | match? |
|---|---|---|---|
| CV(J) mean | 0.2024 | 0.3810 | **No** — 53% of real. See the resolution measurement below: the gap is real and is *understated* by this 4×4-vs-18×18 row. |
| static-variance share | 99.6% | 99.6% | Matches, but see Sec. 5d note — near-automatic given the instructed noise-ratio range, not independent confirmation |
| gradient vs. row/col index | 0.0000 / 0.0000 (exact, by symmetry) | -0.065 / -0.026 | Matches qualitatively — neither is a linear flow-path gradient |
| correlation with distance-from-center | 0.986 | 0.936 | Matches qualitatively — both strongly radial; synthetic is *more* purely radial since it has no other structure by construction |

#### SUPERSEDED — the original autocorrelation reading

The row below, and the scope caveat that accompanied it, stood here until the
resolution confound was actually measured. Preserved struck through rather
than deleted, because the error was one of comparison design, not arithmetic,
and that is worth keeping visible:

> ~~| spatial autocorrelation (mean lag-1 neighbor) | 0.294 | 0.935 | **No** —
> real field is far smoother cell-to-cell. Partly a resolution artifact (4×4
> vs. real 18×18 — fewer, coarser neighbor pairs), partly a genuine gap; not
> disentangled here |~~
>
> ~~**What it doesn't get right**: the absolute CV(J) magnitude (still ~1.9×
> low even at the upper sensitivity bound) **and the fine-grained
> smoothness/autocorrelation of the field.**~~
>
> ~~**Scope**: single real reference run (Normal_Flow); `p_edge=2` and the
> overall radial framing are physically motivated but not independently
> verified against a second real configuration in this pass. Autocorrelation
> comparison is confounded by grid-resolution mismatch (4×4 vs. 18×18) and
> should be read as directional, not precise.~~

That comparison was not like-for-like. It put synthetic-at-4×4 against
real-at-18×18, and lag-1 autocorrelation is a function of grid spacing: the
same smooth field yields a different value at a different resolution, because
"adjacent cell" means a different physical distance. The measurement below
replaces it.

#### Resolution measurement (`scripts/measure_resolution_effects.py`)

Real 18×18 block-averaged to 4×4 via `np.array_split` ([5,5,4,4] groups per
axis — all 324 cells used, nothing cropped); CV(J) computed per timestep
*after* downsampling, since std/mean is nonlinear; autocorrelation computed on
the static template, where block-averaging and time-averaging commute. The
synthetic model is evaluated at 18×18 through a grid-size-parameterized
re-expression of its own formula, with constants imported from
`fault_generator.py` and an assertion that the parameterized 4×4 path is
byte-identical to `generate_healthy_grid_v2` before any 18×18 number is used.

| CV(J) | 4×4 | 18×18 |
|---|---|---|
| real | 0.3344 | 0.3810 |
| synthetic | 0.2024 | 0.1713 |
| **real / synthetic** | **1.65×** | **2.22×** |

| lag-1 autocorrelation | 4×4 | 18×18 |
|---|---|---|
| real | 0.3225 | 0.9349 |
| synthetic | 0.2941 | 0.9545 |

**Autocorrelation: resolved, and it was an artifact.** At matched resolution
the two agree closely — 0.29 vs 0.32 at 4×4, 0.95 vs 0.93 at 18×18 — while
each moves by ~0.6 *across* resolutions. The model's cell-to-cell smoothness
was never wrong; the original comparison was. This is closed as a concern.

**CV(J): real, and worse than previously reported.** The gap survives
resolution matching and *widens* on the finer grid: 1.65× at 4×4, 2.22× at
18×18. Matching resolution does not shrink it, so it is a genuine amplitude
deficit in the synthetic model. Note the direction — the headline "~1.9×"
figure came from comparing synthetic-4×4 to real-18×18, which flattered the
model; the honest same-resolution figure at the validation data's native
18×18 is 2.22×.

**Where the deficit actually lives** (from
`scripts/analyze_real_structure.py`, measurement only — no parameter was
changed in response): decomposing the real template against its own quadratic
fit gives a radial component of std 0.3619 and a non-radial residual of std
0.1167 (Normal_Flow). A purely radial model, even one reproducing the real
cell's radial term exactly, would therefore top out near 0.36 — and ours
produces 0.1713 at 18×18. So the deficit is *within* the radial term's
amplitude, not merely the absence of the residual structure. Separately, the
axis-collapsed edge/center profile ratio of the real template (1.72–1.99
across sessions) sits close to the declared `RADIAL_EDGE_RATIO = 2.0`, but
these are **different quantities** — the model's parameter is an idealized
centre-to-corner ratio on the continuous field, the profile ratio collapses an
axis and averages strong and weak cells together. The real template's true 2D
spread is far wider than the profile ratio suggests (max/min 7.1×–17.6×
across sessions). The apparent closeness of 1.9 and 2.0 is therefore not
validation of the parameter, and was not treated as such.

**What this model gets right**: the qualitative FORM — radial, not
directional; no false flow-path gradient; CV(J) load-invariant by construction
and verified as such; cell-to-cell smoothness correct at matched resolution.
**What it doesn't get right**: the absolute CV(J) amplitude, by 2.22× at
18×18. Reported, not swept under a retuned parameter — nothing was adjusted
after seeing these numbers.

**Scope, updated**: the radial framing is now verified across **all four**
flow configurations, not one (`scripts/analyze_real_structure.py`):
distance-from-center correlation 0.930–0.936, quadratic R² 0.902–0.915,
row/column correlations near zero in every session, and cross-session template
correlation r = 0.976–0.999. The static template is reproducible across
independent sessions, which supports the fixture/compression reading rather
than a per-session or per-flow-configuration effect. Real CV(J) is *not* a
single number across sessions: 0.3810 / 0.3266 / 0.3812 / 0.3247 (Normal,
Inverse, Inverse_Air, Inverse_Hydrogen) — so the target band is ~0.325–0.381,
not 0.381 alone.

Reproducibility: `scripts/analyze_real_structure.py` (real structure, all four
configurations) and `scripts/measure_resolution_effects.py` (the 2×2 table
above) regenerate every real-data number in this section from the corrected
loader. Regression test: `scripts/test_fault_generator.py` — asserts CV(J) stays in
`[0.15, 0.25]` at the default `edge_ratio=2.0` and remains load-invariant to
1e-9 tolerance across 5 load levels.
