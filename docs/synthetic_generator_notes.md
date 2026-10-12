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
only ever been computed on real data before this notebook.

> **Later correction (cross-session measurement):** the 0.381 figure is
> Normal_Flow's alone. Real healthy CV(J) is a **band across the four
> sessions — 0.3810 / 0.3266 / 0.3812 / 0.3247, mean 0.353** (see
> `docs/real_data_notes.md`, "Spatial structure: a radial template"). Gap
> figures stated against 0.381 alone throughout this section are therefore
> against the top of the band, not its centre. Section 5f restates the
> current gap against the band.

The gap is **scale-invariant and unaffected by the units correction
below** — CV(J)'s value doesn't change no matter what units the underlying
grid is in, since it's a ratio of two quantities in the same units.

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

**Cross-session (added later):** static dominance holds in all four
configurations (99.39%–99.96% of variance), but the static/temporal *ratio*
is **not** universal — it spans **12.8× to 48.8×** (Normal_Flow 15.0×,
Inverse_Air 12.8×, Inverse_Flow 48.8×, Inverse_Hydrogen 47.5×). Normal_Flow,
the session this section was derived from, is at the noisy end. The
qualitative conclusion — static-dominant, by more than an order of magnitude
everywhere — is unaffected.

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

- **`p_edge = 2` — SUPERSEDED. The "physics-derived" justification does not
  hold, and the data contradicts p=2 specifically.**

  > ~~**`p_edge = 2` — physics-derived, not chosen.** A quadratic falloff is
  > the natural leading-order term for a smooth, radially-symmetric pressure
  > field peaked at the clamped frame and dipping at the center (the first
  > non-trivial even term in a radial expansion around the center). The real
  > radial fit (R²=0.906, Sec. 5b) was checked against this assumed form, not
  > used to derive it.~~

  **The broken link.** That argument derives a quadratic **pressure** field,
  then silently transfers the exponent to the **current-density** template.
  The chain from one to the other — contact pressure → contact resistance →
  local overpotential/polarization → local current density — is not linear at
  any step. Contact resistance falls nonlinearly with pressure and saturates;
  current density responds to overpotential through Butler–Volmer kinetics
  plus ohmic and transport terms. So a quadratic pressure field does **not**
  imply a quadratic current-density template. The original text carried that
  linearity assumption without stating it, and it was never justified. The
  pressure argument may well be correct about pressure; it says much less than
  claimed about what this model actually generates.

  **What the data says.** Fitting `a + b·rᵖ` to the real template (two free
  parameters at every p, so the comparison is like-for-like) gives R² =
  0.8760 (p=1), 0.8282 (p=1.5), **0.7639 (p=2)**, 0.6255 (p=3), 0.5008 (p=4)
  — monotonically decreasing in p. The best free quadratic, `a + b·r + c·r²`,
  fits at R²=0.9057 with `b=+0.2517, c=−0.0087`: a dominant positive *linear*
  term with slight *negative* curvature, i.e. rising and gently saturating.
  The model's form is flat at the centre with *positive* curvature — the
  opposite sign on the second-order term. p=2 is the worst-fitting of the
  plausible exponents tested, not the best.

  **Decisive evidence that it is a shape problem, not an amplitude problem**:
  under this form no single amplitude reproduces both the real template's
  spread and its depth. Matching the radial-component std (0.3619) requires
  `edge_ratio` 4.63, which yields max/min 4.57× against the real 17.5×;
  matching the real minimum requires `edge_ratio` ≈ 32, which overshoots the
  std by ~60%. Off-centre structure was checked and ruled out as an
  explanation (best-fit re-centring gains only +0.008 R²; the template's
  argmin sits at (9,9) against a geometric centre of (8.5,8.5)).

  **Status**: the exponent is now treated as an *unknown*, not a derived
  constant. It is not being replaced with p=1 — that would substitute a
  value justified by the validation data for one justified by a broken
  argument, which is worse, not better. See the healthy-family design
  discussion for how the uncertainty is carried forward instead.
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
  target. **The 15 should not be read as a universal value**: the observed
  real static/temporal ratio spans 12.8×–48.8× across the four sessions, so
  15 sits near two of them (Normal_Flow 15.0×, Inverse_Air 12.8×) and far
  from the other two (~48×). It is a defensible choice inside the instructed
  1/10–1/20 band, not a measured constant.
  (Note: any choice in the 1/10–1/20 range mathematically forces
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

All three land well below the real band (0.3247–0.3812 across the four
sessions, mean 0.353; 0.381 is Normal_Flow specifically) — the chosen value
(0.2024) recovers roughly half the gap from the original synthetic figure
(0.067 → 0.2024 is a 3× improvement toward the band), not the full gap. **This
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
| CV(J) mean | 0.2024 | 0.3810 (Normal_Flow); band 0.3247–0.3812, mean 0.353 | **No** — 53% of Normal_Flow, 57% of the four-session mean. See the resolution measurement below: the gap is real and is *understated* by this 4×4-vs-18×18 row. **Applies to the superseded point model only — see note below.** |
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

> **Superseded in part by Sec. 7.** The measurement below covers only the
> Sec. 5 form (now superseded) and only Normal_Flow. Sec. 7 repeats it for
> all four configurations and for the adopted logistic family, and finds
> that this form's resolution response runs in the *opposite direction* to
> the real data's. The conclusions below about autocorrelation being a
> resolution artifact still hold.

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
18×18 (against Normal_Flow; **2.06× against the four-session mean of 0.353**).
Matching resolution does not shrink it, so it is a genuine amplitude deficit
in the synthetic model. Note the direction — the headline "~1.9×" figure came
from comparing synthetic-4×4 to real-18×18, which flattered the model; the
honest same-resolution figure at the validation data's native 18×18 is 2.22×
against Normal_Flow, 2.06× against the band.

> **Scope note on every gap ratio in this section (2.22×, 2.06×, 1.65×, 53%,
> 57%).** These compare a *point* model — the superseded Sec. 5 form at a
> single `edge_ratio` — against the real value. For a point estimate that is
> the correct comparison, and these figures stand as descriptions of that
> model. They do **not** transfer to the adopted logistic family, which is a
> distribution rather than a point: for a family the meaningful statements
> are containment and placement (§7d), not a mean-to-mean ratio. Do not carry
> these ratios forward as the adopted family's "residual gap".

#### Where the deficit lives — sharpest characterization to date

From `scripts/analyze_real_structure.py`. **Measurement only; no parameter
was changed in response to any number here.**

Decomposing each real template against its own quadratic-in-radius fit splits
it into a radial component and a non-radial residual:

| session | radial component std | non-radial residual std |
|---|---|---|
| Normal_Flow | 0.3619 | 0.1167 |
| Inverse_Flow | 0.3123 | 0.0953 |
| Inverse_Air_Flow | 0.3609 | 0.1191 |
| Inverse_Hydrogen_Flow | 0.3097 | 0.0974 |
| **mean** | **0.3362** | **0.1071** |

**The deficit is inside the radial term's amplitude, not in the missing
residual structure.** A purely radial model that reproduced the real cell's
radial term exactly would reach ~0.34–0.36. Ours produces **0.1713** at
18×18. So even granting our model zero credit for the non-radial residual it
cannot represent, it is still short by roughly a factor of two *on the
component it does model*. Adding the missing residual structure would not
close the gap; the radial term itself is too weak.

This supersedes the earlier framing that treated the shortfall as partly a
smoothness/structure problem — it is an amplitude problem, localized to one
term.

**A caution on the edge/center ratio.** The axis-collapsed edge/center
profile ratio of the real templates (1.72–1.99 across sessions) sits close to
the declared `RADIAL_EDGE_RATIO = 2.0`. These are **different quantities**:
the model's parameter is an idealized centre-to-corner ratio on the
continuous field, while the profile ratio collapses an axis and averages
strong and weak cells together. The real templates' true 2D spread is far
wider than the profile ratio suggests — **max/min 7.1×–17.6×**. The apparent
closeness of 1.9 and 2.0 is therefore coincidental, is not validation of the
parameter, and was not treated as such.

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

## 6. Healthy-template functional form: two forms tested, one withdrawn

Sec. 5 established *that* the synthetic healthy template is too weak (CV(J)
2.22× low at 18×18) and *where* the deficit sits (inside the radial term's
amplitude, not the missing non-radial residual). This section records the
search for a form that can actually carry that amplitude. Everything here is
measurement; no bounds are set and no parameter family is adopted.

### 6a. The two-parameter power form — proposed, tested, WITHDRAWN

Form: `S_raw = d + (1−d)·(r/r_max)^p`, normalized to mean 1. `d` was intended
as a centre floor and `p` as the falloff shape, giving depth and spread
independent control — which the previous one-parameter form (`p` pinned at 2)
could not do.

Fitted independently to all four configurations (both the real template and
the model have mean exactly 1, so the fit is over `(d, p)` only, with no
amplitude nuisance parameter):

| configuration | d | p | R² | R² at p=2 (previous form) | gain |
|---|---|---|---|---|---|
| Normal_Flow | 0.0006 | 0.911 | 0.8804 | 0.7639 | +0.117 |
| Inverse_Flow | 0.0006 | 0.768 | 0.8864 | 0.7442 | +0.142 |
| Inverse_Air_Flow | 0.0006 | 0.911 | 0.8775 | 0.7623 | +0.115 |
| Inverse_Hydrogen_Flow | 0.0006 | 0.755 | 0.8817 | 0.7393 | +0.142 |

**WITHDRAWN — `d` is not identifiable in the regime the real data occupies.**
`d` railed at the search grid's lower edge in all four configurations.
Probing it directly: R² stays within 0.002 of optimum across **d ∈ [1×10⁻⁶,
0.018]** — roughly four orders of magnitude — with `p` sliding 0.910→0.925 to
compensate. The optimum sits at the `d→0` boundary. In the region where the
real cell lives the two parameters collapse onto a degenerate ridge, and the
second degree of freedom buys ~0.004 R² over a plain linear-in-radius profile
(0.876). Ranging `d` would have ranged a parameter the data cannot constrain.

**Correction to a previously stated figure.** An earlier note put the real
sessions at `d ≈ 0.02–0.14`. That came from a *different* objective — a grid
search for the `(d, p)` pair that simultaneously matched the template's std
and its max/min — not from a least-squares fit to the template. A proper fit
puts `d` at ~0.0006 or lower (degenerate to zero). The two objectives
disagree precisely *because* the form cannot fit the profile shape, so
"match two summary statistics" and "least-squares the whole map" land in
different places. The `0.02–0.14` figure is superseded and should not be
used.

### 6b. Why every power law stalls: the real profile is sigmoidal

Mean template value by radius bin (Normal_Flow), model columns at each form's
own best fit:

| r/r_max | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|
| **real** | 0.148 | 0.316 | 0.381 | 0.638 | 0.958 | 1.178 | 1.315 | 1.370 | 1.392 |
| power form (d/p) | 0.126 | 0.336 | 0.533 | 0.712 | 0.903 | 1.090 | 1.258 | 1.415 | 1.548 |
| inflected logistic | 0.209 | 0.273 | 0.411 | 0.639 | 0.947 | 1.189 | 1.310 | 1.364 | 1.385 |

The real profile has a flattish inner region, a steep mid-radius rise, and
**saturates at the edge** (1.315 → 1.370 → 1.392). The power form keeps
climbing (1.258 → 1.415 → 1.548) because a power law has monotone curvature
and **cannot produce an inflection**. This is the structural reason every
power-law variant stalls: p=2 at R²=0.764, the d/p form at 0.880, and even a
free quadratic at 0.906 — the last doing best of the three only because its
negative `c` term lets it bend over slightly at the edge.

### 6c. Inflected (logistic) form — capability and identifiability

```
rho    = r / r_max                                           in [0, 1]
g      = [σ((rho−r0)/w) − σ(−r0/w)] / [σ((1−r0)/w) − σ(−r0/w)]
S_raw  = c + (1−c)·g            so S_raw(0)=c, S_raw(r_max)=1 exactly
S      = S_raw / mean(S_raw)
```
`c` centre floor (centre-to-edge ratio before normalization), `r0` transition
radius, `w` transition sharpness. Edge saturation is **implied** by `r0` and
`w` rather than given its own parameter — an extra knob duplicating existing
ones is the degeneracy trap 6a fell into.

Fitted per configuration:

| configuration | c | r0 | w | R² |
|---|---|---|---|---|
| Normal_Flow | 0.1390 | 0.4478 | 0.1123 | **0.9368** |
| Inverse_Flow | 0.2000 | 0.4273 | 0.1146 | **0.9410** |
| Inverse_Air_Flow | 0.1403 | 0.4482 | 0.1131 | **0.9322** |
| Inverse_Hydrogen_Flow | 0.2055 | 0.4273 | 0.1142 | **0.9362** |

Against the benchmarks on Normal_Flow, at equal or lower parameter count:

| form | params | R² |
|---|---|---|
| current: p=2, amplitude free | 1 | 0.7639 |
| withdrawn d/p power form | 2 | 0.8804 |
| free quadratic `a+b·r+c·r²` | 3 | 0.9057 |
| **inflected logistic** | 3 | **0.9368** |

It reproduces the edge saturation the power form could not (6b table,
residuals −0.005 to −0.007 in the outer three bins). Its remaining error is
concentrated in the inner two bins (+0.061 at r/r_max=0.1, −0.042 at 0.2):
the real centre is deeper, and the rise just outside it sharper, than a
single logistic produces.

**Identifiability — the test that killed `d`, applied to all three
parameters.** Span of each parameter keeping R² within 0.002 of optimum,
others held at their fitted values:

| parameter | within-0.002 span (Normal_Flow) | ratio | compare |
|---|---|---|---|
| `c` | [0.1113, 0.1675] | 1.5× | — |
| `r0` | [0.4333, 0.4623] | 1.1× | — |
| `w` | [0.0991, 0.1281] | 1.3× | — |
| *(withdrawn `d`)* | *[1×10⁻⁶, 0.018]* | *~18,000×* | *degenerate* |

All three are tightly constrained, and all three genuinely matter: over their
full sweeps R² falls to 0.078 (`c`), −1.18 (`r0`), and 0.566 (`w`). Same
pattern in all four configurations. **This form does not have the degeneracy
that withdrew the previous one.**

### 6d. What is NOT settled here

No bounds, no parameter family, and no adoption decision. The physical
picture (a bolted frame saturating at both limits — fully compressed at the
perimeter, relaxed at the centre) is a *motivation* for a sigmoid, not a
derivation of one: the same pressure → contact-resistance → overpotential →
current-density chain that broke the p=2 argument (Sec. 5d) applies here
unchanged, and nothing in it fixes `r0` or `w`. More pointedly, this form was
selected **because the real templates show an inflection** — which is a
stronger dependence on the validation data than p=2 ever had, since p=2 at
least carried an independent (if invalid) argument. That dependence is real
and is not cancelled by the form fitting well.

### 6e. The inner-region residual: diagnosed, and it is not fixable by a fourth parameter

The §6c fit leaves residuals of +0.061 and −0.042 in the inner two radius
bins while the outer bins sit within 0.007. Diagnosed before treating the
form as settled, and deliberately **without** adding a parameter to absorb it.

**The decisive measurement: the purely-radial ceiling is R² = 0.9416.** That
is what a *free arbitrary function of radius* achieves — each radius shell
assigned its own mean, roughly 40 effective parameters, the most flexible
radial model that can exist on this grid. **5.84% of the template's variance
is irreducibly non-radial** and no radial form of any complexity can reach
it. The three-parameter logistic achieves 0.9368, i.e. **99.5% of all
radially-explainable variance**. There is ~0.005 of headroom left. A fourth
parameter cannot buy more than that, and would be spent chasing structure a
radial model is the wrong tool for.

**What the inner residual actually is.** Not a distinct central feature
(answer (a)), and not discretization noise (answer (b)) — a third thing:

1. *It is reproducible, so not sampling noise.* Shell-level residuals at
   rho = 0.0588 / 0.1315 / 0.1765 / 0.2121 are +0.061 / −0.024 / −0.015 /
   −0.075 for Normal_Flow and +0.062 / −0.024 / −0.017 / −0.076 for
   Inverse_Air_Flow — near-identical. Inverse_Flow and Inverse_Hydrogen_Flow
   agree with each other equally closely at their own (different) values.
   Random small-sample error would not reproduce like this.
2. *The non-radial component is proportionally largest where the radial
   signal is weakest.* Within-shell scatter runs ~18–28% of the shell mean
   across most radii but **45% at the innermost shell**, because the radial
   signal there is small (shell mean 0.148) while the non-radial component is
   not. The inner "misfit" is mostly the irreducible 5.84% showing up where
   it is proportionally worst.
3. *There is a genuine directional feature at the centre, and it is
   non-radial by construction.* The four innermost cells are 0.2600, 0.1386,
   0.1064, 0.0886 — a 2.9× spread across four adjacent cells, in a clear
   diagonal gradient, reproducing almost exactly in Inverse_Air_Flow (0.2593,
   0.1397, 0.1077, 0.0912). This is real structure, but it is *directional*,
   so no radial profile — not the logistic, not a free radial function — can
   represent it. It belongs to the 5.84%.
4. *A mild non-monotonicity exists in one cluster only.* Shell means at
   rho = 0.2121 vs 0.2425 differ by +2.0 and +2.1 sem in Normal_Flow and
   Inverse_Air_Flow, but only +0.3 and +0.2 sem in the other two. A monotone
   radial form cannot carry a bump; this contributes to the inner residual in
   the first cluster and barely at all in the second.

**Conclusion: settled, not open.** The form is at 99.5% of its achievable
ceiling. The remaining error is non-radial structure that is out of scope for
a radial template by definition. No fourth parameter.

### 6f. What Phase 5's no-false-alarm test can and cannot claim

Stated plainly because the answer changed, and not in the project's favour.

**The functional form was selected from the validation data.** The family is
radial because the real templates are radial (`docs/real_data_notes.md`,
cross-session section), and inflected/saturating because the real radial
profile inflects and saturates (§6b). That is a stronger dependence on the
Phase 5 validation set than the superseded `p = 2` ever had — p=2 at least
carried an independent argument, even though §5d shows it was invalid.

**Phase 5 can still claim**: that a model trained across a span of healthy
*parameter values* stays quiet on a healthy cell whose parameter values it
did not see in training. That is a real generalization test and a meaningful
one — it is the difference between a model that memorized one healthy
baseline and one that tolerates a range of them.

**Phase 5 can no longer claim**: independence from the real data's
*structure*. A model trained on this family has the real cell's structural
form — radial, inflected, saturating at both ends — built in by construction.
A quiet false-alarm result partly reflects that the family was built to look
structurally like the thing it is validated against. The test is of
parameter-value generalization inside an assumed form, not of the form.

**The earlier held-out-region framing was relying on an independence we do
not have.** That proposal (now withdrawn for a separate reason — all four
real sessions occupy the same corner, §6a/finding 3) would have held out the
real cell's parameter region and claimed the resulting test was independent.
Even had the parameter spread supported it, holding out parameter *values*
would not have restored *structural* independence, because the form is shared
across the whole family by construction. Worth recording as a second,
independent reason that framing does not deliver what it promised.

**What would restore structural independence**: a second real cell with a
materially different template, or a form derived without reference to this
dataset. Neither is available. The honest scope is the narrow claim above.

### 6g. ADOPTED bounds — and an asymmetry that must be carried forward

**Adopted.** Stated with what sets each bound and where the real sessions
fall inside it.

| parameter | adopted range | what sets it | where real sessions fall |
|---|---|---|---|
| `c` centre floor | **[0.05, 0.60]** | Observed cross-session variation (1.48×, two clusters) gives a measured scale for how much this parameter moves; bracketed to ~3× beyond the observed band in each direction. Upper bound 0.60 is a markedly more uniform cell (max/min ≈ 1.7 against the observed 7–17×); lower bound 0.05 a markedly deeper one. | 0.139, 0.140, 0.200, 0.206 — lower-middle of the range |
| `r0` transition radius | **[0.25, 0.70]** | Geometry only. Lower: compression falling off close to the centre. Upper: only the outer third affected. No empirical content — see below. | 0.427–0.448 — slightly below centre of the range, spanning 5% of its width |
| `w` sharpness | **[0.04, 0.35]** | Shape limits only. Lower: ~3× sharper than observed, approaching a step. Upper: ~3× broader, where the inflection becomes imperceptible and the profile degenerates toward the power-law shapes §6b rejected. No empirical content. | 0.112–0.115 — lower-middle |

**The asymmetry, stated rather than smoothed.** These three bounds are not
equally grounded and should not be read as though they were:

- **`c` has thin but real empirical content.** It varies 1.48× across the
  four sessions, in two clean clusters. That is a *measured* scale for how
  much centre depth moves between states of this cell. It is thin because the
  variation is confounded (H₂ routing vs. session block — see
  `docs/real_data_notes.md`), so we know something moves by 1.48× but not
  what drives it, and therefore not whether 1.48× is the right scale for
  "a different cell" as opposed to "the same cell on another day".
- **`r0` and `w` have essentially none.** They are fixed to within 5% and 2%
  respectively in every session available (CV 2.4% and 0.8%). The data says
  what they *are*; it says nothing whatever about how much they could vary
  across cells. Ranging them is asserting variability that no observation
  supports. A Phase 5 result that depends on `r0`/`w` breadth is resting on
  assumption, and should carry less weight than one that depends on `c`.

**The midpoint criterion is RETIRED.** It previously stood as a guard: a
range whose midpoint lands on the real values was to be treated as suspect.
It is withdrawn because it is not merely strict, it is **incompatible with
the bracketing goal it was meant to police**. Bracketing symmetrically around
an observed band necessarily places the midpoint on that band — for `c`, the
log-midpoint of [0.05, 0.60] is 0.173, inside the observed 0.139–0.206, as an
arithmetic consequence rather than as evidence of tuning. Satisfying the
criterion would require deliberately skewing a range, making the family
asymmetrically wrong in order to look untuned. That is worse than the problem
the criterion guards against.

**What replaces it**: state explicitly, for every ranged parameter, where the
real sessions fall inside the range — the "where real sessions fall" column
above. A reader can then judge the placement directly instead of trusting a
summary statistic that cannot distinguish bracketing from fitting.

### 6h. Standing instruction for the Phase 5 writeup

Not a footnote. This must survive to the writeup, so it is recorded as an
instruction rather than an observation.

**Any Phase 5 finding whose conclusion depends on the breadth of `r0` or `w`
must be marked, at the point it is stated, as resting on assumed rather than
observed variability.** The two are not interchangeable:

| parameter | observed cross-session variation | status of its range |
|---|---|---|
| `c` | **1.48×** (CV 18.5%, two clusters) | empirically anchored, though the variation is itself confounded between H₂ routing and session block |
| `r0` | 1.05× (CV 2.4%) | **assumed** — the data fixes a value, not a range |
| `w` | 1.02× (CV 0.8%) | **assumed** — same |

Concretely, a claim of the form "the model stays quiet across the full
healthy family" is carrying two different kinds of weight at once: the `c`
span is a (thin, confounded) empirical scale, while the `r0` and `w` spans
assert variability **no observation in this dataset supports**. If a Phase 5
result survives only because the family ranged `r0`/`w` widely, that result is
a statement about an assumption, not about healthy cells. It should be
reported as such, and should carry less weight than a result that turns on
`c`.

The same applies in reverse: if a model *fails* only at extreme `r0`/`w`, that
failure is also against assumed conditions and should not be over-read as a
real robustness gap.

## 7. Resolution confound, re-measured against the adopted family

Source: `scripts/measure_resolution_effects.py` (re-runnable). Measurement
only — no parameter was adjusted in response to anything below. This closes
the last open measurement before notebook 03.

Supersedes the Sec. 5f measurement, which covered only the Sec. 5 form (now
superseded) on Normal_Flow alone. Here: all four configurations, both forms.

**Method.** Real 18×18 → 4×4 by block-averaging with `np.array_split`,
partitioning each axis into four groups of sizes [5,5,4,4]. 18 does not
divide evenly by 4, so this uses all 324 cells and crops nothing, unlike a
crop-to-16×16-then-reshape. Block-averaging is the physically right operator:
it is what a coarser sensor does — integrate current over a larger segment
area. CV(J) = std/mean is nonlinear, so each timestep is downsampled *first*
and its CV computed after. Autocorrelation is computed on the static
template, where block-averaging and time-averaging commute. For the family,
the **same** (c, r0, w) is evaluated at both resolutions — a paired
comparison that isolates the resolution effect from parameter variation.

### 7a. The 2×2

**CV(J)**

| | 4×4 | 18×18 | 18×18 / 4×4 |
|---|---|---|---|
| real (4-config mean) | 0.3108 | 0.3534 | **1.14** |
| synthetic, superseded Sec. 5 form | 0.2024 | 0.1713 | **0.85** |
| synthetic, adopted logistic family (mean) | 0.2281 | 0.2546 | **1.12** |
| *logistic at the real sessions' own fitted parameters* | *0.2774–0.3248* | *0.3141–0.3680* | *1.13* |

Real is 1.14 in **all four configurations individually** (Normal_Flow,
Inverse_Flow, Inverse_Air_Flow, Inverse_Hydrogen_Flow all 1.14).

**lag-1 autocorrelation (static template)**

| | 4×4 | 18×18 |
|---|---|---|
| real (4-config mean) | 0.3249 | 0.9389 |
| synthetic, superseded form | 0.2941 | 0.9545 |
| synthetic logistic @ Normal_Flow params | 0.2793 | 0.9628 |

### 7b. The adopted form reproduces real's resolution scaling; the superseded one had it backwards

Real CV(J) rises by **1.14×** from 4×4 to 18×18. The adopted logistic rises
by **1.12×** (family mean) / **1.13×** (at the real sessions' fitted
parameters). The superseded form **fell** by 0.85× — not merely a different
magnitude but the **opposite direction**.

This was not fitted. The form was selected on shape grounds alone — that the
real radial profile inflects and saturates (§6b). Its resolution response is
a consequence of that shape, not a target, and it lands on the real value.
**Weighted honestly**: this is one scalar, and it follows from getting the
radial profile approximately right, so it is not independent of the shape
selection in the way a genuinely held-out test would be. But the superseded
form, which also had a radial profile, got the sign wrong — so the agreement
is not automatic, and it is the first quantity the adopted form predicts
that was not used in choosing it.

A second consequence concerns the superseded form's gap behaviour: it
*widened* with resolution (1.54× → 2.06×), which is what made the Sec. 5f
figure resolution-dependent and the "~1.9×" headline misleading. That
pathology is gone — the adopted family's resolution response matches real's.

> **SUPERSEDED — mean-to-mean gap framing for the family.** This paragraph
> originally continued: ~~"the real-vs-family gap is now **stable across
> resolution** — 1.36× at 4×4, 1.39× at 18×18."~~ Those numbers are correct
> arithmetic but the wrong comparison. A *family* is deliberately broad;
> comparing its mean to the real value measures where the family's centre
> happens to sit, not whether the family covers the real cell. The right
> statements are containment and placement — see §7d: real sits at the
> **78th–89th percentile** inside the family at both resolutions, contained
> but not central. **"1.39× residual gap" should not be carried forward as a
> headline number.** It is retained struck through because the earlier
> sections reasoned from it.

### 7c. The form is NOT resolution-independent, and "by construction" was the wrong expectation

Checked rather than assumed, as instructed. The answer is no, with a precise
statement of what *is* invariant:

- **The continuous field is resolution-independent by construction.** `S_raw`
  depends only on ρ = r/r_max, so the underlying profile is identical at any
  grid size.
- **CV(J) is not**, because it is a *sample statistic* computed over a
  resolution-dependent sampling of radius. Which ρ values exist, and with
  what multiplicity, changes with N. Paired family mean ratio **1.123**
  (p5 0.977, p95 1.357).
- **The sensitivity is wildly uneven across the family**: max deviation from
  unity is **2.90×**. The extreme cases are a small, identifiable corner —
  109 of 4000 draws exceed 1.5×, and they cluster at sharp transitions close
  to the centre (mean w ≈ 0.068 against the family's [0.04, 0.35]; mean
  r0 ≈ 0.288 against [0.25, 0.70]). Correlation of the ratio with `w` is
  −0.40 and with `r0` is −0.41. A 4×4 grid has only three distinct radii
  inside ρ = 0.29, so it cannot resolve a sharp transition sitting there.

**Practical consequence, and it supports the pending generator change**:
running the generator natively at 18×18 — the validation data's own
resolution — removes this entire class of error. At 4×4 a meaningful corner
of the adopted family is misrepresented by up to 2.9×.

### 7d. Where the real sessions fall inside the family

Per the convention adopted in §6g (replacing the retired midpoint criterion),
stated explicitly rather than summarized:

| configuration | real CV(J) 18×18 | percentile within family | real CV(J) 4×4 | percentile |
|---|---|---|---|---|
| Normal_Flow | 0.3810 | 89% | 0.3344 | 87% |
| Inverse_Flow | 0.3266 | 79% | 0.2877 | 77% |
| Inverse_Air_Flow | 0.3812 | 89% | 0.3353 | 88% |
| Inverse_Hydrogen_Flow | 0.3247 | 78% | 0.2858 | 76% |

Family at 18×18: p5 = 0.1262, median = 0.2346, p95 = 0.4503. Real band
0.3247–0.3812.

**The real sessions sit in the upper quartile of the family, comfortably
contained but not central.** 11% of the family lies above the real maximum;
**78% lies below the real minimum**. Under uniform sampling of the adopted
ranges, the family is therefore skewed toward cells *more uniform* than the
real one.

Two things follow, neither of which is acted on here:

1. **The "gap" framing is partly obsolete.** Comparing family *mean* to real
   (1.39×) is the wrong comparison for a family whose whole purpose is to
   span a range. The meaningful statements are containment and placement —
   real is inside, at the 78th–89th percentile — not a mean-to-mean ratio.
   Residual-gap language from Sec. 5/6 should be read with that correction.
2. **Uniform sampling is RETAINED — a decision, not an open question.** The
   skew is a known and accepted property of the family, with a directional
   consequence that works in the project's favour:

   Under uniform sampling of the §6g ranges, **78% of the family lies below
   the real minimum and 11% above the real maximum** — the family is weighted
   toward cells *more spatially uniform* than any observed. A more uniform
   healthy cell has a *lower* CV(J) baseline, so an anomaly threshold that
   must stay quiet across it has less headroom: **the skew makes
   false-alarm avoidance harder, not easier.** A model that stays quiet
   across this family is therefore making a **stronger** claim than one
   validated on a family centred on the real data.

   Log-uniform sampling of `c` would shift mass toward the real end and make
   the test easier. It is rejected for that reason. Re-skewing a family to
   centre it on the validation data is the same reflex §6g's retirement of
   the midpoint criterion warns against, pointed the other way — and here it
   would additionally weaken the claim being tested.

   Recorded as a property to state in the writeup, not a defect pending
   correction.

