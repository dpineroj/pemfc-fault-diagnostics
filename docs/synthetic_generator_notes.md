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


## 8. Healthy family implemented; fault states deliberately left on the old baseline

The adopted family (Sec. 6c, bounds Sec. 6g) is now the default severity=0
path in `fault_generator.py` — `generate_healthy_grid_v3`, selected via
`generate_labeled_run(healthy_model="logistic")`. The legacy notebook 01
gradient stays reachable as `healthy_model="notebook01"`;
`generate_healthy_grid_v2` and the frozen `generate_normal_grid` are
unchanged.

**Fault-state spatial patterns were NOT rebased onto the new baseline, and
this is a deliberate choice rather than deferred work.**

The reason is not merely scope. Notebook 01's fault patterns were *designed as
modifications of a diagonal flow-path gradient*: flooding suppresses the
cathode-outlet corner of that gradient, drying shifts weight toward the H₂
inlet along it, starvation collapses it along the cathode flow path. Each
endpoint is defined relative to the gradient it modifies.

Rebasing only the interpolation's *start* point — swapping healthy from the
diagonal gradient to the radial template while leaving those endpoints
untouched — would not produce "faults on a better baseline". It would produce
a **trajectory between two points never designed to connect**: a severity
sweep from a radial bowl toward a pattern constructed as a perturbation of a
diagonal gradient. The intermediate severities would be geometrically
arbitrary. Flooding is the clearest case, since its defining feature is a
corner suppression that only means anything relative to a flow-path gradient.

Such output would look plausible — smooth, continuous, severity-ordered — while
being unjustified at every intermediate point. **An openly inconsistent
generator is better than a plausible-looking unjustified one**, because the
first can be reported and reasoned about and the second quietly contaminates
anything built on it.

**Consequence to state wherever this generator's output is shown**:
~~the healthy baseline is the adopted radial family; the fault states are
still the superseded diagonal-gradient interpolation. These rest on different
foundations.~~ Fault-state spatial patterns from this generator are **not
validated output** and must not be read as such.

**PARTIALLY SUPERSEDED by §11i — read the two halves separately, because only
one of them stopped applying.**

The **mixed-basis clause is struck**: Phase 3b is complete, `PHASE3B_FAULTS`
covers all three fault types, and every fault now modulates the cell's own
radial healthy template. No run mixes a radial healthy baseline with
diagonal-gradient faults any more. The superseded interpolation forms remain
reachable only via `healthy_params=None`, i.e. by taking the legacy
notebook-01 path as a whole, which is self-consistent.

The **"not validated output" clause stands unchanged, and permanently**. Per
§11a that has nothing to do with the basis mismatch: it follows from there
being no fault-labelled spatially-resolved data for this cell, so no
capability check, no identifiability check and no resolution corroboration
exists for any fault pattern. Rebasing them onto a sounder basis made them
*mutually coherent*, not *validated*, and no downstream result may describe
them as validated.

~~**Phase 3b** (see README) covers the fault-state rework, with the same
treatment the healthy form received: literature grounding, capability check,
identifiability check, and an honest statement of data-dependence. Not
started, not scoped in detail, and explicitly after notebook 03.~~

**SUPERSEDED by §11a — kept visible deliberately.** The struck sentence
promised Phase 3b a treatment that **is not achievable**, and §11a exists
specifically to explain why, so deleting the promise would remove the thing
§11a is answering.

Every load-bearing step of the healthy rework consumed **real templates**: the
capability check (R² against a measured purely-radial ceiling), the
identifiability check (parameter spans against real templates), the
resolution-scaling corroboration, the cross-session replication. There is no
fault-labelled real data for this cell and none is obtainable without
hardware, so for fault states there is nothing to compute R² against, nothing
for parameters to be identifiable *with respect to*, and no real fault
template whose resolution response could be matched.

Phase 3b is scoped in §11 instead, and produces **mutually coherent,
literature-grounded fault patterns — not validated ones**. Per §11a that is a
**permanent limitation**, not a gap pending future work within this project's
scope.

## 9. Native resolution moved to 18×18

`generate_labeled_run` now defaults to `grid_shape=(18, 18)`, matching the
real validation data natively. 4×4 stays reachable via `grid_shape=(4, 4)`.
Every grid function still *defaults* to 4×4 when called directly, so the
frozen path, the v2 regression test, and any existing direct call are
byte-for-byte unaffected — only `generate_labeled_run` opts in.

Why: CV(J) from this family is resolution-dependent (§7c), by up to 2.9× for
sharp transitions near the centre. Generating at the validation resolution
removes that error class rather than correcting for it afterwards.

**Acceptance (both resolutions, all passing):** frozen `generate_normal_grid`
byte-identical to notebook 01 at its 4×4 default across four seeds; the v2
regression test unchanged; the temperature grid byte-identical at 4×4 to the
pre-change formula; load-invariance exact at both resolutions (spread 0.00 at
18×18, 5.6e-17 at 4×4); and episode counts **34/24/9 with bit-identical
severity_targets across all four combinations** of {notebook01, logistic} ×
{4×4, 18×18} — grid resolution touches the episode stream no more than the
healthy-model choice did, verified rather than assumed.

**Two generalizations were required and are not pure parameterization**, so
they are called out rather than buried:

1. *Temperature centre-bias.* Notebook 01 hardcodes 4-element profiles
   (`[0.85, 1.0, 1.0, 0.85]`). These are interpolated onto the actual grid
   size — reproducing the original arrays exactly at 4×4 and generalizing
   above it. Leaving them as fixed 4-vectors simply fails to broadcast.
2. *Temperature smoothing σ.* σ is a smoothing *length*, so it scales with
   resolution (`0.8 × N/4`) to keep its physical meaning. Holding 0.8 cells
   fixed would leave the 18×18 temperature field nearly unsmoothed — a
   behaviour change disguised as a resolution change. Exactly 0.8 at 4×4.

Both touch the temperature model, which is otherwise Phase 3b's business.
They are included here because the alternative is silently different
behaviour at the new default, not because fault-state rework has begun.
**Consequence for Phase 3b: it inherits a partially-generalized temperature
path, not an untouched one** — the centre-bias profile and the smoothing
length are already resolution-aware, while every other temperature constant
(the per-state `base_t`/`center_scale`/`j_scale` triples) remains
notebook-01-derived and untouched. Recorded so it does not have to be
rediscovered.

### 9a. SUPERSEDED — the 0.1719 run-level figure from Sec. 8

> ~~Run-level healthy CV(J) moves 0.0762 → **0.1719** under the adopted
> family.~~

**That figure was 4×4-only and must not be carried forward as "the healthy
baseline's CV(J)".** It is exactly the kind of number that gets quoted later
without its resolution attached. The same run, same drawn cell
(`c=0.2290, r0=0.3235, w=0.0689`), at the new 18×18 default reports
**0.2499** — a 1.45× difference arising purely from grid resolution, because
that cell sits in the sharp-transition-near-centre corner §7c identified.

**Any CV(J) from this family must be stated with its grid size.** A bare
CV(J) number from the generator is not interpretable.

For the same reason, the notebook01 legacy path also moves with resolution:
0.0762 at 4×4, **0.0649** at 18×18.

### 9b. Stale artifact, not regenerated

`figures/fault_generator_run_normal_flow.png` was produced at the 4×4 default
and is now stale: its CV(J) and severity panels reflect a 4×4 run of a
superseded healthy baseline. It is gitignored (not a committed artifact) and
has deliberately **not** been regenerated here. It will be replaced when
notebook 03 produces its figures.

No other stale numbers were found. The 4×4-derived figures that remain in
§5 (0.2024 and the sensitivity table) describe `generate_healthy_grid_v2`,
which still defaults to 4×4 and is unchanged — they are correct for what they
describe, and §5f already scopes them to the superseded point model.

---

## 10. Standing test: does this choice decide the comparison before it is measured?

**A question to ask of every future generator change, not a one-time
observation.**

The project's central deliverable is a *delta between two arms* — same
simulation, same architecture, one fed spatial grids and one fed bulk signals
only. The justification for that design is that **simulation artifacts hit
both arms equally**, so whatever is wrong with the simulation cancels in the
comparison.

**That justification is conditional, and it does not hold universally.**

| artifact class | hits both arms equally? |
|---|---|
| onset timescales, severity ranges | **Yes** — both arms see the same episode timing |
| temporal noise level | **Yes** — both arms see the same fluctuation |
| the load trace and its segmentation | **Yes** — drives triggers, not geometry |
| **spatial pattern geometry** | **NO** — this is by construction the thing one arm sees and the other cannot |

A spatial-geometry artifact is invisible to the bulk arm and fully visible to
the spatial arm. It therefore inflates or deflates the delta directly, and
cancels nothing. The two-arm design provides **no protection at all** against
errors in spatial structure.

### The test to apply

> **Before adopting any generator change, ask: if this choice is wrong, does
> the error reach both arms, or only one?**
>
> If only one — and in practice that means the spatial arm — then the two-arm
> design does not absorb it, the error propagates directly into the headline
> result, and the change needs its own justification rather than inheriting
> the "artifacts cancel" argument.

**The sharper formulation — use this one.** The question above was first
written with a free win in mind, and that framing is too narrow. Promoted here
from §11h item 2, where the limitation surfaced:

> **The test is not "does this flatter the hypothesis". It is "does this
> decide the comparison before it is measured".**

Both failure directions are caught by the second phrasing and only one by the
first. A generator choice that hands the spatial arm a **guaranteed loss** is
exactly as corrupting as one handing it a free win, for the same reason — the
headline comparison follows from the generator's construction rather than from
the measurement — and a reader asking only "is this flattering?" would wave it
through.

**Worked example, and the reason this section exists.** With the healthy
baseline rebased to a radial template while fault states remained on notebook
01's diagonal gradients, the two classes came from structurally different
families. A single scalar — each grid's correlation with distance-from-centre
— separated healthy from fault at **100.0% accuracy**, and did so **at
severity 0.01**, before any meaningful fault had developed:

| severity | starvation | flooding | drying |
|---|---|---|---|
| 0.01 | 1.000 | 1.000 | 1.000 |
| 0.05 | 1.000 | 1.000 | 1.000 |
| 0.50 | 1.000 | 1.000 | 1.000 |
| 0.90 | 1.000 | 1.000 | 1.000 |

Chance is 0.500. Perfect separation at severity 0.01 cannot be fault physics —
there is essentially no fault present. It is the basis mismatch. Had Phase 4
run against this, the spatial arm would have scored near-perfectly on a
feature the bulk arm is blind to, and the project would have confirmed its own
central hypothesis for entirely the wrong reason.

**Worked example in the inverting direction** (§11h item 2, recorded here so
§10 carries both). Starvation's no-axis arm could have been built as
*spatially uniform* suppression, `M = 1 − s²A` — the simplest and most
honest-looking response to not knowing a flow direction. It was rejected. A
uniform modulation carries **zero spatial information by construction**, so a
spatially-resolved classifier could not beat a bulk one on it as *arithmetic*
rather than as a finding. That would have settled §11e branch 3 before Phase 4
measured anything.

Note that this choice is not flattering in the slightest — it hands the
project's own hypothesis a built-in defeat — and the original phrasing of the
test would have let it through. The second phrasing catches it immediately.

## 11. Phase 3b — fault-state rework onto the radial baseline

**COMPLETE.** 40 of 41 acceptance checks pass; the single failure was
pre-registered (§11h) and is recorded as a finding, not a defect (§11i).

**What each subsection settles** — this section is long, so: a map, not a
summary.

| § | settles |
|---|---|
| **11a** | what Phase 3b can claim: coherent and literature-grounded, **never validated** — a permanent limitation, not a gap |
| **11b** | the acceptance tests, with Test 1 and Test 2's thresholds fixed before implementation and a measured null behind 0.560 |
| **11c** | the form: multiplicative modulation of the shared healthy template, and each fault's grounding |
| **11d** | whether a flow-path axis is defensible — measured, and resolved as fixture-locked, not flow-locked |
| **11e** | how Phase 4 must interpret starvation's result, in three branches fixed in advance |
| **11f** | the internal-consistency checks that substitute for the validation that does not exist |
| **11g** | the severity convexity exponent, and the chronology of adopting it after a test failure |
| **11h** | starvation's two arms, pre-registered, including the predicted Test 3 failure |
| **11i** | the result, what it is not, what Phase 4 inherits, and the one threshold with weaker provenance |

### 11a. What Phase 3b can and cannot claim — a permanent limitation

Phase 3b was originally described as receiving "the same treatment the healthy
form received: literature grounding, capability check, identifiability check,
honest statement of data-dependence." **That is not achievable, and the reason
is structural rather than a matter of effort.**

What made the healthy rework rigorous was **real templates to check against**.
Every one of its load-bearing steps consumed real data: the capability check
(R² = 0.9368 against a measured ceiling of 0.9416), the identifiability check
(parameter spans of 1.1×–1.5× against real templates), the resolution-scaling
corroboration (1.13 against real's 1.14), the cross-session replication
(r ≥ 0.976 across four sessions).

**There is no fault-labelled real data for this cell, and none is obtainable
without hardware.** Therefore:

- There is **no capability check** — nothing to compute R² against.
- There is **no identifiability check** — no data for parameters to be
  identifiable *with respect to*.
- There is **no resolution-scaling corroboration** — no real fault template
  whose resolution response could be matched.

**Phase 3b produces mutually coherent, literature-grounded fault patterns. It
does NOT produce validated ones.** No downstream result may describe fault
spatial patterns as validated, and no writeup may imply the fault side
received the treatment the healthy side did.

**This is a permanent limitation, not a gap pending future work within this
project's scope.** It is not resolved by more careful modelling, a better
literature search, or additional analysis of the existing dataset. It is
resolved only by fault-labelled spatially-resolved data, which this project
has established does not exist and cannot be produced without hardware. Stated
in the same register as §6d: the limitation is to be reported, not managed
away.

### 11b. Acceptance test — specified BEFORE implementation, thresholds fixed now

Recorded in advance so they cannot drift to accommodate whatever the rework
produces.

**Is one scalar sufficient? No.** Radial correlation alone is Goodhart-able: a
rework could preserve radial character while introducing, say, a strong
diagonal component — passing the letter of a radial-correlation test while
remaining just as trivially separable on a diagonal feature. A single
discriminability statistic tests one escape route and leaves the others open.

**More importantly, "must not separate" is the wrong criterion at full
severity.** Healthy and a fully-developed fault *should* be separable — that
is the physics the project is trying to detect. The artifact is not that they
separate; it is that they separate **immediately**, at severity 0.01, by a
feature that reflects differing baseline geometry rather than fault
development. The test must therefore be **severity-resolved**.

**Test 1 — Continuity (PRIMARY, structural, hardest to game).**
The fault template must converge to the healthy template as severity → 0,
for the same cell parameters, comparing deterministic templates with the
temporal noise draw excluded:

> relative RMS difference d(ε) = ‖J_fault(ε) − J_healthy‖ / ‖J_healthy‖
> **≤ 0.02** at ε = 0.01, and increasing in ε.

**The monotonicity clause, specified to the same standard as Test 2's
threshold.** "Increasing monotonically" is unadjudicable on its own: a
trajectory sampled at 3 points and one sampled at 30 would be judged
differently, and small non-monotonic jitter has no stated allowance. Fixed
now, before implementation, for the same reason 0.560 was fixed:

> **Evaluation grid** (fixed): ε ∈ {0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75,
> 0.90, 1.00} — nine points, including every severity the §10 separability
> measurement used.
>
> **Sample**: 200 drawn cells. The criteria must hold for **every** cell, not
> on average. Continuity is claimed to hold *by construction*, so it is a
> structural property of the form; a single failing cell falsifies that claim
> and is a failure.
>
> **(i) Convergence**: d(0.01) ≤ 0.02.
>
> **(ii) Local monotonicity, with tolerance**: d(εₖ₊₁) ≥ d(εₖ) − **0.002**
> for every consecutive pair on the grid. The 0.002 absorbs discrete-grid and
> floating-point jitter *only* — Test 1 compares deterministic templates, so
> no sampling noise enters and no larger allowance is warranted.
>
> **(iii) Trend**: strict increase across widely-spaced checkpoints —
> d(0.10) > d(0.01), d(0.50) > d(0.10), d(1.00) > d(0.50). Jitter within the
> 0.002 tolerance cannot satisfy this spuriously, which is why both (ii) and
> (iii) are required rather than either alone.
>
> **Flooding's oscillation is held at its mean** for this test. The
> accumulate/breakthrough factor is a *temporal* modulation, and Test 1
> compares templates with temporal variation excluded; leaving it free would
> make d(ε) depend on oscillation phase rather than on severity.

This is a limit condition, not a discriminability statistic, so it cannot be
satisfied by shuffling which feature carries the signal. The current generator
fails it grossly: at ε = 0.01 the fault grid is already essentially notebook
01's diagonal gradient.

**Test 2 — Severity-resolved separability across a battery (BACKSTOP).**
For each feature below, the best single-threshold accuracy separating healthy
from fault grids:

> **at severity ≤ 0.05, with N = 1000 grids per class, no feature in the
> battery may exceed 0.560** (chance = 0.500). No constraint is placed at
> high severity — separability there is the intended physics.
>
> **This threshold is CONDITIONAL ON THE FORM, and the condition travels with
> the number.** Requiring near-null separability at severity ≤ 0.05 is
> slightly stricter than physics alone demands — a genuine 5%-severity fault
> may be marginally detectable. What makes it the right expectation here is
> specific to the proposed multiplicative form of §11c: continuity is
> guaranteed *by construction* (Test 1, relative RMS ≤ 0.02 at ε = 0.01), so
> the perturbation at ε = 0.05 really is tiny. **Under a form that does not
> guarantee continuity by construction, this rationale does not hold and the
> 0.560 / severity ≤ 0.05 pair must be re-derived rather than carried
> forward.** If a form fails Test 2 while passing Test 1, that is a decision
> point to be argued explicitly — not a licence to move the threshold.

Battery (global-geometry / structure-type features — deliberately **not**
mean(J) or CV(J), which legitimately shift for real faults and whose
suppression would wrongly demand that faults be undetectable):

1. correlation with distance-from-centre
2. correlation with row index
3. correlation with column index
4. correlation with each diagonal (row+col, row−col)
5. lag-1 spatial autocorrelation
6. correlation against the cell's own healthy template

**Count convention — the battery is SEVEN features across SIX numbered
items**, because item 4 contributes two (row+col and row−col). Seven is the
number that matters, since the acceptance test maximizes over *features*, not
items, and the null's battery-max inflation is driven by the feature count.
Use "seven-feature battery" in any future reference;
`scripts/measure_battery_null.py`'s `FEATURE_NAMES` is the authority and has
seven entries.

**The 0.560 is measured, not asserted — and it is inseparable from N.**
An earlier draft of this section proposed 0.60 from intuition. That number
was wrong in *both* directions depending on sample size, which is the real
lesson: **a separability threshold is meaningless without fixing N**, because
best-threshold accuracy on finite samples is upward-biased even when the two
classes are drawn from an identical distribution.

`scripts/measure_battery_null.py` measures the floor directly — grids drawn
from the healthy family **alone**, no fault and no modulation, split into two
arbitrary groups and run through the same battery. Whatever it scores there is
attributable to finite samples and nothing else. Battery-max null at 18×18:

| N / class | median | p95 | p99 | max observed | per-feature p95 | KS p95 predicted |
|---|---|---|---|---|---|---|
| 25 | 0.660 | 0.720 | 0.760 | 0.800 | 0.680 | 0.692 |
| 50 | 0.610 | 0.660 | 0.690 | 0.750 | 0.630 | 0.636 |
| 100 | 0.580 | 0.615 | 0.630 | 0.680 | 0.595 | 0.596 |
| 200 | 0.560 | 0.583 | 0.593 | 0.618 | 0.568 | 0.568 |
| 500 | 0.538 | 0.552 | 0.559 | 0.573 | 0.542 | 0.543 |
| **1000** | 0.527 | **0.537** | **0.543** | **0.553** | 0.530 | 0.530 |

**Why the null is principled rather than merely observed.** For two
equal-size samples and a threshold rule, accuracy at threshold *t* is
[F_B(t) + 1 − F_A(t)]/2, so maximizing over *t* and both orientations gives

> best-threshold accuracy ≡ 0.5 + D/2,  D = two-sample Kolmogorov–Smirnov statistic

The per-feature null is therefore the KS null, which is **distribution-free**:
it depends only on N, not on the feature and not on the parameter spread,
because under the null both groups share one distribution. The measured
per-feature p95 matches the large-N prediction 0.5 + 1.358·√(2/N)/2 to within
0.001 at every N — the last two columns above. Two consequences worth
recording:

- The **within-cell** and **across-cell** designs give the same null to three
  decimals (0.537 vs 0.537 at N=1000), exactly as the identity predicts. The
  choice between them does not need to be defended on null grounds.
- The battery-max sits only ~0.007 above the per-feature p95, because the
  seven features are strongly correlated; each drives the max 13–17% of the
  time. Widening the battery therefore costs very little threshold headroom,
  which is what makes a battery affordable as a Goodhart backstop.

**Adopted: N = 1000 per class, threshold 0.560** = measured p99 (0.543) plus a
stated ~0.02 margin, which also clears the largest value seen in 1000 null
trials (0.553). N = 1000 is chosen over smaller N because the generator can
produce arbitrarily many grids — nothing rations the sample — and a larger N
buys a tighter threshold and therefore more power to detect residual
artifact.

**False-failure rate of the adopted threshold, measured.** P(battery-max >
threshold) under the null, 3000 trials per cell:

| N / class | > 0.54 | > 0.56 | > 0.58 | > 0.60 |
|---|---|---|---|---|
| 100 | 1.0000 | 0.9263 | 0.5570 | 0.1633 |
| 200 | 0.9693 | 0.3937 | 0.0563 | 0.0037 |
| 500 | 0.3590 | 0.0057 | 0.0000 | 0.0000 |
| **1000** | 0.0150 | **0.0000** | 0.0000 | 0.0000 |

At the adopted (N = 1000, 0.560) the null never once exceeded the threshold in
3000 trials, so the false-failure rate is below 1/3000. The adjacent choice
0.54 would have carried a 1.5% false-failure rate, which is why the margin
sits where it does rather than one step tighter.

**For the record, what the discarded 0.60 would have done.** At N = 100 it
would have failed a perfectly correct generator **16.3% of the time** —
roughly one run in six, not the "more than 5%" that reading the p95 alone
suggested. At N ≥ 500 it would never fail a correct generator at all, but the
null's maximum there is 0.573, so it would have permitted genuine separation
well above chance at severity 0.05 — precisely what the test exists to
prevent. Both failure modes come from the same root cause: the threshold was
not anchored to a sample size.

*(The strictness caveat that was here has been moved up to sit immediately
beneath the threshold itself, so the form-dependence cannot be separated from
the number it qualifies.)*

**Test 3 — Cross-fault distinguishability must survive.** The three fault
types must remain distinguishable *from each other* at full severity,
otherwise the rework has made the classification task degenerate. A rework
that passes Tests 1–2 by making all faults look like healthy has not
succeeded.

**Test 4 — Preserved invariants.** Load-invariance of CV(J) must hold as it
does for the healthy family; episode counts (34/24/9) and severity_targets
must remain bit-identical, since triggers key off the load trace and nothing
in the spatial rework may touch that stream.

### 11c. Proposed form — perturbation of a shared basis, not interpolation between endpoints

The structural cause of the current problem (§8) is that fault states are
defined as *independent endpoints* and reached by interpolation. The fix is to
define them as *modulations of the healthy template*:

```
J_fault(i, j; s) = J_healthy(i, j) · M_fault(i, j; s)
with  M_fault(i, j; 0) = 1  everywhere
```

Continuity (Test 1) then holds **by construction** rather than by tuning, in
the same way the healthy family's load-invariance holds by construction
through the `S / mean(S)` renormalization. A multiplicative modulation is also
the physically natural choice: these mechanisms suppress local current as a
*fractional* effect on whatever the cell was already producing.

Per fault type, what `M` must express. Two of the three are grounded on the
thermal field the generator already produces, and **with opposite radial
sign**; only starvation needs a directional axis, which §11d resolves.

- **Drying** — membrane ionic resistance rises where water content is lowest,
  i.e. where the cell is hottest. The generator's thermal field is
  centre-biased, so drying suppresses the **centre**, deepening the healthy
  bathtub and raising CV(J). Grounded internally, no imposed axis.

  **Caveat at the point of use, applying to flooding equally.** This grounding
  is only as good as the thermal field it leans on, and the thermal model's
  per-state `(base_t, center_scale, j_scale)` triples are still
  notebook-01-derived and were **not** touched by the healthy rework (§9 —
  only the grid-size generalizations reached the temperature model). So
  "grounded internally" means *grounded on a quantity the generator already
  produces*, *not* grounded on validated physics: drying and flooding inherit
  a centre-bias assumption from the superseded baseline.

  This does not invalidate the choice — what both faults use is the *sign* of
  the thermal gradient, which is the robust part, and nothing here depends on
  the triples' magnitudes. But the dependency must be reported rather than
  implied away, and the thermal model is Phase 3b-adjacent scope that this
  grounding rests on.
- **Flooding** — liquid water condenses preferentially in the **coldest**
  regions of the active area, so suppression is weighted toward edges and
  corners, flattening the bathtub and lowering CV(J). Paired with the
  accumulate/breakthrough oscillation already in the temporal model. Grounded
  on the same thermal field as drying, with the sign reversed.

  **Correction to an earlier draft of this section**, which stated that
  flooding carried "the same flow-path caveat as starvation for *where* the
  water pools". It does not. The thermal-condensation route grounds flooding
  internally, exactly as it grounds drying. The earlier wording is withdrawn.
- **Starvation** — O₂ depletion along the cathode flow path. **The only fault
  that genuinely requires a directional axis**, and the literature does not
  describe that axis as a clean monotonic ramp: current falls from inlet
  downstream, but the outlet region can show *higher* current under voltage
  reversal, with upstream overshoots specifically under galvanostatic
  loading. Resolved in §11d (orientation-randomized), with its sensitivity
  arm elevated in §11e.

**A reasoned expectation about Test 3 — explicitly NOT a measured claim.**
Drying and flooding land on *opposite* radial signs, and the three faults
already differ in temporal character (fast / oscillatory / slow, implemented
and literature-grounded since Phase 2). Together that suggests Test 3
(cross-fault distinguishability) is satisfiable with no imposed axis at all.
That is an expectation recorded so it can be checked — **it is not evidence,
and it must be measured during implementation.** If it fails, Test 3 is not
satisfied and the form needs revisiting; the expectation does not get to
stand in for the measurement.

### 11d. The flow-path question — measured, then resolved

§11c once asserted that "the real templates show *no* flow-path signal
(row/column correlations ≈ 0)". **That reading was too strong, and this
section corrects it before relying on it.** The question was reopened and
measured: `scripts/measure_residual_directionality.py`.

**The residual IS directional — it was swamped, not absent.** Row, column and
diagonal correlations on the *full* template run only 0.023–0.104, which is
what the earlier claim rested on. But the radial component carries ~94% of
template variance. After removing the **purely-radial ceiling** (each cell
replaced by its radius-shell mean, ~40 effective parameters, §6e — so no
surviving structure can be an artifact of an imperfect 3-parameter fit), the
same correlations run **0.11–0.43**:

| configuration | air | H₂ | row | col | diag_sum | diag_diff |
|---|---|---|---|---|---|---|
| Normal_Flow | fwd | fwd | −0.269 | −0.110 | −0.268 | −0.113 |
| Inverse_Air_Flow | rev | fwd | +0.121 | −0.329 | −0.147 | +0.318 |
| Inverse_Hydrogen_Flow | fwd | rev | −0.425 | +0.094 | −0.234 | −0.367 |
| Inverse_Flow | rev | rev | −0.369 | +0.188 | −0.128 | −0.394 |

**Why "row/col ≈ 0" never ruled depletion out.** The cell is documented as
parallel-serpentine with **cross-flow** distribution between anode and cathode
(`docs/real_data_notes.md`), i.e. air and hydrogen run along perpendicular
axes. In a cross-flow cell combined reactant depletion is therefore
**diagonal** — the sum of two perpendicular gradients — and a diagonal has
near-zero correlation with *both* row and column index. The original test was
blind to exactly the geometry this cell has.

**The decisive test: does the direction track the gas routing?** Reversing a
gas inlet/outlet is physically a *reflection* of that gas's axis, so a
flow-locked residual must match a **mirrored** copy of its partner better than
an unmirrored one. Every candidate transform was tested, since the sensor-grid
orientation of the two axes is undocumented:

| reversal pair | identity | flip rows | flip cols | rot180 | transpose |
|---|---|---|---|---|---|
| Normal → Inverse_Air (air flips) | **+0.885** | +0.161 | −0.285 | −0.419 | +0.277 |
| Inverse_H₂ → Inverse (air flips) | **+0.990** | −0.147 | −0.020 | −0.577 | +0.204 |
| Normal → Inverse_H₂ (H₂ flips) | **+0.886** | −0.070 | −0.032 | −0.538 | +0.290 |
| Inverse_Air → Inverse (H₂ flips) | **+0.634** | +0.071 | −0.118 | −0.333 | +0.364 |

**Identity wins every pair, by a wide margin.** Correspondingly, **85.3% of
residual energy is config-invariant** (per-configuration specific shares 4.0%,
14.1%, 25.5%, 14.0%), and the 2.9× inner-block gradient of §6e is essentially
unchanged by air reversal (2.93× → 2.84×).

**Conclusion: the directional residual is fixture-locked, not flow-locked.**
Reversing either gas feed does not relocate it. This is consistent with the
compression / contact-resistance mechanism already established for the radial
component (`docs/real_data_notes.md`) and inconsistent with reactant
depletion, which would have to move with the flow.

**On the n=1-per-configuration confound.** Configuration is confounded with
session, so a difference between configurations could not be attributed to gas
routing alone. That confound would undermine a *positive* result. This is a
**null** result — "the direction did not move when the flow reversed" — which
session identity does not explain away. The confound therefore does not weaken
this particular finding.

#### The strongest argument AGAINST the no-axis decision

Recorded explicitly, because it is the best counter-argument available and
must not survive only in conversation.

**A healthy cell at adequate stoichiometry has little reactant depletion by
definition.** Starvation is precisely the regime where λ → 1 and depletion
along the channel becomes a first-order effect. So the measurement above
establishes that there is no flow-path structure in **healthy operation**; it
*cannot* establish that fault states would lack one. Every template analysed
here is healthy, and healthy data is close to uninformative about the
fault-state flow-path gradient.

**This is absence of evidence, not evidence of absence.** The decision below
is taken in spite of this argument, not because the argument has been
answered. If fault-labelled spatially-resolved data ever became available,
this is the first thing it would settle.

#### The practical blocker, which is a separate argument

Distinct from defensibility, and not to be conflated with it: **even granting
that an axis belongs there, its orientation in sensor-grid coordinates is
unrecoverable.** The dataset does not document how the air and H₂ axes map
onto grid rows and columns, and — because the data contains no flow-tracking
component — there is nothing in it from which to infer the mapping.

So orientation-randomization is partly a response to **ignorance of
direction**, not only to doubt about the mechanism's existence. Two separate
arguments support the same choice, and they should be reported as two.

#### What the literature adds, and why it cuts both ways

The spatial manifestation of these faults is **design-dependent, not
intrinsic**: flow-field pattern has decisive influence on the current
distribution, down to sharp drops at individual serpentine bends. Flooding
accumulates first in the **middle channels** of a serpentine design and
**under the ribs** in parallel fields; water propagates outlet→inlet on the
cathode while dry effects propagate inlet→outlet.

Design-dependence is the reading under which imposing *this* cell's documented
geometry would be defensible. But it also means a generic monotonic
inlet→outlet ramp is a crude caricature of what a parallel-serpentine cell
actually produces, whose real structure is channel-to-channel and
bend-specific. The literature supports the *mechanism* and undercuts the
*simple ramp*.

#### DECISION

1. **No imposed flow-path axis** in the primary configuration.
2. **Drying and flooding** grounded on the thermal field with opposite radial
   sign (§11c). Neither needs an axis.
3. **Starvation** carries a directional term whose orientation is **drawn
   uniformly per simulated cell and held fixed across that cell's episodes.**
   Per-episode randomization would be physically incoherent — a real cell has
   one flow field — so the draw belongs at cell scope, alongside `c`, `r0`,
   `w`. This encodes the literature-supported mechanism while refusing to
   encode a direction the data cannot support, so no result can depend on one
   specific unsupported axis.
4. The orientation is labelled **tuned-or-assumed** at its definition, per the
   §5d convention.
5. The **no-axis arm** is run as well, and is elevated rather than optional —
   see §11e.

This is deliberately not "ranging an uncertain parameter", which would treat a
structurally unsupported assumption as merely imprecise. It concedes that the
direction is unknown and denies the model any benefit from guessing it.

**What this decision depends on, and what invalidates it.** Point 1 is
tolerable only because points 2 and 3 leave the fault set still mutually
distinguishable. That rests on the **unmeasured** expectation recorded in
§11c: that drying and flooding separate on *opposite radial signs*, which —
with the already-implemented temporal character — is what would satisfy Test 3
without any imposed axis.

**That expectation is not evidence, and it is load-bearing here.** If it
fails, starvation is no longer the only fault lacking a defensible spatial
signature, and the no-axis decision loses its main practical support. **In
that case the decision must be revisited, not patched** — specifically, not by
reintroducing an axis for flooding or drying to recover separability, since
§11d's measurement argues against a flow-locked axis for any fault, and
§11c's thermal grounding is the whole reason those two need none. Revisiting
means returning to the form, with §11a's limits and §11b's fixed thresholds
both still binding.

### 11e. Starvation's no-axis arm, and its interpretation pre-committed

**Why this arm is elevated, not a side robustness check.** Starvation is
simultaneously:

- the **novelty extension** over Kim et al. (2023), who did not cover it;
- the fault with the **least defensible spatial signature**, per §11d;
- the one that, without an axis, reduces largely to a **magnitude** effect —
  the regime where spatial sensing helps least and a bulk sensor is closest to
  sufficient.

Those three facts coincide. So the no-axis arm does not test a peripheral
assumption; **it tests whether the headline extension survives at all.**

**The three branches, written down before anything is implemented or
measured**, so the outcome cannot be reframed once it is known. This is the
same discipline §11b applies to thresholds, applied to an interpretation
rather than a number.

1. **The spatial-vs-bulk delta for starvation survives both arms.** The result
   is robust to the axis assumption. Report it as such, noting that it holds
   with and without an imposed direction.
2. **It survives only with the directional term.** Then the starvation result
   **depends on an imposed direction the real data does not support**, and
   that must be reported as a stated limitation on the starvation finding
   specifically. It is **not** resolved by selecting the directional arm as
   the headline, and it is **not** grounds for dropping the no-axis arm from
   the writeup.
3. **It survives in neither.** Then starvation's spatial advantage **is not
   demonstrated by this work**, and the claim narrows to flooding and drying.
   The extension over Kim et al. is then reported as attempted and
   unsupported, not quietly omitted.

None of the three branches is a failure of the project. Branch 3 is a
legitimate negative result about one fault type, and reporting it as such is
the point of fixing the branches in advance.

> **Phase 4 must also read §11i before applying these branches.** The branches
> above fix *how* the starvation result is interpreted; §11i item 3 adds a
> **confound that determines why branch 3 would trigger**, measured after this
> section was written. In short: the no-axis arm reaches Phase 4 with
> starvation and flooding overlapping on spatial geometry (0.638 against a 0.95
> threshold), so an underperforming no-axis starvation result admits two
> explanations — genuine absence of spatial advantage, or confusability with
> flooding — which the spatial-vs-bulk delta alone cannot separate. Branch 3's
> conclusion holds either way, but the reason differs and must be reported.
> §11i instructs Phase 4 to run a **binary healthy-vs-starvation** comparison
> alongside the multi-class one to separate them, or to state explicitly that
> no separating measurement is available. Arriving at these branches without
> that instruction would mean reporting branch 3 without its reason.

### 11f. What replaces the validation steps that are unavailable

Internal-consistency checks only. These establish coherence, **not**
correctness, and must never be described as validation:

1. **Continuity and monotonicity** in severity (Test 1).
2. **Severity-resolved separability profile** (Test 2) — chance at low
   severity, rising with severity.
3. **Magnitude / dispersion trajectory direction** — each fault must move its
   diagnostic in the direction the literature claims, relative to its *own*
   healthy baseline.

   **For flooding, CV(J) is not a weak check — it is a vacuous one, and the
   earlier "weak for flooding specifically" hedge is withdrawn.** §2
   establishes that CV(J) *cannot* discriminate flooding from healthy on this
   generator at all: flooding lowers σ(J) and mean(J) together
   (0.047→0.041 and 0.697→0.432), and the normalization that makes CV(J)
   load-invariant cancels almost exactly the signal that would distinguish
   it. A check with no discriminating power is not insensitive, it is
   inapplicable, and recording it as "weak" would leave Phase 3b appearing to
   have a coherence check on flooding that it does not have.

   **Substitute, per fault**: drying and starvation are checked on **CV(J)**;
   flooding is checked on **mean(|J|)**, which §2 records flooding does
   depress and which §2 already recommends evaluating jointly with CV(J).

   Note the deliberate asymmetry with Test 2: mean(|J|) is *excluded* from the
   acceptance battery for precisely the reason it works here — genuine fault
   physics should move it. It is therefore usable as a coherence check and
   must **never** be promoted into an acceptance threshold.
4. **Cross-fault separation** at full severity (Test 3).
5. **Preserved invariants** (Test 4).
6. **Parameter labelling** — every constant tagged real / literature-grounded
   / tuned-or-assumed at its definition, as in §5d, with starvation's
   orientation-randomized axis explicitly in the third category.
7. **Both starvation arms run**, directional and no-axis, with §11e's
   pre-committed branches applied as written.

Together these answer "is this generator self-consistent and does it behave as
the literature describes?" They do not, and cannot, answer "is this what a
real faulty cell looks like?"

### 11g. The severity convexity exponent — adopted after a test failure

**This subsection exists so the sequence is on the record.** Convexity in
severity is presented nowhere in this document as a prior design choice,
because it was not one.

#### 1. What actually happened, in order

1. Drying and flooding were implemented with amplitude **linear** in severity:
   `M = 1 − s·(A_uniform + A_spatial·w)`. Linear was not a reasoned choice; it
   was the unexamined default.
2. §11b Test 1 passed comfortably (max d(0.01) = 0.0047 / 0.0052 against the
   ≤ 0.02 criterion), Test 3 passed at 0.972, Test 4 passed, §11f item 3
   passed.
3. **Test 2 failed at severity 0.05** — battery-max 0.591 (drying) and 0.598
   (flooding) against the fixed threshold of 0.560. It passed at severity 0.01
   and 0.02, and marginally at 0.04 (0.558 / 0.557).
4. A **convex** map, amplitude ∝ severity^p, was then proposed — **after
   seeing that failure, and because of it.**

**The physical argument for convexity is sound but was not reached
independently.** Flooding has a percolation threshold — liquid water does not
impede current until it actually blocks pores — and drying follows a nonlinear
conductivity-vs-water-content curve. Both genuinely imply threshold-like
rather than proportional onset. That reasoning holds up on its own terms. It
is nonetheless *post-hoc*: it was constructed to explain a fix, not derived in
advance and then tested.

A reader should be able to judge that sequence for themselves. The adoption
rests on two further points: that a linear map was never justified either (so
this replaces an arbitrary mapping with a literature-shaped one, not a
principled mapping with a convenient one), and that the §11b guards were
**not** touched — the threshold stayed at 0.560, severity ≤ 0.05 stayed, and
the feature that failed stayed in the battery.

#### 2. p is a free parameter the acceptance test cannot constrain

| p | sev 0.01 | sev 0.02 | sev 0.05 | verdict |
|---|---|---|---|---|
| 1.0 (linear) | 0.529 / 0.538 | 0.525 / 0.527 | 0.585 / 0.588 | **FAIL** |
| 1.5 | 0.527 / 0.522 | 0.521 / 0.522 | 0.522 / 0.526 | PASS |
| 2.0 **(adopted)** | 0.530 / 0.522 | 0.521 / 0.528 | 0.523 / 0.523 | PASS |

(drying / flooding; threshold 0.560.)

**p = 1.5 and p = 2.0 both pass, and nothing in the acceptance battery
distinguishes them.** The test constrains p only to "greater than about 1.3";
its value within the passing range is unconstrained by any measurement in this
project. `SEVERITY_CONVEXITY` is therefore labelled **tuned-or-assumed** at
its definition, and **tuned under knowledge of the test**, which is a weaker
status than the other tuned-or-assumed constants.

**Reporting instruction, in the same spirit as §6h's r0/w breadth
requirement**: any downstream result that is sensitive to p *specifically*
must say so and must report the sensitivity, not quote a single-p number. A
result that holds across the passing range of p can be reported without the
caveat; one that does not is a result about p, not about spatial resolution.

#### 3. The bound on any severity-linear formulation — a finding in its own right

Worth recording independently of the fix, because it bounds what *no* linear
form can achieve, not just this one.

At severity 0.05 a linear map produces a systematic relative-RMS shape change
of **d = 0.0220**. The healthy family's quiet end has temporal noise of
**1/60 = 0.0167** of the template's standard deviation
(`HEALTHY_TEMPORAL_RATIO_RANGE` reaches 60). So the perturbation a 5%-severity
fault imposes is *larger than the fluctuation it would have to hide under* in
the quietest cells — a 2.2% systematic change in a cell fluctuating at 1.7%.

With N = 1000 per class that is reliably detectable, and it is arithmetic
rather than a defect: **no form whose amplitude is linear in severity can
satisfy Test 2 at severity 0.05 while still reaching a physically meaningful
full-severity amplitude.** The two requirements are in direct tension at
p = 1. Any future reconsideration of the form should start here rather than
rediscovering it.

#### 4. The battery worked, and that is the other half of the finding

The failure was **isolated to one feature**. At severity 0.05 the six
geometric features scored 0.512–0.539, comfortably inside the threshold; only
`template_match` — correlation of the grid against its own cell's healthy
template — bound, at 0.591 / 0.598.

`template_match` was put in the battery specifically to catch escapes of this
kind: a change that preserves geometric character while still shifting how
much the grid looks like the healthy cell. **It caught exactly that, and it
was the only feature that did.** Recorded here as evidence the battery is
doing its job, not merely as the thing that failed — and as the concrete
reason §11b's answer to "is one scalar sufficient?" was no. A single-scalar
radial test would have passed this form at 0.539 and reported success.

It also follows that dropping `template_match` to pass Test 2 would have been
the textbook Goodhart move, and was never on the table.

### 11h. Starvation's two arms — pre-registered before implementation

Written before any starvation code existed, so the prediction below cannot be
reframed once the number is known. §11e fixed how the *Phase 4* result is
interpreted; this section fixes what Phase 3b expects to see and what it is
not allowed to do about it.

#### 1. Pre-registered prediction: the no-axis arm may fail Test 3 against flooding

**The prediction.** The no-axis arm couples starvation's suppression to local
current density — O₂ depletes fastest where current is highest, a genuine
self-limiting mechanism needing no flow path. But this cell's healthy template
is a **bathtub, high at the edges**, so current-coupled starvation suppresses
the edges. Flooding also suppresses the edges, being weighted to the coldest
regions. **Both arms of the comparison end up edge-weighted**, differing only
in that starvation's weight tracks each cell's own `c`/`r0`/`w` logistic
template while flooding's tracks a fixed thermal outer product. That
difference is real but weak, and it is the kind of difference that separates
at ~0.6 rather than ~0.97.

**So the no-axis arm is predicted to risk a genuine Test 3 failure on the
flooding–starvation pair specifically.** This is why Test 3 must be reported
**pairwise across all three fault pairs, per arm** — an aggregate figure would
hide it behind drying–flooding's 0.972.

**What the honest report is, if it happens.** *"The no-axis arm does not keep
starvation distinguishable from flooding in this cell's geometry."* That is a
finding about the arm and about the cell's bathtub geometry. It is **not** an
occasion to adjust amplitudes until the pair separates, and the amplitudes are
not to be touched in response to this number.

**Option C was available and was rejected for the reason it would have
worked.** A third option existed: lean harder on the per-cell `c`/`r0`/`w`
dependence to widen the starvation–flooding gap. It would very likely improve
*this specific number*. Its only advantage over the adopted form is passing a
test, which is the definition of what §11b's fixed thresholds exist to
prevent. Recorded here so that choosing it later would be visible as a
reversal rather than look like a refinement.

#### 2. Why a uniform no-axis arm (option A) was rejected

The simplest no-axis form would be **spatially uniform** suppression,
`M = 1 − s²A` with no spatial weighting at all. It was rejected.

A uniform modulation carries **zero spatial information by construction**. A
spatially-resolved classifier could not beat a bulk one on it — not as a
measured finding, but as arithmetic. That would settle §11e branch 3 before
Phase 4 measured anything, and Phase 4 would be computing a tautology.

**This is §10's standing test applied in the inverting direction.** §10 was
written about an artifact that handed the spatial arm a *free win* (the basis
mismatch, 100% separability at severity 0.01). The same question — *does this
choice reach both arms, or only one?* — catches the mirror image here: a
choice that hands the spatial arm a **guaranteed loss** is exactly as
corrupting as one handing it a free win, and for the same reason. Both
determine the headline comparison from the generator's construction rather
than from the measurement.

Recorded as a generalization of §10: the test is not "does this flatter the
hypothesis", it is "does this decide the comparison before it is measured".

#### 3. §11f item 3 for starvation — specified before measuring

Fixed in advance, the same way §11b's thresholds were.

**PASS/FAIL criterion: starvation must depress mean(|J|).** Starvation causes
severe current loss, and that holds whether or not a flow-path gradient
exists — so the criterion is robust to the very question §11d left open, and
it adjudicates both arms on the same footing.

**CV(J) is explicitly NOT a criterion for starvation.** The two arms disagree
on its *direction* by construction:

- the **directional** arm imposes a gradient, which *increases* spatial spread
  → CV(J) rises, matching the literature;
- the **no-axis** arm suppresses the bathtub's high-current edges, which
  *flattens* the profile → CV(J) falls.

The literature's CV(J) prediction for starvation is **contingent on a
flow-path gradient existing** — precisely what §11d found no support for in
this cell. Making CV(J) pass/fail would therefore encode the axis assumption
into a coherence check and **auto-fail the arm specifically built to avoid
that assumption.** A criterion that picks the winner before measuring is not a
criterion.

CV(J) direction is therefore **reported for both arms as an arm-dependent
diagnostic**, never as a threshold. Fixed here rather than discovered
mid-measurement.

### 11i. RESULT — the §11h prediction was confirmed, and what follows from it

Phase 3b is complete. 40 of 41 acceptance checks pass. The one failure is the
one §11h pre-registered.

#### 1. The prediction, and the measurement

§11h, written before any starvation code existed, predicted that the no-axis
arm would risk failing Test 3 on the flooding–starvation pair, and said the
difference was "the kind of difference that separates at ~0.6 rather than
~0.97."

**Measured: 0.638.**

| arm | drying–flooding | drying–starvation | flooding–starvation |
|---|---|---|---|
| directional | 0.968 | 1.000 | 0.986 |
| **no_axis** | 0.968 | 0.981 | **0.638 — FAIL** (threshold 0.95) |

The mechanism is the one pre-registered, not a reconstruction after the fact.
Radial-correlation means make it legible:

| arm | drying | flooding | starvation |
|---|---|---|---|
| directional | +0.976 | +0.423 | **+0.678** (between the two) |
| no_axis | +0.976 | +0.423 | **+0.087** (past flooding, overlapping) |

Current-coupled starvation suppresses the bathtub's high-current **edges**;
flooding suppresses the **coldest** regions, which in this cell are also the
edges. Both land in the same place. The only thing left distinguishing them is
that starvation's weight tracks each cell's own `c`/`r0`/`w` logistic template
while flooding's tracks a fixed thermal outer product — and that residual is
exactly what `template_match` picks up at 0.638.

**Pre-registration is what makes this informative.** The same 0.638 explained
after the fact would carry much less weight: a mechanism fitted to a known
number is nearly unfalsifiable, and this document would have no way to show
the explanation was not reverse-engineered. Because the prediction, the
mechanism and the rough magnitude were all fixed in §11h in advance, the
result functions as a test of the reasoning rather than a story about it.

**The pairwise requirement is what surfaced it.** An aggregate "best
cross-fault separability" for the no-axis arm would have reported **0.981**
— drying–starvation — and passed Test 3 cleanly. The collapse would have
gone unseen behind a number that looks healthy.

#### 2. The finding, and three things it is not

> **"The no-axis arm does not keep starvation distinguishable from flooding in
> this cell's geometry."**

That wording is §11h's, fixed before the measurement, and it is the claim —
not a paraphrase of one.

**It is NOT a defect to fix.** The arm did exactly what it was built to do:
couple suppression to local current density, with no imposed flow axis. The
overlap is a **finding about the interaction between a no-axis constraint and
a bathtub-shaped healthy template**, not a failure of implementation.
Amplitudes were not adjusted in response to it, and option C — leaning on the
per-cell `c`/`r0`/`w` dependence to widen the gap — was not revisited. Per
§11h, its only advantage would have been improving this specific number.

**It is NOT a reason to prefer the directional arm.** That arm passes Test 3
at 0.986, but it does so **by imposing a direction the real data does not
support** (§11d: identity beat every mirror in all four reversal pairs, 85.3%
of the residual is config-invariant). **Passing a test by way of an
unsupported assumption is not evidence for the assumption.** If anything the
contrast sharpens §11d's point: the directional arm's separability is
manufactured by the assumption under dispute. Both arms proceed to Phase 4;
neither is promoted on the strength of this number.

**It is NOT generalizable beyond this cell's geometry.** The collision is
specific to a healthy template that is **high at the edges**, where
current-coupled suppression and cold-region suppression necessarily land
together. A cell whose healthy profile peaked at the centre, or was flatter,
would put these two mechanisms in different places and might show no overlap
at all. Nothing here licenses "starvation and flooding are spatially
confusable in PEMFCs" — only in *this* cell's bathtub.

#### 3. What Phase 4 inherits — a confound it must plan for

Stated sharply, because it is easy to misread later.

**If the no-axis arm's starvation result underperforms in Phase 4, two
explanations are available and the spatial-vs-bulk delta alone cannot
distinguish them:**

1. **Genuine absence of spatial advantage** — starvation without an imposed
   axis really does reduce largely to a magnitude effect, which is the regime
   where a bulk sensor is closest to sufficient (§11e).
2. **Confusability with flooding on that arm** — the spatial arm has the
   information but cannot use it, because starvation and flooding occupy
   overlapping spatial geometry at 0.638, so a multi-class model loses
   starvation to flooding rather than to the bulk baseline.

§11e **branch 3's conclusion holds either way** — starvation's spatial
advantage would not be demonstrated by this work — but **the reason differs
and must be reported**, because the two imply different things about whether
spatial sensing could help starvation at all under a different cell geometry.

**§11e's three branches remain binding and are not reopened by this result.**
They were fixed before any starvation code existed and this section adds a
confound to them, not an amendment: nothing measured in Phase 3b adjudicates
any branch, because no branch is adjudicable without a spatial-vs-bulk delta
and Phase 3b produces none. Anyone arriving here first should read §11e before
interpreting a Phase 4 starvation number.

**Instruction for Phase 4**: plan a measurement that separates these two if
one is available. The obvious candidate is a **binary healthy-vs-starvation**
comparison alongside the multi-class one: explanation 2 predicts starvation's
spatial advantage recovers when flooding is removed from the label set,
whereas explanation 1 predicts it does not. If no such measurement is
available, **say so explicitly** rather than reporting the delta as though it
were unambiguous.

#### 4. The CV(J) decision was load-bearing, and it is now demonstrable

§11h ruled, before measuring, that CV(J) is **not** a pass/fail criterion for
starvation, on the grounds that the two arms would disagree on its direction
by construction. Measured, from a healthy baseline of 0.2514:

| arm | starvation CV(J) | direction |
|---|---|---|
| directional | 0.3045 | **rises** |
| no_axis | 0.1224 | **falls** |

Opposite directions, as predicted. **Had §11h made CV(J) pass/fail, it would
have failed the no-axis arm — for disagreeing with the flow-path assumption
that arm exists specifically to avoid.** The criterion would have selected the
winner before the measurement ran.

That decision was taken in advance and is now demonstrably load-bearing, which
is the clearest vindication in this project of fixing criteria before
measuring rather than after.

#### 5. Test 3's 0.95 threshold — weaker provenance than §11b's numbers, and bounded

**Stated because §11 should not read as though every threshold in it carries
equal weight.** It does not.

**When 0.95 was chosen, exactly.** It was hardcoded when the acceptance script
was first written, **before any Test 3 measurement of any kind existed** — no
pair had been computed, for either arm. That is the stronger of the two
possible claims. The weaker half: when the script was extended from two faults
to three and the threshold was promoted to the constant `T3_MIN`, drying–
flooding's 0.972 was already known. **The value was carried forward unchanged,
not re-chosen**, but the reuse was not blind to that one number.

What it was *not*: **0.95 was never recorded in §11 in advance**, unlike
Test 2's 0.560, which was fixed in §11b with a measured null behind it before
any fault code was written. "Chosen before measuring" is not the same standard
as "pre-registered in the design document", and only the latter is
tamper-evident. Honesty about which one applies here matters more than the
number.

**So the number is bounded instead.** The question a reader should be able to
settle directly is *how much does the conclusion depend on 0.95?* Test 3
pairwise across 10 seeds, N = 1000 per fault:

| arm | pair | min | max |
|---|---|---|---|
| directional | drying–flooding | 0.970 | 0.976 |
| directional | drying–starvation | 0.998 | 1.000 |
| directional | flooding–starvation | 0.980 | 0.986 |
| no_axis | drying–flooding | 0.970 | 0.976 |
| no_axis | drying–starvation | 0.972 | 0.988 |
| **no_axis** | **flooding–starvation** | **0.646** | **0.667** |

Worst case for the failing pair is 0.667; worst case across every other pair
is 0.970. So:

> **The conclusion — flooding–starvation fails on the no-axis arm while every
> other pair passes — holds for ANY threshold T in (0.667, 0.970].** Band
> width 0.302. The adopted 0.95 sits inside it with 0.282 of margin below and
> 0.020 above.

**The dependence is therefore weak, but not symmetric.** Nothing in the band's
lower two-thirds changes anything: the failing pair would still fail at any
threshold above 0.667. **The upper edge is where the real content is**, and it
deserves more weight than a parenthetical.

> **drying–flooding is marginal under any reading, on BOTH arms.** It measures
> **0.968–0.976**, i.e. within **0.03** of its own failure point, and it is the
> binding pair at the top of the band. At any threshold T > 0.970 it fails on
> the directional arm *and* the no-axis arm, which converts Phase 3b's result
> from **"one pair collapsed on one arm"** into **"Test 3 is failing broadly
> across both arms"** — a materially different conclusion about the whole
> rework, not a stricter grading of the same one.
>
> So §11i item 1's presentation of drying–flooding as a clean pass is **true
> only relative to the adopted 0.95**. At 0.98 it would read as a failure. The
> 0.638 result is robust; the *drying–flooding* result is not.

**Phase 4 inherits this as a known soft spot and should not rediscover it.**
The pair that §11c predicted would separate most cleanly — opposite radial
signs, the expectation the whole no-axis decision rests on (§11d DECISION) —
is in practice the *least* comfortable margin in Test 3. Two consequences
worth carrying forward: any Phase 4 difficulty separating drying from flooding
should be read against this margin rather than attributed to the classifier,
and any future tightening of Test 3's threshold must be understood as putting
drying–flooding, not starvation, first in the line of fire.

That is the honest shape of it: the 0.638 finding is robust across a wide
band, the adopted threshold is not centred in that band, the pair nearest the
edge is the one §11c was most confident about, and all of it is on the record
rather than resting on an assurance that the threshold did not matter.

**Asymmetry with §11b, stated plainly**: Test 1's 0.02 / 0.002 / nine-point
grid and Test 2's 0.560 at N = 1000 were pre-registered in the design document
before implementation, with a measured null justifying the latter. Test 3's
0.95 and Test 4's invariants were not — they were chosen while writing the
test code. Any future reader weighing these should treat the Test 1 and Test 2
thresholds as commitments and the Test 3 threshold as a bounded judgement
call.

