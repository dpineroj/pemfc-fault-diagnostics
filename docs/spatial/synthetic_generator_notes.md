# Synthetic Fault Generator Notes — `scripts/spatial/fault_generator.py`

> **SCOPE — this document describes a CLOSED line of work.** The
> spatially-resolved fault-detection phase was closed in November 2026; see
> [`../phase_closure_spatial.md`](../phase_closure_spatial.md) for the result, the
> reasoning and what stands independently.
>
> **Its findings stand as recorded.** Where this document says "the project",
> read "the spatial fault-detection phase". Forward-looking commitments —
> §11e's interpretation branches, §11i's Model B instruction, and §12's Model A
> and Model B scoping — are **retired with the phase and are not live
> obligations**.
>
> **§11a is the exception and is NOT scoped down**: its permanent-limitation
> statement is accurate verbatim and the closure document quotes it as still
> standing.

Reference notes for the synthetic fault-onset generator built in Task 2.
Companion to `scripts/spatial/fault_generator.py` (implementation), grounded in
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
argument of the spatial phase (a spatially resolved signal carries
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
`scripts/spatial/cdm_parser.py` at the time, were per-segment current in Amps,
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
`scripts/spatial/cdm_parser.py`; regression test in
`scripts/spatial/test_load_real_data.py`): the real CDM_C grid's raw values are
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

**Downstream implication — the headline finding of this phase, and the one
that carries into the RUL line — not the
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

#### Resolution measurement (`scripts/spatial/measure_resolution_effects.py`)

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

From `scripts/spatial/analyze_real_structure.py`. **Measurement only; no parameter
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
flow configurations, not one (`scripts/spatial/analyze_real_structure.py`):
distance-from-center correlation 0.930–0.936, quadratic R² 0.902–0.915,
row/column correlations near zero in every session, and cross-session template
correlation r = 0.976–0.999. The static template is reproducible across
independent sessions, which supports the fixture/compression reading rather
than a per-session or per-flow-configuration effect. Real CV(J) is *not* a
single number across sessions: 0.3810 / 0.3266 / 0.3812 / 0.3247 (Normal,
Inverse, Inverse_Air, Inverse_Hydrogen) — so the target band is ~0.325–0.381,
not 0.381 alone.

Reproducibility: `scripts/spatial/analyze_real_structure.py` (real structure, all four
configurations) and `scripts/spatial/measure_resolution_effects.py` (the 2×2 table
above) regenerate every real-data number in this section from the corrected
loader. Regression test: `scripts/spatial/test_fault_generator.py` — asserts CV(J) stays in
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

#### A SECOND narrowing, from Phase 4's adoption of all four load traces

Recorded here when the decision was taken, not discovered during the writeup.

Phase 4 Model A trains on synthetic data driven by **all four** of the real
dataset's load traces, not `Normal_Flow` alone (§12c). That buys episode-timing
diversity, which is the only available defence against the load-leakage risk
§12b measures. It also couples training to the Phase 5 holdout, and the
coupling must be stated precisely rather than as a general flag:

- **What is used**: `INTENSIDAD`, a **bulk scalar** channel from the Seville
  recordings, min-max normalized per run to a 0–1 load ratio.
- **What remains held out**: the **spatial current grids**. No real healthy
  spatial grid enters training, model selection, or feature construction in
  any form. The §6f claim above is unaffected on that axis.
- **What this costs**: Phase 5 can **no longer claim quietness on unseen
  operating conditions**. It can claim quietness only on **unseen spatial
  realizations of operating conditions the model has already seen**, because
  the load trajectories it is validated against are the ones that drove its
  training data.

So of the two generalization claims Phase 5 might have made, **the spatial one
survives and the operating-condition one does not.** This is a real reduction
in what Phase 5 establishes, accepted deliberately in exchange for Model A
being testable at all, and it stacks with the structural narrowing above
rather than replacing it.

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

Source: `scripts/spatial/measure_resolution_effects.py` (re-runnable). Measurement
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

The central deliverable of the spatial phase was a *delta between two arms*
— same
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

**Substantiated, not merely asserted — see `docs/spatial/external_fault_datasets.md`
§2.** Two independent real fault-labelled PEMFC datasets have now been
audited, Kim et al. (2023) and Zhou et al. (2022), and **both are
static-load**; neither is spatially resolved, neither covers starvation, and
neither captures fault onset as a transition. The apparent structural reason
is recorded there: inducing a *controlled, labelled* fault appears to require
holding operating conditions fixed, which is in tension with capturing the
same fault under dynamic loading, where the label becomes ambiguous about when
the fault began. That explains why this limitation is permanent rather than
merely current.

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
`scripts/spatial/measure_battery_null.py`'s `FEATURE_NAMES` is the authority and has
seven entries.

**The 0.560 is measured, not asserted — and it is inseparable from N.**
An earlier draft of this section proposed 0.60 from intuition. That number
was wrong in *both* directions depending on sample size, which is the real
lesson: **a separability threshold is meaningless without fixing N**, because
best-threshold accuracy on finite samples is upward-biased even when the two
classes are drawn from an identical distribution.

`scripts/spatial/measure_battery_null.py` measures the floor directly — grids drawn
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
measured: `scripts/spatial/measure_residual_directionality.py`.

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

> **RETIRED WITH THE PHASE.** These branches governed interpretation of a
> Phase 4 spatial-vs-bulk delta. **Model B was never built**, so branches 1–3
> were never adjudicated and are not live obligations. Kept because the
> discipline of fixing an interpretation before measuring is the transferable
> part, not because the branches await resolution.

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

#### Scope of the 0.638, clarified by Model A

**Model A did not reproduce the predicted confusability, and the reason bounds
this finding rather than contradicting it.** Model A's spatial arm confuses
flooding with starvation at **0.011 (directional) and 0.015 (no_axis)** — the
no_axis arm is barely worse than the directional one, where §11i item 3
anticipated a Phase 4 confound.

The 0.638 above was **best-threshold separability on the seven-feature
global-geometry battery at full severity**. Model A's spatial arm uses **all
324 grid values** plus the two bulk scalars, and separates the pair far better
than any single geometric scalar does. **The finding stands exactly as stated —
about that battery — and does not transfer to a full-grid classifier.**

Consequently **§11i item 3's Phase 4 confound is not triggered for Model A**,
and the binary healthy-vs-starvation comparison is not needed to disambiguate
its result, because no_axis starvation does not underperform there (spatial
recall 0.933, precision 0.882, against bulk's 0.718 / 0.519).

**That instruction stays in force for Model B.** Model B works on sequences at
short lead times, where the precursor measurement found starvation's entire
onset is 2–5 s and severity is identically zero more than ~5 s before the
threshold crossing. With far less signal to separate the classes, the
confound could still bite there, and the binary comparison remains the way to
tell the two explanations apart.

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


---

## 12. Phase 4 Model A — design, fixed before implementation

Snapshot fault classifier: one timestep in, class out (healthy / drying /
flooding / starvation). Two arms on identical underlying data; the delta
between them is the project's central result. **No time windows, no sequences,
no prediction** — that is Model B.

Recorded before any Model A code was written, in the §11b discipline: the
decisions below cannot drift to accommodate a result.

### 12a. The episode-vs-timestep inversion — a finding in its own right

Measured on one run, near-seam timesteps excluded (362 of 3739, 9.7%):

| | healthy | starvation | flooding | drying |
|---|---|---|---|---|
| **episodes** | — | **34** | 24 | **9** |
| **timesteps** | 795 | **212** | 1333 | **1037** |

**The two orderings invert.** Starvation is the episode *majority* (34) and the
timestep *minority* (212); drying is the episode minority (9) and the second
largest by timesteps (1037). The cause is onset duration: starvation episodes
last 2–5 s, drying episodes 30–90 s.

**Consequence: Model A and Model B need different class-balance decisions from
the same generator**, and that is not obvious from either count alone. Model A
is a snapshot classifier, so it works on timesteps — its minority class is
**starvation**, 6.3× under flooding. Model B works on sequences or episodes —
its scarce class is **drying**, at 9. Any balance decision must state which
unit it is in. The precursor measurement compounds this: drying is Model B's
scarce class *and* the only fault with a usable spatial precursor.

### 12b. Three measured risks that shaped this design

All measured before the design was fixed; each one changes it.

**Risk 1 — the temperature grid leaks the label.** At severity 0.9:

| | mean(T) | std(T) | spatial corr(T, J) |
|---|---|---|---|
| healthy | 81.48 | 0.62 | 0.97 |
| flooding | 75.01 | 0.16 | 0.67 |
| drying | 85.48 | 1.12 | 0.98 |
| starvation[directional] | 78.53 | 0.94 | 0.90 |
| starvation[no_axis] | 77.97 | 0.23 | 0.70 |

Between-class gaps of 3–10 K dwarf within-class spreads of 0.2–1.1 K, so
`mean(T)` alone is a near-perfect four-class label — and it derives from the
hardcoded per-fault `base_t` constants (70 / 69 / 68 / 73), which are
notebook-01-derived tuned-or-assumed values, not physics.

**Risk 2 — `mean(J)` alone is already near-perfect.** Pairwise best-threshold
accuracy on that single scalar at severity 0.9: healthy against every fault
**1.000**; drying–starvation 1.000 / 0.996; flooding–starvation 0.995 / 0.990.
Only **flooding–drying at 0.790** is genuinely hard.

**Risk 3 — `load_ratio` alone reveals the class**, because episodes are
*triggered by* load events. In-sample four-class upper bounds on a real run:

| features | accuracy | macro-F1 |
|---|---|---|
| load_ratio alone | 0.872 | 0.684 |
| mean(J) alone | 0.762 | 0.629 |
| **mean(J) + load_ratio** | **0.938** | **0.871** |

**Together these mean the bulk arm starts near ceiling**, leaving the spatial
arm almost no headroom — and for artifactual reasons, not physical ones. This
is §10's standing test firing in the **deflating** direction: a design that
hands the spatial arm a guaranteed near-tie decides the comparison before it
is measured, exactly as a free win would.

### 12c. Decisions

**Temperature: EXCLUDED from both arms.** Three reasons, and the third is the
structurally decisive one:

1. Risk 1's measured label leakage, from tuned-or-assumed constants.
2. The synthetic temperature field is a deterministic affine function of the
   current grid plus noise (spatial corr 0.67–0.98), so it adds little
   independent spatial information.
3. **Our synthetic temperature is 18×18; the real sensor's is 9×9.** Including
   it would build a resolution mismatch into the model that Phase 5's
   validation path cannot accommodate — **incompatible regardless of the
   leakage question**, which is why this reason stands alone.

**Arms.**

- **Spatial**: the 324 raw 18×18 current values **plus the two bulk scalars**.
  A spatially-resolved sensor also reports the mean, so spatial is a strict
  **superset** of bulk. The delta therefore measures the *added value of
  resolution*, and a zero delta is a clean negative result rather than an
  artifact of withholding information.
- **Bulk**: `mean(J)` and `load_ratio`. Nothing else.
- **Excluded from bulk, and why**: CV(J), σ(J) and every other spatial
  statistic — they require the grid, so including them would make the "bulk"
  arm secretly spatial and destroy the comparison; the temperature grid and
  `mean(T)` (Risk 1); `severity`, `onset_field_s`, `active_fault_type` as
  label-adjacent; `segment_id` and timestep index as identity proxies.

**Load traces: all four gas configurations**, grouping by (trace, cell). The
Phase 5 coupling this creates is recorded in §6f, precisely rather than as a
flag: a bulk scalar channel is used, the spatial grids stay held out, and
Phase 5's operating-condition generalization claim is surrendered while its
spatial one survives.

**Class balance: accept the imbalance.** `class_weight='balanced'`, per-class
metrics reported, **no resampling** — oversampling adjacent timesteps
duplicates near-identical grids and manufactures variety, the same pathology
the Zhou audit found (`docs/spatial/external_fault_datasets.md` §1d).

**Severity: train on all, evaluate in fixed bins** [0, 0.1), [0.1, 0.3),
[0.3, 0.5), [0.5, 1.0]. Excluding low severity would hide the regime where the
question is interesting; including it means some grids are near-healthy *by
construction* (§11b Test 2 guarantees near-null below 0.05), so aggregate
accuracy is depressed by a designed-in amount. Binning makes that visible
instead of letting it contaminate one headline number.

**Model family: multinomial logistic regression on standardized inputs,
identical for both arms** — same regularization grid, same CV protocol,
differing only in the feature matrix, so the delta reflects the input and not
the architecture. A small random forest is reported as a secondary robustness
check. No CNN and no torch: this is a hypothesis test, not a leaderboard; a
CNN would introduce architecture choices capable of creating the delta by
themselves. **If a linear model on raw grids cannot beat two scalars, that is
itself the finding.**

**Split: leave-(trace, cell)-out as primary**, because Phase 5 applies the
model to a real cell it has never seen, so this rehearses the actual
generalization demand, and because holding out whole cells means no adjacent
timestep ever crosses the fold boundary. Leave-one-segment-out is reported as
robustness. What each does and does not prevent:

| split | prevents | does NOT prevent |
|---|---|---|
| leave-(trace, cell)-out | all adjacency leakage; cell memorization | nothing further, once traces differ |
| leave-one-segment-out | episode leakage across segments | cell leakage — same cell both sides |
| time-based within a run | gross temporal leakage | adjacent timesteps still correlated across the boundary |

**near_seam_buffer timesteps are excluded from all sets** (9.7% of a run).

**Both starvation arms run end to end, separately. Neither is promoted**
(§11d/§11e). Fault states are **not validated output** (§11a) and no Model A
result may describe them as validated.

**No external dataset trains anything.** Zhou et al. is audited and not
integrated. The real spatial grids are Phase 5's and must not enter training
or model selection.

### 12d. Metrics

- **Primary: macro-F1**, not accuracy — 212 starvation against 1333 flooding
  makes accuracy a majority-class report.
- Per-class precision and recall, all four classes.
- **The full 4×4 confusion matrix, reported in full, never collapsed.**
- **Flooding↔starvation confusion as a separately named metric, per arm.**
  §11i predicts it specifically for the no_axis arm (0.638 pairwise), so it
  must be visible rather than buried in an aggregate.
- Severity-binned macro-F1, per arm.
- **Delta = spatial − bulk**, per class and per severity bin, with a bootstrap
  CI over **cell groups** — bootstrapping timesteps would ignore adjacency and
  understate the interval.

### 12e. Amplitude sensitivity — REQUIRED, not a caveat

Risk 2 and Risk 3 put the bulk arm at 0.938 / 0.871 in-sample against a
falsification criterion that declares the test uninformative above 0.95. That
outcome is **close to the expected one, not a remote possibility**, and the
cause is identified: the per-fault uniform-loss constants (0.20 / 0.25 / 0.25)
make `mean(J)` nearly diagnostic by itself, and they were chosen for plausible
full-severity magnitude loss **with no consideration of whether they would
make the bulk arm trivially strong**.

So Model A is run **across a range of those amplitudes**, and the delta is
reported as a function of them. The spatial-component amplitudes are held
fixed, so spatial information content stays constant while bulk information
varies — which isolates bulk headroom as the variable.

**Per §11 the amplitudes are NOT changed. This is measurement across them,
not retuning**, and the committed defaults remain the committed defaults.

**Interpretation, pre-committed before measuring:**

1. **Delta positive and stable across the amplitude range** — spatial sensing
   helps, robustly, and the result does not hinge on the amplitude choice.
2. **Delta appears only at low amplitudes** — the result **depends on a
   tuned-or-assumed constant** and must be reported as conditional on it, not
   as a general finding about spatial sensing.
3. **Bulk at ceiling across the whole range** — **Model A has no headroom and
   cannot test the hypothesis on this generator.** Report that, and state what
   would be needed to make it testable.

### 12f. Falsification — committed before implementation

1. **If spatial − bulk macro-F1 ≤ 0 with a CI including zero, in the severity
   bins where bulk is not at ceiling, then spatial sensing does not help
   snapshot classification on this generator**, and that is reported as the
   result.
2. **A positive delta confined to bins where bulk is already at ceiling is not
   evidence** and will not be reported as support.
3. **If bulk macro-F1 > 0.95 in every bin, Model A is uninformative regardless
   of the delta** — report that the test had no headroom rather than
   presenting a small delta as a finding.
4. Any reported delta carries the §12e amplitude conditionality.

### 12g. A measurement-integrity finding: arm-asymmetric fitting failure

Caught by a smoke test before Model A produced any reportable number, and
recorded here rather than only in `CLAUDE.md`, because it is a finding about
how to measure a two-arm delta, not an environment quirk.

**What happened.** The first implementation used
`LogisticRegression(max_iter=300)`. At that setting the **spatial arm (326
features) did not converge, while the bulk arm (2 features) converged
cleanly.** A non-converged fit understates its own score, so the effect was
to penalise the spatial arm specifically and **drive the delta toward zero.**

**Why it matters more than an ordinary bug.** It would have fired §12f's "no
headroom / delta absent" branch for a reason with **nothing to do with the
hypothesis** — a tooling choice, not the data — and the output would have
looked like a clean negative result. `sklearn` emits a convergence warning,
but warnings are easy to filter and easier to miss, so the defect was close to
invisible. **This is §10's failure mode in the deflating direction**: the
comparison decided before it is measured, this time against the hypothesis
rather than for it.

**The fix is the assertion, not the `max_iter` bump.** `max_iter` is now 5000,
but the durable part is that `evaluate()` **checks `n_iter_` against
`max_iter` on every fit**, collects any failure into `NONCONVERGED`, and prints
it at the end of the run. A raised limit silently stops being enough when the
feature count or sample size changes; an assertion does not.

**Standing practice, generalized** (and recorded in `CLAUDE.md` as a question
to ask): any multi-arm comparison in this project must assert convergence — or
the equivalent fit-quality condition for whatever estimator is used — **per
arm**, and report it. The reason is specific and worth stating in full: the
arms differ in dimensionality by construction, so a fitting failure is
*expected* to hit them asymmetrically, and an asymmetric fitting failure
biases the delta in a direction that **looks exactly like a substantive
result**.

**The catch was luck, not process, and that is the part worth recording.**
`sklearn` *did* emit a convergence warning — but the real job runs with
**`-W ignore`**, so on the actual experiment the warning was **actively
suppressed**. It surfaced only because a smoke test happened to print to a
terminal that was being read at the time. Had the full run completed first,
the biased delta would have been reported with no indication anything was
wrong.

**So: a suppressed warning is not a weaker signal than an assertion — it is no
signal at all, and the two cases are indistinguishable in the output.** Any
check in this project that depends on a warning being *visible* must be
converted into an **assertion** or an **explicitly returned status**. The real
job's `-W ignore` is not going away: it exists because `sklearn` and `numpy`
emit high-volume warnings that would otherwise bury the measurement output.
That makes warning-based checks structurally unreliable here, not merely
fragile.

**Reporting rule**: convergence status is reported **before** any numbers. If
any fit failed to converge, the affected numbers are **not reportable**.

### 12h. Three things the first Model A run got wrong, recorded

Same register as §12g. A project that logs only its correct predictions is not
keeping score.

#### 1. A runtime constant silently equalized the traces

Model A caps each `(trace, cell)` group at `MAX_PER_GROUP` timesteps. The cap
was introduced at 2500 **for runtime**, before the leave-one-trace-out
configuration existed.

Every trace has at least 2500 usable timesteps per cell, so the cap **bound on
all four of them** and equalized the design: 5 cells × 2500 = 12,500 samples
per trace, 25.0% each, every fold exactly 12,500. Inverse_Flow's ~6,500 usable
timesteps per cell were cut to 2,500 — about 62% of its data unused.

**That contradicted an explicit instruction** — do not reweight or subsample to
equalize; make the asymmetry visible — as a *side effect* of a constant that
predated it. The leave-one-trace-out test exists specifically to expose that
Inverse_Flow carries roughly twice the weight of any other trace, and the cap
removed the thing being tested.

**The reporting made it worse.** The per-trace table printed the header
"the asymmetry, NOT equalized" directly above four identical numbers. An
assertion that the opposite of the truth had happened is worse than printing
the numbers bare, because it discourages the reader from noticing. Both the cap
and that header are fixed; `MAX_PER_GROUP` now defaults to `None` with a
comment stating it may be set only to bind on nothing, never to make folds
comparable.

The capped run's numbers are **kept, not overwritten**, and reported beside the
uncapped ones: if the deltas are materially unchanged that is a robustness
result, and if they move the uncapped run is the one that answers the question.

**Generalization**: a constant adopted for one reason (speed) can satisfy or
violate a requirement adopted later (exposing an asymmetry), and nothing
connects them. When an instruction is about the *shape* of the data, the
constants that shape it have to be re-read, not just the code written since.

#### 1b. Capped vs uncapped — the comparison, and its three limits

Recorded as the true state rather than held open for a comparison that will
not arrive.

**Fact one — the capped numbers are TRANSCRIBED, and their authoritative log
is lost.** The capped run completed all 16 cells, and its figures throughout
§12 come from the report written off that run. The log file itself, and a copy
deliberately preserved beside it, were both lost to temp-directory clearing.
**The numbers are therefore second-hand within this project**, which is a
weaker provenance than every other figure in this document, and the weakness
is stated rather than paid off: re-running a complete capped configuration to
regenerate a log, for a closed phase whose numbers are already corroborated,
is not worth a multi-hour run. The lesson about *where* preserved artifacts
must live is in `CLAUDE.md`.

**Fact two — the uncapped run is partial, and consistent where it ran.** 3 of
16 cells persisted (directional arm, scales 0.00 / 0.25 / 0.50, both splits),
at 85,960 samples with Inverse_Flow carrying its real ~2.1× weight:

| | capped (complete) | uncapped (partial) |
|---|---|---|
| samples | 50,000 | **85,960** |
| per-trace share | 25.0% each (equalised by the cap) | **unequal; Inverse_Flow ~2.1×** |
| delta, split=cell | +0.103 to +0.123 | +0.113 to +0.118 |
| delta, split=trace | +0.109 to +0.130 | +0.124 to +0.128 |

**The cap did not change the conclusion.** That is the whole question the
uncapped run existed to answer, and it is answered.

**Fact three — the uncapped run is incomplete and was deliberately not
resumed.** It died twice at exit code 4 with no traceback either time. It
never covered the complete scale 1.00 cell or **any** of the no_axis arm, and
printed no convergence summary. **The capped run remains the phase's result**;
uncapped figures are corroboration and must not be cited as the phase's
numbers.

**On publishing this rather than holding it.** §12h was held uncommitted
pending the full uncapped comparison. That comparison will never exist, so
holding the document indefinitely would be worse than publishing it with its
limits stated — which is the same principle §11a applies to the fault patterns
themselves.

#### 2. A pre-registered expectation was falsified — §12f criterion 3

§12b measured the bulk arm at **0.938 accuracy / 0.871 macro-F1** and §12e
recorded that §12f's "no headroom" branch was therefore "close to the expected
one, not a remote possibility." That expectation was stated twice and **it was
wrong.** Criterion 3 did not fire: bulk reaches **0.822–0.836** macro-F1
out-of-sample and never approaches the 0.95 ceiling.

**Why the expectation was wrong, which is the part worth keeping.** The
0.938/0.871 figures were **in-sample, on a decision tree**, computed as an
upper bound on what two features could carry. They were used as though they
estimated what the bulk *arm* would score. They do not: the real bulk arm is a
2-feature multinomial logistic regression evaluated **out-of-fold under a
grouped split**, which is a different estimator on unseen groups. An in-sample
tree bound overstates an out-of-sample linear score, and nothing in §12b
flagged the substitution.

**Generalization**: an in-sample upper bound on a feature set's information
content is not a forecast of a specific estimator's out-of-sample score, and
should not be used to predict whether a test will have headroom.

#### 3. A third of the delta is the linear boundary, not information content

The random-forest check is specified in §12c as secondary, but one of its
numbers belongs in how the headline result is framed rather than in a
robustness appendix.

| estimator | spatial | bulk | delta |
|---|---|---|---|
| logistic regression | 0.945 | 0.822–0.828 | **+0.117 to +0.123** |
| random forest | 0.962–0.969 | 0.887–0.896 | **+0.066 to +0.082** |

The stronger estimator raises **both** arms but raises bulk more, closing
roughly **a third** of the delta. So part of the logistic delta is **the
2-feature linear decision boundary**, not the information the bulk channel
carries: two features with a nonlinear boundary do better than two features
with a linear one, and the spatial arm's advantage shrinks accordingly.

The delta remains clearly positive under both estimators, with the same sign
and the same per-class story. But the honest statement of magnitude is
**"+0.07 to +0.12 depending on estimator"**, not "+0.12", and any headline
number quoted from the logistic arm alone overstates the information-content
gap by about a third.

### 12i. The final uncapped attempt — rule fixed before launch

Recorded and committed before the run is launched (completion campaign,
November 2026).

- **One attempt, no more.** Resumes from the three persisted cells
  (`run_logs/model_a_results/`), incremental writes per `(arm, scale)` cell,
  design matrix freed between cells — both already in the script.
- The launch wrapper must propagate **Python's** exit status and write it as
  the log's terminal line (the earlier wrapper reported its own `echo`; see
  `CLAUDE.md`, "verify the operation, not the report").
- BLAS threads are capped for this job so it cannot starve Model B. That is a
  scheduling choice with no effect on any fitted number.
- **If it dies: no third attempt.** Partial coverage is recorded and the capped
  run remains the phase's result.
- **A fork the brief does not cover, decided here:** thirteen remaining cells
  at the observed ~2.4 h each is ~31 h. If the run is still alive when every
  other campaign item is complete, it is **stopped and its coverage recorded
  exactly as a death would be.** Reason: this is corroboration of a closed
  phase, not the campaign's purpose, and holding the close-out for it would
  invert that priority. Whatever cells exist at that point are reported, and
  what they do not cover is stated.
- Convergence is asserted per fit as before; the capped numbers stay
  transcribed (§12h) and the capped configuration is **not** re-run.

### 12j. The final uncapped attempt — result

From `scripts/spatial/phase4_model_a.py` with `MAX_PER_GROUP=None`, run log
`run_logs/model_a_attempt3.log`. **The run completed: the log's own terminal
line is `EXIT=0`**, and all eight `(arm, scale)` result files exist in
`run_logs/model_a_results/`. This was the single attempt §12i allowed. It
resumed three cells persisted by the earlier attempts and fitted the other
five.

**Convergence, first — and what the run's own summary does not cover.** The
log ends "All logistic-regression fits converged before max_iter." **That
line is true of 80 of the 128 fits and silent on the other 48.** The
non-convergence list lives in process memory, and the three resumed cells
(directional arm, scales 0.00, 0.25, 0.50) were fitted by earlier processes
that died before printing anything. Their predictions were persisted; their
convergence status was not. This is the same defect found and fixed in
`phase4_model_b.py` before its launch (the fit log is stored in each result
file there); it was not back-ported here, and should have been.

So, stated exactly:

| cells | fits | convergence |
|---|---|---|
| directional, scale 1.00; no_axis, all four scales | 80 | **asserted, all converged** |
| directional, scales 0.00 / 0.25 / 0.50 | 48 | **not recorded** |

The three unrecorded cells are **not re-fitted**: §12i allowed one attempt,
and re-running cells to repair the record is a further attempt under another
name. What can be said of them is an inference, labelled as one — identical
estimator, cap and data size to the 80 fits that converged, and their scores
sit on the same line as the verified scale-1.00 cell (0.943 → 0.946 → 0.948
→ **0.950**). That is corroboration, not an assertion. **Only the verified
cells are quoted as the uncapped result.**

**Capped and uncapped, side by side.** Macro-F1 delta, spatial − bulk,
logistic regression, range across the four amplitude scales:

| | capped (50,000; traces equalised) | uncapped (85,960; Inverse_Flow 41.2%) |
|---|---|---|
| directional, split = cell | +0.103 to +0.123 across arms | +0.119 at scale 1.00 *(+0.113 to +0.117 at the unrecorded scales)* |
| directional, split = trace | +0.109 to +0.130 across arms | +0.130 at scale 1.00 *(+0.124 to +0.128 unrecorded)* |
| no_axis, split = cell | *(as above)* | **+0.100 to +0.111** |
| no_axis, split = trace | *(as above)* | **+0.112 to +0.122** |
| random forest, scale 1.00, split = cell | +0.066 to +0.082 | **+0.083** directional, **+0.080** no_axis |
| `bulk > 0.95` (no headroom) | never | never |

Every uncapped interval excludes zero (lowest lower bound +0.090). The
capped column is transcribed from §12h, whose provenance limits still apply.

**What the uncapped run answers.** The cap had equalised the four load
traces and so could have hidden a dependence on Inverse_Flow. It did not:

- With Inverse_Flow at its real 41.2% share, the delta is the same size as
  under equal shares.
- Leave-one-trace-out, per fold at scale 1.00: the delta is **+0.108 to
  +0.140 whichever trace is held out**, in both arms. Inverse_Flow is the
  *easiest* trace to hold out for both arms (spatial 0.972, bulk 0.835), not
  an outlier driving the result.
- The amplitude sweep moves the delta by about 0.01 from scale 0.00 to 1.00.
  Zeroing the uniform-loss constants does not manufacture or remove it.

**Per class (scale 1.00, cell split), the picture of §12 is unchanged.**
Starvation recall: spatial 0.938 in both arms; bulk 0.709 (directional) and
0.730 (no_axis). Flooding↔starvation confusion: spatial 0.010 / 0.016, bulk
0.119 / 0.114. The advantage is still starvation, and still the
flooding–starvation confusion the bulk arm cannot resolve.

**The honest magnitude stands as "+0.07 to +0.12 depending on estimator."**
On the verified uncapped cells at the generator's actual amplitude, cell
split, it is +0.111 to +0.119 under logistic regression and +0.080 to +0.083
under a random forest. The quoted range is not narrowed to that: the capped
run is the one with a complete convergence record, the two agree, and
re-drawing the range after the fact from the more favourable of two runs is
the move this record exists to refuse.

**Two caveats that arrived later and apply here.** Both are recorded where
they were found and repeated so this section is not read without them:

- §15e: the leave-one-trace-out split does not hold out cells — the same
  five cells recur in every trace. The trace-split deltas above (the larger
  ones) carry that caveat. **Quote the cell split.**
- §16: neither arm's model stays quiet on the real healthy cell. These are
  accuracies at separating this generator's classes and nothing more (§11a).

**Status of the capped run.** §12i said the capped run remains the phase's
result *if the uncapped attempt died*. It did not die, so that clause does
not decide the matter; this does: the capped run is complete **and** fully
convergence-asserted, the uncapped run is complete and 62% asserted. **The
capped run remains the phase's quoted result; the uncapped run is its
corroboration, now across both starvation arms and the full amplitude sweep
rather than three cells of one arm.**

---

## 13. Model B — predictive model, design fixed before implementation

**Committed before any Model B code exists and before any Model B measurement
is taken.** The git history is the evidence: this section predates
`scripts/spatial/phase4_model_b.py`. Every threshold and interpretation branch
below is binding; §13k says what happens if one of them is later regretted.

Nothing here reopens the closure (`../phase_closure_spatial.md`). The economic
objection stands however this comes out. §11a applies in full: fault patterns
are coherent and literature-grounded, **not validated**, and no Model B number
is real-fault prediction accuracy.

### 13a. Why generation is the only route — and what that costs

**Bounded search, three queries, run before this section was written.**
Nothing published since the last audit pairs labeled faults with dynamic
loading in a public deposit, spatial or not. What surfaced: two Mendeley
deposits from one group (flooding/dehydration, 2019 and 2021; bulk signals; no
indication of dynamic loading — not downloaded, not audited, and by the Zhou
precedent a bulk-only static set would be audited and not integrated anyway);
a 2025 durability deposit that is not fault-labeled; and several 2025 papers
classifying faults with no public data visible. The one live lead — a
physics-guided LSTM study with flooding, drying, air- and hydrogen-starvation
classes under dynamic conditions, bulk-only, referenced rather than deposited
— gets a drafted, **unsent** request in `data_requests.md`. The campaign does
not wait on it.

**So the generator is the dataset**, for the reason both audits established
(`external_fault_datasets.md` §2): inducing a *controlled, labeled* fault
requires holding conditions fixed, which conflicts with capturing it under
dynamic loading, and no one has deposited spatially resolved labeled faults at
all.

**What that costs, stated before any result exists:** every Model B figure is
a statement about this generator's severity trajectories and spatial
modulations. The onset shapes are literature-grounded and tuned
(`SEVERITY_CONVEXITY`, the per-fault time constants); the measured precursor
windows that fix the horizons below are *properties of those tuned
trajectories*. A real cell could have longer or shorter precursors. Model B
can show whether a model *achieves the horizons the generator permits*; it
cannot show those horizons are real.

### 13b. Generation volume — drying minimum, PRE-REGISTERED BEFORE MEASUREMENT

Drying is the only fault with a precursor long enough for a multi-second
horizon, and it is the scarce class by episodes (9 of 67 on the Normal_Flow
trace, 6 crossing threshold).

**Unit of independence.** If trigger timing is a deterministic function of the
load trace (to be verified empirically, §13b-i), then the five cells drawn per
trace replay the *same* drying dwell events with different cells and severity
trajectories. The independent unit is then the **distinct load-trace event**,
not the (trace, cell) episode.

**Pre-registered minimum: 30 distinct drying trigger events across the four
traces, with at least 5 in every trace.**

- 30, because the headline drying quantity is a recall-type proportion over
  events and the worst-case binomial 95% half-width at n = 30 is
  1.96·√(0.25/30) ≈ 0.18 — the coarsest resolution at which "predicts" and
  "does not predict" remain distinguishable. Below 30 the interval swallows
  the claim.
- ≥ 5 per trace, because the leave-one-trace-out split evaluates on one
  trace's events alone, and a fold with fewer than 5 is an anecdote.

**(i) Verification owed before designing around determinism:** confirm
empirically that for a fixed trace, different seeds give identical trigger
event counts and timing. If seeds *do* move triggers, the unit becomes the
(trace, cell) episode, the minimum of 30 applies to those, and this paragraph
is superseded in place.

**(ii) Decision rule.** If the achievable count meets the minimum: proceed.
If not: the minimum is **not lowered** and Model B is **not** quietly run as
though it were met. Drying is run on the episodes that exist and reported as
**explicitly underpowered** — count achieved, count required, finding
*indicative rather than established*. Starvation and flooding are unaffected.

**(iii) §10 check on volume.** More traces and more cells change which load
contexts and which cells both arms see; they do not touch spatial geometry and
reach both arms identically. No volume change may alter a fault's spatial
modulation or amplitude. Seeds beyond the existing five cells per trace are
*not* added to inflate the drying count: if events are trace-deterministic,
extra cells add no independent events, and counting them would manufacture n.

### 13c. The task — early warning, not persistence

**Given a window of recent history ending at t, predict the fault state at
t + h.** Two arms on identical data, identical model, differing only in input.

**Restricted to windows in which no fault is yet confirmed.** A window is
*eligible* iff max severity over the window is below
`CLASSIFICATION_THRESHOLD` (0.5). Reason: for a window that already contains a
confirmed fault, "the state at t + h" is almost always "the state at t" — the
snapshot question Model A already answered. Including those windows would let
both arms score on persistence and report it as prediction. Eligibility is
defined on true severity, which neither arm sees, and applies to both arms
identically (§10: it cannot move the delta).

**Label** = state at t + h, constructed as:

| severity at t + h | label |
|---|---|
| exactly 0 (no active episode) | `none` |
| ≥ 0.5, fault type f | `f` (confirmed) |
| in (0, 0.5) | **window excluded** — target is "developing, unconfirmed"; calling it `none` would punish a model for detecting a real precursor, calling it `f` would label an unconfirmed fault |

A **healthy window** is an eligible window whose target is `none`. It may
contain sub-threshold severity inside the window (a fault that recedes); those
are legitimate hard negatives and are kept.

**Positives are stratified, because the strata mean different things:**

- **precursor-visible** — max severity in the window > 0. Spatial information
  exists in the input.
- **blind** — severity is exactly 0 throughout the window. The grids are
  healthy grids. Nothing spatial distinguishes them, by construction; any
  skill here can only come from the load trace, which the bulk arm sees too.

### 13d. Horizons and lookbacks — per fault, each justified

A shared horizon is incoherent: the measured precursor windows are starvation
1.2–3.9 s (median 2.1), flooding 2.3–8.4 s (median 4.1), drying 25.9–67.3 s
(median 47.8), and at 30 s starvation and flooding grids *are* healthy grids.

**Horizon rule (one rule, applied to all three):** the largest lead in the
already-measured sweep {1, 2, 5, 10, 20, 30, 45, 60 s} that does not exceed
**half the fault's median precursor window**. Half, so that a typical episode
offers precursor-visible windows at that lead rather than exactly one.

| fault | median window | half | **horizon h** | measured separability at that lead |
|---|---|---|---|---|
| starvation | 2.1 s | 1.05 | **1 s** | 0.711 directional / 0.567 no_axis |
| flooding | 4.1 s | 2.05 | **2 s** | 0.653 |
| drying | 47.8 s | 23.9 | **20 s** | 0.655 |

**Lookback rule:** Kim et al.'s 45 s is not inherited — it came from 10-hour
static holds and has the same transfer problem as their horizon. The lookback
is **the fault's median precursor window in samples, rounded up, minimum 3**
(three samples is the fewest that fit a trend with a residual): the window
should span the dynamic it is trying to see, and no more, since every extra
sample costs seam-excluded windows.

| fault | **lookback L** |
|---|---|
| starvation | **3 s** |
| flooding | **5 s** |
| drying | **48 s** |

Each fault is run as its own configuration `(h, L)`; inside each, the model is
4-class (`none` + three faults at *that* horizon) and **only the
configuration's own fault is its result.** Results are never aggregated
across faults.

**Negative control (pipeline check, not a result):** starvation and flooding
configurations re-run at **h = 30 s** (Kim et al.'s horizon). There every
positive is blind by construction, so any skill is load-trace reading and the
spatial-minus-bulk delta must be indistinguishable from zero. A nonzero delta
there is an artifact alarm (§13i criterion 4).

### 13e. Windows, seams, and continuity

- Windows are index-based on the 1 Hz series: input samples `t−L+1 … t`,
  target at `t+h`.
- **No window may touch a seam.** A window is dropped if *any* sample in
  `t−L+1 … t+h` carries `near_seam_buffer`, or if the span crosses a
  `segment_id` change. This extends the per-timestep flag to windows.
- A window is dropped if its span contains a time gap (consecutive stamps more
  than 1.5 s apart), since the slope feature assumes uniform spacing.
- Counts dropped by each rule are reported.

### 13f. Architecture — the simplest thing that can represent a trend

**Per input channel, two numbers summarise the window: its last value
(level) and its least-squares slope over the lookback (trend). A multinomial
logistic regression is fit on those, identically in both arms.**

- spatial arm: 324 grid cells + mean(J) + load_ratio = 326 channels → **652
  features**
- bulk arm: mean(J) + load_ratio = 2 channels → **4 features**

Why this and not a recurrent network:

- It *is* a sequence model in the sense that matters — a linear model over a
  fixed two-element temporal basis (constant, linear) per channel, i.e. a
  constrained version of logistic regression on the flattened window. A
  snapshot classifier cannot represent a trend; this can, and it is the
  smallest step past Model A that can.
- **It cannot create the delta.** No architecture choice differs between arms;
  the temporal basis is fixed, not learned, so the spatial arm gets no extra
  representational freedom beyond its extra channels.
- Continuity with Model A: same estimator family, same convergence assertion,
  so a Model A → Model B difference is attributable to the temporal features.
- No new dependency. torch is not installed and nothing here needs it.

What this deliberately gives up: nonlinear temporal structure (curvature,
oscillation phase). If neither arm predicts, that is one of the things the
result cannot distinguish from "no precursor is learnable" — and §13i says so.

**Asymmetric-tooling check (CLAUDE.md), answered here:** 652 vs 4 features.
Things that scale with dimension and how each is held: *solver* — `lbfgs`
(cost linear in features; newton-cholesky's Hessian is (652·4)² and
impractical), with Model A's measured solver-neutrality between the two
(delta shift +0.0003) as the reason this is not itself a confound;
*iterations* — `max_iter` set far above need and **convergence asserted per
fit** into an explicit `NONCONVERGED` list reported before any number;
*regularisation* — identical `C = 1.0` on standardised features in both arms
(a fixed C does regularise 652 coefficients more in aggregate than 4; that
handicaps the *spatial* arm if anything, the conservative direction for a
claim of spatial advantage, and it is recorded rather than tuned);
*standardisation* — fit on the training fold only.

Temperature stays excluded from both arms, for all three §12c reasons.

### 13g. Balance, splits, both starvation arms, the binary comparison

- **Class balance will shift again under windowing** and is measured and
  reported per configuration before any model result. Handling:
  `class_weight='balanced'`. **No oversampling** — adjacent windows are
  near-duplicates and duplicating them manufactures variety.
- **Splits:** GroupKFold on (trace, cell), and LeaveOneGroupOut on trace. Both
  reported. If they disagree materially, the disagreement is a finding about
  what the model generalises over and is reported as one.
- **Both starvation arms run end to end; neither is promoted.** §11e's three
  branches govern interpretation and are binding. The arms already differ at
  1 s lead before any classifier exists (0.711 vs 0.567 separability), so a
  Model B difference between them is read against that gap, not attributed to
  the model.
- **Binary healthy-vs-starvation** is run alongside the 4-class starvation
  configuration, on the same windows with other-fault targets removed —
  §11i kept this instruction live for Model B because leads are short.

### 13h. Metrics and reporting order

For fault f in its own configuration, one-vs-rest on the 4-class predictions:
precision, recall, F1, and **Youden's J = recall − false-positive rate**
(FPR over non-f targets), each for the spatial and bulk arms; recall split
into precursor-visible and blind positives; the full confusion matrix,
uncollapsed; **delta = spatial − bulk** on F1 and on J with a 95% bootstrap
CI resampling **(trace, cell) groups**, never windows.

Reported in this order, always: convergence status; which §13i criterion
fires; per-fault results at their own horizons; per arm and per split; both
starvation arms; the binary comparison; confusion matrices; CIs.

### 13i. Falsification and interpretation — fixed in advance

For each fault f, in order:

1. **Uninformative — the test cannot speak.** If bulk F1 for f exceeds 0.95
   (no headroom), or if fewer than 30 positive windows exist in total for f,
   the configuration is declared uninformative and no spatial claim is made
   from it.
2. **Neither arm predicts.** If J's 95% CI includes or lies below zero in
   *both* arms, the horizon is **not achieved by this model** for f. Reported
   as: the separability measurement found a precursor that this predictor
   did not exploit — or the precursor is not learnable from level-and-trend
   features. The record's per-fault horizon claim is then *not confirmed* for
   f and must be amended to say so.
3. **Spatial does not help.** If delta-F1's CI includes zero or lies below
   it, spatial sensing does not improve prediction of f at its horizon, and
   that is reported as the result. A bulk arm that out-predicts the spatial
   arm is reported as readily as the reverse.
4. **Artifact alarm.** If on **blind** positives the spatial-minus-bulk recall
   difference has a CI excluding zero, or if the 30 s negative control shows
   a delta whose CI excludes zero, the pipeline is leaking something that is
   not a precursor. The affected fault's delta is **not reportable as spatial
   advantage** until the leak is identified, and the alarm is reported.
5. **Spatial helps.** Only if none of 1–4 fires: delta-F1 CI above zero is
   reported as a spatial advantage in prediction, with the share of it carried
   by precursor-visible positives stated.

**Horizon achieved (the question the record has only asserted):** for fault f
and an arm, the horizon is achieved iff J's CI lower bound is above zero.

**Starvation, §11e mapping:** branch 1 if criterion 5 holds in both
starvation arms; branch 2 if only in the directional arm (then the finding
depends on an imposed direction the data does not support, reported as that
limitation, the no_axis arm not dropped); branch 3 if in neither (starvation's
predictive spatial advantage is not demonstrated).

**Drying:** additionally subject to §13b — if the event minimum is not met,
whatever fires is reported as indicative, not established.

**Split disagreement:** if a criterion fires under one split and not the
other, both are reported and the fault's verdict is stated as
split-dependent.

### 13j. Runs, writes, failures

Incremental, resumable writes per (configuration, starvation arm, split) to
`run_logs/model_b_results/`; log inside the repo; the log's own terminal line
is the exit status. Two attempts maximum per experiment; after that the
partial result is recorded with what it does not cover. A fit that fails to
converge makes its experiment's numbers unreportable: remedy is a solver
change with a neutrality check, never fewer samples, arms or features.

### 13k. If a threshold is later regretted

It stands. The wish to change it, the result that prompted the wish, and the
reason it stands are recorded next to the result.

### 13l. Measured before any model code: volume, balance, and two corrections

From `scripts/spatial/measure_generation_volume.py` (labels only; no grids, no
fits). Recorded and committed **before `phase4_model_b.py` exists.**

**Triggers are deterministic in the load trace — verified, not assumed.** For
all four traces, the episodes generated under five different seeds have
identical (fault, onset) sets, and those sets equal what the trigger finders
return on the bare trace. Seeds change the drawn cell and the severity
trajectory; they do not change which events fire or when. The unit of
independence in §13b therefore stands as the distinct load-trace event.

**Yield per trace** (trigger events; threshold-crossing episodes per cell):

| trace | starvation | flooding | drying |
|---|---|---|---|
| normal | 34 (21–26 cross) | 24 (24) | 9 (5–9) |
| inv_flow | 184 (125–140) | 31 (31) | 6 (5–6) |
| inv_air | 57 (38–41) | 24 (24) | 9 (5–8) |
| inv_h2 | 92 (60–68) | 26 (26) | 9 (4–7) |

**Drying against the pre-registered minimum: MET AS WRITTEN — 33 distinct
trigger events (minimum 30), smallest trace 6 (minimum 5).**

**Correction 1 — the minimum was operationalised on the wrong count, and the
conservative reading is the one adopted.** §13b's stated *purpose* was a
recall-type proportion over events with a half-width no worse than 0.18. The
events that proportion is actually taken over are those contributing positive
windows at the drying configuration, and after the eligibility rule (no
confirmed fault anywhere in a 48 s window) that is **18 distinct events, not
33**. At n = 18 the worst-case half-width is 0.23.

So the criterion as written is met and the purpose it was written for is
not. Choosing between those readings after the fact is exactly the freedom
pre-registration exists to remove, and the favourable reading is not taken:
**drying is reported as INDICATIVE, NOT ESTABLISHED**, with both counts
stated (33 trigger events; 18 contributing). The written minimum is not
edited. The lesson — a pre-registered count must be the count the estimate is
actually taken over — is recorded here rather than papered over.

**Correction 2 — the negative control's premise was wrong as stated.** §13d
says that at h = 30 s "every positive is blind by construction". Measured:
of 2,409 starvation-control positives only 1,001 are blind; of 10,495
flooding-control positives, 7,032. The rest are precursor-visible, because
"visible" means *any* nonzero severity in the window, and a window can carry
sub-threshold severity from an earlier or different episode — most plainly
flooding's own accumulate/drain oscillation, where a drained-but-not-ended
episode precedes its next crossing by a cycle period of 20–45 s. That residue
is genuine spatial information about what follows; a spatial advantage on it
would not be an artifact.

~~§13i criterion 4: "…or if the 30 s negative control shows a delta whose CI
excludes zero…"~~ **Superseded before any model result existed:** the
artifact alarm is evaluated on the **blind stratum of the controls only** —
the spatial-minus-bulk recall difference on blind positives must have a CI
including zero. Those are the windows whose grids are healthy grids by
construction, which is the property the control was meant to rest on. The
controls' overall and visible-stratum deltas are still reported, as
description. This amendment follows from a label-only measurement and
predates every fit.

It also matters that **the three main configurations contain no blind
positives at all** (0 of 534, 0 of 1,273, 0 of 647): every eligible positive
window already contains some severity. So criterion 4's blind-positive clause
is vacuous in the main configurations, and the controls' blind strata are the
only place the artifact check lives.

**Window class balance** (kept windows by target, all 20 groups; labels are
identical across starvation arms):

| configuration | none | starvation | flooding | drying | own-fault events |
|---|---|---|---|---|---|
| starvation (h 1, L 3) | 13,507 | **534** | 825 | 90 | 232 |
| flooding (h 2, L 5) | 12,333 | 892 | **1,273** | 175 | 98 |
| drying (h 20, L 48) | **48** | 145 | 796 | **647** | 18 |
| starvation control (h 30) | 4,609 | 2,409 | 10,870 | 2,186 | 231 |
| flooding control (h 30) | 3,441 | 2,097 | 10,495 | 2,137 | 104 |

Of 92,315 timesteps, each configuration keeps 1.6k–20k windows: 6.5–8.7k are
dropped at seams, **52k–78k because a confirmed fault is already inside the
window**, and 3–16k because the target is developing-but-unconfirmed. No
window is dropped for a time gap.

Three consequences, stated before any model runs:

- **The inversion moves again.** Starvation, the episode majority and Model
  A's timestep minority, is again the smallest own-fault class (534) — one or
  two eligible windows per crossing at a 1 s horizon.
- **The drying configuration has almost no healthy targets: 48.** This
  generator is fault-dense — a 68 s span free of any confirmed fault that
  also ends healthy barely exists. The drying model therefore discriminates
  "drying ahead" from "a different fault ahead" far more than from "nothing
  ahead". Its false-positive rate is estimated over 989 non-drying targets,
  of which 48 are healthy; that is stated wherever the drying result is.
- **1,636 windows against 652 spatial features** in the drying configuration
  is close to one sample per two parameters per class. Over-parameterisation
  bites the spatial arm only. That is the conservative direction for a claim
  of spatial advantage, and a reason to read a drying *null* for the spatial
  arm cautiously.

Balance handling stands as registered: `class_weight='balanced'`, no
oversampling.

---

## 14. Phase 5 — the no-false-alarm test, pre-registered

The only point in the project where a trained model meets measured data.
**Committed before any Phase 5 code runs.** The real grids contain no faults,
so this does not test detection. It tests whether models trained on synthetic
faults stay quiet on a genuinely healthy cell.

### 14a. What is run

Final models — Model A (snapshot) and Model B (each fault configuration) —
trained on **all** synthetic groups at the committed amplitudes, both
starvation arms, both input arms; applied to every post-warm-up timestep of
all four real Seville configurations (~5 h each). Seams do not exist in a
continuous real recording; windows spanning a timestamp gap are dropped as in
§13e.

### 14b. The input transform — stated because it decides the bulk arm's result

The generator expresses grids in **load-normalised units**: a healthy
synthetic grid has mean 0.697 at every load, and a fault lowers the mean as a
*fraction of demanded current not delivered*. A real grid's mean tracks load
(0.05–0.8 A/cm²). Fed raw, every low-load real timestep would look like a
massive fault to both arms — a units mismatch, not a finding.

**Transform, fixed here:** each real grid is divided by the bulk-measured
current density at that instant (`INTENSIDAD` / active area) and multiplied
by 0.697. `load_ratio` is `INTENSIDAD` min–max normalised per run, exactly as
in generation. Timesteps with non-finite `INTENSIDAD` are dropped and counted.

**Consequence, stated before the result:** under this transform the bulk
arm's mean(J) feature is the grid-integral-to-`INTENSIDAD` ratio, which the
loader's own regression test pins at 1.005 ± 0.008. **The bulk arm is
therefore quiet nearly by construction** — a real cell under galvanostatic
load delivers its demanded current. Its quietness is close to a tautology and
must be reported as one. **The informative arm is the spatial one**, which
sees the real template: 5.84% irreducibly non-radial, four sessions
concentrated in one corner of the family. A bulk arm that false-alarms less
than the spatial arm is the *expected* direction and is reported as readily
as the reverse.

### 14c. Independence check — required before running

Confirm that no real spatial grid has entered training, model selection or
feature construction in any form. Known and already recorded (§6f): the
healthy family's *form* was chosen with the real template in view, and
`INTENSIDAD` drove generation. So the claim available is **quietness on
unseen spatial realisations of operating conditions the model has seen** —
not on unseen operating conditions, and not structural independence of form.
That narrowing is stated *at the result*, not in a footnote.

### 14d. Criterion

False-alarm rate (FAR) = fraction of evaluated timesteps (or windows) whose
predicted class is anything other than healthy / `none`.

| FAR, per model, per input arm, per configuration | verdict |
|---|---|
| ≤ 5% | **PASS** — quiet |
| > 5% and ≤ 50% | **FAIL** — not quiet; reported with which class it alarms |
| > 50% | **FAIL, simulator-learned** — the model's healthy region does not contain the real cell |

5%, because it is the conventional tolerance for a false-positive rate and
because Model A's own held-out synthetic false-positive rate on healthy was
1.6% (spatial) and 0.2% (bulk): a real-data rate several times the synthetic
one is the signature of a model that learned the simulator. An arm passes
Phase 5 only if it passes on **all four** real configurations. Also reported,
not criteria: FAR against the model's own cross-validated synthetic FPR; FAR
split by conditioning-hold versus dynamic segments; which fault class the
alarms name; and a sustained-alarm rate (≥ 3 consecutive alarmed samples).

**A failure is a result.** If the spatial arm fails, the phase's completed
record says so: a model trained on this generator does not stay quiet on the
real cell, which bounds what every Model A and Model B number can mean.

## 15. Model B — result

From `scripts/spatial/phase4_model_b.py`, run log `run_logs/model_b.log`
(terminal line `EXIT=0`), reported in the order §13h fixed. Every number is a
statement about this generator (§13a) and none is real-fault prediction
accuracy (§11a).

### 15a. Convergence

**All 192 fits converged.** Maximum iterations used: spatial 1,162, bulk 119,
against a cap of 20,000. The fit log is stored inside each result file, so
this covers every fit behind every number below.

### 15b. Which pre-registered criterion fires

**Criterion 3 — SPATIAL DOES NOT HELP — fires for every fault, in both
starvation arms, under both splits.** In no configuration is the spatial arm's
F1 above the bulk arm's. Criterion 5 (spatial helps) fires nowhere.

**The horizon is achieved in both arms for all three faults** (Youden's J CI
lower bound above zero everywhere, lowest +0.711). Criterion 2 fires nowhere.

Criterion 1 (uninformative) fires once, as written: flooding, no_axis
starvation arm, trace split, where bulk F1 = 0.951 exceeds the 0.95 ceiling.
The same configuration under the other three arm/split combinations sits at
0.945–0.949, just under it. Flooding is therefore at the edge of having no
headroom at all, and its result is read that way.

Criterion 4 (artifact alarm, blind stratum of the 30 s controls) fires in six
of eight control cells — see §15e.

### 15c. Per-fault results at their own horizons

Split = (trace, cell) GroupKFold unless stated; the trace split agrees
throughout and is in the log. Delta = spatial − bulk, CI over (trace, cell)
groups.

**Starvation, h = 1 s, L = 3 s** — 534 positives from 211 events.

| starvation arm | input | precision | recall | F1 | J | delta-F1 [95% CI] |
|---|---|---|---|---|---|---|
| directional | spatial | 0.837 | 0.820 | 0.829 | 0.814 | **−0.072 [−0.163, +0.002]** |
| directional | bulk | 0.922 | 0.880 | **0.900** | 0.877 | |
| no_axis | spatial | 0.859 | 0.835 | 0.847 | 0.830 | **−0.054 [−0.112, −0.006]** |
| no_axis | bulk | 0.946 | 0.860 | **0.901** | 0.858 | |

**Flooding, h = 2 s, L = 5 s** — 1,273 positives from 98 events.

| starvation arm | input | precision | recall | F1 | J | delta-F1 [95% CI] |
|---|---|---|---|---|---|---|
| directional | spatial | 0.867 | 0.873 | 0.870 | 0.860 | **−0.077 [−0.120, −0.040]** |
| directional | bulk | 0.924 | 0.972 | **0.947** | 0.964 | |
| no_axis | spatial | 0.868 | 0.876 | 0.872 | 0.863 | **−0.077 [−0.116, −0.039]** |
| no_axis | bulk | 0.926 | 0.973 | **0.949** | 0.966 | |

**Drying, h = 20 s, L = 48 s** — 647 positives from **13 events**; 989
non-drying targets of which 48 are healthy. **INDICATIVE, NOT ESTABLISHED**
(§13l correction 1).

| starvation arm | input | precision | recall | F1 | J | delta-F1 [95% CI] |
|---|---|---|---|---|---|---|
| directional | spatial | 0.912 | 0.898 | 0.905 | 0.841 | **−0.031 [−0.072, −0.002]** |
| directional | bulk | 0.954 | 0.920 | **0.936** | 0.890 | |
| no_axis | spatial | 0.932 | 0.887 | 0.909 | 0.845 | **−0.023 [−0.053, +0.005]** |
| no_axis | bulk | 0.953 | 0.912 | **0.932** | 0.883 | |

**The bulk arm out-predicts the spatial arm for every fault.** For flooding
the interval lies wholly below zero under both splits and both arms: the bulk
arm is not merely unbeaten, it is better. §13i said a bulk arm that
out-predicts the spatial arm is reported as readily as the reverse; this is
that report.

A further count correction on drying: §13l recorded 18 contributing events
from a label-only pass; the model run finds **13** events whose positives
survive into the kept windows. The lower number is the operative one and
strengthens the *indicative* status.

### 15d. Both starvation arms, and the binary comparison

**§11e branch 3.** Criterion 5 holds in neither starvation arm, so
**starvation's predictive spatial advantage is not demonstrated by this
work.** The two arms are indistinguishable here (bulk F1 0.900 vs 0.901;
spatial 0.829 vs 0.847), so the 0.711-vs-0.567 baseline separability gap
between them did not carry into prediction.

**Binary healthy-vs-starvation**: both arms at F1 0.996–1.000 under both
splits, delta within ±0.003. "Is starvation coming, or nothing?" is answered
perfectly by two bulk features. So the 4-class starvation errors are all
*between fault types* — the confusion matrix shows it (directional arm,
spatial: 63 starvation→flooding, 69 flooding→starvation; bulk: 54 and 40) —
and the spatial arm confuses them slightly *more*, not less. §11i's confound
does not rescue a spatial advantage; there is none to rescue.

### 15e. The controls, and the alarm

At h = 30 s both arms still score above chance (starvation control J
0.15–0.31; flooding control J 0.58–0.76), and the bulk arm again leads.
**Skill at 30 s for faults whose whole onset is under 9 s is load-trace
reading** — exactly what the precursor measurement predicted — and the bulk
arm, which sees load directly through two features, does it better than a
model that must find those two features among 652.

**Criterion 4 fires as written in six of eight control cells**, and the sign
matters:

- **Flooding control, all four cells: bulk recall on blind positives exceeds
  spatial** (0.92–0.97 vs 0.71–0.79; dBlind CI −0.32 to −0.10). The spatial
  arm is *worse* on windows whose grids are healthy grids. That is not a leak
  in the spatial arm's favour. It is the cost of 648 uninformative features
  under the same regularisation — the over-parameterisation §13f and §13l
  flagged in advance as biting the spatial arm only.
- **Starvation control, trace split only: spatial recall on blind positives
  exceeds bulk** (0.33 vs 0.11; dBlind CI +0.14 to +0.30). Under the cell
  split the interval includes zero. This is the direction an artifact would
  take, and it appears only in the split where **cells are not held out**: the
  five drawn cells recur in every trace, so under leave-one-trace-out the
  spatial arm can recognise *which cell* it is looking at and the bulk arm
  cannot. The spatial arm also runs at twice the false-positive rate there
  (0.155 vs 0.070), so part of the gap is operating point rather than
  information. **Candidate leak identified, not proven: cell identity shared
  across traces in the trace split.**

Per §13i the affected delta is not reportable as spatial advantage — and none
is being claimed, since the delta is negative. But the alarm reaches backward:
**Model A's trace split shares the same construction.** Its trace-split delta
was the slightly more favourable one (+0.130 vs +0.123, §12). That 0.007
difference is now of uncertain origin, and Model A's (trace, cell) split is
the one to quote. Recorded here; §12's numbers are not edited.

### 15f. What the result means, and what it does not

**The per-fault horizon finding is confirmed by a trained predictor, in both
arms:** 1 s for starvation, 2 s for flooding, 20 s for drying (the last
indicative). That was only a separability claim before; it is now measured.

**Spatial resolution adds nothing to prediction on this generator, and costs
something.** The spatial arm is a superset of the bulk arm, so it cannot be
worse on information; it is worse on estimation. Two bulk features — mean(J)
and load — with their slopes already carry the precursor, because **in this
generator a developing fault is a shortfall in delivered current**, which
mean(J) reads directly, and because faults are triggered by load events,
which load reads directly.

**§10, applied to the result rather than to a design choice.** The bulk
arm's strength is itself partly a construction of the generator. A real cell
under galvanostatic load delivers its demanded current; its mean(J) is pinned
(Phase 5's transform shows it: 0.7007 ± 0.005) and a fault appears in voltage
and in redistribution, not as lost mean current. So "bulk predicts as well as
spatial" is a statement about a generator whose magnitude channel carries the
fault. It does not show that bulk signals would suffice on a real cell — and
§16 shows how little any of these models transfers.

**A wish recorded and declined (§13k).** After seeing this, the obvious move
is to tune regularisation per arm, or select features for the spatial arm,
and re-run. That would be choosing the analysis after the result. `C = 1.0`
in both arms was registered with its asymmetry stated; it stands. What can be
said honestly is the limit of the null: **this shows a level-and-trend linear
model gains nothing from the grid, not that no model could.**

---

## 16. Phase 5 — result

From `scripts/spatial/phase5_false_alarm.py`, run log `run_logs/phase5.log`
(terminal line `EXIT=0`), with `phase5_diagnostics.py` run afterwards.

### 16a. Before any number

**Convergence: all 16 final models converged.**

**Independence (§14c), checked rather than asserted:** every final model was
trained before the real-data loader was imported; the script asserts
`cdm_parser` is absent from `sys.modules` at that point, and the assertion
passed. No real spatial grid entered training.

**The claim available is narrow and is stated here, at the result:** because
`INTENSIDAD` drove generation and the healthy family's form was chosen with
the real template in view (§6f), a pass would mean *quietness on unseen
spatial realisations of operating conditions the models have seen* — not on
unseen operating conditions, and not structural independence.

### 16b. The verdict, as registered

FAR on all post-warm-up timesteps, per real configuration (Normal /
Inverse / Inverse_Air / Inverse_H₂). Pass requires ≤ 5% on all four.

| model | starvation arm | input | FAR per configuration | verdict |
|---|---|---|---|---|
| A snapshot | directional | spatial | 0.986 0.976 0.987 0.961 | **FAIL, simulator-learned** |
| A snapshot | directional | bulk | 0.893 0.847 0.880 0.888 | **FAIL, simulator-learned** |
| A snapshot | no_axis | spatial | 0.069 0.976 0.091 0.959 | **FAIL** |
| A snapshot | no_axis | bulk | 0.893 0.876 0.882 0.889 | **FAIL, simulator-learned** |
| B starvation | directional | spatial | 0.015 0.014 0.014 0.024 | **PASS** |
| B starvation | directional | bulk | 0.053 0.017 0.053 0.063 | **FAIL** |
| B starvation | no_axis | spatial | 0.010 0.015 0.014 0.027 | **PASS** |
| B starvation | no_axis | bulk | 0.055 0.016 0.057 0.066 | **FAIL** |
| B flooding | directional | spatial | 0.180 0.060 0.087 0.079 | **FAIL** |
| B flooding | directional | bulk | 0.063 0.062 0.070 0.078 | **FAIL** |
| B flooding | no_axis | spatial | 0.061 0.041 0.075 0.061 | **FAIL** |
| B flooding | no_axis | bulk | 0.061 0.063 0.070 0.078 | **FAIL** |
| B drying | directional | spatial | 0.987 0.942 0.994 0.967 | **FAIL, simulator-learned** |
| B drying | directional | bulk | 0.934 0.927 0.928 0.923 | **FAIL, simulator-learned** |
| B drying | no_axis | spatial | 0.956 0.928 0.954 0.941 | **FAIL, simulator-learned** |
| B drying | no_axis | bulk | 0.936 0.929 0.928 0.923 | **FAIL, simulator-learned** |

**Fourteen of sixteen fail. Model A fails in every arm: three of its four
arm/input combinations exceed 50% on all four configurations, and the fourth
exceeds it on two.** The only
passes are Model B's starvation configuration, spatial input, in both
starvation arms.

**The models trained on this generator do not stay quiet on the real healthy
cell. They learned the simulator.** That is the registered failure, and per
§14d it bounds what every Model A and Model B number can mean.

### 16c. A prediction in the pre-registration itself was wrong

§14b stated, before the run, that the bulk arm would be *"quiet nearly by
construction"* because the transform pins its mean(J) feature. **It alarms on
85–89% of real timesteps.** The premise held — transformed real mean(J) is
0.7007 ± 0.0053 against a synthetic healthy 0.697 — and the conclusion was
still wrong, because the bulk arm has a second feature and I reasoned as
though it had one. Recorded with the result it mis-predicted; §14b is not
edited.

### 16d. What the models actually learned (diagnostics, verdicts unchanged)

**The prediction path is sound.** The same stored models through the same
`_predict` alarm on 0.1–0.6% of *synthetic* healthy rows. The failure is not
a pipeline bug.

**The bulk arm learned the generator's trigger rule.** At a healthy mean(J),
sweeping load alone flips its prediction: `flooding` at load 0, `healthy`
from 0.1 to 0.4, **`drying` from about 0.5–0.7 upward.** That is
`find_dwell_events` read back out of a classifier — low-load dwell triggers
flooding, high-load dwell triggers drying.

**And the training data left it no alternative.** Synthetic training rows by
load level:

| load_ratio | healthy | drying | flooding | starvation | healthy share |
|---|---|---|---|---|---|
| [0.0, 0.1) | 223 | 91 | 22,820 | 1,856 | 0.9% |
| [0.1, 0.2) | 3,274 | 820 | 1,514 | 3,467 | 36.1% |
| [0.2, 0.4) | 10,236 | 5,506 | 0 | 2,638 | 55.7% |
| [0.4, 0.6) | 35 | 8,289 | 0 | 751 | **0.4%** |
| [0.6, 0.8) | 10 | 2,017 | 0 | 148 | **0.5%** |
| [0.8, 1.0] | 10 | 22,186 | 0 | 69 | **0.0%** |

**A healthy cell at sustained high load does not exist in the training
distribution — 55 rows of 33,515 above load 0.4.** The generator makes a
fault a near-deterministic consequence of the load regime. The real cell
spends 78.5% of its recording in conditioning holds at sustained constant
load, healthy throughout; every model alarms on essentially 100% of hold
timesteps where its training associates that load with a fault.

This is §12b's Risk 3 — *"`load_ratio` alone reveals the class, because
episodes are triggered by load events"*, measured at 0.872 accuracy before
Model A was built — arriving at the one test that could expose its cost. It
was treated then as a threat to the *delta*. It was a threat to the
*models*.

**The spatial arm has a second failure of its own.** Model A's spatial input
names `flooding` on the real cell (directional arm, 93–99% of timesteps),
and in the no_axis arm splits by recording session: 7–9% FAR on Normal and
Inverse_Air, 96–98% on Inverse and Inverse_H₂ — the same two session
clusters `real_data_notes.md` identified by static spread (CV 0.38 vs 0.325).
Flooding in this generator *flattens the bathtub*; the flatter real sessions
are read as flooded. The healthy region the spatial model learned does not
contain the real cell, and where it appears to, that depends on which
starvation arm happened to be in the training set — a decision boundary that
moves with something unrelated to health.

### 16e. The pass, and why it should not be over-read

Model B's starvation configuration with spatial input passes in both arms
(1.0–2.7%). Two things qualify it:

- **It passes on dilution.** Its FAR is exactly 0.000 on hold timesteps and
  **3.4–12.7% on dynamic segments** — above 5% on dynamic segments in three
  of four configurations for the directional arm. The registered criterion
  counts all post-warm-up timesteps and the verdict stands as PASS; but the
  holds are operating conditions the model never saw and happens to be silent
  on, and they are 78% of the denominator.
- **It is the model with the least to say.** A 3 s window and a 1 s horizon,
  on a class whose synthetic positives are tied to steep load ramps. On a
  recording dominated by constant load it has little occasion to fire.

The same model with bulk input fails narrowly (5.3–6.6% in three
configurations), so here — the one place it does — **the spatial arm is
quieter than the bulk arm.** Reported because §14 required whichever arm
false-alarms less to be reported as readily as the reverse; the reverse holds
for Model A, where no_axis spatial is quieter than bulk on two sessions and
louder on the other two.

### 16f. What Phase 5 establishes

- **Negative, and decisive:** classifiers trained on this generator are not
  deployable on the real cell even as healthy/not-healthy detectors. The
  synthetic-to-real gap is not a calibration offset; the models encode the
  generator's load-to-fault rule and its template family's boundaries.
- **It retroactively bounds Model A and Model B.** Their accuracies are
  accuracies at separating this generator's classes, a task in which the load
  regime is close to the label. The spatial-vs-bulk *delta* is a comparison
  between two arms that share that defect, so it is not thereby void — but no
  absolute number from either model says anything about a real cell, healthy
  or faulted. §11a said this of fault patterns; Phase 5 extends it, by
  measurement, to the models.
- **What it does not establish:** anything about detection. There are no
  faults in the real data.

---


## 17. POST-HOC follow-up — plan, fixed before any fit

**Everything in §17 onward is post-hoc.** It was conceived after the Model A,
Model B and Phase 5 results were seen, in response to problems those results
exposed. It carries less evidential weight than §12–§16 for that reason, and
every table, heading, commit and README line derived from it says POST-HOC.

Written and committed before any model in it is fitted. Repository state at
writing: `442e57c`. Nothing below has been run except one label-only check
(§17b: no grids, no fits).

### 17a. What does not change, whatever comes out

- **Nothing already reported is edited, re-run in place or overwritten.** The
  falsified §14b expectation, every §15 and §16 table, and §12 stay as
  written. Where a result here bears on one of them, the new result is
  reported *alongside* and the old one keeps its place.
- **The capped Model A run stays the quoted Model A result** and "+0.07 to
  +0.12 depending on estimator" stays as written.
- **The §15 null stays the pre-registered Model B result.** Nothing
  exploratory overrides it.
- **The closure reasoning** (`phase_closure_spatial.md` §2) is untouched.
- **§11a.** No number here is real-fault accuracy.
- **The real healthy recording is a burned test set.** Its Phase 5 failures
  have been inspected (§16d). Any re-test on it is labelled **NOT AN
  INDEPENDENT TEST**, is run once, and cannot validate anything.
- **The healthy family is not touched**: not its form, its parameter ranges,
  or its sampling. Nothing is fit to, tuned against or selected using the
  real recording.
- `fault_generator.py` stays byte-for-byte unchanged. SHA-256 at writing:
  `32c57c020e3d39890efd8cdb02ca9be3b25e59e4867ba312b0b022fef14d0958`.
  Checked again at close-out, with `test_fault_generator.py` and
  `test_phase3b_acceptance.py` run before the first fit and after the last.

### 17b. A finding that precedes the plan: the leak is wider than §15e said

From `scripts/spatial/posthoc_confirm_cell_leak.py`
(`run_logs/posthoc/cell_leak.log`, `EXIT=0`). Label-only.

Both models seed a cell as `BASE_SEED + 1000 * cell_idx`, with no trace term,
and the generator draws healthy parameters from that seed alone. Measured:
**each `cell_idx` has one parameter set across all four traces — 5 distinct
cells behind 20 groups.** Fold by fold:

| split | test cells also present in training | test traces also present in training |
|---|---|---|
| "cell" — GroupKFold on (trace, cell) | **all of them, every fold** | **all of them, every fold** |
| "trace" — LeaveOneGroupOut on trace | **all of them, every fold** | none |

**So neither split ever holds out a cell.** The split §12 and §13 call
"unseen CELL" holds out only a *combination*: the model has seen that cell
under three other traces and that trace under four other cells.

**This corrects §15e and §12j, which are not edited.** Both say the trace
split is the leaky one and tell the reader to quote the (trace, cell) split
instead. That advice rested on a premise I did not check — that the
(trace, cell) split holds out cells. It does not. It also weakens §15e's
explanation of the starvation-control alarm: if shared cell identity were the
whole cause, the alarm should have fired under the cell split too, where
cells are equally shared. It fired only under the trace split. Cell identity
may still matter in combination with an unseen trace; §17c is built to
separate the two rather than assume.

### 17c. Workstream 1 — corrected splits

**Design: a 2 × 2 over what is held out.** Two splits exist; two are added.

| | cell seen in training | cell held out |
|---|---|---|
| **trace seen** | S0: GroupKFold(4) on (trace, cell) — *original* | **S2: leave-one-cell-out**, 5 folds, 16 training groups |
| **trace held out** | S1: leave-one-trace-out, 4 folds — *original* | **S3: crossed**, 20 folds: test = group (T, c); train = every group with trace ≠ T **and** cell ≠ c (12 groups) |

**The fix chosen: rebuild, do not retire. S3 becomes the corrected split.**
Retiring the trace split in favour of (trace, cell) is not available — S0
leaks more than S1, not less. S3 is chosen over regenerating 20 distinct
cells because it changes only the split: same samples, same features, same
cells as §12 and §15, so a difference between S1 and S3 is attributable to
holding out the cell and to nothing else. Its cost is stated: each fold
trains on 12 groups instead of 15, which lowers every score somewhat in both
arms. S2 is there to show how much of any change is the cell and how much is
the smaller training set and unseen trace.

**Runs** (new outputs under `run_logs/posthoc/`; originals untouched):

- *Model B* — all five configurations and the binary task, both starvation
  arms, both inputs, under S2 and S3. Model, features, windows, labels,
  `C = 1.0`, `max_iter = 20000`, bootstrap (1000 resamples over
  (trace, cell) groups) all imported from `phase4_model_b.py`, not retyped.
  S0 and S1 are read from the persisted §15 files.
- *Model A* — the **capped** configuration (`MAX_PER_GROUP = 2500`, sampling
  seeds `777 + cell_idx`, as in the quoted run), four amplitude scales, both
  arms, logistic regression under S0–S3; random forest at scale 1.00 under S0
  and S3. S0 and S1 are re-fitted here only because the capped run's
  per-sample predictions were not kept. **Reproduction gate:** if re-fitted
  S0/S1 macro-F1 differs from §12's recorded value by more than 0.005 in
  either arm, the run stops there and I ask before going on — a
  re-implementation that cannot reproduce the original cannot correct it.

**Criteria, fixed now.**

1. *Starvation control, artifact alarm* (§13i criterion 4 as amended by
   §13l, unchanged): spatial − bulk recall on blind positives, CI excluding
   zero = alarm. Reported for S0–S3, both arms.
   - Alarm under S1 but not S3 → holding out the cell removes it; shared
     cell identity under an unseen trace was the leak. §15e's candidate is
     confirmed in that narrower form.
   - Alarm still fires under S3 → **cell identity is not the explanation and
     §15e's candidate is falsified.** The artifact is then unexplained, is
     reported as unexplained, and the control's spatial delta stays
     unreportable as advantage. No further split is tried.
2. *Model B main configurations:* §13i criteria applied as written to S3.
3. *Model A:* delta with 95% group-bootstrap CI under S3. Lower bound > 0 →
   the snapshot advantage survives holding out cell and trace. CI includes 0
   → it does not, and every quotation of Model A's delta carries that caveat
   from then on (the quoted figure itself is not edited).

**Expectation, stated so it can be wrong.** I expect the spatial arm to lose
more than the bulk arm from S1 to S3 in Model A, because 324 segments can
identify a cell and two bulk features cannot (every cell's healthy mean is
0.697). I expect the Model A delta to shrink and stay positive. For Model B
I expect criterion 3 to fire again. I have no confident expectation for the
control alarm — §17b removed the reasoning I had.

### 17d. Workstream 2 — the convergence-record gap (verification only)

Re-fit exactly the three cells with no convergence record — directional arm,
scales 0.00, 0.25, 0.50, uncapped, both original splits (48 fits) — with the
configuration and seeds of §12j, writing `n_iter_` against `max_iter` for
**every fit** to a JSON beside the predictions
(`run_logs/posthoc/model_a_refit/`). The persisted §12j files are read, never
written.

§12i allowed one attempt and §12j declined to re-fit on that ground. This
re-fit is made because it has now been explicitly requested as verification;
it is not a further attempt at the result and changes no quoted number.

- **Match** = all 48 fits converged, and for each cell/split/arm the re-fit
  macro-F1 is within 0.001 of the resumed one, with per-sample agreement
  reported. Then those cells' numbers in §12j become verified.
- **Any fit not converged** → that cell's uncapped numbers are unreportable.
- **Converged but not matching** → reported as a reproducibility failure of
  the resumed cells; neither version is quoted.
- In every case the capped run stays the quoted result.

### 17e. Workstream 3 — Phase 5 false alarms

**3a. Re-analysis, no refitting.** For the 16 stored models, false-alarm rate
on conditioning holds and on dynamic segments separately, per real
configuration, recomputed from the stored coefficients through the Phase 5
prediction path (`run_logs/posthoc/phase5_split/`). Descriptive: §16b's
verdicts stand. The 5% line is shown against the dynamic-only rate for
reference and is **not** a re-adjudication — the registered criterion counted
all post-warm-up timesteps.

**3b. One generator change: a healthy companion run.** For every
(trace, cell), a second run is added over **the same load trace** with **the
same cell parameters** and **no fault episode**: severity 0 throughout, same
segment ids and seam flags, independent grid noise (seed = the model's
existing noise-seed formula + 32452843). It belongs to the same (trace, cell)
group, so no split separates a cell from its companion. Ratio 1 : 1. Call the
original data v1 and v1 + companions v2.

- *Why this change.* §16d's diagnosis is that the generator makes a fault a
  near-certain consequence of the load regime, so healthy operation above
  40% load is 55 of 33,515 rows. The companion makes healthy operation cover
  exactly the load distribution of the real dynamic segments, at every load
  level the trace visits.
- *Why it is physically grounded and not fit to the real cell.* It asserts
  only that a healthy cell can run the FC-DLC cycle without faulting — which
  is what a healthy cell is. It uses the load trace, already the generator's
  input. It uses no real grid and no real bulk signal, and no parameter of it
  was chosen by looking at the real recording: the ratio is 1 : 1 because that
  is the uninformative choice, and it is not revisited.
- *Why a companion rather than thinning the triggers.* Thinning redraws every
  fault episode. The companion leaves every v1 faulted run **bit-identical**
  (asserted against the v1 cache), so v1 and v2 share every positive window
  and the cost of the fix can be measured as a paired difference (3d).
- *What it does not fix, by design.* The spatial arm's healthy region did not
  contain two of the real sessions (§16d). Fixing that would mean moving the
  healthy family toward the real cell, which rule 17a forbids.
- **One change, tested once. No second variant, whatever the result.**

**3c. Re-test on the real recording — NOT AN INDEPENDENT TEST.** The 16 final
models are retrained on all v2 synthetic data (same estimators and settings
as `phase5_false_alarm.py`; the `cdm_parser` independence assertion kept) and
run through the identical transform and the identical registered criterion:
≤ 5% on all four configurations to pass, > 50% = simulator-learned. Hold and
dynamic rates reported separately as in 3a.

What each outcome would mean:

- *A model passes.* The specific defect diagnosed in §16d is removed for that
  model. **It is not validated**: the change was designed after seeing this
  recording fail, on this recording.
- *A model still fails.* Generator-trained models remain unusable on the real
  cell even after the diagnosed defect is addressed; §16f is strengthened.
- Either way §16b's table is the Phase 5 result.

Expectations, stated so they can be wrong — noting that my last expectation
about the bulk arm (§14b) was falsified:

- **Bulk Model A falls below 50% in all four configurations, both arms.** I
  do **not** predict it passes 5%. The synthetic healthy mean(J) is far
  tighter than the real one (s.d. 0.0053), so real excursions below 0.697 at
  high load may still read as early drying.
- **Spatial Model A still fails on Inverse and Inverse_H₂** (the flatter
  sessions), because nothing here addresses that.
- **Model B drying falls below 50% with bulk input**; spatial uncertain.

**3d. Does the fix cost prediction? (synthetic)** Model B on v2, three main
configurations and the binary task, both starvation arms, both inputs, under
S3; Model A on v2 at scale 1.00 under S3. Reported: precision, recall, F1,
FPR, J per arm; spatial − bulk delta with CI; §13i criteria as written.
**The paired cost measure:** recall on own-fault positives, v2 minus v1 under
S3, on the identical positive windows, 95% group-bootstrap CI, per fault and
arm. *Costs detection* iff the CI lies wholly below zero; the size is
reported either way. Precision is not paired (v2 has more negatives) and is
reported unpaired.

**This is the model the project's stated goal points at — predicting a fault
before it is confirmed without alarming on a healthy cell — and it is
post-hoc.** A good result here is a statement about v2 of a generator.

### 17f. Workstream 4 — matched tuning for both arms (EXPLORATORY)

Question: does "this model gains nothing from the grid" (§15f) survive as "a
tuned model gains nothing from the grid"?

- Data: **v1**, the §15 data, three main configurations, both starvation
  arms, 4-class task. Same model class, features and class weighting.
- Outer split: **S3**. The request named the (trace, cell) split; §17b shows
  it holds out neither cell nor trace, so S3 is used and the substitution is
  recorded here.
- Inner split: leave-one-cell-out among the four training cells (4 folds),
  so `C` is never selected on a cell the candidate has seen.
- Grid, identical for both arms: `C ∈ {0.001, 0.01, 0.1, 1, 10}`. It
  contains the registered value and extends further toward strong
  regularisation than away from it, because the concern recorded in §15f is
  that `C = 1.0` leaves 652 features under-regularised.
- Selection: own-fault F1 pooled over inner validation folds; ties go to the
  smaller `C`. Same budget per arm: 5 candidates × 4 inner folds + 1 refit
  per outer fold.
- Convergence asserted and recorded for every inner and outer fit. A
  candidate with a non-converged inner fit is ineligible in that fold, for
  that arm, and is listed. A non-converged outer refit makes that
  configuration's tuned numbers unreportable.
- Reported: selected `C` per fold and arm; tuned F1 per arm; tuned delta with
  95% group-bootstrap CI; the untuned S3 delta from §17c beside it.
- Reading: CI includes or lies below zero → the null survives matched tuning,
  for this model class. Lower bound > 0 → *an exploratory indication* that
  regularisation, not information, produced the null. **Neither outcome
  overrides §15.** A positive would justify a properly pre-registered
  replication on fresh draws and nothing more.

### 17g. Workstream 5 — drying statistical power

**Target, fixed before any drying result here: at least 30 distinct drying
trigger events contributing positive windows at the drying configuration,
with at least 5 from each trace.** Justification: §13b's purpose was a
recall-type proportion with worst-case half-width ≤ 0.18, which needs
n ≥ 30; §13l found the minimum had been applied to trigger events (33)
rather than contributing events (18 by labels, 13 in the fitted data). This
applies the same number to the count the estimate is actually taken over.

**The ceiling, stated first.** Triggers are deterministic in the load trace
(§13l) and the four traces contain 33 drying trigger events in total. No
amount of generation produces a 34th. The target asks that 30 of those 33
contribute.

**Method.** An event fails to contribute when a confirmed fault — typically
the starvation episode triggered by the ramp into the high-load dwell — sits
in its 48 s window. Whether that episode crosses the threshold depends on the
cell's seed. So: **40 cells per trace instead of 5** (`cell_idx` 0–39, same
seed rule, the first five identical to §15's; v1 generator, unchanged).
40 is fixed now on this arithmetic: 13 of 33 contributing with 5 cells
implies roughly a 0.10 per-cell chance, and 1 − 0.90⁴⁰ ≈ 0.98. That assumes
events are alike, which they need not be.

- A **label-only volume measurement** runs first and reports contributing
  events per trace, (cell, event) instances, and the count of healthy
  targets (48 in §15).
- **Target met** → drying is reported as established *for this generator, in
  a post-hoc analysis*.
- **Target not met at 40 cells** → the count is not raised and the target is
  not lowered. The comparison is still run once, as registered here, and
  reported as **underpowered**; the shortfall is then a structural limit of
  four load traces.

**Comparison.** §15's protocol: h = 20 s, L = 48 s, same features, model and
criteria (§13i), both starvation arms. Splits: S3 primary, with cells folded
into five blocks (`cell_idx mod 5`) so the 20-fold structure is kept; S0 and
S1 also run and labelled as leaking cell identity, for continuity with §15c.
Group bootstrap over the 160 (trace, cell) groups.

A limit recorded now: more cells multiply (cell, event) instances far faster
than distinct events. The CI comes from resampling groups, so it will tighten
with 160 groups even though the number of distinct load events behind it is
at most 33. Both counts are reported beside every drying number.

### 17h. Execution rules

- **Order:** WS2 → WS1 → WS3a → WS5 volume → WS3b–d → WS5 comparison → WS4.
- **One attempt per run.** A crash is logged and I ask before retrying.
  Runs write results incrementally to `run_logs/posthoc/`; a completed cell
  on disk is not re-fitted.
- **Every run's log ends `EXIT=<code>`**, written from the interpreter's own
  exit status, and that line — not the task notification — is what is read.
- **Every fit writes a convergence record** (`n_iter_`, `max_iter`,
  converged) into the result file it belongs to, so a resumed run cannot
  report fits it did not make. A cell with a non-converged fit is
  unreportable.
- Runs use `-u -W ignore`; no check depends on a warning.
- Seeds: generator `BASE_SEED = 20260810` and all existing seed formulas
  unchanged; companion noise offset `32452843`; post-hoc bootstrap generator
  seeded `20261201`; cap sampling `777 + cell_idx`.
- Convergence status is reported before any number; faults are never
  aggregated across horizons; both starvation arms are always reported.

### 17i. What each workstream can change

| workstream | can change | cannot change |
|---|---|---|
| 1 splits | which split is quoted from now on; a standing caveat on Model A's delta; whether §15e's leak explanation stands | §12 and §15 tables; the quoted Model A range |
| 2 convergence | whether three §12j cells count as verified | the capped run as quoted result |
| 3 Phase 5 | whether the diagnosed defect is shown removable; what a v2 predictor scores on synthetic data | §16b's verdicts; the status of the real recording as burned |
| 4 tuning | whether §15f's stated limit is narrowed | the §15 null |
| 5 drying | whether drying moves from indicative to established (post-hoc) | §15c's drying row |

### 17j. Amendment, 12 November 2026 — approval and four decisions, before the first fit

§17 was approved as written. Four decisions are recorded here before any
model in §17 is fitted; none changes a criterion.

1. **Workstream 4's outer split is the crossed split S3.** It replaces the
   (trace, cell) split named in the original request. The leaky split is
   **not** run for comparison. (Already the plan in §17f; now confirmed.)
2. **Drying keeps the registered unit** — distinct contributing trigger
   events, per §13b. If 40 cells per trace does not reach 30, the comparison
   runs once and is reported as underpowered. (§17g, confirmed.)
3. **Companion runs, not trigger thinning.** (§17e 3b, confirmed.)
4. **Each fault-free companion inherits its faulted twin's (trace, cell)
   group ID**, so it lands on the same side of every split as its twin.
   **New requirement:** this is verified by a label-only check — for S0, S1,
   S2 and S3, every fold must place each companion in the same partition as
   its twin, and no fold may have a twin in training and its companion in
   test or the reverse — and that check must pass, with its log ending
   `EXIT=0`, before any v2 model is fitted.

§17 runs on the existing generator exactly as registered. A new generator
version is being built separately for the detector tool
(`docs/tool_design.md`); it plays no part in §17 and §17's results are not
re-derived on it.

---


## 18. POST-HOC follow-up — results

**Everything in this section is post-hoc** (§17a). It was run as registered
in §17 and amended in §17j, on the existing generator, one attempt per run.
Nothing in §12–§16 is edited; where a result here bears on one of them it is
reported alongside. No number here is real-fault accuracy (§11a).

Logs are in `run_logs/posthoc/`, each ending `EXIT=<code>`; every fit's
convergence record is stored in the result file it belongs to. Subsections
are in the order the runs finished being written up, not the order of §17h.

`fault_generator.py` was not modified: `test_fault_generator.py` and
`test_phase3b_acceptance.py` were run before the first fit
(`run_logs/posthoc/regression_before.log`; 40 of 41, the one failure being
the pre-registered §11h finding) and again at close-out (§18k).

### 18a. Workstream 2 — the convergence-record gap: closed

`run_logs/posthoc/ws2_model_a_refit.log`, `EXIT=0`.

**Convergence: 48 fits recorded, 0 non-converged**, most iterations 1,887
against a cap of 5,000.

**Verdict as registered (§17d): MATCH.** Every re-fitted macro-F1 is within
0.0002 of the resumed value (tolerance 0.001); bulk predictions agree on
100% of samples, spatial on 99.68–99.98%.

| scale | split | resumed delta | re-fitted delta |
|---|---|---|---|
| 0.00 | cell | +0.1133 | +0.1134 |
| 0.00 | trace | +0.1239 | +0.1239 |
| 0.25 | cell | +0.1156 | +0.1154 |
| 0.25 | trace | +0.1263 | +0.1264 |
| 0.50 | cell | +0.1171 | +0.1171 |
| 0.50 | trace | +0.1281 | +0.1280 |

So the three §12j cells that had no convergence record are now verified, and
the uncapped run's convergence is established for all 128 fits (80 asserted
in §12j, 48 here). **This changes no quoted number**: the capped run remains
the quoted Model A result and "+0.07 to +0.12 depending on estimator" stands
as written. Both of those splits leak cell identity (§17b), which §18f
addresses.

### 18b. Workstream 1 — Model B under the corrected splits

`run_logs/posthoc/ws1_model_b_v1.log`, `EXIT=0`. Same data, features, model
and criteria as §15; only the split differs. S0 and S1 are §15's own
predictions, read from disk; their intervals differ from §15's in the third
decimal because the bootstrap generator is seeded differently here.

**Convergence: 600 fits recorded, 0 non-converged** (spatial at most 1,201
iterations, bulk 121, cap 20,000).

**Criterion 3 — SPATIAL DOES NOT HELP — fires for every fault, both
starvation arms, under all four splits.** §15's null survives the corrected
split, and gets stronger.

F1, directional starvation arm (no_axis in the log; same picture):

| fault (horizon) | split | spatial | bulk | spatial − bulk [95% CI] |
|---|---|---|---|---|
| starvation (1 s) | S0 original | 0.829 | 0.900 | −0.072 [−0.174, +0.006] |
| | S1 original | 0.865 | 0.900 | −0.035 [−0.097, +0.021] |
| | S2 cell held out | 0.683 | 0.904 | −0.222 [−0.385, −0.087] |
| | **S3 cell and trace held out** | **0.721** | **0.893** | **−0.171 [−0.323, −0.048]** |
| flooding (2 s) | S0 original | 0.870 | 0.947 | −0.077 [−0.117, −0.037] |
| | S1 original | 0.880 | 0.945 | −0.065 [−0.098, −0.032] |
| | S2 cell held out | 0.619 | 0.947 | −0.328 [−0.465, −0.157] |
| | **S3 cell and trace held out** | **0.656** | **0.945** | **−0.289 [−0.426, −0.114]** |
| drying (20 s, indicative) | S0 original | 0.905 | 0.936 | −0.031 [−0.070, −0.004] |
| | S1 original | 0.911 | 0.932 | −0.020 [−0.045, −0.002] |
| | S2 cell held out | 0.790 | 0.943 | −0.153 [−0.296, −0.060] |
| | **S3 cell and trace held out** | **0.865** | **0.947** | **−0.082 [−0.184, −0.017]** |

Under S3 with the no_axis arm: starvation −0.376 [−0.542, −0.137], flooding
−0.254 [−0.371, −0.103], drying −0.070 [−0.173, −0.009].

**What holding out the cell does, plainly.** The bulk arm does not move
(0.89–0.95 whichever split). **The spatial arm loses 0.10 to 0.25 F1 the
moment the test cell is one it has not seen** (S0→S2, S1→S3), and holding out
the trace as well changes little after that. So **§15's spatial scores were
inflated by cell identity** — the spatial arm had seen each test cell's
template under other traces — and the spatial deficit §15 reported was an
*underestimate*. This is the over-parameterisation §13l and §15e named,
measured: 652 features fitted to 4 training cells.

The horizon is still achieved in both arms for every fault (J lower bounds
+0.50 to +0.87), so §15's horizon finding stands. Binary
healthy-versus-starvation is 0.992–0.999 in both arms under S3.

**The expectation in §17c ("criterion 3 fires again") held.** The size of the
spatial arm's drop I did not predict.

### 18c. The starvation-control alarm — §15e's explanation is falsified

Spatial minus bulk recall on blind positives of the 30 s starvation control
(the registered artifact check, §13i criterion 4 as amended):

| split | cell held out | trace held out | dBlind 95% CI, directional / no_axis | alarm |
|---|---|---|---|---|
| S0 | no | no | [−0.219, +0.090] / [−0.223, +0.109] | no |
| S1 | no | **yes** | [+0.142, +0.296] / [+0.151, +0.295] | **yes** |
| S2 | **yes** | no | [−0.208, +0.081] / [−0.220, +0.076] | no |
| S3 | **yes** | **yes** | [+0.065, +0.321] / [+0.060, +0.282] | **yes** |

**The alarm fires when the trace is held out and only then, whether or not
the cell is held out.** §17c fixed what that means in advance: *cell identity
is not the explanation and §15e's candidate is falsified.* It is. I proposed
shared cell identity as the leak in §15e on a premise I had not checked
(§17b), and the test built to check it says no.

Per §17c the artifact is now **unexplained and is reported as unexplained**,
and no further split is tried. What the table does show is where it lives:
on an unseen load trace, the bulk arm's recall on blind positives collapses
(0.38 → 0.11) while the spatial arm's does not (0.30–0.33). Both arms are
reading the load trace at a 30 s horizon (§15e); the bulk arm, which has only
load and mean current, appears to learn trace-specific timing that does not
transfer, and the spatial arm's noisier, higher-false-positive predictions
(FPR 0.19–0.23 against 0.07) degrade less. That is a description, not a
tested explanation. The control's spatial delta remains unreportable as an
advantage — and is negative in any case.

The flooding control alarms under all four splits with the opposite sign
(bulk better on blind positives, e.g. S3 [−0.655, −0.280]), as in §15e.

### 18d. Workstream 3a — the stored Phase 5 models, holds versus driving

`run_logs/posthoc/ws3a_phase5_split.log`, `EXIT=0`. No refitting: the 16
stored models through the stored prediction path. **Descriptive. §16b's
verdicts stand**; the 5% line against the dynamic-only rate is not a
re-adjudication.

False-alarm rate per real configuration (Normal / Inverse / Inverse_Air /
Inverse_H₂), directional arm; no_axis in the log:

| model | input | conditioning holds | dynamic segments (driving) |
|---|---|---|---|
| A snapshot | spatial | 1.000 ×4 | 0.933 0.942 0.941 0.818 |
| A snapshot | bulk | 1.000 ×4 | 0.501 0.631 0.437 0.477 |
| B starvation | spatial | 0.000 ×4 | 0.070 0.034 0.064 0.111 |
| B starvation | bulk | 0.000 ×4 | 0.249 0.040 0.249 0.294 |
| B flooding | spatial | 0.108 0.000 0.000 0.000 | 0.444 0.145 0.409 0.370 |
| B flooding | bulk | 0.000 ×4 | 0.292 0.149 0.329 0.363 |
| B drying | spatial | 1.000 ×4 | 0.941 0.859 0.972 0.844 |
| B drying | bulk | 1.000 ×4 | 0.693 0.823 0.658 0.641 |

**No stored model is quiet while the cell is actually being driven.** None
has a dynamic-segment rate at or below 5% on all four configurations; the
best (Model B starvation, spatial) is 3–11%. The two §16b passes were passes
on dilution, as §16e said: exactly zero on the holds, which are 78% of the
denominator. And the models that fail do so in two different ways — Model A
and the drying predictor alarm on **every** hold sample, the flooding and
starvation predictors on **none** — which is the load-regime rule of §16d
seen from the other side.

### 18e. Workstream 3b–d — companions: the fix, and what it costs (synthetic)

**The pre-fit check passed** (§17j item 4;
`run_logs/posthoc/check_companions.log`, `EXIT=0`): in every fold of S0–S3,
for all five configurations, each fault-free companion lands on the same side
as its faulted twin.

**Model B on v2 (v1 + companions), S3** —
`run_logs/posthoc/ws3d_model_b_v2.log`, `EXIT=0`. **Convergence: 320 fits, 0
non-converged** (spatial at most 979 iterations, bulk 611).

| fault (horizon) | arm | spatial P / R / F1 | bulk P / R / F1 | spatial − bulk F1 [95% CI] |
|---|---|---|---|---|
| starvation (1 s) | directional | 0.047 / 0.552 / 0.087 | 0.929 / 0.861 / **0.894** | −0.807 [−0.860, −0.675] |
| | no_axis | 0.065 / 0.657 / 0.118 | 0.940 / 0.850 / **0.893** | −0.775 [−0.842, −0.562] |
| flooding (2 s) | directional | 0.064 / 0.731 / 0.117 | 0.438 / 0.918 / **0.593** | −0.476 [−0.559, −0.299] |
| | no_axis | 0.081 / 0.778 / 0.146 | 0.386 / 0.925 / **0.545** | −0.398 [−0.497, −0.169] |
| drying (20 s, 13 events) | directional | 0.656 / 0.604 / 0.629 | 0.636 / 0.941 / **0.759** | −0.130 [−0.382, +0.128] |
| | no_axis | 0.500 / 0.564 / 0.530 | 0.633 / 0.944 / **0.758** | −0.228 [−0.496, +0.059] |

Criterion 3 fires for every fault and arm. Binary healthy-versus-starvation:
both arms 0.994–0.999.

**The paired cost** (§17e-3d: recall on own-fault positives, v2 minus v1,
S3, identical positive windows; `run_logs/posthoc/ws3d_paired.log`):

| fault | arm | input | recall v1 → v2 | difference [95% CI] | as registered |
|---|---|---|---|---|---|
| starvation | directional | bulk | 0.863 → 0.861 | −0.002 [−0.009, +0.007] | no measurable cost |
| | | spatial | 0.712 → 0.552 | −0.159 [−0.256, −0.085] | **costs detection** |
| | no_axis | bulk | 0.846 → 0.850 | +0.004 [+0.000, +0.010] | no measurable cost |
| | | spatial | 0.787 → 0.657 | −0.129 [−0.244, −0.037] | **costs detection** |
| flooding | directional | bulk | 0.967 → 0.918 | −0.049 [−0.062, −0.038] | **costs detection** |
| | | spatial | 0.866 → 0.731 | −0.136 [−0.260, −0.037] | **costs detection** |
| | no_axis | bulk | 0.973 → 0.925 | −0.047 [−0.061, −0.036] | **costs detection** |
| | | spatial | 0.881 → 0.778 | −0.103 [−0.215, −0.010] | **costs detection** |
| drying | directional | bulk | 0.958 → 0.941 | −0.017 [−0.057, +0.000] | no measurable cost |
| | | spatial | 0.804 → 0.604 | −0.199 [−0.384, −0.029] | **costs detection** |
| | no_axis | bulk | 0.958 → 0.944 | −0.014 [−0.047, +0.000] | no measurable cost |
| | | spatial | 0.835 → 0.564 | −0.270 [−0.490, −0.059] | **costs detection** |

**Reading it.** Giving the models healthy operation across the whole load
range costs the **bulk** arm almost nothing in recall — nothing measurable
for starvation and drying, five points for flooding. Its *precision* is
another matter: against five times as many healthy windows, bulk flooding
precision falls from 0.92 to 0.39–0.44, because a healthy low-load dwell and
the first seconds of a flooding episode look alike in two bulk features.
**The spatial arm is hurt badly in both**: it loses 10 to 27 points of recall
and its precision on starvation and flooding collapses to 0.05–0.08. With
652 features and four training cells it cannot separate "fault ahead" from
the healthy variety the companions add.

So on this generator the honest predictor — one that has seen a healthy cell
working hard — is the bulk one: starvation F1 0.89, drying 0.76 (13 events;
indicative), flooding 0.55–0.59.

### 18f. Workstream 5 — drying power: UNDERPOWERED, as feared

Label-only volume first (`run_logs/posthoc/ws5_drying_volume.log`):

| cells per trace | distinct contributing drying events (normal / inv_flow / inv_air / inv_h2) | total |
|---|---|---|
| 5 | 6 / 1 / 5 / 6 | 18 |
| 10 | 6 / 3 / 5 / 6 | 20 |
| 20 | 6 / 4 / 6 / 6 | 22 |
| 40 | 6 / 5 / 6 / 6 | **23** |

**Target (§17g): 30, at least 5 per trace. Reached: 23 by labels, 21 in the
fitted data. NOT MET.** The count was not raised and the target was not
lowered.

**My arithmetic in §17g was wrong.** I estimated 0.98 of events would
contribute at 40 cells, assuming every event had the same per-cell chance.
They do not: 10 of the 33 trigger events never contribute in any of 160
cells. Their windows are blocked by something that does not depend on the
cell's seed. Eight times the cells bought five more events. **The limit is
the four load traces, as §17g said it might be, and more cells from this
generator cannot lift it.** (The new generator's randomised cycles do; that
is a different generator and a different result — `hfc_models/MODEL_CARD.md`.)

The comparison was run once, as registered
(`run_logs/posthoc/ws5_drying_dry40.log`, `EXIT=0`; **112 fits, 0
non-converged**): 13,090 windows, 5,281 positives from 21 distinct events,
414 healthy targets (48 in §15).

| split | arm | spatial F1 | bulk F1 | spatial − bulk [95% CI] |
|---|---|---|---|---|
| S3 | directional | 0.926 | 0.951 | −0.025 [−0.041, −0.013] |
| S3 | no_axis | 0.924 | 0.951 | −0.028 [−0.043, −0.014] |
| S0 *(leaks)* | directional | 0.923 | 0.953 | −0.030 [−0.047, −0.015] |
| S1 *(leaks)* | directional | 0.939 | 0.951 | −0.012 [−0.022, −0.004] |

Criteria as written: **1 (uninformative: bulk F1 above 0.95)** and **3
(spatial does not help)** both fire in every row. **Drying stays INDICATIVE,
NOT ESTABLISHED.** The intervals are tight because 160 groups are resampled;
the number of distinct load events behind them is 21. Both counts are given
for exactly that reason (§17g).

With forty cells the spatial arm's deficit under S3 is far smaller than with
five (−0.03 against −0.08): more training cells is what the spatial arm was
short of, consistent with §18b.
