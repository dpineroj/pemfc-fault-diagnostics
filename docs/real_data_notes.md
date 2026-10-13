# Real Dataset Notes — PEMFC Parallel-Serpentine (Toharias et al., 2024)

Reference notes for the real dataset integrated in Phase 1 alongside the
synthetic work in `01_spatial_viz_sandbox.ipynb`. Companion to
`scripts/load_real_data.py` (loader) and `02_real_data_exploration.ipynb`
(analysis).

## Citation & provenance

- **Dataset**: Toharias, B., Suárez, C., Iranzo, A., Salva, M., Rosa, F.
  (2024), *Dataset and measurements from a current density sensor during
  experimental testing of dynamic load cycling for a parallel-serpentine
  design of a proton exchange membrane fuel cell*, Data in Brief.
  DOI: [10.12795/11441/153760](https://doi.org/10.12795/11441/153760).
  https://www.sciencedirect.com/science/article/pii/S2352340924003615
- **Methodology / related publication**: C. Suárez, B. Toharias, M. Salva,
  A. Chesalkin, F. Rosa, A. Iranzo, *Experimental dynamic load cycling and
  current density measurements of different inlet/outlet configurations of
  a parallel-serpentine PEMFC*, Energy 283 (2023) 128455.
  https://doi.org/10.1016/j.energy.2023.128455
- **Authors' institution**: University of Seville.
- **License**: CC-BY-NC.
- **Sensor**: S++ CDM (current distribution mapping) sensor — measures
  current density and temperature simultaneously, in situ.
- **Cell under test**: ~50 cm² active area, parallel-serpentine flow
  channels, cross-flow field distribution between anode and cathode.
- **Operating conditions (fixed across all runs)**: pressure 0.5 bar,
  temperature 65 °C, anode and cathode relative humidity 60%, anode
  stoichiometry λ=1.3, cathode stoichiometry λ=2.5.
- **Four gas inlet/outlet configurations**: `Normal_Flow` (original),
  `Inverse_Hydrogen_Flow` (H₂ inlet/outlet reversed), `Inverse_Air_Flow` (air
  inlet/outlet reversed), `Inverse_Flow` (both reversed).
- **Two test types per configuration**: polarization curve (`PC`) and
  dynamic load cycle (`FC-DLC`). FC-DLC follows the JRC (EU Joint Research
  Centre) protocol: cycle segments of 4×195 s + 400 s, repeated three times
  per configuration.

## File structure (as discovered — Task 1 audit)

```
data/raw/PEMFC_Parallel-Serpentine/
  {Normal_Flow, Inverse_Flow, Inverse_Air_Flow, Inverse_Hydrogen_Flow}/
    <prefix>_PC_<date>_.../       CDM_C_*.dat, CDM_T_*.dat, PC_*.dat
    <prefix>_FC-DLC_<date>_.../   CDM_C_*.dat, CDM_T_*.dat, FC-DLC_*.dat
```

**Filenames contain a literal space** before the final segment (e.g.
`..._a13RH60_ c25RH60air.dat`) in every run folder we checked — always
`glob()`, never hardcode a filename.

All three `.dat` files per run: UTF-8, tab-delimited, **comma decimal**
(European/Excel locale — `"67,7509"` = 67.7509).

- **`CDM_C_*.dat`** (current density, ~54 MB/run): repeating per-timestep
  **blocks**, not a flat table — 26 lines/block: human timestamp, LabVIEW-
  epoch timestamp, blank, `"voltage"` label, scalar value, blank, `"current"`
  label, an **18×18** tab-delimited grid, blank separator. **Raw grid values
  are per-segment current in Amps, NOT A/cm² density** — see "Units fix"
  below. `load_real_data.load_run()` converts to true A/cm² before
  returning; `parse_cdm_blocks()` itself returns the file's raw,
  unconverted per-segment-amp values.
- **`CDM_T_*.dat`** (temperature, ~14 MB/run): same block style, 14
  lines/block, a **9×9** grid (°C), no scalar sub-block.
- **`FC-DLC_*.dat` / `PC_*.dat`** (bulk/system channels, ~3.5 MB/run):
  conventional flat table, 1 header row + 1 row/second. Columns: `FECHA`,
  `HORA`, 7× `PT0xx` (pressure), `BP001-2` (back-pressure), 5× `FT0xx`
  (flow), `HT001-2` (humidity), `V001-7` (voltage channels), 12×
  thermocouples (`TT0xx`), `INTENSIDAD` (the real load-current
  setpoint/measurement).
- CDM_C and CDM_T are emitted in lockstep — identical timestamps, identical
  block counts (verified: 18,008 blocks each for the audited run) — safe to
  join by row index. The bulk file runs on its **own clock**: different
  start time (CDM starts ~13 s earlier), different end time (CDM ends ~18 s
  later), whole-second resolution only (vs. CDM's millisecond resolution),
  and a different row count (17,951 vs. CDM's 18,008 for the audited run).
  **Must be joined by nearest timestamp** (`pd.merge_asof`), never by row
  index.
- Nominal sampling rate settles to a very stable ~1.000 s/sample (stdev
  ~0.001–0.002 s) from roughly the 25% mark onward, but **the first
  ~150–300 samples of every CDM file are a DAQ warm-up transient**:
  irregular inter-sample timing (0.46–1.07 s) and, in the very first block,
  anomalous near-zero/alternating current values. `load_run()` excludes
  this via a `warmup_seconds` parameter (default 300 s, configurable, not
  hardcoded to a sample count).
- Structural quality is otherwise clean: a full scan of all 18,008 CDM
  blocks and all 17,951 bulk rows (for the audited run) found zero
  malformed rows, zero wrong-width rows, zero unparseable values.

## Known limitations

- **No fault labels.** This dataset is **healthy operation** under four gas
  inlet/outlet configurations plus dynamic load cycling — it does **not**
  contain flooding, drying, or oxygen-starvation conditions. It is a
  different kind of data from notebook 01's synthetic generator, which
  models four *fault* states. Any comparison between the two is a
  "real-healthy vs. synthetic-assumed-healthy" comparison only.
- **Not the same physical cell as the synthetic model.** Notebook 01 models
  a 25 cm², 4×4 (16-segment) PCB-style cell. This real dataset is a
  physically different 50 cm², 18×18-current/9×9-temperature parallel-
  serpentine cell. Different active area, channel geometry, and sensor
  resolution — magnitude comparisons across the two are same-order-of-
  magnitude sanity checks, not validated apples-to-apples measurements.
- **Current-density sign convention — RESOLVED (Task 3 review), not an open
  question.** Raw `current_grid` values are 100% negative in every run
  checked (`frac negative = 1.0000`), while the bulk file's independent
  `INTENSIDAD` channel is 100% positive. This is a **uniform CDM sign
  convention**, not a parse defect or an ambiguity: integrating the (now
  unit-corrected, see below) grid magnitude per timestep and comparing
  against `INTENSIDAD` gives a stable ratio of 1.0053 (n=17,663,
  Normal_Flow FC-DLC; reproduced on Inverse_Hydrogen_Flow at 1.0055 — see
  `test_load_real_data.py`), confirming the two track each other in
  magnitude with a consistent, opposite sign. What remains genuinely
  unresolved is only the *physical meaning* of the sign (e.g. whether it
  encodes current direction relative to some reference) — not whether it's
  a defect. **Not inverted or rescaled anywhere** in the loader or either
  notebook; every use that doesn't care about sign applies `np.abs()`
  explicitly, at the point of use, with a caption/comment noting it.
- **Units fix (Task 3 review, formerly an unstated assumption, not a
  documented limitation until this bug was found).** The raw `CDM_C` grid
  values were treated as already being A/cm² density throughout this
  project (this loader, this doc, notebook 02's figures) until directly
  checked against the bulk file's `INTENSIDAD` channel. They are actually
  **per-segment current in Amps**. Confirmed by integrating the raw grid
  two ways and comparing to `INTENSIDAD`: summing raw values directly (no
  area factor) gives ratio≈1.005 (correct); summing and then multiplying by
  segment area (the old assumption) gives ratio≈0.155 — off by ~6.45×, the
  same factor as `1/SEGMENT_AREA_CM2`, confirming a double application of
  the area factor under the old assumption. `load_real_data.load_run()` now
  divides by `SEGMENT_AREA_CM2` (50 cm² / 324 segments) before returning
  `current_grid`, so it is true A/cm² again. **This was invisible under the
  original Task 1 raw-σ(J) real-vs-synthetic comparison** (below) because
  that comparison only checked σ(J), never mean(J) — a σ(J) that happened
  to be "same order of magnitude" masked a ~7.8x mean(J) gap that CV(J)
  (postdating that comparison) later exposed. See
  `docs/synthetic_generator_notes.md` for the full investigation and the
  downstream implication.
- **CDM "voltage" vs. bulk `V001`–`V007`: unreconciled.** The CDM_C file's
  per-block scalar "voltage" field (~1.9 V) doesn't match the bulk file's
  `V001` channel (~0.5 V) at the same real time, and a single ~50 cm² PEMFC
  cell's OCV shouldn't reach ~1.9 V. Both are loaded as separate, clearly
  labeled fields; neither is assumed to be "the" cell voltage.
  **`V003`–`V007` are dead channels** (found during the flow-config
  discrimination premise check, all four configs, `FC-DLC`): `V003`–`V005`
  are exactly 0 throughout, `V006`/`V007` are ~0.001 or exactly 0 with no
  real variation. Only `V001` (mean ~0.52–0.55 V across configs) and `V002`
  (mean ~-0.17 to -0.18 V, opposite sign) carry actual signal. Not
  previously written down anywhere in the repo before this note.
- **A pre-existing internal inconsistency was found in notebook 01 itself**
  (discovered during the real-vs-synthetic comparison in
  `02_real_data_exploration.ipynb`; **not fixed** — notebook 01 is out of
  scope to modify). Cell 8's actual generated synthetic NORMAL grid has
  `J_std = 0.047` A/cm², but cell 12's simulated time-series figure uses a
  hardcoded `nui_target = 0.10` for the NORMAL phase — a ~2× mismatch. The
  cell 12 docstring calls this "derived from map std values," but it isn't;
  it's an independently chosen illustrative constant. Flagged for
  awareness only.

## Spatial-nonuniformity feature: raw σ(J) vs. normalized CV(J)

`02_real_data_exploration.ipynb`'s closing analysis (Normal_Flow FC-DLC,
n=17,681 timesteps) tested whether notebook 01's raw-σ(J) formula
(`current_grid.std()`, no normalization) is a good nonuniformity feature, or
whether a load-normalized version is better, given that raw σ(J) was
observed tracking `INTENSIDAD` almost exactly over the FC-DLC cycle:

| metric | range across the run | mean | CoV of the metric itself |
|---|---|---|---|
| raw σ(J) = std(J) | 0.0149 – 0.2988 A/cm² | 0.2192 A/cm² | **33.6%** |
| CV(J) = std(J) / mean(\|J\|) | 0.3775 – 0.4084 (dimensionless) | 0.3810 | **1.0%** |

(raw σ(J) figures corrected post-Task-3-review to reflect the units fix
above — CV(J) is a ratio and was, and remains, completely unaffected by
that fix, since it's scale-invariant. The relationship between the two
rows — CV(J) 34x more load-invariant than raw σ(J) — is unchanged by the
units correction; only the raw σ(J) row's absolute A/cm² figures moved.

**Note for anyone reading an older commit of this file**: the σ(J) values
above are POST-correction. Any commit prior to the units fix in
`scripts/load_real_data.py` states σ(J) (and any other absolute, non-ratio
current-density figure) too low by a factor of `1/SEGMENT_AREA_CM2`
(≈6.48×) — those older figures reflect the raw per-segment-Amp values being
misread as A/cm² density, not a different measurement. CV(J) values are
unaffected at every commit, old or new.)

CV(J) is **~34× more load-invariant** than raw σ(J) on this run, with no
instability at low load (mean CV(J) at INTENSIDAD < 5 A: 0.387 vs. 0.381 at
higher load — a division-by-small-mean blow-up was checked for and not
found).

**Recommendation: use CV(J), not raw σ(J), as the nonuniformity feature
going forward.** Raw σ(J) — notebook 01's formula — is largely a proxy for
how hard the cell is being driven, not a signal specific to spatial
(un)health; a fault detector built on it would need to separately control
for load. CV(J) factors that out almost entirely here.

**Scope of this recommendation**: verified on one healthy configuration
only (Normal_Flow). There is no fault-labeled data in this dataset (see
limitations above), so this shows CV(J) is more load-invariant under
healthy operation — not that it still discriminates fault states as well
as, or better than, raw σ(J) would. That remains open until fault-labeled
data is available.

## Spatial structure: a radial template, reproducible across all four sessions

Source: `scripts/analyze_real_structure.py` (re-runnable; re-derives every
figure below from the corrected loader).

Normalizing each timestep by its own spatial mean (which removes load level
and leaves the *shape*), then separating the time-averaged persistent
template from the per-timestep residual, gives the same result in every one
of the four flow configurations:

| configuration | static std | temporal std | static share | corr vs. distance-from-centre | quadratic-in-radius R² | CV(J) |
|---|---|---|---|---|---|---|
| Normal_Flow | 0.3802 | 0.0253 | 99.56% | +0.9360 | 0.9057 | 0.3810 |
| Inverse_Flow | 0.3265 | 0.0067 | 99.96% | +0.9331 | 0.9149 | 0.3266 |
| Inverse_Air_Flow | 0.3800 | 0.0298 | 99.39% | +0.9344 | 0.9018 | 0.3812 |
| Inverse_Hydrogen_Flow | 0.3247 | 0.0068 | 99.96% | +0.9303 | 0.9100 | 0.3247 |

Row- and column-index correlations are near zero in all four (|r| ≤ 0.104),
so this is **not** an inlet-to-outlet flow-path gradient. It is a radially
symmetric "bathtub": lowest in the centre of the active area, highest toward
the edges and corners.

> **Do not over-read the near-zero row/column correlations.** They describe
> the *dominant* structure, which is radial, and they are correct for that.
> They do **not** mean the template has no directional content: once the
> radial component is removed, residual row/column/diagonal correlations rise
> to 0.11–0.43. Nor do they rule out a flow-path mechanism on their own, since
> this cell is cross-flow and combined depletion in a cross-flow cell is
> *diagonal*, which correlates with neither row nor column. That question is
> measured and resolved separately in
> `docs/synthetic_generator_notes.md` §11d (the residual direction turns out
> to be fixture-locked, not flow-locked), via
> `scripts/measure_residual_directionality.py`.

**Cross-session template correlation** (flattened static templates,
pairwise):

| pair | r |
|---|---|
| Normal_Flow vs. Inverse_Flow | +0.9922 |
| Normal_Flow vs. Inverse_Air_Flow | +0.9930 |
| Normal_Flow vs. Inverse_Hydrogen_Flow | +0.9919 |
| Inverse_Flow vs. Inverse_Air_Flow | +0.9768 |
| Inverse_Flow vs. Inverse_Hydrogen_Flow | +0.9994 |
| Inverse_Air_Flow vs. Inverse_Hydrogen_Flow | +0.9755 |

**Finding: the radial structure is a property of the cell and its fixture,
not of any individual run or flow configuration.** Four independent recording
sessions, on different days, under four different gas inlet/outlet
configurations, produce essentially the same spatial template (r ≥ 0.976 in
every pair, ≥ 0.992 against Normal_Flow). Physically this is consistent with
a compression/contact-resistance mechanism — clamping pressure falling off
from a bolt-clamped periphery toward the centre of the active area — which is
a fixed property of how the cell is assembled and would not be expected to
change with gas routing. It is not consistent with a reactant-depletion
gradient, which would have to track the flow path and therefore differ
between configurations that reverse it.

**Why this claim survives the n=1-per-configuration confound.** Elsewhere in
this document, the fact that each configuration was recorded in exactly one
session — with its own pinned steady-state current setpoint — is treated as
disqualifying, and it rules out any claim that *distinguishes* configurations
from one another (see the configuration-discrimination thread in the
README's Open threads). That confound does not apply here, because this is
the opposite kind of claim: it is about what is **common** across the four
sessions, not what separates them. Session identity being confounded with
configuration identity would, if anything, work *against* finding a shared
template — four independent sessions agreeing to r ≥ 0.976 despite differing
in both session and configuration is stronger evidence for a fixture-level
cause, not weaker. The claim made here is only "this structure is present and
near-identical in all four", which is directly observed.

**Caveats.** All four sessions are the same physical cell in the same test
fixture, so "property of the cell and fixture" is as far as this generalizes
— it says nothing about whether other cells, or this cell reassembled with
different clamping, would show the same template. No independent measurement
of the actual compression distribution exists in this dataset, so the
compression mechanism is an interpretation consistent with the observation,
not a measured cause.

### Does the template vary between sessions? Two states, and an unbreakable confound

The templates are near-identical in *shape* (r ≥ 0.976 above), but that does
not mean they are identical in every respect. Fitting an identifiable
three-parameter radial profile to each session
(`docs/synthetic_generator_notes.md` §6c) separates what varies from what
doesn't:

| configuration | session date | H₂ routing | centre depth `c` | transition radius `r0` | sharpness `w` |
|---|---|---|---|---|---|
| Normal_Flow | 2022-02-09 | normal | 0.1390 | 0.4478 | 0.1123 |
| Inverse_Air_Flow | 2022-02-11 | normal | 0.1403 | 0.4482 | 0.1131 |
| Inverse_Flow | 2022-02-17 | reversed | 0.2000 | 0.4273 | 0.1146 |
| Inverse_Hydrogen_Flow | 2022-02-18 | reversed | 0.2055 | 0.4273 | 0.1142 |

**The transition geometry is fixed; the centre depth is not.** `r0` varies by
1.05× (CV 2.4%) and `w` by 1.02× (CV 0.8%) across all four — essentially
constant. `c` varies by **1.48× (CV 18.5%)**, in two tight clusters: ≈0.139
(within-cluster spread 0.9%) and ≈0.203 (2.8%).

**The clustering is perfectly confounded between two explanations.** The two
clusters are {Feb 9, Feb 11} and {Feb 17, Feb 18} — which is *simultaneously*
a split by H₂ routing (normal vs. reversed) and a split by session block (two
sittings six days apart). Both explanations fit the data exactly and nothing
in this dataset can separate them:

- *H₂ routing*: reversing hydrogen flow changes how deep the central current
  depression goes while leaving the frame-driven transition geometry alone.
- *Assembly/session*: the cell was plausibly disassembled, re-clamped, or the
  rig otherwise adjusted between the two sittings, changing centre
  compression while the bolt pattern (hence `r0`, `w`) stayed put.

This is the n=1-per-configuration confound (above) reappearing at a finer
level. It cannot be resolved without replicate runs of the same configuration
across sittings, which this dataset does not contain.

**The specific missing experiment.** Naming it so a later reader sees it was
identified rather than overlooked. The dataset contains:

- no repeat of any routing configuration across the two session blocks, and
- no session block containing both routings.

Either one alone would break the confound. A second `Normal_Flow` run in the
Feb 17–18 block would show whether centre depth tracks the sitting (torque /
assembly) or the routing: if that run's `c` came out near 0.20 it is the
sitting; near 0.139 it is the routing. Equivalently, an `Inverse_Flow` run in
the Feb 9–11 block answers the same question from the other side. Neither
exists, and no amount of reanalysis of the four runs we have can substitute
for one — the two candidate causes are collinear across every sample in the
dataset.

**Status of the "clamping torque varies by session" hypothesis: NOT rejected
— the earlier rejection was an artifact and is withdrawn.** An earlier pass
fitted a two-parameter power form whose depth parameter turned out to be
unidentifiable (degenerate across four orders of magnitude,
`docs/synthetic_generator_notes.md` §6a). That fit showed *zero* spread in
depth across the four sessions, which was read as evidence against
session-to-session variation. It was nothing of the kind: an unconstrainable
parameter shows no spread whether or not the underlying quantity varies.
Re-fitted with an identifiable form, the depth parameter *does* vary, by
1.48×. Torque/assembly variation remains a live candidate explanation — tied
with H₂ routing, and not separable from it here.

This entry is the empirical grounding cited by
`docs/synthetic_generator_notes.md` §5b/§5f for modelling the synthetic
healthy baseline as a radial template rather than a flow-path gradient.

## What this dataset does / doesn't let us claim

**Does let us claim:**
- Realistic magnitude, shape, and load-response benchmarks for a real
  segmented-cell spatial current/temperature map under real dynamic
  loading.
- A real empirical baseline for σ(J) and CV(J) under healthy operation, and
  how sharply each responds to load — grounds for the CV(J) recommendation
  above.
- A concrete real-vs-synthetic sanity check: real Normal_Flow mean σ(J)
  (0.2192 A/cm², post-units-fix) is the same order of magnitude as, but
  **higher than**, notebook 01's synthetic NORMAL figures (0.047 / ~0.10
  A/cm², themselves mutually inconsistent — see above). This direction
  flipped from the pre-fix reading (which had real σ(J) *lower* than
  synthetic, at the old, wrong 0.0338 A/cm² value) — see
  `docs/synthetic_generator_notes.md` for the full corrected comparison,
  including mean(J), which the original Task 1 comparison never checked.

**Does not let us claim:**
- Anything about fault discrimination — there is no flooding/drying/
  starvation ground truth in this dataset.
- That notebook 01's synthetic fault-state assumptions are validated *or*
  invalidated by this data — different cell, different sensor resolution,
  and no fault data to compare against regardless.
- That the CV(J) load-invariance finding generalizes beyond one healthy
  configuration on this specific cell, or would hold under actual fault
  conditions.
- That the CDM "voltage" scalar and the bulk file's `V001`–`V007` channels
  are interchangeable, or that either is confirmed to be true cell voltage.
- Any physical interpretation of `current_grid`'s sign (e.g., which
  direction is "positive") — that remains unresolved pending the source
  paper's methods section.
