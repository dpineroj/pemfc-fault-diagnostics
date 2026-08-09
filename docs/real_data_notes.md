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
  label, an **18×18** tab-delimited grid (A/cm²), blank separator.
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
- **Current-density sign convention is unresolved.** Raw `current_grid`
  values are entirely (or almost entirely) negative in every run checked.
  Cause unconfirmed — sensor convention, calibration offset, or something
  else. **Not inverted or rescaled anywhere** in the loader or either
  notebook; every use that doesn't care about sign applies `np.abs()`
  explicitly, at the point of use, with a caption/comment noting it.
- **CDM "voltage" vs. bulk `V001`–`V007`: unreconciled.** The CDM_C file's
  per-block scalar "voltage" field (~1.9 V) doesn't match the bulk file's
  `V001` channel (~0.5 V) at the same real time, and a single ~50 cm² PEMFC
  cell's OCV shouldn't reach ~1.9 V. Both are loaded as separate, clearly
  labeled fields; neither is assumed to be "the" cell voltage.
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
| raw σ(J) = std(J) | 0.0023 – 0.0461 A/cm² | 0.0338 A/cm² | **33.6%** |
| CV(J) = std(J) / mean(\|J\|) | 0.3775 – 0.4084 (dimensionless) | 0.3810 | **1.0%** |

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

## What this dataset does / doesn't let us claim

**Does let us claim:**
- Realistic magnitude, shape, and load-response benchmarks for a real
  segmented-cell spatial current/temperature map under real dynamic
  loading.
- A real empirical baseline for σ(J) and CV(J) under healthy operation, and
  how sharply each responds to load — grounds for the CV(J) recommendation
  above.
- A concrete real-vs-synthetic sanity check: real Normal_Flow mean σ(J)
  (0.0338 A/cm²) is the same order of magnitude as, but not equal to,
  notebook 01's synthetic NORMAL figures (0.047 / ~0.10 A/cm², themselves
  mutually inconsistent — see above).

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
