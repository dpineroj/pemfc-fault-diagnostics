# Tool design — synthetic dataset generator and fault detector

**Status: design, written before the code it describes.** November 2026.
**Everything here is post-hoc** relative to the spatial phase's pre-registered
experiments (§12–§16 of `spatial/synthetic_generator_notes.md`) and runs
alongside the §17 follow-up without changing it. §17 is run on the existing
generator exactly as registered; nothing in this document feeds back into it.

The tool does three things:

1. **Generates** synthetic PEM fuel cell datasets shaped like the ones this
   project has worked with.
2. **Classifies** a cell's current state — healthy, flooding, drying,
   starvation — from any dataset that carries the measurements it needs.
3. **Predicts** an upcoming fault, each fault at its own horizon.

## 0. What this tool is not

**There is no labelled real fault data in this project, and none of the
tool's accuracy figures are verified on real faults.** Every fault it has
ever seen is generated. The one real recording available (Seville; healthy
only) can test a single thing — whether the tool stays quiet on a healthy
cell — and it has already been used as a test set, so any result on it is
**not an independent test**. The README, data cards, model card and CLI
output all say this; none of them may imply otherwise.

Two results from the completed phase shape every decision below:

- **Phase 5 (§16):** models trained on the old generator alarmed on the real
  healthy cell, because that generator made a fault a near-certain
  consequence of the load regime, and because its *bulk* signal was lost mean
  current — which a real cell under current control does not show.
- **§17b:** cells were seeded without a trace term, so no split ever held out
  a cell.

The generator below is built to remove both defects by construction, and the
detector is built so that it does not depend on the generator's healthy
baseline being right.

---

## 1. Generator (`hfc.gen`, version `hfcgen-4.0`)

Additive. `scripts/spatial/fault_generator.py`, the notebook-01 reference
generators and notebook 01 itself are imported read-only and stay
byte-for-byte unchanged; the existing regression tests must pass before and
after.

### 1.1 Signal model

The cell is **current-controlled**, as the FC-DLC protocol is. The load
profile fixes total current; a fault cannot remove current, it can only
**redistribute** it across the plate and **lower the voltage**.

```
load profile        ->  i(t)  mean current density, A/cm2  (= rated * load_ratio)
healthy template    ->  S(x)  radial clamping template, mean 1
fault modulation    ->  M(x; fault, severity)   = 1 at severity 0
current density     ->  J(x,t) = i(t) * normalise( S * M ) * (1 + noise)
voltage             ->  V(t)  = V_pol(i; cell) - dV_transient - dV_fault + noise
temperature         ->  T(x,t) from local heat  q = J * (E_tn - V)
```

- `S(x)` is the **radial clamping template of `generate_healthy_grid_v2`**
  (`_radial_healthy_shape`: quadratic in radius, edge ratio drawn per cell
  from the bracket already documented for `RADIAL_EDGE_RATIO`), plus an
  optional per-cell load-dependence of the edge ratio.
- `M` reuses the reference generator's fault patterns: drying suppresses the
  hot centre, flooding the cool edges, starvation along a gas axis and/or
  where current is highest.
- **The same convexity `severity ** p` enters `M` and `dV_fault`.** Neither
  the spatial channel nor the voltage channel is given an earlier precursor
  by construction. This is §10's question — *does this choice decide the
  comparison before it is measured?* — answered in the design: it would, so
  the two are tied, and the spatial-versus-bulk result on this generator is
  declared assumption-driven wherever it is reported.
- Fault severity trajectories are the reference generator's episode builders,
  called unmodified: starvation a seconds-long bump, flooding a fast envelope
  with accumulate/breakthrough/drain cycles, drying a slow saturating rise
  with recovery.
- **Flooding voltage overshoot:** at each drain (severity falling fast) the
  voltage briefly overshoots; amplitude is an engineering assumption.

### 1.2 Decisions on the fault set

- **Starvation: one class, two generated subtypes.** Oxygen and hydrogen
  starvation are generated differently — the depleted axis follows the air or
  the hydrogen path, and hydrogen starvation drives the voltage toward
  reversal — and the subtype is written to the labels. **The classifier ships
  one `starvation` class.** Reason: which grid axis is the air path and which
  the hydrogen path is undocumented for the Seville cell and the real data
  shows the residual direction is fixture-locked, not flow-locked
  (`real_data_notes.md`). Splitting the class would rest on an axis mapping
  that is an assumption twice over. The labels let anyone with a known
  geometry train the split.
- **No concurrent faults.** Nothing in the repo's literature grounds how two
  fault patterns combine. An episode that would overlap an active one is
  skipped and counted in the manifest. Stated in every data card.
- **Severity levels and recovery** come from the reference episode builders
  (per-episode severity target, decay after the trigger ends).

### 1.3 Healthy operation covers the whole load range

- A trigger event (ramp, low-load dwell, high-load dwell) starts a fault with
  probability `p_trigger < 1`, not always.
- **Every cell also gets a fault-free companion run** over the same load
  profile, in the same group.
- Static profiles induce faults at scheduled times independent of load.

"Working hard" therefore never implies "faulty", at the row level or the
run level.

### 1.4 Load profiles

| family | what it is | needs third-party data |
|---|---|---|
| `fcdlc_real` | the four real FC-DLC load traces, dynamic segments only | yes (`data/processed/`, not redistributed) |
| `fcdlc_random` | new cycles assembled from the FC-DLC's structure — urban blocks (~195 s) and a highway block (~400 s) — with block order, levels, dwell lengths and ramp rates randomised | no |
| `static` | constant-load holds with externally induced faults, after Kim et al.'s lab protocol | no |
| conditioning holds | optional labelled `hold` segments before and between cycles | no |

Trigger events are counted per profile and written to the manifest. New
randomised cycles are how the drying ceiling (33 trigger events in the four
real traces) is lifted.

### 1.5 Flow configurations

The four Seville configurations (`Normal_Flow`, `Inverse_Flow`,
`Inverse_Air_Flow`, `Inverse_Hydrogen_Flow`) are a parameter that sets the
direction of the air and hydrogen axes for starvation. **This is an
engineering assumption**: the real data has one session per configuration and
cannot ground any difference between them.

### 1.6 Messiness (each switchable; default off except sensor noise)

sensor noise · dropouts and NaN gaps · DAQ start-up transient · CDM negative
sign convention · per-segment amps instead of A/cm² · bulk stream on its own
clock (offset, whole-second) · sampling slower than the contract minimum.

### 1.7 Independence, labels, reproducibility

- **Every cell has a unique ID and appears under exactly one load profile**
  unless a config explicitly asks for recurrence. Seeds come from
  `numpy.random.SeedSequence(master_seed, spawn_key=(profile, cell,
  flow_config, companion, stream))`: every design factor is in the key.
- Group IDs exported with every dataset: `profile_id`, `profile_family`,
  `cell_id`, `pair_id` (faulted run + companion), `flow_config`, `split`.
- **`check-leaks`** (label-only) verifies that a split shares no cell, pair or
  profile between its sides.
- Labels per timestep: state, fault type, starvation subtype, severity,
  time since onset, time-to-onset for each fault, segment type (dynamic /
  hold); per episode: onset, threshold crossing, offset, severity target.
- Same config + seed ⇒ byte-identical output. Tested.

### 1.8 Formats

- **Canonical:** a directory with `manifest.json` (SYNTHETIC flag, generator
  version, seed, full config, provenance table, event counts), `index.csv`
  (one row per run with all group IDs), one compressed `runs/<run_id>.npz`
  per run (so the large preset never has to fit in memory), and
  `DATA_CARD.md`.
- **Seville raw layout export:** `CDM_C_*.dat`, `CDM_T_*.dat`, `FC-DLC_*.dat`
  in the block/flat formats the real files use (per-segment amps, negative
  sign, comma decimals, separate bulk clock). Files and folders are named
  `SYNTHETIC`, and a `SYNTHETIC_README.json` sits beside them. Bulk channels
  the generator does not model are **left out**, never filled.
- **Round-trip test:** generate → export → read back with the Seville adapter
  → compare with the canonical arrays.

### 1.9 Bulk channels

Named and unit-matched to the Seville bulk stream where modelled:
`FECHA`, `HORA`, `INTENSIDAD` (A), `V001` (V). Nothing else is emitted — the
pressures, flows, humidities and thermocouples are not modelled.

### 1.10 Presets

- `smoke` — a handful of cells, short runs, coarse grid; seconds to generate.
- `extensive` — hundreds of independent cells, each on its own randomised
  cycle, plus real-trace and static cells held out for testing; chunked;
  size, runtime and contributing events per fault recorded in its data card.
  If a fault cannot reach the §13b power bar (30 contributing events) without
  ungrounded assumptions it is reported as underpowered.

### 1.11 Parameter provenance

Every parameter carries one of three tags, in the table in `hfc/provenance.py`
(printed into every data card): **literature-grounded**, **physics-derived**,
or **engineering assumption with a sensitivity bracket**. Assumptions are
drawn per cell from their bracket, so no result rests on a single
precise-looking value. A test fails if a config field has no provenance row.

Rule 3 applies throughout: **no parameter is fit to, tuned on or selected
with the Seville recording's healthy behaviour.** Its format, channel names,
units, grid layout, operating set-points and load trace are copied; its
CV(J), template parameters, voltage level and autocorrelation are not
targets. The known synthetic-versus-real gaps are reported in the data card
as diagnostics.

---

## 2. Detector (`hfc.detect`)

### 2.1 Data contract

| capability | minimum measurements | units | sampling |
|---|---|---|---|
| state classification, bulk-only | time; stack current *or* mean current density (+ active area); cell voltage | s; A or A/cm²; cm²; V | ≥ 1 Hz (level-only fallback 0.2–1 Hz) |
| state classification, with spatial | the above + current-density grid, any resolution ≥ 3×3 | A/cm² or A per segment | same |
| fault-ahead, starvation (1 s) and flooding (2 s) | as classification | | ≥ 1 Hz |
| fault-ahead, drying (20 s) | as classification | | ≥ 1 Hz |
| healthy calibration | a stretch of the user's own data they assert is healthy, covering the loads to be diagnosed | | same as the data |

Temperature grids are read and carried but **not used by the shipped
models**; the contract does not require them.

### 2.2 Adapters

A dataset is mapped in by a small JSON config: file locations, column names,
units, sign convention, active area, timestamp format. Two adapters ship:
`canonical` (the tool's own format) and `seville` (the existing
`cdm_parser.load_run`, wrapped, not rewritten). A generic `table` adapter
reads CSV-like bulk data from a mapping.

**Unit sanity checks**, because this project has already made this error
once (per-segment amps read as A/cm²):

- if both a grid and a stack current exist, the grid integral must match the
  stack current within tolerance — under the declared units. If it matches
  under the *other* units instead, the adapter says exactly that and refuses.
- cell voltage must be plausible for one cell; current density must be
  plausible for a PEMFC; sign must be consistent.
- Failures are errors, not warnings. The run does not proceed on bad units.

### 2.3 Features — relative to the user's own healthy baseline

Built on what is verified: CV(J) is load-invariant where σ(J) is not.

- **Bulk:** load fraction; voltage residual against the calibrated healthy
  V(i); its slope and rolling spread; load ramp rate.
- **Spatial (optional, any resolution):** CV(J) residual; centre-to-edge
  contrast residual; residual energy against the calibrated healthy template;
  the residual's radial correlation (its sign separates centre loss from edge
  loss); dipole magnitude.

A handful of physically named features, not hundreds of raw segments. That
removes the over-parameterisation that cost the spatial arm in §15 and makes
it impossible to memorise a cell's identity from its template.

### 2.4 Healthy calibration at runtime

The user points the tool at a known-healthy stretch of **their own** cell.
The tool learns that cell's V(i) curve, its healthy template and CV by load,
and the healthy spread of each residual. Features are residuals against that
baseline, scaled by its spread.

This is why the generator's healthy baseline does not need to match any real
cell, and why calibration does not break rule 3: the baseline comes from the
user's cell at runtime, never from training data. In training, each synthetic
cell is calibrated on its own fault-free companion — the same procedure.

Without calibration the tool does not classify. It says so and stops.

### 2.5 Models and what ships

- State: one multinomial logistic regression. Fault-ahead: one binary
  logistic regression per fault, at horizons 1 s / 2 s / 20 s (the horizons
  §15 confirmed), on windows with no confirmed fault.
- Two variants of each: **bulk-only** and **bulk + spatial**.
- Trained on the extensive preset's training cells. **Evaluated on held-out
  cells under load profiles the models never saw** — including, where the
  third-party traces are present, the real FC-DLC cycles, which are never in
  training.
- Metrics with 95% CIs from a bootstrap over cells. Every fit writes a
  convergence record into the model file.
- **Shipping rule, fixed here before any tool model is trained:**
  bulk-only is the default and always ships. The spatial variant ships as an
  option only if, on the held-out set, it is not worse than bulk-only
  (delta CI not wholly below zero) for classification and for every fault's
  prediction. If it does not earn its place it is not shipped, and the model
  card says why. §17's crossed-split results are read alongside.
- Alarm thresholds are set on validation cells drawn from the training side,
  never on test cells, for a target false-positive rate.

### 2.6 Knowing when not to answer

The tool abstains — it does not alarm — when:

- a required channel is missing (falls back to the mode the data supports, or
  abstains);
- sampling is slower than a capability needs (that capability is disabled and
  flagged);
- load is outside the calibrated range or the range the models were trained
  on;
- a feature is outside the training envelope;
- the sample is inside a gap, a dropout or the start-up transient.

### 2.7 Outputs

Per timestep: state, class probabilities, per-fault fault-ahead probability
and alarm with its horizon, input-quality flags, abstain reason. A summary
report: coverage, time in each state, alarms, abstentions and why, the
calibration used, and the standing statement that real-fault accuracy is
unverified.

---

## 3. Package, CLI, tests

```
hfc/            the package (Python API)
python -m hfc generate     --preset smoke|extensive | --config cfg.json
python -m hfc check-leaks  DATASET [--split-file split.json]
python -m hfc train        DATASET --out MODELS
python -m hfc run          --mapping map.json --models MODELS --calibrate ... --out OUT
```

Tests (`tests/`, standard-library `unittest`): data contract, adapters, unit
checks, abstain logic, generator reproducibility, Seville round-trip, leak
checker, end-to-end on the smoke preset, provenance coverage, and the
existing generator regression tests.

Every long run logs to a file ending `EXIT=<code>`.

## 4. Demonstrations and their labels

- Extensive synthetic set, held-out cells and load profiles — *synthetic
  faults; not real-fault accuracy.*
- Seville recording: calibrate on one part, measure false alarms on held-out
  parts — ***not an independent test***.

## 5. What would verify it

Fault-labelled recordings from a real cell under dynamic load, with onset
times, at ≥ 1 Hz, carrying at least stack current and cell voltage — and a
segmented current map if the spatial variant is to be judged. The draft
request in `spatial/data_requests.md` is updated to ask for exactly that.
