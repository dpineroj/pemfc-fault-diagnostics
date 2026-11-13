# PEMFC Fault Diagnostics

This project asks whether a spatially resolved sensor grid — current density
over 324 segments and temperature over 81, mapped across a single PEM fuel
cell — **classifies and predicts faults** (flooding, membrane drying,
starvation) better than the **bulk** signals a production stack actually
has: stack current and cell voltage. It ends with three findings:

- **The earlier spatial advantage came from a leak, and it reverses.** The
  snapshot classifier (Model A) seemed to gain from the grid, but every test
  cell had also been in training. With the cell held out, the spatial model
  loses to two bulk numbers by −0.18 to −0.22 macro-F1
  ([final_results §10a](docs/spatial/final_results.md)).
- **The prediction result got stronger.** The fault-ahead model (Model B)
  already showed no spatial gain; on held-out cells the spatial arm is
  clearly *worse*, and that survives matched tuning
  ([notes §18b, §18h](docs/spatial/synthetic_generator_notes.md)).
- **Simulation cannot settle spatial versus bulk.** The project's two
  generators give opposite answers, and the disagreement traces to three
  modelling assumptions: whether a fault removes current or redistributes it,
  how large a fault's voltage signature is, and whether per-segment sensor
  noise is independent ([final_results §10d](docs/spatial/final_results.md)).
  Real spatially resolved fault data can settle it; none exists here.

The project also produced **`hfc`**, a tool that generates synthetic fuel
cell datasets, classifies a cell's state, and predicts faults ahead of time.

> **Claim ceiling.** This is a **simulation study**. Every fault any model
> here has seen is generated. The only real data is one **healthy-only**
> recording (one cell, four sessions), used to test false alarms — and it has
> been used that way more than once, so results on it are labelled **not an
> independent test**. **Accuracy on real faults is unverified.**

---

## Key results

| question | result | data | source |
|---|---|---|---|
| Snapshot classifier: spatial − bulk macro-F1, **test cell held out** | **−0.18 to −0.22**, every 95% CI below zero (e.g. −0.192 [−0.307, −0.085]); random forest −0.30 to −0.38. Overturns the earlier ~~+0.07 to +0.12~~ (**struck November 2026**: measured with every test cell also in training) | synthetic, original generator | [final_results §10a](docs/spatial/final_results.md), [notes §18i](docs/spatial/synthetic_generator_notes.md) |
| Fault prediction: spatial − bulk F1, cell **and** load trace held out | starvation, 1 s ahead: −0.171 [−0.323, −0.048]; flooding, 2 s: −0.289 [−0.426, −0.114]; drying, 20 s: −0.082 [−0.184, −0.017] | synthetic, original generator | [notes §18b](docs/spatial/synthetic_generator_notes.md) |
| Does tuning both arms equally rescue the grid? *(exploratory)* | No: spatial still worse in all six configurations, every CI wholly below zero | synthetic | [notes §18h](docs/spatial/synthetic_generator_notes.md) |
| Is drying adequately powered? | No: 23 contributing events against a target of 30 fixed in advance; drying stays *indicative* | synthetic | [notes §18f](docs/spatial/synthetic_generator_notes.md) |
| Original models on the real healthy recording | 14 of 16 fail the pre-registered false-alarm test | real, healthy only | [notes §18l](docs/spatial/synthetic_generator_notes.md) |
| Same, retrained with fault-free "companion" runs | 8 of 16 pass; none stays at or below 5% while the cell is being driven — **not an independent test** | real, healthy only | [notes §18g](docs/spatial/synthetic_generator_notes.md) |
| `hfc` bulk default, held-out cells and unseen load profiles | state macro-F1 0.755 [0.730, 0.776]; false alarms on fault-free runs 0.35% [0.17, 0.59] | synthetic, new generator | [model card](hfc_models/MODEL_CARD.md), [metrics](hfc_models/metrics_heldout.json) |
| `hfc` spatial option, same cells | state macro-F1 0.989 [0.982, 0.994]. **Assumption-driven**: it follows from the new generator's independent per-segment noise and its assumed voltage signatures, and is **not** evidence about real cells. The original generator gives the opposite answer. | synthetic, new generator | [model card](hfc_models/MODEL_CARD.md) |
| `hfc` bulk default on the real recording, calibrated within the session | false alarms 1.1–2.7% of samples; 0% on steady load, **5.7–13.7% while driving** — **not an independent test** | real, healthy only | [model card](hfc_models/MODEL_CARD.md) |
| Same, calibrated on a *different* session | false alarms up to 73% — **not an independent test** | real, healthy only | [model card](hfc_models/MODEL_CARD.md) |

---

## The tool: `hfc`

`hfc` is a Python package with a command-line interface. It does three things:

1. **Generates** synthetic datasets shaped like the Seville recording:
   current-density and temperature grids at any resolution, Seville-named
   bulk channels (`INTENSIDAD`, `V001`), per-sample fault labels, and an
   optional export in Seville's raw file layout. Every generated file carries
   a `SYNTHETIC` flag, the generator version, the seed and the full config.
2. **Classifies** a cell's current state: healthy, starvation, flooding or
   drying.
3. **Predicts** a fault before it is confirmed, each at its own horizon:
   starvation 1 s, flooding 2 s, drying 20 s.

It calibrates on a known-healthy stretch of **your own** recording and judges
everything else as a residual from that baseline. When the input is outside
what it can judge, it **abstains** rather than raising an alarm. Design:
[`docs/tool_design.md`](docs/tool_design.md).

### What it needs

From the data contract in [`docs/tool_design.md`](docs/tool_design.md) §2.1:

| capability | minimum measurements | units | sampling |
|---|---|---|---|
| state classification, **bulk-only (default)** | time; stack current (or mean current density plus active area); cell voltage | s; A or A/cm², cm²; V | ≥ 1 Hz (a level-only classifier runs down to 0.2 Hz) |
| fault-ahead prediction | the same | | ≥ 1 Hz |
| spatial mode (opt-in) | the above plus a current-density grid, any resolution ≥ 3×3 | A/cm² or A per segment | the same |
| healthy calibration | a stretch of the **same recording** you know is healthy, covering the loads you want diagnosed (at least 120 usable samples) | | the same |

Temperature grids are read but not used by the shipped models.

### Quick start

Requires Python 3.12 (other 3.x likely work; 3.12 is what was tested).

```bash
git clone https://github.com/dpineroj/pemfc-fault-diagnostics.git
cd pemfc-fault-diagnostics
python -m venv hfc_env
# Windows: hfc_env\Scripts\activate      macOS/Linux: source hfc_env/bin/activate
pip install -r requirements.txt

# 1. Generate a small SYNTHETIC dataset (12 cells, under a minute) and check
#    that its train and test sides share no cell, run pair or load profile.
python -m hfc generate --preset smoke --out hfc_data/smoke
python -m hfc check-leaks hfc_data/smoke

# 2. Run the shipped models on one generated test run, calibrating on its
#    first 400 s (fault-free in this run). Bulk-only is the default.
python -m hfc run --mapping examples/mapping_canonical.json --models hfc_models \
    --calibrate-range 0 400 --out hfc_data/out

# 3. Spatial mode is opt-in and needs a current-density grid in the data.
python -m hfc run --mapping examples/mapping_canonical.json --models hfc_models \
    --calibrate-range 0 400 --mode spatial --out hfc_data/out_spatial
```

For your own data: copy [`examples/mapping_table.json`](examples/mapping_table.json),
point it at your file, and set `--calibrate-range START_S END_S` to a stretch
you know is healthy. To calibrate on a separate healthy recording instead,
use `--calibrate-mapping healthy.json`. Without a calibration the tool
refuses to classify.

Other commands: `python -m hfc generate --preset extensive` (320 cells, about
2.5 minutes and 1.8 GB), `python -m hfc train`, `python -m hfc evaluate`,
`python -m hfc export-seville`. Tests: `python -m unittest discover -s tests`.

### Writing a mapping file

A mapping is a small JSON file that says where the data is and what its
columns and units mean. For a delimited bulk file:

```json
{
  "adapter": "table",
  "path": "my_recording.csv",
  "sep": ",",
  "decimal": ".",
  "time":    {"column": "time_s",          "unit": "s"},
  "current": {"column": "stack_current_A", "unit": "A"},
  "voltage": {"column": "cell_voltage_V",  "unit": "V"},
  "active_area_cm2": 50.0,
  "n_cells": 1
}
```

- `time.unit`: `s`, `ms`, `min` or `h`. Or give `time.format` (a `strptime`
  format); `column` may then be a list of columns to join, such as date and
  time.
- `current.unit`: `A` (needs `active_area_cm2`), `A/cm2` or `mA/cm2`.
- `voltage.unit`: `V` or `mV`. If the column is a stack voltage, set
  `n_cells`.
- Other adapters: `seville` for the Seville raw layout
  ([example](examples/mapping_seville.json)) and `canonical` for datasets
  written by `hfc generate` ([example](examples/mapping_canonical.json)). See
  [`examples/README.md`](examples/README.md).

**Unit checks stop the run; they are not warnings.** A current grid that only
matches the stack current when read as per-segment amps, a voltage that looks
like millivolts or a whole-stack voltage, or a sign convention that
contradicts the mapping are all errors. This project made the first of these
mistakes once (see [errors caught](#errors-caught-along-the-way)).

### What it outputs

For each recording, in the `--out` directory:

- `<name>_predictions.csv`: one row per sample. Columns are `t`, `state`
  (healthy / starvation / flooding / drying / abstain), `abstain_reason`,
  class probabilities `p_<state>`, and for each fault `ahead_<fault>_prob`,
  `ahead_<fault>_alarm` and `ahead_<fault>_status`.
- `<name>_report.md` and `<name>_report.json`: share of time in each state,
  alarm events, fault-ahead alarms with their horizons, how often and why the
  tool abstained, the calibration used, and the statement that real-fault
  accuracy is unverified.

### Limitations

From the [model card](hfc_models/MODEL_CARD.md):

- **Calibration must come from the same session.** A healthy baseline from
  one session reads another as faulty: false alarms reached 69%, 73% and 9%
  on the three other real sessions.
- **The bulk default catches few floods.** Flooding recall is 0.178 for state
  classification, and fault-ahead prediction catches 29% of flooding
  episodes (synthetic, held-out).
- **Spatial mode false-alarmed heavily while driving on the real recording:**
  up to 37.3% of driving samples with within-session calibration (not an
  independent test). That is why bulk is the default.
- **One starvation class.** Oxygen and hydrogen starvation are generated and
  labelled separately, but the classifier does not split them, because the
  grid axis each gas follows is an assumption.
- **No concurrent faults.** Two simultaneous faults were never generated;
  behaviour on them is unknown.
- **False alarms rise on real driving.** The bulk default's 5.7–13.7% during
  driving is far above its 0.35% synthetic rate: real healthy transients do
  not look like the generator's.

---

## Errors caught along the way

Each was caught by checking an operation through a channel other than the one
that performed or described it, and each is documented where it was found.

- **The cell leak.** Simulated cells were seeded without a load-trace term,
  so the same five cells sat on both sides of every split, including the one
  named "unseen cell". This one was *not* caught in time: the Model A
  advantage was reported before it was found, and it reverses on held-out
  cells. → [notes §17b, §18i](docs/spatial/synthetic_generator_notes.md),
  [final_results §10e](docs/spatial/final_results.md)
- **The units error.** The real recording's per-segment amps were read as
  A/cm². Caught by integrating the grid and comparing it against the
  independent stack-current channel. → [real_data_notes.md](docs/real_data_notes.md)
  ("Units fix"), [phase closure §5](docs/phase_closure_spatial.md)
- **Conditioning-hold contamination.** Most of the "dynamic" real recording
  was steady pre-test conditioning holds, not driving. Holds are now detected
  and separated, and every real-data false-alarm rate is reported both with
  and without them. → [phase closure §5](docs/phase_closure_spatial.md),
  [notes §18d](docs/spatial/synthetic_generator_notes.md)
- **Falsified predictions, recorded beside the results they got wrong:**
  - "the bulk arm will be quiet nearly by construction" (§14b), falsified in
    §16c;
  - "Model A's delta will shrink and stay positive" (§17c), falsified in §18i;
  - "40 cells will recover nearly every drying event" (§17g), falsified in §18f;
  - "shared cell identity explains the control alarm" (§15e), falsified in
    §18c.

  Scorecard: [notes §18k](docs/spatial/synthetic_generator_notes.md).
- **Others:** a convergence failure hidden by suppressed warnings that hit
  only the spatial arm, a basis mismatch that let one scalar separate healthy
  from fault, a speed cap that quietly equalised the load traces, and a
  commit that claimed to untrack third-party files and did not. →
  [phase closure §5](docs/phase_closure_spatial.md)

---

## Open questions, and what would settle them

1. **Does any of this hold on a real faulted cell?** It would take
   fault-labelled recordings under dynamic load, with onset times: stack
   current and cell voltage at ≥ 1 Hz, a healthy stretch from the same
   session, and a segmented current map to judge the spatial option. A
   request for such data is drafted in
   [`docs/spatial/data_requests.md`](docs/spatial/data_requests.md) §1 and §3.
   **It has not been sent.**
2. **Spatial versus bulk.** The generators disagree because of three
   assumptions no simulation here can test:
   - Does a fault remove mean current, or redistribute it under current
     control? (Original generator: removes. New generator: redistributes.)
   - How large is a fault's voltage signature?
   - Is per-segment sensor noise independent?

   Real spatially resolved fault data would settle all three.
   → [final_results §10d](docs/spatial/final_results.md),
   [model card](hfc_models/MODEL_CARD.md)
3. **The unexplained control alarm.** In the 30 s starvation control, an
   artifact alarm fires whenever the load trace is held out, whether or not
   the cell is. The explanation proposed for it was falsified, and it remains
   unexplained. → [notes §18c](docs/spatial/synthetic_generator_notes.md)
4. **Remaining useful life (RUL) from bulk signals.** This line is open and
   not started. The candidate dataset (Zuo et al., 2021) has a DOI that does
   not resolve. → [`docs/rul/open_questions.md`](docs/rul/open_questions.md)

---

## Repo map

| read this | for |
|---|---|
| [`docs/spatial/final_results.md`](docs/spatial/final_results.md) | the whole story in one cold-readable document; **§10 is the current statement** |
| [`docs/phase_closure_spatial.md`](docs/phase_closure_spatial.md) | why the spatial line was closed; §9 records the reversal |
| [`docs/spatial/synthetic_generator_notes.md`](docs/spatial/synthetic_generator_notes.md) | full design record: pre-registrations (§12–§14, §17) and results (§15, §16, §18) |
| [`docs/tool_design.md`](docs/tool_design.md) | design of the `hfc` generator and detector |
| [`hfc_models/MODEL_CARD.md`](hfc_models/MODEL_CARD.md) | what the shipped models are, their metrics, and their failure modes |
| [`hfc_models/TRAINING_DATA_CARD.md`](hfc_models/TRAINING_DATA_CARD.md) | the synthetic dataset the models were trained on: what is grounded and what is assumed |
| [`docs/rul/open_questions.md`](docs/rul/open_questions.md) | the RUL direction, as future work |
| [`docs/real_data_notes.md`](docs/real_data_notes.md) | the Seville dataset: format, quirks, what it can and cannot support |

```
hfc/                 the tool: generator + detector package (python -m hfc)
hfc_models/          shipped models, model card, training data card, held-out metrics
examples/            mapping files for `python -m hfc run`
tests/               unittest suite for hfc
docs/                the documents above; spatial/ = closed phase, rul/ = future work
scripts/common/      load-trace extraction and hold detection (shared)
scripts/spatial/     the closed phase: Seville parser, original generator, Models A/B,
                     Phase 5, and the post-hoc follow-up (posthoc_*.py)
notebooks/spatial/   01 (frozen), 02, 03 (superseded)
data/, hfc_data/, run_logs/   third-party data, generated data, logs (all git-ignored)
```

---

## Data sources and citations

**Real recording (healthy only).** Toharias, B., Suárez, C., Iranzo, A.,
Salva, M., Rosa, F. (2024). *"Dataset and measurements from a current density
sensor during experimental testing of dynamic load cycling for a
parallel-serpentine design of a proton exchange membrane fuel cell."* Data in
Brief. DOI: [10.12795/11441/153760](https://doi.org/10.12795/11441/153760)

Companion paper: Suárez, C., Toharias, B., Salva, M., Chesalkin, A., Rosa,
F., Iranzo, A. (2023). *"Experimental dynamic load cycling and current
density measurements of different inlet/outlet configurations of a
parallel-serpentine PEMFC."* Energy, 283, 128455. DOI:
[10.1016/j.energy.2023.128455](https://doi.org/10.1016/j.energy.2023.128455)

Licence **CC-BY-NC**. Neither the raw files nor the load traces derived from
them are redistributed here. To use them, place the dataset under
`data/raw/PEMFC_Parallel-Serpentine/` and see
[`docs/real_data_notes.md`](docs/real_data_notes.md). Everything in the quick
start runs without them.

**Static-load comparison baseline.** Kim, K., Kim, J., Choi, H. (2023).
*"Pre-diagnosis of flooding and drying in proton exchange membrane fuel cells
by bagging ensemble deep learning models using long short-term memory and
convolutional neural networks."* Energy, 266.
[IDEAS record](https://ideas.repec.org/a/eee/energy/v266y2023ics0360544222033278.html).
Bulk-only, 30-second-ahead prediction under static load. This project's
measured fault onsets are shorter than that horizon for starvation and
flooding ([phase closure §3](docs/phase_closure_spatial.md)). Audit:
[`docs/spatial/external_fault_datasets.md`](docs/spatial/external_fault_datasets.md).

**Audited and not integrated:** Zhou et al. (2022), AC voltage response / EIS
data (Mendeley `10.17632/yhg3ks3jzc.1`, CC BY 4.0). It is real and
fault-labelled, but static-load with no starvation; see
[`docs/spatial/external_fault_datasets.md`](docs/spatial/external_fault_datasets.md).

**Generated data is synthetic and flagged as such.** Every file `hfc
generate` writes carries `SYNTHETIC: true`, the generator version, the seed
and the full config, alongside a data card. Seville-layout exports are
written to folders and files named `SYNTHETIC_…`, with a
`SYNTHETIC_README.json` beside them. The Seville adapter reports
`SYNTHETIC: true` when it reads one.

---

## Project history

Kept for attribution. The current statement of results is
[final_results §10](docs/spatial/final_results.md).

### Superseded headline statements

These are struck through rather than deleted, so a reader can see what was
claimed and where it was overturned.

- ~~Health-state estimation and remaining-useful-life prediction for PEM fuel
  cells from **bulk** measurements under real automotive drive cycles — and
  whether a **load-invariant health indicator** improves either, relative to
  the load-confounded indicators the dynamic-load RUL literature currently
  uses.~~ **Superseded, November 2026:** the RUL line was never started, and
  is recorded as open future work in
  [`docs/rul/open_questions.md`](docs/rul/open_questions.md). The project's
  completed work is the fault classification and prediction study above.
- ~~**Completed phase — spatially-resolved fault detection (CLOSED November
  2026).** The project's first line asked whether a spatially resolved
  current-density grid improves fault detection over bulk signals. It does,
  by +0.07–0.12 macro-F1 depending on estimator, concentrated almost entirely
  in starvation~~ — but a 324-segment current-density plate is a laboratory
  instrument with no deployment path, which is an objection to the premise
  that cannot be answered by measuring harder. **Superseded, November 2026:**
  the +0.07–0.12 was measured with every test cell also in training; with the
  cell held out it reverses to −0.18 to −0.22
  ([final_results §10a](docs/spatial/final_results.md),
  [phase closure §9](docs/phase_closure_spatial.md)). The deployment
  objection, which is what closed the line, stands unchanged.
- ~~a single bulk voltage sensor cannot reliably distinguish between fault
  modes (flooding, drying, oxygen/hydrogen starvation) — different faults can
  produce similar voltage drops. A spatially resolved grid of current-density
  and temperature sensors can, because each fault type leaves a distinct
  spatial signature. … The result was +0.07–0.12 macro-F1, concentrated in
  starvation, with no deployment path for a 324-sensor plate.~~
  **Superseded, November 2026:** this was the project's premise, and the
  result quoted for it does not hold on held-out cells (same sources as
  above). Whether a spatial grid helps on a real cell is
  [open question 2](#open-questions-and-what-would-settle-them).

### What was done

1. **Literature review.** Twelve papers across segmented and spatial sensing,
   dynamic automotive loading, ML fault diagnosis, EIS diagnostics and
   flow-field/water management identified spatial + labelled-fault +
   dynamic-load as an open combination.
2. **Synthetic prototype**, `notebooks/spatial/01_spatial_viz_sandbox.ipynb`
   (frozen). It generates 4×4 grids for four states and compares bulk voltage
   against a spatial non-uniformity index. §2 of the generator notes later
   showed that CV(J) does **not** separate flooding from healthy on that
   generator.
3. **Real dataset integration.** A validated loader
   (`scripts/spatial/cdm_parser.py`) for an undocumented, block-structured,
   comma-decimal format, with timestamp joins across sensors on different
   clocks and handling of a DAQ start-up transient.
4. **CV(J) over σ(J).** On real data, raw σ(J) varies with load even when the
   cell is healthy. CV(J) = σ(J)/mean(|J|) was found to be 34× more
   load-invariant ([phase closure §3](docs/phase_closure_spatial.md)). It is
   verified on healthy data only.
5. **Fault dynamics.** Starvation is fast (seconds), flooding is fast and
   oscillatory, and drying is slow (tens of seconds to minutes). Kim et al.'s
   30 s horizon does not transfer to dynamically triggered faults, because
   starvation and flooding onsets (1.2–3.9 s and 2.3–8.4 s) are shorter than
   the lead time ([phase closure §3](docs/phase_closure_spatial.md)).

### Phases

> **Phases 1–5 and 3b below are the CLOSED spatial line.** Retained for
> attribution and because their findings stand; see
> [`docs/phase_closure_spatial.md`](docs/phase_closure_spatial.md). ~~Phases 4
> and 5 were scoped under that line and are **retired with it** — Model B was
> never built and the no-false-alarm test is not pending.~~ **Superseded,
> November 2026:** the line stays closed, but its experimental arc was
> afterwards run to completion — Model B was built and Phase 5 was run. Results
> in one place: [`docs/spatial/final_results.md`](docs/spatial/final_results.md).
> ~~**The active line is RUL** and does not inherit this numbering.~~
> **Superseded, November 2026:** RUL is open future work, not started — see
> [`docs/rul/open_questions.md`](docs/rul/open_questions.md).

This list is the single source of truth for phase numbering in this repo. It
supersedes any other phase reference, and retires an earlier ad-hoc
"Phase 7" reference to the no-false-alarm check, which is Phase 5 below.

- **Phase 1 — Real data integration, validation, CV(J) adoption**: **COMPLETE**
- **Phase 2 — Real-load-driven synthetic fault generator**: **COMPLETE**
  (real FC-DLC load-trace extraction and conditioning-hold correction;
  `scripts/spatial/fault_generator.py`; validation notebook
  `notebooks/spatial/03_dynamic_fault_generation.ipynb`, now superseded).
- **Phase 3 — Synthetic healthy-baseline rebuild (radial model)**:
  **COMPLETE**. An inflected (logistic) radial family, grounded physically
  and **never fit to the real healthy data**, which was reserved as the
  Phase 5 false-alarm set. See `docs/spatial/synthetic_generator_notes.md`
  §5–§7, §9.
- **Phase 3b — Fault-state spatial pattern rework**: ~~**NOT STARTED**~~
  **Superseded, November 2026: completed (next item).** Scoped in
  `docs/spatial/synthetic_generator_notes.md` §11, with acceptance
  thresholds fixed in §11b before implementation. **Scope correction
  (§11a):** with no fault-labelled real data for this cell, Phase 3b produces
  **mutually coherent, literature-grounded fault patterns, not validated
  ones**. This is a **permanent limitation**.
- **Phase 3b — COMPLETE.** All three fault types modulate the radial healthy
  template. 40 of 41 acceptance checks pass; the single failure was
  pre-registered (§11h) and recorded as a finding (§11i): the no-axis
  starvation arm does not keep starvation distinguishable from flooding.
- **Phase 4 — Fault classification models**: a snapshot spatial-vs-bulk
  classifier (Model A) and a predictive model (Model B; a windowed logistic
  model — torch is deliberately not a dependency). **COMPLETE.** ~~Model A:
  spatial beats bulk by +0.07 to +0.12 macro-F1 depending on estimator,
  almost entirely on starvation.~~ **Superseded, November 2026:** measured
  with every test cell also in training; with the cell held out, spatial is
  worse by 0.18–0.22 ([notes §18i](docs/spatial/synthetic_generator_notes.md)).
  Model B: prediction works at horizons of 1 s (starvation), 2 s (flooding)
  and 20 s (drying, indicative only), and **the pre-registered criterion
  "spatial does not help" fired for every fault**. All numbers are on
  synthetic faults.
- **Phase 5 — No-false-alarm validation**: **COMPLETE — and failed as
  pre-registered.** 14 of 16 models false-alarm on the real healthy cell; the
  models learned the generator's load-to-fault rule rather than a fault
  signature ([final_results §6](docs/spatial/final_results.md)).
- **POST-HOC follow-up (generator notes §17–§18) — COMPLETE.** Pre-registered
  as a plan before any fit. The results are in [Key results](#key-results).
- **The `hfc` tool — BUILT (post-hoc).** See [the tool](#the-tool-hfc).
- **RUL direction — OPEN, not started.** See
  [`docs/rul/open_questions.md`](docs/rul/open_questions.md).

### Rebuilding the derived load traces

Only needed for the real-trace scripts and for the extensive preset's
real-load-profile test cohort, which is skipped if the traces are absent.
`data/processed/` is git-ignored because the traces are derived from
CC-BY-NC data. Rebuild them from `data/raw/` with
`python scripts/common/extract_load_trace.py` (Normal_Flow). The other three
configurations use the same `extract_load_trace` → `detect_holds` →
`extract_dynamic_segments` → `concatenate_segments` sequence
([`docs/spatial/external_fault_datasets.md`](docs/spatial/external_fault_datasets.md) §3).

### Parked, and known quirks

- **Flow-configuration discrimination experiment** (real data only):
  **parked, undecided**. The four configurations are single sessions, so
  configuration is confounded with session.
- **`notebooks/spatial/03_dynamic_fault_generation.ipynb` is superseded.** A
  banner says so; use `hfc` to generate data.
- **Current-density sign:** raw sensor values are negative. Magnitude is
  verified, the meaning of the sign is not; `abs()` is applied only where
  sign is irrelevant.
- **V001–V007 bulk channels:** V003–V007 are dead; the meaning of V001 and
  V002 is unconfirmed.
- **Notebook 01 internal inconsistency:** two cells compute the healthy
  non-uniformity differently. It is documented and left unmodified, because
  the notebook is frozen.

Details: [`docs/real_data_notes.md`](docs/real_data_notes.md).
