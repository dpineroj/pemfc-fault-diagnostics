# PEMFC Fault Diagnostics

Health-state estimation and remaining-useful-life prediction for PEM fuel cells from **bulk** measurements under real automotive drive cycles — and whether a **load-invariant health indicator** improves either, relative to the load-confounded indicators the dynamic-load RUL literature currently uses.

> **Correction, POST-HOC (November 2026).** The figure quoted in the next two
> paragraphs — spatial beats bulk by +0.07–0.12 macro-F1 — was measured with
> every test cell also present in training. **With the cell held out it
> reverses: −0.18 to −0.22.** The paragraphs are left as written; see
> [`docs/spatial/final_results.md`](docs/spatial/final_results.md) §10.

**Completed phase — spatially-resolved fault detection (CLOSED November 2026).** The project's first line asked whether a spatially resolved current-density grid improves fault detection over bulk signals. It does, by +0.07–0.12 macro-F1 depending on estimator, concentrated almost entirely in starvation — but a 324-segment current-density plate is a laboratory instrument with no deployment path, which is an objection to the premise that cannot be answered by measuring harder. Closed deliberately rather than abandoned: see **[`docs/phase_closure_spatial.md`](docs/phase_closure_spatial.md)** for the result, the reasoning, what stands independently, and what was never validated.

**Completed phase — spatial fault detection (closed Nov 2026):** a single bulk voltage sensor cannot reliably distinguish between fault modes (flooding, drying, oxygen/hydrogen starvation) — different faults can produce similar voltage drops. A spatially resolved grid of current-density and temperature sensors can, because each fault type leaves a distinct spatial signature. Closed; see [`docs/phase_closure_spatial.md`](docs/phase_closure_spatial.md). The result was +0.07–0.12 macro-F1, concentrated in starvation, with no deployment path for a 324-sensor plate.

**Status:** see [Phases](#phases) below — the single source of truth for phase numbering in this repo (supersedes any other phase reference, including an earlier ad-hoc "Phase 7").

---

## The tool: `hfc` (POST-HOC, November 2026)

> **Real-fault accuracy is unverified.** This project has no labelled real
> fault data. Every fault the tool has been trained and evaluated on is
> synthetic. Its output is a screening aid, not a diagnosis.

`hfc` is a Python package with a CLI that does three things:

1. **Generates** synthetic PEM fuel cell datasets shaped like the ones this
   project worked with (`hfcgen-4.0`): current-density and temperature grids
   at any resolution, Seville-named bulk channels, fault labels, and a
   Seville-raw-layout export. Every file is flagged `SYNTHETIC`.
2. **Classifies** a cell's state — healthy, flooding, drying, starvation —
   from stack current and cell voltage, optionally with a current-density
   grid.
3. **Predicts** an upcoming fault at its own horizon: starvation 1 s,
   flooding 2 s, drying 20 s.

It calibrates on a known-healthy stretch of **your own** cell and reports
residuals against that; it abstains instead of alarming when the input is
outside what it can judge.

```
# generate a dataset (seconds for smoke; ~2.5 min and 1.8 GB for extensive)
python -m hfc generate --preset smoke --out hfc_data/smoke
python -m hfc check-leaks hfc_data/smoke

# run the shipped models on a new recording
python -m hfc run --mapping examples/mapping_table.json --models hfc_models \
       --calibrate-range 0 1800 --out hfc_out
```

`--calibrate-range START_S END_S` names a stretch of the recording you assert
is healthy; `--calibrate-mapping other.json` uses a separate healthy
recording instead. The mapping file says where the data is and what its
columns and units mean — see [`examples/`](examples/). Outputs: a
per-sample predictions CSV, and a report in Markdown and JSON.

**What the evidence says** (full detail in
[`hfc_models/MODEL_CARD.md`](hfc_models/MODEL_CARD.md)):

- On 80 held-out synthetic cells under load profiles never seen in training,
  the bulk default scores macro-F1 0.755 [0.730, 0.776] with 0.35% false
  alarms; it is nearly blind to flooding (recall 0.18). The opt-in spatial
  variant scores 0.989 on the same cells — a difference that follows from the
  generator's assumptions and is not evidence about real cells.
- On the real healthy recording (**not an independent test**), calibrated
  within a session, the bulk default false-alarms on 1–3% of samples overall
  and **6–14% during actual driving**. Calibrated on a different session it
  false-alarms on up to 73%: a baseline does not transfer between sessions.

Design: [`docs/tool_design.md`](docs/tool_design.md). Tests:
`python -m unittest discover -s tests`.

---

## Project structure

```
PEMFC-FAULT-DIAGNOSTICS/
├── docs/
│   ├── phase_closure_spatial.md       # READ FIRST: closes the spatial line
│   ├── real_data_notes.md             # Seville dataset — shared by both lines
│   ├── spatial/                       # CLOSED phase
│   │   ├── final_results.md           # one cold-readable account of the results
│   │   ├── synthetic_generator_notes.md   # full design record, §1–§16
│   │   ├── data_requests.md           # drafted, unsent dataset request
│   │   └── external_fault_datasets.md
│   ├── tool_design.md                 # design of the hfc generator and detector
│   └── rul/                           # future work (health state + RUL); open_questions.md
├── scripts/
│   ├── common/                        # transfers to both lines
│   │   ├── extract_load_trace.py      # FC-DLC segmentation, hold detection
│   │   └── bulk_timeseries.py         # comma-decimal bulk channel reading
│   ├── spatial/                       # CLOSED phase
│   │   ├── cdm_parser.py              # Seville CDM block parser + load_run
│   │   ├── fault_generator.py         # load-trace-driven synthetic generator
│   │   ├── posthoc_*.py               # POST-HOC follow-up runs (notes §17-§18)
│   │   └── ...                        # measurements, tests, Models A and B, Phase 5
│   ├── hfc_run.sh                     # run an hfc command with an EXIT=<code> log
│   ├── hfc_demo_seville.py            # detector on the real recording (not independent)
│   └── rul/                           # ACTIVE line
├── notebooks/
│   └── spatial/                       # 01 (frozen), 02, 03 (superseded)
├── hfc/                               # THE TOOL: generator + detector package
├── hfc_models/                        # shipped models, MODEL_CARD.md, metrics
├── examples/                          # mapping files for `python -m hfc run`
├── tests/                             # unittest suite for hfc
├── data/
│   ├── raw/                           # third-party data (git-ignored)
│   └── processed/                     # regenerable traces (git-ignored)
├── hfc_data/                          # generated SYNTHETIC datasets (git-ignored)
├── run_logs/                          # run logs + results (git-ignored)
├── figures/                           # generated outputs (git-ignored)
└── hfc_env/                           # Python 3.12.9 virtualenv (git-ignored)
```

---

## What's been done

### 1. Literature review
12 papers reviewed across segmented/spatial sensing, dynamic automotive loading, ML fault diagnosis, EIS diagnostics, and flow-field/water management, establishing the specific combination of spatial + labeled-fault + dynamic-load (later refined to include *predictive* forecasting) as an open gap.

### 2. Synthetic prototype — `notebooks/spatial/01_spatial_viz_sandbox.ipynb` (frozen)
Generates literature-grounded synthetic 4×4 current-density/temperature grids for four states (NORMAL, FLOODING, DRYING, OXYGEN STARVATION). Its central figure compares bulk voltage (non-discriminating across faults) against a spatial nonuniformity index (distinct per fault) — the closed spatial phase's core argument in miniature, and see §2 of the generator notes, which establishes that CV(J) does **not** in fact discriminate flooding from healthy on this generator. This notebook and its output figures are treated as a completed deliverable and are not modified going forward.

### 3. Real dataset integration
Integrated an open dataset (see [Data](#data) below) providing real 18×18 current-density and 9×9 temperature measurements from a commercial CDM sensor under real automotive-style dynamic load cycling. Since this dataset has **no labeled fault conditions**, its role is calibration/grounding — validating the synthetic prototype's assumptions — not classifier training data.

Work included:
- Reverse-engineering an undocumented, block-structured, European-locale (comma-decimal) raw file format
- Building a validated loader (`scripts/spatial/cdm_parser.py`) with structural assertions, timestamp-based joins across sensors with different sampling behavior, and explicit handling of a DAQ startup transient
- Computing the same spatial-nonuniformity metric on real data as on synthetic data for direct comparison

### 4. Key finding: CV(J) over raw σ(J)
Real-data validation revealed that raw spatial-nonuniformity (σ(J), standard deviation of the current-density grid) varies 3–4× with load level alone, even in fault-free data — it confounds "fault present" with "load is high." A normalized alternative, **CV(J) = σ(J) / mean(|J|)**, was tested and found to be **34× more load-invariant** on the real dynamic-load data. This is now the project's adopted diagnostic metric (verified on healthy data only — no fault-labeled data exists to validate against yet).

### 5. Literature-grounded fault dynamics
Confirmed, across multiple sources, that PEMFC faults have distinct onset timescales and shapes: **starvation** (fast, seconds, driven by air-supply lag during load ramps), **flooding** (fast but oscillatory — accumulate/breakthrough/drain cycles), **drying** (slow, tens of seconds to minutes, driven by membrane humidification dynamics). Also identified closely related prior work (Kim et al., *Energy* 2023) doing bulk-only 30-second-ahead fault prediction — but only under static lab loading, never real dynamic driving conditions, sharpening the spatial phase's novelty claim (closed). Note the phase's own closing measurement on that comparison: Kim et al.'s 30-second horizon does **not** transfer to dynamically-triggered faults, because starvation and flooding onsets (1.2–3.9 s and 2.3–8.4 s) are shorter than the lead time itself — see `docs/phase_closure_spatial.md` §3.

---

## Data

**Audited and deliberately NOT integrated:** Zhou et al. (2022) AC voltage
response / EIS data (Mendeley `10.17632/yhg3ks3jzc.1`, **CC BY 4.0**) is real
and fault-labelled, but covers no starvation and is static-load, so it is not
used as a Phase 4 bulk-arm anchor. Reasoning, full audit, and the split-design
evidence it did produce are in `docs/spatial/external_fault_datasets.md`. The files are
retained under `data/raw/zhou_ac_voltage/` (gitignored) in case a later use
emerges. **No loader exists and none is planned.**

**Primary dataset:** Toharias, B., Suárez, C., Iranzo, A., Salva, M., Rosa, F. (2024). *"Dataset and measurements from a current density sensor during experimental testing of dynamic load cycling for a parallel-serpentine design of a proton exchange membrane fuel cell."* Data in Brief. DOI: [10.12795/11441/153760](https://doi.org/10.12795/11441/153760)

**Companion methodology paper:** Suárez, C., Toharias, B., Salva, M., Chesalkin, A., Rosa, F., Iranzo, A. (2023). *"Experimental dynamic load cycling and current density measurements of different inlet/outlet configurations of a parallel-serpentine PEMFC."* Energy, 283, 128455. DOI: [10.1016/j.energy.2023.128455](https://doi.org/10.1016/j.energy.2023.128455)

License: **CC-BY-NC**. Raw data files are not committed to this repository — see `docs/real_data_notes.md` for full details on obtaining and placing the dataset in `data/raw/`.

---

## Setup

```powershell
python -m venv hfc_env
.\hfc_env\Scripts\Activate.ps1
pip install -r requirements.txt   # numpy, pandas, matplotlib, scipy, scikit-learn, seaborn, jupyter
```

Place the downloaded dataset under `data/raw/PEMFC_Parallel-Serpentine/` (git-ignored; not included in this repo per the dataset's license terms).

**Regenerating the derived load traces.** `data/processed/` is git-ignored — the
load traces are derived from CC-BY-NC third-party data and are deterministically
regenerable, so they are not redistributed here (reasoning in
`docs/spatial/external_fault_datasets.md` §3). To rebuild them from `data/raw/`:

```powershell
python scripts/common/extract_load_trace.py          # Normal_Flow (full + dynamic-only)
```

Phase 4 additionally needs the other three gas configurations' dynamic-only
traces (`Inverse_Flow`, `Inverse_Air_Flow`, `Inverse_Hydrogen_Flow`), built with
the same `extract_load_trace` / `detect_holds` / `extract_dynamic_segments` /
`concatenate_segments` sequence per configuration.

---

## Phases

> **Phases 1–5 and 3b below are the CLOSED spatial line.** Retained for
> attribution and because their findings stand; see
> [`docs/phase_closure_spatial.md`](docs/phase_closure_spatial.md). ~~Phases 4
> and 5 were scoped under that line and are **retired with it** — Model B was
> never built and the no-false-alarm test is not pending.~~ **Superseded,
> November 2026:** the line stays closed, but its experimental arc was
> afterwards run to completion — Model B was built and Phase 5 was run. Results
> in one place: [`docs/spatial/final_results.md`](docs/spatial/final_results.md).
> **The active line is RUL** and does not inherit this numbering.

The single source of truth for phase numbering in this repo — supersedes
any other phase reference elsewhere (including notes/discussion outside
version control), and retires an earlier ad-hoc "Phase 7" reference to the
no-false-alarm check, which is Phase 5 below.

- **Phase 1 — Real data integration, validation, CV(J) adoption**: **COMPLETE**
- **Phase 2 — Real-load-driven synthetic fault generator**: **COMPLETE**
  (Tasks 1/1b/2: real FC-DLC load trace extraction + conditioning-hold
  correction; `scripts/spatial/fault_generator.py`'s literature-grounded fault-onset
  trajectories driven by that trace. Task 3: validation notebook
  `notebooks/spatial/03_dynamic_fault_generation.ipynb`, which validates the healthy
  baseline and demonstrates the fault paths with Phase 3b caveats stated
  per panel).
- **Phase 3 — Synthetic healthy-baseline rebuild (radial model)**:
  **COMPLETE**. An inflected (logistic) radial family, grounded physically
  and **never fit to the real healthy data** — which is reserved as the
  Phase 5 false-alarm validation set — carried forward as a sampled family
  rather than a point estimate, with native resolution at 18×18 to match
  the real sensor. See `docs/spatial/synthetic_generator_notes.md` §5–§7, §9.
- **Phase 3b — Fault-state spatial pattern rework**: rebase the fault
  patterns onto the radial healthy baseline. **NOT STARTED** — scoped in
  `docs/spatial/synthetic_generator_notes.md` §11, with acceptance thresholds fixed
  in §11b *before* implementation so they cannot drift. Required because
  notebook 01's fault patterns were designed as modifications of a
  *diagonal flow-path gradient*; rebasing the interpolation's start point
  without re-examining its endpoints produces a trajectory between two
  points never designed to connect (§8). The measured consequence: a single
  scalar — correlation with distance-from-centre — currently separates
  healthy from fault at **100% accuracy at severity 0.01**, before any
  meaningful fault has developed, which is basis mismatch rather than fault
  physics (§10).

  **Scope correction (§11a):** this phase **cannot** receive the treatment the
  healthy form received. Capability, identifiability and resolution checks
  all consumed real templates, and **no fault-labelled real data for this
  cell exists or is obtainable without hardware**. Phase 3b therefore
  produces **mutually coherent, literature-grounded fault patterns, not
  validated ones**, and no downstream result may describe them as validated.
  This is a **permanent limitation**, not a gap pending future work.
- **Phase 3b — COMPLETE.** All three fault types now modulate the radial
  healthy template (`J_fault = J_healthy · M_fault`, continuity by
  construction). 40/41 acceptance checks pass; the single failure was
  pre-registered in `docs/spatial/synthetic_generator_notes.md` §11h and is recorded
  as a finding in §11i — the no-axis starvation arm does not keep starvation
  distinguishable from flooding in this cell's bathtub geometry (0.638 against
  a 0.95 threshold, predicted at ~0.6 before measuring). Both starvation arms
  proceed to Phase 4; neither is the default.

- **Phase 4 — Fault classification models**: a snapshot spatial-vs-bulk
  classifier (Model A), and a predictive/precursor model (Model B; built as a
  windowed logistic model, not an LSTM — torch is deliberately not a
  dependency). **COMPLETE.** Model A: spatial beats bulk by +0.07 to +0.12
  macro-F1 depending on estimator, almost entirely on starvation. Model B:
  prediction works at per-fault horizons of 1 s (starvation), 2 s (flooding)
  and 20 s (drying, indicative only), and **the pre-registered criterion
  "spatial does not help" fired for every fault** — the bulk arm predicts at
  least as well. All numbers are on synthetic faults and none is real-fault
  accuracy.
- **Phase 5 — No-false-alarm validation**: both Phase 4 models run against the
  real healthy dynamic-load data. **COMPLETE — and failed as pre-registered:**
  fourteen of sixteen models false-alarm on the real healthy cell (most above
  85% of timesteps). The models learned the generator's load-to-fault rule
  rather than a fault signature. Only Model B's starvation predictor with
  spatial input passes (1.0–2.7%).

  Full account: [`docs/spatial/final_results.md`](docs/spatial/final_results.md).

- **POST-HOC follow-up (generator notes §17–§18) — COMPLETE.** Pre-registered
  as a plan before any fit. Found that neither Phase 4 split held out a cell;
  with the cell held out, **Model A's spatial advantage reverses** (−0.18 to
  −0.22, intervals wholly below zero) and Model B's null strengthens and
  survives matched tuning. Drying remains underpowered (23 of a target 30
  events). Fault-free companion runs remove the bulk arm's Phase 5 defect in
  a re-test that is **not independent**; no model is quiet during driving.
- **The `hfc` tool — BUILT (post-hoc).** See
  [the tool](#the-tool-hfc-post-hoc-november-2026) above and
  [`hfc_models/MODEL_CARD.md`](hfc_models/MODEL_CARD.md). Real-fault accuracy
  unverified.
- **RUL direction — OPEN, not started.** See
  [`docs/rul/open_questions.md`](docs/rul/open_questions.md).

---

## Open threads

Work that's been started, paused, and not yet decided — kept here so it
isn't lost between sessions:

- **Flow-configuration discrimination experiment** (real data only, no
  synthetic data involved): tests whether spatial features can distinguish
  the dataset's four gas-flow configurations when bulk signals cannot.
  Premise check found the four configs are **not independent replicates** —
  each is a single recording session with its own near-deterministic
  hold-current setpoint, so configuration identity is confounded with
  session identity in a way no split or feature engineering can separate
  with this data. Three ways forward were proposed (proceed with that
  limitation stated plainly; restrict to the dynamic, non-hold portions of
  each run and re-check; or treat the dataset as unable to support this
  specific claim). **Parked, undecided** — no code committed for this yet.

---

## Known stale artifacts

- **`notebooks/spatial/03_dynamic_fault_generation.ipynb` — SUPERSEDED
  (November 2026), no longer stale-and-unmarked.** It opens with a banner
  saying so. It was superseded rather than refreshed: its fault cells use the
  pre-Phase-3b interpolation and will raise against the current generator
  (which requires an explicit `starvation_arm`), and every figure it held is
  now produced by a re-runnable script recorded in the generator notes. Use
  the `hfc` package to generate data.

## Known open questions

- **Current-density sign convention**: raw sensor values are consistently negative; magnitude is physically plausible, sign meaning is unconfirmed. Not inverted anywhere in the pipeline — `abs()` applied only at analysis/visualization sites where sign is irrelevant.
- **V001–V007 bulk channel meaning**: likely raw transducer outputs paired with `PT001–PT007`, not distinct fuel-cell voltage taps — unconfirmed.
- **Internal inconsistency in the synthetic prototype**: two cells in `notebooks/spatial/01_spatial_viz_sandbox.ipynb` compute the synthetic NORMAL state's nonuniformity differently (~2× disagreement). Discovered during real-data validation, documented, and deliberately left unmodified since the notebook is frozen.

Full details on all of the above are in `docs/real_data_notes.md`.
