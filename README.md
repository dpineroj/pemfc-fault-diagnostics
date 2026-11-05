# PEMFC Fault Diagnostics

Health-state estimation and remaining-useful-life prediction for PEM fuel cells from **bulk** measurements under real automotive drive cycles — and whether a **load-invariant health indicator** improves either, relative to the load-confounded indicators the dynamic-load RUL literature currently uses.

**Completed phase — spatially-resolved fault detection (CLOSED November 2026).** The project's first line asked whether a spatially resolved current-density grid improves fault detection over bulk signals. It does, by +0.07–0.12 macro-F1 depending on estimator, concentrated almost entirely in starvation — but a 324-segment current-density plate is a laboratory instrument with no deployment path, which is an objection to the premise that cannot be answered by measuring harder. Closed deliberately rather than abandoned: see **[`docs/phase_closure_spatial.md`](docs/phase_closure_spatial.md)** for the result, the reasoning, what stands independently, and what was never validated.

**Completed phase — spatial fault detection (closed Nov 2026):** a single bulk voltage sensor cannot reliably distinguish between fault modes (flooding, drying, oxygen/hydrogen starvation) — different faults can produce similar voltage drops. A spatially resolved grid of current-density and temperature sensors can, because each fault type leaves a distinct spatial signature. Closed; see [`docs/phase_closure_spatial.md`](docs/phase_closure_spatial.md). The result was +0.07–0.12 macro-F1, concentrated in starvation, with no deployment path for a 324-sensor plate.

**Status:** see [Phases](#phases) below — the single source of truth for phase numbering in this repo (supersedes any other phase reference, including an earlier ad-hoc "Phase 7").

---

## Project structure

```
PEMFC-FAULT-DIAGNOSTICS/
├── docs/
│   ├── phase_closure_spatial.md       # READ FIRST: closes the spatial line
│   ├── real_data_notes.md             # Seville dataset — shared by both lines
│   ├── spatial/                       # CLOSED phase
│   │   ├── synthetic_generator_notes.md
│   │   └── external_fault_datasets.md
│   └── rul/                           # ACTIVE line (health state + RUL)
├── scripts/
│   ├── common/                        # transfers to both lines
│   │   ├── extract_load_trace.py      # FC-DLC segmentation, hold detection
│   │   └── bulk_timeseries.py         # comma-decimal bulk channel reading
│   ├── spatial/                       # CLOSED phase (13 files)
│   │   ├── cdm_parser.py              # Seville CDM block parser + load_run
│   │   ├── fault_generator.py         # load-trace-driven synthetic generator
│   │   └── ...                        # measurements, tests, Model A
│   └── rul/                           # ACTIVE line
├── notebooks/
│   └── spatial/                       # 01 (frozen), 02, 03 (stale)
├── data/
│   ├── raw/                           # third-party data (git-ignored)
│   └── processed/                     # regenerable traces (git-ignored)
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
> [`docs/phase_closure_spatial.md`](docs/phase_closure_spatial.md). Phases 4
> and 5 were scoped under that line and are **retired with it** — Model B was
> never built and the no-false-alarm test is not pending. **The active line is
> RUL** and does not inherit this numbering.

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
  classifier, and a sequence-based (LSTM) predictive/precursor model. **NOT
  STARTED — queued behind Phase 3b, deliberately.** Training the spatial arm
  against the current fault patterns would let it score near-perfectly on a
  basis-mismatch artifact the bulk arm is blind to, confirming the project's
  central hypothesis for the wrong reason (§10).
- **Phase 5 — No-false-alarm validation**: validate both Phase 4 models
  against real healthy dynamic-load data. **NOT STARTED.**

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

- **`notebooks/spatial/03_dynamic_fault_generation.ipynb` is stale and will raise.**
  Phase 3b made `starvation_arm` a required argument to
  `generate_labeled_run` (Sec. 11d/11h: starvation is generated two ways and
  neither is the default), so the notebook's bare `generate_labeled_run()`
  call now fails deliberately rather than silently picking an arm. Its
  starvation panels also describe the superseded diagonal-gradient
  interpolation. **Needs the arm argument plus a re-execution, and its
  starvation commentary rewritten against §11c/§11d.** Scoped as its own task,
  not folded into Phase 3b. Recorded here so it is not rediscovered as a bug.

## Known open questions

- **Current-density sign convention**: raw sensor values are consistently negative; magnitude is physically plausible, sign meaning is unconfirmed. Not inverted anywhere in the pipeline — `abs()` applied only at analysis/visualization sites where sign is irrelevant.
- **V001–V007 bulk channel meaning**: likely raw transducer outputs paired with `PT001–PT007`, not distinct fuel-cell voltage taps — unconfirmed.
- **Internal inconsistency in the synthetic prototype**: two cells in `notebooks/spatial/01_spatial_viz_sandbox.ipynb` compute the synthetic NORMAL state's nonuniformity differently (~2× disagreement). Discovered during real-data validation, documented, and deliberately left unmodified since the notebook is frozen.

Full details on all of the above are in `docs/real_data_notes.md`.
