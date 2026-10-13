# PEMFC Fault Diagnostics

Spatially-resolved electro-thermal fault signature detection in PEM fuel cells, targeting **spatial current-density/temperature resolution + labeled fault conditions + real automotive dynamic loading** for PEMFC fault diagnosis and prediction.

**Core hypothesis:** a single bulk voltage sensor cannot reliably distinguish between fault modes (flooding, drying, oxygen/hydrogen starvation) — different faults can produce similar voltage drops. A spatially resolved grid of current-density and temperature sensors can, because each fault type leaves a distinct spatial signature.

**Status:** see [Phases](#phases) below — the single source of truth for phase numbering in this repo (supersedes any other phase reference, including an earlier ad-hoc "Phase 7").

---

## Project structure

```
PEMFC-FAULT-DIAGNOSTICS/
├── notebooks/
│   ├── 01_spatial_viz_sandbox.ipynb   # Synthetic prototype (frozen — see below)
│   ├── 02_real_data_exploration.ipynb # Real-data validation & metric analysis
│   └── 03_dynamic_fault_generation.ipynb # Healthy-baseline validation + fault demo
├── scripts/
│   ├── load_real_data.py              # Loader for the real open PEMFC dataset
│   ├── fault_generator.py             # Load-trace-driven synthetic fault generator
│   ├── analyze_real_structure.py      # Real healthy spatial-template analysis
│   └── measure_resolution_effects.py  # Grid-resolution confound measurement
├── data/
│   └── raw/                            # Real dataset (git-ignored, see Data section)
├── docs/
│   ├── real_data_notes.md             # Dataset citation, format, limitations, findings
│   ├── synthetic_generator_notes.md   # Generator decisions, limits, Phase 3b scope
│   └── external_fault_datasets.md     # Audits of external fault-labelled data, + decisions
├── figures/                            # Generated outputs (git-ignored)
└── hfc_env/                            # Python 3.12.9 virtualenv (git-ignored)
```

---

## What's been done

### 1. Literature review
12 papers reviewed across segmented/spatial sensing, dynamic automotive loading, ML fault diagnosis, EIS diagnostics, and flow-field/water management, establishing the specific combination of spatial + labeled-fault + dynamic-load (later refined to include *predictive* forecasting) as an open gap.

### 2. Synthetic prototype — `01_spatial_viz_sandbox.ipynb` (frozen)
Generates literature-grounded synthetic 4×4 current-density/temperature grids for four states (NORMAL, FLOODING, DRYING, OXYGEN STARVATION). Its central figure compares bulk voltage (non-discriminating across faults) against a spatial nonuniformity index (distinct per fault) — the project's core argument in miniature. This notebook and its output figures are treated as a completed deliverable and are not modified going forward.

### 3. Real dataset integration
Integrated an open dataset (see [Data](#data) below) providing real 18×18 current-density and 9×9 temperature measurements from a commercial CDM sensor under real automotive-style dynamic load cycling. Since this dataset has **no labeled fault conditions**, its role is calibration/grounding — validating the synthetic prototype's assumptions — not classifier training data.

Work included:
- Reverse-engineering an undocumented, block-structured, European-locale (comma-decimal) raw file format
- Building a validated loader (`scripts/load_real_data.py`) with structural assertions, timestamp-based joins across sensors with different sampling behavior, and explicit handling of a DAQ startup transient
- Computing the same spatial-nonuniformity metric on real data as on synthetic data for direct comparison

### 4. Key finding: CV(J) over raw σ(J)
Real-data validation revealed that raw spatial-nonuniformity (σ(J), standard deviation of the current-density grid) varies 3–4× with load level alone, even in fault-free data — it confounds "fault present" with "load is high." A normalized alternative, **CV(J) = σ(J) / mean(|J|)**, was tested and found to be **34× more load-invariant** on the real dynamic-load data. This is now the project's adopted diagnostic metric (verified on healthy data only — no fault-labeled data exists to validate against yet).

### 5. Literature-grounded fault dynamics
Confirmed, across multiple sources, that PEMFC faults have distinct onset timescales and shapes: **starvation** (fast, seconds, driven by air-supply lag during load ramps), **flooding** (fast but oscillatory — accumulate/breakthrough/drain cycles), **drying** (slow, tens of seconds to minutes, driven by membrane humidification dynamics). Also identified closely related prior work (Kim et al., *Energy* 2023) doing bulk-only 30-second-ahead fault prediction — but only under static lab loading, never real dynamic driving conditions, sharpening this project's specific novelty claim.

---

## Data

**Audited and deliberately NOT integrated:** Zhou et al. (2022) AC voltage
response / EIS data (Mendeley `10.17632/yhg3ks3jzc.1`, **CC BY 4.0**) is real
and fault-labelled, but covers no starvation and is static-load, so it is not
used as a Phase 4 bulk-arm anchor. Reasoning, full audit, and the split-design
evidence it did produce are in `docs/external_fault_datasets.md`. The files are
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

---

## Phases

The single source of truth for phase numbering in this repo — supersedes
any other phase reference elsewhere (including notes/discussion outside
version control), and retires an earlier ad-hoc "Phase 7" reference to the
no-false-alarm check, which is Phase 5 below.

- **Phase 1 — Real data integration, validation, CV(J) adoption**: **COMPLETE**
- **Phase 2 — Real-load-driven synthetic fault generator**: **COMPLETE**
  (Tasks 1/1b/2: real FC-DLC load trace extraction + conditioning-hold
  correction; `scripts/fault_generator.py`'s literature-grounded fault-onset
  trajectories driven by that trace. Task 3: validation notebook
  `notebooks/03_dynamic_fault_generation.ipynb`, which validates the healthy
  baseline and demonstrates the fault paths with Phase 3b caveats stated
  per panel).
- **Phase 3 — Synthetic healthy-baseline rebuild (radial model)**:
  **COMPLETE**. An inflected (logistic) radial family, grounded physically
  and **never fit to the real healthy data** — which is reserved as the
  Phase 5 false-alarm validation set — carried forward as a sampled family
  rather than a point estimate, with native resolution at 18×18 to match
  the real sensor. See `docs/synthetic_generator_notes.md` §5–§7, §9.
- **Phase 3b — Fault-state spatial pattern rework**: rebase the fault
  patterns onto the radial healthy baseline. **NOT STARTED** — scoped in
  `docs/synthetic_generator_notes.md` §11, with acceptance thresholds fixed
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
  pre-registered in `docs/synthetic_generator_notes.md` §11h and is recorded
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

- **`notebooks/03_dynamic_fault_generation.ipynb` is stale and will raise.**
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
- **Internal inconsistency in the synthetic prototype**: two cells in `01_spatial_viz_sandbox.ipynb` compute the synthetic NORMAL state's nonuniformity differently (~2× disagreement). Discovered during real-data validation, documented, and deliberately left unmodified since the notebook is frozen.

Full details on all of the above are in `docs/real_data_notes.md`.
