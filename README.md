# PEMFC Fault Diagnostics

Spatially-resolved electro-thermal fault signature detection in PEM fuel cells, targeting a gap in the literature: no existing work combines **spatial current-density/temperature resolution + labeled fault conditions + real automotive dynamic loading** for PEMFC fault diagnosis and prediction.

**Core hypothesis:** a single bulk voltage sensor cannot reliably distinguish between fault modes (flooding, drying, oxygen/hydrogen starvation) — different faults can produce similar voltage drops. A spatially resolved grid of current-density and temperature sensors can, because each fault type leaves a distinct spatial signature.

**Status:** Phase 1 (real-data integration + test build) complete. Phase 2 (predictive fault modeling) in progress.

---

## Project structure

```
ProjectHFC/
├── notebooks/
│   ├── 01_spatial_viz_sandbox.ipynb   # Synthetic prototype (frozen — see below)
│   └── 02_real_data_exploration.ipynb # Real-data validation & metric analysis
├── scripts/
│   └── load_real_data.py              # Loader for the real open PEMFC dataset
├── data/
│   └── raw/                            # Real dataset (git-ignored, see Data section)
├── docs/
│   └── real_data_notes.md             # Dataset citation, format, limitations, findings
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

## Roadmap (Phase 2, in progress)

1. Drive the synthetic fault generator with the real FC-DLC load trace instead of an invented load profile
2. Rebuild fault states as literature-grounded onset trajectories (severity-over-time, distinct per-fault timescales) using CV(J) as the diagnostic metric
3. Build and compare a spatial-vs-bulk snapshot classifier and a sequence-based (LSTM) predictive/precursor model
4. Validate both models against real healthy dynamic-load data as a no-false-alarm check
5. Document findings with explicit separation of real, synthetic, and literature-derived components

---

## Known open questions

- **Current-density sign convention**: raw sensor values are consistently negative; magnitude is physically plausible, sign meaning is unconfirmed. Not inverted anywhere in the pipeline — `abs()` applied only at analysis/visualization sites where sign is irrelevant.
- **V001–V007 bulk channel meaning**: likely raw transducer outputs paired with `PT001–PT007`, not distinct fuel-cell voltage taps — unconfirmed.
- **Internal inconsistency in the synthetic prototype**: two cells in `01_spatial_viz_sandbox.ipynb` compute the synthetic NORMAL state's nonuniformity differently (~2× disagreement). Discovered during real-data validation, documented, and deliberately left unmodified since the notebook is frozen.

Full details on all of the above are in `docs/real_data_notes.md`.
