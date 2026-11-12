# MODEL CARD — hfc detector models (`hfc_models/models.json`)

> **Real-fault accuracy is UNVERIFIED.** Every fault these models were
> trained and evaluated on is synthetic. This project has no labelled real
> fault data. The numbers below say how well the models separate the
> generator's own fault states on cells and drive cycles they have not seen.
> They do not say how well the models detect faults in a real cell.

November 2026 · post-hoc work, after the spatial phase's pre-registered
experiments · design in `docs/tool_design.md`.

## What the models are

Logistic regressions on a handful of named features, each a residual against
a healthy baseline calibrated on the same cell.

| model | question | lookback | horizon |
|---|---|---|---|
| `state` | healthy, starvation, flooding or drying, now? | 5 s | — |
| `state_level` | the same, for data sampled slower than 1 Hz | none | — |
| `ahead/starvation` | will starvation be confirmed? | 3 s | **1 s** |
| `ahead/flooding` | will flooding be confirmed? | 5 s | **2 s** |
| `ahead/drying` | will drying be confirmed? | 48 s | **20 s** |

Two variants of each: **bulk** (stack current + cell voltage; 4–6 features)
and **spatial** (adds 6–9 features from a current-density grid of any
resolution). **Bulk is the default.** Spatial is opt-in (`--mode spatial`).

All 10 fits converged (at most 69 of 5,000 iterations); each model stores its
own convergence record.

## What they were trained on

Synthetic dataset `extensive`, generator `hfcgen-4.0`, seed `20261112`
(`TRAINING_DATA_CARD.md` beside this file):

- 240 training cells (220 on randomised FC-DLC-style cycles, 20 on static
  holds with induced faults). 192 used for fitting, 48 held back to set alarm
  thresholds at a 0.5% false-positive target.
- Every cell is unique to one load profile, and has a fault-free calibration
  run and a fault-free companion run. `python -m hfc check-leaks` passes: no
  cell, pair or load profile is shared between train and test.
- The cell is current-controlled in this generator: a fault redistributes
  current and lowers voltage; it does not remove current.

## Results on held-out synthetic cells and load profiles

**80 test cells that share no cell and no load profile with training**: 40 on
new randomised cycles, 16 on static holds, and 24 on the **real FC-DLC load
traces, which are never in training**. 95% CIs from a bootstrap over cells.
*Synthetic faults. Not real-fault accuracy.*

### State classification

| variant | macro-F1 [95% CI] | recall healthy / starvation / flooding / drying | false alarms on fault-free runs [95% CI] |
|---|---|---|---|
| **bulk (default)** | 0.755 [0.730, 0.776] | 0.996 / 0.887 / **0.178** / 0.919 | 0.35% [0.17, 0.59] |
| spatial (opt-in) | 0.989 [0.982, 0.994] | 1.000 / 0.918 / 1.000 / 1.000 | 0.00% [0.00, 0.00] |

On the cells driven by the real FC-DLC cycles: bulk 0.759, spatial 0.992.

### Fault-ahead prediction, each at its own horizon

Never pooled across faults.

| fault | horizon | variant | precision | recall | F1 [95% CI] | false-positive rate | episodes caught |
|---|---|---|---|---|---|---|---|
| starvation | 1 s | bulk | 0.806 | 0.653 | 0.721 [0.619, 0.806] | 0.01% | 65% of 121 |
| starvation | 1 s | spatial | 0.968 | 0.496 | 0.656 [0.560, 0.741] | 0.00% | 50% of 121 |
| flooding | 2 s | bulk | 0.136 | 0.159 | **0.147 [0.109, 0.188]** | 0.48% | 29% of 299 |
| flooding | 2 s | spatial | 0.559 | 0.975 | 0.711 [0.666, 0.757] | 0.37% | 100% of 299 |
| drying | 20 s | bulk | 0.194 | 0.446 | **0.271 [0.168, 0.380]** | 0.96% | 65% of 46 |
| drying | 20 s | spatial | 0.761 | 0.990 | 0.861 [0.724, 0.951] | 0.16% | 100% of 46 |

Each fault has more than the 30 contributing events the project's power bar
asks for (121 / 299 / 46 in the test set).

### Spatial minus bulk, and the shipping decision

| task | spatial − bulk [95% CI] |
|---|---|
| state (macro-F1) | +0.234 [+0.215, +0.258] |
| starvation ahead (F1) | −0.066 [−0.174, +0.033] |
| flooding ahead (F1) | +0.564 [+0.510, +0.616] |
| drying ahead (F1) | +0.590 [+0.479, +0.702] |

The rule was fixed in the design document before any model was trained:
bulk always ships and is the default; spatial ships as an option only if it
is not worse than bulk (interval not wholly below zero) on every task. It is
not worse on any, so **it ships as an option.**

**Read this difference with care. It is assumption-driven, in both
directions.**

- It is large here because (a) the generator assumes each segment's noise is
  independent, so 324 segments average to a far cleaner signal than one
  voltage, and (b) it assumes flooding's voltage loss scales with current,
  so flooding at low load is nearly invisible in voltage. Neither is
  measured on a real cell.
- On the project's *previous* generator the opposite held: under the
  corrected crossed split the spatial arm was **worse** than bulk for every
  fault (generator notes §18). That generator let a fault remove mean
  current, which handed the bulk arm the answer.
- Two generators, two opposite answers, each traceable to an assumption.
  **Neither is evidence about real cells.** That is the honest state of the
  spatial-versus-bulk question in this project.

## The real-data false-alarm result — NOT AN INDEPENDENT TEST

The Seville recording is healthy-only, so every alarm on it is a false alarm
and nothing about detection can be measured. It has been used as a test set
before and its earlier failures were inspected before this tool was designed.
Protocol fixed in `scripts/hfc_demo_seville.py` before its single run; no
threshold or feature was adjusted for this data.

**A. Calibrate on the first half of a session, score the second half.**

| variant | false alarms, all scored samples (4 sessions) | during conditioning holds | during driving | answered |
|---|---|---|---|---|
| **bulk (default)** | 1.1% · 2.4% · 2.2% · 2.7% | 0.0% in all four | **5.7% · 12.1% · 11.1% · 13.7%** | 99.7–99.8% |
| spatial (opt-in) | 0.0% · 0.5% · 3.4% · 7.5% | ≤ 0.03% | 0.0% · 2.3% · 17.2% · **37.3%** | 99.7–99.8% |

**B. Calibrate on one session (Normal_Flow), score the other three.**

| variant | false alarms |
|---|---|
| bulk | **69% · 73% · 9%** |
| spatial | **21% · 4% · 99%** |

What this shows, plainly:

- **Within a session the bulk default is mostly quiet** (1–3% of samples,
  against 85–99% for the previous models, §16), and silent on steady load.
  **During actual driving it false-alarms on 6–14% of samples** — roughly
  twenty to forty times its synthetic rate of 0.35%. Real healthy transients
  are not what the generator's are.
- **A calibration does not transfer between sessions.** A baseline learned on
  one day and gas configuration reads another as faulty most of the time.
  The tool must be calibrated on the session it is judging. This matches the
  earlier finding that the real cell sits in two different static states
  across sessions.
- **The spatial option is less trustworthy on the real cell than its
  synthetic score suggests**: perfect on synthetic data, up to 37% false
  alarms while driving on real data. Bulk stays the default for that reason.

## Known failure modes

- **Flooding is close to invisible to the bulk variant** (recall 0.18,
  fault-ahead F1 0.15). If flooding matters and only bulk signals are
  available, this tool will mostly miss it.
- **Bulk drying prediction at 20 s is weak** (F1 0.27, precision 0.19).
- **Starvation prediction catches half to two-thirds of episodes** at a 1 s
  horizon — and one second is very little warning.
- **False alarms rise sharply on real dynamic load** (above).
- **Cross-session or cross-condition calibration fails** (above).
- **Abstention.** The tool does not answer outside the calibrated load range,
  outside the training feature envelope, across gaps, with a required channel
  missing, or below the sampling rate a capability needs. A calibration
  stretch that does not span the loads to be diagnosed produces abstentions,
  not answers.
- **One starvation class.** Oxygen and hydrogen starvation are not
  distinguished. **No concurrent faults** were ever generated; behaviour on
  two simultaneous faults is unknown.
- **Sensor layout.** Spatial features assume a roughly square plate with the
  clamping pattern centred; nothing else was generated.

## What would verify it

Fault-labelled recordings from a real cell **under dynamic load**, with
onset times, sampled at ≥ 1 Hz, carrying stack current and cell voltage —
and a segmented current-density map if the spatial variant is to be judged —
plus a healthy stretch from the same session for calibration. Even a handful
of induced flooding, drying and starvation episodes would replace every
synthetic number above with a real one. The draft request for such data is in
`docs/spatial/data_requests.md`; it has not been sent.

## Reproduce

```
python -m hfc generate   --preset extensive --out hfc_data/extensive
python -m hfc check-leaks hfc_data/extensive
python -m hfc train      hfc_data/extensive --out hfc_models
python -m hfc evaluate   hfc_data/extensive --models hfc_models --out hfc_models/metrics_heldout.json --ship
python scripts/hfc_demo_seville.py        # needs the third-party recording
```

Generation is byte-identical for a given config and seed. Logs:
`run_logs/hfc/*.log`, each ending `EXIT=<code>`.
