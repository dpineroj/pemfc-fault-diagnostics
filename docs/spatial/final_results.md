# Final results — the spatially-resolved fault-detection phase

**November 2026.** One document, readable without the rest of the repo. It
says what was asked, what was measured, what was fixed in advance, which
pre-set criteria fired, what that means, and which errors were caught on the
way and what each would have cost.

> **Amended, POST-HOC — read section 10 first.** A follow-up found that the
> "held-out" cells in sections 4 and 5 were never held out. With the cell
> held out, **section 4's spatial advantage reverses** (−0.18 to −0.22);
> section 5's finding stands and strengthens. Sections 1–9 are left as
> written.

The phase is **closed** (`docs/phase_closure_spatial.md`). This document
records how its experiments ended; it does not reopen it.

**Read this first.** Every fault in this work is synthetic. The real dataset
contains a healthy cell only. So no number below is a real-fault detection or
prediction accuracy — they are measurements on a fault generator whose
patterns are literature-grounded and mutually coherent but were never
validated, because nothing real exists to validate them against. Section 7
shows, by measurement, how far the trained models are from transferring.

---

## 1. What was asked

> Does a spatially resolved grid of current-density measurements improve
> fault detection and prediction over bulk signals alone, under dynamic
> automotive load, for flooding, membrane drying and oxygen starvation?

The cell is a 50 cm² PEM fuel cell with an 18 × 18 segmented current-density
plate, recorded by a third party under the FC-DLC automotive drive cycle in
four gas-routing configurations. "Spatial" means a model sees all 324
segments. "Bulk" means it sees only what a production stack could measure —
here mean current density and the load level.

The plan was a two-step one: **build a classifier that can name a fault from
a snapshot (Model A), then use the same machinery to predict a fault before
it is confirmed (Model B), then check on the real cell that the models at
least stay quiet when nothing is wrong (Phase 5).**

## 2. What there was to work with

- **Real data: healthy only.** Four recording sessions, about 17,650 usable
  timesteps each. No faults. It was reserved from the start as the
  false-alarm test set and never used for training or model selection.
- **No real fault-labelled spatial data exists.** One bounded search during
  this campaign and two full dataset audits earlier found fault-labelled
  datasets that are bulk-only and static-load. A request email for one
  further candidate dataset is drafted in `docs/spatial/data_requests.md`
  and has **not** been sent.
- **So faults are generated.** A healthy spatial template (radial "bathtub",
  grounded physically, never fit to the real data) is multiplied by a fault
  pattern whose severity grows from an onset. Onsets are triggered by events
  in the real load trace: steep ramps trigger starvation, low-load dwells
  trigger flooding, high-load dwells trigger drying.

**What the generator fallback costs.** Everything downstream inherits the
generator's assumptions — which fault looks like what, and above all *when*
faults happen. Section 7 is where that bill arrives.

## 3. What was fixed before measuring

Each of these was written to `docs/spatial/synthetic_generator_notes.md` and
committed before the code that could test it existed; the git history is the
evidence of order.

| what | where | commit |
|---|---|---|
| Model A falsification criteria | §12f | before Model A ran |
| Model B design, horizons, splits, five ordered criteria | §13a–k | `fc3fb28` |
| Minimum number of drying events (30, at least 5 per trace) | §13b | `fc3fb28` |
| One final attempt at the uncapped Model A run, no third | §12i | `fc3fb28` |
| Phase 5 pass rule: false-alarm rate ≤ 5% on all four real configurations; > 50% = "learned the simulator" | §14 | `fc3fb28` |
| Two corrections found by a label-only measurement, before any fit | §13l | `df434c5` |

Standing rules throughout: splits are grouped, never random; both arms get
the identical model; every fit's convergence is asserted and reported before
any number (the jobs run with warnings suppressed, so a warning is no signal
at all); thresholds are never moved after the fact.

## 4. Model A — naming a fault from a snapshot

Four classes (healthy, drying, flooding, starvation), multinomial logistic
regression, 326 spatial features against 2 bulk features.

**Result: spatial input helps, narrowly.** +0.07 to +0.12 macro-F1 depending
on the estimator — +0.10 to +0.12 with logistic regression, +0.07 to +0.08
with a random forest, which closes about a third of the gap. Almost all of
it is starvation: bulk is right about half the time it says "starvation"
(precision 0.505) against 0.875 for spatial. Flooding and drying the bulk arm
handles competently.

**The robustness re-run.** The first complete run capped the samples per
group for speed, which silently made the four load traces equal in weight.
A re-run without the cap died twice; one final attempt, with its rule fixed
in advance, completed during this campaign. With the traces at their real,
unequal weights the difference is the same size:

| | capped run | uncapped run |
|---|---|---|
| logistic, split by trace-and-cell | +0.103 to +0.123 | +0.100 to +0.119 |
| logistic, split by trace | +0.109 to +0.130 | +0.112 to +0.130 |
| random forest | +0.066 to +0.082 | +0.080 to +0.083 |

The cap did not change the conclusion. One limit: the uncapped run resumed
three of its eight cells from the attempts that died, and those carry no
convergence record (48 of 128 fits). They were not re-fitted, because the
rule allowed one attempt. So the capped run, whose convergence record is
complete, stays the quoted result and the uncapped run is its corroboration
(§12j).

## 5. Model B — predicting a fault before it is confirmed

**The task.** Take a short window in which no fault has yet been confirmed.
Predict the state a fixed time ahead. Each fault has its own horizon, set
from how fast it develops in the generator: **starvation 1 s, flooding 2 s,
drying 20 s.** Features are each channel's latest value and its trend over
the window. Same logistic model in both arms. 192 fits, **all converged**.

**Which pre-registered criterion fired: "spatial does not help" — for every
fault, in both versions of the starvation pattern, under both ways of
splitting the data.**

| fault | horizon | spatial F1 | bulk F1 | spatial − bulk [95% CI] |
|---|---|---|---|---|
| starvation | 1 s | 0.829 | 0.900 | −0.072 [−0.163, +0.002] |
| flooding | 2 s | 0.870 | 0.947 | −0.077 [−0.120, −0.040] |
| drying *(indicative)* | 20 s | 0.905 | 0.936 | −0.031 [−0.072, −0.002] |

*(Directional starvation arm, split by trace-and-cell. The other three
arm/split combinations agree in sign and size for every fault; all twelve
rows are in §15c and `run_logs/model_b.log`. Results are never pooled across
faults — the horizons differ twentyfold.)*

What this says:

- **Prediction works at these horizons, in both arms.** Every model clears
  chance with room to spare. The horizons — about 1 s, 2 s and 20 s — are the
  phase's finding about *how early*, and they are very short for two of three
  faults. A 30-second horizon, the figure used in the nearest prior work
  under static load, is longer than a starvation or flooding onset here.
- **The grid adds nothing to prediction and costs a little.** For flooding
  the bulk arm is measurably better.
- **Starvation's snapshot advantage does not carry into prediction.** Asked
  only "starvation ahead, or nothing?", both arms score 0.996–1.000.
- **Drying is indicative only.** The pre-set minimum of 30 drying events was
  met as written (33), but only 13 of them produce usable positive windows.
  The minimum had been written against the wrong count; the stricter reading
  was adopted rather than the convenient one.

**Why the bulk arm does so well, and why that is not good news for bulk.**
In this generator a developing fault removes current, so mean current
density falls — the bulk arm reads the fault directly. A real cell under
controlled load does not behave that way: its mean current is whatever is
demanded. The bulk arm's strength is partly a property of the generator.

**The limit of the null.** Both arms used the same regularisation, which
penalises a 652-feature model more than a 4-feature one. Tuning each arm
separately after seeing the result was considered and declined — it would be
choosing the analysis after the outcome. So: *this* model gains nothing from
the grid. That is not a proof that no model could.

**An alarm in the controls.** As a check, both arms were also asked to
predict starvation and flooding 30 s ahead, where no precursor can exist.
Both still beat chance — by reading the load trace, since faults follow load
events. In one control, under the split that holds out a load trace but not
a cell, the spatial arm did better than bulk on windows whose grids are
purely healthy. The likely cause is that the same five simulated cells
appear in every trace, so the spatial arm can recognise the cell. This is
identified, not proven, and it puts a caveat on Model A's leave-one-trace-out
split as well; the trace-and-cell split is the one to quote.

## 6. Phase 5 — do the models stay quiet on the real healthy cell?

Final models were trained on all synthetic data, then shown the real healthy
recordings. A healthy cell should raise no alarm. The script asserts the
real-data loader had not been imported when training finished.

**The claim that was available was narrow**, and is stated before the
result: the real load trace drove the generator, and the healthy template's
form was chosen with the real one in view. A pass would have meant "quiet on
unseen spatial realisations of familiar operating conditions" — nothing
broader.

**Result: fourteen of sixteen models fail.**

| model | spatial input | bulk input |
|---|---|---|
| Model A, directional starvation arm | 96 – 99% false alarms | 85 – 89% |
| Model A, no-axis starvation arm | 7 – 9% on two sessions, 96 – 98% on the other two | 88 – 89% |
| Model B, starvation | **1.0 – 2.7% — passes** | 1.6 – 6.6% — fails |
| Model B, flooding | 4.1 – 18.0% — fails | 6.1 – 7.8% — fails |
| Model B, drying | 93 – 99% | 92 – 94% |

**The models learned the simulator.** Follow-up diagnostics (which change no
verdict) show what that means concretely:

- The pipeline is sound: the same stored models alarm on 0.1–0.6% of
  *synthetic* healthy data.
- **The bulk model learned "high load means drying".** Hold mean current at
  a healthy value, raise only the load level, and its answer flips from
  healthy to drying. In the generator, sustained high load *causes* drying,
  so a healthy cell at high load barely exists in training — 55 rows of
  33,515 above 40% load. The real cell spends 78% of its recording healthy at
  steady load.
- **The spatial model's idea of "healthy" does not contain the real cell.**
  It reads the two recording sessions with flatter profiles as flooded.

The single pass, Model B's starvation predictor with spatial input, is real
as registered but thin: it is silent on the steady-load 78% of the recording
and alarms on 3–13% of the dynamic part.

**A prediction of mine, written into the pre-registration, was wrong.** I
stated the bulk arm would be quiet "nearly by construction". It alarms on
85–89% of real timesteps. I had reasoned about one of its two inputs.

## 7. What it all means

1. **The question as posed gets a mixed and mostly negative answer on this
   generator.** Spatial input helps a snapshot classifier modestly, through
   one fault. It does not help prediction.
2. **The more important result is Phase 5.** Models trained on generated
   faults are not usable on the real cell even to say "nothing is wrong".
   The generator ties each fault to a load regime so tightly that the load
   regime becomes the label. That was noticed early — load alone reached
   0.872 accuracy before Model A was built — and treated as a threat to the
   spatial-versus-bulk *difference*. It was a threat to the *models*.
3. **The spatial-versus-bulk differences survive as comparisons** between
   two arms sharing the same flaw. The absolute accuracies do not survive as
   statements about any real cell.
4. **None of this changes why the phase was closed.** A 324-segment plate is
   a laboratory instrument with no route into a vehicle; that was decisive
   whichever way these numbers fell.

**What stands on real data, independent of all of the above:** the
coefficient of variation of current density is 34 times more load-invariant
than its standard deviation; the healthy spatial pattern is radial,
belongs to the cell and its fixture rather than to the gas routing, and is
stable across sessions (r ≥ 0.976).

**What is not established:** any real-fault accuracy; that bulk signals
suffice on a real cell; that spatial resolution cannot help under another
model class; anything about detection sensitivity from Phase 5, which has no
faults in it.

## 8. Errors caught, and what each would have cost

In rough chronological order. The first seven are from before this campaign
and are detailed in the closure document; the last six are from it.

| error | how caught | cost if missed |
|---|---|---|
| Per-segment amps read as A/cm² | cross-check against the independent bulk current channel | every absolute figure wrong by 6.5×, hidden because ratios looked fine |
| 78% of the "dynamic" recording was steady conditioning holds | inspecting the load trace | a dynamic-load analysis of mostly static data |
| Healthy and fault states built on different spatial bases | a one-number separability check gave 100% at negligible severity | the spatial arm winning near-perfectly for a reason unrelated to faults |
| Spatial arm not converging at the default iteration cap, bulk arm converging | a smoke test that happened to be watched; warnings were suppressed in the real job | a false "no advantage", reported cleanly |
| A speed cap silently equalised the four load traces | reading the numbers under a header that claimed otherwise | a robustness test that never ran, reported as run |
| A commit claimed to untrack third-party-derived files and did not | `git ls-files` | **not avoided** — the files stayed in a public tree until fixed |
| Run logs kept in a temp directory | they vanished | **not avoided** — a multi-hour log lost |
| Drying minimum counted trigger events, not the events the estimate is taken over | a label-only measurement before any fit | drying reported as established on 13 events |
| Negative control assumed "no precursor at 30 s" for every window; 58% and 33% of windows carried residue from earlier episodes | the same label-only measurement | a legitimate spatial signal flagged as an artifact, or the reverse |
| Convergence log kept per process, so a resumed run would report zero failures | reading the resume path before launching | unconverged fits reported as converged |
| The same per-process convergence log, left unfixed in the Model A script | reading the script while its final run was reporting | "all fits converged" quoted for 48 fits it never covered — **partly not avoided**: those fits stay unverified |
| The control alarm above (cell identity shared across traces) | the pre-registered blind-stratum check | a leak favouring the spatial arm, undetected in Model A's trace split |
| My pre-stated expectation that the bulk arm would pass Phase 5 | running the test | believing the bulk arm transfers because the argument sounded right |

The pattern across them: each wrong result would have looked plausible, and
most were caught by checking an operation through a channel other than the
one that performed or described it.

## 9. Reproducing

All from the repo root with `./hfc_env/Scripts/python.exe -u -W ignore`.
Raw data is third-party licensed and not redistributed; see the README for
how to obtain it and regenerate the processed load traces.

| result | script | output |
|---|---|---|
| Event counts, drying minimum, window balance (§13l) | `scripts/spatial/measure_generation_volume.py` | `run_logs/generation_volume.log` |
| Model A (§12, §12j) | `scripts/spatial/phase4_model_a.py` | `run_logs/model_a_results/` |
| Model B (§15) | `scripts/spatial/phase4_model_b.py` | `run_logs/model_b.log`, `run_logs/model_b_summary.json` |
| Phase 5 (§16b) | `scripts/spatial/phase5_false_alarm.py` | `run_logs/phase5.log`, `run_logs/phase5_summary.json` |
| Phase 5 diagnostics (§16d) | `scripts/spatial/phase5_diagnostics.py` | `run_logs/phase5_diagnostics.log` |

Section numbers refer to `docs/spatial/synthetic_generator_notes.md`.
`run_logs/` is gitignored; the scripts regenerate it.

---

## 10. POST-HOC follow-up — read this before quoting anything above

**Everything in this section is post-hoc.** It was done after sections 1–9
were written, in response to problems they exposed. The plan, configurations,
seeds and pass/fail criteria were written and committed before any model was
fitted (§17 of the generator notes, commit `33d78c4`, amended `718034a`);
results are in §18 there. Sections 1–9 are not edited. Where this section
contradicts one of them, **this section is the current statement.**

Still true throughout: every fault is synthetic and **no number here is
real-fault accuracy.**

### 10a. The one conclusion that is overturned

**Section 4 says spatial input helps a snapshot classifier by +0.07 to +0.12.
For a cell the model has not seen, that is wrong in sign.**

The simulated cells were seeded by a cell number alone, so the same five
cells appeared under every load trace. Both ways of splitting the data put
every test cell's spatial template into training. A label-only check
confirmed it: no fold of either split ever held out a cell.

Re-run with the cell held out — same data, same models, after first
reproducing section 4's numbers to three decimals:

| | test cells also in training (section 4) | **cell held out** |
|---|---|---|
| logistic regression, spatial − bulk | +0.10 to +0.13 | **−0.18 to −0.22** |
| random forest, spatial − bulk | +0.07 to +0.08 | **−0.30 to −0.38** |
| spatial macro-F1 | 0.94 | 0.62 |
| bulk macro-F1 | 0.82 | 0.82 |

Every held-out interval lies wholly below zero (for example
−0.192 [−0.307, −0.085]). 528 fits, all converged.

**What section 4 measured was recognition of fault patterns on cells already
seen.** With four training cells and 326 features, the spatial model learns
the cells. This does not show spatial information is useless — one result
with forty cells per trace suggests the gap narrows with more cells — but
with the data this project's conclusions were built on, the advantage is not
there.

I predicted, in writing, that the advantage would shrink and stay positive.
It did not.

### 10b. What stands, and is stronger

| earlier finding | after the follow-up |
|---|---|
| **Model B: the grid does not help prediction** (section 5) | Stands. With the cell held out the spatial arm is worse by 0.17 (starvation), 0.29 (flooding), 0.08 (drying) F1, intervals wholly below zero. It also survives giving both arms the same tuning budget (exploratory; 5,040 fits). |
| **Horizons of 1 s, 2 s, 20 s** (section 5) | Stand, in both arms, with the cell held out. |
| **Drying is indicative only** (section 5) | Stands. A target of 30 contributing events was set in advance; 40 cells per trace reached 23. Ten of the 33 trigger events never contribute whatever the cell. The four load traces are the limit. |
| **Phase 5: the models learned the simulator** (section 6) | Stands. Split by segment type, **no model was quiet while the cell was being driven**; the two passes were on steady load. |
| the three Model A cells with no convergence record (section 4) | Closed: re-fitted, all converged, results match to 0.0002. |

### 10c. Can the Phase 5 failure be fixed? Partly — and not independently tested

One change was made to the training data, once: every simulated cell got a
second, **fault-free** run over the same load profile, so that a healthy cell
working hard exists in training.

- **On synthetic data, with the cell held out:** the bulk predictor keeps its
  recall (starvation 0.86, drying 0.94, flooding 0.92 — a five-point loss on
  flooding only). The spatial predictor loses 10 to 27 points of recall and
  its precision collapses.
- **On the real healthy recording — NOT AN INDEPENDENT TEST**, since the
  change was designed after that recording's failures were examined: eight of
  sixteen models now pass the original criterion, against two. The bulk
  snapshot model goes from 85–89% false alarms to 2–4%. **None is at or below
  5% during driving on all four sessions** (bulk: 5–18%).

So the specific defect is removable, and a healthy real cell under transient
load still does not look like the generator's.

### 10d. The tool

A package, `hfc`, now generates datasets, classifies state and predicts
faults (README; `docs/tool_design.md`; `hfc_models/MODEL_CARD.md`). It uses
a **new generator**, built to remove the two defects above: a fault
redistributes current and lowers voltage rather than removing current; every
cell is unique to one load profile; healthy operation covers the full load
range. The detector calibrates on a healthy stretch of the user's own cell.

Held-out synthetic results (80 unseen cells, unseen load profiles, including
the real FC-DLC cycles; 95% CIs over cells):

| | bulk (default) | spatial (opt-in) |
|---|---|---|
| state, macro-F1 | 0.755 [0.730, 0.776] | 0.989 [0.982, 0.994] |
| false alarms on fault-free runs | 0.35% [0.17, 0.59] | 0.00% |
| starvation ahead, 1 s, F1 | 0.721 [0.619, 0.806] | 0.656 [0.560, 0.741] |
| flooding ahead, 2 s, F1 | 0.147 [0.109, 0.188] | 0.711 [0.666, 0.757] |
| drying ahead, 20 s, F1 | 0.271 [0.168, 0.380] | 0.861 [0.724, 0.951] |

Real healthy recording, **not an independent test**: calibrated within a
session, bulk false-alarms on 1.1–2.7% of samples (0% on steady load,
**5.7–13.7% during driving**); calibrated on a different session, up to 73%.

**The two generators disagree about spatial versus bulk.** On the old one,
with the cell held out, spatial is worse. On the new one it is much better.
Each answer follows from an assumption — whether a fault removes current,
how large its voltage signature is, whether segment noise is independent.
**Neither is evidence about a real cell.** After everything in this
document, that question is open, and only fault-labelled real data can
close it.

### 10e. Errors caught in the follow-up, and what each cost

| error | how caught | cost |
|---|---|---|
| Cells seeded without a trace term, so no split held out a cell | a control experiment raised an artifact alarm; a label-only fold check then showed the overlap | **Not avoided.** The headline spatial advantage was reported for months and is wrong in sign for a new cell |
| My advice to quote the "trace-and-cell" split as the clean one | the same fold check: it leaks more than the split I had blamed | avoided — corrected before anything was built on it |
| My explanation of that control alarm (shared cell identity) | a test registered in advance to be able to falsify it; it fired with the cell held out too | avoided — it is now recorded as falsified and the alarm as unexplained |
| My estimate that 40 cells would yield ~98% of drying events | the label-only count: 23 of 33 | avoided — the target had been fixed in advance and was not lowered, so the result is an honest "underpowered" |
| The tool's abstain reason silently truncated to 24 characters | a test asserting the exact reason string | avoided — out-of-range inputs would have been labelled with a reason no downstream filter matched |
| The generator assumed every real load trace has three dynamic segments | the first full-scale generation crashed | a two-minute run; the failed log is kept |

Pattern, same as section 8: the expensive one was a property nobody thought
to check because the split's *name* said it was handled.

### 10f. Reproducing the follow-up

| result | script | log |
|---|---|---|
| cell-leak confirmation | `scripts/spatial/posthoc_confirm_cell_leak.py` | `run_logs/posthoc/cell_leak.log` |
| Model A, corrected splits | `scripts/spatial/posthoc_model_a.py --dataset capped` | `ws1_model_a_capped.log` |
| Model B, corrected splits | `scripts/spatial/posthoc_model_b.py --dataset v1 --splits S2,S3 --cfgs all --binary` | `ws1_model_b_v1.log` |
| convergence re-fit | `scripts/spatial/posthoc_ws2_model_a_refit.py` | `ws2_model_a_refit.log` |
| Phase 5 by segment type; re-test | `scripts/spatial/posthoc_phase5.py --mode split` / `--mode v2` | `ws3a_phase5_split.log`, `ws3c_phase5_v2.log` |
| companions: Model B, paired cost, Model A | `posthoc_model_b.py --dataset v2 …`, `posthoc_ws3d_paired.py`, `posthoc_model_a.py --dataset v2` | `ws3d_*.log` |
| matched tuning | `scripts/spatial/posthoc_ws4_nested.py` | `ws4_nested.log` |
| drying power | `posthoc_checks.py --check drying`, `posthoc_model_b.py --dataset dry40 …` | `ws5_*.log` |
| the tool | `python -m hfc …` (README) | `run_logs/hfc/*.log` |

Launch through `scripts/spatial/posthoc_run.sh <name> <script> [args]`, which
writes a log ending `EXIT=<code>` and refuses to overwrite an existing one.
6,744 logistic fits across the follow-up, none non-converged.
