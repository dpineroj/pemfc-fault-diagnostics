# Phase closure — the spatially-resolved fault-detection line

**Status: CLOSED, November 2026.** Completed, not abandoned. This document ends
the line deliberately so that what it established stays legible and
attributable, and so that a reader arriving later can tell which claims it
supports and which it never did.

**Amended November 2026 (§8):** the experimental arc was afterwards run to
completion through Phase 5 — the predictive model and the false-alarm test on
the real cell. The closure and its reasoning are unchanged. §8 changes how
§1's result should be read, and says how.

The work is not deleted. It becomes background to the active line (bulk-signal
health-state estimation and remaining-useful-life prediction under dynamic
load). Every section below describes a finished thing.

---

## 1. The question, and the answer

**The question.** Does a spatially resolved grid of current-density
measurements improve predictive fault detection over bulk signals alone, under
realistic dynamic automotive loading, for flooding, drying and
oxygen starvation?

**The answer, as measured.** Yes, but narrowly, and smaller than the framing
implied.

On a snapshot four-class classifier (healthy / drying / flooding /
starvation), spatial resolution beat bulk-only by **+0.07 to +0.12 macro-F1
depending on estimator** — +0.103 to +0.123 under multinomial logistic
regression, **+0.066 to +0.082 under a random forest**. The honest figure is
the range, not the upper end: the stronger estimator raises both arms but
raises bulk more, closing roughly **a third** of the logistic delta. So part
of the apparent gap was the 2-feature linear decision boundary rather than
information the bulk channel does not carry.

**The advantage was almost entirely starvation.** Per-class, bulk handled
flooding and drying competently and failed specifically on starvation:

| class | spatial precision / recall | bulk precision / recall |
|---|---|---|
| healthy | 0.885 / 0.984 | 0.812 / 0.998 |
| drying | 0.986 / 0.929 | 0.971 / 0.837 |
| flooding | 0.994 / 0.982 | 0.947 / 0.874 |
| **starvation** | **0.875 / 0.934** | **0.505 / 0.698** |

Bulk calls starvation correctly about half the times it says so. Remove
starvation and the case largely evaporates.

The result was obtained with the falsification criteria fixed in writing
before implementation (§12f), under grouped splits, with per-fit convergence
asserted, and with the delta stable across an amplitude sweep that zeroed the
constants most likely to have manufactured it. **It is a real result about this
generator.** Section 4 states precisely what it is not.

---

## 2. Why the line is closed

Two reasons. They are independent, they point the same way, and neither is a
matter of taste.

### 2a. The economic objection — no deployment path

A 324-segment current-density mapping plate is a **laboratory instrument**. No
production vehicle carries 324 current sensors, at any price, and practical
fuel-cell systems are designed toward *fewer* sensors because sensors cost
money and add failure modes.

So the finding "spatial beats bulk" is academically clean and operationally
inert. **This is an objection to the premise, not to the execution**, and the
distinction matters: it cannot be answered by measuring more carefully, by a
better classifier, or by more data. A larger delta would not help. The result
could be entirely correct and still have nothing to deploy into.

That is the decisive reason, and it was decisive regardless of how the numbers
had come out.

### 2b. The evidentiary objection — the advantage was narrower than the framing

Independently of deployability, the evidence came in weaker than the project's
own headline:

- the delta was **concentrated in one fault type** (§1), so the general claim
  "spatial resolution distinguishes fault modes" overstates a result that is
  really "spatial resolution distinguishes starvation";
- a random forest **closed about a third** of it, locating part of the gap in
  the estimator rather than in the information;
- **no real fault-labelled spatially-resolved data exists or is obtainable
  without hardware.** Roughly six months went into building a defensible
  synthetic substitute and documenting precisely what it could not establish.

The last point is the structural one, and it recurred from every direction the
project approached it: the fault-diagnosis literature has no dynamic-load
labelled spatial data, and both external fault-labelled datasets audited
(`docs/spatial/external_fault_datasets.md`) turned out bulk-only **and** static-load.
The same wall, hit three times.

### 2c. What this closure is not

**It is not a pivot forced by a preference.** Prof. Litster's feedback
identified the economic objection, but the objection stands on its own terms
and the project's own measurements had already narrowed the case. The
conclusion the evidence supports is that this line produces a correct result
with no route to use, and that is a reason to close it rather than to continue.

Recorded this way deliberately: a closure justified by "someone senior
preferred otherwise" is not reviewable, and a closure justified by evidence is.

---

## 3. What was established, and stands independently of the framing

These are **findings, not scaffolding**. Each is true whether or not anyone
pursues spatial fault detection, and each is attributable to this phase.

**The load-invariance result.** Raw σ(J) tracks how hard the cell is being
driven, not how healthy it is: measured CoV of the metric itself across a real
FC-DLC run is **33.6%** for σ(J) against **1.0%** for CV(J) = σ(J)/mean(|J|) —
**34× more load-invariant**, on n = 17,681 real timesteps. This is the
phase's most portable finding and it becomes the *central* contribution of the
RUL line rather than a side correction, because the dynamic-load RUL
literature names load-confounded health indicators as its open problem.

**The real cell's healthy spatial structure.** The persistent template is
radially symmetric — a "bathtub", low at the centre, high at the edges — and
**not** an inlet-to-outlet flow-path gradient. It is a property of the cell and
its fixture, not of any run: cross-session template correlation **r ≥ 0.976**
across four independent recording sessions on different days under four
different gas inlet/outlet configurations. With a measured ceiling: a *free
arbitrary function of radius* reaches **R² = 0.9416**, so **5.84% of the
template's variance is irreducibly non-radial**, and the adopted
three-parameter logistic form achieves 0.9368 — 99.5% of everything radially
explainable.

**The residual direction is fixture-locked, not flow-locked.** After removing
all radially-explainable structure, the residual *is* directional (row/column/
diagonal correlations 0.11–0.43, against 0.023–0.104 on the full template).
But reversing a gas feed is physically a reflection of that gas's axis, and
identity beat every candidate mirror in all four reversal pairs (+0.885 to
+0.990 against the best mirror's +0.364), with **85.3% of residual energy
config-invariant**. Consistent with compression/contact resistance; not with
reactant depletion.

**Per-fault precursor windows, and what they do to Kim et al.'s horizon.**
Measured from onset to the severity threshold crossing: starvation
**1.2–3.9 s** (median 2.1), flooding **2.3–8.4 s** (median 4.1), drying
**25.9–67.3 s** (median 47.8). At a **30 s** lead — the prior art's prediction
horizon — starvation and flooding severity is **identically zero for every
episode**, because their entire onsets are shorter than the lead time. Only
drying retains a precursor that far out. **So Kim et al.'s 30 s horizon does
not transfer to dynamically-triggered faults**: under 10-hour static holds 30 s
is a negligible fraction of onset, whereas here it exceeds two of three onsets
entirely. That is a finding about comparability to the prior art, and it
survives the closure.

**A separability identity worth reusing.** Best-single-threshold accuracy
between two equal-size samples is exactly **0.5 + D/2**, where D is the
two-sample Kolmogorov–Smirnov statistic. The null is therefore
distribution-free and depends only on N. Used here to calibrate an acceptance
threshold against a measured null instead of intuition; general-purpose.

**Two dataset audits with transferable conclusions.** Zhou et al. (2022) and
Kim et al. (2023) are both **static-load**, neither spatially resolved, neither
capturing fault onset as a transition. The apparent structural reason:
inducing a *controlled, labelled* fault requires holding conditions fixed,
which conflicts with capturing it under dynamic load, where the label becomes
ambiguous about when the fault began. That explains *why* the gap exists rather
than merely asserting it.

---

## 4. What was never validated, and never will be

Quoted verbatim from §11a of `docs/spatial/synthetic_generator_notes.md`, **and still
accurate as written**:

> **This is a permanent limitation, not a gap pending future work within this
> project's scope.** It is not resolved by more careful modelling, a better
> literature search, or additional analysis of the existing dataset. It is
> resolved only by fault-labelled spatially-resolved data, which this project
> has established does not exist and cannot be produced without hardware.
> Stated in the same register as §6d: the limitation is to be reported, not
> managed away.

Concretely, and for the avoidance of any doubt in later writeups:

- **Every fault label in this phase is synthetic.** The real dataset is
  healthy-only and was reserved as the false-alarm validation set.
- The fault spatial patterns are **mutually coherent and literature-grounded,
  not validated**. There was no capability check, no identifiability check and
  no resolution corroboration available for them, because there was nothing
  real to check against.
- The §1 delta is therefore **a delta between two arms on coherent synthetic
  data**, not a measured real-fault detection accuracy. It must never be
  quoted as the latter.
- The healthy family was grounded physically and **never fit to the real
  healthy data**, by design, because fitting it would have made the
  false-alarm test a tautology.

The closure does not weaken any of this, and nothing in the RUL line inherits
it — the new direction has real labels, which is most of why it is viable.

---

## 5. The errors caught, and what each would have cost

The most transferable part of this record. Each was caught before it reached a
conclusion; each would have produced a confident, wrong, plausible-looking
result.

**A units bug in the real-data loader.** Raw per-segment amps were being read
as A/cm², making every absolute current-density figure low by ≈6.48×. Caught by
an independent cross-channel check: integrating the grid and comparing against
the bulk file's own `INTENSIDAD` channel gave a ratio of 1.0053 with no area
factor, and 0.1551 with one. *Cost avoided*: every absolute σ(J) figure in the
project wrong by a factor of six, while all the ratio-based conclusions looked
fine — the most insidious kind, because CV(J) is scale-invariant and would have
masked it.

**Conditioning-hold contamination.** **231 of 295 minutes** of the real FC-DLC
recording were pre-test steady-state conditioning holds, not driving. *Cost
avoided*: a "dynamic load" analysis in which 78% of the data was not dynamic,
with ramp-rate and dwell statistics dominated by an artifact.

**The basis mismatch.** The healthy baseline was rebased to a radial template
while fault states still interpolated notebook 01's *diagonal* gradient. A
single scalar — correlation with distance-from-centre — separated healthy from
fault at **100% accuracy at severity 0.01**, before any meaningful fault
existed. *Cost avoided*: the spatial arm scoring near-perfectly on a feature the
bulk arm is structurally blind to, confirming the project's central hypothesis
for entirely the wrong reason. This produced the standing test now in §10.

**A suppressed convergence failure.** The 326-feature spatial arm did not
converge at `max_iter=300` while the 2-feature bulk arm did — an
**arm-asymmetric** failure that understates the spatial arm specifically and
drives the delta toward zero. It surfaced only because a smoke test printed to
a terminal that happened to be under observation; the real job runs with
`-W ignore`, so sklearn's warning was actively suppressed. *Cost avoided*: a
false negative, reported as a clean one, that would have fired §12f's "no
headroom" branch for a reason with nothing to do with the hypothesis. **The
catch was luck, not process**, which is why the fix is a per-fit assertion
rather than a raised limit, and why "a suppressed warning is not a weak signal,
it is no signal" is now recorded.

**A runtime constant that violated an instruction by side effect.**
`MAX_PER_GROUP = 2500`, introduced for speed, silently equalised all four load
traces and erased the very asymmetry the leave-one-trace-out split existed to
expose. The reporting compounded it by printing "the asymmetry, NOT equalized"
above four identical numbers. *Cost avoided*: a robustness test that did not
run, reported as though it had.

**A commit that asserted a state it did not achieve.** `188cd2a` claimed to
stop tracking two CC-BY-NC-derived npz files. Supplying pathspecs to
`git commit` commits working-tree state for those paths and silently discards
staged deletions, so the files stayed tracked while the message and the docs
both said otherwise — and `.gitignore` has no effect on tracked files.
*Cost*: not avoided. Third-party derived data sat in a public tree for the
interval, and the record was wrong until `f8688c2` fixed it. The practice that
follows — verify tracking changes with `git ls-files`, because a commit
reporting success is not evidence the index changed — is in `CLAUDE.md`.

A seventh, recorded for completeness because it was a loss rather than a catch:
a multi-hour run log and a deliberately preserved copy of a superseded run were
both lost to temp-directory clearing. The protection failed on **where** it was
put, not whether it was done.

---

## 6. What carries into the RUL line, and what does not

**Carries over.**

- The **load-invariance finding** and its reasoning — promoted from side
  correction to central contribution.
- **FC-DLC domain knowledge** and the load-trace machinery:
  `extract_load_trace.py`'s hold detection, ramp-event and dwell detection,
  dynamic-segment extraction and concatenation. All of it operates on
  `(t, load)` arrays and is protocol-level, not dataset-specific. The
  conditioning-hold detection is the highest-value transfer, since a 1008-hour
  durability run is very likely to contain characterisation interruptions.
- The **reusable half of `load_real_data.py`**: comma-decimal European locale
  handling, timestamp assembly, nearest-timestamp joining, explicit NaN
  accounting, and above all the practice of validating a derived channel
  against an independent one.
- **Data-engineering discipline**: shape assertions, grouped and chronological
  splits, cross-channel validation.
- **Methodological practice**: pre-registered criteria, measured nulls, the
  standing question *does this choice decide the comparison before it is
  measured*, per-arm fit-quality assertions, and the
  real / literature-grounded / tuned tagging convention.
- `docs/real_data_notes.md` as the template for how a dataset gets documented.

**Does not carry over.** Background from here:

- the healthy spatial template model — logistic radial form, bounds,
  identifiability (§5–§7)
- the fault generator and its three onset models (§8, §11, Phase 3b)
- the spatial-vs-bulk two-arm comparison as the headline experiment (§12)
- Model A and Model B as previously scoped

Kept, attributable, and no longer on the critical path.

---

## 7. Status of the final Model A run

A re-run without the `MAX_PER_GROUP` cap (§5) was attempted after the line was
closed, to test whether the cap had changed the conclusion. **It died twice,
both times with exit code 4 and no traceback**, and was then **stopped
deliberately rather than relaunched a third time.**

**Corrected statement.** An earlier version of this section said the run "was
allowed to finish rather than be killed". That was written while it was still
running and is **false**; it is corrected here rather than quietly amended,
because this document's entire value is that it is accurate, and a false line
in it is worse than a gap.

**Configuration**: uncapped, **85,960 samples** (against 50,000 capped), with
Inverse_Flow carrying ~2.1× any other trace — the asymmetry the capped run had
erased. Four amplitude scales, both split configurations, both starvation arms,
full 326-feature spatial arm. Results persist incrementally per
`(arm, scale)`, so the run is resumable and a mid-run failure costs one cell.

**What the partial run covers.** 3 of 16 cells persisted — the **directional
arm only**, at amplitude scales 0.00, 0.25 and 0.50, both split
configurations. 85,960 samples, with Inverse_Flow at its real ~2.1× weight
rather than the capped run's equalised 25%:

| scale | split | spatial | bulk | delta |
|---|---|---|---|---|
| 0.00 | cell | 0.943 | 0.830 | +0.113 |
| 0.00 | trace | 0.942 | 0.818 | +0.124 |
| 0.25 | cell | 0.946 | 0.831 | +0.115 |
| 0.25 | trace | 0.945 | 0.818 | +0.127 |
| 0.50 | cell | 0.948 | 0.831 | +0.117 |
| 0.50 | trace | 0.947 | 0.819 | +0.128 |
| 1.00 | cell | 0.950 | 0.832 | +0.118 | *(logged, not persisted — the death came during this cell's trace split)* |

**What it does NOT cover, stated so nothing over-reads it**: the complete
scale 1.00 cell, and **the entire no_axis arm**. No convergence summary was
printed, because the run never reached its reporting section.

**It is corroboration, not a replacement.** The deltas are +0.113 to +0.128
where it ran, consistent with the capped run's +0.103 to +0.130, which answers
the robustness question the uncapped run existed to ask. **The capped run —
complete across all 16 cells — remains the phase's result, and no figure
anywhere should cite uncapped numbers as the phase's.**

**Why it was stopped rather than relaunched.** A third multi-hour attempt,
after two unexplained deaths, on a closed phase, to refine a number that was
already corroborated, is sunk cost. The incremental writes added for the
second attempt are what make this a partial result rather than nothing at all
— the first attempt died having materialised no output whatsoever.

The capped-vs-uncapped comparison, with its provenance limits, is recorded in
§12h of `spatial/synthetic_generator_notes.md`.

---

## 8. Amendment — the experimental arc completed through Phase 5

**Added November 2026, after the closure.** The line stays closed. What
changed is that the experiments it had scoped and not run — the predictive
model (Model B) and the false-alarm test on the real cell (Phase 5) — were
run to completion, so that the phase ends on measurements rather than on
intentions. Sections 1–7 are left as written; where this section changes how
one of them should be read, it says so here rather than editing it there.

Full record: §13–§16 of `spatial/synthetic_generator_notes.md`. Single
cold-readable account: `spatial/final_results.md`.

### 8a. Model B — prediction, at each fault's own horizon

Model B asks the question the project was actually framed around: given a
window in which **no fault is yet confirmed**, will one be confirmed a fixed
time ahead? Design, horizons, splits and falsification criteria were written
and committed before any model code (§13, commit `fc3fb28`). All 192 fits
converged.

Results are per fault at that fault's own horizon and are **never aggregated
across faults**, because the horizons differ by a factor of twenty.

| fault | horizon | spatial F1 | bulk F1 | spatial − bulk [95% CI] |
|---|---|---|---|---|
| starvation | 1 s | 0.829 – 0.865 | 0.897 – 0.901 | −0.035 to −0.072; intervals reach zero in three of four arm/split cells |
| flooding | 2 s | 0.870 – 0.880 | 0.945 – 0.951 | −0.065 to −0.080; **every interval wholly below zero** |
| drying *(indicative)* | 20 s | 0.905 – 0.912 | 0.932 – 0.936 | −0.020 to −0.031 |

**The pre-registered criterion that fired is "spatial does not help" — for
every fault, in both starvation arms, under both splits.** The bulk arm
predicts at least as well everywhere and measurably better for flooding.

**Both arms achieve the horizon for all three faults.** The per-fault
precursor windows of §3 were a separability measurement; they are now
confirmed by a trained predictor.

Drying is *indicative, not established*: its pre-registered minimum of 30
trigger events was met as written (33), but only 13 events contribute
positive windows at its configuration, and the minimum had been written
against the wrong count. The favourable reading was not taken.

**How this changes the reading of §1.** §1 reports that spatial resolution
helped a *snapshot* classifier, almost entirely through starvation. Model B
shows that advantage **does not extend to prediction**: per the branch fixed
in advance (§11e, branch 3), *starvation's predictive spatial advantage is
not demonstrated by this work*. Asked only "starvation ahead, or nothing?",
both arms score 0.996–1.000 — two bulk features answer it.

The honest limit of that null: it is a statement about a level-and-trend
linear model under equal regularisation in both arms, which costs the
652-feature arm more than the 4-feature one. Tuning per arm after seeing the
result was considered and declined, because it would have been choosing the
analysis after the outcome. The null shows that *this* model gains nothing
from the grid, not that no model could.

### 8b. Phase 5 — the false-alarm test on the real healthy cell

The real spatial dataset is healthy-only and was held out of every training
and selection step for this purpose; the script asserts the real-data loader
had not been imported when the final models were trained. Criterion, fixed
and committed before any Phase 5 code existed (§14, commit `fc3fb28`): false-alarm rate
≤ 5% on **all four** real recording configurations; > 50% means the model
learned the simulator.

**The claim that was available was narrow, and is stated at the result.**
The real load channel drove generation, and the healthy family's form was
chosen with the real template in view. A pass would therefore have meant
quietness on *unseen spatial realisations of operating conditions the models
had seen* — not on unseen operating conditions, and not structural
independence.

**Fourteen of sixteen models fail.**

| model | spatial input | bulk input |
|---|---|---|
| Model A (snapshot), directional | 96 – 99% — simulator-learned | 85 – 89% — simulator-learned |
| Model A (snapshot), no_axis | 7 – 9% on two sessions, 96 – 98% on two | 88 – 89% — simulator-learned |
| Model B starvation | **1.0 – 2.7% — PASS (both arms)** | 1.6 – 6.6% — fail |
| Model B flooding | 4.1 – 18.0% — fail | 6.1 – 7.8% — fail |
| Model B drying | 93 – 99% — simulator-learned | 92 – 94% — simulator-learned |

**The models trained on this generator do not stay quiet on the real healthy
cell.** Diagnostics run afterwards (verdicts unchanged) show why:

- The prediction path is sound — the same stored models alarm on 0.1–0.6% of
  *synthetic* healthy rows.
- **The bulk arm learned the generator's trigger rule.** At a healthy mean
  current density, changing the load level alone flips its output from
  healthy to drying. In the generator, sustained high load *causes* drying;
  healthy operation above 40% load is 55 of 33,515 training rows. The real
  cell spends 78% of its recording healthy at sustained constant load.
- **The spatial arm's healthy region does not contain the real cell.** It
  reads the two flatter recording sessions as flooded, and whether it reads
  the other two as healthy depends on which starvation arm was in training.

The one pass — Model B's starvation configuration with spatial input — is
real as registered and should not be over-read: its false-alarm rate is
exactly zero on conditioning holds (78% of the timesteps) and 3.4–12.7% on
the dynamic segments.

**A pre-registered expectation of my own was wrong.** §14b predicted the bulk
arm would be quiet "nearly by construction". It alarms on 85–89% of real
timesteps. Recorded with the result; the prediction is not edited.

**How this changes the reading of §1.** The §1 delta compares two arms that
share this defect, so it is not void as a comparison. But Phase 5 extends §4
from the fault *patterns* to the *models*: no absolute accuracy from Model A
or Model B says anything about a real cell, healthy or faulted.

### 8c. The closure reasoning is unchanged

**The line was closed for the reason in §2a — a 324-segment plate has no
deployment path — and that reason does not depend on any number in this
section.** It was decisive when the snapshot result favoured the spatial
arm, and it would have been decisive had Model B favoured it too.

The new results bear only on §2b, the evidentiary objection, and they
tighten it: the advantage was narrow in snapshot classification, is absent
in prediction, and the models do not transfer to the real cell. They are
recorded as the completion of the arc, not as a new argument for a decision
already made on other grounds.

### 8d. What the phase does and does not establish, complete

**Establishes** (all on this generator unless marked *real*):

- *real* — CV(J) is 34× more load-invariant than σ(J).
- *real* — the healthy template is radial, fixture-locked, and stable across
  sessions (r ≥ 0.976).
- Per-fault precursor horizons of roughly 1 s, 2 s and 20 s, now confirmed by
  a trained predictor in both arms; a 30 s horizon does not transfer to
  dynamically triggered faults.
- A snapshot classifier gains +0.07 to +0.12 macro-F1 from spatial input,
  depending on estimator, almost entirely on starvation.
- A predictive classifier gains nothing from spatial input, and loses a
  little.
- *real* — classifiers trained on this generator false-alarm on the real
  healthy cell; they encode the generator's load-to-fault rule.

**Does not establish:**

- any real-fault detection or prediction accuracy, for either arm;
- that bulk signals suffice on a real cell — the bulk arm's strength rests
  partly on the generator expressing a fault as lost mean current, which a
  real galvanostatic cell does not do;
- that spatial resolution cannot help prediction under a different model
  class;
- anything about detection sensitivity from Phase 5, which contains no
  faults.

### 8e. The permanent limitation, restated

§11a of `spatial/synthetic_generator_notes.md`, verbatim, and still accurate
as written:

> **Phase 3b produces mutually coherent, literature-grounded fault patterns. It
> does NOT produce validated ones.** No downstream result may describe fault
> spatial patterns as validated, and no writeup may imply the fault side
> received the treatment the healthy side did.
>
> **This is a permanent limitation, not a gap pending future work within this
> project's scope.** It is not resolved by more careful modelling, a better
> literature search, or additional analysis of the existing dataset. It is
> resolved only by fault-labelled spatially-resolved data, which this project
> has established does not exist and cannot be produced without hardware.
> Stated in the same register as §6d: the limitation is to be reported, not
> managed away.

**No Model A or Model B number in this document or any other is real-fault
accuracy.**
