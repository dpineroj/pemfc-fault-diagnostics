# External fault-labelled datasets — audits and decisions

This project's own real dataset (Toharias et al., `docs/real_data_notes.md`) is
**healthy-only**, and it is reserved as the Phase 5 false-alarm validation set.
Every fault label the project generates is synthetic. This file records audits
of *external* fault-labelled PEMFC datasets, and the decisions taken on each.

A decision not to integrate a dataset is recorded here with its reasoning, so
that it reads as a judgement that can be revisited rather than as the dataset
being unusable.

---

## 1. Zhou et al. (2022) — AC voltage response / EIS

**Citation**: Zhou, S. et al., *"Fault diagnosis of PEMFC based on the AC
voltage response and 1D convolutional neural network"*, Cell Reports Physical
Science. Data: Mendeley Data, DOI
[10.17632/yhg3ks3jzc.1](https://doi.org/10.17632/yhg3ks3jzc.1).
Companion code (NOT vendored into this repo):
`github.com/Shangwei-ZHOU/Fuel_Cell_Diagnosis`.

**Licence: CC BY 4.0 — attribution only, no non-commercial restriction.**
Confirmed on the Mendeley landing page for the DOI. This is *less* restrictive
than the Toharias dataset's CC-BY-NC, so anything built on Zhou et al. is not
bound by the NC clause that governs the Seville data — the two must not be
conflated when terms are being reasoned about.

Note for anyone re-checking: **there is no LICENSE or README file inside the
archive**, and that is normal for Mendeley Data — terms live on the landing
page, not in the download. The absence is not an open question and should not
be re-raised as one.

**Location**: `data/raw/zhou_ac_voltage/` (gitignored, same convention as the
Seville data). **Retained deliberately** even though it is not integrated, so
it remains available if a later use emerges.

### 1a. What is in it

12 files, all **MATLAB MAT5/v7** (version flag `0x0100`) — *not* v7.3/HDF5, so
`scipy.io` reads them directly and **no new dependency is required**.

| set | Normal | Flooding | Drying | rows | content |
|---|---|---|---|---|---|
| EIS spectra | 300 | 270 | 300 | 15 freqs × 2 | cell array, col 0 = Re(Z), col 1 = Im(Z), ohms |
| sequential original | 300 | 270 | 300 | 40000 | AC voltage response, 15-frequency sweep |
| sequential downsampled | 300 | 270 | 300 | 16800 | same runs, resampled |
| multi-sine | 600 | 600 | 600 | 10001 | ~12 superposed sine components |

Columns are instances, rows are time samples. Confirmed two independent ways:
the EIS instance counts match the sequential column counts exactly (same runs,
two representations), and columns are distinct records rather than repeats
(corr between the first few columns: +0.03, −0.44, +0.24, +0.20).

Structural details established by inspection rather than from the paper:

- Each sequential record resolves into **15 blocks of ~2667 samples with
  monotonically descending frequency** (normalized 0.498 → 0.00038, roughly
  1.6× per step). 15 matches the EIS frequency-point count exactly, which is
  what identifies these records as swept EIS.
- **Downsampling is resampling, not decimation**: 16800 = 15 × 1120 exactly,
  while 40000/15 = 2666.67. Each frequency block was resampled to a fixed
  1120 samples. The 2.381 row ratio is non-integer.
- **Absolute sampling rate is not recorded**, so record duration cannot be
  stated from the data. (If the top frequency is ~1 kHz then fs ≈ 2 kHz and a
  record is ~20 s, consistent with the lowest block holding exactly one cycle
  — but that is inference from a round number, not measurement.)
- Coarse quantization: 1.6e-4 V LSB, only 657 unique values across ±0.058 V.
- The stored signal is the **AC perturbation response, DC-stripped**.
  Multi-sine columns have mean exactly 0; sequential means are ~1e-4 against a
  ±0.058 range.

### 1b. Findings about the dataset

Recorded as facts about Zhou et al., not as shortcomings relative to our
needs:

- **No severity levels.** One condition per class, defined by an RH setpoint
  (Normal 60% RH, Flooding 85–90%, Dehydration 30–35% — from the paper). The
  structure is three-class categorical, not graded.
- **No operating-condition channels.** Every file contains *exactly one*
  variable. No time vector, no current or load, no temperature, no RH, no
  stoichiometry. All operating conditions are paper-only and cannot be
  recovered from the data.
- **Labels are filename-encoded only.**
- **No fault-onset transitions.** Each record is a sweep at a held condition,
  so onset timescales are not directly measurable from this dataset.
- **The drift that does exist is experimental, not fault development.**
  Impedance drifts monotonically with instance index in all three classes —
  and **Normal drifts as hard as the fault classes** (r = +0.90). Attributing
  the drift to fault progression would therefore be wrong; it is more
  consistent with thermal equilibration, conditioning, or slow water
  redistribution.

### 1c. DECISION — not integrated as the Phase 4 bulk arm's training anchor

**Decided, with reasoning, not a finding that the dataset is unusable.**

The candidate role was to anchor the Phase 4 **bulk** arm against real
fault-labelled data, so that the headline spatial-vs-bulk comparison would not
be synthetic-versus-synthetic. That role is declined, for three reasons:

1. **It is silent exactly where external grounding is most needed.** Zhou et
   al. covers normal / flooding / drying. It contains **no starvation**. Per
   §11i, starvation is simultaneously the novelty extension over Kim et al.,
   the fault with the least defensible spatial signature, and the one whose
   no-axis arm collapses into flooding (0.638 against a 0.95 threshold). The
   one fault most in need of an external anchor is the one this dataset cannot
   provide.
2. **It is static-load**, established structurally rather than from the paper:
   no load channel exists anywhere in the download, EIS sweeps presuppose a
   stationary DC bias (a sweep across a moving operating point is not an
   impedance), and the multi-sine files are named `VR_Fixed_*`. A bulk arm
   anchored here would be anchored **under static conditions only** — the same
   limitation as Kim et al., and dynamic loading is half this project's
   novelty claim.
3. **Partial anchoring is worse than none, because it invites over-reading.**
   A result anchored this way would be externally grounded for two fault types
   in one load regime and purely synthetic everywhere else, with **nothing in
   the headline number marking the difference**. A reader — including a future
   version of this project — would be invited to credit the whole comparison
   with grounding that only part of it has. An openly synthetic bulk arm is
   easier to report honestly than a selectively anchored one.

This decision can be revisited if the project's claim narrows to flooding and
drying under static load, which is one of the outcomes §11e branch 3
contemplates. The data is retained for that reason.

### 1d. Split-design evidence — now justified by a real measurement

The strongest thing this audit produced, and it survives the decision above.

**Instance index correlates strongly with the discriminative features inside
every class**, measured on the EIS spectra:

| class | Re(Z) @ high freq vs index | Re(Z) @ low freq vs index | detrended lag-1 |
|---|---|---|---|
| Normal | r = **+0.903** | r = +0.899 | +0.762 |
| Flooding | r = +0.702 | r = **+0.905** | +0.815 |
| Drying | r = **−0.801** | r = +0.837 | +0.927 |

Flooding's low-frequency resistance drifts by **+33% of its own mean** across
the record. Neighbouring instances are therefore near-duplicates, and instance
index is itself informative about the target.

**This explains the paper's own accuracy gap as leakage.** Zhou et al. report
**88.46%** with a chronological train/test split and **96.15%** with a random
split on *identical data*. A random split places near-duplicate neighbours on
both sides of the boundary; the 7.7-point gap is the size of that leak. The
chronological figure is the honest one.

**Consequence for Phase 4**: our synthetic data has the same adjacency
property — consecutive timesteps within an episode are near-duplicates by
construction — so **our Phase 4 split must be chronological or grouped, never
random**. That requirement was previously an argument; it is now backed by a
measured number from real data.

**Zhou et al.'s 100% multi-sine figure must be treated as UNVERIFIED.**
Nothing about multi-sine excitation removes the drift, and no chronological
check was reported for it. **If this work is ever used as a benchmark,
benchmark against 88.46%**, not against 96.15% or 100%.

### 1e. Observation only — a real-data reference point for drying's direction

High-frequency real part, mean over instances:

| | Normal | Flooding | Drying |
|---|---|---|---|
| Re(Z) @ high freq (Ω) | 0.001514 | 0.001611 | **0.001816** |

Ordering **Normal < Flooding < Drying**, i.e. dry-out raises the ohmic /
high-frequency resistance most, which is the expected physics. Our generator's
drying direction — suppress the hottest, driest region, raising resistance and
CV(J) — is **consistent with** this ordering.

**Recorded as an observation, not as grounds to change anything.** It is a
different cell, a different measurement modality, and a bulk rather than
spatial quantity, so it cannot calibrate an amplitude or validate a pattern.
Any generator change goes through the §11 acceptance process.

(For completeness, the low-frequency real part orders the same way: 0.0138 <
0.0185 < 0.0218. And drying's high-frequency resistance *falls* across
instances while its low-frequency resistance rises — not the uniform monotonic
worsening our generator assumes. Also an observation only.)

---

## 2. Documented negative result — both known fault-labelled datasets are static-load

**This substantiates §11a's permanent-limitation claim rather than merely
asserting it, and it is now a second confirmed instance.**

Two independent real, fault-labelled PEMFC datasets have been examined:

| | fault conditions | load regime | onset captured? |
|---|---|---|---|
| **Kim et al. (2023)** | flooding, drying | **static** — 10-hour holds at fixed current density | no |
| **Zhou et al. (2022)** | flooding, dehydration | **static** — EIS sweeps at held RH setpoints | no |

Neither is spatially resolved, neither covers starvation, and **neither
captures fault onset as a transition**. Both record each condition as a held
steady state.

**The apparent structural reason, stated as the explanation it is**: inducing
a *controlled, labelled* fault appears to require holding operating conditions
fixed — a known RH setpoint, a known current density, held long enough for the
condition to establish and be confidently labelled. That requirement is in
direct tension with capturing the same fault under *dynamic* loading, where
conditions are by definition moving and the label becomes ambiguous about when
the fault began. The two goals pull against each other experimentally.

So the gap this project targets is not an oversight in the literature that
better searching would close. It appears to follow from how labelled fault
data has to be produced. That is a stronger basis for the novelty claim than
"no one has done it", and simultaneously a stronger basis for §11a's
limitation: **the data that would validate synthetic dynamic-load fault
patterns is not merely absent, it is difficult to produce by the means that
produce labelled fault data at all.**

This does not upgrade §11a from permanent to anything else. It explains why it
is permanent.


---

## 3. Derived load traces are no longer tracked — an earlier precedent reversed

**Supersession, not a silent change.** Until this point
`data/processed/normal_flow_fc-dlc_load_trace_full.npz` and
`..._dynamic_only.npz` were **committed to the repository**. Phase 4's adoption
of all four gas configurations (§12c) would have extended that precedent from
two files to five, which prompted examining it rather than inheriting it. It
was reversed: `data/processed/` is now gitignored and the two tracked files
were removed from the index with `git rm --cached` (they remain on disk).

Two reasons.

**1. Consistency of licence handling.** The repository is public and these
traces are **derived from CC-BY-NC third-party data** (Toharias et al.).
Redistribution of a normalized bulk scalar channel is *probably* permitted,
but **"probably permitted under someone else's licence" is a weaker position
than "we don't redistribute third-party data"**, and the project already
takes the stronger line elsewhere: the Zhou data is gitignored despite being
**CC BY 4.0, i.e. MORE permissive**. Treating NC-derived data more liberally
than permissively-licensed raw data was inconsistent, and the inconsistency
was the argument for changing it.

**2. Stale-artifact risk on regenerable files.** The traces are
**deterministically regenerable** by `scripts/extract_load_trace.py` from
`data/raw/`. Tracking regenerable artifacts whose *inputs* are not shipped
invites exactly one failure mode: the extraction logic changes and the
committed artifact silently does not. **This project has already hit that
once** — the conditioning-hold correction changed what counts as a dynamic
segment, which changed the trace. A tracked npz would have survived that
change unnoticed.

Regeneration is documented in the README. Nothing is lost: the traces are
reproducible from the raw dataset, which the README already explains how to
obtain and place.
