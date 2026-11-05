# Literature gate — is "health state + RUL on Zuo" already saturated?

**Purpose.** Establish what is left before scoping anything. Written to the
instruction *do not go looking for a gap you can fill; report what is there* —
a finding that the area is saturated costs a week instead of a semester and is
a useful result.

**Verdict up front: substantially more worked than the pivot brief assumed, and
two of the three angles the project might have claimed are occupied.** Details
in §3. But the evidence base is thin in a specific, correctable way, and §1
says exactly how thin.

---

## 1. Access reality — one full text out of five

This matters more than it looks, because the three questions in §3 are about
*method*, and method lives in sections that abstracts do not contain.

| paper | what was obtained |
|---|---|
| Zuo et al. 2021, Applied Energy 281:115937 | **abstract only** (Semantic Scholar). Not OA. |
| Meng et al. 2025, J. Power Sources 625:235634 | **TITLE ONLY.** No abstract in Crossref, OpenAlex or Semantic Scholar. |
| Wang et al. 2023, RESS 233:109123 | **FULL TEXT**, via the arXiv preprint (2302.10771) |
| Nagulapati et al. 2023, Energy and AI 12:100237 | **abstract only**, reconstructed from OpenAlex. Gold OA but no free PDF exists at any non-publisher location. |
| Lv et al. 2024, IEEE TTE 10:1747, DOI 10.1109/TTE.2023.3266803 | **abstract only.** 77 citations. |

**A correction to the brief**: the Data in Brief descriptor was read through
**Europe PMC open access**, not institutional access. There is no authenticated
route available here, so publisher-side paywalls are hard stops —
ScienceDirect returns 403 and IEEE is unreachable. **Four of these five papers
need CMU institutional access to resolve properly**, and that is the single
cheapest action available to improve this assessment.

**Nagulapati's type is unresolved**: `publicationTypes` is null in Semantic
Scholar and OpenAlex gives only the generic `article`. Its reconstructed
abstract lists candidate indicators and describes intrusive methods as
background, which *reads* like a review — but that is an inference and is not
relied on below. **The brief's description of it as "trains GPR, SVM and ANN on
this dataset" could not be confirmed from any accessible source**, and its
OpenAlex reference list does not include the Zuo 2021 record (though OpenAlex
reference lists are often incomplete, so that establishes nothing).

---

## 2. Per paper

### Wang et al. 2023 — the only one read in full

| field | finding |
|---|---|
| health indicator | **The long-term degradation component of dynamic stack voltage, extracted by Hilbert–Huang transform.** Then compressed by adaptive Brownian-bridge aggregation (ABBA) into a reduced-dimension symbolic sequence. |
| load dependence | **Explicitly acknowledged as the central problem and attacked directly.** *"the measurements collected under dynamic mission profiles are difficult to use directly as health indicator (HI)"*. HHT is their answer to it. |
| reversible recovery | **NOT ADDRESSED. The words "reversible" and "recovery" appear zero times in the full text.** |
| model | symbolic-sequence GRU-based RNN; trend predicted, HI reconstructed, RUL estimated |
| horizon | *"a competitive prognostics horizon with lower computational cost"*; no horizon value stated |
| split | **Same-cell, explicitly and by design**: *"The prognostics process is implemented based only on historical data of the fuel cell stack in question. Training and testing of prediction models can be achieved using data from a single long-term fuel cell test."* |
| channels | **Stack voltage only, and the choice is justified on instrumentation-cost grounds** — see §3(iii) |
| dataset | two dynamic-load conditions. **Cites the Zuo Data in Brief descriptor as reference [44]**, so it almost certainly uses this data. |
| stated future work | *"explore prognostics methods incorporating fault diagnostics, e.g., adaptively setting failure thresholds by analyzing abnormal behavior"* and prognostics-based operational control. **Not** HI comparison, **not** transfer, **not** channel selection. |

One further quote worth keeping, because it bears on §4: *"PEMFC long-term
durability tests are extremely sparse, and open-source datasets in particular
are lacking."*

### Zuo et al. 2021 — abstract only

Attention-based RNN (attention-LSTM / attention-GRU vs plain LSTM / GRU),
predicting *"output voltage degradation"*. RMSE 0.015518–0.018206 on a dynamic
dataset, with a second "pseudo-steady" dataset also used. **Health indicator,
load handling, recovery, horizon and split are all NOT STATED in the
abstract.** Whether a model trained on one of its two datasets was tested on
the other is not stated — so it cannot be counted as a transfer test.

### Meng et al. 2025 — title only, and the title is the finding

*"Fuel cell life prediction considering the recovery phenomenon of reversible
voltage loss."*

**The title alone establishes that reversible-recovery-aware life prediction is
already a published contribution.** Nothing about method is available —
Crossref carries no abstract, and both OpenAlex and Semantic Scholar
rate-limited. OpenAlex index keywords (assigned, not authorial) are "health
indicators", "polarization curves", "LSTM". Its Crossref reference list cites
**both** Zuo 2021 and the Zuo dataset descriptor, so it plausibly uses this
data, though that is not stated.

### Nagulapati et al. 2023 — abstract only, type unresolved

Title: *"Machine learning based fault detection and state of health estimation
of proton exchange membrane fuel cells."* The abstract lists candidate
indicators — *"Indicators that influence voltages, current density
distribution, impedance spectra, acoustic emission and magnetic fields"* — and
notes that intrusive methods (EIS, cyclic voltammetry, galvanostatic analysis)
require operation to be paused. **It does not state which indicator is used, a
model, a dataset, a split or a result.** Whether it compares indicators or
merely lists them is **unresolved without the full text.**

### Lv et al. 2024 — abstract only

Transformer with a *"series-attention mechanism"* replacing self-attention, for
*"multistep-ahead prediction and long-term prognostic conditions"* on *"PEMFC
dynamic durability test data"*. Motivation: *"the long-term prediction
performance is still limited by the model structure and accumulative errors."*
**HI definition, load handling, recovery, horizon, split and numeric results
all NOT STATED.** 77 citations.

---

## 3. The three questions, answered without softening

### (i) Has anyone treated CONSTRUCTION OF THE HEALTH INDICATOR as the object of study?

**The premise behind this question is wrong, and Wang's full text is what
shows it. HI construction under dynamic load is not an unexamined choice made
en route to a model — it is a contested, actively worked problem.**

Wang's related-work section surveys at least four distinct prior HI
constructions, each a different answer to the load-confounding problem:

- reconstruction of a **virtual steady-state stack voltage** from segmented
  data [their ref 14]
- a **polarization-function model** with voltage segmentation and nonlinear
  regression [15]
- **equivalent internal resistance** tracked by a sliding-window degradation
  model [16]
- a **periodic polarization test** at intervals [17]
- plus mechanistic-model parameter identification via filters [11–13]

and then adds a fifth, HHT. So the project's framing — that the literature
picks a load-confounded indicator and works around it — **does not survive
contact with this paper.** Several groups have built explicitly
load-decoupled indicators.

**What remains genuinely unestablished**: whether anyone has *systematically
compared* candidate HIs against each other as the object of a study, rather
than proposing one and benchmarking the downstream model. **No paper I could
read does that** — but I could only read one in full, so this is an absence of
evidence across four inaccessible methods sections, not a demonstrated gap.
Wang's own stated future work does not include it.

### (ii) Has anyone tested whether a model TRANSFERS to another cell or dataset?

**No paper among the five shows a cross-cell or cross-dataset transfer test.**

Wang is explicit that it does not, and frames same-cell operation as the
design: *"implemented based only on historical data of the fuel cell stack in
question."* Zuo uses two datasets but does not state that a model crossed
between them. The other three say nothing.

**This is the most defensible of the three openings — with one important
qualification.** It may be open because it is *infeasible* with what exists
rather than because it is unexamined: Wang states that long-term durability
tests are *"extremely sparse, and open-source datasets in particular are
lacking."* Transfer needs two comparable long-term dynamic-load datasets, and
if only one is public, nobody can do it regardless of interest.

**So before claiming this: identify a second usable dataset.** If none exists,
the question is open and will stay open, and the project cannot answer it
either.

### (iii) Has anyone asked WHICH CHANNELS are needed?

**No systematic channel-selection or minimum-instrumentation study was found.
But the sensor-cost argument is already in use, which largely answers the
question before it is asked.**

Wang justifies a voltage-only indicator in exactly those terms: *"the PEMFC
stack voltage is easy to collect and **does not require the installation of
additional test equipment**. Numerous government agencies, research institutes
and industries directly and/or indirectly use stack voltages when measuring
PEMFC performance."*

**This is the sensor-cost objection already applied, by the field, as a
justification for using one channel.** The consequence for the project is
uncomfortable and should be stated plainly: the field has effectively settled
on minimum instrumentation *by default*, so a study concluding that one or two
channels suffice would **confirm existing practice rather than challenge it**.
And the inverse result — that Zuo's sixteen channels are needed — would argue
*against* deployability, which is the objection that closed the spatial phase.

There is a narrower, still-unoccupied version: nobody appears to have
*measured* what the additional fifteen channels buy, as opposed to assuming
they are unnecessary. That is a real question. It is also a modest one, and it
is a feature-ablation study rather than a research programme.

---

## 4. What this leaves

**Occupied:** load-decoupled HI construction (at least five distinct
approaches, Wang's among them). Reversible-recovery-aware life prediction
(Meng 2025, by title). Deep-sequence prognostics on dynamic-load data (Zuo,
Lv, Wang).

**Plausibly open, both weaker than hoped:** systematic comparison of candidate
HIs as an object of study; and cross-dataset transfer, which may be blocked by
data scarcity rather than neglect.

**Not a gap:** "the literature uses load-confounded indicators." Wang's
related work falsifies that as a general claim.

**What would change this assessment**, in order of value per unit effort:

1. **Meng et al. 2025 full text.** Its title claims the recovery angle. If its
   method is narrow, the recovery problem may be partly open; if broad, it is
   closed. One paper decides this.
2. **Zuo et al. 2021 full text.** Settles whether the dataset's own authors
   already handled load confounding.
3. **Nagulapati 2023 full text.** Settles review-vs-study, and whether anyone
   has compared indicators.
4. Wang's published RESS version, to confirm the preprint matches.

All four need institutional access. **Until at least items 1 and 2 are read,
no novelty claim should be made.** The cost of reading them is hours; the cost
of skipping them is a semester spent re-deriving a published result.
