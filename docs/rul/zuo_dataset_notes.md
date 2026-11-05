# Zuo et al. (2021) long-term dynamic durability dataset — audit from the descriptor

**No data downloaded. No loader written. No model proposed.** Written from the
data descriptor alone, deliberately: the Seville units error was caught because
structure was characterised before a parser existed, and the same discipline
applies here.

**Citation.** Zuo, J., Lv, H., Zhou, D., Xue, Q., Jin, L., Zhou, W., Yang, D.,
Zhang, C. (2021). *"Long-term dynamic durability test datasets for single proton
exchange membrane fuel cell."* Data in Brief 35:106775.
DOI [10.1016/j.dib.2021.106775](https://doi.org/10.1016/j.dib.2021.106775).

**How it was read, and the limits of that.** The publisher routes
(doi.org → linkinghub, ScienceDirect) are closed to automated fetching: 403 and
an empty redirect stub. **The full text was read via Europe PMC,
PMC7851326 / PMID 33553524**, as open-access XML. Everything attributed below
is from that full text. Where the descriptor is silent, this document says
**NOT STATED** rather than inferring — those silences are the audit's main
product.

**The companion paper was NOT obtained in full** — see §6. Only its abstract,
via Semantic Scholar. That matters, and §6 says why.

---

## 1. LIVE BLOCKER — the dataset DOI does not resolve

> **This is a blocker, not a footnote. If the DOI does not resolve, the dataset
> is unusable regardless of what the literature says, and that answer is needed
> before any further investment.**

Measured, not assumed — every URL and what it returned:

| URL | result |
|---|---|
| `https://doi.org/10.17632/w65jjt8v5w` | **HTTP 404 Not Found** |
| `https://data.mendeley.com/datasets/w65jjt8v5w/1` | HTTP 200, page reads **"Dataset Not Found"** |
| `https://data.mendeley.com/datasets/w65jjt8v5w/2` | same **"Dataset Not Found"** |
| `https://api.datacite.org/dois/10.17632/w65jjt8v5w` | **HTTP 404** — the DOI is not registered with DataCite |

**The DataCite 404 is the significant one.** A registered Mendeley DOI should
have DataCite metadata. Its absence, together with the doi.org 404, is
consistent with **the deposit never having been published out of draft** —
which is exactly what the paper's own link suggests, since the descriptor
prints only a tokenised draft URL:

```
https://data.mendeley.com/datasets/w65jjt8v5w/draft?a=5b37947d-ea28-48cd-a5f2-d8c61c8ed8b1
```

A tokenised draft link is **not a durable citation**. It can be revoked, it
cannot be cited, and a project built on it has no reproducible data provenance.

**This does not prove the data is gone.** The token may still work, the deposit
may have been republished under a different identifier, and a 404 from an
automated fetch is weaker evidence than a browser check. **What it does mean is
that viability must be established before anything else**, by:

1. opening the tokenised draft URL in a browser — the cheapest test, and the
   only one that settles whether the files are reachable at all;
2. searching Mendeley Data for the title, in case the published deposit sits
   under a different accession;
3. **emailing the corresponding author** if neither works. The dataset is the
   subject of a Data in Brief paper, whose entire purpose is data availability,
   so a broken deposit is a defect the authors would want reported.

**Fallback if it is unrecoverable.** The nearest substitutes, with what each
costs:

- **IEEE PHM 2014 / FCLAB** — the standard benchmark, genuinely available, but
  FC2 is *quasi-dynamic* (current ripple on a setpoint, not a drive cycle), so
  the dynamic-load framing weakens to ripple-load. Also heavily worked over.
- **The Nature Sci. Data (2024) catalyst-degradation set** — 36 tests, 4000+ h,
  real ASTs, ships evaluation code. Materials-oriented rather than automotive
  load profiles.
- **Zenodo 7054555** — real-world automotive operation with operando per-cell
  voltage and HFR. Possible, unaudited.

None of these is FC-DLC under a real drive cycle to end of life, which is what
made Zuo attractive. **If Zuo is unrecoverable, the RUL direction needs
re-scoping around a different dataset, and the "our FC-DLC tooling transfers
directly" argument weakens accordingly** — it transfers to an FC-DLC protocol,
and none of the fallbacks use one.

## 1a. Licence — also unresolved

**The dataset link printed in the paper is a DRAFT URL with an access token:**

```
https://data.mendeley.com/datasets/w65jjt8v5w/draft?a=5b37947d-ea28-48cd-a5f2-d8c61c8ed8b1
```

**The published dataset pages do not resolve.** Both
`data.mendeley.com/datasets/w65jjt8v5w/1` and
`data.mendeley.com/datasets/w65jjt8v5w` return HTTP 200 with **"Dataset Not
Found"**. So the file listing, the real file extensions, the file sizes and the
dataset's own licence could not be read from the repository.

**Licence status:**

| | |
|---|---|
| the **article** | **CC BY-NC-ND** — stated verbatim: *"This is an open access article under the CC BY-NC-ND license"* |
| the **dataset** | **NOT STATED anywhere in the descriptor**, and unreadable from Mendeley |

Two consequences. First, **the licence must be established from the Mendeley
landing page before anything is built on this data**, exactly as was done for
Zhou et al. — and unlike Zhou, it cannot currently be read. Second, the
article's **ND** term is worth noting even though it governs the article rather
than the data: a non-derivatives clause is unusual for a data descriptor and
makes the data licence less safely assumed.

**Applying the established rule regardless**:
third-party data stays **gitignored** (`data/raw/` is already ignored). That
rule was adopted precisely so the licence question stops being load-bearing —
see `../spatial/external_fault_datasets.md` §3.

**Before planning a download**: confirm the published DOI
`10.17632/w65jjt8v5w` resolves. Right now it does not appear to, and the only
working link is a tokenised draft, which is not a durable citation.

---

## 2. Verification of the briefed summary

Checked line by line against the full text.

**Confirmed exactly as briefed:**

- Greenlight **G20 (P17-2390, Burnaby, B.C. Canada)** test station.
- Single commercial PEMFC from **Wuhan New Energy Co., Ltd.**
- Recording at **1 Hz** — *"The data record frequency in this work is set as 1 Hz."*
- **3076 FC-DLC cycles, around 1008 h.**
- Folder `Durability_test_dataset`, **twenty** files; nineteen at **152 cycles**
  each, last at **188**. 19 × 152 + 188 = 3076 ✓.
- Column list and units as briefed (with one qualification, below).
- Setpoints: H₂ and air inlet pressure **both 110 kPa**; H₂ inlet **70 °C** /
  dewpoint **55 °C**; air inlet **70 °C** / dewpoint **65 °C**. The paper also
  notes recorded values differ slightly from setpoints.
- Polarization folder and both filenames; **`current_set` and `current`** both
  exist; the after-rest file **begins at 100 h**.
- Sweep: **0.02 A·cm⁻²** steps from 0 to 0.1, **0.2 A·cm⁻²** above; dwell
  **30 s** at OCV, **40 s** to 0.1, **190 s** above 0.1; ascending **and**
  descending modes described.
- Shutdown rest **every 304 cycles**, 12 h, durability restarting after the
  polarization test.

**Three corrections or additions to the briefing:**

1. **Table 1 does not give separate headers for current, voltage and power.**
   It has one combined row — *"The operating load (A), output voltage (V) and
   power (W)"*. The briefing listed `Current (A), Voltage (V), Power (W)` as
   three named columns. The **exact header spellings for these three are NOT
   STATED** and must be read off the file.
2. **One printed header contains a stray space**: `Temp_ cathode_dewpoint_water`.
   Whether that is typesetting or the actual column name is unknown — a loader
   must not assume either.
3. **The characterisation interval is internally inconsistent in the paper**,
   and the briefing reproduced both halves without noting they conflict.
   §2.1 says characterisation follows **every 152 FC-DLC cycles (~50 h)**;
   the abstract and §1 list characterisation at **0, 100, 200 … 1000 h**, i.e.
   **100 h** intervals. **The paper does not reconcile these.** Twenty
   durability files at ~50 h each against eleven polarization time points at
   100 h each is consistent with characterisation happening every ~50 h but
   only every *other* one being retained — but that is inference, not text.

---

## 3. Item A — "RUL labels come free" is WRONG, and this is the audit's main finding

**The descriptor does not state that the cell reached end of life. It gives no
end-of-life criterion, no failure threshold, no degradation rate, and no reason
why testing stopped.**

The only duration statement is *"The overall durability test is composed of
3076 FC-DLC cycles, around 1008 h."* Nothing says whether 1008 h was a planned
duration or a stopping point forced by cell condition.

**It also gives no voltage-degradation numbers at all** — no mV, no mV·h⁻¹, no
percentage loss. The abstract says only that the degradation trend *"[is]
plotted and described"*, and the text refers to *"the reversible degradation
trend as shown in Fig. 1 b)"*. Figure 1b is an image; no values are recoverable
from the text.

**So the earlier framing — "degradation labels itself with time; at hour 300 of
a 1008-hour run to end-of-life, remaining life is 708 hours" — does not hold.**
It assumed the run terminated at end of life. The descriptor does not support
that, and until the data is inspected it cannot be assumed.

**THE CORRECTION, PROPAGATED.** The project's claim is now stated as:
**health-state estimation at time t**, with remaining useful life obtainable
**only by extrapolation against an analyst-chosen threshold**. That is a
materially weaker claim than observed RUL and must be written that way
everywhere — the target is constructed rather than measured, the horizon is
unvalidated, and the prediction error cannot be checked against any ground
truth beyond the end of the data.

**The settling test requires the data and therefore stays OPEN.** Computing a
fixed-current voltage against aging time and checking for a conventional
threshold crossing is a measurement, not a model — but it cannot be run until
the dataset is in hand, which §1 shows is currently blocked. **Until that test
runs, the weaker framing is the operative one**, not a placeholder to be
upgraded on optimism.

**External material carries the wrong framing and needs correcting before it
goes anywhere.** Any abstract, proposal, slide or summary drafted on the
"degradation labels itself with time — at hour 300 of a 1008-hour run to
end-of-life, remaining life is 708 hours" framing asserts observed RUL labels
that this dataset is not known to provide. **That claim must be corrected
before any such material is sent, submitted or presented**, and it is listed
here rather than only in conversation so the correction is not lost.

**What this changes about the project's claim.** Two different projects follow,
and they are not interchangeable:

| if | then the project is |
|---|---|
| the cell **did** reach a definable EOL within 1008 h | **RUL prediction** with observed labels |
| it **did not** | **health-state estimation**, with RUL only by **extrapolation** past the end of the data |

Extrapolated RUL is a materially weaker claim: the target is constructed by the
analyst, the horizon is unvalidated, and the error cannot be measured against
anything. **Which of these the project is must be settled from the data before
any RUL claim is scoped**, and the honest default until then is health-state
estimation.

**Determinable from the data without a model**: compute a fixed-current voltage
over aging time and check whether any conventional threshold (often 10% voltage
loss) is crossed within 1008 h. That is a measurement, not a model, and it
should be the first thing run after loading.

---

## 4. Item B — the 12 h rests cause recovery, and the descriptor says so

**Confirmed, and it is a first-order modelling constraint rather than a
footnote.** Quoting:

> *"the V-I curves before and after shutdown rest can help to further
> investigate the recovery phenomenon in PEMFC."*

> *"the reversible degradation trend as shown in Fig. 1 b) can also be
> explained by the polarization curves."*

> *"Whenever the G20 test station is been shut down, the PEMFC is rest for
> 12 h, then the durability test is restarted after the polarization test."*
> — and *"the shutdown rest is carried out every 304 FC-DLC cycles."*

So the voltage history is **not a monotonic decline**: a sawtooth from periodic
recovery is superimposed on the irreversible trend, roughly every 304 cycles
(~100 h), eleven-ish times across the run.

**Any health indicator that does not separate reversible from irreversible
degradation will mistake recovery for repair** — and worse, a model trained
across rests can learn the rest schedule rather than the degradation, scoring
well for the wrong reason. That is structurally the §10 failure mode from the
closed phase: a result decided by the data's construction rather than measured.

**This is a SECOND confound stacked on the first.** The project exists because
the health signal is confounded by **load**. It is also confounded by
**time-since-last-rest**. The two are independent and both must be handled:

- load confounding → the load-invariance work that carries over
- recovery confounding → needs its own treatment, and the project has nothing
  built for it yet

The before/after-rest polarization pairs are the lever: they give a direct
measurement of the reversible component at eleven points, which is the only
clean separation the dataset affords.

---

## 5. Item C — single cell, no replicates

One unit, one run. **No cross-unit generalisation is available at all.**

**What that forbids.** The leave-cells-out protocol from the closed phase is
impossible — there is one cell. Any split is within the same cell's own
history, so it tests **temporal extrapolation**, not generalisation to a new
cell. No result from this dataset can claim to generalise across cells, and
nothing should be phrased as if it does.

**What it permits, and it is the honest test anyway.** A **chronological**
split — train on early aging hours, test on later ones — is the only
defensible protocol, and it is *harder* than a random split rather than a
concession. It also matches what a deployed system actually faces: predicting
forward from history it has already seen.

**Carried over from the Zhou audit, and it applies with more force here**: a
random split on this data would leak badly. Adjacent 1 Hz samples are
near-duplicates, and the Zhou audit measured the cost of exactly that mistake
(88.46% chronological against 96.15% random on identical data). With 1 Hz
sampling over 1008 h — on the order of 3.6 million rows — the adjacency is far
denser than Zhou's. **Chronological or grouped-by-aging-block, never random.**

---

## 6. Item E — the companion paper is prior art, and the key question is UNRESOLVED

Zuo, J. et al. (2021). *"Deep learning based prognostic framework towards
proton exchange membrane fuel cell for automotive application."* Applied Energy
281:115937. DOI 10.1016/j.apenergy.2020.115937.

**Only the abstract was obtainable.** Not in Europe PMC (hitCount 0), not open
access per OpenAlex (`is_oa: false`, `has_fulltext: false`), and the publisher
routes are closed to automated fetching. The abstract came from Semantic
Scholar.

**What the abstract establishes:**

- **Model**: an **attention-based RNN**, comparing attention-LSTM and
  attention-GRU against plain LSTM and GRU.
- **Target**: *"the output voltage degradation of proton exchange membrane fuel
  cell (PEMFC) based on the original long-term dynamic loading cycle durability
  test data."*
- **Results** (RMSE): attention-LSTM **0.016409**, attention-GRU **0.015518**,
  LSTM 0.017637, GRU 0.018206, on "dataset 1" (dynamic). A second
  "pseudo-steady" dataset is also used, results not quantified in the abstract.

**What the abstract does NOT establish — and these are the ones that matter:**

| question | status |
|---|---|
| what **health indicator** they used, and how computed | **NOT STATED** |
| whether they handled **load confounding** | **NOT STATED** — no mention of current-dependence, normalisation, or fixed-current evaluation |
| whether they handled the **recovery / rest** effect | **NOT STATED** |
| prediction **horizon** | **NOT STATED** |
| **train/test split** protocol | **NOT STATED** |

**This is the single largest open question in the audit, and it is not
resolvable from what is accessible.** The project's premise is that
load-confounded health indicators are the field's open problem. If the
companion paper already conditioned on current — or if feeding an RNN the
current channel alongside voltage implicitly handles it — then the gap the
project aims at may be substantially narrower than assumed, on this very
dataset, by the people who produced it.

Two honest readings, both live:

- Predicting *"output voltage degradation"* with current as an input feature
  could let the network absorb load-dependence implicitly, which would make
  explicit load-invariance an interpretability and robustness improvement
  rather than an enabling one.
- Or they predicted an aggregate voltage trend and inherited exactly the
  confound the literature complains about, in which case the gap is real.

**The abstract cannot distinguish these.** The full text is required before any
novelty claim is made. **Getting it is a prerequisite, not a nicety** — it is
cheap to obtain through CMU's institutional access, and expensive to skip.

---

## 7. Structural questions the descriptor leaves open

Each of these is where a units-error-class defect would hide.

**Is `Time` cumulative across the run, or per file?** **NOT STATED.** Table 1
says *"Aging time (s)"*; the text says *"Durability time is recorded in the
'time' column."* The file naming — *"50_h.csv represents the data within this
file was recorded from 0 to 50 h"* — hints the file spans 0→50 h of test time,
but says nothing about the column's origin. **This determines whether
cross-file stitching is concatenation or offset reconstruction**, and getting
it wrong silently misaligns the entire aging axis.

**Do the durability files contain the polarization periods inline?** **NOT
STATED.** The schedule implies characterisation happens between cycling blocks
and the curves live in their own folder, which suggests cycling-only. But if
polarization segments are embedded they **must** be split out — they are a
different operating mode, and a slow current sweep averaged into a drive-cycle
health indicator would corrupt it. Structurally identical to the **231 of 295
minutes** of conditioning holds in the Seville data.

Note a concrete interaction: `normalize_load_trace`'s new **constant-current
guard will fire** on a polarization dwell (190 s at a fixed current), which is
useful as a detector rather than a nuisance.

**What is the cell's active area?** **NOT STATED in the descriptor** — it
refers the reader to the companion paper for cell parameters. This is a direct
units hazard: the **polarization sweep is specified in A·cm⁻²** while the
**durability `Current` column is in A**. Converting between them requires the
active area, which the descriptor does not provide. **This is the Seville units
error's exact shape** — a per-unit versus per-area confusion with a missing
constant — and it should be treated as the single most likely source of a
silent factor error.

**What are the FC-DLC current levels?** **NOT STATED** — *"shown in the
experimental part of the linked research article."* Another dependency on the
companion paper.

**Cell operating temperature setpoint?** Not given; controlled by an external
circulating water pump, with `Temp_anode_endplate` recorded as the operating
temperature.

---

## 8. Internal consistency checks the data affords

Seville's lesson was that a derived channel must be validated against an
independent one. Here:

1. **`Power` = `Current` × `Voltage`** — assert on load, as the primary check.
   It is only a partial check: it validates the recording chain's arithmetic,
   not the calibration of any individual channel.
2. **`current_set` vs `current`** in the polarization files — a direct measure
   of how closely the station tracked its own setpoints, and the best available
   proxy for instrument fidelity.
3. **Recorded conditions vs stated setpoints** — pressures against 110 kPa,
   temperatures against 70 °C, dewpoints against 55/65 °C. The paper
   acknowledges small deviations, so this bounds the drift rather than
   detecting error.
4. **Flow against current** — stoichiometry links `Total_*_stack_flow` to
   `Current`; a plausibility rather than exact check without the stoichiometry
   setpoints.
5. **Cycle count against duration** — 19 files × 152 cycles + 188 = 3076, and
   ~50 h per 152-cycle block. If the files are 1 Hz and ~50 h, each should hold
   on the order of 180,000 rows; a materially different count signals missing
   data or a different sampling behaviour.

**Weaker than Seville.** There, `INTENSIDAD` was a genuinely independent
measurement of the same physical quantity as the integrated grid. Here, `Power`
is almost certainly *computed* by the station from current and voltage, so the
check may be a tautology. **It should be run and its result interpreted
cautiously** — if `Power` matches to floating-point precision it was computed,
and provides no validation at all.

---

## 9. What existing code gives for free, and what is genuinely new

**Free, or nearly:**

- **`normalize_load_trace(timestamps, current)`** accepts this shape
  **directly**. It was decoupled from the Seville loader for exactly this, and
  takes either a datetime64 axis or a **numeric seconds axis** — which is what
  `Time (s)` provides. No change needed.
- **`detect_holds`** plausibly finds the characterisation interruptions: it
  flags contiguous windows where rolling std of load ratio stays below a
  threshold. A polarization sweep is a staircase of 40–190 s dwells, so
  individual dwells should trip it. **Its `min_hold_s` default is 1800 s**,
  tuned for Seville's half-hour-plus conditioning holds, and will need
  retuning — a polarization sweep is shorter. That is a parameter change, not
  new machinery.
- **`find_ramp_events`, `find_dwell_fraction`, `extract_dynamic_segments`,
  `concatenate_segments`** all operate on `(t_seconds, load_ratio)` and are
  protocol-level. The FC-DLC protocol is the same one.
- **`bulk_timeseries.read_bulk`'s discipline** transfers even though its
  parsing does not: this data is unlikely to be comma-decimal, but the
  explicit-format-string lesson applies to whatever datetime or numeric parsing
  the new loader does.

**Genuinely new, in rough order of risk:**

1. **Cross-file stitching with continuity validation** — twenty files into one
   monotonic ~1008 h axis, with gaps and overlaps at boundaries explicitly
   accounted for. Conceptually the segment-seam handling from the closed phase,
   but across files and unresolved until `Time`'s origin is known.
2. **Polarization / cycling separation** — identifying and splitting a
   different operating mode, whether it sits in its own files or inline.
3. **Reversible-degradation handling** — nothing exists for this. It is the
   genuinely new modelling problem (§4).
4. **An end-of-life definition, pre-registered** — per §3, and per the closed
   phase's practice of fixing criteria before measuring, the EOL threshold must
   be chosen and written down *before* looking at where it falls.
5. **Excel/CSV reading across many files** — trivial, but `openpyxl`
   availability should be confirmed rather than assumed if the files really are
   `.xlsx`.

---

## 10. Ambiguities to resolve before writing a parser

Listed because the Seville experience says this is where a silent factor error
hides. **None of these should be guessed.**

| # | ambiguity | why it matters |
|---|---|---|
| 1 | `.csv` vs `.xlsx` — text says *"twenty excel data files"*, names them `.csv` | wrong reader fails loudly; harmless |
| 2 | last file is `1000_h.csv` **and** `1000h.csv` in different sentences | a glob must tolerate both; an exact-name loader breaks on one |
| 3 | `Time` origin: cumulative or per-file | **silent** aging-axis misalignment |
| 4 | exact headers for current / voltage / power — Table 1 merges them | fails loudly |
| 5 | `Temp_ cathode_dewpoint_water` stray space | fails loudly, but only if that column is used |
| 6 | **active area missing** → A vs A·cm⁻² conversion | **silent factor error — the Seville class** |
| 7 | polarization inline in durability files, or not | **silent** corruption of any health indicator |
| 8 | characterisation every ~50 h or every 100 h — paper contradicts itself | affects how recovery epochs are located |
| 9 | whether EOL was reached | determines what the project can claim (§3) |
| 10 | dataset licence, and whether the DOI resolves at all | determines whether this is usable |

**Items 3, 6 and 7 are the dangerous ones** — they fail silently and produce
plausible numbers. Items 1, 2, 4 and 5 fail loudly and are therefore cheap.

---

## 11. Summary of what this audit changes

- **The RUL framing was too optimistic.** The descriptor does not establish
  that end of life was reached, so "labels come free" does not hold. Until the
  data says otherwise the project is **health-state estimation**, with RUL as
  an extrapolated and weaker claim (§3).
- **There is a second confound**, time-since-rest, stacked on the load
  confound the project was conceived around. Nothing exists to handle it (§4).
- **No cross-cell generalisation is possible**; chronological splitting is the
  only defensible protocol and is the harder test (§5).
- **The prior art on this exact dataset, by these authors, has not been
  read in full**, and whether it already addresses load confounding is
  unresolved. That question gates the novelty claim (§6).
- **Access is a LIVE BLOCKER**: the DOI 404s and is unregistered with
  DataCite, and the only link is a tokenised draft (§1). Viability must be
  settled before further investment.
- **The literature is more worked than assumed.** See
  [`literature_gate.md`](literature_gate.md): load-decoupled HI construction
  is a contested, actively worked problem with at least five distinct
  approaches, and reversible-recovery-aware life prediction is already
  published by title (Meng et al. 2025). Two of the three angles this project
  might have claimed are occupied.
- **Our load-trace machinery transfers directly**; the new work is stitching,
  mode separation, recovery handling and an EOL definition (§9).
