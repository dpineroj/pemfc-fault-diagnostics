# Data requests — drafted, NOT sent

Drafts only. Nothing here has been sent, and the completion campaign does not
wait on any reply (`synthetic_generator_notes.md` §13a). Recipient details
must be verified by a human against the paper before sending.

---

## 1. Physics-guided LSTM fault-diagnosis dataset

**Why it matters.** It is the one lead found in two audits and one bounded
search that pairs *labeled* faults — flooding, membrane drying, air starvation
and hydrogen starvation — with *dynamic* operating conditions. Reported class
counts 8437 / 3846 / 1692 / 477 / 485 at a 20 s sliding window. It is
bulk-signal only and is referenced in the paper rather than deposited.

**What it could and could not do for this project.** Bulk-only, so it can
externally benchmark the **bulk** arm and cannot test the spatial arm at all:
it would *bound* the spatial-vs-bulk comparison, not validate it. It would be
the first real data in this project carrying a starvation label.

**Identification — unverified.** Journal of Power Sources (2024), article PII
S0378775324016483. A literature review surfaced in the search attributes a
physics-model-plus-LSTM study of exactly these four faults to Pettorossi et
al.; whether that is the same paper was **not confirmed**. Check the PII, the
corresponding author and their current address before sending.

**Draft.**

> Subject: Request for the fault-diagnosis dataset from your Journal of Power
> Sources paper (PII S0378775324016483)
>
> Dear Dr. [corresponding author],
>
> I am an undergraduate researcher at Carnegie Mellon University working on
> PEM fuel cell fault detection under dynamic automotive loading. Your paper
> on physics-guided LSTM fault diagnosis is, as far as I have been able to
> find, the only study that pairs labeled flooding, drying, air-starvation and
> hydrogen-starvation conditions with dynamic operation.
>
> My own work compares spatially resolved and bulk-signal classifiers. Because
> no public dataset combines labeled faults with dynamic loading, every fault
> label I have is synthetic, and I report that as a permanent limitation. Your
> data would let me benchmark the bulk-signal arm against real labeled faults
> — it would not validate the spatial arm, and I would say so explicitly.
>
> Would you be willing to share the dataset, or the subset underlying the
> classification results? I would use it for non-commercial academic research
> only, would not redistribute it, and would cite your paper in any resulting
> work. I am glad to sign a data-use agreement if your institution requires
> one.
>
> If the data cannot be shared, I would be grateful for any detail on how the
> fault conditions were induced under dynamic load and how the 20 s windows
> were labeled, since that question is central to my own labeling.
>
> Thank you for your time.
>
> David Piñero
> Electrical and Computer Engineering, Carnegie Mellon University

**Status: drafted, not sent.**

---

## 2. Leads recorded, not pursued

From the same bounded search (three queries). None pairs labeled faults with
dynamic loading; none was downloaded or audited.

| lead | what it is | why not pursued |
|---|---|---|
| Mendeley `10.17632/m9m2gmcr6k` (2021) | flooding and dehydration test data from two PEMFC systems, "2D data" diagnosis method | no starvation; no indication of dynamic loading; bulk signals. By the Zhou precedent a bulk-only static set is audited and not integrated — and it cannot test the spatial arm in any case |
| Mendeley `10.17632/4sfyzcc39j` (2019) | stack flooding and humidification-sensor fault data | same |
| Zenodo record 16748652 (2025) | catalyst-loading durability study at varying humidity and temperature | not fault-labeled |
| several 2025 papers classifying normal / drying / starvation | reported accuracies up to 100% on a single cell | no public data visible; data-availability statements unchecked |

"No indication of dynamic loading" is a statement about what the search
snippets showed, **not** a structural finding. If either Mendeley deposit is
ever considered, it gets a Zhou-style audit first.

---

## 3. What the tool changes about what to ask for (added November 2026)

**Still a draft. Nothing has been sent.** This section updates *what* a
request should ask for, now that the `hfc` detector exists
(`docs/tool_design.md`); it does not change the status of §1.

The detector's data contract is narrower and more specific than the original
request assumed, and the Seville demo showed two things that matter for any
dataset worth asking for (`hfc_models/MODEL_CARD.md`):

1. **Bulk signals are enough to test the default variant.** Stack current and
   cell voltage at ≥ 1 Hz. The §1 dataset is bulk-only, which was recorded
   above as a limitation; for the shipped default it is exactly what is
   needed.
2. **A healthy stretch from the same session is required.** The tool
   calibrates on the cell it is judging, and a baseline does not transfer
   between sessions (cross-session false alarms of 69–73% on the real
   recording). A fault recording without same-session healthy data cannot be
   scored by this tool at all.

So a request should ask, in this order of importance:

| ask | why |
|---|---|
| per-sample **stack current and cell (or per-cell mean) voltage**, with timestamps, at ≥ 1 Hz | the bulk contract; fault-ahead horizons are 1 s, 2 s and 20 s |
| a **healthy stretch in the same session** before each fault, spanning the load range used | runtime calibration |
| **fault onset time** for each episode, and how it was induced | the fault-ahead models predict a confirmation time; a label with no onset can only score classification |
| fault type, including **oxygen versus hydrogen starvation** if distinguished | the tool ships one starvation class because nothing here could ground the split |
| whether load was **dynamic or static** during each fault | the central gap in every dataset audited so far |
| active area, number of cells, and units for every channel | the unit checks refuse to run on implausible units |
| a segmented current-density map, if one exists | only needed to judge the opt-in spatial variant |

Sentences to add to the §1 draft before it is ever sent, replacing its
general description of intended use:

> Specifically, I would use per-sample stack current and cell voltage with
> timestamps, the onset time and type of each induced fault, and a stretch of
> healthy operation recorded in the same session before each fault. I have a
> detector trained only on simulated faults and no way to measure whether it
> works on real ones; your data would be the first real test it has had. I
> would report the result whichever way it comes out.

**Status: drafted, not sent.**
