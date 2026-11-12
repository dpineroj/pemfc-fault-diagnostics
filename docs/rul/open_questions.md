# The RUL direction — open future work

**Status: open. Nothing here has been started.** Written November 2026 so the
question is not lost now that the spatial phase and its follow-up tool are
finished.

## The question

Can a load-invariant health indicator improve remaining-useful-life
estimation for a PEM fuel cell under dynamic automotive load, using **bulk**
signals only?

It was adopted as the project's active line when the spatial line was closed
(`docs/phase_closure_spatial.md` §6), because it has what the spatial line
never had: real labels, in the form of measured degradation over a real
durability run.

## Where it stands

| item | state | where |
|---|---|---|
| Candidate dataset: Zuo et al. (2021), long-term dynamic durability, single cell | audited **from the descriptor only**; no data downloaded | `zuo_dataset_notes.md` |
| **Dataset DOI does not resolve** | **live blocker, unresolved** | `zuo_dataset_notes.md` §1 |
| Literature gate | done: dynamic-load RUL is heavily worked; load-confounded health indicators are named as an open problem | `literature_gate.md` |
| Companion paper (Zuo et al., methodology) | **abstract only**; full text needs institutional access | `zuo_dataset_notes.md` §6 |
| Meng et al. (2025) | not read in full; needs institutional access | `literature_gate.md` |

## What has to happen before any modelling

1. **Resolve the dataset's availability.** Find a working route to the data
   (publisher supplementary files, Mendeley, the authors). If there is none,
   this direction needs a different dataset and the audit starts again.
2. **Read the two papers in full** through the university library. The
   descriptor is silent on several things the audit marked NOT STATED, and
   the methodology paper is where they would be.
3. **Decide whether the question is still open** after reading them. The
   literature gate found the area crowded; the specific claim — that a
   load-invariant indicator helps — has to survive contact with what Meng
   2025 already did.
4. Only then: a loader, validated against an independent channel before it
   is trusted, as the Seville loader was.

## What carries over from the finished work

- **CV(J) is 34× more load-invariant than σ(J)** on real data. That result is
  about a *spatial* statistic; the RUL line is bulk-only, so what transfers is
  the method — normalise an indicator by the load-dependent quantity it
  scales with, then measure the invariance rather than assert it — not the
  statistic itself.
- **The detector's calibration idea**: express a signal as a residual against
  a healthy baseline learned at the same load (`hfc/features.py`). For
  degradation, the baseline is the cell's own beginning-of-life V(i) curve,
  and the residual's slow drift is a candidate health indicator. The Seville
  demo is a warning attached to that idea: a baseline did not transfer
  between sessions on the same cell, so session-to-session shifts have to be
  separated from degradation before any drift is read as ageing.
- **Conditioning-hold detection and FC-DLC segmentation**
  (`scripts/common/extract_load_trace.py`), which a 1,000-hour durability run
  with characterisation interruptions will need.
- **The working rules**: pre-register criteria, split by time or by group and
  check the split with a label-only leak check, assert convergence per fit,
  verify an operation through a channel other than the one that reported it.

## What does not carry over

The fault generator, the fault classifier and the fault-ahead models. Faults
are fast, reversible events; degradation is slow and irreversible. None of
the `hfc` models says anything about remaining useful life, and none should
be repurposed for it.
