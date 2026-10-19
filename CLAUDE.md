# Working notes for this repo

Environment and tooling properties learned the hard way. Project *content*
decisions live in `docs/` — this file is only for things about the setup that
would otherwise be rediscovered.

## Writing files

**Do not write large multi-line Python through a bash heredoc. It fails
intermittently in this environment.** Observed three separate times in one
session: bash reports `unexpected EOF while looking for matching '` and the
heredoc body is never delivered, so the target file is left untouched (no
corruption, but no change either). It is not correlated with any obvious
feature of the content — short heredocs usually work, long ones with nested
quotes often do not.

Use a file write instead (the Write tool, or write the new content to a
separate file and splice it with `head`/`cat`). For an edit to part of a large
file, prefer a small patch script written to disk and then executed, over
piping the patch through a heredoc.

Three occurrences is a property of the setup, not bad luck. Budget for it.

## Shell quoting

Paths in `data/raw/zhou_ac_voltage/` contain **spaces** (`AC voltage
response/`, `In sequential/`). Unquoted `for f in $(find ...)` word-splits on
them and produces a cascade of "No such file or directory". Use Python with
`pathlib` for anything iterating those trees, or quote rigorously.

## Python environment

- Virtualenv is `hfc_env/`; invoke as `./hfc_env/Scripts/python.exe` from the
  repo root. It is NOT on `PATH` by default in this shell.
- `scipy` 1.18, `sklearn` 1.9, `numpy` 2.x. **torch is not installed** and
  should not be added without a stated reason (see `docs/` §12c).
- **numpy 2.x removed `ndarray.ptp()`** — use `np.ptp(arr)`. Easy to trip on
  when adapting older snippets.
- Run throwaway inspection code with `python -I` when it touches
  `data/raw/`, which is untrusted downloaded data.

## Long-running scripts

Several measurement scripts take minutes to tens of minutes (`measure_battery_null.py`,
`phase4_model_a.py`). Run them in the background rather than in the
foreground, and **pass `-u`** — without it, output is buffered and the progress
file stays empty, which looks indistinguishable from a hung process.

`phase4_model_a.py` caches its expensive per-`(trace, cell)` label-building
step to `.model_a_cache/` (gitignored). Delete that directory to force a
rebuild; keep it to iterate on the modelling cheaply.

## Data directories

`data/raw/` and `data/processed/` are both gitignored and neither is
redistributed — the raw data is third-party licensed, and the processed load
traces are deterministically regenerable from it. Reasoning and regeneration
instructions: `docs/external_fault_datasets.md` §3 and the README.

## Standing test: could this tooling choice fail asymmetrically between the arms?

**Ask this of every multi-arm comparison, before trusting a delta.** Phrased as
a question, in the manner of `docs/synthetic_generator_notes.md` §10.

The arms of a spatial-vs-bulk comparison differ in dimensionality **by
construction** — hundreds of features against a handful. So any tooling
limitation that scales with dimensionality (iteration caps, regularization
defaults, convergence tolerances, feature-scaling assumptions, memory limits)
is *expected* to bite one arm harder than the other. **An arm-asymmetric
failure biases the delta in a direction that looks exactly like a substantive
result**, and it does so quietly.

Therefore: **assert the fit-quality condition per arm and report it** —
convergence for an iterative estimator, or whatever the equivalent is for the
estimator in use — and report that status *before* reporting any delta.

**Never rely on a library warning.** Not because warnings are easy to miss,
but because **this project's measurement scripts are run with `-W ignore`** —
`sklearn` and `numpy` otherwise emit enough noise to bury the output. On the
real job a warning is therefore **not a weak signal, it is no signal**, and a
suppressed warning looks identical in the output to a check that passed.

Any check that depends on a warning being visible must be rewritten as an
assertion or an explicitly returned status. This is structural, not a matter
of care: the suppression is deliberate and will stay.

See §12g for the instance that prompted this.

## scikit-learn convergence

`LogisticRegression` on the 326-feature spatial arm does NOT converge at
`max_iter=300`, while the 2-feature bulk arm does. Left unchecked that
understates the spatial arm specifically and biases a spatial-vs-bulk delta
toward zero. `phase4_model_a.py` runs at `max_iter=5000` and asserts
convergence per fit, collecting any failures into `NONCONVERGED` and printing
them at the end. Keep that check in any model comparison added later.
