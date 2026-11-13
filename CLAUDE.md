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

Several measurement scripts take minutes to tens of minutes
(`scripts/spatial/measure_battery_null.py`,
`scripts/spatial/phase4_model_a.py`). Run them in the background rather than
in the foreground, and **pass `-u`** — without it, output is buffered and the progress
file stays empty, which looks indistinguishable from a hung process.

`scripts/spatial/phase4_model_a.py` caches its expensive per-`(trace, cell)` label-building
step to `.model_a_cache/` (gitignored). Delete that directory to force a
rebuild; keep it to iterate on the modelling cheaply.

## Data directories

`data/raw/` and `data/processed/` are both gitignored and neither is
redistributed — the raw data is third-party licensed, and the processed load
traces are deterministically regenerable from it. Reasoning and regeneration
instructions: `docs/spatial/external_fault_datasets.md` §3 and the README.

## Standing test: could this tooling choice fail asymmetrically between the arms?

**Ask this of every multi-arm comparison, before trusting a delta.** Phrased as
a question, in the manner of `docs/spatial/synthetic_generator_notes.md` §10.

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
toward zero. `scripts/spatial/phase4_model_a.py` runs at `max_iter=5000` and asserts
convergence per fit, collecting any failures into `NONCONVERGED` and printing
them at the end. Keep that check in any model comparison added later.

## Where to put artifacts worth keeping

**In the repo (gitignored if need be), never `/tmp` or a scratchpad.** Both are
cleared without warning — a multi-hour run log and a deliberately preserved
copy of a superseded run were both lost to a 12-day gap, along with the
background task output that held the traceback.

The protection failed because of WHERE it was put, not whether it was done.
Copying a log "somewhere safe" is not safe if that somewhere is temporary.
Use a gitignored directory inside the repo (`run_logs/` is already ignored),
so the file survives anything short of a deliberate clean.

Corollary for long runs: write results incrementally to such a path as they
are produced. A run that only materialises its output at the end loses
everything to a mid-run death, including the evidence of why it died.

## Verifying a commit changed tracking state

**After any commit meant to change what is TRACKED, verify with
`git ls-files` rather than trusting that the commit succeeded. A commit that
reports success is not evidence the index changed the way you meant.**

The specific trap: **supplying pathspecs to `git commit` commits the
WORKING-TREE state for those paths and silently discards whatever was staged
for them.** So

```
git rm --cached some/file      # stages a deletion
git commit -- some/ other/     # DISCARDS it, if the file still exists on disk
```

commits nothing for `some/`, reports success, and leaves the file tracked.
This happened once here: 188cd2a asserted two CC-BY-NC-derived npz files had
been untracked, they were still tracked, and `.gitignore` has no effect on
tracked files -- so third-party data stayed in a public tree while the commit
message and the docs both said it had been withdrawn. Fixed in a83784d by
staging and committing with NO pathspec.

Pathspecs on `git commit` are still the right tool for splitting unrelated
changes across commits -- that use is fine, since it only ever commits content
that is present. The failure mode is specific to DELETIONS and other
index-only state.

## Standing check: verify the operation, not the report

**A mechanism that REPORTS on an operation is not evidence the operation
happened.** Three instances in this project, all different mechanisms, all the
same shape:

| what reported | what it asserted | what was true |
|---|---|---|
| a commit message (188cd2a) | two npz files were untracked | they stayed tracked; `git commit` with pathspecs had discarded the staged deletion |
| a results table header (Sec. 12h) | "the asymmetry, NOT equalized" | printed directly above four identical numbers; a cap had equalised them |
| a launch wrapper | run completed, exit 0 | Python died at exit 4; the wrapper reported its own `echo` |

So: **check the operation through an independent channel, not through the
thing whose job was to perform or describe it.** `git ls-files` after a
tracking change, not the commit's success. The data, not the header you wrote
over it. The log's own terminal line, not the task notification.

The specific wrapper bug, because it will recur: a command ending

```
python script.py > log 2>&1; echo "EXIT=$?" >> log
```

exits with the status of the `echo`, which is always 0. The harness therefore
reports success for a failed run, and a crash is indistinguishable from a
clean finish in the notification. The `EXIT=` line inside the log is correct;
the reported status is not. **Read the log's terminal line.**

This is the second standing question in this file, alongside the
arm-asymmetry one above. Both exist because a plausible-looking wrong answer
is more dangerous than an obvious failure.

## Standing check: a split's name is not evidence of what it holds out

**Before trusting any "held-out X" result, list which X are in each fold's
train and test sides and count the overlap.** Label-only; takes seconds.

The instance (generator notes Sec. 17b, 18i): cells were seeded as
`BASE_SEED + 1000 * cell_idx`, with no trace term, so the same five cells
recurred under every load trace. The split called "unseen CELL"
(`GroupKFold` on `(trace, cell)`) held out a *combination* and never a cell.
Model A's headline spatial advantage (+0.07 to +0.12) was measured that way
and **reverses** (-0.18 to -0.22) when the cell is actually held out.

- For the old experiments: `scripts/spatial/posthoc_confirm_cell_leak.py`.
- For `hfc` datasets: `python -m hfc check-leaks DATASET [--split-file ...]`.
  `hfc train` refuses to run if it fails.
- When generating: every design factor goes into the seed
  (`SeedSequence(seed, spawn_key=(...))` in `hfc/gen.py`), and group IDs are
  exported with the data.

This is the third standing check in this file. Same shape as the other two:
the thing that *describes* the operation ("grouped by cell") was taken as
evidence the operation happened.

## The `hfc` package and post-hoc runs

- `python -m hfc ...` from the repo root; tests with
  `./hfc_env/Scripts/python.exe -W ignore -m unittest discover -s tests`
  (**pytest is not installed**).
- Launch long hfc commands through `scripts/hfc_run.sh <name> <args>` and
  Sec. 17-style runs through `scripts/spatial/posthoc_run.sh`; both write a
  log ending `EXIT=<python's code>`. `posthoc_run.sh` refuses to overwrite an
  existing log -- one attempt per pre-registered run.
- Generated datasets go in `hfc_data/` (gitignored, 1.8 GB for `extensive`).
  `np.savez` is NOT byte-reproducible (zip timestamps); use
  `hfc.gen.save_npz_deterministic`.
- joblib workers at 2 BLAS threads each: 3 concurrent jobs x 6 workers
  oversubscribes 20 cores, and on battery the machine throttles hard and may
  sleep mid-run. Runs write results per cell, so a sleep costs time, not
  results.
- A numpy string array silently truncates to its dtype width (`U24` cut
  `outside_training_envelope`). Size reason/label arrays generously and test
  the exact string.
