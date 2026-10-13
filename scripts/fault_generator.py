"""
Synthetic fault-onset generator driven by the real dynamic-only load trace.

Grounding / conventions (verified by reading the source before writing this,
not assumed):

- Grid shape and per-state spatial patterns are grounded in notebook 01
  (`01_spatial_viz_sandbox.ipynb`, cells `d3fb30fd`/`899e7d48`/`a1edd97b`):
  4x4 segment grid, the same row/column gradient + noise + clip formulas per
  state. Notebook 01 itself is NOT imported or modified -- its generators are
  static, single-shot, discretely switched by state name. Here they are
  reimplemented as continuous functions of a `severity` in [0, 1], built by
  linearly interpolating notebook 01's NORMAL formula (severity=0) toward its
  fully-formed fault formula (severity=1), so severity=1 reproduces notebook
  01's numbers exactly and intermediate severities are a physically smooth
  blend, not a new invented pattern.
- CV(J) is copied verbatim from notebook 02's verified formula (cells
  `7f90ce08` + `a70af676`): population std (ddof=0) of |current_grid| divided
  by mean(|current_grid|), per timestep. Not reimplemented independently.
- Ramp-rate and dwell thresholds (0.02/s, 0.10, 0.40, 10s minimum sustained
  duration) are the exact values `extract_load_trace.py`'s dynamic-only
  report already derived from this trace's own distribution (see that
  script's "DYNAMIC-ONLY TRACE ... USE THESE FOR TASK 2 TRIGGERS" section).
  `find_ramp_events` is imported and reused as-is from that module; a sibling
  `find_dwell_events` is added here (mirrors `find_dwell_fraction`'s
  contiguous-run detection, but returns each run's own boundaries -- needed
  to build onset trajectories, not just an aggregate fraction).
- The dynamic-only trace concatenates 3 real driving segments with 2
  artificial 1s seams (`segment_id` marks which). Every trigger-detection and
  episode-building step below operates strictly within one segment's own
  index range; nothing is ever computed or carried across a segment boundary.

Fault dynamics (severity(t) shape per type) are engineering choices grounded
in the literal onset-speed/shape description from the Task 2 brief -- see
each `_build_*_episode` function's comment for the specific reasoning. Every
numeric constant in those functions that is a calibration choice rather than
something derived from the real trace or notebook 01/02 is labeled inline as
such, at its point of definition (same real/literature/tuned separation
`docs/real_data_notes.md` uses for the dataset itself).

SCOPE NOTE -- starvation is a single combined fault type, not split into air-
vs. hydrogen-starvation subtypes, even though both are physically distinct
failure modes with (per the literature) different spatial signatures (air
starvation: cathode-flow-path column gradient, per Yu et al. 2024, which is
what notebook 01's `generate_starvation` and this file's
`generate_starvation_grid` model; hydrogen starvation would need a different,
anode-flow-path-oriented spatial pattern that isn't implemented here). This
is a deliberate current-scope simplification carried over from the Task 2
brief's own phrasing ("STARVATION (air or hydrogen)"), not an oversight --
splitting them into two distinct fault types with their own spatial grounding
is deferred to a later task.

KNOWN LIMITATION -- CV(J) does NOT discriminate flooding from healthy
operation on this generator's output: baseline (no-fault) CV(J) mean is
0.076 vs. flooding-at-severity>0.5 mean 0.086 -- flooding is not lower, the
opposite of the raw sigma(J) intuition from notebook 01 / the Task 1
real-data finding. Cause: CV(J) normalizes sigma(J) by mean(|J|), and
flooding drives both down together (notebook 01: J_std 0.047->0.041, J_mean
0.697->0.432), so the normalization cancels almost all of the raw-sigma
signal that made flooding look distinct. CV(J) remains valid as the
headline spatial-vs-bulk diagnostic argument (spatial nonuniformity carries
information bulk voltage doesn't), but it is NOT sufficient alone as a
fault-TYPE classifier feature -- downstream classification must use the full
spatial grid, or CV(J) jointly with mean(|J|), to actually discriminate
flooding from healthy. Full writeup: docs/synthetic_generator_notes.md.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from extract_load_trace import find_ramp_events

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

DEFAULT_TRACE_PATH = os.path.join(PROCESSED_DIR, "normal_flow_fc-dlc_load_trace_dynamic_only.npz")

ROWS, COLS = 4, 4  # notebook 01's grid convention -- the FROZEN default

# Native resolution for generated runs. 4x4 was a vestige of an abandoned PCB
# hardware design; 18x18 matches the real validation data natively, which
# matters because CV(J) computed on this family is resolution-dependent
# (Sec. 7c: paired 18x18/4x4 ratio 1.12 on average, up to 2.9x for sharp
# transitions near the centre). Generating at the validation resolution
# removes that class of error entirely.
#
# Every grid function below still DEFAULTS to ROWS/COLS = 4x4 when called
# directly, so the frozen path, the v2 regression test, and any existing
# direct call are byte-for-byte unaffected. Only generate_labeled_run opts in.
DEFAULT_GRID_SHAPE = (18, 18)

# Trigger thresholds -- lifted directly from extract_load_trace.py's
# dynamic-only-trace report, not re-derived here.
RAMP_THRESHOLD = 0.02           # /s, starvation trigger
LOW_LOAD_THRESH = 0.10          # flooding trigger
HIGH_LOAD_THRESH = 0.40         # drying trigger
MIN_DWELL_S = 10.0              # sustained-dwell minimum, matches extract_load_trace.py

# Placeholder "confirmed fault" cutoff -- per Task 2 brief this is
# "still-to-be-defined". Used only to derive each episode's own precursor
# window from its actual onset speed (never to gate/hide labeling elsewhere).
CLASSIFICATION_THRESHOLD = 0.5

# CALIBRATION CHOICE: window (seconds) on either side of an internal
# segment-boundary seam flagged via near_seam_buffer. 90s is set to roughly
# match the longest precursor window actually observed across all fault
# types in this run (drying, ~76s -- see the Task 2 review's onset-event
# summary) plus a margin, not the starvation precursor window specifically
# (starvation's is much shorter, ~1-3s). Any window built from a sample this
# close to a seam would include the hard reset-then-reonset artifact
# described in generate_labeled_run's docstring.
NEAR_SEAM_WINDOW_S = 90.0

BASE_SEED = 20260810  # fixed seed -> fully reproducible noise realizations

FAULT_COLORS = {"starvation": "#d7191c", "flooding": "#2166ac", "drying": "#e08214"}


# ---------------------------------------------------------------------------
# Spatial grid generators -- continuous-by-severity, grounded in notebook 01
# ---------------------------------------------------------------------------

def generate_normal_grid(seed: int, rows: int = ROWS, cols: int = COLS) -> np.ndarray:
    """
    Verbatim notebook 01 `generate_normal` (cell 899e7d48), reseeded per call.

    FROZEN at the 4x4 default: generate_normal_grid(seed) must stay
    byte-identical to notebook 01 forever, and that is asserted in the tests.
    Larger rows/cols evaluate the same formula at finer resolution -- a
    generalization of the formula, NOT notebook 01's output, and not covered
    by the frozen contract.
    """
    rng = np.random.default_rng(seed)
    row_grad = np.linspace(1.0, 0.65, rows)
    col_grad = np.linspace(1.0, 0.72, cols)
    gradient = np.outer(row_grad, col_grad)
    base = 0.45 + 0.35 * gradient
    noise = rng.normal(0, 0.02, (rows, cols))
    return np.clip(base + noise, 0.30, 0.90)


# ---------------------------------------------------------------------------
# Healthy spatial-structure model v2 (Phase 3, Task 3 review) -- ADDITIVE.
# generate_normal_grid above is untouched: it remains notebook 01's exact
# formula and the severity=0 baseline every fault-state interpolation in
# this file is still built from. This section does not wire into that
# interpolation yet -- fault-state generation is not modified here.
#
# Motivation: Task 3 review found synthetic healthy CV(J) (~0.067,
# generate_normal_grid) is ~5.7x below real healthy CV(J) (~0.381,
# Normal_Flow), traced to sigma(J) being ~4.67x too low once the units bug
# was corrected (not the operating point -- see
# docs/synthetic_generator_notes.md Sec. 4). Decomposing real Normal_Flow's
# spatial pattern (normalize each timestep by its own spatial mean, isolate
# the persistent time-averaged shape from the per-timestep residual) found
# the persistent shape explains 99.6% of the spatial variance (static_std=
# 0.3802 vs temporal_std=0.0253) -- static-dominant, not a temporal-noise
# problem. That persistent shape correlates with distance-from-grid-center
# at r=0.936, NOT with row/column index (both ~0) -- i.e. NOT the
# inlet-to-outlet flow-path gradient notebook 01 assumed. A radial quadratic
# fit explains R^2=0.906 of the shape by itself.
#
# This model reproduces that FORM (radially symmetric, quadratic falloff),
# grounded in a real physical mechanism documented in segmented large-area
# PEMFC literature: non-uniform bolt-frame clamping pressure produces higher
# contact pressure (lower contact resistance, higher local current) toward
# the clamped periphery and lower pressure/current near the center --
# see the literature note on RADIAL_EDGE_RATIO below for sources and the
# amplitude-sourcing decision. The specific real-data regression
# coefficients (a=-0.264, b=0.252, c=-0.0087 from the quadratic fit above)
# are NOT used here -- only the qualitative form and the R^2 finding that
# validated it. Parameter values below come from physical reasoning and
# (where found) literature, never from fitting to the real grids.
# ---------------------------------------------------------------------------

# Physics-derived, NOT chosen to fit anything: contact pressure under
# bolt-frame clamping falls off smoothly from the clamped periphery toward
# the cell center. A quadratic falloff in radius is the natural leading-order
# term for a smooth pressure field peaked at the frame and dipping at the
# center (the first non-trivial even term in a radially-symmetric Taylor
# expansion around the center). The real-data radial fit (R^2=0.906, found
# without fitting this exponent -- it was assumed then checked) supports
# this form; it is not derived from that fit.
RADIAL_EDGE_EXPONENT = 2

# CALIBRATION CHOICE -- literature search performed (4 targeted queries
# across this and the prior review pass: segmented-cell edge/clamping
# current-density distribution, clamping-pressure current-density
# correlation, commercial-size PEMFC non-uniform current density, bolt-frame
# periphery current-density ratios). Multiple independent papers confirm the
# MECHANISM (uneven clamping pressure -> uneven contact resistance -> higher
# local current at higher-pressure regions, typically the bolted periphery
# of large-area cells) but none gave a stated, citable edge-to-center or
# max/min current-density RATIO precise enough to use directly:
#   - Analysis of local current density, temperature, and mechanical
#     pressure distributions in an operating PEMFC under variable
#     compression -- https://www.sciencedirect.com/science/article/pii/S0306261925009171
#   - Advanced parametric model for analysis of the influence of channel
#     cross section dimensions and clamping pressure on current density
#     distribution in PEMFC -- https://www.sciencedirect.com/science/article/abs/pii/S0306261921014094
#   - Investigation of the non-uniform distribution of current density in
#     commercial-size PEMFCs -- https://www.sciencedirect.com/science/article/abs/pii/S0378775320301397
#   - Current density and temperature distribution measurement and
#     homogeneity analysis for a large-area PEMFC -- https://www.sciencedirect.com/science/article/abs/pii/S0360544221021708
# STATED LIMITATION: absence of a citable magnitude. RADIAL_EDGE_RATIO=2.0 is
# therefore an order-of-magnitude engineering assumption (idealized
# center-to-corner ratio), not a literature value, bracketed by 1.5x/2.5x in
# the Phase 3 sensitivity analysis (docs/synthetic_generator_notes.md) so the
# result's dependence on this specific choice is visible, not hidden behind
# one precise-looking number. Never adjusted to make CV(J) hit 0.381.
RADIAL_EDGE_RATIO = 2.0

# CALIBRATION CHOICE: temporal fluctuation kept clearly subordinate to the
# static structure by construction -- 1/15 of the static shape's own spatial
# std (within the instructed 1/10-1/20 range). NOT fit to the real ~1/15
# static:temporal ratio found during Task 3 review (that number is used only
# as a post-hoc comparison in docs/synthetic_generator_notes.md); documented
# here as a realism add-on, not a calibrated quantity.
TEMPORAL_SUBORDINATION_FACTOR = 15.0

# ---------------------------------------------------------------------------
# ADOPTED healthy family (docs/synthetic_generator_notes.md Sec. 6c, bounds
# Sec. 6g). Supersedes the v2 form above for the severity=0 path; v2 itself is
# kept unchanged rather than deleted, same as every other superseded artifact
# in this project.
#
# Form:  g     = [sig((rho-r0)/w) - sig(-r0/w)] / [sig((1-r0)/w) - sig(-r0/w)]
#        S_raw = c + (1-c)*g        so S_raw(0)=c, S_raw(r_max)=1 exactly
#        S     = S_raw / mean(S_raw)
#
# Chosen because the real templates inflect and saturate (Sec. 6b) -- a
# data-dependence stated plainly in Sec. 6d and NOT to be softened. The
# physical picture (a bolted frame saturating at both limits) motivates the
# shape; it does not derive it, and it fixes neither r0 nor w.
#
# Bounds below are ADOPTED (Sec. 6g). Asymmetry that must be carried forward
# (Sec. 6h): c has thin but real empirical content (1.48x observed
# cross-session variation, itself confounded); r0 and w are fixed to within 5%
# and 2% in every session available, so their ranges are ASSUMED, not
# observed. Any downstream finding depending on r0/w breadth must be reported
# as resting on assumption.
HEALTHY_C_RANGE = (0.05, 0.60)
HEALTHY_R0_RANGE = (0.25, 0.70)
HEALTHY_W_RANGE = (0.04, 0.35)
HEALTHY_TEMPORAL_RATIO_RANGE = (8.0, 60.0)

# Sampling is UNIFORM over these ranges, by decision, not oversight
# (Sec. 7d). It skews the family toward cells more spatially uniform than any
# observed (78% below the real minimum, 11% above the real maximum). That
# direction makes false-alarm avoidance HARDER, not easier, so a model that
# stays quiet across this family makes a stronger claim than one validated on
# a family centred on the real data. Log-uniform sampling of c would shift
# mass toward the real end and weaken the test; it is rejected for that reason.


def _radial_healthy_shape(edge_ratio: float = RADIAL_EDGE_RATIO,
                          rows: int = ROWS, cols: int = COLS) -> np.ndarray:
    """
    Persistent (static) normalized spatial shape S(i,j), mean 1 over the
    grid -- quadratic falloff in distance from grid center
    (RADIAL_EDGE_EXPONENT=2), amplitude set so the IDEALIZED center (r=0) to
    corner (r=r_max) ratio equals `edge_ratio`. On this file's 4x4 grid, no
    cell sits exactly at r=0, so the realized ratio between the actual
    nearest-to-center and corner CELLS is somewhat less extreme than
    `edge_ratio` -- a known, stated consequence of the coarser (4x4 vs.
    real 18x18) discretization, not a separate free parameter.
    """
    row_idx, col_idx = np.indices((rows, cols))
    center_r, center_c = (rows - 1) / 2.0, (cols - 1) / 2.0
    r = np.sqrt((row_idx - center_r) ** 2 + (col_idx - center_c) ** 2)
    r_max = r.max()
    a_edge = edge_ratio - 1.0
    s_raw = 1.0 + a_edge * (r / r_max) ** RADIAL_EDGE_EXPONENT
    return s_raw / s_raw.mean()  # renormalize -> shape only (mean=1), carries no load information


def generate_healthy_grid_v2(load_level: float = 0.697, seed: int = 0,
                             edge_ratio: float = RADIAL_EDGE_RATIO,
                             rows: int = ROWS, cols: int = COLS) -> np.ndarray:
    """
    Phase 3 healthy spatial-structure model (additive -- see section header
    above). J(i,j) = load_level * S(i,j) * (1 + noise(i,j)).

    `load_level` defaults to 0.697 A/cm^2, matching generate_normal_grid's
    (notebook 01's) synthetic NORMAL operating point -- for comparability
    with the existing generator, not fit to real data.

    S is the persistent radial shape, renormalized to mean 1, so it carries
    NO load information; `load_level` scales the whole grid separately. This
    makes CV(J) load-invariant BY CONSTRUCTION: sigma(J)/mean(J) depends only
    on S and the noise term, never on load_level. Stated here as the design
    reasoning -- verified explicitly (not just argued) in the Phase 3
    validation script / docs/synthetic_generator_notes.md.
    """
    rng = np.random.default_rng(seed)
    S = _radial_healthy_shape(edge_ratio, rows, cols)
    sigma_temporal = S.std() / TEMPORAL_SUBORDINATION_FACTOR
    noise = rng.normal(0, sigma_temporal, S.shape)
    return load_level * S * (1.0 + noise)


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def logistic_healthy_shape(c: float, r0: float, w: float,
                           rows: int = ROWS, cols: int = COLS) -> np.ndarray:
    """
    Persistent healthy spatial template -- the ADOPTED form (Sec. 6c).

    Normalized to mean 1, so it carries NO load information and CV(J) stays
    load-invariant by construction, exactly as in v2.

    NOTE (Sec. 7c): the continuous field is resolution-independent by
    construction (it depends only on rho = r/r_max), but CV(J) computed on it
    is NOT -- it is a sample statistic over a resolution-dependent sampling of
    radius. Measured paired ratio 18x18/4x4 is 1.12 on average, up to 2.9x for
    sharp transitions near the centre (small w, small r0). Grid size therefore
    matters and must be stated wherever CV(J) from this form is reported.
    """
    row_idx, col_idx = np.indices((rows, cols))
    r = np.sqrt((row_idx - (rows - 1) / 2.0) ** 2 + (col_idx - (cols - 1) / 2.0) ** 2)
    rho = r / r.max()
    w = max(float(w), 1e-4)
    lo, hi = _logistic(-r0 / w), _logistic((1.0 - r0) / w)
    g = (_logistic((rho - r0) / w) - lo) / (hi - lo)
    raw = c + (1.0 - c) * g
    return raw / raw.mean()


def sample_healthy_params(rng: np.random.Generator) -> dict:
    """
    Draw ONE simulated cell's healthy parameters, uniform over the adopted
    ranges. Drawn once per run, not per timestep: the real template is
    persistent (99.6% of spatial variance is static, Sec. 5a), so a template
    that changed every timestep would contradict the structure being modelled.
    """
    return {
        "c": float(rng.uniform(*HEALTHY_C_RANGE)),
        "r0": float(rng.uniform(*HEALTHY_R0_RANGE)),
        "w": float(rng.uniform(*HEALTHY_W_RANGE)),
        "temporal_ratio": float(rng.uniform(*HEALTHY_TEMPORAL_RATIO_RANGE)),
        # Starvation's directional-arm flow axis (Sec. 11d DECISION point 3):
        # drawn ONCE per cell and held fixed across that cell's episodes,
        # because a real cell has one flow field -- per-episode randomization
        # would be physically incoherent. Appended last on purpose, so the
        # four draws above are byte-identical to before this field existed.
        "starvation_axis_rad": float(rng.uniform(0.0, 2.0 * np.pi)),
    }


def generate_healthy_grid_v3(params: dict, seed: int, load_level: float = 0.697,
                             rows: int = ROWS, cols: int = COLS) -> np.ndarray:
    """
    One timestep of a healthy grid for the cell described by `params`.

    The template is fixed by `params` (persistent); only the temporal noise
    varies with `seed`. `load_level` keeps the same 0.697 default as v2 and
    generate_normal_grid so that swapping the healthy model changes the grid
    SHAPE only, not its mean -- which is what makes the swap's effect
    attributable.
    """
    rng = np.random.default_rng(seed)
    S = logistic_healthy_shape(params["c"], params["r0"], params["w"], rows, cols)
    sigma_temporal = S.std() / params["temporal_ratio"]
    noise = rng.normal(0, sigma_temporal, S.shape)
    return load_level * S * (1.0 + noise)


# ---------------------------------------------------------------------------
# Phase 3b: fault states as MODULATIONS of the shared healthy template
# (Sec. 11c). Supersedes the interpolation-toward-independent-endpoints form
# below, which is kept reachable and undeleted per the project's supersession
# convention.
#
# J_fault(i,j;s) = J_healthy(i,j) * M_fault(i,j;s),  M_fault(i,j;0) = 1
#
# Continuity (Sec. 11b Test 1) therefore holds BY CONSTRUCTION, not by tuning,
# the same way load-invariance holds by construction through S / mean(S).
#
# SCOPE, Sec. 11a: these patterns are mutually coherent and literature-
# grounded. They are NOT validated. No fault-labelled spatially-resolved data
# for this cell exists or is obtainable without hardware, so there is no
# capability check, no identifiability check and no resolution corroboration
# available for them -- permanently, not pending future work.
# ---------------------------------------------------------------------------

# TUNED-OR-ASSUMED (Sec. 5d labelling convention). Not fitted to anything --
# nothing exists to fit them to. Set so that full severity gives a physically
# plausible magnitude loss and the literature-reported DIRECTION of CV(J)
# movement. The direction is the claim; the magnitudes are not.
DRYING_UNIFORM_LOSS = 0.20    # severity-1 fractional loss everywhere
DRYING_CENTRE_LOSS = 0.45     # additional severity-1 loss at the hottest cell
FLOODING_UNIFORM_LOSS = 0.25
FLOODING_EDGE_LOSS = 0.40     # additional severity-1 loss at the coolest cell

# TUNED-OR-ASSUMED, and tuned under knowledge of the acceptance test -- read
# Sec. 11g before relying on it. The severity -> amplitude map is CONVEX:
# amplitude scales as severity**SEVERITY_CONVEXITY, not linearly.
#
# Physically this is threshold-like behaviour: flooding has a percolation
# threshold (liquid water does not impede current until it actually blocks
# pores) and drying follows a nonlinear conductivity-vs-water-content curve.
# But per Sec. 11g this rationale was reached AFTER a linear map failed
# Sec. 11b Test 2, not independently of it.
#
# The acceptance battery CANNOT constrain this value: p=1.5 and p=2.0 both
# pass every test and nothing distinguishes them. It is therefore a free
# parameter, and any downstream result sensitive to p specifically must be
# flagged the way Sec. 6h flags r0/w breadth.
#
# severity**p leaves severity=1 bit-identical for any p, so this choice
# affects only the low-severity region -- the only region that failed.
SEVERITY_CONVEXITY = 2.0

# Starvation amplitudes. Same tuned-or-assumed status as the pair above, and
# the same bound: 1 - (0.25 + 0.45) = 0.30 > 0, so M stays positive and
# clip-free at full severity.
STARVATION_UNIFORM_LOSS = 0.25
STARVATION_AXIS_LOSS = 0.45      # directional arm, along the drawn axis
STARVATION_COUPLED_LOSS = 0.45   # no-axis arm, proportional to local current

# Sec. 11d/11h: starvation is generated TWO ways and NEITHER is the default.
#   "directional" -- O2 depletion along a flow-path axis whose orientation is
#       drawn uniformly per simulated cell and held fixed across that cell's
#       episodes. Encodes the literature-supported mechanism while refusing to
#       encode a direction the real data supports (Sec. 11d).
#   "no_axis" -- suppression proportional to LOCAL CURRENT DENSITY. O2 depletes
#       fastest where current is highest, a self-limiting mechanism needing no
#       flow path at all.
# Callers must choose explicitly; there is no fallback. A uniform third option
# was rejected in Sec. 11h because it would carry zero spatial information and
# so decide Phase 4's starvation comparison by arithmetic.
STARVATION_ARMS = ("directional", "no_axis")

# Every fault is now rebased onto the radial baseline, so the knowingly-
# inconsistent mixed-basis state described in Sec. 8 is resolved: no run mixes
# a radial healthy template with diagonal-gradient faults any more. The
# superseded interpolation forms remain reachable only when
# healthy_params=None, i.e. the legacy notebook-01 path taken as a whole.
PHASE3B_FAULTS = ("drying", "flooding", "starvation")


def _thermal_centre_bias(rows: int = ROWS, cols: int = COLS) -> np.ndarray:
    """
    The centre-bias profile the TEMPERATURE model uses -- edge segments cooler
    via heat-sinking at the clamping plates. Factored out so the thermal
    grounding of drying and flooding (Sec. 11c) is literally the same array
    the temperature field is built from, rather than a re-derivation that
    could silently drift away from it.

    CAVEAT carried from Sec. 11c: the per-state (base_t, center_scale,
    j_scale) triples remain notebook-01-derived and untouched by the healthy
    rework. "Grounded internally" means grounded on a quantity the generator
    already produces -- NOT grounded on validated physics. What drying and
    flooding use is the SIGN of this gradient, which is the robust part; no
    amplitude here depends on those triples.
    """
    row_center = np.interp(np.linspace(0, 1, rows), np.linspace(0, 1, 4),
                           np.array([0.85, 1.0, 1.0, 0.85]))
    col_center = np.interp(np.linspace(0, 1, cols), np.linspace(0, 1, 4),
                           np.array([0.88, 1.0, 1.0, 0.88]))
    return np.outer(row_center, col_center)


def _thermal_hot_weight(rows: int = ROWS, cols: int = COLS) -> np.ndarray:
    """Centre-bias rescaled to [0, 1]: 1 at the hottest cell, 0 at the coolest."""
    cb = _thermal_centre_bias(rows, cols)
    lo, hi = float(cb.min()), float(cb.max())
    if hi - lo < 1e-12:
        return np.zeros_like(cb)
    return (cb - lo) / (hi - lo)


def _axis_weight(theta: float, rows: int, cols: int) -> np.ndarray:
    """
    Normalized position along an axis at angle `theta`, in [0, 1]: 0 at the
    inlet end, 1 at the outlet end. Used only by starvation's DIRECTIONAL arm.

    The orientation is a per-cell draw, not a fixed direction -- Sec. 11d: the
    sensor-grid orientation of the air and H2 axes is undocumented and the real
    data holds no flow-tracking component to infer it from, so randomizing it
    answers ignorance of direction as well as doubt about the mechanism.
    """
    row_idx, col_idx = np.indices((rows, cols))
    x = col_idx - (cols - 1) / 2.0
    y = row_idx - (rows - 1) / 2.0
    proj = np.cos(theta) * x + np.sin(theta) * y
    lo, hi = float(proj.min()), float(proj.max())
    if hi - lo < 1e-12:
        return np.zeros_like(proj, dtype=float)
    return (proj - lo) / (hi - lo)


def _current_coupled_weight(params: dict, rows: int, cols: int) -> np.ndarray:
    """
    Normalized local current density of this cell's own healthy template, in
    [0, 1]: 1 where the cell produces most current. Used only by starvation's
    NO-AXIS arm.

    Grounding: O2 depletes fastest where current is highest, which is a genuine
    self-limiting mechanism and needs no flow path. Note the consequence
    pre-registered in Sec. 11h -- this cell's template is a bathtub, high at
    the EDGES, so this weight is edge-heavy, which is also where flooding's
    suppression falls. The predicted Test 3 risk on the flooding-starvation
    pair follows directly from that, and is NOT to be tuned away.
    """
    S = logistic_healthy_shape(params["c"], params["r0"], params["w"], rows, cols)
    lo, hi = float(S.min()), float(S.max())
    if hi - lo < 1e-12:
        return np.zeros_like(S)
    return (S - lo) / (hi - lo)


def fault_modulation(fault_type: str, severity: float,
                     rows: int = ROWS, cols: int = COLS,
                     params: dict | None = None,
                     starvation_arm: str | None = None) -> np.ndarray:
    """
    M_fault(i,j;s). Exactly 1.0 everywhere at s=0, for every fault type --
    which is what makes Sec. 11b Test 1 structural rather than tuned.

    drying   -- membrane ionic resistance rises where water content is lowest,
                i.e. where the cell is HOTTEST, so suppression is weighted to
                the centre. Deepens the healthy bathtub => CV(J) rises.
    flooding -- liquid water condenses preferentially in the COLDEST regions,
                so suppression is weighted to edges and corners. Flattens the
                bathtub => CV(J) falls. Its accumulate/breakthrough
                oscillation lives in the episode's severity trajectory
                (_build_flooding_episode), not here, so M stays monotone in s.

    Both amplitudes are bounded so M > 0 everywhere at s=1 (worst cases 0.35),
    which keeps the form clip-free and therefore keeps d(s) smooth for Test 1.

    The severity axis enters as severity**SEVERITY_CONVEXITY (Sec. 11g), so
    amplitude is convex in severity rather than linear. severity=0 still gives
    M == 1 exactly and severity=1 is unchanged for any exponent.
    """
    s = float(np.clip(severity, 0.0, 1.0)) ** SEVERITY_CONVEXITY
    w_hot = _thermal_hot_weight(rows, cols)
    if fault_type == "drying":
        return 1.0 - s * (DRYING_UNIFORM_LOSS + DRYING_CENTRE_LOSS * w_hot)
    if fault_type == "flooding":
        return 1.0 - s * (FLOODING_UNIFORM_LOSS
                          + FLOODING_EDGE_LOSS * (1.0 - w_hot))
    if fault_type == "starvation":
        if starvation_arm not in STARVATION_ARMS:
            raise ValueError(
                f"starvation requires an explicit arm, one of "
                f"{STARVATION_ARMS}; got {starvation_arm!r}. Sec. 11d/11h: "
                f"neither arm is the default and both are run.")
        if params is None:
            raise ValueError(
                "starvation needs the cell's healthy params: the directional "
                "arm reads its drawn axis, the no-axis arm its template.")
        if starvation_arm == "directional":
            if "starvation_axis_rad" not in params:
                raise ValueError(
                    "params lacks 'starvation_axis_rad'; draw params with "
                    "sample_healthy_params so the axis is a per-cell draw.")
            w = _axis_weight(params["starvation_axis_rad"], rows, cols)
            amp = STARVATION_AXIS_LOSS
        else:
            w = _current_coupled_weight(params, rows, cols)
            amp = STARVATION_COUPLED_LOSS
        return 1.0 - s * (STARVATION_UNIFORM_LOSS + amp * w)
    raise ValueError(
        f"fault_modulation: unknown fault_type {fault_type!r} "
        f"(Phase 3b covers {PHASE3B_FAULTS})")


def fault_template_v3(fault_type: str, severity: float, params: dict,
                      load_level: float = 0.697,
                      rows: int = ROWS, cols: int = COLS,
                      starvation_arm: str | None = None) -> np.ndarray:
    """
    The DETERMINISTIC fault template -- no temporal noise draw. This is what
    Sec. 11b Test 1 compares, since including noise would make d(s) depend on
    the draw rather than on severity.
    """
    S = logistic_healthy_shape(params["c"], params["r0"], params["w"], rows, cols)
    return load_level * S * fault_modulation(
        fault_type, severity, rows, cols, params, starvation_arm)


def generate_fault_grid_v3(fault_type: str, severity: float, params: dict,
                           seed: int, load_level: float = 0.697,
                           rows: int = ROWS, cols: int = COLS,
                           starvation_arm: str | None = None) -> np.ndarray:
    """
    One timestep of a faulted grid for the cell described by `params`.

    The healthy grid is generated first -- same params, same seed, so the same
    temporal noise realization -- and then modulated. Multiplying AFTER the
    noise is applied is deliberate: these mechanisms suppress local current as
    a fractional effect on whatever the cell was actually producing, so the
    fluctuation scales down with the signal rather than persisting at full
    amplitude over a suppressed mean.
    """
    healthy = generate_healthy_grid_v3(params, seed, load_level, rows, cols)
    return healthy * fault_modulation(
        fault_type, severity, rows, cols, params, starvation_arm)


# ---------------------------------------------------------------------------
# SUPERSEDED fault-state forms (Sec. 8) -- interpolation from notebook 01's
# diagonal flow-path gradient toward independent endpoints. Kept reachable,
# not deleted. Still the active path for starvation until Sec. 11e's work
# lands, and for any call that passes healthy_params=None.
# ---------------------------------------------------------------------------

def generate_starvation_grid(severity: float, seed: int,
                         rows: int = ROWS, cols: int = COLS) -> np.ndarray:
    """
    O2 starvation, continuous. Interpolates NORMAL's row/col gradient, base
    range, and clip bounds toward notebook 01's `generate_starvation` (cell
    899e7d48) fully-formed values as severity -> 1. At severity=0 this is
    exactly generate_normal_grid; at severity=1 it's exactly notebook 01's
    starvation formula.
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    row_grad = (1 - severity) * np.linspace(1.0, 0.65, rows) + severity * np.linspace(0.95, 1.0, rows)
    col_grad = (1 - severity) * np.linspace(1.0, 0.72, cols) + severity * np.linspace(1.0, 0.04, cols)
    gradient = np.outer(row_grad, col_grad)
    base_lo = (1 - severity) * 0.45 + severity * 0.04
    base_scale = (1 - severity) * 0.35 + severity * 0.62
    base = base_lo + base_scale * gradient
    noise = rng.normal(0, 0.02, (rows, cols))
    clip_lo = (1 - severity) * 0.30 + severity * 0.0
    clip_hi = (1 - severity) * 0.90 + severity * 0.75
    return np.clip(base + noise, clip_lo, clip_hi)


def generate_flooding_grid(severity: float, seed: int,
                         rows: int = ROWS, cols: int = COLS) -> np.ndarray:
    """
    Flooding, continuous. Interpolates toward notebook 01's
    `generate_flooding` (cell 899e7d48), including the bottom-right water-
    pooling suppression term, scaled by severity.
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    row_grad = (1 - severity) * np.linspace(1.0, 0.65, rows) + severity * np.linspace(1.0, 0.82, rows)
    col_grad = (1 - severity) * np.linspace(1.0, 0.72, cols) + severity * np.linspace(1.0, 0.87, cols)
    gradient = np.outer(row_grad, col_grad)
    base_lo = (1 - severity) * 0.45 + severity * 0.28
    base_scale = (1 - severity) * 0.35 + severity * 0.20
    base = base_lo + base_scale * gradient
    suppression = np.zeros((rows, cols))
    suppression[2:, 2:] = 0.07 * severity
    noise_std = (1 - severity) * 0.02 + severity * 0.012
    noise = rng.normal(0, noise_std, (rows, cols))
    clip_lo = (1 - severity) * 0.30 + severity * 0.10
    clip_hi = (1 - severity) * 0.90 + severity * 0.58
    return np.clip(base - suppression + noise, clip_lo, clip_hi)


def generate_drying_grid(severity: float, seed: int,
                         rows: int = ROWS, cols: int = COLS) -> np.ndarray:
    """Drying, continuous. Interpolates toward notebook 01's `generate_drying` (cell 899e7d48)."""
    severity = float(np.clip(severity, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    row_grad = (1 - severity) * np.linspace(1.0, 0.65, rows) + severity * np.linspace(1.0, 0.42, rows)
    col_grad = (1 - severity) * np.linspace(1.0, 0.72, cols) + severity * np.linspace(1.0, 0.38, cols)
    gradient = np.outer(row_grad, col_grad)
    base_lo = (1 - severity) * 0.45 + severity * 0.08
    base_scale = (1 - severity) * 0.35 + severity * 0.52
    base = base_lo + base_scale * gradient
    noise = rng.normal(0, 0.02, (rows, cols))
    clip_lo = (1 - severity) * 0.30 + severity * 0.04
    clip_hi = (1 - severity) * 0.90 + severity * 0.68
    return np.clip(base + noise, clip_lo, clip_hi)


_GRID_GENERATORS = {
    "starvation": generate_starvation_grid,
    "flooding": generate_flooding_grid,
    "drying": generate_drying_grid,
}


def generate_current_grid(fault_type: str | None, severity: float, seed: int,
                          healthy_params: dict | None = None,
                          rows: int = ROWS, cols: int = COLS,
                          starvation_arm: str | None = None) -> np.ndarray:
    """
    `healthy_params` selects the severity=0 path:
      dict  -> adopted logistic family (generate_healthy_grid_v3)
      None  -> legacy notebook 01 form (generate_normal_grid), kept reachable

    Fault-state paths (Phase 3b, Sec. 11c): a fault in PHASE3B_FAULTS is
    generated as a MODULATION of this cell's healthy template, so it shares
    the radial basis and converges to healthy as severity -> 0. A fault not
    yet rebased -- currently starvation only -- still takes the superseded
    interpolation from notebook 01's diagonal gradient, as does every fault
    when healthy_params is None.

    That mixture is a KNOWN, temporary inconsistency while Sec. 11e's
    starvation work is outstanding, and it is visible here rather than
    smoothed over: see PHASE3B_FAULTS and Sec. 8.
    """
    if fault_type is None or severity <= 0:
        if healthy_params is None:
            return generate_normal_grid(seed, rows, cols)
        return generate_healthy_grid_v3(healthy_params, seed, rows=rows, cols=cols)
    if healthy_params is not None and fault_type in PHASE3B_FAULTS:
        return generate_fault_grid_v3(fault_type, severity, healthy_params,
                                      seed, rows=rows, cols=cols,
                                      starvation_arm=starvation_arm)
    return _GRID_GENERATORS[fault_type](severity, seed, rows, cols)


def generate_temperature_grid(current_grid: np.ndarray, fault_type: str | None, severity: float, seed: int) -> np.ndarray:
    """
    Continuous version of notebook 01's `generate_temperature` (cell
    a1edd97b): same center-bias / current-correlation / gaussian-smoothing
    mechanism, but the per-state (base_t, center_scale, j_scale) triple is
    interpolated by severity instead of switched by a state string.
    """
    severity = float(np.clip(severity, 0.0, 1.0)) if fault_type else 0.0
    rng = np.random.default_rng(seed + 100)
    t_rows, t_cols = current_grid.shape

    # Notebook 01 hardcodes these as 4-element profiles (edge segments cooler
    # via heat-sinking at the clamping plates). Interpolating them onto the
    # actual grid size reproduces the original arrays EXACTLY at 4x4 and
    # generalizes smoothly above it -- the alternative, leaving them as fixed
    # 4-vectors, simply fails to broadcast at any other resolution.
    # Shared with the Phase 3b drying/flooding modulations, which are grounded
    # on this exact array (Sec. 11c) -- one definition so the two cannot drift.
    center_bias = _thermal_centre_bias(t_rows, t_cols)
    j_norm = current_grid / 0.80

    normal_params = (70.0, 5.0, 8.0)
    fault_params = {
        "starvation": (69.0, 4.0, 11.0),
        "flooding": (68.0, 3.5, 6.0),
        "drying": (73.0, 6.0, 14.0),
    }
    target = fault_params.get(fault_type, normal_params) if fault_type else normal_params

    base_t = (1 - severity) * normal_params[0] + severity * target[0]
    center_scale = (1 - severity) * normal_params[1] + severity * target[1]
    j_scale = (1 - severity) * normal_params[2] + severity * target[2]

    temp = base_t + center_scale * center_bias + j_scale * j_norm
    noise = rng.normal(0, 0.4, (t_rows, t_cols))
    # sigma is a smoothing LENGTH, so it must scale with resolution to keep its
    # physical meaning: notebook 01's sigma=0.8 on a 4x4 grid is 0.2 of the grid
    # width. Holding 0.8 cells fixed at 18x18 would silently leave the
    # temperature field nearly unsmoothed -- a behaviour change disguised as a
    # resolution change. Exactly 0.8 at 4x4, so frozen-resolution output is
    # unaffected.
    sigma = 0.8 * (max(t_rows, t_cols) / 4.0)
    return gaussian_filter(temp + noise, sigma=sigma)


# ---------------------------------------------------------------------------
# CV(J) -- copied verbatim from notebook 02's verified formula
# ---------------------------------------------------------------------------

def compute_cv(current_grid: np.ndarray) -> float:
    """
    CV(J) = std(|J|) / mean(|J|), copied verbatim from notebook 02's verified
    formula (cells 7f90ce08 + a70af676): population std (ddof=0, numpy
    default) of the grid's magnitude, divided by the grid's magnitude mean.
    abs() kept for formula fidelity even though synthetic current values here
    are non-negative by construction (unlike the real dataset's unresolved
    sign convention that motivated it there).
    """
    j_abs = np.abs(current_grid).reshape(-1)
    return float(j_abs.std() / j_abs.mean())


# ---------------------------------------------------------------------------
# Trigger detection
# ---------------------------------------------------------------------------

def find_dwell_events(
    t_seconds: np.ndarray, load_ratio: np.ndarray, threshold: float, mode: str, min_dwell_s: float
) -> list[dict]:
    """
    Contiguous runs (duration >= min_dwell_s) where load_ratio stays
    continuously below (mode="low") or above (mode="high") threshold.
    Mirrors extract_load_trace.find_dwell_fraction's contiguous-run
    detection loop exactly, but returns each qualifying run's own boundaries
    (start_s/end_s/duration_s) instead of an aggregate fraction, since
    building onset trajectories needs individual event timing. Same seam
    caveat as find_ramp_events: caller must not pass samples spanning a
    segment boundary.
    """
    if mode == "low":
        mask = load_ratio <= threshold
    elif mode == "high":
        mask = load_ratio >= threshold
    else:
        raise ValueError(f"mode must be 'low' or 'high', got {mode!r}")

    events = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            end_idx = min(j, n - 1)
            duration = t_seconds[end_idx] - t_seconds[i]
            if duration >= min_dwell_s:
                events.append(
                    {
                        "start_s": float(t_seconds[i]),
                        "end_s": float(t_seconds[end_idx]),
                        "duration_s": float(duration),
                    }
                )
            i = j
        else:
            i += 1
    return events


# ---------------------------------------------------------------------------
# Per-fault-type severity(t) episode builders
# ---------------------------------------------------------------------------

def _build_starvation_episode(event: dict, rng: np.random.Generator) -> dict:
    """
    Fast onset (~2-5s ramp, per Task 2 brief), triggered by a single ramp
    event. Ramps linearly to a target severity, holds briefly, then decays --
    a short symmetric bump, since O2 starvation during a load ramp is a
    transient that resolves once air supply catches up.

    Target severity is sampled independently of the triggering ramp's
    magnitude (NOT derived from peak_rate). Earlier this was a linear map of
    peak_rate onto [0.5, 1.0], on the reasoning that harder ramps should
    produce more severe starvation -- but this trace's ramp-rate distribution
    is bimodal with a gap (routine load steps cluster ~0.0135/s, well below
    the 0.02 trigger; genuine fast transitions cluster tightly ~0.0264-
    0.0275/s, per extract_load_trace.py's report), so 33/34 triggering events
    landed within that narrow top slice and the map saturated severity_target
    at ~[0.85, 1.0] for nearly every event instead of spanning a meaningful
    range. Decoupled sampling is used instead, purely for severity diversity
    in generated training data -- see the CALIBRATION CHOICE comment below.
    """
    onset_start_s = event["start_s"]
    # CALIBRATION CHOICE, not literature-derived: [2, 5]s draw for each of the
    # ramp/plateau/decay stages, matching the brief's "~2-5s" but with no
    # source for the exact distribution shape (uniform) or for splitting the
    # ~2-5s three ways rather than some other split.
    onset_dur = float(rng.uniform(2.0, 5.0))
    plateau_dur = float(rng.uniform(2.0, 5.0))
    decay_dur = float(rng.uniform(2.0, 5.0))
    # CALIBRATION CHOICE: severity_target drawn independently and uniformly
    # from [0.3, 1.0], deliberately wider than flooding's/drying's [0.7-0.8,
    # 1.0] target ranges -- starvation is a fast transient with no real
    # magnitude signal available to scale it by (see docstring above), so a
    # broad, unconditioned range is used to give downstream training data
    # severity diversity rather than clustering near full severity.
    severity_target = float(rng.uniform(0.3, 1.0))

    def severity_fn(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        dt = t - onset_start_s
        sev = np.zeros_like(dt)

        ramp = (dt >= 0) & (dt < onset_dur)
        sev[ramp] = severity_target * dt[ramp] / onset_dur

        plateau = (dt >= onset_dur) & (dt < onset_dur + plateau_dur)
        sev[plateau] = severity_target

        decay = (dt >= onset_dur + plateau_dur) & (dt < onset_dur + plateau_dur + decay_dur)
        frac = (dt[decay] - onset_dur - plateau_dur) / decay_dur
        sev[decay] = severity_target * (1.0 - frac)

        return sev

    episode_end_s = onset_start_s + onset_dur + plateau_dur + decay_dur
    return {
        "fault_type": "starvation",
        "onset_start_s": onset_start_s,
        "episode_end_s": episode_end_s,
        "severity_fn": severity_fn,
        "severity_target": severity_target,
        "trigger_duration_s": event["duration_s"],
        "trigger_metric": event["peak_rate"],  # kept as metadata only, no longer drives severity_target
        "trigger_metric_name": "peak_ramp_rate_per_s",
    }


def _build_flooding_episode(event: dict, rng: np.random.Generator) -> dict:
    """
    Fast onset but OSCILLATORY (per Task 2 brief): an envelope ramps up fast
    (~2-5s) at dwell start, holds through the sustained low-load dwell, then
    decays fast (~2-5s) once the dwell ends (load rises, drainage improves).
    Within the envelope, severity oscillates on a tens-of-seconds cycle:
    accumulate toward a ceiling over the first 70% of each cycle, then
    breakthrough/drain sharply over the remaining 30% back down to a partial
    floor (not fully clearing) -- water doesn't fully drain between
    breakthrough events.
    """
    onset_start_s = event["start_s"]
    dwell_end_s = event["end_s"]
    dwell_dur = dwell_end_s - onset_start_s
    # CALIBRATION CHOICE, not literature-derived: same [2, 5]s envelope
    # draw as starvation's onset/decay, reused here for the fast ramp-up/
    # recovery of the oscillation's amplitude envelope; matches the brief's
    # "fast onset" qualifier but the exact distribution is tuned.
    fast_dur = float(rng.uniform(2.0, 5.0))
    decay_dur = float(rng.uniform(2.0, 5.0))
    # CALIBRATION CHOICE: oscillation period drawn from [20, 45]s -- matches
    # the brief's "tens of seconds" qualifier but the specific range and
    # uniform-draw shape are tuned, not derived from any accumulate/
    # breakthrough/drain timing reported in the literature.
    cycle_period = float(rng.uniform(20.0, 45.0))
    # CALIBRATION CHOICE: partial-drain floor -- each breakthrough cycle only
    # drains back down to 35% of severity_target, not to 0, on the modeling
    # assumption that water doesn't fully clear between breakthrough events.
    # The 0.35 value itself is an illustrative constant, not measured.
    floor = 0.35
    # CALIBRATION CHOICE: flooding's target severity once fully established,
    # drawn uniformly from [0.7, 1.0] -- no literature source for this
    # specific range, chosen only to make flooding "quite severe" per the
    # brief's general framing.
    severity_target = float(rng.uniform(0.7, 1.0))

    def severity_fn(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        dt = t - onset_start_s
        dt_clip = np.clip(dt, 0, None)

        env = np.ones_like(dt)
        ramp = dt < fast_dur
        env[ramp] = np.clip(dt[ramp] / fast_dur, 0.0, 1.0)
        after = dt >= dwell_dur
        frac = np.clip((dt[after] - dwell_dur) / decay_dur, 0.0, 1.0)
        env[after] = np.clip(1.0 - frac, 0.0, 1.0)
        env = np.clip(env, 0.0, 1.0)

        # CALIBRATION CHOICE: 70/30 accumulate/drain split of each cycle, and
        # the -3.0 decay-rate constant shaping the accumulate curve -- both
        # illustrative, not measured or literature-derived.
        phase = np.mod(dt_clip, cycle_period) / cycle_period
        osc = np.empty_like(phase)
        acc = phase < 0.7
        osc[acc] = floor + (1.0 - floor) * (1.0 - np.exp(-3.0 * phase[acc] / 0.7))
        drain = ~acc
        frac2 = (phase[drain] - 0.7) / 0.3
        osc[drain] = 1.0 - (1.0 - floor) * frac2

        sev = severity_target * env * osc
        sev[dt < 0] = 0.0
        return sev

    episode_end_s = dwell_end_s + decay_dur
    return {
        "fault_type": "flooding",
        "onset_start_s": onset_start_s,
        "episode_end_s": episode_end_s,
        "severity_fn": severity_fn,
        "severity_target": severity_target,
        "trigger_duration_s": event["duration_s"],
        "trigger_metric": dwell_dur,
        "trigger_metric_name": "dwell_duration_s",
    }


def _build_drying_episode(event: dict, rng: np.random.Generator) -> dict:
    """
    Slow, roughly monotonic onset (tens of seconds to minutes, per Task 2
    brief), triggered by a sustained high-load dwell: severity rises along a
    saturating exponential with time constant tau in [30, 90]s toward a
    target, for as long as the dwell lasts -- if the dwell ends before
    saturating, severity simply doesn't reach the target within this episode
    (explicitly allowed by the brief). After the dwell ends, severity decays
    back toward 0 (membrane rehumidifies once load drops) on a comparable
    time constant.
    """
    onset_start_s = event["start_s"]
    dwell_end_s = event["end_s"]
    dwell_dur = dwell_end_s - onset_start_s
    # CALIBRATION CHOICE: [30, 90]s time constant -- matches the brief's
    # "tens of seconds to minutes" qualifier, but the specific range, the
    # uniform-draw shape, and the choice of a saturating exponential (vs.
    # some other slow-monotonic curve) are tuned, not literature-derived.
    tau = float(rng.uniform(30.0, 90.0))
    # CALIBRATION CHOICE: recovery time constant, drawn independently from
    # [20, 60]s -- "comparable, maybe a bit faster than onset" is a modeling
    # assumption (membrane rehydration once load drops), not a measured rate.
    recovery_tau = float(rng.uniform(20.0, 60.0))
    # CALIBRATION CHOICE: target severity drawn from [0.8, 1.0] -- no
    # literature source, chosen only to make drying "severe once fully
    # developed" per the brief's general framing.
    severity_target = float(rng.uniform(0.8, 1.0))
    sev_at_dwell_end = severity_target * (1.0 - np.exp(-dwell_dur / tau))

    def severity_fn(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        dt = t - onset_start_s
        sev = np.zeros_like(dt)

        rising = (dt >= 0) & (dt < dwell_dur)
        sev[rising] = severity_target * (1.0 - np.exp(-dt[rising] / tau))

        after = dt >= dwell_dur
        sev[after] = sev_at_dwell_end * np.exp(-(dt[after] - dwell_dur) / recovery_tau)

        return sev

    episode_end_s = dwell_end_s + 3.0 * recovery_tau  # ~95% decayed by 3 time constants
    return {
        "fault_type": "drying",
        "onset_start_s": onset_start_s,
        "episode_end_s": episode_end_s,
        "severity_fn": severity_fn,
        "severity_target": severity_target,
        "trigger_duration_s": event["duration_s"],
        "trigger_metric": tau,
        "trigger_metric_name": "onset_time_constant_s",
    }


_EPISODE_BUILDERS = {
    "starvation": _build_starvation_episode,
    "flooding": _build_flooding_episode,
    "drying": _build_drying_episode,
}


def find_near_seam_buffer(
    t_seconds: np.ndarray, segment_id: np.ndarray, window_s: float = NEAR_SEAM_WINDOW_S
) -> np.ndarray:
    """
    Boolean mask, True for timesteps within `window_s` seconds of either side
    of an INTERNAL segment-boundary seam (i.e. the two concatenation points
    between the 3 real driving segments) -- not the true start of segment 0
    or the true end of the last segment, since those are real boundaries of
    the driving data, not artificial seams, and carry no reset artifact.

    Flags both the tail of the earlier segment and the head of the later
    segment around each seam, since the artifact described in
    generate_labeled_run's docstring (a fault episode hard-truncated at one
    segment's end, immediately followed by a fresh episode re-onsetting at
    the very first sample of the next) can land on either side.
    """
    near = np.zeros(len(t_seconds), dtype=bool)
    segs = np.unique(segment_id)
    for s in segs:
        idx = np.flatnonzero(segment_id == s)
        t_seg = t_seconds[idx]
        if s != segs[-1]:
            near[idx[(t_seg[-1] - t_seg) <= window_s]] = True
        if s != segs[0]:
            near[idx[(t_seg - t_seg[0]) <= window_s]] = True
    return near


def _install_episode(
    t_seg: np.ndarray,
    severity_seg: np.ndarray,
    fault_type_seg: np.ndarray,
    onset_field_seg: np.ndarray,
    episode: dict,
) -> None:
    """
    Writes one episode's severity_fn into the segment-local arrays (views
    into the global per-timestep arrays), only where this episode's severity
    exceeds whatever is currently installed -- the merge rule for the rare
    case of two episodes' spans overlapping (physically uncommon, since the
    three trigger conditions are close to mutually exclusive by
    construction, but handled defensively).
    """
    idx0 = int(np.searchsorted(t_seg, episode["onset_start_s"], side="left"))
    idx1 = int(np.searchsorted(t_seg, episode["episode_end_s"], side="right"))
    idx1 = min(idx1, len(t_seg))
    if idx0 >= idx1:
        return

    local_t = t_seg[idx0:idx1]
    sev_vals = episode["severity_fn"](local_t)

    for k in range(idx0, idx1):
        v = sev_vals[k - idx0]
        if v > severity_seg[k]:
            severity_seg[k] = v
            fault_type_seg[k] = episode["fault_type"]
            onset_field_seg[k] = max(0.0, local_t[k - idx0] - episode["onset_start_s"])


# ---------------------------------------------------------------------------
# Full labeled run
# ---------------------------------------------------------------------------

def generate_labeled_run(trace_path: str | None = None, base_seed: int = BASE_SEED,
                         healthy_model: str = "logistic",
                         grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE,
                         starvation_arm: str | None = None) -> dict:
    """
    `starvation_arm` is REQUIRED whenever the logistic healthy family is in use,
    because starvation is generated two ways and Sec. 11d/11h fixes that neither
    is the default: pass "directional" or "no_axis" explicitly. Both are meant
    to be run and compared, so defaulting one of them would quietly privilege
    it. It is ignored on the legacy `healthy_model="notebook01"` path, where the
    whole superseded interpolation form applies.

    `grid_shape` sets the generated resolution, defaulting to 18x18 to match
    the real validation data natively. CV(J) from this family is
    resolution-dependent (Sec. 7c), so grid size must be stated wherever its
    CV(J) is reported. 4x4 stays reachable via grid_shape=(4, 4).

    `healthy_model` selects the severity=0 spatial template:
      "logistic"   -- ADOPTED family (Sec. 6c/6g), default
      "notebook01" -- legacy diagonal flow-path gradient, superseded by the
                      radial finding (Sec. 5b) but kept reachable

    Fault-state patterns are unaffected by this switch; they still interpolate
    from notebook 01's gradient pending Phase 3b (Sec. 8).

    Loads the real dynamic-only load trace and produces a full labeled
    synthetic run: for every timestep, a (current_grid, temp_grid,
    active_fault_type_or_none, severity, onset_field_s, cv_j,
    near_seam_buffer) tuple.

    onset_field_s is signed: >= 0 and counting up from an active episode's
    own onset ("seconds since onset") while inside that episode's span;
    negative and counting down to the next known episode's onset within the
    same segment ("seconds until onset") while no fault is active; NaN if no
    episode (past or future) is relevant at that index within its segment.

    Every trigger-detection and episode-installation step is scoped to a
    single segment_id's own index range -- fault state never carries across
    the trace's 2 artificial seams, and ramp rate is never computed across
    one either (find_ramp_events/find_dwell_events only ever see one
    segment's own t_seconds/load_ratio slice). This is correct in the sense
    that no state leaks across a seam, but it produces a physically
    unrealistic artifact right at each seam: a fault episode active at one
    segment's last sample gets hard-truncated there (severity drops straight
    to 0 instead of decaying naturally), and because the 3 real driving
    segments are similarly-shaped cycles that tend to both start and end
    inside a qualifying dwell, a fresh episode often re-onsets immediately at
    the very first sample of the next segment. The returned `near_seam_buffer`
    boolean array flags timesteps close enough to either of the 2 internal
    seams to contain this artifact -- any training set built from this
    generator's output should exclude those timesteps, not because the
    no-carryover logic is wrong, but because the concatenation itself isn't
    physically continuous there.
    """
    if healthy_model == "logistic" and starvation_arm not in STARVATION_ARMS:
        raise ValueError(
            f"generate_labeled_run requires starvation_arm={STARVATION_ARMS} "
            f"when healthy_model='logistic'; got {starvation_arm!r}. "
            f"Sec. 11d/11h: starvation is generated two ways and neither is "
            f"the default -- both are run and compared, so defaulting one "
            f"would quietly privilege it.")
    grid_rows, grid_cols = int(grid_shape[0]), int(grid_shape[1])

    if healthy_model not in ("logistic", "notebook01"):
        raise ValueError(f"healthy_model must be 'logistic' or 'notebook01', got {healthy_model!r}")

    # Drawn from a SEPARATE stream, deliberately: the episode RNG below must
    # produce an identical sequence under either healthy_model, so that
    # swapping the healthy baseline cannot perturb fault-episode parameters.
    # That isolation is what makes the acceptance test meaningful.
    healthy_params = (
        sample_healthy_params(np.random.default_rng(base_seed + 100003))
        if healthy_model == "logistic" else None
    )

    if trace_path is None:
        trace_path = DEFAULT_TRACE_PATH
    d = np.load(trace_path)
    t_seconds = d["t_seconds"]
    load_ratio = d["load_ratio"]
    segment_id = d["segment_id"]

    n = len(t_seconds)
    severity = np.zeros(n, dtype=float)
    fault_type = np.full(n, None, dtype=object)
    onset_field = np.full(n, np.nan, dtype=float)
    all_episodes: list[dict] = []

    rng = np.random.default_rng(base_seed)

    for seg in np.unique(segment_id):
        idx = np.flatnonzero(segment_id == seg)
        idx0, idx1 = int(idx[0]), int(idx[-1]) + 1
        t_seg = t_seconds[idx0:idx1]
        lr_seg = load_ratio[idx0:idx1]

        ramp_events = find_ramp_events(t_seg, lr_seg, RAMP_THRESHOLD)
        low_events = find_dwell_events(t_seg, lr_seg, LOW_LOAD_THRESH, "low", MIN_DWELL_S)
        high_events = find_dwell_events(t_seg, lr_seg, HIGH_LOAD_THRESH, "high", MIN_DWELL_S)

        seg_episodes = (
            [_build_starvation_episode(e, rng) for e in ramp_events]
            + [_build_flooding_episode(e, rng) for e in low_events]
            + [_build_drying_episode(e, rng) for e in high_events]
        )

        sev_seg = severity[idx0:idx1]
        ft_seg = fault_type[idx0:idx1]
        onset_seg = onset_field[idx0:idx1]

        for ep in seg_episodes:
            _install_episode(t_seg, sev_seg, ft_seg, onset_seg, ep)

        # "seconds until onset" for indices with no active episode: countdown
        # to the next episode's onset within THIS segment only, never a
        # neighboring one.
        onset_times_sorted = sorted(ep["onset_start_s"] for ep in seg_episodes)
        for i in range(len(t_seg)):
            if ft_seg[i] is None and onset_times_sorted:
                j = int(np.searchsorted(onset_times_sorted, t_seg[i], side="left"))
                if j < len(onset_times_sorted):
                    onset_seg[i] = -(onset_times_sorted[j] - t_seg[i])

        # Derive each episode's own precursor window + achieved peak severity
        # by finely sampling its own severity_fn -- never a hardcoded table.
        for ep in seg_episodes:
            fine_t = np.linspace(ep["onset_start_s"], ep["episode_end_s"], 200)
            fine_sev = ep["severity_fn"](fine_t)
            ep["achieved_peak_severity"] = float(fine_sev.max())
            above = np.flatnonzero(fine_sev >= CLASSIFICATION_THRESHOLD)
            ep["precursor_window_s"] = float(fine_t[above[0]] - ep["onset_start_s"]) if above.size else None
            ep["segment_id"] = int(seg)
            ep["episode_duration_s"] = ep["episode_end_s"] - ep["onset_start_s"]
            all_episodes.append(ep)

    current_grids = np.empty((n, grid_rows, grid_cols), dtype=np.float64)
    temp_grids = np.empty((n, grid_rows, grid_cols), dtype=np.float64)
    cv_j = np.empty(n, dtype=float)
    for i in range(n):
        ft, sev = fault_type[i], severity[i]
        cg = generate_current_grid(ft, sev, seed=base_seed + i, healthy_params=healthy_params,
                                   rows=grid_rows, cols=grid_cols,
                                   starvation_arm=starvation_arm)
        tg = generate_temperature_grid(cg, ft, sev, seed=base_seed + i)
        current_grids[i] = cg
        temp_grids[i] = tg
        cv_j[i] = compute_cv(cg)

    near_seam_buffer = find_near_seam_buffer(t_seconds, segment_id, NEAR_SEAM_WINDOW_S)

    return {
        "t_seconds": t_seconds,
        "load_ratio": load_ratio,
        "segment_id": segment_id,
        "current_grid": current_grids,
        "temp_grid": temp_grids,
        "active_fault_type": fault_type,
        "severity": severity,
        "onset_field_s": onset_field,
        "cv_j": cv_j,
        "near_seam_buffer": near_seam_buffer,
        "episodes": all_episodes,
        "grid_shape": (grid_rows, grid_cols),
        "healthy_model": healthy_model,
        "healthy_params": healthy_params,
    }


if __name__ == "__main__":
    # The demo has to name an arm, like every caller: Sec. 11h fixes that
    # neither is the default. "directional" here is a DEMO choice only and
    # carries no precedence -- the no_axis arm is equally part of the design.
    run = generate_labeled_run(starvation_arm="directional")
    t_seconds = run["t_seconds"]
    load_ratio = run["load_ratio"]
    segment_id = run["segment_id"]
    severity = run["severity"]
    cv_j = run["cv_j"]
    near_seam_buffer = run["near_seam_buffer"]
    episodes = run["episodes"]

    os.makedirs(FIGURES_DIR, exist_ok=True)
    t_min = t_seconds / 60.0

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(
        "Synthetic Fault-Onset Generator -- Real Dynamic-Only Load Trace (Normal_Flow FC-DLC)",
        fontsize=12, fontweight="bold",
    )

    ax1.plot(t_min, load_ratio, color="#333333", linewidth=0.8)
    ax1.set_ylabel("load_ratio")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_min, cv_j, color="#762a83", linewidth=0.8)
    ax2.set_ylabel("CV(J)")
    ax2.grid(True, alpha=0.3)

    ax3.plot(t_min, severity, color="#333333", linewidth=0.6)
    ax3.set_ylabel("severity")
    ax3.set_xlabel("Elapsed time (min)")
    ax3.grid(True, alpha=0.3)

    for s in np.unique(segment_id):
        mask = segment_id == s
        for ax in (ax1, ax2, ax3):
            ax.axvspan(t_min[mask][0], t_min[mask][-1], alpha=0.04, color="black" if s % 2 else "gray", zorder=0)

    # Hatch the near-seam buffer windows -- visual flag for the hard
    # reset-then-reonset artifact described in generate_labeled_run's
    # docstring; these windows should be excluded from any training set.
    nsb = near_seam_buffer.astype(int)
    starts = np.flatnonzero(np.diff(np.concatenate(([0], nsb))) == 1)
    ends = np.flatnonzero(np.diff(np.concatenate((nsb, [0]))) == -1)
    for s0, s1 in zip(starts, ends):
        for ax in (ax1, ax2, ax3):
            ax.axvspan(t_min[s0], t_min[s1], color="black", alpha=0.08, hatch="//", zorder=0.5)

    for ep in episodes:
        color = FAULT_COLORS[ep["fault_type"]]
        span = (ep["onset_start_s"] / 60.0, ep["episode_end_s"] / 60.0)
        for ax in (ax1, ax2, ax3):
            ax.axvspan(*span, color=color, alpha=0.25, zorder=1)

    legend_handles = [
        plt.Line2D([0], [0], color=c, lw=6, alpha=0.5, label=k) for k, c in FAULT_COLORS.items()
    ]
    ax3.legend(handles=legend_handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "fault_generator_run_normal_flow.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {save_path}")

    print()
    print("=== Onset-event summary (dynamic-only trace, Normal_Flow FC-DLC) ===")
    for label in ("starvation", "flooding", "drying"):
        evs = [e for e in episodes if e["fault_type"] == label]
        print(f"\n{label.upper()} -- {len(evs)} onset events")
        if not evs:
            continue
        durs = [e["episode_duration_s"] for e in evs]
        peaks = [e["achieved_peak_severity"] for e in evs]
        pws = [e["precursor_window_s"] for e in evs if e["precursor_window_s"] is not None]
        n_no_cross = sum(1 for e in evs if e["precursor_window_s"] is None)
        print(f"  episode duration:  min={min(durs):.1f}s  max={max(durs):.1f}s  mean={np.mean(durs):.1f}s")
        print(f"  achieved severity: min={min(peaks):.3f}  max={max(peaks):.3f}  mean={np.mean(peaks):.3f}")
        if pws:
            print(
                f"  precursor window:  min={min(pws):.1f}s  max={max(pws):.1f}s  mean={np.mean(pws):.1f}s  "
                f"(n={len(pws)}/{len(evs)} crossed the {CLASSIFICATION_THRESHOLD} classification threshold)"
            )
        if n_no_cross:
            print(f"  {n_no_cross}/{len(evs)} event(s) never crossed the {CLASSIFICATION_THRESHOLD} threshold within their episode")

    starv_targets = np.array([e["severity_target"] for e in episodes if e["fault_type"] == "starvation"])
    print()
    print("=== STARVATION severity_target distribution (decoupled from ramp magnitude) ===")
    print(f"  n={len(starv_targets)}  min={starv_targets.min():.3f}  max={starv_targets.max():.3f}  "
          f"mean={starv_targets.mean():.3f}  median={np.median(starv_targets):.3f}")
    bin_edges = np.linspace(0.3, 1.0, 8)
    hist, edges = np.histogram(starv_targets, bins=bin_edges)
    for count, lo, hi in zip(hist, edges[:-1], edges[1:]):
        print(f"    [{lo:.2f}, {hi:.2f}): {'#' * count} ({count})")

    n_near_seam = int(near_seam_buffer.sum())
    print()
    print(f"=== near_seam_buffer (window={NEAR_SEAM_WINDOW_S:.0f}s each side of the 2 internal seams) ===")
    print(f"  {n_near_seam}/{len(near_seam_buffer)} timesteps flagged ({100 * n_near_seam / len(near_seam_buffer):.1f}% of the full run)")
