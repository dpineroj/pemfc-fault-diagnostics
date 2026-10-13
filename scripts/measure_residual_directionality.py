"""
Is there flow-path structure in the real healthy template, hiding under the
radial component?

WHY THIS EXISTS (Sec. 11c). The proposed fault form needs a flow-path axis for
starvation and flooding, but the real templates show row/column correlations
~0 in all four configurations (docs/real_data_notes.md). Two readings are
possible, and they have opposite consequences:

  (a) no flow-path structure exists in this cell -> imposing an axis is
      contradicted by the data, not merely unsupported by it;
  (b) flow-path structure exists but is swamped by the dominant radial
      component -> imposing an axis is defensible, and the data can even
      calibrate its direction.

Reading (b) is testable, because the residual is not small: 5.84% of template
variance is irreducibly non-radial (Sec. 6e), and it contains a known
directional feature -- a 2.9x diagonal gradient across the four innermost
cells.

THE DECISIVE TEST. The cell is documented as parallel-serpentine with
CROSS-FLOW distribution between anode and cathode (docs/real_data_notes.md),
i.e. air and hydrogen run along perpendicular axes. Two consequences:

  1. In a cross-flow cell, combined reactant depletion is DIAGONAL -- the sum
     of two perpendicular depletion gradients. So a diagonal residual is
     exactly what a flow-path mechanism would look like here, and "row/col
     correlations are ~0" does NOT by itself rule it out. A diagonal gradient
     has near-zero correlation with BOTH row and column index.
  2. Reversing the air inlet/outlet alone flips one of the two perpendicular
     components, which ROTATES the diagonal by ~90 degrees: a residual
     aligned with (row+col) becomes aligned with (row-col). Reversing
     hydrogen alone does the same to the other component.

So the test is not "is the residual directional" but "does its direction
TRACK the gas routing". The dataset provides exactly the needed contrast:

  air-reversal pairs (air flips, H2 fixed):  Normal vs Inverse_Air
                                             Inverse_Hydrogen vs Inverse
  H2-reversal pairs  (H2 flips, air fixed):  Normal vs Inverse_Hydrogen
                                             Inverse_Air vs Inverse

If the diagonal residual is depletion-driven, the diag_sum / diag_diff
loadings must swap within each reversal pair. If it is fixture-driven
(clamping, sensor mounting), it stays put.

CAVEAT carried into the report: configuration is confounded with session
(n=1 recording per configuration), so a difference between configurations
cannot be attributed to gas routing alone. That confound weakens a POSITIVE
result; it does not weaken a NULL result, which is what makes this test worth
running -- "the direction did not move when the flow reversed" is not
explained away by session identity.

Radial removal uses the purely-radial CEILING (each cell replaced by its
radius-shell mean, ~40 effective parameters, Sec. 6e) rather than the fitted
3-parameter logistic. That removes ALL radially-explainable structure, so any
surviving directionality cannot be an artifact of an imperfect radial fit.

Run:  python scripts/measure_residual_directionality.py
"""

from __future__ import annotations

import numpy as np

from analyze_real_structure import CONFIGS, static_template
from load_real_data import load_run

# which gas is reversed in each configuration, relative to Normal_Flow
ROUTING = {
    "Normal_Flow":           {"air": "fwd", "h2": "fwd"},
    "Inverse_Air_Flow":      {"air": "rev", "h2": "fwd"},
    "Inverse_Hydrogen_Flow": {"air": "fwd", "h2": "rev"},
    "Inverse_Flow":          {"air": "rev", "h2": "rev"},
}

AIR_PAIRS = (("Normal_Flow", "Inverse_Air_Flow"),
             ("Inverse_Hydrogen_Flow", "Inverse_Flow"))
H2_PAIRS = (("Normal_Flow", "Inverse_Hydrogen_Flow"),
            ("Inverse_Air_Flow", "Inverse_Flow"))


def _fields(rows, cols):
    ri, ci = np.indices((rows, cols))
    r = np.sqrt((ri - (rows - 1) / 2.0) ** 2 + (ci - (cols - 1) / 2.0) ** 2)
    return {
        "radial": r,
        "row": ri.astype(float),
        "col": ci.astype(float),
        "diag_sum": (ri + ci).astype(float),
        "diag_diff": (ri - ci).astype(float),
    }


def _corr(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    da, db = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return float((a * b).sum() / (da * db))


def radial_ceiling_residual(m, r):
    """
    m minus the best possible purely-radial model: each cell replaced by the
    mean of its radius shell. Returns (residual, R2 of the radial ceiling).
    """
    rr = np.round(r.ravel(), 6)
    flat = m.ravel()
    fitted = np.empty_like(flat)
    for val in np.unique(rr):
        sel = rr == val
        fitted[sel] = flat[sel].mean()
    resid = flat - fitted
    ss_tot = ((flat - flat.mean()) ** 2).sum()
    r2 = 1.0 - (resid ** 2).sum() / ss_tot
    return resid.reshape(m.shape), float(r2)


def main() -> None:
    f = None
    templates, residuals = {}, {}

    print(__doc__.split("Run:")[0].strip()[:0] or "", end="")
    print("Residual directionality after removing the purely-radial ceiling")
    print()

    for cfg in CONFIGS:
        run = load_run(cfg)
        m, _, _ = static_template(run["current_grid"])
        if f is None:
            f = _fields(*m.shape)
        resid, r2 = radial_ceiling_residual(m, f["radial"])
        templates[cfg] = m
        residuals[cfg] = resid
        share = float((resid ** 2).sum() / ((m - m.mean()) ** 2).sum())
        print(f"{cfg:>22}  grid={m.shape[0]}x{m.shape[1]}  "
              f"radial-ceiling R2={r2:.4f}  non-radial share={100*share:5.2f}%")

    print()
    print("=" * 78)
    print("Residual correlation with each directional field")
    print("=" * 78)
    hdr = f"{'configuration':>22} {'air':>4} {'h2':>4} | " + " ".join(
        f"{k:>10}" for k in ("row", "col", "diag_sum", "diag_diff"))
    print(hdr)
    print("-" * len(hdr))
    for cfg in CONFIGS:
        rsd = residuals[cfg]
        vals = [_corr(rsd, f[k]) for k in ("row", "col", "diag_sum", "diag_diff")]
        print(f"{cfg:>22} {ROUTING[cfg]['air']:>4} {ROUTING[cfg]['h2']:>4} | "
              + " ".join(f"{v:>+10.3f}" for v in vals))

    print()
    print("  for reference, the same correlations on the FULL template")
    print("  (radial component NOT removed):")
    for cfg in CONFIGS:
        vals = [_corr(templates[cfg], f[k])
                for k in ("row", "col", "diag_sum", "diag_diff")]
        print(f"{cfg:>22}      | " + " ".join(f"{v:>+10.3f}" for v in vals))

    print()
    print("=" * 78)
    print("THE TEST: does the residual's direction track the gas routing?")
    print("=" * 78)
    print("If depletion-driven, diag_sum/diag_diff loadings SWAP within a")
    print("reversal pair, and the residual maps should ANTI-correlate or")
    print("decorrelate. If fixture-driven, they stay put and maps correlate.")
    print()
    for name, pairs in (("AIR reversed (H2 fixed)", AIR_PAIRS),
                        ("H2 reversed (air fixed)", H2_PAIRS)):
        print(f"  {name}:")
        for a, b in pairs:
            rm = _corr(residuals[a], residuals[b])
            da = _corr(residuals[a], f["diag_sum"])
            db = _corr(residuals[b], f["diag_sum"])
            ea = _corr(residuals[a], f["diag_diff"])
            eb = _corr(residuals[b], f["diag_diff"])
            print(f"    {a:>22} vs {b:<22} residual-map r={rm:>+6.3f}")
            print(f"{'':>26}   diag_sum  {da:>+6.3f} -> {db:>+6.3f}"
                  f"    diag_diff {ea:>+6.3f} -> {eb:>+6.3f}")
        print()

    print("=" * 78)
    print("The four innermost cells (reproduces Sec. 6e point 3)")
    print("=" * 78)
    for cfg in CONFIGS:
        m = templates[cfg]
        rows, cols = m.shape
        i0, j0 = (rows - 1) // 2, (cols - 1) // 2
        blk = m[i0:i0 + 2, j0:j0 + 2]
        print(f"{cfg:>22}  {blk[0,0]:.4f} {blk[0,1]:.4f} "
              f"{blk[1,0]:.4f} {blk[1,1]:.4f}   "
              f"spread={blk.max()/blk.min():.2f}x")
    print()
    print("  orientation of that 2x2: high corner -> low corner tells which")
    print("  diagonal it runs along; it must ROTATE under air reversal if it")
    print("  is cathode-side depletion.")

    print()
    print("=" * 78)
    print("How much of the residual is SHARED across all four configurations?")
    print("=" * 78)
    stack = np.stack([residuals[c] for c in CONFIGS])
    common = stack.mean(axis=0)
    specific = stack - common[None]
    tot = float((stack ** 2).sum())
    common_energy = float((np.repeat(common[None], len(CONFIGS), 0) ** 2).sum())
    specific_energy = float((specific ** 2).sum())
    print(f"  common (config-invariant) part : "
          f"{100 * common_energy / tot:5.1f}% of residual energy")
    print(f"  config-specific part           : "
          f"{100 * specific_energy / tot:5.1f}%")
    print()
    print("  A depletion-driven residual would be mostly config-SPECIFIC,")
    print("  since reversing a gas feed relocates the depleted region.")
    print()
    for i, cfg in enumerate(CONFIGS):
        print(f"{cfg:>22}  r(residual, common)={_corr(stack[i], common):>+6.3f}"
              f"   specific share={100*float((specific[i]**2).sum()/(stack[i]**2).sum()):5.1f}%")

    print()
    print("=" * 78)
    print("MIRROR TEST -- the sharpest available discriminator")
    print("=" * 78)
    print("Reversing a gas inlet/outlet is physically a REFLECTION of that")
    print("gas's axis. So if the residual is flow-locked, config B's residual")
    print("should match a MIRRORED copy of config A's better than an")
    print("unmirrored one. If it is fixture-locked, identity wins.")
    print("(The sensor-grid orientation of the air and H2 axes is not")
    print("documented, so every candidate transform is tested.)")
    print()
    transforms = {
        "identity": lambda x: x,
        "flip_rows": np.flipud,
        "flip_cols": np.fliplr,
        "flip_both(rot180)": lambda x: np.flipud(np.fliplr(x)),
        "transpose": lambda x: x.T,
    }
    for name, pairs in (("AIR reversed", AIR_PAIRS), ("H2 reversed", H2_PAIRS)):
        print(f"  {name}:")
        for a, b in pairs:
            scores = {k: _corr(fn(residuals[a]), residuals[b])
                      for k, fn in transforms.items()}
            best = max(scores, key=lambda k: scores[k])
            print(f"    {a[:18]:>18} -> {b[:18]:<18}  "
                  + "  ".join(f"{k}={scores[k]:+.3f}" for k in transforms))
            print(f"{'':>40}  best = {best}"
                  + ("   <-- flow-locked" if best != "identity" else
                     "   <-- fixture-locked"))
        print()


if __name__ == "__main__":
    main()
