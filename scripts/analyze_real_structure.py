"""
Characterizes the real healthy cell's spatial structure, from the corrected
loader, for every flow configuration.

Reconstructs an analysis that was originally done ad-hoc in-session and whose
conclusions were written into docs/synthetic_generator_notes.md Sec. 5b
without a re-runnable script behind them. Everything here is re-derived from
`load_real_data.load_run` -- no figure is transcribed from those docs. If a
number here disagrees with Sec. 5b, the script is the authority and the docs
need correcting, not the reverse.

Method (stated because each choice changes what the numbers mean):

  1. J = abs(current_grid). Magnitude only -- the sign convention is a
     resolved, uniform CDM convention (docs/real_data_notes.md); every
     quantity below is a dispersion or a correlation, so sign is irrelevant.
  2. Load-normalize each timestep by its own spatial mean:
        J_norm(t,i,j) = J(t,i,j) / mean_ij( J(t,i,j) )
     This strips the load level and leaves the SHAPE. Necessary because raw
     sigma(J) tracks load 3-4x (the load confound from notebook 02); without
     this step the "temporal" term below would be dominated by the whole grid
     scaling up and down together, which is a load effect, not spatial
     structure.
  3. Static template M(i,j) = mean_t( J_norm(t,i,j) ) -- the persistent shape.
     Static term      = M.std()                    (spread ACROSS cells)
     Temporal term    = RMS over cells of std_t( J_norm - M )   (spread OVER time)
     These are the two components of the per-timestep spatial variance; their
     ratio says whether the cell's nonuniformity is a fixed pattern or
     per-timestep fluctuation.

Run:  python scripts/analyze_real_structure.py
"""

from __future__ import annotations

import numpy as np

from load_real_data import load_run

CONFIGS = (
    "Normal_Flow",
    "Inverse_Flow",
    "Inverse_Air_Flow",
    "Inverse_Hydrogen_Flow",
)


def static_template(current_grid: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Returns (static template M, static std, temporal std) -- see module docstring."""
    j = np.abs(current_grid)
    n_t = j.shape[0]
    per_t_mean = j.reshape(n_t, -1).mean(axis=1)
    j_norm = j / per_t_mean[:, None, None]

    m = j_norm.mean(axis=0)
    static_std = float(m.std())

    residual = j_norm - m[None, :, :]
    temporal_std = float(np.sqrt((residual.std(axis=0) ** 2).mean()))
    return m, static_std, temporal_std


def radial_diagnostics(m: np.ndarray) -> dict:
    """Correlations of the static template against geometry, + quadratic-in-radius fit."""
    rows, cols = m.shape
    row_idx, col_idx = np.indices((rows, cols))
    r = np.sqrt((row_idx - (rows - 1) / 2.0) ** 2 + (col_idx - (cols - 1) / 2.0) ** 2)

    y = m.ravel()
    # Quadratic in radius: M ~ a + b*r + c*r^2. Least squares, R^2 vs. the mean model.
    design = np.stack([np.ones(y.size), r.ravel(), r.ravel() ** 2], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    r2 = float(1.0 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum())

    return {
        "corr_dist_from_center": float(np.corrcoef(r.ravel(), y)[0, 1]),
        "corr_row_index": float(np.corrcoef(row_idx.ravel(), y)[0, 1]),
        "corr_col_index": float(np.corrcoef(col_idx.ravel(), y)[0, 1]),
        "quadratic_r2": r2,
        "quadratic_coef": coef,
        "residual_std": float(resid.std()),
        # Std of the fitted radial component alone = sqrt(R^2) * std(M).
        # This is the like-for-like comparator against a purely-radial
        # synthetic model's CV(J): it is what the real cell's CV(J) would be
        # if its ONLY structure were the quadratic-in-radius term.
        "radial_component_std": float(np.sqrt(r2) * y.std()),
    }


def profile_ratios(m: np.ndarray) -> dict:
    """
    Row/column profiles (collapse the other axis) and an edge/center ratio.

    Edge   = mean of the two outermost entries of the profile.
    Center = mean of the two innermost entries (grids here are even-sized, so
             no single center row exists).
    """
    row_profile = m.mean(axis=1)
    col_profile = m.mean(axis=0)

    def edge_center(profile: np.ndarray) -> float:
        mid = len(profile) // 2
        edge = (profile[0] + profile[-1]) / 2.0
        center = (profile[mid - 1] + profile[mid]) / 2.0
        return float(edge / center)

    return {
        "row_profile": row_profile,
        "col_profile": col_profile,
        "row_edge_center_ratio": edge_center(row_profile),
        "col_edge_center_ratio": edge_center(col_profile),
        # The profile ratios above collapse one axis, which averages strong
        # and weak cells together and UNDERSTATES the map's true spread. The
        # 2D extremes below are the honest measure of how far the template
        # actually ranges -- report both, they answer different questions.
        "template_min": float(m.min()),
        "template_max": float(m.max()),
        "template_max_over_min": float(m.max() / m.min()),
    }


def main() -> None:
    templates: dict[str, np.ndarray] = {}

    for config in CONFIGS:
        run = load_run(config, run_type="FC-DLC")
        m, static_std, temporal_std = static_template(run["current_grid"])
        templates[config] = m

        total_var = static_std ** 2 + temporal_std ** 2
        frac_static = 100.0 * static_std ** 2 / total_var

        rad = radial_diagnostics(m)
        prof = profile_ratios(m)

        # Cross-check: per-timestep CV(J) should equal std(J_norm) per timestep,
        # and its mean should land near the static term when static dominates.
        j = np.abs(run["current_grid"])
        n_t = j.shape[0]
        flat = j.reshape(n_t, -1)
        cv_mean = float((flat.std(axis=1) / flat.mean(axis=1)).mean())

        print(f"\n=== {config} / FC-DLC  (n={n_t} timesteps, grid {m.shape[0]}x{m.shape[1]}) ===")
        print(f"  static std          : {static_std:.4f}")
        print(f"  temporal std        : {temporal_std:.4f}")
        print(f"  static share of var : {frac_static:.2f}%   (static/temporal ratio {static_std / temporal_std:.1f}x)")
        print(f"  CV(J) mean          : {cv_mean:.4f}   (cross-check: static std alone = {static_std:.4f})")
        print(f"  corr vs dist-from-center : {rad['corr_dist_from_center']:+.4f}")
        print(f"  corr vs row index        : {rad['corr_row_index']:+.4f}")
        print(f"  corr vs col index        : {rad['corr_col_index']:+.4f}")
        print(f"  quadratic-in-radius R^2  : {rad['quadratic_r2']:.4f}")
        print(f"    radial component std   : {rad['radial_component_std']:.4f}   (what CV(J) would be from the radial term alone)")
        print(f"    non-radial residual std: {rad['residual_std']:.4f}   (structure a pure radial model cannot produce)")
        print(f"  edge/center ratio  rows={prof['row_edge_center_ratio']:.3f}  cols={prof['col_edge_center_ratio']:.3f}  (axis-collapsed)")
        print(f"  template 2D extremes: min={prof['template_min']:.3f}  max={prof['template_max']:.3f}  max/min={prof['template_max_over_min']:.1f}x")
        print(f"  row profile: {np.array2string(prof['row_profile'], precision=3, max_line_width=100)}")
        print(f"  col profile: {np.array2string(prof['col_profile'], precision=3, max_line_width=100)}")

    print("\n=== cross-session correlation of static templates ===")
    print("(how reproducible the spatial template is across independent sessions;")
    print(" high values mean the structure is a property of the cell/fixture, not of one run)")
    names = list(templates)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            corr = float(np.corrcoef(templates[a].ravel(), templates[b].ravel())[0, 1])
            print(f"  {a:22s} vs {b:22s}: r = {corr:+.4f}")


if __name__ == "__main__":
    main()
