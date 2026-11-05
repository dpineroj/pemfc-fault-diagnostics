"""
Poster-quality figure export. RENDERING ONLY -- no analysis is changed here and
no generator constant is touched. Every number plotted is recomputed from the
same code paths the docs cite.

Output: figures/poster/*.png, PNG at dpi=300, tight bbox, white background,
NO baked-in titles (captions belong on the poster). Font sizes are set
explicitly for legibility at ~10 inches wide on a 48-inch board.

PROVENANCE is printed for every figure, because it determines how each can be
captioned honestly:
    MEASURED      -- real data only
    MODEL         -- synthetic/generator output only
    BOTH          -- real data with model curves overlaid
    ILLUSTRATIVE  -- hand-set values, neither measured nor generated

Run:  python scripts/export_poster_figures.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- path bootstrap: siblings moved into scripts/{common,spatial} by the
# November 2026 restructure. Keeps `python scripts/<dir>/<file>.py` working
# without turning scripts/ into a package, which would change every
# documented invocation.
import os as _os
import sys as _sys
_S = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(_S, "common"), _os.path.join(_S, "spatial")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path bootstrap ---

import fault_generator as F  # noqa: E402
from analyze_real_structure import static_template  # noqa: E402
from cdm_parser import load_run  # noqa: E402

OUT = os.path.join(F.PROJECT_ROOT, "figures", "poster")
DPI = 300

plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 16,
    "axes.labelsize": 19,
    "axes.titlesize": 19,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "axes.linewidth": 1.4,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "lines.linewidth": 2.6,
})

MANIFEST = []


def save(fig, name, provenance, note=""):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    w, h = fig.get_size_inches()
    px = (int(round(w * DPI)), int(round(h * DPI)))
    MANIFEST.append((name, px, provenance, note))
    print(f"  wrote {name}  {px[0]}x{px[1]} px  [{provenance}]")
    return path


# ===================================================================== fig 1
def fig_real_templates():
    print("[1] real persistent healthy templates (MEASURED)")
    tpl = {}
    for cfg in ("Normal_Flow", "Inverse_Hydrogen_Flow"):
        run = load_run(cfg)
        m, s_std, t_std = static_template(run["current_grid"])
        tpl[cfg] = m
        print(f"    {cfg:<24} static std {s_std:.4f}  temporal std {t_std:.4f}")
    # SHARED colour limits, so the cross-session reproducibility claim is
    # visually comparable rather than hidden by independent autoscaling
    vmin = min(float(m.min()) for m in tpl.values())
    vmax = max(float(m.max()) for m in tpl.values())
    print(f"    SHARED COLOUR LIMITS  vmin={vmin:.4f}  vmax={vmax:.4f}")
    r = np.corrcoef(tpl["Normal_Flow"].ravel(),
                    tpl["Inverse_Hydrogen_Flow"].ravel())[0, 1]
    print(f"    cross-session template correlation r = {r:+.4f}")

    for cfg, fname in (("Normal_Flow", "real_template_normal_flow.png"),
                       ("Inverse_Hydrogen_Flow", "real_template_inverse_h2.png")):
        fig, ax = plt.subplots(figsize=(7.2, 6.0))
        im = ax.imshow(tpl[cfg], cmap="viridis", vmin=vmin, vmax=vmax,
                       origin="upper", interpolation="nearest")
        ax.set_xlabel("sensor column")
        ax.set_ylabel("sensor row")
        ax.set_xticks([0, 4, 9, 14, 17])
        ax.set_yticks([0, 4, 9, 14, 17])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("normalised current density  $J/\\langle J\\rangle$")
        save(fig, fname, "MEASURED",
             f"{cfg}, 18x18, per-timestep spatial-mean normalised then "
             f"time-averaged; colour limits [{vmin:.4f}, {vmax:.4f}]")
    return vmin, vmax, r


# ===================================================================== fig 2
def _radial_shells(m):
    rows, cols = m.shape
    ri, ci = np.indices((rows, cols))
    r = np.sqrt((ri - (rows - 1) / 2.0) ** 2 + (ci - (cols - 1) / 2.0) ** 2)
    rho = r / r.max()
    key = np.round(rho, 6)
    u = np.unique(key)
    return u, np.array([m[key == v].mean() for v in u]), key


def _r2(obs, pred):
    obs = np.asarray(obs, float)
    return float(1.0 - ((obs - pred) ** 2).sum()
                 / ((obs - obs.mean()) ** 2).sum())


def fig_radial_profile():
    print("[2] radial profile vs adopted logistic and superseded p=2 (BOTH)")
    run = load_run("Normal_Flow")
    m, _, _ = static_template(run["current_grid"])
    rows, cols = m.shape
    u_rho, shell_mean, key = _radial_shells(m)

    # purely-radial CEILING: every cell replaced by its own shell mean
    fitted_ceiling = np.empty(m.size)
    flat = m.ravel()
    kf = key.ravel()
    for v in u_rho:
        sel = kf == v
        fitted_ceiling[sel] = flat[sel].mean()
    r2_ceiling = _r2(flat, fitted_ceiling)
    print(f"    purely-radial ceiling R2 = {r2_ceiling:.4f}")

    # capability-check fit of the ADOPTED FORM. This is Sec. 6c's capability
    # check -- what the form can achieve -- NOT the adopted parameters. The
    # adopted family is sampled from bounds and is NEVER fit to real data,
    # because the real healthy data is Phase 5's validation set.
    from scipy.optimize import minimize

    def logistic_pred(theta):
        c, r0, w = theta
        return F.logistic_healthy_shape(c, r0, max(w, 1e-3), rows, cols).ravel()

    best, best_r2 = None, -np.inf
    for c0 in (0.1, 0.3, 0.5):
        for r00 in (0.3, 0.5, 0.65):
            for w0 in (0.08, 0.2, 0.3):
                res = minimize(lambda th: -_r2(flat, logistic_pred(th)),
                               [c0, r00, w0], method="Nelder-Mead",
                               options={"maxiter": 2000, "xatol": 1e-4})
                if -res.fun > best_r2:
                    best_r2, best = -res.fun, res.x
    c, r0, w = best
    print(f"    logistic capability fit: c={c:.4f} r0={r0:.4f} w={w:.4f}"
          f"  R2={best_r2:.4f}")
    log_grid = F.logistic_healthy_shape(c, r0, w, rows, cols)
    _, log_shell, _ = _radial_shells(log_grid)

    # superseded p=2 power form (v2, edge_ratio 2.0)
    pow_grid = F._radial_healthy_shape(F.RADIAL_EDGE_RATIO, rows, cols)
    r2_pow = _r2(flat, pow_grid.ravel())
    _, pow_shell, _ = _radial_shells(pow_grid)
    print(f"    superseded p=2 power form R2 = {r2_pow:.4f}")

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    ax.plot(u_rho, shell_mean, "o", color="black", ms=8, zorder=5,
            label="measured shell means (real cell)")
    ax.plot(u_rho, log_shell, "-", color="#1b7837",
            label=f"adopted logistic form, capability fit ($R^2$={best_r2:.4f})")
    ax.plot(u_rho, pow_shell, "--", color="#d95f02",
            label=f"superseded $p=2$ power form ($R^2$={r2_pow:.4f})")
    ax.axhline(shell_mean.mean(), color="0.75", lw=1.2, ls=":", zorder=1)
    ax.annotate(f"purely-radial ceiling  $R^2$ = {r2_ceiling:.4f}\n"
                f"({100*(1-r2_ceiling):.2f}% of variance is irreducibly non-radial)",
                xy=(0.03, 0.97), xycoords="axes fraction", va="top", ha="left",
                fontsize=14, bbox=dict(boxstyle="round,pad=0.45",
                                       fc="#f0f0f0", ec="0.5"))
    ax.set_xlabel(r"normalised radius  $\rho = r/r_{\max}$")
    ax.set_ylabel(r"normalised current density  $J/\langle J\rangle$")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.25)
    save(fig, "radial_profile_with_ceiling.png", "BOTH",
         f"measured shell means (real) + two model forms overlaid; the "
         f"logistic curve is Sec. 6c's CAPABILITY fit, not the adopted "
         f"parameters, which are never fit to real data")
    return r2_ceiling, best_r2, r2_pow


# ===================================================================== fig 3
def fig_load_invariance():
    print("[3] sigma(J) vs CV(J) load invariance (MEASURED)")
    run = load_run("Normal_Flow")
    j = np.abs(run["current_grid"])
    n = j.shape[0]
    flat = j.reshape(n, -1)
    sigma = flat.std(axis=1)
    mean = flat.mean(axis=1)
    cv = sigma / mean
    load = run["bulk"]["INTENSIDAD"].to_numpy().astype(float)
    load = load[:n] if len(load) >= n else np.pad(load, (0, n - len(load)),
                                                  mode="edge")
    # INTENSIDAD carries 18 NaN rows (unmatched CDM<->bulk merges). Left in,
    # they propagate through np.quantile and silently empty every load bin.
    good = np.isfinite(load) & np.isfinite(sigma) & np.isfinite(cv)
    print(f"    dropped {int((~good).sum())} rows with non-finite "
          f"load/sigma/CV of {n}")
    sigma, mean, cv, load = sigma[good], mean[good], cv[good], load[good]

    # spread across load DECILES -- the load confound, quantified the same way
    # for both metrics so the contrast is like-for-like
    qs = np.quantile(load, np.linspace(0, 1, 11))
    bs, bc, bl = [], [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (load >= lo) & (load <= hi)
        if m.sum() < 20:
            continue
        bs.append(sigma[m].mean()); bc.append(cv[m].mean())
        bl.append(load[m].mean())
    bs, bc, bl = np.array(bs), np.array(bc), np.array(bl)
    # THE DOCUMENTED DEFINITION (docs/real_data_notes.md): "CoV of the metric
    # itself" = std/mean over ALL timesteps. This is what the 33.6% / 1.0%
    # figures and the ~34x claim are, so it is what the figure annotates.
    cov_s = 100 * sigma.std() / sigma.mean()
    cov_c = 100 * cv.std() / cv.mean()
    print(f"    DOCUMENTED definition (CoV of the metric over all timesteps):")
    print(f"      sigma(J) CoV = {cov_s:.1f}%   (docs: 33.6%)")
    print(f"      CV(J)    CoV = {cov_c:.1f}%   (docs: 1.0%)")
    print(f"      ratio        = {cov_s / cov_c:.1f}x  (docs: ~34x)")
    print(f"      sigma range {sigma.min():.4f}-{sigma.max():.4f} "
          f"mean {sigma.mean():.4f}")
    print(f"      CV    range {cv.min():.4f}-{cv.max():.4f} "
          f"mean {cv.mean():.4f}")
    # A SECOND definition, reported for transparency only and NOT annotated:
    # max-min across load deciles. Larger because it is a range, not a std.
    sp_s = 100 * (bs.max() - bs.min()) / bs.mean()
    sp_c = 100 * (bc.max() - bc.min()) / bc.mean()
    print(f"    alternative definition (decile max-min, NOT the documented "
          f"one): {sp_s:.1f}% vs {sp_c:.1f}% = {sp_s / sp_c:.1f}x")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    a1.plot(bl, bs, "o-", color="#b2182b", ms=9)
    a1.set_xlabel("load current  (A)")
    a1.set_ylabel(r"raw  $\sigma(J)$")
    a1.text(0.04, 0.93, f"CoV of metric = {cov_s:.1f}%",
            transform=a1.transAxes, fontsize=17, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fde0dd", ec="#b2182b"))
    a1.grid(alpha=0.25)
    a2.plot(bl, bc, "o-", color="#2166ac", ms=9)
    a2.set_xlabel("load current  (A)")
    a2.set_ylabel(r"$\mathrm{CV}(J) = \sigma(J)/\langle |J|\rangle$")
    a2.text(0.04, 0.93,
            f"CoV of metric = {cov_c:.1f}%"
            f"   ({cov_s / cov_c:.0f}x more load-invariant)",
            transform=a2.transAxes, fontsize=16, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#d1e5f0", ec="#2166ac"))
    a2.grid(alpha=0.25)
    for a in (a1, a2):
        a.margins(y=0.18)
    save(fig, "sigma_vs_cv_load_invariance.png", "MEASURED",
         f"Normal_Flow FC-DLC, n={len(sigma):,}; lines are load-decile "
         f"means, annotations are the DOCUMENTED CoV-of-metric figures "
         f"{cov_s:.1f}% vs {cov_c:.1f}% ({cov_s / cov_c:.0f}x)")
    return cov_s, cov_c


# ===================================================================== fig 4
def fig_bulk_vs_spatial():
    print("[4] bulk voltage vs spatial NUI (ILLUSTRATIVE -- hand-set values)")
    print("    reimplemented from notebook 01's simulate_time_series literals;")
    print("    notebook 01 is NOT modified or imported.")
    rng = np.random.default_rng(42)
    t = np.linspace(0, 120, 300)
    phases = [(0, 30, "NORMAL", 0.73, 0.10),
              (30, 60, "FLOODING", 0.62, 0.04),
              (60, 90, "DRYING", 0.59, 0.15),
              (90, 120, "STARVATION", 0.55, 0.22)]
    v = np.zeros(300); nui = np.zeros(300)
    for t0, t1, _lab, vt, nt in phases:
        m = (t >= t0) & (t < t1)
        v[m] = vt + rng.normal(0, 0.007, m.sum())
        nui[m] = nt + rng.normal(0, 0.005, m.sum())

    cols = ["#d4edda", "#cce5ff", "#fff3cd", "#f8d7da"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11.5, 7.0), sharex=True)
    for (t0, t1, lab, _v, _n), cc in zip(phases, cols):
        for a in (a1, a2):
            a.axvspan(t0, t1, color=cc, zorder=0)
        a1.text((t0 + t1) / 2, 0.755, lab, ha="center", fontsize=14,
                fontweight="bold")
    a1.plot(t, v, color="#333333")
    a1.set_ylabel("bulk voltage proxy  (V)")
    a1.set_ylim(0.52, 0.78)
    a1.grid(alpha=0.25)
    a2.plot(t, nui, color="#2166ac")
    a2.set_ylabel("spatial nonuniformity index")
    a2.set_xlabel("time  (s)")
    a2.grid(alpha=0.25)
    save(fig, "bulk_vs_spatial_motivation.png", "ILLUSTRATIVE",
         "hand-set phase targets from notebook 01 (voltage 0.73/0.62/0.59/0.55, "
         "NUI 0.10/0.04/0.15/0.22) plus noise -- NOT measured and NOT generator "
         "output; a schematic of the argument only")


# ===================================================================== fig 5
def fig_generator_run():
    print("[5] generator run, 18x18, adopted family + rebased faults (MODEL)")
    run = F.generate_labeled_run(starvation_arm="directional")
    t = run["t_seconds"] / 60.0
    seg = run["segment_id"]
    nsb = run["near_seam_buffer"]
    print(f"    grid_shape {run['grid_shape']}  healthy_model "
          f"{run['healthy_model']}  n={len(t)}")
    print(f"    drawn cell: " + ", ".join(
        f"{k}={v:.4f}" for k, v in run["healthy_params"].items()))
    print(f"    near_seam flagged {int(nsb.sum())} of {len(nsb)}")

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(13.5, 9.5), sharex=True)
    a1.plot(t, run["load_ratio"], color="#333333", lw=1.8)
    a1.set_ylabel("load ratio")
    a2.set_ylabel("fault severity")
    for ftype, col in F.FAULT_COLORS.items():
        sev = np.where(np.asarray(run["active_fault_type"],
                                  dtype=object) == ftype, run["severity"], np.nan)
        a2.plot(t, sev, color=col, lw=2.2, label=ftype)
    a2.axhline(F.CLASSIFICATION_THRESHOLD, color="0.5", ls=":", lw=1.6)
    a2.legend(loc="upper right", ncol=3, frameon=True)
    a2.set_ylim(-0.04, 1.05)
    a3.plot(t, run["cv_j"], color="#1b7837", lw=1.6)
    a3.set_ylabel(r"$\mathrm{CV}(J)$")
    a3.set_xlabel("time  (min)")
    # near-seam hatching, retained
    edges = np.flatnonzero(np.diff(nsb.astype(int)) != 0) + 1
    bounds = np.concatenate([[0], edges, [len(nsb)]])
    for i0, i1 in zip(bounds[:-1], bounds[1:]):
        if nsb[i0]:
            for a in (a1, a2, a3):
                a.axvspan(t[i0], t[min(i1, len(t) - 1)], color="black",
                          alpha=0.10, hatch="//", zorder=0.5, lw=0)
    for s in np.unique(seg)[:-1]:
        x = t[np.flatnonzero(seg == s)[-1]]
        for a in (a1, a2, a3):
            a.axvline(x, color="0.4", ls="--", lw=1.2)
    for a in (a1, a2, a3):
        a.grid(alpha=0.22)
    save(fig, "fault_generator_run.png", "MODEL",
         "18x18 default, adopted logistic healthy family, Phase 3b rebased "
         "fault states, directional starvation arm; hatching = "
         "near_seam_buffer; dashed = segment boundaries")


def main():
    print(f"writing to {OUT}")
    print()
    fig_real_templates()
    print()
    fig_radial_profile()
    print()
    fig_load_invariance()
    print()
    fig_bulk_vs_spatial()
    print()
    fig_generator_run()
    print()
    print("=" * 76)
    print("MANIFEST")
    print("=" * 76)
    print(f"{'file':<38}{'pixels':>13}  provenance")
    for name, px, prov, note in MANIFEST:
        print(f"{name:<38}{px[0]}x{px[1]:>6}  {prov}")
        print(f"{'':<38}{'':>13}  {note}")
    print()
    print("provenance key: MEASURED = real data only | MODEL = generator only")
    print("                BOTH = real + model overlay | ILLUSTRATIVE = hand-set")


if __name__ == "__main__":
    main()
