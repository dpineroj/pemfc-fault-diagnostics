"""
Provenance of every generator parameter.

Three tags, the convention used for RADIAL_EDGE_RATIO in the reference
generator:

  LIT   literature-grounded
  PHYS  physics-derived
  ASSUME engineering assumption, with a sensitivity bracket. Where the
        bracket is a range the value is DRAWN PER CELL from it, so no result
        rests on one precise-looking number.

NOTHING here is fit to, tuned on or selected with the Seville recording's
healthy behaviour (docs/tool_design.md, rule 3). "Seville format" entries
copy layout, units and set-points only.

Literature values marked (*) are quoted from standard references and were
not re-verified against the source in this pass; verify before citing.

tests/test_generator.py fails if a physics parameter in DEFAULT_PHYSICS has
no row here.
"""

LIT, PHYS, ASSUME, FORMAT = "literature", "physics", "assumption", "format"

# key -> (tag, value or per-cell bracket, unit, basis)
PROVENANCE = {
    # ---------------------------------------------------------- spatial
    "edge_exponent": (PHYS, 2, "-",
        "Leading even term of a smooth radially symmetric clamping-pressure "
        "field (reference generator, RADIAL_EDGE_EXPONENT)."),
    "edge_ratio": (ASSUME, (1.5, 2.5), "-",
        "Idealised centre-to-corner current ratio under bolt-frame clamping. "
        "Mechanism is literature-supported, magnitude is not citable "
        "(reference generator, RADIAL_EDGE_RATIO = 2.0, bracket 1.5-2.5)."),
    "edge_load_slope": (ASSUME, (-0.15, 0.15), "per unit load",
        "Whether and how the healthy template changes with load is unknown; "
        "randomised in sign so the detector cannot assume load-invariance."),
    "temporal_factor": (ASSUME, (10.0, 20.0), "-",
        "Per-segment temporal noise = template spatial std / factor "
        "(reference generator: 15, instructed range 10-20)."),
    "severity_convexity": (ASSUME, 2.0, "-",
        "Fault amplitude ~ severity**p (reference generator, Sec. 11g; p is a "
        "free parameter there). Applied IDENTICALLY to the spatial pattern "
        "and to the voltage loss so neither channel gets an earlier "
        "precursor by construction."),
    "drying_centre_loss": (ASSUME, 0.45, "fraction at severity 1",
        "Reference generator DRYING_CENTRE_LOSS; sign is thermally grounded "
        "(driest where hottest), amplitude is not."),
    "flooding_edge_loss": (ASSUME, 0.40, "fraction at severity 1",
        "Reference generator FLOODING_EDGE_LOSS; sign grounded (water "
        "condenses where coolest), amplitude is not."),
    "starvation_loss": (ASSUME, 0.45, "fraction at severity 1",
        "Reference generator STARVATION_AXIS_LOSS / COUPLED_LOSS."),
    "fault_amplitude_scale": (ASSUME, (0.6, 1.2), "-",
        "Per-cell multiplier on the three amplitudes above."),
    "starvation_axis_mix": (ASSUME, (0.0, 1.0), "-",
        "Starvation pattern = mix of a gas-axis gradient and a "
        "current-coupled pattern; the project ran these as two arms and "
        "could not choose between them (Sec. 11), so the mix is drawn."),
    "flow_axes": (ASSUME, "per flow_config", "rad",
        "Air axis along grid rows, hydrogen axis along columns (cross-flow); "
        "an 'Inverse' configuration reverses that gas's direction. The real "
        "data has one session per configuration and cannot ground this."),
    "h2_starvation_fraction": (ASSUME, 0.3, "-",
        "Share of starvation episodes that are hydrogen rather than oxygen "
        "starvation. Bracket 0.1-0.5. Air-supply lag on load ramps is the "
        "commonly described dynamic cause."),
    # ---------------------------------------------------------- electrical
    "rated_current_density": (ASSUME, (0.5, 1.0), "A/cm2",
        "Current density at 100% load. Typical PEMFC operating range; upper "
        "end limited so the mass-transport term stays physical."),
    "idle_fraction": (ASSUME, (0.03, 0.10), "of rated",
        "Current at 0% load ratio. Keeps current density nonzero at idle."),
    "pol_E_oc": (LIT, 1.031, "V",
        "(*) Empirical polarisation curve V = E_oc - r i - A ln(i) - m "
        "exp(n i), Larminie & Dicks, Fuel Cell Systems Explained (2nd ed.), "
        "values for a Ballard Mark V cell at 70 C, after Kim et al. (1995). "
        "Per-cell multiplier 0.97-1.03."),
    "pol_A": (LIT, 0.03, "V", "(*) as pol_E_oc. Per-cell multiplier 0.8-1.2."),
    "pol_r": (LIT, 2.45e-4, "kOhm cm2",
              "(*) as pol_E_oc. Per-cell multiplier 0.7-1.5."),
    "pol_m": (LIT, 2.11e-5, "V", "(*) as pol_E_oc. Per-cell multiplier 0.5-2."),
    "pol_n": (LIT, 8e-3, "cm2/mA", "(*) as pol_E_oc. Per-cell multiplier 0.9-1.1."),
    "v_transient_gain": (ASSUME, (0.02, 0.10), "V per A/cm2 step",
        "Voltage under/overshoot on a load step (double-layer and hydration "
        "lag). Direction is standard; magnitude is not grounded."),
    "v_transient_tau": (ASSUME, (2.0, 15.0), "s", "Recovery time of the above."),
    "drying_resistance_gain": (ASSUME, (0.5, 2.0), "-",
        "Ohmic resistance multiplier at severity 1 is (1 + gain). Drying "
        "raises membrane resistance (physics); the factor is assumed."),
    "flooding_voltage_loss": (ASSUME, (0.05, 0.20), "V at rated load, severity 1",
        "Mass-transport loss from liquid water."),
    "flooding_overshoot": (ASSUME, (0.005, 0.03), "V",
        "Voltage overshoot when a flooded channel drains. The oscillation "
        "and overshoot are described qualitatively in the literature the "
        "repo cites for flooding onset; the amplitude is assumed."),
    "o2_starvation_voltage_loss": (ASSUME, (0.4, 0.9), "fraction of healthy V",
        "Oxygen starvation collapses cell voltage toward zero."),
    "h2_starvation_reversal": (ASSUME, (0.1, 0.8), "V below zero at severity 1",
        "Hydrogen starvation drives cell reversal (negative voltage)."),
    "voltage_noise": (ASSUME, (0.001, 0.004), "V", "Sensor noise, 1 sigma."),
    # ---------------------------------------------------------- thermal
    "thermoneutral_voltage": (PHYS, 1.254, "V",
        "Heat flux q = J (E_tn - V), lower-heating-value thermoneutral "
        "voltage."),
    "temperature_setpoint": (FORMAT, 65.0, "degC",
        "Seville operating set-point (a protocol condition, not a healthy "
        "statistic)."),
    "thermal_centre_gain": (ASSUME, (2.0, 8.0), "degC per W/cm2",
        "Centre-over-edge temperature rise per unit mean heat flux."),
    "thermal_local_gain": (ASSUME, (1.0, 4.0), "degC per W/cm2",
        "Local temperature response to local heat-flux deviation."),
    "temperature_noise": (ASSUME, 0.2, "degC", "Sensor noise, 1 sigma."),
    # ---------------------------------------------------------- dynamics
    "episode_builders": (LIT, "reference generator", "-",
        "Onset shapes and time constants: starvation ramp 2-5 s; flooding "
        "fast envelope with 20-45 s accumulate/drain cycles; drying "
        "saturating rise, tau 30-90 s, recovery 20-60 s. From the literature "
        "review recorded in the README (Phase 2) and Sec. 8 of the generator "
        "notes; called unmodified."),
    "classification_threshold": (ASSUME, 0.5, "severity",
        "Severity at which a fault counts as confirmed (reference generator)."),
    "ramp_trigger": (ASSUME, 0.02, "load ratio per s",
        "Ramp rate that can trigger starvation (reference generator)."),
    "low_load_trigger": (ASSUME, 0.10, "load ratio",
        "Sustained low load that can trigger flooding (reference generator)."),
    "high_load_trigger": (ASSUME, 0.40, "load ratio",
        "Sustained high load that can trigger drying (reference generator)."),
    "min_dwell": (ASSUME, 10.0, "s", "Minimum dwell for the two above."),
    "p_trigger": (ASSUME, 0.5, "per trigger event",
        "Probability that a trigger event starts a fault. Bracket 0.2-0.8. "
        "The reference generator used 1.0, which made load regime a proxy "
        "for the label (Sec. 16d); anything below 1 breaks that."),
    "concurrent_faults": (ASSUME, "none", "-",
        "Not modelled: nothing grounds how two patterns combine. An episode "
        "that would overlap an active one is skipped and counted."),
    # ---------------------------------------------------------- profiles
    "fcdlc_block_structure": (LIT, "4 x ~195 s urban + ~400 s highway", "-",
        "JRC FC-DLC protocol structure as documented in real_data_notes.md."),
    "fcdlc_random_levels": (ASSUME, "see hfc/profiles.py", "-",
        "Levels, dwell lengths and ramp rates of randomised cycles. Ramp "
        "rates are drawn around the two clusters seen in the real load "
        "trace (routine ~0.0135/s, fast ~0.027/s) -- a property of the "
        "protocol's load trace, which rule 3 allows copying."),
    "static_induced_faults": (LIT, "Kim et al., Energy 2023 (protocol)", "-",
        "Static-load holds with externally induced faults, for comparison "
        "with that baseline. Induction times are assumed."),
    "conditioning_holds": (FORMAT, "optional", "-",
        "Real recordings contain them (78% of the Seville FC-DLC files)."),
    # ---------------------------------------------------------- format
    "grid_layout": (FORMAT, "18x18 current, 9x9 temperature (default)", "-",
                    "Seville S++ CDM layout; configurable."),
    "active_area": (FORMAT, 50.0, "cm2", "Seville cell."),
    "bulk_channels": (FORMAT, "FECHA, HORA, INTENSIDAD [A], V001 [V]", "-",
        "Seville names and units for the channels the generator models. "
        "The other 34 are NOT emitted."),
}


def table_markdown():
    rows = ["| parameter | tag | value / bracket | unit | basis |",
            "|---|---|---|---|---|"]
    for k, (tag, val, unit, basis) in PROVENANCE.items():
        v = f"{val[0]} – {val[1]} (per cell)" if isinstance(val, tuple) else str(val)
        rows.append(f"| `{k}` | {tag} | {v} | {unit} | {basis} |")
    return "\n".join(rows)
