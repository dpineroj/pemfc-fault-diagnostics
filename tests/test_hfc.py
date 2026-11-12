"""
Tests for the hfc package. Standard library unittest; run from the repo root:

    ./hfc_env/Scripts/python.exe -W ignore -m unittest discover -s tests -v

Covers: data contract, adapters, unit checks, abstain logic, generator
reproducibility, provenance coverage, the Seville round-trip, the leak
checker, messiness, and an end-to-end run on a small preset.
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hfc import detect, features, gen, io, leaks, models, profiles  # noqa: E402
from hfc import provenance, seville_export  # noqa: E402
from hfc._ref import F, REPO  # noqa: E402

TINY = {"name": "tiny", "grid": {"current": [6, 6], "temperature": [3, 3]},
        "cohorts": [{"family": "fcdlc_random", "n_cells": 5, "split": "train"},
                    {"family": "static", "n_cells": 1, "split": "train"},
                    {"family": "fcdlc_random", "n_cells": 2, "split": "test"}]}


def _sha_dir(d):
    h = hashlib.sha256()
    for root, _dirs, files in sorted(os.walk(d)):
        for f in sorted(files):
            h.update(os.path.relpath(os.path.join(root, f), d).encode())
            with open(os.path.join(root, f), "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


class Shared(unittest.TestCase):
    """One tiny dataset + models, built once for the end-to-end tests."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hfc_test_")
        cls.ds = os.path.join(cls.tmp, "tiny")
        cls.cfg = gen.make_config(None, TINY)
        gen.generate(cls.cfg, cls.ds, n_jobs=1, log=lambda *a: None)
        cls.models_dir = os.path.join(cls.tmp, "models")
        cls.bundle = models.train(cls.ds, cls.models_dir, n_jobs=1,
                                  log=lambda *a: None)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def rec(self, run_id, **kw):
        return io.load_canonical_run(os.path.join(self.ds, "runs",
                                                  run_id + ".npz"), **kw)


class TestGenerator(Shared):
    def test_reproducible_byte_identical(self):
        d2 = os.path.join(self.tmp, "tiny_again")
        gen.generate(self.cfg, d2, n_jobs=1, log=lambda *a: None)
        self.assertEqual(_sha_dir(self.ds), _sha_dir(d2))

    def test_seed_changes_output(self):
        d3 = os.path.join(self.tmp, "tiny_seed")
        gen.generate(dict(self.cfg, seed=self.cfg["seed"] + 1), d3, n_jobs=1,
                     log=lambda *a: None)
        self.assertNotEqual(_sha_dir(self.ds), _sha_dir(d3))

    def test_every_file_is_flagged_synthetic(self):
        with open(os.path.join(self.ds, "manifest.json")) as fh:
            man = json.load(fh)
        self.assertTrue(man["SYNTHETIC"])
        self.assertEqual(man["seed"], self.cfg["seed"])
        self.assertIn("config", man)
        for f in os.listdir(os.path.join(self.ds, "runs")):
            meta = json.loads(str(np.load(os.path.join(self.ds, "runs", f))
                                  ["meta_json"]))
            self.assertTrue(meta["SYNTHETIC"])
            for k in ("generator_version", "seed", "config"):
                self.assertIn(k, meta)
        with open(os.path.join(self.ds, "DATA_CARD.md"), encoding="utf-8") as fh:
            card = fh.read()
        self.assertIn("SYNTHETIC DATA", card)
        self.assertIn("must NOT be used to claim", card)

    def test_provenance_covers_every_physics_parameter(self):
        for k in list(gen.DEFAULT_PHYSICS) + list(gen.POL):
            self.assertIn(k, provenance.PROVENANCE, k)
        for k, (tag, _v, _u, basis) in provenance.PROVENANCE.items():
            self.assertIn(tag, ("literature", "physics", "assumption", "format"))
            self.assertTrue(len(basis) > 10, k)

    def test_current_controlled(self):
        """A fault redistributes current; it does not remove it."""
        z = np.load(os.path.join(self.ds, "runs", "c00000_fault.npz"))
        J, I = z["current"].astype(float), z["INTENSIDAD"].astype(float)
        seg = self.cfg["active_area_cm2"] / 36
        ratio = J.sum(axis=(1, 2)) * seg / I
        faulty = z["severity"] >= 0.5
        self.assertTrue(faulty.any())
        self.assertLess(abs(np.median(ratio[faulty]) - 1.0), 0.02)

    def test_companion_is_fault_free_and_same_profile(self):
        f = np.load(os.path.join(self.ds, "runs", "c00000_fault.npz"))
        c = np.load(os.path.join(self.ds, "runs", "c00000_companion.npz"))
        self.assertEqual(float(c["severity"].max()), 0.0)
        np.testing.assert_array_equal(f["load_ratio"], c["load_ratio"])
        self.assertGreater(float(f["severity"].max()), 0.0)

    def test_no_concurrent_faults_and_labels_consistent(self):
        z = np.load(os.path.join(self.ds, "runs", "c00001_fault.npz"))
        eps = json.loads(str(z["episodes_json"]))
        for a, b in zip(eps, eps[1:]):
            self.assertLess(a["offset_s"], b["onset_s"])
        st, sev = z["state"], z["severity"]
        self.assertTrue((sev[st == "healthy"] == 0).all())
        self.assertTrue((sev[st == "developing"] < 0.5).all())
        self.assertTrue(set(np.unique(z["segment_type"])) <= {"dynamic", "hold"})

    def test_healthy_covers_high_load(self):
        """'Working hard' must not imply 'faulty'."""
        hi = tot = 0
        for r in leaks.read_index(self.ds):
            z = np.load(os.path.join(self.ds, "runs", r["run_id"] + ".npz"))
            m = z["load_ratio"] >= 0.4
            tot += int(m.sum())
            hi += int((m & (z["severity"] == 0)).sum())
        self.assertGreater(hi / tot, 0.5)

    def test_any_resolution(self):
        cfg = gen.make_config(None, dict(TINY, grid={"current": [4, 4],
                                                     "temperature": [4, 4]}))
        cells, _ = gen.plan(cfg)
        run, _m, _t = gen.make_run(cfg, cells[0], "fault")
        self.assertEqual(run["current"].shape[1:], (4, 4))
        cfg = gen.make_config("extensive")
        self.assertEqual(cfg["grid"], {"current": [18, 18],
                                       "temperature": [9, 9]})

    def test_reference_generator_untouched(self):
        import subprocess
        out = subprocess.run(["git", "status", "--porcelain", "--",
                              "scripts/spatial/fault_generator.py",
                              "notebooks/spatial/01_spatial_viz_sandbox.ipynb"],
                             cwd=REPO, capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "")
        # frozen 4x4 reference: same call, same numbers
        np.testing.assert_array_equal(F.generate_normal_grid(7),
                                      F.generate_normal_grid(7))


class TestLeaks(Shared):
    def test_shipped_split_is_clean(self):
        ok, rep = leaks.check(self.ds)
        self.assertTrue(ok, rep)
        self.assertEqual(rep["cells_under_multiple_profiles"], [])

    def test_cell_ids_unique_per_profile(self):
        rows = [r for r in leaks.read_index(self.ds) if r["role"] == "fault"]
        self.assertEqual(len({r["cell_id"] for r in rows}), len(rows))
        self.assertEqual(len({r["profile_id"] for r in rows}), len(rows))

    def test_detects_a_leaky_split(self):
        p = os.path.join(self.tmp, "bad_split.json")
        with open(p, "w") as fh:     # companion on the other side of its twin
            json.dump({"train": ["c00000_fault"], "test": ["c00000_companion"]},
                      fh)
        ok, rep = leaks.check(self.ds, p)
        self.assertFalse(ok)
        self.assertTrue(rep["shared"]["test~train:pair_id"])

    def test_recurring_cells_are_reported(self):
        cfg = gen.make_config(None, dict(
            TINY, cells_recur_across_profiles=True,
            cohorts=[{"family": "fcdlc_random", "n_cells": 2, "split": "train"},
                     {"family": "static", "n_cells": 2, "split": "test"}]))
        d = os.path.join(self.tmp, "recur")
        gen.generate(cfg, d, n_jobs=1, log=lambda *a: None)
        ok, rep = leaks.check(d)
        self.assertFalse(ok)                      # the Sec. 17b failure mode
        self.assertTrue(rep["shared"]["test~train:cell_id"])
        self.assertEqual(len(rep["cells_under_multiple_profiles"]), 2)


class TestUnitsAndAdapters(Shared):
    def _raw(self):
        z = np.load(os.path.join(self.ds, "runs", "c00000_companion.npz"))
        return z["current"].astype(float), z["INTENSIDAD"].astype(float)

    def test_grid_units_ok(self):
        J, I = self._raw()
        r = io.check_grid_units(J, I, "A/cm2", 50.0)
        self.assertLess(abs(r - 1), 0.05)

    def test_per_segment_amps_declared_as_density_is_an_error(self):
        """The error this project actually made."""
        J, I = self._raw()
        amps = J * (50.0 / 36)
        with self.assertRaises(io.UnitError) as cm:
            io.check_grid_units(amps, I, "A/cm2", 50.0)
        self.assertIn("A_per_segment", str(cm.exception))
        io.check_grid_units(amps, I, "A_per_segment", 50.0)   # correct label

    def test_wrong_area_is_an_error(self):
        J, I = self._raw()
        with self.assertRaises(io.UnitError):
            io.check_grid_units(J, I, "A/cm2", 25.0)

    def test_negative_sign_must_be_declared(self):
        J, _ = self._raw()
        with self.assertRaises(io.UnitError):
            io.check_sign(-J, 1)
        io.check_sign(-J, -1)

    def test_bulk_plausibility(self):
        i = np.full(100, 0.4)
        with self.assertRaises(io.UnitError):
            io.check_bulk(i, np.full(100, 650.0))          # mV given as V
        with self.assertRaises(io.UnitError):
            io.check_bulk(i * 50, np.full(100, 0.65))      # A given as A/cm2
        with self.assertRaises(io.UnitError):
            io.check_bulk(i, np.full(100, 48.0))           # stack voltage
        io.check_bulk(i, np.full(100, 0.65))

    def test_table_adapter_and_mapping(self):
        rec = self.rec("c00006_fault")
        p = os.path.join(self.tmp, "bulk.csv")
        with open(p, "w") as fh:
            fh.write("time_ms;I_A;U_mV\n")
            for t, i, v in zip(rec["t"], rec["i"] * 50.0, rec["v"]):
                fh.write(f"{t * 1000:.0f};{i:.4f};{v * 1000:.2f}\n")
        mp = {"adapter": "table", "path": p, "sep": ";",
              "time": {"column": "time_ms", "unit": "ms"},
              "current": {"column": "I_A", "unit": "A"},
              "voltage": {"column": "U_mV", "unit": "mV"},
              "active_area_cm2": 50.0}
        got = io.load(mp)[0]
        self.assertIsNone(got["grid"])
        np.testing.assert_allclose(got["i"], rec["i"], atol=2e-5)
        np.testing.assert_allclose(got["v"], rec["v"], atol=2e-5)
        bad = dict(mp, voltage={"column": "U_mV", "unit": "V"})
        with self.assertRaises(io.UnitError):
            io.load(bad)
        with self.assertRaises(io.ContractError):
            io.load({"adapter": "nope"})

    def test_seville_round_trip(self):
        """generate -> Seville raw layout -> Seville adapter -> compare."""
        cfg = gen.make_config(None, dict(
            TINY, grid={"current": [18, 18], "temperature": [9, 9]},
            cohorts=[{"family": "static", "n_cells": 1, "split": "test"}],
            profiles={"static_duration_s": [300, 320]}))
        cells, _ = gen.plan(cfg)
        run, meta, _t = gen.make_run(cfg, cells[0], "fault")
        root = os.path.join(self.tmp, "seville_layout")
        d = seville_export.export_run(run, meta, root)
        self.assertIn("SYNTHETIC", os.path.basename(d))
        for f in os.listdir(d):
            self.assertIn("SYNTHETIC", f)
        with open(os.path.join(d, "SYNTHETIC_README.json")) as fh:
            rd = json.load(fh)
        self.assertTrue(rd["SYNTHETIC"])
        self.assertEqual(rd["seed"], cfg["seed"])
        got = io.load({"adapter": "seville", "data_root": root,
                       "config": meta["flow_config"], "warmup_seconds": 0})[0]
        self.assertTrue(got["meta"]["SYNTHETIC"])
        n = len(run["t"])
        self.assertEqual(got["grid"].shape, (n, 18, 18))
        seg = 50.0 / 324
        # file precision: 4 decimals of per-segment amps
        np.testing.assert_allclose(got["grid"], run["current"],
                                   atol=0.6e-4 / seg)
        np.testing.assert_allclose(got["temperature"], run["temperature"],
                                   atol=1e-3)
        np.testing.assert_allclose(got["i"] * 50.0, run["INTENSIDAD"], atol=2e-3)
        np.testing.assert_allclose(got["v"], run["V001"], atol=1e-4)
        # the raw file really is per-segment amps with a negative sign
        import cdm_parser
        import glob
        raw = cdm_parser.parse_cdm_blocks(glob.glob(os.path.join(d, "CDM_C_*"))[0])
        self.assertTrue((raw["grid"] <= 0).all())
        with self.assertRaises(io.UnitError):
            io.check_grid_units(raw["grid"].astype(float),
                                run["INTENSIDAD"].astype(float), "A/cm2", 50.0)

    def test_messy_dataset_goes_through_the_adapter(self):
        cfg = gen.make_config("messy")
        cells, _ = gen.plan(cfg)
        run, meta, _t = gen.make_run(cfg, cells[0], "fault")
        self.assertTrue((run["current"][np.isfinite(run["current"])] <= 0).all())
        self.assertNotEqual(len(run["t"]), len(run["t_bulk"]))
        self.assertTrue((run["quality_truth"] == "startup").any())
        p = os.path.join(self.tmp, "messy.npz")
        gen.save_npz_deterministic(p, run)
        rec = io.load_canonical_run(p)            # sign + units + clock join
        ok = np.isfinite(rec["grid"]).all(axis=(1, 2))
        self.assertGreater(float(np.nanmin(rec["grid"][ok][5:])), 0.0)
        seg = 50.0 / 64
        r = np.nanmedian(rec["grid"][ok].sum(axis=(1, 2)) * seg
                         / (rec["i"][ok] * 50.0))
        self.assertLess(abs(r - 1), 0.05)


class TestContractAndAbstain(Shared):
    def setUp(self):
        self.cal_rec = self.rec("c00006_cal")
        self.cal = features.calibrate(dict(self.cal_rec, grid=None))
        self.rec0 = self.rec("c00006_companion")

    def run_det(self, rec, cal=None, mode=None):
        return detect.run_recording(rec, cal or self.cal, self.bundle, mode)

    def test_bulk_only_works_without_a_grid(self):
        out, info = self.run_det(dict(self.rec0, grid=None))
        self.assertEqual(info["mode"], "bulk")
        self.assertGreater((out["state"] != "abstain").mean(), 0.9)

    def test_no_voltage_is_a_contract_error(self):
        with self.assertRaises(io.ContractError):
            self.run_det(dict(self.rec0, v=None))

    def test_no_calibration_no_answer(self):
        from hfc import cli
        mp = os.path.join(self.tmp, "m.json")
        with open(mp, "w") as fh:
            json.dump({"adapter": "canonical", "path": self.ds,
                       "runs": ["c00006_companion"]}, fh)
        with self.assertRaises(SystemExit) as cm:
            cli.main(["run", "--mapping", mp, "--models", self.models_dir,
                      "--out", os.path.join(self.tmp, "o")])
        self.assertIn("calibration", str(cm.exception))

    def test_short_calibration_rejected(self):
        with self.assertRaises(ValueError):
            features.calibrate(self.cal_rec, self.cal_rec["t"] < 30)

    def test_abstains_outside_calibrated_load(self):
        lo = features.calibrate(dict(self.cal_rec, grid=None),
                                self.cal_rec["i"] < np.median(self.cal_rec["i"]))
        out, _ = self.run_det(dict(self.rec0, grid=None), lo)
        hi = self.rec0["i"] > lo.d["i_max"] * 1.1
        self.assertTrue(hi.any())
        self.assertTrue((out["state"][hi] == "abstain").all())
        self.assertTrue((out["abstain_reason"][hi]
                         == "load_outside_calibration").all())
        for f in models.FAULTS:
            self.assertFalse(out[f"ahead_{f}_alarm"][hi].any())

    def test_abstains_on_gaps_and_missing_input(self):
        rec = dict(self.rec0, grid=None, v=self.rec0["v"].copy())
        rec["v"][200:230] = np.nan
        out, _ = self.run_det(rec)
        self.assertTrue((out["state"][200:230] == "abstain").all())
        self.assertTrue((out["abstain_reason"][200:230] == "missing_input").all())
        self.assertEqual(out["state"][232], "abstain")     # window still dirty
        self.assertFalse(out["ahead_drying_alarm"][200:260].any())

    def test_abstains_outside_training_envelope(self):
        rec = dict(self.rec0, grid=None, v=self.rec0["v"].copy())
        base, _ = self.run_det(rec)
        ans = base["state"] != "abstain"
        run = np.convolve(ans.astype(int), np.ones(30, dtype=int), "valid")
        k = int(np.flatnonzero(run == 30)[0])   # 30 answered samples in a row
        rec["v"][k:k + 30] += 0.9             # absurd, far above any training
        out, _ = self.run_det(rec)
        self.assertTrue((out["abstain_reason"][k + 2:k + 28]
                         == "outside_training_envelope").all())
        self.assertFalse(out["ahead_starvation_alarm"][k + 2:k + 28].any())

    def test_slow_sampling_disables_capabilities(self):
        slow = {k: (v[::3] if isinstance(v, np.ndarray) else v)
                for k, v in dict(self.rec0, grid=None).items()}
        slow["labels"] = None
        out, info = self.run_det(slow)
        self.assertIn("level-only", info["capabilities"]["state"])
        self.assertIn("disabled", info["capabilities"]["ahead"])
        self.assertFalse(out["ahead_starvation_alarm"].any())
        self.assertTrue((out["ahead_drying_status"] == "disabled").all())
        self.assertGreater((out["state"] != "abstain").mean(), 0.5)
        very = {k: (v[::8] if isinstance(v, np.ndarray) else v)
                for k, v in dict(self.rec0, grid=None).items()}
        very["labels"] = None
        out, info = self.run_det(very)
        self.assertTrue((out["state"] == "abstain").all())
        self.assertTrue((out["abstain_reason"] == "sampling_too_slow").all())

    def test_spatial_handles_other_resolutions(self):
        """Features are resolution-free: a 6x6-trained bundle runs on 10x10."""
        cfg = gen.make_config(None, dict(
            TINY, grid={"current": [10, 10], "temperature": [5, 5]},
            cohorts=[{"family": "fcdlc_random", "n_cells": 1, "split": "test"}]))
        cells, _ = gen.plan(cfg)
        recs = {}
        for role in ("cal", "companion"):
            run, _m, _t = gen.make_run(cfg, cells[0], role)
            p = os.path.join(self.tmp, f"r10_{role}.npz")
            gen.save_npz_deterministic(p, run)
            recs[role] = io.load_canonical_run(p)
        cal = features.calibrate(recs["cal"])
        out, info = detect.run_recording(recs["companion"], cal, self.bundle,
                                         "spatial")
        self.assertEqual(info["mode"], "spatial")
        self.assertGreater((out["state"] != "abstain").mean(), 0.8)
        with self.assertRaises(ValueError):       # mismatched calibration grid
            detect.run_recording(self.rec0, cal, self.bundle, "spatial")

    def test_bulk_is_default_and_spatial_is_opt_in(self):
        cal = features.calibrate(self.cal_rec)
        for shipped in (["bulk"], ["bulk", "spatial"]):
            b = dict(self.bundle, shipped_modes=shipped)
            _o, info = detect.run_recording(self.rec0, cal, b)
            self.assertEqual(info["mode"], "bulk")
        _o, info = detect.run_recording(
            self.rec0, cal, dict(self.bundle, shipped_modes=["bulk"]), "spatial")
        self.assertIn("NOT shipped", info["mode_note"])
        _o, info = detect.run_recording(
            self.rec0, cal, dict(self.bundle, shipped_modes=["bulk", "spatial"]),
            "spatial")
        self.assertEqual((info["mode"], info["mode_note"]), ("spatial", ""))

    def test_shipping_rule(self):
        d = os.path.join(self.tmp, "ship")
        os.makedirs(d, exist_ok=True)
        shutil.copy(os.path.join(self.models_dir, "models.json"), d)
        ok = {"delta": 0.1, "ci": [-0.05, 0.2]}
        bad = {"delta": -0.2, "ci": [-0.3, -0.1]}
        tasks = ["state"] + [f"ahead/{f}" for f in models.FAULTS]
        shipped, worse = models.apply_shipping_rule(
            d, {"spatial_minus_bulk": {t: ok for t in tasks}})
        self.assertEqual(shipped, ["bulk", "spatial"])
        m = {t: ok for t in tasks}
        m["ahead/flooding"] = bad
        shipped, worse = models.apply_shipping_rule(d, {"spatial_minus_bulk": m})
        self.assertEqual((shipped, worse), (["bulk"], ["ahead/flooding"]))


class TestEndToEnd(Shared):
    def test_models_have_convergence_records(self):
        for k, m in self.bundle["models"].items():
            c = m["convergence"]
            self.assertTrue(c["converged"], k)
            self.assertLess(c["n_iter"], c["max_iter"])
        self.assertEqual(self.bundle["nonconverged"], [])

    def test_cli_generate_check_run(self):
        from hfc import cli
        mp = os.path.join(self.tmp, "run.json")
        cp = os.path.join(self.tmp, "cal.json")
        with open(mp, "w") as fh:
            json.dump({"adapter": "canonical", "path": self.ds,
                       "runs": ["c00006_fault", "c00006_companion"]}, fh)
        with open(cp, "w") as fh:
            json.dump({"adapter": "canonical", "path": self.ds,
                       "runs": ["c00006_cal"]}, fh)
        out = os.path.join(self.tmp, "out")
        cli.main(["run", "--mapping", mp, "--models", self.models_dir,
                  "--calibrate-mapping", cp, "--out", out])
        for stem in ("c00006_fault", "c00006_companion"):
            with open(os.path.join(out, stem + "_report.json")) as fh:
                rep = json.load(fh)
            self.assertIn("UNVERIFIED", rep["statement"])
            self.assertTrue(rep["input_is_synthetic"])
            with open(os.path.join(out, stem + "_report.md"),
                      encoding="utf-8") as fh:
                self.assertIn("UNVERIFIED", fh.read())
            with open(os.path.join(out, stem + "_predictions.csv")) as fh:
                head = fh.readline().strip().split(",")
            for col in ("t", "state", "abstain_reason", "p_healthy",
                        "ahead_drying_prob", "ahead_drying_alarm"):
                self.assertIn(col, head)
        with self.assertRaises(SystemExit) as cm:
            cli.main(["check-leaks", self.ds])
        self.assertEqual(cm.exception.code, 0)

    def test_evaluate_reports_cis_on_held_out_cells(self):
        m = models.evaluate(self.ds, self.bundle, n_jobs=1)
        self.assertEqual(m["n_test_cells"], 2)
        s = m["tasks"]["state|bulk"]
        self.assertEqual(len(s["macro_f1_ci"]), 2)
        self.assertIn("false_alarm_companion", s)
        self.assertIn("ahead/drying", m["spatial_minus_bulk"])

    def test_profiles(self):
        ss = np.random.SeedSequence(1, spawn_key=(1, 2))
        a = profiles.build("fcdlc_random", 3, ss, {"dt_s": 1.0})
        b = profiles.build("fcdlc_random", 3, ss, {"dt_s": 1.0})
        np.testing.assert_array_equal(a["load_ratio"], b["load_ratio"])
        self.assertGreaterEqual(a["load_ratio"].min(), 0.0)
        self.assertLessEqual(a["load_ratio"].max(), 1.0)
        self.assertGreater(a["t"][-1], 900)
        s = profiles.build("static", 0, ss, {"dt_s": 1.0})
        self.assertTrue(len(s["induced"]) >= 1)
        self.assertLess(np.ptp(s["load_ratio"][s["segment_type"] == "hold"]),
                        1e-9)
        c = profiles.build("calibration", 0, ss, {"dt_s": 1.0})
        self.assertAlmostEqual(float(c["load_ratio"].max()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
