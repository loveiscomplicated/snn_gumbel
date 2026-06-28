import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.lsm.initialization.fdi_calibration import (
    _scale_input_projection,
    _scale_recurrent_weight,
    calibrate_fdi_style_initial_regime,
    collect_initial_regime_stats,
)
from src.lsm.trainer import build_model
from src.utils.config import Config


def _make_config(neuron_type: str = "lif") -> Config:
    cfg = Config(
        dataset="shd",
        n_input=8,
        n_output=3,
        T=6,
        batch_size=2,
    )
    cfg.liquid.n_liquid = 5
    cfg.liquid.neuron_type = neuron_type
    cfg.liquid.p_input = 0.6
    cfg.liquid.input_weight_scale = 0.5
    cfg.liquid.recurrent_mode = "random_sparse"
    cfg.liquid.recurrent_sparsity = 0.5
    cfg.liquid.w_raw_init_mean = -4.0
    cfg.liquid.w_raw_init_std = 0.01
    cfg.liquid.w_raw_max = -3.0
    cfg.liquid.threshold_min = 0.3
    cfg.liquid.threshold_max = 0.6
    cfg.liquid.beta_min = 0.7
    cfg.liquid.beta_max = 0.9
    cfg.liquid.fdi_probe_batches = 2
    cfg.liquid.fdi_candidate_input_scales = [0.75, 1.0]
    cfg.liquid.fdi_candidate_recurrent_scales = [1.0]
    cfg.liquid.fdi_candidate_threshold_scales = [1.0]
    return cfg


def _make_loader(cfg: Config) -> DataLoader:
    torch.manual_seed(123)
    x = (torch.rand(6, cfg.T, cfg.n_input) < 0.35).float()
    y = torch.zeros(6, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=cfg.batch_size, shuffle=False)


class FDICalibrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.device = torch.device("cpu")

    def test_manual_mode_default_forward_unchanged(self):
        cfg = _make_config()
        self.assertEqual(cfg.liquid.init_mode, "manual")
        model = build_model(cfg, self.device)
        x, _ = next(iter(_make_loader(cfg)))

        out = model(x)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(tuple(out.shape), (cfg.batch_size, cfg.n_output))

    def test_collect_initial_regime_stats_returns_finite_lif_stats(self):
        cfg = _make_config()
        model = build_model(cfg, self.device)
        probe_batches = list(_make_loader(cfg))[: cfg.liquid.fdi_probe_batches]

        stats = collect_initial_regime_stats(model, probe_batches, cfg, self.device)

        for key in (
            "mean_rate_hz",
            "median_rate_hz",
            "max_rate_hz",
            "silent_neuron_frac",
            "overactive_neuron_frac",
            "membrane_mean",
            "membrane_std",
            "threshold_mean",
            "xi_mean",
            "input_current_std",
            "recurrent_current_std",
            "recurrent_to_input_std_ratio",
        ):
            self.assertIn(key, stats)
            self.assertIsNotNone(stats[key], key)
            self.assertTrue(math.isfinite(float(stats[key])), key)
        self.assertNotIn("adaptation_to_threshold_ratio", stats)

    def test_calibration_runs_without_optimizer_and_applies_only_selected(self):
        cfg = _make_config()
        model = build_model(cfg, self.device)
        loader = _make_loader(cfg)
        original_input = model.input_proj.weight.detach().clone()
        original_threshold = model.liquid.threshold.detach().clone()
        original_w_raw = model.liquid.w_raw.detach().clone()

        with tempfile.TemporaryDirectory() as tmp:
            report = calibrate_fdi_style_initial_regime(
                model, loader, cfg, self.device, output_dir=tmp
            )
            report_path = Path(tmp) / "init_fdi_calibration_report.json"
            self.assertTrue(report_path.exists())
            persisted = json.loads(report_path.read_text())

        selected = report["selected_candidate"]
        self.assertEqual(selected, persisted["selected_candidate"])
        self.assertTrue(
            torch.allclose(
                model.input_proj.weight,
                original_input * float(selected["input_scale"]),
            )
        )
        self.assertTrue(
            torch.allclose(
                model.liquid.threshold,
                original_threshold * float(selected["threshold_scale"]),
            )
        )
        self.assertTrue(torch.allclose(model.liquid.w_raw, original_w_raw))
        self.assertIn("skipped_scale_dimensions", report)
        self.assertIn("warnings", report)
        self.assertGreaterEqual(len(report["all_candidates"]), 1)

    def test_input_scale_helper_scales_fixed_sparse_effective_projection(self):
        cfg = _make_config()
        model = build_model(cfg, self.device)
        original_effective_input = model.input_proj.effective_weight().detach().clone()
        original_norm = original_effective_input.norm().item()

        ok, reason = _scale_input_projection(model, 0.75)

        self.assertTrue(ok, reason)
        self.assertIsNone(reason)
        self.assertAlmostEqual(
            model.input_proj.effective_weight().norm().item(),
            original_norm * 0.75,
            places=6,
        )
        self.assertTrue(
            torch.allclose(
                model.input_proj.effective_weight(),
                original_effective_input * 0.75,
            )
        )

    def test_input_scale_helper_scales_learned_sparse_effective_projection(self):
        cfg = _make_config()
        cfg.liquid.input_projection_mode = "learned_sparse"
        cfg.liquid.train_input_projection = True
        model = build_model(cfg, self.device)
        original_effective_input = model.input_proj.effective_weight().detach().clone()
        original_norm = original_effective_input.norm().item()
        original_mask = model.input_proj.mask.detach().clone()

        with torch.no_grad():
            ok, reason = _scale_input_projection(model, 0.75)

        self.assertTrue(ok, reason)
        self.assertIsNone(reason)
        self.assertTrue(model.input_proj.weight.requires_grad)
        self.assertTrue(torch.equal(model.input_proj.mask, original_mask))
        self.assertAlmostEqual(
            model.input_proj.effective_weight().norm().item(),
            original_norm * 0.75,
            places=6,
        )
        self.assertTrue(
            torch.allclose(
                model.input_proj.effective_weight(),
                original_effective_input * 0.75,
            )
        )
        self.assertTrue(
            torch.all(model.input_proj.effective_weight()[model.input_proj.mask == 0] == 0)
        )

    def test_recurrent_scale_helper_uses_soft_gate_scale_buffer(self):
        cfg = _make_config()
        cfg.liquid.recurrent_mode = "soft_gate_lowrank"
        cfg.liquid.train_w_raw = False
        cfg.liquid.noise_scale = 0.0
        cfg.liquid.temp_init = 1.0
        cfg.liquid.target_density_init = 0.3
        model = build_model(cfg, self.device)
        model.liquid.sample_mask()
        original_w = model.liquid.get_effective_weight().detach().clone()
        original_score = model.liquid.get_theta().detach().clone()

        with torch.no_grad():
            ok, reason = _scale_recurrent_weight(model, 0.75)

        self.assertTrue(ok, reason)
        self.assertIsNone(reason)
        self.assertAlmostEqual(float(model.liquid.recurrent_weight_scale.item()), 0.75)
        self.assertTrue(torch.allclose(model.liquid.get_theta(), original_score))
        self.assertTrue(torch.allclose(model.liquid.get_effective_weight(), original_w * 0.75))

    def test_calibration_scales_learned_sparse_input_projection_parameter(self):
        cfg = _make_config()
        cfg.liquid.input_projection_mode = "learned_sparse"
        cfg.liquid.train_input_projection = True
        cfg.liquid.fdi_candidate_input_scales = [0.75]
        cfg.liquid.fdi_candidate_recurrent_scales = [1.0]
        cfg.liquid.fdi_candidate_threshold_scales = [1.0]
        model = build_model(cfg, self.device)
        loader = _make_loader(cfg)
        original_input = model.input_proj.weight.detach().clone()
        original_effective_input = model.input_proj.effective_weight().detach().clone()
        original_mask = model.input_proj.mask.detach().clone()

        with tempfile.TemporaryDirectory() as tmp:
            report = calibrate_fdi_style_initial_regime(
                model, loader, cfg, self.device, output_dir=tmp
            )

        selected = report["selected_candidate"]
        self.assertEqual(float(selected["input_scale"]), 0.75)
        self.assertTrue(model.input_proj.weight.requires_grad)
        self.assertTrue(torch.equal(model.input_proj.mask, original_mask))
        self.assertTrue(
            torch.allclose(
                model.input_proj.weight,
                original_input * float(selected["input_scale"]),
            )
        )
        self.assertTrue(
            torch.allclose(
                model.input_proj.effective_weight(),
                original_effective_input * float(selected["input_scale"]),
            )
        )
        self.assertEqual(report["skipped_scale_dimensions"], [])

    def test_alif_adaptation_stats_are_optional_and_available_for_alif(self):
        cfg = _make_config(neuron_type="alif")
        cfg.liquid.alif_beta_init = 0.1
        model = build_model(cfg, self.device)
        probe_batches = list(_make_loader(cfg))[: cfg.liquid.fdi_probe_batches]

        stats = collect_initial_regime_stats(model, probe_batches, cfg, self.device)

        self.assertIn("adaptation_mean", stats)
        self.assertIn("adaptation_max", stats)
        self.assertIn("adaptation_to_threshold_ratio", stats)
        self.assertTrue(math.isfinite(float(stats["adaptation_to_threshold_ratio"])))

    def test_strict_mode_raises_on_selected_hard_constraint_violation(self):
        cfg = _make_config()
        cfg.liquid.fdi_strict_mode = True
        cfg.liquid.fdi_target_rate_hz = 1000.0
        cfg.liquid.fdi_target_rate_hz_min = 1000.0
        cfg.liquid.fdi_target_rate_hz_max = 2000.0
        cfg.liquid.fdi_candidate_input_scales = [1.0]
        cfg.liquid.fdi_candidate_recurrent_scales = [1.0]
        cfg.liquid.fdi_candidate_threshold_scales = [1.0]
        model = build_model(cfg, self.device)

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                calibrate_fdi_style_initial_regime(
                    model, _make_loader(cfg), cfg, self.device, output_dir=tmp
                )
            self.assertTrue((Path(tmp) / "init_fdi_calibration_report.json").exists())


if __name__ == "__main__":
    unittest.main()
