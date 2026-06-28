import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from src.lsm.diagnostics import (
    DiagnosticsLogger,
    collect_epoch_diagnostics,
    collect_topology_metrics,
    deterministic_edge_probabilities,
    entropy,
    gini,
    relative_change,
    safe_argmax,
    safe_float,
    safe_max,
    safe_mean,
    safe_min,
)
from src.lsm.model import LiquidLayer
from src.utils.config import Config, load_config


class _FakeLiquid:
    def __init__(self, values: torch.Tensor, mode: str = "learned_lowrank"):
        self.mode = mode
        self._values = values
        self.self_conn_mask = 1.0 - torch.eye(values.shape[0])
        self.theta_bias = torch.tensor(-1.0)
        self.fixed_mask = values if mode in {"fixed", "random_sparse"} else None

    def get_theta(self):
        return self._values


class _FakeModel:
    def __init__(self, values: torch.Tensor, mode: str = "learned_lowrank"):
        self.liquid = _FakeLiquid(values, mode=mode)
        self._last_spike_rates = torch.tensor(
            [[0.0, 0.10, 0.30], [0.0, 0.20, 0.40]]
        )


class DiagnosticsTest(unittest.TestCase):
    def test_safe_stat_utilities_ignore_missing_and_non_finite(self):
        values = [None, 1.0, float("nan"), torch.tensor(3.0), float("inf")]

        self.assertIsNone(safe_float(float("nan")))
        self.assertEqual(safe_float(torch.tensor(2.5)), 2.5)
        self.assertEqual(safe_mean(values), 2.0)
        self.assertEqual(safe_max(values), 3.0)
        self.assertEqual(safe_min(values), 1.0)
        self.assertEqual(safe_argmax(values), 3)
        self.assertEqual(relative_change(2.0, 3.0), 0.5)
        self.assertIsNone(relative_change(0.0, 3.0))

    def test_gini_and_entropy_are_finite_and_conservative(self):
        self.assertEqual(gini([0.0, 0.0, 0.0]), 0.0)
        self.assertAlmostEqual(gini([1.0, 1.0, 1.0]), 0.0)
        self.assertGreater(gini([0.0, 0.0, 10.0]), 0.6)

        low_entropy = entropy([0.01, 0.99])
        high_entropy = entropy([0.5, 0.5])
        self.assertIsNotNone(low_entropy)
        self.assertIsNotNone(high_entropy)
        self.assertLess(low_entropy, high_entropy)

    def test_topology_metrics_use_expected_degrees_and_top_50_probs(self):
        probs = torch.tensor(
            [
                [0.5, 0.9, 0.1],
                [0.8, 0.5, 0.2],
                [0.7, 0.6, 0.5],
            ]
        )
        logits = torch.logit(probs.clamp(1e-6, 1.0 - 1e-6))
        model = _FakeModel(logits)

        metrics = collect_topology_metrics(model)

        self.assertEqual(metrics["topology_probability_source"], "logits_sigmoid")
        self.assertAlmostEqual(metrics["top_edge_prob_mean"], 0.55, places=6)
        self.assertAlmostEqual(metrics["max_in_degree"], 1.5, places=6)
        self.assertAlmostEqual(metrics["max_out_degree"], 1.3, places=6)
        self.assertGreater(metrics["in_degree_gini"], metrics["out_degree_gini"])

    def test_probability_source_can_be_direct_probabilities(self):
        probs = torch.tensor([[0.2, 0.8], [0.6, 0.4]])
        model = _FakeModel(probs, mode="future_probability_mode")

        edge_probs, _, source = deterministic_edge_probabilities(model)

        self.assertEqual(source, "probabilities_direct")
        self.assertTrue(torch.allclose(edge_probs, probs))

    def test_soft_gate_probability_source_uses_actual_gate(self):
        liquid = LiquidLayer(
            3,
            mode="soft_gate_edgewise",
            theta_init_std=0.0,
            train_w_raw=False,
            self_connection=False,
            soft_gate_temp_init=0.5,
            soft_gate_target_density_init=0.3,
        )
        with torch.no_grad():
            liquid.theta.copy_(
                torch.tensor(
                    [
                        [0.0, -1.0, 1.0],
                        [2.0, 0.0, -2.0],
                        [0.5, -0.5, 0.0],
                    ]
                )
            )
            liquid.theta_offset.zero_()
        model = type("Model", (), {"liquid": liquid})()

        edge_probs, valid, source = deterministic_edge_probabilities(model)

        self.assertEqual(source, "soft_gate")
        self.assertTrue(
            torch.equal(
                valid,
                torch.tensor(
                    [
                        [False, True, True],
                        [True, False, True],
                        [True, True, False],
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.allclose(edge_probs, torch.sigmoid(liquid.get_theta() / 0.5))
        )

    def test_fixed_mask_topology_metrics_include_degree_gini(self):
        mask = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
        model = _FakeModel(mask, mode="fixed")

        metrics = collect_topology_metrics(model)

        self.assertEqual(metrics["topology_probability_source"], "fixed_mask:fixed")
        self.assertIsNotNone(metrics["edge_prob_entropy"])
        self.assertLess(metrics["edge_prob_entropy"], 1e-4)
        self.assertIsNotNone(metrics["in_degree_gini"])
        self.assertIsNotNone(metrics["out_degree_gini"])

    def test_collect_epoch_diagnostics_uses_raw_activity_and_interval_skips(self):
        cfg = Config()
        cfg.diagnostics.topology_log_interval = 5
        model = _FakeModel(torch.zeros(3, 3))
        raw = {
            "epoch": 2,
            "train_loss": 1.0,
            "train_acc": 0.2,
            "val_loss": 0.9,
            "val_acc": 0.3,
            "mean_firing_rate": 0.1,
            "max_firing_rate": 0.4,
            "mean_adaptation": 0.2,
            "max_adaptation": 0.5,
            "membrane_mean": 0.05,
            "membrane_max": 0.2,
            "input_current_abs_mean": 0.4,
            "input_current_abs_max": 0.9,
            "recurrent_current_abs_mean": 0.8,
            "recurrent_current_abs_max": 1.5,
            "rec_input_abs_ratio": 2.0,
            "topology_grad_pre_clip": 2.0,
            "topology_grad_post_clip": 1.0,
            "test_at_best_val_expected": False,
        }

        row = collect_epoch_diagnostics(model, raw, cfg)

        self.assertIn("edge_prob_entropy", row["interval_skipped_metrics"])
        self.assertEqual(row["unsupported_without_extra_forward_trace_metrics"], [])
        self.assertIsNone(row["edge_prob_entropy"])
        self.assertEqual(row["rec_input_abs_ratio"], 2.0)
        self.assertEqual(row["input_current_abs_mean"], 0.4)
        self.assertEqual(row["recurrent_current_abs_mean"], 0.8)
        self.assertAlmostEqual(row["silent_fraction"], 1.0 / 3.0)
        self.assertAlmostEqual(row["overactive_fraction"], 1.0 / 3.0)

    def test_logger_outputs_and_missing_categories(self):
        cfg = Config()
        cfg.diagnostics.enabled = True
        cfg.diagnostics.save_trend_plots = False
        with tempfile.TemporaryDirectory() as tmp:
            logger = DiagnosticsLogger(tmp, cfg)
            logger.log_epoch(
                1,
                {
                    "epoch": 1,
                    "train_loss": 1.0,
                    "val_loss": 1.0,
                    "train_acc": 0.1,
                    "val_acc": 0.2,
                    "best_val_acc_so_far": 0.2,
                    "test_at_best_val": None,
                    "test_at_best_val_expected": False,
                    "mean_firing_rate": 0.1,
                    "max_firing_rate": 0.2,
                    "silent_fraction": 0.0,
                    "overactive_fraction": 0.0,
                    "adaptation_mean": 0.1,
                    "adaptation_max": 0.2,
                    "membrane_mean": None,
                    "membrane_max": None,
                    "theta_grad_norm_pre_clip": 1.0,
                    "theta_grad_norm_post_clip": 1.0,
                    "theta_bias": -1.0,
                    "edge_prob_entropy": 0.5,
                    "edge_prob_mean": 0.25,
                    "edge_prob_std": 0.1,
                    "top_edge_prob_mean": 0.8,
                    "in_degree_gini": 0.1,
                    "out_degree_gini": 0.1,
                    "max_in_degree": 1.0,
                    "max_out_degree": 1.0,
                    "rec_input_abs_ratio": 1.5,
                    "interval_skipped_metrics": [],
                    "unsupported_without_extra_forward_trace_metrics": [],
                },
            )

            summary = logger.summarize_run()
            root = Path(tmp) / "diagnostics"

            self.assertTrue((root / "epoch_metrics.jsonl").exists())
            self.assertTrue((root / "run_summary.json").exists())
            self.assertTrue((root / "red_flags.json").exists())
            self.assertTrue((root / "diagnostic_report.md").exists())
            self.assertEqual(
                summary["unsupported_without_extra_forward_trace_metrics"],
                [],
            )
            self.assertNotIn(
                "test_at_best_val",
                summary["unexpectedly_missing_metrics"],
            )
            persisted = json.loads((root / "run_summary.json").read_text())
            self.assertEqual(persisted["status"], "healthy")
            self.assertEqual(persisted["primary_status"], "healthy")

    def test_red_flag_classification_topology_collapse(self):
        cfg = Config()
        cfg.diagnostics.save_raw_jsonl = False
        with tempfile.TemporaryDirectory() as tmp:
            logger = DiagnosticsLogger(tmp, cfg)
            rows = [
                {
                    "epoch": 1,
                    "train_loss": 1.0,
                    "val_loss": 1.0,
                    "train_acc": 0.1,
                    "val_acc": 0.2,
                    "best_val_acc_so_far": 0.2,
                    "test_at_best_val": 0.1,
                    "test_at_best_val_expected": True,
                    "mean_firing_rate": 0.1,
                    "max_firing_rate": 0.2,
                    "silent_fraction": 0.0,
                    "overactive_fraction": 0.0,
                    "adaptation_mean": 0.1,
                    "adaptation_max": 0.2,
                    "membrane_mean": 0.0,
                    "membrane_max": 0.1,
                    "theta_grad_norm_pre_clip": 1.0,
                    "theta_grad_norm_post_clip": 1.0,
                    "theta_bias": -1.0,
                    "edge_prob_entropy": 0.6,
                    "edge_prob_mean": 0.2,
                    "edge_prob_std": 0.1,
                    "top_edge_prob_mean": 0.6,
                    "in_degree_gini": 0.1,
                    "out_degree_gini": 0.1,
                    "max_in_degree": 1.0,
                    "max_out_degree": 1.0,
                    "rec_input_abs_ratio": 1.0,
                    "unsupported_without_extra_forward_trace_metrics": [],
                },
                {
                    "epoch": 2,
                    "train_loss": 0.9,
                    "val_loss": 0.9,
                    "train_acc": 0.2,
                    "val_acc": 0.25,
                    "best_val_acc_so_far": 0.25,
                    "test_at_best_val": 0.15,
                    "test_at_best_val_expected": True,
                    "mean_firing_rate": 0.1,
                    "max_firing_rate": 0.2,
                    "silent_fraction": 0.0,
                    "overactive_fraction": 0.0,
                    "adaptation_mean": 0.1,
                    "adaptation_max": 0.2,
                    "membrane_mean": 0.0,
                    "membrane_max": 0.1,
                    "theta_grad_norm_pre_clip": 1.0,
                    "theta_grad_norm_post_clip": 1.0,
                    "theta_bias": -1.0,
                    "edge_prob_entropy": 0.3,
                    "edge_prob_mean": 0.3,
                    "edge_prob_std": 0.1,
                    "top_edge_prob_mean": 0.8,
                    "in_degree_gini": 0.4,
                    "out_degree_gini": 0.3,
                    "max_in_degree": 2.0,
                    "max_out_degree": 2.0,
                    "rec_input_abs_ratio": 1.2,
                    "unsupported_without_extra_forward_trace_metrics": [],
                },
            ]
            logger.rows = rows

            summary = logger.summarize_run()

            self.assertEqual(summary["status"], "topology_collapse")

    def test_config_loading_defaults_and_diag_yaml(self):
        base = load_config()
        self.assertFalse(base.diagnostics.enabled)

        diag = load_config(
            "configs/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_diag.yaml"
        )

        self.assertTrue(diag.diagnostics.enabled)
        self.assertEqual(diag.diagnostics.topology_log_interval, 5)
        self.assertTrue(diag.experiment_name.endswith("_diag"))


if __name__ == "__main__":
    unittest.main()
