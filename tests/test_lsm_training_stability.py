import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.analyze_lsm_training_run import build_report, load_rows
from src.lsm.trainer import (
    _build_checkpoint_payload,
    _build_optimizer_param_groups,
    _checkpoint_metadata,
    _rank_validation_rows,
    _restore_topology_bundle,
    _selection_row_is_better,
    _snapshot_topology_bundle,
    build_model,
)
from src.utils.config import Config


def _cfg() -> Config:
    cfg = Config(dataset="shd", n_input=4, n_output=2, T=3, batch_size=2)
    cfg.liquid.n_liquid = 5
    cfg.liquid.recurrent_mode = "learned_lowrank"
    cfg.liquid.theta_rank = 2
    cfg.liquid.theta_lr_scale = 0.3
    cfg.liquid.recurrent_sparsity = 0.4
    return cfg


class ValidationComparatorTest(unittest.TestCase):
    def test_val_loss_tie_break_wins_without_using_test_acc(self):
        cfg = Config()
        cfg.selection_val_loss_tie_break = True
        cfg.selection_tie_epsilon = 1e-12
        incumbent = {"epoch": 1, "val_acc": 0.8, "val_loss": 0.5, "test_acc": 0.99}
        candidate = {"epoch": 2, "val_acc": 0.8, "val_loss": 0.4, "test_acc": 0.10}

        improved, reason = _selection_row_is_better(
            candidate, incumbent, cfg, "val_acc"
        )

        self.assertTrue(improved)
        self.assertEqual(reason, "val_acc_tie_lower_val_loss")

    def test_missing_val_loss_keeps_existing_unless_later_flag_enabled(self):
        cfg = Config()
        incumbent = {"epoch": 1, "val_acc": 0.8, "test_acc": 0.1}
        candidate = {"epoch": 2, "val_acc": 0.8, "test_acc": 0.9}

        improved, reason = _selection_row_is_better(
            candidate, incumbent, cfg, "val_acc"
        )
        self.assertFalse(improved)
        self.assertEqual(reason, "val_acc_tie_missing_val_loss_keep_existing")

        cfg.selection_tie_break_later_if_loss_missing = True
        improved, reason = _selection_row_is_better(
            candidate, incumbent, cfg, "val_acc"
        )
        self.assertTrue(improved)
        self.assertEqual(reason, "val_acc_tie_missing_val_loss_later_epoch")

    def test_top_k_uses_same_validation_ranking(self):
        cfg = Config()
        cfg.selection_val_loss_tie_break = True
        rows = [
            {"epoch": 1, "val_acc": 0.8, "val_loss": 0.5},
            {"epoch": 2, "val_acc": 0.8, "val_loss": 0.4},
            {"epoch": 3, "val_acc": 0.79, "val_loss": 0.1},
        ]

        ranked = _rank_validation_rows(rows, cfg)

        self.assertEqual([row["epoch"] for row in ranked], [2, 1, 3])


class CheckpointAndSnapshotTest(unittest.TestCase):
    def test_checkpoint_payload_contains_requested_metadata(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        row = {
            "epoch": 3,
            "train_loss": 0.3,
            "train_acc": 0.7,
            "val_loss": 0.4,
            "val_acc": 0.8,
            "test_acc": 0.75,
            "theta_grad_norm_pre_clip": 12.0,
            "max_firing_rate": 0.91,
        }
        metadata = _checkpoint_metadata(
            row,
            topo_frozen=False,
            topology_rollback_target_epoch=2,
            checkpoint_kind="best",
            selection_tie_break_reason="higher_val_acc",
            checkpoint_in_top_k_val=True,
        )

        payload = _build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            row=row,
            history=[row],
            best_row=row,
            best_metric_name="val_acc",
            topology_freeze_enabled=True,
            topology_freeze_reason=None,
            topology_frozen_epoch=None,
            topology_best_epoch=2,
            topology_best_metric_name="val_acc",
            topology_best_metric_value=0.8,
            topology_rollback_applied_any=False,
            metadata=metadata,
        )

        self.assertIn("checkpoint_metadata", payload)
        self.assertEqual(payload["checkpoint_metadata"]["epoch"], 3)
        self.assertEqual(payload["checkpoint_metadata"]["test_acc"], 0.75)
        self.assertEqual(
            payload["checkpoint_metadata"]["topology_rollback_target_epoch"], 2
        )

    def test_topology_snapshot_restores_params_and_adam_state(self):
        cfg = _cfg()
        model = build_model(cfg, torch.device("cpu"))
        param_groups, _ = _build_optimizer_param_groups(model, cfg)
        optimizer = torch.optim.Adam(param_groups)
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        optimizer.step()

        row = {"epoch": 4, "val_acc": 0.7, "val_loss": 0.6}
        snapshot = _snapshot_topology_bundle(model, optimizer, row)
        saved_src = model.liquid.src_embed.detach().clone()
        saved_exp_avg = optimizer.state[model.liquid.src_embed]["exp_avg"].clone()

        with torch.no_grad():
            model.liquid.src_embed.add_(1.0)
        optimizer.state[model.liquid.src_embed]["exp_avg"].zero_()

        restored_epoch = _restore_topology_bundle(model, optimizer, snapshot)

        self.assertEqual(restored_epoch, 4)
        self.assertTrue(torch.allclose(model.liquid.src_embed, saved_src))
        self.assertTrue(
            torch.allclose(
                optimizer.state[model.liquid.src_embed]["exp_avg"], saved_exp_avg
            )
        )


class AnalyzerReportTest(unittest.TestCase):
    def test_report_includes_requested_fields_and_threshold_warnings(self):
        rows = [
            {
                "epoch": 1,
                "val_acc": 0.8,
                "val_loss": 0.5,
                "test_acc": 0.7,
                "topology_grad_norm_pre_clip": 10.0,
                "max_firing_rate": 0.8,
            },
            {
                "epoch": 2,
                "val_acc": 0.8,
                "val_loss": 0.4,
                "test_acc": 0.76,
                "topology_grad_norm_pre_clip": 60.0,
                "max_firing_rate": 0.91,
                "topology_rollback_target_epoch": 1,
            },
        ]

        report = build_report(
            rows=rows,
            metadata={},
            sources=["synthetic"],
            tie_epsilon=1e-12,
            topology_grad_threshold=50.0,
            low_test_gap=0.03,
        )

        self.assertEqual(report["best_val_epoch"], 2)
        self.assertEqual(report["best_val_acc"], 0.8)
        self.assertEqual(report["test_at_best_val_epoch"], 0.76)
        self.assertEqual(report["all_epochs_tied_at_best_val_acc"], [1, 2])
        self.assertTrue(
            any("topo_grad_norm_pre_clip exceeds threshold" in w for w in report["warnings"])
        )
        self.assertTrue(
            any("max_firing_rate exceeds 0.9" in w for w in report["warnings"])
        )

    def test_log_file_alias_loads_stdout_when_jsonl_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stdout_path = tmp_path / "copied_stdout.txt"
            stdout_path.write_text(
                "[001/002|P2] loss=1.0  train=0.5  val=0.6  test=0.7  "
                "topo_grad=2.0/1.0  fr=0.1/0.2\n"
            )

            rows, _, sources = load_rows(tmp_path, str(stdout_path))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["epoch"], 1)
        self.assertEqual(rows[0]["val_acc"], 0.6)
        self.assertEqual(sources, [str(stdout_path)])


if __name__ == "__main__":
    unittest.main()
