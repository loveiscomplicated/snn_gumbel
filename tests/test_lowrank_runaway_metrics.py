import math
import tempfile
import types
import unittest
from pathlib import Path

import torch

from src.analysis.lowrank_runaway import (
    generate_report,
    gini,
    lagged_correlations,
    materialize_lowrank_topology,
    sign_aware_strength_metrics,
    topk_overlap,
)


class LowrankRunawayMetricsTest(unittest.TestCase):
    def test_gini_handles_uniform_skewed_and_empty_vectors(self):
        self.assertEqual(gini([]), 0.0)
        self.assertEqual(gini([0.0, 0.0, 0.0]), 0.0)
        self.assertAlmostEqual(gini([1.0, 1.0, 1.0]), 0.0)
        self.assertGreater(gini([0.0, 0.0, 10.0]), 0.6)

    def test_topk_overlap_returns_expected_counts(self):
        overlap = topk_overlap([10.0, 9.0, 1.0, 0.0], [10.0, 0.0, 9.0, 8.0], 2)

        self.assertEqual(overlap["overlap_count"], 1)
        self.assertAlmostEqual(overlap["overlap_fraction"], 0.5)
        self.assertAlmostEqual(overlap["jaccard"], 1.0 / 3.0)
        self.assertEqual(overlap["overlap_ids"], [0])

    def test_lagged_correlations_shape_and_nan_handling(self):
        rows = lagged_correlations([1.0, 1.0, 1.0], [2.0, 3.0, 4.0])

        self.assertEqual(len(rows), 3)
        self.assertEqual({row["lag"] for row in rows}, {
            "x_t_vs_y_t",
            "x_t_minus_1_vs_y_t",
            "y_t_minus_1_vs_x_t",
        })
        self.assertTrue(math.isnan(rows[0]["pearson"]))
        self.assertEqual(rows[0]["n"], 3)

    def test_role_logit_materialization_shape_and_self_mask(self):
        src = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        dst = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self_mask = 1.0 - torch.eye(2)

        mats = materialize_lowrank_topology(src, dst, -0.25, self_mask)

        self.assertEqual(tuple(mats["topology_logit"].shape), (2, 2))
        self.assertEqual(tuple(mats["edge_prob"].shape), (2, 2))
        self.assertFalse(bool(mats["hard_mask"][0, 0]))
        self.assertFalse(bool(mats["valid_mask"][1, 1]))

    def test_missing_key_report_generation_does_not_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = generate_report(
                Path(tmp),
                run_summaries=[
                    {
                        "run_name": "synthetic_missing",
                        "seed": 1,
                        "train_log_available": False,
                        "best_checkpoint_available": False,
                    }
                ],
                epoch_rows=[],
                event_rows=[],
                lagged_rows=[],
                role_rows=[
                    {
                        "run_name": "synthetic_missing",
                        "lowrank_available": True,
                        "hard_density_match": False,
                        "reconstructed_hard_density": 0.10,
                        "model_eval_hard_density": 0.20,
                        "current_mask_density": 0.20,
                    }
                ],
                neuron_rows=[],
                correlation_rows=[],
                overlap_rows=[],
                warnings=["synthetic_missing: missing train.jsonl"],
            )

            self.assertIn("insufficient_temporal_evidence", report)
            self.assertIn("Hard-mask reconstruction warnings", report)
            self.assertTrue((Path(tmp) / "report.md").exists())

    def test_sign_aware_recurrent_strength_metrics(self):
        hard_mask = torch.tensor(
            [
                [0, 1, 1],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=torch.bool,
        )
        liquid = types.SimpleNamespace(
            w_raw=torch.zeros(3, 3),
            w_raw_max=10.0,
            dale_sign=torch.tensor([[1.0], [-1.0], [1.0]]),
            self_conn_mask=1.0 - torch.eye(3),
        )
        model = types.SimpleNamespace(liquid=liquid)

        metrics = sign_aware_strength_metrics(model, hard_mask)
        unit = torch.nn.functional.softplus(torch.tensor(0.0))

        self.assertTrue(
            torch.allclose(
                metrics["weighted_in_abs_strength"],
                torch.tensor([unit, 2 * unit, unit]),
            )
        )
        self.assertTrue(
            torch.allclose(
                metrics["weighted_in_signed_strength"],
                torch.tensor([-unit, 2 * unit, unit]),
            )
        )
        self.assertAlmostEqual(
            float(metrics["incoming_ei_abs_balance"][0].item()),
            -1.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()

