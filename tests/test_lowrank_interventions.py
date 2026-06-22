import math
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from src.analysis.lowrank_interventions import (
    compute_metric_deltas,
    degree_preserving_directed_swap,
    density_preserving_random_shuffle,
    generate_report,
    sample_ei_matched_random_neurons,
    sample_random_neurons,
    select_top_neurons,
    shuffle_adaptation_features,
    temporary_recurrent_mask,
)


def _rows():
    return [
        {
            "neuron_id": str(idx),
            "ei_type": "E" if idx < 3 else "I",
            "recurrent_current_abs_mean": str(value),
            "total_degree": str(idx),
        }
        for idx, value in enumerate([0.1, 0.9, 0.2, 0.8, 0.3])
    ]


class LowrankInterventionsTest(unittest.TestCase):
    def test_neuron_selection_topk_and_topfrac(self):
        rows = _rows()

        top2, reason = select_top_neurons(
            rows,
            "top_recurrent_current_abs_mean",
            top_k=2,
        )
        frac, frac_reason = select_top_neurons(
            rows,
            "top_recurrent_current_abs_mean",
            top_frac=0.4,
        )

        self.assertEqual(reason, "")
        self.assertEqual(frac_reason, "")
        self.assertEqual(top2, [1, 3])
        self.assertEqual(frac, [1, 3])

    def test_random_controls_are_reproducible_and_ei_matched(self):
        rows = _rows()

        same_a = sample_random_neurons(rows, 3, seed=123)
        same_b = sample_random_neurons(rows, 3, seed=123)
        matched, matching = sample_ei_matched_random_neurons(rows, [0, 3], seed=456)

        self.assertEqual(same_a, same_b)
        self.assertEqual(matching, "ei_matched")
        ei_by_id = {int(row["neuron_id"]): row["ei_type"] for row in rows}
        self.assertEqual(
            sorted(ei_by_id[idx] for idx in matched),
            ["E", "I"],
        )

    def test_adaptation_shuffle_preserves_shape(self):
        adaptation = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        for mode in (
            "adaptation_all_zero",
            "adaptation_selected_zero",
            "adaptation_neuron_shuffle",
            "adaptation_batch_shuffle",
        ):
            shuffled = shuffle_adaptation_features(
                adaptation,
                mode,
                selected=[1, 2],
                seed=7,
            )
            self.assertEqual(tuple(shuffled.shape), tuple(adaptation.shape))
        self.assertTrue(
            torch.equal(
                shuffle_adaptation_features(adaptation, "adaptation_all_zero"),
                torch.zeros_like(adaptation),
            )
        )

    def test_density_preserving_shuffle_preserves_edge_count(self):
        mask = torch.tensor(
            [
                [0, 1, 1],
                [0, 0, 1],
                [1, 0, 0],
            ],
            dtype=torch.bool,
        )

        shuffled = density_preserving_random_shuffle(mask, seed=99)

        self.assertEqual(int(shuffled.sum().item()), int(mask.sum().item()))
        self.assertFalse(bool(torch.diag(shuffled).any().item()))

    def test_degree_preserving_swap_failure_is_graceful(self):
        mask = torch.tensor(
            [
                [0, 1],
                [0, 0],
            ],
            dtype=torch.bool,
        )

        swapped, status, reason = degree_preserving_directed_swap(mask, seed=1)

        self.assertEqual(tuple(swapped.shape), tuple(mask.shape))
        self.assertIn(status, {"insufficient_evidence", "fallback_density_preserving_random_shuffle"})
        self.assertTrue(reason)

    def test_intervention_delta_calculation(self):
        delta = compute_metric_deltas(
            {"accuracy": 0.7, "loss": 0.6},
            {"accuracy": 0.8, "loss": 0.5},
            metric_keys=["accuracy", "loss", "adaptation_mean"],
        )

        self.assertAlmostEqual(delta["delta_accuracy"], -0.1)
        self.assertAlmostEqual(delta["delta_loss"], 0.1)
        self.assertTrue(math.isnan(delta["delta_adaptation_mean"]))

    def test_lif_missing_adaptation_report_generation_does_not_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            report = generate_report(
                output_dir,
                run_rows=[
                    {
                        "run_name": "lif_run",
                        "intervention_type": "original_subset",
                        "accuracy": 0.5,
                        "loss": 1.0,
                    }
                ],
                summary_rows=[],
                verdict_rows=[],
                warnings=["lif_run: adaptation intervention requires ALIF"],
                metadata={"num_batches": 1},
            )

            self.assertIn("checkpoint-level intervention sensitivity", report.lower())
            self.assertIn("Retraining recovery is not tested", report)
            self.assertTrue((output_dir / "report.md").exists())

    def test_temporary_recurrent_mask_restores_state(self):
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.zeros(()))
                self.liquid = types.SimpleNamespace(current_mask=torch.ones(2, 2))

        model = DummyModel()
        original = model.liquid.current_mask.clone()

        with temporary_recurrent_mask(model, torch.zeros(2, 2)):
            self.assertTrue(torch.equal(model.liquid.current_mask, torch.zeros(2, 2)))

        self.assertTrue(torch.equal(model.liquid.current_mask, original))


if __name__ == "__main__":
    unittest.main()
