import math
import types
import unittest

import torch

from scripts.neuron_diagnostics import (
    PerNeuronAccumulator,
    graph_node_metrics,
    pearson_corr,
    readout_importance,
    spearman_corr,
)


class NeuronDiagnosticsTest(unittest.TestCase):
    def test_graph_node_metrics_counts_degree_reciprocal_and_triangles(self):
        mask = torch.tensor(
            [
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
            ],
            dtype=torch.bool,
        )

        metrics = graph_node_metrics(mask)

        self.assertTrue(torch.equal(metrics["in_degree"], torch.tensor([1.0, 2.0, 1.0])))
        self.assertTrue(torch.equal(metrics["out_degree"], torch.tensor([1.0, 1.0, 2.0])))
        self.assertTrue(
            torch.equal(metrics["reciprocal_degree"], torch.tensor([0.0, 1.0, 1.0]))
        )
        self.assertTrue(torch.equal(metrics["triangle_count"], torch.tensor([1.0, 1.0, 1.0])))

    def test_readout_importance_splits_spike_adaptation_concat_weights(self):
        readout = torch.nn.Linear(6, 2, bias=False)
        with torch.no_grad():
            readout.weight.copy_(
                torch.tensor(
                    [
                        [3.0, 0.0, 4.0, 0.0, 5.0, 12.0],
                        [0.0, 8.0, 0.0, 6.0, 0.0, 0.0],
                    ]
                )
            )
        model = types.SimpleNamespace(
            n_liquid=3,
            readout_mode="spike_adaptation_concat",
            readout=readout,
        )

        importance = readout_importance(model)

        self.assertTrue(
            torch.allclose(
                importance["readout_spike_weight_norm"],
                torch.tensor([3.0, 8.0, 4.0]),
            )
        )
        self.assertTrue(
            torch.allclose(
                importance["readout_adapt_weight_norm"],
                torch.tensor([6.0, 5.0, 12.0]),
            )
        )
        self.assertTrue(
            torch.allclose(
                importance["readout_total_weight_norm"],
                torch.tensor([math.sqrt(45.0), math.sqrt(89.0), math.sqrt(160.0)]),
            )
        )

    def test_readout_importance_uses_single_block_for_spike_count(self):
        readout = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            readout.weight.copy_(torch.tensor([[3.0, 0.0, 4.0], [0.0, 8.0, 0.0]]))
        model = types.SimpleNamespace(n_liquid=3, readout_mode="spike_count", readout=readout)

        importance = readout_importance(model)

        self.assertTrue(
            torch.allclose(
                importance["readout_total_weight_norm"],
                torch.tensor([3.0, 8.0, 4.0]),
            )
        )
        self.assertTrue(torch.isnan(importance["readout_adapt_weight_norm"]).all())

    def test_correlations_return_nan_for_constant_vectors(self):
        self.assertTrue(math.isnan(pearson_corr([1, 1, 1], [2, 3, 4])))
        self.assertTrue(math.isnan(spearman_corr([1, 1, 1], [2, 3, 4])))

    def test_correlations_handle_valid_vectors(self):
        self.assertAlmostEqual(pearson_corr([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(spearman_corr([1, 2, 3], [3, 2, 1]), -1.0)

    def test_per_neuron_accumulator_tracks_adaptation_abs_mean(self):
        accumulator = PerNeuronAccumulator(n_liquid=2, has_adaptation=True)
        accumulator.update(
            {
                "spikes": torch.zeros(1, 2, 2),
                "input_current": torch.zeros(1, 2, 2),
                "recurrent_current": torch.zeros(1, 2, 2),
                "adaptation": torch.tensor([[[1.0, -2.0], [3.0, -4.0]]]),
            }
        )

        metrics = accumulator.finalize()

        self.assertTrue(torch.allclose(metrics["adaptation_mean"], torch.tensor([2.0, -3.0])))
        self.assertTrue(
            torch.allclose(metrics["adaptation_abs_mean"], torch.tensor([2.0, 3.0]))
        )


if __name__ == "__main__":
    unittest.main()
