import json
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from scripts.run_edge_control_interventions import (
    ACTIVE_EDGE_RULE,
    BASELINE_RULE_NATIVE,
    TOPK_ACTIVE_EDGE_RULE,
    active_density,
    apply_edge_removal,
    build_native_active_mask,
    build_topk_active_mask,
    expected_degrees,
    hub_edges,
    ranked_edges_by_score,
    result_row,
    sample_degree_matched_neurons,
    sample_ei_matched_neurons,
    sample_random_active_edges,
    select_top_neurons_by_score,
    summarize_results,
    top_probability_active_edges,
    valid_nonself_mask,
    verify_mask_override_effect,
    write_csv,
    write_report,
    _json_safe,
)
from src.analysis.lowrank_interventions import temporary_recurrent_mask


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(()))
        self.liquid = types.SimpleNamespace()
        self.liquid.current_mask = torch.ones(3, 3)
        self.liquid.eval_mask = torch.tensor(
            [
                [1, 1, 0],
                [0, 1, 1],
                [1, 0, 1],
            ],
            dtype=torch.float32,
        )
        self.liquid.self_conn_mask = 1.0 - torch.eye(3)

        def sample_mask(tau=1.0):
            del tau
            self.liquid.current_mask = self.liquid.eval_mask.clone()
            return self.liquid.current_mask

        def get_effective_weight():
            return self.liquid.current_mask * self.liquid.self_conn_mask

        self.liquid.sample_mask = sample_mask
        self.liquid.get_effective_weight = get_effective_weight


class EdgeControlInterventionsTest(unittest.TestCase):
    def test_topk_active_edge_selection_excludes_self_connections(self):
        probs = torch.tensor(
            [
                [0.99, 0.90, 0.10],
                [0.80, 0.70, 0.60],
                [0.50, 0.40, 0.30],
            ]
        )
        valid = valid_nonself_mask(tuple(probs.shape))

        active, count = build_topk_active_mask(probs, valid, recurrent_sparsity=0.5)

        self.assertEqual(count, 3)
        self.assertFalse(bool(torch.diag(active).any().item()))
        self.assertEqual(
            set(map(tuple, active.nonzero(as_tuple=False).tolist())),
            {(0, 1), (1, 0), (1, 2)},
        )

    def test_top_probability_edge_selection_returns_requested_count(self):
        probs = torch.arange(100, dtype=torch.float32).reshape(10, 10)
        valid = valid_nonself_mask(tuple(probs.shape))
        active = valid.clone()

        edges = top_probability_active_edges(active, probs, top_k=50)

        self.assertEqual(len(edges), 50)
        self.assertEqual(edges[0], (9, 8))
        self.assertNotIn((9, 9), edges)

    def test_random_active_edges_same_count_and_reproducible(self):
        active = torch.tensor(
            [
                [0, 1, 1],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=torch.bool,
        )

        a = sample_random_active_edges(active, 3, seed=123)
        b = sample_random_active_edges(active, 3, seed=123)

        self.assertEqual(a, b)
        self.assertEqual(len(a), 3)
        keep, removed = apply_edge_removal(active, a)
        self.assertEqual(len(removed), 3)
        self.assertEqual(int(active.sum().item()) - int(keep.sum().item()), 3)

    def test_expected_in_and_out_degree_computation(self):
        probs = torch.tensor(
            [
                [0.0, 0.2, 0.4],
                [0.3, 0.0, 0.5],
                [0.7, 0.1, 0.0],
            ]
        )
        valid = valid_nonself_mask(tuple(probs.shape))

        expected_in, expected_out = expected_degrees(probs, valid)

        self.assertTrue(torch.allclose(expected_out, torch.tensor([0.6, 0.8, 0.8])))
        self.assertTrue(torch.allclose(expected_in, torch.tensor([1.0, 0.3, 0.9])))
        self.assertEqual(select_top_neurons_by_score(expected_in, 2), [0, 2])
        self.assertEqual(select_top_neurons_by_score(expected_out, 2), [1, 2])

    def test_hub_edges_follow_requested_direction(self):
        active = torch.tensor(
            [
                [0, 1, 1],
                [1, 0, 0],
                [1, 1, 0],
            ],
            dtype=torch.bool,
        )

        self.assertEqual(set(hub_edges(active, [1], "incoming")), {(0, 1), (2, 1)})
        self.assertEqual(set(hub_edges(active, [2], "outgoing")), {(2, 0), (2, 1)})

    def test_native_active_mask_uses_model_eval_mask_and_excludes_self_connections(self):
        model = DummyModel()
        valid = valid_nonself_mask((3, 3), model.liquid.self_conn_mask)

        active = build_native_active_mask(model, valid, tau=1.0)

        self.assertEqual(
            set(map(tuple, active.nonzero(as_tuple=False).tolist())),
            {(0, 1), (1, 2), (2, 0)},
        )
        self.assertFalse(bool(torch.diag(active).any().item()))
        self.assertAlmostEqual(active_density(active, valid), 3 / 6)

    def test_ei_matched_sampling_preserves_composition(self):
        ei_types = ["E", "E", "E", "I", "I", "I"]

        sampled, reason = sample_ei_matched_neurons([0, 3], ei_types, seed=7)

        self.assertEqual(reason, "")
        self.assertIsNotNone(sampled)
        assert sampled is not None
        self.assertEqual(sorted(ei_types[idx] for idx in sampled), ["E", "I"])
        self.assertTrue(set(sampled).isdisjoint({0, 3}))

    def test_degree_matched_sampling_quality(self):
        degrees = torch.tensor([10.0, 11.0, 12.0, 40.0, 41.0, 42.0])

        sampled, reason, info = sample_degree_matched_neurons([0, 3], degrees, seed=9)

        self.assertEqual(reason, "")
        self.assertIsNotNone(sampled)
        assert sampled is not None
        self.assertEqual(len(sampled), 2)
        self.assertLessEqual(info.mean_abs_degree_gap or 999.0, 2.0)
        self.assertAlmostEqual(info.mean_target_expected_out_degree or 0.0, 25.0)

    def test_temporary_mask_restores_after_success_and_exception(self):
        model = DummyModel()
        original = model.liquid.current_mask.clone()

        with temporary_recurrent_mask(model, torch.zeros(3, 3)):
            self.assertTrue(torch.equal(model.liquid.current_mask, torch.zeros(3, 3)))
        self.assertTrue(torch.equal(model.liquid.current_mask, original))

        with self.assertRaises(RuntimeError):
            with temporary_recurrent_mask(model, torch.zeros(3, 3)):
                raise RuntimeError("boom")
        self.assertTrue(torch.equal(model.liquid.current_mask, original))

    def test_mask_override_verification_uses_effective_recurrent_path(self):
        model = DummyModel()
        active = torch.tensor(
            [
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
            ],
            dtype=torch.bool,
        )

        info = verify_mask_override_effect(model, active)

        self.assertTrue(info["mask_override_verified"])
        self.assertGreater(info["active_effective_nonzero"], 0)
        self.assertEqual(info["zero_effective_nonzero"], 0)

    def test_ranked_edges_has_deterministic_tie_break(self):
        scores = torch.tensor(
            [
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.1, 0.1, 0.0],
            ]
        )
        valid = valid_nonself_mask(tuple(scores.shape))

        edges = ranked_edges_by_score(scores, valid)

        self.assertEqual(edges[:4], [(0, 1), (0, 2), (1, 0), (1, 2)])

    def test_output_csv_json_and_markdown_smoke(self):
        rows = [
            result_row(
                intervention="baseline",
                repeat_id="",
                baseline_acc=0.8,
                intervention_acc=0.8,
                removed_edges=0,
                control_type="baseline",
                seed=42,
                split="test",
                checkpoint=Path("ckpt.pt"),
            ),
            result_row(
                intervention="hub_outgoing_remove",
                repeat_id="",
                baseline_acc=0.8,
                intervention_acc=0.6,
                removed_edges=12,
                target_neurons=[1, 2],
                control_type="target",
                seed=42,
                split="test",
                checkpoint=Path("ckpt.pt"),
            ),
        ]
        summaries = summarize_results(rows)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_csv(output_dir / "edge_control_results.csv", rows)
            summary = {
                "baseline_mask_rule": BASELINE_RULE_NATIVE,
                "interventions": summaries,
                "nan_field": float("nan"),
            }
            (output_dir / "edge_control_summary.json").write_text(
                json.dumps(_json_safe(summary))
            )
            report = write_report(
                output_dir / "edge_control_report.md",
                config=Path("config.yaml"),
                checkpoint=Path("ckpt.pt"),
                split="test",
                seed=42,
                original_model_eval_acc=0.81,
                native_active_baseline_acc=0.81,
                topk_active_baseline_acc=0.80,
                baseline_mask_rule=BASELINE_RULE_NATIVE,
                native_active_density=0.05,
                topk_active_density=0.10,
                baseline_matches_original_eval=True,
                active_edge_rule=TOPK_ACTIVE_EDGE_RULE,
                topology_probability_source="logits_sigmoid",
                summaries=summaries,
                unavailable_rows=[],
            )

            self.assertTrue((output_dir / "edge_control_results.csv").exists())
            self.assertTrue((output_dir / "edge_control_summary.json").exists())
            self.assertTrue((output_dir / "edge_control_report.md").exists())
            loaded = json.loads((output_dir / "edge_control_summary.json").read_text())
            self.assertIsNone(loaded["nan_field"])
            self.assertIn("top-k configured active-mask baseline", report)
            self.assertIn("native_active_baseline_acc", report)


if __name__ == "__main__":
    unittest.main()
