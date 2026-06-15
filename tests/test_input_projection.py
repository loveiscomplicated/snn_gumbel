import unittest

import torch
import torch.nn as nn

from src.lsm.model import InputProjection
from src.lsm.trainer import (
    _build_optimizer_param_groups,
    build_model,
)
from src.utils.config import Config


def _small_lsm_config() -> Config:
    cfg = Config(
        dataset="shd",
        n_input=4,
        n_output=3,
        T=3,
        batch_size=2,
        lr=0.01,
        weight_decay=0.123,
    )
    cfg.liquid.n_liquid = 5
    cfg.liquid.p_input = 0.6
    cfg.liquid.input_weight_scale = 0.5
    cfg.liquid.recurrent_mode = "random_sparse"
    cfg.liquid.recurrent_sparsity = 0.4
    cfg.liquid.bptt_truncate = 0
    return cfg


def _projection_with_known_mask(trainable: bool) -> InputProjection:
    proj = InputProjection(
        n_input=2,
        n_liquid=2,
        p_input=1.0,
        weight_scale=1.0,
        mode="learned_sparse",
        trainable=trainable,
    )
    with torch.no_grad():
        proj.mask.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        proj.weight.copy_(torch.tensor([[0.5, 0.0], [0.0, -0.5]]))
    return proj


class InputProjectionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_fixed_sparse_preserves_old_initialization_and_shape(self):
        torch.manual_seed(123)
        proj = InputProjection(
            n_input=4,
            n_liquid=5,
            p_input=0.4,
            weight_scale=0.2,
        )

        torch.manual_seed(123)
        expected_mask = (torch.rand(4, 5) < 0.4).float()
        expected_weight = torch.randn(4, 5) * 0.2 * expected_mask

        self.assertFalse(any(param.requires_grad for param in proj.parameters()))
        self.assertIn("mask", dict(proj.named_buffers()))
        self.assertIn("weight", dict(proj.named_buffers()))
        self.assertTrue(torch.equal(proj.mask, expected_mask))
        self.assertTrue(torch.allclose(proj.weight, expected_weight))

        x = torch.randn(3, 4)
        self.assertEqual(tuple(proj(x).shape), (3, 5))
        self.assertTrue(torch.allclose(proj(x), x @ expected_weight))

    def test_default_lsm_input_projection_is_not_trainable(self):
        cfg = _small_lsm_config()
        model = build_model(cfg, torch.device("cpu"))
        x = torch.zeros(cfg.batch_size, cfg.T, cfg.n_input)

        out = model(x)

        self.assertEqual(tuple(out.shape), (cfg.batch_size, cfg.n_output))
        self.assertEqual(list(model.input_proj.parameters()), [])
        self.assertFalse(model.input_proj.trainable)

    def test_learned_sparse_uses_parameter_and_buffer_mask(self):
        proj = InputProjection(
            n_input=4,
            n_liquid=5,
            p_input=0.5,
            weight_scale=0.2,
            mode="learned_sparse",
            trainable=True,
        )

        self.assertIsInstance(proj.weight, nn.Parameter)
        self.assertTrue(proj.weight.requires_grad)
        self.assertIn("mask", dict(proj.named_buffers()))

        x = torch.randn(3, 4)
        self.assertEqual(tuple(proj(x).shape), (3, 5))
        self.assertTrue(torch.all(proj.effective_weight()[proj.mask == 0] == 0))

    def test_frozen_learned_sparse_keeps_parameter_frozen(self):
        proj = InputProjection(
            n_input=4,
            n_liquid=5,
            p_input=0.5,
            weight_scale=0.2,
            mode="learned_sparse",
            trainable=False,
        )

        self.assertIsInstance(proj.weight, nn.Parameter)
        self.assertFalse(proj.weight.requires_grad)
        self.assertFalse(proj.trainable)

    def test_learned_sparse_train_step_updates_effective_weights(self):
        proj = _projection_with_known_mask(trainable=True)
        optimizer = torch.optim.SGD(proj.parameters(), lr=0.1, weight_decay=0.0)
        before = proj.effective_weight().detach().clone()
        x = torch.tensor([[1.0, 2.0], [-1.0, 1.0]])

        loss = proj(x).pow(2).sum()
        loss.backward()
        optimizer.step()

        after = proj.effective_weight().detach()
        self.assertFalse(torch.allclose(after[proj.mask.bool()], before[proj.mask.bool()]))
        self.assertTrue(torch.all(after[proj.mask == 0] == 0))

    def test_frozen_learned_sparse_train_step_does_not_update(self):
        proj = _projection_with_known_mask(trainable=False)
        optimizer = torch.optim.SGD(proj.parameters(), lr=0.1, weight_decay=0.0)
        before = proj.effective_weight().detach().clone()
        x = torch.tensor([[1.0, 2.0], [-1.0, 1.0]], requires_grad=True)

        loss = proj(x).pow(2).sum()
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.allclose(proj.effective_weight().detach(), before))
        self.assertIsNone(proj.weight.grad)

    def test_masked_out_weights_have_zero_gradient_and_effective_weight(self):
        proj = _projection_with_known_mask(trainable=True)
        optimizer = torch.optim.SGD(proj.parameters(), lr=0.1, weight_decay=0.0)
        x = torch.tensor([[1.0, 2.0], [-1.0, 1.0]])

        loss = proj(x).pow(2).sum()
        loss.backward()
        self.assertTrue(torch.all(proj.weight.grad[proj.mask == 0] == 0))
        optimizer.step()

        self.assertTrue(torch.all(proj.effective_weight().detach()[proj.mask == 0] == 0))

    def test_optimizer_group_for_trainable_input_projection(self):
        cfg = _small_lsm_config()
        cfg.liquid.input_projection_mode = "learned_sparse"
        cfg.liquid.train_input_projection = True
        cfg.liquid.input_proj_lr_scale = 0.25
        model = build_model(cfg, torch.device("cpu"))

        param_groups, metadata = _build_optimizer_param_groups(model, cfg)
        input_group = next(
            group for group in param_groups if group["name"] == "input_projection"
        )
        all_group_param_ids = [
            id(param) for group in param_groups for param in group["params"]
        ]

        self.assertEqual(metadata["input_projection_params"], [model.input_proj.weight])
        self.assertIs(input_group["params"][0], model.input_proj.weight)
        self.assertAlmostEqual(input_group["lr"], cfg.lr * cfg.liquid.input_proj_lr_scale)
        self.assertEqual(input_group["weight_decay"], 0.0)
        self.assertEqual(len(all_group_param_ids), len(set(all_group_param_ids)))
        self.assertNotIn(
            id(model.input_proj.weight),
            {id(param) for param in metadata["other_params"]},
        )

    def test_frozen_learned_sparse_is_not_in_optimizer_groups(self):
        cfg = _small_lsm_config()
        cfg.liquid.input_projection_mode = "learned_sparse"
        cfg.liquid.train_input_projection = False
        model = build_model(cfg, torch.device("cpu"))

        param_groups, metadata = _build_optimizer_param_groups(model, cfg)
        group_names = {group["name"] for group in param_groups}
        all_group_param_ids = {
            id(param) for group in param_groups for param in group["params"]
        }

        self.assertNotIn("input_projection", group_names)
        self.assertEqual(metadata["input_projection_params"], [])
        self.assertNotIn(id(model.input_proj.weight), all_group_param_ids)


if __name__ == "__main__":
    unittest.main()
