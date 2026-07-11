import unittest

import torch
import torch.nn.functional as F

from src.lsm.model import LiquidLayer
from src.lsm.trainer import build_model, get_soft_gate_schedule
from src.utils.config import Config, load_config


def _small_cfg(mode: str, train_w_raw: bool) -> Config:
    cfg = Config(dataset="shd", n_input=6, n_output=2, T=4, batch_size=2)
    cfg.liquid.n_liquid = 5
    cfg.liquid.recurrent_mode = mode
    cfg.liquid.train_w_raw = train_w_raw
    cfg.liquid.theta_rank = 2
    cfg.liquid.theta_init_mean = -1.0
    cfg.liquid.theta_init_std = 0.1
    cfg.liquid.theta_lowrank_init_std = 0.1
    cfg.liquid.w_raw_init_mean = -2.25
    cfg.liquid.w_raw_init_std = 0.0
    cfg.liquid.w_raw_max = -2.0
    cfg.liquid.input_projection_mode = "learned_sparse"
    cfg.liquid.train_input_projection = True
    return cfg


class RecurrentModeFormulaTest(unittest.TestCase):
    def test_soft_gate_edgewise_formula_uses_gate_and_score_mag(self):
        liquid = LiquidLayer(
            4,
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
                        [0.0, -2.0, -1.0, 0.5],
                        [1.0, 0.0, -0.5, 2.0],
                        [-3.0, 0.25, 0.0, 1.5],
                        [0.75, -1.5, -0.25, 0.0],
                    ]
                )
            )
            liquid.theta_offset.zero_()
        liquid.set_soft_gate_temperature(0.5)
        liquid.sample_mask()

        score = liquid.theta + liquid.theta_offset
        gate = torch.sigmoid(score / 0.5)
        mag = F.softplus(score)
        expected = liquid.recurrent_weight_scale * liquid.self_conn_mask * liquid.dale_sign * gate * mag

        self.assertFalse(liquid.w_raw.requires_grad)
        self.assertTrue(liquid.theta.requires_grad)
        self.assertTrue(torch.allclose(liquid.current_mask, gate))
        self.assertTrue(torch.allclose(liquid.get_effective_weight(), expected))

    def test_soft_gate_density_penalty_depends_on_gate_not_mag(self):
        liquid = LiquidLayer(
            4,
            mode="soft_gate_edgewise",
            theta_init_std=0.1,
            train_w_raw=False,
            mag_from_separate_param=True,
            self_connection=False,
            soft_gate_temp_init=1.0,
            soft_gate_target_density_init=0.3,
        )
        liquid.sample_mask()
        before = liquid.soft_gate_density_penalty(0.3).detach().clone()
        with torch.no_grad():
            liquid.w_core.add_(10.0)
        after = liquid.soft_gate_density_penalty(0.3).detach().clone()

        self.assertTrue(torch.allclose(before, after))

    def test_soft_gate_target_quadratic_penalty_protects_lower_bound(self):
        target = torch.tensor(0.3)
        below = (torch.tensor(0.05) - target) ** 2
        near = (torch.tensor(0.25) - target) ** 2

        self.assertGreater(float(below.item()), float(near.item()))

    def test_soft_gate_quantile_init_matches_initial_soft_density(self):
        for mode in ("soft_gate_lowrank", "soft_gate_edgewise"):
            with self.subTest(mode=mode):
                liquid = LiquidLayer(
                    16,
                    mode=mode,
                    theta_rank=3,
                    theta_init_std=0.05,
                    theta_lowrank_init_std=0.2,
                    train_w_raw=False,
                    self_connection=False,
                    soft_gate_temp_init=1.0,
                    soft_gate_target_density_init=0.3,
                )
                density = liquid.soft_gate_density().item()

                self.assertAlmostEqual(density, 0.3, delta=1e-3)

    def test_soft_gate_ignores_epoch_gumbel_noise(self):
        liquid = LiquidLayer(
            5,
            mode="soft_gate_edgewise",
            theta_init_std=0.1,
            train_w_raw=False,
            self_connection=False,
            soft_gate_temp_init=1.0,
            soft_gate_target_density_init=0.3,
        )
        deterministic = liquid._soft_gate_gate().detach().clone()
        liquid.sample_epoch_mask(tau=0.01, epoch_noise=torch.randn_like(liquid.theta) * 100.0)
        sampled = liquid.sample_mask().detach()

        self.assertTrue(torch.allclose(sampled, deterministic))

    def test_soft_gate_negative_scores_still_have_gradient(self):
        liquid = LiquidLayer(
            4,
            mode="soft_gate_edgewise",
            theta_init_std=0.0,
            train_w_raw=False,
            self_connection=False,
            soft_gate_temp_init=1.0,
            soft_gate_target_density_init=0.3,
        )
        with torch.no_grad():
            liquid.theta.fill_(-8.0)
            liquid.theta_offset.zero_()
        liquid.sample_mask()
        loss = liquid.get_effective_weight().abs().sum()
        loss.backward()

        self.assertIsNotNone(liquid.theta.grad)
        self.assertTrue(torch.isfinite(liquid.theta.grad).all())
        self.assertGreater(float(liquid.theta.grad.abs().sum().item()), 0.0)

    def test_learned_lowrank_grad_r_uses_deterministic_lowrank_ste(self):
        liquid = LiquidLayer(
            3,
            mode="learned_lowrank_grad_r",
            theta_rank=1,
            theta_lowrank_init_std=0.0,
            w_raw_init_mean=-2.0,
            w_raw_init_std=0.0,
            train_w_raw=False,
            self_connection=False,
        )
        with torch.no_grad():
            liquid.src_embed.copy_(torch.tensor([[1.0], [-1.0], [0.5]]))
            liquid.dst_embed.copy_(torch.tensor([[1.0], [2.0], [-3.0]]))
            liquid.theta_bias.zero_()

        theta = liquid.get_theta()
        expected_hard = (theta > 0).float()
        expected_ste = (
            expected_hard - torch.sigmoid(theta).detach() + torch.sigmoid(theta)
        )

        liquid.train()
        liquid.sample_epoch_mask(
            tau=0.01,
            epoch_noise=torch.full_like(theta, 1000.0),
        )
        sampled = liquid.sample_mask()

        self.assertTrue(torch.allclose(sampled, expected_ste))
        self.assertTrue(torch.allclose(sampled.detach(), expected_hard))

        loss = liquid.get_effective_weight().abs().sum()
        loss.backward()

        self.assertIsNotNone(liquid.src_embed.grad)
        self.assertIsNotNone(liquid.dst_embed.grad)
        self.assertIsNotNone(liquid.theta_bias.grad)
        self.assertGreater(float(liquid.src_embed.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(liquid.dst_embed.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(liquid.theta_bias.grad.abs().sum().item()), 0.0)

    def test_grad_r_uses_deterministic_edgewise_ste(self):
        liquid = LiquidLayer(
            3,
            mode="grad_r",
            theta_init_std=0.0,
            w_raw_init_mean=-2.0,
            w_raw_init_std=0.0,
            train_w_raw=False,
            self_connection=False,
        )
        with torch.no_grad():
            liquid.theta.copy_(
                torch.tensor(
                    [
                        [-1.0, 2.0, -0.5],
                        [0.25, -1.0, 0.75],
                        [-0.25, 0.5, -1.0],
                    ]
                )
            )

        theta = liquid.get_theta()
        expected_hard = (theta > 0).float()
        expected_ste = (
            expected_hard - torch.sigmoid(theta).detach() + torch.sigmoid(theta)
        )

        liquid.train()
        liquid.sample_epoch_mask(
            tau=0.01,
            epoch_noise=torch.full_like(theta, 1000.0),
        )
        sampled = liquid.sample_mask()

        self.assertTrue(torch.allclose(sampled, expected_ste))
        self.assertTrue(torch.allclose(sampled.detach(), expected_hard))

        loss = liquid.get_effective_weight().abs().sum()
        loss.backward()

        self.assertIsNotNone(liquid.theta.grad)
        self.assertGreater(float(liquid.theta.grad.abs().sum().item()), 0.0)

    def test_grad_r_regularizers_apply_to_edgewise_theta(self):
        cfg = _small_cfg("grad_r", train_w_raw=False)
        model = build_model(cfg, torch.device("cpu"))

        loss = model.sparsity_loss() + model.commitment_loss()
        loss.backward()

        self.assertIsNotNone(model.liquid.theta.grad)
        self.assertGreater(float(model.liquid.theta.grad.abs().sum().item()), 0.0)

    def test_soft_gate_schedule_anneals_after_warmup(self):
        cfg = Config()
        cfg.liquid.theta_warmup_epochs = 10
        cfg.liquid.temp_init = 1.0
        cfg.liquid.temp_final = 0.2
        cfg.liquid.target_density_init = 0.3
        cfg.liquid.target_density_final = 0.05
        cfg.liquid.target_anneal_epochs = 40

        self.assertEqual(get_soft_gate_schedule(0, cfg), (1.0, 0.3))
        self.assertEqual(get_soft_gate_schedule(10, cfg), (1.0, 0.3))
        temp, target = get_soft_gate_schedule(50, cfg)
        self.assertAlmostEqual(temp, 0.2)
        self.assertAlmostEqual(target, 0.05)

    def test_soft_gate_config_rejects_sampling_and_w_raw_conflicts(self):
        with self.assertRaisesRegex(ValueError, "train_w_raw=false"):
            load_config(
                None,
                overrides=[
                    "liquid.recurrent_mode=soft_gate_lowrank",
                    "liquid.train_w_raw=true",
                    "liquid.noise_scale=0.0",
                ],
            )
        with self.assertRaisesRegex(ValueError, "noise_scale=0.0"):
            load_config(
                None,
                overrides=[
                    "liquid.recurrent_mode=soft_gate_lowrank",
                    "liquid.train_w_raw=false",
                    "liquid.noise_scale=0.1",
                ],
            )
        with self.assertRaisesRegex(ValueError, "density_penalty_lambda"):
            load_config(
                None,
                overrides=[
                    "liquid.recurrent_mode=learned_lowrank",
                    "liquid.density_penalty_lambda=1.0",
                ],
            )
        with self.assertRaisesRegex(ValueError, "noise_scale=0.0"):
            load_config(
                None,
                overrides=[
                    "liquid.recurrent_mode=learned_lowrank_grad_r",
                    "liquid.noise_scale=0.1",
                ],
            )

    def test_edgewise_soft_conductance_is_single_channel_theta(self):
        liquid = LiquidLayer(
            4,
            mode="edgewise_soft_conductance",
            theta_init_mean=-2.0,
            theta_init_std=0.0,
            train_w_raw=False,
            self_connection=False,
        )
        liquid.sample_mask()

        expected = liquid.self_conn_mask * liquid.dale_sign * F.softplus(liquid.theta)

        self.assertFalse(liquid.w_raw.requires_grad)
        self.assertTrue(liquid.theta.requires_grad)
        self.assertTrue(torch.allclose(liquid.get_effective_weight(), expected))

    def test_grad_r_recurrent_weight_scale_multiplies_final_weight_only(self):
        liquid = LiquidLayer(
            4,
            mode="grad_r",
            theta_init_mean=1.0,
            theta_init_std=0.0,
            w_raw_init_mean=-2.25,
            w_raw_init_std=0.0,
            train_w_raw=False,
            self_connection=False,
            recurrent_weight_scale=3.0,
        )
        liquid.sample_mask()
        theta_before = liquid.theta.detach().clone()
        mask_before = liquid.current_mask.detach().clone()

        expected = (
            liquid.recurrent_weight_scale
            * liquid.current_mask
            * liquid.self_conn_mask
            * liquid.dale_sign
            * F.softplus(torch.clamp(liquid.w_raw, max=liquid.w_raw_max))
        )

        self.assertFalse(liquid.w_raw.requires_grad)
        self.assertTrue(torch.allclose(liquid.get_effective_weight(), expected))
        self.assertTrue(torch.allclose(liquid.theta, theta_before))
        self.assertTrue(torch.allclose(liquid.current_mask, mask_before))

    def test_softplus_w_only_is_dense_valid_weight(self):
        liquid = LiquidLayer(
            4,
            mode="softplus_w_only",
            w_raw_init_mean=-2.0,
            w_raw_init_std=0.0,
            train_w_raw=True,
            self_connection=False,
        )
        liquid.sample_mask()

        expected = (
            liquid.self_conn_mask
            * liquid.dale_sign
            * F.softplus(torch.clamp(liquid.w_raw, max=liquid.w_raw_max))
        )

        self.assertTrue(liquid.w_raw.requires_grad)
        self.assertTrue(torch.allclose(liquid.get_effective_weight(), expected))
        self.assertEqual(float((liquid.get_effective_weight().diag() == 0).float().mean()), 1.0)

    def test_smooth_lowrank_scale_matching_is_initial_only_norm_match(self):
        liquid = LiquidLayer(
            5,
            mode="smooth_lowrank_conductance",
            theta_init_mean=1.0,
            theta_rank=2,
            theta_lowrank_init_std=0.0,
            w_raw_init_mean=-2.25,
            w_raw_init_std=0.0,
            train_w_raw=False,
            self_connection=False,
            recurrent_weight_scale=1.0,
            match_initial_w_eff_scale=True,
        )
        liquid.sample_mask()

        hard_mask = (torch.sigmoid(liquid.get_theta()) >= 0.5).float()
        target = (
            hard_mask
            * liquid.self_conn_mask
            * liquid.dale_sign
            * F.softplus(torch.clamp(liquid.w_raw, max=liquid.w_raw_max))
        )

        self.assertFalse(liquid.w_raw.requires_grad)
        self.assertTrue(torch.allclose(liquid.get_effective_weight().norm(), target.norm()))
        with torch.no_grad():
            before = liquid.get_effective_weight().norm().item()
            liquid.src_embed.add_(0.25)
            liquid.dst_embed.add_(0.25)
            after = liquid.get_effective_weight().norm().item()
        self.assertNotEqual(before, after)

    def test_lowrank_frozen_w_constant_g_uses_single_active_magnitude(self):
        liquid = LiquidLayer(
            4,
            mode="learned_lowrank_frozen_w",
            theta_init_mean=1.0,
            theta_rank=2,
            theta_lowrank_init_std=0.0,
            train_w_raw=False,
            frozen_w_mode="constant_g",
            self_connection=False,
        )
        liquid.sample_mask()
        w_eff = liquid.get_effective_weight()
        active = (liquid.get_binary_mask() > 0).bool()

        self.assertFalse(liquid.w_raw.requires_grad)
        self.assertGreater(float(liquid.frozen_w_constant_g.item()), 0.0)
        self.assertTrue(
            torch.allclose(
                w_eff[active].abs(),
                torch.full_like(w_eff[active].abs(), liquid.frozen_w_constant_g.item()),
            )
        )

    def test_random_floor_and_fixed_random_learned_w_trainability(self):
        floor = build_model(_small_cfg("random_sparse", False), torch.device("cpu"))
        learned_w = build_model(_small_cfg("random_sparse", True), torch.device("cpu"))

        self.assertFalse(floor.liquid.w_raw.requires_grad)
        self.assertTrue(learned_w.liquid.w_raw.requires_grad)
        self.assertEqual(floor.liquid.topology_parameters(), [])
        self.assertEqual(learned_w.liquid.topology_parameters(), [])

    def test_ablation_configs_load(self):
        paths = [
            "configs/ablation/lsm_shd_alif_A_current_learned_lowrank_no_rollback.yaml",
            "configs/ablation/lsm_shd_alif_B_random_floor_fixed_sparse_frozen_w.yaml",
            "configs/ablation/lsm_shd_alif_C_softplus_w_only_dense.yaml",
            "configs/ablation/lsm_shd_alif_D_smooth_lowrank_conductance_matched_scale.yaml",
            "configs/ablation/lsm_shd_alif_E_edgewise_soft_conductance.yaml",
            "configs/ablation/lsm_shd_alif_F_fixed_random_sparse_learned_w.yaml",
            "configs/ablation/lsm_shd_alif_G_lowrank_frozen_w_constant_g.yaml",
            "configs/ablation/lsm_shd_alif_H_lowrank_frozen_w_initialized_w.yaml",
            "configs/ablation/lsm_shd_alif_SG_lowrank.yaml",
            "configs/ablation/lsm_shd_alif_SG_edgewise.yaml",
            "configs/ablation/lsm_shd_alif_gradR.yaml",
            "configs/ablation/lsm_shd_alif_lowrank_gradR.yaml",
        ]
        for path in paths:
            with self.subTest(path=path):
                cfg = load_config(path)
                self.assertEqual(cfg.seed, 42)


if __name__ == "__main__":
    unittest.main()
