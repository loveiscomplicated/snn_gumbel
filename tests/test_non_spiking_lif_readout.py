import unittest

import torch

from src.lsm.model import NonSpikingLIFReadout
from src.lsm.trainer import build_model
from src.utils.config import Config, load_config


def _small_lsm_config() -> Config:
    cfg = Config(
        dataset="shd",
        n_input=4,
        n_output=3,
        T=4,
        batch_size=2,
    )
    cfg.liquid.n_liquid = 5
    cfg.liquid.p_input = 0.7
    cfg.liquid.input_weight_scale = 0.5
    cfg.liquid.recurrent_mode = "random_sparse"
    cfg.liquid.recurrent_sparsity = 0.4
    cfg.liquid.bptt_truncate = 0
    return cfg


class NonSpikingLIFReadoutTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_default_readout_mode_is_spike_count_and_shape_is_unchanged(self):
        cfg = _small_lsm_config()
        self.assertEqual(cfg.liquid.readout_mode, "spike_count")
        model = build_model(cfg, torch.device("cpu"))
        x = torch.zeros(cfg.batch_size, cfg.T, cfg.n_input)

        out = model(x)

        self.assertEqual(model.readout_mode, "spike_count")
        self.assertFalse(model.is_non_spiking_lif_readout)
        self.assertEqual(tuple(out.shape), (cfg.batch_size, cfg.n_output))

    def test_non_spiking_lif_readout_model_output_shape(self):
        cfg = _small_lsm_config()
        cfg.liquid.readout_mode = "non_spiking_lif_final_mem"
        cfg.liquid.readout_lif_beta = 0.95
        model = build_model(cfg, torch.device("cpu"))
        x = torch.zeros(cfg.batch_size, cfg.T, cfg.n_input)

        out = model(x)

        self.assertTrue(model.is_non_spiking_lif_readout)
        self.assertEqual(tuple(out.shape), (cfg.batch_size, cfg.n_output))

    def test_temporal_order_changes_final_membrane(self):
        readout = NonSpikingLIFReadout(n_liquid=1, n_output=1, beta=0.5)
        with torch.no_grad():
            readout.linear.weight.fill_(1.0)
            readout.linear.bias.zero_()
        early = torch.tensor([[[1.0], [0.0], [0.0]]])
        late = torch.tensor([[[0.0], [0.0], [1.0]]])
        self.assertTrue(torch.allclose(early.mean(dim=1), late.mean(dim=1)))

        logits_early = readout(early)
        logits_late = readout(late)

        self.assertFalse(torch.allclose(logits_early, logits_late))

    def test_update_matches_beta_mem_plus_linear_without_reset(self):
        readout = NonSpikingLIFReadout(n_liquid=2, n_output=1, beta=0.5)
        with torch.no_grad():
            readout.linear.weight.copy_(torch.tensor([[2.0, -1.0]]))
            readout.linear.bias.copy_(torch.tensor([0.25]))
        spikes = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])

        logits = readout(spikes)

        expected = torch.tensor([[1.4375]])
        self.assertTrue(torch.allclose(logits, expected))

    def test_gradient_flows_to_readout_linear_weights(self):
        readout = NonSpikingLIFReadout(n_liquid=4, n_output=3, beta=0.95)
        spikes = torch.randn(2, 5, 4, requires_grad=True)

        loss = readout(spikes).sum()
        loss.backward()

        self.assertIsNotNone(readout.linear.weight.grad)
        self.assertIsNotNone(spikes.grad)

    def test_new_config_loads_expected_readout_and_baseline_fields(self):
        cfg = load_config(
            "configs/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_nonspiking_lif_readout.yaml"
        )

        self.assertEqual(cfg.liquid.readout_mode, "non_spiking_lif_final_mem")
        self.assertAlmostEqual(cfg.liquid.readout_lif_beta, 0.95)
        self.assertFalse(cfg.liquid.readout_lif_learn_beta)
        self.assertEqual(cfg.liquid.input_projection_mode, "learned_sparse")
        self.assertTrue(cfg.liquid.train_input_projection)
        self.assertEqual(cfg.liquid.init_mode, "fdi_calibrated")
        self.assertEqual(cfg.liquid.recurrent_mode, "learned_lowrank")
        self.assertEqual(cfg.liquid.theta_rank, 16)


if __name__ == "__main__":
    unittest.main()
