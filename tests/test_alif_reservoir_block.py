import math
import unittest

import torch
import torch.nn as nn

from src.lsm.model import ALIFReservoirBlock, ALIFReservoirState, LiquidLayer
from src.lsm.trainer import build_model
from src.models.layers import spike_fn
from src.utils.config import Config, load_config


def _make_alif_liquid(dtype: torch.dtype = torch.float32) -> LiquidLayer:
    torch.manual_seed(17)
    liquid = LiquidLayer(
        n_liquid=4,
        exc_ratio=0.75,
        neuron_type="alif",
        mode="fixed",
        target_sparsity=1.0,
        self_connection=True,
        beta_min=0.45,
        beta_max=0.45,
        threshold_min=0.55,
        threshold_max=0.55,
        alif_rho_init=0.70,
        alif_beta_init=0.20,
        alif_adapt_increment=0.35,
        w_raw_init_mean=-2.0,
        w_raw_init_std=0.0,
        w_raw_max=-1.5,
    )
    liquid = liquid.to(dtype=dtype)
    with torch.no_grad():
        liquid.fixed_mask.fill_(1.0)
        liquid.w_raw.fill_(-2.0)
    liquid.sample_mask()
    return liquid


def _old_inline_alif(liquid: LiquidLayer, input_current: torch.Tensor) -> dict:
    batch_size, timesteps, n_liquid = input_current.shape
    membrane = torch.zeros(
        batch_size,
        n_liquid,
        device=input_current.device,
        dtype=input_current.dtype,
    )
    spike = torch.zeros_like(membrane)
    adaptation = torch.zeros_like(membrane)
    traces = {
        "spike": [],
        "adaptation": [],
        "membrane": [],
        "recurrent_current": [],
        "theta_eff": [],
    }

    for t in range(timesteps):
        recurrent_current = liquid(spike)
        membrane_pre_reset = (
            liquid.beta * membrane + input_current[:, t] + recurrent_current
        )
        membrane_pre_reset = torch.clamp(membrane_pre_reset, -3.0, 3.0)
        adaptation = (
            liquid.alif_rho * adaptation + liquid.alif_adapt_increment * spike
        )
        theta_eff = liquid.threshold + liquid.alif_beta * adaptation
        spike = spike_fn(membrane_pre_reset - theta_eff.clamp(min=0.01))

        traces["spike"].append(spike)
        traces["adaptation"].append(adaptation)
        traces["membrane"].append(membrane_pre_reset)
        traces["recurrent_current"].append(recurrent_current)
        traces["theta_eff"].append(theta_eff)

        membrane = membrane_pre_reset * (1.0 - spike)

    return {key: torch.stack(value, dim=1) for key, value in traces.items()}


def _small_alif_config() -> Config:
    cfg = Config(
        dataset="shd",
        n_input=5,
        n_output=3,
        T=4,
        batch_size=2,
    )
    cfg.liquid.n_liquid = 6
    cfg.liquid.neuron_type = "alif"
    cfg.liquid.recurrent_mode = "fixed"
    cfg.liquid.recurrent_sparsity = 0.5
    cfg.liquid.p_input = 0.7
    cfg.liquid.input_weight_scale = 0.4
    cfg.liquid.threshold_min = 0.35
    cfg.liquid.threshold_max = 0.60
    cfg.liquid.alif_beta_init = 0.10
    cfg.liquid.alif_adapt_increment = 0.125
    cfg.liquid.readout_mode = "spike_adaptation_concat"
    return cfg


class ALIFReservoirBlockTest(unittest.TestCase):
    def test_init_state_preserves_shape_device_dtype_and_fields(self):
        liquid = _make_alif_liquid(dtype=torch.float64)
        block = ALIFReservoirBlock(liquid)

        state = block.init_state(
            batch_size=3,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

        self.assertIsInstance(state, ALIFReservoirState)
        for value in (
            state.spike,
            state.membrane,
            state.adaptation,
            state.recurrent_current,
            state.membrane_pre_reset,
            state.theta_eff,
        ):
            self.assertEqual(tuple(value.shape), (3, liquid.n_liquid))
            self.assertEqual(value.device.type, "cpu")
            self.assertEqual(value.dtype, torch.float64)

    def test_forward_returns_spike_and_complete_next_state(self):
        liquid = _make_alif_liquid()
        block = ALIFReservoirBlock(liquid)
        state = block.init_state(batch_size=2, device=torch.device("cpu"))
        input_current = torch.randn(2, liquid.n_liquid)

        spike, next_state = block(input_current, state)

        self.assertEqual(tuple(spike.shape), (2, liquid.n_liquid))
        for value in (
            next_state.spike,
            next_state.membrane,
            next_state.adaptation,
            next_state.recurrent_current,
            next_state.membrane_pre_reset,
            next_state.theta_eff,
        ):
            self.assertEqual(tuple(value.shape), (2, liquid.n_liquid))
        self.assertTrue(torch.equal(spike, next_state.spike))

    def test_theta_eff_is_last_step_observable_not_carried_input(self):
        liquid = _make_alif_liquid()
        block = ALIFReservoirBlock(liquid)
        state = block.init_state(batch_size=2, device=torch.device("cpu"))
        input_current = torch.randn(2, liquid.n_liquid)
        mutated_state = ALIFReservoirState(
            spike=state.spike,
            membrane=state.membrane,
            adaptation=state.adaptation,
            recurrent_current=state.recurrent_current,
            membrane_pre_reset=state.membrane_pre_reset,
            theta_eff=torch.full_like(state.theta_eff, 999.0),
        )

        spike_a, next_a = block(input_current, state)
        spike_b, next_b = block(input_current, mutated_state)

        self.assertTrue(torch.allclose(spike_a, spike_b))
        self.assertTrue(torch.allclose(next_a.membrane, next_b.membrane))
        self.assertTrue(torch.allclose(next_a.adaptation, next_b.adaptation))
        self.assertTrue(torch.allclose(next_a.theta_eff, next_b.theta_eff))

    def test_multi_step_block_matches_previous_inline_alif_equations(self):
        liquid = _make_alif_liquid()
        block = ALIFReservoirBlock(liquid)
        torch.manual_seed(23)
        input_current = torch.randn(2, 5, liquid.n_liquid) * 0.45

        expected = _old_inline_alif(liquid, input_current)
        state = block.init_state(
            batch_size=input_current.shape[0],
            device=input_current.device,
            dtype=input_current.dtype,
        )
        actual = {
            "spike": [],
            "adaptation": [],
            "membrane": [],
            "recurrent_current": [],
            "theta_eff": [],
        }
        for t in range(input_current.shape[1]):
            spike, state = block(input_current[:, t], state)
            actual["spike"].append(spike)
            actual["adaptation"].append(state.adaptation)
            actual["membrane"].append(state.membrane_pre_reset)
            actual["recurrent_current"].append(state.recurrent_current)
            actual["theta_eff"].append(state.theta_eff)
        actual = {key: torch.stack(value, dim=1) for key, value in actual.items()}

        for key in expected:
            self.assertTrue(torch.allclose(actual[key], expected[key]), key)

    def test_model_uses_plain_wrapper_without_state_dict_duplication(self):
        cfg = _small_alif_config()
        model = build_model(cfg, torch.device("cpu"))
        state_keys = list(model.state_dict())

        self.assertIsInstance(model.alif_reservoir, ALIFReservoirBlock)
        self.assertNotIn("alif_reservoir", dict(model.named_modules()))
        self.assertFalse(
            any(key.startswith("alif_reservoir.") for key in state_keys)
        )
        self.assertFalse(any(".alif_reservoir." in key for key in state_keys))
        self.assertEqual(len(state_keys), len(set(state_keys)))
        self.assertIn("liquid.w_raw", model.state_dict())

    def test_model_traces_diagnostics_and_readout_contract(self):
        cfg = _small_alif_config()
        model = build_model(cfg, torch.device("cpu"))
        x = (torch.rand(cfg.batch_size, cfg.T, cfg.n_input) < 0.4).float()
        labels = torch.tensor([0, 1], dtype=torch.long)

        logits, traces, diagnostics = model(
            x,
            return_traces=True,
            return_diagnostics=True,
        )
        loss = nn.CrossEntropyLoss()(logits, labels)

        self.assertEqual(tuple(logits.shape), (cfg.batch_size, cfg.n_output))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.readout.in_features, cfg.liquid.n_liquid * 2)
        for key in (
            "spikes",
            "membrane",
            "input_current",
            "recurrent_current",
            "adaptation",
            "theta_eff",
        ):
            self.assertIn(key, traces)
            self.assertEqual(
                tuple(traces[key].shape),
                (cfg.batch_size, cfg.T, cfg.liquid.n_liquid),
            )
            self.assertTrue(torch.isfinite(traces[key]).all(), key)
        readout_input = torch.cat(
            [traces["spikes"].mean(dim=1), traces["adaptation"].mean(dim=1)],
            dim=1,
        )
        self.assertEqual(
            tuple(readout_input.shape),
            (cfg.batch_size, cfg.liquid.n_liquid * 2),
        )
        for key in (
            "mean_spike_rate",
            "max_spike_rate",
            "adaptation_mean",
            "adaptation_max",
            "membrane_mean",
            "membrane_max",
            "input_current_abs_mean",
            "input_current_abs_max",
            "recurrent_current_abs_mean",
            "recurrent_current_abs_max",
            "rec_input_abs_ratio",
        ):
            self.assertIn(key, diagnostics)
            self.assertTrue(math.isfinite(float(diagnostics[key])), key)

    def test_return_contract_preserves_old_traces_behavior(self):
        cfg = _small_alif_config()
        model = build_model(cfg, torch.device("cpu"))
        x = (torch.rand(cfg.batch_size, cfg.T, cfg.n_input) < 0.4).float()

        trace_result = model(x, return_traces=True)
        self.assertIsInstance(trace_result, tuple)
        self.assertEqual(len(trace_result), 2)
        logits, traces = trace_result
        self.assertEqual(tuple(logits.shape), (cfg.batch_size, cfg.n_output))
        self.assertEqual(
            set(traces),
            {
                "spikes",
                "membrane",
                "input_current",
                "recurrent_current",
                "adaptation",
                "theta_eff",
            },
        )

        diagnostics_result = model(x, return_diagnostics=True)
        self.assertIsInstance(diagnostics_result, tuple)
        self.assertEqual(len(diagnostics_result), 2)
        logits, diagnostics = diagnostics_result
        self.assertEqual(tuple(logits.shape), (cfg.batch_size, cfg.n_output))
        for value in diagnostics.values():
            self.assertTrue(math.isfinite(float(value)))

        combined_result = model(
            x,
            return_traces=True,
            return_diagnostics=True,
        )
        self.assertIsInstance(combined_result, tuple)
        self.assertEqual(len(combined_result), 3)
        logits, traces, diagnostics = combined_result
        self.assertEqual(tuple(logits.shape), (cfg.batch_size, cfg.n_output))
        self.assertIn("membrane", traces)
        self.assertIn("recurrent_current_abs_mean", diagnostics)

    def test_reference_config_loads_and_instantiates_model(self):
        cfg = load_config(
            "configs/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05.yaml"
        )
        model = build_model(cfg, torch.device("cpu"))

        self.assertEqual(cfg.liquid.neuron_type, "alif")
        self.assertEqual(cfg.liquid.recurrent_mode, "learned_lowrank")
        self.assertEqual(cfg.liquid.readout_mode, "spike_adaptation_concat")
        self.assertIsInstance(model.alif_reservoir, ALIFReservoirBlock)


if __name__ == "__main__":
    unittest.main()
