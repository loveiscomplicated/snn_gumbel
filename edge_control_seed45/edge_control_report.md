# Edge-Control Intervention Report

## Setup
- config: `configs/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05.yaml`
- checkpoint: `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_45_260616083948/checkpoints/best.pt`
- split: `test`
- seed: `45`
- original_model_eval_acc: `0.722173`
- native_active_baseline_acc: `0.722173`
- topk_active_baseline_acc: `0.068021`
- baseline_mask_rule: `native_eval`
- native_active_density: `0.064573`
- topk_active_density: `0.100000`
- baseline_matches_original_eval: `true`
- active_edge_rule: `native_eval_recurrent_mask`
- topology_probability_source: `logits_sigmoid`

Delta accuracy is measured against the native eval active-mask baseline.
The top-k configured active-mask baseline is retained as secondary sensitivity analysis and may differ from the original checkpoint eval path.

## Results Summary

| Intervention | Removed edges | Mean Δacc | Std Δacc | Available | Interpretation |
|---|---:|---:|---:|---|---|
| `degree_matched_outgoing_control` | 1015.9 | -0.0298 | 0.0081 | yes | accuracy lower than the native_eval baseline |
| `ei_matched_outgoing_control` | 283.4 | -0.0057 | 0.0069 | yes | accuracy lower than the native_eval baseline |
| `hub_incoming_remove` | 1034.0 | -0.1471 | 0.0000 | yes | accuracy lower than the native_eval baseline |
| `hub_outgoing_remove` | 1191.0 | -0.0106 | 0.0000 | yes | accuracy lower than the native_eval baseline |
| `random_edges_same_as_top_prob` | 50.0 | 0.0010 | 0.0021 | yes | accuracy above the native_eval baseline |
| `random_edges_same_count_as_hub_incoming` | 1034.0 | -0.0256 | 0.0082 | yes | accuracy lower than the native_eval baseline |
| `random_edges_same_count_as_hub_outgoing` | 1191.0 | -0.0380 | 0.0125 | yes | accuracy lower than the native_eval baseline |
| `recurrent_current_top_neuron_outgoing_remove` | 442.0 | -0.3370 | 0.0000 | yes | accuracy lower than the native_eval baseline |
| `top_prob_edges_remove` | 50.0 | -0.0013 | 0.0000 | yes | accuracy lower than the native_eval baseline |

## Key Comparisons

### Hub incoming vs random same-count
Hub incoming vs random same-count: target removal caused a larger accuracy drop than its control (-0.1471 vs -0.0256), consistent with checkpoint-level decision sensitivity for that edge bundle.

### Hub outgoing vs random same-count
Hub outgoing vs random same-count: target removal did not exceed the matched control drop (-0.0106 vs -0.0380); this weakens a targeted-specificity interpretation.

### Hub outgoing vs E/I-matched
Hub outgoing vs E/I-matched: target removal caused a larger accuracy drop than its control (-0.0106 vs -0.0057), consistent with checkpoint-level decision sensitivity for that edge bundle.

### Hub outgoing vs degree-matched
Hub outgoing vs degree-matched: target removal did not exceed the matched control drop (-0.0106 vs -0.0298); this weakens a targeted-specificity interpretation.

## Unavailable Interventions
- None

## Interpretation Boundary
This is fixed-checkpoint decision sensitivity, not retraining recovery and not proof of training-time causality.
