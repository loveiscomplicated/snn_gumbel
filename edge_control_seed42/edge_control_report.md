# Edge-Control Intervention Report

## Setup
- config: `configs/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05.yaml`
- checkpoint: `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_42_260616014632/checkpoints/best.pt`
- split: `test`
- seed: `42`
- original_model_eval_acc: `0.730565`
- native_active_baseline_acc: `0.730565`
- topk_active_baseline_acc: `0.083481`
- baseline_mask_rule: `native_eval`
- native_active_density: `0.046136`
- topk_active_density: `0.100000`
- baseline_matches_original_eval: `true`
- active_edge_rule: `native_eval_recurrent_mask`
- topology_probability_source: `logits_sigmoid`

Delta accuracy is measured against the native eval active-mask baseline.
The top-k configured active-mask baseline is retained as secondary sensitivity analysis and may differ from the original checkpoint eval path.

## Results Summary

| Intervention | Removed edges | Mean Δacc | Std Δacc | Available | Interpretation |
|---|---:|---:|---:|---|---|
| `degree_matched_outgoing_control` | 586.5 | 0.0021 | 0.0030 | yes | accuracy above the native_eval baseline |
| `ei_matched_outgoing_control` | 232.8 | -0.0027 | 0.0039 | yes | accuracy lower than the native_eval baseline |
| `hub_incoming_remove` | 935.0 | -0.2323 | 0.0000 | yes | accuracy lower than the native_eval baseline |
| `hub_outgoing_remove` | 792.0 | -0.0009 | 0.0000 | yes | accuracy lower than the native_eval baseline |
| `random_edges_same_as_top_prob` | 50.0 | 0.0008 | 0.0015 | yes | accuracy above the native_eval baseline |
| `random_edges_same_count_as_hub_incoming` | 935.0 | -0.0145 | 0.0086 | yes | accuracy lower than the native_eval baseline |
| `random_edges_same_count_as_hub_outgoing` | 792.0 | -0.0088 | 0.0066 | yes | accuracy lower than the native_eval baseline |
| `recurrent_current_top_neuron_outgoing_remove` | 555.0 | -0.2385 | 0.0000 | yes | accuracy lower than the native_eval baseline |
| `top_prob_edges_remove` | 50.0 | 0.0000 | 0.0000 | yes | no mean accuracy change from the native_eval baseline |

## Key Comparisons

### Hub incoming vs random same-count
Hub incoming vs random same-count: target removal caused a larger accuracy drop than its control (-0.2323 vs -0.0145), consistent with checkpoint-level decision sensitivity for that edge bundle.

### Hub outgoing vs random same-count
Hub outgoing vs random same-count: target removal did not exceed the matched control drop (-0.0009 vs -0.0088); this weakens a targeted-specificity interpretation.

### Hub outgoing vs E/I-matched
Hub outgoing vs E/I-matched: target removal did not exceed the matched control drop (-0.0009 vs -0.0027); this weakens a targeted-specificity interpretation.

### Hub outgoing vs degree-matched
Hub outgoing vs degree-matched: target removal caused a larger accuracy drop than its control (-0.0009 vs 0.0021), consistent with checkpoint-level decision sensitivity for that edge bundle.

## Unavailable Interventions
- None

## Interpretation Boundary
This is fixed-checkpoint decision sensitivity, not retraining recovery and not proof of training-time causality.
