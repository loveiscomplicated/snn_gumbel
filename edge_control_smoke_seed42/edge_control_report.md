# Edge-Control Intervention Report

## Setup
- config: `configs/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05.yaml`
- checkpoint: `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_42_260616014632/checkpoints/best.pt`
- split: `test`
- seed: `42`
- original_model_eval_acc: `0.742188`
- topk_active_baseline_acc: `0.078125`
- baseline_mask_rule: `topk_by_configured_recurrent_sparsity`
- active_edge_rule: `topk_by_configured_recurrent_sparsity`
- topology_probability_source: `logits_sigmoid`

This analysis uses a deterministic top-k active-mask baseline and may differ from the original checkpoint eval path.

## Results Summary

| Intervention | Removed edges | Mean Δacc | Std Δacc | Available | Interpretation |
|---|---:|---:|---:|---|---|
| `degree_matched_outgoing_control` | 1023.5 | -0.0078 | 0.0000 | yes | accuracy lower than the top-k active-mask baseline |
| `ei_matched_outgoing_control` | 557.5 | -0.0156 | 0.0000 | yes | accuracy lower than the top-k active-mask baseline |
| `hub_incoming_remove` | 1344.0 | -0.0078 | 0.0000 | yes | accuracy lower than the top-k active-mask baseline |
| `hub_outgoing_remove` | 1269.0 | -0.0156 | 0.0000 | yes | accuracy lower than the top-k active-mask baseline |
| `random_edges_same_as_top_prob` | 50.0 | -0.0078 | 0.0078 | yes | accuracy lower than the top-k active-mask baseline |
| `random_edges_same_count_as_hub_incoming` | 1344.0 | 0.0000 | 0.0000 | yes | no mean accuracy change from the top-k active-mask baseline |
| `random_edges_same_count_as_hub_outgoing` | 1269.0 | -0.0117 | 0.0039 | yes | accuracy lower than the top-k active-mask baseline |
| `recurrent_current_top_neuron_outgoing_remove` |  |  |  | partial/no | missing_recurrent_current_diagnostics |
| `top_prob_edges_remove` | 50.0 | 0.0000 | 0.0000 | yes | no mean accuracy change from the top-k active-mask baseline |

## Key Comparisons

### Hub incoming vs random same-count
Hub incoming vs random same-count: target removal caused a larger accuracy drop than its control (-0.0078 vs 0.0000), consistent with checkpoint-level decision sensitivity for that edge bundle.

### Hub outgoing vs random same-count
Hub outgoing vs random same-count: target removal caused a larger accuracy drop than its control (-0.0156 vs -0.0117), consistent with checkpoint-level decision sensitivity for that edge bundle.

### Hub outgoing vs E/I-matched
Hub outgoing vs E/I-matched: target removal did not exceed the matched control drop (-0.0156 vs -0.0156); this weakens a targeted-specificity interpretation.

### Hub outgoing vs degree-matched
Hub outgoing vs degree-matched: target removal caused a larger accuracy drop than its control (-0.0156 vs -0.0078), consistent with checkpoint-level decision sensitivity for that edge bundle.

## Unavailable Interventions
- `recurrent_current_top_neuron_outgoing_remove`: missing_recurrent_current_diagnostics

## Interpretation Boundary
This is fixed-checkpoint decision sensitivity, not retraining recovery and not proof of training-time causality.
