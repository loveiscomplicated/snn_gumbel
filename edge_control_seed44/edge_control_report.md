# Edge-Control Intervention Report

## Setup
- config: `configs/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05.yaml`
- checkpoint: `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_44_260616083940/checkpoints/best.pt`
- split: `test`
- seed: `44`
- original_model_eval_acc: `0.724382`
- native_active_baseline_acc: `0.724382`
- topk_active_baseline_acc: `0.140018`
- baseline_mask_rule: `native_eval`
- native_active_density: `0.057639`
- topk_active_density: `0.100000`
- baseline_matches_original_eval: `true`
- active_edge_rule: `native_eval_recurrent_mask`
- topology_probability_source: `logits_sigmoid`

Delta accuracy is measured against the native eval active-mask baseline.
The top-k configured active-mask baseline is retained as secondary sensitivity analysis and may differ from the original checkpoint eval path.

## Results Summary

| Intervention | Removed edges | Mean Δacc | Std Δacc | Available | Interpretation |
|---|---:|---:|---:|---|---|
| `degree_matched_outgoing_control` | 963.9 | -0.0157 | 0.0049 | yes | accuracy lower than the native_eval baseline |
| `ei_matched_outgoing_control` | 262.8 | -0.0026 | 0.0051 | yes | accuracy lower than the native_eval baseline |
| `hub_incoming_remove` | 1169.0 | -0.1952 | 0.0000 | yes | accuracy lower than the native_eval baseline |
| `hub_outgoing_remove` | 1331.0 | -0.0671 | 0.0000 | yes | accuracy lower than the native_eval baseline |
| `random_edges_same_as_top_prob` | 50.0 | 0.0012 | 0.0021 | yes | accuracy above the native_eval baseline |
| `random_edges_same_count_as_hub_incoming` | 1169.0 | -0.0265 | 0.0111 | yes | accuracy lower than the native_eval baseline |
| `random_edges_same_count_as_hub_outgoing` | 1331.0 | -0.0348 | 0.0084 | yes | accuracy lower than the native_eval baseline |
| `recurrent_current_top_neuron_outgoing_remove` | 566.0 | -0.1440 | 0.0000 | yes | accuracy lower than the native_eval baseline |
| `top_prob_edges_remove` | 50.0 | -0.0009 | 0.0000 | yes | accuracy lower than the native_eval baseline |

## Key Comparisons

### Hub incoming vs random same-count
Hub incoming vs random same-count: target removal caused a larger accuracy drop than its control (-0.1952 vs -0.0265), consistent with checkpoint-level decision sensitivity for that edge bundle.

### Hub outgoing vs random same-count
Hub outgoing vs random same-count: target removal caused a larger accuracy drop than its control (-0.0671 vs -0.0348), consistent with checkpoint-level decision sensitivity for that edge bundle.

### Hub outgoing vs E/I-matched
Hub outgoing vs E/I-matched: target removal caused a larger accuracy drop than its control (-0.0671 vs -0.0026), consistent with checkpoint-level decision sensitivity for that edge bundle.

### Hub outgoing vs degree-matched
Hub outgoing vs degree-matched: target removal caused a larger accuracy drop than its control (-0.0671 vs -0.0157), consistent with checkpoint-level decision sensitivity for that edge bundle.

## Unavailable Interventions
- None

## Interpretation Boundary
This is fixed-checkpoint decision sensitivity, not retraining recovery and not proof of training-time causality.
