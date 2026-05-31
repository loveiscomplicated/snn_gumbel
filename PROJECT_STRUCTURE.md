# PROJECT_STRUCTURE

## .DS_Store `unknown`

## .gitignore `unknown`

## LICENSE `unknown`

## PROJECT_STRUCTURE.md `MD`

## _legacy/config.py `Python`

## _legacy/evaluate.py `Python`

### 함수
- `get_device()` (L15)
- `load_model(checkpoint_path: str, device)` (L23)
- `run_evaluation(checkpoint_path: str = None)` (L36)

## _legacy/model.py `Python`

### 클래스 / 타입
- **SurrogateSpike** (L19)
  Forward:  Heaviside(x) = 1 if x >= 0 else 0
  - `forward`
  - `backward`
- **GumbelLIFLayer** (L69)
  One feedforward layer of LIF neurons whose connections are learned
  - `__init__`
  - `beta`
  - `forward`
  - `get_binary_mask`
  - `sparsity`
- **SNNModel** (L137)
  Two-layer feedforward SNN:
  - `__init__`
  - `forward`
  - `sparsity_loss`
  - `commitment_loss`
  - `sparsity_info`

### 함수
- `spike_fn(x)` (L38)
- `gumbel_sigmoid(
    logits: torch.Tensor, tau: float = 1.0, hard: bool = False
) torch.Tensor` (L47)
  Bernoulli Gumbel-Softmax relaxation (PGExplainer style).

## _legacy/train.py `Python`

### 함수
- `get_device()` (L19)
- `get_tau(epoch: int) float` (L27)
  Cosine anneal tau from tau_start to tau_end over tau_anneal_epochs.
- `build_dataloaders()` (L36)
- `evaluate(model, loader, device, tau)` (L70)
- `train(resume: bool = False)` (L85)

## _legacy/visualize.py `Python`

### 함수
- `plot_training_curves(history: list, save_path: str = "training_curves.png")` (L16)
- `plot_topology(model, save_path: str = "topology.png")` (L58)
  Visualise binary edge masks as heatmaps.
- `plot_theta_distribution(model, save_path: str = "theta_distribution.png")` (L96)
  Histogram of sigmoid(theta) — distribution of connection probabilities.
- `plot_threshold_distribution(model, save_path: str = "threshold_distribution.png")` (L123)
  Distribution of learned per-neuron firing thresholds.
- `plot_input_connectivity(model, save_path: str = "input_receptive_field.png")` (L147)
  For each hidden neuron, count how many input pixels it receives.
- `run_all(checkpoint_path: str = None)` (L179)

## commands.txt `TXT`

## configs/.DS_Store `unknown`

## configs/ablation_full.yaml `YAML`

## configs/ablation_learned.yaml `YAML`

## configs/ablation_random_sparse.yaml `YAML`

## configs/ablation_transfer.yaml `YAML`

## configs/base.yaml `YAML`

## configs/fashion_mnist_baseline.yaml `YAML`

## configs/lsm_shd_C_valrollback_m50p10.yaml `YAML`

## configs/lsm_shd_alif_learned_lowrank_m50p10.yaml `YAML`

## configs/lsm_shd_alif_lowrank_readout_motor_lif.yaml `YAML`

## configs/lsm_shd_alif_random_sparse_p045_readout_motor_lif.yaml `YAML`

## configs/lsm_shd_alif_random_sparse_p045_fixed.yaml `YAML`

## configs/lsm_shd_alif_random_sparse_p045_midadapt.yaml `YAML`

## configs/lsm_shd_baseline.yaml `YAML`

## configs/lsm_shd_grad_r_valrollback_m50p10.yaml `YAML`

## configs/lsm_shd_lif_lowrank_readout_membrane_trace.yaml `YAML`

## configs/lsm_shd_lif_lowrank_readout_motor_lif.yaml `YAML`

## configs/mnist_baseline.yaml `YAML`

## docs/.DS_Store `unknown`

## docs/alif_implementation.md `MD`

## docs/context_2_lowrank_updated_v7_vision_aligned.md `MD`

## docs/context_lowrank_updated_v7_vision_aligned.md `MD`

## docs/dev_guide_lowrank_updated_v7_vision_aligned.md `MD`

## docs/lsm_implementation_plan_lowrank_updated_v7_vision_aligned.md `MD`

## docs/research_memo_v_0_5_kr_vision_aligned.md `MD`

## docs/research_vision_roadmap_v0_2_kr.md `MD`

## experiments_manifest.csv `CSV`

## requirements.txt `TXT`

## scripts/analyze_topology.py `Python`

### 함수
- `parse_args()` (L70)
- `tensor_summary(x: torch.Tensor) str` (L92)
- `pearson_corr(a: torch.Tensor, b: torch.Tensor) float` (L103)
- `print_header(title: str) None` (L116)
- `infer_label(exp_dir: Path, cfg) str` (L133)
- `load_experiment(exp_dir: Path, device: torch.device) ExperimentTopology` (L143)
- `topk_pairs(values: torch.Tensor, k: int) str` (L181)
- `sign_masks(dale_sign: torch.Tensor) tuple[torch.Tensor, torch.Tensor]` (L189)
- `subset_summary(mask: torch.Tensor, neuron_mask: torch.Tensor) str` (L195)
- `reciprocal_counts(
    mask: torch.Tensor, exc: torch.Tensor, inh: torch.Tensor
) dict[str, int]` (L200)
- `directed_3cycle_count(mask: torch.Tensor) int` (L217)
- `feedforward_triplet_count(mask: torch.Tensor) int` (L223)
- `print_single_experiment_report(exp: ExperimentTopology, topk: int) None` (L233)
- `jaccard(mask_a: torch.Tensor, mask_b: torch.Tensor) tuple[float, int, int, int]` (L292)
- `shared_topk_fraction(a: torch.Tensor, b: torch.Tensor, k: int) float` (L301)
- `print_cross_seed_report(
    experiments: list[ExperimentTopology], topk: int, compare_seed: str
) None` (L308)
- `main()` (L370)

## scripts/build_performance_table.py `Python`

### 함수
- `parse_args() argparse.Namespace` (L66)
- `read_jsonl(path: Path) list[dict[str, Any]]` (L79)
- `finite_float(value: Any) float` (L90)
- `has_metric(row: dict[str, Any], metric: str) bool` (L96)
- `choose_by_metric(
    rows: list[dict[str, Any]], metric: str, realized_rule: str
) tuple[dict[str, Any], str]` (L101)
- `select_row(
    log_rows: list[dict[str, Any]], selection_rule: str
) tuple[dict[str, Any], str]` (L114)
- `build_run_row(manifest_row: dict[str, str]) dict[str, Any]` (L129)
- `summarize(rows: list[dict[str, Any]]) list[dict[str, Any]]` (L167)
- `write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) None` (L206)
- `main() None` (L214)

## scripts/diagnose_input_projection.py `Python`

### 함수
- `parse_args()` (L42)
- `_pairwise_corr(x: torch.Tensor, y: torch.Tensor) float` (L81)
- `compute_projection_matrix_stats(model) dict` (L92)
- `collect_nonzero_timestep_inputs(
    batches: list[tuple[torch.Tensor, torch.Tensor]], max_samples: int
) torch.Tensor` (L150)
- `compute_projection_geometry_stats(
    model,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    max_samples: int,
) dict` (L164)
- `compute_projected_current_stats(model, batches, device) dict` (L228)
- `print_projection_matrix_stats(stats: dict) None` (L253)
- `print_projection_geometry_stats(stats: dict) None` (L304)
- `print_projected_current_stats(stats: dict) None` (L333)
- `_json_default(obj)` (L349)
- `save_json(path: str, payload: dict) None` (L357)
- `main()` (L365)

## scripts/diagnose_liquid.py `Python`

### 클래스 / 타입
- **RunningStats** (L36)
  Streaming scalar stats for tensor values.
  - `__init__`
  - `update`
  - `as_dict`

### 함수
- `parse_args()` (L71)
- `load_checkpoint_if_requested(model, checkpoint_path: str | None, device) None` (L136)
- `fmt_stats(stats: dict[str, float]) str` (L149)
- `print_header(title: str) None` (L156)
- `print_config_summary(cfg, model, checkpoint_path: str | None) None` (L160)
- `print_parameter_sanity(cfg, model) None` (L181)
- `print_recurrent_sparsity(
    model, tau: float, skip_cycle_metrics: bool = False, skip_clustering: bool = False
) dict` (L227)
- `tensor_summary(x: torch.Tensor) str` (L276)
- `print_recurrent_weight_stats(model, binary_mask: torch.Tensor) None` (L287)
- `connected_component_sizes(mask: torch.Tensor) list[int]` (L331)
  Weakly connected component sizes on the undirected version of the graph.
- `gini(x: torch.Tensor) float` (L356)
  Gini coefficient for a non-negative vector; all-zero vectors return 0.
- `_adjacency_without_self_loops(mask: torch.Tensor) torch.Tensor` (L372)
- `ei_block_counts(mask: torch.Tensor, dale_sign: torch.Tensor) dict[str, float]` (L380)
- `reciprocity_metrics(mask: torch.Tensor) dict[str, float]` (L407)
- `directed_3cycle_count(mask: torch.Tensor) int` (L417)
- `average_undirected_clustering(mask: torch.Tensor) float` (L426)
- `compute_graph_metrics(
    mask: torch.Tensor,
    dale_sign: torch.Tensor,
    skip_cycle_metrics: bool = False,
    skip_clustering: bool = False,
) dict[str, float]` (L447)
- `_sanity_check_graph_metrics() None` (L468)
  Small checks documenting expected graph metric behavior.
- `print_graph_topology_metrics(metrics: dict[str, float]) None` (L493)
- `print_graph_structure_stats(mask: torch.Tensor) dict` (L523)
- `collect_batches(loader, n_batches: int) list[tuple[torch.Tensor, torch.Tensor]]` (L562)
- `print_input_spike_stats(batches: list[tuple[torch.Tensor, torch.Tensor]]) None` (L571)
- `run_liquid_diagnostics(model, batches, device, tau: float) dict` (L599)
- `print_current_and_firing_stats(diag: dict) dict` (L652)
- `collect_samples_by_class(
    loader, n_classes: int, samples_per_class: int
) dict[int, list]` (L712)
- `liquid_mean_rate(model, batch: torch.Tensor, device, tau: float) torch.Tensor` (L732)
- `print_class_separation(
    model, loader, device, tau: float, n_classes: int, samples_per_class: int
) tuple[dict[int, list], dict]` (L753)
- `readout_logits_mean(model, batch: torch.Tensor, device, tau: float) torch.Tensor` (L826)
- `print_readout_separation(
    model,
    samples_by_class: dict[int, list],
    device,
    tau: float,
    n_classes: int,
    samples_per_class: int,
) dict` (L832)
- `diagnostic_batch_readout_stats(model, batches, device, tau: float) dict` (L902)
  Compute readout accuracy and margins on the diagnostic batches.
- `_save_json(path: str, data: dict) None` (L939)
- `_append_csv(path: str, row: dict) None` (L953)
- `_save_embeddings(path: str, class_vecs: dict) None` (L979)
- `main()` (L993)

## scripts/evaluate.py `Python`

### 함수
- `main()` (L23)

## scripts/fill_activity_metrics.py `Python`

### 함수
- `parse_args() argparse.Namespace` (L39)
- `read_csv(path: Path) list[dict[str, str]]` (L59)
- `write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) None` (L64)
- `as_float(value: Any) float` (L71)
- `config_experiment_name(experiment_dir: str) str` (L80)
- `dedupe_activity_rows(rows: list[dict[str, str]]) list[dict[str, str]]` (L87)
- `populated_row(
    placeholder: dict[str, str],
    activity: dict[str, str] | None,
    num_batches: int,
) dict[str, Any]` (L99)
- `main() None` (L134)

## scripts/run_activity_diagnostics.py `Python`

### 함수
- `parse_args() argparse.Namespace` (L16)
- `read_manifest(path: Path, groups: set[str]) list[dict[str, str]]` (L39)
- `main() None` (L48)

## scripts/run_topology_diagnostics_from_manifest.py `Python`

### 함수
- `parse_args() argparse.Namespace` (L12)
- `selected_experiments(manifest_path: Path, groups: set[str]) list[str]` (L32)
- `main() None` (L48)

## scripts/summarize_activity_diagnostics.py `Python`

### 함수
- `parse_args() argparse.Namespace` (L37)
- `read_csv(path: Path) list[dict[str, str]]` (L57)
- `to_float(value: Any) float` (L62)
- `mean_std(rows: list[dict[str, str]], metric: str) tuple[float, float]` (L71)
- `dedupe_exact_repeated_runs(rows: list[dict[str, str]]) list[dict[str, str]]` (L81)
  Drop exact repeated rows for the same experiment.
- `main() None` (L99)

## scripts/summarize_diagnostics.py `Python`

### 함수
- `parse_args() argparse.Namespace` (L39)
- `read_csv(path: Path) list[dict[str, str]]` (L51)
- `exp_key(path_or_name: str) str` (L56)
- `to_float(value: Any) float` (L60)
- `mean_std(values: list[float]) tuple[float, float]` (L69)
- `write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) None` (L78)
- `main() None` (L86)

## scripts/topology_diagnostics.ipynb `IPYNB`

## scripts/topology_diagnostics.py `Python`

### 함수
- `parse_args() argparse.Namespace` (L181)
- `select_device(device_arg: str) torch.device` (L217)
  Select device with explicit auto semantics: CUDA > MPS > CPU.
- `resolve_experiment_specs(args: argparse.Namespace) list[dict[str, str | None]]` (L236)
  Resolve CLI inputs into normalized experiment specs.
- `load_config_and_model(
    config_path: str, checkpoint_path: str, device: torch.device
) tuple[Any, torch.nn.Module]` (L281)
  Load config, instantiate the model, and restore checkpoint state.
- `infer_method_label(exp_dir: str | None, cfg: Any) str` (L309)
  Infer a paper-facing method label from experiment name and config.
- `extract_recurrent_mask(model: torch.nn.Module) torch.Tensor` (L339)
  Extract the deterministic recurrent adjacency as a CPU bool tensor.
- `safe_float(x: Any) float` (L368)
  Convert scalar-like values to a Python float, preserving NaN.
- `safe_pearson_corr(x: torch.Tensor, y: torch.Tensor) float` (L383)
  Pearson correlation with NaN on size mismatch or zero variance.
- `_adjacency_without_self_loops(mask: torch.Tensor) torch.Tensor` (L397)
- `_possible_edges(mask: torch.Tensor, model: torch.nn.Module) int` (L403)
- `_dale_sign_vector(model: torch.nn.Module, cfg: Any, n_nodes: int) torch.Tensor` (L410)
- `_edge_type_metrics(mask: torch.Tensor, dale_sign: torch.Tensor) dict[str, float]` (L423)
- `_reciprocity_metrics(mask: torch.Tensor) dict[str, float]` (L446)
- `_strong_component_sizes(mask: torch.Tensor) list[int]` (L457)
- `_path_metrics(mask: torch.Tensor) dict[str, float]` (L505)
- `compute_core_graph_metrics(
    mask: torch.Tensor,
    model: torch.nn.Module,
    cfg: Any,
    *,
    skip_path_metrics: bool,
    use_networkx: bool,
) dict[str, float]` (L557)
  Compute core graph metrics from a deterministic recurrent mask.
- `compute_lowrank_metrics(model: torch.nn.Module, mask: torch.Tensor) dict[str, float]` (L636)
  Compute learned-lowrank-specific embedding/logit diagnostics.
- `compute_readout_topology_metrics(
    model: torch.nn.Module, mask: torch.Tensor
) dict[str, float]` (L702)
  Correlate readout importance with topology centrality.
- `make_activity_placeholder(row_meta: dict[str, Any]) dict[str, Any]` (L726)
  Create the Phase 1 placeholder row for activity metrics.
- `build_summary_row(graph_row: dict[str, Any], activity_row: dict[str, Any]) dict[str, Any]` (L750)
  Merge the report rows into the compact summary schema.
- `write_outputs(
    graph_rows: list[dict[str, Any]],
    activity_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    output_dir: Path,
) None` (L756)
  Write standardized CSV and JSON outputs.
- `_selection_metadata(spec: dict[str, str | None], cfg: Any) dict[str, Any]` (L783)
- `main() None` (L801)

## scripts/train.py `Python`

### 함수
- `main()` (L26)

## scripts/train_lsm.py `Python`

### 함수
- `main()` (L13)

## scripts/upload_wandb.py `Python`

### 함수
- `load_config_yaml(exp_dir: Path) dict` (L27)
- `load_jsonl(exp_dir: Path) list[dict]` (L35)
- `upload(exp_dir: Path, project: str, entity: str | None)` (L49)
- `main()` (L95)

## scripts/visualize.py `Python`

### 함수
- `main()` (L32)

## src/.DS_Store `unknown`

## src/__init__.py `Python`

## src/evaluation/__init__.py `Python`

## src/evaluation/evaluate.py `Python`

### 함수
- `_is_lsm(cfg: Config) bool` (L10)
  Detect if config describes an LSM model.
- `get_device() torch.device` (L15)
- `load_model(checkpoint_path: str, cfg: Config, device: torch.device)` (L23)
- `run_evaluation(checkpoint_path: str, cfg: Config) tuple` (L34)

## src/evaluation/visualize.py `Python`

### 함수
- `plot_training_curves(history: list, save_path: str)` (L23)
- `plot_topology(model, save_path: str)` (L53)
- `plot_theta_distribution(model, save_path: str)` (L71)
- `plot_threshold_distribution(model, save_path: str)` (L90)
- `plot_input_connectivity(model, save_path: str)` (L111)
  Visualise first layer's input→hidden connectivity as a 28×28 heatmap.
- `_history_metric(row: dict, new_key: str, old_key: str, default=0)` (L132)
- `lsm_plot_training_curves(history: list, save_path: str)` (L138)
- `lsm_plot_topology(model, save_path: str)` (L203)
  Visualise liquid recurrent connectivity mask.
- `lsm_plot_theta_distribution(model, save_path: str)` (L236)
  Visualise sigma(theta) distribution for the liquid layer.
- `lsm_plot_threshold_distribution(model, save_path: str)` (L254)
  Visualise learned neuron thresholds and beta (membrane decay).
- `lsm_plot_weight_distribution(model, save_path: str)` (L274)
  Visualise effective weight magnitude distribution.
- `run_all(checkpoint_path: str, cfg, figures_dir: str | None = None)` (L297)

## src/lsm/__init__.py `Python`

## src/lsm/model.py `Python`

### 클래스 / 타입
- **InputProjection** (L25)
  Fixed random sparse connections from input to liquid. Mixed excitatory/inhibitory.
  - `__init__`
  - `forward`
- **LiquidLayer** (L49)
  Recurrent liquid layer with topology learning.
  - `__init__`
  - `beta`
  - `alif_rho`
  - `alif_beta`
  - `get_theta`
  - `topology_parameters`
  - `set_topology_requires_grad`
  - `topology_state_dict`
  - `load_topology_state_dict`
  - `freeze_topology`
  - `sample_epoch_mask`
  - `unlock_epoch_mask`
  - `sample_mask`
  - `get_effective_weight`
  - `forward`
  - `sparsity`
  - `get_binary_mask`
- **LSMModel** (L374)
  - `__init__`
  - `forward`
  - `sparsity_loss`
  - `commitment_loss`
  - `sparsity_info`
  - `firing_rate_info`
  - `adaptation_info`
  - `prediction_loss`
  - `prediction_info`

## src/lsm/trainer.py `Python`

### 함수
- `get_device() torch.device` (L30)
- `get_tau(epoch: int, cfg: Config, warmup_epochs: int | None = None) float` (L38)
- `_make_experiment_dir(cfg: Config) Path` (L52)
- `build_model(cfg: Config, device: torch.device) LSMModel` (L67)
- `_compute_loss(rates, labels, model, cfg)` (L105)
- `_evaluate_metrics(
    model: LSMModel, loader, device: torch.device, tau: float
) tuple[float, float]` (L113)
- `_evaluate(model: LSMModel, loader, device: torch.device, tau: float) float` (L131)
- `_metric_improved(
    metric_name: str, value: float, best_value: float | None, min_delta: float
) bool` (L136)
- `_select_warmup_metric(metric_name: str, row: dict) float` (L146)
- `_warmup_score(metric_name: str, value: float) float` (L160)
- `_warmup_slope(scores: list[float], window: int) float | None` (L164)
- `_grad_norm(params: list[torch.nn.Parameter]) float` (L171)
- `_param_group_lr(optimizer: optim.Optimizer, name: str) float` (L179)
- `_selection_state(val_loader) str` (L186)
- `train(cfg: Config) tuple` (L190)

## src/models/__init__.py `Python`

## src/models/layers.py `Python`

### 클래스 / 타입
- **SurrogateSpike** (L15)
  - `forward`
  - `backward`
- **GumbelLIFLayer** (L88)
  Single LIF layer with topology controlled by `mode`.
  - `__init__`
  - `beta`
  - `forward`
  - `get_binary_mask`
  - `sparsity`

### 함수
- `spike_fn(x)` (L29)
- `gumbel_sigmoid(logits, tau=1.0, hard=False)` (L33)
- `sigmoid_ste(logits)` (L41)
  Deterministic Straight-Through Estimator for binary mask.
- `gumbel_sigmoid_ste(logits, tau=1.0)` (L57)
  Gumbel-Sigmoid with Straight-Through Estimator.

## src/models/snn.py `Python`

### 클래스 / 타입
- **SNNModel** (L20)
  - `__init__`
  - `forward`
  - `sparsity_loss`
  - `commitment_loss`
  - `sparsity_info`
  - `load_topology_from_checkpoint`

## src/training/__init__.py `Python`

## src/training/losses.py `Python`

### 함수
- `total_loss(
    rates, labels, model, lambda_sparse: float, lambda_commit: float
) torch.Tensor` (L12)

## src/training/trainer.py `Python`

### 함수
- `get_device() torch.device` (L28)
- `get_tau(epoch: int, cfg: Config) float` (L36)
- `_make_experiment_dir(cfg: Config) Path` (L44)
- `_cfg_to_dict(cfg: Config) dict` (L60)
- `_evaluate(model: SNNModel, loader, device: torch.device, tau: float) float` (L66)
- `build_model(cfg: Config, device: torch.device) SNNModel` (L79)
- `train(cfg: Config, resume: bool = False) list` (L96)

## src/utils/__init__.py `Python`

## src/utils/config.py `Python`

### 함수
- `_deep_merge(base: dict, override: dict) dict` (L176)
  Recursively merge override into base (in-place on a copy).
- `_load_yaml(path: str | Path) dict` (L187)
- `_resolve_inheritance(yaml_path: str | Path) dict` (L192)
  Load a YAML file, resolving a `base:` key by merging parent first.
- `_apply_cli_overrides(d: dict, overrides: List[str]) dict` (L206)
  Apply overrides of the form "key=value" or "section.key=value".
- `load_config(
    config_path: str | Path | None = None,
    overrides: List[str] | None = None,
) Config` (L234)
  Load config with optional YAML file and CLI overrides.
