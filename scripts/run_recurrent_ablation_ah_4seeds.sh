#!/usr/bin/env bash
set -euo pipefail

# Run recurrent ablation configs A-H sequentially for seeds 42-45.
#
# Examples:
#   bash scripts/run_recurrent_ablation_ah_4seeds.sh
#   DRY_RUN=1 bash scripts/run_recurrent_ablation_ah_4seeds.sh
#   DISABLE_TQDM=0 bash scripts/run_recurrent_ablation_ah_4seeds.sh
#   DEVICE=cuda DATA_DIR=/workspace/data bash scripts/run_recurrent_ablation_ah_4seeds.sh
#   EXTRA_OVERRIDES="epochs=1 batch_size=16" bash scripts/run_recurrent_ablation_ah_4seeds.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_CMD="${PYTHON_CMD:-uv run python -u}"
DEVICE="${DEVICE:-auto}"
DATA_DIR="${DATA_DIR:-./data}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-logs/recurrent_ablation_ah_$(date +%y%m%d%H%M%S)}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"

SEEDS=(42 43 44 45)
CONFIGS=(
  "configs/ablation/lsm_shd_alif_A_current_learned_lowrank_no_rollback.yaml"
  "configs/ablation/lsm_shd_alif_B_random_floor_fixed_sparse_frozen_w.yaml"
  "configs/ablation/lsm_shd_alif_C_softplus_w_only_dense.yaml"
  "configs/ablation/lsm_shd_alif_D_smooth_lowrank_conductance_matched_scale.yaml"
  "configs/ablation/lsm_shd_alif_E_edgewise_soft_conductance.yaml"
  "configs/ablation/lsm_shd_alif_F_fixed_random_sparse_learned_w.yaml"
  "configs/ablation/lsm_shd_alif_G_lowrank_frozen_w_constant_g.yaml"
  "configs/ablation/lsm_shd_alif_H_lowrank_frozen_w_initialized_w.yaml"
)

mkdir -p "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

run_cmd() {
  log "CMD: $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

total=$(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))
job=0

log "Starting recurrent ablation A-H: jobs=$total seeds=${SEEDS[*]}"
log "device=$DEVICE data_dir=$DATA_DIR log_dir=$LOG_DIR dry_run=$DRY_RUN disable_tqdm=$DISABLE_TQDM"
if [[ -n "$EXTRA_OVERRIDES" ]]; then
  log "extra_overrides=$EXTRA_OVERRIDES"
fi

for config in "${CONFIGS[@]}"; do
  if [[ ! -f "$config" ]]; then
    log "Missing config: $config"
    exit 1
  fi

  label="$(basename "$config" .yaml)"
  for seed in "${SEEDS[@]}"; do
    job=$((job + 1))
    exp_name="${label}_${seed}"
    log_path="$LOG_DIR/${exp_name}.log"
    log "[$job/$total] config=$config seed=$seed experiment_name=$exp_name"
    {
      printf '[%s] job=%s/%s\n' "$(date '+%F %T')" "$job" "$total"
      printf '[%s] config=%s\n' "$(date '+%F %T')" "$config"
      printf '[%s] seed=%s\n' "$(date '+%F %T')" "$seed"
      printf '[%s] experiment_name=%s\n' "$(date '+%F %T')" "$exp_name"
      printf '[%s] device=%s data_dir=%s disable_tqdm=%s\n' "$(date '+%F %T')" "$DEVICE" "$DATA_DIR" "$DISABLE_TQDM"
    } | tee "$log_path" >/dev/null

    # shellcheck disable=SC2086
    cmd=( $PYTHON_CMD scripts/train.py
      --config "$config"
      --seed "$seed"
      "seed=$seed"
      "val_seed=$seed"
      "device=$DEVICE"
      "data_dir=$DATA_DIR"
      "experiment_name=$exp_name"
    )
    if [[ -n "$EXTRA_OVERRIDES" ]]; then
      # shellcheck disable=SC2206
      extra_args=( $EXTRA_OVERRIDES )
      cmd+=( "${extra_args[@]}" )
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
      run_cmd "${cmd[@]}"
    else
      log "Writing stdout/stderr to $log_path"
      PYTHONUNBUFFERED=1 DISABLE_TQDM="$DISABLE_TQDM" "${cmd[@]}" 2>&1 | tee -a "$log_path"
    fi
  done
done

log "Completed recurrent ablation A-H: jobs=$total"
