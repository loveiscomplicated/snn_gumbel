#!/usr/bin/env bash
set -euo pipefail

# Vast.ai setup for running SHD LSM experiments in this repository.
#
# Typical use from a fresh instance:
#   git clone <repo-url> snn_gumbel
#   cd snn_gumbel
#   bash scripts/setup_vastai.sh
#
# Optional:
#   RUN_SMOKE=1 bash scripts/setup_vastai.sh
#   CONFIG=configs/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi.yaml bash scripts/setup_vastai.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
DATA_DIR="${DATA_DIR:-./data}"
CONFIG="${CONFIG:-configs/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi.yaml}"
SEED="${SEED:-42}"
RUN_SMOKE="${RUN_SMOKE:-0}"
PREPARE_SHD="${PREPARE_SHD:-1}"
INSTALL_APT="${INSTALL_APT:-1}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_sudo() {
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    "$@"
  fi
}

log "Repository: $REPO_ROOT"
log "Config: $CONFIG"

if [[ "$INSTALL_APT" == "1" ]] && command -v apt-get >/dev/null 2>&1; then
  log "Installing system packages"
  run_sudo apt-get update
  if command -v sudo >/dev/null 2>&1; then
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
      git \
      curl \
      ca-certificates \
      build-essential \
      python3-venv \
      python3-dev
  else
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      git \
      curl \
      ca-certificates \
      build-essential \
      python3-venv \
      python3-dev
  fi
fi

log "Creating Python virtualenv at $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

log "Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

if [[ "$INSTALL_TORCH" == "1" ]]; then
  log "Installing CUDA PyTorch from $TORCH_INDEX_URL"
  python -m pip install --upgrade torch torchvision --index-url "$TORCH_INDEX_URL"
fi

log "Installing project requirements"
python -m pip install -r requirements.txt

log "Installing SHD/Tonic dependencies"
python -m pip install --upgrade tonic h5py

mkdir -p "$DATA_DIR" experiments runs

log "Environment check"
python - <<'PY'
import importlib
import torch

print("python ok")
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
for name in ("torchvision", "yaml", "tonic", "h5py"):
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "unknown")
    print(f"{name}: {version}")
PY

log "Config load check"
python - "$CONFIG" <<'PY'
import sys
from src.utils.config import load_config

cfg = load_config(sys.argv[1])
print("experiment_name:", cfg.experiment_name)
print("dataset:", cfg.dataset)
print("data_dir:", cfg.data_dir)
print("recurrent_mode:", cfg.liquid.recurrent_mode)
print("input_projection_mode:", cfg.liquid.input_projection_mode)
print("train_input_projection:", cfg.liquid.train_input_projection)
print("init_mode:", cfg.liquid.init_mode)
PY

if [[ "$PREPARE_SHD" == "1" ]]; then
  log "Preparing SHD dataset under $DATA_DIR"
  python - "$CONFIG" "$DATA_DIR" <<'PY'
import sys
from src.utils.config import load_config
from src.data.loaders import get_train_val_test_dataloaders

cfg = load_config(sys.argv[1], overrides=[f"data_dir={sys.argv[2]}"])
train_loader, val_loader, test_loader = get_train_val_test_dataloaders(cfg)
print("train_samples:", len(train_loader.dataset))
print("val_samples:", len(val_loader.dataset) if val_loader is not None else 0)
print("test_samples:", len(test_loader.dataset))
PY
fi

if [[ "$RUN_SMOKE" == "1" ]]; then
  log "Running 1-epoch smoke training"
  python scripts/train_lsm.py "$CONFIG" \
    data_dir="$DATA_DIR" \
    seed="$SEED" \
    epochs=1 \
    experiment_name="vastai_smoke_${SEED}"
fi

cat <<EOF

Setup complete.

Activate the environment:
  source $VENV_DIR/bin/activate

Example full run:
  python scripts/train_lsm.py $CONFIG \\
    data_dir=$DATA_DIR \\
    seed=$SEED \\
    experiment_name=lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_${SEED}

Useful options:
  RUN_SMOKE=1 bash scripts/setup_vastai.sh
  DATA_DIR=/workspace/data bash scripts/setup_vastai.sh
  TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash scripts/setup_vastai.sh
  INSTALL_APT=0 bash scripts/setup_vastai.sh
  INSTALL_TORCH=0 bash scripts/setup_vastai.sh
EOF
