"""LSM training CLI entry point."""

import sys
from pathlib import Path

# ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.lsm.trainer import train


def main():
    print("asdf")
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/lsm_shd_baseline.yaml"
    overrides = sys.argv[2:] if len(sys.argv) > 2 else []
    cfg = load_config(config_path, overrides)
    train(cfg)


if __name__ == "__main__":
    main()
