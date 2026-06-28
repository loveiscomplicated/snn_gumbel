"""
CLI entry point for training.

Examples:
    python scripts/train.py --config configs/mnist_baseline.yaml
    python scripts/train.py --config configs/ablation_learned.yaml --resume
    python scripts/train.py --config configs/mnist_baseline.yaml lr=0.0005 epochs=50
    python scripts/train.py --config configs/mnist_baseline.yaml architecture.hidden_layers=[512,256]
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
from pathlib import Path

# Make project root importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import Config, load_config


def _is_lsm_config(cfg: Config) -> bool:
    return str(cfg.dataset).lower() == "shd" or (
        cfg.liquid.n_liquid > 0 and not cfg.architecture.hidden_layers
    )


def main():
    parser = argparse.ArgumentParser(description="Train SNN with Gumbel topology")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: use base defaults)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last checkpoint in the experiment dir",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override cfg.seed without using key=value syntax",
    )
    # All remaining args are treated as key=value overrides
    parser.add_argument(
        "overrides", nargs="*", help="CLI overrides in key=value form, e.g. lr=0.0005"
    )

    args = parser.parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    if args.seed is not None:
        cfg.seed = args.seed
    if _is_lsm_config(cfg):
        from src.lsm.trainer import train as train_lsm

        if args.resume:
            raise ValueError("LSM trainer does not support --resume via scripts/train.py")
        train_lsm(cfg)
    else:
        from src.training.trainer import train

        train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
