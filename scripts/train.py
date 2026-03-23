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

from src.utils.config import load_config
from src.training.trainer import train


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
    # All remaining args are treated as key=value overrides
    parser.add_argument(
        "overrides", nargs="*", help="CLI overrides in key=value form, e.g. lr=0.0005"
    )

    args = parser.parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
