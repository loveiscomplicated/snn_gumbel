"""
CLI entry point for visualisation.

Examples:
    python scripts/visualize.py --config experiments/mnist_baseline_2603231200/config.yaml \
                                 --checkpoint experiments/mnist_baseline_2603231200/checkpoints/best.pt
    # Save figures directly to the experiment directory:
    python scripts/visualize.py --config experiments/mnist_baseline_2603231200/config.yaml \
                                 --checkpoint experiments/mnist_baseline_2603231200/checkpoints/best.pt \
                                 --figures-dir experiments/mnist_baseline_2603231200/figures
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.evaluation.visualize import run_all


def main():
    parser = argparse.ArgumentParser(description="Visualise a trained SNN checkpoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=None,
        help="Where to save figures (default: ./figures)",
    )
    parser.add_argument("overrides", nargs="*")

    args = parser.parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    run_all(args.checkpoint, cfg, figures_dir=args.figures_dir)


if __name__ == "__main__":
    main()
