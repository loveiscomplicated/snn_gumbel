"""
CLI entry point for evaluation.

Examples:
    python scripts/evaluate.py --config experiments/mnist_baseline_2603231200/config.yaml \
                                --checkpoint experiments/mnist_baseline_2603231200/checkpoints/best.pt
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.evaluation.evaluate import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained SNN checkpoint")
    parser.add_argument(
        "--config", type=str, required=True, help="Config YAML used for this experiment"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to best.pt checkpoint"
    )
    parser.add_argument("overrides", nargs="*")

    args = parser.parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    run_evaluation(args.checkpoint, cfg)


if __name__ == "__main__":
    main()
