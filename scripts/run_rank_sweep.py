"""Launch multi-seed low-rank sweep experiments.

Examples:
  uv run python scripts/run_rank_sweep.py \
    --config configs/lsm_shd_alif_lowrank_readout_spike_adaptation_concat.yaml \
    --tag lsm_shd_alif_lowrank_readout_spike_adaptation_concat_rank_sweep \
    --ranks 1 2 4 8 16 32 64 128 256 500 \
    --seeds 42 43 44 45

  uv run python scripts/run_rank_sweep.py \
    --config configs/lsm_shd_alif_lowrank_readout_spike_adaptation_concat.yaml \
    --tag smoke_rank_sweep \
    --ranks 16 32 \
    --seeds 42 \
    --dry-run \
    -- liquid.theta_warmup_epochs=2 epochs=4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rank sweep jobs by calling scripts/train_lsm.py with overrides."
    )
    parser.add_argument("--config", required=True, help="Base config YAML.")
    parser.add_argument(
        "--tag",
        required=True,
        help="Experiment name prefix. Each run becomes <tag>_r<RANK>_s<SEED>.",
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        required=True,
        help="Rank values to sweep.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="Seed values to sweep.",
    )
    parser.add_argument(
        "--python-prefix",
        nargs="+",
        default=["uv", "run", "python"],
        help="Interpreter command prefix used to launch training.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Extra key=value overrides. Prefix with `--` to terminate option parsing.",
    )
    return parser.parse_args()


def normalize_overrides(overrides: list[str]) -> list[str]:
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]
    return overrides


def build_command(
    python_prefix: list[str],
    config: str,
    tag: str,
    rank: int,
    seed: int,
    extra_overrides: list[str],
) -> list[str]:
    exp_name = f"{tag}_r{rank}_s{seed}"
    overrides = [
        f"experiment_name={exp_name}",
        f"seed={seed}",
        "use_validation=true",
        f"val_seed={seed}",
        "liquid.recurrent_mode=learned_lowrank",
        f"liquid.theta_rank={rank}",
    ]
    overrides.extend(extra_overrides)
    return [*python_prefix, "scripts/train_lsm.py", config, *overrides]


def main() -> None:
    args = parse_args()
    extra_overrides = normalize_overrides(args.overrides)
    planned = [
        build_command(args.python_prefix, args.config, args.tag, rank, seed, extra_overrides)
        for seed in args.seeds
        for rank in args.ranks
    ]

    print(f"Planned jobs: {len(planned)}")
    for idx, cmd in enumerate(planned, start=1):
        print(f"[{idx}/{len(planned)}] {' '.join(cmd)}")
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)

    if args.dry_run:
        print("Dry run only; no jobs executed.")


if __name__ == "__main__":
    main()
