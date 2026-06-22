"""Diagnose learned-lowrank recurrent runaway evidence from existing runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.lowrank_runaway import DiagnosticOptions, run_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc learned-lowrank recurrent runaway diagnostics."
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Experiment run directories to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for CSV, report.md, and figures.",
    )
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--firing-threshold", type=float, default=0.9)
    parser.add_argument("--theta-grad-threshold", type=float, default=50.0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device for diagnostic forward passes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_diagnostics(
        DiagnosticOptions(
            run_dirs=[Path(path) for path in args.run_dirs],
            output_dir=Path(args.output_dir),
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            top_k=args.top_k,
            firing_threshold=args.firing_threshold,
            theta_grad_threshold=args.theta_grad_threshold,
            device=args.device,
        )
    )
    print(f"[OK] wrote diagnostics to {result['output_dir']}")
    for path in result["artifact_paths"]:
        print(f"[OK] {path}")
    if result["warnings"]:
        print("[WARN] missing / insufficient evidence was recorded:")
        for warning in result["warnings"][:20]:
            print(f"  - {warning}")
        if len(result["warnings"]) > 20:
            print(f"  - ... {len(result['warnings']) - 20} more warnings")


if __name__ == "__main__":
    main()

