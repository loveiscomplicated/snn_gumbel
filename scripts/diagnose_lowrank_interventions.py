"""Run checkpoint-level lowrank intervention sensitivity diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.lowrank_interventions import (
    InterventionOptions,
    run_intervention_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Checkpoint-level intervention sensitivity diagnostics for lowrank LSM runs."
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Experiment run directories containing config.yaml and checkpoints/best.pt.",
    )
    parser.add_argument(
        "--diagnostic-dir",
        required=True,
        help="Existing lowrank runaway diagnostic directory with neuron_table.csv.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for intervention CSVs, report.md, and figures.",
    )
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", nargs="+", type=int, default=[50])
    parser.add_argument("--top-frac", nargs="*", type=float, default=[])
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument(
        "--intervention-set",
        choices=["core", "adaptation", "topology", "edge", "all"],
        default="core",
        help="Subset of interventions to run. Default core avoids a large first run.",
    )
    parser.add_argument(
        "--neuron-intervention-mode",
        choices=["communication_knockout", "full_neuron_silence", "both"],
        default="communication_knockout",
        help="Neuron intervention semantics. full_neuron_silence must be selected explicitly.",
    )
    parser.add_argument(
        "--degree-bin-controls",
        action="store_true",
        help="Add optional degree-bin + E/I matched random neuron controls when possible.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device for diagnostic forward passes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_intervention_diagnostics(
        InterventionOptions(
            run_dirs=[Path(path) for path in args.run_dirs],
            diagnostic_dir=Path(args.diagnostic_dir),
            output_dir=Path(args.output_dir),
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            top_k=args.top_k,
            top_frac=args.top_frac,
            random_repeats=args.random_repeats,
            device=args.device,
            intervention_set=args.intervention_set,
            neuron_intervention_mode=args.neuron_intervention_mode,
            degree_bin_controls=args.degree_bin_controls,
            seed=args.seed,
        )
    )
    print(f"[OK] wrote intervention diagnostics to {result['output_dir']}")
    for path in result["artifact_paths"]:
        print(f"[OK] {path}")
    if result["warnings"]:
        print("[WARN] insufficient evidence was recorded:")
        for warning in result["warnings"][:20]:
            print(f"  - {warning}")
        if len(result["warnings"]) > 20:
            print(f"  - ... {len(result['warnings']) - 20} more warnings")


if __name__ == "__main__":
    main()
