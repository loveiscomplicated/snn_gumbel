"""Run diagnose_liquid.py for manifest-selected experiments.

Outputs are collected under the diagnostics directory instead of mutating each
experiment directory.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run activity diagnostics from manifest.")
    parser.add_argument("--manifest", default="experiments_manifest.csv")
    parser.add_argument("--groups", nargs="+", default=["R1", "R4", "R8"])
    parser.add_argument("--output-dir", default="runs/diagnostics/main_g1_g3_g4/activity")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument(
        "--fresh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove the output activity CSV before running to avoid append duplicates.",
    )
    parser.add_argument(
        "--skip-expensive-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip cycle/clustering recomputation in diagnose_liquid.py.",
    )
    return parser.parse_args()


def read_manifest(path: Path, groups: set[str]) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    selected = [row for row in rows if row["group_id"] in groups]
    if not selected:
        raise RuntimeError(f"No manifest rows matched groups: {sorted(groups)}")
    return selected


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    json_dir = output_dir / "json"
    embedding_dir = output_dir / "embeddings"
    csv_path = output_dir / "activity_diagnostics.csv"
    json_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    if args.fresh and csv_path.exists():
        csv_path.unlink()

    rows = read_manifest(Path(args.manifest), set(args.groups))
    for index, row in enumerate(rows, start=1):
        exp_dir = Path(row["experiment_dir"])
        run_name = exp_dir.name
        config_path = exp_dir / "config.yaml"
        checkpoint_path = exp_dir / "checkpoints" / "best.pt"
        out_json = json_dir / f"{run_name}.json"
        out_embedding = embedding_dir / f"{run_name}.csv"

        cmd = [
            sys.executable,
            "scripts/diagnose_liquid.py",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--classes",
            str(args.classes),
            "--samples-per-class",
            str(args.samples_per_class),
            "--batches",
            str(args.batches),
            "--out-json",
            str(out_json),
            "--out-csv",
            str(csv_path),
            "--save-embeddings",
            str(out_embedding),
        ]
        if args.skip_expensive_graph:
            cmd.extend(["--skip-cycle-metrics", "--skip-clustering"])

        print(f"[{index}/{len(rows)}] {run_name}", flush=True)
        subprocess.run(cmd, check=True)

    print(f"Wrote {csv_path}")
    print(f"Wrote JSON files under {json_dir}")
    print(f"Wrote embeddings under {embedding_dir}")


if __name__ == "__main__":
    main()
