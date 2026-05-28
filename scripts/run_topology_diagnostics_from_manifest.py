"""Run topology_diagnostics.py for selected manifest groups."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run topology diagnostics for experiments selected from manifest."
    )
    parser.add_argument("--manifest", default="experiments_manifest.csv")
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--skip-path-metrics",
        action="store_true",
        help="Forward --skip-path-metrics to topology_diagnostics.py.",
    )
    return parser.parse_args()


def selected_experiments(manifest_path: Path, groups: set[str]) -> list[str]:
    with manifest_path.open() as f:
        rows = list(csv.DictReader(f))
    experiments = [
        row["experiment_dir"]
        for row in rows
        if row["group_id"] in groups and row.get("main_table", "").lower() == "yes"
    ]
    if not experiments:
        raise RuntimeError(f"No experiments matched groups: {sorted(groups)}")
    missing = [exp for exp in experiments if not Path(exp).exists()]
    if missing:
        raise FileNotFoundError(f"Missing experiment directories: {missing}")
    return experiments


def main() -> None:
    args = parse_args()
    experiments = selected_experiments(Path(args.manifest), set(args.groups))
    cmd = [
        sys.executable,
        "scripts/topology_diagnostics.py",
        "--experiments",
        *experiments,
        "--output-dir",
        args.output_dir,
        "--device",
        args.device,
    ]
    if args.skip_path_metrics:
        cmd.append("--skip-path-metrics")

    print(f"Running topology diagnostics for {len(experiments)} experiments")
    print("Groups:", ", ".join(args.groups))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
